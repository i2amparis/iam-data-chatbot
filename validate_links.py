import argparse
import json
import socket
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib import error, request
from urllib.parse import urlparse

from link_catalog import DEFAULT_OUTPUT as DEFAULT_LINK_CATALOG


DEFAULT_REPORT = Path("docs/link_validation_results.md")


@dataclass
class LinkValidationResult:
    url: str
    status: str
    http_status: int | None = None
    method: str = ""
    final_url: str = ""
    error: str = ""
    titles: list[str] | None = None
    categories: list[str] | None = None


def load_catalog(path: Path = DEFAULT_LINK_CATALOG) -> list[dict]:
    return json.loads(path.read_text())


def unique_catalog_urls(catalog: list[dict], domain: str = "") -> list[dict]:
    grouped: dict[str, dict] = {}
    domain = str(domain or "").strip().lower()
    for item in catalog:
        url = str(item.get("url") or "").strip()
        if not url:
            continue
        if domain:
            host = urlparse(url).netloc.lower()
            if host != domain and not host.endswith("." + domain):
                continue
        bucket = grouped.setdefault(url, {"url": url, "titles": [], "categories": []})
        title = str(item.get("title") or "").strip()
        category = str(item.get("category") or "").strip()
        if title and title not in bucket["titles"]:
            bucket["titles"].append(title)
        if category and category not in bucket["categories"]:
            bucket["categories"].append(category)
    return sorted(grouped.values(), key=lambda row: row["url"])


def _request_url(url: str, method: str, timeout: float, opener=request.urlopen) -> tuple[int, str]:
    req = request.Request(
        url,
        method=method,
        headers={
            "User-Agent": "iam-data-chatbot-link-validator/1.0",
            "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
        },
    )
    with opener(req, timeout=timeout) as response:
        status = int(getattr(response, "status", response.getcode()))
        final_url = str(response.geturl() or url)
        return status, final_url


def _status_label(http_status: int, final_url: str, original_url: str) -> str:
    if 200 <= http_status < 300:
        return "redirected" if final_url and final_url.rstrip("/") != original_url.rstrip("/") else "ok"
    if 300 <= http_status < 400:
        return "redirected"
    if 400 <= http_status < 500:
        return "broken"
    if http_status >= 500:
        return "server_error"
    return "unknown"


def check_url(url: str, timeout: float = 10.0, opener=request.urlopen) -> LinkValidationResult:
    last_error = ""
    for method in ("HEAD", "GET"):
        try:
            status, final_url = _request_url(url, method, timeout, opener)
            return LinkValidationResult(
                url=url,
                status=_status_label(status, final_url, url),
                http_status=status,
                method=method,
                final_url=final_url,
            )
        except error.HTTPError as exc:
            status = int(getattr(exc, "code", 0) or 0)
            if method == "HEAD" and status in {403, 405, 406, 429}:
                last_error = f"{method} {status}"
                continue
            return LinkValidationResult(
                url=url,
                status=_status_label(status, str(getattr(exc, "url", "") or url), url),
                http_status=status,
                method=method,
                final_url=str(getattr(exc, "url", "") or url),
                error=str(exc.reason or ""),
            )
        except (error.URLError, TimeoutError, socket.timeout) as exc:
            reason = getattr(exc, "reason", exc)
            last_error = str(reason)
            if method == "HEAD":
                continue
            return LinkValidationResult(
                url=url,
                status="error",
                method=method,
                error=last_error,
            )

    return LinkValidationResult(url=url, status="error", method="GET", error=last_error)


def validate_catalog_links(
    catalog: list[dict],
    *,
    domain: str = "",
    timeout: float = 10.0,
    opener=request.urlopen,
) -> list[LinkValidationResult]:
    results: list[LinkValidationResult] = []
    for row in unique_catalog_urls(catalog, domain=domain):
        result = check_url(row["url"], timeout=timeout, opener=opener)
        result.titles = row["titles"]
        result.categories = row["categories"]
        results.append(result)
    return results


def render_markdown(results: list[LinkValidationResult], *, catalog_path: Path, domain: str = "") -> str:
    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1

    lines = [
        "# Link Validation Results",
        "",
        f"- Catalog: `{catalog_path}`",
        f"- Domain filter: `{domain or 'all domains'}`",
        f"- Unique URLs checked: {len(results)}",
    ]
    for status, count in sorted(counts.items()):
        lines.append(f"- `{status}`: {count}")

    lines.extend([
        "",
        "| Status | HTTP | Method | URL | Final URL | Titles | Error |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ])
    for result in results:
        titles = ", ".join((result.titles or [])[:5])
        if result.titles and len(result.titles) > 5:
            titles += f" and {len(result.titles) - 5} more"
        lines.append(
            "| {status} | {http_status} | {method} | {url} | {final_url} | {titles} | {error} |".format(
                status=_md(result.status),
                http_status=_md(result.http_status if result.http_status is not None else ""),
                method=_md(result.method),
                url=_md(result.url),
                final_url=_md(result.final_url),
                titles=_md(titles),
                error=_md(result.error),
            )
        )
    return "\n".join(lines) + "\n"


def _md(value: object) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate URLs from the IAM PARIS link catalog.")
    parser.add_argument("--catalog", default=str(DEFAULT_LINK_CATALOG), help="Path to generated link catalog JSON.")
    parser.add_argument("--output", default=str(DEFAULT_REPORT), help="Markdown report path.")
    parser.add_argument("--json-output", default="", help="Optional JSON results path.")
    parser.add_argument("--domain", default="", help="Optional domain filter, e.g. iamparis.eu.")
    parser.add_argument("--timeout", type=float, default=10.0, help="Per-request timeout in seconds.")
    args = parser.parse_args()

    catalog_path = Path(args.catalog)
    catalog = load_catalog(catalog_path)
    results = validate_catalog_links(catalog, domain=args.domain, timeout=args.timeout)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_markdown(results, catalog_path=catalog_path, domain=args.domain))

    if args.json_output:
        json_path = Path(args.json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps([asdict(result) for result in results], indent=2) + "\n")

    broken = sum(1 for result in results if result.status in {"broken", "server_error", "error"})
    print(f"Wrote {output_path} for {len(results)} unique URLs ({broken} problem links).")
    return 1 if broken else 0


if __name__ == "__main__":
    raise SystemExit(main())
