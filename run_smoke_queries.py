import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import requests

from validate_links import check_url


DEFAULT_API_URL = os.getenv("IAM_API_URL", "http://127.0.0.1:8000/query")
MANUAL_BULK_MD_PATH = os.getenv("SMOKE_QUERIES_MD", "manual_bulk_queries.md")

# Cache HTTP results per URL so we don't re-hit the same link across queries.
_LINK_STATUS_CACHE: Dict[str, bool] = {}


def _link_works(url: str) -> bool:
    """Return True when the URL resolves (HTTP 2xx/3xx)."""
    url = (url or "").strip()
    if not url:
        return False
    if url not in _LINK_STATUS_CACHE:
        try:
            result = check_url(url, timeout=10.0)
            _LINK_STATUS_CACHE[url] = result.status in {"ok", "redirected"}
        except Exception:
            _LINK_STATUS_CACHE[url] = False
    return _LINK_STATUS_CACHE[url]


def _check_links(links: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate that each relevant link's URL actually resolves."""
    urls = [str(l.get("url") or "").strip() for l in (links or []) if isinstance(l, dict)]
    urls = [u for u in urls if u]
    ok, broken = [], []
    for u in urls:
        (ok if _link_works(u) else broken).append(u)
    return {
        "links_total": len(urls),
        "links_ok": len(ok),
        "broken_links": broken,
        "links_all_ok": bool(urls) and not broken,
    }


def _ensure_api_ready(api_url: str) -> None:
    # api_url is /query; health is one level up
    base = api_url.rsplit("/", 1)[0]
    health_url = f"{base}/health"
    deadline = time.time() + 240
    while time.time() < deadline:
        try:
            r = requests.get(health_url, timeout=2)
            if r.status_code == 200:
                return
        except Exception:
            time.sleep(1)
    raise RuntimeError(f"Timed out waiting for API at {health_url}")


def _start_uvicorn_if_needed(api_url: str) -> None:
    base = api_url.rsplit("/", 1)[0]
    # base looks like http://127.0.0.1:8000
    m = re.search(r"http://127\.0\.0\.1:(\d+)", base)
    port = int(m.group(1)) if m else 8000

    try:
        _ensure_api_ready(api_url)
        return
    except Exception:
        pass

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "fastapi_app:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    print(f"[smoke] Starting uvicorn: {' '.join(cmd)}")
    subprocess.Popen(cmd)
    _ensure_api_ready(api_url)


def _extract_queries_from_md(md_text: str) -> List[str]:
    """
    Extract unchecked checklist queries in appearance order:
      - `- [ ] `query``
      - `1. [ ] `query``
    """
    queries: List[str] = []
    for line in md_text.splitlines():
        m = re.match(r"^\s*-\s*\[\s*\]\s*`([^`]+)`\s*$", line)
        if m:
            queries.append(m.group(1).strip())
            continue
        m2 = re.match(r"^\s*\d+\.\s*\[\s*\]\s*`([^`]+)`\s*$", line)
        if m2:
            queries.append(m2.group(1).strip())
            continue
        m3 = re.match(r"^\s*-\s*\[\s*[xX]\s*\]\s*`([^`]+)`\s*$", line)
        if m3:
            queries.append(m3.group(1).strip())
            continue
    return queries


def _post_query(
    query: str,
    session_id: str,
    reset_session: bool,
    api_url: str,
    *,
    max_retries: int = 5,
    base_backoff_s: float = 1.0,
) -> Dict[str, Any]:
    payload = {"query": query, "session_id": session_id, "reset_session": reset_session}
    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("IAM_API_KEY", "").strip()
    if api_key:
        headers["X-API-Key"] = api_key

    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            r = requests.post(api_url, headers=headers, json=payload, timeout=300)

            # Handle rate limiting with backoff
            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After")
                if retry_after:
                    try:
                        sleep_s = float(retry_after)
                    except ValueError:
                        sleep_s = None
                else:
                    sleep_s = None

                if attempt >= max_retries:
                    r.raise_for_status()

                if sleep_s is None:
                    sleep_s = base_backoff_s * (2**attempt)
                time.sleep(sleep_s)
                continue

            r.raise_for_status()
            return r.json()

        except requests.exceptions.RequestException as e:
            last_exc = e
            # For non-429 errors, don't retry by default—fail fast
            # (429 is handled above using status_code).
            break

    # If we get here, we failed without returning JSON
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Request failed without a captured exception")


def _row(query: str, res: Dict[str, Any], check_links: bool = False) -> Dict[str, Any]:
    answer = str(res.get("answer") or "")
    answer_lower = answer.lower()
    plot_present = bool(res.get("plot_base64")) or bool(res.get("plot_caption"))
    no_data = ("i could not find data" in answer_lower) or ("no data found" in answer_lower)
    links = res.get("relevant_links") or []
    row = {
        "query": query,
        "plot_present": plot_present,
        "no_data": no_data,
        "relevant_links_count": len(links),
        "suggested_next_questions_count": len(res.get("suggested_next_questions") or []),
        "route": res.get("route") or {},
        "entities": res.get("entities") or {},
        "data_provenance": res.get("data_provenance") or {},
        "answer_preview": answer[:400],
    }
    if check_links:
        row["link_check"] = _check_links(links)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--md", default=MANUAL_BULK_MD_PATH, help="Path to manual_bulk_queries.md")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="POST endpoint, default /query")
    parser.add_argument("--port", type=int, default=8000, help="Kept for compatibility; uvicorn uses auto from api url")
    parser.add_argument("--session-prefix", default="smoke", help="Prefix for generated session_id")
    parser.add_argument("--reuse-session", action="store_true", help="Reuse a single session_id for all queries")
    parser.add_argument("--debug", action="store_true", help="Print full responses")
    parser.add_argument("--check-links", action="store_true",
                        help="Verify that returned relevant_links URLs actually resolve (HTTP)")
    args = parser.parse_args()

    md_path = Path(args.md)
    if not md_path.exists():
        raise FileNotFoundError(str(md_path))
    md_text = md_path.read_text(encoding="utf-8")

    queries = _extract_queries_from_md(md_text)
    if not queries:
        raise RuntimeError("No queries extracted from markdown")

    _start_uvicorn_if_needed(args.api_url)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("manual_smoke_results") / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = out_dir / "results.jsonl"
    summary_path = out_dir / "results_summary.md"
    summary_rows_json = out_dir / "results_summary.json"

    session_id_static = f"{args.session_prefix}_all"
    rows: List[Dict[str, Any]] = []

    with jsonl_path.open("w", encoding="utf-8") as f:
        for i, q in enumerate(queries):
            session_id = session_id_static if args.reuse_session else f"{args.session_prefix}_{i+1}"
            reset = True if (not args.reuse_session or i == 0) else False
            print(f"[smoke] {i+1}/{len(queries)} session={session_id} query={q}")

            try:
                res = _post_query(q, session_id=session_id, reset_session=reset, api_url=args.api_url)
                session_out = res.get("session_id") or session_id

                f.write(
                    json.dumps({"query": q, "session_id": session_out, "response": res}, ensure_ascii=False) + "\n"
                )

                r = _row(q, res, check_links=args.check_links)
                r["session_id"] = session_out
                rows.append(r)

                if args.debug:
                    print(json.dumps(res, ensure_ascii=False, indent=2))

            except Exception as e:
                # Continue smoke suite even if one query fails (e.g., 429 after retries, 500, etc.)
                err_payload = {"error": repr(e)}
                f.write(json.dumps({"query": q, "session_id": session_id, "response": err_payload}, ensure_ascii=False) + "\n")

                # Minimal row so summary generation still works
                r = {
                    "query": q,
                    "plot_present": False,
                    "no_data": True,
                    "relevant_links_count": 0,
                    "suggested_next_questions_count": 0,
                    "route": {},
                    "entities": {},
                    "data_provenance": {},
                    "answer_preview": "",
                    "session_id": session_id,
                }
                rows.append(r)

                print(f"[smoke] Query failed (continuing): {q} error={repr(e)}")

            time.sleep(0.35)

    # Human summary (lightweight)
    lines = ["# Manual Smoke Test Results", f"Generated: {ts}", f"Total queries: {len(rows)}", ""]
    if args.check_links:
        checked = [r for r in rows if r.get("link_check")]
        with_links = [r for r in checked if r["link_check"]["links_total"] > 0]
        only_working = [r for r in with_links if r["link_check"]["links_all_ok"]]
        broken = [r for r in with_links if not r["link_check"]["links_all_ok"]]

        # Top summary: how many queries returned ONLY working links
        lines.append(
            f"Links-only working: {len(only_working)}/{len(with_links)} "
            f"(broken={len(broken)})."
        )
        lines.append("")

    for idx, r in enumerate(rows, 1):
        status = "NO_DATA" if r["no_data"] else ("PLOT" if r["plot_present"] else "OK")
        link_note = ""
        lc = r.get("link_check")
        if lc and lc["links_total"]:
            tag = "LINKS_OK" if lc["links_all_ok"] else "LINKS_BROKEN"
            link_note = f" [{tag} {lc['links_ok']}/{lc['links_total']}]"
            # Include broken=<url> when there are broken links
            if lc.get("broken_links"):
                link_note += " broken=" + ", ".join(lc["broken_links"])
        lines.append(f"{idx}. [{status}] `{r['query']}`{link_note}")
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    summary_rows_json.write_text(json.dumps({"generated": ts, "count": len(rows), "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[smoke] Done. JSONL: {jsonl_path}")
    print(f"[smoke] Summary: {summary_path}")
    print(f"[smoke] Summary JSON: {summary_rows_json}")


if __name__ == "__main__":
    main()
