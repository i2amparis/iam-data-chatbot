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

_CLARIFICATION_MARKERS = (
    "choose the variable",
    "choose the region",
    "choose the scenario",
    "which variable should i use",
    "which region should i use",
    "which scenario should i use",
    "i need one more detail",
    "reply with a number",
    "closest valid options",
)

_NO_DATA_MARKERS = (
    "i could not find data",
    "no data found",
    "no time series data",
    "no timeseries data",
    "can't combine these series",
    "cannot combine these series",
    "can't plot the complete",
    "could not find any projection years",
)


def _query_requests_plot(query: str) -> bool:
    return bool(re.search(r"\b(?:plot|chart|graph|visuali[sz]e|draw)\b", query, re.IGNORECASE))


def _smoke_status(row: Dict[str, Any]) -> str:
    if row.get("error"):
        return "ERROR"
    if row.get("plot_present"):
        return "PLOT"
    if row.get("plot_requested") and row.get("no_data"):
        return "PLOT_NO_DATA"
    if row.get("plot_requested"):
        return "PLOT_FAILED"
    if row.get("no_data"):
        return "NO_DATA"
    if row.get("clarification"):
        return "CLARIFICATION"
    return "OK"


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
    return [case["query"] for case in _extract_query_cases_from_md(md_text)]


def _extract_query_cases_from_md(md_text: str) -> List[Dict[str, str]]:
    """Extract checklist queries with category and optional follow-up block."""
    cases: List[Dict[str, str]] = []
    category = "Uncategorized"
    conversation_group = ""
    for line in md_text.splitlines():
        heading = re.match(r"^\s*##\s+(.+?)\s*$", line)
        if heading:
            category = heading.group(1).strip()
            conversation_group = ""
            continue
        subheading = re.match(r"^\s*###\s+(.+?)\s*$", line)
        if subheading:
            conversation_group = subheading.group(1).strip()
            continue
        def append_case(query: str) -> None:
            case = {"query": query.strip(), "category": category}
            if conversation_group:
                case["conversation_group"] = conversation_group
            cases.append(case)

        m = re.match(r"^\s*-\s*\[\s*\]\s*`([^`]+)`\s*$", line)
        if m:
            append_case(m.group(1))
            continue
        m2 = re.match(r"^\s*\d+\.\s*\[\s*\]\s*`([^`]+)`\s*$", line)
        if m2:
            append_case(m2.group(1))
            continue
        m3 = re.match(r"^\s*-\s*\[\s*[xX]\s*\]\s*`([^`]+)`\s*$", line)
        if m3:
            append_case(m3.group(1))
            continue
    return cases


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
    no_data = any(marker in answer_lower for marker in _NO_DATA_MARKERS)
    clarification = any(marker in answer_lower for marker in _CLARIFICATION_MARKERS)
    links = res.get("relevant_links") or []
    row = {
        "query": query,
        "plot_requested": _query_requests_plot(query),
        "plot_present": plot_present,
        "no_data": no_data,
        "clarification": clarification,
        "relevant_links_count": len(links),
        "suggested_next_questions_count": len(res.get("suggested_next_questions") or []),
        "route": res.get("route") or {},
        "entities": res.get("entities") or {},
        "data_provenance": res.get("data_provenance") or {},
        "answer": answer,
        "answer_preview": answer[:400],
    }
    if check_links:
        row["link_check"] = _check_links(links)
    row["status"] = _smoke_status(row)
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

    cases = _extract_query_cases_from_md(md_text)
    if not cases:
        raise RuntimeError("No queries extracted from markdown")

    _start_uvicorn_if_needed(args.api_url)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("manual_smoke_results") / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = out_dir / "results.jsonl"
    summary_path = out_dir / "results_summary.md"
    summary_rows_json = out_dir / "results_summary.json"
    query_answers_path = out_dir / "query_answers.md"

    session_id_static = f"{args.session_prefix}_all"
    group_sessions: Dict[str, str] = {}
    group_turns: Dict[str, int] = {}
    rows: List[Dict[str, Any]] = []

    with jsonl_path.open("w", encoding="utf-8") as f:
        for i, case in enumerate(cases):
            q = case["query"]
            category = case["category"]
            conversation_group = str(case.get("conversation_group") or "").strip()
            if args.reuse_session:
                session_id = session_id_static
                reset = i == 0
            elif conversation_group:
                if conversation_group not in group_sessions:
                    safe_group = re.sub(r"[^a-z0-9]+", "_", conversation_group.casefold()).strip("_")
                    group_sessions[conversation_group] = f"{args.session_prefix}_{safe_group or len(group_sessions)+1}"
                    group_turns[conversation_group] = 0
                session_id = group_sessions[conversation_group]
                reset = group_turns[conversation_group] == 0
                group_turns[conversation_group] += 1
            else:
                session_id = f"{args.session_prefix}_{i+1}"
                reset = True
            print(f"[smoke] {i+1}/{len(cases)} category={category} session={session_id} query={q}")

            try:
                res = _post_query(q, session_id=session_id, reset_session=reset, api_url=args.api_url)
                session_out = res.get("session_id") or session_id

                f.write(
                    json.dumps({"query": q, "session_id": session_out, "response": res}, ensure_ascii=False) + "\n"
                )

                r = _row(q, res, check_links=args.check_links)
                r["session_id"] = session_out
                r["category"] = category
                if conversation_group:
                    r["conversation_group"] = conversation_group
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
                    "category": category,
                    "conversation_group": conversation_group,
                    "plot_requested": _query_requests_plot(q),
                    "plot_present": False,
                    "no_data": True,
                    "clarification": False,
                    "relevant_links_count": 0,
                    "suggested_next_questions_count": 0,
                    "route": {},
                    "entities": {},
                    "data_provenance": {},
                    "answer": "",
                    "answer_preview": "",
                    "session_id": session_id,
                    "error": repr(e),
                    "status": "ERROR",
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

    status_counts: Dict[str, int] = {}
    for row in rows:
        status = str(row.get("status") or _smoke_status(row))
        status_counts[status] = status_counts.get(status, 0) + 1
    lines.append(
        "Status counts: "
        + ", ".join(f"{name}={count}" for name, count in sorted(status_counts.items()))
    )
    lines.append("")

    category_counts: Dict[str, Dict[str, int]] = {}
    for row in rows:
        category = str(row.get("category") or "Uncategorized")
        status = str(row.get("status") or _smoke_status(row))
        bucket = category_counts.setdefault(category, {})
        bucket[status] = bucket.get(status, 0) + 1
    lines.append("## Results by category")
    lines.append("")
    for category, counts in category_counts.items():
        total = sum(counts.values())
        details = ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
        lines.append(f"- **{category}** ({total}): {details}")
    lines.append("")

    for idx, r in enumerate(rows, 1):
        status = str(r.get("status") or _smoke_status(r))
        link_note = ""
        lc = r.get("link_check")
        if lc and lc["links_total"]:
            tag = "LINKS_OK" if lc["links_all_ok"] else "LINKS_BROKEN"
            link_note = f" [{tag} {lc['links_ok']}/{lc['links_total']}]"
            # Include broken=<url> when there are broken links
            if lc.get("broken_links"):
                link_note += " broken=" + ", ".join(lc["broken_links"])
        lines.append(f"{idx}. [{status}] [{r.get('category', 'Uncategorized')}] `{r['query']}`{link_note}")
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    summary_rows_json.write_text(json.dumps({"generated": ts, "count": len(rows), "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    query_answer_lines = [
        "# Query and Answer Transcript",
        f"Generated: {ts}",
        f"Total queries: {len(rows)}",
        "",
    ]
    for idx, row in enumerate(rows, 1):
        answer = str(row.get("answer") or row.get("error") or "(No answer returned.)").strip()
        query_answer_lines.extend(
            [
                f"## {idx}. {row['query']}",
                "",
                f"Category: {row.get('category', 'Uncategorized')}",
                "",
                answer,
                "",
            ]
        )
    query_answers_path.write_text("\n".join(query_answer_lines), encoding="utf-8")

    print(f"[smoke] Done. JSONL: {jsonl_path}")
    print(f"[smoke] Summary: {summary_path}")
    print(f"[smoke] Summary JSON: {summary_rows_json}")
    print(f"[smoke] Query/answer transcript: {query_answers_path}")


if __name__ == "__main__":
    main()
