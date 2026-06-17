import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib import error, request


VALID_LINK_ACTIONS = {"open", "search"}


@dataclass
class AuditIssue:
    id: str
    query: str
    field: str
    message: str


def _iter_result_rows(payload: object) -> list[dict]:
    if isinstance(payload, list):
        rows: list[dict] = []
        for item in payload:
            if isinstance(item, dict) and isinstance(item.get("turns"), list):
                rows.extend(turn for turn in item["turns"] if isinstance(turn, dict))
            elif isinstance(item, dict):
                rows.append(item)
        return rows
    if isinstance(payload, dict) and isinstance(payload.get("turns"), list):
        return [turn for turn in payload["turns"] if isinstance(turn, dict)]
    return []


def audit_response(row: dict) -> list[AuditIssue]:
    row_id = str(row.get("id") or row.get("query") or "unknown")
    query = str(row.get("query") or "")
    issues: list[AuditIssue] = []

    answer = str(row.get("answer") or row.get("answer_preview") or "")
    if not answer.strip():
        issues.append(AuditIssue(row_id, query, "answer", "missing answer text"))

    links = row.get("links") or row.get("relevant_links") or []
    if links and not isinstance(links, list):
        issues.append(AuditIssue(row_id, query, "links", "links must be a list"))
        links = []
    for index, link in enumerate(links):
        if not isinstance(link, dict):
            issues.append(AuditIssue(row_id, query, f"links[{index}]", "link must be an object"))
            continue
        if not str(link.get("title") or "").strip():
            issues.append(AuditIssue(row_id, query, f"links[{index}].title", "missing link title"))
        if not str(link.get("url") or "").strip():
            issues.append(AuditIssue(row_id, query, f"links[{index}].url", "missing link URL"))
        action = str(link.get("action") or "open")
        if action not in VALID_LINK_ACTIONS:
            issues.append(AuditIssue(row_id, query, f"links[{index}].action", f"invalid action `{action}`"))
        if action == "search" and not str(link.get("search_hint") or "").strip():
            issues.append(AuditIssue(row_id, query, f"links[{index}].search_hint", "search links need a search_hint"))

    route = row.get("route") or {}
    if route and not str(route.get("agent") or "").strip():
        issues.append(AuditIssue(row_id, query, "route.agent", "missing route agent"))

    provenance = row.get("data_provenance") or {}
    if provenance and not isinstance(provenance.get("display_rows", []), list):
        issues.append(AuditIssue(row_id, query, "data_provenance.display_rows", "display_rows must be a list"))

    if row.get("plot_base64") and not str(row.get("plot_caption") or "").strip():
        issues.append(AuditIssue(row_id, query, "plot_caption", "plot responses should include plot_caption"))

    return issues


def audit_results(rows: list[dict]) -> list[AuditIssue]:
    issues: list[AuditIssue] = []
    for row in rows:
        issues.extend(audit_response(row))
    return issues


def render_markdown(issues: list[AuditIssue], *, source: str, total_rows: int) -> str:
    lines = [
        "# Frontend Response Audit",
        "",
        f"- Source: `{source}`",
        f"- Responses checked: {total_rows}",
        f"- Issues: {len(issues)}",
        "",
    ]
    if not issues:
        lines.append("Status: pass")
        return "\n".join(lines) + "\n"

    lines.extend([
        "Status: review",
        "",
        "| ID | Query | Field | Issue |",
        "| --- | --- | --- | --- |",
    ])
    for issue in issues:
        lines.append(
            "| {id} | {query} | {field} | {message} |".format(
                id=_md(issue.id),
                query=_md(issue.query),
                field=_md(issue.field),
                message=_md(issue.message),
            )
        )
    return "\n".join(lines) + "\n"


def _md(value: object) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")


def post_query(live_url: str, query_text: str) -> dict:
    payload = json.dumps({"query": query_text}).encode("utf-8")
    req = request.Request(
        live_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit chatbot responses for frontend rendering safety.")
    parser.add_argument("--live-results", default="docs/evaluation_live_results.json", help="Saved live-results JSON to audit.")
    parser.add_argument("--live-url", default="", help="Optional FastAPI /query URL for direct sample queries.")
    parser.add_argument("--query", action="append", default=[], help="Query to send when --live-url is provided. Can be repeated.")
    parser.add_argument("--output", default="docs/frontend_response_audit.md", help="Markdown report path.")
    parser.add_argument("--json-output", default="", help="Optional JSON issue report path.")
    args = parser.parse_args()

    source = args.live_results
    if args.live_url:
        sample_queries = args.query or [
            "show me carbon dioxide emissions for Europe",
            "What is the REMIND model?",
            "where can I find Climate Watch",
            "show solar capacity for Greece under Policy in 2100",
        ]
        rows = [post_query(args.live_url, query_text) | {"query": query_text, "id": f"sample-{idx}"} for idx, query_text in enumerate(sample_queries, start=1)]
        source = args.live_url
    else:
        payload = json.loads(Path(args.live_results).read_text())
        rows = _iter_result_rows(payload)

    issues = audit_results(rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_markdown(issues, source=source, total_rows=len(rows)))

    if args.json_output:
        json_path = Path(args.json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps([asdict(issue) for issue in issues], indent=2) + "\n")

    print(f"Wrote {output_path} for {len(rows)} responses ({len(issues)} issues).")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
