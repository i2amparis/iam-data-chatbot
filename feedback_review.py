import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from model_profiles import find_model_profile


DEFAULT_SOURCE = Path("docs/eval_feedback_candidates.jsonl")
DEFAULT_REPORT = Path("docs/eval_feedback_review.md")
DEFAULT_CSV = Path("docs/eval_feedback_candidates.csv")

EVAL_COLUMNS = [
    "id",
    "query",
    "expected_route",
    "expected_variable",
    "expected_region",
    "expected_scenario",
    "expected_model",
    "useful_clarification",
    "useful_link",
    "no_hallucinated_data",
]


def _normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", str(query or "").strip().lower())


def load_feedback_candidates(path: Path = DEFAULT_SOURCE) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(item.get("query", "")).strip():
                rows.append(item)
    return rows


def dedupe_feedback_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_query: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = _normalize_query(str(row.get("query", "")))
        if not key:
            continue
        current = by_query.get(key)
        if current is None or str(row.get("timestamp", "")) > str(current.get("timestamp", "")):
            by_query[key] = row
    return sorted(by_query.values(), key=lambda item: str(item.get("timestamp", "")), reverse=True)


def _expected_link_from_query(query: str, route: str) -> str:
    q = _normalize_query(query)
    if route == "model_explanation":
        return "models"
    if any(term in q for term in ["application", "library", "aqueduct", "climate watch", "portal", "dashboard"]):
        return "application_library"
    if any(term in q for term in ["data story", "catalogue", "catalog", "barrier", "technology inventory"]):
        return "data_stories"
    if "ndc" in q:
        return "ndc_aspects"
    if any(term in q for term in ["fit for 55", "net zero", "cost of capital"]):
        return "iam_compact"
    if "contact" in q:
        return "contact"
    return "results"


def _is_context_only_query(query: str) -> bool:
    q = _normalize_query(query)
    return bool(
        q in {"emissions", "co2 emissions", "ghg emissions", "temperature", "capacity", "solar", "wind"}
        or
        re.fullmatch(r"(same|same for|what about|plot it|graph it|show it|compare it)(?: .*)?", q)
        or re.fullmatch(r"compare (?:to|with|against) (?:current policy|baseline|policy|the scenario)", q)
        or re.fullmatch(r"use the (?:first|second|third|fourth|fifth) scenario", q)
    )


def _looks_like_site_navigation_query(query: str) -> bool:
    q = _normalize_query(query)
    if any(term in q for term in ["transformation results", "transformation workspace", "land use results"]):
        return True
    if (
        "ndc" in q
        and any(term in q for term in ["results", "workspace"])
        and any(term in q for term in ["transport", "buildings", "afolu"])
    ):
        return True
    if any(term in q for term in ["where can i find", "open ", "application library", "climate watch", "aqueduct"]):
        return True
    return False


def to_eval_row(row: dict[str, Any], index: int) -> dict[str, str]:
    entities = row.get("entities") if isinstance(row.get("entities"), dict) else {}
    query = str(row.get("query", "")).strip()
    profile = find_model_profile(query)
    route = str(row.get("route") or "data_query")
    expected_model = str(entities.get("model", "") or "")
    if expected_model == "China-MORE" and "china" in query.lower() and not re.search(r"\b(model|using|use|with)\b", query, flags=re.IGNORECASE):
        expected_model = ""
    if profile and re.search(r"\b(model|assumptions?|about|explain|info|information|what\s+is)\b", query, flags=re.IGNORECASE):
        route = "model_explanation"
        expected_model = str(profile.get("name", "") or expected_model)
    elif _looks_like_site_navigation_query(query):
        route = "general_qa"
        entities = {}
    expected_scenario = str(entities.get("scenario", "") or "")
    if route == "data_query" and re.search(r"\b(ndc|nationally determined contributions?)\b", query, flags=re.IGNORECASE):
        expected_scenario = "NDC"
    useful_clarification = (
        "no"
        if route in {"model_explanation", "general_qa"}
        else ("yes" if row.get("no_data_reason") or row.get("route_confidence", 1.0) < 0.55 else "no")
    )
    if route == "data_query" and "ndc impacts" in query.lower():
        useful_clarification = "no"
    return {
        "id": f"feedback-{index:03d}",
        "query": query,
        "expected_route": route,
        "expected_variable": str(entities.get("variable", "") or ""),
        "expected_region": str(entities.get("region", "") or ""),
        "expected_scenario": expected_scenario,
        "expected_model": expected_model,
        "useful_clarification": useful_clarification,
        "useful_link": _expected_link_from_query(str(row.get("query", "")), route),
        "no_hallucinated_data": "yes",
    }


def write_feedback_csv(rows: list[dict[str, Any]], output: Path = DEFAULT_CSV) -> list[dict[str, str]]:
    output.parent.mkdir(parents=True, exist_ok=True)
    filtered = [
        row for row in rows
        if not _is_context_only_query(str(row.get("query", "")))
    ]
    eval_rows = [to_eval_row(row, index) for index, row in enumerate(filtered, start=1)]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVAL_COLUMNS)
        writer.writeheader()
        writer.writerows(eval_rows)
    return eval_rows


def render_feedback_report(rows: list[dict[str, Any]], eval_rows: list[dict[str, str]], source: Path, csv_output: Path) -> str:
    eval_by_query = {_normalize_query(row["query"]): row for row in eval_rows}
    lines = [
        "# Eval Feedback Review",
        "",
        f"Source log: `{source}`",
        f"CSV export: `{csv_output}`",
        f"Unique candidate queries: {len(rows)}",
        "",
        "| Query | Route | No-data reason | Route confidence | Eval CSV id |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        eval_row = eval_by_query.get(_normalize_query(str(row.get("query", ""))))
        if not eval_row:
            continue
        query = str(row.get("query", "")).replace("|", "\\|")
        route = str(row.get("route", "")).replace("|", "\\|")
        reason = str(row.get("no_data_reason", "")).replace("|", "\\|")
        confidence = str(row.get("route_confidence", ""))
        lines.append(f"| {query} | {route} | {reason} | {confidence} | {eval_row['id']} |")
    return "\n".join(lines) + "\n"


def write_feedback_report(
    source: Path = DEFAULT_SOURCE,
    report: Path = DEFAULT_REPORT,
    csv_output: Path = DEFAULT_CSV,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    rows = dedupe_feedback_candidates(load_feedback_candidates(source))
    eval_rows = write_feedback_csv(rows, csv_output)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_feedback_report(rows, eval_rows, source, csv_output), encoding="utf-8")
    return rows, eval_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Review low-confidence/no-data feedback and export eval-ready CSV rows.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()
    rows, eval_rows = write_feedback_report(args.source, args.report, args.csv_output)
    print(f"Wrote {args.report} and {args.csv_output} for {len(eval_rows)} unique feedback queries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
