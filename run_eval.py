import argparse
import csv
import json
from pathlib import Path
from urllib import error, request


EVAL_FILE = Path("eval_queries.csv")
HOLDOUT_EVAL_FILE = Path("eval_holdout_queries.csv")
FEEDBACK_EVAL_FILE = Path("docs/eval_feedback_candidates.csv")
CONVERSATION_FILE = Path("eval_conversations.json")
RESULTS_FILE = Path("docs/evaluation_results.md")
HOLDOUT_RESULTS_FILE = Path("docs/evaluation_holdout_results.md")
FEEDBACK_RESULTS_FILE = Path("docs/evaluation_feedback_results.md")
CONVERSATION_RESULTS_FILE = Path("docs/evaluation_conversation_results.md")
LIVE_RESULTS_FILE = Path("docs/evaluation_live_results.json")
HOLDOUT_LIVE_RESULTS_FILE = Path("docs/evaluation_holdout_live_results.json")
FEEDBACK_LIVE_RESULTS_FILE = Path("docs/evaluation_feedback_live_results.json")
CONVERSATION_LIVE_RESULTS_FILE = Path("docs/evaluation_conversation_live_results.json")
REQUIRED_COLUMNS = {
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
}
MIN_EVAL_QUERIES = 150
MIN_HOLDOUT_QUERIES = 50
MIN_FEEDBACK_QUERIES = 1
MIN_CONVERSATIONS = 10
CLARIFICATION_MARKERS = (
    "which variable",
    "which model",
    "which region",
    "which scenario",
    "choose the variable",
    "choose the model",
    "choose the region",
    "choose the scenario",
    "closest valid options",
    "reply with",
    "i need one more detail",
    "confidence is low",
    "recommended variables",
    "recommended regions",
    "recommended scenarios",
    "closest variables",
    "closest regions",
    "closest scenarios",
)


def load_eval_rows(path: Path, min_queries: int = MIN_EVAL_QUERIES) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
        rows = [
            {key: (value or "").strip() for key, value in row.items()}
            for row in reader
        ]
        if len(rows) < min_queries:
            raise ValueError(
                f"Expected at least {min_queries} evaluation queries, found {len(rows)}"
            )
        return rows


def load_conversations(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    conversations = data.get("conversations", data) if isinstance(data, dict) else data
    if not isinstance(conversations, list):
        raise ValueError("Conversation eval file must contain a list or a `conversations` list.")
    if len(conversations) < MIN_CONVERSATIONS:
        raise ValueError(
            f"Expected at least {MIN_CONVERSATIONS} evaluation conversations, found {len(conversations)}"
        )

    normalized = []
    for index, conversation in enumerate(conversations, start=1):
        if not isinstance(conversation, dict):
            raise ValueError(f"Conversation {index} must be an object.")
        turns = conversation.get("turns")
        if not isinstance(turns, list) or not turns:
            raise ValueError(f"Conversation {conversation.get('id', index)} must include non-empty turns.")
        normalized_turns = []
        for turn_index, turn in enumerate(turns, start=1):
            if not isinstance(turn, dict) or not str(turn.get("query", "")).strip():
                raise ValueError(f"Conversation {conversation.get('id', index)} turn {turn_index} needs a query.")
            expected = {key: str(turn.get(key, "") or "").strip() for key in REQUIRED_COLUMNS}
            expected["id"] = str(turn.get("id") or f"{conversation.get('id', index)}.{turn_index}")
            expected["query"] = str(turn.get("query") or "").strip()
            normalized_turns.append(expected)
        normalized.append({
            "id": str(conversation.get("id") or index),
            "title": str(conversation.get("title") or f"Conversation {index}"),
            "turns": normalized_turns,
        })
    return normalized


def markdown_cell(value: str) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")


def _contains_expected(response: dict, field: str, expected: str) -> bool:
    expected = str(expected or "").strip()
    if not expected:
        return True

    expected_lower = expected.lower()
    equivalent_tokens = {
        "region": {
            "india": ("india", "ind"),
            "china": ("china", "chn"),
            "greece": ("greece", "greece", "grc"),
            "united states": ("united states", "usa", "us"),
            "eu": ("eu", "europe", "european union"),
            "world": ("world", "global"),
        },
        "scenario": {
            "current policies": ("current policies", "current policy", "pr_curpol", "curpol"),
            "ndc": ("ndc", "nationally determined"),
            "baseline": ("baseline", "bau", "business as usual"),
        },
        "model": {
            "gcam-pr 7.0": ("gcam-pr", "gcam pr", "gcampr"),
            "messageix-globiom 2.0": ("messageix-globiom", "messageix", "message ix", "message-ix"),
        },
    }
    entities = response.get("entities") or {}
    entity_values = [str(entities.get(field) or "")]
    if field == "variable":
        entity_values.extend(str(item) for item in (entities.get("variables") or []) if item)
    if field == "scenario":
        entity_values.extend(str(item) for item in (entities.get("scenarios") or []) if item)
    if field == "model":
        entity_values.extend(str(item) for item in (entities.get("models") or []) if item)
    answer = str(response.get("answer") or "")
    links = response.get("relevant_links") or []
    haystack = " ".join(
        [
            " ".join(entity_values),
            answer,
            " ".join(
                " ".join(str(link.get(key) or "") for key in ("title", "url", "reason", "search_hint"))
                for link in links
                if isinstance(link, dict)
            ),
        ]
    ).lower()
    tokens = equivalent_tokens.get(field, {}).get(expected_lower, (expected_lower,))
    return any(token in haystack for token in tokens)


def _has_useful_clarification(response: dict) -> bool:
    answer = str(response.get("answer") or "")
    answer = answer.split("Relevant IAM PARIS links:", 1)[0].lower()
    return any(marker in answer for marker in CLARIFICATION_MARKERS)


def _has_useful_link(response: dict, expected: str) -> bool:
    expected = str(expected or "").strip().lower()
    if not expected:
        return True

    links = response.get("relevant_links") or []
    if not links:
        return False

    aliases = {
        "application_library": ("application_library", "application library", "/application_library"),
        "data_stories": ("data_stories", "data stories", "data-story", "datastories", "/datastories/", "story"),
        "iam_compact": ("iam compact", "iam-compact"),
        "ndc_aspects": ("ndc aspects", "ndc-aspects"),
        "buildings": ("buildings", "building"),
        "transport": ("transport", "transportation", "mobility"),
        "afolu": ("afolu", "agriculture", "forestry", "land"),
    }
    tokens = aliases.get(expected, (expected.replace("_", " "), expected))
    for link in links:
        if not isinstance(link, dict):
            continue
        text = " ".join(str(link.get(key) or "") for key in ("title", "url", "reason", "search_hint")).lower()
        if any(token in text for token in tokens):
            return True
    return False


def _has_no_hallucinated_data(response: dict, expected: str) -> bool:
    expected = str(expected or "").strip().lower()
    if expected not in {"yes", "true", "1"}:
        return True

    answer = str(response.get("answer") or "").lower()
    if "i could not find data" in answer or "no data found" in answer:
        return True

    if response.get("plot_base64"):
        return True

    route = response.get("route") or {}
    if route.get("agent") in {"model_explanation", "general_qa"}:
        return True

    entities = response.get("entities") or {}
    return bool(entities or response.get("relevant_links"))


def score_response(row: dict[str, str], response: dict) -> dict[str, bool | str]:
    route = response.get("route") or {}
    expected_clarification = row.get("useful_clarification", "").lower() == "yes"
    has_clarification = _has_useful_clarification(response)
    return {
        "correct_route": str(route.get("agent") or "") == row.get("expected_route", ""),
        "correct_variable": _contains_expected(response, "variable", row.get("expected_variable", "")),
        "correct_region": _contains_expected(response, "region", row.get("expected_region", "")),
        "correct_scenario": _contains_expected(response, "scenario", row.get("expected_scenario", "")),
        "correct_model": _contains_expected(response, "model", row.get("expected_model", "")),
        "useful_clarification": has_clarification if expected_clarification else not has_clarification,
        "useful_link": _has_useful_link(response, row.get("useful_link", "")),
        "no_hallucinated_data": _has_no_hallucinated_data(response, row.get("no_hallucinated_data", "")),
    }


def post_query(live_url: str, query: str, session_id: str = "") -> dict:
    payload = {"query": query}
    if session_id:
        payload["session_id"] = session_id
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        live_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))


def run_live_eval(rows: list[dict[str, str]], live_url: str) -> list[dict]:
    results = []
    session_id = ""
    for row in rows:
        try:
            response = post_query(live_url, row["query"], session_id=session_id)
            session_id = response.get("session_id") or session_id
            scores = score_response(row, response)
            status = "pass" if all(value is True for value in scores.values()) else "review"
            results.append({
                "id": row.get("id", ""),
                "query": row.get("query", ""),
                "expected": row,
                "status": status,
                "scores": scores,
                "route": response.get("route") or {},
                "entities": response.get("entities") or {},
                "links": response.get("relevant_links") or [],
                "answer": str(response.get("answer") or ""),
                "answer_preview": str(response.get("answer") or "").replace("\n", " ")[:240],
            })
        except (error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            results.append({
                "id": row.get("id", ""),
                "query": row.get("query", ""),
                "expected": row,
                "status": "error",
                "scores": {},
                "error": str(exc),
            })
    return results


def run_live_conversation_eval(conversations: list[dict], live_url: str) -> list[dict]:
    results = []
    for conversation in conversations:
        session_id = ""
        turn_results = []
        for turn_index, row in enumerate(conversation.get("turns", []), start=1):
            try:
                response = post_query(live_url, row["query"], session_id=session_id)
                next_session_id = response.get("session_id") or session_id
                scores = score_response(row, response)
                scores["session_continuity"] = bool(not session_id or next_session_id == session_id)
                session_id = next_session_id
                status = "pass" if all(value is True for value in scores.values()) else "review"
                turn_results.append({
                    "id": row.get("id", ""),
                    "turn": turn_index,
                    "query": row.get("query", ""),
                    "expected": row,
                    "status": status,
                    "scores": scores,
                    "session_id": session_id,
                    "route": response.get("route") or {},
                    "entities": response.get("entities") or {},
                    "links": response.get("relevant_links") or [],
                    "answer": str(response.get("answer") or ""),
                    "answer_preview": str(response.get("answer") or "").replace("\n", " ")[:240],
                })
            except (error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
                turn_results.append({
                    "id": row.get("id", ""),
                    "turn": turn_index,
                    "query": row.get("query", ""),
                    "expected": row,
                    "status": "error",
                    "scores": {},
                    "error": str(exc),
                    "session_id": session_id,
                })
        results.append({
            "id": conversation.get("id", ""),
            "title": conversation.get("title", ""),
            "status": "pass" if turn_results and all(item.get("status") == "pass" for item in turn_results) else "review",
            "turns": turn_results,
        })
    return results


def render_conversation_results(conversations: list[dict], live_results: list[dict] | None = None) -> str:
    live_by_id = {str(item.get("id", "")): item for item in live_results or []}
    total_turns = sum(len(conversation.get("turns", [])) for conversation in conversations)
    live_turns = [
        turn
        for conversation in live_results or []
        for turn in conversation.get("turns", [])
    ]
    live_scores: dict[str, int] = {}
    for turn in live_turns:
        for key, value in (turn.get("scores") or {}).items():
            live_scores.setdefault(key, 0)
            live_scores[key] += 1 if value is True else 0

    lines = [
        "# Conversation Evaluation Results",
        "",
        "Status: live evaluation complete" if live_results else "Status: pending live/manual review",
        "",
        "This report is generated from `eval_conversations.json` and checks multi-turn session behavior.",
        "",
        "## Summary",
        "",
        f"- Total conversations: {len(conversations)}",
        f"- Total turns: {total_turns}",
    ]
    if live_results:
        passed_conversations = sum(1 for item in live_results if item.get("status") == "pass")
        passed_turns = sum(1 for item in live_turns if item.get("status") == "pass")
        lines.extend([
            f"- Live `pass` conversations: {passed_conversations}/{len(live_results)}",
            f"- Live `pass` turns: {passed_turns}/{len(live_turns)}",
            "",
            "## Live Score Summary",
            "",
        ])
        for key in sorted(live_scores):
            lines.append(f"- `{key}`: {live_scores[key]}/{len(live_turns)}")

    lines.extend([
        "",
        "## Conversation Set",
        "",
        "| Conversation | Turn | Query | Expected route | Actual route | Variable | Region | Scenario | Model | Status | Notes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ])

    for conversation in conversations:
        live_conversation = live_by_id.get(str(conversation.get("id", ""))) or {}
        live_turns_by_id = {
            str(turn.get("id", "")): turn
            for turn in live_conversation.get("turns", [])
        }
        for turn_index, row in enumerate(conversation.get("turns", []), start=1):
            live_turn = live_turns_by_id.get(str(row.get("id", ""))) or {}
            route = ((live_turn or {}).get("route") or {}).get("agent", "")
            scores = live_turn.get("scores") or {}
            failed = [key for key, value in scores.items() if value is not True]
            if live_turn.get("error"):
                notes = live_turn["error"]
            elif failed:
                notes = "Review: " + ", ".join(failed)
            elif live_turn:
                notes = (live_turn.get("answer_preview") or "")[:120]
            else:
                notes = "pending_manual_review"
            lines.append(
                "| {conversation} | {turn} | {query} | {expected_route} | {actual_route} | {variable} | {region} | {scenario} | {model} | {status} | {notes} |".format(
                    conversation=markdown_cell(f"{conversation.get('id', '')}: {conversation.get('title', '')}"),
                    turn=turn_index,
                    query=markdown_cell(row.get("query", "")),
                    expected_route=markdown_cell(row.get("expected_route", "")),
                    actual_route=markdown_cell(route),
                    variable=markdown_cell(row.get("expected_variable", "")),
                    region=markdown_cell(row.get("expected_region", "")),
                    scenario=markdown_cell(row.get("expected_scenario", "")),
                    model=markdown_cell(row.get("expected_model", "")),
                    status=markdown_cell(live_turn.get("status") or "pending_manual_review"),
                    notes=markdown_cell(notes),
                )
            )

    return "\n".join(lines) + "\n"


def conversation_eval_is_green(live_results: list[dict] | None) -> bool:
    return bool(live_results) and all(item.get("status") == "pass" for item in live_results)


def render_results(
    rows: list[dict[str, str]],
    live_results: list[dict] | None = None,
    source_file: str = "eval_queries.csv",
) -> str:
    route_counts: dict[str, int] = {}
    for row in rows:
        route = row.get("expected_route", "").strip() or "unknown"
        route_counts[route] = route_counts.get(route, 0) + 1

    live_by_id = {str(item.get("id", "")): item for item in live_results or []}
    live_scores: dict[str, int] = {}
    live_total = 0
    if live_results:
        for item in live_results:
            scores = item.get("scores") or {}
            for key, value in scores.items():
                live_scores.setdefault(key, 0)
                live_scores[key] += 1 if value is True else 0
            if scores:
                live_total += 1

    lines = [
        "# Evaluation Results",
        "",
        "Status: live evaluation complete" if live_results else "Status: pending live/manual review",
        "",
        f"This report is generated from `{source_file}`."
        + (" It includes deterministic scoring from the local FastAPI response; rows marked `review` still need human inspection." if live_results else " It records the expected coverage set and can be extended to compare live chatbot outputs."),
        "",
        "## Summary",
        "",
        f"- Total evaluation queries: {len(rows)}",
    ]
    for route, count in sorted(route_counts.items()):
        lines.append(f"- Expected `{route}` queries: {count}")

    if live_results:
        status_counts: dict[str, int] = {}
        for item in live_results:
            status = str(item.get("status") or "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        for status, count in sorted(status_counts.items()):
            lines.append(f"- Live `{status}` rows: {count}")
        if live_total:
            lines.append("")
            lines.append("## Live Score Summary")
            lines.append("")
            for key in sorted(live_scores):
                lines.append(f"- `{key}`: {live_scores[key]}/{live_total}")

    lines.extend([
        "",
        "## Tracking Fields",
        "",
        "- correct route",
        "- correct variable",
        "- correct region",
        "- correct scenario",
        "- correct model",
        "- useful clarification",
        "- useful link",
        "- no hallucinated data",
        "",
        "## Query Set",
        "",
        "| ID | Query | Expected route | Actual route | Variable | Region | Scenario | Model | Useful clarification | Useful link | No hallucinated data | Status | Notes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ])

    for row in rows:
        live = live_by_id.get(row.get("id", ""))
        scores = (live or {}).get("scores") or {}
        route = ((live or {}).get("route") or {}).get("agent", "")
        status = (live or {}).get("status") or "pending_manual_review"
        failed = [key for key, value in scores.items() if value is not True]
        notes = ""
        if live and live.get("error"):
            notes = live["error"]
        elif failed:
            notes = "Review: " + ", ".join(failed)
        elif live:
            notes = (live.get("answer_preview") or "")[:120]
        lines.append(
            "| {id} | {query} | {expected_route} | {actual_route} | {variable} | {region} | {scenario} | {model} | {clarification} | {link} | {hallucination} | {status} | {notes} |".format(
                id=markdown_cell(row.get("id", "")),
                query=markdown_cell(row.get("query", "")),
                expected_route=markdown_cell(row.get("expected_route", "")),
                actual_route=markdown_cell(route),
                variable=markdown_cell(row.get("expected_variable", "")),
                region=markdown_cell(row.get("expected_region", "")),
                scenario=markdown_cell(row.get("expected_scenario", "")),
                model=markdown_cell(row.get("expected_model", "")),
                clarification=markdown_cell(row.get("useful_clarification", "")),
                link=markdown_cell(row.get("useful_link", "")),
                hallucination=markdown_cell(row.get("no_hallucinated_data", "")),
                status=markdown_cell(status),
                notes=markdown_cell(notes),
            )
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate IAM PARIS chatbot evaluation report.")
    parser.add_argument("--eval-file", default=str(EVAL_FILE), help="CSV file with evaluation queries.")
    parser.add_argument("--holdout", action="store_true", help="Run/render the hidden holdout query set.")
    parser.add_argument("--feedback", action="store_true", help="Run/render reviewed feedback candidate queries.")
    parser.add_argument("--conversation-eval", action="store_true", help="Run/render multi-turn conversation evaluation instead of the single-query CSV.")
    parser.add_argument("--conversation-file", default=str(CONVERSATION_FILE), help="JSON file with multi-turn evaluation conversations.")
    parser.add_argument("--output", default=str(RESULTS_FILE), help="Markdown report path.")
    parser.add_argument("--live-url", default="", help="Optional FastAPI /query URL for live scoring.")
    parser.add_argument("--live-results", default="", help="Existing live-results JSON to render without calling the API.")
    parser.add_argument("--json-output", default=str(LIVE_RESULTS_FILE), help="JSON live-results path.")
    args = parser.parse_args()

    if args.conversation_eval:
        conversations = load_conversations(Path(args.conversation_file))
        live_results = None
        json_path = Path(args.json_output)
        if json_path == LIVE_RESULTS_FILE:
            json_path = CONVERSATION_LIVE_RESULTS_FILE
        if args.live_results:
            live_results = json.loads(Path(args.live_results).read_text())
        elif args.live_url:
            live_results = run_live_conversation_eval(conversations, args.live_url)
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(json.dumps(live_results, indent=2, sort_keys=True) + "\n")

        output_path = Path(args.output)
        if output_path == RESULTS_FILE:
            output_path = CONVERSATION_RESULTS_FILE
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(render_conversation_results(conversations, live_results=live_results))
        if live_results is None:
            print(f"Wrote {output_path} for {len(conversations)} conversations.")
        else:
            passed = sum(1 for item in live_results if item.get("status") == "pass")
            print(f"Wrote {output_path} and {json_path} for {len(conversations)} live conversations ({passed} pass).")
            if not conversation_eval_is_green(live_results):
                return 1
        return 0

    eval_file = Path(args.eval_file)
    min_queries = MIN_EVAL_QUERIES
    output_path = Path(args.output)
    json_output = Path(args.json_output)
    if args.holdout:
        if eval_file == EVAL_FILE:
            eval_file = HOLDOUT_EVAL_FILE
        if output_path == RESULTS_FILE:
            output_path = HOLDOUT_RESULTS_FILE
        if json_output == LIVE_RESULTS_FILE:
            json_output = HOLDOUT_LIVE_RESULTS_FILE
        min_queries = MIN_HOLDOUT_QUERIES
    if args.feedback:
        if eval_file == EVAL_FILE:
            eval_file = FEEDBACK_EVAL_FILE
        if output_path == RESULTS_FILE:
            output_path = FEEDBACK_RESULTS_FILE
        if json_output == LIVE_RESULTS_FILE:
            json_output = FEEDBACK_LIVE_RESULTS_FILE
        min_queries = MIN_FEEDBACK_QUERIES

    rows = load_eval_rows(eval_file, min_queries=min_queries)
    live_results = None
    if args.live_results:
        live_results = json.loads(Path(args.live_results).read_text())
    elif args.live_url:
        live_results = run_live_eval(rows, args.live_url)
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(json.dumps(live_results, indent=2, sort_keys=True) + "\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_results(rows, live_results=live_results, source_file=eval_file.name))
    if live_results is None:
        print(f"Wrote {output_path} for {len(rows)} queries.")
    else:
        passed = sum(1 for item in live_results if item.get("status") == "pass")
        print(f"Wrote {output_path} and {json_output} for {len(rows)} live queries ({passed} pass).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
