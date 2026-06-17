import argparse
import subprocess
import sys
from dataclasses import dataclass
from typing import Sequence


UNIT_TEST_MODULES = [
    "test_model_profiles.py",
    "test_model_aliases.py",
    "test_canonical_aliases.py",
    "test_query_extractor_confidence.py",
    "test_query_normalizer.py",
    "test_link_catalog.py",
    "test_link_router.py",
    "test_runtime_context.py",
    "test_manager_fallback.py",
    "test_query_regressions.py",
    "test_fastapi_smoke.py",
    "test_run_eval.py",
    "test_validate_links.py",
    "test_data_metadata.py",
    "test_year_filters.py",
    "test_clarification_prompts.py",
    "test_feedback_review.py",
    "test_frontend_response_audit.py",
    "test_qualitative_followups.py",
    "test_quality_gate.py",
]


@dataclass
class GateCommand:
    name: str
    args: list[str]
    required: bool = True


def build_commands(
    *,
    live_url: str = "",
    include_link_validation: bool = False,
    include_static_eval: bool = True,
) -> list[GateCommand]:
    commands = [
        GateCommand(
            "unit tests",
            [sys.executable, "-m", "unittest", *UNIT_TEST_MODULES],
        )
    ]
    if include_static_eval:
        commands.extend(
            [
                GateCommand("main eval report", [sys.executable, "run_eval.py"]),
                GateCommand("holdout eval report", [sys.executable, "run_eval.py", "--holdout"]),
                GateCommand("feedback eval report", [sys.executable, "run_eval.py", "--feedback"]),
                GateCommand("conversation eval report", [sys.executable, "run_eval.py", "--conversation-eval"]),
            ]
        )
    if live_url:
        commands.extend(
            [
                GateCommand("main live eval", [sys.executable, "run_eval.py", "--live-url", live_url]),
                GateCommand("holdout live eval", [sys.executable, "run_eval.py", "--holdout", "--live-url", live_url]),
                GateCommand("feedback live eval", [sys.executable, "run_eval.py", "--feedback", "--live-url", live_url]),
                GateCommand("conversation live eval", [sys.executable, "run_eval.py", "--conversation-eval", "--live-url", live_url]),
                GateCommand(
                    "frontend response audit",
                    [sys.executable, "frontend_response_audit.py", "--live-results", "docs/evaluation_live_results.json"],
                ),
            ]
        )
    if include_link_validation:
        commands.append(
            GateCommand(
                "IAM PARIS link validation",
                [sys.executable, "validate_links.py", "--domain", "iamparis.eu"],
            )
        )
    return commands


def run_commands(commands: Sequence[GateCommand], *, runner=subprocess.run) -> int:
    failures: list[str] = []
    for command in commands:
        print(f"==> {command.name}")
        completed = runner(command.args)
        code = int(getattr(completed, "returncode", 0) or 0)
        if code != 0:
            failures.append(f"{command.name} exited with {code}")
            if command.required:
                break
    if failures:
        print("\nQuality gate failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("\nQuality gate passed.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run IAM PARIS chatbot quality gates.")
    parser.add_argument("--live-url", default="", help="Optional local FastAPI /query URL for live eval gates.")
    parser.add_argument("--skip-static-eval", action="store_true", help="Skip static Markdown eval report generation.")
    parser.add_argument("--validate-links", action="store_true", help="Validate iamparis.eu links from the generated catalog.")
    args = parser.parse_args()

    return run_commands(
        build_commands(
            live_url=args.live_url,
            include_link_validation=args.validate_links,
            include_static_eval=not args.skip_static_eval,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
