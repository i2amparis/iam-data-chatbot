import argparse
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
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
    "test_query_plan.py",
    "test_query_regressions.py",
    "test_known_query_regressions.py",
    "test_region_equivalence.py",
    "test_plot_improvements.py",
    "test_fastapi_smoke.py",
    "test_run_eval.py",
    "test_run_smoke_queries.py",
    "test_validate_links.py",
    "test_site_navigation.py",
    "test_data_metadata.py",
    "test_year_filters.py",
    "test_clarification_prompts.py",
    "test_feedback_review.py",
    "test_frontend_response_audit.py",
    "test_qualitative_followups.py",
    "test_main_fetch.py",
    "test_operational_safety.py",
    "test_partial_latest_results.py",
    "test_quality_gate.py",
    "test_response_fixes.py",
    "test_golden_real_data.py",
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
    report_dir: str | None = None,
) -> list[GateCommand]:
    """Gate commands. ``report_dir`` redirects the static eval reports, which
    otherwise overwrite the tracked reports in ``docs/``."""

    def report(name: str) -> list[str]:
        return ["--output", str(Path(report_dir) / name)] if report_dir else []

    commands = [
        GateCommand(
            "unit tests",
            [sys.executable, "-m", "unittest", *UNIT_TEST_MODULES],
        )
    ]
    if include_static_eval:
        commands.extend(
            [
                GateCommand("main eval report", [sys.executable, "run_eval.py", *report("evaluation_results.md")]),
                GateCommand("holdout eval report", [sys.executable, "run_eval.py", "--holdout", *report("evaluation_holdout_results.md")]),
                GateCommand("feedback eval report", [sys.executable, "run_eval.py", "--feedback", *report("evaluation_feedback_results.md")]),
                GateCommand("conversation eval report", [sys.executable, "run_eval.py", "--conversation-eval", *report("evaluation_conversation_results.md")]),
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
    parser.add_argument(
        "--write-reports",
        action="store_true",
        help="Write static eval reports to docs/ (default: a temporary directory).",
    )
    args = parser.parse_args()

    # A gate run must not change tracked reports or the runtime's feedback log
    # and monitoring counters; send them to a scratch directory unless asked.
    scratch = tempfile.mkdtemp(prefix="iam-quality-gate-")
    os.environ.setdefault("IAM_EVAL_FEEDBACK_LOG", str(Path(scratch) / "eval_feedback_candidates.jsonl"))
    os.environ.setdefault("IAM_MONITORING_STATE", str(Path(scratch) / "monitoring_counters.json"))
    report_dir = None if args.write_reports else scratch
    if report_dir:
        print(f"Static eval reports: {report_dir} (use --write-reports to update docs/)")

    return run_commands(
        build_commands(
            live_url=args.live_url,
            include_link_validation=args.validate_links,
            include_static_eval=not args.skip_static_eval,
            report_dir=report_dir,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
