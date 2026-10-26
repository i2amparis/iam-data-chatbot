"""Benchmark local (Ollama) models on the chatbot's real routing and answers.

Runs every model in its own process (model settings are read at import),
against the cached IAM PARIS data, and reports for each model:

- route accuracy on the eval CSVs, LLM calls per question and latency;
- general-QA answers to conceptual questions, saved for manual review.

Nothing is written to docs/ or to the runtime's feedback/monitoring files.
The knowledge index uses the app's own cache (cache/faiss_index) and is only
rebuilt when its content changed.

Usage (from the repository root, with Ollama running): by default it runs the
project's configured model (LOCAL_LLM_MODEL in .env, e.g. qwen3:0.6b-q4_K_M).

    python scripts/benchmark_local_model.py
    python scripts/benchmark_local_model.py --eval-files eval_natural_queries.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_FILES = ["eval_queries.csv", "eval_holdout_queries.csv", "eval_natural_queries.csv"]
CONCEPTUAL_QUESTIONS = [
    "What are the main uncertainties in IAM projections of net zero pathways?",
    "Why might energy transitions differ between countries?",
    "How does carbon pricing affect energy transitions?",
    "What role does carbon dioxide removal play in mitigation pathways?",
    "Why do different models give different results for the same scenario?",
    "How should I compare results from two studies?",
    "What limits how fast renewables can grow?",
    "Why does electricity demand rise in most net zero scenarios?",
]


def _worker(model: str, eval_files: list[str], output: Path) -> None:
    """Run one model in this process and write its raw results to *output*."""
    os.environ["LOCAL_LLM_MODEL"] = model
    os.environ.setdefault("LOCAL_LLM_REASONING_EFFORT", "none")
    os.environ["IAM_EVAL_FEEDBACK_ENABLED"] = "0"
    os.environ["IAM_MONITORING_STATE"] = ""
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    from dotenv import load_dotenv

    load_dotenv(override=False)
    os.environ["LOCAL_LLM_MODEL"] = model

    import pandas as pd
    from langchain_ollama import ChatOllama

    calls: list[dict] = []
    original_invoke = ChatOllama.invoke

    def timed_invoke(self, *args, **kwargs):
        started = time.time()
        try:
            return original_invoke(self, *args, **kwargs)
        finally:
            calls.append({"secs": round(time.time() - started, 2)})

    ChatOllama.invoke = timed_invoke

    import fastapi_app as app
    from agents import GeneralQAAgent
    from main import IAMParisBot, load_definitions
    from runtime_context import build_runtime_context

    bot = IAMParisBot(streaming=False)
    models = pd.read_json(max(glob.glob("cache/models*.json"), key=os.path.getsize)).to_dict("records")
    records = bot.fetch_json(
        bot.env["REST_API_FULL"],
        payload={"workspace_code": app._load_workspaces(), "limit": -1},
        cache=True,
    )
    region_docs, variable_docs = load_definitions()
    index = app._build_knowledge_index(models, region_docs, variable_docs, bot.env.get("OPENAI_API_KEY"))
    with tempfile.TemporaryDirectory() as scratch:
        context = build_runtime_context(
            models=models, ts=records, vector_store=index, env=bot.env, bot=bot,
            metadata_cache_file=os.path.join(scratch, "metadata.pkl"),
        )
    app._cached_resources = context
    app._initialization_status = "ready"
    app._warm_runtime_caches(context)

    qa = GeneralQAAgent(context, streaming=False)
    qa.handle("hello", [])  # load the model so timings exclude the cold start

    questions = []
    for path in eval_files:
        with open(path, newline="") as handle:
            questions += [dict(row, _file=path) for row in csv.DictReader(handle)]

    results = []
    for row in questions:
        before = len(calls)
        started = time.time()
        try:
            response = app._process_session_query(
                app.QueryRequest(query=row["query"]), time.monotonic() + 600,
            )
            answer, route = response.answer, response.route
        except Exception as err:  # keep benchmarking the remaining questions
            answer, route = f"ERROR {err!r}", {}
        expected = row.get("expected_route", "")
        got = route.get("agent", "")
        results.append({
            "file": row["_file"], "id": row.get("id"), "query": row["query"],
            "expected_route": expected, "route": got, "reason": route.get("reason", ""),
            "route_ok": got == expected or (expected == "data_plotting" and "data explorer" in answer.lower()),
            "secs": round(time.time() - started, 2), "llm_calls": len(calls) - before,
            "answer": answer[:1500],
        })

    conceptual = []
    for question in CONCEPTUAL_QUESTIONS:
        started = time.time()
        try:
            answer = qa.handle(question, [])
        except Exception as err:
            answer = f"ERROR {err!r}"
        conceptual.append({"query": question, "secs": round(time.time() - started, 2), "answer": answer})

    output.write_text(json.dumps({"model": model, "eval": results, "conceptual": conceptual}, indent=1))


def _summary(raw: dict) -> dict:
    results = raw["eval"]
    latencies = sorted(row["secs"] for row in results) or [0.0]
    llm_rows = [row for row in results if row["llm_calls"]]
    conceptual = [row["secs"] for row in raw["conceptual"]] or [0.0]
    return {
        "model": raw["model"],
        "questions": len(results),
        "route_ok": sum(row["route_ok"] for row in results),
        "used_llm": len(llm_rows),
        "median_secs": statistics.median(latencies),
        "p90_secs": latencies[int(0.9 * (len(latencies) - 1))],
        "conceptual_median_secs": statistics.median(conceptual),
    }


def _write_report(summaries: list[dict], raws: list[dict], output_dir: Path) -> Path:
    lines = [
        "# Local model benchmark",
        "",
        f"Generated {datetime.now().isoformat(timespec='seconds')} on the cached IAM PARIS data.",
        "",
        "| Model | Route accuracy | Questions using the LLM | Median s | p90 s | Conceptual answer median s |",
        "|---|---|---|---|---|---|",
    ]
    for item in summaries:
        lines.append(
            f"| {item['model']} | {item['route_ok']}/{item['questions']} | {item['used_llm']} | "
            f"{item['median_secs']:.2f} | {item['p90_secs']:.2f} | {item['conceptual_median_secs']:.2f} |"
        )
    lines += ["", "## Conceptual answers (review manually)", ""]
    for index, question in enumerate(CONCEPTUAL_QUESTIONS):
        lines += [f"### {question}", ""]
        for raw in raws:
            answer = raw["conceptual"][index]["answer"].replace("\n", " ")
            lines += [f"- **{raw['model']}** ({raw['conceptual'][index]['secs']} s): {answer}", ""]
    report = output_dir / "report.md"
    report.write_text("\n".join(lines) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Ollama model names. Default: the project's LOCAL_LLM_MODEL from .env.",
    )
    parser.add_argument("--eval-files", nargs="+", default=DEFAULT_EVAL_FILES)
    parser.add_argument("--output-dir", default="", help="Default: a new temporary directory.")
    parser.add_argument("--worker", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        _worker(args.worker, args.eval_files, Path(args.worker_output))
        return 0

    if not args.models:
        from dotenv import dotenv_values

        configured = (os.getenv("LOCAL_LLM_MODEL") or dotenv_values(REPO / ".env").get("LOCAL_LLM_MODEL") or "").strip()
        if not configured:
            parser.error("no --models given and LOCAL_LLM_MODEL is not set in the environment or .env")
        args.models = [configured]

    output_dir = Path(args.output_dir or tempfile.mkdtemp(prefix="iam-model-benchmark-"))
    output_dir.mkdir(parents=True, exist_ok=True)
    raws = []
    for model in args.models:
        print(f"==> {model}", flush=True)
        raw_path = output_dir / f"{model.replace(':', '_').replace('/', '_')}.json"
        completed = subprocess.run([
            sys.executable, __file__, "--models", model, "--worker", model,
            "--worker-output", str(raw_path), "--eval-files", *args.eval_files,
        ])
        if completed.returncode != 0 or not raw_path.exists():
            print(f"    failed (exit {completed.returncode}); is the model pulled in Ollama?", flush=True)
            continue
        raws.append(json.loads(raw_path.read_text()))
        summary = _summary(raws[-1])
        print(
            f"    routes {summary['route_ok']}/{summary['questions']}, LLM used on {summary['used_llm']}, "
            f"median {summary['median_secs']:.2f}s, conceptual median {summary['conceptual_median_secs']:.2f}s",
            flush=True,
        )
    if not raws:
        return 1
    report = _write_report([_summary(raw) for raw in raws], raws, output_dir)
    print(f"Report: {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
