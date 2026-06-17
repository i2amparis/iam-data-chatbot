"""Reproducible live evaluation battery for the IAM PARIS chatbot.

Runs a fixed set of real queries (data / model descriptions / plots / links)
through the production routing path, with per-query state reset so each query is
scored independently. Reports heuristic pass/fail and latency per category and
writes a JSON artifact.

Usage:
    python scripts/eval_battery.py [--json docs/eval_battery_results.json]

Notes:
- Requires the same resources as the server (cache + a valid OPENAI_API_KEY for
  the LLM-dependent paths). Deterministic paths (data/model/plot/link-catalog)
  run even when the OpenAI account is over quota, which makes this a useful
  degradation check too.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
logging.disable(logging.CRITICAL)

import fastapi_app
from manager import MultiAgentManager

BATTERY = {
    "data": [
        "show me carbon dioxide emissions for Europe",
        "what models are available",
        "list the scenarios",
        "GDP for China in 2050",
        "primary energy in World for the net-zero scenario",
    ],
    "model_description": [
        "what is GCAM",
        "explain the REMIND model",
        "tell me about the PROMETHEUS model",
    ],
    "plot": [
        "plot CO2 emissions for World",
        "graph primary energy for Europe",
        "visualize final energy demand over time",
    ],
    "links": [
        "where can I find the IAM PARIS results",
        "give me the link to the documentation",
        "how do I access the scenario explorer",
    ],
}


def score(category, answer, links):
    a = answer or ""
    has_table = "| Year |" in a or "| Value |" in a
    has_plot = "data:image/png;base64" in a
    has_links = bool(links)
    no_data = "no data found" in a.lower() or "no data available" in a.lower()
    err = a.startswith("Sorry,") or "encountered an error" in a.lower()
    if category == "data":
        ok = (has_table or any(c.isdigit() for c in a)) and not err
    elif category == "model_description":
        ok = len(a.strip()) > 120 and not err
    elif category == "plot":
        ok = has_plot and not err
    elif category == "links":
        ok = has_links and not err
    else:
        ok = not err
    return ok, dict(table=has_table, plot=has_plot, links=len(links or []), no_data=no_data, error=err)


def main():
    parser = argparse.ArgumentParser(description="Run the live eval battery.")
    parser.add_argument("--json", default="docs/eval_battery_results.json", help="JSON output path.")
    args = parser.parse_args()

    t0 = time.time()
    fastapi_app.initialize_resources()
    res = fastapi_app._cached_resources
    if res is None:
        print(f"Initialization failed: {fastapi_app._initialization_error}")
        return 1
    print(f"Init {time.time()-t0:.1f}s | models={len(res.get('models',[]))} "
          f"ts={len(res.get('ts',[]))} links={len(res.get('link_catalog',[]))}\n")

    manager = MultiAgentManager(res, streaming=False)
    rows = []
    for category, queries in BATTERY.items():
        for q in queries:
            manager.last_entities = {}
            manager.clarification_context = None
            manager.last_links = []
            t = time.time()
            try:
                ans = manager.route_query(q, [])
            except Exception as e:
                ans = f"Sorry, EXC: {e}"
            dt = time.time() - t
            links = getattr(manager, "last_links", []) or []
            route = (getattr(manager, "last_route_decision", {}) or {}).get("agent", "?")
            ok, flags = score(category, ans, links)
            rows.append(dict(category=category, query=q, ok=ok, latency=round(dt, 2),
                             route=route, flags=flags, preview=(ans or "")[:200].replace("\n", " ")))
            print(f"[{'PASS' if ok else 'FAIL'}] {category:18s} {dt:5.1f}s route={route:16s} :: {q}")

    print("\n" + "=" * 70)
    cats = {}
    for r in rows:
        c = cats.setdefault(r["category"], dict(n=0, ok=0, lat=0.0))
        c["n"] += 1
        c["ok"] += int(r["ok"])
        c["lat"] += r["latency"]
    total_n = sum(c["n"] for c in cats.values())
    total_ok = sum(c["ok"] for c in cats.values())
    total_lat = sum(c["lat"] for c in cats.values())
    for cat, c in cats.items():
        print(f"{cat:18s} pass {c['ok']}/{c['n']}  avg latency {c['lat']/c['n']:.1f}s")
    print("-" * 70)
    print(f"{'TOTAL':18s} pass {total_ok}/{total_n} ({100*total_ok/total_n:.0f}%)  avg latency {total_lat/total_n:.1f}s")

    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": {"pass": total_ok, "total": total_n}, "rows": rows},
                              indent=2, sort_keys=True) + "\n")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
