"""Golden questions against the real cached IAM PARIS data.

Unit-test fixtures hold one or two studies; most response defects in
response_fixes_todo.md only appeared with the real multi-study data. This
module replays eval_golden_real_data.json through the real router with the
LLM disabled. It is skipped when no results cache is present (e.g. in CI) or
when IAM_SKIP_REAL_DATA_TESTS=1.
"""

import glob
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

GOLDEN_FILE = Path("eval_golden_real_data.json")
REDIRECT_OR_DEAD_END = (
    "plot these results",
    "don't have an active",
    "i need one more detail",
    "sorry, i encountered",
)


def _largest(pattern: str) -> str:
    files = glob.glob(pattern)
    return max(files, key=os.path.getsize) if files else ""


def _results_cache_file() -> str:
    """The largest cached results file, as loaded by the server at startup."""
    candidates = [
        path for path in glob.glob("cache/*.json")
        if not os.path.basename(path).startswith(("models", "monitoring"))
    ]
    return max(candidates, key=os.path.getsize) if candidates else ""


@unittest.skipIf(
    os.getenv("IAM_SKIP_REAL_DATA_TESTS", "").strip() == "1"
    or not _results_cache_file()
    or not _largest("cache/models*.json"),
    "real data cache not available",
)
class GoldenRealDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import pandas as pd
        from runtime_context import build_runtime_context

        models = pd.read_json(_largest("cache/models*.json")).to_dict("records")
        ts = pd.read_json(_results_cache_file()).to_dict("records")
        cls._tmp = tempfile.TemporaryDirectory()
        cls.resources = build_runtime_context(
            models=models,
            ts=ts,
            vector_store=None,
            env={"OPENAI_API_KEY": "test"},
            metadata_cache_file=str(Path(cls._tmp.name) / "metadata.pkl"),
        )
        cls.golden = json.loads(GOLDEN_FILE.read_text())

    @classmethod
    def tearDownClass(cls):
        from model_aliases import register_model_display_names

        register_model_display_names([])
        cls._tmp.cleanup()

    def _manager(self):
        from manager import MultiAgentManager

        failing_llm = MagicMock(side_effect=RuntimeError("LLM disabled in golden tests"))
        self.resources["router_llm"] = failing_llm
        with patch("manager.ChatOpenAI", return_value=failing_llm), patch(
            "query_extractor.ChatOpenAI", return_value=failing_llm
        ):
            return MultiAgentManager(self.resources, streaming=False)

    def _replay(self, turns):
        manager = self._manager()
        for turn in turns:
            started = time.monotonic()
            answer = manager.route_query(turn["query"])
            elapsed = time.monotonic() - started
            yield manager, turn, answer, elapsed

    def test_golden_conversations(self):
        import fastapi_app

        budget = float(self.golden.get("max_seconds_per_turn", 3.0))
        for conversation in self.golden["conversations"]:
            history = []
            for manager, turn, answer, elapsed in self._replay(conversation["turns"]):
                label = f"{conversation['id']}: {turn['query']}"
                history.append(turn["query"])
                with self.subTest(turn=label):
                    route = manager.last_route_decision or {}
                    if "agent" in turn:
                        self.assertEqual(route.get("agent"), turn["agent"], answer[:300])
                    if "reason" in turn:
                        self.assertEqual(route.get("reason"), turn["reason"], answer[:300])
                    for text in turn.get("contains", []):
                        self.assertIn(text, answer)
                    for text in turn.get("absent", []):
                        self.assertNotIn(text.casefold(), answer.casefold())
                    self.assertLess(elapsed, budget, f"{label} took {elapsed:.2f}s")
                if conversation.get("check_suggestions"):
                    suggestions = fastapi_app._suggested_next_questions(turn["query"], answer, manager)
                    for suggestion in suggestions:
                        follow_up = self._manager()
                        for previous in history:
                            follow_up.route_query(previous)
                        reply = follow_up.route_query(suggestion).casefold()
                        with self.subTest(turn=label, suggestion=suggestion):
                            for marker in REDIRECT_OR_DEAD_END:
                                self.assertNotIn(marker, reply)


if __name__ == "__main__":
    unittest.main()
