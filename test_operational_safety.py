import os
import subprocess
import sys
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from langchain.schema import Document

import fastapi_app
import manager as manager_module
from main import build_faiss_index, load_best_cached_results


class _SlowManager:
    def __init__(self, resources, streaming=False):
        self.shared_resources = resources
        self.last_links = []
        self.last_entities = {}
        self.last_route_decision = {}
        self.clarification_context = None

    def route_query(self, query, history=None):
        time.sleep(0.05)
        return "late answer"


class _StatefulSlowManager(_SlowManager):
    def __init__(self, resources, streaming=False):
        super().__init__(resources, streaming=streaming)
        self.last_entities = {"workspace_code": "old-study"}

    def route_query(self, query, history=None):
        if query == "switch":
            time.sleep(0.05)
            self.last_entities = {"workspace_code": "new-study"}
            return "switched"
        return self.last_entities["workspace_code"]


class OperationalSafetyTests(unittest.TestCase):
    def setUp(self):
        self.originals = {
            "status": fastapi_app._initialization_status,
            "resources": fastapi_app._cached_resources,
            "manager": fastapi_app.MultiAgentManager,
            "sessions": fastapi_app._sessions,
            "api_key": fastapi_app.API_KEY,
            "rate": fastapi_app.RATE_LIMIT_PER_MINUTE,
            "trust_proxy": fastapi_app.TRUST_PROXY,
            "timeout": fastapi_app.API_REQUEST_TIMEOUT,
            "feedback_enabled": fastapi_app.EVAL_FEEDBACK_ENABLED,
        }
        fastapi_app._initialization_status = "ready"
        fastapi_app._cached_resources = {
            "models": [], "ts": [], "env": {}, "vector_store": object(),
        }
        fastapi_app._sessions = fastapi_app.OrderedDict()
        fastapi_app._rate_buckets.clear()
        fastapi_app.API_KEY = ""
        fastapi_app.EVAL_FEEDBACK_ENABLED = False

    def tearDown(self):
        fastapi_app._initialization_status = self.originals["status"]
        fastapi_app._cached_resources = self.originals["resources"]
        fastapi_app.MultiAgentManager = self.originals["manager"]
        fastapi_app._sessions = self.originals["sessions"]
        fastapi_app.API_KEY = self.originals["api_key"]
        fastapi_app.RATE_LIMIT_PER_MINUTE = self.originals["rate"]
        fastapi_app.TRUST_PROXY = self.originals["trust_proxy"]
        fastapi_app.API_REQUEST_TIMEOUT = self.originals["timeout"]
        fastapi_app.EVAL_FEEDBACK_ENABLED = self.originals["feedback_enabled"]
        fastapi_app._rate_buckets.clear()

    def test_current_results_are_authoritative_over_historical_caches(self):
        fresh = [{"resultId": 1, "scenario": "Current"}]
        with patch("main.glob.glob", return_value=["cache/results_old.json"]), patch(
            "main.pd.read_json"
        ) as read_json:
            records, source = load_best_cached_results(fresh)

        self.assertEqual(records, fresh)
        self.assertEqual(source, "current")
        read_json.assert_not_called()

    def test_empty_current_result_is_not_repopulated_from_history(self):
        with patch("main.glob.glob", return_value=["cache/results_old.json"]), patch(
            "main.pd.read_json"
        ) as read_json:
            records, source = load_best_cached_results([])

        self.assertEqual(records, [])
        self.assertEqual(source, "current")
        read_json.assert_not_called()

    def test_manager_passes_runtime_api_key_to_entity_extractor(self):
        resources = {
            "models": [{"modelName": "Example"}],
            "ts": [],
            "env": {"OPENAI_API_KEY": "runtime-key"},
        }
        with patch.object(
            manager_module.MultiAgentManager, "_initialize_agents"
        ), patch("manager.QueryEntityExtractor") as extractor, patch("manager.ChatOpenAI"):
            manager_module.MultiAgentManager(resources, streaming=False)

        extractor.assert_called_once_with(
            models=resources["models"],
            ts_data=resources["ts"],
            api_key="runtime-key",
        )

    def test_faiss_cache_rebuilds_when_document_content_changes(self):
        embeddings = type("Embeddings", (), {"model": "test-model", "dimensions": 3})()
        first_store = MagicMock()
        changed_store = MagicMock()

        def save_index(index_dir):
            path = Path(index_dir)
            path.mkdir(parents=True, exist_ok=True)
            (path / "index.faiss").write_bytes(b"index")

        first_store.save_local.side_effect = save_index
        changed_store.save_local.side_effect = save_index
        fake_faiss = MagicMock()
        fake_faiss.from_documents.side_effect = [first_store, changed_store]
        cached_store = object()
        fake_faiss.load_local.return_value = cached_store

        original_cwd = os.getcwd()
        with TemporaryDirectory() as tmpdir, patch("main.FAISS", fake_faiss):
            try:
                os.chdir(tmpdir)
                first = build_faiss_index([Document(page_content="old", metadata={})], embeddings)
                cached = build_faiss_index([Document(page_content="old", metadata={})], embeddings)
                changed = build_faiss_index([Document(page_content="new", metadata={})], embeddings)
            finally:
                os.chdir(original_cwd)

        self.assertIs(first, first_store)
        self.assertIs(cached, cached_store)
        self.assertIs(changed, changed_store)
        self.assertEqual(fake_faiss.from_documents.call_count, 2)
        self.assertEqual(fake_faiss.load_local.call_count, 1)

    def test_feedback_write_failure_is_best_effort(self):
        trace = {"no_data_reason": "missing", "query": "test"}
        fastapi_app.EVAL_FEEDBACK_ENABLED = True
        with patch.object(Path, "open", side_effect=PermissionError("read-only")):
            wrote = fastapi_app._write_eval_feedback_candidate(
                trace, "valid answer", log_path="feedback.jsonl"
            )
        self.assertFalse(wrote)

    def test_public_monitoring_redacts_query_and_session_details(self):
        with TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "feedback.jsonl"
            log_path.write_text('{"query":"private query","session_id":"secret-session"}\n')
            with patch.dict(os.environ, {"IAM_EVAL_FEEDBACK_LOG": str(log_path)}):
                response = TestClient(fastapi_app.app).get("/monitoring")

        self.assertEqual(response.status_code, 200)
        feedback = response.json()["feedback_candidates"]
        self.assertEqual(feedback["count"], 1)
        self.assertTrue(feedback["details_redacted"])
        self.assertNotIn("recent", feedback)
        self.assertNotIn("private query", response.text)
        self.assertNotIn("secret-session", response.text)

    def test_provenance_timestamp_uses_loaded_results_source(self):
        resources = {
            "results_timestamp": "2026-07-15T14:17:27+00:00",
            "ts": [],
        }

        self.assertEqual(
            fastapi_app._latest_cache_timestamp(resources),
            "2026-07-15T14:17:27+00:00",
        )

    def test_forwarded_header_is_ignored_without_trusted_proxy(self):
        fastapi_app.MultiAgentManager = _SlowManager
        fastapi_app.RATE_LIMIT_PER_MINUTE = 1
        fastapi_app.TRUST_PROXY = False
        fastapi_app.API_REQUEST_TIMEOUT = 1
        client = TestClient(fastapi_app.app)

        first = client.post("/query", json={"query": "one"}, headers={"X-Forwarded-For": "1.1.1.1"})
        second = client.post("/query", json={"query": "two"}, headers={"X-Forwarded-For": "2.2.2.2"})

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 429)

    def test_query_deadline_returns_gateway_timeout(self):
        fastapi_app.MultiAgentManager = _SlowManager
        fastapi_app.RATE_LIMIT_PER_MINUTE = 0
        fastapi_app.API_REQUEST_TIMEOUT = 0.01

        response = TestClient(fastapi_app.app).post("/query", json={"query": "slow"})

        self.assertEqual(response.status_code, 504)
        self.assertIn("timed out", response.json()["detail"].lower())
        # Let the worker release its session lock before restoring globals.
        time.sleep(0.06)

    def test_timed_out_turn_does_not_change_follow_up_context(self):
        fastapi_app.MultiAgentManager = _StatefulSlowManager
        fastapi_app.RATE_LIMIT_PER_MINUTE = 0
        fastapi_app.API_REQUEST_TIMEOUT = 0.01
        client = TestClient(fastapi_app.app)

        timed_out = client.post(
            "/query", json={"query": "switch", "session_id": "timeout-state"}
        )
        time.sleep(0.06)
        fastapi_app.API_REQUEST_TIMEOUT = 1
        follow_up = client.post(
            "/query", json={"query": "this study", "session_id": "timeout-state"}
        )

        self.assertEqual(timed_out.status_code, 504)
        self.assertEqual(follow_up.status_code, 200)
        self.assertEqual(follow_up.json()["answer"], "old-study")
        self.assertEqual(follow_up.json()["history"], [["this study", "old-study"]])

    def test_dotenv_api_key_is_loaded_before_fastapi_configuration(self):
        project_root = Path(__file__).resolve().parent
        with TemporaryDirectory() as directory:
            Path(directory, ".env").write_text(
                "IAM_API_KEY=dotenv-secret\n"
                "OPENAI_API_KEY=test-key\n"
                "REST_MODELS_URL=https://example.test/models\n"
                "REST_API_FULL=https://example.test/results\n"
            )
            environment = dict(os.environ)
            environment.pop("IAM_API_KEY", None)
            environment["PYTHONPATH"] = str(project_root)
            completed = subprocess.run(
                [sys.executable, "-c", "import fastapi_app; print(fastapi_app.API_KEY)"],
                cwd=directory,
                env=environment,
                capture_output=True,
                text=True,
                check=True,
            )

        self.assertEqual(completed.stdout.strip(), "dotenv-secret")

    def test_response_preparation_timeout_rolls_back_context_and_history(self):
        class FastSwitch(_StatefulSlowManager):
            def route_query(self, query, history=None):
                if query == "switch":
                    self.last_entities = {"workspace_code": "new-study"}
                    return "switched"
                return self.last_entities["workspace_code"]

        fastapi_app.MultiAgentManager = FastSwitch
        fastapi_app.RATE_LIMIT_PER_MINUTE = 0
        fastapi_app.API_REQUEST_TIMEOUT = .03
        original = fastapi_app._build_data_provenance

        def slow_provenance(*args, **kwargs):
            time.sleep(.10)
            return original(*args, **kwargs)

        client = TestClient(fastapi_app.app)
        with patch.object(fastapi_app, "_build_data_provenance", side_effect=slow_provenance):
            response = client.post("/query", json={"query": "switch", "session_id": "format-timeout"})
            time.sleep(.15)
        self.assertEqual(response.status_code, 504)
        fastapi_app.API_REQUEST_TIMEOUT = 1
        follow_up = client.post("/query", json={"query": "this study", "session_id": "format-timeout"})
        self.assertEqual(follow_up.json()["answer"], "old-study")
        self.assertEqual(follow_up.json()["history"], [["this study", "old-study"]])


if __name__ == "__main__":
    unittest.main()
