import json
import unittest
from collections import OrderedDict
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

import fastapi_app


class _ManagerStub:
    def __init__(self, _resources, streaming=False):
        self.streaming = streaming
        self.calls = 0
        self.shared_resources = {
            "ts": [
                {"variable": "Emissions|CO2", "region": "World", "scenario": "Baseline", "modelName": "GCAM", "unit": "Mt CO2/yr", "2030": 100},
                {"variable": "GDP|MER", "region": "World", "scenario": "Baseline", "modelName": "GCAM"},
                {"variable": "Emissions|CO2", "region": "EU", "scenario": "Baseline", "modelName": "GCAM"},
            ]
        }
        self.last_links = [
            {
                "title": "IAM PARIS Results",
                "url": "https://iamparis.eu/results",
                "reason": "Smoke test link",
                "confidence": 1.0,
                "search_hint": "",
            }
        ]
        self.last_entities = {"region": "World"}
        self.clarification_context = None
        self.last_route_decision = {
            "agent": "data_query",
            "confidence": 0.9,
            "source": "deterministic",
            "reason": "smoke test route",
        }

    def route_query(self, query, _history=None):
        self.calls += 1
        if query == "needs clarification":
            self.last_entities = {}
            self.clarification_context = {
                "suggested_options": ["Emissions|CO2"],
                "suggested_kind": "variable",
            }
            return "Choose the variable: 1. `Emissions|CO2` (CO2 emissions) Reply with a number (1-1), or `yes` for option 1."
        if query == "1":
            self.last_entities = {"variable": "Emissions|CO2", "region": "World"}
            self.clarification_context = None
            return "### Emissions|CO2 in World\n\nAnswer:\ncontinued from option 1"
        if query == "numeric answer":
            self.last_entities = {"variable": "Emissions|CO2", "region": "World", "scenario": "Baseline", "model": "GCAM"}
            return (
                "### Emissions|CO2 in World\n\n"
                "Scope: scenario `Baseline`, model `GCAM`, years `2030`\n"
                "Unit: `Mt CO2/yr`\n\n"
                "Answer:\n"
                "**GCAM - Baseline**\n"
                "| Year | Value | Unit |\n|------|-------|------|\n| 2030 | 100.00 | Mt CO2/yr |"
            )
        if query == "plot it":
            return f"plotted {self.last_entities.get('variable', 'missing')} for {self.last_entities.get('region', 'missing')}"
        return f"Smoke answer for: {query}"


class FastAPISmokeTests(unittest.TestCase):
    def setUp(self):
        self._orig_status = fastapi_app._initialization_status
        self._orig_error = fastapi_app._initialization_error
        self._orig_resources = fastapi_app._cached_resources
        self._orig_manager = fastapi_app.MultiAgentManager
        self._orig_sessions = fastapi_app._sessions
        self._orig_monitoring = dict(fastapi_app._monitoring_counters)
        self._orig_api_key = fastapi_app.API_KEY
        self._orig_rate_limit = fastapi_app.RATE_LIMIT_PER_MINUTE
        self._orig_max_sessions = fastapi_app.MAX_SESSIONS

        fastapi_app._initialization_status = "ready"
        fastapi_app._initialization_error = None
        fastapi_app._cached_resources = {
            "models": [],
            "ts": [],
            "vector_store": object(),
            "env": {},
            "bot": None,
            "link_catalog": [{"title": "IAM PARIS Results"}],
            "metadata": None,
        }
        fastapi_app._sessions = OrderedDict()
        fastapi_app._rate_buckets.clear()
        for key in fastapi_app._monitoring_counters:
            fastapi_app._monitoring_counters[key] = 0
        fastapi_app.MultiAgentManager = _ManagerStub

    def tearDown(self):
        fastapi_app._initialization_status = self._orig_status
        fastapi_app._initialization_error = self._orig_error
        fastapi_app._cached_resources = self._orig_resources
        fastapi_app.MultiAgentManager = self._orig_manager
        fastapi_app._sessions = self._orig_sessions
        fastapi_app._monitoring_counters.clear()
        fastapi_app._monitoring_counters.update(self._orig_monitoring)
        fastapi_app.API_KEY = self._orig_api_key
        fastapi_app.RATE_LIMIT_PER_MINUTE = self._orig_rate_limit
        fastapi_app.MAX_SESSIONS = self._orig_max_sessions

    def test_health_endpoint_reports_ready(self):
        client = TestClient(fastapi_app.app)

        response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ready")
        self.assertTrue(body["resources_loaded"])

    def test_query_endpoint_returns_answer_and_history(self):
        client = TestClient(fastapi_app.app)

        response = client.post("/query", json={"query": "show me CO2 emissions for World"})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["answer"], "Smoke answer for: show me CO2 emissions for World")
        self.assertTrue(body["session_id"])
        self.assertEqual(body["history"], [["show me CO2 emissions for World", body["answer"]]])
        self.assertEqual(body["plot_base64"], "")
        self.assertEqual(body["plot_caption"], "")
        self.assertEqual(body["notices"], [])
        self.assertEqual(body["relevant_links"][0]["title"], "IAM PARIS Results")
        self.assertEqual(body["relevant_links"][0]["display_label"], "Open IAM PARIS Results")
        self.assertEqual(body["relevant_links"][0]["action"], "open")
        self.assertIn("category", body["relevant_links"][0])
        self.assertIn("verified_direct_url", body["relevant_links"][0])
        self.assertEqual(body["entities"], {"region": "World"})
        self.assertEqual(body["data_scope"], {"region": "World"})
        self.assertEqual(body["route"]["agent"], "data_query")
        self.assertEqual(body["route"]["source"], "deterministic")
        self.assertEqual(body["route"]["confidence"], 0.9)
        self.assertIn("Plot it", body["suggested_next_questions"])
        self.assertIn("matched_record_count", body["data_provenance"])

    def test_application_library_link_fallback_has_search_action(self):
        links = fastapi_app._prepare_relevant_links([
            {
                "title": "Climate Watch",
                "url": "https://iamparis.eu/application_library",
                "reason": "Matched: Climate Watch",
                "confidence": 0.9,
                "search_hint": "Climate Watch",
                "category": "application_library",
                "verified_direct_url": False,
                "fallback_instruction": "Open the Application Library and search for: Climate Watch",
            }
        ])

        self.assertEqual(links[0]["action"], "search")
        self.assertEqual(links[0]["display_hint"], "Open the Application Library and search for: Climate Watch")
        self.assertFalse(links[0]["verified_direct_url"])

    def test_query_endpoint_returns_numeric_data_provenance(self):
        client = TestClient(fastapi_app.app)

        response = client.post("/query", json={"query": "numeric answer"})

        self.assertEqual(response.status_code, 200)
        provenance = response.json()["data_provenance"]
        self.assertEqual(provenance["matched_record_count"], 1)
        self.assertEqual(provenance["selected_filters"]["variable"], "Emissions|CO2")
        self.assertEqual(provenance["selected_filters"]["region"], "World")
        self.assertEqual(provenance["selected_filters"]["scenario"], "Baseline")
        self.assertEqual(provenance["selected_filters"]["model"], "GCAM")
        self.assertEqual(provenance["selected_filters"]["years"], "2030")
        self.assertEqual(provenance["selected_filters"]["unit"], "Mt CO2/yr")
        self.assertIn("cache_timestamp", provenance)
        self.assertEqual(provenance["display_title"], "Data provenance")
        self.assertTrue(any(row["label"] == "Matched records" for row in provenance["display_rows"]))

    def test_query_trace_contains_monitoring_fields(self):
        manager = _ManagerStub({}, streaming=False)
        trace = fastapi_app._build_query_trace(
            "session-1",
            "show me CO2",
            manager,
            "I could not find data for `Emissions|CO2` in `World`.",
        )

        self.assertEqual(trace["session_id"], "session-1")
        self.assertEqual(trace["query"], "show me CO2")
        self.assertEqual(trace["route"], "data_query")
        self.assertEqual(trace["route_confidence"], 0.9)
        self.assertEqual(trace["selected_region"], "World")
        self.assertEqual(trace["matched_records"], 2)
        self.assertEqual(trace["no_data_reason"], "region combination unavailable")
        self.assertEqual(trace["selected_links"], ["IAM PARIS Results"])
        self.assertEqual(trace["link_scores"], {"IAM PARIS Results": 1.0})

    def test_eval_feedback_candidate_logging_writes_jsonl_for_no_data(self):
        trace = {
            "session_id": "session-1",
            "query": "missing data",
            "route": "data_query",
            "route_confidence": 0.9,
            "entities": {"variable": "Emissions|CO2"},
            "entity_confidence": {},
            "matched_records": 0,
            "no_data_reason": "scenario combination unavailable",
        }

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "feedback.jsonl"
            wrote = fastapi_app._write_eval_feedback_candidate(trace, "No data found", log_path=path)

            self.assertTrue(wrote)
            text = path.read_text()
            self.assertIn("missing data", text)
            self.assertIn("eval_holdout_queries.csv", text)

    def test_monitoring_endpoint_reports_runtime_rates(self):
        client = TestClient(fastapi_app.app)

        response = client.post("/query", json={"query": "show me CO2 emissions for World"})
        monitoring = client.get("/monitoring")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(monitoring.status_code, 200)
        body = monitoring.json()
        self.assertEqual(body["counters"]["total_queries"], 1)
        self.assertIn("failed_route_rate", body["rates"])
        self.assertIn("thresholds", body)
        self.assertIn("alerts", body)
        self.assertEqual(body["status"], "ok")
        self.assertIn("feedback_candidates", body)

    def test_monitoring_endpoint_reports_alerts_when_thresholds_are_exceeded(self):
        fastapi_app._monitoring_counters.update({
            "total_queries": 10,
            "failed_queries": 2,
            "no_data_queries": 0,
            "low_confidence_route_queries": 0,
            "low_confidence_entity_queries": 0,
        })
        client = TestClient(fastapi_app.app)

        response = client.get("/monitoring")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "warning")
        self.assertTrue(any(alert["metric"] == "failed_route_rate" for alert in body["alerts"]))

    def test_query_endpoint_reuses_session_history(self):
        client = TestClient(fastapi_app.app)

        first = client.post("/query", json={"query": "first question"}).json()
        second = client.post(
            "/query",
            json={"query": "second question", "session_id": first["session_id"]},
        ).json()

        self.assertEqual(second["session_id"], first["session_id"])
        self.assertEqual(len(second["history"]), 2)
        self.assertEqual(second["history"][0][0], "first question")
        self.assertEqual(second["history"][1][0], "second question")

    def test_query_endpoint_can_reset_session(self):
        client = TestClient(fastapi_app.app)

        first = client.post("/query", json={"query": "first question"}).json()
        reset = client.post(
            "/query",
            json={
                "query": "fresh question",
                "session_id": first["session_id"],
                "reset_session": True,
            },
        ).json()

        self.assertEqual(reset["session_id"], first["session_id"])
        self.assertEqual(len(reset["history"]), 1)
        self.assertEqual(reset["history"][0][0], "fresh question")

    def test_session_clarification_then_number_continues(self):
        client = TestClient(fastapi_app.app)

        first = client.post("/query", json={"query": "needs clarification"}).json()
        second = client.post(
            "/query",
            json={"query": "1", "session_id": first["session_id"]},
        ).json()

        self.assertIn("Choose the variable", first["answer"])
        self.assertIn("Use the first option", first["suggested_next_questions"])
        self.assertIn("continued from option 1", second["answer"])
        self.assertEqual(second["entities"], {"variable": "Emissions|CO2", "region": "World"})
        self.assertEqual(len(second["history"]), 2)

    def test_session_plot_it_uses_previous_scope(self):
        client = TestClient(fastapi_app.app)

        first = client.post("/query", json={"query": "1"}).json()
        second = client.post(
            "/query",
            json={"query": "plot it", "session_id": first["session_id"]},
        ).json()

        self.assertEqual(second["answer"], "plotted Emissions|CO2 for World")

    def test_status_endpoint_includes_catalog_and_metadata_fields(self):
        client = TestClient(fastapi_app.app)

        response = client.get("/status")

        self.assertEqual(response.status_code, 200)
        cache = response.json()["cache"]
        self.assertEqual(cache["link_catalog_count"], 1)
        self.assertIn("metadata", cache)

    def test_query_requires_api_key_when_configured(self):
        fastapi_app.API_KEY = "secret-token"
        client = TestClient(fastapi_app.app)

        missing = client.post("/query", json={"query": "hello"})
        self.assertEqual(missing.status_code, 401)

        wrong = client.post("/query", json={"query": "hello"}, headers={"X-API-Key": "nope"})
        self.assertEqual(wrong.status_code, 401)

        ok = client.post("/query", json={"query": "hello"}, headers={"X-API-Key": "secret-token"})
        self.assertEqual(ok.status_code, 200)

    def test_protected_get_endpoints_require_api_key(self):
        fastapi_app.API_KEY = "secret-token"
        client = TestClient(fastapi_app.app)

        self.assertEqual(client.get("/status").status_code, 401)
        self.assertEqual(client.get("/monitoring").status_code, 401)
        # Public endpoints stay open.
        self.assertEqual(client.get("/health").status_code, 200)

    def test_rate_limit_returns_429_when_exceeded(self):
        fastapi_app.RATE_LIMIT_PER_MINUTE = 3
        client = TestClient(fastapi_app.app)

        statuses = [client.post("/query", json={"query": "q"}).status_code for _ in range(4)]
        self.assertEqual(statuses[:3], [200, 200, 200])
        self.assertEqual(statuses[3], 429)

    def test_query_rejects_overlong_query(self):
        client = TestClient(fastapi_app.app)

        response = client.post("/query", json={"query": "x" * 2001})
        self.assertEqual(response.status_code, 422)

    def test_sessions_are_capped_with_lru_eviction(self):
        fastapi_app.MAX_SESSIONS = 2
        client = TestClient(fastapi_app.app)

        first = client.post("/query", json={"query": "a"}).json()["session_id"]
        client.post("/query", json={"query": "b"})
        client.post("/query", json={"query": "c"})

        self.assertLessEqual(len(fastapi_app._sessions), 2)
        # The oldest session should have been evicted.
        self.assertNotIn(first, fastapi_app._sessions)

    def test_health_does_not_leak_error_details(self):
        fastapi_app._initialization_error = "Connection error: postgres://secret-host:5432"
        client = TestClient(fastapi_app.app)

        body = client.get("/health").json()

        self.assertNotIn("error", body)
        self.assertTrue(body["has_error"])
        self.assertNotIn("secret-host", json.dumps(body))


if __name__ == "__main__":
    unittest.main()
