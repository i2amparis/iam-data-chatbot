import json
import unittest
from collections import OrderedDict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi.testclient import TestClient

import fastapi_app
from data_utils import format_time_series_data
from manager import MultiAgentManager as ProductionManager
from model_aliases import UNLABELLED_MODEL_LABEL
from resolved_scope import ConversationState


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
        if query == "electricity for India":
            self.last_entities = {}
            self.clarification_context = {
                "clarification_id": "clarification-1",
                "entities": {"region": "IND", "action": "query"},
                "suggested_options": ["Final Energy", "Secondary Energy|Electricity"],
                "suggested_option_kinds": ["variable", "variable"],
                "suggested_kind": "variable",
            }
            return "Choose the variable: 1. `Final Energy` 2. `Secondary Energy|Electricity`"
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
        if query == "incompatible plot":
            self.last_route_decision = {
                "agent": "data_plotting",
                "confidence": 0.95,
                "source": "deterministic",
                "reason": "plot smoke test",
            }
            return (
                "I can't combine these variables on one axis because the loaded "
                "series use incompatible units: `EJ/yr`, `Mt CO2/yr`."
            )
        if query == "partial plot":
            self.last_entities = {
                "variable": "Emissions|CO2",
                "region": "World",
                "unit": "Mt CO2/yr",
                "chart_type": "line",
                "displayed_series": ["GCAM"],
                "displayed_series_count": 1,
            }
            self.last_route_decision = {
                "agent": "data_plotting",
                "confidence": 0.95,
                "source": "deterministic",
                "reason": "plot smoke test",
            }
            return (
                "Note: no timeseries data for model `Missing Model` in this slice; "
                "plotting `GCAM`.\n\n"
                "Showing Emissions|CO2 in World.\n"
                "![Plot](data:image/png;base64,ZmFrZQ==)"
            )
        if query == "CO2 for Atlantis":
            self.last_entities = {
                "action": "query",
                "variable": "Emissions|CO2",
                "unmatched_region": "Atlantis",
                "entity_confidence": {
                    "action": 0.75,
                    "variable": 0.9,
                    "region": 0.0,
                },
            }
            self.clarification_context = None
            return (
                "I couldn't find `Atlantis` as a region in the IAM PARIS data, "
                "so I can't return results for it."
            )
        return f"Smoke answer for: {query}"


class _ResolvedTableUnitManager(ProductionManager):
    """Minimal production-state manager used to exercise the API boundary."""

    _CASES = {
        "population unit": ("Population", "NGA", "EJ/yr"),
        "secondary energy unit": ("Secondary Energy", "EU", "Mt CO2/yr"),
        "solar electricity unit": (
            "Secondary Energy|Electricity|Solar", "EU", "Mt CO2/yr",
        ),
        "wind electricity unit": (
            "Secondary Energy|Electricity|Wind", "World", "Mt CO2/yr",
        ),
    }

    def __init__(self, resources, streaming=False):
        self.shared_resources = resources
        self.streaming = streaming
        self.conversation_state = ConversationState()
        self.last_result_models = []
        self.last_links = []
        self.last_route_decision = {
            "agent": "data_query",
            "confidence": 1.0,
            "source": "test",
            "reason": "resolved table unit regression",
        }
        self.turn_counter = 0
        self.current_turn = 0
        self.clarification_context = None

    def route_query(self, query, _history=None):
        variable, region, stale_unit = self._CASES[query]
        records = [
            record for record in (self.shared_resources.get("ts") or [])
            if record.get("variable") == variable and record.get("region") == region
        ]
        answer = format_time_series_data(records, variable, region)
        self._persist_last_entities(
            {
                "action": "query",
                "variable": variable,
                "region": region,
                "unit": stale_unit,
            },
            answer,
        )
        return answer


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
        self._orig_history_max_turns = fastapi_app.HISTORY_MAX_TURNS

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
        fastapi_app.HISTORY_MAX_TURNS = self._orig_history_max_turns

    def test_health_endpoint_reports_ready(self):
        client = TestClient(fastapi_app.app)

        response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ready")
        self.assertTrue(body["resources_loaded"])

    def test_numeric_model_identity_is_sanitized_at_api_boundary(self):
        manager = type("Manager", (), {
            "last_entities": {
                "model": "42",
                "models": ["42", "GCAM"],
                "result_models": ["42"],
                "displayed_series": ["42 - Baseline", "GCAM - Baseline"],
            },
            "clarification_context": None,
        })()

        entities = fastapi_app._manager_response_entities(manager)
        answer, _image, _caption, _notices = fastapi_app._split_answer_payload(
            "| Model | Scenario |\n|---|---|\n| 42 | Baseline |"
        )

        self.assertEqual(entities["model"], UNLABELLED_MODEL_LABEL)
        self.assertEqual(entities["models"], [UNLABELLED_MODEL_LABEL, "GCAM"])
        self.assertEqual(entities["result_models"], [UNLABELLED_MODEL_LABEL])
        self.assertEqual(
            entities["displayed_series"][0],
            f"{UNLABELLED_MODEL_LABEL} - Baseline",
        )
        self.assertIn(f"| {UNLABELLED_MODEL_LABEL} | Baseline |", answer)
        self.assertNotRegex(answer, r"\b42\b")

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
        self.assertIn("Open the data explorer", body["suggested_next_questions"])
        self.assertNotIn("Plot it", body["suggested_next_questions"])
        self.assertIn("matched_record_count", body["data_provenance"])

    def test_suggestions_use_scenarios_from_selected_study(self):
        manager = _ManagerStub({})
        manager.last_entities = {
            "workspace_code": "world-headed", "variable": "Emissions|CO2",
            "region": "World", "scenario": "Policy",
        }
        manager.entity_extractor = type("Extractor", (), {
            "available_scenarios": ["AFOLU_Baseline", "World Baseline", "Policy"],
        })()
        manager.shared_resources["ts"] = [
            {"workspace_code": "world-headed", "scenario": "World Baseline"},
            {"workspace_code": "world-headed", "scenario": "Policy"},
            {"workspace_code": "afolu", "scenario": "AFOLU_Baseline"},
        ]

        suggestions = fastapi_app._suggested_next_questions(
            "show emissions", "### Emissions|CO2 in World\n\nAnswer: data", manager,
        )

        self.assertIn("Compare with World Baseline", suggestions)
        self.assertNotIn("Compare with AFOLU_Baseline", suggestions)

    def test_workspace_ingestion_uses_verified_decipher_code(self):
        workspaces = fastapi_app._load_workspaces()

        self.assertIn("decipher", workspaces)
        self.assertNotIn("decipher_1", workspaces)

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

    def test_endpoint_uses_rendered_table_unit_for_scope_and_provenance(self):
        rows = [
            ("Population", "NGA", "million"),
            ("Secondary Energy", "EU", "EJ/yr"),
            ("Secondary Energy|Electricity|Solar", "EU", "EJ/yr"),
            ("Secondary Energy|Electricity|Wind", "World", "EJ/yr"),
        ]
        fastapi_app._cached_resources["ts"] = [
            {
                "variable": variable,
                "region": region,
                "scenario": "Path",
                "modelName": "Model",
                "unit": unit,
                "years": {"2050": 1},
            }
            for variable, region, unit in rows
        ]
        fastapi_app.MultiAgentManager = _ResolvedTableUnitManager
        fastapi_app._sessions = OrderedDict()
        client = TestClient(fastapi_app.app)

        expected = {
            "population unit": ("Population", "NGA", "million"),
            "secondary energy unit": ("Secondary Energy", "EU", "EJ/yr"),
            "solar electricity unit": (
                "Secondary Energy|Electricity|Solar", "EU", "EJ/yr",
            ),
            "wind electricity unit": (
                "Secondary Energy|Electricity|Wind", "World", "EJ/yr",
            ),
        }
        for query, (variable, region, unit) in expected.items():
            with self.subTest(query=query):
                body = client.post("/query", json={"query": query}).json()

                self.assertEqual(body["entities"]["variable"], variable)
                self.assertEqual(body["entities"]["region"], region)
                self.assertEqual(body["entities"]["unit"], unit)
                provenance = body["data_provenance"]
                self.assertEqual(provenance["selected_filters"]["unit"], unit)
                self.assertEqual(provenance["matched_record_count"], 1)

    def test_invalid_region_has_zero_attempted_scope_provenance_and_safe_suggestions(self):
        client = TestClient(fastapi_app.app)

        response = client.post("/query", json={"query": "CO2 for Atlantis"})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["entities"]["unmatched_region"], "Atlantis")
        provenance = body["data_provenance"]
        self.assertEqual(provenance["selected_filters"]["variable"], "Emissions|CO2")
        self.assertEqual(provenance["selected_filters"]["region"], "Atlantis")
        self.assertEqual(provenance["matched_record_count"], 0)
        self.assertEqual(
            provenance["no_data_reason"],
            "region combination unavailable",
        )
        self.assertIn("Show available regions", body["suggested_next_questions"])
        self.assertIn("Help me choose a region", body["suggested_next_questions"])
        self.assertNotIn("Plot it", body["suggested_next_questions"])
        self.assertFalse(any(
            suggestion.startswith("Compare with")
            for suggestion in body["suggested_next_questions"]
        ))

    def test_provenance_resolves_gemini_e3_alias_to_runtime_family_members(self):
        resources = {
            "ts": [
                {"variable": "GDP|MER", "region": "EU", "scenario": "Path A", "modelName": "GEMINI-E3 7.0"},
                {"variable": "GDP|MER", "region": "World", "scenario": "Path B", "modelName": "gemini_e3"},
                {"variable": "GDP|MER", "region": "World", "scenario": "Path C", "modelName": "Other"},
            ]
        }

        provenance = fastapi_app._build_data_provenance(
            resources,
            {"model": "GEMINI-E3"},
            "Model `GEMINI-E3` reports data for years 2020–2050.",
            {"agent": "data_query", "confidence": 1.0, "source": "test"},
        )

        self.assertEqual(
            provenance["selected_filters"]["models"],
            ["GEMINI-E3 7.0", "gemini_e3"],
        )
        self.assertNotIn("model", provenance["selected_filters"])
        self.assertEqual(provenance["matched_record_count"], 2)

    def test_provenance_does_not_resolve_gem_e3_to_gemini_e3(self):
        resources = {
            "ts": [
                {"variable": "GDP|MER", "region": "EU", "scenario": "Path A", "modelName": "GEMINI-E3 7.0"},
                {"variable": "GDP|MER", "region": "World", "scenario": "Path B", "modelName": "gemini_e3"},
            ]
        }

        provenance = fastapi_app._build_data_provenance(
            resources,
            {"model": "GEM-E3"},
            "I could not find any years recorded for model `GEM-E3`.",
            {"agent": "data_query", "confidence": 1.0, "source": "test"},
        )

        self.assertEqual(provenance["selected_filters"]["model"], "GEM-E3")
        self.assertNotIn("models", provenance["selected_filters"])
        self.assertEqual(provenance["matched_record_count"], 0)

    def test_numeric_model_scope_uses_neutral_label_and_preserves_count(self):
        resources = {
            "ts": [
                {
                    "variable": "Emissions|CO2", "region": "World",
                    "scenario": "Baseline", "modelName": "42",
                    "unit": "Mt CO2/yr", "years": {"2030": 1},
                },
                {
                    "variable": "Emissions|CO2", "region": "World",
                    "scenario": "Baseline", "modelName": "GCAM",
                    "unit": "Mt CO2/yr", "years": {"2030": 2},
                },
            ]
        }

        provenance = fastapi_app._build_data_provenance(
            resources,
            {
                "variable": "Emissions|CO2", "region": "World",
                "scenario": "Baseline", "result_models": ["42"],
                "start_year": 2030, "end_year": 2030,
            },
            "### Emissions|CO2 in World\n\nAnswer: available data.",
            {"agent": "data_query", "confidence": 1.0, "source": "test"},
        )

        self.assertEqual(
            provenance["selected_filters"]["models"],
            [UNLABELLED_MODEL_LABEL],
        )
        self.assertEqual(provenance["matched_record_count"], 1)
        model_rows = [
            row["value"] for row in provenance["display_rows"]
            if row["label"] in {"Model", "Models"}
        ]
        self.assertEqual(model_rows, [UNLABELLED_MODEL_LABEL])

    def test_provenance_prefers_prometheus_result_model_over_display_alias(self):
        resources = {
            "ts": [
                {
                    "variable": "Final Energy", "region": "CHN",
                    "scenario": "CP_EI", "modelName": "PROMETHEUS V1",
                    "years": {"2005": 4, "2050": 8, "2100": 12},
                },
                {
                    "variable": "Final Energy", "region": "CHN",
                    "scenario": "CP_EI", "modelName": "Other",
                    "years": {"2005": 3, "2050": 6, "2100": 9},
                },
            ]
        }

        provenance = fastapi_app._build_data_provenance(
            resources,
            {
                "variable": "Final Energy", "region": "CHN",
                "model": "PROMETHEUS", "result_models": ["PROMETHEUS V1"],
                "scenarios": ["Baseline", "CP_EI", "NDC_EI"],
                "start_year": 2005, "end_year": 2100,
            },
            "### Final Energy in CHN\n\nAnswer: available data.",
            {"agent": "data_query", "confidence": 1.0, "source": "test"},
        )

        self.assertEqual(
            provenance["selected_filters"]["models"],
            ["PROMETHEUS V1"],
        )
        self.assertNotIn("model", provenance["selected_filters"])
        self.assertEqual(
            provenance["selected_filters"]["scenarios"],
            ["Baseline", "CP_EI", "NDC_EI"],
        )
        self.assertEqual(provenance["matched_record_count"], 1)

    def test_region_data_availability_heading_is_not_treated_as_variable(self):
        resources = {
            "ts": [
                {"variable": "Metric A", "region": "GREECE", "scenario": "Path"},
                {"variable": "Metric B", "region": "GREECE", "scenario": "Path"},
                {"variable": "Metric A", "region": "EU", "scenario": "Path"},
            ]
        }

        provenance = fastapi_app._build_data_provenance(
            resources,
            {"region": "GREECE"},
            "### Data available for GREECE (Greece)\n\n- Variables: 2",
            {"agent": "data_query", "confidence": 1.0, "source": "test"},
        )

        self.assertEqual(provenance["selected_filters"], {"region": "GREECE"})
        self.assertEqual(provenance["matched_record_count"], 2)

    def test_latest_year_catalogue_answer_omits_slice_provenance(self):
        provenance = fastapi_app._build_data_provenance(
            {"ts": [{"variable": "Metric", "region": "World", "years": {"2100": 1}}]},
            {"start_year": -1, "end_year": -1},
            "The latest available projection year is **2100**.",
            {"agent": "data_query", "confidence": 1.0, "source": "test"},
        )

        self.assertEqual(provenance, {})

    def test_plot_provenance_preserves_plural_comparison_scope(self):
        resources = {
            "ts": [
                {"variable": "Metric", "region": "R1", "scenario": "Path", "modelName": "M"},
                {"variable": "Metric", "region": "R2", "scenario": "Path", "modelName": "M"},
            ]
        }
        provenance = fastapi_app._build_data_provenance(
            resources,
            {
                "variable": "Metric", "regions": ["R1", "R2"],
                "scenario": "Path", "models": ["M"],
                "comparison": "region", "comparison_dimension": "region",
                "chart_type": "line",
                "displayed_series": ["R1 - M", "R2 - M"],
                "displayed_series_count": 2,
                "omitted_series": 3,
            },
            "Showing a comparison plot.",
            {"agent": "data_plotting", "confidence": 0.9, "source": "test"},
        )

        self.assertEqual(provenance["selected_filters"]["regions"], ["R1", "R2"])
        self.assertEqual(provenance["selected_filters"]["models"], ["M"])
        self.assertEqual(provenance["matched_record_count"], 2)
        self.assertEqual(provenance["comparison_dimension"], "region")
        self.assertEqual(provenance["chart_type"], "line")
        self.assertEqual(provenance["displayed_series_count"], 2)
        self.assertEqual(provenance["omitted_series"], 3)
        self.assertEqual(provenance["displayed_series"], ["R1 - M", "R2 - M"])

    def test_record_count_understands_scenario_family_scope(self):
        resources = {
            "ts": [
                {"variable": "Metric", "region": "EU", "scenario": "PR_CurPol_CP"},
                {"variable": "Metric", "region": "EU", "scenario": "PR_CurPol_EI"},
                {"variable": "Metric", "region": "EU", "scenario": "PR_Baseline"},
            ]
        }

        count = fastapi_app._count_matching_records(
            resources,
            {"variable": "Metric", "region": "EU", "scenario": "Current Policies"},
        )

        self.assertEqual(count, 2)

    def test_record_count_respects_workspace_scope(self):
        resources = {
            "ts": [
                {
                    "workspace_code": "world-headed", "variable": "Emissions|CO2",
                    "region": "World", "scenario": "Baseline", "modelName": "GCAM",
                    "years": {"2050": 1},
                },
                {
                    "workspace_code": "afolu", "variable": "Emissions|CO2",
                    "region": "World", "scenario": "Baseline", "modelName": "GCAM",
                    "years": {"2050": 2},
                },
            ]
        }

        count = fastapi_app._count_matching_records(resources, {
            "workspace_code": "world-headed",
            "variable": "Emissions|CO2",
            "region": "World",
            "start_year": 2050,
            "end_year": 2050,
        })

        self.assertEqual(count, 1)

    def test_multi_variable_provenance_counts_all_units_without_fake_shared_unit(self):
        resources = {"ts": [
            {
                "workspace_code": "world-headed", "variable": "Emissions|CO2",
                "region": "World", "scenario": "Baseline", "modelName": "GCAM",
                "unit": "Mt CO2/yr", "years": {"2050": 100},
            },
            {
                "workspace_code": "world-headed", "variable": "Emissions|CH4",
                "region": "World", "scenario": "Baseline", "modelName": "GCAM",
                "unit": "Mt CH4/yr", "years": {"2050": 10},
            },
        ]}
        entities = {
            "workspace_code": "world-headed",
            "variables": ["Emissions|CO2", "Emissions|CH4"],
            "region": "World", "scenarios": ["Baseline"],
            "result_models": ["GCAM"], "unit": "multiple",
            "start_year": 2050, "end_year": 2050,
        }
        answer = (
            "### Emissions|CO2 in World\n\nUnit: `Mt CO2/yr`\n\n"
            "### Emissions|CH4 in World\n\nUnit: `Mt CH4/yr`"
        )

        provenance = fastapi_app._build_data_provenance(
            resources, entities, answer,
            {"agent": "data_query", "confidence": 1.0, "source": "test"},
        )

        self.assertEqual(provenance["matched_record_count"], 2)
        self.assertNotIn("unit", provenance["selected_filters"])

    def test_record_count_respects_visible_models_years_and_equivalent_units(self):
        resources = {
            "ts": [
                {
                    "variable": "Metric", "region": "R1", "scenario": "Path",
                    "modelName": "Visible A", "unit": "EJ/y", "years": {"2050": 1},
                },
                {
                    "variable": "Metric", "region": "R1", "scenario": "Path",
                    "modelName": "Visible B", "unit": "EJ/yr", "2050": 2,
                },
                {
                    "variable": "Metric", "region": "R1", "scenario": "Path",
                    "modelName": "Empty", "unit": "EJ/yr", "years": {"2040": 3},
                },
                {
                    "variable": "Metric", "region": "R1", "scenario": "Path",
                    "modelName": "Outlier", "unit": "Mt CO2/yr", "years": {"2050": 4},
                },
            ]
        }

        count = fastapi_app._count_matching_records(
            resources,
            {
                "variable": "Metric", "region": "R1", "scenario": "Path",
                "models": ["Visible A", "Visible B"], "years": "2050", "unit": "EJ/yr",
            },
        )

        self.assertEqual(count, 2)

    def test_query_endpoint_classifies_plot_validation_failure_as_no_data(self):
        client = TestClient(fastapi_app.app)

        with patch("fastapi_app._write_eval_feedback_candidate", return_value=False):
            response = client.post("/query", json={"query": "incompatible plot"})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["data_provenance"]["no_data_reason"], "incompatible units")
        self.assertEqual(fastapi_app._monitoring_counters["no_data_queries"], 1)
        self.assertIn("Show available variables", body["suggested_next_questions"])
        self.assertEqual(body["plot_base64"], "")

    def test_query_endpoint_does_not_misclassify_partial_plot_notice(self):
        client = TestClient(fastapi_app.app)

        response = client.post("/query", json={"query": "partial plot"})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["plot_base64"], "ZmFrZQ==")
        self.assertNotIn("no_data_reason", body["data_provenance"])
        self.assertEqual(fastapi_app._monitoring_counters["no_data_queries"], 0)
        self.assertNotIn("Show available variables", body["suggested_next_questions"])

    def test_record_count_normalizes_unit_whitespace_and_per_year_aliases(self):
        resources = {
            "ts": [
                {"variable": "Price|Carbon", "unit": "US$2010/tCO2/y", "2050": 1},
                {"variable": "Price|Carbon", "unit": "US$2010/t CO2/yr", "2050": 2},
                {"variable": "Price|Carbon", "unit": "US$2010/tCO2/a", "2050": 3},
            ]
        }

        count = fastapi_app._count_matching_records(
            resources,
            {"variable": "Price|Carbon", "unit": "US$2010/t CO2/yr"},
        )

        self.assertEqual(count, 3)

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

    def test_pending_clarification_exposes_resolved_base_scope(self):
        client = TestClient(fastapi_app.app)

        body = client.post("/query", json={"query": "electricity for India"}).json()

        self.assertEqual(body["entities"]["region"], "IND")
        self.assertEqual(body["clarification"]["missing_dimension"], "variable")
        self.assertEqual(body["clarification"]["base_scope"]["region"], "IND")
        self.assertEqual(
            body["clarification"]["options"][1],
            {"kind": "variable", "value": "Secondary Energy|Electricity"},
        )

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

    def test_session_history_is_capped_in_memory(self):
        fastapi_app.HISTORY_MAX_TURNS = 2
        client = TestClient(fastapi_app.app)

        first = client.post("/query", json={"query": "one"}).json()
        client.post("/query", json={"query": "two", "session_id": first["session_id"]})
        third = client.post("/query", json={"query": "three", "session_id": first["session_id"]}).json()

        self.assertEqual([turn[0] for turn in third["history"]], ["two", "three"])
        state = fastapi_app._sessions[first["session_id"]]
        self.assertEqual(len(state["chat_history"]), 2)
        self.assertIn("lock", state)

    def test_health_does_not_leak_error_details(self):
        fastapi_app._initialization_error = "Connection error: postgres://secret-host:5432"
        client = TestClient(fastapi_app.app)

        body = client.get("/health").json()

        self.assertNotIn("error", body)
        self.assertTrue(body["has_error"])
        self.assertNotIn("secret-host", json.dumps(body))


if __name__ == "__main__":
    unittest.main()
