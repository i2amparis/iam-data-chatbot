import json
import logging
import unittest
from unittest.mock import patch

from data_utils import _looks_like_capability_question, format_time_series_data
from agents import DataQueryAgent
from manager import MultiAgentManager, _looks_like_site_navigation_request
from main import _normalize_cli_query
from model_aliases import UNLABELLED_MODEL_LABEL
from model_profiles import format_model_profile_answer
from resolved_scope import record_resolved_scope


class _ExtractorStub:
    def __init__(self, entities):
        self._entities = entities

    def extract(self, _query):
        if callable(self._entities):
            return dict(self._entities(_query))
        return dict(self._entities)


class _PromptThatFails:
    class _Chain:
        def invoke(self, _payload):
            raise RuntimeError("Provider Error: router unavailable")

    def __or__(self, _other):
        return self._Chain()


class _PromptShouldNotRun:
    def __or__(self, _other):
        raise AssertionError("Router LLM should not run for deterministic routes")


class _PromptReturns:
    def __init__(self, route):
        self.route = route

    class _Response:
        def __init__(self, content):
            self.content = content

    class _Chain:
        def __init__(self, route):
            self.route = route

        def invoke(self, _payload):
            return _PromptReturns._Response(self.route)

    def __or__(self, _other):
        return self._Chain(self.route)


class _AgentStub:
    def __init__(self, response="", error=None):
        self.response = response
        self.error = error
        self.calls = 0
        self.last_query = None
        self.last_entities = None

    def handle(self, _query, _history=None):
        self.calls += 1
        self.last_query = _query
        if self.error:
            raise RuntimeError(self.error)
        return self.response

    def handle_with_entities(self, _query, _entities, _history=None):
        self.last_query = _query
        self.last_entities = dict(_entities or {})
        return self.handle(_query, _history)


class ManagerFallbackTests(unittest.TestCase):
    def _build_manager(self, entities):
        mgr = MultiAgentManager.__new__(MultiAgentManager)
        mgr.logger = logging.getLogger("ManagerFallbackTests")
        mgr.shared_resources = {"models": []}
        mgr.entity_extractor = _ExtractorStub(entities)
        mgr.routing_prompt = _PromptThatFails()
        mgr.router_llm = object()
        mgr.last_entities = {}
        mgr.clarification_context = None
        mgr.turn_counter = 0
        mgr.current_turn = 0
        return mgr

    def test_assistant_capability_question_routes_to_general_qa(self):
        mgr = self._build_manager(
            {
                "action": "query",
                "scenario": "Policy",
                "entity_confidence": {"action": 0.75, "scenario": 0.9},
            }
        )
        data_agent = _AgentStub(response="data handled")
        general_agent = _AgentStub(response="general handled")
        mgr.agents = {
            "data_query": data_agent,
            "general_qa": general_agent,
            "model_explanation": _AgentStub(response="model"),
            "data_plotting": _AgentStub(response="plot"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        response = mgr._route_single(
            "How can IAM PARIS help with climate policy research?"
        )

        self.assertEqual(response, "general handled")
        self.assertEqual(mgr.last_route_decision["agent"], "general_qa")
        self.assertEqual(
            mgr.last_route_decision["reason"], "assistant capability question"
        )
        self.assertEqual(data_agent.calls, 0)
        self.assertNotIn("Policy", json.dumps(mgr.last_entities))

    def test_data_request_is_not_mistaken_for_a_capability_question(self):
        mgr = self._build_manager({})
        data_agent = _AgentStub(response="data handled")
        general_agent = _AgentStub(response="general handled")
        mgr.agents = {
            "data_query": data_agent,
            "general_qa": general_agent,
            "model_explanation": _AgentStub(response="model"),
            "data_plotting": _AgentStub(response="plot"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        response = mgr._route_single("Show me CO2 emissions for Europe in 2050")

        self.assertEqual(response, "data handled")
        self.assertEqual(mgr.last_route_decision["agent"], "data_query")
        self.assertEqual(general_agent.calls, 0)

    def test_capability_detector_matches_general_shapes_only(self):
        for question in (
            "How can IAM PARIS help with climate policy research?",
            "What can you do?",
            "How could this platform help me explore mitigation pathways?",
        ):
            self.assertTrue(_looks_like_capability_question(question), question)

        for question in (
            "Can you help me find CO2 emissions data?",
            "Show me CO2 emissions for Europe in 2050",
            "What models are available?",
        ):
            self.assertFalse(_looks_like_capability_question(question), question)

    def test_plot_request_redirects_to_data_explorer_without_calling_plot_agent(self):
        mgr = self._build_manager({})
        plot_agent = _AgentStub(response="plot should not be rendered")
        mgr.agents = {"data_plotting": plot_agent}

        response = mgr._route_single("Plot CO2 emissions for the 1.5 TECH scenario")

        self.assertIn("I do not generate figures in the chat", response)
        self.assertIn("https://iamparis.eu/results", response)
        self.assertEqual(plot_agent.calls, 0)
        self.assertEqual(mgr.last_route_decision["reason"], "charts delegated to data explorer")

    def test_plot_request_explicit_study_overrides_previous_study_link(self):
        mgr = self._build_manager({})
        mgr.shared_resources["ts"] = [
            {"workspace_code": "world-headed", "variable": "Emissions|CO2"},
            {"workspace_code": "afolu", "variable": "Emissions|CO2"},
        ]
        mgr.last_entities = {"workspace_code": "world-headed"}
        mgr.agents = {"data_plotting": _AgentStub(response="must not render")}

        response = mgr._route_single(
            "Plot CO2 emissions in AFOLU transformation for 2050"
        )

        self.assertIn("/ndc-aspects/afolu-transformation/graphs", response)
        self.assertNotIn("/where-is-the-world-headed/graphs", response)
        self.assertEqual(mgr.last_entities, {"workspace_code": "afolu"})

    def test_partial_comparison_persists_displayed_scope(self):
        mgr = self._build_manager({})
        mgr.entity_extractor.available_scenarios = ["Baseline", "Policy"]
        mgr.last_entities = {
            "variable": "Emissions|CO2", "region": "World",
            "scenario": "Policy", "workspace_code": "world-headed",
            "start_year": 2050, "end_year": 2050,
        }
        mgr.shared_resources["ts"] = [{
            "workspace_code": "world-headed", "variable": "Emissions|CO2",
            "region": "World", "scenario": "Baseline", "modelName": "GCAM",
            "unit": "Mt CO2/yr", "years": {"2050": 100},
        }]
        mgr.agents = {"data_query": DataQueryAgent(mgr.shared_resources, streaming=False)}

        response = mgr.route_query("Compare with Baseline")

        self.assertIn("Some requested comparison members have no data", response)
        self.assertEqual(mgr.last_entities["scenario"], "Baseline")
        self.assertEqual(mgr.last_entities["scenarios"], ["Baseline"])
        self.assertNotEqual(mgr.last_attempted_entities.get("scenario"), "Policy")

    def test_numeric_followup_keeps_selected_study_without_explicit_study_reference(self):
        records = [
            {
                "workspace_code": "world-headed", "variable": "Emissions|CO2",
                "region": "World", "scenario": "Baseline", "modelName": "Model A",
                "unit": "Mt CO2/yr", "years": {"2050": 100},
            },
            {
                "workspace_code": "afolu", "variable": "Emissions|CO2",
                "region": "World", "scenario": "Other", "modelName": "Wrong study",
                "unit": "Mt CO2/yr", "years": {"2050": 999},
            },
        ]

        def entities(query):
            if "where is the world headed" in query.lower():
                return {"action": "query"}
            return {
                "action": "query", "variable": "Emissions|CO2", "region": "World",
                "start_year": 2050, "end_year": 2050,
            }

        mgr = self._build_manager(entities)
        mgr.entity_extractor.available_models = []
        mgr.entity_extractor.available_variables = ["Emissions|CO2"]
        mgr.entity_extractor.available_regions = ["World"]
        mgr.entity_extractor.available_scenarios = ["Baseline", "Other"]
        mgr.shared_resources["ts"] = records
        mgr.agents = {"data_query": DataQueryAgent(mgr.shared_resources, streaming=False)}

        mgr.route_query("Tell me about Where is the world headed?")
        response = mgr.route_query("Can you share results for Emissions|CO2 in World in 2050?")

        self.assertIn("| 2050 | 100.00 | Mt CO2/yr |", response)
        self.assertNotIn("999", response)

    def test_scenario_comparison_returns_table_and_keeps_workspace_scope(self):
        mgr = self._build_manager({})
        mgr.entity_extractor.available_models = []
        mgr.entity_extractor.available_variables = ["Emissions|CO2"]
        mgr.entity_extractor.available_regions = ["World"]
        mgr.entity_extractor.available_scenarios = ["Policy", "Baseline"]
        mgr.last_entities = {
            "action": "query",
            "variable": "Emissions|CO2",
            "region": "World",
            "scenario": "Policy",
            "workspace_code": "world-headed",
            "start_year": 2050,
            "end_year": 2050,
        }
        data_agent = _AgentStub(response="numeric comparison table")
        plot_agent = _AgentStub(response="plot should not run")
        mgr.agents = {"data_query": data_agent, "data_plotting": plot_agent}

        response = mgr._route_single("Compare with Baseline")

        self.assertEqual(response, "numeric comparison table")
        self.assertEqual(data_agent.calls, 1)
        self.assertEqual(plot_agent.calls, 0)
        self.assertEqual(data_agent.last_entities["action"], "query")
        self.assertEqual(data_agent.last_entities["workspace_code"], "world-headed")
        self.assertEqual(mgr.last_route_decision["agent"], "data_query")

    def test_results_available_uses_study_catalogue_without_router(self):
        mgr = self._build_manager({})
        mgr.shared_resources["ts"] = [
            {"workspace_code": "world-headed", "variable": "A", "scenario": "Baseline"},
            {"workspace_code": "afolu", "variable": "B", "scenario": "Policy"},
        ]
        data_agent = DataQueryAgent(mgr.shared_resources, streaming=False)
        mgr.agents = {"data_query": data_agent}

        response = mgr._route_single("What results are available?")

        self.assertIn("I found these workspaces (studies)", response)
        self.assertIn("Where is the world headed?", response)
        self.assertIn("AFOLU transformation", response)
        self.assertEqual(mgr.last_route_decision["agent"], "data_query")

    def test_scenario_comparison_real_handler_returns_bounded_numeric_table(self):
        mgr = self._build_manager({})
        mgr.entity_extractor.available_scenarios = ["Baseline", "Policy", "Other"]
        mgr.last_entities = {
            "variable": "Emissions|CO2", "region": "World", "scenario": "Policy",
            "workspace_code": "world-headed", "start_year": 2050, "end_year": 2050,
        }
        mgr.shared_resources["ts"] = [
            {"workspace_code": workspace, "variable": "Emissions|CO2", "region": "World",
             "scenario": scenario, "modelName": "GCAM", "unit": "Mt CO2/yr", "years": {"2050": value}}
            for workspace, scenario, value in (
                ("world-headed", "Baseline", 100), ("world-headed", "Policy", 80),
                ("world-headed", "Other", 777), ("afolu", "Baseline", 999),
            )
        ]
        mgr.agents = {
            "data_query": DataQueryAgent(mgr.shared_resources, streaming=False),
            "data_plotting": _AgentStub(error="Plotting must not run"),
        }
        with patch("data_utils.simple_plot_query", side_effect=AssertionError("Legacy plotter must not run")):
            response = mgr.route_query("Compare with Baseline")
        self.assertIn("| 2050 | 100.00 |", response)
        self.assertIn("| 2050 | 80.00 |", response)
        self.assertNotIn("999", response)
        self.assertNotIn("777", response)
        self.assertNotIn("![Plot]", response)
        self.assertEqual(mgr.last_entities["workspace_code"], "world-headed")

    def test_study_followup_resolves_methane_instead_of_defaulting_to_co2(self):
        records = [
            {
                "workspace_code": "world-headed", "variable": "Emissions|CH4",
                "region": "World", "scenario": "Baseline", "modelName": "Model A",
                "unit": "Mt CH4/yr", "years": {"2050": 12},
            },
        ]
        mgr = self._build_manager({"action": "query", "region": "World"})
        mgr.entity_extractor.available_models = []
        mgr.entity_extractor.available_variables = ["Emissions|CO2", "Emissions|CH4"]
        mgr.entity_extractor.available_regions = ["World"]
        mgr.entity_extractor.available_scenarios = ["Baseline"]
        mgr.shared_resources["ts"] = records
        mgr.last_entities = {"workspace_code": "world-headed"}
        mgr.agents = {"data_query": DataQueryAgent(mgr.shared_resources, streaming=False)}

        response = mgr._route_single(
            "What are global methane emissions in this study for 2050?"
        )

        self.assertIn("### Emissions|CH4 in World", response)
        self.assertIn("| 2050 | 12.00 | Mt CH4/yr |", response)
        self.assertNotIn("Which variable should I use?", response)

    def test_natural_language_study_result_with_year_is_not_navigation(self):
        self.assertFalse(_looks_like_site_navigation_request(
            "What are global CO2 emissions in AFOLU transformation for 2050?",
            [{"title": "AFOLU transformation", "url": "https://iamparis.eu/results"}],
        ))

    def test_unresolved_variable_candidates_trigger_grounded_clarification(self):
        mgr = self._build_manager({})

        response = mgr._low_confidence_entity_prompt({
            "variable": None,
            "variable_candidates": ["Candidate A", "Candidate B"],
            "unmatched_variable_terms": ["novel", "qualifier"],
        })

        self.assertIn("could not confidently match", response.lower())
        self.assertIn("`novel`", response)
        self.assertIn("1. `Candidate A`", response)

    def test_broad_electricity_clarification_prefers_generic_output_in_scope(self):
        def extract(query):
            if str(query).strip().casefold() == "electricity for r1":
                return {
                    "action": "query",
                    "region": "R1",
                    "variable_candidates": [
                        "Capacity|Electricity",
                        "Price|Electricity|Steel",
                        "Capacity|Electricity|Gas",
                    ],
                    "entity_confidence": {"action": 0.75, "region": 0.85},
                }
            return {"action": "plot"}

        mgr = self._build_manager(extract)
        mgr.entity_extractor.available_variables = [
            "Capacity|Electricity",
            "Final Energy|Electricity",
            "Price|Electricity|Steel",
            "Secondary Energy|Electricity",
        ]
        mgr.entity_extractor.available_regions = ["R1", "R2"]
        mgr.entity_extractor.available_scenarios = []
        mgr.entity_extractor.available_models = []
        mgr.entity_extractor.ts_data = [
            {"region": "R1", "variable": variable}
            for variable in mgr.entity_extractor.available_variables
        ]
        data_agent = _AgentStub(response="data handled")
        plot_agent = _AgentStub(response="plot handled")
        mgr.agents = {
            "data_query": data_agent,
            "data_plotting": plot_agent,
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        prompt = mgr._route_single("electricity for R1")
        resolved = mgr._route_single("1")
        plotted = mgr._route_single("plot it")

        self.assertIn("1. `Secondary Energy|Electricity`", prompt)
        self.assertEqual(resolved, "data handled")
        self.assertEqual(
            data_agent.last_entities["variable"],
            "Secondary Energy|Electricity",
        )
        self.assertIn("I do not generate figures in the chat", plotted)
        self.assertEqual(plot_agent.calls, 0)
        self.assertEqual(mgr.last_entities["variable"], "Secondary Energy|Electricity")
        self.assertEqual(mgr.last_entities["region"], "R1")

    def test_oil_demand_clarification_only_offers_final_liquid_energy(self):
        mgr = self._build_manager({})
        mgr.entity_extractor.available_variables = [
            "Capacity|Oil",
            "Price|Primary Energy|Oil",
            "Primary Energy|Oil",
            "Final Energy|Liquids|Bioenergy",
            "Final Energy|Industry|Liquids",
            "Final Energy|Transportation|Liquids",
            "Final Energy|Liquids|Fossil",
        ]
        mgr.entity_extractor.ts_data = [
            {"region": "R1", "variable": variable}
            for variable in mgr.entity_extractor.available_variables
        ]
        entities = {
            "region": "R1",
            "variable_candidates": [
                "Capacity|Oil",
                "Price|Primary Energy|Oil",
                "Primary Energy|Oil",
            ],
            "unmatched_variable_terms": ["oil"],
        }

        response = mgr._low_confidence_entity_prompt(
            entities,
            "oil demand for R1",
        )

        self.assertIn("1. `Final Energy|Liquids|Fossil`", response)
        self.assertIn("2. `Final Energy|Transportation|Liquids`", response)
        self.assertIn("3. `Final Energy|Industry|Liquids`", response)
        self.assertNotIn("`Capacity|Oil`", response)
        self.assertNotIn("`Price|Primary Energy|Oil`", response)
        self.assertNotIn("`Primary Energy|Oil`", response)

    def test_rejected_oil_variable_drops_its_orphaned_unit(self):
        mgr = self._build_manager({
            "action": "query",
            "variable": None,
            "variable_candidates": [
                "Primary Energy|Oil",
                "Primary Energy|Oil|Electricity",
                "Primary Energy|Oil|Hydrogen",
            ],
            "unit": "BEUR",
            "entity_confidence": {"action": 0.9, "variable": 0.35},
        })
        mgr.entity_extractor.available_variables = [
            "Capacity|Electricity|Oil",
            "Primary Energy|Oil",
            "Primary Energy|Oil|Electricity",
            "Primary Energy|Oil|Hydrogen",
        ]
        mgr.entity_extractor.available_regions = []
        mgr.entity_extractor.available_scenarios = []
        mgr.entity_extractor.available_models = []
        mgr.entity_extractor.variable_units = {
            "Capacity|Electricity|Oil": "BEUR",
            "Primary Energy|Oil": "EJ/yr",
        }
        mgr.entity_extractor.ts_data = [
            {"variable": variable}
            for variable in mgr.entity_extractor.available_variables
        ]
        mgr.agents = {
            "data_query": _AgentStub(response="must clarify"),
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        response = mgr._route_single("oil")

        self.assertIn("1. `Primary Energy|Oil`", response)
        self.assertNotIn("BEUR", response)
        self.assertNotIn("unit", mgr.clarification_context["entities"])

    def test_bare_emissions_clarifies_region_and_keeps_resolved_co2_scope(self):
        mgr = self._build_manager({
            "action": "query",
            "variable": "Emissions|CO2",
            "entity_confidence": {"action": 0.75, "variable": 0.9},
        })
        mgr.entity_extractor.available_variables = ["Emissions|CO2"]
        mgr.entity_extractor.available_regions = ["EU", "CHN", "World", "IND"]
        mgr.entity_extractor.available_scenarios = ["Baseline"]
        mgr.entity_extractor.available_models = []
        data_agent = _AgentStub(response="must not render an unscoped table")
        mgr.agents = {
            "data_query": data_agent,
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        response = mgr._route_single("emissions")

        self.assertIn("treated bare **emissions** as `Emissions|CO2`", response)
        self.assertIn("Choose the region:", response)
        self.assertIn("1. `World`", response)
        self.assertEqual(data_agent.calls, 0)
        payload = mgr.pending_clarification_payload()
        self.assertEqual(payload["missing_dimension"], "region")
        self.assertEqual(payload["base_scope"]["variable"], "Emissions|CO2")
        self.assertEqual(
            [option["value"] for option in payload["options"]],
            ["World", "EU", "CHN"],
        )

    def test_invalid_region_is_turn_visible_without_replacing_successful_scope(self):
        mgr = self._build_manager({
            "action": "query",
            "variable": "Emissions|CO2",
            "unmatched_region": "Atlantis",
            "entity_confidence": {"action": 0.75, "variable": 0.9, "region": 0.0},
        })
        mgr.entity_extractor.available_variables = ["Emissions|CO2"]
        mgr.entity_extractor.available_regions = ["World", "EU"]
        mgr.entity_extractor.available_scenarios = ["Baseline"]
        mgr.entity_extractor.available_models = []
        mgr.last_entities = {
            "action": "query", "variable": "Population", "region": "World",
        }
        data_agent = _AgentStub(response=(
            "I couldn't find `Atlantis` as a region in the IAM PARIS data, "
            "so I can't return results for it."
        ))
        mgr.agents = {
            "data_query": data_agent,
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        mgr._route_single("CO2 for Atlantis")

        self.assertEqual(mgr.last_entities["variable"], "Population")
        self.assertEqual(mgr.last_entities["region"], "World")
        self.assertEqual(mgr.last_attempted_entities["unmatched_region"], "Atlantis")
        self.assertNotIn("region", mgr.last_attempted_entities)
        self.assertEqual(mgr.response_entities()["unmatched_region"], "Atlantis")
        self.assertEqual(mgr.response_entities()["variable"], "Emissions|CO2")
        self.assertNotIn("region", mgr.response_entities())

    def test_explicit_unknown_region_reaches_data_agent_without_variable_prompt(self):
        mgr = self._build_manager({
            "action": "query",
            "variable": "Emissions|CO2",
            "variable_candidates": ["Emissions|CO2"],
            "unmatched_region": "Atlantis",
            "entity_confidence": {"variable": 0.95, "region": 0.0},
        })
        mgr.entity_extractor.available_variables = ["Emissions|CO2"]
        mgr.entity_extractor.available_regions = ["World"]
        mgr.entity_extractor.available_scenarios = ["Path"]
        mgr.entity_extractor.available_models = []
        data_agent = _AgentStub(response="unknown region handled")
        mgr.agents = {
            "data_query": data_agent,
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        response = mgr._route_single("CO2 for Atlantis")

        self.assertEqual(response, "unknown region handled")
        self.assertEqual(data_agent.last_entities["variable"], "Emissions|CO2")
        self.assertEqual(data_agent.last_entities["unmatched_region"], "Atlantis")

    def test_referenced_model_links_use_carried_plural_models(self):
        mgr = self._build_manager({})
        mgr.shared_resources["link_catalog"] = [{
            "title": "Models", "url": "https://example.test/models",
            "category": "models", "keywords": ["model", "models"],
            "verified_direct_url": True,
        }]
        mgr.last_entities = {"models": ["Model A", "Model B"]}
        mgr.agents = {}

        response = mgr._route_single("Give me links for these models")

        self.assertIn("Model A, Model B", response)
        self.assertIn("https://example.test/models", response)
        self.assertEqual(mgr.last_route_decision["source"], "conversation_state")

    def test_both_of_them_resolves_carried_model_links(self):
        mgr = self._build_manager({})
        mgr.shared_resources["link_catalog"] = [{
            "title": "Models", "url": "https://example.test/models",
            "category": "models", "keywords": ["model", "models"],
            "verified_direct_url": True,
        }]
        mgr.last_entities = {"models": ["Model A", "Model B"]}
        mgr.agents = {}

        response = mgr._route_single("Where can I read more about both of them?")

        self.assertIn("Model A, Model B", response)
        self.assertIn("https://example.test/models", response)

    def test_qualitative_followup_stays_bounded_to_referenced_model_pair(self):
        mgr = self._build_manager({})
        mgr.entity_extractor.available_models = ["Alpha 1.0", "Beta 2.0", "Gamma 3.0"]
        mgr.entity_extractor.available_variables = ["Metric"]
        mgr.entity_extractor.available_regions = ["World"]
        mgr.entity_extractor.available_scenarios = ["Path"]
        mgr.shared_resources = {
            "models": [
                {
                    "modelName": "Alpha 1.0",
                    "description": "Alpha catalogue description.",
                    "model_type": "Simulation framework",
                },
                {
                    "modelName": "Beta 2.0",
                    "description": "Beta catalogue description.",
                    "model_type": "Linear optimization framework",
                },
                {
                    "modelName": "Gamma 3.0",
                    "description": "Gamma catalogue description.",
                    "model_type": "Unrelated framework",
                },
            ],
            "link_catalog": [{
                "title": "Models",
                "url": "https://example.test/models",
                "category": "models",
                "item_type": "route",
                "keywords": ["Model documentation directory"],
                "verified_direct_url": True,
            }],
        }
        mgr.last_entities = {"models": ["Alpha 1.0", "Beta 2.0"]}
        data_agent = _AgentStub(response="global model list")
        mgr.agents = {"data_query": data_agent}

        response = mgr._route_single(
            "Which of these two models uses an optimization approach?"
        )

        self.assertIn("Alpha 1.0", response)
        self.assertIn("Simulation framework", response)
        self.assertIn("Beta 2.0", response)
        self.assertIn("Linear optimization framework", response)
        self.assertNotIn("Gamma 3.0", response)
        self.assertEqual(data_agent.calls, 0)
        self.assertEqual(mgr.last_entities["models"], ["Alpha 1.0", "Beta 2.0"])
        self.assertEqual(mgr.last_route_decision["source"], "conversation_state")
        self.assertEqual(
            mgr.last_route_decision["reason"],
            "referenced-model qualitative comparison",
        )

        pronoun_response = mgr._route_single("How do they differ in methodology?")

        self.assertIn("Simulation framework", pronoun_response)
        self.assertIn("Linear optimization framework", pronoun_response)
        self.assertNotIn("Gamma 3.0", pronoun_response)
        self.assertEqual(data_agent.calls, 0)
        self.assertEqual(mgr.last_entities["models"], ["Alpha 1.0", "Beta 2.0"])
        self.assertEqual(mgr.last_route_decision["source"], "conversation_state")

        links_response = mgr._route_single("Give me documentation links for both models")

        self.assertIn("Alpha 1.0, Beta 2.0", links_response)
        self.assertIn("https://example.test/models", links_response)
        self.assertEqual(mgr.last_entities["models"], ["Alpha 1.0", "Beta 2.0"])

    def test_no_context_plural_model_link_reference_asks_for_models(self):
        mgr = self._build_manager({})
        mgr.shared_resources["link_catalog"] = [{
            "title": "Models", "url": "https://example.test/models",
            "category": "models", "item_type": "route",
            "keywords": ["model", "models"], "verified_direct_url": True,
        }]
        mgr.agents = {}

        response = mgr._route_single("Give me documentation links for both models")

        self.assertIn("do not have a referenced set of two models", response)
        self.assertIn("Name or compare the models first", response)
        self.assertNotIn("https://example.test/models", response)
        self.assertEqual(mgr.last_route_decision["reason"], "missing referenced model set")

    def test_singular_pronoun_compares_carried_and_new_model(self):
        mgr = self._build_manager({"model": "GCAM"})
        mgr.last_entities = {"model": "WITCH"}
        mgr.shared_resources["link_catalog"] = []
        mgr.agents = {}

        response = mgr._route_single("How does it compare with GCAM in sector coverage?")

        self.assertIn("WITCH", response)
        self.assertIn("GCAM", response)
        self.assertEqual(mgr.last_entities["models"], ["WITCH", "GCAM"])
        self.assertEqual(mgr.last_route_decision["source"], "conversation_state")

    def test_singular_pronoun_comparison_keeps_runtime_only_models(self):
        mgr = self._build_manager({"model": "Alpha"})
        mgr.entity_extractor.available_models = ["Alpha", "Beta"]
        mgr.entity_extractor.available_variables = []
        mgr.last_entities = {"model": "Alpha"}
        mgr.shared_resources = {
            "models": [
                {"modelName": "Alpha", "description": "Alpha runtime description."},
                {"modelName": "Beta", "description": "Beta runtime description."},
            ],
            "link_catalog": [],
        }
        mgr.agents = {}

        response = mgr._route_single("How does it differ from Beta?")

        self.assertIn("Alpha", response)
        self.assertIn("Beta", response)
        self.assertEqual(mgr.last_entities["models"], ["Alpha", "Beta"])
        self.assertEqual(mgr.last_route_decision["source"], "conversation_state")

    def test_singular_model_metadata_supports_deictic_and_elliptical_followups(self):
        mgr = self._build_manager({})
        mgr.entity_extractor.available_models = ["Alpha 1.0"]
        mgr.entity_extractor.available_variables = []
        mgr.shared_resources = {
            "models": [{
                "modelName": "Alpha 1.0",
                "description": "Runtime model description.",
                "institute": "Example Institute",
            }],
            "link_catalog": [],
        }
        mgr.last_entities = {"model": "Alpha 1.0"}
        mgr.last_route_decision = {"agent": "model_explanation"}
        mgr.agents = {}

        developer = mgr._route_single("Who develops this model?")
        limitations = mgr._route_single("What limitations should I keep in mind?")

        self.assertIn("Example Institute", developer)
        self.assertIn("Interpretation notes", limitations)
        self.assertEqual(mgr.last_entities["model"], "Alpha 1.0")
        self.assertEqual(mgr.last_route_decision["agent"], "model_explanation")

    def test_typed_qualitative_model_followup_supersedes_stale_data_clarification(self):
        mgr = self._build_manager({})
        mgr.entity_extractor.available_models = ["Alpha 1.0", "Beta 2.0"]
        mgr.entity_extractor.available_variables = ["Metric"]
        mgr.shared_resources = {
            "models": [
                {"modelName": "Alpha 1.0", "model_type": "Simulation method"},
                {"modelName": "Beta 2.0", "model_type": "Optimization method"},
            ],
            "link_catalog": [],
        }
        mgr.last_entities = {"models": ["Alpha 1.0", "Beta 2.0"]}
        mgr.clarification_context = {
            "agent_type": "data_query",
            "entities": {"region": "R1"},
            "suggested_kind": "variable",
            "response": "Which variable should I use?",
            "issued_turn": 0,
        }
        data_agent = _AgentStub(response="wrong route")
        mgr.agents = {"data_query": data_agent}

        response = mgr._route_single("How do these two differ in methodology?")

        self.assertIn("Simulation method", response)
        self.assertIn("Optimization method", response)
        self.assertEqual(data_agent.calls, 0)
        self.assertIsNone(mgr.clarification_context)
        self.assertEqual(mgr.last_entities["models"], ["Alpha 1.0", "Beta 2.0"])

    def test_runtime_methodology_fills_missing_curated_profile_field(self):
        mgr = self._build_manager({})
        mgr.entity_extractor.available_models = ["Alpha", "Beta"]
        mgr.entity_extractor.available_variables = []
        mgr.shared_resources = {
            "models": [
                {"modelName": "Alpha", "model_type": "Simulation method"},
                {"modelName": "Beta", "model_type": "Optimization method"},
            ],
        }
        curated = {
            "Alpha": {"name": "Alpha", "aliases": ["alpha"], "description": "Alpha summary."},
            "Beta": {"name": "Beta", "aliases": ["beta"], "description": "Beta summary."},
        }

        with patch.dict("model_profiles.CURATED_MODEL_PROFILES", curated, clear=True):
            profiles = mgr._runtime_model_profiles("Compare Alpha with Beta")

        self.assertEqual([profile["name"] for profile in profiles], ["Alpha", "Beta"])
        self.assertEqual(profiles[0]["methodology_note"], "Simulation method")
        self.assertEqual(profiles[1]["methodology_note"], "Optimization method")

    def test_runtime_profile_preserves_model_id_for_specific_documentation_link(self):
        mgr = self._build_manager({})
        mgr.entity_extractor.available_models = ["ALADIN"]
        mgr.entity_extractor.available_variables = []
        mgr.shared_resources = {
            "models": [
                {
                    "id": 6,
                    "modelName": "ALADIN",
                    "description": "Alternative automobiles diffusion model.",
                },
            ],
            "link_catalog": [],
        }

        profiles = mgr._runtime_model_profiles("Can you provide more details for ALADIN?")
        response = format_model_profile_answer(profiles[0])

        self.assertIn("[IAM PARIS model page](https://iamparis.eu/models/6)", response)
        self.assertNotIn("[IAM PARIS Models](https://iamparis.eu/models)", response)

    def test_bare_comparison_resolves_versioned_runtime_model_family(self):
        mgr = self._build_manager({"model": "Alpha 1.0"})
        mgr.entity_extractor.available_models = ["Alpha 1.0", "Beta 2.0"]
        mgr.entity_extractor.available_variables = []
        mgr.last_entities = {"model": "Alpha 1.0"}
        mgr.shared_resources = {
            "models": [
                {"modelName": "Alpha 1.0", "description": "Alpha runtime description."},
                {"modelName": "Beta 2.0", "description": "Beta runtime description."},
            ],
            "link_catalog": [],
        }
        mgr.agents = {}

        response = mgr._route_single("How does it compare with Beta?")

        self.assertIn("Alpha 1.0", response)
        self.assertIn("Beta 2.0", response)
        self.assertEqual(mgr.last_entities["models"], ["Alpha 1.0", "Beta 2.0"])
        self.assertEqual(mgr.last_route_decision["source"], "conversation_state")

    def test_runtime_models_support_qualitative_technology_comparison(self):
        mgr = self._build_manager({})
        mgr.entity_extractor.available_models = ["Alpha 1.0", "Beta 2.0"]
        mgr.entity_extractor.available_variables = []
        mgr.shared_resources = {
            "models": [
                {
                    "modelName": "Alpha 1.0",
                    "description": "Alpha is a systems model.",
                    "measures_technologies": "Represents technologies using detailed engineering choices.",
                },
                {
                    "modelName": "Beta 2.0",
                    "description": "Beta is an economy model.",
                    "measures_technologies": "Represents technologies through aggregate production functions.",
                },
            ],
            "link_catalog": [],
        }
        mgr.agents = {}

        response = mgr._route_single(
            "Compare Alpha 1.0 with Beta 2.0 in their treatment of technologies."
        )

        self.assertIn("Alpha 1.0", response)
        self.assertIn("Beta 2.0", response)
        self.assertIn("detailed engineering choices", response)
        self.assertIn("aggregate production functions", response)
        self.assertEqual(mgr.last_route_decision["agent"], "model_explanation")
        self.assertEqual(mgr.last_entities["models"], ["Alpha 1.0", "Beta 2.0"])

    def test_named_model_method_question_returns_focused_grounded_answer(self):
        mgr = self._build_manager({})
        mgr.entity_extractor.available_models = []
        mgr.entity_extractor.available_variables = []
        mgr.shared_resources = {"models": [], "link_catalog": []}
        mgr.agents = {}

        response = mgr._route_single("Is REMIND a general equilibrium model?")

        self.assertIn(
            "does not state whether `REMIND` is a general equilibrium model",
            response,
        )
        self.assertNotIn("Description:", response)
        self.assertEqual(mgr.last_route_decision["agent"], "model_explanation")
        self.assertEqual(mgr.last_entities["model"], "REMIND")

    def test_named_model_technology_question_returns_only_requested_dimension(self):
        mgr = self._build_manager({})
        mgr.entity_extractor.available_models = []
        mgr.entity_extractor.available_variables = []
        mgr.shared_resources = {"models": [], "link_catalog": []}
        mgr.agents = {}

        response = mgr._route_single("How does REMIND handle technological change?")

        self.assertIn("Technology information in the loaded profile:", response)
        self.assertIn("does not provide a more specific mechanism", response)
        self.assertNotIn("Model scope:", response)
        self.assertNotIn("Useful for:", response)
        self.assertEqual(mgr.last_route_decision["agent"], "model_explanation")

    def test_image_developer_question_uses_grounded_profile_not_fuzzy_model(self):
        mgr = self._build_manager({"model": "TIAM_Grantham"})
        mgr.entity_extractor.available_models = ["TIAM_Grantham", "MANAGE"]
        mgr.entity_extractor.available_variables = []
        mgr.shared_resources = {
            "models": [
                {"modelName": "TIAM_Grantham", "institute": "Unrelated Institute"},
                {"modelName": "MANAGE", "institute": "Another Institute"},
            ],
            "link_catalog": [],
        }
        mgr.agents = {}

        response = mgr._route_single("Who develops the IMAGE model?")

        self.assertIn("IMAGE team", response)
        self.assertIn("PBL Netherlands Environmental Assessment Agency", response)
        self.assertNotIn("TIAM", response)
        self.assertNotIn("MANAGE", response)
        self.assertEqual(mgr.last_entities["model"], "IMAGE")
        self.assertEqual(mgr.last_route_decision["agent"], "model_explanation")

    def test_integrated_assessment_model_definition_bypasses_navigation_and_aliases(self):
        mgr = self._build_manager({"model": "TIAM_Grantham"})
        mgr.entity_extractor.available_models = ["TIAM_Grantham", "MANAGE"]
        mgr.entity_extractor.available_variables = []
        mgr.shared_resources = {
            "models": [{"modelName": "TIAM_Grantham"}],
            "link_catalog": [{
                "title": "Models",
                "url": "https://iamparis.eu/models",
                "category": "models",
                "keywords": ["model", "models", "integrated assessment"],
                "verified_direct_url": True,
            }],
        }
        mgr.agents = {
            "general_qa": _AgentStub(response="wrong LLM answer"),
            "model_explanation": _AgentStub(response="wrong model answer"),
        }

        response = mgr._route_single("What is an integrated assessment model?")

        self.assertIn("### Integrated assessment model (IAM)", response)
        self.assertIn("combines knowledge from two or more domains", response)

    def test_generic_model_class_and_scenario_assumption_questions_are_direct(self):
        mgr = self._build_manager({})
        mgr.entity_extractor.available_models = []
        mgr.entity_extractor.available_variables = []
        mgr.shared_resources = {"models": [], "link_catalog": []}
        mgr.agents = {}

        comparison = mgr._route_single(
            "What is the difference between an IAM and an energy-system model?"
        )
        assumptions = mgr._route_single(
            "Explain how scenario assumptions affect IAM outputs."
        )

        self.assertIn("breadth versus energy-detail", comparison)
        self.assertNotIn("cannot resolve", comparison.lower())
        self.assertIn("conditional pathways", assumptions)
        self.assertNotIn("specify the variable", assumptions.lower())

    def test_overview_purpose_and_policy_use_questions_use_profiles(self):
        mgr = self._build_manager({})
        mgr.entity_extractor.available_models = ["REMIND-MAgPIE", "GCAM-PR", "E3ME-FTT"]
        mgr.entity_extractor.available_variables = ["Land Cover"]
        mgr.shared_resources = {"models": [], "link_catalog": []}
        mgr.agents = {}

        overview = mgr._route_single("Give me an overview of REMIND-MAgPIE.")
        purpose = mgr._route_single("What problems is GCAM-PR designed to study?")
        policy = mgr._route_single("What kinds of policy questions can E3ME answer?")

        self.assertIn("### REMIND-MAgPIE", overview)
        self.assertIn("Common uses described", purpose)
        self.assertIn("Puerto Rico regional transition pathways", purpose)
        self.assertIn("Common uses described", policy)
        self.assertIn("economy-energy-environment policy analysis", policy)

    def test_runtime_only_models_and_negative_comparison_clause_keep_identity(self):
        mgr = self._build_manager({})
        mgr.entity_extractor.available_models = []
        mgr.entity_extractor.available_variables = []
        mgr.shared_resources = {
            "models": [
                {
                    "modelName": "MANAGE",
                    "description": "MANAGE is used for economy-wide energy and climate policy analysis.",
                    "institute": "National Technical University of Athens",
                },
                {
                    "modelName": "OSeMOSYS",
                    "description": "An open-source energy-system framework for long-term energy planning.",
                    "institute": "KTH Royal Institute of Technology",
                },
                {"modelName": "IMAGE", "description": "Wrong model."},
            ],
            "link_catalog": [],
        }
        mgr.agents = {}

        manage = mgr._route_single(
            "Describe the MANAGE model without confusing it with IMAGE."
        )
        osemosys = mgr._route_single("Tell me what OSeMOSYS is intended for.")

        self.assertIn("### MANAGE", manage)
        self.assertIn("economy-wide energy and climate policy", manage)
        self.assertNotIn("### IMAGE", manage)
        self.assertIn("### OSeMOSYS", osemosys)
        self.assertIn("long-term energy planning", osemosys)

    def test_compound_coverage_and_hyphenated_method_questions_are_focused(self):
        mgr = self._build_manager({})
        mgr.entity_extractor.available_models = ["IMAGE", "GCAM", "WITCH"]
        mgr.entity_extractor.available_variables = []
        mgr.shared_resources = {"models": [], "link_catalog": []}
        mgr.agents = {}

        image = mgr._route_single("Does IMAGE include biodiversity and human development?")
        gcam = mgr._route_single("Does GCAM represent water and land systems?")
        witch = mgr._route_single("Is WITCH a partial-equilibrium energy model?")

        self.assertIn("Yes.", image)
        self.assertIn("`biodiversity`", image)
        self.assertIn("`human development`", image)
        self.assertIn("Yes.", gcam)
        self.assertIn("`water`", gcam)
        self.assertIn("`land`", gcam)
        self.assertIn("does not state whether `WITCH` is a partial equilibrium model", witch)
        self.assertNotIn("Description:", witch)

    def test_qualitative_suitability_choice_is_not_routed_as_land_data(self):
        mgr = self._build_manager({})
        mgr.entity_extractor.available_models = ["REMIND", "REMIND-MAgPIE"]
        mgr.entity_extractor.available_variables = ["Land Cover", "Final Energy"]
        mgr.shared_resources = {"models": [], "link_catalog": []}
        mgr.agents = {}

        response = mgr._route_single(
            "Which model is better suited to detailed land-energy interactions, "
            "REMIND or REMIND-MAgPIE?"
        )

        self.assertIn("REMIND", response.splitlines()[0])
        self.assertIn("REMIND-MAgPIE", response.splitlines()[0])
        self.assertIn("Common uses:", response)
        self.assertEqual(mgr.last_route_decision["agent"], "model_explanation")
        self.assertIn("energy-land-climate", response)
        self.assertIn("https://iamparis.eu/models", response)
        self.assertNotIn("TIAM", response)
        self.assertNotIn("Relevant IAM PARIS links", response)
        self.assertEqual(mgr.last_route_decision["source"], "query_plan")

    def test_e3me_what_does_it_do_returns_grounded_profile_description(self):
        mgr = self._build_manager({"model": "E3ME-FTT"})
        mgr.entity_extractor.available_models = ["E3ME-FTT", "E3ME 6.1"]
        mgr.entity_extractor.available_variables = []
        mgr.shared_resources = {"models": [], "link_catalog": []}
        mgr.agents = {
            "data_query": _AgentStub(response="wrong data answer"),
            "model_explanation": _AgentStub(response="wrong model-agent answer"),
        }

        response = mgr._route_single("What does the E3ME model do?")

        self.assertIn("### E3ME-FTT", response)
        self.assertIn("dynamic global macroeconomic model", response)
        self.assertIn("Cambridge Econometrics", response)
        self.assertNotIn("wrong data answer", response)
        self.assertEqual(mgr.agents["data_query"].calls, 0)
        self.assertEqual(mgr.agents["model_explanation"].calls, 0)
        self.assertEqual(mgr.last_route_decision["agent"], "model_explanation")

    def test_broad_runtime_model_comparison_defaults_to_metadata(self):
        mgr = self._build_manager({})
        mgr.entity_extractor.available_models = ["Alpha 1.0", "Beta 2.0"]
        mgr.entity_extractor.available_variables = []
        mgr.shared_resources = {
            "models": [
                {"modelName": "Alpha 1.0", "description": "Alpha systems model."},
                {"modelName": "Beta 2.0", "description": "Beta economy model."},
            ],
            "link_catalog": [],
        }
        mgr.agents = {}

        response = mgr._route_single("Compare Alpha 1.0 with Beta 2.0.")

        self.assertIn("Alpha 1.0", response)
        self.assertIn("Beta 2.0", response)
        self.assertEqual(mgr.last_route_decision["agent"], "model_explanation")
        self.assertEqual(mgr.last_entities["models"], ["Alpha 1.0", "Beta 2.0"])

    def test_version_shaped_unknown_model_is_not_misrouted_as_data(self):
        mgr = self._build_manager({"scenario": "Policy"})
        mgr.entity_extractor.available_models = ["KnownModel 1.0"]
        mgr.agents = {}

        response = mgr._route_single(
            "What is UNKNOWN-IAM 9.4, and what kinds of policy questions is it suited for?"
        )

        self.assertIn("cannot resolve `UNKNOWN-IAM 9.4`", response)
        self.assertEqual(mgr.last_entities["unresolved_model"], "UNKNOWN-IAM 9.4")
        self.assertEqual(mgr.last_route_decision["agent"], "model_explanation")

    def test_unresolved_model_followup_does_not_return_global_scenarios(self):
        mgr = self._build_manager({})
        mgr.last_entities = {"unresolved_model": "UNKNOWN-IAM 9.4"}
        mgr.entity_extractor.available_models = ["KnownModel 1.0"]
        mgr.agents = {}

        response = mgr._route_single("Which scenarios are available for this model?")

        self.assertIn("still cannot resolve `UNKNOWN-IAM 9.4`", response)
        self.assertIn("exact catalogue name", response)

    def test_topic_model_availability_aggregates_variable_family(self):
        class _Metadata:
            all_model_names = {"M1", "M2"}

            def models_covering_topic(self, _query):
                return "Topic", ["M1", "M2"]

        mgr = self._build_manager({"variable": "Topic|Leaf"})
        mgr.shared_resources["metadata"] = _Metadata()
        mgr.shared_resources["link_catalog"] = []
        mgr.entity_extractor.available_models = ["M1", "M2"]
        mgr.agents = {}

        response = mgr._route_single("Which models report topic-related variables?")

        self.assertIn("M1, M2", response)
        self.assertEqual(mgr.last_route_decision["reason"], "topic-wide model availability")

    def test_exact_variable_model_availability_precedes_topic_aggregation(self):
        class _Metadata:
            all_variables = {"Topic|Leaf", "Topic|Other"}

            def models_covering_topic(self, _query):
                return "Topic", ["BroadModel", "ExactModel"]

            def get_available_for_variable(self, variable):
                if variable != "Topic|Leaf":
                    return {}
                return {
                    "variable": "Topic|Leaf",
                    "models": ["ExactModel"],
                    "regions": ["R1"],
                    "scenarios": ["S1"],
                    "unit": "unit",
                }

        mgr = self._build_manager({"variable": "Topic|Leaf", "region": "R1"})
        mgr.shared_resources.update({
            "metadata": _Metadata(),
            "link_catalog": [],
            "ts": [{
                "variable": "Topic|Leaf", "region": "R1", "scenario": "S1",
                "modelName": "ExactModel", "2030": 1,
            }],
        })
        mgr.agents = {}

        response = mgr._route_single("Which models report Topic|Leaf for R1?")

        self.assertIn("Availability for Topic|Leaf", response)
        self.assertIn("ExactModel", response)
        self.assertNotIn("BroadModel", response)
        self.assertEqual(mgr.last_route_decision["reason"], "exact-variable scoped availability")

    def test_contextual_model_availability_uses_active_data_scope(self):
        class _Metadata:
            all_variables = {"Metric|Leaf"}

            def get_available_for_variable(self, variable):
                return {
                    "variable": variable,
                    "models": ["M1"],
                    "regions": ["R1"],
                    "scenarios": ["S1"],
                    "unit": "unit",
                }

        mgr = self._build_manager({})
        mgr.entity_extractor.available_variables = ["Metric|Leaf"]
        mgr.entity_extractor.available_regions = ["R1"]
        mgr.entity_extractor.available_scenarios = ["S1"]
        mgr.entity_extractor.available_models = ["M1"]
        mgr.last_entities = {
            "action": "query", "variable": "Metric|Leaf", "region": "R1",
            "scenario": "S1", "start_year": 2030, "end_year": 2030,
        }
        mgr.shared_resources.update({
            "metadata": _Metadata(),
            "link_catalog": [],
            "ts": [{
                "variable": "Metric|Leaf", "region": "R1", "scenario": "S1",
                "modelName": "M1", "2030": 1,
            }],
        })
        mgr.agents = {}

        response = mgr._route_single("Which models provide this variable there?")

        self.assertIn("Availability for Metric|Leaf", response)
        self.assertIn("M1", response)
        self.assertEqual(mgr.last_route_decision["reason"], "scoped availability request")

    def test_contextual_availability_preserves_variable_and_projects_requested_dimension(self):
        class _Metadata:
            all_variables = {"Metric|Leaf", "Other", "Outside"}
            availability_matrix = {
                "Metric|Leaf": {
                    "R1": {
                        "S1": {"M1": {"2030"}, "M2": {"2030"}},
                        "S2": {"M1": {"2030"}},
                    },
                },
                "Other": {"R1": {"S1": {"M1": {"2030"}}}},
                "Outside": {"R2": {"S1": {"M1": {"2030"}}}},
            }

            def get_available_for_variable(self, variable):
                if variable not in self.all_variables:
                    return {"variable": None}
                return {"variable": variable, "unit": "unit"}

        ts = [
            {"variable": "Metric|Leaf", "region": "R1", "scenario": "S1", "modelName": "M1", "2030": 1},
            {"variable": "Metric|Leaf", "region": "R1", "scenario": "S2", "modelName": "M1", "2030": 1},
            {"variable": "Metric|Leaf", "region": "R1", "scenario": "S1", "modelName": "M2", "2030": 1},
            {"variable": "Other", "region": "R1", "scenario": "S1", "modelName": "M1", "2030": 1},
            {"variable": "Outside", "region": "R2", "scenario": "S1", "modelName": "M1", "2030": 1},
        ]
        mgr = self._build_manager({})
        mgr.entity_extractor.available_variables = ["Metric|Leaf", "Other", "Outside"]
        mgr.entity_extractor.available_regions = ["R1", "R2"]
        mgr.entity_extractor.available_scenarios = ["S1", "S2"]
        mgr.entity_extractor.available_models = ["M1", "M2"]
        mgr.shared_resources = {
            "models": [], "metadata": _Metadata(), "ts": ts, "link_catalog": [],
        }
        mgr.agents = {}

        regions = mgr._route_single("Which regions report Metric|Leaf?")
        scenarios = mgr._route_single("Which scenarios are available for that variable?")
        models = mgr._route_single("Which models report it in R1 under S1?")
        variables = mgr._route_single("Which variables are available for M1 in R1?")

        self.assertIn("R1", regions)
        self.assertIn("S1, S2", scenarios)
        self.assertIn("M1, M2", models)
        self.assertIn("Metric|Leaf", variables)
        self.assertIn("Other", variables)
        self.assertNotIn("Outside", variables)

    def test_scenario_pair_availability_ignores_stale_singular_scenario(self):
        class _Metadata:
            all_variables = {"Metric"}
            availability_matrix = {
                "Metric": {
                    "R1": {
                        "S1": {"M1": {"2030"}, "M2": {"2030"}},
                        "S2": {"M1": {"2030"}},
                    },
                },
            }

            def get_available_for_variable(self, variable):
                return {"variable": variable if variable == "Metric" else None, "unit": "unit"}

        mgr = self._build_manager({})
        mgr.entity_extractor.available_variables = ["Metric"]
        mgr.entity_extractor.available_regions = ["R1"]
        mgr.entity_extractor.available_scenarios = ["S1", "S2"]
        mgr.entity_extractor.available_models = ["M1", "M2"]
        mgr.shared_resources = {"models": [], "metadata": _Metadata(), "ts": [], "link_catalog": []}
        mgr.last_entities = {
            "variable": "Metric", "region": "R1", "scenario": "stale",
            "scenarios": ["S1", "S2"],
        }
        mgr.agents = {}

        response = mgr._route_single("Which models provide that variable there?")

        self.assertIn("**Models (1):** M1", response)
        self.assertNotIn("M1, M2", response)

    def test_show_only_model_preserves_exact_versioned_runtime_label(self):
        mgr = self._build_manager({
            "action": "query", "variable": "Metric", "region": "R1",
            "scenario": "S1", "model": "Alpha",
        })
        mgr.entity_extractor.available_variables = ["Metric"]
        mgr.entity_extractor.available_regions = ["R1"]
        mgr.entity_extractor.available_scenarios = ["S1"]
        mgr.entity_extractor.available_models = ["Alpha 2.0", "Beta 1.0"]
        mgr.last_entities = {
            "action": "query", "variable": "Metric", "region": "R1",
            "scenario": "S1", "model": "Beta 1.0",
        }
        data_agent = _AgentStub(response="data handled")
        mgr.agents = {
            "data_query": data_agent,
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        response = mgr._route_single("Show only model Alpha 2.0.")

        self.assertEqual(response, "data handled")
        self.assertEqual(data_agent.last_entities["model"], "Alpha 2.0")
        self.assertEqual(data_agent.last_entities["variable"], "Metric")
        self.assertEqual(data_agent.last_entities["region"], "R1")
        self.assertEqual(data_agent.last_entities["scenario"], "S1")

    def test_change_only_variable_preserves_plot_action(self):
        mgr = self._build_manager({
            "action": "plot", "variable": "Metric|New", "region": "R1",
            "scenario": "S1", "start_year": 2030, "end_year": 2050,
        })
        mgr.entity_extractor.available_variables = ["Metric|Old", "Metric|New"]
        mgr.entity_extractor.available_regions = ["R1"]
        mgr.entity_extractor.available_scenarios = ["S1"]
        mgr.entity_extractor.available_models = []
        mgr.last_entities = {
            "action": "plot", "variable": "Metric|Old", "region": "R1",
            "scenario": "S1", "start_year": 2030, "end_year": 2050,
        }
        plot_agent = _AgentStub(response="plot handled")
        mgr.agents = {
            "data_plotting": plot_agent,
            "data_query": _AgentStub(response="data handled"),
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        response = mgr._route_single("Change only the variable to Metric|New.")

        self.assertEqual(response, mgr.agents["data_query"].response)
        self.assertEqual(mgr.agents["data_query"].last_entities["action"], "query")
        self.assertEqual(mgr.agents["data_query"].last_entities["variable"], "Metric|New")

    def test_variable_only_plot_switch_keeps_single_region_and_later_bar_override(self):
        mgr = self._build_manager({
            "action": "plot",
            "variable": "Metric|New",
            "region": "R1",
            "scenario": "S1",
            "chart_type": "line",
            "entity_confidence": {
                "variable": 0.9, "region": 0.9, "scenario": 0.9,
            },
        })
        mgr.entity_extractor.available_variables = ["Metric|Old", "Metric|New"]
        mgr.entity_extractor.available_regions = ["R1", "R2"]
        mgr.entity_extractor.available_scenarios = ["S1"]
        mgr.entity_extractor.available_models = []
        mgr.last_entities = {
            "action": "plot",
            "variable": "Metric|Old",
            "region": "R1",
            "regions": ["R1"],
            "scenario": "S1",
            "scenarios": ["S1"],
            "start_year": 2030,
            "end_year": 2050,
            "chart_type": "line",
        }
        plot_agent = _AgentStub(response="plot handled")
        mgr.agents = {
            "data_plotting": plot_agent,
            "data_query": _AgentStub(response="data handled"),
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        first = mgr._route_single("change only the variable to Metric|New")

        self.assertEqual(first, mgr.agents["data_query"].response)
        self.assertEqual(mgr.agents["data_query"].last_entities["variable"], "Metric|New")
        self.assertEqual(mgr.agents["data_query"].last_entities["region"], "R1")
        self.assertEqual(mgr.agents["data_query"].last_entities["scenario"], "S1")
        self.assertNotIn("regions", mgr.agents["data_query"].last_entities)
        self.assertNotIn("scenarios", mgr.agents["data_query"].last_entities)

        scope_before_plot = dict(mgr.last_entities)
        redirected = mgr._route_single('plot it as a bar chart')

        self.assertIn("I do not generate figures in the chat", redirected)
        self.assertIn("https://iamparis.eu/results", redirected)
        self.assertEqual(mgr.agents["data_plotting"].calls, 0)
        self.assertEqual(mgr.last_entities, scope_before_plot)
        self.assertEqual(mgr.last_route_decision["reason"], "charts delegated to data explorer")

    def test_exact_catalogue_variable_overrides_clarification_sibling(self):
        mgr = self._build_manager({})
        mgr.entity_extractor.available_variables = ["Metric|PPP", "Metric|MER"]
        mgr.entity_extractor.available_regions = ["R1"]
        mgr.entity_extractor.available_scenarios = []
        mgr.entity_extractor.available_models = []
        mgr._update_clarification_context(
            "data_query",
            "Show data for R1 in 2045",
            "I found the region `R1`. Which variable should I use?",
            {"action": "query", "region": "R1", "start_year": 2045, "end_year": 2045},
        )
        data_agent = _AgentStub(response="### Metric|PPP in R1\n\nAnswer:\n\n| Year | Value |")
        mgr.agents = {
            "data_query": data_agent,
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        mgr._route_single("Use Metric|PPP.")

        self.assertEqual(data_agent.last_entities["variable"], "Metric|PPP")
        self.assertEqual(data_agent.last_entities["region"], "R1")
        self.assertEqual(data_agent.last_entities["start_year"], 2045)

    def test_generic_plot_followup_preserves_exact_carried_scenario(self):
        mgr = self._build_manager({
            "action": "plot", "variable": "Metric", "region": "R1",
            "scenario": "Baseline",
        })
        mgr.last_entities = {
            "variable": "Metric", "region": "R1", "scenario": "Exact_Scenario",
            "start_year": 2040, "end_year": 2040,
        }
        plot_agent = _AgentStub(response="plot handled")
        mgr.agents = {
            "data_plotting": plot_agent,
            "data_query": _AgentStub(response="data"),
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        scope_before_plot = dict(mgr.last_entities)
        redirected = mgr._route_single('Now plot the same data from 2030 to 2050')

        self.assertIn("I do not generate figures in the chat", redirected)
        self.assertIn("https://iamparis.eu/results", redirected)
        self.assertEqual(mgr.agents["data_plotting"].calls, 0)
        self.assertEqual(mgr.last_entities, scope_before_plot)
        self.assertEqual(mgr.last_route_decision["reason"], "charts delegated to data explorer")

    def test_dimension_only_followup_preserves_carried_years(self):
        mgr = self._build_manager({
            "action": "query", "variable": "Metric", "region": "R2",
            "scenario": "Scenario", "start_year": 2000, "end_year": 2100,
        })
        mgr.last_entities = {
            "variable": "Metric", "region": "R1", "scenario": "Scenario",
            "start_year": 2030, "end_year": 2030,
        }
        data_agent = _AgentStub(response="data handled")
        plot_agent = _AgentStub(response="plot handled")
        mgr.agents = {
            "data_query": data_agent,
            "data_plotting": plot_agent,
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        response = mgr._route_single("Keep everything else, but use R2 instead")

        self.assertEqual(response, "data handled")
        self.assertEqual(data_agent.last_entities["start_year"], 2030)
        self.assertEqual(data_agent.last_entities["end_year"], 2030)

    def test_scenario_family_followup_replaces_exact_carried_scenario(self):
        mgr = self._build_manager({
            "action": "query", "variable": "Metric", "region": "EU",
            "scenario": "PR_Baseline", "start_year": 2030, "end_year": 2030,
        })
        mgr.entity_extractor.available_scenarios = ["PR_Baseline", "PR_CurPol_CP"]
        mgr.entity_extractor.available_variables = ["Metric"]
        mgr.entity_extractor.available_regions = ["EU"]
        mgr.last_entities = {
            "variable": "Metric", "region": "EU", "scenario": "PR_Baseline",
            "start_year": 2030, "end_year": 2030,
        }
        data_agent = _AgentStub(response="data handled")
        plot_agent = _AgentStub(response="plot handled")
        mgr.agents = {
            "data_query": data_agent,
            "data_plotting": plot_agent,
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        response = mgr._route_single("Use current-policy scenarios instead, keeping 2030.")

        self.assertEqual(response, "data handled")
        self.assertNotIn("PR_Baseline", data_agent.last_query)
        self.assertEqual(data_agent.last_entities["scenario"], "Current Policies")
        self.assertNotIn("scenarios", data_agent.last_entities)
        self.assertEqual(data_agent.last_entities["start_year"], 2030)
        self.assertEqual(data_agent.last_entities["end_year"], 2030)

    def test_exact_runtime_scenario_switch_overrides_carried_scope_with_model(self):
        mgr = self._build_manager({
            "action": "query", "variable": "Metric", "region": "R1",
            "scenario": "Old_Path", "model": "gcam",
            "start_year": 2040, "end_year": 2040,
        })
        mgr.entity_extractor.available_scenarios = ["Old_Path", "Future_Path"]
        mgr.entity_extractor.available_variables = ["Metric"]
        mgr.entity_extractor.available_regions = ["R1"]
        mgr.entity_extractor.available_models = ["gcam"]
        mgr.last_entities = {
            "variable": "Metric", "region": "R1", "scenario": "Old_Path",
            "model": "gcam", "scenarios": ["Old_Path"],
            "start_year": 2040, "end_year": 2040,
        }
        data_agent = _AgentStub(response="data handled")
        mgr.agents = {
            "data_query": data_agent,
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": _AgentStub(response="model description"),
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        response = mgr._route_single(
            "Switch the scenario to Future_Path and keep the same year."
        )

        self.assertEqual(response, "data handled")
        self.assertEqual(data_agent.last_entities["scenario"], "Future_Path")
        self.assertNotIn("scenarios", data_agent.last_entities)
        self.assertEqual(data_agent.last_entities["model"].casefold(), "gcam")
        self.assertEqual(data_agent.last_entities["start_year"], 2040)
        self.assertNotIn("Old_Path", data_agent.last_query)
        self.assertEqual(mgr.last_entities["scenario"], "Future_Path")
        self.assertNotIn("scenarios", mgr.last_entities)

    def test_no_data_patch_preserves_last_successful_scope(self):
        mgr = self._build_manager({})
        mgr.last_entities = {
            "variable": "Metric", "scenario": "Old_Path",
            "scenarios": ["Old_Path"], "start_year": 2040,
        }

        mgr._persist_last_entities(
            {"variable": "Metric", "scenario": "Future_Path", "start_year": 2040},
            "No data found for this exact combination.",
        )

        self.assertEqual(mgr.last_entities["scenario"], "Old_Path")
        self.assertEqual(mgr.last_entities["scenarios"], ["Old_Path"])
        self.assertEqual(mgr.last_entities["start_year"], 2040)
        self.assertEqual(mgr.last_attempted_entities["scenario"], "Future_Path")

    def test_plot_failures_preserve_last_successful_scope(self):
        failure_answers = (
            (
                "I can't combine these variables on one axis because the loaded "
                "series use incompatible units: `EJ/yr`, `Mt CO2/yr`.",
                {"variables": ["Energy", "Emissions"], "region": "World"},
            ),
            (
                "No time series data is available in the requested year range.",
                {"variable": "Metric", "region": "R2", "start_year": 2050},
            ),
        )

        for answer, attempted in failure_answers:
            with self.subTest(answer=answer):
                mgr = self._build_manager({})
                prior = {"variable": "Prior Metric", "region": "R1"}
                mgr.last_entities = prior

                mgr._persist_last_entities(attempted, answer)

                self.assertEqual(mgr.last_entities, prior)
                self.assertEqual(mgr.last_attempted_entities, attempted)

    def test_rendered_plot_wins_over_partial_no_timeseries_notice(self):
        mgr = self._build_manager({})
        mgr.last_entities = {"variable": "Prior Metric", "region": "R1"}
        record_resolved_scope(
            variable="Metric",
            region="World",
            unit="EJ/yr",
            displayed_series=["Available Model"],
            displayed_series_count=1,
            action="plot",
        )
        answer = (
            "Note: no timeseries data for model `Missing Model` in this slice; "
            "plotting `Available Model`.\n\n"
            "Showing Metric in World.\n"
            "![Plot](data:image/png;base64,ZmFrZQ==)"
        )

        mgr._persist_last_entities(
            {"variable": "Metric", "region": "World", "action": "plot"},
            answer,
        )

        self.assertEqual(mgr.last_entities["variable"], "Metric")
        self.assertEqual(mgr.last_entities["region"], "World")
        self.assertEqual(mgr.last_entities["unit"], "EJ/yr")
        self.assertEqual(mgr.last_attempted_entities, {})

    def test_rendered_table_unit_overrides_stale_extractor_unit(self):
        cases = (
            ("Population", "NGA", "million", "EJ/yr"),
            ("Secondary Energy", "EU", "EJ/yr", "Mt CO2/yr"),
            ("Secondary Energy|Electricity|Solar", "EU", "EJ/yr", "Mt CO2/yr"),
            ("Secondary Energy|Electricity|Wind", "World", "EJ/yr", "Mt CO2/yr"),
        )

        for variable, region, resolved_unit, stale_unit in cases:
            with self.subTest(variable=variable, region=region):
                records = [{
                    "variable": variable,
                    "region": region,
                    "scenario": "Path",
                    "modelName": "Model",
                    "unit": resolved_unit,
                    "years": {"2050": 1},
                }]
                answer = format_time_series_data(records, variable, region)
                mgr = self._build_manager({})

                mgr._persist_last_entities(
                    {
                        "action": "query",
                        "variable": variable,
                        "region": region,
                        "unit": stale_unit,
                    },
                    answer,
                )

                self.assertEqual(mgr.last_entities["unit"], resolved_unit)
                self.assertEqual(mgr.last_entities["variable"], variable)
                self.assertEqual(mgr.last_entities["region"], region)

    def test_auto_dimension_patch_resolves_region_from_runtime_catalog(self):
        mgr = self._build_manager({
            "action": "query", "variable": "Metric", "region": "IND",
            "scenario": "Path", "start_year": 2040, "end_year": 2040,
        })
        mgr.entity_extractor.available_variables = ["Metric"]
        mgr.entity_extractor.available_regions = ["CHN", "IND"]
        mgr.entity_extractor.available_scenarios = ["Path"]
        mgr.entity_extractor.available_models = []
        mgr.last_entities = {
            "action": "query", "variable": "Metric", "region": "CHN",
            "scenario": "Path", "start_year": 2040, "end_year": 2040,
        }
        data_agent = _AgentStub(response="data handled")
        mgr.agents = {
            "data_query": data_agent,
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        response = mgr._route_single("Now use India but keep everything else.")

        self.assertEqual(response, "data handled")
        self.assertEqual(data_agent.last_entities["region"], "IND")
        self.assertEqual(data_agent.last_entities["variable"], "Metric")
        self.assertEqual(data_agent.last_entities["scenario"], "Path")

    def test_plot_comparison_uses_two_consecutive_successful_regions(self):
        mgr = self._build_manager({
            "action": "plot", "variable": "Metric", "region": "R2",
            "scenario": "Path", "start_year": 2040, "end_year": 2050,
        })
        mgr.entity_extractor.available_variables = ["Metric"]
        mgr.entity_extractor.available_regions = ["R1", "R2"]
        mgr.entity_extractor.available_scenarios = ["Path"]
        mgr.entity_extractor.available_models = []
        mgr.previous_entities = {
            "variable": "Metric", "region": "R1", "scenario": "Path",
            "start_year": 2040, "end_year": 2050,
        }
        mgr.last_entities = {
            "variable": "Metric", "region": "R2", "scenario": "Path",
            "start_year": 2040, "end_year": 2050,
        }
        plot_agent = _AgentStub(response="plot handled")
        mgr.agents = {
            "data_query": _AgentStub(response="data"),
            "data_plotting": plot_agent,
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        scope_before_plot = dict(mgr.last_entities)
        redirected = mgr._route_single('Plot the comparison between the two regions.')

        self.assertIn("I do not generate figures in the chat", redirected)
        self.assertIn("https://iamparis.eu/results", redirected)
        self.assertEqual(mgr.agents["data_plotting"].calls, 0)
        self.assertEqual(mgr.last_entities, scope_before_plot)
        self.assertEqual(mgr.last_route_decision["reason"], "charts delegated to data explorer")

    def test_router_fallback_uses_heuristic_data_query(self):
        mgr = self._build_manager({"variable": "Emissions|CO2", "region": "World"})
        data_query_agent = _AgentStub(response="ok from data query")
        mgr.agents = {
            "data_query": data_query_agent,
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="fresh question handled"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        response = mgr._route_single("show me CO2 emissions for world")
        self.assertEqual(response, "ok from data query")
        self.assertEqual(data_query_agent.calls, 1)
        self.assertEqual(mgr.last_route_decision["source"], "deterministic")
        self.assertGreaterEqual(mgr.last_route_decision["confidence"], 0.7)

    def test_obvious_plot_request_does_not_use_router_llm(self):
        mgr = self._build_manager({"action": "plot"})
        mgr.routing_prompt = _PromptShouldNotRun()
        plot_agent = _AgentStub(response="plot handled")
        mgr.agents = {
            "data_query": _AgentStub(response="data"),
            "data_plotting": plot_agent,
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        scope_before_plot = dict(mgr.last_entities)
        redirected = mgr._route_single('plot solar capacity for EU')

        self.assertIn("I do not generate figures in the chat", redirected)
        self.assertIn("https://iamparis.eu/results", redirected)
        self.assertEqual(mgr.agents["data_plotting"].calls, 0)
        self.assertEqual(mgr.last_entities, scope_before_plot)
        self.assertEqual(mgr.last_route_decision["reason"], "charts delegated to data explorer")

    def test_variable_comparison_routes_to_plotting_without_plot_word(self):
        mgr = self._build_manager({"variable": "Capacity|Electricity|Wind"})
        mgr.routing_prompt = _PromptShouldNotRun()
        plot_agent = _AgentStub(response="plot handled")
        mgr.agents = {
            "data_query": _AgentStub(response="data"),
            "data_plotting": plot_agent,
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        response = mgr._route_single("compare wind power and solar PV")

        self.assertEqual(response, mgr.agents["data_query"].response)
        self.assertEqual(plot_agent.calls, 0)
        self.assertEqual(mgr.agents["data_query"].calls, 1)
        self.assertEqual(mgr.last_route_decision["agent"], "data_query")
        self.assertEqual(mgr.last_route_decision["reason"], "chat plotting disabled; return numeric table")

    def test_wind_solar_comparison_repairs_primary_trace_variable(self):
        mgr = self._build_manager({"variable": "Capacity|Electricity|Solar"})
        mgr.routing_prompt = _PromptShouldNotRun()
        plot_agent = _AgentStub(response="plot handled")
        mgr.agents = {
            "data_query": _AgentStub(response="data"),
            "data_plotting": plot_agent,
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        mgr._route_single("compare wind power and solar PV")

        self.assertEqual(mgr.agents["data_query"].last_entities["variable"], "Capacity|Electricity|Wind")
        self.assertEqual(
            mgr.agents["data_query"].last_entities["variables"][:2],
            ["Capacity|Electricity|Wind", "Capacity|Electricity|Solar"],
        )

    def test_greenhouse_gas_request_repairs_primary_trace_variable(self):
        mgr = self._build_manager({"variable": "Emissions|Kyoto Gases|AFOLU"})
        data_agent = _AgentStub(response="data handled")
        mgr.agents = {
            "data_query": data_agent,
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        mgr._route_single("greenhouse gas pathways by country")

        self.assertEqual(data_agent.last_entities["variable"], "Emissions|GHG")

    def test_current_policy_request_repairs_primary_trace_scenario(self):
        mgr = self._build_manager({"variable": "Emissions|CO2", "region": "EU"})
        data_agent = _AgentStub(response="data handled")
        mgr.agents = {
            "data_query": data_agent,
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        mgr._route_single("current policy scenario emissions for EU")

        self.assertEqual(data_agent.last_entities["scenario"], "Current Policies")

    def test_model_comparison_routes_to_plotting_without_plot_word(self):
        mgr = self._build_manager({"variable": "Emissions|CO2", "model": "GCAM"})
        mgr.routing_prompt = _PromptShouldNotRun()
        plot_agent = _AgentStub(response="plot handled")
        mgr.agents = {
            "data_query": _AgentStub(response="data"),
            "data_plotting": plot_agent,
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        response = mgr._route_single("compare GCAM and MESSAGE for CO2 emissions")

        self.assertEqual(response, mgr.agents["data_query"].response)
        self.assertEqual(plot_agent.calls, 0)
        self.assertEqual(mgr.agents["data_query"].calls, 1)
        self.assertEqual(mgr.last_route_decision["agent"], "data_query")

    def test_year_only_followup_reuses_previous_data_scope(self):
        def extract(query):
            self.assertIn("Emissions|CO2", query)
            self.assertIn("World", query)
            self.assertIn("Baseline", query)
            self.assertIn("after 2030", query)
            return {
                "action": "query",
                "variable": "Emissions|CO2",
                "region": "World",
                "scenario": "Baseline",
                "start_year": 2031,
                "end_year": None,
            }

        mgr = self._build_manager(extract)
        mgr.last_entities = {
            "variable": "Emissions|CO2",
            "region": "World",
            "scenario": "Baseline",
        }
        mgr.routing_prompt = _PromptShouldNotRun()
        data_agent = _AgentStub(response="data handled")
        mgr.agents = {
            "data_query": data_agent,
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        response = mgr._route_single("after 2030")

        self.assertEqual(response, "data handled")
        self.assertEqual(data_agent.calls, 1)
        self.assertEqual(mgr.last_route_decision["agent"], "data_query")

    def test_after_year_followup_keeps_open_upper_bound(self):
        mgr = self._build_manager({
            "action": "query", "variable": "Metric", "region": "R1",
            "start_year": 2031, "end_year": 2100,
        })
        mgr.entity_extractor.available_variables = ["Metric"]
        mgr.entity_extractor.available_regions = ["R1"]
        mgr.entity_extractor.available_scenarios = ["Path"]
        mgr.entity_extractor.available_models = []
        mgr.last_entities = {
            "variable": "Metric", "region": "R1", "scenario": "Path",
            "start_year": 2020, "end_year": 2030,
        }
        data_agent = _AgentStub(response="data handled")
        mgr.agents = {
            "data_query": data_agent,
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        mgr._route_single("after 2030")

        self.assertEqual(data_agent.last_entities["start_year"], 2031)
        self.assertNotIn("end_year", data_agent.last_entities)

    def test_open_requested_year_scope_survives_observed_data_and_comparison(self):
        class _ObservedAgent(_AgentStub):
            def handle_with_entities(self, query, entities, history=None):
                self.last_query = query
                self.last_entities = dict(entities or {})
                self.calls += 1
                record_resolved_scope(
                    variable=entities.get("variable"),
                    region=entities.get("region"),
                    scenarios=entities.get("scenarios"),
                    start_year=2035,
                    end_year=2100,
                    action=entities.get("action"),
                )
                return self.response

        def extract(query):
            return {
                "action": "query",
                "variable": "Metric",
                "region": "R1",
                "scenario": "Baseline",
                # Deliberately mimic an extractor that materialised the data
                # maximum; QueryPlan must restore the requested open bound.
                "start_year": 2031,
                "end_year": 2100,
            }

        mgr = self._build_manager(extract)
        mgr.entity_extractor.available_variables = ["Metric"]
        mgr.entity_extractor.available_regions = ["R1"]
        mgr.entity_extractor.available_scenarios = ["Baseline", "Current Policies"]
        mgr.entity_extractor.available_models = []
        mgr.last_entities = {
            "variable": "Metric",
            "region": "R1",
            "scenario": "Baseline",
            "start_year": 2030,
            "end_year": 2030,
        }
        data_agent = _ObservedAgent(response="data handled")
        plot_agent = _ObservedAgent(response="plot handled")
        mgr.agents = {
            "data_query": data_agent,
            "data_plotting": plot_agent,
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        mgr._route_single("after 2030")

        self.assertEqual(mgr.last_entities["start_year"], 2031)
        self.assertNotIn("end_year", mgr.last_entities)
        self.assertEqual(mgr.last_entities["observed_start_year"], 2035)
        self.assertEqual(mgr.last_entities["observed_end_year"], 2100)

        mgr._route_single("compare with current policy")

        self.assertEqual(mgr.agents["data_query"].last_entities["start_year"], 2031)
        self.assertNotIn("end_year", mgr.agents["data_query"].last_entities)
        self.assertEqual(mgr.last_entities["start_year"], 2031)
        self.assertNotIn("end_year", mgr.last_entities)
        self.assertEqual(mgr.last_entities["observed_start_year"], 2035)
        self.assertEqual(mgr.last_entities["observed_end_year"], 2100)

    def test_until_year_followup_clears_carried_lower_bound(self):
        mgr = self._build_manager({
            "action": "query", "variable": "Metric", "region": "R1",
            "start_year": 1900, "end_year": 2050,
        })
        mgr.entity_extractor.available_variables = ["Metric"]
        mgr.entity_extractor.available_regions = ["R1"]
        mgr.entity_extractor.available_scenarios = ["Path"]
        mgr.entity_extractor.available_models = []
        mgr.last_entities = {
            "variable": "Metric", "region": "R1", "scenario": "Path",
            "start_year": 2040, "end_year": 2100,
        }
        data_agent = _AgentStub(response="data handled")
        mgr.agents = {
            "data_query": data_agent,
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        mgr._route_single("until 2050")

        self.assertNotIn("start_year", data_agent.last_entities)
        self.assertEqual(data_agent.last_entities["end_year"], 2050)

    def test_plot_it_preserves_structured_region_comparison(self):
        mgr = self._build_manager({
            "action": "plot", "variable": "Metric", "scenario": "Path",
        })
        mgr.entity_extractor.available_variables = ["Metric"]
        mgr.entity_extractor.available_regions = ["R1", "R2"]
        mgr.entity_extractor.available_scenarios = ["Path"]
        mgr.entity_extractor.available_models = []
        mgr.last_entities = {
            "action": "plot", "variable": "Metric", "regions": ["R1", "R2"],
            "scenario": "Path", "comparison": "region", "chart_type": "line",
        }
        plot_agent = _AgentStub(response="plot handled")
        mgr.agents = {
            "data_query": _AgentStub(response="data"),
            "data_plotting": plot_agent,
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        scope_before_plot = dict(mgr.last_entities)
        redirected = mgr._route_single('plot it')

        self.assertIn("I do not generate figures in the chat", redirected)
        self.assertIn("https://iamparis.eu/results", redirected)
        self.assertEqual(mgr.agents["data_plotting"].calls, 0)
        self.assertEqual(mgr.last_entities, scope_before_plot)
        self.assertEqual(mgr.last_route_decision["reason"], "charts delegated to data explorer")

    def test_plot_it_uses_bar_for_carried_exact_year(self):
        mgr = self._build_manager({
            "action": "plot", "variable": "Metric", "region": "R1",
        })
        mgr.entity_extractor.available_variables = ["Metric"]
        mgr.entity_extractor.available_regions = ["R1"]
        mgr.entity_extractor.available_scenarios = ["Path"]
        mgr.entity_extractor.available_models = []
        mgr.last_entities = {
            "action": "query", "variable": "Metric", "region": "R1",
            "scenario": "Path", "start_year": 2050, "end_year": 2050,
        }
        plot_agent = _AgentStub(response="plot handled")
        mgr.agents = {
            "data_query": _AgentStub(response="data"),
            "data_plotting": plot_agent,
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        scope_before_plot = dict(mgr.last_entities)
        redirected = mgr._route_single('plot it')

        self.assertIn("I do not generate figures in the chat", redirected)
        self.assertIn("https://iamparis.eu/results", redirected)
        self.assertEqual(mgr.agents["data_plotting"].calls, 0)
        self.assertEqual(mgr.last_entities, scope_before_plot)
        self.assertEqual(mgr.last_route_decision["reason"], "charts delegated to data explorer")

    def test_scenario_comparison_followup_reuses_previous_scope_for_plotting(self):
        def extract(query):
            self.assertIn("plot compare", query)
            self.assertIn("Emissions|CO2", query)
            self.assertIn("World", query)
            self.assertIn("current policy", query)
            return {
                "action": "plot",
                "variable": "Emissions|CO2",
                "region": "World",
                "scenario": "Current Policies",
            }

        mgr = self._build_manager(extract)
        mgr.last_entities = {
            "variable": "Emissions|CO2",
            "region": "World",
            "scenario": "Baseline",
        }
        mgr.routing_prompt = _PromptShouldNotRun()
        plot_agent = _AgentStub(response="plot handled")
        mgr.agents = {
            "data_query": _AgentStub(response="data"),
            "data_plotting": plot_agent,
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        response = mgr._route_single("compare with current policy")

        self.assertEqual(response, mgr.agents["data_query"].response)
        self.assertEqual(plot_agent.calls, 0)
        self.assertEqual(mgr.agents["data_query"].calls, 1)
        self.assertEqual(mgr.last_route_decision["agent"], "data_query")

    def test_obvious_model_info_request_does_not_use_router_llm(self):
        mgr = self._build_manager({"model": "GCAM"})
        mgr.shared_resources = {"models": [{"modelName": "GCAM"}]}
        mgr.routing_prompt = _PromptShouldNotRun()
        model_agent = _AgentStub(response="model handled")
        mgr.agents = {
            "data_query": _AgentStub(response="data"),
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": model_agent,
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        response = mgr._route_single("tell me about GCAM model")

        self.assertEqual(response, "model handled")
        self.assertEqual(model_agent.calls, 1)
        self.assertEqual(mgr.last_route_decision["agent"], "model_explanation")
        self.assertEqual(mgr.last_route_decision["source"], "deterministic")

    def test_qualitative_named_model_question_bypasses_variable_clarification(self):
        for query, model in (
            ("what does the E3ME model do?", "E3ME"),
        ):
            with self.subTest(query=query):
                mgr = self._build_manager({
                    "action": "query",
                    "model": model,
                    "variable_candidates": ["Land Cover|Cropland", "Land Cover|Forest"],
                    "entity_confidence": {"model": 0.9},
                })
                mgr.shared_resources = {"models": [{"modelName": model}]}
                mgr.entity_extractor.available_models = [model]
                mgr.entity_extractor.available_variables = [
                    "Land Cover|Cropland", "Land Cover|Forest",
                ]
                mgr.entity_extractor.available_regions = []
                mgr.entity_extractor.available_scenarios = []
                model_agent = _AgentStub(response="model handled")
                data_agent = _AgentStub(response="data handled")
                mgr.agents = {
                    "data_query": data_agent,
                    "data_plotting": _AgentStub(response="plot"),
                    "model_explanation": model_agent,
                    "general_qa": _AgentStub(response="general"),
                    "modelling_suggestions": _AgentStub(response="suggest"),
                }

                response = mgr._route_single(query)

                self.assertNotIn("Choose the variable", response)
                self.assertEqual(response, "model handled")
                self.assertEqual(model_agent.calls, 1)
                self.assertEqual(data_agent.calls, 0)
                self.assertEqual(
                    mgr.last_route_decision["agent"],
                    "model_explanation",
                )

    def test_witch_land_use_question_leads_with_grounded_verdict(self):
        mgr = self._build_manager({
            "action": "query",
            "model": "WITCH",
            "variable_candidates": ["Land Cover|Cropland", "Land Cover|Forest"],
            "entity_confidence": {"model": 0.9},
        })
        mgr.shared_resources = {"models": [{"modelName": "WITCH"}], "link_catalog": []}
        mgr.entity_extractor.available_models = ["WITCH"]
        mgr.entity_extractor.available_variables = [
            "Land Cover|Cropland", "Land Cover|Forest",
        ]
        model_agent = _AgentStub(response="generic model dump")
        data_agent = _AgentStub(response="wrong data route")
        mgr.agents = {
            "data_query": data_agent,
            "model_explanation": model_agent,
        }

        response = mgr._route_single("does WITCH model land use?")

        self.assertIn("I cannot verify", response)
        self.assertIn("Please consult the model documentation", response)
        self.assertIn("https://iamparis.eu/models", response)
        self.assertNotIn("Description:", response)
        self.assertEqual(model_agent.calls, 0)
        self.assertEqual(data_agent.calls, 0)
        self.assertEqual(mgr.last_route_decision["agent"], "model_explanation")

    def test_tiam_kind_question_uses_structured_runtime_model_type(self):
        mgr = self._build_manager({"model": "TIAM_Grantham"})
        mgr.entity_extractor.available_models = ["TIAM_Grantham", "TIAM_Grantham 3.2"]
        mgr.entity_extractor.available_variables = []
        mgr.shared_resources = {
            "models": [
                {
                    "modelName": "TIAM_Grantham",
                    "description": "A replacement version was expected to become available in 2020.",
                    "model_type": "Partial Equilibrium",
                },
                {
                    "modelName": "TIAM_Grantham 3.2",
                    "model_type": "Partial Equilibrium",
                },
            ],
            "link_catalog": [],
        }
        model_agent = _AgentStub(response="stale generic prose")
        mgr.agents = {"model_explanation": model_agent}

        response = mgr._route_single("what kind of model is TIAM?")

        self.assertIn("Methodology/model type: Partial Equilibrium", response)
        self.assertNotIn("available in 2020", response)
        self.assertNotIn("Description:", response)
        self.assertEqual(model_agent.calls, 0)

    def test_bare_aim_family_discloses_single_loaded_match(self):
        mgr = self._build_manager({"model": "AIM/Enduse India 3.3"})
        mgr.entity_extractor.available_models = ["AIM/Enduse India 3.3"]
        mgr.entity_extractor.available_variables = []
        mgr.shared_resources = {
            "models": [{
                "modelName": "AIM/Enduse India 3.3",
                "overview": "A national energy, water and land systems model.",
                "model_type": "Partial Equilibrium",
            }],
            "link_catalog": [],
        }
        model_agent = _AgentStub(response="silently selected")
        mgr.agents = {"model_explanation": model_agent}

        response = mgr._route_single("what is the AIM model?")

        self.assertIn("I matched `AIM`", response)
        self.assertIn("only loaded catalogue entry", response)
        self.assertIn("`AIM/Enduse India 3.3`", response)
        self.assertEqual(mgr.last_entities["model"], "AIM/Enduse India 3.3")
        self.assertEqual(model_agent.calls, 0)

    def test_bare_aim_family_clarifies_multiple_variants_and_resolves_selection(self):
        mgr = self._build_manager({"model": "AIM/Enduse India 3.3"})
        mgr.entity_extractor.available_models = [
            "AIM/CGE Global 2.0", "AIM/Enduse India 3.3",
        ]
        mgr.entity_extractor.available_variables = []
        mgr.shared_resources = {
            "models": [
                {"modelName": "AIM/CGE Global 2.0", "description": "Global CGE variant."},
                {"modelName": "AIM/Enduse India 3.3", "overview": "Indian end-use variant."},
            ],
            "link_catalog": [],
        }
        mgr.agents = {}

        response = mgr._route_single("what is the AIM model?")

        self.assertIn("matches multiple loaded IAM PARIS model entries", response)
        self.assertIn("1. `AIM/CGE Global 2.0`", response)
        self.assertIn("2. `AIM/Enduse India 3.3`", response)

        selected = mgr._route_single("2")

        self.assertIn("### AIM/Enduse India 3.3", selected)
        self.assertIn("Indian end-use variant", selected)
        self.assertEqual(mgr.last_entities["model"], "AIM/Enduse India 3.3")

    def test_bounded_named_model_availability_stays_data_query(self):
        mgr = self._build_manager({
            "action": "query",
            "model": "WITCH",
            "variable": "Price|Carbon",
            "region": "EU",
            "entity_confidence": {"model": 0.9, "variable": 0.95, "region": 0.95},
        })
        mgr.shared_resources = {"models": [{"modelName": "WITCH"}]}
        mgr.entity_extractor.available_models = ["WITCH"]
        mgr.entity_extractor.available_variables = ["Price|Carbon"]
        mgr.entity_extractor.available_regions = ["EU"]
        mgr.entity_extractor.available_scenarios = []
        data_agent = _AgentStub(response="data handled")
        model_agent = _AgentStub(response="model handled")
        mgr.agents = {
            "data_query": data_agent,
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": model_agent,
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        response = mgr._route_single("does WITCH report carbon price for EU?")

        self.assertEqual(response, "data handled")
        self.assertEqual(data_agent.calls, 1)
        self.assertEqual(model_agent.calls, 0)
        self.assertEqual(mgr.last_route_decision["agent"], "data_query")

    def test_latest_projection_year_routes_to_catalogue_data(self):
        mgr = self._build_manager({
            "action": "query",
            "variable_candidates": ["Population", "GDP|MER"],
            "entity_confidence": {"action": 0.75},
        })
        mgr.entity_extractor.available_models = []
        mgr.entity_extractor.available_variables = ["Population", "GDP|MER"]
        mgr.entity_extractor.available_regions = []
        mgr.entity_extractor.available_scenarios = []
        data_agent = _AgentStub(response="Latest loaded year: 2100")
        mgr.agents = {
            "data_query": data_agent,
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        response = mgr._route_single("what is the latest year in the projections?")

        self.assertEqual(response, "Latest loaded year: 2100")
        self.assertEqual(data_agent.calls, 1)
        self.assertEqual(mgr.last_route_decision["agent"], "data_query")
        self.assertEqual(
            mgr.last_route_decision["reason"],
            "catalogue year coverage request",
        )

    def test_model_metadata_fallback_uses_curated_profile_for_known_model(self):
        mgr = self._build_manager({"model": "REMIND"})

        response = mgr._model_metadata_fallback_answer(
            "What are the assumptions in the REMIND model?",
            "I need one more detail: please specify the variable, region, or scenario.",
            {"model": "REMIND"},
        )

        self.assertIn("### REMIND", response)
        self.assertIn("Assumptions:", response)
        self.assertIn("scenario-dependent", response)

    def test_vague_model_information_request_routes_to_data_query_for_eval_parity(self):
        mgr = self._build_manager({"model": "GCAM"})
        mgr.shared_resources = {"models": [{"modelName": "GCAM"}]}
        mgr.routing_prompt = _PromptShouldNotRun()
        data_agent = _AgentStub(response="data handled")
        mgr.agents = {
            "data_query": data_agent,
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        response = mgr._route_single("information on gcam")

        self.assertEqual(response, "data handled")
        self.assertEqual(data_agent.calls, 1)
        self.assertEqual(mgr.last_route_decision["agent"], "data_query")

    def test_application_library_navigation_routes_to_general_qa_before_data(self):
        mgr = self._build_manager({})
        mgr.routing_prompt = _PromptShouldNotRun()
        general_agent = _AgentStub(response="general handled")
        mgr.agents = {
            "data_query": _AgentStub(response="data"),
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": _AgentStub(response="model"),
            "general_qa": general_agent,
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        response = mgr._route_single("open the Aqueduct raw data application")

        self.assertEqual(response, "general handled")
        self.assertEqual(general_agent.calls, 1)
        self.assertEqual(mgr.last_route_decision["agent"], "general_qa")
        self.assertEqual(mgr.last_route_decision["reason"], "site/navigation link request")

    def test_climate_watch_navigation_routes_to_general_qa_before_data(self):
        mgr = self._build_manager({})
        mgr.routing_prompt = _PromptShouldNotRun()
        general_agent = _AgentStub(response="general handled")
        mgr.agents = {
            "data_query": _AgentStub(response="data"),
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": _AgentStub(response="model"),
            "general_qa": general_agent,
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        response = mgr._route_single("where can I find Climate Watch")

        self.assertEqual(response, "general handled")
        self.assertEqual(general_agent.calls, 1)
        self.assertEqual(mgr.last_route_decision["agent"], "general_qa")

    def test_afolu_result_navigation_routes_to_general_qa_before_data(self):
        mgr = self._build_manager({})
        mgr.routing_prompt = _PromptShouldNotRun()
        general_agent = _AgentStub(response="general handled")
        mgr.agents = {
            "data_query": _AgentStub(response="data"),
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": _AgentStub(response="model"),
            "general_qa": general_agent,
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        response = mgr._route_single("AFOLU agriculture land forestry transformation results")

        self.assertEqual(response, "general handled")
        self.assertEqual(general_agent.calls, 1)
        self.assertEqual(mgr.last_route_decision["agent"], "general_qa")

    def test_plain_agriculture_forestry_land_results_route_to_general_qa_before_data(self):
        mgr = self._build_manager({})
        mgr.routing_prompt = _PromptShouldNotRun()
        general_agent = _AgentStub(response="general handled")
        mgr.agents = {
            "data_query": _AgentStub(response="data"),
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": _AgentStub(response="model"),
            "general_qa": general_agent,
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        response = mgr._route_single("agriculture forestry land results")

        self.assertEqual(response, "general handled")
        self.assertEqual(general_agent.calls, 1)
        self.assertEqual(mgr.last_route_decision["agent"], "general_qa")

    def test_cdp_open_data_portal_navigation_routes_to_general_qa_before_data(self):
        mgr = self._build_manager({"region": "IS"})
        mgr.routing_prompt = _PromptShouldNotRun()
        general_agent = _AgentStub(response="portal handled")
        mgr.agents = {
            "data_query": _AgentStub(response="data"),
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": _AgentStub(response="model"),
            "general_qa": general_agent,
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        response = mgr._route_single("where is the CDP Open Data Portal")

        self.assertEqual(response, "portal handled")
        self.assertEqual(general_agent.calls, 1)
        self.assertEqual(mgr.last_route_decision["agent"], "general_qa")

    def test_project_workspace_query_routes_to_general_qa_before_data(self):
        mgr = self._build_manager({"scenario": "Policy", "variable": "Emissions|CO2"})
        mgr.routing_prompt = _PromptShouldNotRun()
        general_agent = _AgentStub(response="workspace handled")
        mgr.agents = {
            "data_query": _AgentStub(response="data"),
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": _AgentStub(response="model"),
            "general_qa": general_agent,
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        response = mgr._route_single("IAM COMPACT renewable energy metrics")

        self.assertEqual(response, "workspace handled")
        self.assertEqual(general_agent.calls, 1)
        self.assertEqual(mgr.last_route_decision["agent"], "general_qa")

    def test_data_story_query_routes_to_general_qa_before_data(self):
        mgr = self._build_manager({"scenario": "Policy"})
        mgr.routing_prompt = _PromptShouldNotRun()
        general_agent = _AgentStub(response="story handled")
        mgr.agents = {
            "data_query": _AgentStub(response="data"),
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": _AgentStub(response="model"),
            "general_qa": general_agent,
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        response = mgr._route_single("recovery policy database")

        self.assertEqual(response, "story handled")
        self.assertEqual(general_agent.calls, 1)
        self.assertEqual(mgr.last_route_decision["agent"], "general_qa")

    def test_workspace_query_interrupts_stale_data_clarification(self):
        mgr = self._build_manager({"variable": "Bad Carryover", "region": "EU"})
        mgr.routing_prompt = _PromptShouldNotRun()
        general_agent = _AgentStub(response="workspace handled")
        data_agent = _AgentStub(response="data handled")
        mgr.agents = {
            "data_query": data_agent,
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": _AgentStub(response="model"),
            "general_qa": general_agent,
            "modelling_suggestions": _AgentStub(response="suggest"),
        }
        mgr.clarification_context = {
            "original_query": "previous bad data",
            "base_query": "previous bad data",
            "agent_type": "data_query",
            "entities": {"variable": "Bad Carryover", "region": "EU"},
            "suggested_options": ["A", "B"],
            "suggested_kind": "variable",
            "response": "Choose the variable",
        }

        response = mgr._route_single("transportation transformation workspace")

        self.assertEqual(response, "workspace handled")
        self.assertEqual(general_agent.calls, 1)
        self.assertEqual(data_agent.calls, 0)
        self.assertEqual(mgr.last_route_decision["agent"], "general_qa")

    def test_profile_model_info_routes_before_stale_region_entity(self):
        mgr = self._build_manager({"region": "IS"})
        mgr.routing_prompt = _PromptShouldNotRun()
        model_agent = _AgentStub(response="model handled")
        mgr.agents = {
            "data_query": _AgentStub(response="data"),
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": model_agent,
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        response = mgr._route_single("what is the WITCH model")

        self.assertEqual(response, "model handled")
        self.assertEqual(model_agent.calls, 1)
        self.assertEqual(mgr.last_route_decision["agent"], "model_explanation")

    def test_global_impacts_of_ndcs_routes_to_general_qa_before_data(self):
        mgr = self._build_manager({"scenario": "NDC"})
        mgr.routing_prompt = _PromptShouldNotRun()
        general_agent = _AgentStub(response="workspace handled")
        mgr.agents = {
            "data_query": _AgentStub(response="data"),
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": _AgentStub(response="model"),
            "general_qa": general_agent,
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        response = mgr._route_single("global impacts of NDCs")

        self.assertEqual(response, "workspace handled")
        self.assertEqual(general_agent.calls, 1)
        self.assertEqual(mgr.last_route_decision["agent"], "general_qa")

    def test_unclear_query_uses_llm_router_as_fallback(self):
        mgr = self._build_manager({})
        mgr.routing_prompt = _PromptReturns("general_qa")
        general_agent = _AgentStub(response="general handled")
        mgr.agents = {
            "data_query": _AgentStub(response="data"),
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": _AgentStub(response="model"),
            "general_qa": general_agent,
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        response = mgr._route_single("hello there")

        self.assertEqual(response, "general handled")
        self.assertEqual(general_agent.calls, 1)
        self.assertEqual(mgr.last_route_decision["source"], "llm")
        self.assertEqual(mgr.last_route_decision["agent"], "general_qa")

    def test_low_confidence_entity_asks_short_clarification(self):
        mgr = self._build_manager(
            {
                "region": "Uncertain Region",
                "entity_confidence": {"region": 0.35},
            }
        )
        data_query_agent = _AgentStub(response="should not be called")
        mgr.agents = {
            "data_query": data_query_agent,
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="fresh question handled"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        response = mgr._route_single("show me uncertain data")

        self.assertEqual(
            response,
            "I matched `Uncertain Region` as the region, but confidence is low. Which region should I use?",
        )
        self.assertEqual(data_query_agent.calls, 0)

    def test_low_confidence_data_clarification_retains_results_link(self):
        mgr = self._build_manager({
            "variable": "Candidate Variable",
            "variable_candidates": ["Candidate Variable", "Other Variable"],
            "entity_confidence": {"variable": 0.35},
        })
        mgr.entity_extractor.available_models = []
        mgr.shared_resources["link_catalog"] = [{
            "title": "Results",
            "url": "https://example.test/results",
            "category": "results",
            "item_type": "route",
            "keywords": ["results"],
            "verified_direct_url": True,
            "search_hint": "",
        }]
        mgr.agents = {
            "data_query": _AgentStub(response="should not be called"),
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        response = mgr._route_single("show candidate data")

        self.assertIn("Choose the variable:", response)
        self.assertEqual([link["url"] for link in mgr.last_links], ["https://example.test/results"])

    def test_general_qa_provider_error_falls_back_to_data_query(self):
        mgr = self._build_manager({})
        data_query_agent = _AgentStub(response="fallback from data query")
        mgr.agents = {
            "data_query": data_query_agent,
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(error="Provider Error: authentication"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        response = mgr._route_single("explain climate policy basics")
        self.assertEqual(response, "fallback from data query")
        self.assertEqual(data_query_agent.calls, 1)

    def test_fresh_query_clears_old_clarification_context(self):
        def extractor(query):
            q = str(query).lower()
            if "electricity" in q and "india" in q:
                return {"variable": "Secondary Energy|Electricity", "region": "India"}
            return {}

        mgr = self._build_manager(extractor)
        data_query_agent = _AgentStub(response="fresh question handled")
        mgr.agents = {
            "data_query": data_query_agent,
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="fresh question handled"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }
        mgr.clarification_context = {
            "original_query": "What is the renewable energy share in Europe by 2050 in net zero scenario",
            "base_query": "What is the renewable energy share in Europe by 2050 in net zero scenario",
            "agent_type": "data_query",
            "entities": {"variable": "Biomass Investment Share", "region": "EU", "scenario": "NZE"},
            "suggested_options": ["PV Investment Share", "Biofuels Investment Share"],
            "suggested_kind": "variable",
            "suggested_variable": "PV Investment Share",
            "suggested_region": "EU",
            "suggested_scenario": "NZE",
            "response": "Choose the variable: 1. `PV Investment Share` 2. `Biofuels Investment Share`",
        }

        response = mgr._route_single("Electricity for India")
        self.assertEqual(response, "fresh question handled")
        self.assertIsNone(mgr.clarification_context)
        self.assertEqual(data_query_agent.calls, 1)

    def test_numeric_reply_keeps_clarification_context(self):
        mgr = self._build_manager({})
        data_query_agent = _AgentStub(response="clarification reply handled")
        mgr.agents = {
            "data_query": data_query_agent,
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="fresh question handled"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }
        mgr.clarification_context = {
            "original_query": "Oil demand for EU",
            "base_query": "Oil demand for EU",
            "agent_type": "data_query",
            "entities": {"region": "EU"},
            "suggested_options": ["Final Energy|Non-Energy Use|Oil", "Secondary Energy|Liquids|Oil"],
            "suggested_kind": "variable",
            "suggested_variable": "Final Energy|Non-Energy Use|Oil",
            "suggested_region": "EU",
            "suggested_scenario": "",
            "response": "Choose the variable: 1. `Final Energy|Non-Energy Use|Oil` 2. `Secondary Energy|Liquids|Oil`",
        }

        response = mgr._route_single("2")
        self.assertEqual(response, "clarification reply handled")
        self.assertEqual(data_query_agent.calls, 1)

    def test_yes_reply_accepts_current_clarification_option(self):
        mgr = self._build_manager({})
        data_query_agent = _AgentStub(response="clarification reply handled")
        mgr.agents = {
            "data_query": data_query_agent,
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="fresh question handled"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }
        mgr.clarification_context = {
            "original_query": "Oil demand for EU",
            "base_query": "Oil demand for EU",
            "agent_type": "data_query",
            "entities": {"region": "EU"},
            "suggested_options": ["Final Energy|Non-Energy Use|Oil"],
            "suggested_kind": "variable",
            "suggested_variable": "Final Energy|Non-Energy Use|Oil",
            "suggested_region": "EU",
            "suggested_scenario": "",
            "response": "Choose the variable: 1. `Final Energy|Non-Energy Use|Oil`",
        }

        response = mgr._route_single("yes")

        self.assertEqual(response, "clarification reply handled")
        self.assertEqual(data_query_agent.last_entities["variable"], "Final Energy|Non-Energy Use|Oil")

    def test_no_reply_advances_to_remaining_clarification_options(self):
        mgr = self._build_manager({})
        mgr.agents = {
            "data_query": _AgentStub(response="data"),
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="fresh question handled"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }
        mgr.clarification_context = {
            "original_query": "Oil demand for EU",
            "base_query": "Oil demand for EU",
            "agent_type": "data_query",
            "entities": {"region": "EU"},
            "suggested_options": ["Final Energy|Non-Energy Use|Oil", "Secondary Energy|Liquids|Oil"],
            "suggested_kind": "variable",
            "suggested_variable": "Final Energy|Non-Energy Use|Oil",
            "suggested_region": "EU",
            "suggested_scenario": "",
            "response": "Choose the variable: 1. `Final Energy|Non-Energy Use|Oil` 2. `Secondary Energy|Liquids|Oil`",
        }

        response = mgr._route_single("no")

        self.assertIn("Okay, here are the next closest options.", response)
        self.assertIn("Secondary Energy|Liquids|Oil", response)

    def test_numeric_reply_without_context_gets_friendly_message(self):
        mgr = self._build_manager({})
        mgr.agents = {
            "data_query": _AgentStub(response="data"),
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="fresh question handled"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        response = mgr._route_single("2")
        self.assertIn("don't have an active numbered choice", response.lower())

    def test_clarification_expires_after_grace_window(self):
        mgr = self._build_manager({})
        data_query_agent = _AgentStub(response="fresh question handled")
        mgr.agents = {
            "data_query": data_query_agent,
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="fresh question handled"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }
        mgr.clarification_context = {
            "original_query": "Oil demand for EU",
            "base_query": "Oil demand for EU",
            "agent_type": "data_query",
            "entities": {"region": "EU"},
            "suggested_options": ["Final Energy|Non-Energy Use|Oil", "Secondary Energy|Liquids|Oil"],
            "suggested_kind": "variable",
            "suggested_variable": "Final Energy|Non-Energy Use|Oil",
            "suggested_region": "EU",
            "suggested_scenario": "",
            "response": "Choose the variable: 1. `Final Energy|Non-Energy Use|Oil` 2. `Secondary Energy|Liquids|Oil`",
            "issued_turn": 1,
        }
        # Within the grace window the numbered choice is still honoured.
        mgr.current_turn = 3
        response = mgr._route_single("2")
        self.assertNotIn("don't have an active numbered choice", response.lower())
        self.assertEqual(data_query_agent.calls, 1)

        # Past the grace window the pending choice expires.
        mgr.clarification_context = {
            "original_query": "Oil demand for EU",
            "base_query": "Oil demand for EU",
            "agent_type": "data_query",
            "entities": {"region": "EU"},
            "suggested_options": ["Final Energy|Non-Energy Use|Oil", "Secondary Energy|Liquids|Oil"],
            "suggested_kind": "variable",
            "suggested_variable": "Final Energy|Non-Energy Use|Oil",
            "suggested_region": "EU",
            "suggested_scenario": "",
            "response": "Choose the variable: 1. `Final Energy|Non-Energy Use|Oil` 2. `Secondary Energy|Liquids|Oil`",
            "issued_turn": 1,
        }
        mgr.current_turn = 5
        response = mgr._route_single("2")
        self.assertIn("don't have an active numbered choice", response.lower())
        self.assertIsNone(mgr.clarification_context)

    def test_repeated_query_prefixes_are_stripped(self):
        self.assertEqual(
            _normalize_cli_query("Query: Query: Plot solar capacity for EU"),
            "Plot solar capacity for EU",
        )

    def test_plot_it_reuses_last_entities(self):
        mgr = self._build_manager({"action": "plot"})
        plot_agent = _AgentStub(response="plot handled")
        mgr.agents = {
            "data_query": _AgentStub(response="data"),
            "data_plotting": plot_agent,
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="fresh question handled"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }
        mgr.last_entities = {
            "variable": "Secondary Energy|Electricity",
            "region": "IND",
            "scenario": "PR_Baseline",
        }

        scope_before_plot = dict(mgr.last_entities)
        redirected = mgr._route_single('plot it')

        self.assertIn("I do not generate figures in the chat", redirected)
        self.assertIn("https://iamparis.eu/results", redirected)
        self.assertEqual(mgr.agents["data_plotting"].calls, 0)
        self.assertEqual(mgr.last_entities, scope_before_plot)
        self.assertEqual(mgr.last_route_decision["reason"], "charts delegated to data explorer")

    def test_same_for_region_reuses_last_scope(self):
        mgr = self._build_manager({"region": "China"})
        data_agent = _AgentStub(response="data handled")
        mgr.agents = {
            "data_query": data_agent,
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }
        mgr.last_entities = {
            "variable": "Emissions|CO2",
            "region": "World",
            "scenario": "Baseline",
        }

        response = mgr._route_single("same for China")

        self.assertEqual(response, "data handled")
        self.assertEqual(data_agent.last_query, "show Emissions|CO2 for China under Baseline")

    def test_region_switch_keeps_one_scenario_filter_and_enables_comparison(self):
        def extracted(query):
            return {
                "action": "plot" if "plot" in query.casefold() else "query",
                "variable": "Synthetic Value",
                "region": "R2" if "R2" in query else "R1",
                "scenario": "Path",
                "entity_confidence": {
                    "variable": 0.9, "region": 0.9, "scenario": 0.9,
                },
            }

        mgr = self._build_manager(extracted)
        mgr.entity_extractor.available_variables = ["Synthetic Value"]
        mgr.entity_extractor.available_regions = ["R1", "R2"]
        mgr.entity_extractor.available_scenarios = ["Path"]
        mgr.entity_extractor.available_models = []
        mgr.last_entities = {
            "action": "query",
            "variable": "Synthetic Value",
            "region": "R1",
            "regions": ["R1"],
            "scenario": "Path",
            "scenarios": ["Path"],
            "start_year": 2050,
            "end_year": 2050,
        }
        data_agent = _AgentStub(response="data handled")
        plot_agent = _AgentStub(response="plot handled")
        mgr.agents = {
            "data_query": data_agent,
            "data_plotting": plot_agent,
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        switched = mgr._route_single("same for R2")

        self.assertEqual(switched, "data handled")
        self.assertEqual(data_agent.last_entities["region"], "R2")
        self.assertEqual(data_agent.last_entities["scenario"], "Path")
        self.assertNotIn("scenarios", data_agent.last_entities)

        scope_before_plot = dict(mgr.last_entities)
        redirected = mgr._route_single('plot the comparison between the two regions')

        self.assertIn("I do not generate figures in the chat", redirected)
        self.assertIn("https://iamparis.eu/results", redirected)
        self.assertEqual(mgr.agents["data_plotting"].calls, 0)
        self.assertEqual(mgr.last_entities, scope_before_plot)
        self.assertEqual(mgr.last_route_decision["reason"], "charts delegated to data explorer")

    def test_what_about_year_reuses_last_scope(self):
        mgr = self._build_manager({"start_year": 2050, "end_year": 2050})
        data_agent = _AgentStub(response="data handled")
        mgr.agents = {
            "data_query": data_agent,
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }
        mgr.last_entities = {
            "variable": "Emissions|CO2",
            "region": "World",
            "scenario": "Baseline",
        }

        response = mgr._route_single("what about 2050")

        self.assertEqual(response, "data handled")
        self.assertEqual(data_agent.last_query, "show Emissions|CO2 for World under Baseline 2050")

    def test_compare_with_scenario_reuses_last_scope(self):
        mgr = self._build_manager({"action": "plot", "scenario": "Baseline"})
        plot_agent = _AgentStub(response="plot handled")
        mgr.agents = {
            "data_query": _AgentStub(response="data"),
            "data_plotting": plot_agent,
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }
        mgr.last_entities = {
            "variable": "Emissions|CO2",
            "region": "World",
            "scenario": "Policy",
        }

        response = mgr._route_single("compare with baseline")

        # The carried scenario is paired with the newly named one and handed to
        # the plotting agent as structured entities (families expanded to member
        # codes), so the new scenario can never be re-read as a region.
        self.assertEqual(response, mgr.agents["data_query"].response)
        self.assertEqual(mgr.agents["data_query"].last_entities.get("variable"), "Emissions|CO2")
        self.assertEqual(mgr.agents["data_query"].last_entities.get("region"), "World")
        self.assertEqual(mgr.agents["data_query"].last_entities.get("comparison"), "scenario")
        self.assertEqual(
            [str(value).casefold() for value in mgr.agents["data_query"].last_entities.get("scenarios", [])],
            ["policy", "baseline"],
        )

    def test_show_all_scenarios_reuses_last_scope(self):
        mgr = self._build_manager({})
        data_agent = _AgentStub(response="data handled")
        mgr.agents = {
            "data_query": data_agent,
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }
        mgr.last_entities = {
            "variable": "Emissions|CO2",
            "region": "World",
            "scenario": "Policy",
        }

        response = mgr._route_single("show all scenarios")

        self.assertEqual(response, "data handled")
        self.assertEqual(data_agent.last_query, "show all scenarios for Emissions|CO2 in World")

    def test_use_first_scenario_selects_first_clarification_option(self):
        mgr = self._build_manager({})
        data_query_agent = _AgentStub(response="clarification reply handled")
        mgr.agents = {
            "data_query": data_query_agent,
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }
        mgr.clarification_context = {
            "original_query": "CO2 for World",
            "base_query": "CO2 for World",
            "agent_type": "data_query",
            "entities": {"variable": "Emissions|CO2", "region": "World"},
            "suggested_options": ["Baseline", "Policy"],
            "suggested_kind": "scenario",
            "suggested_variable": "",
            "suggested_region": "World",
            "suggested_scenario": "Baseline",
            "response": "Choose the scenario: 1. `Baseline` 2. `Policy`",
        }

        response = mgr._route_single("use the first scenario")

        self.assertEqual(response, "clarification reply handled")
        self.assertEqual(data_query_agent.last_entities["scenario"], "Baseline")

    def test_use_first_scenario_without_context_gets_data_query_guidance(self):
        mgr = self._build_manager({})
        mgr.agents = {
            "data_query": _AgentStub(response="data"),
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        response = mgr._route_single("use the first scenario")

        self.assertIn("active scenario choice", response)
        self.assertIn("Reply with a scenario name", response)
        self.assertEqual(mgr.last_route_decision["agent"], "data_query")

    def test_relevant_links_are_appended_to_final_answer(self):
        mgr = self._build_manager({})
        mgr.shared_resources = {
            "models": [],
            "link_catalog": [
                {
                    "title": "GCAM",
                    "url": "https://iamparis.eu/models",
                    "category": "models",
                    "keywords": ["GCAM"],
                    "verified_direct_url": False,
                    "search_hint": "GCAM",
                }
            ],
        }

        response = mgr._append_relevant_links(
            "### GCAM\nA model description.",
            "Tell me about GCAM",
            {"model": "GCAM"},
            "model_explanation",
        )

        self.assertIn("Relevant IAM PARIS links:", response)
        self.assertIn("[GCAM](https://iamparis.eu/models)", response)

    def test_relevant_links_are_not_appended_to_clarification(self):
        mgr = self._build_manager({})
        mgr.shared_resources = {
            "models": [],
            "link_catalog": [
                {
                    "title": "IAM PARIS Results",
                    "url": "https://iamparis.eu/results",
                    "category": "results",
                    "keywords": ["results"],
                    "verified_direct_url": True,
                    "search_hint": "",
                }
            ],
        }

        response = mgr._append_relevant_links(
            "Choose the variable: 1. `Emissions|CO2` Reply with a number (1-1), or `yes` for option 1.",
            "show me data",
            {},
            "data_query",
        )

        self.assertNotIn("Relevant IAM PARIS links:", response)
        self.assertEqual(len(mgr.last_links), 1)
        self.assertEqual(mgr.last_links[0]["url"], "https://iamparis.eu/results")

    def test_site_navigation_answer_is_grounded_in_link_catalog(self):
        mgr = self._build_manager({})
        mgr.shared_resources = {
            "models": [],
            "link_catalog": [
                {
                    "title": "Aqueduct",
                    "url": "https://iamparis.eu/application_library/474",
                    "category": "application_library",
                    "keywords": ["Aqueduct", "Raw Data"],
                    "verified_direct_url": True,
                    "search_hint": "",
                }
            ],
        }
        general_agent = _AgentStub(response="Use the external WRI page at https://www.wri.org/aqueduct")
        mgr.agents = {
            "data_query": _AgentStub(response="data"),
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": _AgentStub(response="model"),
            "general_qa": general_agent,
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        response = mgr._route_single("open Aqueduct details")

        self.assertEqual(general_agent.calls, 0)
        self.assertIn("Use these IAM PARIS links", response)
        self.assertIn("[Aqueduct](https://iamparis.eu/application_library/474)", response)
        self.assertNotIn("wri.org", response)
        self.assertEqual(mgr.last_links[0]["url"], "https://iamparis.eu/application_library/474")

    def test_afolu_land_use_results_routes_to_specific_workspace(self):
        mgr = self._build_manager({})
        mgr.shared_resources = {
            "models": [],
            "link_catalog": [
                {
                    "title": "Results",
                    "url": "https://iamparis.eu/results",
                    "category": "results",
                    "item_type": "route",
                    "keywords": ["scenario results", "project outputs", "workspaces"],
                    "verified_direct_url": True,
                    "search_hint": "",
                },
                {
                    "title": "AFOLU transformation",
                    "url": "https://iamparis.eu/results/ndc-aspects/afolu-transformation/policy_questions",
                    "category": "results",
                    "item_type": "workspace",
                    "keywords": ["AFOLU", "land use", "agriculture", "forestry"],
                    "verified_direct_url": False,
                    "search_hint": "",
                },
            ],
        }
        general_agent = _AgentStub(response="general")
        mgr.agents = {
            "data_query": _AgentStub(response="data"),
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": _AgentStub(response="model"),
            "general_qa": general_agent,
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        response = mgr._route_single("AFOLU land use results")

        self.assertIn(
            "[AFOLU transformation](https://iamparis.eu/results/ndc-aspects/afolu-transformation/policy_questions)",
            response,
        )
        self.assertNotIn("[Results](https://iamparis.eu/results)", response)
        self.assertEqual(general_agent.calls, 0)

    def test_site_navigation_answer_does_not_persist_false_data_entities(self):
        mgr = self._build_manager({"region": "CAN"})
        mgr.shared_resources = {
            "models": [],
            "link_catalog": [
                {
                    "title": "Aqueduct",
                    "url": "https://iamparis.eu/application_library/474",
                    "category": "application_library",
                    "keywords": ["Aqueduct", "Raw Data"],
                    "verified_direct_url": True,
                    "search_hint": "",
                }
            ],
        }
        mgr.agents = {
            "data_query": _AgentStub(response="data"),
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        response = mgr._route_single("where can I find Aqueduct")

        self.assertIn("[Aqueduct](https://iamparis.eu/application_library/474)", response)
        self.assertEqual(mgr.last_entities, {})

    def test_navigation_hides_turn_entities_but_preserves_plot_followup_scope(self):
        def extract(query):
            if "plot" in str(query).casefold():
                return {"action": "plot"}
            return {
                "action": "query",
                "variable": "Emissions|CO2",
                "region": "World",
                "start_year": 2050,
                "end_year": 2050,
            }

        mgr = self._build_manager(extract)
        mgr.entity_extractor.available_variables = ["Emissions|CO2"]
        mgr.entity_extractor.available_regions = ["World"]
        mgr.entity_extractor.available_scenarios = []
        mgr.entity_extractor.available_models = []
        mgr.shared_resources = {
            "models": [],
            "link_catalog": [{
                "title": "Aqueduct",
                "url": "https://iamparis.eu/application_library/474",
                "category": "application_library",
                "keywords": ["Aqueduct", "Raw Data"],
                "verified_direct_url": True,
                "search_hint": "",
            }],
        }
        data_agent = _AgentStub(response="data handled")
        plot_agent = _AgentStub(response="plot handled")
        mgr.agents = {
            "data_query": data_agent,
            "data_plotting": plot_agent,
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        mgr._route_single("show Emissions|CO2 for World in 2050")
        navigation = mgr._route_single("open Aqueduct details")

        self.assertIn("[Aqueduct]", navigation)
        self.assertEqual(mgr.response_entities(), {})
        self.assertEqual(mgr.last_entities["variable"], "Emissions|CO2")
        self.assertEqual(mgr.last_entities["region"], "World")

        scope_before_plot = dict(mgr.last_entities)
        redirected = mgr._route_single('plot it')

        self.assertIn("I do not generate figures in the chat", redirected)
        self.assertIn("https://iamparis.eu/results", redirected)
        self.assertEqual(mgr.agents["data_plotting"].calls, 0)
        self.assertEqual(mgr.last_entities, scope_before_plot)
        self.assertEqual(mgr.last_route_decision["reason"], "charts delegated to data explorer")

    def test_site_navigation_answer_includes_application_library_search_hint(self):
        mgr = self._build_manager({})
        mgr.shared_resources = {
            "models": [],
            "link_catalog": [
                {
                    "title": "Climate Watch",
                    "url": "https://iamparis.eu/application_library",
                    "category": "application_library",
                    "keywords": ["Climate Watch"],
                    "verified_direct_url": False,
                    "search_hint": "Climate Watch",
                    "fallback_instruction": "Open the Application Library and search for: Climate Watch",
                }
            ],
        }

        response = mgr._grounded_site_navigation_answer("where can I find Climate Watch", {})

        self.assertIn("[Climate Watch](https://iamparis.eu/application_library)", response)
        self.assertIn("Search for: Climate Watch", response)
        self.assertIn("Open the Application Library and search for: Climate Watch", response)

    def test_generic_model_catalogue_navigation_uses_runtime_root(self):
        mgr = self._build_manager({})
        mgr.shared_resources = {
            "models": [],
            "link_catalog": [
                {
                    "title": "Models",
                    "url": "https://example.test/model-directory",
                    "category": "models",
                    "item_type": "route",
                    "keywords": ["Model documentation directory"],
                    "verified_direct_url": True,
                    "search_hint": "",
                    "fallback_instruction": "",
                }
            ],
        }
        general_agent = _AgentStub(response="general")
        mgr.agents = {
            "data_query": _AgentStub(response="data"),
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": _AgentStub(response="model"),
            "general_qa": general_agent,
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        response = mgr._route_single(
            "Give me the IAM PARIS page where I can browse valid model names."
        )

        self.assertIn("[Models](https://example.test/model-directory)", response)
        self.assertEqual(general_agent.calls, 0)
        self.assertEqual(mgr.last_route_decision["agent"], "general_qa")
        self.assertIn("navigation", mgr.last_route_decision["reason"])

    def test_model_availability_answer_leads_with_models_link(self):
        mgr = self._build_manager({})
        mgr.shared_resources = {
            "models": [],
            "link_catalog": [
                {
                    "title": "Models",
                    "url": "https://iamparis.eu/models",
                    "category": "models",
                    "item_type": "route",
                    "keywords": ["model directory", "available models"],
                    "verified_direct_url": True,
                    "search_hint": "",
                },
                {
                    "title": "Results",
                    "url": "https://iamparis.eu/results",
                    "category": "results",
                    "item_type": "route",
                    "keywords": ["data", "results"],
                    "verified_direct_url": True,
                    "search_hint": "",
                },
            ],
        }
        mgr.agents = {
            "data_query": _AgentStub(response="Available models answer."),
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        response = mgr._route_single("which models are available")

        self.assertIn("[Models](https://iamparis.eu/models)", response)
        self.assertEqual(mgr.last_links[0]["title"], "Models")
        self.assertEqual(mgr.last_links[0]["url"], "https://iamparis.eu/models")

    def test_model_availability_request_clears_stale_entities(self):
        mgr = self._build_manager({})
        mgr.last_entities = {"variable": "Emissions|CO2", "region": "Greece"}
        mgr.agents = {
            "data_query": _AgentStub(response="models available"),
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        response = mgr._route_single("which models are available")

        self.assertEqual(response, "models available")
        self.assertEqual(mgr.last_entities, {})

    def test_followup_guidance_is_added_to_open_data_answer(self):
        mgr = self._build_manager({})

        response = mgr._maybe_add_followup_guidance(
            "### Emissions|CO2 in EU\nScope: scenario `multiple`.",
            "show me carbon dioxide emissions for Europe",
            "data_query",
        )

        self.assertIn("Reply with a scenario, model, region, or year", response)

    def test_followup_guidance_skips_complete_baseline_year_answer(self):
        mgr = self._build_manager({})

        response = mgr._maybe_add_followup_guidance(
            "### Emissions|CO2 in World\nScope: scenario `Baseline`, years `2030`.",
            "show Emissions|CO2 for World under Baseline in 2030",
            "data_query",
        )

        self.assertNotIn("Reply with a scenario", response)

    def test_fit_for_55_workspace_query_replaces_generic_clarification(self):
        mgr = self._build_manager({"region": "EU"})

        response = mgr._workspace_result_answer(
            "Fit-for-55 EU net zero results",
            "I need one more detail to continue. Please specify the variable, region, or scenario.",
        )

        self.assertIn("IAM COMPACT", response)
        self.assertIn("Fit-for-55", response)
        self.assertNotIn("I need one more detail", response)

    def test_data_answer_keeps_results_fallback_when_model_links_dominate(self):
        mgr = self._build_manager({})
        links = [
            {
                "title": "China-MORE",
                "url": "https://iamparis.eu/models",
                "reason": "Matched: China-MORE",
                "confidence": 1.0,
                "search_hint": "China-MORE",
            }
        ]
        catalog = [
            {
                "title": "IAM PARIS Results",
                "url": "https://iamparis.eu/results",
                "category": "results",
                "item_type": "route",
                "keywords": ["results"],
            }
        ]

        updated = mgr._ensure_results_link_for_data_answer(links, catalog, "data_query")

        self.assertTrue(any(link["url"] == "https://iamparis.eu/results" for link in updated))

    def test_data_answer_uses_runtime_results_root_and_deduplicates_it(self):
        mgr = self._build_manager({})
        root_url = "https://example.test/data-explorer"
        catalog = [{
            "title": "Results",
            "url": root_url,
            "category": "results",
            "item_type": "route",
            "keywords": [],
            "search_hint": "",
        }]
        model_link = [{
            "title": "A model",
            "url": "https://example.test/models",
            "category": "models",
        }]

        updated = mgr._ensure_results_link_for_data_answer(model_link, catalog, "data_query")
        deduplicated = mgr._ensure_results_link_for_data_answer(
            [{"title": "Runtime data root", "url": f"{root_url}/"}],
            catalog,
            "data_query",
        )

        self.assertEqual(updated[-1]["url"], root_url)
        self.assertEqual(len(deduplicated), 1)

    def test_scenario_comparison_followup_preserves_both_scenarios(self):
        mgr = self._build_manager(
            {
                "action": "plot",
                "variable": "Emissions|CO2",
                "region": "World",
                "scenario": "Baseline",
            }
        )
        mgr.shared_resources = {
            "models": [],
            "ts": [
                {"scenario": "Baseline", "variable": "Emissions|CO2"},
                {"scenario": "Policy", "variable": "Emissions|CO2"},
            ],
        }

        entities = mgr._repair_comparison_entities(
            "plot compare Emissions|CO2 for World under Baseline versus Policy",
            {
                "action": "plot",
                "variable": "Emissions|CO2",
                "region": "World",
                "scenario": "Baseline",
            },
        )

        self.assertEqual(entities["comparison"], "scenario")
        self.assertEqual(entities["scenarios"], ["Baseline", "Policy"])
        self.assertIsNone(entities["scenario"])

    def test_industry_emissions_repairs_stale_steel_extraction(self):
        mgr = self._build_manager({})
        mgr.shared_resources = {
            "models": [],
            "ts": [
                {
                    "variable": "Emissions|CO2|Energy|Demand|Industry",
                    "region": "EU",
                },
                {
                    "variable": "Emissions|CO2|Industry|Steel",
                    "region": "EU",
                },
            ],
        }

        entities = mgr._repair_comparison_entities(
            "industry emissions in Europe",
            {
                "action": "query",
                "variable": "Emissions|CO2|Industry|Steel",
                "region": "EU",
                "entity_confidence": {"variable": 0.9, "region": 0.85},
            },
        )

        self.assertEqual(
            entities["variable"],
            "Emissions|CO2|Energy|Demand|Industry",
        )

    def test_scenario_comparison_followup_keeps_carried_variable(self):
        def extract(query):
            self.assertIn("Emissions|CO2", query)
            return {
                "action": "plot",
                "variable": "Emissions|CO2|Energy and Industrial Processes",
                "region": "World",
                "scenario": "Baseline",
                "entity_confidence": {"variable": 0.7, "region": 0.85, "scenario": 0.9},
            }

        mgr = self._build_manager(extract)
        mgr.last_entities = {
            "variable": "Emissions|CO2",
            "region": "World",
            "scenario": "Baseline",
        }
        mgr.shared_resources = {
            "models": [],
            "ts": [
                {"scenario": "Baseline", "variable": "Emissions|CO2"},
                {"scenario": "Policy", "variable": "Emissions|CO2"},
            ],
        }
        mgr.routing_prompt = _PromptShouldNotRun()
        plot_agent = _AgentStub(response="plot handled")
        mgr.agents = {
            "data_query": _AgentStub(response="data"),
            "data_plotting": plot_agent,
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        response = mgr._route_single("compare with policy")

        self.assertEqual(response, mgr.agents["data_query"].response)
        self.assertEqual(mgr.agents["data_query"].last_entities["variable"], "Emissions|CO2")
        self.assertEqual(mgr.agents["data_query"].last_entities["scenarios"], ["Baseline", "Policy"])

    def test_curated_model_profile_bypasses_low_confidence_model_clarification(self):
        mgr = self._build_manager(
            {
                "action": "query",
                "model": "REMIND",
                "entity_confidence": {"model": 0.35},
            }
        )
        model_agent = _AgentStub(response="model profile")
        mgr.agents = {
            "data_query": _AgentStub(response="data"),
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": model_agent,
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        response = mgr._route_single("What is the REMIND model?")

        self.assertEqual(response, "model profile")
        self.assertEqual(mgr.last_route_decision["agent"], "model_explanation")
        self.assertGreaterEqual(model_agent.last_entities["entity_confidence"]["model"], 0.9)

    def test_dimension_switch_followups_are_recognized(self):
        mgr = self._build_manager({})
        for phrase in (
            "under PR_NDC_CP", "now for CHN", "and under PR_CurPol_CP?", "for India",
            "Now show the same value for R2.",
        ):
            self.assertTrue(mgr._is_contextual_dimension_followup(phrase), msg=phrase)
        # A genuine list request or a fresh question must not be treated as one.
        for phrase in ("list models", "what models are available", "for the EU what are the emissions of CO2"):
            self.assertFalse(mgr._is_contextual_dimension_followup(phrase), msg=phrase)

    def test_compose_under_switch_keeps_scope_and_swaps_scenario(self):
        mgr = self._build_manager({})
        composed = mgr._compose_contextual_query(
            "under PR_NDC_CP",
            {"variable": "Emissions|CO2", "region": "EU"},
        )
        self.assertIn("Emissions|CO2", composed)
        self.assertIn("EU", composed)
        self.assertIn("under PR_NDC_CP", composed)

    def test_compose_region_switch_carries_model_without_list_trigger(self):
        mgr = self._build_manager({})
        composed = mgr._compose_contextual_query(
            "now for CHN",
            {"variable": "Final Energy", "model": "gcam"},
        )
        self.assertIn("Final Energy", composed)
        self.assertIn("CHN", composed)
        self.assertIn("gcam", composed)
        # The literal word "model" would trip the model-list detector downstream.
        self.assertNotIn("for model gcam", composed)

    def test_explicit_runtime_region_overrides_stale_extracted_context(self):
        mgr = self._build_manager({
            "action": "query", "variable": "Metric", "region": "R1",
            "scenario": "Path", "start_year": 2040, "end_year": 2040,
        })
        mgr.entity_extractor.available_variables = ["Metric"]
        mgr.entity_extractor.available_regions = ["R1", "R2"]
        mgr.entity_extractor.available_scenarios = ["Path"]
        mgr.entity_extractor.available_models = []
        mgr.last_entities = {
            "variable": "Metric", "region": "R1", "scenario": "Path",
            "start_year": 2040, "end_year": 2040,
        }
        data_agent = _AgentStub(response="data handled")
        mgr.agents = {
            "data_query": data_agent,
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        response = mgr._route_single("Now show the same value for R2.")

        self.assertEqual(response, "data handled")
        self.assertEqual(data_agent.last_entities["region"], "R2")
        self.assertEqual(data_agent.last_entities["variable"], "Metric")
        self.assertEqual(data_agent.last_entities["scenario"], "Path")
        self.assertEqual(data_agent.last_entities["start_year"], 2040)

    def test_buildings_results_is_navigation_not_data(self):
        for phrase in ("can I find the buildings results?", "buildings results",
                       "show me the buildings results", "transport results", "afolu results"):
            self.assertTrue(_looks_like_site_navigation_request(phrase), msg=phrase)
        # A data query that merely mentions a workspace word must stay data.
        for phrase in ("buildings emissions for EU", "final energy for buildings in India",
                       "show me the results for buildings emissions"):
            self.assertFalse(_looks_like_site_navigation_request(phrase), msg=phrase)

    def test_result_provenance_followup_answers_from_last_models(self):
        mgr = self._build_manager({})
        # Capture models from a rendered data answer.
        mgr._persist_last_entities(
            {},
            "### Final Energy in IND\n**42 - PR_Baseline**\n| Year |\n**gcam - PR_NDC_CP**\n| Year |",
        )
        self.assertEqual(
            mgr.last_result_models,
            [UNLABELLED_MODEL_LABEL, "gcam"],
        )
        for phrase in (
            "which model is this from?",
            "what model is this?",
            "which models are these from",
            "Which models are these results from?",
        ):
            self.assertTrue(mgr._is_result_provenance_question(phrase), msg=phrase)
        answer = mgr._result_provenance_answer()
        self.assertIn(f"`{UNLABELLED_MODEL_LABEL}`", answer)
        self.assertNotRegex(answer, r"\b42\b")
        self.assertIn("`gcam`", answer)

    def test_result_provenance_single_model(self):
        mgr = self._build_manager({})
        mgr._persist_last_entities({}, "### GDP|MER in EU\nScope: scenario `multiple`, model `gcam`, years")
        self.assertEqual(mgr.last_result_models, ["gcam"])
        self.assertEqual(mgr._result_provenance_answer(), "That result comes from model `gcam`.")

    def test_result_scope_followups_use_latest_plot_scope(self):
        mgr = self._build_manager({})
        mgr.entity_extractor.available_scenarios = ["Policy_A", "Policy_B", "Baseline"]
        mgr.last_entities = {
            "variable": "Metric",
            "region": "R1",
            "scenarios": ["Policy_A", "Policy_B"],
            "action": "plot",
        }
        mgr.last_result_models = ["Alpha", "Beta"]
        mgr.agents = {}

        scenario_answer = mgr._route_single("Which scenarios are shown in this plot?")
        self.assertIn("`Policy_A`", scenario_answer)
        self.assertIn("`Policy_B`", scenario_answer)
        self.assertNotIn("Baseline", scenario_answer)

        model_answer = mgr._route_single("Which models contributed to it?")
        self.assertIn("`Alpha`", model_answer)
        self.assertIn("`Beta`", model_answer)
        self.assertNotIn("available models", model_answer.lower())

    def test_plot_model_scope_followup_falls_back_to_structured_result_models(self):
        mgr = self._build_manager({})
        mgr.last_entities = {
            "action": "plot",
            "variable": "Metric",
            "region": "R1",
            "result_models": ["Alpha", "Beta"],
        }
        # Simulate a restored/session state where the auxiliary attribute was
        # not populated but the latest successful plot scope is intact.
        mgr.last_result_models = []
        mgr.agents = {}

        response = mgr._route_single("Which models contributed to it?")

        self.assertIn("The latest result includes these models", response)
        self.assertIn("`Alpha`", response)
        self.assertIn("`Beta`", response)
        self.assertNotIn("Availability", response)
        self.assertEqual(mgr.last_route_decision["reason"], "latest-result scope follow-up")

    def test_failed_scope_is_not_silently_used_as_one_sided_comparison(self):
        mgr = self._build_manager({})
        mgr.last_entities = {"variable": "Metric", "region": "R1", "scenario": "Path"}
        mgr.last_attempted_entities = {
            "variable": "Metric", "region": "R2", "scenario": "Path"
        }
        mgr.agents = {"data_plotting": _AgentStub(response="plot handled")}

        scope_before_plot = dict(mgr.last_entities)
        redirected = mgr._route_single('Plot the comparison between the two regions.')

        self.assertIn("I do not generate figures in the chat", redirected)
        self.assertIn("https://iamparis.eu/results", redirected)
        self.assertEqual(mgr.agents["data_plotting"].calls, 0)
        self.assertEqual(mgr.last_entities, scope_before_plot)
        self.assertEqual(mgr.last_route_decision["reason"], "charts delegated to data explorer")

    def test_plot_scope_persists_contributing_models_structurally(self):
        mgr = self._build_manager({})
        record_resolved_scope(
            variable="Metric", region="R1", action="plot", result_models=["Alpha-1", "Beta-2"],
        )

        mgr._persist_last_entities({}, "Showing Metric in R1.\n![Plot](data:image/png;base64,x)")

        self.assertEqual(mgr.last_result_models, ["Alpha-1", "Beta-2"])
        self.assertNotIn("models", mgr.last_entities)

    def test_availability_followup_is_bounded_to_referenced_models(self):
        mgr = self._build_manager({})
        mgr.entity_extractor.available_variables = ["Metric"]
        mgr.entity_extractor.available_regions = ["World"]
        mgr.entity_extractor.available_scenarios = ["Path"]
        mgr.entity_extractor.available_models = ["Alpha-1", "Beta-1", "Gamma-1"]
        mgr.shared_resources = {
            "models": [],
            "ts": [
                {"modelName": "Alpha-1", "variable": "Metric", "region": "World", "scenario": "Path", "years": {"2050": 1}},
                {"modelName": "Beta-1", "variable": "Other", "region": "World", "scenario": "Path", "years": {"2050": 2}},
                {"modelName": "Gamma-1", "variable": "Metric", "region": "World", "scenario": "Path", "years": {"2050": 3}},
            ],
            "link_catalog": [],
        }
        mgr.last_entities = {"models": ["Alpha", "Beta"]}
        mgr.agents = {}

        response = mgr._route_single("Which one reports Metric for World?")

        self.assertIn("`Alpha`: available", response)
        self.assertIn("`Beta`: no matching rows", response)
        self.assertNotIn("Gamma", response)
        self.assertEqual(mgr.last_route_decision["source"], "conversation_state")

    def test_plural_model_availability_precedes_global_topic_discovery(self):
        class _Metadata:
            all_model_names = {"Alpha-1", "Beta-1", "Gamma-1"}

            def models_covering_topic(self, _query):
                return "Metric family", ["Alpha-1", "Beta-1", "Gamma-1"]

        mgr = self._build_manager({})
        mgr.entity_extractor.available_variables = ["Metric"]
        mgr.entity_extractor.available_regions = ["World"]
        mgr.entity_extractor.available_scenarios = ["Path"]
        mgr.entity_extractor.available_models = ["Alpha-1", "Beta-1", "Gamma-1"]
        mgr.shared_resources = {
            "models": [],
            "metadata": _Metadata(),
            "ts": [
                {"modelName": "Alpha-1", "variable": "Metric", "region": "World", "scenario": "Path", "years": {"2050": 1}},
                {"modelName": "Beta-1", "variable": "Other", "region": "World", "scenario": "Path", "years": {"2050": 2}},
                {"modelName": "Gamma-1", "variable": "Metric", "region": "World", "scenario": "Path", "years": {"2050": 3}},
            ],
            "link_catalog": [],
        }
        mgr.last_entities = {"models": ["Alpha", "Beta"]}
        mgr.agents = {}

        response = mgr._route_single("Which of those models reports Metric for World?")

        self.assertIn("`Alpha`: available", response)
        self.assertIn("`Beta`: no matching rows", response)
        self.assertNotIn("Gamma", response)
        self.assertEqual(mgr.last_route_decision["reason"], "referenced-model availability")
        self.assertEqual(mgr.last_entities["models"], ["Alpha", "Beta"])
        self.assertEqual(mgr.last_entities["variable"], "Metric")

    def test_referenced_models_survive_availability_then_link_followup(self):
        mgr = self._build_manager({})
        mgr.entity_extractor.available_variables = ["Metric"]
        mgr.entity_extractor.available_regions = ["World"]
        mgr.entity_extractor.available_scenarios = ["Path"]
        mgr.entity_extractor.available_models = ["Alpha-1", "Beta-1"]
        mgr.shared_resources = {
            "models": [],
            "ts": [
                {"modelName": "Alpha-1", "variable": "Metric", "region": "World", "scenario": "Path", "years": {"2050": 1}},
                {"modelName": "Beta-1", "variable": "Other", "region": "World", "scenario": "Path", "years": {"2050": 2}},
            ],
            "link_catalog": [
                {
                    "title": "Models",
                    "url": "https://example.test/models",
                    "category": "models",
                    "item_type": "route",
                    "keywords": ["Model documentation directory"],
                    "verified_direct_url": True,
                    "search_hint": "",
                }
            ],
        }
        mgr.last_entities = {"models": ["Alpha", "Beta"]}
        mgr.agents = {}

        mgr._route_single("Which of those models reports Metric for World?")
        response = mgr._route_single("Give me documentation links for both models")

        self.assertIn("Alpha, Beta", response)
        self.assertIn("https://example.test/models", response)
        self.assertEqual(mgr.last_entities["models"], ["Alpha", "Beta"])

    def test_no_context_plural_model_availability_does_not_fall_back_globally(self):
        class _Metadata:
            all_model_names = {"Alpha-1", "Beta-1"}

            def models_covering_topic(self, _query):
                return "Metric family", ["Alpha-1", "Beta-1"]

        mgr = self._build_manager({})
        mgr.entity_extractor.available_variables = ["Metric"]
        mgr.entity_extractor.available_models = ["Alpha-1", "Beta-1"]
        mgr.shared_resources = {
            "models": [],
            "metadata": _Metadata(),
            "ts": [],
            "link_catalog": [],
        }
        mgr.agents = {}

        response = mgr._route_single("Which of those models reports Metric?")

        self.assertIn("do not have a referenced set of two models", response)
        self.assertNotIn("Models covering", response)
        self.assertEqual(mgr.last_route_decision["reason"], "missing referenced model set")

    def test_short_catalog_code_does_not_match_lowercase_pronoun(self):
        mgr = self._build_manager({})

        self.assertEqual(mgr._match_catalog_value_from_text("Plot it.", ["EU", "IT"]), "")
        self.assertEqual(mgr._match_catalog_value_from_text("Plot IT.", ["EU", "IT"]), "IT")
        self.assertEqual(mgr._match_catalog_value_from_text("use region it", ["EU", "IT"]), "IT")

    def test_model_scope_followup_keeps_previous_model_for_region_question(self):
        mgr = self._build_manager({"action": "query"})
        mgr.entity_extractor.available_models = ["SyntheticModel"]
        mgr.entity_extractor.available_variables = ["Synthetic Metric"]
        mgr.entity_extractor.available_regions = ["R1", "R2"]
        mgr.entity_extractor.available_scenarios = ["Path"]
        mgr.last_entities = {"model": "SyntheticModel"}
        data_agent = _AgentStub(response="model-scoped regions")
        mgr.agents = {
            "data_query": data_agent,
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        response = mgr._route_single("Which regions does it cover?")

        self.assertEqual(response, "model-scoped regions")
        self.assertEqual(data_agent.last_entities["model"], "SyntheticModel")
        self.assertIn("SyntheticModel", data_agent.last_query)

    def test_model_metadata_region_followup_lists_only_that_models_data(self):
        mgr = self._build_manager({})
        mgr.entity_extractor.available_models = ["SyntheticModel", "OtherModel"]
        mgr.entity_extractor.available_variables = ["Metric"]
        mgr.entity_extractor.available_regions = ["R1", "R2", "R3"]
        mgr.entity_extractor.available_scenarios = ["Path"]
        mgr.shared_resources["ts"] = [
            {
                "modelName": "SyntheticModel", "variable": "Metric",
                "region": "R1", "scenario": "Path", "years": {"2050": 1},
            },
            {
                "modelName": "SyntheticModel", "variable": "Metric",
                "region": "R2", "scenario": "Path", "years": {"2050": 2},
            },
            {
                "modelName": "OtherModel", "variable": "Metric",
                "region": "R3", "scenario": "Path", "years": {"2050": 3},
            },
        ]
        mgr.last_entities = {"model": "SyntheticModel"}
        data_agent = _AgentStub(response="wrong variable clarification")
        mgr.agents = {
            "data_query": data_agent,
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        response = mgr._route_single("Which regions does it cover?")

        self.assertIn("Model `SyntheticModel` has these regions: R1, R2", response)
        self.assertNotIn("R3", response)
        self.assertEqual(data_agent.calls, 0)
        self.assertEqual(mgr.last_route_decision["reason"], "model-scoped catalogue follow-up")

    def test_model_metadata_region_followup_reports_no_loaded_data(self):
        mgr = self._build_manager({})
        mgr.entity_extractor.available_models = ["EmptyModel", "OtherModel"]
        mgr.entity_extractor.available_variables = ["Metric"]
        mgr.entity_extractor.available_regions = ["R3"]
        mgr.entity_extractor.available_scenarios = ["Path"]
        mgr.shared_resources["ts"] = [{
            "modelName": "OtherModel", "variable": "Metric",
            "region": "R3", "scenario": "Path", "years": {"2050": 3},
        }]
        mgr.last_entities = {"model": "EmptyModel"}
        mgr.agents = {}

        response = mgr._route_single("Which regions does it cover?")

        self.assertEqual(
            response,
            "I could not find any regions recorded for model `EmptyModel`.",
        )
        self.assertNotIn("R3", response)

    def test_multi_intent_pronoun_receives_structured_state_from_prior_segment(self):
        mgr = self._build_manager({})
        routed_queries = []

        def route_segment(query, _history=None, context=None):
            routed_queries.append((query, dict((context or {}).get("last_entities") or {})))
            mgr.last_entities = {"model": "SyntheticModel"}
            return "handled"

        with patch.object(mgr, "_route_single", side_effect=route_segment):
            response = mgr.route_query(
                "Explain SyntheticModel and show its variables."
            )

        self.assertEqual(len(routed_queries), 2)
        self.assertEqual(routed_queries[1][1], {"model": "SyntheticModel"})
        self.assertIn("model SyntheticModel", routed_queries[1][0])
        self.assertIn("**2. show its variables.**", response)

    def test_multi_intent_keeps_parent_model_when_first_segment_clears_state(self):
        mgr = self._build_manager({})
        routed_queries = []

        def route_segment(query, _history=None, context=None):
            routed_queries.append((query, dict((context or {}).get("last_entities") or {})))
            mgr.last_entities = {}
            return "handled"

        with patch.object(mgr, "_route_single", side_effect=route_segment):
            mgr.route_query("List scenarios for GCAM-PR and explain what the model is designed for.")

        self.assertEqual(routed_queries[0][1]["model"], "GCAM-PR")
        self.assertEqual(routed_queries[1][1]["model"], "GCAM-PR")
        self.assertIn("model GCAM-PR", routed_queries[1][0])

    def test_dependent_model_metadata_clauses_stay_in_one_intent(self):
        mgr = self._build_manager({})

        parts = mgr._split_multi_intent(
            "Explain MESSAGEix-GLOBIOM and describe its applications and limitations."
        )

        self.assertEqual(parts, [
            "Explain MESSAGEix-GLOBIOM and describe its applications and limitations."
        ])

    def test_runtime_family_metadata_clauses_stay_in_one_intent(self):
        mgr = self._build_manager({})
        mgr.entity_extractor.available_models = ["Alpha 1.0"]
        mgr.shared_resources["models"] = [
            {"modelName": "Alpha 1.0", "description": "Runtime description."}
        ]

        parts = mgr._split_multi_intent(
            "Describe Alpha and explain its main use cases."
        )

        self.assertEqual(parts, [
            "Describe Alpha and explain its main use cases."
        ])

    def test_plain_plot_followup_keeps_latest_scope_instead_of_comparing_history(self):
        mgr = self._build_manager({
            "action": "plot", "variable": "Metric", "region": "R2",
            "scenario": "Path", "start_year": 2040, "end_year": 2050,
        })
        mgr.entity_extractor.available_variables = ["Metric"]
        mgr.entity_extractor.available_regions = ["R1", "R2"]
        mgr.entity_extractor.available_scenarios = ["Path"]
        mgr.entity_extractor.available_models = []
        mgr.previous_entities = {
            "variable": "Metric", "region": "R1", "scenario": "Path",
            "start_year": 2040, "end_year": 2050,
        }
        mgr.last_entities = {
            "variable": "Metric", "region": "R2", "scenario": "Path",
            "start_year": 2040, "end_year": 2050,
        }
        plot_agent = _AgentStub(response="plot handled")
        mgr.agents = {
            "data_query": _AgentStub(response="data"),
            "data_plotting": plot_agent,
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        scope_before_plot = dict(mgr.last_entities)
        redirected = mgr._route_single('Plot that.')

        self.assertIn("I do not generate figures in the chat", redirected)
        self.assertIn("https://iamparis.eu/results", redirected)
        self.assertEqual(mgr.agents["data_plotting"].calls, 0)
        self.assertEqual(mgr.last_entities, scope_before_plot)
        self.assertEqual(mgr.last_route_decision["reason"], "charts delegated to data explorer")

    def test_scenario_comparison_followup_keeps_bounded_pair_on_replot(self):
        def extracted(query):
            return {
                "action": "plot" if "plot" in query.lower() else "query",
                "variable": "Metric",
                "region": "R1",
                "scenario": "Path B" if "Path B" in query else None,
                "start_year": 2050,
                "end_year": 2050,
                "entity_confidence": {
                    "action": 0.9,
                    "variable": 0.9,
                    "region": 0.9,
                    "scenario": 0.9,
                    "years": 0.9,
                },
            }

        mgr = self._build_manager(extracted)
        mgr.entity_extractor.available_variables = ["Metric"]
        mgr.entity_extractor.available_regions = ["R1"]
        mgr.entity_extractor.available_scenarios = ["Path A", "Path B", "Path C"]
        mgr.entity_extractor.available_models = []
        mgr.last_entities = {
            "action": "query",
            "variable": "Metric",
            "region": "R1",
            "scenario": "Path A",
            "start_year": 2050,
            "end_year": 2050,
        }
        plot_agent = _AgentStub(response="plot handled")
        mgr.agents = {
            "data_query": _AgentStub(response="data"),
            "data_plotting": plot_agent,
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        first = mgr._route_single("Compare it with Path B.")

        self.assertEqual(first, mgr.agents["data_query"].response)
        self.assertEqual(mgr.agents["data_query"].last_entities["scenarios"], ["Path A", "Path B"])
        self.assertEqual(mgr.agents["data_query"].last_entities["comparison"], "scenario")
        self.assertNotIn("variables", mgr.agents["data_query"].last_entities)

        scope_before_plot = dict(mgr.last_entities)
        redirected = mgr._route_single('Plot the comparison.')

        self.assertIn("I do not generate figures in the chat", redirected)
        self.assertIn("https://iamparis.eu/results", redirected)
        self.assertEqual(mgr.agents["data_plotting"].calls, 0)
        self.assertEqual(mgr.last_entities, scope_before_plot)
        self.assertEqual(mgr.last_route_decision["reason"], "charts delegated to data explorer")

    def test_region_patch_preserves_bounded_scenario_comparison(self):
        def extracted(query):
            return {
                "action": "query",
                "variable": "Synthetic Output",
                "region": "R2" if "R2" in query else "R1",
                "scenarios": ["Path A", "Path B", "Path C"],
                "comparison": "scenario",
                "all_scenarios": True,
                "start_year": 2050,
                "end_year": 2050,
                "entity_confidence": {"variable": 0.9, "region": 0.9},
            }

        mgr = self._build_manager(extracted)
        mgr.entity_extractor.available_variables = ["Synthetic Output"]
        mgr.entity_extractor.available_regions = ["R1", "R2"]
        mgr.entity_extractor.available_scenarios = ["Path A", "Path B", "Path C"]
        mgr.entity_extractor.available_models = []
        mgr.last_entities = {
            "action": "plot",
            "variable": "Synthetic Output",
            "region": "R1",
            "scenarios": ["Path A", "Path B"],
            "comparison": "scenario",
            "all_scenarios": False,
            "start_year": 2050,
            "end_year": 2050,
        }
        data_agent = _AgentStub(response="data handled")
        plot_agent = _AgentStub(response="plot handled")
        mgr.agents = {
            "data_query": data_agent,
            "data_plotting": plot_agent,
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        response = mgr._route_single("Now show the same comparison for R2.")

        self.assertEqual(response, mgr.agents["data_query"].response)
        self.assertEqual(plot_agent.calls, 0)
        self.assertEqual(mgr.agents["data_query"].last_entities["region"], "R2")
        self.assertEqual(mgr.agents["data_query"].last_entities["scenarios"], ["Path A", "Path B"])
        self.assertEqual(mgr.agents["data_query"].last_entities["comparison"], "scenario")
        self.assertFalse(mgr.agents["data_query"].last_entities["all_scenarios"])
        self.assertNotIn("scenario", mgr.agents["data_query"].last_entities)

    def test_scenario_comparison_synonyms_keep_bounded_pair(self):
        def extracted(query):
            return {
                "action": "query",
                "variable": "Metric",
                "region": "R1",
                "scenario": "Path B" if "Path B" in query else None,
                "entity_confidence": {"variable": 0.9, "region": 0.9, "scenario": 0.9},
            }

        mgr = self._build_manager(extracted)
        mgr.entity_extractor.available_variables = ["Metric"]
        mgr.entity_extractor.available_regions = ["R1"]
        mgr.entity_extractor.available_scenarios = ["Path A", "Path B", "Path C"]
        mgr.entity_extractor.available_models = []
        plot_agent = _AgentStub(response="plot handled")
        mgr.agents = {
            "data_query": _AgentStub(response="data"),
            "data_plotting": plot_agent,
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        for query in ("Contrast it with Path B.", "Show the difference against Path B."):
            with self.subTest(query=query):
                mgr.last_entities = {
                    "variable": "Metric", "region": "R1", "scenario": "Path A"
                }
                response = mgr._route_single(query)
                self.assertEqual(response, mgr.agents["data_query"].response)
                self.assertEqual(mgr.agents["data_query"].last_entities["scenarios"], ["Path A", "Path B"])
                self.assertEqual(mgr.agents["data_query"].last_entities["comparison"], "scenario")
                self.assertFalse(mgr.agents["data_query"].last_entities.get("all_scenarios"))

    def test_compound_variable_descendant_is_not_replaced_by_literal_parent(self):
        mgr = self._build_manager({
            "action": "query",
            "variable": "Family|Carrier",
            "region": "R1",
            "scenario": "Path",
            "entity_confidence": {"variable": 0.9, "region": 0.9, "scenario": 0.9},
        })
        mgr.entity_extractor.available_variables = ["Family", "Family|Carrier"]
        mgr.entity_extractor.available_regions = ["R1"]
        mgr.entity_extractor.available_scenarios = ["Path"]
        mgr.entity_extractor.available_models = []
        data_agent = _AgentStub(response="data handled")
        mgr.agents = {
            "data_query": data_agent,
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        response = mgr._route_single("Show Family Carrier for R1 under Path.")

        self.assertEqual(response, "data handled")
        self.assertEqual(data_agent.last_entities["variable"], "Family|Carrier")

    def test_broad_parent_query_does_not_keep_unmentioned_child(self):
        mgr = self._build_manager({
            "action": "query",
            "variable": "Family|Carrier",
            "region": "R1",
            "scenario": "Path",
            "entity_confidence": {"variable": 0.9, "region": 0.9, "scenario": 0.9},
        })
        mgr.entity_extractor.available_variables = ["Family", "Family|Carrier"]
        mgr.entity_extractor.available_regions = ["R1"]
        mgr.entity_extractor.available_scenarios = ["Path"]
        mgr.entity_extractor.available_models = []
        data_agent = _AgentStub(response="data handled")
        mgr.agents = {
            "data_query": data_agent,
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        response = mgr._route_single("Show Family for R1 under Path.")

        self.assertEqual(response, "data handled")
        self.assertEqual(data_agent.last_entities["variable"], "Family")

    def test_supported_direct_child_beats_unmentioned_intermediate_branch(self):
        mgr = self._build_manager({
            "action": "query",
            "variable": "Family|Sector|Carrier",
            "region": "R1",
            "scenario": "Path",
            "entity_confidence": {"variable": 0.9, "region": 0.9, "scenario": 0.9},
        })
        mgr.entity_extractor.available_variables = [
            "Family", "Family|Carrier", "Family|Sector|Carrier",
        ]
        mgr.entity_extractor.available_regions = ["R1"]
        mgr.entity_extractor.available_scenarios = ["Path"]
        mgr.entity_extractor.available_models = []
        data_agent = _AgentStub(response="data handled")
        mgr.agents = {
            "data_query": data_agent,
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        mgr._route_single("Show Family Carrier for R1 under Path.")

        self.assertEqual(data_agent.last_entities["variable"], "Family|Carrier")

    def test_previous_scope_comparison_requires_matching_noncompared_scope(self):
        mgr = self._build_manager({})
        mgr.previous_entities = {
            "variable": "Metric A", "region": "R1", "scenario": "Path",
            "start_year": 2040, "end_year": 2050,
        }

        changed_variable = {
            "variable": "Metric B", "region": "R2", "scenario": "Path",
            "start_year": 2040, "end_year": 2050,
        }
        changed_year = {
            "variable": "Metric A", "region": "R2", "scenario": "Path",
            "start_year": 2050, "end_year": 2050,
        }

        self.assertEqual(mgr._render_previous_scope_comparison(changed_variable), "")
        self.assertEqual(mgr._render_previous_scope_comparison(changed_year), "")

    def test_explicit_region_comparison_preserves_unchanged_plural_scenarios(self):
        mgr = self._build_manager({
            "action": "plot", "variable": "Metric", "regions": ["R1", "R2"],
            "start_year": 2040, "end_year": 2050,
        })
        mgr.entity_extractor.available_variables = ["Metric"]
        mgr.entity_extractor.available_regions = ["R1", "R2"]
        mgr.entity_extractor.available_scenarios = ["Path A", "Path B"]
        mgr.entity_extractor.available_models = []
        mgr.previous_entities = {
            "variable": "Metric", "region": "R1", "scenarios": ["Path A", "Path B"],
            "start_year": 2040, "end_year": 2050,
        }
        mgr.last_entities = {
            "variable": "Metric", "region": "R2", "scenarios": ["Path A", "Path B"],
            "start_year": 2040, "end_year": 2050,
        }
        plot_agent = _AgentStub(response="plot handled")
        mgr.agents = {
            "data_query": _AgentStub(response="data"),
            "data_plotting": plot_agent,
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        scope_before_plot = dict(mgr.last_entities)
        redirected = mgr._route_single('Plot the comparison.')

        self.assertIn("I do not generate figures in the chat", redirected)
        self.assertIn("https://iamparis.eu/results", redirected)
        self.assertEqual(mgr.agents["data_plotting"].calls, 0)
        self.assertEqual(mgr.last_entities, scope_before_plot)
        self.assertEqual(mgr.last_route_decision["reason"], "charts delegated to data explorer")

    def test_runtime_catalog_drives_comparison_navigation_answer_and_link(self):
        mgr = self._build_manager({})
        target_url = "https://example.test/widgets/compare"
        mgr.shared_resources = {
            "models": [],
            "link_catalog": [{
                "title": "Widget Comparison",
                "url": target_url,
                "category": "tools",
                "item_type": "route",
                "keywords": [
                    "Side-by-side widget comparison",
                    "Use when a visitor wants to compare widgets",
                ],
                "verified_direct_url": True,
            }],
        }
        general_agent = _AgentStub(response="ungrounded")
        mgr.agents = {
            "data_query": _AgentStub(response="data"),
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": _AgentStub(response="model"),
            "general_qa": general_agent,
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        for query in (
            "Open the widget comparison page.",
            "Also give me the page used specifically to compare widgets.",
        ):
            response = mgr._route_single(query)
            self.assertIn(f"[Widget Comparison]({target_url})", response)
            self.assertEqual(mgr.last_links[0]["url"], target_url)
            self.assertEqual(mgr.last_route_decision["agent"], "general_qa")
            self.assertEqual(general_agent.calls, 0)

    def test_catalog_navigation_precedes_named_model_metadata_comparison(self):
        mgr = self._build_manager({})
        mgr.entity_extractor.available_models = ["Alpha 1.0", "Beta 2.0"]
        target_url = "https://example.test/widgets/compare"
        mgr.shared_resources = {
            "models": [
                {"modelName": "Alpha 1.0", "description": "Alpha description."},
                {"modelName": "Beta 2.0", "description": "Beta description."},
            ],
            "link_catalog": [{
                "title": "Widget Comparison",
                "url": target_url,
                "category": "tools",
                "item_type": "route",
                "keywords": ["Compare widgets side by side"],
                "verified_direct_url": True,
            }],
        }
        model_agent = _AgentStub(response="metadata comparison")
        mgr.agents = {
            "data_query": _AgentStub(response="data"),
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": model_agent,
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        response = mgr._route_single(
            "Open the Widget Comparison page to compare Alpha 1.0 and Beta 2.0."
        )

        self.assertIn(f"[Widget Comparison]({target_url})", response)
        self.assertEqual(mgr.last_route_decision["agent"], "general_qa")
        self.assertEqual(mgr.last_route_decision["source"], "deterministic")
        self.assertEqual(model_agent.calls, 0)

    def test_numeric_clarification_cleans_candidates_and_refreshes_route(self):
        mgr = self._build_manager({})
        mgr.clarification_context = {
            "original_query": "show an uncertain metric for R1",
            "base_query": "show an uncertain metric for R1",
            "agent_type": "data_query",
            "entities": {
                "region": "R1",
                "variable_candidates": ["Metric A", "Metric B"],
                "unmatched_variable_terms": ["uncertain"],
                "entity_confidence": {"variable": 0.35, "region": 0.95},
                "confidence": 0.65,
            },
            "suggested_variable": "Metric A",
            "suggested_options": ["Metric A", "Metric B"],
            "suggested_kind": "variable",
            "suggested_region": "R1",
            "suggested_scenario": "",
            "issued_turn": 0,
        }
        data_agent = _AgentStub(response="### Metric B in R1\n\nAnswer:\n\n| Year | Value |")
        mgr.agents = {
            "data_query": data_agent,
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        mgr._route_single("2")

        self.assertEqual(data_agent.last_entities["variable"], "Metric B")
        self.assertNotIn("variable_candidates", data_agent.last_entities)
        self.assertNotIn("unmatched_variable_terms", data_agent.last_entities)
        self.assertEqual(data_agent.last_entities["entity_confidence"]["variable"], 1.0)
        self.assertNotIn("variable_candidates", mgr.last_entities)
        self.assertEqual(mgr.last_route_decision["agent"], "data_query")
        self.assertEqual(mgr.last_route_decision["source"], "conversation_state")
        self.assertIn("resolved", mgr.last_route_decision["reason"])

    def test_second_clarification_preserves_successful_scope_and_uses_clean_attempt(self):
        mgr = self._build_manager({})
        successful_scope = {"variable": "Earlier Metric", "region": "R0"}
        mgr.last_entities = dict(successful_scope)
        mgr.clarification_context = {
            "original_query": "show an uncertain metric",
            "base_query": "show an uncertain metric",
            "agent_type": "data_query",
            "entities": {
                "variable_candidates": ["Metric A", "Metric B"],
                "unmatched_variable_terms": ["uncertain"],
                "entity_confidence": {"variable": 0.35},
            },
            "suggested_variable": "Metric A",
            "suggested_options": ["Metric A", "Metric B"],
            "suggested_kind": "variable",
            "suggested_region": "",
            "suggested_scenario": "",
            "issued_turn": 0,
        }
        next_prompt = (
            "Choose the scenario:\n\n"
            "1. `Path A`\n"
            "2. `Path B`\n\n"
            "Reply with a number (1-2)."
        )
        mgr.agents = {
            "data_query": _AgentStub(response=next_prompt),
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        response = mgr._route_single("1")

        self.assertEqual(response, next_prompt)
        self.assertEqual(mgr.last_entities, successful_scope)
        self.assertEqual(mgr.last_attempted_entities["variable"], "Metric A")
        self.assertNotIn("variable_candidates", mgr.last_attempted_entities)
        self.assertNotIn("unmatched_variable_terms", mgr.last_attempted_entities)
        self.assertIsNotNone(mgr.clarification_context)
        self.assertNotIn("variable_candidates", mgr.clarification_context["entities"])
        self.assertEqual(mgr.last_route_decision["source"], "conversation_state")
        self.assertIn("additional clarification required", mgr.last_route_decision["reason"])

    def test_missing_variable_prompt_keeps_resolved_scope_for_followup(self):
        mgr = self._build_manager({
            "action": "query",
            "region": "R1",
            "start_year": 2050,
            "end_year": 2050,
            "entity_confidence": {"region": 0.9, "years": 0.9},
        })
        mgr.entity_extractor.available_variables = ["Metric"]
        mgr.entity_extractor.available_regions = ["R1", "R2"]
        mgr.entity_extractor.available_scenarios = ["Path"]
        mgr.entity_extractor.available_models = []
        prompt = "I found the region `R1`. Which variable should I use?"
        mgr.agents = {
            "data_query": _AgentStub(response=prompt),
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        response = mgr._route_single("Show data for R1 in 2050.")

        self.assertEqual(response, prompt)
        self.assertIsNotNone(mgr.clarification_context)
        self.assertEqual(mgr.clarification_context["suggested_kind"], "variable")
        self.assertEqual(mgr.clarification_context["entities"]["region"], "R1")
        self.assertEqual(mgr.clarification_context["entities"]["start_year"], 2050)
        self.assertEqual(mgr.clarification_context["suggested_options"], [])

    def test_pending_clarification_exposes_resolved_region_without_overwriting_success(self):
        mgr = self._build_manager({
            "action": "query",
            "region": "IND",
            "entity_confidence": {"region": 0.95},
        })
        mgr.entity_extractor.available_variables = ["Final Energy", "Secondary Energy|Electricity"]
        mgr.entity_extractor.available_regions = ["IND"]
        mgr.entity_extractor.available_scenarios = ["Path"]
        mgr.entity_extractor.available_models = []
        prompt = "I found the region `IND`. Which variable should I use?"
        mgr.agents = {
            "data_query": _AgentStub(response=prompt),
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        mgr._route_single("electricity for India")

        self.assertEqual(mgr.last_entities, {})
        self.assertEqual(mgr.response_entities()["region"], "IND")
        payload = mgr.pending_clarification_payload()
        self.assertEqual(payload["missing_dimension"], "variable")
        self.assertEqual(payload["base_scope"]["region"], "IND")

    def test_typed_state_records_failed_attempt_without_replacing_success(self):
        mgr = self._build_manager({})
        mgr._persist_last_entities(
            {"variable": "Metric", "region": "R1"},
            "### Metric in R1\n\nAnswer:\n1",
        )

        mgr._persist_last_entities(
            {"variable": "Metric", "region": "R2"},
            "No data found for this exact combination.",
        )

        self.assertEqual(mgr.last_entities["region"], "R1")
        self.assertEqual(mgr.last_attempted_entities["region"], "R2")
        self.assertEqual(mgr.conversation_state.active_scope["region"], "R1")

    def test_mixed_clarification_choice_updates_its_own_dimension(self):
        mgr = self._build_manager({})
        mixed_prompt = (
            "Closest valid options:\n\n"
            "1. region `R2`\n\n"
            "2. scenario `Path B`\n\n"
            "Reply with `1` or `2`."
        )
        mgr._update_clarification_context(
            "data_query",
            "show Metric for R1",
            mixed_prompt,
            {"variable": "Metric", "region": "R1"},
        )
        self.assertEqual(
            mgr.clarification_context["suggested_option_kinds"],
            ["region", "scenario"],
        )
        data_agent = _AgentStub(response="### Metric in R1\n\nAnswer:\n\n| Year | Value |")
        mgr.agents = {
            "data_query": data_agent,
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        mgr._route_single("2")

        self.assertEqual(data_agent.last_entities["region"], "R1")
        self.assertEqual(data_agent.last_entities["scenario"], "Path B")
        self.assertNotEqual(data_agent.last_entities["region"], "Path B")
        self.assertEqual(data_agent.last_entities["entity_confidence"]["scenario"], 1.0)
        self.assertEqual(mgr.last_route_decision["source"], "conversation_state")

    def test_plot_recovery_prompt_uses_typed_clarification_state(self):
        mgr = self._build_manager({})
        mgr.entity_extractor.available_variables = ["Metric A", "Metric B"]
        mgr.entity_extractor.available_regions = ["R1", "R2"]
        mgr.entity_extractor.available_scenarios = ["S1", "S2"]
        mgr.entity_extractor.available_models = []
        prompt = (
            "No rows matched.\n"
            "- Closest variables: `Metric A`, `Metric B`\n"
            "- Closest regions: `R2`\n"
            "- Closest scenarios: `S2`\n\n"
            "Reply with the option you want to use."
        )
        mgr._update_clarification_context(
            "data_plotting",
            "plot unknown scope",
            prompt,
            {"action": "plot", "region": "R1", "scenario": "S1"},
        )

        self.assertEqual(
            mgr.clarification_context["suggested_option_kinds"],
            ["variable", "variable", "region", "scenario"],
        )
        updated = mgr._route_single("Change only the region to R2.")
        self.assertIn("Updated region to `R2`", updated)

        plot_agent = _AgentStub(response="plot handled")
        mgr.agents = {
            "data_plotting": plot_agent,
            "data_query": _AgentStub(response="data"),
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }
        mgr._route_single("1")

        self.assertEqual(plot_agent.calls, 0)
        self.assertEqual(mgr.agents["data_query"].last_entities["variable"], "Metric A")
        self.assertEqual(mgr.agents["data_query"].last_entities["region"], "R2")
        self.assertEqual(mgr.agents["data_query"].last_entities["scenario"], "S1")

    def test_named_clarification_option_is_resolved_unambiguously(self):
        mgr = self._build_manager({})

        self.assertEqual(
            mgr._extract_named_option_choice(
                "use Path B",
                ["R2", "Path B", "Path C"],
            ),
            1,
        )
        self.assertIsNone(
            mgr._extract_named_option_choice("Path", ["Path B", "Path C"]),
        )

    def test_plot_reference_does_not_become_fuzzy_clarification_value(self):
        mgr = self._build_manager({
            "action": "plot",
            "region": "THA",
            "entity_confidence": {"action": 0.9, "region": 0.6},
        })
        prompt = (
            "Closest valid options:\n\n"
            "1. region `R2`\n\n"
            "2. scenario `Path B`\n\n"
            "Reply with `1` or `2`."
        )
        mgr._update_clarification_context(
            "data_query",
            "show uncertain data for R1",
            prompt,
            {"variable": "Metric", "region": "R1"},
        )
        data_agent = _AgentStub(response="should not run")
        mgr.agents = {
            "data_query": data_agent,
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        response = mgr._route_single("Plot that.")

        self.assertIn("[IAM PARIS data explorer]", response)
        self.assertEqual(data_agent.calls, 0)
        self.assertEqual(mgr.agents["data_plotting"].calls, 0)
        self.assertIsNone(mgr.clarification_context)
        self.assertEqual(mgr.last_route_decision["source"], "deterministic")

    def test_plot_reference_paraphrases_preserve_pending_clarification(self):
        prompt = (
            "Closest valid options:\n\n"
            "1. region `R2`\n\n"
            "2. scenario `Path B`\n\n"
            "Reply with `1` or `2`."
        )
        for query in (
            "Visualize that.",
            "Plot this one.",
            "Graph this one.",
            "Show that as a chart.",
        ):
            with self.subTest(query=query):
                mgr = self._build_manager({
                    "action": "plot",
                    "region": "THA",
                    "entity_confidence": {"action": 0.9, "region": 0.6},
                })
                mgr._update_clarification_context(
                    "data_query",
                    "show uncertain data for R1",
                    prompt,
                    {"variable": "Metric", "region": "R1"},
                )
                data_agent = _AgentStub(response="should not run")
                mgr.agents = {
                    "data_query": data_agent,
                    "data_plotting": _AgentStub(response="plot"),
                    "model_explanation": _AgentStub(response="model"),
                    "general_qa": _AgentStub(response="general"),
                    "modelling_suggestions": _AgentStub(response="suggest"),
                }

                response = mgr._route_single(query)

                self.assertIn("[IAM PARIS data explorer]", response)
                self.assertEqual(data_agent.calls, 0)
                self.assertEqual(mgr.agents["data_plotting"].calls, 0)
                self.assertIsNone(mgr.clarification_context)
                self.assertEqual(
                    mgr.last_route_decision["reason"],
                    "charts delegated to data explorer",
                )

    def test_scope_patch_during_clarification_keeps_pending_choice(self):
        mgr = self._build_manager({
            "variable": "Unrelated Metric",
            "scenario": "Path B",
            "entity_confidence": {"variable": 0.4, "scenario": 0.9},
        })
        mgr.entity_extractor.available_variables = ["Metric A", "Metric B", "Unrelated Metric"]
        mgr.entity_extractor.available_regions = ["R1"]
        mgr.entity_extractor.available_scenarios = ["Path A", "Path B"]
        mgr.entity_extractor.available_models = []
        prompt = (
            "Choose the variable:\n\n"
            "1. `Metric A`\n"
            "2. `Metric B`\n\n"
            "Reply with a number (1-2)."
        )
        mgr._update_clarification_context(
            "data_query",
            "show an uncertain metric for R1 under Path A",
            prompt,
            {
                "region": "R1",
                "scenario": "Path A",
                "variable_candidates": ["Metric A", "Metric B"],
                "entity_confidence": {"region": 0.9, "scenario": 0.9, "variable": 0.4},
            },
        )
        data_agent = _AgentStub(response="data handled")
        mgr.agents = {
            "data_query": data_agent,
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        response = mgr._route_single("Now use Path B but keep everything else.")

        self.assertIn("Updated scenario to `Path B`", response)
        self.assertIn(prompt, response)
        self.assertEqual(data_agent.calls, 0)
        self.assertIsNotNone(mgr.clarification_context)
        self.assertEqual(mgr.clarification_context["entities"]["scenario"], "Path B")
        self.assertEqual(
            mgr.clarification_context["suggested_options"],
            ["Metric A", "Metric B"],
        )

        resolved = mgr._route_single("2")

        self.assertEqual(resolved, "data handled")
        self.assertEqual(data_agent.last_entities["variable"], "Metric B")
        self.assertEqual(data_agent.last_entities["scenario"], "Path B")
        self.assertNotIn("variable_candidates", data_agent.last_entities)

    def test_variable_patch_during_region_clarification_survives_confirmation(self):
        mgr = self._build_manager({"variable": "Metric B"})
        mgr.entity_extractor.available_variables = ["Metric A", "Metric B"]
        mgr.entity_extractor.available_regions = ["R1", "R2"]
        mgr.entity_extractor.available_scenarios = ["Path"]
        mgr.entity_extractor.available_models = []
        prompt = (
            "Choose the region:\n\n"
            "1. `R1`\n"
            "2. `R2`\n\n"
            "Reply with a number (1-2)."
        )
        mgr._update_clarification_context(
            "data_query",
            "show Metric A in an uncertain region",
            prompt,
            {
                "variable": "Metric A",
                "region_candidates": ["R1", "R2"],
                "entity_confidence": {"variable": 0.9, "region": 0.4},
            },
        )
        data_agent = _AgentStub(response="data handled")
        mgr.agents = {
            "data_query": data_agent,
            "data_plotting": _AgentStub(response="plot"),
            "model_explanation": _AgentStub(response="model"),
            "general_qa": _AgentStub(response="general"),
            "modelling_suggestions": _AgentStub(response="suggest"),
        }

        response = mgr._route_single("Change the variable to Metric B.")

        self.assertIn("Updated variable to `Metric B`", response)
        self.assertEqual(mgr.clarification_context["entities"]["variable"], "Metric B")
        self.assertEqual(mgr.clarification_context["suggested_variable"], "Metric B")

        resolved = mgr._route_single("1")

        self.assertEqual(resolved, "data handled")
        self.assertEqual(data_agent.last_entities["variable"], "Metric B")
        self.assertEqual(data_agent.last_entities["region"], "R1")

    def test_unknown_scope_patch_is_rejected_without_losing_clarification(self):
        mgr = self._build_manager({})
        mgr.entity_extractor.available_variables = ["Metric A", "Metric B"]
        mgr.entity_extractor.available_regions = ["R1", "R2"]
        mgr.entity_extractor.available_scenarios = ["Path"]
        mgr.entity_extractor.available_models = []
        prompt = (
            "Choose the variable:\n\n"
            "1. `Metric A`\n"
            "2. `Metric B`\n\n"
            "Reply with a number (1-2)."
        )
        mgr._update_clarification_context(
            "data_query",
            "show an uncertain metric for R1",
            prompt,
            {
                "region": "R1",
                "variable_candidates": ["Metric A", "Metric B"],
                "entity_confidence": {"region": 0.9, "variable": 0.4},
            },
        )
        mgr.agents = {"data_query": _AgentStub(response="should not run")}

        response = mgr._route_single("Switch the region to Atlantis.")

        self.assertIn("could not resolve `Atlantis` as an available region", response)
        self.assertIn(prompt, response)
        self.assertEqual(mgr.clarification_context["entities"]["region"], "R1")
        self.assertEqual(
            mgr.clarification_context["suggested_options"],
            ["Metric A", "Metric B"],
        )
        self.assertEqual(mgr.agents["data_query"].calls, 0)
        self.assertEqual(
            mgr.last_route_decision["reason"],
            "invalid clarification scope patch rejected",
        )

    def test_explicit_patch_of_pending_dimension_confirms_named_value(self):
        mgr = self._build_manager({})
        mgr.entity_extractor.available_variables = ["Metric"]
        mgr.entity_extractor.available_regions = ["R1", "R2"]
        mgr.entity_extractor.available_scenarios = ["Path"]
        mgr.entity_extractor.available_models = []
        prompt = (
            "Choose the region:\n\n"
            "1. `R1`\n"
            "2. `R2`\n\n"
            "Reply with a number (1-2)."
        )
        mgr._update_clarification_context(
            "data_query",
            "show Metric in an uncertain region",
            prompt,
            {
                "variable": "Metric",
                "region_candidates": ["R1", "R2"],
                "entity_confidence": {"variable": 0.9, "region": 0.4},
            },
        )
        data_agent = _AgentStub(response="data handled")
        mgr.agents = {"data_query": data_agent}

        response = mgr._route_single("Switch the region to R2.")

        self.assertEqual(response, "data handled")
        self.assertEqual(data_agent.last_entities["variable"], "Metric")
        self.assertEqual(data_agent.last_entities["region"], "R2")
        self.assertEqual(data_agent.last_entities["entity_confidence"]["region"], 1.0)
        self.assertIsNone(mgr.clarification_context)


if __name__ == "__main__":
    unittest.main()
