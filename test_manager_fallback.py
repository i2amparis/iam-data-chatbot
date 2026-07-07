import logging
import unittest

from manager import MultiAgentManager
from main import _normalize_cli_query


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

        response = mgr._route_single("plot solar capacity for EU")

        self.assertEqual(response, "plot handled")
        self.assertEqual(plot_agent.calls, 1)
        self.assertEqual(mgr.last_route_decision["agent"], "data_plotting")
        self.assertEqual(mgr.last_route_decision["source"], "deterministic")

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

        self.assertEqual(response, "plot handled")
        self.assertEqual(plot_agent.calls, 1)
        self.assertEqual(mgr.last_route_decision["agent"], "data_plotting")
        self.assertEqual(mgr.last_route_decision["reason"], "comparison plot request")

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

        self.assertEqual(plot_agent.last_entities["variable"], "Capacity|Electricity|Wind")
        self.assertEqual(
            plot_agent.last_entities["variables"][:2],
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

        self.assertEqual(response, "plot handled")
        self.assertEqual(plot_agent.calls, 1)
        self.assertEqual(mgr.last_route_decision["agent"], "data_plotting")

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

        self.assertEqual(response, "plot handled")
        self.assertEqual(plot_agent.calls, 1)
        self.assertEqual(mgr.last_route_decision["agent"], "data_plotting")

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

        response = mgr._route_single("plot it")
        self.assertEqual(response, "plot handled")
        self.assertEqual(
            plot_agent.last_query,
            "plot Secondary Energy|Electricity for IND under PR_Baseline",
        )

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

        self.assertEqual(response, "plot handled")
        self.assertEqual(plot_agent.last_query, "plot compare Emissions|CO2 for World under Policy versus baseline")

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
                }
            ],
        }

        response = mgr._grounded_site_navigation_answer("where can I find Climate Watch", {})

        self.assertIn("[Climate Watch](https://iamparis.eu/application_library)", response)
        self.assertIn("Search for: Climate Watch", response)
        self.assertIn("open the Application Library and search for: Climate Watch", response)

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
                "keywords": ["results"],
            }
        ]

        updated = mgr._ensure_results_link_for_data_answer(links, catalog, "data_query")

        self.assertTrue(any(link["url"] == "https://iamparis.eu/results" for link in updated))

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

        self.assertEqual(response, "plot handled")
        self.assertEqual(plot_agent.last_entities["variable"], "Emissions|CO2")
        self.assertEqual(plot_agent.last_entities["scenarios"], ["Baseline", "Policy"])

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
        for phrase in ("under PR_NDC_CP", "now for CHN", "and under PR_CurPol_CP?", "for India"):
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


if __name__ == "__main__":
    unittest.main()
