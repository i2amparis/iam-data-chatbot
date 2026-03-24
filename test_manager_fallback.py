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

    def test_clarification_expires_after_one_missed_turn(self):
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
        mgr.current_turn = 3

        response = mgr._route_single("2")
        self.assertIn("don't have an active numbered choice", response.lower())
        self.assertIsNone(mgr.clarification_context)
        self.assertEqual(data_query_agent.calls, 0)

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


if __name__ == "__main__":
    unittest.main()
