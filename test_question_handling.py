"""Regression tests for question handling with a small local LLM.

Covers the knowledge base (YAML definitions, model names, glossary, batched
embedding), the grounded general-QA prompt, glossary definition answers,
yes/no without a pending question, and analytical data questions.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

from test_response_fixes import _record, build_manager


def _series(region, scenario, model, years, variable="Primary Energy|Coal", unit="EJ/yr"):
    return {
        "workspace_code": "world-headed", "variable": variable, "region": region,
        "scenario": scenario, "modelName": model, "unit": unit, "years": years,
    }


class KnowledgeBaseTests(unittest.TestCase):
    def test_variable_entries_become_named_documents(self):
        from utils.yaml_loader import yaml_to_documents

        docs = yaml_to_documents({
            "/defs/variable/energy/final-energy.yaml": [
                {"Final Energy": {"description": "Total final energy consumption", "unit": "EJ/yr"}},
            ],
            "/defs/variable/energy/tag_fuels.yaml": [{"Fuels": [{"Coal": {"description": "coal"}}]}],
        })

        self.assertEqual(len(docs), 1)
        self.assertIn("`Final Energy`", docs[0].page_content)
        self.assertIn("Total final energy consumption", docs[0].page_content)
        self.assertIn("EJ/yr", docs[0].page_content)
        self.assertEqual(docs[0].metadata["variable"], "Final Energy")

    def test_region_groups_become_documents_and_native_regions_are_skipped(self):
        from utils.yaml_loader import yaml_to_documents

        docs = yaml_to_documents({
            "/defs/region/common.yaml": [
                {"common": ["World", {"International": {"description": "Bunker fuels"}}]},
            ],
            "/defs/region/native_regions/GCAM/GCAM_7.0.yaml": [
                {"GCAM 7.0": [{"GCAM 7.0|Africa": {"countries": ["Kenya"]}}]},
            ],
        })

        texts = [doc.page_content for doc in docs]
        self.assertEqual(len(docs), 2)
        self.assertTrue(any("`World`" in text for text in texts))
        self.assertTrue(any("`International`" in text and "Bunker fuels" in text for text in texts))

    def test_real_definitions_produce_documents(self):
        from pathlib import Path
        from utils.yaml_loader import load_all_yaml_files, yaml_to_documents

        variables = yaml_to_documents(load_all_yaml_files(str(Path("definitions/variable").resolve())))
        self.assertGreater(len(variables), 100)
        self.assertTrue(all(doc.page_content.strip() for doc in variables))

    def test_definition_cache_key_includes_conversion_version(self):
        import main

        with patch.object(main, "YAML_DOCUMENTS_VERSION", "test-a"):
            first = main._definition_source_signature()
        with patch.object(main, "YAML_DOCUMENTS_VERSION", "test-b"):
            second = main._definition_source_signature()
        self.assertNotEqual(first, second)

    def test_model_documents_name_the_model(self):
        from main import docs_from_records

        docs = docs_from_records([{"modelName": "ATOM", "description": "Agent-based adoption model."}])
        self.assertTrue(docs[0].page_content.startswith("Model ATOM: Agent-based"))

    def test_knowledge_documents_include_glossary(self):
        from main import knowledge_documents

        docs = knowledge_documents([], [], [])
        self.assertTrue(any("Glossary — Carbon budget" in doc.page_content for doc in docs))

    def test_index_is_embedded_in_batches(self):
        import tempfile
        import main
        from langchain.schema import Document

        docs = [Document(page_content=f"doc {i}", metadata={}) for i in range(70)]
        store = MagicMock()
        fake_faiss = MagicMock()
        fake_faiss.from_documents.return_value = store
        embeddings = type("Embeddings", (), {"model": "m", "dimensions": 3})()
        previous = os.getcwd()
        with tempfile.TemporaryDirectory() as directory, patch("main.FAISS", fake_faiss):
            try:
                os.chdir(directory)
                main.build_faiss_index(docs, embeddings)
            finally:
                os.chdir(previous)
        self.assertEqual(len(fake_faiss.from_documents.call_args[0][0]), main.EMBEDDING_BATCH_SIZE)
        self.assertEqual(store.add_documents.call_count, 2)
        added = sum(len(call[0][0]) for call in store.add_documents.call_args_list)
        self.assertEqual(added + main.EMBEDDING_BATCH_SIZE, 70)


class PromptTests(unittest.TestCase):
    def test_general_qa_prompt_is_grounded_and_has_no_developer_notes(self):
        from agents import GENERAL_QA_SYSTEM_PROMPT

        self.assertIn("{context}", GENERAL_QA_SYSTEM_PROMPT)
        self.assertIn("Never invent", GENERAL_QA_SYSTEM_PROMPT)
        self.assertNotIn("RuntimeContext", GENERAL_QA_SYSTEM_PROMPT)
        self.assertNotIn("Skill guidance", GENERAL_QA_SYSTEM_PROMPT)

    def test_router_prompt_has_no_skill_guidance(self):
        manager = build_manager()
        system = manager.routing_prompt.messages[0].prompt.template
        self.assertNotIn("Skill guidance", system)
        self.assertTrue(system.rstrip().endswith("Respond with ONLY the category name, nothing else."))

    def test_local_max_tokens_maps_to_num_predict(self):
        from llm_factory import get_chat_openai

        with patch.dict(os.environ, {"LOCAL_LLM_MODEL": "qwen3:0.6b"}):
            llm = get_chat_openai("qwen3:0.6b", max_tokens=123)
        self.assertEqual(llm.num_predict, 123)


class GlossaryTests(unittest.TestCase):
    def test_definition_questions_match_glossary_terms(self):
        from glossary import definition_terms

        cases = {
            "what does net zero mean": ["Net zero emissions"],
            "explain the NDC scenario": ["Nationally Determined Contribution (NDC)"],
            "what's the difference between NDC and current policies?": [
                "Nationally Determined Contribution (NDC)", "Current policies scenario",
            ],
            "what’s a carbon budget": ["Carbon budget"],
            "define SSP": ["Shared Socioeconomic Pathways (SSPs)"],
        }
        for question, expected in cases.items():
            with self.subTest(question=question):
                self.assertEqual([entry["term"] for entry in definition_terms(question)], expected)

    def test_data_and_listing_questions_are_not_definitions(self):
        from glossary import definition_terms

        for question in (
            "what is the carbon price in 2050",
            "what are the scenarios for the EU",
            "what are the scenarios",
            "what scenarios are available",
            "what is the highest carbon price in 2050?",
            "what does net zero mean for electricity prices",
            "what is final energy for India",
            "what is GCAM",
            "tell me about AFOLU emissions in Brazil",
        ):
            with self.subTest(question=question):
                self.assertEqual(definition_terms(question), [])

    def test_manager_answers_definitions_from_glossary(self):
        manager = build_manager()
        answer = manager.route_query("what does net zero mean?")

        self.assertEqual(manager.last_route_decision.get("reason"), "glossary definition")
        self.assertIn("### Net zero emissions", answer)
        self.assertIn("Source: IPCC AR6 glossary", answer)

    def test_yes_without_pending_question_does_not_reach_the_llm(self):
        manager = build_manager()
        answer = manager.route_query("yes")

        self.assertEqual(manager.last_route_decision.get("reason"), "yes/no without active question")
        self.assertIn("open question", answer)


class AnalyticalQuestionTests(unittest.TestCase):
    TS = [
        _series("IND", "Baseline", "GCAM", {"2020": 20, "2030": 22, "2050": 25}),
        _series("IND", "NDC", "GCAM", {"2020": 20, "2030": 12, "2050": 0.5}),
        _series("IND", "LTT", "TIMES", {"2020": 18, "2030": 10, "2050": 4}),
    ]

    def _ask(self, question):
        from data_utils import data_query

        return data_query(question, [], self.TS, forced_entities={
            "variable": "Primary Energy|Coal", "region": "IND",
        })

    def test_intent_detection(self):
        from data_utils import analysis_intent

        self.assertEqual(analysis_intent("which scenario has the lowest emissions in 2050?"), "min")
        self.assertEqual(analysis_intent("what is the highest carbon price"), "max")
        self.assertEqual(analysis_intent("average CO2 across models"), "mean")
        self.assertEqual(analysis_intent("by how much do emissions fall between 2020 and 2050"), "change")
        self.assertEqual(analysis_intent("Will Germany's electricity demand grow?"), "trend")
        self.assertEqual(analysis_intent("is coal phased out in India by 2050?"), "phaseout")
        self.assertEqual(analysis_intent("CO2 emissions for World in 2050"), "")

    def test_lowest_and_highest(self):
        self.assertIn("**Lowest in 2050:** 0.5 EJ/yr (GCAM, NDC), out of 3 series.",
                      self._ask("which scenario has the lowest coal use in India in 2050"))
        self.assertIn("**Highest in 2050:** 25.00 EJ/yr (GCAM, Baseline)",
                      self._ask("highest coal use in India in 2050"))

    def test_phase_out(self):
        answer = self._ask("is coal phased out in India by 2050?")
        self.assertIn("falls below 5% of its starting level in 1 of 3 series** (GCAM, NDC)", answer)

    def test_change_and_trend(self):
        change = self._ask("by how much does coal use change between 2020 and 2050 in India")
        self.assertIn("**Change from 2020 to 2050:** median -78%", change)
        trend = self._ask("will coal use in India decline?")
        self.assertIn("rises in 1 and falls in 2 of 3 series", trend)

    def test_plain_questions_keep_the_range_summary(self):
        self.assertIn("ranges from 0.5 (GCAM, NDC)", self._ask("coal use in India in 2050"))

    def test_ranking_question_is_not_a_catalogue_listing(self):
        from query_plan import build_query_plan

        self.assertNotEqual(
            build_query_plan("which scenario has the lowest emissions in 2050 for the world?").intent,
            "availability",
        )
        self.assertEqual(build_query_plan("which scenarios are available for the EU?").intent, "availability")

    def test_demand_and_phase_out_aliases(self):
        from canonical_aliases import preferred_variable_from_query

        catalogue = [
            "Final Energy|Electricity", "Secondary Energy|Electricity",
            "Primary Energy|Coal", "Price|Coal", "Exports|CCS coal",
        ]
        self.assertEqual(
            preferred_variable_from_query("Will Germany's electricity demand grow?", catalogue),
            "Final Energy|Electricity",
        )
        self.assertEqual(
            preferred_variable_from_query("is coal phased out in India by 2050?", catalogue),
            "Primary Energy|Coal",
        )


class VagueQuestionTests(unittest.TestCase):
    def test_vague_about_question_gets_examples_not_model_error(self):
        from data_utils import data_query

        answer = data_query("tell me something interesting about the data", [{"modelName": "GCAM"}], [
            _record("world-headed", "World", "Baseline", "GCAM", 1),
        ])
        self.assertNotIn("known model", answer)
        self.assertIn("which scenario has the lowest emissions", answer)

    def test_bare_region_lists_available_data(self):
        from data_utils import data_query

        answer = data_query("Greece", [], [
            _record("net-zero", "GREECE", "NZE", "TIMES", 5, variable="Capacity|Electricity|Solar", unit="GW"),
        ])
        self.assertIn("Data available for GREECE", answer)


def _multi(workspace, region, scenario, model, variable, unit, years):
    return {
        "workspace_code": workspace, "variable": variable, "region": region,
        "scenario": scenario, "modelName": model, "unit": unit, "years": years,
    }


SESSION_TS = [
    _multi("world-headed", region, scenario, model, variable, unit, years)
    for region in ("World", "IND")
    for scenario, model in (("Baseline", "GCAM"), ("NDC", "GCAM"), ("NDC", "MUSE"))
    for variable, unit, years in (
        ("Primary Energy|Coal", "EJ/yr", {"2020": 20, "2050": 10 if scenario == "NDC" else 25}),
        ("Price|Carbon", "US$2010/t CO2", {"2020": 5, "2050": 100 if scenario == "NDC" else 0}),
        ("Emissions|CH4", "Mt CH4/yr", {"2020": 300, "2050": 250}),
        ("Emissions|CO2", "Mt CO2/yr", {"2020": 30000, "2050": 20000 if scenario == "NDC" else 40000}),
    )
]
SESSION_MODELS = [{"modelName": "GCAM"}, {"modelName": "MUSE"}]


class SessionFollowupTests(unittest.TestCase):
    """Multi-turn behaviour found by the 2026-10-25 conversation tests."""

    def test_new_variable_is_not_overridden_by_carried_one(self):
        manager = build_manager(SESSION_TS, SESSION_MODELS)
        manager.route_query("will India's coal use fall by 2050?")
        answer = manager.route_query("what is the average carbon price in 2050 for the world?")

        self.assertIn("Price|Carbon in World", answer)
        self.assertNotIn("Primary Energy|Coal", answer)
        self.assertEqual(manager.last_entities.get("variable"), "Price|Carbon")

    def test_ranking_followup_ranks_previous_result(self):
        manager = build_manager(SESSION_TS, SESSION_MODELS)
        manager.route_query("CO2 emissions for World in 2050")
        answer = manager.route_query("which scenario is lowest?")

        self.assertEqual(manager.last_route_decision.get("reason"), "ranking of previous result")
        self.assertIn("**Lowest in 2050:** 20,000 Mt CO2/yr", answer)

    def test_provenance_question_is_not_a_catalogue_listing(self):
        manager = build_manager(SESSION_TS, SESSION_MODELS)
        manager.route_query("CO2 emissions for World in 2050")
        answer = manager.route_query("which model is this from?")

        self.assertEqual(manager.last_route_decision.get("reason"), "result provenance follow-up")
        self.assertIn("GCAM", answer)
        self.assertIn("MUSE", answer)

    def test_observed_single_scenario_is_not_a_filter_for_followups(self):
        ts = [
            _multi("w", "EU", "S1", "GCAM", "Final Energy", "EJ/yr", {"2050": 10}),
            _multi("w", "DEU", "S2", "MUSE", "Final Energy", "EJ/yr", {"2050": 3}),
        ]
        manager = build_manager(ts, SESSION_MODELS)
        manager.route_query("final energy for EU in 2050")
        self.assertIn("observed_dimensions", manager.last_entities)
        answer = manager.route_query("now for Germany")

        self.assertIn("Final Energy in DEU", answer)
        self.assertIn("3.00", answer)

    def test_two_named_variables_are_both_answered(self):
        manager = build_manager(SESSION_TS, SESSION_MODELS)
        answer = manager.route_query("show methane and CO2 for World in 2050")

        self.assertIn("### Emissions|CH4 in World", answer)
        self.assertIn("### Emissions|CO2 in World", answer)

    def test_unrecognized_input_gets_examples(self):
        manager = build_manager(SESSION_TS, SESSION_MODELS)
        answer = manager.route_query("asdfgh")

        self.assertEqual(manager.last_route_decision.get("reason"), "unrecognized input")
        self.assertIn("CO2 emissions for World in 2050", answer)

    def test_question_during_pending_choice_is_a_new_question(self):
        manager = build_manager(SESSION_TS, SESSION_MODELS)
        manager.clarification_context = {
            "agent_type": "data_query", "suggested_kind": "variable",
            "suggested_options": ["Primary Energy|Coal", "Price|Carbon"],
            "entities": {"region": "World"}, "issued_turn": 1, "original_query": "coal",
        }
        answer = manager.route_query("what does net zero mean?")
        self.assertIn("### Net zero emissions", answer)


class ModelAnswerTests(unittest.TestCase):
    def test_who_developed_it_is_a_focused_model_answer(self):
        from data_utils import _looks_like_model_info_request
        from model_profiles import _focused_profile_dimension

        self.assertTrue(_looks_like_model_info_request("who developed it?"))
        self.assertEqual(_focused_profile_dimension("who developed it?"), "developer")
        self.assertEqual(_focused_profile_dimension("what are its limitations?"), "limitations")

    def test_comparison_without_sector_data_claims_no_differences(self):
        from model_profiles import format_model_comparison_answer

        answer = format_model_comparison_answer(
            [{"name": "Alpha", "sectors": ["energy", "land"]}, {"name": "Beta", "sectors": []}],
            "compare the coverage of Alpha and Beta",
        )
        self.assertNotIn("Distinctive in", answer)

    def test_iam_paris_has_its_own_definition(self):
        from glossary import definition_terms

        self.assertEqual([t["term"] for t in definition_terms("what is IAM PARIS?")], ["IAM PARIS"])
        self.assertEqual(
            [t["term"] for t in definition_terms("what is an IAM?")], ["Integrated assessment model (IAM)"],
        )


class GeneralQaHistoryTests(unittest.TestCase):
    def test_history_only_for_questions_that_refer_back(self):
        from agents import _REFERS_BACK

        self.assertIsNone(_REFERS_BACK.search("why do energy transitions differ between countries?"))
        self.assertIsNotNone(_REFERS_BACK.search("why does it differ?"))
        self.assertIsNotNone(_REFERS_BACK.search("what about Asia?"))


class NoDataAndDeadEndTests(unittest.TestCase):
    """Items 2-4: accurate no-data reasons, unrecorded attributes, clarification loops."""

    def test_region_gap_reason_and_region_options(self):
        from data_utils import _matrix_recovery_prompt
        from data_metadata import DataMetadata

        records = [
            _multi("w", "EU", "S1", "GCAM", "Price|Coal", "US$2010/GJ", {"2050": 2.7}),
            _multi("w", "World", "S1", "GCAM", "Price|Agriculture|Corn|Index", "Index", {"2050": 1}),
            _multi("w", "World", "S2", "GCAM", "Emissions|CO2", "Mt CO2/yr", {"2050": 1}),
        ]
        prompt = _matrix_recovery_prompt(
            DataMetadata(records, []),
            "No data found for **Price|Coal** in region `World` for any scenario.",
            variable="Price|Coal", region="World",
        )
        self.assertIn("`Price|Coal` is not reported for `World` in any scenario", prompt)
        self.assertIn("1. region `EU`", prompt)
        self.assertNotIn("Corn", prompt)
        self.assertNotIn("scenario `", prompt)
        self.assertIn("Reply with `1`, or type the region you want.", prompt)

    def test_licence_questions_say_the_catalogue_has_no_such_field(self):
        manager = build_manager(SESSION_TS, SESSION_MODELS)
        answer = manager.route_query("which models are open source?")

        self.assertEqual(manager.last_route_decision.get("reason"), "model attribute not in catalogue")
        self.assertIn("does not record licensing", answer)
        self.assertIn("the models are open source", answer)
        named = manager.route_query("is GCAM open source?")
        self.assertIn("`GCAM`", named)

    def test_year_reply_during_pending_choice_is_kept(self):
        manager = build_manager(SESSION_TS, SESSION_MODELS)
        manager.clarification_context = {
            "agent_type": "data_query", "suggested_kind": "variable",
            "suggested_options": ["Primary Energy|Coal", "Price|Carbon"],
            "suggested_option_kinds": ["variable", "variable"],
            "entities": {"region": "World", "workspace_code": "world-headed"},
            "issued_turn": 0, "original_query": "coal for World", "base_query": "coal for World",
            "response": "Choose the variable: 1. `Primary Energy|Coal` 2. `Price|Carbon`",
        }
        noted = manager.route_query("by 2050")
        self.assertEqual(manager.last_route_decision.get("reason"), "year noted for pending clarification")
        self.assertIn("Noted: I'll use years", noted)
        self.assertEqual(manager.clarification_context["entities"].get("end_year"), 2050)

    def test_year_only_reply_detection(self):
        manager = build_manager(SESSION_TS, SESSION_MODELS)
        self.assertTrue(manager._is_year_only_reply("by 2040"))
        self.assertTrue(manager._is_year_only_reply("in 2050 please"))
        self.assertTrue(manager._is_year_only_reply("between 2030 and 2050"))
        self.assertFalse(manager._is_year_only_reply("Germany in 2050"))
        self.assertFalse(manager._is_year_only_reply("2"))


class StartupAndDataQualityTests(unittest.TestCase):
    def test_knowledge_index_failure_does_not_fail_startup(self):
        import fastapi_app

        with patch("fastapi_app.build_faiss_index", side_effect=RuntimeError("embedding server down")), \
                patch("fastapi_app.get_embeddings", return_value=MagicMock()):
            index = fastapi_app._build_knowledge_index([], [], [], "test")
        self.assertIsNone(index)
        self.assertIn("embedding server down", fastapi_app._knowledge_index_error)

    def test_labels_are_normalized_once(self):
        from record_labels import normalize_record_labels

        records = [
            {"variable": "Emissions|CO2 ", "region": "World", "modelName": "GCAM", "unit": "EJ/yr", "scenario": "Baseline"},
            {"variable": "Emissions|CO2", "region": "WORLD", "modelName": "gcam", "unit": "EJ/Yr", "scenario": "baseline"},
            {"variable": "Emissions|CO2", "region": "World", "modelName": "GCAM", "unit": "EJ/yr", "scenario": "Baseline"},
        ]
        normalize_record_labels(records)

        self.assertEqual({r["variable"] for r in records}, {"Emissions|CO2"})
        self.assertEqual({r["region"] for r in records}, {"World"})
        self.assertEqual({r["modelName"] for r in records}, {"GCAM"})
        self.assertEqual({r["unit"] for r in records}, {"EJ/yr"})
        # Scenario names are study identifiers: only whitespace is normalized.
        self.assertEqual({r["scenario"] for r in records}, {"Baseline", "baseline"})

    def test_runtime_context_normalizes_labels(self):
        from runtime_context import build_runtime_context
        import tempfile

        ts = [{"variable": "Population ", "region": "World", "modelName": "M", "years": {"2050": 1}}]
        with tempfile.TemporaryDirectory() as directory:
            build_runtime_context(models=[], ts=ts, vector_store=None, env={},
                                  metadata_cache_file=os.path.join(directory, "m.pkl"))
        self.assertEqual(ts[0]["variable"], "Population")

    def test_variable_templates_are_expanded(self):
        from variable_definitions import variable_definitions

        definitions = variable_definitions()
        self.assertIn("Emissions of carbon dioxide (CO2)", definitions["Emissions|CO2"]["description"])
        self.assertEqual(definitions["Emissions|CH4"]["unit"], "Mt CH4/yr")
        # A note about the CO2 balance is not copied onto CH4.
        self.assertNotIn("Emissions|CO2", definitions["Emissions|CH4"]["notes"])

    def test_variable_definition_questions(self):
        manager = build_manager(SESSION_TS, SESSION_MODELS)
        answer = manager.route_query("what does Emissions|CO2 include?")
        self.assertEqual(manager.last_route_decision.get("reason"), "variable definition")
        self.assertIn("Emissions of carbon dioxide (CO2)", answer)

        data = manager.route_query("what is Emissions|CO2 for India in 2050")
        self.assertNotEqual(manager.last_route_decision.get("reason"), "variable definition")
        self.assertIn("in IND", data)


class HundredQueryFindingsTests(unittest.TestCase):
    """Problems found by the 100-question local run (2026-10-25)."""

    def test_change_intent_survives_rewritten_followup(self):
        manager = build_manager(SESSION_TS, SESSION_MODELS)
        manager.route_query("Primary Energy|Coal for World in 2050")
        answer = manager.route_query("how much does it change between 2020 and 2050?")
        self.assertIn("**Change from 2020 to 2050:**", answer)

    def test_years_only_followup_reruns_previous_question(self):
        manager = build_manager(SESSION_TS, SESSION_MODELS)
        manager.route_query("Primary Energy|Coal for World in 2050")
        answer = manager.route_query("between 2020 and 2050")
        self.assertEqual(manager.last_route_decision.get("reason"), "year change for previous question")
        self.assertIn("Primary Energy|Coal in World", answer)
        self.assertIn("| 2020 |", answer)

    def test_different_model_is_not_presented_as_requested_one(self):
        from data_utils import _different_model_note

        self.assertIn("no exact `REMIND` entry", _different_model_note("tell me about REMIND", "REMIND-MFA"))
        self.assertEqual(_different_model_note("AIM", "AIM/CGE Global 2.0"), "")
        self.assertEqual(_different_model_note("GCAM", "GCAM 7.0"), "")

    def test_curated_profile_of_another_model_is_not_merged(self):
        manager = build_manager(models=[{"modelName": "REMIND-MFA", "description": "Material flow analysis."}])
        profiles = manager._runtime_model_profiles("tell me about REMIND-MFA")
        self.assertTrue(profiles)
        self.assertNotIn("macroeconomic", str(profiles[0].get("description", "")))

    def test_conceptual_questions_go_to_general_qa_even_mid_session(self):
        manager = build_manager(SESSION_TS, SESSION_MODELS)
        manager.route_query("carbon price for World in 2050")
        manager.route_query("why do models disagree?")
        self.assertEqual(manager.last_route_decision.get("reason"), "conceptual climate explanation")
        manager.route_query("what limits how fast renewables can grow?")
        self.assertEqual(manager.last_route_decision.get("reason"), "conceptual climate explanation")

    def test_listing_and_year_questions_are_not_conceptual(self):
        manager = build_manager(SESSION_TS, SESSION_MODELS)
        manager.route_query("what are the scenarios for the EU")
        self.assertNotEqual(manager.last_route_decision.get("reason"), "conceptual climate explanation")

    def test_mix_and_hydrogen_aliases(self):
        from canonical_aliases import preferred_variable_from_query

        catalogue = ["Primary Energy", "Population", "Secondary Energy|Electricity",
                     "Secondary Energy|Hydrogen", "Production|Hydrogen", "Exports|Hydrogen"]
        self.assertEqual(preferred_variable_from_query("energy mix of China in 2050", catalogue), "Primary Energy")
        self.assertEqual(
            preferred_variable_from_query("hydrogen production in Europe 2050", catalogue), "Secondary Energy|Hydrogen",
        )
        self.assertEqual(
            preferred_variable_from_query("electricity mix for Germany", catalogue), "Secondary Energy|Electricity",
        )

    def test_bare_study_name_gives_study_summary(self):
        ts = SESSION_TS + [_multi("eu-headed", "EU", "PR_WWH_CP", "GCAM", "Emissions|CO2", "Mt CO2/yr", {"2050": 1})]
        manager = build_manager(ts, SESSION_MODELS)
        answer = manager.route_query("Where is the EU headed?")
        self.assertEqual(manager.last_route_decision.get("reason"), "study summary")
        self.assertIn("### Where is the EU headed?", answer)

    def test_list_scenarios_in_active_study(self):
        ts = SESSION_TS + [_multi("post-glasgow", "World", "LTS_only", "GCAM", "Emissions|CO2", "Mt CO2/yr", {"2050": 1})]
        manager = build_manager(ts, SESSION_MODELS)
        manager.route_query("CO2 emissions for World in 2050")
        manager.route_query("Post-Glasgow targets")
        answer = manager.route_query("which scenarios are in this study?")
        self.assertEqual(manager.last_route_decision.get("reason"), "listing for active study")
        self.assertIn("LTS_only", answer)
        self.assertNotIn("Baseline", answer)


class FollowupGuidanceTests(unittest.TestCase):
    def test_narrowing_hint_is_shown_once_per_conversation(self):
        manager = build_manager(SESSION_TS, SESSION_MODELS)
        answer = "### Emissions|CO2 in EU\nScope: scenario `multiple`."
        first = manager._maybe_add_followup_guidance(answer, "show CO2 emissions for Europe", "data_query")
        second = manager._maybe_add_followup_guidance(answer, "same for China", "data_query")
        self.assertIn("Reply with a scenario, model, region, or year", first)
        self.assertNotIn("Reply with a scenario", second)

    def test_no_hint_when_table_already_explains_narrowing(self):
        manager = build_manager(SESSION_TS, SESSION_MODELS)
        answer = (
            "### Emissions|CO2 in EU\nScope: scenario `multiple`.\n"
            "Narrow by model or scenario to view the 26 remaining series."
        )
        self.assertNotIn(
            "Reply with a scenario",
            manager._maybe_add_followup_guidance(answer, "show CO2 emissions for Europe", "data_query"),
        )


class OpenItemFixesTests(unittest.TestCase):
    def test_reused_region_is_stated(self):
        manager = build_manager(SESSION_TS, SESSION_MODELS)
        manager.route_query("Primary Energy|Coal for IND in 2050")
        answer = manager.route_query("what is the highest carbon price in 2050?")
        self.assertTrue(answer.startswith("Using `IND` from your previous question"), answer[:120])
        followup = manager.route_query("what about 2020?")
        self.assertNotIn("from your previous question", followup)

    def test_single_typo_is_corrected_instead_of_a_menu(self):
        manager = build_manager(SESSION_TS, SESSION_MODELS)
        corrected, corrections = manager._correct_typos("carbon prise for World")
        self.assertEqual(corrections, [("prise", "price")])
        self.assertEqual(corrected, "carbon price for World")
        # Known and short words are never rewritten.
        self.assertEqual(manager._correct_typos("show me data please")[1], [])

    def test_one_edit_distance(self):
        from manager import MultiAgentManager

        one = MultiAgentManager._one_edit_apart
        self.assertTrue(one("capacty", "capacity"))
        self.assertTrue(one("capasity", "capacity"))
        self.assertTrue(one("emsisions", "emissions"))
        self.assertFalse(one("capcty", "capacity"))

    def test_new_topic_does_not_keep_previous_variable(self):
        ts = SESSION_TS + [
            _multi("world-headed", "EU", "Baseline", "GCAM", "Final Energy|Buildings", "EJ/yr", {"2050": 5}),
        ]
        manager = build_manager(ts, SESSION_MODELS)
        manager.route_query("Price|Carbon for World in 2050")
        self.assertTrue(manager._states_new_topic("energy in buildings for EU", manager.last_entities))
        self.assertFalse(manager._states_new_topic("only coal", manager.last_entities))
        self.assertFalse(manager._states_new_topic("same for China", manager.last_entities))

    def test_new_glossary_entries(self):
        from glossary import definition_terms, load_glossary

        terms = {entry["term"] for entry in load_glossary()}
        for term in ("Electrification", "Model intercomparison", "Renewable expansion constraints",
                     "Comparing results across studies"):
            self.assertIn(term, terms)
        self.assertEqual([t["term"] for t in definition_terms("what is electrification?")], ["Electrification"])


class TwoHundredFiftyQueryFindingsTests(unittest.TestCase):
    """Problems found by the 250-question local run (2026-10-25)."""

    def test_region_word_does_not_select_a_model(self):
        from model_aliases import match_model_name, resolve_model_candidates

        names = ["EU MFA", "China-MORE", "OPEN-PROM", "GCAM", "E3ME-FTT"]
        self.assertEqual(match_model_name("electricity sector emissions for EU in 2030", names), "")
        self.assertEqual(resolve_model_candidates("CO2 for China in 2050", names), [])
        self.assertEqual(match_model_name("info on EU MFA", names), "EU MFA")
        self.assertEqual(match_model_name("CO2 for World from GCAM", names), "GCAM")
        self.assertEqual(resolve_model_candidates("tell me about China-MORE", names), ["China-MORE"])

    def test_empty_answer_gives_examples_not_a_generic_request(self):
        from data_utils import _VAGUE_REQUEST_HELP

        manager = build_manager(SESSION_TS, SESSION_MODELS)
        self.assertTrue(manager._is_unsuccessful_response(_VAGUE_REQUEST_HELP))

    def test_data_source_and_update_questions(self):
        manager = build_manager(SESSION_TS, SESSION_MODELS)
        manager.shared_resources["results_timestamp"] = "2026-10-01 12:00"
        for question in ("what is the source of the data?", "when was the data last updated?",
                         "where does the data come from?"):
            answer = manager.route_query(question)
            self.assertEqual(manager.last_route_decision.get("reason"), "data source question", question)
            self.assertIn("IAM PARIS results database", answer)
            self.assertIn("2026-10-01 12:00", answer)
        manager.route_query("Primary Energy|Coal for World in 2050")
        manager.route_query("which model is this from?")
        self.assertNotEqual(manager.last_route_decision.get("reason"), "data source question")

    def test_dataset_year_coverage(self):
        manager = build_manager(SESSION_TS, SESSION_MODELS)
        for question in ("what years are covered?", "which years are available?"):
            answer = manager.route_query(question)
            self.assertIn("2020–2050", answer, question)
            self.assertNotIn("for this scope", answer)

    def test_forecast_and_ssp_glossary(self):
        from glossary import definition_terms

        self.assertEqual(
            [t["term"] for t in definition_terms("are the results forecasts?")], ["Projections, not forecasts"],
        )
        self.assertEqual(
            [t["term"] for t in definition_terms("explain SSP2")], ["Shared Socioeconomic Pathways (SSPs)"],
        )
        # "FORECAST" is also a model name: a bare "what is forecast" stays open.
        self.assertEqual(definition_terms("what is FORECAST?"), [])

    def test_how_to_use_is_a_capability_question(self):
        from data_utils import _looks_like_capability_question

        self.assertTrue(_looks_like_capability_question("how do I use this chatbot?"))

    def test_unknown_name_is_reported(self):
        manager = build_manager(SESSION_TS, SESSION_MODELS)
        for question, name in (("what is PRIMES?", "PRIMES"), ("what is FooBar 2.0?", "FooBar 2.0")):
            answer = manager.route_query(question)
            self.assertEqual(manager.last_route_decision.get("reason"), "unknown named subject", question)
            self.assertIn(f"`{name}`", answer)
        manager.route_query("what is GCAM?")
        self.assertNotEqual(manager.last_route_decision.get("reason"), "unknown named subject")
        manager.route_query("what is CCS?")
        self.assertEqual(manager.last_route_decision.get("reason"), "glossary definition")

    def test_uncatalogued_model_with_results_points_to_its_data(self):
        ts = SESSION_TS + [_multi("buildings", "EU", "Baseline", "PRIMES-Buildings", "Final Energy", "EJ/yr",
                                  {"2030": 1})]
        manager = build_manager(ts, SESSION_MODELS)
        answer = manager.route_query("what is PRIMES?")
        self.assertIn("`PRIMES-Buildings`", answer)
        self.assertNotIn("couldn't find", answer)

    def test_bare_model_name_describes_the_model(self):
        manager = build_manager(SESSION_TS, [{"modelName": "GCAM", "description": "A dynamic-recursive model."}])
        answer = manager.route_query("GCAM")
        self.assertIn("### GCAM", answer)

    def test_average_across_models_uses_previous_result(self):
        manager = build_manager(SESSION_TS, SESSION_MODELS)
        manager.route_query("Primary Energy|Coal for World in 2050")
        answer = manager.route_query("average across models")
        self.assertEqual(manager.last_route_decision.get("reason"), "ranking of previous result")
        self.assertIn("Average across", answer)

    def test_region_spelling_and_korea(self):
        from record_labels import normalize_record_labels
        from utils_query import extract_region_from_query

        records = [{"region": "South Korea"}, {"region": "South Korea"}, {"region": "South_Korea"},
                   {"variable": "Final Energy|Residential and Commercial"},
                   {"variable": "Final Energy|Residential and Commercial"},
                   {"variable": "Final Energy|Residential & Commercial"}]
        normalize_record_labels(records)
        self.assertEqual({r["region"] for r in records if "region" in r}, {"South Korea"})
        self.assertEqual(
            {r["variable"] for r in records if "variable" in r}, {"Final Energy|Residential and Commercial"},
        )
        self.assertEqual(extract_region_from_query("CO2 for Korea in 2050", {}, ["KOR", "World"]), "KOR")
        self.assertNotEqual(extract_region_from_query("CO2 for North Korea in 2050", {}, ["KOR", "World"]), "KOR")

    def test_only_refines_a_sector_variable(self):
        ts = SESSION_TS + [
            _multi("world-headed", "World", "Baseline", "GCAM", "Final Energy|Industry", "EJ/yr", {"2050": 50}),
            _multi("world-headed", "World", "Baseline", "GCAM", "Final Energy|Industry|Electricity", "EJ/yr",
                   {"2050": 20}),
        ]
        manager = build_manager(ts, SESSION_MODELS)
        manager.route_query("Final Energy|Industry for World in 2050")
        answer = manager.route_query("only electricity")
        self.assertIn("Final Energy|Industry|Electricity in World", answer)

    def test_power_and_transport_aliases(self):
        from canonical_aliases import preferred_variable_from_query

        catalogue = ["Secondary Energy|Electricity", "Secondary Energy|Electricity|Coal",
                     "Secondary Energy|Electricity|Gas", "Primary Energy|Coal", "Exports|Air transport",
                     "Final Energy|Transportation|Hydrogen", "Final Energy|Electricity"]
        self.assertEqual(preferred_variable_from_query("is coal power phased out in India by 2050?", catalogue),
                         "Secondary Energy|Electricity|Coal")
        self.assertEqual(preferred_variable_from_query("gas power generation for EU", catalogue),
                         "Secondary Energy|Electricity|Gas")
        self.assertEqual(preferred_variable_from_query("hydrogen in transport for EU in 2050", catalogue),
                         "Final Energy|Transportation|Hydrogen")

    def test_study_switch_without_data_describes_the_study(self):
        ts = SESSION_TS + [_multi("post-glasgow", "World", "LTS_only", "GCAM", "Emissions|CO2", "Mt CO2/yr",
                                  {"2050": 1})]
        manager = build_manager(ts, SESSION_MODELS)
        manager.route_query("Primary Energy|Coal for World in 2050")
        answer = manager.route_query("Post-Glasgow targets")
        self.assertEqual(manager.last_route_decision.get("reason"), "study summary")
        self.assertIn("is not reported in this study", answer)
        self.assertNotIn("Choose the variable", answer)

    def test_definition_and_data_request_are_split(self):
        manager = build_manager(SESSION_TS, SESSION_MODELS)
        self.assertEqual(
            manager._split_multi_intent("what is net zero and show CO2 emissions for EU in 2050"),
            ["what is net zero", "show CO2 emissions for EU in 2050"],
        )

    def test_which_is_better_compares_named_models(self):
        manager = build_manager(SESSION_TS, [{"modelName": "GCAM", "description": "Global model."},
                                             {"modelName": "MUSE", "description": "Agent-based model."}])
        manager.route_query("which is better for transport, MUSE or GCAM?")
        self.assertEqual(manager.last_route_decision.get("reason"), "multi-model metadata comparison")

    def test_raw_model_codes_show_catalogue_names(self):
        from model_aliases import UNLABELLED_MODEL_LABEL, display_model_label, register_model_display_names

        register_model_display_names(["42", "ICES-XPS 1.0", "ICES-XPS 1.1", "TIAM_Grantham",
                                      "TIAM_Grantham 3.2", "GCAM 7.0", "GCAM-PR 5.3"])
        try:
            self.assertEqual(display_model_label("42"), "42")
            self.assertEqual(display_model_label("ices"), "ICES-XPS")
            self.assertEqual(display_model_label("tiam"), "TIAM_Grantham")
            self.assertEqual(display_model_label("gcam"), "GCAM")
            self.assertEqual(display_model_label("12345"), UNLABELLED_MODEL_LABEL)
        finally:
            register_model_display_names([])
        self.assertEqual(display_model_label("42"), UNLABELLED_MODEL_LABEL)

    def test_case_variant_model_shows_catalogue_spelling(self):
        from model_aliases import display_model_label, register_model_display_names

        register_model_display_names(["GEMINI-E3 7.0", "GCAM 7.0"])
        try:
            self.assertEqual(display_model_label("Gemini-E3"), "GEMINI-E3")
            self.assertEqual(display_model_label("GEMINI-E3 7.0"), "GEMINI-E3 7.0")
        finally:
            register_model_display_names([])

    def test_by_2050_is_not_suggested_for_a_2050_answer(self):
        import fastapi_app

        def suggestions(query):
            manager = build_manager(SESSION_TS, SESSION_MODELS)
            answer = manager.route_query(query)
            return fastapi_app._suggested_next_questions(query, answer, manager)

        self.assertNotIn("By 2050", suggestions("Emissions|CO2 for World in 2050"))
        self.assertIn("By 2050", suggestions("Emissions|CO2 for World in 2020"))

    def test_fuel_price_and_buildings_demand_aliases(self):
        from canonical_aliases import preferred_variable_from_query

        catalogue = ["Price|Primary Energy|Oil", "Price|Secondary Energy|Liquids|Oil", "Price|Primary Energy|Gas",
                     "Price|Final Energy|Transportation|Gases", "Price|Primary Energy|Coal", "Price|Coal",
                     "Final Energy|Residential and Commercial", "Final Energy|Residential and Commercial|Gases",
                     "Final Energy|Residential"]
        self.assertEqual(preferred_variable_from_query("oil price in 2030", catalogue), "Price|Primary Energy|Oil")
        self.assertEqual(preferred_variable_from_query("gas price for EU in 2040", catalogue),
                         "Price|Primary Energy|Gas")
        self.assertEqual(preferred_variable_from_query("coal price for the EU in 2040", catalogue),
                         "Price|Primary Energy|Coal")
        self.assertEqual(
            preferred_variable_from_query("residential and commercial energy demand for Japan in 2050", catalogue),
            "Final Energy|Residential and Commercial",
        )

    def test_child_refinement_ignores_parent_words_and_matches_plurals(self):
        from manager import MultiAgentManager

        catalogue = ["Final Energy|Residential and Commercial", "Final Energy|Residential and Commercial|Gases",
                     "Final Energy|Residential and Commercial|Residential"]
        self.assertEqual(
            MultiAgentManager._best_supported_variable_path(
                "Final Energy|Residential and Commercial", "",
                "only gas variable Final Energy|Residential and Commercial region JPN", catalogue,
            ),
            "Final Energy|Residential and Commercial|Gases",
        )

    def test_new_sector_word_is_a_new_topic_but_refinements_are_not(self):
        manager = build_manager(SESSION_TS, SESSION_MODELS)
        carried = {"variable": "Price|Secondary Energy|Electricity"}
        manager.entity_extractor.available_variables = ["Final Energy|Buildings", carried["variable"],
                                                        "Final Energy|Transportation"]
        manager._variable_word_vocabulary = None
        self.assertTrue(manager._states_new_topic("energy in buildings for EU", carried))
        self.assertFalse(manager._states_new_topic("same for transport", {"variable": "Final Energy"}))
        self.assertFalse(manager._states_new_topic("only transport", {"variable": "Final Energy"}))

    def test_variable_templates_are_not_indexed(self):
        from utils.yaml_loader import yaml_to_documents

        docs = yaml_to_documents({
            "/defs/variable/emissions.yaml": [
                {"Emissions|{Level-1 Species}": {"description": "Emissions of {Level-1 Species}"}},
                {"Emissions|Kyoto Gases": {"description": "Total Kyoto GHG emissions"}},
            ],
        })
        self.assertEqual([doc.metadata["variable"] for doc in docs], ["Emissions|Kyoto Gases"])



class ConceptualAnswerTests(unittest.TestCase):
    """LLM answers checked on 25 conceptual questions (2026-10-26)."""

    def test_explanations_match_their_questions(self):
        from explanations import matching_explanations

        cases = {
            "why are carbon prices so different across models?": "Why carbon prices differ across models",
            "why is carbon capture important for net zero?": "Carbon capture and storage versus carbon removal",
            "how do IAMs represent technology learning?": "Technology learning in IAMs",
            "what are the trade-offs of bioenergy with CCS?": "Trade-offs of bioenergy with CCS (BECCS)",
            "how do renewables affect electricity prices?": "How renewables affect electricity prices",
            "why do emissions differ between scenarios?": "Why emissions differ between scenarios",
        }
        for question, topic in cases.items():
            self.assertEqual(matching_explanations(question)[0].metadata["topic"], topic, question)
        self.assertEqual(matching_explanations("CO2 emissions for World in 2050"), [])

    def test_ccs_explanation_does_not_call_it_removal(self):
        from explanations import matching_explanations

        text = matching_explanations("why is carbon capture important for net zero?")[0].page_content
        self.assertIn("does not remove CO2 from the atmosphere", text)

    def test_matched_explanation_is_the_only_context(self):
        from agents import _CuratedFirstRetriever

        base = MagicMock()
        base.invoke.return_value = [MagicMock(page_content="Variable Price|Carbon [weighted]")]
        retriever = _CuratedFirstRetriever(base=base)
        docs = retriever.invoke("why are carbon prices so different across models?")
        self.assertEqual([doc.metadata["type"] for doc in docs], ["explanation"])
        self.assertEqual(len(retriever.invoke("why do models disagree?")), 1)

    def test_explanation_questions_route_to_general_qa(self):
        manager = build_manager(SESSION_TS, SESSION_MODELS)
        for question in (
            "what role does carbon dioxide removal play in mitigation pathways?",
            "what are the trade-offs of bioenergy with CCS?",
            "how should I compare results from two studies?",
            "why is the power sector usually decarbonised first?",
            "why do baseline scenarios still show emissions growth?",
        ):
            self.assertTrue(manager._is_conceptual_climate_question(question), question)
        for question in ("CO2 emissions for World in 2050", "why is CO2 for World in 2050 so high?",
                         "compare solar and wind capacity for EU in 2050", "how much does nuclear grow?"):
            self.assertFalse(manager._is_conceptual_climate_question(question), question)

    def test_explanations_are_indexed(self):
        from main import knowledge_documents

        docs = knowledge_documents([], [], [])
        self.assertTrue(any(doc.metadata.get("type") == "explanation" for doc in docs))



class RiskyQueryFindingsTests(unittest.TestCase):
    """Problems found by the 59 risky follow-up questions (2026-10-26)."""

    def test_named_model_replaces_carried_model(self):
        ts = SESSION_TS + [_multi("world-headed", "World", "NDC", "TIAM_Grantham", "Emissions|CO2", "Mt CO2/yr",
                                  {"2050": 15000})]
        manager = build_manager(ts, SESSION_MODELS + [{"modelName": "TIAM_Grantham"}])
        manager.route_query("Primary Energy|Coal for World in 2050")
        manager.route_query("same from MUSE")
        answer = manager.route_query("Emissions|CO2 for World in 2050 from TIAM_Grantham")
        self.assertIn("TIAM_Grantham", answer)
        self.assertNotIn("model `MUSE`", answer)

    def test_full_new_question_does_not_keep_previous_model(self):
        manager = build_manager(SESSION_TS, SESSION_MODELS)
        manager.route_query("Primary Energy|Coal for World in 2050")
        manager.route_query("same from MUSE")
        answer = manager.route_query("Emissions|CO2 for IND in 2050")
        self.assertIn("GCAM", answer)
        # A bare follow-up still keeps the model.
        manager.route_query("Primary Energy|Coal for World in 2050")
        manager.route_query("same from MUSE")
        followup = manager.route_query("same for IND")
        self.assertNotIn("| GCAM |", followup)

    def test_two_named_models_are_both_compared(self):
        manager = build_manager(SESSION_TS, SESSION_MODELS)
        answer = manager.route_query("compare GCAM and MUSE for Emissions|CO2 in World in 2050")
        self.assertIn("GCAM", answer)
        self.assertIn("MUSE", answer)

    def test_comparison_members_match_raw_model_codes(self):
        from data_utils import _model_member_ok

        self.assertTrue(_model_member_ok("gcam", "GCAM"))
        self.assertTrue(_model_member_ok("GCAM 7.0", "GCAM"))
        self.assertFalse(_model_member_ok("GCAM 7.0", "GCAM 5.3"))
        self.assertFalse(_model_member_ok("GCAM-PR 5.3", "GCAM"))
        self.assertFalse(_model_member_ok("MUSE", "GCAM"))

    def test_explanation_words_do_not_pick_a_variable(self):
        from canonical_aliases import preferred_variable_from_query

        catalogue = ["Emissions|CO2", "Emissions|CO2|Industrial Processes|Chemicals|High value chemicals",
                     "Price|Primary Energy|Gas"]
        self.assertEqual(
            preferred_variable_from_query("why is CO2 for World in 2050 so high under the baseline?", catalogue),
            "Emissions|CO2",
        )
        self.assertEqual(preferred_variable_from_query("price of natural gas in 2050", catalogue),
                         "Price|Primary Energy|Gas")

    def test_only_carrier_swaps_a_carrier_leaf(self):
        manager = build_manager(SESSION_TS, SESSION_MODELS)
        manager.entity_extractor.available_variables = [
            "Capacity|Electricity|Solar", "Capacity|Electricity|Wind", "Primary Energy|Coal", "Primary Energy|Gas",
            "Secondary Energy|Electricity", "Secondary Energy|Electricity|Solar", "Final Energy|Industry",
        ]
        swap = manager._only_carrier_swap
        self.assertEqual(swap("only wind", "Capacity|Electricity|Solar"), "Capacity|Electricity|Wind")
        self.assertEqual(swap("only gas", "Primary Energy|Coal"), "Primary Energy|Gas")
        # Children stay children.
        self.assertEqual(swap("only solar", "Secondary Energy|Electricity"), "")
        self.assertEqual(swap("only electricity", "Final Energy|Industry"), "")
        self.assertEqual(swap("same for China", "Primary Energy|Coal"), "")

    def test_api_sanitizer_keeps_catalogue_numeric_model(self):
        import fastapi_app
        from model_aliases import UNLABELLED_MODEL_LABEL, register_model_display_names

        table = "| Model | Scenario | 2050 |\n|---|---|---|\n| 42 | NDC | 1 |\n| 12345 | NDC | 2 |"
        register_model_display_names(["42", "GCAM 7.0"])
        try:
            text = fastapi_app._sanitize_model_presentation_text(table)
            self.assertIn("| 42 | NDC", text)
            self.assertIn(f"| {UNLABELLED_MODEL_LABEL} | NDC | 2", text)
        finally:
            register_model_display_names([])
        self.assertNotIn("| 42 |", fastapi_app._sanitize_model_presentation_text(table))

    def test_new_explanations_match(self):
        from explanations import matching_explanations

        for question, topic in (
            ("why is methane important for near-term warming?", "Methane and near-term warming"),
            ("how do IAMs handle air pollution?", "Air pollution in IAMs"),
            ("how do IAMs represent behaviour change?", "Behaviour and lifestyle change in IAMs"),
        ):
            self.assertEqual(matching_explanations(question)[0].metadata["topic"], topic)


if __name__ == "__main__":
    unittest.main()
