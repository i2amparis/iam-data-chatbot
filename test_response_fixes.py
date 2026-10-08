"""Regression tests for response_fixes_todo.md (2026-10-07 review).

The fixtures load several studies at once, like the real data. Single-study
fixtures hid most of these defects because the multi-study paths never ran.
"""

import unittest
from unittest.mock import MagicMock, patch

from data_metadata import DataMetadata
from data_utils import format_number, format_time_series_data
from manager import MultiAgentManager


def _record(workspace, region, scenario, model, value, variable="Emissions|CO2", unit="Mt CO2/yr"):
    return {
        "workspace_code": workspace,
        "variable": variable,
        "region": region,
        "scenario": scenario,
        "modelName": model,
        "unit": unit,
        "years": {"2030": value / 2, "2050": value},
    }


TWO_STUDY_TS = [
    _record("world-headed", "World", "Baseline", "GCAM", 45000),
    _record("world-headed", "World", "NDC", "GCAM", 30000),
    _record("world-headed", "CHN", "Baseline", "GCAM", 12000),
    _record("world-headed", "CHN", "NDC", "GCAM", 9000),
    _record("post-glasgow", "World", "LTT", "GCAM", 9200),
    _record("post-glasgow", "CHN", "LTT", "GCAM", 2100),
]


def build_manager(ts=None, models=None):
    """A real manager whose LLM clients fail, so only deterministic paths run."""
    failing_llm = MagicMock(side_effect=RuntimeError("LLM disabled in tests"))
    models = models if models is not None else [{"modelName": "GCAM"}]
    ts = list(ts if ts is not None else TWO_STUDY_TS)
    resources = {
        "models": models,
        "ts": ts,
        "metadata": DataMetadata(ts, models),
        "env": {"OPENAI_API_KEY": "test"},
        "link_catalog": [],
        "vector_store": None,
    }
    with patch("manager.ChatOpenAI", return_value=failing_llm), patch(
        "query_extractor.ChatOpenAI", return_value=failing_llm
    ):
        return MultiAgentManager(resources, streaming=False)


class StudyDefaultTests(unittest.TestCase):
    """Item 1: a slice present in several studies answers from one of them."""

    def test_multi_study_question_returns_numbers_with_study_note(self):
        manager = build_manager()
        answer = manager.route_query("CO2 emissions for World in 2050")

        self.assertNotIn("Please choose a study", answer)
        self.assertIn("Showing results from the study **Where is the world headed?**", answer)
        self.assertIn("Also available in: Post-Glasgow targets", answer)
        self.assertEqual(manager.last_entities.get("workspace_code"), "world-headed")

    def test_region_followup_keeps_study_year_and_does_not_invent_comparison(self):
        manager = build_manager()
        manager.route_query("CO2 emissions for World in 2050")
        answer = manager.route_query("same for China")

        self.assertIn("in CHN", answer)
        self.assertNotIn("comparison members", answer)
        self.assertNotIn("2030", answer)
        self.assertEqual(manager.last_entities.get("workspace_code"), "world-headed")

    def test_bare_study_name_switches_previous_question(self):
        manager = build_manager()
        manager.route_query("CO2 emissions for World in 2050")
        answer = manager.route_query("Post-Glasgow targets")

        self.assertEqual(manager.last_route_decision.get("reason"), "study switch for previous question")
        self.assertIn("LTT", answer)
        self.assertNotIn("NDC", answer)
        self.assertEqual(manager.last_entities.get("workspace_code"), "post-glasgow")

    def test_study_name_without_previous_question_is_not_a_switch(self):
        manager = build_manager()
        manager.route_query("Post-Glasgow targets")

        self.assertNotEqual(
            manager.last_route_decision.get("reason"), "study switch for previous question",
        )

    def test_comparison_prefers_study_covering_both_regions(self):
        variable = "Capacity|Electricity|Solar"
        ts = [
            _record("world-headed", "DEU", scenario, "GCAM", value, variable, "GW")
            for scenario, value in (("Baseline", 100), ("NDC", 90), ("LTT", 80))
        ] + [
            _record("post-glasgow", "DEU", "Baseline", "GCAM", 50, variable, "GW"),
            _record("post-glasgow", "GREECE", "Baseline", "GCAM", 20, variable, "GW"),
        ]
        manager = build_manager(ts)

        answer = manager.route_query("compare solar capacity for Germany and Greece in 2030")

        self.assertIn("Showing results from the study **Post-Glasgow targets**", answer)
        self.assertIn("in DEU", answer)
        self.assertIn("in GREECE", answer)
        self.assertNotIn("Please choose a study", answer)
        self.assertEqual(manager.last_entities.get("workspace_code"), "post-glasgow")

    def test_latest_year_in_results_uses_loaded_records(self):
        manager = build_manager()

        answer = manager.route_query("what is the latest year in the results")

        self.assertEqual(manager.last_route_decision.get("reason"), "catalogue year coverage request")
        self.assertIn("**2050**", answer)
        self.assertNotIn("2020", answer)


class ConceptualRoutingTests(unittest.TestCase):
    def test_causal_climate_questions_reach_general_qa(self):
        questions = (
            "What are the main uncertainties in IAM projections of net zero pathways?",
            "Why might energy transitions differ between countries?",
            "How does carbon pricing affect energy transitions?",
        )
        for question in questions:
            with self.subTest(question=question):
                manager = build_manager()
                manager.agents["general_qa"].handle = MagicMock(return_value="Conceptual answer")

                answer = manager.route_query(question)

                self.assertIn("Conceptual answer", answer)
                self.assertEqual(manager.last_route_decision.get("agent"), "general_qa")
                self.assertEqual(manager.last_route_decision.get("reason"), "conceptual climate explanation")

    def test_numeric_and_named_model_questions_keep_their_routes(self):
        manager = build_manager()
        manager.route_query("CO2 emissions for World in 2050")
        self.assertEqual(manager.last_route_decision.get("agent"), "data_query")

        manager = build_manager()
        manager.route_query("compare GCAM and REMIND")
        self.assertEqual(manager.last_route_decision.get("agent"), "model_explanation")


class NoPlotDeadEndTests(unittest.TestCase):
    """Item 2: follow-ups and the bot's own suggestions never dead-end on plotting."""

    REDIRECT_OR_DEAD_END = (
        "plot these results",
        "don't have an active",
        "i need one more detail",
        "sorry, i encountered",
    )

    def test_compare_with_baseline_returns_a_table(self):
        manager = build_manager()
        manager.route_query("CO2 emissions for World in 2050")
        answer = manager.route_query("compare with baseline")

        self.assertNotIn("plot these results", answer)
        self.assertIn("GCAM - Baseline", answer)
        self.assertIn("| 2050 |", answer)

    def test_scenario_comparison_expands_baseline_only_within_active_study(self):
        ts = TWO_STUDY_TS + [
            _record("post-glasgow", "World", "Baseline_HP", "GCAM", 50000),
        ]
        manager = build_manager(ts)
        manager.route_query("CO2 emissions for World under NDC in 2050")
        answer = manager.route_query("compare with baseline")

        self.assertNotIn("Baseline_HP", answer)
        self.assertNotIn("comparison members", answer)

    def test_open_data_explorer_links_active_study(self):
        manager = build_manager()
        manager.route_query("CO2 emissions for World in 2050")
        answer = manager.route_query("Open the data explorer")

        self.assertIn("where-is-the-world-headed/graphs", answer)
        # The data scope survives for the next follow-up.
        self.assertEqual(manager.last_entities.get("variable"), "Emissions|CO2")

    def test_study_explorer_navigation_variations_preserve_data_scope(self):
        queries = (
            "take me to the active study data explorer",
            "take me to the data explorer for the active study",
            "show me this study's data explorer",
            "open the data explorer for this study",
            "give me the link to the current study data explorer",
        )
        manager = build_manager()
        manager.route_query("CO2 emissions for World in 2050")
        manager.route_query("same for China")
        manager.route_query("Post-Glasgow targets")
        scope = dict(manager.last_entities)
        for query in queries:
            with self.subTest(query=query):
                answer = manager.route_query(query)
                self.assertIn("post-glasgow-targets/graphs", answer)
                self.assertEqual(manager.last_route_decision.get("agent"), "general_qa")
                self.assertNotIn("| Model |", answer)
                self.assertEqual(manager.last_entities, scope)
        answer = manager.route_query("same for World")
        self.assertIn("in World", answer)
        self.assertEqual(manager.last_entities.get("workspace_code"), "post-glasgow")

    def test_explorer_navigation_without_scope_does_not_invent_a_model(self):
        manager = build_manager()
        answer = manager.route_query("take me to the data explorer for the active study")
        self.assertIn("https://iamparis.eu/results", answer)
        self.assertNotIn("MARIO", answer)
        self.assertNotIn("/models", answer)

    def test_before_year_excludes_values_at_the_requested_boundary(self):
        manager = build_manager()
        answer = manager.route_query("CO2 emissions for World before 2050")
        self.assertIn("2030", answer)
        self.assertNotIn("| 2050 |", answer)
        self.assertEqual(manager.last_entities.get("end_year"), 2049)

    def test_show_both_followup_returns_a_table_not_a_figure(self):
        manager = build_manager()
        manager.route_query("CO2 emissions for World in 2050")
        manager.route_query("same for China")
        answer = manager.route_query("show both together")

        self.assertNotIn("![Plot]", answer)
        self.assertNotIn("data:image/png", answer)
        self.assertEqual(manager.last_route_decision.get("agent"), "data_query")

    def test_capability_text_does_not_offer_chat_plots(self):
        manager = build_manager()
        self.assertNotIn("plot", manager.route_query("hello").casefold())

    def test_every_suggestion_is_answerable(self):
        import fastapi_app

        starts = (
            "CO2 emissions for World in 2050",
            "tell me about GCAM",
            "CO2 for Atlantis",
            "hello",
        )
        for start in starts:
            manager = build_manager()
            answer = manager.route_query(start)
            suggestions = fastapi_app._suggested_next_questions(start, answer, manager)
            self.assertTrue(suggestions, start)
            for suggestion in suggestions:
                with self.subTest(start=start, suggestion=suggestion):
                    fresh = build_manager()
                    fresh.route_query(start)
                    reply = fresh.route_query(suggestion).casefold()
                    for marker in self.REDIRECT_OR_DEAD_END:
                        self.assertNotIn(marker, reply)


class RegionNotModelTests(unittest.TestCase):
    """Item 3: "does <region> have" names a region, not an unknown model."""

    TS = [
        _record("net-zero", "DEU", "NZE", "TIMES", 40, variable="Capacity|Electricity|Solar", unit="GW"),
        _record("net-zero", "DEU", "NZE", "TIMES", 600, variable="Emissions|CO2"),
    ]

    def test_how_much_question_returns_values_for_region(self):
        manager = build_manager(self.TS, models=[{"modelName": "TIMES"}])
        answer = manager.route_query("how much solar capacity does Germany have")

        self.assertNotIn("could not match model", answer)
        self.assertIn("Capacity|Electricity|Solar in DEU", answer)
        self.assertIn("40.00", answer)

    def test_region_availability_question_is_answered_for_region(self):
        manager = build_manager(self.TS, models=[{"modelName": "TIMES"}])
        answer = manager.route_query("does Germany report CO2 emissions")

        self.assertTrue(answer.startswith("Yes."), answer)
        self.assertIn("DEU", answer)

    def test_unknown_model_subject_is_still_reported(self):
        manager = build_manager(self.TS, models=[{"modelName": "TIMES"}])
        answer = manager.route_query("does FooModel report CO2 emissions")

        self.assertIn("could not match model `FooModel`", answer)


class TopicModelCoverageTests(unittest.TestCase):
    """Item 4: a named term outside the sector list filters the model list."""

    TS = [
        _record("net-zero", "World", "NZE", "TIMES", 5, variable="Capacity|Hydrogen|Electrolysis", unit="GW"),
        _record("net-zero", "World", "NZE", "GCAM", 600, variable="Emissions|CO2"),
    ]
    MODELS = [{"modelName": "TIMES"}, {"modelName": "GCAM"}]

    def test_models_reporting_named_term(self):
        manager = build_manager(self.TS, self.MODELS)
        answer = manager.route_query("which models report hydrogen")

        self.assertIn("Models reporting hydrogen variables", answer)
        self.assertIn("TIMES", answer)
        self.assertNotIn("GCAM", answer)

    def test_unknown_term_says_so_instead_of_listing_all_models(self):
        manager = build_manager(self.TS, self.MODELS)
        answer = manager.route_query("which models report unobtainium")

        self.assertIn("No loaded IAM PARIS variable mentions `unobtainium`", answer)

    def test_generic_model_list_is_unchanged(self):
        manager = build_manager(self.TS, self.MODELS)
        answer = manager.route_query("which models are available")

        self.assertNotIn("Models reporting", answer)
        self.assertIn("GCAM", answer)


class RegionDetectionTests(unittest.TestCase):
    """Items 5-6 and the regionless-table guard."""

    TS = [
        _record("net-zero", "DEU", "NZE", "TIMES", 30, variable="Price|Secondary Energy|Electricity", unit="US$2010/GJ"),
        _record("net-zero", "FRA", "NZE", "TIMES", 20, variable="Price|Secondary Energy|Electricity", unit="US$2010/GJ"),
        _record("net-zero", "World", "NZE", "TIMES", 600, variable="Emissions|CO2"),
    ]
    MODELS = [{"modelName": "TIMES"}]

    def test_data_wording_after_for_is_not_a_region(self):
        manager = build_manager(self.TS, self.MODELS)
        answer = manager.route_query("what does net zero mean for electricity prices")

        self.assertNotIn("as a region", answer)

    def test_regionless_multi_region_slice_asks_for_region(self):
        manager = build_manager(self.TS, self.MODELS)
        answer = manager.route_query("electricity prices")

        self.assertIn("Choose the region:", answer)
        # Never a single unlabeled row mixing DEU and FRA values.
        self.assertNotIn("30.00", answer)
        follow = manager.route_query("1")
        self.assertIn("Price|Secondary Energy|Electricity in", follow)

    def test_unknown_region_is_reported_before_study_choice(self):
        manager = build_manager()
        answer = manager.route_query("CO2 for Atlantis")

        self.assertIn("couldn't find `Atlantis` as a region", answer)
        self.assertNotIn("Showing results from the study", answer)


class NamedStudyNavigationTests(unittest.TestCase):
    """Item 7: a question naming a study is grounded in that exact study."""

    def test_named_study_question_links_that_study(self):
        ts = TWO_STUDY_TS + [_record("eu-headed", "EU", "PR_WWH_CP", "GCAM", 3000)]
        manager = build_manager(ts)
        manager.shared_resources["link_catalog"] = [
            {"title": "Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU",
             "url": "https://iamparis.eu/results/iam-compact/fit-for-55", "category": "results"},
        ]
        answer = manager.route_query(
            "what are the main differences between the scenarios in Where is the EU headed?"
        )

        self.assertIn("### Where is the EU headed?", answer)
        self.assertIn("PR_WWH_CP", answer)
        urls = [link["url"] for link in manager.last_links]
        self.assertTrue(urls and all("where-is-the-eu-headed" in url for url in urls), urls)
        self.assertEqual(manager.last_entities.get("workspace_code"), "eu-headed")


class StudySuggestionTests(unittest.TestCase):
    """Item 8: suggestion requests naming a sector still get suggestions."""

    def test_suggestion_with_topic_is_not_a_variable_picker(self):
        ts = TWO_STUDY_TS + [
            _record("transp-transf", "World", "NDC", "GCAM", 5, variable="Exports|Air transport", unit="bn USD"),
        ]
        manager = build_manager(ts)
        answer = manager.route_query("suggest research ideas on transport")

        self.assertEqual(manager.last_route_decision.get("agent"), "modelling_suggestions")
        self.assertIn("modelling study suggestions related to transport", answer)
        self.assertNotIn("Choose the variable", answer)


class ModelComparisonScopeTests(unittest.TestCase):
    """Item 9: an exact family name does not pull in longer catalogue variants."""

    MODELS = [{"modelName": "GCAM"}, {"modelName": "REMIND-MFA"}]

    def test_family_name_selects_only_that_profile(self):
        manager = build_manager(models=self.MODELS)
        names = [profile["name"] for profile in manager._runtime_model_profiles("compare GCAM and REMIND")]

        self.assertEqual(sorted(names), ["GCAM", "REMIND"])

    def test_explicit_variant_is_kept(self):
        manager = build_manager(models=self.MODELS)
        names = [profile["name"] for profile in manager._runtime_model_profiles("compare GCAM and REMIND-MFA")]

        self.assertIn("REMIND-MFA", names)
        self.assertIn("GCAM", names)


class AnswerFormattingTests(unittest.TestCase):
    """Items 10-11: readable numbers and a computed one-line takeaway."""

    def test_format_number_keeps_precision(self):
        self.assertEqual(format_number(35612.4), "35,612")
        self.assertEqual(format_number(2_500_000), "2,500,000")
        self.assertEqual(format_number(12.5), "12.50")
        self.assertEqual(format_number(0.0034), "0.0034")
        self.assertEqual(format_number(0), "0.00")
        self.assertEqual(format_number(-1234.5), "-1,234")

    def test_single_series_trajectory_headline(self):
        response = format_time_series_data(
            [{"scenario": "NDC", "modelName": "GCAM", "unit": "Mt CO2/yr",
              "years": {"2030": 40000, "2050": 20000}}],
            "Emissions|CO2", "World", 2030, 2050,
        )
        self.assertIn(
            "Emissions|CO2 in World goes from 40,000 in 2030 to 20,000 Mt CO2/yr in 2050 (-50%) (GCAM, NDC).",
            response,
        )

    def test_units_differing_only_by_case_are_one_unit(self):
        response = format_time_series_data(
            [
                {"scenario": "A", "modelName": "M1", "unit": "Million", "years": {"2050": 80}},
                {"scenario": "A", "modelName": "M2", "unit": "million", "years": {"2050": 84}},
                {"scenario": "B", "modelName": "M2", "unit": "million", "years": {"2050": 82}},
            ],
            "Population", "DEU", 2050, 2050,
        )
        self.assertIn("Unit: `million`", response)
        self.assertIn("ranges from 80.00", response)


class ModelDisplayNameTests(unittest.TestCase):
    """Item 12: raw result aliases show with the catalogue family spelling."""

    def tearDown(self):
        from model_aliases import register_model_display_names
        register_model_display_names([])

    def test_raw_lowercase_alias_uses_catalogue_family_spelling(self):
        from model_aliases import display_model_label, register_model_display_names

        register_model_display_names(["GCAM 7.0", "GEMINI-E3 7.0", "TIAM_Grantham 3.2"])

        self.assertEqual(display_model_label("gcam"), "GCAM")
        self.assertEqual(display_model_label("gemini_e3"), "GEMINI-E3")
        # No unique catalogue family: unchanged rather than guessed.
        self.assertEqual(display_model_label("tiam"), "tiam")
        # Already-cased and versioned labels are never rewritten.
        self.assertEqual(display_model_label("GCAM 7.0"), "GCAM 7.0")

    def test_provenance_counts_records_behind_a_display_label(self):
        import fastapi_app
        from model_aliases import register_model_display_names

        register_model_display_names(["GEMINI-E3 7.0"])
        resources = {"ts": [
            _record("w", "World", "NDC", "gemini_e3", 10),
            _record("w", "World", "NDC", "GEMINI-E3 7.0", 20),
        ]}

        self.assertEqual(
            fastapi_app._count_matching_records(resources, {"models": ["GEMINI-E3"]}), 1,
        )
        self.assertEqual(
            fastapi_app._count_matching_records(resources, {"model": "GEMINI-E3"}), 1,
        )

    def test_model_list_and_topic_counts_use_the_same_denominator(self):
        ts = [
            _record("w", "World", "NDC", "TIMES", 5, variable="Final Energy|Transportation", unit="EJ/yr"),
            _record("w", "World", "NDC", "GCAM", 5, variable="Emissions|CO2"),
        ]
        models = [{"modelName": name} for name in ("TIMES", "GCAM", "A", "B", "C", "D", "E")]
        manager = build_manager(ts, models)

        listing = manager.route_query("which models are available")
        topic = manager.route_query("which models cover transport")

        self.assertIn("2 of them have loaded results", listing)
        self.assertIn("of the 2 with loaded results", topic)


class LatencyCacheTests(unittest.TestCase):
    """P1 latency: indexes and caches must not change any answer or count."""

    def test_narrowed_counts_equal_full_scan_counts(self):
        import fastapi_app

        ts = TWO_STUDY_TS + [
            _record("post-glasgow", "deu", "LTT", "TIMES", 7),
            _record("post-glasgow", "DEU", "LTT", "TIMES", 8, variable="Population", unit="million"),
        ]
        resources = {"ts": ts}
        scopes = [
            ({"variable": "Emissions|CO2"}, 7),
            ({"region": "World"}, 3),
            ({"regions": ["CHN", "DEU"]}, 5),
            ({"model": "TIMES"}, 2),
            ({"workspace_code": "post-glasgow"}, 4),
            ({"variable": "Emissions|CO2", "region": "CHN", "scenario": "NDC"}, 1),
        ]
        for scope, expected in scopes:
            with self.subTest(scope=scope):
                self.assertEqual(fastapi_app._count_matching_records(resources, scope), expected)

    def test_record_cache_invalidates_when_records_change(self):
        import record_cache

        records = [{"variable": f"V{i % 3}"} for i in range(record_cache.MIN_CACHED_RECORDS)]
        self.assertEqual(record_cache.distinct_values(records, "variable"), {"V0", "V1", "V2"})
        changed = records + [{"variable": "V9"}]
        self.assertIn("V9", record_cache.distinct_values(changed, "variable"))
        # A copy of the same records reuses the cached result.
        self.assertIs(
            record_cache.distinct_values(list(records), "variable"),
            record_cache.distinct_values(records, "variable"),
        )


class LlmSetupTests(unittest.TestCase):
    """P2 LLM setup: JSON mode, context size, router replies, history size."""

    def test_openai_json_output_sets_response_format(self):
        import os
        from llm_factory import get_chat_openai

        with patch.dict(os.environ, {"LOCAL_LLM_MODEL": "", "USE_LOCAL_LLM": "false", "IAM_LOCAL_MODELS": ""}):
            llm = get_chat_openai("gpt-4o-mini", json_output=True, openai_api_key="test")
        self.assertEqual(llm.model_kwargs.get("response_format"), {"type": "json_object"})

    def test_local_json_output_and_context_size(self):
        import os
        from llm_factory import get_chat_openai

        with patch.dict(os.environ, {"LOCAL_LLM_MODEL": "qwen3:0.6b", "LOCAL_LLM_NUM_CTX": "16384"}):
            llm = get_chat_openai("qwen3:0.6b", json_output=True, timeout=5)
            plain = get_chat_openai("qwen3:0.6b")
        self.assertEqual(llm.format, "json")
        self.assertEqual(llm.num_ctx, 16384)
        self.assertIsNone(plain.format)

    def test_router_reply_normalization(self):
        normalize = MultiAgentManager._normalize_route_reply
        self.assertEqual(normalize("`data_query`."), "data_query")
        self.assertEqual(normalize("Category: general_qa"), "general_qa")
        self.assertEqual(normalize("<think>x</think> model_explanation"), "model_explanation")
        self.assertNotIn(normalize("data_query or general_qa"), ("data_query", "general_qa"))

    def test_history_answers_are_compacted(self):
        from agents import _compact_history_answer

        answer = (
            "### Emissions|CO2 in World\n\nAnswer:\nIn 2050 it ranges from 9,173 to 52,211.\n\n"
            "| Model | Scenario | 2050 |\n|---|---|---|\n| GCAM | NDC | 9,173 |\n\n"
            "Use the [data explorer](https://iamparis.eu/results)."
        )
        compact = _compact_history_answer(answer)
        self.assertNotIn("|---", compact)
        self.assertNotIn("https://", compact)
        self.assertIn("ranges from 9,173", compact)
        self.assertLessEqual(len(_compact_history_answer("word " * 500)), 302)



class QueryHandlingRetestTests(unittest.TestCase):
    def test_catalogue_year_count_uses_distinct_years_not_regions(self):
        manager = build_manager()
        answer = manager.route_query("how many distinct years are in the results")
        self.assertIn("**2 distinct years**", answer)
        self.assertNotIn("as a region", answer)

    def test_electricity_model_coverage_excludes_oil_only_models(self):
        ts = [
            _record("study", "World", "Baseline", "ElectricModel", 1, "Secondary Energy|Electricity"),
            _record("study", "World", "Baseline", "OilModel", 1, "Primary Energy|Oil"),
        ]
        manager = build_manager(ts)
        answer = manager.route_query("which models cover electricity")
        self.assertIn("ElectricModel", answer)
        self.assertNotIn("OilModel", answer)

    def test_natural_energy_and_co2_aliases_return_the_requested_year(self):
        for phrase, variable in (
            ("energy demand", "Final Energy"),
            ("energy production", "Secondary Energy"),
            ("carbon dioxide", "Emissions|CO2"),
        ):
            with self.subTest(phrase=phrase):
                manager = build_manager([_record("study", "World", "Baseline", "GCAM", 4, variable)])
                answer = manager.route_query(f"{phrase} for World in 2030")
                self.assertIn(f"### {variable} in World", answer)
                self.assertIn("2030", answer)
                self.assertNotIn("Choose the variable", answer)

    def test_renewable_capacity_returns_technology_totals_without_double_counting(self):
        variables = ("Capacity|Electricity|Solar", "Capacity|Electricity|Wind", "Capacity|Electricity|Solar|PV")
        ts = [_record("study", "EU", "Baseline", "GCAM", 10, v, "GW") for v in variables]
        manager = build_manager(ts)
        answer = manager.route_query("compare renewable capacity in EU between 2030 and 2050")
        self.assertIn("capacities separately", answer)
        self.assertIn("### Capacity|Electricity|Solar in EU", answer)
        self.assertIn("### Capacity|Electricity|Wind in EU", answer)
        self.assertNotIn("### Capacity|Electricity|Solar|PV", answer)
        self.assertIn("2030", answer)
        self.assertIn("2050", answer)

    def test_unknown_second_region_does_not_return_a_one_sided_comparison(self):
        manager = build_manager([_record("study", "EU", "Baseline", "GCAM", 10, "Final Energy")])
        answer = manager.route_query("compare energy use in Europe and Asia in 2050")
        self.assertIn("`Asia` as a region", answer)
        self.assertNotIn("Choose the variable", answer)
        self.assertNotIn("| Year |", answer)

    def test_scenario_comparison_is_not_misread_as_a_region_comparison(self):
        from data_utils import unknown_comparison_region

        self.assertIsNone(unknown_comparison_region(
            "compare CO2 emissions for World under Baseline and Current Policies in 2050",
            ["World"], [], ["Baseline", "Current Policies"], ["Emissions|CO2"],
        ))

    def test_causal_question_with_scenarios_is_not_availability(self):
        manager = build_manager()
        manager.agents["general_qa"].handle = MagicMock(return_value="Conceptual answer")
        answer = manager.route_query("What causes emissions pathways to differ across climate scenarios?")
        self.assertIn("Conceptual answer", answer)
        self.assertEqual(manager.last_route_decision.get("agent"), "general_qa")

    def test_model_classification_uses_runtime_metadata(self):
        manager = build_manager(models=[{
            "modelName": "EnergyPLAN", "model_type": "Simulation model",
            "description": "Annual energy system operation in hourly time steps.",
        }])
        answer = manager.route_query("is EnergyPLAN an integrated assessment model?")
        self.assertIn("Simulation model", answer)
        self.assertIn("integrated assessment model", answer)
        self.assertNotIn("Description:", answer)

    def test_full_region_and_study_followup_workflow(self):
        manager = build_manager()
        manager.route_query("CO2 emissions for World in 2050")
        region_answer = manager.route_query("same for China")
        study_answer = manager.route_query("Post-Glasgow targets")
        self.assertIn("in CHN", region_answer)
        self.assertIn("in CHN", study_answer)
        self.assertEqual(manager.last_entities.get("start_year"), 2050)
        self.assertEqual(manager.last_entities.get("workspace_code"), "post-glasgow")
        self.assertNotIn("Choose", study_answer)

if __name__ == "__main__":
    unittest.main()
