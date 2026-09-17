import logging
import unittest

from query_extractor import QueryEntityExtractor


def build_extractor_stub():
    extractor = object.__new__(QueryEntityExtractor)
    extractor.logger = logging.getLogger("QueryEntityExtractorTest")
    extractor.available_models = ["GCAM", "GCAM-PR 5.3", "GCAM-PR 7.0", "GLOBIO", "MESSAGEix-GLOBIOM 2.0", "REMIND"]
    extractor.available_scenarios = ["Baseline", "Policy"]
    extractor.available_variables = [
        "CO2 emissions cuts|Absolute",
        "Emissions|CO2",
        "Emissions|GHG",
        "Emissions|CH4",
        "Emissions|N2O",
        "Capacity|Electricity|Solar",
        "Capacity|Hydrogen|Solar",
        "Capacity|Electricity|Wind",
        "Secondary Energy|Electricity",
        "Price|Carbon",
        "GDP|MER",
        "Agricultural Production",
        "Final Energy|Transportation|Aviation",
        "Population",
    ]
    extractor.available_regions = ["World", "EU", "IS"]
    extractor.variable_units = {"Emissions|CO2": "Mt CO2/yr"}
    extractor.variable_dict = {}
    extractor.region_dict = {}
    extractor.model_alias_map = extractor._build_model_alias_map(extractor.available_models)
    return extractor


class QueryExtractorConfidenceTests(unittest.TestCase):
    def test_single_generic_variable_token_does_not_become_a_hard_match(self):
        extractor = build_extractor_stub()

        result = extractor._fallback_extraction(
            "show synthetic aviation fuel production in 2050"
        )

        self.assertIsNone(result["variable"])
        self.assertIn("synthetic", result["unmatched_variable_terms"])

    def test_exact_multi_token_variable_still_resolves(self):
        extractor = build_extractor_stub()

        result = extractor._fallback_extraction("show agricultural production in 2050")

        self.assertEqual(result["variable"], "Agricultural Production")
        self.assertGreaterEqual(result["entity_confidence"]["variable"], 0.9)

    def test_natural_language_region_is_excluded_from_variable_evidence(self):
        extractor = build_extractor_stub()

        result = extractor._fallback_extraction("show data for Europe in 2060")

        self.assertEqual(result["region"], "EU")
        self.assertIsNone(result["variable"])
        self.assertNotIn("europe", result.get("unmatched_variable_terms", []))

    def test_invalid_llm_region_is_not_fuzzily_forced_into_runtime_scope(self):
        extractor = build_extractor_stub()

        result = extractor._validate_result(
            {"action": "query", "region": "Atlantis"},
            "show data for Atlantis in 2050",
        )

        self.assertIsNone(result["region"])
        self.assertEqual(result["unmatched_region"], "Atlantis")
        self.assertEqual(result["entity_confidence"]["region"], 0.0)

    def test_valid_catalogue_region_from_llm_still_requires_query_evidence(self):
        extractor = build_extractor_stub()
        extractor.available_regions.append("IND")

        result = extractor._validate_result(
            {"action": "query", "model": "REMIND", "region": "IND"},
            "what kind of model is REMIND?",
        )

        self.assertIsNone(result["region"])
        self.assertNotIn("unmatched_region", result)
        self.assertEqual(result["entity_confidence"]["region"], 0.0)

    def test_llm_only_unit_guess_is_dropped_without_grounded_variable(self):
        extractor = build_extractor_stub()

        result = extractor._validate_result(
            {"action": "query", "unit": "BEUR"},
            "oil",
        )

        self.assertNotIn("unit", result)

    def test_exact_gem_e3_query_rejects_llm_gemini_e3_substitution(self):
        extractor = build_extractor_stub()
        extractor.available_models = ["GEMINI-E3 7.0", "gemini_e3"]
        extractor.model_alias_map = extractor._build_model_alias_map(
            extractor.available_models,
        )

        result = extractor._validate_result(
            {"action": "query", "model": "GEMINI-E3 7.0"},
            "what years are available for GEM-E3 data?",
        )

        self.assertEqual(result["model"], "GEM-E3")
        self.assertFalse(result["model_matched"])
        self.assertNotIn("GEMINI-E3", result["model"])

    def test_unambiguous_variable_typo_still_resolves(self):
        extractor = build_extractor_stub()

        result = extractor._fallback_extraction("show populaton in 2050")

        self.assertEqual(result["variable"], "Population")
        self.assertGreaterEqual(result["entity_confidence"]["variable"], 0.7)

    def test_plain_china_region_query_does_not_allow_model_match(self):
        extractor = QueryEntityExtractor.__new__(QueryEntityExtractor)

        self.assertFalse(
            extractor._query_allows_model_match("show Emissions|CO2 for China")
        )

    def test_explicit_model_query_allows_model_match(self):
        extractor = QueryEntityExtractor.__new__(QueryEntityExtractor)

        self.assertTrue(
            extractor._query_allows_model_match("show data using GCAM")
        )

    def test_model_information_question_does_not_extract_is_as_region(self):
        extractor = build_extractor_stub()

        result = extractor._fallback_extraction("What is the REMIND model?")

        self.assertEqual(result["model"], "REMIND")
        self.assertIsNone(result["region"])

    def test_validate_result_adds_entity_confidence(self):
        extractor = build_extractor_stub()
        result = extractor._validate_result(
            {
                "action": "query",
                "variable": "Emissions|CO2",
                "region": "World",
                "scenario": "Baseline",
                "model": "GCAM",
            },
            "CO2 emissions for World under Baseline with GCAM",
        )
        result = extractor._finalize_confidence(result)

        self.assertGreaterEqual(result["confidence"], 0.9)
        self.assertEqual(result["entity_confidence"]["variable"], 0.95)
        self.assertEqual(result["entity_confidence"]["region"], 0.95)
        self.assertEqual(result["entity_confidence"]["scenario"], 0.95)
        self.assertEqual(result["entity_confidence"]["model"], 0.95)

    def test_fallback_extraction_returns_confidence_and_years(self):
        extractor = build_extractor_stub()
        result = extractor._fallback_extraction(
            "plot CO2 emissions for World under Baseline after 2030 with GCAM"
        )

        self.assertEqual(result["action"], "plot")
        self.assertEqual(result["variable"], "Emissions|CO2")
        self.assertEqual(result["region"], "World")
        self.assertEqual(result["scenario"], "Baseline")
        self.assertEqual(result["model"], "GCAM")
        self.assertEqual(result["start_year"], 2031)
        self.assertIsNone(result["end_year"])
        self.assertIn("entity_confidence", result)
        self.assertGreater(result["confidence"], 0.0)

    def test_extract_prefers_deterministic_for_obvious_query(self):
        class PromptShouldNotRun:
            def __or__(self, _other):
                raise AssertionError("LLM path should not run for obvious deterministic queries")

        extractor = build_extractor_stub()
        extractor.prompt = PromptShouldNotRun()
        extractor.llm = object()

        result = extractor.extract("plot CO2 emissions for World under Baseline with GCAM")

        self.assertEqual(result["extraction_method"], "deterministic")
        self.assertEqual(result["action"], "plot")
        self.assertEqual(result["variable"], "Emissions|CO2")
        self.assertEqual(result["region"], "World")

    def test_carbon_dioxide_emissions_prefers_standard_co2_variable(self):
        extractor = build_extractor_stub()

        result = extractor._fallback_extraction("show me carbon dioxide emissions for Europe")

        self.assertEqual(result["variable"], "Emissions|CO2")
        self.assertGreaterEqual(result["entity_confidence"]["variable"], 0.9)

    def test_photovoltaic_capacity_prefers_electricity_solar_capacity(self):
        extractor = build_extractor_stub()

        result = extractor._fallback_extraction("plot photovoltaic capacity for Greece")

        self.assertEqual(result["variable"], "Capacity|Electricity|Solar")
        self.assertGreaterEqual(result["entity_confidence"]["variable"], 0.9)

    def test_gross_domestic_product_prefers_gdp_mer(self):
        extractor = build_extractor_stub()

        result = extractor._fallback_extraction("gross domestic product for World")

        self.assertEqual(result["variable"], "GDP|MER")

    def test_greenhouse_gas_prefers_broad_ghg_variable(self):
        extractor = build_extractor_stub()

        result = extractor._fallback_extraction("greenhouse gas pathways by country")

        self.assertEqual(result["variable"], "Emissions|GHG")

    def test_methane_alias_prefers_ch4_variable(self):
        extractor = build_extractor_stub()

        result = extractor._fallback_extraction("methane emissions for Europe")

        self.assertEqual(result["variable"], "Emissions|CH4")

    def test_current_policy_alias_sets_canonical_scenario(self):
        extractor = build_extractor_stub()
        extractor.available_scenarios = ["Baseline", "Policy", "Current Policies"]

        result = extractor._fallback_extraction("current policy scenario emissions for EU")

        self.assertEqual(result["scenario"], "Current Policies")

    def test_message_ix_alias_resolves_to_messageix_globiom(self):
        extractor = build_extractor_stub()

        result = extractor._fallback_extraction("show data with message ix")

        self.assertEqual(result["model"], "MESSAGEix-GLOBIOM 2.0")

    def test_gcam_pr_alias_prefers_gcam_pr_over_gcam(self):
        extractor = build_extractor_stub()

        result = extractor._fallback_extraction("tell me about GCAM PR")

        self.assertEqual(result["model"], "GCAM-PR 7.0")

    def test_message_ix_query_rejects_unrelated_exact_model(self):
        extractor = build_extractor_stub()

        result = extractor._validate_result(
            {"action": "query", "model": "GLOBIO"},
            "show data with message ix",
        )

        self.assertEqual(result["model"], "MESSAGEix-GLOBIOM 2.0")

    def test_message_ix_alias_survives_when_model_is_not_in_local_cache(self):
        extractor = build_extractor_stub()
        extractor.available_models = ["GCAM", "GLOBIO"]
        extractor.model_alias_map = extractor._build_model_alias_map(extractor.available_models)

        result = extractor._validate_result(
            {"action": "query", "model": "MESSAGEix-GLOBIOM 2.0"},
            "show data with message ix",
        )

        self.assertEqual(result["model"], "MESSAGEix-GLOBIOM 2.0")
        self.assertGreaterEqual(result["entity_confidence"]["model"], 0.75)

    def test_extract_uses_llm_when_deterministic_is_insufficient(self):
        class Response:
            content = '{"action": "query", "variable": null, "region": null, "scenario": null, "model": null}'

        class PromptShouldRun:
            class Chain:
                def invoke(self, _payload):
                    return Response()

            def __or__(self, _other):
                return self.Chain()

        extractor = build_extractor_stub()
        extractor.prompt = PromptShouldRun()
        extractor.llm = object()

        result = extractor.extract("help me understand the dataset")

        self.assertEqual(result["extraction_method"], "llm")

    def test_draw_is_a_plot_action_and_bare_emissions_defaults_to_co2(self):
        extractor = build_extractor_stub()

        result = extractor._fallback_extraction("draw emissions for World")

        self.assertEqual(result["action"], "plot")
        self.assertEqual(result["variable"], "Emissions|CO2")

    def test_multi_region_comparison_preserves_every_region(self):
        extractor = build_extractor_stub()
        extractor.available_regions.extend(["BRA", "IND"])

        result = extractor._fallback_extraction(
            "draw emissions for Brazil and India together"
        )

        self.assertEqual(result["region"], "BRA")
        self.assertEqual(result["regions"], ["BRA", "IND"])
        self.assertEqual(result["comparison"], "region")

    def test_domain_words_and_catalogue_commands_do_not_invent_regions(self):
        extractor = build_extractor_stub()
        extractor.available_regions.extend([
            "IND", "COL", "DEU", "GREECE", "AFR", "ZAF",
        ])
        extractor.available_models.append("TIAM")
        extractor.model_alias_map = extractor._build_model_alias_map(
            extractor.available_models
        )
        cases = (
            ("plot coal and gas primary energy for EU", ["EU"]),
            ("compare wind power and solar PV for Greece", ["GREECE"]),
            ("what kind of model is TIAM?", []),
            ("where can I find project publications?", []),
            ("how many scenarios are in the database?", []),
        )

        for query, expected_regions in cases:
            with self.subTest(query=query):
                result = extractor._fallback_extraction(query)
                actual = list(result.get("regions") or [])
                if not actual and result.get("region"):
                    actual = [result["region"]]
                self.assertEqual(actual, expected_regions)

    def test_exact_africa_alias_outranks_south_africa_fuzzy_match(self):
        extractor = build_extractor_stub()
        extractor.available_regions.extend(["AFR", "ZAF"])

        result = extractor._fallback_extraction(
            "which variables exist for Africa?"
        )

        self.assertEqual(result["region"], "AFR")
        self.assertNotEqual(result["region"], "ZAF")

    def test_explicit_bar_chart_type_is_preserved(self):
        extractor = build_extractor_stub()

        result = extractor._fallback_extraction(
            "bar chart of carbon price for World in 2050"
        )

        self.assertEqual(result["action"], "plot")
        self.assertEqual(result["chart_type"], "bar")
        self.assertEqual(result["start_year"], 2050)
        self.assertEqual(result["end_year"], 2050)

    def test_area_in_variable_wording_does_not_invent_chart_type(self):
        extractor = build_extractor_stub()
        extractor.available_variables.append("Land Cover|Cropland")

        result = extractor._fallback_extraction("cropland area for Brazil")
        plotted = extractor._fallback_extraction("make an area chart of cropland for Brazil")

        self.assertEqual(result["action"], "query")
        self.assertIsNone(result["chart_type"])
        self.assertEqual(plotted["action"], "plot")
        self.assertEqual(plotted["chart_type"], "area")

    def test_region_word_with_use_does_not_trigger_colliding_model(self):
        extractor = build_extractor_stub()
        extractor.available_models.append("China-MORE")
        extractor.available_regions.append("CHN")
        extractor.model_alias_map = extractor._build_model_alias_map(extractor.available_models)

        result = extractor._fallback_extraction(
            "Coal use in China's primary energy mix in 2040"
        )

        self.assertEqual(result["region"], "CHN")
        self.assertIsNone(result["model"])

    def test_unknown_region_does_not_erase_a_clear_variable(self):
        extractor = build_extractor_stub()

        result = extractor._fallback_extraction("CO2 for Atlantis")

        self.assertEqual(result["variable"], "Emissions|CO2")
        self.assertIsNone(result["region"])
        self.assertEqual(result["unmatched_region"], "Atlantis")

    def test_availability_grammar_and_model_are_not_variable_terms(self):
        extractor = build_extractor_stub()
        extractor.available_models.append("WITCH")
        extractor.model_alias_map = extractor._build_model_alias_map(
            extractor.available_models
        )

        result = extractor._fallback_extraction(
            "does WITCH report carbon price for EU?"
        )

        self.assertEqual(result["variable"], "Price|Carbon")
        self.assertEqual(result["model"], "WITCH")
        self.assertEqual(result.get("unmatched_variable_terms"), [])

    def test_common_emissions_and_region_typos_are_recovered_together(self):
        extractor = build_extractor_stub()

        result = extractor._fallback_extraction("emisions for europ")

        self.assertEqual(result["variable"], "Emissions|CO2")
        self.assertEqual(result["region"], "EU")


if __name__ == "__main__":
    unittest.main()
