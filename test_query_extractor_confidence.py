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
        "GDP|MER",
    ]
    extractor.available_regions = ["World", "EU", "IS"]
    extractor.variable_units = {"Emissions|CO2": "Mt CO2/yr"}
    extractor.variable_dict = {}
    extractor.region_dict = {}
    extractor.model_alias_map = extractor._build_model_alias_map(extractor.available_models)
    return extractor


class QueryExtractorConfidenceTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
