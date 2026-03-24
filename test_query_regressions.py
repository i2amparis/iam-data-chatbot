import glob
import logging
import unittest

import pandas as pd

from data_utils import data_query
from fastapi_app import _split_answer_payload
from main import _extract_plot_markdown
from main import load_best_cached_results
from simple_plotter import simple_plot_query


def _load_cached_fixtures():
    model_file = max(glob.glob("cache/models*.json"), key=lambda f: len(pd.read_json(f)))
    results_file = max(glob.glob("cache/results*.json"), key=lambda f: len(pd.read_json(f)))
    models = pd.read_json(model_file).to_dict("records")
    ts = pd.read_json(results_file).to_dict("records")
    ts, _ = load_best_cached_results(ts)
    return models, ts


class QueryRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)
        cls.models, cls.ts = _load_cached_fixtures()

    def ask(self, query: str) -> str:
        return data_query(query, self.models, self.ts)

    def test_vague_region_only_query_asks_for_variable(self):
        response = self.ask("show me data for EU")
        self.assertIn("I found the region", response)
        self.assertIn("Which variable should I use?", response)

    def test_buildings_query_returns_ranked_variable_choices(self):
        response = self.ask("energy in buildings for EU")
        self.assertIn("Choose the variable:", response)
        self.assertIn("Reply with a number", response)
        self.assertIn("buildings", response.lower())

    def test_transport_carbon_query_requests_next_missing_piece(self):
        response = self.ask("carbon from transport in Europe")
        self.assertTrue(
            "Choose the variable:" in response or "Choose the scenario:" in response
        )

    def test_electricity_query_for_india_requests_scenario(self):
        response = self.ask("electricity for India")
        self.assertIn("which variable should i use?", response.lower())
        self.assertNotIn("Capacity|Electricity|Wind", response)
        self.assertTrue(
            "Secondary Energy|Electricity" in response
            or "Final Energy|Electricity" in response
            or "Capacity|Electricity" in response
        )

    def test_oil_query_returns_relevant_choices(self):
        response = self.ask("oil demand for EU")
        self.assertTrue(
            "Choose the variable:" in response or "Choose the scenario:" in response
        )
        self.assertIn("oil", response.lower())
        self.assertTrue(
            "Final Energy" in response
            or "Primary Energy" in response
            or "Secondary Energy" in response
        )

    def test_solar_capacity_query_prefers_capacity_over_capacity_additions(self):
        response = self.ask("solar capacity for EU")
        self.assertNotIn("### Capacity Additions|Electricity|Solar", response)
        self.assertNotIn("Capacity Additions|Electricity|Solar in EU", response)
        self.assertTrue(
            "Capacity|Electricity|Solar" in response
            or "Choose the scenario:" in response
            or "Choose the variable:" in response
        )

    def test_solar_energy_query_prefers_energy_or_capacity_family(self):
        response = self.ask("solar energy for EU under PR_WWH_CP")
        self.assertNotIn("Investment Share", response)
        self.assertTrue(
            "### Capacity|Electricity|Solar" in response
            or "### Secondary Energy|Electricity|Solar" in response
            or "Choose the scenario:" in response
            or "Choose the variable:" in response
        )

    def test_no_data_query_returns_guided_recovery(self):
        response = self.ask("CO2 emissions for EU28 under WWH")
        self.assertTrue(
            "Choose the region:" in response
            or "Closest variables:" in response
            or "Closest regions:" in response
        )

    def test_model_filtered_no_data_uses_compact_recovery(self):
        response = self.ask("CO2 emissions for EU28 under WWH for GCAM")
        self.assertIn("No data found", response)
        self.assertIn("Closest variables:", response)

    def test_unknown_model_returns_clear_guidance(self):
        response = self.ask("CO2 emissions for EU28 under WWH model REMIND")
        self.assertIn("couldn't match that to a known model", response.lower())

    def test_model_alias_lookup_handles_compact_form(self):
        response = self.ask("tell me about gcampr 7")
        self.assertIn("GCAM-PR", response)

    def test_model_alias_filter_does_not_trip_unknown_model(self):
        response = self.ask("CO2 emissions for World for model gcampr")
        self.assertNotIn("couldn't match that to a known model", response.lower())

    def test_generic_indicator_phrasing_lists_variables(self):
        response = self.ask("which indicators are included in the dataset")
        self.assertIn("I can work with these variables", response)

    def test_plot_variable_discovery_phrase_lists_variables(self):
        response = self.ask("What variables can you plot?")
        self.assertIn("I can work with these variables", response)
        self.assertNotIn("Could not identify a variable to plot", response)

    def test_generic_country_phrasing_lists_regions(self):
        response = self.ask("what countries are included")
        self.assertIn("I found regions like", response)

    def test_generic_model_info_phrasing_resolves_model(self):
        response = self.ask("information on gcam")
        self.assertIn("### GCAM", response)

    def test_list_available_models_stays_in_model_listing(self):
        response = self.ask("List available models")
        self.assertIn("There are", response)
        self.assertIn("models available", response)

    def test_renewable_share_query_does_not_fall_back_to_overview(self):
        response = self.ask("What is the renewable energy share in Europe by 2050 in the net zero scenario")
        self.assertNotIn("### What I can help you with", response)
        self.assertTrue(
            "Which variable should I use?" in response
            or "Choose the variable:" in response
            or "No data found for" in response
        )

    def test_carbon_price_query_is_not_treated_as_scenario_list(self):
        response = self.ask("Show me the carbon price trajectory for the EU under different scenarios")
        self.assertNotIn("I found scenarios like", response)
        self.assertNotIn("### What I can help you with", response)
        self.assertTrue(
            "Price|Carbon" in response
            or "Which variable should I use?" in response
            or "Choose the variable:" in response
            or "### Price|Carbon" in response
        )

    def test_model_assumptions_query_is_not_treated_as_overview(self):
        response = self.ask("What are the assumptions in the REMIND model regarding carbon capture and storage technology")
        self.assertNotIn("### What I can help you with", response)
        self.assertIn("couldn't match that to a known model", response.lower())

    def test_gcam_assumptions_query_reports_missing_assumptions_field(self):
        response = self.ask("What are the assumptions in the GCAM model?")
        self.assertIn("### GCAM", response)
        self.assertIn("No explicit assumptions field is available", response)

    def test_renewable_share_query_uses_honest_closest_variable_prompt(self):
        response = self.ask("What is the renewable energy share in Europe by 2050 in net zero scenario?")
        self.assertIn("could not find an explicit renewable share variable", response.lower())
        self.assertNotIn("Investment Share", response)

    def test_plot_with_explicit_scenario_does_not_claim_showing_all_scenarios(self):
        response = simple_plot_query("Plot Capacity|Electricity|Solar for EU under PR_WWH_CP", self.models, self.ts)
        self.assertIn("Showing Solar Capacity in EU for scenario `PR_WWH_CP`.", response)
        self.assertNotIn("Multiple scenarios exist; showing all.", response)

    def test_plot_solar_energy_query_prefers_solar_energy_or_capacity_family(self):
        response = simple_plot_query("Plot solar energy for EU under PR_WWH_CP", self.models, self.ts)
        self.assertNotIn("Investment|Energy Supply|Electricity|Solar", response)
        self.assertTrue(
            "Showing Solar Capacity in EU for scenario `PR_WWH_CP`." in response
            or "Showing Solar Electricity in EU for scenario `PR_WWH_CP`." in response
            or "Recommended variables: Capacity|Electricity|Solar" in response
            or "Recommended variables: Secondary Energy|Electricity|Solar" in response
        )

    def test_main_extracts_text_and_plot_markdown_cleanly(self):
        message, plot_data = _extract_plot_markdown(
            "Showing Solar Capacity in EU for scenario `PR_WWH_CP`.\n![Plot](data:image/png;base64,abc123)"
        )
        self.assertEqual(message, "Showing Solar Capacity in EU for scenario `PR_WWH_CP`.")
        self.assertEqual(plot_data, "data:image/png;base64,abc123")

    def test_fastapi_splits_plot_payload_and_notice(self):
        answer, plot_base64, plot_caption, notices = _split_answer_payload(
            "Showing Solar Capacity in EU for scenario `PR_WWH_CP`.\n"
            "No explicit assumptions field is available in the model metadata.\n"
            "![Plot](data:image/png;base64,abc123)"
        )
        self.assertEqual(answer, "Showing Solar Capacity in EU for scenario `PR_WWH_CP`.")
        self.assertEqual(plot_base64, "abc123")
        self.assertEqual(plot_caption, "Showing Solar Capacity in EU for scenario `PR_WWH_CP`.")
        self.assertEqual(notices, ["No explicit assumptions field is available in the model metadata."])


if __name__ == "__main__":
    unittest.main()
