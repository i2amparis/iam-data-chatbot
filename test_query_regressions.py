import glob
import logging
import unittest
from unittest.mock import patch

import pandas as pd

from canonical_aliases import explicit_scenarios_from_query
from data_metadata import DataMetadata
from data_utils import (
    data_query,
    format_time_series_data,
    unknown_named_region,
    _suggest_variable_candidates,
    _unmatched_variable_qualifiers,
)
from fastapi_app import _split_answer_payload
from main import _extract_plot_markdown
from main import load_best_cached_results
from resolved_scope import consume_resolved_scope
from simple_plotter import simple_plot_query, simple_plot_query_with_entities
from utils_query import region_mentions_from_query


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

    def test_baseline_family_plot_handles_case_variant_catalogue_labels(self):
        records = [
            {"variable": "Metric", "region": "R1", "scenario": "PR_Baseline",
             "modelName": "Model", "unit": "unit", "years": {"2050": 1}},
            {"variable": "Metric", "region": "R1", "scenario": "PR_NDC_CP",
             "modelName": "Model", "unit": "unit", "years": {"2050": 9}},
            {"variable": "Other", "region": "R2", "scenario": "Baseline",
             "modelName": "Model", "unit": "unit", "years": {"2050": 2}},
            {"variable": "Other", "region": "R3", "scenario": "baseline",
             "modelName": "Model", "unit": "unit", "years": {"2050": 3}},
        ]

        response = simple_plot_query_with_entities(
            "plot Metric for R1 under baseline in 2050",
            [],
            records,
            {
                "action": "plot",
                "variable": "Metric",
                "region": "R1",
                "scenario": "Baseline",
                "start_year": 2050,
                "end_year": 2050,
            },
        )
        scope = consume_resolved_scope()

        self.assertIn("![Plot]", response)
        self.assertNotIn("No data", response)
        self.assertIn("PR_Baseline", response)
        self.assertEqual(scope["scenarios"], ["PR_Baseline"])

    def test_baseline_family_table_filters_concrete_runtime_members(self):
        records = [
            {"variable": "Metric", "region": "R1", "scenario": "PR_Baseline",
             "modelName": "Model", "unit": "unit", "years": {"2050": 1}},
            {"variable": "Metric", "region": "R1", "scenario": "PR_NDC_CP",
             "modelName": "Model", "unit": "unit", "years": {"2050": 9}},
            {"variable": "Other", "region": "R2", "scenario": "Baseline",
             "modelName": "Model", "unit": "unit", "years": {"2050": 2}},
            {"variable": "Other", "region": "R3", "scenario": "baseline",
             "modelName": "Model", "unit": "unit", "years": {"2050": 3}},
        ]

        response = data_query(
            "show Metric for R1 under baseline in 2050",
            [],
            records,
            forced_entities={
                "variable": "Metric",
                "region": "R1",
                "scenario": "Baseline",
                "start_year": 2050,
                "end_year": 2050,
            },
        )

        self.assertIn("PR_Baseline", response)
        self.assertNotIn("PR_NDC_CP", response)

    def test_specific_unloaded_plot_wording_does_not_choose_unrelated_variable(self):
        response = simple_plot_query(
            "plot synthetic aviation fuel production for R1 in 2050",
            [],
            [
                {"variable": "Agricultural Production", "region": "R1", "scenario": "Path",
                 "modelName": "Model", "unit": "unit", "years": {"2050": 1}},
                {"variable": "Final Energy|Transportation|Aviation", "region": "R1", "scenario": "Path",
                 "modelName": "Model", "unit": "unit", "years": {"2050": 2}},
            ],
        )

        self.assertIn("could not confidently match", response.lower())
        self.assertNotIn("Showing Agricultural Production", response)

    def test_specific_unavailable_variable_family_has_no_unrelated_fallback(self):
        records = [
            {"variable": "Primary Energy|Nuclear", "region": "R1", "scenario": "Path", "years": {"2050": 1}},
            {"variable": "Final Energy|Oil", "region": "R1", "scenario": "Path", "years": {"2050": 2}},
        ]

        candidates = _suggest_variable_candidates(
            "hydrogen production", {}, records, region="R1", scenario="Path",
        )

        self.assertEqual(candidates, [])

    def test_unrepresented_modifier_is_reported_without_treating_region_as_qualifier(self):
        qualifiers = _unmatched_variable_qualifiers(
            "Show novel hydrogen production for South Korea in 2050",
            ["Secondary Energy|Hydrogen|Coal", "Secondary Energy|Hydrogen|Gas"],
            region="KOR",
        )

        self.assertEqual(qualifiers, ["novel"])

    def test_region_comparison_uses_single_scenario_from_plural_scope(self):
        records = [
            {"variable": "Metric", "region": region, "scenario": scenario,
             "modelName": "Model", "unit": "unit", "years": {"2050": value}}
            for region, scenario, value in (
                ("R1", "Path", 1), ("R2", "Path", 2),
                ("R1", "Other", 9), ("R2", "Other", 9),
            )
        ]

        response = simple_plot_query_with_entities(
            "Plot the comparison",
            [],
            records,
            {
                "variable": "Metric",
                "regions": ["R1", "R2"],
                "scenarios": ["Path"],
                "start_year": 2050,
                "end_year": 2050,
            },
        )
        scope = consume_resolved_scope()

        self.assertIn("for scenario `Path`", response)
        self.assertEqual(scope["scenario"], "Path")
        self.assertEqual(scope["scenarios"], ["Path"])
        self.assertEqual(scope["regions"], ["R1", "R2"])

    def test_region_comparison_recovers_verbatim_runtime_scenario_from_query(self):
        records = [
            {"variable": "Metric", "region": region, "scenario": scenario,
             "modelName": "Model", "unit": "unit", "years": {"2050": value}}
            for region, scenario, value in (
                ("R1", "Path_A", 1), ("R2", "Path_A", 2),
                ("R1", "Path_B", 9), ("R2", "Path_B", 9),
            )
        ]

        response = simple_plot_query_with_entities(
            "Plot Metric for R1 and R2 under Path_A in 2050",
            [],
            records,
            {"variable": "Metric", "regions": ["R1", "R2"], "start_year": 2050, "end_year": 2050},
        )
        scope = consume_resolved_scope()

        self.assertIn("for scenario `Path_A`", response)
        self.assertEqual(scope["scenarios"], ["Path_A"])

    def test_single_followup_scenario_does_not_collapse_structured_pair(self):
        records = [
            {
                "variable": "Metric",
                "region": "R1",
                "scenario": scenario,
                "modelName": "Alpha",
                "unit": "unit",
                "years": {"2050": value},
            }
            for scenario, value in (("Path A", 1), ("Path B", 2), ("Path C", 3))
        ]

        with (
            patch("simple_plotter.plt.plot") as plot_mock,
            patch("simple_plotter.plt.legend"),
        ):
            response = simple_plot_query_with_entities(
                "Compare it with Path B.",
                [],
                records,
                {
                    "action": "plot",
                    "variable": "Metric",
                    "region": "R1",
                    "scenarios": ["Path A", "Path B"],
                    "comparison": "scenario",
                    "start_year": 2050,
                    "end_year": 2050,
                },
            )
        scope = consume_resolved_scope()

        self.assertIn("![Plot]", response)
        self.assertEqual(plot_mock.call_count, 2)
        self.assertIn("for selected scenarios `Path A` and `Path B`", response)
        self.assertNotIn("across available scenarios", response)
        self.assertEqual(scope["scenarios"], ["Path A", "Path B"])
        self.assertFalse(scope["all_scenarios"])
        self.assertNotIn("Path C", scope["scenarios"])

    def test_plot_keeps_every_model_scenario_series_in_broad_scope(self):
        records = [
            {
                "variable": "Metric",
                "region": "R1",
                "scenario": scenario,
                "modelName": model,
                "unit": "unit",
                "years": {"2040": value, "2050": value + 1},
            }
            for model, scenario, value in (
                ("Alpha", "Path A", 1),
                ("Alpha", "Path B", 2),
                ("Beta", "Path A", 3),
                ("Beta", "Path B", 4),
            )
        ]

        with (
            patch("simple_plotter.plt.plot") as plot_mock,
            patch("simple_plotter.plt.legend"),
        ):
            response = simple_plot_query_with_entities(
                "Plot Metric for R1 across the selected scenarios",
                [],
                records,
                {
                    "action": "plot",
                    "variable": "Metric",
                    "region": "R1",
                    "scenarios": ["Path A", "Path B"],
                },
            )
        scope = consume_resolved_scope()

        self.assertIn("![Plot]", response)
        self.assertEqual(plot_mock.call_count, 4)
        labels = {call.kwargs["label"] for call in plot_mock.call_args_list}
        self.assertEqual(labels, {
            "Path A - Alpha", "Path A - Beta",
            "Path B - Alpha", "Path B - Beta",
        })
        self.assertIn("for selected scenarios `Path A` and `Path B`", response)
        self.assertNotIn("across available scenarios", response)
        self.assertFalse(scope["all_scenarios"])
        self.assertEqual(scope["result_models"], ["Alpha", "Beta"])

    def test_truly_broad_plot_records_and_captions_all_scenarios(self):
        records = [
            {
                "variable": "Metric",
                "region": "R1",
                "scenario": scenario,
                "modelName": "Alpha",
                "unit": "unit",
                "years": {"2050": value},
            }
            for scenario, value in (("Path A", 1), ("Path B", 2), ("Path C", 3))
        ]

        with (
            patch("simple_plotter.plt.plot") as plot_mock,
            patch("simple_plotter.plt.legend"),
        ):
            response = simple_plot_query_with_entities(
                "Plot Metric for R1 across available scenarios",
                [],
                records,
                {
                    "action": "plot",
                    "variable": "Metric",
                    "region": "R1",
                    "all_scenarios": True,
                },
            )
        scope = consume_resolved_scope()

        self.assertIn("![Plot]", response)
        self.assertEqual(plot_mock.call_count, 3)
        self.assertIn("across available scenarios", response)
        self.assertTrue(scope["all_scenarios"])
        self.assertEqual(scope["scenarios"], ["Path A", "Path B", "Path C"])

    def test_region_comparison_keeps_every_selected_scenario(self):
        records = [
            {
                "variable": "Metric", "region": region, "scenario": scenario,
                "modelName": "Alpha", "unit": "unit",
                "years": {"2040": value, "2050": value + 1},
            }
            for region, scenario, value in (
                ("R1", "Path A", 1), ("R1", "Path B", 2),
                ("R2", "Path A", 3), ("R2", "Path B", 4),
            )
        ]

        with (
            patch("simple_plotter.plt.plot") as plot_mock,
            patch("simple_plotter.plt.legend"),
        ):
            response = simple_plot_query_with_entities(
                "Plot the region comparison",
                [],
                records,
                {
                    "action": "plot", "variable": "Metric",
                    "regions": ["R1", "R2"],
                    "scenarios": ["Path A", "Path B"],
                    "comparison": "region",
                },
            )
        scope = consume_resolved_scope()

        self.assertIn("![Plot]", response)
        self.assertEqual(plot_mock.call_count, 4)
        self.assertEqual(
            {call.kwargs["label"] for call in plot_mock.call_args_list},
            {"R1 (Path A)", "R1 (Path B)", "R2 (Path A)", "R2 (Path B)"},
        )
        self.assertIn("for selected scenarios `Path A` and `Path B`", response)
        self.assertNotIn("across available scenarios", response)
        self.assertEqual(scope["scenarios"], ["Path A", "Path B"])
        self.assertFalse(scope["all_scenarios"])

    def test_vague_region_only_query_asks_for_variable(self):
        response = self.ask("show me data for EU")
        self.assertIn("I found the region", response)
        self.assertIn("Which variable should I use?", response)

    def test_generic_by_country_request_asks_for_concrete_region(self):
        records = [
            {
                "variable": "Emissions|Kyoto Gases",
                "region": region,
                "scenario": "Baseline",
                "modelName": "Model",
                "unit": "Mt CO2-equiv/yr",
                "years": {"2050": value},
            }
            for region, value in (("CHN", 1), ("IND", 2), ("USA", 3))
        ]

        response = data_query(
            "greenhouse gas pathways by country",
            [],
            records,
            forced_entities={"variable": "Emissions|Kyoto Gases"},
        )

        self.assertIn("specific country or region", response)
        self.assertIn("Choose the region:", response)
        self.assertIn("`CHN`", response)
        self.assertNotIn("Summary:", response)

    def test_carbon_price_table_normalizes_equivalent_unit_spellings(self):
        records = [
            {
                "variable": "Price|Carbon",
                "region": "World",
                "scenario": scenario,
                "modelName": "Model",
                "unit": unit,
                "years": {"2050": value},
            }
            for scenario, unit, value in (
                ("A", "US$2010/tCO2", 1),
                ("B", "US$2010/t CO2", 2),
                ("C", "US$2010/t", 3),
            )
        ]

        response = format_time_series_data(
            records,
            "Price|Carbon",
            "World",
            2050,
            2050,
        )

        self.assertIn("Unit: `US$2010/t CO2`", response)
        self.assertNotIn("Unit: `multiple`", response)

    def test_table_scope_records_multiple_units_instead_of_stale_single_unit(self):
        records = [
            {
                "variable": "GDP|MER",
                "region": "World",
                "scenario": scenario,
                "modelName": model,
                "unit": unit,
                "years": {"2050": value},
            }
            for scenario, model, unit, value in (
                ("Path A", "Model A", "billion US$2010/yr", 1),
                ("Path B", "Model B", "billion US$2010/yr OR local currency", 2),
            )
        ]

        response = format_time_series_data(records, "GDP|MER", "World", 2050, 2050)
        scope = consume_resolved_scope()

        self.assertIn("Unit: `multiple`", response)
        self.assertEqual(scope["unit"], "multiple")

    def test_vague_natural_language_region_and_year_asks_for_variable(self):
        response = self.ask("show data for Europe in 2060")

        self.assertIn("I found the region", response)
        self.assertIn("Which variable should I use?", response)
        self.assertNotIn("Emissions|CO", response)

    def test_runtime_region_mention_is_not_a_variable_qualifier(self):
        query = "plot electricity for Aurora Basin in 2050"
        mentions = region_mentions_from_query(
            query,
            {},
            ["Aurora Basin"],
            resolved_region="Aurora Basin",
        )
        qualifiers = _unmatched_variable_qualifiers(
            query,
            ["Final Energy|Electricity", "Secondary Energy|Electricity"],
            region="Aurora Basin",
            ignored_values=mentions,
        )

        self.assertEqual(mentions, ["Aurora Basin"])
        self.assertEqual(qualifiers, [])

    def test_arbitrary_runtime_region_is_excluded_from_candidate_ranking(self):
        records = [{
            "variable": "Aurora Output",
            "region": "Aurora Basin",
            "scenario": "Path",
            "modelName": "Model",
            "years": {"2050": 1},
        }]

        response = data_query("show data for Aurora Basin in 2050", [], records)

        self.assertIn("I found the region", response)
        self.assertIn("Which variable should I use?", response)
        self.assertNotIn("Aurora Output", response)

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

    def test_broad_oil_query_requests_clarification(self):
        response = self.ask("oil")
        self.assertTrue("Choose the variable:" in response or "Which variable should I use?" in response)
        self.assertIn("oil", response.lower())
        if "Choose the variable:" in response:
            self.assertIn("1. `Primary Energy|Oil`", response)

    def test_broad_emissions_query_defaults_to_co2_and_requests_scope(self):
        response = self.ask("emissions")
        self.assertIn("Choose the region:", response)
        self.assertIn("Emissions|CO2", response)
        self.assertIn("emissions", response.lower())

    def test_broad_policy_query_does_not_return_empty_answer(self):
        response = self.ask("policy")
        self.assertTrue(response.strip())
        self.assertIn("policy", response.lower())
        self.assertIn("one more detail", response.lower())

    def test_broad_ndc_query_does_not_return_empty_answer(self):
        response = self.ask("NDC")
        self.assertTrue(response.strip())
        self.assertIn("ndc", response.lower())
        self.assertIn("one more detail", response.lower())
        self.assertIn("NDC emissions for EU", response)
        self.assertNotIn("Current Policies", response)

    def test_region_scoped_variable_listing_keeps_region_in_examples(self):
        response = data_query(
            "which variables exist for Africa?",
            [],
            [{
                "variable": "Primary Energy",
                "region": "AFR",
                "scenario": "Baseline",
                "modelName": "Model",
                "unit": "EJ/yr",
                "years": {"2050": 1},
            }],
        )

        self.assertIn("region `AFR", response)
        self.assertIn("for AFR", response)
        self.assertNotIn("for Greece", response)

    def test_broad_electricity_query_prefers_high_level_choices(self):
        response = self.ask("electricity")
        self.assertIn("Choose the variable:", response)
        self.assertIn("Electricity", response)
        self.assertNotIn("Carbon Capture|Energy|Supply|Electricity|Synthetic Fuels|Industrial Processes", response)

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
        self.assertTrue(
            "Closest variables:" in response
            or "Closest regions:" in response
            or "Choose the region:" in response
        )

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

    def test_data_categories_phrase_returns_discovery_overview(self):
        response = self.ask("show all data categories")
        self.assertIn("What I can help you with", response)
        self.assertNotIn("Choose the variable:", response)

    def test_what_data_do_you_have_returns_discovery_overview(self):
        for phrase in (
            "what data do you have?",
            "what data is available",
            "what data can you show me",
            "what's in the dataset",
        ):
            response = self.ask(phrase)
            self.assertIn("What I can help you with", response, msg=phrase)
            self.assertNotIn("Choose the variable:", response, msg=phrase)

    def test_generic_model_info_phrasing_resolves_model(self):
        response = self.ask("information on gcam")
        self.assertIn("### GCAM", response)

    def test_list_available_models_stays_in_model_listing(self):
        response = self.ask("List available models")
        self.assertIn("There are", response)
        self.assertIn("models available", response)
        self.assertNotIn("Showing ", response)

    def test_long_variable_lists_offer_show_all(self):
        response = self.ask("list variables")
        self.assertIn("I can work with these variables", response)
        self.assertIn("show all variables", response)

    def test_variable_catalog_sanitizes_source_only_label_artifacts(self):
        response = self.ask("which variables are available?")
        self.assertNotIn("- 'Economy", response)
        self.assertNotIn("|Energys", response)

    def test_long_scenario_lists_offer_show_all(self):
        response = self.ask("list scenarios")
        self.assertIn("I found scenarios like", response)
        self.assertIn("show all scenarios", response)

    def test_carbon_dioxide_synonym_query_routes_to_emissions(self):
        response = self.ask("show me carbon dioxide emissions for Europe")
        self.assertNotIn("### What I can help you with", response)
        self.assertIn("Emissions|CO2", response)
        self.assertNotIn("CO2 emissions cuts|Absolute", response)

    def test_photovoltaic_synonym_query_routes_to_solar_capacity(self):
        response = self.ask("plot photovoltaic capacity for Greece")
        self.assertNotIn("### What I can help you with", response)
        self.assertIn("Solar", response)
        self.assertIn("Capacity|Electricity|Solar", response)
        self.assertTrue(
            "Which variable should I use?" in response
            or "Choose the variable:" in response
            or "![Plot]" in response
            or "No data found" in response
        )

    def test_primary_energy_from_coal_keeps_carrier_qualifier(self):
        response = self.ask("show primary energy from coal for Europe in 2030 across available scenarios")
        self.assertIn("Showing Coal", response)
        self.assertNotIn("Showing Primary Energy in", response)

    def test_solar_pv_generation_never_silently_becomes_capacity(self):
        response = self.ask("plot electricity generation from solar PV for Greece from 2020 to 2050")
        if "Capacity|" in response:
            self.assertIn("No data found", response)
        self.assertNotIn("No data found for **Capacity|Electricity|Solar**", response)

    def test_generic_solar_data_for_greece_prefers_core_solar_variable(self):
        response = self.ask("show solar data for Greece")
        self.assertIn("Solar", response)
        self.assertNotIn("Capacity Additions|Electricity|Solar", response)
        self.assertNotIn("Investment|Energy Supply|Electricity|Solar", response)
        self.assertNotIn("Average Annual Investment", response)
        self.assertNotIn("Battery", response)

    def test_generic_wind_data_for_greece_prefers_core_wind_variable(self):
        response = self.ask("show wind data for Greece")
        self.assertIn("Wind", response)
        self.assertNotIn("Capacity Additions|Electricity|Wind", response)
        self.assertNotIn("Investment|Energy Supply|Electricity|Wind", response)

    def test_gross_domestic_product_routes_to_gdp_mer(self):
        response = self.ask("gross domestic product for World")
        self.assertNotIn("### What I can help you with", response)
        self.assertIn("GDP|MER", response)

    def test_gdp_no_data_recovery_stays_in_economic_family(self):
        response = self.ask("show GDP for World under Baseline")
        self.assertIn("GDP|MER", response)
        self.assertNotIn("Agricultural Demand", response)
        self.assertNotIn("Final Energy", response)

    def test_which_models_are_available_uses_model_listing(self):
        response = self.ask("which models are available")
        self.assertIn("models available", response)
        self.assertNotIn("Showing ", response)

    def test_compare_wind_power_and_solar_pv_routes_to_plot_or_choice(self):
        response = self.ask("compare wind power and solar PV")
        self.assertNotIn("### What I can help you with", response)
        self.assertTrue("![Plot]" in response or "Which variable should I use?" in response)

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
        self.assertIn("### REMIND", response)
        self.assertIn("Assumptions:", response)
        self.assertIn("scenario-dependent", response)
        self.assertIn("[IAM PARIS Models](https://iamparis.eu/models)", response)

    def test_witch_model_profile_fills_missing_local_metadata(self):
        response = self.ask("Explain the WITCH model")
        self.assertIn("### WITCH", response)
        self.assertIn("integrated assessment model", response.lower())
        self.assertIn("[IAM PARIS Models](https://iamparis.eu/models)", response)

    def test_messageix_model_profile_handles_alias(self):
        response = self.ask("Tell me about MESSAGEix model")
        self.assertIn("### MESSAGEix-GLOBIOM", response)
        self.assertIn("energy-system", response.lower())
        self.assertIn("[IAM PARIS Models](https://iamparis.eu/models)", response)

    def test_gcampr_model_profile_handles_compact_alias(self):
        response = self.ask("What is gcampr 7?")
        self.assertIn("### GCAM-PR", response)
        self.assertIn("regional", response.lower())
        self.assertIn("[IAM PARIS Models](https://iamparis.eu/models)", response)

    def test_gcam_assumptions_query_reports_missing_assumptions_field(self):
        response = self.ask("What are the assumptions in the GCAM model?")
        self.assertIn("### GCAM", response)
        self.assertIn("Description:", response)
        self.assertIn("Assumptions:", response)
        self.assertIn("No explicit assumptions field is available", response)
        self.assertRegex(response, r"\[IAM PARIS model page\]\(https://iamparis\.eu/models/\d+\)")

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

    def test_co2_emissions_keeps_canonical_variable_not_fuzzy_oc(self):
        # Regression: a confident canonical alias ("co2 emissions" -> Emissions|CO2)
        # was nulled as "ambiguous" and replaced by a YAML fuzzy match
        # (Emissions|OC). The canonical match must survive.
        response = self.ask("CO2 emissions for Greece")
        self.assertIn("Emissions|CO2", response)
        self.assertNotIn("Emissions|OC", response)

    def test_under_current_policies_extracts_scenario(self):
        # Regression: a double-escaped regex in _match_scenario_name never matched,
        # so "under current policies" failed to resolve a scenario.
        response = self.ask("emissions under current policies for EU")
        self.assertNotIn("Emissions|OC", response)
        self.assertIn("Emissions|CO2", response)

    def test_current_policies_scenario_returns_data_via_family(self):
        # Regression: canonical scenario "Current Policies" has no verbatim code in
        # the dataset (codes look like PR_CurPol_CP). Family-aware matching must
        # return data instead of a false no-data.
        response = data_query(
            "emissions under current policies for EU",
            self.models, self.ts,
            forced_entities={"variable": "Emissions|CO2", "region": "EU", "scenario": "Current Policies"},
        )
        self.assertIn("Emissions|CO2 in EU", response)
        self.assertNotIn("could not find data", response.lower())
        self.assertNotIn("no data found", response.lower())

    def test_net_zero_no_data_names_scope_and_only_offers_compatible_changes(self):
        records = [
            {
                "variable": "Emissions|CO2", "region": "World",
                "scenario": "Baseline", "modelName": "Global Model",
                "unit": "Mt CO2/yr", "years": {"2050": 100},
            },
            {
                "variable": "Emissions|CO2", "region": "World",
                "scenario": "PR_NDC_CP", "modelName": "Global Model",
                "unit": "Mt CO2/yr", "years": {"2050": 80},
            },
            {
                "variable": "Emissions|CO2", "region": "EU27",
                "scenario": "NZE_Bench_H", "modelName": "EU Model",
                "unit": "Mt CO2/yr", "years": {"2050": 10},
            },
            {
                "variable": "Emissions|CO2", "region": "AFR",
                "scenario": "RegionalOnly", "modelName": "Regional Model",
                "unit": "Mt CO2/yr", "years": {"2050": 50},
            },
            {
                "variable": "Other", "region": "World",
                "scenario": "NZE_Bench_L", "modelName": "Other Model",
                "unit": "unit", "years": {"2050": 1},
            },
        ]

        response = data_query(
            "plot emissions for World under net zero",
            [],
            records,
            forced_entities={
                "variable": "Emissions|CO2",
                "region": "World",
                "scenario": "Net Zero",
            },
            metadata=DataMetadata(records),
        )

        self.assertIn("`Emissions|CO2`", response)
        self.assertIn("region `World`", response)
        self.assertIn("scenario family `Net Zero`", response)
        self.assertIn("region `EU27`", response)
        self.assertIn("scenario `Baseline`", response)
        self.assertIn("scenario `PR_NDC_CP`", response)
        self.assertNotIn("RegionalOnly", response)
        self.assertNotIn("NZE_Bench_L", response)

    def test_net_zero_plot_no_data_uses_the_same_compatible_recovery_scope(self):
        records = [
            {
                "variable": "Emissions|CO2", "region": "World",
                "scenario": "Baseline", "modelName": "Global Model",
                "unit": "Mt CO2/yr", "years": {"2050": 100},
            },
            {
                "variable": "Emissions|CO2", "region": "World",
                "scenario": "CP_EI", "modelName": "Global Model",
                "unit": "Mt CO2/yr", "years": {"2050": 80},
            },
            {
                "variable": "Emissions|CO2", "region": "EU27",
                "scenario": "NZE_Bench_H", "modelName": "EU Model",
                "unit": "Mt CO2/yr", "years": {"2050": 10},
            },
            {
                "variable": "Emissions|CO2", "region": "AFR",
                "scenario": "RegionalOnly", "modelName": "Regional Model",
                "unit": "Mt CO2/yr", "years": {"2050": 50},
            },
        ]

        response = simple_plot_query_with_entities(
            "plot emissions for World under net zero",
            [],
            records,
            {
                "action": "plot", "variable": "Emissions|CO2",
                "region": "World", "scenario": "Net Zero",
            },
        )

        self.assertIn("`Emissions|CO2`", response)
        self.assertIn("region `World`", response)
        self.assertIn("scenario family `Net Zero`", response)
        self.assertIn("region `EU27`", response)
        self.assertIn("scenario `Baseline`", response)
        self.assertIn("scenario `CP_EI`", response)
        self.assertNotIn("RegionalOnly", response)
        self.assertNotIn("![Plot]", response)

    def test_baseline_scenario_returns_data_via_family(self):
        response = data_query(
            "CO2 emissions for EU under baseline",
            self.models, self.ts,
            forced_entities={"variable": "Emissions|CO2", "region": "EU", "scenario": "Baseline"},
        )
        self.assertIn("Emissions|CO2 in EU", response)
        self.assertNotIn("could not find data", response.lower())

    def test_forced_model_without_data_does_not_substitute_other_models(self):
        # An explicit model with no matching rows must remain a hard scope.  A
        # plausible table from unrelated models is more misleading than no data.
        response = data_query(
            "GDP for China from GCAM",
            self.models, self.ts,
            forced_entities={"variable": "GDP|MER", "region": "CHN", "model": "GCAM 7.0"},
        )
        self.assertIn("explicitly requested model `GCAM 7.0`", response)
        self.assertIn("did not substitute results from a different model", response)
        self.assertNotIn("### GDP|MER in CHN", response)

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

    def test_workspace_listing_phrasings_return_workspace_list(self):
        for phrase in ("list workspaces", "show all workspaces", "what workspaces are there?"):
            response = self.ask(phrase)
            self.assertIn("I found these workspaces", response, msg=phrase)
            self.assertNotIn("I need one more detail", response, msg=phrase)
            self.assertNotIn("Which variable should I use?", response, msg=phrase)

    def test_forced_model_canonicalized_to_record_name(self):
        # The entity extractor emits a display-cased model name ("GCAM") that
        # differs from the record name ("gcam"); the model filter must still
        # find the data instead of falsely reporting "no timeseries data".
        response = data_query(
            "Primary Energy for gcam",
            self.models,
            self.ts,
            forced_entities={"variable": "Primary Energy", "model": "GCAM"},
        )
        self.assertIn("model `gcam`", response)
        self.assertNotIn("no timeseries data for model", response.lower())

    def test_complete_data_scope_with_model_stays_strict_and_does_not_describe_model(self):
        response = data_query(
            "show Primary Energy|Coal for EU under PR_WWH_CP in 2040 for model gcam",
            self.models,
            self.ts,
            forced_entities={
                "variable": "Primary Energy|Coal",
                "region": "EU",
                "scenario": "PR_WWH_CP",
                "model": "GCAM",
                "start_year": 2040,
                "end_year": 2040,
            },
        )

        self.assertIn("explicitly requested model `GCAM`", response)
        self.assertIn("did not substitute results from a different model", response)
        self.assertNotIn("### GCAM\n\nDescription:", response)

    def test_explicit_pipe_variable_beats_extractor_superstring(self):
        # The extractor can drift "Secondary Energy|Electricity" to the
        # superstring "Price|Secondary Energy|Electricity"; the exact variable
        # the user typed must win.
        response = data_query(
            "Secondary Energy|Electricity for USA",
            self.models,
            self.ts,
            forced_entities={
                "variable": "Price|Secondary Energy|Electricity",
                "region": "USA",
            },
        )
        self.assertIn("Secondary Energy|Electricity in USA", response)
        self.assertNotIn("Price|Secondary Energy|Electricity", response)


    def test_model_scoped_scenarios_direct_for_named_model(self):
        response = self.ask("what scenarios does GCAM have?")
        self.assertIn("Model `GCAM` has these scenarios", response)
        self.assertIn("Runtime labels included:", response)
        self.assertIn("Baseline", response)
        self.assertNotIn("What I can help you with", response)
        self.assertNotIn("I found scenarios like", response)

    def test_model_scoped_listing_works_for_any_model(self):
        # Not just gcam: a different model must be scoped to its own records.
        response = self.ask("what scenarios does ices have?")
        self.assertIn("Model `ICES-XPS 1.1` has these scenarios", response)
        self.assertNotIn("What I can help you with", response)

    def test_model_scoped_regions_and_variables(self):
        regions = self.ask("which regions does gcam cover?")
        self.assertIn("Model `GCAM` has these regions", regions)
        self.assertIn("Runtime labels included:", regions)
        variables = self.ask("what variables does gcam have?")
        self.assertIn("Model `GCAM` has these variables", variables)
        self.assertIn("Runtime labels included:", variables)

    def test_model_scoped_followup_uses_carried_model(self):
        # Mirrors the manager injecting the carried model for "what scenarios
        # does it have" after a model_explanation turn.
        response = data_query(
            "what scenarios does it have model GCAM",
            self.models,
            self.ts,
            forced_entities={"model": "GCAM"},
        )
        self.assertIn("Model `gcam` has these scenarios", response)
        self.assertNotIn("What I can help you with", response)

    def test_generic_scenario_list_stays_unscoped(self):
        response = self.ask("list scenarios")
        self.assertIn("I found scenarios like", response)
        self.assertNotIn("Model `", response)


    def test_plot_model_filter_matches_record_name(self):
        # Extractor emits "GCAM"; records use "gcam". The plot model filter must
        # canonicalize so it is not silently emptied (previously it also read the
        # wrong 'model' field instead of 'modelName').
        response = simple_plot_query_with_entities(
            "plot Primary Energy|Coal for gcam in CHN under PR_CurPol_CP",
            self.models,
            self.ts,
            {
                "variable": "Primary Energy|Coal",
                "model": "GCAM",
                "region": "CHN",
                "scenario": "PR_CurPol_CP",
            },
        )
        self.assertNotIn("No data found", response)
        self.assertIn("data:image", response)

    def test_plot_prefers_verbatim_variable_over_extractor_drift(self):
        # Extractor drifts "Final Energy|Industry" to a superstring; the plot path
        # must honour the variable the user typed verbatim.
        response = simple_plot_query_with_entities(
            "plot Final Energy|Industry for IND under NDC_EI",
            self.models,
            self.ts,
            {
                "variable": "Final Energy (excl. feedstocks)|Industry",
                "region": "IND",
                "scenario": "NDC_EI",
            },
        )
        self.assertNotIn("No data found", response)
        self.assertIn("data:image", response)

    def test_verbatim_scenario_beats_extractor_family_collapse(self):
        # "BAU" is a real scenario the extractor collapsed to "Baseline"; the
        # typed name must win so GREECE+BAU data is found instead of no-data.
        response = data_query(
            "Primary Energy|Coal in GREECE under BAU from 2020 to 2050",
            self.models,
            self.ts,
            forced_entities={
                "variable": "Primary Energy|Coal",
                "region": "GREECE",
                "scenario": "Baseline",
            },
        )
        self.assertIn("scenario `BAU`", response)
        self.assertNotIn("could not find data", response.lower())

    def test_explicit_scenarios_respect_word_boundaries(self):
        available = ["Baseline", "PR_Baseline", "PR_NDC_CP", "BAU", "WWH"]
        # A generic family label must not match inside a distinct code.
        self.assertEqual(
            explicit_scenarios_from_query("emissions under PR_Baseline and PR_NDC_CP", available),
            ["PR_Baseline", "PR_NDC_CP"],
        )
        self.assertEqual(
            explicit_scenarios_from_query("coal under BAU", available),
            ["BAU"],
        )


    def test_tell_me_about_model_does_not_match_stopword_models(self):
        # "tell me about REMIND": the stopword "me" must not match E3ME/MEDEAS,
        # and the model profile should be returned instead of a spurious list.
        for name, header in (("REMIND", "### REMIND"), ("MESSAGEix", "### MESSAGEix")):
            response = self.ask(f"tell me about {name}")
            self.assertIn(header, response, msg=name)
            self.assertNotIn("multiple model matches", response, msg=name)


    def test_hyphenated_model_name_is_not_shadowed_by_substring(self):
        # "MESSAGEix-GLOBIOM" contains the unrelated ts model "GLOBIO" as a
        # substring; the profile the user named must win.
        response = self.ask("tell me about MESSAGEix-GLOBIOM")
        self.assertIn("### MESSAGEix-GLOBIOM", response)
        self.assertNotIn("### GLOBIO ", response)

    def test_from_model_clause_applies_model_filter(self):
        # The extractor resolves the variable/region but misses "from E3ME"; the
        # explicit clause must still scope the query to that model (canonicalized
        # to the record name "e3me").
        response = data_query(
            "CO2 emissions for EU from E3ME",
            self.models,
            self.ts,
            forced_entities={"variable": "Emissions|CO2", "region": "EU"},
        )
        self.assertIn("model `e3me`", response)

    def test_from_model_clause_reports_absent_model_honestly(self):
        # MUSE is a real model but carries no Final Energy timeseries; the clause
        # must resolve the model and the answer must name it, not silently ignore.
        response = data_query(
            "final energy for EU from MUSE",
            self.models,
            self.ts,
            forced_entities={"variable": "Final Energy", "region": "EU"},
        )
        self.assertIn("MUSE", response)


    def test_what_data_does_model_have_lists_its_variables(self):
        # "what data does GCAM have for the EU" names a model, so it must scope to
        # that model's variables instead of the generic dataset overview.
        response = data_query(
            "what data does GCAM have for the EU?",
            self.models,
            self.ts,
            forced_entities={"model": "GCAM", "region": "EU"},
        )
        self.assertIn("Model `gcam` has these variables", response)
        self.assertNotIn("What I can help you with", response)


    def test_unknown_named_region_helper(self):
        regions = ["EU", "World", "CHN", "Japan", "Brazil", "USA"]
        self.assertEqual(unknown_named_region("CO2 for Gotham", regions), "Gotham")
        self.assertEqual(unknown_named_region("co2 for gotham", regions), "gotham")
        # Strict matching: a partial must not be accepted as a real region.
        self.assertEqual(unknown_named_region("GDP for Middle Earth", regions), "Middle Earth")
        self.assertEqual(unknown_named_region("population for mars in 2050", regions), "mars")
        # Real regions / aliases / no place must not be flagged.
        self.assertIsNone(unknown_named_region("CO2 for EU", regions))
        self.assertIsNone(unknown_named_region("GDP for China", regions))
        self.assertIsNone(unknown_named_region("emissions for World", regions))
        self.assertIsNone(unknown_named_region("final energy in industry for China", regions))
        self.assertIsNone(unknown_named_region("CO2 emissions", regions))

    def test_data_query_rejects_unknown_region_not_inventing_data(self):
        for q in (
            "CO2 emissions for Gotham",
            "GDP for Middle Earth",
            "population for Mars in 2050",
            "population for mars in 2050",
        ):
            response = self.ask(q)
            self.assertIn("as a region in the IAM PARIS data", response, msg=q)
            self.assertNotIn("| Year |", response, msg=q)

    def test_unknown_region_is_rejected_before_plot_variable_ranking(self):
        response = self.ask("plot synthetic aviation fuel production for Atlantis in 2050")

        self.assertIn("as a region in the IAM PARIS data", response)
        self.assertNotIn("qualifier", response.lower())
        self.assertNotIn("![Plot]", response)

    def test_data_query_still_serves_known_regions(self):
        for q in ("CO2 emissions for EU", "final energy demand in industry for China",
                  "carbon capture and storage for World"):
            response = self.ask(q)
            self.assertNotIn("as a region in the IAM PARIS data", response, msg=q)


if __name__ == "__main__":
    unittest.main()
