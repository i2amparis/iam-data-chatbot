import tempfile
import unittest
from pathlib import Path

from data_metadata import DataMetadata, build_metadata_with_cache
from data_utils import data_query, format_time_series_data
from model_aliases import UNLABELLED_MODEL_LABEL
from resolved_scope import consume_resolved_scope
from simple_plotter import simple_plot_query


class DataMetadataTests(unittest.TestCase):
    def test_numeric_source_model_ids_are_retained_but_not_presented(self):
        metadata = DataMetadata(
            [
                {
                    "variable": "Emissions|CO2", "region": "World",
                    "scenario": "Baseline", "modelName": "42",
                    "years": {"2030": 1},
                },
                {
                    "variable": "Emissions|CO2", "region": "World",
                    "scenario": "Baseline", "modelName": "GCAM 7.0",
                    "years": {"2030": 2},
                },
            ],
            models=[{"modelName": "42"}, {"modelName": "GCAM 7.0"}],
        )

        self.assertIn("42", metadata.all_model_names)
        self.assertEqual(metadata.distinct_model_labels(), ["GCAM 7.0"])
        self.assertEqual(
            metadata.get_available_for_region("World")["models"],
            ["GCAM 7.0"],
        )

    def test_model_listing_omits_numeric_source_ids(self):
        models = [
            {"modelName": "42"},
            {"modelName": "GCAM 7.0"},
            {"modelName": "WITCH 6.0"},
        ]

        response = data_query("list models", models, [])

        self.assertIn("GCAM 7.0", response)
        self.assertIn("WITCH 6.0", response)
        self.assertNotIn("42", response)

    def test_model_listing_does_not_append_showing_count_footer(self):
        models = [
            {"modelName": f"Model {index:02d}"}
            for index in range(1, 10)
        ]

        response = data_query("what models are available?", models, [])

        self.assertIn("There are 9 models available.", response)
        self.assertNotIn("Showing 8 of 9", response)
        self.assertNotIn("show all models", response)

    def test_unknown_model_coverage_question_uses_model_documentation_not_variables(self):
        models = [{
            "modelName": "ALADIN",
            "id": 6,
            "description": "Alternative automobiles diffusion model.",
        }]
        ts = [
            {
                "modelName": "ALADIN",
                "variable": "Final Energy|Transportation|Rail",
                "region": "World",
                "scenario": "Baseline",
                "years": {"2030": 1},
            },
            {
                "modelName": "ALADIN",
                "variable": "Emissions|CO2|Energy|Demand|Transportation|Road|Freight",
                "region": "World",
                "scenario": "Baseline",
                "years": {"2030": 1},
            },
        ]

        response = data_query("Does ALADIN covers the energy sector?", models, ts)

        self.assertIn("I cannot verify", response)
        self.assertIn("https://iamparis.eu/models/6", response)
        self.assertNotIn("closest match", response.lower())
        self.assertNotIn("Final Energy|Transportation|Rail", response)

    def test_workspace_listing_uses_public_study_titles_not_internal_codes(self):
        ts = [
            {"workspace_code": "world-headed", "variable": "A", "scenario": "Baseline"},
            {"workspace_code": "afolu", "variable": "B", "scenario": "NDC"},
        ]

        response = data_query("list workspaces", [], ts)

        self.assertIn("Where is the world headed?", response)
        self.assertIn("AFOLU transformation", response)
        self.assertNotIn("world-headed", response)
        self.assertNotIn("workspace_code", response)

    def test_numeric_query_requires_study_when_multiple_studies_are_loaded(self):
        ts = [
            {
                "workspace_code": "world-headed", "variable": "Emissions|CO2",
                "region": "World", "scenario": "Baseline", "modelName": "Model",
                "unit": "Mt CO2/yr", "years": {"2050": 100},
            },
            {
                "workspace_code": "afolu", "variable": "Emissions|CO2",
                "region": "World", "scenario": "Baseline", "modelName": "Model",
                "unit": "Mt CO2/yr", "years": {"2050": 999},
            },
        ]

        response = data_query(
            "What are global emissions in this study for 2050?",
            [],
            ts,
            forced_entities={
                "variable": "Emissions|CO2", "region": "World",
                "start_year": 2050, "end_year": 2050,
            },
        )

        self.assertIn("Please choose a study", response)
        self.assertNotIn("100.00", response)
        self.assertNotIn("999.00", response)

    def test_numeric_query_is_independent_of_cross_study_record_order(self):
        records = [
            {
                "workspace_code": "world-headed", "variable": "Emissions|CO2",
                "region": "World", "scenario": "Baseline", "modelName": "Model",
                "unit": "Mt CO2/yr", "years": {"2050": 100},
            },
            {
                "workspace_code": "afolu", "variable": "Emissions|CO2",
                "region": "World", "scenario": "Baseline", "modelName": "Model",
                "unit": "Mt CO2/yr", "years": {"2050": 999},
            },
        ]
        entities = {
            "variable": "Emissions|CO2", "region": "World",
            "scenario": "Baseline", "start_year": 2050, "end_year": 2050,
        }

        for ordered_records in (records, list(reversed(records))):
            with self.subTest(first=ordered_records[0]["workspace_code"]):
                response = data_query(
                    "What are global CO2 emissions for Baseline in 2050?",
                    [], ordered_records, forced_entities=entities,
                )
                self.assertIn("Please choose a study", response)
                self.assertNotIn("100.00", response)
                self.assertNotIn("999.00", response)

    def test_region_availability_uses_only_selected_study_records(self):
        records = [
            {
                "workspace_code": "world-headed", "variable": "Emissions|CO2",
                "region": "World", "scenario": "World Baseline", "modelName": "GCAM",
                "years": {"2050": 100},
            },
            {
                "workspace_code": "afolu", "variable": "Land|Only in AFOLU",
                "region": "World", "scenario": "AFOLU Policy", "modelName": "Land Model",
                "years": {"2050": 200},
            },
        ]

        response = data_query(
            "What data do you have for World?", [], records,
            forced_entities={"workspace_code": "world-headed", "region": "World"},
            metadata=DataMetadata(records, []),
        )

        self.assertIn("Emissions|CO2", response)
        self.assertNotIn("Land|Only in AFOLU", response)
        self.assertIn("**Scenarios:** 1", response)

    def test_named_study_overrides_carried_workspace(self):
        ts = [
            {
                "workspace_code": "world-headed", "variable": "Emissions|CO2",
                "region": "World", "scenario": "Baseline", "modelName": "Model",
                "unit": "Mt CO2/yr", "years": {"2050": 100},
            },
            {
                "workspace_code": "afolu", "variable": "Emissions|CO2",
                "region": "World", "scenario": "Baseline", "modelName": "Model",
                "unit": "Mt CO2/yr", "years": {"2050": 999},
            },
        ]

        response = data_query(
            "Emissions|CO2 in AFOLU transformation for World in 2050",
            [],
            ts,
            forced_entities={
                "workspace_code": "world-headed", "variable": "Emissions|CO2",
                "region": "World", "start_year": 2050, "end_year": 2050,
            },
        )

        self.assertIn("999.00", response)
        self.assertNotIn("100.00", response)

    def test_model_description_uses_direct_model_page(self):
        response = data_query(
            "info ALADIN",
            [{"modelName": "ALADIN", "id": 6, "description": "A transport model."}],
            [],
        )

        self.assertIn("https://iamparis.eu/models/6", response)
        self.assertNotIn("](https://iamparis.eu/models)", response)

    def test_public_model_id_takes_precedence_over_internal_model_id(self):
        response = data_query(
            "info ALADIN",
            [{
                "modelName": "ALADIN", "id": 6, "modelId": 33369,
                "description": "A transport model.",
            }],
            [],
        )

        self.assertIn("https://iamparis.eu/models/6", response)
        self.assertNotIn("https://iamparis.eu/models/33369", response)

    def test_comparison_discloses_unavailable_requested_scenario(self):
        records = [{
            "workspace_code": "world-headed", "variable": "Emissions|CO2",
            "region": "World", "scenario": "Baseline", "modelName": "GCAM",
            "unit": "Mt CO2/yr", "years": {"2050": 100},
        }]

        response = data_query(
            "Compare Baseline with Policy for global CO2 emissions in 2050",
            [], records,
            forced_entities={
                "workspace_code": "world-headed", "variable": "Emissions|CO2",
                "region": "World", "scenarios": ["Baseline", "Policy"],
                "start_year": 2050, "end_year": 2050,
            },
            allow_plots=False,
        )

        self.assertIn("Some requested comparison members have no data", response)
        self.assertIn("`Policy`", response)
        self.assertIn("| 2050 | 100.00 |", response)

    def test_comparison_discloses_scenario_missing_requested_year(self):
        records = [
            {
                "workspace_code": "world-headed", "variable": "Emissions|CO2",
                "region": "World", "scenario": "Baseline", "modelName": "GCAM",
                "unit": "Mt CO2/yr", "years": {"2050": 100},
            },
            {
                "workspace_code": "world-headed", "variable": "Emissions|CO2",
                "region": "World", "scenario": "Policy", "modelName": "GCAM",
                "unit": "Mt CO2/yr", "years": {"2030": 80},
            },
        ]

        response = data_query(
            "Compare Baseline with Policy for global CO2 emissions in 2050",
            [], records,
            forced_entities={
                "workspace_code": "world-headed", "variable": "Emissions|CO2",
                "region": "World", "scenarios": ["Baseline", "Policy"],
                "start_year": 2050, "end_year": 2050,
            },
            allow_plots=False,
        )

        self.assertIn("Some requested comparison members have no data", response)
        self.assertIn("`Policy`", response)
        self.assertIn("| 2050 | 100.00 |", response)
        self.assertNotIn("| 2030 | 80.00 |", response)

    def test_scenario_family_alias_is_not_reported_missing(self):
        records = [
            {
                "workspace_code": "world-headed", "variable": "Emissions|CO2",
                "region": "World", "scenario": scenario, "modelName": "GCAM",
                "unit": "Mt CO2/yr", "years": {"2050": value},
            }
            for scenario, value in (("PR_Baseline", 100), ("Policy", 80))
        ]

        response = data_query(
            "Compare Baseline with Policy for global CO2 emissions in 2050",
            [], records,
            forced_entities={
                "workspace_code": "world-headed", "variable": "Emissions|CO2",
                "region": "World", "scenarios": ["Baseline", "Policy"],
                "start_year": 2050, "end_year": 2050,
            },
            allow_plots=False,
        )

        self.assertNotIn("requested comparison members have no data", response)
        self.assertIn("GCAM - PR_Baseline", response)

    def test_multi_variable_comparison_records_aggregate_scope(self):
        records = [
            {
                "workspace_code": "world-headed", "variable": variable,
                "region": "World", "scenario": "Baseline", "modelName": "GCAM",
                "unit": unit, "years": {"2050": value},
            }
            for variable, unit, value in (
                ("Emissions|CO2", "Mt CO2/yr", 100),
                ("Emissions|CH4", "Mt CH4/yr", 10),
            )
        ]
        consume_resolved_scope()

        data_query(
            "Compare CO2 and methane emissions for World in 2050", [], records,
            forced_entities={
                "workspace_code": "world-headed",
                "variables": ["Emissions|CO2", "Emissions|CH4"],
                "region": "World", "scenarios": ["Baseline"],
                "start_year": 2050, "end_year": 2050,
            },
            allow_plots=False,
        )
        scope = consume_resolved_scope()

        self.assertEqual(scope["variables"], ["Emissions|CO2", "Emissions|CH4"])
        self.assertEqual(scope["unit"], "multiple")
        self.assertNotIn("variable", scope)

    def test_workspace_summary_is_limited_and_includes_direct_study_links(self):
        ts = [
            {
                "workspace_code": "world-headed",
                "variable": f"Variable {number:02d}",
                "scenario": f"Scenario {number:02d}",
            }
            for number in range(1, 13)
        ]

        response = data_query("tell me about Where is the world headed?", [], ts)

        self.assertIn("### Where is the world headed?", response)
        self.assertIn("`Variable 10`", response)
        self.assertNotIn("`Variable 11`", response)
        self.assertIn("`Scenario 10`", response)
        self.assertNotIn("`Scenario 11`", response)
        self.assertIn(
            "https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs",
            response,
        )
        self.assertIn(
            "https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/policy_questions",
            response,
        )

    def test_workspace_scoped_global_emissions_returns_only_2050_table_values(self):
        ts = [
            {
                "workspace_code": "world-headed", "variable": "Emissions|CO2",
                "region": "World", "scenario": "Baseline", "modelName": "Model A",
                "unit": "Mt CO2/yr", "years": {"2050": 100},
            },
            {
                "workspace_code": "world-headed", "variable": "Emissions|CO2",
                "region": "World", "scenario": "Policy", "modelName": "Model B",
                "unit": "Mt CO2/yr", "years": {"2050": 80},
            },
            {
                "workspace_code": "afolu", "variable": "Emissions|CO2",
                "region": "World", "scenario": "Other", "modelName": "Wrong study",
                "unit": "Mt CO2/yr", "years": {"2050": 999},
            },
        ]

        response = data_query(
            "What are the global emissions estimated in this study for 2050?",
            [],
            ts,
            forced_entities={
                "variable": "Emissions|CO2", "region": "World",
                "start_year": 2050, "end_year": 2050,
                "workspace_code": "world-headed",
            },
        )

        self.assertIn("### Emissions|CO2 in World", response)
        self.assertIn("years `2050`", response)
        self.assertIn("| 2050 | 100.00 | Mt CO2/yr |", response)
        self.assertIn("| 2050 | 80.00 | Mt CO2/yr |", response)
        self.assertNotIn("999", response)
        self.assertNotIn("plot Emissions", response)

    def test_model_availability_alternatives_omit_numeric_source_ids(self):
        ts = [
            {
                "variable": "Price|Carbon", "region": "EU",
                "scenario": "Policy", "modelName": "42",
                "unit": "US$2010/t CO2", "years": {"2030": 1},
            },
            {
                "variable": "Price|Carbon", "region": "EU",
                "scenario": "Policy", "modelName": "GCAM 7.0",
                "unit": "US$2010/t CO2", "years": {"2030": 2},
            },
        ]
        models = [
            {"modelName": "42"},
            {"modelName": "GCAM 7.0"},
            {"modelName": "WITCH 6.0"},
        ]

        response = data_query(
            "does WITCH report carbon price for EU?",
            models,
            ts,
            forced_entities={
                "variable": "Price|Carbon", "region": "EU", "model": "WITCH",
            },
            metadata=DataMetadata(ts, models),
        )

        self.assertTrue(response.startswith("No."))
        self.assertIn("GCAM 7.0", response)
        self.assertNotIn("42", response)

    def test_availability_matrix_tracks_model_name_and_years(self):
        metadata = DataMetadata(
            [
                {
                    "variable": "Emissions|CO2",
                    "region": "World",
                    "scenario": "Baseline",
                    "modelName": "GCAM",
                    "unit": "Mt CO2/yr",
                    "years": {"2030": 10, "2050": 5},
                }
            ],
            models=[{"modelName": "GCAM"}],
        )

        self.assertTrue(metadata.combination_exists("Emissions|CO2", "World", "Baseline", "GCAM"))
        self.assertFalse(metadata.combination_exists("Emissions|CO2", "EU", "Baseline", "GCAM"))
        self.assertEqual(metadata.get_available_years("Emissions|CO2", "World", "Baseline", "GCAM"), ["2030", "2050"])
        self.assertIn("GCAM", metadata.all_model_names)

    def test_suggest_valid_options_uses_existing_combinations(self):
        metadata = DataMetadata(
            [
                {
                    "variable": "Emissions|CO2",
                    "region": "World",
                    "scenario": "Baseline",
                    "modelName": "GCAM",
                    "years": {"2030": 1},
                },
                {
                    "variable": "Emissions|CO2",
                    "region": "EU",
                    "scenario": "Policy",
                    "modelName": "GCAM",
                    "years": {"2030": 1},
                },
                {
                    "variable": "GDP|MER",
                    "region": "World",
                    "scenario": "Policy",
                    "modelName": "GCAM",
                    "years": {"2030": 1},
                },
            ],
            models=[{"modelName": "GCAM"}],
        )

        options = metadata.suggest_valid_options(
            variable="Emissions|CO2",
            region="World",
            scenario="Policy",
            model="GCAM",
        )

        self.assertEqual(options["variables"], [])
        self.assertEqual(options["regions"], ["EU"])
        self.assertEqual(options["scenarios"], ["Baseline"])

        variable_options = metadata.suggest_valid_options(
            region="World",
            scenario="Policy",
            model="GCAM",
        )
        self.assertEqual(variable_options["variables"], ["GDP|MER"])

    def test_region_availability_is_intersection_not_global_catalogue(self):
        metadata = DataMetadata(
            [
                {
                    "variable": "Emissions|CO2", "region": "EU",
                    "scenario": "Policy", "modelName": "GCAM",
                    "years": {"2030": 1},
                },
                {
                    "variable": "Population", "region": "World",
                    "scenario": "Baseline", "modelName": "REMIND 3.4",
                    "years": {"2050": 2},
                },
            ]
        )

        info = metadata.get_available_for_region("EU")

        self.assertEqual(info["variables"], ["Emissions|CO2"])
        self.assertEqual(info["scenarios"], ["Policy"])
        self.assertEqual(info["models"], ["GCAM"])
        self.assertEqual(info["years"], ["2030"])

    def test_model_availability_expands_curated_family_aliases(self):
        metadata = DataMetadata(
            [
                {
                    "variable": "Price|Carbon", "region": "EU",
                    "scenario": "Policy", "modelName": "GEMINI-E3 7.0",
                    "years": {"2030": 1, "2050": 2},
                },
                {
                    "variable": "GDP|MER", "region": "World",
                    "scenario": "Baseline", "modelName": "gemini_e3",
                    "years": {"2020": 3},
                },
            ]
        )

        info = metadata.get_available_for_model("GEMINI-E3")

        self.assertEqual(info["models"], ["GEMINI-E3 7.0", "gemini_e3"])
        self.assertEqual(info["variables"], ["GDP|MER", "Price|Carbon"])
        self.assertEqual(info["regions"], ["EU", "World"])
        self.assertEqual(info["years"], ["2020", "2030", "2050"])

        distinct_model = metadata.get_available_for_model("GEM-E3")
        self.assertIsNone(distinct_model["model"])
        self.assertEqual(distinct_model["models"], [])
        self.assertEqual(distinct_model["variables"], [])

    def test_timeseries_table_hides_numeric_model_id_without_dropping_rows(self):
        response = format_time_series_data(
            [
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
            ],
            "Emissions|CO2",
            "World",
        )
        scope = consume_resolved_scope()

        self.assertIn(UNLABELLED_MODEL_LABEL, response)
        self.assertNotRegex(response, r"\b42\b")
        self.assertEqual(response.count("| 2030 |"), 2)
        self.assertEqual(
            scope["result_models"],
            ["GCAM", UNLABELLED_MODEL_LABEL],
        )

    def test_suggest_scenarios_by_scope_prioritizes_current_variable_and_region(self):
        metadata = DataMetadata(
            [
                {
                    "variable": "Emissions|CO2",
                    "region": "World",
                    "scenario": "Baseline",
                    "modelName": "GCAM",
                    "years": {"2030": 1},
                },
                {
                    "variable": "Emissions|CO2",
                    "region": "World",
                    "scenario": "RegionalPolicy",
                    "modelName": "GCAM",
                    "years": {"2030": 1},
                },
                {
                    "variable": "Emissions|CO2",
                    "region": "EU",
                    "scenario": "Policy",
                    "modelName": "GCAM",
                    "years": {"2030": 1},
                },
                {
                    "variable": "Emissions|CH4",
                    "region": "World",
                    "scenario": "MethanePolicy",
                    "modelName": "GCAM",
                    "years": {"2030": 1},
                },
            ],
            models=[{"modelName": "GCAM"}],
        )

        scenarios = metadata.suggest_scenarios_by_scope(
            variable="Emissions|CO2",
            region="World",
            model="GCAM",
            exclude="MissingPolicy",
            limit=4,
        )

        self.assertEqual(
            scenarios,
            ["Baseline", "RegionalPolicy", "Policy", "MethanePolicy"],
        )

    def test_metadata_cache_invalidates_when_data_changes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_file = str(Path(temp_dir) / "metadata.pkl")
            first = build_metadata_with_cache(
                [
                    {
                        "variable": "Population",
                        "region": "World",
                        "scenario": "Baseline",
                        "modelName": "GCAM",
                        "years": {"2030": 1},
                    }
                ],
                [{"modelName": "GCAM"}],
                cache_file=cache_file,
            )
            second = build_metadata_with_cache(
                [
                    {
                        "variable": "Population",
                        "region": "World",
                        "scenario": "Baseline",
                        "modelName": "GCAM",
                        "years": {"2030": 1},
                    },
                    {
                        "variable": "GDP|MER",
                        "region": "World",
                        "scenario": "Baseline",
                        "modelName": "GCAM",
                        "years": {"2030": 2},
                    },
                ],
                [{"modelName": "GCAM"}],
                cache_file=cache_file,
            )

            self.assertEqual(len(first.all_variables), 1)
            self.assertEqual(len(second.all_variables), 2)

    def test_data_query_uses_metadata_for_invalid_combination_recovery(self):
        ts = [
            {
                "variable": "Emissions|CO2",
                "region": "World",
                "scenario": "Baseline",
                "modelName": "GCAM",
                "unit": "Mt CO2/yr",
                "years": {"2030": 1},
            },
            {
                "variable": "Emissions|CO2",
                "region": "EU",
                "scenario": "Policy",
                "modelName": "GCAM",
                "unit": "Mt CO2/yr",
                "years": {"2030": 1},
            },
            {
                "variable": "GDP|MER",
                "region": "World",
                "scenario": "Policy",
                "modelName": "GCAM",
                "unit": "billion US$",
                "years": {"2030": 1},
            },
        ]
        models = [{"modelName": "GCAM"}]
        metadata = DataMetadata(ts, models)

        response = data_query(
            "Show Emissions|CO2 for World under Policy",
            models,
            ts,
            forced_entities={
                "variable": "Emissions|CO2",
                "region": "World",
                "scenario": "Policy",
            },
            metadata=metadata,
        )

        self.assertIn("I could not find data for `Emissions|CO2` in `World` under `Policy`.", response)
        self.assertIn("Closest valid options:", response)
        self.assertIn("region `EU`", response)
        self.assertIn("scenario `Baseline`", response)
        self.assertNotIn("variable `GDP|MER`", response)

    def test_data_query_orders_scenario_recovery_by_scope(self):
        ts = [
            {
                "variable": "Emissions|CO2",
                "region": "World",
                "scenario": "Baseline",
                "modelName": "GCAM",
                "unit": "Mt CO2/yr",
                "years": {"2030": 1},
            },
            {
                "variable": "Emissions|CO2",
                "region": "World",
                "scenario": "RegionalPolicy",
                "modelName": "GCAM",
                "unit": "Mt CO2/yr",
                "years": {"2030": 1},
            },
            {
                "variable": "Emissions|CO2",
                "region": "EU",
                "scenario": "Policy",
                "modelName": "GCAM",
                "unit": "Mt CO2/yr",
                "years": {"2030": 1},
            },
            {
                "variable": "Emissions|CH4",
                "region": "World",
                "scenario": "MethanePolicy",
                "modelName": "GCAM",
                "unit": "Mt CH4/yr",
                "years": {"2030": 1},
            },
        ]
        models = [{"modelName": "GCAM"}]
        metadata = DataMetadata(ts, models)

        response = data_query(
            "Show Emissions|CO2 for World under MissingPolicy",
            models,
            ts,
            forced_entities={
                "variable": "Emissions|CO2",
                "region": "World",
                "scenario": "MissingPolicy",
            },
            metadata=metadata,
        )

        baseline_index = response.index("`Baseline`")
        regional_index = response.index("`RegionalPolicy`")
        policy_index = response.index("`Policy`")
        self.assertLess(baseline_index, regional_index)
        self.assertLess(regional_index, policy_index)

    def test_plot_query_uses_metadata_for_invalid_combination_recovery(self):
        ts = [
            {
                "variable": "Emissions|CO2",
                "region": "World",
                "scenario": "Baseline",
                "modelName": "GCAM",
                "unit": "Mt CO2/yr",
                "years": {"2030": 1},
            },
            {
                "variable": "Emissions|CO2",
                "region": "EU",
                "scenario": "Policy",
                "modelName": "GCAM",
                "unit": "Mt CO2/yr",
                "years": {"2030": 1},
            },
            {
                "variable": "GDP|MER",
                "region": "World",
                "scenario": "Policy",
                "modelName": "GCAM",
                "unit": "billion US$",
                "years": {"2030": 1},
            },
        ]
        models = [{"modelName": "GCAM"}]

        response = simple_plot_query(
            "Plot Emissions|CO2 for World under Policy",
            models,
            ts,
        )

        self.assertIn("No data found for **Emissions|CO2** in region `World` under scenario `Policy`.", response)
        self.assertIn("Closest regions: `EU`", response)
        self.assertIn("Closest scenarios: `Baseline`", response)
        self.assertNotIn("Closest variables: `GDP|MER`", response)


class CatalogueMetadataQueryTests(unittest.TestCase):
    def setUp(self):
        self.ts = [
            {
                "variable": "Emissions|CO2", "region": "GREECE",
                "scenario": "PR_CurPol_CP", "modelName": "GCAM",
                "unit": "Mt CO2/yr", "years": {"2030": 1, "2050": 2},
            },
            {
                "variable": "Population", "region": "AFR",
                "scenario": "Baseline", "modelName": "POLES 3.0",
                "unit": "million", "years": {"2020": 3, "2100": 4},
            },
            {
                "variable": "Price|Carbon", "region": "EU",
                "scenario": "Policy", "modelName": "WITCH 6.0",
                "unit": "US$2010/t CO2", "years": {"2030": 5, "2050": 6},
            },
            {
                "variable": "GDP|MER", "region": "EU",
                "scenario": "Policy", "modelName": "GEMINI-E3 7.0",
                "unit": "billion US$", "years": {"2015": 7, "2050": 8},
            },
            {
                "variable": "Price|Carbon", "region": "World",
                "scenario": "Baseline", "modelName": "gemini_e3",
                "unit": "US$2010/t CO2", "years": {"2020": 9},
            },
        ]
        self.models = [
            {"modelName": name}
            for name in ["GCAM", "POLES 3.0", "WITCH 6.0", "GEMINI-E3 7.0"]
        ]
        self.metadata = DataMetadata(self.ts, self.models)

    def test_region_data_overview_is_scoped(self):
        response = data_query(
            "what data do you have for Greece?",
            self.models,
            self.ts,
            forced_entities={"region": "GREECE"},
            metadata=self.metadata,
        )

        self.assertIn("Data available for GREECE (Greece)", response)
        self.assertIn("**Variables:** 1", response)
        self.assertIn("`Emissions|CO2`", response)
        self.assertNotIn("`Population`", response)

    def test_region_variable_listing_is_scoped(self):
        response = data_query(
            "which variables exist for Africa?",
            self.models,
            self.ts,
            forced_entities={"region": "AFR"},
            metadata=self.metadata,
        )

        self.assertIn("for region `AFR (Africa)`", response)
        self.assertIn("Population", response)
        self.assertNotIn("GDP|MER", response)

    def test_catalogue_count_and_latest_year_are_grounded(self):
        scenario_response = data_query(
            "how many scenarios are in the database?",
            self.models,
            self.ts,
            metadata=self.metadata,
        )
        latest_response = data_query(
            "what is the latest year in the projections?",
            self.models,
            self.ts,
            metadata=self.metadata,
        )

        self.assertIn("3 distinct scenarios", scenario_response)
        self.assertIn("**2100**", latest_response)
        self.assertNotIn("Prometheus", latest_response)

    def test_model_category_queries_keep_gem_and_gemini_distinct(self):
        years_response = data_query(
            "what years are available for GEM-E3 data?",
            self.models,
            self.ts,
            forced_entities={"model": "GEM-E3"},
            metadata=self.metadata,
        )
        consume_resolved_scope()
        gemini_years_response = data_query(
            "what years are available for GEMINI-E3 data?",
            self.models,
            self.ts,
            forced_entities={"model": "GEMINI-E3"},
            metadata=self.metadata,
        )
        years_scope = consume_resolved_scope()
        variables_response = data_query(
            "list the variables reported by POLES",
            self.models,
            self.ts,
            forced_entities={"model": "POLES"},
            metadata=self.metadata,
        )

        self.assertIn("could not find any years recorded for model `GEM-E3`", years_response)
        self.assertNotIn("GEMINI-E3", years_response)
        self.assertIn("2015–2050", gemini_years_response)
        self.assertEqual(
            years_scope["result_models"],
            ["GEMINI-E3 7.0", "gemini_e3"],
        )
        self.assertIn("Population", variables_response)
        self.assertNotIn("Emissions|CO2", variables_response)

    def test_model_scoped_listing_without_runtime_rows_does_not_fall_back_global(self):
        response = data_query(
            "list the variables reported by EMPTY-IAM",
            [{"modelName": "EMPTY-IAM"}],
            self.ts,
            metadata=self.metadata,
        )

        self.assertIn("could not find any variables recorded for model `EMPTY-IAM`", response)
        self.assertNotIn("I can work with these variables", response)

    def test_yes_no_availability_checks_exact_model_variable_region_slice(self):
        yes_response = data_query(
            "does WITCH report carbon price for EU?",
            self.models,
            self.ts,
            forced_entities={
                "variable": "Price|Carbon", "region": "EU", "model": "WITCH",
            },
            metadata=self.metadata,
        )
        no_response = data_query(
            "does WITCH report population for EU?",
            self.models,
            self.ts,
            forced_entities={
                "variable": "Population", "region": "EU", "model": "WITCH",
            },
            metadata=self.metadata,
        )

        self.assertTrue(yes_response.startswith("Yes."))
        self.assertIn("Price|Carbon", yes_response)
        self.assertTrue(no_response.startswith("No."))

    def test_yes_no_availability_does_not_treat_unknown_runtime_model_as_all_models(self):
        response = data_query(
            "does REMIND report carbon price for EU?",
            self.models,
            self.ts,
            forced_entities={
                "variable": "Price|Carbon", "region": "EU", "model": "REMIND",
            },
            metadata=self.metadata,
        )

        self.assertTrue(response.startswith("No."))
        self.assertIn("model `REMIND`", response)
        self.assertNotIn("Yes.", response)
        self.assertIn("WITCH 6.0", response)

    def test_bare_emissions_answer_discloses_co2_interpretation(self):
        response = data_query(
            "emissions under current policies for Greece",
            self.models,
            self.ts,
            forced_entities={
                "variable": "Emissions|CO2",
                "region": "GREECE",
                "scenario": "Current Policies",
            },
            metadata=self.metadata,
        )

        self.assertIn("treated bare **emissions** as `Emissions|CO2`", response)


if __name__ == "__main__":
    unittest.main()
