import tempfile
import unittest
from pathlib import Path

from data_metadata import DataMetadata, build_metadata_with_cache
from data_utils import data_query
from simple_plotter import simple_plot_query


class DataMetadataTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
