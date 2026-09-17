import unittest
from unittest.mock import patch

from data_utils import format_time_series_data
from resolved_scope import consume_resolved_scope
from year_filters import (
    LATEST_YEAR_SENTINEL,
    extract_year_filter,
    extract_year_range,
    select_years,
)


class YearFilterTests(unittest.TestCase):
    def test_extract_year_range_handles_required_phrases(self):
        self.assertEqual(extract_year_range("show values in 2030"), (2030, 2030))
        self.assertEqual(extract_year_range("show values from 2030 to 2050"), (2030, 2050))
        self.assertEqual(extract_year_range("show values by 2050"), (None, 2050))
        self.assertEqual(extract_year_range("show values after 2030"), (2031, None))
        self.assertEqual(extract_year_range("show the latest available year"), (LATEST_YEAR_SENTINEL, LATEST_YEAR_SENTINEL))

    def test_typed_filter_distinguishes_until_from_no_filter(self):
        no_filter = extract_year_filter("show emissions")
        until = extract_year_filter("until 2050")

        self.assertFalse(no_filter.explicit)
        self.assertTrue(until.explicit)
        self.assertIsNone(until.start_year)
        self.assertEqual(until.end_year, 2050)
        self.assertEqual(
            until.apply({"start_year": 2030, "end_year": 2040}),
            {"end_year": 2050},
        )

    def test_extract_year_range_spans_and_end_year(self):
        # "between X and Y" and "X and Y" are ranges, not just the first year.
        self.assertEqual(extract_year_range("final energy between 2030 and 2060"), (2030, 2060))
        self.assertEqual(extract_year_range("carbon price in 2030 and 2050"), (2030, 2050))
        # 2100 (starts with "21") must be recognised, not silently dropped.
        self.assertEqual(extract_year_range("population in 2100"), (2100, 2100))
        self.assertEqual(extract_year_range("primary energy from 2020 to 2100"), (2020, 2100))
        self.assertEqual(extract_year_range("emissions from 2025 through 2050"), (2025, 2050))
        self.assertEqual(extract_year_range("emissions around 2040"), (2040, 2040))

    def test_common_spoken_projection_years_are_unambiguous(self):
        self.assertEqual(extract_year_range("GDP for US in twenty fifty"), (2050, 2050))
        self.assertEqual(extract_year_range("from twenty thirty to twenty sixty"), (2030, 2060))
        self.assertEqual(extract_year_range("population in twenty one hundred"), (2100, 2100))
        self.assertEqual(extract_year_range("historical data in nineteen ninety five"), (1995, 1995))

    def test_select_years_filters_ranges_and_latest(self):
        years = ["2020", "2030", "2040", "2050"]
        self.assertEqual(select_years(years, 2030, 2040), ["2030", "2040"])
        self.assertEqual(select_years(years, None, 2030), ["2020", "2030"])
        self.assertEqual(select_years(years, 2031, None), ["2040", "2050"])
        self.assertEqual(select_years(years, LATEST_YEAR_SENTINEL, LATEST_YEAR_SENTINEL), ["2050"])

    def test_format_time_series_data_uses_latest_year_filter(self):
        response = format_time_series_data(
            [
                {
                    "variable": "Emissions|CO2",
                    "region": "World",
                    "scenario": "Baseline",
                    "modelName": "GCAM",
                    "unit": "Mt CO2/yr",
                    "years": {"2030": 1, "2040": 2, "2050": 3},
                }
            ],
            "Emissions|CO2",
            "World",
            LATEST_YEAR_SENTINEL,
            LATEST_YEAR_SENTINEL,
        )

        self.assertNotIn("| 2030 |", response)
        self.assertNotIn("| 2040 |", response)
        self.assertIn("| 2050 |", response)
        self.assertIn("Scope: scenario `Baseline`, model `GCAM`, years `latest available`", response)
        self.assertIn("Unit: `Mt CO2/yr`", response)
        self.assertIn("Answer:", response)
        self.assertIn("[IAM PARIS data explorer]", response)

    def test_format_time_series_data_uses_standard_answer_sections(self):
        response = format_time_series_data(
            [
                {
                    "variable": "Emissions|CO2",
                    "region": "World",
                    "scenario": "Baseline",
                    "modelName": "GCAM",
                    "unit": "Mt CO2/yr",
                    "years": {"2030": 1, "2050": 3},
                }
            ],
            "Emissions|CO2",
            "World",
            2030,
            2050,
        )

        self.assertTrue(response.startswith("### Emissions|CO2 in World"))
        self.assertIn("Scope: scenario `Baseline`, model `GCAM`, years `2030-2050`", response)
        self.assertIn("Unit: `Mt CO2/yr`", response)
        self.assertIn("Answer:", response)
        self.assertIn("| 2030 |", response)
        self.assertIn("| 2050 |", response)
        self.assertIn("[IAM PARIS data explorer]", response)

    def test_format_omits_groups_without_values_in_requested_period(self):
        response = format_time_series_data(
            [
                {
                    "scenario": "Path", "modelName": "VisibleModel", "unit": "EJ/yr",
                    "years": {"2030": 1.5},
                },
                {
                    "scenario": "Path", "modelName": "EmptyModel", "unit": "EJ/yr",
                    "years": {"2040": 2.5},
                },
            ],
            "Synthetic Metric",
            "Synthetic Region",
            2030,
            2030,
        )

        self.assertIn("**VisibleModel - Path**", response)
        self.assertNotIn("**EmptyModel - Path**", response)
        self.assertNotIn("No year data available", response)
        scope = consume_resolved_scope()
        self.assertEqual(scope["result_models"], ["VisibleModel"])

    def test_format_normalizes_units_and_excludes_clear_dimension_outlier(self):
        response = format_time_series_data(
            [
                {
                    "scenario": "Path A", "modelName": "Energy One", "unit": "EJ/y",
                    "years": {"2030": 1},
                },
                {
                    "scenario": "Path B", "modelName": "Energy Two", "unit": "EJ/yr",
                    "years": {"2030": 2},
                },
                {
                    "scenario": "Path C", "modelName": "Mass Outlier", "unit": "Mt CO2/yr",
                    "years": {"2030": 3},
                },
            ],
            "Synthetic Metric",
            "Synthetic Region",
            2030,
            2030,
        )

        self.assertIn("Unit: `EJ/yr`", response)
        self.assertNotIn("EJ/y |", response)
        self.assertNotIn("**Mass Outlier - Path C**", response)
        self.assertIn("excluded incompatible unit group", response)
        self.assertIn("`Mass Outlier - Path C` (Mt CO2/yr)", response)
        scope = consume_resolved_scope()
        self.assertEqual(scope["result_models"], ["Energy One", "Energy Two"])

    def test_format_returns_no_data_instead_of_empty_table(self):
        response = format_time_series_data(
            [{
                "scenario": "Path", "modelName": "AnyModel", "unit": "EJ/yr",
                "years": {"2030": 1},
            }],
            "Synthetic Metric",
            "Synthetic Region",
            2050,
            2050,
        )

        self.assertIn("No data found", response)
        self.assertIn("requested period", response)
        self.assertNotIn("| Year |", response)

    @patch("data_utils.DATA_COMPACT_GROUP_THRESHOLD", 2)
    @patch("data_utils.DATA_GROUP_DISPLAY_LIMIT", 3)
    def test_broad_results_use_deterministic_capped_matrix_without_averaging(self):
        response = format_time_series_data(
            [
                {
                    "scenario": "Path Z", "modelName": "Zulu", "unit": "EJ/yr",
                    "years": {"2030": 99, "2050": 999},
                },
                {
                    "scenario": "Path B", "modelName": "Alpha", "unit": "EJ/yr",
                    "years": {"2030": 2, "2050": 20},
                },
                {
                    "scenario": "Path A", "modelName": "Beta", "unit": "EJ/yr",
                    "years": {"2030": 3, "2050": 30},
                },
                {
                    "scenario": "Path A", "modelName": "Alpha", "unit": "EJ/yr",
                    "years": {"2030": 1, "2050": 10},
                },
            ],
            "Synthetic Metric",
            "Synthetic Region",
            2030,
            2050,
        )

        self.assertIn("Summary: 4 series across 3 models and 3 scenarios.", response)
        self.assertIn("Showing series 1-3 of 4", response)
        self.assertIn("Narrow by model or scenario", response)
        self.assertIn("| Model | Scenario | 2030 | 2050 | Unit |", response)
        self.assertEqual(response.count("| Model | Scenario | 2030 | 2050 | Unit |"), 1)
        self.assertNotIn("| Year | Value | Unit |", response)

        alpha_a = "| Alpha | Path A | 1.00 | 10.00 | EJ/yr |"
        alpha_b = "| Alpha | Path B | 2.00 | 20.00 | EJ/yr |"
        beta_a = "| Beta | Path A | 3.00 | 30.00 | EJ/yr |"
        self.assertIn(alpha_a, response)
        self.assertIn(alpha_b, response)
        self.assertIn(beta_a, response)
        self.assertLess(response.index(alpha_a), response.index(alpha_b))
        self.assertLess(response.index(alpha_b), response.index(beta_a))
        self.assertNotIn("| Zulu | Path Z |", response)
        self.assertNotIn("11.00", response)

        scope = consume_resolved_scope()
        self.assertEqual(scope["result_models"], ["Alpha", "Beta", "Zulu"])
        self.assertEqual(scope["scenarios"], ["Path A", "Path B", "Path Z"])

    @patch("data_utils.DATA_COMPACT_GROUP_THRESHOLD", 4)
    @patch("data_utils.DATA_GROUP_DISPLAY_LIMIT", 1)
    def test_configured_compact_threshold_preserves_small_result_layout(self):
        response = format_time_series_data(
            [
                {
                    "scenario": "Path A", "modelName": "Alpha", "unit": "EJ/yr",
                    "years": {"2030": 1},
                },
                {
                    "scenario": "Path B", "modelName": "Beta", "unit": "EJ/yr",
                    "years": {"2030": 2},
                },
            ],
            "Synthetic Metric",
            "Synthetic Region",
            2030,
            2030,
        )

        self.assertNotIn("Summary:", response)
        self.assertEqual(response.count("| Year | Value | Unit |"), 2)
        self.assertIn("**Alpha - Path A**", response)
        self.assertIn("**Beta - Path B**", response)


if __name__ == "__main__":
    unittest.main()
