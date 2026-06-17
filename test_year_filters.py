import unittest

from data_utils import format_time_series_data
from year_filters import LATEST_YEAR_SENTINEL, extract_year_range, select_years


class YearFilterTests(unittest.TestCase):
    def test_extract_year_range_handles_required_phrases(self):
        self.assertEqual(extract_year_range("show values in 2030"), (2030, 2030))
        self.assertEqual(extract_year_range("show values from 2030 to 2050"), (2030, 2050))
        self.assertEqual(extract_year_range("show values by 2050"), (None, 2050))
        self.assertEqual(extract_year_range("show values after 2030"), (2031, None))
        self.assertEqual(extract_year_range("show the latest available year"), (LATEST_YEAR_SENTINEL, LATEST_YEAR_SENTINEL))

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
        self.assertIn("Next:", response)

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
        self.assertIn("Next:", response)


if __name__ == "__main__":
    unittest.main()
