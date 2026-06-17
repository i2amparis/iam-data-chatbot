import unittest

from model_aliases import extract_model_hint, match_model_name, resolve_model_candidates


class ModelAliasTests(unittest.TestCase):
    def setUp(self):
        self.models = [
            "GCAM",
            "GCAM-PR 7.0",
            "PROMETHEUS",
            "LEAP",
            "REMIND-MAgPIE 3.0",
            "MESSAGEix-GLOBIOM 2.0",
            "WITCH 6.0",
        ]

    def test_extract_model_hint_stops_at_next_dimension(self):
        self.assertEqual(
            extract_model_hint("CO2 emissions for gcampr under Baseline"),
            "gcampr",
        )

    def test_curated_aliases_resolve_expected_models(self):
        cases = {
            "show data for gcam": "GCAM",
            "show data for gcampr": "GCAM-PR 7.0",
            "show data using gcam-pr": "GCAM-PR 7.0",
            "show data with prometheus": "PROMETHEUS",
            "show data for leap": "LEAP",
            "show data model remind": "REMIND-MAgPIE 3.0",
            "show data with message": "MESSAGEix-GLOBIOM 2.0",
            "show data using message ix": "MESSAGEix-GLOBIOM 2.0",
            "show data for witch": "WITCH 6.0",
        }
        for query, expected in cases.items():
            with self.subTest(query=query):
                self.assertEqual(match_model_name(query, self.models), expected)

    def test_resolve_model_candidates_returns_alias_matches(self):
        self.assertEqual(resolve_model_candidates("messageix", self.models)[0], "MESSAGEix-GLOBIOM 2.0")
        self.assertEqual(resolve_model_candidates("gcam pr", self.models)[0], "GCAM-PR 7.0")


if __name__ == "__main__":
    unittest.main()
