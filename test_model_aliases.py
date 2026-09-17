import unittest

from model_aliases import (
    UNLABELLED_MODEL_LABEL,
    display_model_label,
    extract_model_hint,
    is_presentable_model_label,
    match_model_name,
    model_family_key,
    resolve_model_candidates,
    resolve_model_family_members,
)


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

    def test_numeric_source_ids_are_not_resolved_as_model_labels(self):
        self.assertFalse(is_presentable_model_label("42"))
        self.assertFalse(is_presentable_model_label(42))
        self.assertTrue(is_presentable_model_label("GCAM 7.0"))
        self.assertTrue(is_presentable_model_label("GEM-E3"))
        self.assertEqual(resolve_model_candidates("42", ["42", "GCAM"]), [])
        self.assertEqual(match_model_name("model 42", ["42", "GCAM"]), "")
        self.assertEqual(display_model_label("42"), UNLABELLED_MODEL_LABEL)

    def test_version_suffix_shares_family_without_stripping_name_digits(self):
        self.assertEqual(model_family_key("Example-E3 7.0"), model_family_key("Example-E3"))
        self.assertNotEqual(model_family_key("Example-E3"), model_family_key("Example"))

    def test_family_members_expand_base_names_but_keep_qualifiers_distinct(self):
        models = [
            "GCAM", "GCAM 7.0", "GCAM-PR 7.0",
            "GEM-E3 2.0", "GEMINI-E3 7.0", "gemini_e3", "POLES 3.0",
        ]

        self.assertEqual(
            resolve_model_family_members("GEM-E3", models),
            ["GEM-E3 2.0"],
        )
        self.assertEqual(
            resolve_model_family_members("GEMINI-E3", models),
            ["GEMINI-E3 7.0", "gemini_e3"],
        )
        self.assertEqual(
            resolve_model_family_members("POLES", models),
            ["POLES 3.0"],
        )
        self.assertEqual(
            resolve_model_family_members("GCAM", models),
            ["GCAM", "GCAM 7.0"],
        )
        self.assertEqual(
            resolve_model_family_members("GCAM-PR 7.0", models),
            ["GCAM-PR 7.0"],
        )

    def test_exact_gem_e3_never_borrows_gemini_runtime_members(self):
        gemini_only = ["GEMINI-E3 7.0", "gemini_e3"]

        self.assertEqual(resolve_model_candidates("GEM-E3", gemini_only), [])
        self.assertEqual(resolve_model_family_members("GEM-E3", gemini_only), [])
        self.assertEqual(match_model_name("data from GEM-E3", gemini_only), "")


if __name__ == "__main__":
    unittest.main()
