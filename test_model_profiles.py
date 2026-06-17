import unittest

from model_profiles import find_model_profile, format_model_profile_answer, has_strong_model_metadata


class ModelProfileTests(unittest.TestCase):
    def test_find_model_profile_handles_common_aliases(self):
        self.assertEqual(find_model_profile("tell me about MESSAGEix")["name"], "MESSAGEix-GLOBIOM")
        self.assertEqual(find_model_profile("What is gcampr 7?")["name"], "GCAM-PR")
        self.assertEqual(find_model_profile("WITCH assumptions")["name"], "WITCH")

    def test_gcampr_does_not_resolve_to_base_gcam(self):
        profile = find_model_profile("tell me about GCAM-PR 7.0")
        self.assertEqual(profile["name"], "GCAM-PR")

    def test_gcam_pr_spaced_alias_does_not_resolve_to_base_gcam(self):
        profile = find_model_profile("tell me about GCAM PR")
        self.assertEqual(profile["name"], "GCAM-PR")

    def test_format_model_profile_answer_includes_assumption_context_and_link(self):
        profile = find_model_profile("REMIND CCS assumptions")
        response = format_model_profile_answer(profile, asks_assumptions=True)

        self.assertIn("### REMIND", response)
        self.assertIn("Assumptions:", response)
        self.assertIn("scenario-dependent", response)
        self.assertIn("[IAM PARIS Models](https://iamparis.eu/models)", response)

    def test_metadata_strength_detects_weak_records(self):
        self.assertFalse(has_strong_model_metadata({"description": ""}))
        self.assertFalse(has_strong_model_metadata({"description": "Short."}))
        self.assertTrue(has_strong_model_metadata({"description": "A" * 140}))


if __name__ == "__main__":
    unittest.main()
