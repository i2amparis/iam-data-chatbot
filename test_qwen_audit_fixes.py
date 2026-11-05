"""Regressions from the 200-question local Qwen audit."""

import unittest
from unittest.mock import MagicMock

from explanations import matching_explanations
from test_response_fixes import _record, build_manager


class QwenAuditFixTests(unittest.TestCase):
    def test_explanatory_questions_reach_general_qa(self):
        for query in (
            "What does net zero mean for electricity prices?",
            "What uncertainties matter for projected electricity demand?",
        ):
            with self.subTest(query=query):
                manager = build_manager()
                manager.agents["general_qa"].handle = MagicMock(return_value="Explanation")
                self.assertIn("Explanation", manager.route_query(query))
                self.assertEqual(manager.last_route_decision["reason"], "conceptual climate explanation")

    def test_negative_emissions_value_is_a_definition(self):
        manager = build_manager()
        answer = manager.route_query("What does a negative emissions value mean?")
        self.assertIn("net removal", answer)
        self.assertIn("sign convention", answer)
        self.assertEqual(manager.last_route_decision["reason"], "negative-emissions value explanation")

    def test_baseline_decline_context_does_not_include_rising_explanation(self):
        docs = matching_explanations("Why do baseline scenarios sometimes show falling emissions?")
        self.assertTrue(docs)
        self.assertIn("baseline can show falling", docs[0].page_content.casefold())
        self.assertFalse(any("baseline scenarios show rising" in doc.page_content.casefold() for doc in docs))

    def test_region_model_coverage_does_not_search_variable_names(self):
        ts = [
            _record("world-headed", "GREECE", "Baseline", "GCAM", 10),
            _record("world-headed", "GREECE", "Baseline", "MUSE", 12),
        ]
        manager = build_manager(ts, [{"modelName": "GCAM"}, {"modelName": "MUSE"}])
        answer = manager.route_query("Which models have data for Greece?")
        self.assertIn("Models with data for", answer)
        self.assertIn("GCAM", answer)
        self.assertIn("MUSE", answer)
        self.assertNotIn("No loaded IAM PARIS variable", answer)

    def test_exact_curated_model_is_not_replaced_by_longer_name(self):
        manager = build_manager(models=[{"modelName": "REMIND-MFA"}])
        profiles = manager._runtime_model_profiles("How does REMIND represent energy transitions?")
        self.assertTrue(profiles)
        self.assertEqual(profiles[0]["name"], "REMIND")

    def test_plural_members_of_one_family_can_be_compared(self):
        manager = build_manager()
        manager.entity_extractor.available_scenarios = [
            "Baseline", "PR_Baseline", "PR_CurPol_CP",
        ]
        values = manager._scenario_comparison_followup_values(
            "compare with current policy",
            {"variable": "Emissions|CO2", "scenarios": ["Baseline", "PR_Baseline"]},
        )
        self.assertEqual(values, ["Baseline", "Current Policies"])

    def test_provenance_after_chart_redirect_uses_previous_data(self):
        manager = build_manager()
        manager.route_query("CO2 emissions for World in 2050")
        manager.route_query("plot it")
        answer = manager.route_query("Which models contributed to it?")
        self.assertIn("GCAM", answer)
        self.assertNotIn("Models (0)", answer)

    def test_explicit_chart_request_preserves_scope_for_edit(self):
        manager = build_manager()
        manager.route_query("plot Emissions|CO2 for World under Baseline from 2030 to 2050")
        self.assertEqual(manager.last_entities.get("variable"), "Emissions|CO2")
        self.assertEqual(manager.last_entities.get("region"), "World")
        self.assertEqual(manager.last_entities.get("start_year"), 2030)
        self.assertEqual(manager.last_entities.get("end_year"), 2050)


if __name__ == "__main__":
    unittest.main()
