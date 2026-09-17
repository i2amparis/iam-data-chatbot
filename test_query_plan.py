import unittest

from query_plan import build_query_plan, render_scope_query


class QueryPlanTests(unittest.TestCase):

    def test_negative_chart_followup_selects_table(self):
        plan = build_query_plan("Do not show a chart now; give the 2040 value in text")
        self.assertTrue(plan.followup)
        self.assertEqual(plan.output_mode, "table")
        self.assertEqual(plan.start_year, 2040)
        self.assertEqual(plan.end_year, 2040)

    def test_comparison_morphology_is_recognized(self):
        plan = build_query_plan(
            "How are Alpha and Beta different?",
            available_models=["Alpha", "Beta"],
        )
        self.assertEqual(plan.intent, "model_comparison")
    def test_detects_two_runtime_models_as_metadata_comparison(self):
        plan = build_query_plan(
            "How do AlphaModel and BetaModel differ in sector coverage?",
            available_models=["AlphaModel", "BetaModel", "GammaModel"],
        )
        self.assertEqual(plan.intent, "model_comparison")
        self.assertEqual(set(plan.mentioned_models), {"AlphaModel", "BetaModel"})

    def test_availability_targets_are_generic_dimensions(self):
        plan = build_query_plan("For this indicator, what regions and pathways are available?")
        self.assertEqual(plan.intent, "availability")
        self.assertIn("region", plan.availability_targets)
        self.assertIn("scenario", plan.availability_targets)

    def test_availability_projection_excludes_filter_dimensions(self):
        scenarios = build_query_plan("Which scenarios are available for that variable?")
        models = build_query_plan("Which models report it in region R1?")
        variables = build_query_plan(
            "Which variables are available for Alpha 2.0 in region R1?",
            available_models=["Alpha 2.0"],
        )

        self.assertEqual(scenarios.availability_targets, ("scenario",))
        self.assertEqual(models.availability_targets, ("model",))
        self.assertEqual(variables.availability_targets, ("variable",))

    def test_negative_chart_request_selects_table_output(self):
        plan = build_query_plan("Instead of a chart, give me only the value for 2050")
        self.assertEqual(plan.output_mode, "table")
        self.assertEqual(plan.start_year, 2050)
        self.assertTrue(plan.followup)

    def test_dimension_switch_mutates_only_requested_scope(self):
        plan = build_query_plan("Keep everything else, but switch the geography to Global Aggregate")
        rendered = render_scope_query(
            {
                "variable": "Metric|Example",
                "region": "Old Region",
                "scenario": "Scenario A",
                "start_year": 2030,
                "end_year": 2040,
            },
            plan,
        )
        self.assertIn("Metric|Example", rendered)
        self.assertIn("for Global Aggregate", rendered)
        self.assertIn("under Scenario A", rendered)
        self.assertIn("from 2030 to 2040", rendered)

    def test_value_before_scenario_dimension_extracts_replacement(self):
        plan = build_query_plan("Use current-policy scenarios instead, keeping 2030.")

        self.assertTrue(plan.followup)
        self.assertEqual(plan.replacement_dimension, "scenario")
        self.assertEqual(plan.replacement_value, "current-policy")

    def test_switch_value_excludes_trailing_preservation_instruction(self):
        plan = build_query_plan(
            "Switch the scenario to Future_Path and keep the same year."
        )

        self.assertEqual(plan.replacement_dimension, "scenario")
        self.assertEqual(plan.replacement_value, "Future_Path")

    def test_switch_to_value_before_dimension(self):
        plan = build_query_plan("Switch to Alternative-Path scenarios.")

        self.assertEqual(plan.replacement_dimension, "scenario")
        self.assertEqual(plan.replacement_value, "Alternative-Path")

    def test_preserved_scope_value_without_dimension_is_runtime_classified(self):
        plan = build_query_plan("Now use R2 but keep everything else.")

        self.assertEqual(plan.replacement_dimension, "auto")
        self.assertEqual(plan.replacement_value, "R2")

    def test_plot_the_comparison_is_a_followup(self):
        plan = build_query_plan("Plot the comparison.")

        self.assertTrue(plan.followup)
        self.assertEqual(plan.output_mode, "plot")

    def test_change_only_variable_is_a_dimension_patch(self):
        plan = build_query_plan("Change only the variable to Metric|Leaf.")

        self.assertEqual(plan.replacement_dimension, "variable")
        self.assertEqual(plan.replacement_value, "Metric|Leaf")

    def test_show_only_exact_runtime_model_is_a_scope_patch(self):
        plan = build_query_plan(
            "Show only model Alpha 2.0.",
            available_models=["Alpha 2.0", "Beta 1.0"],
        )

        self.assertEqual(plan.replacement_dimension, "model")
        self.assertEqual(plan.replacement_value, "Alpha 2.0")

    def test_show_only_runtime_label_can_omit_model_noun(self):
        plan = build_query_plan(
            "Show only Alpha 2.0.",
            available_models=["Alpha 2.0", "Beta 1.0"],
        )

        self.assertEqual(plan.replacement_dimension, "model")
        self.assertEqual(plan.replacement_value, "Alpha 2.0")

    def test_table_switch_preserves_single_plural_scenario(self):
        plan = build_query_plan("Now show the values instead of a plot.")
        rendered = render_scope_query(
            {
                "action": "plot", "variable": "Metric", "region": "R1",
                "scenarios": ["Path_A"], "start_year": 2030, "end_year": 2050,
            },
            plan,
        )

        self.assertTrue(rendered.startswith("show "))
        self.assertIn("under Path_A", rendered)

    def test_after_year_is_an_open_ended_scope_patch(self):
        plan = build_query_plan("after 2030")
        rendered = render_scope_query(
            {
                "variable": "Metric", "region": "R1",
                "start_year": 2020, "end_year": 2030,
            },
            plan,
        )

        self.assertTrue(plan.followup)
        self.assertTrue(plan.year_filter.explicit)
        self.assertEqual(plan.start_year, 2031)
        self.assertIsNone(plan.end_year)
        self.assertIn("after 2030", rendered)

    def test_until_year_clears_carried_lower_bound(self):
        plan = build_query_plan("until 2050")
        updated = plan.scope_patch.apply({"start_year": 2030, "end_year": 2040})
        rendered = render_scope_query(
            {"variable": "Metric", "region": "R1", "start_year": 2030},
            plan,
        )

        self.assertTrue(plan.followup)
        self.assertNotIn("start_year", updated)
        self.assertEqual(updated["end_year"], 2050)
        self.assertIn("until 2050", rendered)


if __name__ == "__main__":
    unittest.main()
