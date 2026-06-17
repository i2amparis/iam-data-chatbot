import unittest

from data_utils import _choice_prompt, _compact_recovery_prompt


class ClarificationPromptTests(unittest.TestCase):
    def test_choice_prompt_asks_for_one_missing_item(self):
        response = _choice_prompt(
            "I need one more detail.",
            "variable",
            ["Emissions|CO2", "Capacity|Electricity|Solar"],
        )

        self.assertIn("Choose the variable:", response)
        self.assertNotIn("Choose the region:", response)
        self.assertNotIn("Choose the scenario:", response)
        self.assertIn("Reply with a number", response)

    def test_choice_prompt_includes_option_reasons(self):
        response = _choice_prompt(
            "I found close options.",
            "variable",
            ["Emissions|CO2", "Capacity|Electricity|Solar"],
        )

        self.assertIn("`Emissions|CO2`", response)
        self.assertIn("(CO2 emissions)", response)
        self.assertIn("`Capacity|Electricity|Solar`", response)
        self.assertIn("(power capacity, solar)", response)

    def test_recovery_prompt_includes_region_and_scenario_reasons(self):
        response = _compact_recovery_prompt(
            "No data found.",
            region_options=["EU"],
            scenario_options=["Baseline"],
        )

        self.assertIn("Closest valid options:", response)
        self.assertIn("1. region `EU` (EU (European Union))", response)
        self.assertIn("2. scenario `Baseline` (baseline)", response)
        self.assertIn("Reply with `1`, `2`, or `3`", response)


if __name__ == "__main__":
    unittest.main()
