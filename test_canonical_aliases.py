import unittest

from canonical_aliases import (
    preferred_variable_from_query,
    scenario_in_family,
    scenario_family_members,
)


class CanonicalScenarioTests(unittest.TestCase):
    def test_current_policies_family_matches_dataset_codes(self):
        self.assertTrue(scenario_in_family("PR_CurPol_CP", "Current Policies"))
        self.assertTrue(scenario_in_family("PR_CurPol_EI", "Current Policies"))
        self.assertFalse(scenario_in_family("PR_Baseline", "Current Policies"))

    def test_baseline_family(self):
        self.assertTrue(scenario_in_family("PR_Baseline", "Baseline"))
        self.assertTrue(scenario_in_family("Unharmonised baseline", "Baseline"))

    def test_family_members(self):
        codes = ["PR_CurPol_CP", "PR_Baseline", "PR_CurPol_EI", "NZE"]
        self.assertEqual(
            scenario_family_members("Current Policies", codes),
            ["PR_CurPol_CP", "PR_CurPol_EI"],
        )


class CanonicalVariableTests(unittest.TestCase):
    AVAIL = {
        "Final Energy", "Final Energy|Geothermal", "Final Energy|Electricity",
        "Primary Energy", "Secondary Energy", "Secondary Energy|Electricity",
        "Emissions|CO2",
    }

    def test_bare_final_energy_resolves_to_base(self):
        # Regression: a bare "final energy" request used to fuzzy-match an
        # over-specific carrier (e.g. Final Energy|Geothermal).
        self.assertEqual(
            preferred_variable_from_query("final energy for EU from MUSE", self.AVAIL),
            "Final Energy",
        )

    def test_specific_carrier_not_stolen_by_energy_base(self):
        # A specific carrier under an energy family must not collapse to the base.
        self.assertIsNone(
            preferred_variable_from_query("final energy electricity for EU", self.AVAIL)
        )
        self.assertIsNone(
            preferred_variable_from_query("secondary energy electricity for EU", self.AVAIL)
        )


if __name__ == "__main__":
    unittest.main()
