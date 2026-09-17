import unittest

from canonical_aliases import (
    canonical_scenario_from_query,
    explicit_scenarios_from_query,
    preferred_variable_from_query,
    rank_catalogue_variable_matches,
    scenario_in_family,
    scenario_family_members,
)


class CanonicalScenarioTests(unittest.TestCase):
    def test_net_zero_alias_resolves_to_stable_family_not_arbitrary_member(self):
        scenarios = {"NZE_Bench_M", "NZE_Bench_H", "NZE_Bench_L", "Baseline"}

        self.assertEqual(
            canonical_scenario_from_query("emissions under net zero", scenarios),
            "Net Zero",
        )
        self.assertEqual(
            scenario_family_members("Net Zero", scenarios),
            ["NZE_Bench_H", "NZE_Bench_L", "NZE_Bench_M"],
        )

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

    def test_explicit_mentions_dedupe_case_only_catalogue_variants(self):
        scenarios = ["Baseline", "baseline", "PR_Baseline"]

        self.assertEqual(
            explicit_scenarios_from_query("show data under baseline", scenarios),
            ["Baseline"],
        )

    def test_explicit_code_accepts_separator_variants_without_matching_suffix(self):
        scenarios = ["Baseline", "PR_Baseline"]

        self.assertEqual(
            explicit_scenarios_from_query("show data under PR-Baseline", scenarios),
            ["PR_Baseline"],
        )


class CanonicalVariableTests(unittest.TestCase):
    def test_single_generic_overlap_is_not_auto_accepted(self):
        ranking = rank_catalogue_variable_matches(
            "synthetic aviation fuel production",
            ["Agricultural Production", "Final Energy|Transportation|Aviation"],
        )

        self.assertFalse(ranking[0]["auto_accept"])

    def test_exact_catalogue_token_outranks_similar_spelling(self):
        ranking = rank_catalogue_variable_matches(
            "synthetic aviation fuel production",
            [
                "Energy Service|Transportation|Freight|Navigation",
                "Final Energy|Industry|Cement|Waste based fuels",
                "Final Energy|Transportation|Aviation",
            ],
        )

        self.assertEqual(ranking[0]["variable"], "Final Energy|Transportation|Aviation")
        self.assertFalse(ranking[0]["auto_accept"])

    def test_exact_phrase_and_unambiguous_typo_are_auto_accepted(self):
        variables = ["Agricultural Production", "Population", "Population|Urban"]

        exact = rank_catalogue_variable_matches("agricultural production", variables)
        typo = rank_catalogue_variable_matches("populaton", variables)

        self.assertEqual(exact[0]["variable"], "Agricultural Production")
        self.assertTrue(exact[0]["auto_accept"])
        self.assertEqual(typo[0]["variable"], "Population")
        self.assertTrue(typo[0]["auto_accept"])

    def test_unambiguous_typo_can_resolve_a_short_hierarchical_variable(self):
        ranking = rank_catalogue_variable_matches(
            "emisions for europ",
            ["Emissions|CO2", "Population"],
            ignored_values=["Europe"],
        )

        self.assertEqual(ranking[0]["variable"], "Emissions|CO2")
        self.assertTrue(ranking[0]["auto_accept"])

    def test_structural_words_do_not_force_an_unrequested_descendant(self):
        variables = ["Emissions|CH4", "Emissions|CH4|Energy|Demand|Residential and Commercial"]
        self.assertEqual(
            preferred_variable_from_query("methane emissions between 2020 and 2060", variables),
            "Emissions|CH4",
        )

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
        # A specific carrier named under an energy family resolves to that exact
        # carrier variable (when it exists in the catalogue), rather than being
        # collapsed to the generic base. The carrier is unambiguous here, so it
        # is resolved rather than turned into a clarification.
        self.assertEqual(
            preferred_variable_from_query("final energy electricity for EU", self.AVAIL),
            "Final Energy|Electricity",
        )
        self.assertEqual(
            preferred_variable_from_query("secondary energy electricity for EU", self.AVAIL),
            "Secondary Energy|Electricity",
        )

    def test_industry_emissions_prefers_sector_aggregate_over_steel(self):
        available = {
            "Emissions|CO2|Energy|Demand|Industry",
            "Emissions|CO2|Energy|Demand|Industry|Steel",
            "Emissions|CO2|Industry|Steel",
        }

        self.assertEqual(
            preferred_variable_from_query("industry emissions in Europe", available),
            "Emissions|CO2|Energy|Demand|Industry",
        )

    def test_richer_energy_phrases_resolve_to_grounded_parent_variables(self):
        available = {
            "Emissions|Kyoto Gases",
            "Final Energy|Industry",
            "Secondary Energy|Electricity|Hydro",
            "Secondary Energy|Electricity|Wind|Onshore",
            "Capacity|Electricity|Solar",
            "Carbon Capture",
            "Carbon Sequestration|CCS|Biomass",
            "Price|Carbon",
        }
        cases = {
            "Kyoto-gas emissions in Africa after 2040": "Emissions|Kyoto Gases",
            "Industrial final energy demand for China": "Final Energy|Industry",
            "Hydropower electricity for Brazil": "Secondary Energy|Electricity|Hydro",
            "Electricity generated from onshore wind for Germany": "Secondary Energy|Electricity|Wind|Onshore",
            "Installed solar power capacity in India": "Capacity|Electricity|Solar",
            "Carbon capture and storage in the World": "Carbon Capture",
            "Carbon removal from biomass with CCS": "Carbon Sequestration|CCS|Biomass",
            "Global carbon price under current-policy scenarios": "Price|Carbon",
        }

        for query, expected in cases.items():
            with self.subTest(query=query):
                self.assertEqual(preferred_variable_from_query(query, available), expected)

    def test_natural_output_sector_and_typo_phrases_keep_exact_scope(self):
        available = {
            "Emissions|CH4",
            "Emissions|CO2",
            "Emissions|CO2|AFOLU",
            "Emissions|CO2|Energy|Supply|Electricity",
            "Final Energy|Liquids",
            "Land Cover|Forest",
            "Primary Energy",
            "Production|Cement",
            "Production|Steel",
        }
        cases = {
            "Methan emisions for indai": "Emissions|CH4",
            "carbon dioxide release for europ": "Emissions|CO2",
            "land-use CO2 emissions in Brazil": "Emissions|CO2|AFOLU",
            "power-sector CO2 emissions for World": "Emissions|CO2|Energy|Supply|Electricity",
            "How much oil demand does the EU report?": "Final Energy|Liquids",
            "Primary enegy for wrold": "Primary Energy",
            "steel output for India": "Production|Steel",
            "cement output for China": "Production|Cement",
            "forest area in Brazil": "Land Cover|Forest",
        }

        for query, expected in cases.items():
            with self.subTest(query=query):
                self.assertEqual(preferred_variable_from_query(query, available), expected)


if __name__ == "__main__":
    unittest.main()
