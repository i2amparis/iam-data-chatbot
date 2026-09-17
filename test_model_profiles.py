import unittest
from unittest.mock import patch

from model_profiles import (
    CURATED_MODEL_PROFILES,
    find_model_profile,
    find_model_profiles,
    format_model_comparison_answer,
    format_model_profile_answer,
    has_strong_model_metadata,
)


class ModelProfileTests(unittest.TestCase):
    def test_longer_literal_model_does_not_add_substring_family(self):
        profiles = {
            "Alpha": {"name": "Alpha", "aliases": ["alpha"]},
            "Alpha-Extended": {"name": "Alpha-Extended", "aliases": ["alpha-extended"]},
            "Beta": {"name": "Beta", "aliases": ["beta"]},
        }
        with patch.dict(CURATED_MODEL_PROFILES, profiles, clear=True):
            matches = find_model_profiles("Compare Alpha-Extended with Beta")

        self.assertEqual([profile["name"] for profile in matches], ["Alpha-Extended", "Beta"])

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

    def test_remind_magpie_prefers_specific_coupled_profile(self):
        profile = find_model_profile("Who develops the REMIND-MAgPIE model?")
        self.assertEqual(profile["name"], "REMIND-MAgPIE")
        response = format_model_profile_answer(profile)
        self.assertIn("Developer:", response)
        self.assertIn("Potsdam Institute", response)

    def test_format_model_profile_answer_includes_assumption_context_and_link(self):
        profile = find_model_profile("REMIND CCS assumptions")
        response = format_model_profile_answer(profile, asks_assumptions=True)

        self.assertIn("### REMIND", response)
        self.assertIn("Assumptions:", response)
        self.assertIn("scenario-dependent", response)
        self.assertIn("[IAM PARIS Models](https://iamparis.eu/models)", response)

    def test_format_model_profile_answer_links_specific_iamparis_model_page_from_id(self):
        response = format_model_profile_answer({
            "name": "ALADIN",
            "description": "Alternative automobiles diffusion model.",
            "id": 6,
        })

        self.assertIn("[IAM PARIS model page](https://iamparis.eu/models/6)", response)
        self.assertNotIn("[IAM PARIS Models](https://iamparis.eu/models)", response)

    def test_metadata_strength_detects_weak_records(self):
        self.assertFalse(has_strong_model_metadata({"description": ""}))
        self.assertFalse(has_strong_model_metadata({"description": "Short."}))
        self.assertTrue(has_strong_model_metadata({"description": "A" * 140}))

    def test_comparison_follows_requested_metadata_dimensions(self):
        profiles = find_model_profiles("Compare REMIND with WITCH")
        response = format_model_comparison_answer(
            profiles,
            "Compare them, focusing on limitations and typical uses",
        )

        self.assertIn("Common uses:", response)
        self.assertIn("Limitations:", response)
        self.assertNotIn("Systems/sectors:", response)

    def test_comparison_can_focus_on_assumptions(self):
        profiles = find_model_profiles("REMIND and WITCH")
        response = format_model_comparison_answer(profiles, "Compare their assumptions")

        self.assertEqual(response.count("- Assumptions:"), 2)
        self.assertNotIn("Common uses:", response)

    def test_transitive_uses_with_approach_selects_methodology_not_common_uses(self):
        profiles = [
            {
                "name": "Alpha",
                "methodology_note": "Simulation framework",
                "technology_note": "Detailed technology inventory",
                "typical_use_cases": ["system planning"],
            },
            {
                "name": "Beta",
                "methodology_note": "Linear optimization framework",
                "technology_note": "Aggregate technology representation",
                "typical_use_cases": ["policy analysis"],
            },
        ]

        response = format_model_comparison_answer(
            profiles,
            "Which of these two models uses an optimization approach?",
        )

        self.assertIn("- Methodology/model type: Simulation framework", response)
        self.assertIn("- Methodology/model type: Linear optimization framework", response)
        self.assertNotIn("Common uses:", response)
        self.assertNotIn("Technology representation:", response)
        self.assertIn("Comparison focus: method.", response)
        self.assertIn("Catalogue wording match for `optimization`: **Beta**.", response)

    def test_general_choice_question_states_catalogue_wording_match(self):
        profiles = [
            {"name": "Alpha", "sectors": ["land", "water"]},
            {"name": "Beta", "sectors": ["economy", "energy"]},
        ]

        response = format_model_comparison_answer(
            profiles, "Which of these models is described as economy-focused?"
        )

        self.assertIn("Catalogue wording match for `economy`: **Beta**.", response)

    def test_technology_focus_uses_technology_metadata_not_methodology(self):
        profiles = [
            {
                "name": "Alpha",
                "methodology_note": "Method Alpha",
                "technology_note": "Technology detail Alpha",
            },
            {
                "name": "Beta",
                "methodology_note": "Method Beta",
                "technology_note": "Technology detail Beta",
            },
        ]

        response = format_model_comparison_answer(
            profiles,
            "Compare their treatment of technologies.",
        )

        self.assertIn("- Technology representation: Technology detail Alpha", response)
        self.assertIn("- Technology representation: Technology detail Beta", response)
        self.assertNotIn("Methodology/model type:", response)
        self.assertIn("Comparison focus: technology.", response)

    def test_common_uses_noun_phrase_still_selects_use_cases(self):
        profiles = [
            {"name": "Alpha", "typical_use_cases": ["system planning"]},
            {"name": "Beta", "typical_use_cases": ["policy analysis"]},
        ]

        response = format_model_comparison_answer(profiles, "Compare their uses")

        self.assertIn("- Common uses: system planning", response)
        self.assertIn("- Common uses: policy analysis", response)
        self.assertIn("Comparison focus: uses.", response)

    def test_model_names_are_not_treated_as_unmatched_comparison_focus(self):
        profiles = [
            {
                "name": "GCAM",
                "aliases": ["gcam"],
                "methodology_note": "Partial equilibrium",
            },
            {
                "name": "GEM-E3",
                "methodology_note": "Recursive Dynamic Computable General Equilibrium",
            },
        ]

        response = format_model_comparison_answer(
            profiles,
            "Compare GCAM and GEM-E3 as model types",
        )

        self.assertIn("- Methodology/model type: Partial equilibrium", response)
        self.assertIn("- Methodology/model type: Recursive Dynamic Computable General Equilibrium", response)
        self.assertNotIn("does not explicitly match", response)
        self.assertNotIn("Catalogue wording match for `gcam`", response)

    def test_method_question_explicitly_reports_missing_profile_dimension(self):
        profile = find_model_profile("REMIND")

        response = format_model_profile_answer(
            profile,
            query="Is REMIND a general equilibrium model?",
        )

        self.assertIn(
            "does not state whether `REMIND` is a general equilibrium model",
            response,
        )
        self.assertNotIn("Description:", response)
        self.assertNotIn("Model scope:", response)

    def test_method_question_directly_answers_from_structured_metadata(self):
        response = format_model_profile_answer(
            {
                "name": "Example",
                "methodology_note": "Computable General Equilibrium",
            },
            query="Is Example a general equilibrium model?",
        )

        self.assertIn("Yes.", response)
        self.assertIn("Computable General Equilibrium", response)

    def test_what_kind_question_leads_with_structured_model_type(self):
        response = format_model_profile_answer(
            {
                "name": "TIAM_Grantham",
                "description": "A stale description about a release planned for 2020.",
                "methodology_note": "Partial Equilibrium",
            },
            query="What kind of model is TIAM?",
        )

        self.assertIn("Methodology/model type: Partial Equilibrium", response)
        self.assertNotIn("planned for 2020", response)
        self.assertNotIn("Description:", response)

    def test_technology_question_uses_only_relevant_grounded_profile_evidence(self):
        profile = find_model_profile("REMIND")

        response = format_model_profile_answer(
            profile,
            query="How does REMIND handle technological change?",
        )

        self.assertIn("Technology information in the loaded profile:", response)
        self.assertIn("technology choices", response)
        self.assertIn("does not provide a more specific mechanism", response)
        self.assertNotIn("Model scope:", response)
        self.assertNotIn("Useful for:", response)

    def test_image_developer_answer_is_grounded_in_curated_profile(self):
        profile = find_model_profile("Who develops the IMAGE model?")

        response = format_model_profile_answer(
            profile,
            query="Who develops the IMAGE model?",
        )

        self.assertEqual(profile["name"], "IMAGE")
        self.assertIn("IMAGE team", response)
        self.assertIn("PBL Netherlands Environmental Assessment Agency", response)
        self.assertIn("https://www.pbl.nl/en/image/home", response)
        self.assertNotIn("Description:", response)

    def test_e3me_profile_uses_repository_model_metadata(self):
        profile = find_model_profile("What does the E3ME model do?")

        response = format_model_profile_answer(profile)

        self.assertEqual(profile["name"], "E3ME-FTT")
        self.assertIn("dynamic global macroeconomic model", response)
        self.assertIn("Cambridge Econometrics", response)
        self.assertIn("economy-energy-environment policy analysis", response)

    def test_coverage_question_returns_explicit_not_stated_verdict(self):
        profile = find_model_profile("WITCH")

        response = format_model_profile_answer(
            profile,
            query="Does WITCH model land use?",
        )

        self.assertIn("I cannot verify", response)
        self.assertIn("Please consult the model documentation", response)
        self.assertIn("https://iamparis.eu/models", response)
        self.assertNotIn("Description:", response)


if __name__ == "__main__":
    unittest.main()
