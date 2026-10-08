"""Behavior regressions from the 2,000-query workflow audit.

Synthetic catalogue values prove that routing and edits use runtime data,
without encoding the audit's specific models, studies or answers.
"""
import unittest
from unittest.mock import MagicMock

from canonical_aliases import preferred_variable_from_query
from data_utils import data_query, unknown_comparison_region
from link_router import infer_navigation_category, suggest_links
from test_response_fixes import build_manager
from year_filters import extract_year_filter


def records():
    return [
        {
            "workspace_code": "sample-study", "study_title": "Sample study",
            "variable": variable, "region": region, "scenario": "Reference",
            "modelName": "Atlas 2.0", "unit": "EJ/yr",
            "years": {"2040": value, "2070": value * 2},
        }
        for variable in ("Final Energy", "Final Energy|Non-Energy Use", "Agricultural Demand")
        for region, value in (("ITA", 10), ("ESP", 20), ("CHN", 30), ("EU", 40))
    ]


class WorkflowReportRegressions(unittest.TestCase):
    def manager(self):
        return build_manager(records(), [{
            "modelName": "Atlas 2.0", "description": "A synthetic energy system model.",
        }])

    def test_catalogue_grammar_does_not_create_regions_from_cache_words(self):
        queries = [
            ("Display the study titles in the results cache", "Sample study"),
            ("Retrieve the model names in the cache", "Atlas"),
            ("Enumerate the scenario labels in the cache", "Reference"),
            ("Identify the regional scopes available for queries", "ITA"),
            ("Display the variables in the time-series cache", "Final Energy"),
            ("Retrieve the first and last years in the cache", "2070"),
        ]
        for query, expected in queries:
            with self.subTest(query=query):
                manager = self.manager()
                answer = manager.route_query(query)
                self.assertIn(expected, answer)
                self.assertNotIn("as a region", answer)
                self.assertNotIn("Which variable", answer)
                self.assertEqual(manager.last_route_decision["agent"], "data_query")

    def test_plain_catalogue_request_does_not_wait_for_model_inference(self):
        manager = self.manager()
        manager.entity_extractor.extract = MagicMock(side_effect=AssertionError("numeric extraction"))
        answer = manager.route_query("Identify the regional scopes available for queries")
        self.assertIn("ITA", answer)
        self.assertEqual(manager.last_route_decision["agent"], "data_query")
        manager.entity_extractor.extract.assert_not_called()

    def test_descriptive_model_questions_do_not_request_numeric_dimensions(self):
        for query in (
            "Which methods does Atlas use?",
            "Give me a concise profile of Atlas.",
            "What kind of scenarios can Atlas analyze?",
        ):
            with self.subTest(query=query):
                manager = self.manager()
                answer = manager.route_query(query)
                self.assertEqual(manager.last_route_decision["agent"], "model_explanation")
                self.assertIn("Atlas", answer)
                self.assertNotIn("Which variable", answer)

    def test_broad_climate_explanation_bypasses_numeric_extraction(self):
        manager = self.manager()
        manager.agents["general_qa"].handle = MagicMock(return_value="A conceptual explanation.")
        manager.entity_extractor.extract = MagicMock(side_effect=AssertionError("numeric extraction"))
        for query in (
            "Explain scenario uncertainty in plain language for an IAM reader.",
            "Explain model calibration in plain language for an IAM reader.",
            "What tradeoffs arise around policy ambition in climate scenarios?",
        ):
            answer = manager.route_query(query)
            self.assertEqual(answer, "A conceptual explanation.")
            self.assertEqual(manager.last_route_decision["agent"], "general_qa")
        manager.entity_extractor.extract.assert_not_called()

    def test_help_request_is_not_a_variable_lookup(self):
        for query in (
            "Can you guide me through energy projections?",
            "I need help finding mitigation pathways.",
            "Please show me how to explore energy projections.",
            "Can you guide me through study comparisons?",
        ):
            manager = self.manager()
            manager.agents["general_qa"].handle = MagicMock(return_value="Usage guidance.")
            manager.entity_extractor.extract = MagicMock(side_effect=AssertionError("numeric extraction"))
            self.assertEqual(manager.route_query(query), "Usage guidance.")
            self.assertEqual(manager.last_route_decision["agent"], "general_qa")
            manager.entity_extractor.extract.assert_not_called()
        manager = self.manager()
        answer = manager.route_query("Could you retrieve final energy for Italy during 2070?")
        self.assertIn("20", answer)
        self.assertEqual(manager.last_route_decision["agent"], "data_query")

    def test_numeric_data_is_not_replaced_by_model_year_or_variable_catalogue(self):
        manager = self.manager()
        answer = manager.route_query("What does Atlas report for final energy in EU in 2070?")
        self.assertIn("80", answer)
        self.assertNotIn("has these variables", answer)
        self.assertNotIn("reports data for years", answer)

    def test_region_alias_validation_accepts_each_comparison_side(self):
        self.assertIsNone(unknown_comparison_region(
            "Compare final energy for Italy versus Spain in 2070.",
            ["ITA", "ESP"], variables=["Final Energy"],
        ))
        self.assertEqual(unknown_comparison_region(
            "Compare final energy for Italy versus Atlantis in 2070.",
            ["ITA", "ESP"], variables=["Final Energy"],
        ), "Atlantis")

    def test_compound_followups_apply_region_year_and_comparison(self):
        manager = self.manager()
        manager.route_query("Final energy for Italy in 2040")
        answer = manager.route_query("Keep final energy and use Spain now.")
        self.assertIn("in ESP", answer)
        self.assertEqual(manager.last_entities["variable"], "Final Energy")
        answer = manager.route_query("Set the reporting year to 2070 for final energy.")
        self.assertIn("In 2070", answer)
        self.assertNotIn("2040", answer)
        answer = manager.route_query("Compare this final energy result with Italy.")
        self.assertIn("in ESP", answer)
        self.assertIn("in ITA", answer)
        self.assertNotIn("2040", answer)

    def test_comparison_after_unavailable_year_does_not_revert_to_old_values(self):
        manager = self.manager()
        manager.route_query("Final energy for Italy in 2040")
        no_data = manager.route_query("Set the reporting year to 2090 for final energy.")
        self.assertIn("No data", no_data)
        self.assertEqual(manager.last_entities["start_year"], 2040)
        self.assertEqual(manager.conversation_state.response_entities()["start_year"], 2090)
        answer = manager.route_query("Compare this final energy result with Spain.")
        self.assertIn("2090", answer)
        self.assertIn("returned no data", answer)
        self.assertEqual(manager.conversation_state.response_entities()["start_year"], 2090)
        self.assertNotIn("| Model |", answer)

    def test_compound_year_question_does_not_reuse_old_region(self):
        manager = self.manager()
        manager.route_query("Final energy for Spain in 2040")
        answer = manager.route_query("What about 2070 for final energy in China?")
        self.assertIn("in CHN", answer)
        self.assertIn("60", answer)
        self.assertNotIn("in ESP", answer)

    def test_assignment_year_is_a_point_but_until_remains_an_interval(self):
        assigned = extract_year_filter("Set the reporting year to 2070 for this metric")
        self.assertEqual((assigned.start_year, assigned.end_year), (2070, 2070))
        replacement = extract_year_filter("Use 2070 instead of 2040 for this metric")
        self.assertEqual((replacement.start_year, replacement.end_year), (2070, 2070))
        until = extract_year_filter("Show the values until 2070")
        self.assertEqual((until.start_year, until.end_year), (None, 2070))

    def test_command_and_measurement_words_do_not_become_variable_qualifiers(self):
        for query in (
            "Display agricultural demand for Italy in 2070.",
            "What agricultural demand is recorded for Italy in 2070?",
            "I want Italy values for agricultural demand in 2070.",
            "Could you retrieve agricultural demand for Italy during 2070?",
        ):
            manager = self.manager()
            answer = manager.route_query(query)
            self.assertIn("Agricultural Demand", answer)
            self.assertIn("20", answer)
            self.assertNotIn("Choose the variable", answer)

    def test_numeric_retrieval_with_a_dimension_label_stays_numeric(self):
        manager = self.manager()
        answer = manager.route_query("Retrieve final energy for region EU in 2070.")
        self.assertIn("80", answer)
        self.assertNotIn("Availability", answer)

    def test_per_capita_never_returns_a_total_measurement(self):
        ts = [{**record, "variable": "Emissions|CO2"} for record in records()]
        self.assertIsNone(preferred_variable_from_query("CO2 per capita", ["Emissions|CO2"]))
        answer = data_query("CO2 per capita for Italy in 2070", [], ts,
                            forced_entities={"variable": "Emissions|CO2"}, allow_plots=False)
        self.assertIn("No matching per-capita series", answer)
        self.assertNotIn("| Model |", answer)

    def test_navigation_roots_are_resolved_from_metadata_without_url_literals(self):
        catalogue = [{
            "title": "Archives", "category": "archives", "item_type": "route",
            "url": "https://example.org/arbitrary-path",
            "keywords": ["Project workspaces hub"], "verified_direct_url": True,
        }]
        query = "Where is the link for study workspaces?"
        category = infer_navigation_category(query, catalogue)
        links = suggest_links(query, catalogue, agent_name="general_qa", navigation_category=category)
        self.assertEqual(category, "archives")
        self.assertEqual(links[0]["url"], catalogue[0]["url"])

    def test_navigation_without_a_resolved_variable_still_returns_a_link(self):
        manager = self.manager()
        answer = manager.route_query("Open the active explorer for an unspecified metric.")
        self.assertIn("](https://", answer)
        self.assertFalse(manager.last_entities.get("variable"))
        manager.shared_resources["link_catalog"] = [{
            "title": "Models", "category": "models", "item_type": "route",
            "url": "https://example.org/profiles", "verified_direct_url": True,
            "keywords": ["Model documentation directory"],
        }]
        answer = manager.route_query("How can I get to model profiles?")
        self.assertIn("](https://example.org/profiles)", answer)
        self.assertEqual(manager.last_route_decision["agent"], "general_qa")

    def test_observed_models_do_not_turn_region_aliases_into_model_constraints(self):
        manager = self.manager()
        manager.entity_extractor.available_models.append("eu_times")
        manager.entity_extractor._model_match_names.append("eu_times")
        extracted = manager.entity_extractor.extract("What about 2070 for final energy in EU?")
        self.assertEqual(extracted["region"], "EU")
        self.assertFalse(extracted.get("model"))
