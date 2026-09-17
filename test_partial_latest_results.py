import unittest
import math

from agents import DataQueryAgent
from data_utils import data_query
from fastapi_app import _build_data_provenance, _count_matching_records, _is_no_data_answer
from resolved_scope import consume_resolved_scope, has_numeric_result_table
import test_manager_fallback


def record(scenario="Baseline", years=None, model="GCAM"):
    return dict(workspace_code="world-headed", variable="Emissions|CO2",
                region="World", scenario=scenario, modelName=model,
                unit="Mt CO2/yr", years=years or {"2050": 100})


class PartialLatestResultsTests(unittest.TestCase):
    def test_comparison_warns_for_missing_variable_scenario_pair(self):
        records = [record(), record("Policy"),
                   {**record(), "variable": "Emissions|CH4", "unit": "Mt CH4/yr"}]
        entities = dict(workspace_code="world-headed", region="World",
                        variables=["Emissions|CO2", "Emissions|CH4"],
                        scenarios=["Baseline", "Policy"], start_year=2050, end_year=2050)
        answer = data_query("Compare CO2 and methane", [], records,
                            forced_entities=entities, allow_plots=False)
        self.assertIn("For `Emissions|CH4` in `World`, no data", answer)
        self.assertIn("scenario: `Policy`", answer)
        self.assertTrue(has_numeric_result_table(answer))
        self.assertFalse(_is_no_data_answer(answer))

    def test_distinct_units_are_not_merged(self):
        records = [record(years={"2030": 100}),
                   {**record(years={"2050": .08}), "unit": "Gt CO2/yr"}]
        entities = dict(workspace_code="world-headed", region="World",
                        variable="Emissions|CO2", start_year=2030, end_year=2050)
        for rows in (records, list(reversed(records))):
            answer = data_query("CO2 from 2030 to 2050", [], rows,
                                forced_entities=entities, allow_plots=False)
            self.assertIn("| 2050 | 0.08 | Gt CO2/yr |", answer)
            self.assertIn("| 2030 | 100.00 | Mt CO2/yr |", answer)
            self.assertNotIn("| 2050 | 0.08 | Mt CO2/yr |", answer)

    def test_latest_uses_last_finite_value_not_last_declared_year(self):
        records = [record(years={"2030": 50, "2050": 100, "2100": None})]
        entities = dict(workspace_code="world-headed", variable="Emissions|CO2",
                        region="World", start_year=-1, end_year=-1)

        answer = data_query("latest CO2 emissions", [], records,
                            forced_entities=entities, allow_plots=False)

        self.assertIn("| 2050 | 100.00 |", answer)
        self.assertNotIn("No data found", answer)
        self.assertEqual(_count_matching_records({"ts": records}, entities), 1)

    def test_latest_excludes_nan_and_infinite_values(self):
        records = [record(years={"2030": 50, "2050": math.nan, "2100": math.inf})]
        entities = dict(workspace_code="world-headed", variable="Emissions|CO2",
                        region="World", start_year=-1, end_year=-1)

        answer = data_query("latest CO2 emissions", [], records,
                            forced_entities=entities, allow_plots=False)

        self.assertIn("| 2030 | 50.00 |", answer)
        self.assertNotIn(" nan ", answer.casefold())
        self.assertNotIn(" inf ", answer.casefold())
        self.assertEqual(_count_matching_records({"ts": records}, entities), 1)

    def test_latest_comparison_warns_and_counts_only_displayed_year(self):
        records = [record(), record("Policy", {"2030": 80})]
        entities = dict(workspace_code="world-headed", variable="Emissions|CO2",
                        region="World", scenarios=["Baseline", "Policy"],
                        start_year=-1, end_year=-1)
        consume_resolved_scope()
        answer = data_query("Compare Baseline and Policy at the latest year", [], records,
                            forced_entities=entities, allow_plots=False)
        scope = consume_resolved_scope()
        self.assertIn("scenario: `Policy`", answer)
        self.assertIn("| 2050 | 100.00 |", answer)
        self.assertNotIn("| 2030 |", answer)
        provenance = _build_data_provenance({"ts": records}, scope, answer, {"agent": "data_query"})
        self.assertEqual(provenance["matched_record_count"], 1)
        self.assertNotIn("no_data_reason", provenance)
        self.assertEqual(_count_matching_records({"ts": records}, entities), 1)

    def test_partial_results_keep_context_and_success_metadata_in_both_formats(self):
        for count in (1, 7):
            with self.subTest(series=count):
                records = [record(model=f"Model {i}") for i in range(count)]
                entities = dict(workspace_code="world-headed", region="World",
                                variables=["Emissions|CO2", "Emissions|CH4"],
                                start_year=2050, end_year=2050)
                manager = test_manager_fallback.ManagerFallbackTests()._build_manager(entities)
                manager.shared_resources["ts"] = records
                agent = DataQueryAgent(manager.shared_resources, streaming=False)
                consume_resolved_scope()
                answer = agent.handle_with_entities("Compare CO2 and methane for 2050", entities)
                manager._persist_last_entities(entities, answer)
                self.assertTrue(has_numeric_result_table(answer))
                self.assertFalse(_is_no_data_answer(answer))
                scope = manager.response_entities()
                self.assertEqual(scope["variables"], ["Emissions|CO2"])
                self.assertEqual(scope["workspace_code"], "world-headed")
                provenance = _build_data_provenance(manager.shared_resources, scope, answer,
                                                     {"agent": "data_query"})
                self.assertEqual(provenance["matched_record_count"], count)
                self.assertNotIn("no_data_reason", provenance)

    def test_empty_headers_are_not_successful_results(self):
        for header in ("| Year | Value | Unit |", "| Model | Scenario | 2050 | Unit |"):
            answer = "No data found.\n" + header + "\n|---|---|---|\n"
            self.assertFalse(has_numeric_result_table(answer))
            self.assertTrue(_is_no_data_answer(answer))


if __name__ == "__main__":
    unittest.main()
