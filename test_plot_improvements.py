import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt

import fastapi_app
from manager import MultiAgentManager
from model_aliases import UNLABELLED_MODEL_LABEL
from resolved_scope import ConversationState, consume_resolved_scope
from year_filters import LATEST_YEAR_SENTINEL
from simple_plotter import (
    MAX_PLOT_SERIES,
    _pretty_variable_name,
    _year_range_text,
    plot_model_comparison,
    plot_multiple_variables,
    plot_variable_across_regions,
    simple_plot_query,
    simple_plot_query_with_entities,
)


def _record(variable, region, scenario, model, value, unit="EJ/yr"):
    return {
        "variable": variable,
        "region": region,
        "scenario": scenario,
        "modelName": model,
        "unit": unit,
        "years": {"2040": value, "2050": value + 1},
    }


class PlotImprovementTests(unittest.TestCase):
    def tearDown(self):
        consume_resolved_scope()
        plt.close("all")

    def test_region_comparison_keeps_models_instead_of_grouping_first(self):
        records = [
            _record("Metric", region, "Path", model, value)
            for value, (region, model) in enumerate(
                (("R1", "Alpha"), ("R1", "Beta"), ("R2", "Alpha"), ("R2", "Beta")),
                start=1,
            )
        ]

        with (
            patch("simple_plotter.plt.plot") as plot_mock,
            patch("simple_plotter._finalize_plot_layout"),
            patch("simple_plotter.save_plot_to_base64", return_value="![Plot](data:test)"),
        ):
            response = plot_variable_across_regions(
                "compare R1 and R2", [], records, "Metric", ["R1", "R2"], scenario="Path",
            )
        scope = consume_resolved_scope()

        self.assertIn("![Plot]", response)
        self.assertEqual(plot_mock.call_count, 4)
        self.assertEqual(
            {call.kwargs["label"] for call in plot_mock.call_args_list},
            {"R1 — Alpha", "R1 — Beta", "R2 — Alpha", "R2 — Beta"},
        )
        self.assertEqual(scope["regions"], ["R1", "R2"])
        self.assertEqual(scope["result_models"], ["Alpha", "Beta"])
        self.assertEqual(scope["comparison_dimension"], "region")
        self.assertEqual(len(scope["displayed_series"]), 4)
        self.assertEqual(scope["omitted_series"], 0)

    def test_region_comparison_prefers_complete_model_scenario_pairs(self):
        records = [
            _record("Metric", "R1", "Path", "Shared", 1),
            _record("Metric", "R2", "Path", "Shared", 2),
            _record("Metric", "R1", "R1 only", "Alpha", 3),
            _record("Metric", "R2", "R2 only", "Beta", 4),
        ]

        with (
            patch("simple_plotter.plt.plot") as plot_mock,
            patch("simple_plotter._finalize_plot_layout"),
            patch("simple_plotter.save_plot_to_base64", return_value="![Plot](data:test)"),
        ):
            response = plot_variable_across_regions(
                "compare Metric between R1 and R2", [], records, "Metric", ["R1", "R2"],
            )
        scope = consume_resolved_scope()

        self.assertEqual(plot_mock.call_count, 2)
        self.assertEqual(
            {tuple(call.args[1]) for call in plot_mock.call_args_list},
            {(1, 2), (2, 3)},
        )
        self.assertIn("uses model/scenario sources available in every requested region", response)
        self.assertEqual(scope["result_models"], ["Shared"])
        self.assertEqual(scope["comparison_pairing"], "shared model/scenario")
        self.assertEqual(scope["omitted_series"], 2)

    def test_region_comparison_discloses_when_sources_cannot_be_paired(self):
        records = [
            _record("Metric", "R1", "Path A", "Alpha", 1),
            _record("Metric", "R2", "Path B", "Beta", 2),
        ]

        with (
            patch("simple_plotter.plt.plot") as plot_mock,
            patch("simple_plotter.plt.title") as title_mock,
            patch("simple_plotter._finalize_plot_layout"),
            patch("simple_plotter.save_plot_to_base64", return_value="![Plot](data:test)"),
        ):
            response = plot_variable_across_regions(
                "compare Metric between R1 and R2", [], records, "Metric", ["R1", "R2"],
            )
        scope = consume_resolved_scope()

        self.assertIn("no shared model-and-scenario source", response)
        self.assertIn("not a like-for-like paired comparison", response)
        self.assertNotIn("R1 vs R2", response)
        self.assertNotIn(" vs ", title_mock.call_args.args[0])
        self.assertEqual(
            {call.kwargs["label"] for call in plot_mock.call_args_list},
            {"R1 — Alpha — Path A", "R2 — Beta — Path B"},
        )
        self.assertEqual(scope["comparison_pairing"], "unpaired source scopes")
        self.assertEqual(scope["comparison_dimension"], "region")

    def test_model_comparison_is_balanced_capped_and_reports_omissions(self):
        records = [
            _record("Metric", "World", f"Path {index}", model, index)
            for model in ("Alpha", "Beta")
            for index in range(6)
        ]

        with (
            patch("simple_plotter.plt.plot") as plot_mock,
            patch("simple_plotter._finalize_plot_layout"),
            patch("simple_plotter.save_plot_to_base64", return_value="![Plot](data:test)"),
        ):
            response = plot_model_comparison(
                "compare Alpha and Beta", [], records, "Metric", ["Alpha", "Beta"], region="World",
            )
        scope = consume_resolved_scope()

        self.assertEqual(plot_mock.call_count, MAX_PLOT_SERIES)
        self.assertIn("Showing 8 of 12 available series", response)
        self.assertEqual(scope["models"], ["Alpha", "Beta"])
        self.assertEqual(scope["result_models"], ["Alpha", "Beta"])
        self.assertEqual(scope["omitted_series"], 4)
        self.assertEqual(scope["displayed_series_count"], MAX_PLOT_SERIES)

    def test_multi_variable_comparison_keeps_each_model_and_labels_it(self):
        records = [
            _record(variable, "World", "Path", model, value)
            for value, (variable, model) in enumerate(
                (("Metric A", "Alpha"), ("Metric A", "Beta"),
                 ("Metric B", "Alpha"), ("Metric B", "Beta")),
                start=1,
            )
        ]

        with (
            patch("simple_plotter.plt.plot") as plot_mock,
            patch("simple_plotter._finalize_plot_layout"),
            patch("simple_plotter.save_plot_to_base64", return_value="![Plot](data:test)"),
        ):
            response = plot_multiple_variables(
                "compare Metric A and Metric B", [], records,
                ["Metric A", "Metric B"], region="World", scenario="Path",
            )
        scope = consume_resolved_scope()

        self.assertIn("![Plot]", response)
        self.assertEqual(plot_mock.call_count, 4)
        labels = {call.kwargs["label"] for call in plot_mock.call_args_list}
        self.assertEqual(labels, {
            "Metric A — Alpha", "Metric A — Beta",
            "Metric B — Alpha", "Metric B — Beta",
        })
        self.assertEqual(scope["variables"], ["Metric A", "Metric B"])
        self.assertEqual(scope["comparison_dimension"], "variable")

    def test_explicit_solar_pv_comparison_keeps_exact_pv_leaf(self):
        records = [
            _record("Capacity|Electricity|Wind", "GREECE", "Path", "Alpha", 10, unit="GW"),
            _record("Capacity|Electricity|Solar", "GREECE", "Path", "Alpha", 20, unit="GW"),
            _record("Capacity|Electricity|Solar|PV", "GREECE", "Path", "Alpha", 30, unit="GW"),
        ]

        with (
            patch("simple_plotter.plt.plot") as plot_mock,
            patch("simple_plotter._finalize_plot_layout"),
            patch("simple_plotter.save_plot_to_base64", return_value="![Plot](data:test)"),
        ):
            response = simple_plot_query_with_entities(
                "compare wind power and solar PV for Greece",
                [],
                records,
                {
                    "variable": "Capacity|Electricity|Wind",
                    "variables": [
                        "Capacity|Electricity|Wind",
                        "Capacity|Electricity|Solar",
                    ],
                    "region": "GREECE",
                    "scenario": "Path",
                    "comparison": "variable",
                },
            )
        scope = consume_resolved_scope()

        self.assertIn("Solar PV Capacity", response)
        self.assertEqual(
            scope["variables"],
            ["Capacity|Electricity|Wind", "Capacity|Electricity|Solar|PV"],
        )
        self.assertNotIn("Capacity|Electricity|Solar", scope["variables"])
        self.assertEqual(plot_mock.call_count, 2)
        self.assertEqual(
            {tuple(call.args[1]) for call in plot_mock.call_args_list},
            {(10, 11), (30, 31)},
        )

    def test_incompatible_units_are_not_put_on_one_axis(self):
        records = [
            _record("Energy", "World", "Path", "Alpha", 1, unit="EJ/yr"),
            _record("Emissions", "World", "Path", "Alpha", 2, unit="Mt CO2/yr"),
        ]

        with patch("simple_plotter.plt.plot") as plot_mock:
            response = plot_multiple_variables(
                "compare energy and emissions", [], records,
                ["Energy", "Emissions"], region="World", scenario="Path",
            )

        self.assertEqual(plot_mock.call_count, 0)
        self.assertIn("incompatible units", response)
        self.assertIn("Plot each variable or unit separately", response)

    def test_harmless_year_unit_aliases_remain_compatible(self):
        records = [
            _record("Metric", "R1", "Path", "Alpha", 1, unit="EJ/yr"),
            _record("Metric", "R2", "Path", "Alpha", 2, unit="EJ/y"),
        ]

        with (
            patch("simple_plotter.plt.plot") as plot_mock,
            patch("simple_plotter._finalize_plot_layout"),
            patch("simple_plotter.save_plot_to_base64", return_value="![Plot](data:test)"),
        ):
            response = plot_variable_across_regions(
                "compare R1 and R2", [], records, "Metric", ["R1", "R2"], scenario="Path",
            )

        self.assertIn("![Plot]", response)
        self.assertEqual(plot_mock.call_count, 2)

    def test_carbon_price_per_tonne_spellings_are_safe_aliases(self):
        records = [
            _record("Price|Carbon", "World", "Path", model, value, unit=unit)
            for model, value, unit in (
                ("Alpha", 1, "US$2010/t CO2"),
                ("Beta", 2, "US$2010/tCO2"),
                ("Gamma", 3, "US$2010/t"),
            )
        ]

        with (
            patch("simple_plotter.plt.plot") as plot_mock,
            patch("simple_plotter._finalize_plot_layout"),
            patch("simple_plotter.save_plot_to_base64", return_value="![Plot](data:test)"),
        ):
            response = simple_plot_query_with_entities(
                "plot carbon price for World under Path",
                [],
                records,
                {"variable": "Price|Carbon", "region": "World", "scenario": "Path"},
            )
        scope = consume_resolved_scope()

        self.assertIn("![Plot]", response)
        self.assertNotIn("incompatible units", response)
        self.assertEqual(plot_mock.call_count, 3)
        self.assertEqual(scope["displayed_series_count"], 3)
        self.assertEqual(scope["omitted_series"], 0)

    def test_mixed_capacity_units_use_disclosed_dominant_subset(self):
        records = [
            _record("Capacity|Electricity|Solar", "World", "Path", "Alpha", 1, unit="GW"),
            _record("Capacity|Electricity|Solar", "World", "Path", "Beta", 2, unit="GW"),
            _record("Capacity|Electricity|Solar", "World", "Path", "Gamma", 3, unit="GW/yr"),
        ]

        with (
            patch("simple_plotter.plt.plot") as plot_mock,
            patch("simple_plotter._finalize_plot_layout"),
            patch("simple_plotter.save_plot_to_base64", return_value="![Plot](data:test)"),
        ):
            response = simple_plot_query_with_entities(
                "plot solar capacity for World under Path",
                [],
                records,
                {
                    "variable": "Capacity|Electricity|Solar",
                    "region": "World",
                    "scenario": "Path",
                },
            )
        scope = consume_resolved_scope()

        self.assertIn("![Plot]", response)
        self.assertEqual(plot_mock.call_count, 2)
        self.assertIn("using the dominant compatible unit `GW`", response)
        self.assertIn("omitted 1 series", response)
        self.assertEqual(scope["unit"], "GW")
        self.assertEqual(scope["omitted_series"], 1)

    def test_substantial_concrete_gdp_unit_outranks_ambiguous_larger_group(self):
        records = [
            _record(
                "GDP|MER", "IND", "Path", f"Ambiguous {index}", index,
                unit="billion US$2010/yr OR local currency",
            )
            for index in range(25)
        ] + [
            _record(
                "GDP|MER", "India", "Path", f"Concrete {index}", index,
                unit="billion US$2010/yr",
            )
            for index in range(16)
        ]

        with (
            patch("simple_plotter.plt.plot") as plot_mock,
            patch("simple_plotter._finalize_plot_layout"),
            patch("simple_plotter.save_plot_to_base64", return_value="![Plot](data:test)"),
        ):
            response = simple_plot_query_with_entities(
                "chart GDP for India",
                [],
                records,
                {"variable": "GDP|MER", "region": "IND"},
            )
        scope = consume_resolved_scope()

        self.assertIn("![Plot]", response)
        self.assertEqual(plot_mock.call_count, MAX_PLOT_SERIES)
        self.assertIn("dominant compatible unit `billion US$2010/yr`", response)
        self.assertIn("omitted 25 series", response)
        self.assertNotIn("GDP (MER): India vs IND", response)
        self.assertEqual(scope["unit"], "billion US$2010/yr")
        self.assertEqual(scope["region"], "IND")
        self.assertEqual(scope["regions"], ["IND"])

    def test_region_comparison_does_not_drop_a_side_to_reconcile_units(self):
        records = [
            _record("Capacity|Electricity|Solar", "R1", "Path", "Alpha", 1, unit="GW"),
            _record("Capacity|Electricity|Solar", "R2", "Path", "Alpha", 2, unit="GW/yr"),
        ]

        with patch("simple_plotter.plt.plot") as plot_mock:
            response = plot_variable_across_regions(
                "compare solar capacity in R1 and R2",
                [],
                records,
                "Capacity|Electricity|Solar",
                ["R1", "R2"],
                scenario="Path",
            )

        self.assertEqual(plot_mock.call_count, 0)
        self.assertNotIn("![Plot]", response)
        self.assertIn("incompatible units", response)

    def test_missing_variable_comparison_returns_a_valid_common_scope(self):
        records = [
            _record("Metric A", "R1", "Path", "Alpha", 1),
            _record("Metric A", "R2", "Path", "Alpha", 2),
            _record("Metric B", "R2", "Path", "Alpha", 3),
        ]

        response = plot_multiple_variables(
            "compare Metric A and Metric B for R1", [], records,
            ["Metric A", "Metric B"], region="R1", scenario="Path",
        )

        self.assertNotIn("![Plot]", response)
        self.assertIn("complete variable comparison", response)
        self.assertIn("Common scopes available", response)
        self.assertIn("region `R2`", response)
        self.assertIn("scenario `Path`", response)

    def test_single_year_bar_chart_uses_bar_renderer_and_records_type(self):
        records = [
            _record("Price|Carbon", "World", "Path", model, value, unit=unit)
            for model, value, unit in (
                ("Alpha", 1, "US$2010/tCO2"),
                ("Beta", 2, "US$2010/t CO2"),
            )
        ]

        with (
            patch("simple_plotter.plt.bar") as bar_mock,
            patch("simple_plotter.plt.plot") as plot_mock,
            patch("simple_plotter.plt.xticks"),
            patch("simple_plotter._finalize_plot_layout"),
            patch("simple_plotter.save_plot_to_base64", return_value="![Plot](data:test)"),
        ):
            response = simple_plot_query_with_entities(
                "bar chart of carbon price for World in 2050",
                [],
                records,
                {
                    "variable": "Price|Carbon",
                    "region": "World",
                    "start_year": 2050,
                    "end_year": 2050,
                    "chart_type": "bar",
                },
            )
        scope = consume_resolved_scope()

        self.assertEqual(bar_mock.call_count, 2)
        self.assertEqual(plot_mock.call_count, 0)
        self.assertEqual(scope["chart_type"], "bar")
        self.assertEqual(scope["displayed_series_count"], 2)
        self.assertIn("Carbon price", response)
        self.assertIn("(2050)", response)
        self.assertNotIn("2050-2050", response)

    def test_latest_year_resolves_to_actual_latest_year_and_bar(self):
        records = [_record("Metric", "World", "Path", "Alpha", 4)]

        with (
            patch("simple_plotter.plt.bar") as bar_mock,
            patch("simple_plotter.plt.plot") as plot_mock,
            patch("simple_plotter.plt.xticks"),
            patch("simple_plotter._finalize_plot_layout"),
            patch("simple_plotter.save_plot_to_base64", return_value="![Plot](data:test)"),
        ):
            response = simple_plot_query_with_entities(
                "plot Metric for World at the latest available year",
                [],
                records,
                {
                    "variable": "Metric",
                    "region": "World",
                    "start_year": LATEST_YEAR_SENTINEL,
                    "end_year": LATEST_YEAR_SENTINEL,
                },
            )
        scope = consume_resolved_scope()

        self.assertEqual(bar_mock.call_count, 1)
        self.assertEqual(plot_mock.call_count, 0)
        self.assertEqual(bar_mock.call_args.args[1], 5)
        self.assertEqual(scope["start_year"], 2050)
        self.assertEqual(scope["end_year"], 2050)
        self.assertEqual(scope["chart_type"], "bar")
        self.assertIn("(2050)", response)
        self.assertNotIn("(-1)", response)

    def test_scatter_and_area_requests_use_the_matching_renderers(self):
        records = [_record("Metric", "World", "Path", "Alpha", 4)]

        for chart_type, expected_renderer in (("scatter", "scatter"), ("area", "fill_between")):
            with self.subTest(chart_type=chart_type):
                with (
                    patch("simple_plotter.plt.plot") as line_mock,
                    patch("simple_plotter.plt.scatter") as scatter_mock,
                    patch("simple_plotter.plt.fill_between") as area_mock,
                    patch("simple_plotter._finalize_plot_layout"),
                    patch("simple_plotter.save_plot_to_base64", return_value="![Plot](data:test)"),
                ):
                    response = simple_plot_query_with_entities(
                        f"{chart_type} chart of Metric for World",
                        [],
                        records,
                        {
                            "variable": "Metric",
                            "region": "World",
                            "chart_type": chart_type,
                        },
                    )
                scope = consume_resolved_scope()

                selected_mock = scatter_mock if expected_renderer == "scatter" else area_mock
                self.assertEqual(selected_mock.call_count, 1)
                self.assertEqual(line_mock.call_count, 0)
                self.assertIn("![Plot]", response)
                self.assertEqual(scope["chart_type"], chart_type)

    def test_legacy_plot_path_preserves_explicit_scatter_type(self):
        records = [_record("Primary Energy|Coal", "World", "Path", "Alpha", 1)]

        with (
            patch("simple_plotter.plt.plot") as line_mock,
            patch("simple_plotter.plt.scatter") as scatter_mock,
            patch("simple_plotter._finalize_plot_layout"),
            patch("simple_plotter.save_plot_to_base64", return_value="![Plot](data:test)"),
        ):
            response = simple_plot_query(
                "scatter chart of Primary Energy|Coal for World",
                [],
                records,
            )
        scope = consume_resolved_scope()

        self.assertEqual(scatter_mock.call_count, 1)
        self.assertEqual(line_mock.call_count, 0)
        self.assertIn("![Plot]", response)
        self.assertEqual(scope["chart_type"], "scatter")

    def test_rendered_unit_reaches_manager_and_api_provenance(self):
        records = [
            _record("Price|Carbon", "World", "Path", "Alpha", 1, unit="US$2010/tCO2"),
            _record("Price|Carbon", "World", "Path", "Beta", 2, unit="US$2010/t CO2"),
            _record("Price|Carbon", "World", "Path", "Gamma", 3, unit="US$2010/t"),
        ]
        with (
            patch("simple_plotter.plt.plot"),
            patch("simple_plotter._finalize_plot_layout"),
            patch("simple_plotter.save_plot_to_base64", return_value="![Plot](data:test)"),
        ):
            response = simple_plot_query_with_entities(
                "plot carbon price for World in 2050",
                [],
                records,
                {
                    "variable": "Price|Carbon",
                    "region": "World",
                    "start_year": 2050,
                    "end_year": 2050,
                },
            )

        manager = MultiAgentManager.__new__(MultiAgentManager)
        manager.conversation_state = ConversationState()
        manager.last_result_models = []
        manager._persist_last_entities(
            {
                "variable": "Price|Carbon",
                "region": "World",
                "start_year": 2050,
                "end_year": 2050,
                "action": "plot",
            },
            response,
        )
        provenance = fastapi_app._build_data_provenance(
            {"ts": records},
            manager.last_entities,
            response,
            {"agent": "data_plotting", "confidence": 1.0, "source": "test"},
        )

        self.assertEqual(manager.last_entities["unit"], "US$2010/tCO2")
        self.assertEqual(provenance["selected_filters"]["unit"], "US$2010/tCO2")
        self.assertEqual(provenance["matched_record_count"], 3)
        self.assertEqual(provenance["displayed_series_count"], 3)
        self.assertEqual(provenance["chart_type"], "line")

    def test_explicit_model_without_requested_slice_does_not_relax_scope(self):
        records = [
            _record("Metric", "EU", "Path", "Requested Model", 1),
            _record("Metric", "World", "Path", "Available Model", 2),
        ]

        with patch("simple_plotter.plt.plot") as plot_mock:
            response = simple_plot_query_with_entities(
                "plot Metric for World under Path with Requested Model",
                [],
                records,
                {
                    "variable": "Metric",
                    "region": "World",
                    "scenario": "Path",
                    "model": "Requested Model",
                },
            )

        self.assertEqual(plot_mock.call_count, 0)
        self.assertNotIn("![Plot]", response)
        self.assertIn("No data found", response)
        self.assertIn("model `Requested Model`", response)
        self.assertIn("Models with data for the requested slice: `Available Model`", response)
        self.assertIn("Regions available for `Requested Model`: EU", response)

    def test_legacy_plot_path_is_also_capped_without_dropping_silently(self):
        records = [
            _record("Primary Energy|Coal", "World", f"Path {index}", f"Model {index % 3}", index)
            for index in range(12)
        ]

        with (
            patch("simple_plotter.plt.plot") as plot_mock,
            patch("simple_plotter._finalize_plot_layout"),
            patch("simple_plotter.save_plot_to_base64", return_value="![Plot](data:test)"),
        ):
            response = simple_plot_query(
                "plot Primary Energy|Coal for World", [], records,
            )
        scope = consume_resolved_scope()

        self.assertEqual(plot_mock.call_count, MAX_PLOT_SERIES)
        self.assertIn("Showing 8 of 12 available series", response)
        self.assertEqual(scope["omitted_series"], 4)
        self.assertEqual(len(scope["displayed_series"]), MAX_PLOT_SERIES)

    def test_numeric_model_id_is_neutral_in_plot_legend_and_scope(self):
        records = [
            _record("Metric", "World", "Path", "42", 1),
            _record("Metric", "World", "Path", "GCAM", 2),
        ]

        with (
            patch("simple_plotter.plt.plot") as plot_mock,
            patch("simple_plotter._finalize_plot_layout"),
            patch("simple_plotter.save_plot_to_base64", return_value="![Plot](data:test)"),
        ):
            response = simple_plot_query_with_entities(
                "plot Metric for World under Path",
                [],
                records,
                {"variable": "Metric", "region": "World", "scenario": "Path"},
            )
        scope = consume_resolved_scope()

        self.assertEqual(
            {call.kwargs["label"] for call in plot_mock.call_args_list},
            {UNLABELLED_MODEL_LABEL, "GCAM"},
        )
        self.assertEqual(
            scope["result_models"],
            ["GCAM", UNLABELLED_MODEL_LABEL],
        )
        self.assertEqual(scope["displayed_series_count"], 2)
        self.assertFalse(any(label == "42" for label in scope["displayed_series"]))
        self.assertNotRegex(response, r"\b42\b")

    def test_friendly_taxonomy_and_single_year_labels(self):
        self.assertEqual(_pretty_variable_name("Price|Carbon"), "Carbon price")
        self.assertEqual(_pretty_variable_name("GDP|MER"), "GDP (MER)")
        self.assertEqual(
            _pretty_variable_name("Secondary Energy|Electricity|Nuclear"),
            "Nuclear electricity",
        )
        self.assertEqual(_year_range_text(2050, 2050), " (2050)")


if __name__ == "__main__":
    unittest.main()
