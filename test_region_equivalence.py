import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt

import fastapi_app
from canonical_aliases import (
    canonical_region_from_query,
    dedupe_equivalent_regions,
    region_family_members,
    regions_equivalent,
)
from data_metadata import DataMetadata
from data_utils import _list_model_category, data_query
from resolved_scope import consume_resolved_scope
from simple_plotter import simple_plot_query_with_entities
from utils_query import extract_region_from_query


def _record(region, variable="Metric", scenario="Path", value=42, model="Model"):
    return {
        "variable": variable,
        "region": region,
        "scenario": scenario,
        "modelName": model,
        "unit": "EJ/yr",
        "years": {"2050": value},
    }


class CountryRegionEquivalenceTests(unittest.TestCase):
    def tearDown(self):
        consume_resolved_scope()
        plt.close("all")

    def test_country_alias_families_are_symmetric_and_aggregates_stay_distinct(self):
        for left, right in (
            ("DEU", "DE"),
            ("FRA", "FR"),
            ("IND", "India"),
            ("GREECE", "GR"),
            ("GREECE", "GRC"),
        ):
            with self.subTest(left=left, right=right):
                self.assertTrue(regions_equivalent(left, right))
                self.assertTrue(regions_equivalent(right, left))

        self.assertFalse(regions_equivalent("EU", "DE"))
        self.assertFalse(regions_equivalent("Europe", "France"))
        self.assertFalse(regions_equivalent("World", "India"))
        self.assertEqual(region_family_members("DEU", ["EU", "DE", "FR"]), ["DE"])
        self.assertEqual(canonical_region_from_query("data for Germany", ["EU", "DE"]), "DE")
        self.assertEqual(canonical_region_from_query("data for France", ["EU", "FR"]), "FR")

    def test_runtime_extractor_selects_available_country_alias(self):
        cases = (
            ("Germany", "DE"),
            ("DEU", "DE"),
            ("France", "FR"),
            ("India", "IND"),
            ("Greece", "GR"),
            ("GRC", "GREECE"),
        )
        for requested, stored in cases:
            with self.subTest(requested=requested, stored=stored):
                self.assertEqual(
                    extract_region_from_query(
                        f"show Metric for {requested}", {}, ["EU", stored]
                    ),
                    stored,
                )

    def test_dotted_region_initialisms_match_exact_runtime_codes(self):
        self.assertEqual(
            extract_region_from_query("GHG emissions for the E.U.", {}, ["EU", "USA"]),
            "EU",
        )
        self.assertEqual(
            extract_region_from_query("population in the U.S.A.", {}, ["EU", "USA"]),
            "USA",
        )

    def test_table_filter_accepts_country_name_and_code_variants(self):
        cases = (
            ("DEU", "DE"),
            ("FRA", "FR"),
            ("India", "IND"),
            ("GREECE", "GR"),
            ("GREECE", "GRC"),
        )
        for requested, stored in cases:
            with self.subTest(requested=requested, stored=stored):
                records = [_record(stored)]
                response = data_query(
                    f"show Metric for {requested} under Path in 2050",
                    [],
                    records,
                    forced_entities={
                        "variable": "Metric",
                        "region": requested,
                        "scenario": "Path",
                        "start_year": 2050,
                        "end_year": 2050,
                    },
                    metadata=DataMetadata(records),
                )
                self.assertNotIn("No data", response)
                self.assertIn("42.00", response)

    def test_greece_availability_unions_all_equivalent_runtime_labels(self):
        records = [
            _record("GREECE", "Metric A", "Path A", 1, "Model A"),
            _record("GR", "Metric B", "Path B", 2, "Model B"),
            _record("GRC", "Metric C", "Path C", 3, "Model C"),
        ]

        available = DataMetadata(records).get_available_for_region("Greece")

        self.assertEqual(available["variables"], ["Metric A", "Metric B", "Metric C"])
        self.assertEqual(available["scenarios"], ["Path A", "Path B", "Path C"])
        self.assertEqual(available["models"], ["Model A", "Model B", "Model C"])

    def test_plot_filter_and_provenance_accept_country_alias(self):
        records = [_record("DE")]
        with (
            patch("simple_plotter.plt.plot") as plot_mock,
            patch("simple_plotter._finalize_plot_layout"),
            patch("simple_plotter.save_plot_to_base64", return_value="![Plot](data:test)"),
        ):
            response = simple_plot_query_with_entities(
                "plot Metric for Germany under Path",
                [],
                records,
                {"variable": "Metric", "region": "DEU", "scenario": "Path"},
            )
        scope = consume_resolved_scope()

        self.assertIn("![Plot]", response)
        self.assertEqual(plot_mock.call_count, 1)
        self.assertEqual(scope["region"], "DEU")
        self.assertEqual(scope["regions"], ["DEU"])
        self.assertEqual(
            fastapi_app._count_matching_records(
                {"ts": records}, {"variable": "Metric", "region": "DEU"}
            ),
            1,
        )
        self.assertEqual(
            fastapi_app._count_matching_records(
                {"ts": records + [_record("FR")]},
                {"variable": "Metric", "regions": ["DEU", "FRA"]},
            ),
            2,
        )

    def test_single_region_plot_collapses_equivalent_raw_labels_to_requested_scope(self):
        cases = (
            ("World", "World", "WORLD"),
            ("IND", "India", "IND"),
            ("CHN", "China", "CHN"),
            ("JPN", "Japan", "JPN"),
            ("FRA", "FR", "FRA"),
        )
        for requested, first_label, second_label in cases:
            with self.subTest(requested=requested):
                records = [
                    _record(first_label, value=1, model="Alpha"),
                    _record(second_label, value=2, model="Beta"),
                ]
                with (
                    patch("simple_plotter.plt.plot") as plot_mock,
                    patch("simple_plotter._finalize_plot_layout"),
                    patch(
                        "simple_plotter.save_plot_to_base64",
                        return_value="![Plot](data:test)",
                    ),
                ):
                    response = simple_plot_query_with_entities(
                        f"plot Metric for {requested}",
                        [],
                        records,
                        {"variable": "Metric", "region": requested},
                    )
                scope = consume_resolved_scope()

                self.assertIn("![Plot]", response)
                self.assertNotIn(" vs ", response)
                self.assertEqual(plot_mock.call_count, 2)
                self.assertEqual(scope["region"], requested)
                self.assertEqual(scope["regions"], [requested])
                self.assertIsNone(scope.get("comparison_dimension"))
                self.assertEqual(scope["displayed_series"], ["Alpha", "Beta"])
                self.assertEqual(
                    fastapi_app._count_matching_records(
                        {"ts": records},
                        {"variable": "Metric", "region": requested},
                    ),
                    2,
                )

    def test_one_structured_variable_still_recovers_query_comparison(self):
        records = [
            _record("EU", "Primary Energy|Coal", value=1),
            _record("EU", "Primary Energy|Gas", value=2),
        ]
        with (
            patch("simple_plotter.plt.plot") as plot_mock,
            patch("simple_plotter._finalize_plot_layout"),
            patch("simple_plotter.save_plot_to_base64", return_value="![Plot](data:test)"),
        ):
            response = simple_plot_query_with_entities(
                "plot coal and gas primary energy for EU",
                [],
                records,
                {
                    "variable": "Primary Energy|Gas",
                    "variables": ["Primary Energy|Gas"],
                    "region": "EU",
                    "scenario": "Path",
                },
            )
        scope = consume_resolved_scope()

        self.assertIn("![Plot]", response)
        self.assertEqual(plot_mock.call_count, 2)
        self.assertEqual(
            scope["variables"], ["Primary Energy|Coal", "Primary Energy|Gas"]
        )
        self.assertEqual(scope["comparison_dimension"], "variable")

    def test_region_lists_dedupe_name_and_iso_variants(self):
        labels = ["ARG", "Argentina", "BRA", "Brazil", "CAN", "Canada", "EU"]
        self.assertEqual(
            dedupe_equivalent_regions(labels),
            ["ARG", "BRA", "CAN", "EU"],
        )
        records = [_record(label, model="GCAM") for label in labels]
        response = _list_model_category(
            "regions", "GCAM", records, show_all=True, model_members=["GCAM"]
        )
        self.assertIn("ARG, BRA, CAN, EU", response)
        self.assertNotIn("Argentina", response)
        self.assertNotIn("Brazil", response)
        self.assertNotIn("Canada", response)

        global_response = data_query("list all regions", [], records)
        self.assertIn("ARG, BRA, CAN and EU", global_response)
        self.assertNotIn("Argentina", global_response)
        self.assertNotIn("Brazil", global_response)
        self.assertNotIn("Canada", global_response)


if __name__ == "__main__":
    unittest.main()
