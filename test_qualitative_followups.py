"""Regression tests for the second qualitative-pass findings N4-N8.

N4 - sector-filtered model list, N5 - context survives a failed turn,
N6 - no-data recovery suggests an aggregate region, N7 - Population resolves,
N8 - typo tolerance for variable/region tokens.
"""
import unittest

from canonical_aliases import preferred_variable_from_query
from data_metadata import DataMetadata
from data_utils import _aggregate_region_candidates, _looks_like_category_list_request
from query_extractor import QueryEntityExtractor


class N4ModelCoverageTests(unittest.TestCase):
    def _metadata(self):
        return DataMetadata(
            [
                {"variable": "Final Energy|Residential", "region": "World", "scenario": "B", "modelName": "GCAM", "years": {"2030": 1}},
                {"variable": "Emissions|CO2|Buildings", "region": "World", "scenario": "B", "modelName": "REMIND", "years": {"2030": 1}},
                {"variable": "Energy Service|Transportation", "region": "World", "scenario": "B", "modelName": "MESSAGE", "years": {"2030": 1}},
                {"variable": "Emissions|CO2", "region": "World", "scenario": "B", "modelName": "WITCH", "years": {"2030": 1}},
            ],
            models=[{"modelName": n} for n in ["GCAM", "REMIND", "MESSAGE", "WITCH"]],
        )

    def test_detects_sector_model_list_request(self):
        self.assertTrue(_looks_like_category_list_request("which models cover buildings", "models"))
        self.assertTrue(_looks_like_category_list_request("what models cover transport", "models"))

    def test_buildings_subset(self):
        category, models = self._metadata().models_covering_topic("which models cover buildings")
        self.assertEqual(category, "Buildings")
        self.assertEqual(models, ["GCAM", "REMIND"])

    def test_transport_subset(self):
        category, models = self._metadata().models_covering_topic("models covering transport")
        self.assertEqual(category, "Transport")
        self.assertEqual(models, ["MESSAGE"])

    def test_no_topic_returns_none(self):
        category, models = self._metadata().models_covering_topic("which models are available")
        self.assertIsNone(category)
        self.assertEqual(models, [])


class N6RegionRecoveryTests(unittest.TestCase):
    def test_germany_prefers_eu_aggregate(self):
        self.assertEqual(_aggregate_region_candidates("Germany", ["EU", "ARG", "AUS"]), ["EU"])

    def test_no_aggregate_when_absent(self):
        self.assertEqual(_aggregate_region_candidates("Germany", ["ARG", "AUS"]), [])

    def test_non_european_region_has_no_eu_aggregate(self):
        self.assertEqual(_aggregate_region_candidates("China", ["EU", "CHN"]), [])


class N7PopulationTests(unittest.TestCase):
    def test_population_exact(self):
        self.assertEqual(
            preferred_variable_from_query("population projection for World", ["Population", "Emissions|CO2"]),
            "Population",
        )

    def test_population_prefix_fallback(self):
        self.assertEqual(
            preferred_variable_from_query("population for World", ["Population|Total", "GDP|MER"]),
            "Population|Total",
        )

    def test_gdp_still_resolves(self):
        self.assertEqual(
            preferred_variable_from_query("gdp for China", ["GDP|MER", "GDP|PPP"]),
            "GDP|MER",
        )


class N8TypoToleranceTests(unittest.TestCase):
    def _extractor(self):
        ts = [
            {"variable": "Emissions|CO2", "region": "Europe", "scenario": "Baseline", "modelName": "GCAM", "years": {"2030": 1}},
            {"variable": "Population", "region": "World", "scenario": "Baseline", "modelName": "GCAM", "years": {"2030": 1}},
        ]
        return QueryEntityExtractor(models=[{"modelName": "GCAM"}], ts_data=ts, api_key="x")

    def test_misspelled_variable_and_region(self):
        result = self._extractor()._fallback_extraction("emisions for europ")
        self.assertEqual(result["variable"], "Emissions|CO2")
        self.assertEqual(result["region"], "Europe")

    def test_population_typo_query(self):
        result = self._extractor()._fallback_extraction("population projection for World")
        self.assertEqual(result["variable"], "Population")
        self.assertEqual(result["region"], "World")


class N5ContextPersistenceTests(unittest.TestCase):
    """A failed/clarification turn must not wipe the scope from the last success."""

    class _Stub:
        # Borrow the real implementations without constructing the full manager.
        from manager import MultiAgentManager as _M
        _UNSUCCESSFUL_RESPONSE_MARKERS = _M._UNSUCCESSFUL_RESPONSE_MARKERS
        _is_unsuccessful_response = _M._is_unsuccessful_response
        _persist_last_entities = _M._persist_last_entities

        def __init__(self):
            self.last_entities = {}

    def test_failed_turn_preserves_prior_scope(self):
        stub = self._Stub()
        stub.last_entities = {"variable": "Emissions|CO2", "region": "World"}
        # A failed follow-up that carries no usable entities.
        stub._persist_last_entities({}, "I need one more detail to continue.")
        self.assertEqual(stub.last_entities.get("variable"), "Emissions|CO2")
        self.assertEqual(stub.last_entities.get("region"), "World")

    def test_successful_turn_updates_scope(self):
        stub = self._Stub()
        stub.last_entities = {"variable": "Emissions|CO2", "region": "World"}
        stub._persist_last_entities(
            {"variable": "GDP|MER", "region": "China"},
            "### GDP|MER in China\n\nScope: ...",
        )
        self.assertEqual(stub.last_entities.get("variable"), "GDP|MER")
        self.assertEqual(stub.last_entities.get("region"), "China")


if __name__ == "__main__":
    unittest.main()
