"""Regression tests for the query/plot/follow-up defects found in manual QA.

Every case here reproduces a specific numbered problem from
`manual_chatbot_problems_100_el.md` (the 2026-07-20 100-question run) or
`manual_chatbot_check_50_el.md`. Those defects kept reappearing across QA rounds
because nothing pinned them, so each fix below is paired with a guard asserting
the behaviour it was NOT supposed to change.
"""

import unittest

from canonical_aliases import (
    preferred_variable_from_query,
    rank_catalogue_variable_matches,
)
from simple_plotter import (
    MAX_PLOT_SERIES,
    _representative_rows,
    _scenario_priority,
    detect_multi_variable_comparison,
)


# A small stand-in for the runtime catalogue. Only the variables these tests
# reason about are listed; the resolver ranks against whatever it is given.
CATALOGUE = [
    "Benchmarking|Industry|GDP per capita",
    "Capacity Additions|Electricity|Nuclear",
    "Capacity|Electricity|Nuclear",
    "Capacity|Electricity|Solar",
    "Capacity|Electricity|Solar|PV",
    "Emissions|CO2",
    "Emissions|CO2|AFOLU",
    "Exports|Steel",
    "Final Energy",
    "Final Energy|Transportation",
    "GDP|MER",
    "GDP|PPP",
    "Imports|Steel",
    "Land Cover|Built-up Area",
    "Land Cover|Cropland",
    "Land Cover|Cropland|Energy Crops",
    "Population",
    "Price|Carbon",
    "Price|Carbon|EUETS",
    "Price|Electricity|Steel",
    "Price|Final Energy|Commercial|Electricity",
    "Price|Secondary Energy|Electricity",
    "Primary Energy",
    "Primary Energy|Coal",
    "Primary Energy|Gas",
    "Production|Steel",
    "Secondary Energy|Electricity",
    "Secondary Energy|Electricity|Hydro",
    "Secondary Energy|Electricity|Non-Biomass Renewables",
    "Secondary Energy|Electricity|Nuclear",
    "Secondary Energy|Electricity|Solar",
    "Secondary Energy|Electricity|Wind",
]


def resolve(query, ignored=()):
    """Resolve as the pipeline does: alias fast-path, then auto-accepted rank."""
    preferred = preferred_variable_from_query(query, CATALOGUE)
    if preferred:
        return preferred
    ranked = rank_catalogue_variable_matches(query, CATALOGUE, ignored_values=ignored)
    if ranked and ranked[0]["auto_accept"]:
        return ranked[0]["variable"]
    return None


class VariableResolutionRegressions(unittest.TestCase):
    """Each case is a query that previously clarified or resolved wrongly."""

    def test_gdp_per_capita_is_not_swallowed_by_gdp(self):
        # #6: the bare "gdp" alias ignored the "per capita" qualifier.
        self.assertEqual(
            resolve("GDP per capita for Mexico", ["Mexico"]),
            "Benchmarking|Industry|GDP per capita",
        )

    def test_plain_gdp_still_resolves_to_total(self):
        self.assertEqual(resolve("GDP for Mexico", ["Mexico"]), "GDP|MER")

    def test_carrier_electricity_reads_as_generation(self):
        # #13: `Capacity|Electricity|X` outranked the generation variable.
        self.assertEqual(
            resolve("graph solar electricity for the EU", ["EU"]),
            "Secondary Energy|Electricity|Solar",
        )
        self.assertEqual(
            resolve("visualize nuclear electricity for France", ["France"]),
            "Secondary Energy|Electricity|Nuclear",
        )

    def test_explicit_capacity_still_wins_for_capacity_questions(self):
        # The generation aliases must not steal capacity queries.
        self.assertEqual(
            resolve("nuclear capacity for France", ["France"]),
            "Capacity|Electricity|Nuclear",
        )
        self.assertEqual(
            resolve("solar capacity for Spain", ["Spain"]),
            "Capacity|Electricity|Solar",
        )

    def test_electricity_price_prefers_economy_wide_price(self):
        # #19: the steel sector's electricity price was ranked first.
        self.assertEqual(
            resolve("electricity price for Germany", ["Germany"]),
            "Price|Secondary Energy|Electricity",
        )

    def test_steel_production_beats_alphabetical_tie(self):
        # #20: Exports/Imports/Production|Steel scored identically.
        self.assertEqual(
            resolve("steel production for India", ["India"]), "Production|Steel"
        )

    def test_cropland_beats_generic_area_token(self):
        # #21: the common "area" token pulled this to Land Cover|Built-up Area.
        self.assertEqual(
            resolve("cropland area for India", ["India"]), "Land Cover|Cropland"
        )

    def test_generic_electricity_generation_unchanged(self):
        self.assertEqual(
            resolve("electricity generation for the EU", ["EU"]),
            "Secondary Energy|Electricity",
        )

    def test_renewable_electricity_unchanged(self):
        self.assertEqual(
            resolve("renewable electricity for the EU", ["EU"]),
            "Secondary Energy|Electricity|Non-Biomass Renewables",
        )

    def test_round_five_fixes_still_hold(self):
        self.assertEqual(
            resolve("final energy in transport for the World", ["World"]),
            "Final Energy|Transportation",
        )
        self.assertEqual(
            resolve("land use emissions for the World", ["World"]),
            "Emissions|CO2|AFOLU",
        )

    def test_scope_tokens_do_not_block_a_clean_variable(self):
        # #11/#14: the region/year must be excluded from variable evidence.
        for query in (
            "bar chart of carbon price for the World in 2050",
            "carbon price for the World after 2040",
        ):
            with self.subTest(query=query):
                self.assertEqual(resolve(query, ["World"]), "Price|Carbon")

    def test_bare_emissions_defaults_to_co2_after_scope_is_removed(self):
        self.assertEqual(
            resolve(
                "emissions under current policies for EU",
                ["Current Policies", "EU"],
            ),
            "Emissions|CO2",
        )


class RankingConfidenceRegressions(unittest.TestCase):
    """Scoring defects that surfaced obscure variables or over-confident matches."""

    def test_query_coverage_never_exceeds_one(self):
        # A fuzzy match used the catalogue token's rarity weight while dividing
        # by the query token's, so coverage reached 1.154 and floated
        # `Emission Factor|CO2|Tailpipe|2W` above every real emissions variable.
        for query in (
            "emissions", "emission", "populaton", "co2", "carbon price",
            "primary energy", "steel", "final energy", "land use emissions",
        ):
            for item in rank_catalogue_variable_matches(query, CATALOGUE)[:5]:
                with self.subTest(query=query, variable=item["variable"]):
                    self.assertLessEqual(item["query_coverage"], 1.0)

    def test_bare_emissions_prefers_real_emissions_variables(self):
        ranked = rank_catalogue_variable_matches(
            "plot emissions for the World under net zero",
            CATALOGUE,
            ignored_values=["World", "Net Zero"],
        )
        self.assertTrue(ranked[0]["variable"].startswith("Emissions"))

    def test_deep_variable_is_not_auto_accepted_from_few_tokens(self):
        # #22: three matched tokens auto-accepted a seven-segment IDA
        # decomposition whose remaining four segments were never asked for.
        deep_catalogue = CATALOGUE + [
            "IDA|Emissions|CO2|Energy|Supply|Electricity|CO2 intensity",
        ]
        ranked = rank_catalogue_variable_matches(
            "emissions intensity of electricity for the EU",
            deep_catalogue,
            ignored_values=["EU"],
        )
        self.assertFalse(ranked[0]["auto_accept"])

    def test_shallow_well_covered_variable_still_auto_accepts(self):
        # Guard: tightening the gate must not stop ordinary matches resolving.
        ranked = rank_catalogue_variable_matches("carbon price", CATALOGUE)
        self.assertEqual(ranked[0]["variable"], "Price|Carbon")
        self.assertTrue(ranked[0]["auto_accept"])

    def test_comparison_and_availability_grammar_is_not_unmatched_taxonomy(self):
        ranked = rank_catalogue_variable_matches(
            "does WITCH report carbon price for EU together",
            CATALOGUE,
            ignored_values=["WITCH", "EU"],
        )

        self.assertEqual(ranked[0]["variable"], "Price|Carbon")
        self.assertNotIn("does", ranked[0]["unmatched_terms"])
        self.assertNotIn("report", ranked[0]["unmatched_terms"])
        self.assertNotIn("together", ranked[0]["unmatched_terms"])


class ShareQueryRegressions(unittest.TestCase):
    def test_carrier_share_resolves_to_the_carrier_quantity(self):
        # #29: "share" matched `PV Investment Share` while "coal" matched nothing.
        self.assertEqual(resolve("coal share for GCAM", ["GCAM"]), "Primary Energy|Coal")
        self.assertEqual(resolve("its coal share"), "Primary Energy|Coal")
        self.assertEqual(resolve("share of gas for the EU", ["EU"]), "Primary Energy|Gas")


class MultiVariableDetectionRegressions(unittest.TestCase):
    def test_conjunction_keeps_the_family_noun_with_each_carrier(self):
        # #12: "coal and gas primary energy" produced a single-variable plot.
        # Returning bare carriers was not enough -- "coal" alone resolves to
        # `Price|Coal`, so the family noun has to travel with each carrier for
        # the alias catalogue to reach `Primary Energy|Coal`.
        self.assertEqual(
            detect_multi_variable_comparison("plot coal and gas primary energy for the EU"),
            ["coal primary energy", "gas primary energy"],
        )
        self.assertEqual(
            detect_multi_variable_comparison("oil and gas production for the World"),
            ["oil production", "gas production"],
        )

    def test_conjunction_without_a_family_noun_returns_bare_terms(self):
        self.assertEqual(
            detect_multi_variable_comparison("compare solar and wind"), ["solar", "wind"]
        )

    def test_region_conjunction_is_not_a_variable_comparison(self):
        # Guard: two regions must not be mistaken for two variables.
        self.assertEqual(
            detect_multi_variable_comparison("compare CO2 for China and India"), []
        )
        self.assertEqual(
            detect_multi_variable_comparison("plot CO2 for Germany and France"), []
        )

    def test_single_variable_query_detects_nothing(self):
        self.assertEqual(detect_multi_variable_comparison("plot emissions for the World"), [])


class PlotSeriesTrimmingTests(unittest.TestCase):
    """A chart carrying every model/scenario pair is unreadable (38-59 series)."""

    @staticmethod
    def _frame(rows):
        pandas = __import__("pandas")
        return pandas.DataFrame(rows)

    def test_small_frame_is_untouched(self):
        frame = self._frame([
            {"scenario": "Baseline", "model": "gcam"},
            {"scenario": "NDC", "model": "muse"},
        ])
        trimmed, omitted = _representative_rows(frame)
        self.assertEqual(omitted, 0)
        self.assertEqual(len(trimmed), 2)

    def test_large_frame_is_capped_and_reports_the_remainder(self):
        rows = [
            {"scenario": f"Scenario{index}", "model": f"model{index % 4}"}
            for index in range(38)
        ]
        trimmed, omitted = _representative_rows(self._frame(rows))
        self.assertEqual(len(trimmed), MAX_PLOT_SERIES)
        self.assertEqual(omitted, 38 - MAX_PLOT_SERIES)

    def test_trimmed_chart_spans_scenarios_not_one_family(self):
        # Selecting purely by scenario priority produced eight baselines, which
        # answers nothing: the policy scenarios are the point of the comparison.
        rows = []
        for model in ("gcam", "muse", "e3me", "gemini_e3"):
            for scenario in ("Baseline", "PR_CurPol_CP", "NDC_EI", "NZE_Bench"):
                rows.append({"scenario": scenario, "model": model})
        trimmed, _ = _representative_rows(self._frame(rows))
        self.assertGreaterEqual(len(set(trimmed["scenario"])), 3)

    def test_priority_scenarios_are_preferred_over_obscure_ones(self):
        self.assertLess(_scenario_priority("Baseline"), _scenario_priority("Obscure_Run_7"))
        self.assertLess(_scenario_priority("PR_CurPol_CP"), _scenario_priority("Obscure_Run_7"))


class FollowupCompositionRegressions(unittest.TestCase):
    """Follow-ups that silently kept the wrong scope."""

    class _Stub:
        from manager import MultiAgentManager as _M

        _FOLLOWUP_FILLER = _M._FOLLOWUP_FILLER
        _VARIABLE_SEGMENT_SWITCH_TOKENS = _M._VARIABLE_SEGMENT_SWITCH_TOKENS
        _is_generic_followup = _M._is_generic_followup
        _is_contextual_dimension_followup = _M._is_contextual_dimension_followup
        _compose_contextual_query = _M._compose_contextual_query
        _from_switch_model = _M._from_switch_model
        _carrier_switch_variable = _M._carrier_switch_variable

        class _Extractor:
            available_variables = CATALOGUE
            available_models: list = []

        entity_extractor = _Extractor()

        def _model_switch_names(self):
            return []

        def _match_scenario_from_text(self, _text):
            return None

        def _resolve_region_from_text(self, text):
            return "China" if str(text).strip().lower() == "china" else ""

        def _resolve_carry_dimension(self, token):
            return ("region", "China") if str(token).lower() == "china" else None

    def test_same_for_carrier_switches_the_variable_not_the_region(self):
        # #15: "same for wind" kept plotting solar and read "wind" as a region,
        # which is what produced the clarification loop.
        composed = self._Stub()._compose_contextual_query(
            "same for wind",
            {"variable": "Secondary Energy|Electricity|Solar", "region": "India"},
        )
        self.assertIn("Secondary Energy|Electricity|Wind", composed)
        self.assertIn("for India", composed)
        self.assertNotIn("Solar", composed)

    def test_same_for_region_still_switches_the_region(self):
        composed = self._Stub()._compose_contextual_query(
            "same for China", {"variable": "Emissions|CO2", "region": "World"}
        )
        self.assertIn("for China", composed)
        self.assertIn("Emissions|CO2", composed)

    def test_plot_the_comparison_keeps_both_scenarios(self):
        # #16: the composer only read the singular `scenario`, so the second
        # side of the comparison was dropped.
        composed = self._Stub()._compose_contextual_query(
            "plot the comparison",
            {
                "variable": "Emissions|CO2",
                "region": "EU",
                "scenario": "Current Policies",
                "scenarios": ["Current Policies", "Baseline"],
                "comparison": "scenario",
            },
        )
        self.assertIn("Current Policies", composed)
        self.assertIn("Baseline", composed)

    def test_plot_it_still_preserves_the_full_scope(self):
        composed = self._Stub()._compose_contextual_query(
            "plot it",
            {
                "variable": "Emissions|CO2",
                "region": "World",
                "scenario": "Baseline",
                "start_year": 2030,
                "end_year": 2030,
            },
        )
        self.assertIn("Emissions|CO2", composed)
        self.assertIn("for World", composed)
        self.assertIn("2030", composed)


class CountryNameLookupRegressions(unittest.TestCase):
    """Country names absent from the YAML definitions never reached pycountry."""

    def test_country_phrases_are_extracted_from_a_sentence(self):
        from utils_query import _country_name_candidates

        phrases = _country_name_candidates("population for Vietnam")
        self.assertIn("Vietnam", phrases)
        # Question wording must not be offered to a fuzzy country lookup.
        self.assertNotIn("for", phrases)
        self.assertNotIn("population", phrases)

    def test_multiword_country_is_tried_before_its_parts(self):
        from utils_query import _country_name_candidates

        phrases = _country_name_candidates("final energy for South Korea")
        self.assertLess(phrases.index("South Korea"), phrases.index("Korea"))

    def test_implausible_fuzzy_hits_are_rejected(self):
        # `search_fuzzy("Middle")` returns United Kingdom; an invented region
        # must not be answered with real data.
        import pycountry

        from utils_query import _country_match_is_plausible

        united_kingdom = pycountry.countries.get(alpha_3="GBR")
        self.assertFalse(_country_match_is_plausible("Middle", united_kingdom))
        self.assertFalse(_country_match_is_plausible("Atlantis", united_kingdom))

    def test_real_country_names_are_accepted(self):
        import pycountry

        from utils_query import _country_match_is_plausible

        for phrase, code in (("Vietnam", "VNM"), ("Kenya", "KEN"),
                             ("Portugal", "PRT"), ("South Korea", "KOR")):
            with self.subTest(phrase=phrase):
                country = pycountry.countries.get(alpha_3=code)
                self.assertTrue(_country_match_is_plausible(phrase, country))


class OpenEndedYearRegressions(unittest.TestCase):
    """Open-ended year phrases were collapsed to a single year."""

    def test_extract_year_range_keeps_bounds_open(self):
        from year_filters import extract_year_range

        self.assertEqual(extract_year_range("population until 2080"), (None, 2080))
        self.assertEqual(extract_year_range("carbon price after 2040"), (2041, None))
        self.assertEqual(extract_year_range("emissions before 2050"), (None, 2050))

    def test_select_years_treats_none_as_unbounded(self):
        from year_filters import select_years

        years = [str(y) for y in range(2005, 2105, 5)]
        until = select_years(years, None, 2080)
        after = select_years(years, 2041, None)
        # "until 2080" must not become the single year 2080.
        self.assertIn("2005", until)
        self.assertIn("2080", until)
        self.assertNotIn("2085", until)
        self.assertIn("2045", after)
        self.assertNotIn("2040", after)

    def test_open_ended_phrase_detection(self):
        from query_extractor import QueryEntityExtractor

        pattern = QueryEntityExtractor._OPEN_ENDED_YEAR_PHRASE
        for query in ("population until 2080", "carbon price after 2040",
                      "emissions before 2050", "energy from 2020"):
            with self.subTest(query=query):
                self.assertTrue(pattern.search(query))
        # A single year is not open-ended and must keep its exact-year handling.
        self.assertFalse(pattern.search("emissions in 2050"))
        self.assertFalse(pattern.search("CO2 between 2030 and 2060"))


class FollowupGateRegressions(unittest.TestCase):
    """`plot its X` matched no follow-up rule, so carried scope was dropped."""

    class _Stub:
        from manager import MultiAgentManager as _M

        _FOLLOWUP_FILLER = _M._FOLLOWUP_FILLER
        _is_contextual_dimension_followup = _M._is_contextual_dimension_followup
        _is_generic_followup = _M._is_generic_followup
        _from_switch_model = _M._from_switch_model

        def _model_switch_names(self):
            return []

        def _match_scenario_from_text(self, _text):
            return None

        def _resolve_carry_dimension(self, _token):
            return None

    def test_plot_possessive_is_a_dimension_followup(self):
        stub = self._Stub()
        for query in ("plot its CO2 emissions", "graph its GDP", "visualize their emissions"):
            with self.subTest(query=query):
                self.assertTrue(stub._is_contextual_dimension_followup(query))

    def test_plain_possessive_still_matches(self):
        stub = self._Stub()
        self.assertTrue(stub._is_contextual_dimension_followup("its population"))

    def test_carrier_token_is_not_treated_as_a_region(self):
        # The region resolver substring-matches codes inside words ("wind" ->
        # IND, "coal" -> COL), which used to abort every carrier switch.
        from utils_query import extract_region_from_query

        self.assertEqual(extract_region_from_query("wind capacity", {}, ["IND", "COL"]), "")


class LinkScoringRegressions(unittest.TestCase):
    """Function words let long unrelated titles out-score the named entry."""

    CATALOG = [
        {
            "title": "GCAM",
            "category": "models",
            "url": "https://iamparis.eu/models",
            "keywords": ["GCAM", "The Global Change Assessment Model"],
        },
        {
            "title": "Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU",
            "category": "results",
            "url": "https://iamparis.eu/results/iam-compact/fit-for-55",
            "keywords": ["Fit-for-55"],
        },
    ]

    def test_stopwords_do_not_inflate_long_titles(self):
        from link_router import (
            _category_boosts,
            _normalize,
            _query_terms,
            _score_item,
            _tokens,
            _LINK_SCORING_STOPWORDS,
        )

        query = "link me to the model documentation for GCAM"
        query_text = _normalize(_query_terms(query, {}, ""))
        tokens = _tokens(query_text) - _LINK_SCORING_STOPWORDS
        boosts = _category_boosts(query, "", {})
        scores = {
            item["title"]: _score_item(item, query_text, tokens, boosts)[0]
            for item in self.CATALOG
        }
        self.assertGreater(
            scores["GCAM"],
            scores["Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU"],
        )

    def test_stopword_set_excludes_only_function_words(self):
        from link_router import _LINK_SCORING_STOPWORDS

        # Topical words must never be filtered out of link scoring.
        for topical in ("model", "models", "scenario", "data", "results", "gcam",
                        "transport", "emissions", "page", "documentation"):
            with self.subTest(word=topical):
                self.assertNotIn(topical, _LINK_SCORING_STOPWORDS)


if __name__ == "__main__":
    unittest.main()
