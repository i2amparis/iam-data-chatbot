from collections import Counter
from difflib import SequenceMatcher
from functools import lru_cache
import math
import re
from typing import Iterable


VARIABLE_ALIASES: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    # "GDP per capita" must be matched before the bare "gdp" alias below, which
    # would otherwise swallow the "per capita" qualifier and answer with total GDP.
    (("gdp per capita", "gross domestic product per capita", "gdp/capita",
      "income per capita", "gdp per head"),
     ("Benchmarking|Industry|GDP per capita",)),
    (("gross domestic product", "gdp"), ("GDP|MER", "GDP|PPP")),
    # "Kyoto Gases" is the IPCC GHG basket; it is the canonical fallback when a
    # dataset does not report an "Emissions|GHG" aggregate itself.
    (("greenhouse gas", "greenhouse gases", "ghg"), ("Emissions|GHG", "Emissions|Kyoto Gases")),
    (("kyoto gas", "kyoto gases", "kyoto-gas", "kyoto-gases"),
     ("Emissions|Kyoto Gases",)),
    (("methane", "methan", "ch4"),
     ("Emissions|CH4", "Emissions|Kyoto Gases|CH4")),
    (("nitrous oxide", "n2o"), ("Emissions|N2O", "Emissions|Kyoto Gases|N2O")),
    (("solar pv capacity", "photovoltaic capacity", "pv capacity"), ("Capacity|Electricity|Solar|PV", "Capacity|Electricity|Solar")),
    (("solar capacity",), ("Capacity|Electricity|Solar",)),
    (("installed solar power capacity", "installed solar capacity"),
     ("Capacity|Electricity|Solar", "Capacity|Electricity|Solar|PV")),
    (("wind capacity", "wind power capacity", "installed wind"), ("Capacity|Electricity|Wind",)),
    (("solar pv generation", "solar photovoltaic generation", "electricity generation from solar pv",
      "solar pv electricity", "photovoltaic electricity"),
     ("Secondary Energy|Electricity|Solar|PV", "Secondary Energy|Electricity|Solar")),
    # "renewable electricity" carries the "renewable" qualifier, so it must map
    # to the renewables aggregate rather than to all electricity. Placed before
    # the generic "electricity generation" alias so both "renewable electricity"
    # and "renewable electricity generation" resolve consistently.
    (("renewable electricity", "renewables electricity", "renewable electricity generation",
      "renewable power generation", "renewable power", "green electricity"),
     ("Secondary Energy|Electricity|Non-Biomass Renewables",)),
    # "<carrier> electricity" reads as generation, not installed capacity, but
    # `Capacity|Electricity|<carrier>` outranks `Secondary Energy|Electricity|
    # <carrier>` on token coverage (three segments against four). These keep the
    # generation reading; the explicit "<carrier> capacity" aliases above still
    # win for capacity questions.
    (("solar electricity", "solar power generation", "solar generation"),
     ("Secondary Energy|Electricity|Solar",)),
    (("wind electricity", "wind power generation", "wind generation"),
     ("Secondary Energy|Electricity|Wind",)),
    (("onshore wind electricity", "onshore wind generation",
      "electricity generated from onshore wind"),
     ("Secondary Energy|Electricity|Wind|Onshore", "Secondary Energy|Electricity|Wind")),
    (("nuclear electricity", "nuclear power generation", "nuclear generation"),
     ("Secondary Energy|Electricity|Nuclear",)),
    (("hydro electricity", "hydropower electricity", "hydropower generation",
      "hydro generation", "hydroelectricity"),
     ("Secondary Energy|Electricity|Hydro",)),
    (("nuclear capacity", "installed nuclear"), ("Capacity|Electricity|Nuclear",)),
    (("electricity generation", "power generation"), ("Secondary Energy|Electricity",)),
    # Without this the ranker prefers `Price|Electricity|Steel` -- the steel
    # sector's electricity price -- over the economy-wide electricity price.
    (("electricity price", "electricity prices", "price of electricity"),
     ("Price|Secondary Energy|Electricity",)),
    # `Exports|Steel`, `Imports|Steel` and `Production|Steel` score identically
    # on "steel production", so the tie broke alphabetically to Exports.
    (("steel production", "production of steel", "steel output"),
     ("Production|Steel",)),
    (("cement production", "production of cement", "cement output"),
     ("Production|Cement",)),
    # "area" is a common catalogue token and pulled "cropland area" to
    # `Land Cover|Built-up Area`; the distinctive token is "cropland".
    (("cropland area", "cropland", "crop land"), ("Land Cover|Cropland",)),
    (("forest area", "forest land area"), ("Land Cover|Forest",)),
    (("primary energy from coal", "coal primary energy"), ("Primary Energy|Coal",)),
    # "<carrier> share" asks for that carrier's slice of the energy mix. The
    # catalogue has no share variable for it (the only "* Share" entries are
    # investment shares), so resolve to the carrier quantity the share is taken
    # of. Without these, "share" matched `PV Investment Share` and friends while
    # the carrier token matched nothing at all.
    (("coal share", "share of coal"), ("Primary Energy|Coal",)),
    (("gas share", "share of gas"), ("Primary Energy|Gas",)),
    (("oil share", "share of oil"), ("Primary Energy|Oil",)),
    (("nuclear share", "share of nuclear"), ("Primary Energy|Nuclear",)),
    (("biomass share", "share of biomass"), ("Primary Energy|Biomass",)),
    (("renewables share", "renewable share", "share of renewables"),
     ("Primary Energy|Non-Biomass Renewables",)),
    # A sector-level industry request must prefer the aggregate industry series.
    # Otherwise catalogue token scoring tends to pick an arbitrary descendant
    # such as steel, even when the aggregate is available for the requested scope.
    (("industry emissions", "industrial emissions", "emissions from industry",
      "emissions in industry"),
     ("Emissions|CO2|Energy|Demand|Industry", "Emissions|Kyoto Gases|Industry")),
    # Land-use / AFOLU emissions (predominantly CO2); fall back to the GHG basket
    # for that sector when a dataset does not report the CO2 slice.
    (("land use emissions", "land-use emissions", "land use co2 emissions",
      "land-use co2 emissions", "co2 emissions from land use", "afolu emissions",
      "emissions from land use", "land use change emissions"),
     ("Emissions|CO2|AFOLU", "Emissions|Kyoto Gases|AFOLU")),
    (("power sector co2 emissions", "power-sector co2 emissions",
      "electricity sector co2 emissions", "co2 emissions from power generation"),
     ("Emissions|CO2|Energy|Supply|Electricity",)),
    # Keep the generic CO2 alias after the sector-specific phrases above.
    # Otherwise its shorter phrase (``CO2 emissions``) wins first and silently
    # erases an explicitly requested AFOLU or power-sector scope.
    (("carbon dioxide emissions", "carbon dioxide release", "co2 emissions",
      "carbon emissions", "co2"),
     ("Emissions|CO2",)),
    (("industrial final energy demand", "industry final energy demand",
      "industrial energy demand", "industry energy demand",
      "industrial final energy", "final energy in industry"),
     ("Final Energy|Industry",)),
    (("residential and commercial final energy", "residential commercial final energy",
      "buildings final energy"),
     ("Final Energy|Residential & Commercial", "Final Energy|Residential and Commercial")),
    (("carbon capture and storage", "carbon capture storage", "ccs deployment"),
     ("Carbon Capture", "Carbon Sequestration|CCS")),
    (("carbon removal from biomass with ccs", "biomass carbon removal",
      "biomass with ccs", "beccs"),
     ("Carbon Sequestration|CCS|Biomass",)),
    (("carbon price", "carbon prices", "price of carbon"), ("Price|Carbon",)),
    (("oil demand", "demand for oil"),
     ("Final Energy|Liquids", "Final Energy|Liquids|Fossil")),
    # In ordinary climate-data questions a bare "emissions" request means CO2
    # unless the user names another gas or the GHG basket.  Keep this after all
    # gas/sector-specific aliases so those more precise readings still win.
    (("emissions", "emission", "emisions", "emisson"),
     ("Emissions|CO2", "Emissions|Kyoto Gases")),
    (("population", "population projection", "population projections", "number of people"), ("Population",)),
    # Energy families: a bare "final/primary/secondary energy" request should map to
    # the base variable, not an over-specific sub-carrier (e.g. "Final Energy|Geothermal").
    (("final energy demand", "final energy"), ("Final Energy",)),
    (("primary energy demand", "primary energy", "primary enegy"), ("Primary Energy",)),
    (("secondary energy",), ("Secondary Energy",)),
)


SCENARIO_ALIASES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("current policy", "current policies", "curpol"), "Current Policies"),
    (("baseline", "business as usual", "bau"), "Baseline"),
    (("policy", "policy scenario"), "Policy"),
    (("net zero", "net-zero", "nze"), "Net Zero"),
    (("nationally determined contribution", "nationally determined contributions", "ndc", "ndcs"), "NDC"),
)


REGION_ALIASES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("world", "global", "globally"), "World"),
    (("europe", "european union", "eu27", "eu"), "EU"),
    (("germany",), "Germany"),
    (("france", "french"), "France"),
    (("greece", "hellas"), "Greece"),
    (("china", "chn"), "China"),
    (("india", "ind", "indian"), "India"),
    (("united states", "usa", "us"), "United States"),
)


# Runtime IAM datasets are not consistent about country labels: the same
# country can be stored as a name, ISO-3 code, ISO-2 code, or (for Greece in
# one source) an upper-case name.  Keep this as an explicit country-only
# allowlist.  In particular, aggregate regions such as EU/Europe and
# World/Global must never become equivalent through this helper.
COUNTRY_REGION_EQUIVALENCE: tuple[tuple[str, ...], ...] = (
    ("Argentina", "ARG", "AR"),
    ("Australia", "AUS", "AU"),
    ("Brazil", "BRA", "BR"),
    ("Canada", "CAN", "CA"),
    ("China", "CHN", "CN"),
    ("Germany", "DEU", "DE"),
    ("Egypt", "EGY", "EG"),
    ("Spain", "ESP", "ES"),
    ("France", "FRA", "FR"),
    ("United Kingdom", "GBR", "GB", "UK"),
    ("Greece", "GREECE", "GRC", "GR", "Hellas"),
    ("Indonesia", "IDN", "ID"),
    ("India", "IND", "IN"),
    ("Iran", "IRN", "IR"),
    ("Iraq", "IRQ", "IQ"),
    ("Italy", "ITA", "IT"),
    ("Japan", "JPN", "JP"),
    ("South Korea", "KOR", "KR"),
    ("Mexico", "MEX", "MX"),
    ("Nigeria", "NGA", "NG"),
    ("Netherlands", "NLD", "NL"),
    ("Norway", "NOR", "NO"),
    ("Pakistan", "PAK", "PK"),
    ("Poland", "POL", "PL"),
    ("Russian Federation", "Russia", "RUS", "RU"),
    ("Saudi Arabia", "SAU", "SA"),
    ("Sweden", "SWE", "SE"),
    ("Switzerland", "CHE", "CH"),
    ("Thailand", "THA", "TH"),
    ("Turkey", "TUR", "TR"),
    ("Ukraine", "UKR", "UA"),
    ("United States", "USA", "US"),
    ("Viet Nam", "Vietnam", "VNM", "VN"),
    ("South Africa", "ZAF", "ZA"),
)


def _region_equivalence_key(value: object) -> str:
    """Normalize only case and whitespace for conservative label matching."""
    return " ".join(str(value or "").strip().casefold().split())


_COUNTRY_REGION_FAMILY_BY_KEY: dict[str, frozenset[str]] = {}
for _family in COUNTRY_REGION_EQUIVALENCE:
    _normalized_family = frozenset(_region_equivalence_key(value) for value in _family)
    for _member_key in _normalized_family:
        _COUNTRY_REGION_FAMILY_BY_KEY[_member_key] = _normalized_family


@lru_cache(maxsize=512)
def _iso_country_identity(label_key: str) -> str | None:
    """Resolve an exact ISO code/name when optional ``pycountry`` is present."""
    if not label_key:
        return None
    try:
        import pycountry  # type: ignore

        country = pycountry.countries.lookup(label_key)
    except Exception:
        return None
    identity = str(getattr(country, "alpha_3", "") or "").strip().casefold()
    return identity or None


def regions_equivalent(left: object, right: object) -> bool:
    """Return whether two runtime labels denote the same explicit country.

    Exact labels remain case-insensitive for every kind of region. Cross-label
    matching is limited to :data:`COUNTRY_REGION_EQUIVALENCE`, so an aggregate
    cannot accidentally match one of its member countries.
    """
    left_key = _region_equivalence_key(left)
    right_key = _region_equivalence_key(right)
    if not left_key or not right_key:
        return False
    if left_key == right_key:
        return True
    family = _COUNTRY_REGION_FAMILY_BY_KEY.get(left_key)
    if family and right_key in family:
        return True
    left_country = _iso_country_identity(left_key)
    return bool(left_country and left_country == _iso_country_identity(right_key))


def region_family_members(requested: object, available_regions: Iterable[object]) -> list[str]:
    """Return available runtime labels equivalent to one requested country."""
    matches: list[str] = []
    seen: set[str] = set()
    for value in available_regions or []:
        label = str(value or "").strip()
        key = _region_equivalence_key(label)
        if label and key not in seen and regions_equivalent(label, requested):
            matches.append(label)
            seen.add(key)
    return matches


def dedupe_equivalent_regions(regions: Iterable[object]) -> list[str]:
    """Stable-dedupe region labels, collapsing only explicit country aliases."""
    result: list[str] = []
    for value in regions or []:
        label = str(value or "").strip()
        if label and not any(regions_equivalent(label, prior) for prior in result):
            result.append(label)
    return result


# Individual word tokens that denote a region (from the alias phrases and their
# canonical names). Used to keep a region mention from being mistaken for a
# variable path segment during variable refinement.
_REGION_ALIAS_TOKENS: frozenset[str] = frozenset(
    token
    for phrases, canonical in REGION_ALIASES
    for source in (*phrases, canonical)
    for token in re.findall(r"[a-z0-9]+", source.lower())
)


# Canonical scenario labels (e.g. "Current Policies") rarely exist verbatim in
# the dataset, which uses model-specific codes like ``PR_CurPol_CP``. These
# patterns map a canonical family to the dataset codes that belong to it so a
# request for "current policies" matches every current-policies scenario.
SCENARIO_FAMILY_PATTERNS: dict[str, tuple[str, ...]] = {
    "Current Policies": ("curpol", "current polic"),
    "Baseline": ("baseline",),
    "Net Zero": ("nze", "net zero", "net-zero"),
    "NDC": ("ndc",),
}


def scenario_in_family(record_scenario: str, requested: str) -> bool:
    """True when a dataset scenario code belongs to the requested canonical family."""
    rs = str(record_scenario or "").strip().lower()
    requested_key = _normalized_catalog_key(requested)
    family = next(
        (
            label for label in SCENARIO_FAMILY_PATTERNS
            if _normalized_catalog_key(label) == requested_key
        ),
        None,
    )
    patterns = SCENARIO_FAMILY_PATTERNS.get(family or "")
    if not patterns:
        return False
    return any(p in rs for p in patterns)


def scenario_family_members(requested: str, available_scenarios: Iterable[str]) -> list[str]:
    """Return the dataset scenario codes that belong to the requested family."""
    return sorted(s for s in (available_scenarios or []) if scenario_in_family(s, requested))


def scenario_scope_tokens_in_query(query: str) -> set[str]:
    """Tokens of any scenario alias phrase present in the query.

    A scenario is scope, not a variable. When the user writes "under net zero"
    or "current policies", those words must not be counted as evidence for a
    variable — but the resolved scenario *code* (e.g. ``NZE_Bench_H``) shares no
    tokens with the phrase the user typed, so ignoring the code is not enough.
    This returns the literal phrase tokens so the caller can exclude them.

    Single-word aliases are only returned when they cannot also be a variable
    descriptor (``policy`` and ``baseline`` appear in real variable names such
    as ``Policy Cost``), keeping the exclusion data-driven and safe. Multi-word
    phrases and distinctive scenario codes are always safe to exclude.
    """
    safe_single_word = {"curpol", "bau", "nze", "ndc"}
    tokens: set[str] = set()
    for phrases, _canonical in SCENARIO_ALIASES:
        for phrase in phrases:
            phrase_tokens = re.findall(r"[a-z0-9]+", phrase.lower())
            multiword = len(phrase_tokens) > 1
            if not multiword and phrase.lower() not in safe_single_word:
                continue
            if _contains_phrase(query, phrase):
                tokens.update(phrase_tokens)
    return tokens


def _normalized_catalog_key(value: object) -> str:
    """Normalize case and separators without erasing token boundaries."""
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").casefold()))


def _contains_phrase(query: str, phrase: str) -> bool:
    # Treat whitespace, hyphens and slashes as equivalent separators so aliases
    # generalize across natural typography (e.g. "solar PV" / "solar-PV").
    tokens = [re.escape(token) for token in re.findall(r"[\w]+", phrase)]
    if not tokens:
        return False
    pattern = r"\b" + r"[\s\-_/]+".join(tokens) + r"\b"
    return bool(re.search(pattern, query, flags=re.IGNORECASE))


def explicit_scenarios_from_query(query: str, available_scenarios: Iterable[str]) -> list[str]:
    """Return dataset scenario names that appear verbatim in the query, longest
    first. Word-boundary matching means a family label like ``Baseline`` will not
    match inside a distinct code such as ``PR_Baseline`` (the underscore keeps
    them separate), so a typed scenario (``BAU``, ``PR_NDC_CP``) is trusted over
    an extractor result that collapsed it to a generic family. Used to override
    a mangled scenario before filtering.
    """
    q = str(query or "")
    found: list[str] = []
    found_keys: set[str] = set()
    canonical_families = set(SCENARIO_FAMILY_PATTERNS)
    scenarios = {str(s).strip() for s in (available_scenarios or []) if str(s).strip()}
    for scenario in sorted(
        scenarios,
        key=lambda value: (
            0 if value in canonical_families else 1,
            -len(value),
            value.casefold(),
            value,
        ),
    ):
        tokens = [re.escape(token) for token in re.findall(r"[A-Za-z0-9]+", scenario)]
        if not tokens:
            continue
        # Separators inside a complete scenario code are interchangeable, while
        # the outer boundary still treats an underscore/hyphen as part of a
        # larger code. Thus ``PR-Baseline`` can match ``PR_Baseline`` without a
        # shorter catalogue value ``Baseline`` stealing that mention.
        pattern = (
            r"(?<![A-Za-z0-9_-])"
            + r"[\s_\-/]+".join(tokens)
            + r"(?![A-Za-z0-9_-])"
        )
        if re.search(pattern, q, flags=re.IGNORECASE):
            normalized = _normalized_catalog_key(scenario)
            if normalized in found_keys:
                continue
            # Skip a scenario already covered by a longer match containing it.
            if any(
                normalized in _normalized_catalog_key(longer)
                and normalized != _normalized_catalog_key(longer)
                for longer in found
            ):
                continue
            found.append(scenario)
            found_keys.add(normalized)
    return found


_VARIABLE_QUERY_FILLER = {
    "a", "an", "and", "as", "at", "between", "but", "by", "can", "chart",
    "compare", "comparison", "data", "display", "do", "for", "from", "give",
    "graph", "in", "into", "is", "me", "of", "on", "or", "please", "plot",
    "provide", "retrieve", "series", "show", "the", "time", "to", "trend",
    "under", "value", "values", "versus", "visualise", "visualize", "what",
    "which", "with", "year", "years", "how", "much", "scenario", "scenarios", "pathway",
    "pathways", "global", "globally", "worldwide",
    # Measurement-operation words describe what the user wants done or
    # measured; they are weak evidence for a taxonomy leaf by themselves.
    "amount", "consume", "consumed", "consumption", "demand", "generate",
    "generated", "generation", "level", "output", "produce", "produced",
    "production", "supply", "total",
    # Chart-type and drawing verbs describe the *output*, never a taxonomy leaf.
    "bar", "line", "pie", "scatter", "histogram", "figure", "diagram",
    "draw", "make", "create", "visualisation", "visualization",
    # Comparison/availability grammar is scope, not taxonomy evidence.
    "together", "does", "report", "reports", "reported", "reporting", "model",
    # Time-window prepositions accompany a year, not a variable name. (These
    # carry no variable evidence; the year itself is parsed separately.)
    "after", "before", "until", "around", "during", "since", "over", "till",
}

# A spelling-similarity hit is useful recovery evidence, but it must not outrank
# a literal catalogue-token match merely because the similar token happens to
# be rarer.  Keep the factor domain-independent and apply it uniformly.
_FUZZY_VARIABLE_EVIDENCE_FACTOR = 0.65


def rank_catalogue_variable_matches(
    query: str,
    available_variables: Iterable[str],
    *,
    ignored_values: Iterable[object] = (),
) -> list[dict]:
    """Rank runtime variables using token coverage and catalogue distinctiveness.

    The evidence is deliberately derived from the supplied catalogue. A single
    common token cannot become an automatic match, while an exact variable
    phrase, a well-supported multi-token phrase, or an unambiguous typo can.
    """
    variables = sorted({str(value).strip() for value in available_variables if str(value).strip()})
    if not variables:
        return []

    query_tokens = [
        token for token in re.findall(r"[a-z0-9]+", str(query or "").casefold())
        if token not in _VARIABLE_QUERY_FILLER and not token.isdigit()
    ]
    ignored_tokens = {
        token
        for value in ignored_values
        for token in re.findall(r"[a-z0-9]+", str(value or "").casefold())
    }

    def belongs_to_ignored_scope(token: str) -> bool:
        if token in ignored_tokens:
            return True
        # Scope values may themselves be misspelled (for example a region).
        # Remove only a very close, sufficiently long spelling variant so it
        # cannot distort variable confidence.
        return any(
            min(len(token), len(ignored)) >= 4
            and SequenceMatcher(None, token, ignored).ratio() >= 0.9
            for ignored in ignored_tokens
        )

    query_tokens = [token for token in query_tokens if not belongs_to_ignored_scope(token)]
    if not query_tokens:
        return []

    variable_tokens = {
        variable: set(re.findall(r"[a-z0-9]+", variable.casefold()))
        for variable in variables
    }
    document_frequency = Counter(
        token for tokens in variable_tokens.values() for token in tokens
    )
    catalogue_size = len(variables)

    def weight(token: str) -> float:
        return 1.0 + math.log((catalogue_size + 1) / (document_frequency.get(token, 0) + 1))

    query_weight = sum(weight(token) for token in set(query_tokens)) or 1.0
    normalized_query = _normalized_catalog_key(query)
    explicit_structured = "|" in str(query or "")
    ranked: list[dict] = []

    for variable in variables:
        candidate_tokens = variable_tokens[variable]
        if not candidate_tokens:
            continue
        exact_terms = {token for token in query_tokens if token in candidate_tokens}
        fuzzy_terms: dict[str, tuple[str, float]] = {}
        for token in set(query_tokens) - exact_terms:
            best_candidate = ""
            best_ratio = 0.0
            for candidate_token in candidate_tokens:
                ratio = SequenceMatcher(None, token, candidate_token).ratio()
                if ratio > best_ratio:
                    best_candidate, best_ratio = candidate_token, ratio
            if best_ratio >= 0.88:
                fuzzy_terms[token] = (best_candidate, best_ratio)

        matched_query_terms = exact_terms | set(fuzzy_terms)
        matched_candidate_terms = exact_terms | {
            candidate for candidate, _ in fuzzy_terms.values()
        }
        # For a fuzzy token (commonly a typo or singular/plural variation), use
        # the catalogue token's document frequency. An unseen query spelling
        # must not receive an artificial rarity boost.
        #
        # The contribution is also capped at what an exact match of the query
        # token itself would be worth. Without the cap a rare catalogue token
        # scored against a common query token (the singular "emission" against
        # the very common "emissions") yields more evidence than a perfect match
        # would, pushing `query_coverage` above 1.0 and floating obscure
        # variables such as `Emission Factor|CO2|Tailpipe|2W` to the top.
        matched_weight = sum(weight(token) for token in exact_terms) + sum(
            min(weight(candidate), weight(query_token))
            * ratio
            * _FUZZY_VARIABLE_EVIDENCE_FACTOR
            for query_token, (candidate, ratio) in fuzzy_terms.items()
        )
        query_coverage = matched_weight / query_weight
        candidate_coverage = len(matched_candidate_terms) / len(candidate_tokens)
        candidate_key = _normalized_catalog_key(variable)
        exact_phrase = bool(
            candidate_key
            and re.search(
                r"(?:^|\s)" + re.escape(candidate_key) + r"(?:$|\s)",
                normalized_query,
            )
        )
        structured_exact = bool(explicit_structured and exact_phrase)
        score = (
            matched_weight
            + (4.0 * query_coverage)
            + (2.0 * candidate_coverage)
            + (8.0 if structured_exact else 0.0)
            + (3.0 if exact_phrase else 0.0)
        )
        ranked.append({
            "variable": variable,
            "score": score,
            "query_coverage": query_coverage,
            "candidate_coverage": candidate_coverage,
            "matched_terms": sorted(matched_query_terms),
            "unmatched_terms": sorted(set(query_tokens) - matched_query_terms),
            "exact_terms": sorted(exact_terms),
            "fuzzy_terms": {
                token: {"candidate": candidate, "ratio": ratio}
                for token, (candidate, ratio) in fuzzy_terms.items()
            },
            "exact_phrase": exact_phrase,
            "structured_exact": structured_exact,
            # How many segments of this variable the query never mentioned. A
            # deep, specialised variable can score well on the few tokens it does
            # share while most of its taxonomy went unasked for.
            "unmentioned_segments": len(candidate_tokens) - len(matched_candidate_terms),
        })

    ranked.sort(
        key=lambda item: (
            -item["score"],
            -item["query_coverage"],
            -item["candidate_coverage"],
            len(item["variable"]),
            item["variable"],
        )
    )
    for index, item in enumerate(ranked):
        runner_up = ranked[index + 1]["score"] if index + 1 < len(ranked) else 0.0
        margin = item["score"] - runner_up
        matched_count = len(item["matched_terms"])
        only_fuzzy_typo = bool(
            len(query_tokens) == 1
            and not item["exact_terms"]
            and len(item["fuzzy_terms"]) == 1
            and item["candidate_coverage"] >= 0.5
            and next(iter(item["fuzzy_terms"].values()))["ratio"] >= 0.92
            and margin >= 0.75
        )
        full_phrase = bool(
            item["exact_phrase"]
            and item["query_coverage"] >= 0.9
            and not item["unmatched_terms"]
        )
        # Auto-accepting a deep variable from a handful of tokens is how
        # "emissions intensity of electricity" landed on the IDA decomposition
        # `IDA|Emissions|CO2|Energy|Supply|Electricity|CO2 intensity`: three
        # matched tokens against four segments the user never asked for. A
        # confident answer needs most of the taxonomy to have been named; when it
        # was not, clarifying is the honest outcome.
        shallow_enough = item["unmentioned_segments"] <= 2
        strong_composition = bool(
            matched_count >= 2
            and shallow_enough
            and item["query_coverage"] >= 0.62
            and item["candidate_coverage"] >= 0.4
            and margin >= 1.0
        )
        # Mutual complete match: the query names every token of this variable and
        # this variable accounts for every variable-token in the query (no
        # missing, no extra segment), and it is strictly the best candidate.
        # This resolves a base variable like `Price|Carbon` for "carbon price"
        # against more specific siblings (`Price|Carbon|EUETS`, ...) whose extra
        # segments the query never mentioned, instead of forcing a clarification.
        complete_match = bool(
            matched_count >= 2
            and not item["unmatched_terms"]
            and item["query_coverage"] >= 0.9
            and item["candidate_coverage"] >= 0.999
            and margin > 0
        )
        item["margin"] = margin
        item["auto_accept"] = bool(
            item["structured_exact"] or full_phrase or only_fuzzy_typo
            or strong_composition or complete_match
        )
    return ranked


# Energy-base aliases ("final/primary/secondary energy") must not steal queries
# that ask for a specific carrier or sector under that energy family — those
# should resolve to the more specific variable instead.
_ENERGY_BASE_CANDIDATES = {"Final Energy", "Primary Energy", "Secondary Energy"}
_ENERGY_SPECIFIC_TOKENS = (
    "electricity", "solar", "wind", "hydro", "nuclear", "gas", "oil", "coal",
    "biomass", "bioenergy", "hydrogen", "heat", "geothermal",
    "industry", "industrial", "transport", "transportation", "buildings",
    "residential", "commercial",
)


def _energy_base_blocked(query: str, candidate: str) -> bool:
    """True when an energy-base candidate should yield to a more specific carrier."""
    if candidate not in _ENERGY_BASE_CANDIDATES:
        return False
    ql = query.lower()
    return any(re.search(r"\b" + tok + r"\b", ql) for tok in _ENERGY_SPECIFIC_TOKENS)


def _token_supports(query_token: str, segment_token: str) -> bool:
    """Whether a query token evidences a variable-path segment token, tolerating
    a common word-form variant so a user word like "transport" still matches the
    catalogue segment "Transportation". Deliberately conservative: only long
    tokens qualify via prefix/spelling similarity, so short unrelated words never
    collide."""
    if query_token == segment_token:
        return True
    if (
        len(query_token) >= 5
        and len(segment_token) >= 5
        and (segment_token.startswith(query_token) or query_token.startswith(segment_token))
    ):
        return True
    return bool(
        min(len(query_token), len(segment_token)) >= 6
        and SequenceMatcher(None, query_token, segment_token).ratio() >= 0.85
    )


def preferred_variable_from_query(query: str, available_variables: Iterable[str]) -> str | None:
    q = str(query or "")
    available = set(available_variables or [])
    structural_words = {
        "a", "an", "and", "as", "at", "between", "but", "by", "for", "from",
        "in", "into", "of", "on", "or", "the", "to", "under", "with",
        "show", "plot", "give", "me", "data", "value", "values", "year", "years",
    }
    query_tokens = {
        token for token in re.findall(r"[a-z0-9]+", q.lower())
        if token not in structural_words
    }
    # A region mention ("for the EU") must never be read as evidence for a
    # variable path segment that happens to share that word (e.g. the
    # "Intra-EU" leaf of a transport-aviation emissions variable). Region words
    # are scope, not variable qualifiers, so strip them from the tokens used to
    # justify narrowing to a more specific descendant.
    variable_tokens = query_tokens - _REGION_ALIAS_TOKENS

    def _refine(candidate: str) -> str:
        """Prefer a descendant only when its extra path segments are explicitly
        supported by user tokens. This is catalogue-driven, not technology-specific."""
        descendants = [value for value in available if value == candidate or value.startswith(candidate + "|")]
        if len(descendants) <= 1:
            return candidate
        base_tokens = set(re.findall(r"[a-z0-9]+", candidate.lower()))

        def _supported(segment_token: str) -> bool:
            return any(_token_supports(qt, segment_token) for qt in variable_tokens)

        def score(value: str):
            value_tokens = set(re.findall(r"[a-z0-9]+", value.lower()))
            extra = value_tokens - base_tokens
            supported_extra = sum(1 for token in extra if _supported(token))
            total_overlap = sum(1 for token in value_tokens if _supported(token))
            return (supported_extra, total_overlap, -len(value_tokens), -len(value), value)

        best = max(descendants, key=score)
        return best if score(best)[0] > 0 else candidate

    for phrases, candidates in VARIABLE_ALIASES:
        if any(_contains_phrase(q, phrase) for phrase in phrases):
            if any(_energy_base_blocked(q, candidate) for candidate in candidates):
                # The base is too generic given a named carrier/sector. Prefer the
                # specific descendant the user's words support (e.g. "final energy
                # in transport" -> Final Energy|Transportation), but only when
                # *every* extra path segment is supported — otherwise the level the
                # user named does not exist as its own variable (e.g. only
                # Buildings|Residential / |Commercial exist), and a clarification
                # is more honest than silently picking one sub-scope.
                for candidate in candidates:
                    if candidate in available:
                        refined = _refine(candidate)
                        if refined == candidate:
                            continue
                        base_seg = set(re.findall(r"[a-z0-9]+", candidate.lower()))
                        extra_seg = set(re.findall(r"[a-z0-9]+", refined.lower())) - base_seg
                        if all(
                            any(_token_supports(qt, token) for qt in variable_tokens)
                            for token in extra_seg
                        ):
                            return refined
                continue
            for candidate in candidates:
                if candidate in available:
                    return _refine(candidate)
            # Fallback: accept a variable that starts with the canonical
            # candidate (e.g. "Population" -> "Population|Total"), but never
            # guess between sibling sub-scopes the user did not name. A
            # descendant is eligible only when its extra path segments are
            # supported by the user's own tokens, or when it is the single
            # descendant (so nothing is being silently narrowed).
            for candidate in candidates:
                prefix = candidate + "|"
                prefixed = sorted(v for v in available if v.startswith(prefix))
                if not prefixed:
                    continue
                base_tokens = set(re.findall(r"[a-z0-9]+", candidate.lower()))

                def _extra_supported(value: str) -> bool:
                    extra = set(re.findall(r"[a-z0-9]+", value.lower())) - base_tokens
                    return any(
                        _token_supports(qt, token) for token in extra for qt in query_tokens
                    )

                supported = [v for v in prefixed if _extra_supported(v)]
                pool = supported or (prefixed if len(prefixed) == 1 else [])
                if pool:
                    return _refine(min(pool, key=lambda value: (value.count("|"), len(value), value)))
    return None


def canonical_scenario_from_query(query: str, available_scenarios: Iterable[str] | None = None) -> str | None:
    q = str(query or "")
    available = set(available_scenarios or [])
    for phrases, canonical in SCENARIO_ALIASES:
        if any(_contains_phrase(q, phrase) for phrase in phrases):
            # Canonical families are filters over every compatible runtime
            # scenario, not a request to pick one arbitrary member.  Iterating
            # the ``available`` set previously made ``net zero`` resolve to a
            # different NZE code across processes.  Keep the stable family
            # label whenever at least one member exists; downstream filters
            # expand it with ``scenario_in_family``.
            if canonical in SCENARIO_FAMILY_PATTERNS and (
                not available
                or canonical in available
                or scenario_family_members(canonical, available)
            ):
                return canonical
            if canonical in available or not available:
                return canonical
            for scenario in available:
                low = scenario.lower()
                if canonical.lower() in low or any(phrase in low for phrase in phrases):
                    return scenario
            return canonical
    return None


def canonical_scenario_family_from_query(query: str, available_scenarios: Iterable[str] | None = None) -> str | None:
    """Return the canonical scenario *family* label a query names (e.g. "Net
    Zero", "Current Policies"), rather than one member code. Useful when both
    sides of a comparison should expand to their full family via
    :func:`scenario_family_members`. Only returns a family that has at least one
    member in the loaded data."""
    available = set(available_scenarios or [])
    for phrases, canonical in SCENARIO_ALIASES:
        if any(_contains_phrase(query, phrase) for phrase in phrases):
            if not available or canonical in available or scenario_family_members(canonical, available):
                return canonical
    return None


def canonical_region_from_query(query: str, available_regions: Iterable[str] | None = None) -> str | None:
    q = str(query or "")
    available = set(available_regions or [])
    for phrases, canonical in REGION_ALIASES:
        if any(_contains_phrase(q, phrase) for phrase in phrases):
            if canonical in available or not available:
                return canonical
            equivalent = region_family_members(canonical, available)
            if equivalent:
                return equivalent[0]
            return canonical
    return None
