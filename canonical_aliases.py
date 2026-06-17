import re
from typing import Iterable


VARIABLE_ALIASES: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("gross domestic product", "gdp"), ("GDP|MER", "GDP|PPP")),
    (("greenhouse gas", "greenhouse gases", "ghg"), ("Emissions|GHG",)),
    (("methane", "ch4"), ("Emissions|CH4", "Emissions|Kyoto Gases|CH4")),
    (("nitrous oxide", "n2o"), ("Emissions|N2O", "Emissions|Kyoto Gases|N2O")),
    (("solar capacity", "solar pv capacity", "photovoltaic capacity", "pv capacity"), ("Capacity|Electricity|Solar",)),
    (("wind capacity", "wind power capacity", "installed wind"), ("Capacity|Electricity|Wind",)),
    (("electricity generation", "power generation"), ("Secondary Energy|Electricity",)),
    (("carbon dioxide emissions", "co2 emissions", "carbon emissions"), ("Emissions|CO2",)),
    (("population", "population projection", "population projections", "number of people"), ("Population",)),
    # Energy families: a bare "final/primary/secondary energy" request should map to
    # the base variable, not an over-specific sub-carrier (e.g. "Final Energy|Geothermal").
    (("final energy demand", "final energy"), ("Final Energy",)),
    (("primary energy demand", "primary energy"), ("Primary Energy",)),
    (("secondary energy",), ("Secondary Energy",)),
)


SCENARIO_ALIASES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("current policy", "current policies", "curpol"), "Current Policies"),
    (("baseline", "business as usual", "bau"), "Baseline"),
    (("policy", "policy scenario"), "Policy"),
    (("net zero", "net-zero", "nze"), "Net Zero"),
    (("nationally determined contribution", "nationally determined contributions", "ndc"), "NDC"),
)


REGION_ALIASES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("world", "global", "globally"), "World"),
    (("europe", "european union", "eu27", "eu"), "EU"),
    (("greece", "hellas"), "Greece"),
    (("china", "chn"), "China"),
    (("india", "ind"), "India"),
    (("united states", "usa", "us"), "United States"),
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
    patterns = SCENARIO_FAMILY_PATTERNS.get(str(requested or "").strip())
    if not patterns:
        return False
    return any(p in rs for p in patterns)


def scenario_family_members(requested: str, available_scenarios: Iterable[str]) -> list[str]:
    """Return the dataset scenario codes that belong to the requested family."""
    return sorted(s for s in (available_scenarios or []) if scenario_in_family(s, requested))


def _contains_phrase(query: str, phrase: str) -> bool:
    pattern = r"\b" + re.escape(phrase).replace(r"\ ", r"\s+") + r"\b"
    return bool(re.search(pattern, query, flags=re.IGNORECASE))


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


def preferred_variable_from_query(query: str, available_variables: Iterable[str]) -> str | None:
    q = str(query or "")
    available = set(available_variables or [])
    for phrases, candidates in VARIABLE_ALIASES:
        if any(_contains_phrase(q, phrase) for phrase in phrases):
            if any(_energy_base_blocked(q, candidate) for candidate in candidates):
                continue
            for candidate in candidates:
                if candidate in available:
                    return candidate
            # Fallback: accept the closest variable that starts with the canonical
            # candidate (e.g. "Population" -> "Population" or "Population|Total").
            for candidate in candidates:
                prefix = candidate + "|"
                prefixed = sorted(v for v in available if v.startswith(prefix))
                if prefixed:
                    return prefixed[0]
    return None


def canonical_scenario_from_query(query: str, available_scenarios: Iterable[str] | None = None) -> str | None:
    q = str(query or "")
    available = set(available_scenarios or [])
    for phrases, canonical in SCENARIO_ALIASES:
        if any(_contains_phrase(q, phrase) for phrase in phrases):
            if canonical == "Current Policies":
                return canonical
            if canonical in available or not available:
                return canonical
            for scenario in available:
                low = scenario.lower()
                if canonical.lower() in low or any(phrase in low for phrase in phrases):
                    return scenario
            return canonical
    return None


def canonical_region_from_query(query: str, available_regions: Iterable[str] | None = None) -> str | None:
    q = str(query or "")
    available = set(available_regions or [])
    for phrases, canonical in REGION_ALIASES:
        if any(_contains_phrase(q, phrase) for phrase in phrases):
            if canonical in available or not available:
                return canonical
            for region in available:
                if region.lower() == canonical.lower():
                    return region
            return canonical
    return None
