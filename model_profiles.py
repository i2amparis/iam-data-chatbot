import re
from typing import Any

from model_aliases import normalize_model_name


CURATED_MODEL_PROFILES: dict[str, dict[str, Any]] = {
    "REMIND": {
        "name": "REMIND",
        "aliases": ["remind"],
        "description": (
            "REMIND is a global integrated assessment model that links macroeconomic "
            "development, the energy system, climate policy and technology choices. "
            "It is commonly used to assess mitigation pathways, carbon pricing, energy "
            "transitions and technology deployment across regions and sectors."
        ),
        "sectors": ["energy", "economy", "climate", "land-use interactions"],
        "typical_use_cases": [
            "global and regional mitigation pathways",
            "carbon pricing and policy scenarios",
            "energy-system transformation",
            "technology deployment including renewables, CCS and negative emissions",
        ],
        "limitations": [
            "assumptions vary by scenario, model version and experiment design",
            "local IAM PARIS metadata may not expose every technology assumption",
        ],
        "assumptions_note": (
            "No explicit REMIND assumptions field is available in the local IAM PARIS "
            "model metadata. For CCS or carbon dioxide removal, treat the answer as "
            "scenario-dependent: REMIND can represent these technology options, but "
            "their availability, cost and deployment depend on the specific scenario "
            "configuration."
        ),
        "search_hint": "REMIND",
    },
    "WITCH": {
        "name": "WITCH",
        "aliases": ["witch"],
        "description": (
            "WITCH is an integrated assessment model focused on the interaction between "
            "the economy, energy system, climate policy and technological change. It is "
            "often used for regional mitigation strategies, policy design and technology "
            "innovation analysis."
        ),
        "sectors": ["economy", "energy", "climate policy", "technology innovation"],
        "typical_use_cases": [
            "regional mitigation and burden-sharing analysis",
            "carbon pricing and policy assessment",
            "technology innovation and diffusion scenarios",
            "long-term climate policy pathways",
        ],
        "limitations": [
            "assumptions are experiment-specific",
            "IAM PARIS local metadata may not include full methodology notes",
        ],
        "assumptions_note": (
            "No explicit WITCH assumptions field is available in the local IAM PARIS "
            "model metadata. Interpret assumptions through the selected scenario, "
            "policy setup and available IAM PARIS results."
        ),
        "search_hint": "WITCH",
    },
    "MESSAGEix-GLOBIOM": {
        "name": "MESSAGEix-GLOBIOM",
        "aliases": ["message", "messageix", "message ix", "message-ix", "messageix-globiom"],
        "description": (
            "MESSAGEix-GLOBIOM is an integrated assessment framework combining the "
            "MESSAGEix energy-system model with GLOBIOM land-use and agriculture "
            "components. It is used for energy, land, emissions and climate mitigation "
            "pathway analysis."
        ),
        "sectors": ["energy", "land use", "agriculture", "emissions", "climate"],
        "typical_use_cases": [
            "energy-land-climate mitigation pathways",
            "technology and resource constraints",
            "emissions trajectories across sectors",
            "scenario analysis for climate policy",
        ],
        "limitations": [
            "results depend on scenario protocol and regional aggregation",
            "local IAM PARIS metadata may not expose all model internals",
        ],
        "assumptions_note": (
            "No explicit MESSAGEix-GLOBIOM assumptions field is available in the local "
            "IAM PARIS model metadata. Use the scenario name, workspace and result "
            "metadata to interpret assumptions for a specific output."
        ),
        "search_hint": "MESSAGEix",
    },
    "GCAM": {
        "name": "GCAM",
        "aliases": ["gcam"],
        "description": (
            "GCAM is an integrated assessment model that represents interactions among "
            "energy, economy, land, water and climate systems. It is commonly used for "
            "scenario analysis of emissions, energy transitions and climate policy."
        ),
        "sectors": ["energy", "economy", "land", "water", "climate"],
        "typical_use_cases": [
            "emissions and energy pathway analysis",
            "climate policy scenarios",
            "land and resource interactions",
            "regional mitigation comparisons",
        ],
        "limitations": [
            "model assumptions vary by scenario and input dataset",
            "local IAM PARIS metadata may not include detailed assumptions",
        ],
        "assumptions_note": (
            "No explicit assumptions field is available in the local IAM PARIS model "
            "metadata. Use the selected scenario and IAM PARIS result metadata to "
            "interpret GCAM assumptions for a specific answer."
        ),
        "search_hint": "GCAM",
    },
    "GCAM-PR": {
        "name": "GCAM-PR",
        "aliases": ["gcam-pr", "gcam pr", "gcampr", "gcam-pr 7.0", "gcampr 7"],
        "description": (
            "GCAM-PR is a GCAM-derived regional model profile used for more detailed "
            "Puerto Rico energy, economy and climate-policy analysis. It is useful "
            "when the question asks about GCAM-PR-specific pathways or local policy "
            "configurations."
        ),
        "sectors": ["regional energy", "economy", "climate policy", "land"],
        "typical_use_cases": [
            "Puerto Rico regional transition pathways",
            "local energy and emissions policy analysis",
            "scenario-specific regional model outputs",
        ],
        "limitations": [
            "not interchangeable with global GCAM outputs",
            "assumptions depend on the local GCAM-PR configuration",
        ],
        "assumptions_note": (
            "No explicit GCAM-PR assumptions field is available in the local IAM PARIS "
            "model metadata. Treat assumptions as GCAM-PR configuration and "
            "scenario-specific."
        ),
        "search_hint": "GCAM-PR",
    },
}


def _profile_aliases(profile: dict[str, Any]) -> set[str]:
    aliases = {str(profile.get("name", ""))}
    aliases.update(str(alias) for alias in profile.get("aliases", []) if alias)
    return {alias.lower() for alias in aliases if alias}


def find_model_profile(text: str) -> dict[str, Any] | None:
    query = str(text or "").lower()
    query_norm = normalize_model_name(query)
    if not query and not query_norm:
        return None

    matches: list[tuple[int, dict[str, Any]]] = []
    for profile in CURATED_MODEL_PROFILES.values():
        for alias in _profile_aliases(profile):
            alias_norm = normalize_model_name(alias)
            if not alias_norm:
                continue
            if re.search(r"(?<![\w-])" + re.escape(alias) + r"(?![\w-])", query):
                matches.append((len(alias_norm), profile))
                continue
            if alias_norm and alias_norm in query_norm:
                if alias_norm == "gcam" and "gcampr" in query_norm:
                    continue
                matches.append((len(alias_norm), profile))
    if matches:
        return sorted(matches, key=lambda item: item[0], reverse=True)[0][1]
    return None


def has_strong_model_metadata(record: dict[str, Any] | None) -> bool:
    if not record:
        return False
    desc = str(record.get("description", "") or "").strip()
    assumptions = str(record.get("assumptions", "") or "").strip()
    return len(desc) >= 120 or len(assumptions) >= 80


def format_model_profile_answer(
    profile: dict[str, Any],
    requested_name: str = "",
    asks_assumptions: bool = False,
) -> str:
    name = str(requested_name or profile.get("name") or "Model").strip()
    parts = [f"### {name}"]
    description = str(profile.get("description", "") or "").strip()
    if description:
        parts.append(f"Description:\n{description}")

    sectors = [str(item) for item in profile.get("sectors", []) if item]
    if sectors:
        parts.append("Model scope:\n- " + "\n- ".join(sectors))

    uses = [str(item) for item in profile.get("typical_use_cases", []) if item]
    if uses:
        parts.append("Useful for:\n- " + "\n- ".join(uses))

    if asks_assumptions:
        note = str(profile.get("assumptions_note", "") or "").strip()
        if note:
            parts.append(f"Assumptions:\n{note}")

    limitations = [str(item) for item in profile.get("limitations", []) if item]
    if limitations:
        parts.append("Interpretation notes:\n- " + "\n- ".join(limitations))

    search_hint = str(profile.get("search_hint", "") or name).strip()
    parts.append(
        "Related model documentation:\n"
        "- [IAM PARIS Models](https://iamparis.eu/models)\n"
        f"- Open the Models page and search for: `{search_hint}`"
    )
    return "\n\n".join(parts)
