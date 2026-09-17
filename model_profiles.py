import re
from difflib import SequenceMatcher
from typing import Any

from model_aliases import normalize_model_name


CURATED_MODEL_PROFILES: dict[str, dict[str, Any]] = {
    "REMIND-MAgPIE": {
        "name": "REMIND-MAgPIE",
        "aliases": ["remind-magpie", "remind magpie", "remind-mAgPIE"],
        "developer": "Potsdam Institute for Climate Impact Research (PIK)",
        "description": (
            "REMIND-MAgPIE couples the REMIND energy-economy-climate model with "
            "the MAgPIE land-use and agriculture model to study interactions among "
            "energy, land, food, emissions and climate mitigation pathways."
        ),
        "sectors": ["energy", "economy", "climate", "land use", "agriculture"],
        "typical_use_cases": [
            "integrated energy-land-climate mitigation pathways",
            "food, land and bioenergy interactions",
            "technology deployment and carbon-removal scenarios",
        ],
        "limitations": [
            "results depend on the model version and scenario protocol",
            "technology and land-use assumptions should be interpreted from the selected experiment",
        ],
        "search_hint": "REMIND-MAgPIE",
    },
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
    "IMAGE": {
        "name": "IMAGE",
        "aliases": ["image"],
        "developer": "IMAGE team under the authority of PBL Netherlands Environmental Assessment Agency",
        "description": (
            "IMAGE (Integrated Model to Assess the Global Environment) is a global "
            "integrated assessment modelling framework representing interactions among "
            "society, the biosphere and the climate system."
        ),
        "sectors": ["energy", "land", "climate", "biodiversity", "human development"],
        "typical_use_cases": [
            "long-term global environmental and sustainable-development pathways",
            "climate, biodiversity and land-use policy assessment",
        ],
        "search_hint": "IMAGE",
        "documentation_label": "Official IMAGE model documentation (PBL)",
        "documentation_url": "https://www.pbl.nl/en/image/home",
    },
    "E3ME-FTT": {
        "name": "E3ME-FTT",
        "aliases": ["e3me", "e3me-ftt", "e3me model"],
        "developer": "Cambridge Econometrics",
        "description": (
            "E3ME is a dynamic global macroeconomic model representing interactions "
            "among the economy, society and environment. Its detailed sectoral "
            "disaggregation and empirical approach support analysis across those systems."
        ),
        "sectors": ["economy", "society", "environment", "energy"],
        "typical_use_cases": [
            "economy-energy-environment policy analysis",
            "sectorally detailed transition and socioeconomic impact analysis",
        ],
        "methodology_note": "Macro-econometric E3 model",
        "search_hint": "E3ME-FTT",
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
    matches = find_model_profiles(text)
    return matches[0] if matches else None


def find_model_profiles(text: str) -> list[dict[str, Any]]:
    """Return every distinct curated profile mentioned, longest match first."""
    query = str(text or "").lower()
    query_norm = normalize_model_name(query)
    if not query and not query_norm:
        return []

    # Prefer boundary-aware literal matches. Normalized containment is useful
    # for punctuation variants, but on its own it makes a family name match
    # inside a longer, distinct model name (``Alpha`` in ``Alpha-Extended``).
    literal_matches: list[tuple[int, dict[str, Any], str]] = []
    fallback_matches: list[tuple[int, dict[str, Any], str]] = []
    for profile in CURATED_MODEL_PROFILES.values():
        for alias in _profile_aliases(profile):
            alias_norm = normalize_model_name(alias)
            if not alias_norm:
                continue
            if re.search(r"(?<![\w-])" + re.escape(alias) + r"(?![\w-])", query):
                literal_matches.append((len(alias_norm), profile, alias_norm))
                continue
            if alias_norm and alias_norm in query_norm:
                fallback_matches.append((len(alias_norm), profile, alias_norm))

    literal_aliases = {alias for _length, _profile, alias in literal_matches}
    matches: list[tuple[int, dict[str, Any]]] = [
        (length, profile) for length, profile, _alias in literal_matches
    ]
    for length, profile, alias_norm in fallback_matches:
        if any(alias_norm != literal and alias_norm in literal for literal in literal_aliases):
            continue
        matches.append((length, profile))
    ordered = sorted(matches, key=lambda item: item[0], reverse=True)
    unique = []
    seen = set()
    for _length, profile in ordered:
        name = str(profile.get("name", ""))
        if name and name not in seen:
            seen.add(name)
            unique.append(profile)
    return unique


def _profile_identity_tokens(profiles: list[dict[str, Any]]) -> set[str]:
    """Tokens which identify the compared models rather than the requested focus.

    A qualitative comparison such as ``GCAM vs GEM-E3 as model types`` should
    inspect the catalogue's model-type fields for the *requested concept*, not
    try to find the strings ``gcam`` and ``gem`` inside those fields.  Runtime
    profiles do not always carry aliases, so include both the display name and
    any curated aliases available on the profile.
    """
    tokens: set[str] = set()
    for profile in profiles:
        labels = [profile.get("name", ""), *(profile.get("aliases", []) or [])]
        for label in labels:
            tokens.update(re.findall(r"[a-z0-9]+", str(label or "").casefold()))
    return tokens


def format_model_comparison_answer(
    profiles: list[dict[str, Any]],
    query: str = "",
) -> str:
    """Compare profiles using metadata dimensions requested in the query.

    The query selects structured fields only; substantive values come from the
    supplied profiles. With no requested dimension, the general comparison
    continues to show developer, coverage and applications.
    """
    if len(profiles) < 2:
        return ""
    q = str(query or "").casefold()
    requested = {
        key
        for key, pattern in {
            "developer": r"\b(?:developer|developed\s+by|institution|organisation|organization)\b",
            "coverage": r"\b(?:systems?|sectors?|cover(?:s|ed|ing|age)?|scope)\b",
            # Match uses as a metadata noun, not the ordinary transitive verb
            # in a question such as "which model uses an ... approach?".
            "uses": (
                r"\b(?:applications?|use\s+cases?|useful\s+for|used\s+(?:for|to)|"
                r"(?:common|typical|main|primary|intended)\s+uses?|their\s+uses?|"
                r"what\s+(?:are\s+)?(?:their|its|the)?\s*uses?|"
                r"better\s+suited|suitable\s+for|designed\s+(?:for|to))\b"
            ),
            # Methodology/model type and technology representation are separate
            # catalogue fields. Keeping their intent cues separate prevents a
            # method question from being answered with a technology inventory.
            "method": (
                r"\b(?:methodolog(?:y|ies|ical)|methods?|approaches?|"
                r"model(?:ling|ing)?\s+types?|types?\s+of\s+models?|"
                r"optimi[sz](?:ation|e|ed|es|ing))\b"
            ),
            "technology": (
                r"\b(?:technolog(?:y|ies|ical)|technology\s+(?:treatment|representation|detail|options?)|"
                r"treatment\s+of\s+technolog(?:y|ies)|how\s+.+\s+works?)\b"
            ),
            "limitations": r"\b(?:limitations?|caveats?|constraints?|interpretation\s+notes?)\b",
            "assumptions": r"\b(?:assumptions?|premises?)\b",
        }.items()
        if re.search(pattern, q)
    }
    focused = bool(requested)
    if not requested:
        requested = {"developer", "coverage", "uses"}

    names = [str(profile.get("name", "Model")) for profile in profiles]
    lines = [f"### {' vs '.join(names)}", ""]
    for profile in profiles:
        name = str(profile.get("name", "Model"))
        sectors = [str(value) for value in profile.get("sectors", []) if value]
        uses = [str(value) for value in profile.get("typical_use_cases", []) if value]
        limitations = [str(value) for value in profile.get("limitations", []) if value]
        assumptions = str(profile.get("assumptions_note", "") or "").strip()
        methodology = str(
            profile.get("methodology_note")
            or profile.get("description")
            or ""
        ).strip()
        technology = str(
            profile.get("technology_note")
            or profile.get("description")
            or ""
        ).strip()
        lines.append(f"**{name}**")
        if "developer" in requested:
            if profile.get("developer"):
                lines.append(f"- Developer: {profile['developer']}")
            elif focused:
                lines.append("- Developer: not described in the loaded catalogue metadata")
        if "coverage" in requested:
            if sectors:
                lines.append("- Systems/sectors: " + ", ".join(sectors))
            elif focused:
                lines.append("- Systems/sectors: not described in the loaded catalogue metadata")
        if "uses" in requested:
            if uses:
                lines.append("- Common uses: " + "; ".join(uses))
            elif focused:
                lines.append("- Common uses: not described in the loaded catalogue metadata")
        if "method" in requested:
            if methodology:
                lines.append("- Methodology/model type: " + methodology)
            else:
                lines.append("- Methodology/model type: not described in the loaded catalogue metadata")
        if "technology" in requested:
            if technology:
                lines.append("- Technology representation: " + technology)
            else:
                lines.append("- Technology representation: not described in the loaded catalogue metadata")
        if "limitations" in requested:
            if limitations:
                lines.append("- Limitations: " + "; ".join(limitations))
            elif focused:
                lines.append("- Limitations: not described in the loaded catalogue metadata")
        if "assumptions" in requested:
            if assumptions:
                lines.append("- Assumptions: " + assumptions)
            elif focused:
                lines.append("- Assumptions: not described in the loaded catalogue metadata")
        lines.append("")

    # For a choice question about a specific method/technology concept, make
    # the catalogue evidence explicit instead of forcing the reader to infer
    # the answer from two paragraphs. Both the concept and the match come from
    # the query/runtime metadata; no model or method names are encoded here.
    focused_field = "methodology_note" if "method" in requested else (
        "technology_note" if "technology" in requested else ""
    )
    if focused_field:
        focus_filler = {
            "a", "an", "and", "approach", "approaches", "are", "both", "compare",
            "comparison", "differ", "different", "do", "does", "for", "how", "in",
            "is", "method", "methods", "methodology", "model", "modeling", "modelling",
            "models", "of", "one", "option", "options", "representation", "technology",
            "technologies", "the", "their", "them", "treatment", "detail", "details",
            "these", "they", "this", "those", "two", "type", "types", "use", "used",
            "uses", "versus", "vs", "what", "which", "with", "works",
        }
        focus_terms = {
            token for token in re.findall(r"[a-z0-9]+", q)
            if (
                token not in focus_filler
                and token not in _profile_identity_tokens(profiles)
                and len(token) > 2
            )
        }

        def _term_matches(term: str, text: str) -> bool:
            field_terms = set(re.findall(r"[a-z0-9]+", str(text or "").casefold()))
            return term in field_terms or any(
                min(len(term), len(candidate)) >= 5
                and SequenceMatcher(None, term, candidate).ratio() >= 0.88
                for candidate in field_terms
            )

        if focus_terms:
            field_matches = []
            for profile in profiles:
                field_text = str(profile.get(focused_field) or "").strip()
                matched_terms = {
                    term for term in focus_terms if _term_matches(term, field_text)
                }
                if matched_terms and len(matched_terms) / len(focus_terms) >= 0.5:
                    field_matches.append(str(profile.get("name", "Model")))
            rendered_focus = ", ".join(f"`{term}`" for term in sorted(focus_terms))
            if field_matches:
                lines.append(
                    "Catalogue wording match for " + rendered_focus + ": "
                    + ", ".join(f"**{name}**" for name in field_matches) + "."
                )
            else:
                lines.append(
                    "The loaded catalogue metadata does not explicitly match "
                    + rendered_focus + " for either model."
                )
    elif re.search(r"\b(?:which|what)\b", q):
        # Produce a direct evidence-based conclusion for general qualitative
        # choices too (for example a sector or application focus). Vocabulary
        # and answers come only from the query and supplied profile metadata.
        choice_filler = {
            "a", "an", "and", "are", "as", "both", "described", "does",
            "better", "detailed", "either", "for", "focused", "focus", "in",
            "interaction", "interactions", "is", "model",
            "models", "of", "one", "the", "their", "them", "these", "those",
            "suited", "suitable", "two", "what", "which", "with",
        }
        identity_terms = _profile_identity_tokens(profiles)
        choice_terms = {
            token for token in re.findall(r"[a-z0-9]+", q)
            if token not in choice_filler and token not in identity_terms and len(token) > 2
        }

        def _concept_match(term: str, candidate: str) -> bool:
            return term == candidate or (
                min(len(term), len(candidate)) >= 5
                and SequenceMatcher(None, term, candidate).ratio() >= 0.84
            )

        scored: list[tuple[int, str]] = []
        for profile in profiles:
            evidence = " ".join([
                str(profile.get("description") or ""),
                str(profile.get("methodology_note") or ""),
                str(profile.get("technology_note") or ""),
                " ".join(str(value) for value in profile.get("sectors", []) if value),
                " ".join(str(value) for value in profile.get("typical_use_cases", []) if value),
            ])
            evidence_terms = set(re.findall(r"[a-z0-9]+", evidence.casefold()))
            score = sum(
                any(_concept_match(term, candidate) for candidate in evidence_terms)
                for term in choice_terms
            )
            scored.append((score, str(profile.get("name", "Model"))))
        best_score = max((score for score, _name in scored), default=0)
        matches = [name for score, name in scored if score == best_score and score > 0]
        if matches:
            rendered = ", ".join(f"`{term}`" for term in sorted(choice_terms))
            lines.append(
                "Catalogue wording match for " + rendered + ": "
                + ", ".join(f"**{name}**" for name in matches) + "."
            )
    if "coverage" in requested:
        sector_sets = [set(str(value) for value in profile.get("sectors", []) if value) for profile in profiles]
        shared = set.intersection(*sector_sets) if sector_sets else set()
        if shared:
            lines.append("Shared coverage: " + ", ".join(sorted(shared)) + ".")
        for index, (profile, sectors) in enumerate(zip(profiles, sector_sets)):
            others = set().union(*(
                other for other_index, other in enumerate(sector_sets)
                if other_index != index
            ))
            distinctive = sorted(sectors - others)
            if distinctive:
                lines.append(f"Distinctive in {profile.get('name', 'Model')}: " + ", ".join(distinctive) + ".")
    if focused:
        lines.append("Comparison focus: " + ", ".join(sorted(requested)) + ".")
    lines.extend(["", "[Browse IAM PARIS model documentation](https://iamparis.eu/models)"])
    return "\n".join(lines)


_PLACEHOLDER_TEXT = {"nan", "none", "null", "n/a", "na", "-", "--", ""}


def clean_metadata_text(value: Any) -> str:
    """Normalize a metadata string, treating serialized empties (e.g. a pandas
    ``NaN`` rendered as ``"nan"``) as absent so they never surface to users."""
    text = str(value if value is not None else "").strip()
    return "" if text.casefold() in _PLACEHOLDER_TEXT else text


def _iamparis_model_url(profile: dict[str, Any]) -> str:
    """Return a direct IAM PARIS model URL when the runtime catalogue exposes it."""
    for key in (
        "iamparis_model_url", "model_url", "modelUrl", "url", "URL",
        "documentation_url", "documentationUrl",
    ):
        value = clean_metadata_text(profile.get(key))
        if value.startswith("https://iamparis.eu/models/"):
            return value

    for key in ("route", "path", "slug", "href"):
        value = clean_metadata_text(profile.get(key))
        if re.fullmatch(r"/?models/\d+/?", value):
            return "https://iamparis.eu/" + value.strip("/")

    # Public routes use the catalogue record id. `modelId` is a separate
    # metadata identifier and can return a 200 page with no selected model.
    for key in ("id", "ID", "pk"):
        value = clean_metadata_text(profile.get(key))
        if re.fullmatch(r"\d+", value):
            return f"https://iamparis.eu/models/{value}"

    return ""


def has_strong_model_metadata(record: dict[str, Any] | None) -> bool:
    if not record:
        return False
    desc = clean_metadata_text(record.get("description"))
    assumptions = clean_metadata_text(record.get("assumptions"))
    return len(desc) >= 120 or len(assumptions) >= 80


def _focused_profile_dimension(query: str) -> str:
    q = str(query or "").casefold()
    if re.search(
        r"\b(?:developers?|develop(?:ed|s|ing)?|institution|organisation|organization|"
        r"maintainers?|created|built)\b",
        q,
    ):
        return "developer"
    if re.search(
        r"\b(?:general[-\s]+equilibrium|partial[-\s]+equilibrium|cge|"
        r"methodolog(?:y|ies|ical)|model(?:ling|ing)?\s+types?|"
        r"types?\s+of\s+models?|what\s+(?:kind|type|sort)\s+of\s+model|"
        r"optimi[sz](?:ation|e|ed|es|ing)|"
        r"simulation|bottom-?up|top-?down|energy\s+system\s+model)\b",
        q,
    ):
        return "method"
    if re.search(
        r"\b(?:technolog(?:y|ies|ical)|innovation|learning(?:\s+by\s+doing)?|"
        r"technical\s+change)\b",
        q,
    ):
        return "technology"
    if re.search(
        r"\b(?:overview|purpose|intended\s+for|designed\s+(?:for|to)|"
        r"used\s+(?:for|to)|use\s+cases?|applications?|"
        r"problems?\s+.+\s+study|policy\s+questions?\s+.+\s+answer|"
        r"better\s+suited|suitable\s+for)\b",
        q,
    ):
        return "uses"
    if _coverage_topic_from_query(query):
        return "coverage"
    return ""


def _coverage_topic_from_query(query: str) -> str:
    """Extract the qualitative system/sector named in a model-coverage question."""
    text = str(query or "").strip().rstrip("?!. ")
    match = re.match(
        r"^\s*does\s+.+?\s+(?:model|cover(?:s|ed|ing)?|include(?:s|d|ing)?|"
        r"represent(?:s|ed|ing)?|simulate(?:s|d|ing)?|capture(?:s|d|ing)?|account\s+for)\s+(.+)$",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    topic = re.sub(r"^(?:the|a|an)\s+", "", match.group(1).strip(), flags=re.IGNORECASE)
    if not topic or len(topic) > 80 or re.search(
        r"\b(?:data|dataset|result|results|output|outputs|value|values|timeseries|time\s+series)\b",
        topic,
        flags=re.IGNORECASE,
    ):
        return ""
    return topic


def _method_concept_from_query(query: str) -> tuple[str, tuple[str, ...]]:
    """Return the user-facing concept and grounded spellings to match."""
    q = str(query or "").casefold().replace("-", " ")
    concepts = (
        ("general equilibrium", ("general equilibrium", "computable general equilibrium", "cge")),
        ("partial equilibrium", ("partial equilibrium",)),
        ("computable general equilibrium (CGE)", ("computable general equilibrium", "cge")),
        ("energy system model", ("energy system",)),
        ("optimization model", ("optimization", "optimisation")),
        ("simulation model", ("simulation",)),
        ("bottom-up model", ("bottom-up", "bottom up")),
        ("top-down model", ("top-down", "top down")),
    )
    for label, spellings in concepts:
        if any(spelling in q for spelling in spellings):
            return label, spellings
    return "", ()


def _focused_model_profile_answer(
    profile: dict[str, Any],
    name: str,
    query: str,
) -> str:
    """Answer one requested metadata dimension without dumping a full profile."""
    dimension = _focused_profile_dimension(query)
    if not dimension:
        return ""

    lines = [f"### {name}", ""]
    if dimension == "developer":
        developer = clean_metadata_text(profile.get("developer"))
        if developer:
            lines.append(f"`{name}` is developed by the {developer}.")
        else:
            lines.append(
                f"The developer or maintaining organization for `{name}` is not described "
                "in the loaded IAM PARIS profile."
            )
    elif dimension == "method":
        methodology = clean_metadata_text(profile.get("methodology_note"))
        concept, spellings = _method_concept_from_query(query)
        concept_noun = (
            concept if concept.endswith("model") else f"{concept} model"
        ) if concept else ""
        if not methodology:
            if concept:
                lines.append(
                    f"The loaded IAM PARIS profile does not state whether `{name}` is a {concept_noun}."
                )
            else:
                lines.append(
                    "Methodology/model type: not described in the loaded IAM PARIS profile."
                )
        else:
            methodology_key = methodology.casefold().replace("optimisation", "optimization")
            normalized_spellings = tuple(
                spelling.casefold().replace("optimisation", "optimization")
                for spelling in spellings
            )
            if concept == "general equilibrium" and (
                "partial equilibrium" in methodology_key
                or re.search(r"\bnot\b[^.]{0,60}\bgeneral equilibrium\b", methodology_key)
            ):
                lines.append(
                    f"No. The loaded catalogue classifies `{name}` as `{methodology}`."
                )
            elif concept and any(spelling in methodology_key for spelling in normalized_spellings):
                lines.append(
                    f"Yes. The loaded catalogue classifies `{name}` as `{methodology}`."
                )
            elif concept:
                lines.append(
                    f"The loaded catalogue classifies `{name}` as `{methodology}`, but it does not "
                    f"explicitly establish whether that is a {concept_noun}."
                )
            else:
                lines.append(f"Methodology/model type: {methodology}")
    elif dimension == "uses":
        uses = [
            str(value).strip()
            for value in (profile.get("typical_use_cases", []) or [])
            if str(value or "").strip()
        ]
        if uses:
            lines.append("Common uses described in the loaded profile:")
            lines.extend(f"- {value}" for value in uses)
        else:
            description = clean_metadata_text(profile.get("description"))
            if description:
                lines.append("The loaded profile describes its purpose as follows:")
                lines.append(description)
            else:
                lines.append(
                    f"The intended uses of `{name}` are not described in the loaded IAM PARIS profile."
                )
    elif dimension == "coverage":
        topic = _coverage_topic_from_query(query)
        topic_parts = [
            re.sub(
                r"\b(?:systems?|sectors?|areas?)\b$",
                "",
                re.sub(r"^(?:the|a|an)\s+", "", part.strip(), flags=re.IGNORECASE),
                flags=re.IGNORECASE,
            ).strip()
            for part in re.split(r"\s*(?:,|/|&|\band\b|\bor\b)\s*", topic, flags=re.IGNORECASE)
        ]
        topic_parts = [part for part in topic_parts if part]
        if not topic_parts and topic:
            topic_parts = [topic]

        def _topic_regex(value: str) -> re.Pattern[str] | None:
            tokens = re.findall(r"[a-z0-9]+", value.casefold())
            if not tokens:
                return None
            pattern = r"[^a-z0-9]+".join(re.escape(token) for token in tokens)
            return re.compile(
                r"(?<![a-z0-9])" + pattern + r"(?![a-z0-9])",
                flags=re.IGNORECASE,
            )

        topic_regexes = {part: _topic_regex(part) for part in topic_parts}
        sectors = [
            str(value).strip() for value in (profile.get("sectors", []) or [])
            if str(value or "").strip()
        ]
        evidence_values = [
            *sectors,
            clean_metadata_text(profile.get("description")),
            clean_metadata_text(profile.get("methodology_note")),
            clean_metadata_text(profile.get("technology_note")),
            *[
                str(value).strip()
                for value in (profile.get("typical_use_cases", []) or [])
                if str(value or "").strip()
            ],
        ]
        evidence_values = [value for value in evidence_values if value]
        matched_parts = [
            part
            for part, regex in topic_regexes.items()
            if regex and any(regex.search(value) for value in evidence_values)
        ]
        sector_matches = list(dict.fromkeys(
            value
            for value in sectors
            if any(regex and regex.search(value) for regex in topic_regexes.values())
        ))
        negative_parts = []
        for part, regex in topic_regexes.items():
            if not regex:
                continue
            negative_pattern = re.compile(
                r"\b(?:does\s+not|doesn't|cannot|can't|excludes?|without|no)\b"
                r"[^.!?]{0,80}" + regex.pattern,
                flags=re.IGNORECASE,
            )
            if any(negative_pattern.search(value) for value in evidence_values):
                negative_parts.append(part)

        if negative_parts:
            lines.append(
                f"No. The loaded IAM PARIS profile explicitly indicates that `{name}` does not "
                "model " + ", ".join(f"`{part}`" for part in negative_parts) + "."
            )
            lines.append(
                "Supporting profile evidence: a loaded profile field explicitly excludes the named scope."
            )
        elif topic_parts and len(matched_parts) == len(topic_parts):
            lines.append(
                f"Yes. The loaded IAM PARIS profile explicitly indicates that `{name}` "
                f"models `{topic}`."
            )
            if sector_matches:
                lines.append(
                    "Supporting profile evidence: listed model scope includes "
                    + ", ".join(f"`{value}`" for value in sector_matches) + "."
                )
            else:
                lines.append(
                    f"Supporting profile evidence: the description or use-case metadata explicitly mentions `{topic}`."
                )
        elif matched_parts:
            missing_parts = [part for part in topic_parts if part not in matched_parts]
            lines.append(
                "Partly stated. The loaded IAM PARIS profile explicitly covers "
                + ", ".join(f"`{part}`" for part in matched_parts)
                + ", but does not explicitly state "
                + ", ".join(f"`{part}`" for part in missing_parts)
                + "."
            )
            if sectors:
                lines.append(
                    "Supporting profile evidence: its listed scope is "
                    + ", ".join(f"`{value}`" for value in sectors) + "."
                )
        else:
            lines.append(
                f"I cannot verify from the loaded metadata whether `{name}` models `{topic}`. "
                "Please consult the model documentation below for the authoritative answer."
            )
    else:
        technology = clean_metadata_text(profile.get("technology_note"))
        if technology:
            lines.append(f"Technology representation:\n{technology}")
        else:
            # A curated description can provide bounded, high-level evidence
            # even when the catalogue lacks a structured technology field. Show
            # only the relevant sentences/use-cases, then make the missing
            # mechanism explicit instead of presenting a generic model dump.
            description = clean_metadata_text(profile.get("description"))
            evidence = [
                sentence.strip()
                for sentence in re.split(r"(?<=[.!?])\s+", description)
                if re.search(r"\btechnolog(?:y|ies|ical)\b", sentence, re.IGNORECASE)
            ]
            evidence.extend(
                str(value).strip()
                for value in (profile.get("typical_use_cases", []) or [])
                if re.search(r"\btechnolog(?:y|ies|ical)\b", str(value), re.IGNORECASE)
            )
            evidence = list(dict.fromkeys(value for value in evidence if value))
            if evidence:
                lines.append("Technology information in the loaded profile:\n- " + "\n- ".join(evidence))
                lines.append(
                    "The loaded profile does not provide a more specific mechanism for how "
                    "technological change is represented."
                )
            else:
                lines.append(
                    "Technology representation: not described in the loaded IAM PARIS profile; "
                    f"I cannot determine how `{name}` handles technological change from the loaded metadata."
                )

    lines.extend(["", *_profile_documentation_lines(profile, name)])
    return "\n".join(lines)


def _profile_documentation_lines(profile: dict[str, Any], name: str) -> list[str]:
    lines = ["Related model documentation:"]
    iamparis_model_url = _iamparis_model_url(profile)
    if iamparis_model_url:
        lines.append(f"- [IAM PARIS model page]({iamparis_model_url})")
        return lines

    documentation_url = clean_metadata_text(profile.get("documentation_url"))
    documentation_label = clean_metadata_text(profile.get("documentation_label"))
    if documentation_url:
        lines.append(f"- [{documentation_label or 'Official model documentation'}]({documentation_url})")
    search_hint = str(profile.get("search_hint", "") or name).strip()
    lines.extend([
        "- [IAM PARIS Models](https://iamparis.eu/models)",
        f"- Open the Models page and search for: `{search_hint}`",
    ])
    return lines


def format_model_profile_answer(
    profile: dict[str, Any],
    requested_name: str = "",
    asks_assumptions: bool = False,
    asks_limitations: bool = False,
    query: str = "",
) -> str:
    name = str(requested_name or profile.get("name") or "Model").strip()
    focused = _focused_model_profile_answer(profile, name, query)
    if focused:
        return focused
    parts = [f"### {name}"]
    description = clean_metadata_text(profile.get("description"))
    if description:
        parts.append(f"Description:\n{description}")

    developer = clean_metadata_text(profile.get("developer"))
    if developer:
        parts.append(f"Developer:\n{developer}")

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
    elif asks_limitations:
        parts.append(
            "Interpretation notes:\n"
            "- The loaded IAM PARIS catalogue does not provide a specific limitation for this model. "
            "Use the model documentation and experiment metadata before interpreting a particular result."
        )

    parts.append("\n".join(_profile_documentation_lines(profile, name)))
    return "\n\n".join(parts)
