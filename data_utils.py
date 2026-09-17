import os
import re
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime
import logging
import base64
from io import BytesIO
from pathlib import Path
import requests.exceptions

from canonical_aliases import (
    canonical_region_from_query,
    dedupe_equivalent_regions,
    explicit_scenarios_from_query,
    preferred_variable_from_query,
    rank_catalogue_variable_matches,
    regions_equivalent,
    scenario_in_family,
    SCENARIO_FAMILY_PATTERNS,
)


def _scenario_match_ok(record_scenario: str, scenario_match: str) -> bool:
    """Scenario filter predicate that understands canonical families.

    Returns True when the record's scenario equals the requested one, or when the
    requested value is a canonical family label (e.g. "Current Policies") and the
    record's scenario code belongs to that family (e.g. ``PR_CurPol_CP``).
    """
    if not scenario_match:
        return True
    rs = str(record_scenario or "").strip()
    if rs == scenario_match:
        return True
    return scenario_in_family(rs, scenario_match)


def _scenario_is_family_label(scenario_match: str) -> bool:
    """True when scenario_match is a canonical family label (e.g. "Current Policies").

    Such labels map to many dataset codes, so an availability precheck must not
    gate on the verbatim label — the family-aware filter does the precise match.
    """
    return bool(scenario_match) and scenario_match in SCENARIO_FAMILY_PATTERNS
from simple_plotter import simple_plot_query, simple_plot_query_with_entities
from model_aliases import (
    display_model_label,
    extract_model_hint,
    is_presentable_model_label,
    match_model_name,
    resolve_model_candidates,
    resolve_model_family_members,
)
from model_profiles import (
    _iamparis_model_url,
    find_model_profile,
    format_model_profile_answer,
    has_strong_model_metadata,
)
from query_normalizer import normalize_query_text, query_tokens
from year_filters import (
    extract_year_range,
    is_finite_numeric_value,
    is_latest_year_filter,
    select_years,
)
from resolved_scope import consume_resolved_scope, record_resolved_scope
from utils_query import (
    match_variable_from_yaml,
    extract_examples_from_data,
    get_available_workspaces,
    extract_variable_and_region_from_query,
    resolve_natural_language_variable_universal,
    resolve_natural_language_variable_with_score,
    resolve_natural_language_variable_candidates,
    resolve_natural_language_variable_ranked,
    extract_region_from_query,
    region_mentions_from_query,
    format_region_label,
)


_WORKSPACE_CATALOG_PATH = Path("config/workspace_catalog.json")


def _workspace_catalog() -> dict[str, dict]:
    """Load public workspace labels and links without hard-coding them in routing."""
    try:
        payload = json.loads(_WORKSPACE_CATALOG_PATH.read_text())
    except (OSError, ValueError):
        return {}
    entries = payload.get("workspaces", []) if isinstance(payload, dict) else []
    return {
        str(entry.get("code", "")).strip(): entry
        for entry in entries
        if isinstance(entry, dict) and str(entry.get("code", "")).strip()
    }


def _workspace_entries(ts_data: list) -> list[dict]:
    """Return loaded workspaces with a public title, retaining their internal code privately."""
    catalogue = _workspace_catalog()
    records_by_code: dict[str, list[dict]] = {}
    for record in ts_data:
        code = str((record or {}).get("workspace_code", "")).strip()
        if code:
            records_by_code.setdefault(code, []).append(record)

    entries = []
    for code, records in records_by_code.items():
        configured = dict(catalogue.get(code, {}))
        runtime_title = next(
            (
                str(record.get(key, "")).strip()
                for record in records
                for key in ("workspace_title", "workspace_name", "study_title", "studyName")
                if str(record.get(key, "")).strip()
            ),
            "",
        )
        entries.append({
            **configured,
            "code": code,
            "title": str(configured.get("title") or runtime_title or code.replace("-", " ").replace("_", " ").title()),
            "records": records,
        })
    return sorted(entries, key=lambda entry: str(entry["title"]).casefold())


def _matched_workspace(question: str, entries: list[dict]) -> dict | None:
    normalized_query = re.sub(r"[^a-z0-9]+", " ", str(question or "").casefold()).strip()
    allows_internal_code = bool(re.search(
        r"\b(?:study|workspace|results?|data\s+explorer|explainer)\b",
        normalized_query,
    ))
    for entry in entries:
        values = [entry.get("title")]
        # Short internal codes such as ``net-zero`` are also ordinary query
        # language. Do not let them select a study unless the user explicitly
        # frames the message as a study/workspace/results request.
        if allows_internal_code:
            values.append(entry.get("code"))
        for value in values:
            normalized_value = re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()
            if normalized_value and len(normalized_value) > 3 and re.search(
                rf"(?<![a-z0-9]){re.escape(normalized_value)}(?![a-z0-9])", normalized_query
            ):
                return entry
    return None


def _workspace_summary(entry: dict) -> str:
    records = entry.get("records", [])
    variables = sorted({str(record.get("variable", "")).strip() for record in records if record.get("variable")})
    scenarios = sorted({str(record.get("scenario", "")).strip() for record in records if record.get("scenario")})
    title = str(entry.get("title") or "Workspace")
    lines = [f"### {title}", "", "Available in this study:"]
    lines.append(f"- **Variables ({len(variables)}):** " + (", ".join(f"`{value}`" for value in variables[:10]) or "none"))
    lines.append(f"- **Scenarios ({len(scenarios)}):** " + (", ".join(f"`{value}`" for value in scenarios[:10]) or "none"))
    explorer_url = str(entry.get("explorer_url") or "").strip()
    explainer_url = str(entry.get("explainer_url") or "").strip()
    if explorer_url or explainer_url:
        lines.extend(["", "Explore this study:"])
        if explorer_url:
            lines.append(f"- [Open the data explorer]({explorer_url})")
        if explainer_url:
            lines.append(f"- [Read the study explainer]({explainer_url})")
    return "\n".join(lines)


def _workspace_choice_response(entries: list[dict]) -> str:
    sample = entries[:10]
    titles = "; ".join(str(entry.get("title") or entry.get("code")) for entry in sample)
    remaining = len(entries) - len(sample)
    suffix = f"; and {remaining} more" if remaining > 0 else ""
    return (
        "Please choose a study before I return numeric results, so I do not mix "
        "records from different studies. Available studies include: "
        f"{titles}{suffix}. Ask about one study to see its variables, scenarios, "
        "and direct data-explorer link."
    )
from utils.yaml_loader import load_all_yaml_files
from difflib import get_close_matches
import pickle
import os

# Create cached versions of YAML loading
def get_cached_yaml_definitions():
    # Try file cache. A corrupt/incompatible cache must not crash import
    # (this runs at module load), so fall back to regenerating from YAML.
    cache_file = "cache/yaml_dicts.pkl"
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception:
            logging.getLogger(__name__).warning(
                "Failed to load %s; regenerating from YAML.", cache_file, exc_info=True
            )

    # Load YAML files (expensive operation)
    variable_dict = load_all_yaml_files('definitions/variable')
    region_dict = load_all_yaml_files('definitions/region')
    result = (variable_dict, region_dict)
    
    # Save to cache
    os.makedirs("cache", exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(result, f)
    
    return result

# Load variable and region definitions from YAML files (replace lines 18-20)
variable_dict, region_dict = get_cached_yaml_definitions()
logger = logging.getLogger(__name__)


def _positive_int_setting(name: str, default: int) -> int:
    """Read a positive integer setting without making formatting fragile."""
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


# Detailed per-series tables remain easier to read for small result sets. Larger
# scopes switch to a compact matrix, with a separate cap on rendered series so a
# catalogue-wide query cannot produce an unbounded response. Deployments can
# tune both policies without changing query- or domain-specific code.
DATA_COMPACT_GROUP_THRESHOLD = _positive_int_setting(
    "IAM_DATA_COMPACT_GROUP_THRESHOLD", 6
)
DATA_GROUP_DISPLAY_LIMIT = _positive_int_setting(
    "IAM_DATA_GROUP_DISPLAY_LIMIT", 12
)


def _normalize_free_text(text: str) -> str:
    return normalize_query_text(text)


def _token_set(text: str) -> set[str]:
    return query_tokens(text)


def _looks_like_plot_request(text: str) -> bool:
    tokens = _token_set(text)
    return bool(tokens & {"plot", "graph", "chart", "visualize", "visualise"})


def _looks_like_comparison_request(text: str) -> bool:
    q = _normalize_free_text(text)
    tokens = _token_set(text)
    return bool(
        {"compare", "comparison", "contrast", "difference", "different", "versus", "vs"} & tokens
        or re.search(r"\bcompar(?:e|ed|es|ing|ison)\b", q)
        or re.search(r"\b(?:contrast|differ(?:s|ed|ent|ently|ence|ences)?)\b", q)
        or re.search(r"\bvs\b", q)
        or re.search(r"\bversus\b", q)
    )


def _looks_like_data_request(text: str) -> bool:
    tokens = _token_set(text)
    if _looks_like_plot_request(text):
        return True
    data_terms = {
        "data", "value", "values", "timeseries", "time", "series", "trend",
        "show", "display", "give", "provide", "retrieve", "fetch",
        "emission", "emissions", "capacity", "energy", "generation",
        "electricity", "demand", "supply", "variable", "variables",
        "gdp", "growth", "share", "shares", "price", "prices",
        "trajectory", "renewable", "renewables", "oil", "gas", "coal",
        "policy", "policies", "ndc", "ndcs",
    }
    if tokens & data_terms:
        return True
    q = _normalize_free_text(text)
    return bool(re.search(r"\btime\s+series\b", q) or re.search(r"\bunder\s+different\s+scenarios\b", q))


# N6: map common sub-regions (countries) to the aggregate region codes that are
# most likely to actually carry data, so no-data recovery suggests EU for Germany
# instead of an unrelated alphabetical region.
_REGION_AGGREGATE_KEYWORDS = {
    "europe": ["EU", "EUR", "EU27", "EU28", "Europe", "European Union", "R5OECD", "OECD"],
}
_EU_COUNTRIES = {
    "germany", "france", "italy", "spain", "poland", "netherlands", "belgium",
    "austria", "portugal", "greece", "sweden", "finland", "denmark", "ireland",
    "czech", "czechia", "hungary", "romania", "bulgaria", "slovakia", "slovenia",
    "croatia", "lithuania", "latvia", "estonia", "luxembourg", "malta", "cyprus",
}


def _aggregate_region_candidates(region: str, scoped_regions) -> list:
    """Return aggregate regions (e.g. EU) for a requested sub-region, restricted to
    those that actually exist in ``scoped_regions``. Empty when none apply."""
    lowered = str(region or "").strip().lower()
    if not lowered:
        return []
    targets: list = []
    if lowered in _EU_COUNTRIES or "europe" in lowered:
        targets = _REGION_AGGREGATE_KEYWORDS["europe"]
    if not targets:
        return []
    scoped_lower = {str(r).strip().lower(): r for r in (scoped_regions or [])}
    result = []
    for candidate in targets:
        actual = scoped_lower.get(candidate.lower())
        if actual and actual not in result:
            result.append(actual)
    return result


def _looks_like_discovery_request(text: str) -> bool:
    q = _normalize_free_text(text)
    tokens = _token_set(text)
    explicit_category_list = any(
        _looks_like_category_list_request(text, category)
        for category in ("models", "variables", "regions", "scenarios", "workspaces")
    )
    if explicit_category_list:
        return False
    availability_terms = {"available", "included", "exist", "exists", "have", "contains", "included"}
    categories = {"model", "models", "variable", "variables", "region", "regions", "scenario", "scenarios", "workspace", "workspaces"}
    if tokens & availability_terms and tokens & categories:
        return True
    if tokens & {"list", "overview", "discover", "browse", "explore"} and tokens & categories:
        return True
    if re.search(r"\bwhat\s+can\s+i\s+ask\b", q):
        return True
    if re.search(r"\bwhat\s+(?:kinds?|types?|sorts?)\s+of\s+data\b", q):
        return True
    # Overview-style questions about the dataset as a whole, e.g.
    # "what data do you have", "what data is available", "what data can you show".
    if re.search(r"\bwhat\s+data\s+(?:do|does|can|could|is|are|'s|s)\b", q):
        return True
    if re.search(r"\bwhat(?:'s|\s+is|\s+s)\s+(?:in|available\s+in)\s+(?:the\s+)?(?:data|dataset|database)\b", q):
        return True
    if re.search(r"\bdata\s+categor(?:y|ies)\b", q):
        return True
    if re.search(r"\bhelp\s+me\s+find\s+data\b", q):
        return True
    return False


def _model_scoped_category(text: str) -> str | None:
    """Detect a request for the scenarios/variables/regions/data of a *specific
    model*, e.g. "what scenarios does GCAM have", "which regions does it cover",
    "what data does GCAM have for the EU". Returns the plural category token
    (``scenarios``/``variables``/``regions``; bare "data" maps to ``variables``)
    or ``None``.

    The possessive verb (does/have/run/…) is required so a plain category list
    like "what scenarios are available" is left to the normal listing path; the
    caller supplies the actual model (named in the query or carried context).
    """
    q = _normalize_free_text(text)
    cat = re.search(r"\b(scenario|scenarios|variable|variables|region|regions|year|years|data)\b", q)
    if not cat:
        return None
    # A possessive verb ("does GCAM have") or an availability phrasing
    # ("what years are available for GCAM") both scope the request to a model;
    # the caller still requires an actual model, so plain listings such as
    # "what scenarios are available" keep using the normal listing path.
    if not re.search(
        r"\b(does|do|has|have|run|runs|use|uses|cover|covers|report|reports|reported|reporting|offer|offers)\b",
        q,
    ) and not re.search(r"\b(?:available|are\s+there|exist|exists)\b", q):
        return None
    token = cat.group(1)
    if token == "data":
        return "variables"
    return token.rstrip("s") + "s"


def _display_variable_label(value: object) -> str:
    """Clean source-only artefacts when presenting a variable catalogue.

    The underlying variable value remains untouched for filtering and
    provenance.  This is deliberately conservative: it removes stray leading
    quote characters and fixes the known standalone ``Energys`` segment.
    """
    label = str(value or "").strip().lstrip("'\"").strip()
    return re.sub(r"(?i)(^|\|)energys(?=\||$)", r"\1Energy", label)


def _display_variable_catalog(values) -> list[str]:
    return sorted({
        label
        for value in values
        if (label := _display_variable_label(value))
    })


def _list_model_category(
    category: str,
    model: str,
    ts_data: list,
    show_all: bool = False,
    model_members: list[str] | None = None,
) -> str:
    """List the distinct scenarios/variables/regions/years recorded for one model."""
    runtime_models = sorted({
        str(record.get("modelName", "")).strip()
        for record in ts_data
        if record and is_presentable_model_label(record.get("modelName"))
    })
    family_members = list(model_members or resolve_model_family_members(model, runtime_models))
    if not family_members and model in runtime_models:
        family_members = [model]
    family_set = set(family_members)
    display_model = str(model or "").strip()
    family_note = (
        f" Runtime labels included: {', '.join(family_members)}."
        if len(family_members) > 1 else ""
    )
    if category == "years":
        years = sorted({
            int(year)
            for r in ts_data
            if r and str(r.get("modelName", "")).strip() in family_set
            for year in (r.get("years") or {})
            if str(year).isdigit()
        })
        if not years:
            return f"I could not find any years recorded for model `{display_model}`."
        listed = ", ".join(str(year) for year in years)
        record_resolved_scope(
            model=display_model,
            models=family_members,
            result_models=family_members,
            action="query",
        )
        return (
            f"Model `{display_model}` reports data for years {years[0]}–{years[-1]} "
            f"({len(years)} distinct years: {listed}).{family_note}"
        )
    field = {"scenarios": "scenario", "variables": "variable", "regions": "region"}[category]
    all_values = {
        str(r.get(field, "")).strip()
        for r in ts_data
        if r and str(r.get("modelName", "")).strip() in family_set and r.get(field)
    }
    primary_values = sorted({
        str(r.get(field, "")).strip()
        for r in ts_data
        if r and str(r.get("modelName", "")).strip() == display_model and r.get(field)
    })
    values = primary_values + sorted(all_values - set(primary_values))
    if category == "regions":
        values = dedupe_equivalent_regions(values)
    elif category == "variables":
        values = _display_variable_catalog(values)
    if not values:
        return f"I could not find any {category} recorded for model `{display_model}`."
    shown = values if show_all else values[:15]
    more = "" if len(values) <= len(shown) else f" and {len(values) - len(shown)} more"
    hint = (
        f"\n\nShowing {len(shown)} of {len(values)}. "
        f"Say `show all {category} for {display_model}` if you need the full list."
        if more else ""
    )
    record_resolved_scope(
        model=display_model,
        models=family_members,
        result_models=family_members,
        action="query",
    )
    return (
        f"Model `{display_model}` has these {category}: {', '.join(shown)}{more}."
        + family_note
        + hint
    )


def _looks_like_model_info_request(text: str) -> bool:
    q = _normalize_free_text(text)
    tokens = _token_set(text)
    if tokens & {
        "info", "information", "details", "describe", "about", "explain",
        "assumption", "assumptions", "methodology", "structure", "works",
        "developer", "developers", "institution", "institutions",
        "limitation", "limitations", "caveat", "caveats", "constraint",
        "constraints", "coverage", "scope", "sector", "sectors",
        "application", "applications", "technology", "technologies",
        "purpose", "overview", "intended", "designed",
    }:
        return True
    return bool(
        re.search(r"\btell\s+me\s+about\b", q)
        or re.search(r"\bgive\s+me\s+(?:an?\s+)?overview\s+of\b", q)
        or re.search(r"\bwhat\s+(?:problems?|questions?)\s+is\b[^?]*\bdesigned\s+to\b", q)
        or re.search(r"\bwhat\s+(?:kinds?\s+of\s+)?policy\s+questions?\s+can\b[^?]*\banswer\b", q)
        or re.search(r"\bwhat\b[^?]*\b(?:is\s+intended|is\s+designed|used)\s+(?:for|to)\b", q)
        or re.search(r"\bhow\s+does\b", q)
        or re.search(
            r"\bwhat\s+does\s+(?:the\s+)?"
            r"(?!it|this|that|they|these|those\b)"
            r"[a-z0-9][\w./-]*\s+model\s+do\b",
            q,
        )
        or re.search(r"\bwhat\s+are\s+the\s+assumptions\b", q)
        or re.search(r"\bwho\s+(?:develops?|built|created)\b", q)
        or re.search(
            r"\bwhich\s+one\s+(?:is\s+associated\s+with|uses?|includes?|represents?)\b",
            q,
        )
        # Descriptive yes/no and "what kind" questions about a named model. The
        # subject after is/does must be a model *name*, never a pronoun — that
        # keeps a model-scoped availability question ("which regions does it
        # cover?") on the data path. The caller also pairs this with an actual
        # model reference, so a data query like "does China have data?" is safe.
        or re.search(r"\bwhat\s+(?:kind|type|sort)\s+of\s+model\b", q)
        or re.search(
            r"\bis\s+(?!it|this|that|they|these|those|the\b)\w[\w./-]*\s+(?:a|an)\b[^?]*\bmodel\b",
            q,
        )
        or re.search(
            r"\bdoes\s+(?!it|this|that|they|these|those|the\b)\w[\w./-]*\s+"
            r"(?:model|cover|covers|include|includes|"
            r"represent|represents|simulate|simulates|capture|captures|account\s+for)\b",
            q,
        )
        or re.search(
            r"\bis\s+(?!it|this|that|they|these|those|the\b)\w[\w./-]*\s+(?:a|an)\s+"
            r"(?:general\s+equilibrium|partial\s+equilibrium|"
            r"computable|recursive|dynamic|optimi[sz]ation|simulation|energy\s+system|"
            r"integrated\s+assessment|macro-?economic|bottom-?up|top-?down)\b",
            q,
        )
    )


def _looks_like_category_list_request(text: str, category: str) -> bool:
    q = _normalize_free_text(text)
    tokens = _token_set(text)
    singular = category.rstrip("s")
    valid_names = {singular, category}
    if category == "variables":
        valid_names.update({"indicator", "indicators", "metric", "metrics"})
    if category == "regions":
        valid_names.update({"country", "countries", "location", "locations", "area", "areas"})
    if category == "scenarios":
        valid_names.update({"pathway", "pathways"})
    if category == "models":
        valid_names.update({"iam", "iams"})
    category_pattern = "|".join(sorted((re.escape(name) for name in valid_names), key=len, reverse=True))
    explicit_list_terms = {"list", "which", "available", "included", "show", "display", "enumerate"}
    if tokens & {"assumption", "assumptions", "about", "explain", "describe", "details", "info", "information"}:
        return False
    if re.search(rf"\bwhat\s+(?:{category_pattern})\s+can\s+i\s+use\b", q):
        return True
    # Count questions ("how many scenarios are there", "number of models") are
    # catalogue-metadata requests, not per-region data queries.
    if re.search(rf"\bhow\s+many\s+(?:{category_pattern})\b", q) or re.search(
        rf"\bnumber\s+of\s+(?:{category_pattern})\b", q
    ):
        return True
    if (
        category == "variables"
        and re.search(
            rf"\b(?:what|which)\s+(?:{category_pattern})\s+can\s+you\s+"
            r"(?:plot|graph|chart|visuali[sz]e|show|display|use)\b",
            q,
        )
    ):
        return True
    if tokens & valid_names and tokens & explicit_list_terms:
        if tokens & {"price", "trajectory", "trend", "plot", "graph", "chart", "compare", "growth", "share", "emissions", "capacity", "gdp"}:
            return False
        return True
    return bool(
        re.search(rf"\bwhat\s+(?:{category_pattern})\s+(?:are\s+)?(?:available|included)\b", q)
        # "what scenarios are there", "what scenarios exist", "what scenarios do you have"
        or re.search(
            rf"\b(?:what|which)\s+(?:{category_pattern})\s+(?:are\s+there|exist|do\s+(?:you|we)\s+have)\b",
            q,
        )
        or re.search(
            rf"\b(?:what|which)\s+(?:{category_pattern})\s+can\s+you\s+"
            r"(?:plot|graph|chart|visuali[sz]e|show|display)\b",
            q,
        )
        or re.search(rf"\bwhich\s+(?:{category_pattern})\b", q)
        # "what models cover buildings", "which models include transport"
        or re.search(
            rf"\b(?:what|which)\s+(?:{category_pattern})\s+"
            r"(?:cover|covers|covering|include|includes|including|support|supports|have|report)\b",
            q,
        )
    )


def _history_has_region_or_workspace(
    history: list | None,
    region_dict: dict,
    ts_data: list
) -> bool:
    if not history:
        return False
    region_candidates = sorted({str(r.get('region', '')).strip() for r in ts_data if r and r.get('region')})
    workspaces = get_available_workspaces(ts_data)

    for turn in history:
        user_msg = ""
        if isinstance(turn, (list, tuple)) and turn:
            user_msg = str(turn[0] or "")
        elif isinstance(turn, dict):
            if turn.get("role") == "user":
                user_msg = str(turn.get("content", "") or "")
        if not user_msg:
            continue
        msg_lower = user_msg.lower()
        if workspaces and any(ws.lower() in msg_lower for ws in workspaces if ws):
            return True
        if extract_region_from_query(user_msg, region_dict, region_candidates):
            return True
    return False


def _rank_variable_candidates(
    question: str,
    variable_dict: dict,
    available_vars: set,
    ranked_vars: list | None = None,
    significant_words: list | None = None,
    ignored_values: list[str] | None = None,
    limit: int = 3,
) -> list[str]:
    """
    Return a short list of the best variable candidates to confirm with the user.
    The list prefers exact semantic matches, then available-variable substring matches,
    then fuzzy fallbacks from the YAML resolver.
    """
    query_lower = question.lower()
    if not available_vars:
        return []
    candidates: list[str] = []
    preferred = _preferred_available_variable(question, available_vars)
    if preferred:
        candidates.append(preferred)
    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
        "show", "plot", "graph", "chart", "visualize", "display", "give", "me", "please", "data",
        "value", "values", "time", "series", "timeseries", "trend", "region", "regions", "workspace",
        "workspaces", "model", "models", "scenario", "scenarios", "what", "which", "want", "need",
        "use", "it", "this", "that", "here", "there", "load", "please"
    }

    if ranked_vars:
        candidates.extend([name for name, _, _, _ in ranked_vars if name in available_vars])

    query_terms = set()
    if significant_words:
        query_terms.update(w for w in significant_words if len(w) > 2 and w not in stop_words)
    query_terms.update(
        w for w in re.findall(r"\b\w+\b", query_lower)
        if len(w) > 2 and w not in stop_words
    )
    ignored_terms = {
        term
        for value in (ignored_values or [])
        for term in _tokenize_text(str(value or ""))
    }
    query_terms -= ignored_terms

    if query_terms:
        term_matches = [
            var for var in available_vars
            if any(term in var.lower() for term in query_terms)
        ]
        candidates.extend(term_matches)

    if "co2" in query_lower or "carbon dioxide" in query_lower:
        candidates.extend([v for v in available_vars if "co2" in v.lower() or "emission" in v.lower()])
    if any(term in query_lower for term in ["emission", "emissions"]):
        candidates.extend([v for v in available_vars if "emission" in v.lower() or "co2" in v.lower()])
    if "solar" in query_lower or "pv" in query_lower or "photovoltaic" in query_lower:
        candidates.extend([v for v in available_vars if "solar" in v.lower() or "pv" in v.lower()])
    if "wind" in query_lower:
        candidates.extend([v for v in available_vars if "wind" in v.lower()])
    if "oil" in query_lower:
        candidates.extend([v for v in available_vars if "oil" in v.lower()])
    if any(term in query_lower for term in ["capacity", "generation", "demand", "investment"]):
        candidates.extend([
            v for v in available_vars
            if any(term in v.lower() for term in ["capacity", "generation", "demand", "investment"])
        ])

    if not candidates:
        candidates = resolve_natural_language_variable_candidates(question, variable_dict, top_k=limit)
        candidates = [c for c in candidates if c in available_vars]

    if not candidates:
        candidates = find_similar_available_variables(
            question,
            available_vars,
            intent=_infer_variable_intent(question, significant_words),
            significant_words=significant_words,
        )

    deduped: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in deduped:
            deduped.append(candidate)

    # Re-rank against the runtime catalogue and require meaningful wording
    # coverage. This prevents one generic overlapping token from surfacing an
    # otherwise unrelated variable merely because it sorts first.
    evidence = rank_catalogue_variable_matches(
        question,
        available_vars,
        ignored_values=ignored_values or [],
    )
    evidence_by_name = {item["variable"]: item for item in evidence}
    supported = [
        candidate for candidate in deduped
        if candidate in evidence_by_name
        and (
            candidate == preferred
            or evidence_by_name[candidate]["auto_accept"]
            or len(evidence_by_name[candidate]["matched_terms"]) >= 2
            or evidence_by_name[candidate]["query_coverage"] >= 0.5
        )
    ]
    supported.sort(
        key=lambda candidate: (
            -evidence_by_name[candidate]["score"],
            candidate,
        )
    )
    return supported[:limit]


def _preferred_available_variable(question: str, available_vars: set[str]) -> str | None:
    return preferred_variable_from_query(question, available_vars)


def unknown_named_region(
    question: str,
    region_candidates,
    model_names=None,
    scenario_names=None,
    variable_names=None,
) -> str | None:
    """Return a syntactically scoped place that is absent from live catalogues.

    Location phrases are read after ``for``/``in`` and validated against the
    runtime region, model, scenario and variable vocabularies.  Capitalization
    is not treated as evidence, so ordinary lowercase input works without
    turning variable qualifiers such as ``in industry`` into fake regions.
    """
    text = str(question or "")
    regions = [str(value).strip() for value in (region_candidates or []) if str(value).strip()]
    models = [str(value).strip() for value in (model_names or []) if str(value).strip()]
    scenarios = [str(value).strip() for value in (scenario_names or []) if str(value).strip()]
    variables = [str(value).strip() for value in (variable_names or []) if str(value).strip()]
    boundary_words = {
        "across", "after", "at", "before", "between", "by", "during", "for",
        "from", "in", "on", "over", "please", "since", "through", "to", "under",
        "until", "using", "versus", "via", "with",
    }
    non_location_phrases = {
        "a chart", "a graph", "a plot", "a table", "all models", "all scenarios",
        "available models", "available scenarios", "both models", "each model",
    }

    def _tokens(value: object) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", str(value or "").casefold()))

    variable_token_sets = [_tokens(value) for value in variables]
    scenario_token_sets = [_tokens(value) for value in scenarios]

    unknown_places: list[str] = []
    found_known_region = False
    for marker in re.finditer(r"\b(?:for|in)\s+", text, re.IGNORECASE):
        tail = text[marker.end():]
        words: list[str] = []
        last_end = 0
        for token_match in re.finditer(r"[A-Za-z][A-Za-z'’-]*|\d{4}", tail):
            # Stop if punctuation separated this token from the previous one.
            if words and re.search(r"[,.?!;:]", tail[last_end:token_match.start()]):
                break
            word = token_match.group(0)
            lower = word.casefold()
            if word.isdigit() or (words and lower in boundary_words):
                break
            if not words and lower in {"a", "an", "the"}:
                words.append(word)
                continue
            words.append(word)
            last_end = token_match.end()
            if len(words) >= 4:
                break
        place = " ".join(words).strip()
        if not place:
            continue
        pl = place.casefold()
        place_tokens = _tokens(place)
        if pl in {"world", "global"} or pl in non_location_phrases:
            if pl in {"world", "global"}:
                found_known_region = True
            continue
        if any(pl == region.casefold() for region in regions):
            found_known_region = True
            continue
        if any(pl == format_region_label(region).strip().casefold() for region in regions):
            found_known_region = True
            continue
        if canonical_region_from_query(place, regions):
            found_known_region = True
            continue
        if models and match_model_name(place, models):
            continue
        if place_tokens and any(place_tokens <= tokens for tokens in scenario_token_sets):
            continue
        if place_tokens and any(place_tokens <= tokens for tokens in variable_token_sets):
            continue
        unknown_places.append(place)
    if found_known_region:
        return None
    return unknown_places[0] if unknown_places else None


def _explicit_variable_in_query(question: str, available_vars: set[str]) -> str | None:
    """When the user types a structured (pipe-delimited) variable name verbatim,
    prefer that exact variable over fuzzy/alias resolution, which can otherwise
    latch onto a superstring such as `Price|Secondary Energy|Electricity`.

    Only structured names (containing `|`) are considered so that a bare word
    like `Population` cannot match incidentally; the longest match wins so a
    parent name never shadows the more specific one the user actually typed.
    """
    q = (question or "").lower()
    matches = [v for v in available_vars if "|" in v and v and v.lower() in q]
    if not matches:
        return None
    return max(matches, key=len)


def _record_has_year_data(record: dict) -> bool:
    record = record or {}
    if any(str(k).isdigit() for k in record.keys()):
        return True
    years = record.get("years")
    return isinstance(years, dict) and any(str(k).isdigit() for k in years.keys())


def _infer_variable_intent(question: str, significant_words: list | None = None) -> str:
    ql = (question or "").lower()
    words = set(w.lower() for w in (significant_words or []) if w)
    joined = " ".join(sorted(words))

    def _has_any(tokens: list[str]) -> bool:
        return any(t in ql or t in joined for t in tokens)

    if _has_any(["methane", "ch4"]):
        return "methane"
    if _has_any(["gdp", "economic", "economy", "growth"]):
        return "gdp"
    if _has_any(["price", "prices", "cost", "costs", "trajectory"]):
        return "price"
    if _has_any(["share", "shares", "fraction"]):
        return "share"
    if _has_any(["renewable", "renewables", "clean energy"]):
        return "renewables"
    if _has_any(["oil"]):
        return "oil"
    if _has_any(["gas"]):
        return "gas"
    if _has_any(["hydrogen"]):
        return "hydrogen"
    if _has_any(["nuclear"]):
        return "nuclear"
    if _has_any(["hydro", "hydropower"]):
        return "hydro"
    if _has_any(["solar", "pv", "photovoltaic"]):
        return "solar"
    if _has_any(["wind"]):
        return "wind"
    if _has_any(["transport", "transportation"]):
        return "transport"
    if _has_any(["industry", "industrial"]):
        return "industry"
    if _has_any(["building", "buildings", "residential", "commercial"]):
        return "buildings"
    if _has_any(["co2", "emission", "emissions", "carbon"]):
        return "emissions_co2"
    if _has_any(["capacity"]):
        return "capacity"
    if _has_any(["electricity", "power", "generation"]):
        return "electricity"
    if _has_any(["demand"]):
        return "demand"
    if _has_any(["supply"]):
        return "supply"
    if _has_any(["investment", "investments", "invest"]):
        return "investment"
    return "general"


def _tokenize_text(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) >= 2}


def _query_terms(question: str, significant_words: list | None = None) -> set[str]:
    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
        "show", "plot", "graph", "chart", "visualize", "display", "give", "me", "please", "data",
        "value", "values", "time", "series", "timeseries", "trend", "region", "regions", "workspace",
        "workspaces", "model", "models", "scenario", "scenarios", "what", "which", "want", "need",
        "use", "it", "this", "that", "here", "there", "load", "under", "list", "available",
        "tell", "about", "compare", "vs", "versus", "can", "you", "from"
    }
    terms = _tokenize_text(question)
    if significant_words:
        terms.update(str(w).lower() for w in significant_words if w)
    return {t for t in terms if t not in stop_words}


def _query_profile(question: str, significant_words: list | None = None) -> dict[str, set[str] | bool]:
    ql = (question or "").lower()
    terms = _query_terms(question, significant_words)

    def _has(*tokens: str) -> bool:
        return any(token in ql or token in terms for token in tokens)

    sector_terms = {
        "transport": _has("transport", "transportation", "mobility", "vehicle", "vehicles"),
        "industry": _has("industry", "industrial", "manufacturing", "steel", "cement"),
        "buildings": _has("building", "buildings", "residential", "commercial", "heating", "cooling"),
        "power": _has("electricity", "power", "generation", "grid"),
        "afolu": _has("afolu", "land", "agriculture", "forestry", "lulucf"),
    }

    energy_terms = {
        "renewables": _has("renewable", "renewables", "clean"),
        "solar": _has("solar", "pv", "photovoltaic"),
        "wind": _has("wind", "onshore", "offshore"),
        "oil": _has("oil", "petroleum", "liquids"),
        "gas": _has("gas", "methane", "naturalgas", "natural", "lng"),
        "hydrogen": _has("hydrogen", "h2"),
        "nuclear": _has("nuclear"),
        "hydro": _has("hydro", "hydropower"),
        "coal": _has("coal"),
        "bioenergy": _has("biomass", "bioenergy", "biofuel", "biofuels"),
    }

    metric_terms = {
        "emissions": _has("emission", "emissions", "co2", "carbon", "ghg", "greenhouse", "ch4", "methane", "n2o"),
        "capacity": _has("capacity", "installed"),
        "generation": _has("generation", "produce", "produced", "production", "electricity"),
        "demand": _has("demand", "consumption", "use", "used"),
        "supply": _has("supply", "supplyside", "production"),
        "investment": _has("investment", "invest", "investments", "spending", "capital"),
        "price": _has("price", "cost", "costs"),
        "share": _has("share", "shares", "fraction"),
        "gdp": _has("gdp", "growth", "economy", "economic"),
    }

    broad = {
        "broad_metric": sum(metric_terms.values()) <= 1,
        "broad_sector": sum(sector_terms.values()) == 0,
        "broad_energy": sum(energy_terms.values()) <= 1,
    }

    return {
        "terms": terms,
        "sector_terms": {k for k, v in sector_terms.items() if v},
        "energy_terms": {k for k, v in energy_terms.items() if v},
        "metric_terms": {k for k, v in metric_terms.items() if v},
        "broad_metric": broad["broad_metric"],
        "broad_sector": broad["broad_sector"],
        "broad_energy": broad["broad_energy"],
    }


def _has_meaningful_query_signal(
    question: str,
    significant_words: list | None = None,
    region: str | None = None,
    scenario: str | None = None,
    model: str | None = None,
    ignored_values: list[str] | None = None,
    available_variables: set[str] | None = None,
) -> bool:
    """
    Decide whether the user's free-text query contains enough semantic signal
    to offer ranked variable choices instead of asking a generic follow-up.
    """
    profile = _query_profile(question, significant_words)
    terms = set(profile["terms"])

    scope_values = [region, scenario, model, *(ignored_values or [])]
    for value in scope_values:
        if value:
            terms -= _tokenize_text(str(value))

    weak_terms = {
        "data", "show", "give", "plot", "graph", "chart", "value", "values",
        "time", "series", "timeseries", "query", "information", "info"
    }
    terms = {
        term for term in terms
        if term not in weak_terms and not term.isdigit()
    }

    if profile["metric_terms"] or profile["sector_terms"] or profile["energy_terms"]:
        return True
    if available_variables:
        ranked = rank_catalogue_variable_matches(
            question,
            available_variables,
            ignored_values=scope_values,
        )
        return bool(ranked and ranked[0]["matched_terms"])
    return len(terms) >= 2


def _variable_relevance_score(
    variable: str,
    question: str,
    intent: str,
    significant_words: list | None = None,
    prefer_with_years: bool = False,
) -> float:
    var = str(variable or "")
    var_lower = var.lower()
    var_terms = _tokenize_text(var_lower.replace("|", " "))
    profile = _query_profile(question, significant_words)
    query_terms = profile["terms"]

    score = 0.0
    overlap = query_terms & var_terms
    score += len(overlap) * 5.0

    if intent != "general":
        strict_family = _filter_vars_by_intent([var], intent, strict=True)
        soft_family = _filter_vars_by_intent([var], intent, strict=False)
        if strict_family:
            score += 20.0
        elif soft_family:
            score += 10.0
        else:
            score -= 12.0

    sector_terms = profile["sector_terms"]
    energy_terms = profile["energy_terms"]
    metric_terms = profile["metric_terms"]

    sector_map = {
        "transport": ["transport", "transportation", "mobility", "vehicle"],
        "industry": ["industry", "industrial", "steel", "cement"],
        "buildings": ["building", "buildings", "residential", "commercial", "heating", "cooling"],
        "power": ["electricity", "power", "generation", "grid"],
        "afolu": ["afolu", "land", "agriculture", "forestry", "lulucf"],
    }
    energy_map = {
        "renewables": ["renewable", "renewables", "solar", "wind", "hydro", "biomass", "bioenergy", "geothermal"],
        "solar": ["solar", "pv", "photovoltaic"],
        "wind": ["wind", "onshore", "offshore"],
        "oil": ["oil", "petroleum", "liquids"],
        "gas": ["gas", "methane", "lng"],
        "hydrogen": ["hydrogen"],
        "nuclear": ["nuclear"],
        "hydro": ["hydro", "hydropower"],
        "coal": ["coal"],
        "bioenergy": ["biomass", "bioenergy", "biofuel"],
    }
    metric_map = {
        "emissions": ["emission", "emissions", "co2", "carbon", "ch4", "methane", "ghg"],
        "capacity": ["capacity", "installed"],
        "generation": ["generation", "electricity", "power", "secondary energy"],
        "demand": ["demand", "consumption", "final energy", "useful energy"],
        "supply": ["supply", "primary energy", "secondary energy", "production"],
        "investment": ["investment", "capital", "spending"],
        "price": ["price", "cost"],
        "share": ["share", "fraction"],
        "gdp": ["gdp", "gross domestic product"],
    }

    for name, tokens in sector_map.items():
        has_in_query = name in sector_terms
        has_in_var = any(token in var_lower for token in tokens)
        if has_in_query and has_in_var:
            score += 8.0
        elif has_in_query and not has_in_var:
            score -= 4.0
        elif not has_in_query and has_in_var and sector_terms:
            score -= 3.0

    for name, tokens in energy_map.items():
        has_in_query = name in energy_terms
        has_in_var = any(token in var_lower for token in tokens)
        if has_in_query and has_in_var:
            score += 8.0
        elif has_in_query and not has_in_var:
            score -= 4.0
        elif not has_in_query and has_in_var and energy_terms:
            score -= 2.0

    for name, tokens in metric_map.items():
        has_in_query = name in metric_terms
        has_in_var = any(token in var_lower for token in tokens)
        if has_in_query and has_in_var:
            score += 9.0
        elif has_in_query and not has_in_var:
            score -= 5.0
        elif not has_in_query and has_in_var and metric_terms:
            score -= 2.0

    if "emissions" in metric_terms and "co2" in query_terms:
        if "emissions|co2" in var_lower or var_lower.startswith("gross emissions|co2"):
            score += 10.0
        elif "co2" in var_lower:
            score += 4.0
        else:
            score -= 8.0

    if "methane" in query_terms or "ch4" in query_terms:
        if "methane" in var_lower or "ch4" in var_lower:
            score += 12.0
        else:
            score -= 8.0

    if "generation" in metric_terms and "investment" in var_lower and "investment" not in metric_terms:
        score -= 10.0
    if "capacity" in metric_terms and "generation" in var_lower and "capacity" not in var_lower:
        score -= 5.0
    if "demand" in metric_terms and "supply" in var_lower and "demand" not in var_lower:
        score -= 4.0
    if "investment" not in metric_terms and "investment" in var_lower:
        score -= 12.0
    if "price" not in metric_terms and "price" in var_lower:
        score -= 10.0
    if "share" in metric_terms and "investment" in var_lower and "investment" not in metric_terms:
        score -= 14.0
    if "renewables" in energy_terms and "investment" in var_lower and "investment" not in metric_terms:
        score -= 10.0
    if "renewables" in energy_terms and any(token in var_lower for token in ["primary energy", "secondary energy", "electricity"]):
        score += 8.0

    # Broad electricity requests should not drift into investment/price families.
    if "electricity" in profile["metric_terms"] or "power" in profile["sector_terms"]:
        if not (metric_terms & {"investment", "price", "capacity", "generation", "demand", "supply"}):
            if any(token in var_lower for token in ["investment", "price", "cost"]):
                score -= 18.0
    if intent == "electricity" and "investment" not in metric_terms and "investment" in var_lower:
        score -= 14.0

    # If the query names a specific fuel or technology, heavily penalize variables that miss it.
    named_energy_terms = set(profile["energy_terms"])
    if named_energy_terms:
        missing_named = [
            term for term in named_energy_terms
            if term in {"solar", "wind", "oil", "gas", "hydrogen", "nuclear", "hydro", "coal", "bioenergy"}
            and not any(token in var_lower for token in energy_map[term])
        ]
        score -= 12.0 * len(missing_named)

    conflicting_fuels = {
        "oil": {"gas", "coal", "hydrogen", "bioenergy"},
        "gas": {"oil", "coal", "hydrogen", "bioenergy"},
        "coal": {"oil", "gas", "hydrogen", "bioenergy"},
        "hydrogen": {"oil", "gas", "coal", "bioenergy"},
    }
    for primary, conflicts in conflicting_fuels.items():
        if primary in named_energy_terms and not any(token in var_lower for token in energy_map[primary]):
            if any(conflict in var_lower for conflict in conflicts):
                score -= 10.0

    if "buildings" in profile["sector_terms"]:
        if any(token in var_lower for token in ["final energy", "lighting", "appliances", "space heating", "space cooling"]):
            score += 8.0
        if "hydrogen" in var_lower:
            score -= 6.0
    if "transport" in profile["sector_terms"] and "emissions" in profile["metric_terms"]:
        if "emission" in var_lower or "co2" in var_lower:
            score += 10.0
        elif "demand|" in var_lower:
            score -= 4.0

    if "solar" in question.lower():
        if "capacity" in question.lower() and "capacity|electricity|solar" in var_lower:
            score += 15.0
            if "capacity additions|electricity|solar" in var_lower and not any(
                token in question.lower() for token in ["addition", "additions", "new capacity", "build rate", "annual build"]
            ):
                score -= 18.0
        if any(token in question.lower() for token in ["energy", "electricity", "power", "generation"]):
            if "secondary energy|electricity|solar" in var_lower or "generation|electricity|solar" in var_lower:
                score += 14.0
            elif "capacity|electricity|solar" in var_lower:
                score += 8.0
        if "investment" in var_lower:
            score -= 10.0

    if "oil" in question.lower():
        if any(token in question.lower() for token in ["demand", "consumption", "energy", "use"]):
            if any(token in var_lower for token in ["final energy", "primary energy", "secondary energy", "demand"]):
                score += 14.0
            if "electricity|oil" in var_lower and "electricity" not in question.lower():
                score -= 6.0
        if "investment" in var_lower:
            score -= 8.0

        # With only a fuel name, lead the clarification with the broad fuel
        # balance rather than a narrow technology such as oil-fired capacity.
        # The user can still choose the narrower candidates that follow.
        if re.fullmatch(r"\s*oil\s*[?!.]*\s*", question, flags=re.IGNORECASE):
            if var_lower == "primary energy|oil":
                score += 18.0
            elif var_lower.startswith(("primary energy|oil|", "final energy|oil", "final energy|liquids")):
                score += 8.0
            elif "capacity|electricity|oil" in var_lower:
                score -= 6.0

    if "electricity" in question.lower() and not (
        set(profile["energy_terms"]) & {"solar", "wind", "oil", "gas", "hydrogen", "nuclear", "hydro", "coal", "bioenergy"}
    ):
        if any(token in var_lower for token in ["|solar", "|wind", "|hydro", "|nuclear", "|oil", "|gas", "|coal", "|hydrogen", "|bioenergy"]):
            score -= 10.0

    if profile["broad_metric"] and profile["broad_sector"]:
        score -= max(0, var.count("|") - 2) * 1.5
    elif profile["broad_sector"]:
        score -= max(0, var.count("|") - 3) * 1.0

    top_family = var.split("|", 1)[0].lower()
    friendly_families = {
        "emissions": 6.0,
        "gross emissions": 5.0,
        "final energy": 5.0,
        "primary energy": 4.0,
        "secondary energy": 5.0,
        "capacity": 5.0,
        "investment": 3.0,
        "price": 2.0,
        "trade": 1.0,
    }
    if top_family in friendly_families:
        score += friendly_families[top_family]

    if not any(term in query_terms for term in {"export", "import", "trade", "forcing"}):
        if any(token in var_lower for token in ["export", "import", "forcing"]):
            score -= 10.0
    if "trade" not in query_terms and top_family == "trade":
        score -= 6.0
    if prefer_with_years:
        score += 1.0

    return score


def _variable_matches_query_signal(
    variable: str,
    question: str,
    intent: str,
    significant_words: list | None = None,
) -> bool:
    if _is_capacity_additions_mismatch(question, variable):
        return False
    score = _variable_relevance_score(variable, question, intent, significant_words)
    if intent == "general":
        return score >= 8
    return score >= 12


def sanitize_variable_for_query(variable, question: str, intent: str | None = None):
    """Shared guard for extracted variables: return the variable when it is
    consistent with explicit cues in the question, else None.

    Single source of truth for the keyword sanity checks previously duplicated
    in the manager and the plotting agent.
    """
    var = str(variable or "")
    if not var:
        return None
    ql = str(question or "").lower()
    vl = var.lower()
    if any(t in ql for t in ("co2", "emission", "emissions")) and not ("co2" in vl or "emission" in vl):
        return None
    if "solar" in ql and "solar" not in vl:
        return None
    if "wind" in ql and "wind" not in vl:
        return None
    if "capacity" in ql and "capacity" not in vl:
        return None
    # An energy question (final/primary/secondary) must never resolve to an
    # emissions variable just because its name contains "Energy".
    if (
        any(t in ql for t in ("final energy", "primary energy", "secondary energy"))
        and "emission" not in ql
        and "emission" in vl
    ):
        return None
    if intent is None:
        intent = _infer_variable_intent(question)
    if not _variable_matches_query_signal(var, question, intent):
        return None
    return var


def _is_capacity_additions_mismatch(question: str, variable: str | None) -> bool:
    ql = str(question or "").lower()
    vl = str(variable or "").lower()
    if "capacity additions" not in vl:
        return False
    if "capacity" not in ql:
        return False
    return not any(
        token in ql
        for token in ["addition", "additions", "new capacity", "build rate", "annual build"]
    )


def _clean_label_text(text: str) -> str:
    cleaned = str(text or "")
    replacements = {
        "commerciall": "commercial",
        "residential and commercial": "buildings",
        "transportation": "transport",
    }
    for src, dst in replacements.items():
        cleaned = re.sub(rf"\b{re.escape(src)}\b", dst, cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("|", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _describe_variable_option(variable: str) -> str:
    v = str(variable or "")
    lower = v.lower()
    parts = []

    if "emissions|co2" in lower or lower.startswith("gross emissions|co2"):
        parts.append("CO2 emissions")
    elif "emissions|ch4" in lower or "methane" in lower:
        parts.append("methane emissions")
    elif "emissions" in lower:
        parts.append("emissions")
    elif "final energy" in lower:
        parts.append("final energy use")
    elif "primary energy" in lower:
        parts.append("primary energy")
    elif "secondary energy|electricity" in lower:
        parts.append("electricity output")
    elif "secondary energy" in lower:
        parts.append("secondary energy")
    elif "capacity|electricity" in lower:
        parts.append("power capacity")
    elif "capacity" in lower:
        parts.append("capacity")
    elif "investment" in lower:
        parts.append("investment")
    elif "price" in lower:
        parts.append("price")
    elif "demand" in lower:
        parts.append("demand")

    if "transport" in lower or "transportation" in lower:
        parts.append("transport")
    elif "industry" in lower or "industrial" in lower:
        parts.append("industry")
    elif any(token in lower for token in ["building", "residential", "commercial"]):
        parts.append("buildings")
    elif "afolu" in lower or "land" in lower:
        parts.append("land use")

    if "solar" in lower:
        parts.append("solar")
    elif "wind" in lower:
        parts.append("wind")
    elif "oil" in lower:
        parts.append("oil")
    elif "gas" in lower:
        parts.append("gas")
    elif "hydrogen" in lower:
        parts.append("hydrogen")
    elif "electricity" in lower and "output" not in " ".join(parts):
        parts.append("electricity")

    if not parts:
        return ""

    seen = []
    for part in parts:
        if part and part not in seen:
            seen.append(_clean_label_text(part))
    return ", ".join(seen[:3])


def _describe_choice_option(kind: str, option: str) -> str:
    if kind == "variable":
        return _describe_variable_option(option)
    if kind == "region":
        opt = str(option or "")
        pretty = format_region_label(opt)
        if pretty != opt:
            return pretty
        return ""
    if kind == "scenario":
        opt = str(option or "")
        lower = opt.lower()
        labels = []
        if "baseline" in lower or lower.endswith("_bau") or lower == "bau":
            labels.append("baseline")
        if "curpol" in lower or "current policy" in lower:
            labels.append("current policy")
        if "ndc" in lower:
            labels.append("NDC")
        if "nze" in lower or "net-zero" in lower or "net zero" in lower:
            labels.append("net zero")
        if "1.5" in lower:
            labels.append("1.5C pathway")
        if "ssp" in lower:
            m = re.search(r"(ssp\d)", lower)
            labels.append(m.group(1).upper() if m else "SSP")
        deduped = []
        for label in labels:
            if label and label not in deduped:
                deduped.append(label)
        return ", ".join(deduped[:2])
    return ""


def _preferred_family_matches(question: str, available_vars: set[str]) -> list[str]:
    """
    Hand-tuned family preferences for common plain-language phrases.
    This keeps high-signal queries like "solar energy" and "oil demand"
    in the right variable family instead of drifting to weaker matches.
    """
    ql = (question or "").lower()
    candidates: list[str] = []

    if "solar" in ql:
        if any(token in ql for token in ["capacity", "data"]):
            candidates.extend(
                v for v in available_vars
                if "solar" in v.lower() and "capacity|electricity" in v.lower()
                and "additions" not in v.lower()
            )
        if any(token in ql for token in ["energy", "electricity", "power", "generation", "data"]):
            candidates.extend(
                v for v in available_vars
                if "solar" in v.lower()
                and (
                    "secondary energy|electricity" in v.lower()
                    or "generation|electricity" in v.lower()
                    or "capacity|electricity" in v.lower()
                )
                and "investment" not in v.lower()
                and "additions" not in v.lower()
            )

    if "wind" in ql:
        if any(token in ql for token in ["capacity", "data"]):
            candidates.extend(
                v for v in available_vars
                if "wind" in v.lower() and "capacity|electricity" in v.lower()
                and "additions" not in v.lower()
            )
        if any(token in ql for token in ["energy", "electricity", "power", "generation", "data"]):
            candidates.extend(
                v for v in available_vars
                if "wind" in v.lower()
                and (
                    "secondary energy|electricity" in v.lower()
                    or "generation|electricity" in v.lower()
                    or "capacity|electricity" in v.lower()
                )
                and "investment" not in v.lower()
                and "additions" not in v.lower()
            )

    if "oil" in ql and any(token in ql for token in ["demand", "consumption", "energy", "use"]):
        candidates.extend(
            v for v in available_vars
            if "oil" in v.lower()
            and any(
                token in v.lower()
                for token in ["final energy", "primary energy", "secondary energy", "demand"]
            )
            and "investment" not in v.lower()
        )

    deduped: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in deduped:
            deduped.append(candidate)
    return deduped


def _broad_electricity_candidates(available_vars: set[str]) -> list[str]:
    """
    Return broad electricity-family variables for generic electricity questions.
    Prefer high-level output, use, capacity, and emissions variables over
    technology- or sector-specific branches.
    """
    preferred_order = [
        "Secondary Energy|Electricity",
        "Final Energy|Electricity",
        "Capacity|Electricity",
        "Emissions|CO2|Energy|Supply|Electricity",
        "Emissions|CO2|Electricity",
    ]

    candidates: list[str] = []
    for name in preferred_order:
        if name in available_vars and name not in candidates:
            candidates.append(name)

    if candidates:
        return candidates

    blocked_tokens = {
        "transport", "transportation", "passenger", "freight",
        "residential", "commercial", "industry", "other sector",
        "solar", "wind", "hydro", "nuclear", "oil", "gas", "coal",
        "biomass", "bioenergy", "geothermal", "hydrogen",
        "investment", "price", "cost", "capital", "sequestration", "additions",
    }

    fallback = []
    for variable in sorted(available_vars):
        lower = variable.lower()
        if "electricity" not in lower:
            continue
        if any(token in lower for token in blocked_tokens):
            continue
        fallback.append(variable)

    return fallback[:3]


def _rank_scored_candidates(
    candidates: list[str],
    question: str,
    intent: str,
    significant_words: list | None = None,
    popularity: dict[str, int] | None = None,
    limit: int = 3,
) -> list[str]:
    seen: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.append(candidate)
    ranked = sorted(
        seen,
        key=lambda var: (
            -(
                _variable_relevance_score(var, question, intent, significant_words)
                + min((popularity or {}).get(var, 0), 8) * 0.75
            ),
            var.count("|"),
            len(var),
            var,
        ),
    )
    return ranked[:limit]


def _filter_vars_by_intent(variables: set[str] | list[str], intent: str, strict: bool = False) -> list[str]:
    items = [v for v in variables if v]
    if not items:
        return []

    def _prioritize(matches: list[str]) -> list[str]:
        if strict:
            return matches
        remainder = [v for v in items if v not in matches]
        return matches + remainder

    if intent == "emissions_co2":
        co2 = [v for v in items if ("co2" in v.lower() and "emission" in v.lower()) or v.lower().startswith("gross emissions|co2")]
        emissions = [v for v in items if "emission" in v.lower()]
        preferred = co2 if co2 else emissions
        if not preferred:
            return []
        return _prioritize(preferred)
    if intent == "methane":
        meth = [v for v in items if "ch4" in v.lower() or "methane" in v.lower()]
        if not meth:
            return []
        return _prioritize(meth)
    if intent == "solar":
        solar = [v for v in items if "solar" in v.lower() or "pv" in v.lower()]
        if not solar:
            return []
        preferred = [
            v for v in solar
            if any(k in v.lower() for k in ["generation", "electricity", "secondary energy", "capacity"])
            and "investment" not in v.lower()
        ]
        if preferred:
            return _prioritize(preferred)
        return _prioritize(solar)
    if intent == "wind":
        wind = [v for v in items if "wind" in v.lower()]
        return _prioritize(wind)
    if intent == "capacity":
        cap = [v for v in items if "capacity" in v.lower()]
        return _prioritize(cap)
    if intent == "electricity":
        elec = [v for v in items if "electricity" in v.lower() or "power" in v.lower() or "generation" in v.lower()]
        return _prioritize(elec)
    if intent == "demand":
        demand = [v for v in items if "demand" in v.lower()]
        return _prioritize(demand)
    if intent == "supply":
        supply = [v for v in items if "supply" in v.lower()]
        return _prioritize(supply)
    if intent == "investment":
        inv = [v for v in items if "investment" in v.lower()]
        return _prioritize(inv)
    if intent == "price":
        price = [v for v in items if "price" in v.lower() or "cost" in v.lower()]
        return _prioritize(price)
    if intent == "gdp":
        gdp = [v for v in items if "gdp" in v.lower() or "gross domestic product" in v.lower()]
        return _prioritize(gdp)
    if intent == "share":
        share = [v for v in items if "share" in v.lower() or "fraction" in v.lower()]
        non_investment = [v for v in share if "investment" not in v.lower()]
        if non_investment:
            return _prioritize(non_investment)
        return _prioritize(share)
    if intent == "renewables":
        renewables = [
            v for v in items
            if any(k in v.lower() for k in ["renewable", "renewables", "solar", "wind", "hydro", "bio", "geothermal"])
        ]
        if any("share" in v.lower() or "fraction" in v.lower() for v in renewables):
            renewable_shares = [
                v for v in renewables
                if ("share" in v.lower() or "fraction" in v.lower()) and "investment" not in v.lower()
            ]
            if renewable_shares:
                return _prioritize(renewable_shares)
        return _prioritize(renewables)
    if intent == "transport":
        transport = [v for v in items if "transport" in v.lower()]
        emissions_transport = [
            v for v in transport
            if "emission" in v.lower() or "co2" in v.lower() or "carbon" in v.lower()
        ]
        if emissions_transport:
            return _prioritize(emissions_transport)
        return _prioritize(transport)
    if intent == "industry":
        industry = [v for v in items if "industry" in v.lower() or "industrial" in v.lower()]
        return _prioritize(industry)
    if intent == "buildings":
        bld = [v for v in items if "building" in v.lower() or "residential" in v.lower() or "commercial" in v.lower()]
        energy_buildings = [
            v for v in bld
            if any(k in v.lower() for k in ["energy", "heating", "cooling", "electricity", "demand"])
        ]
        preferred = [
            v for v in energy_buildings
            if any(k in v.lower() for k in ["final energy", "space heating", "space cooling", "lighting", "appliances"])
            and "hydrogen" not in v.lower()
        ]
        if preferred:
            return _prioritize(preferred)
        if energy_buildings:
            return _prioritize(energy_buildings)
        return _prioritize(bld)
    if intent == "oil":
        oil = [v for v in items if "oil" in v.lower()]
        if not oil:
            return []
        preferred = [
            v for v in oil
            if any(k in v.lower() for k in ["energy", "electricity", "secondary energy", "final energy", "primary energy", "emission"])
        ]
        if preferred:
            return _prioritize(preferred)
        return _prioritize(oil)
    if intent == "gas":
        gas = [v for v in items if "gas" in v.lower()]
        return _prioritize(gas)
    if intent == "hydrogen":
        h2 = [v for v in items if "hydrogen" in v.lower()]
        return _prioritize(h2)
    if intent == "nuclear":
        nuc = [v for v in items if "nuclear" in v.lower()]
        return _prioritize(nuc)
    if intent == "hydro":
        hyd = [v for v in items if "hydro" in v.lower()]
        return _prioritize(hyd)
    return items


def _suggest_variable_candidates(
    question: str,
    variable_dict: dict,
    ts_data: list,
    ranked_vars: list | None = None,
    significant_words: list | None = None,
    region: str | None = None,
    scenario: str | None = None,
    model: str | None = None,
    ignored_values: list[str] | None = None,
    limit: int = 3,
) -> list[str]:
    """
    Return context-aware variable candidates:
    - prioritize variables that have year data in the current scope
    - keep emissions queries in the emissions/CO2 family
    """
    intent = _infer_variable_intent(question, significant_words)
    scoped_vars = {
        str(r.get('variable', '')).strip()
        for r in ts_data
        if r and r.get('variable')
        and _record_has_year_data(r)
        and (not region or regions_equivalent(r.get('region'), region))
        and (not scenario or r.get('scenario') == scenario)
        and (not model or r.get('modelName') == model)
    }
    if not scoped_vars:
        scoped_vars = {
            str(r.get('variable', '')).strip()
            for r in ts_data
            if r and r.get('variable') and _record_has_year_data(r)
        }
    if not scoped_vars:
        scoped_vars = {
            str(r.get('variable', '')).strip()
            for r in ts_data
            if r and r.get('variable')
        }

    preferred_family = _preferred_family_matches(question, scoped_vars)
    if preferred_family:
        return _rank_scored_candidates(
            preferred_family,
            question,
            intent,
            significant_words=significant_words,
            limit=limit,
        )

    popularity = {}
    for record in ts_data:
        if not record or not record.get('variable'):
            continue
        variable = str(record.get('variable', '')).strip()
        if variable not in scoped_vars:
            continue
        if region and not regions_equivalent(record.get('region'), region):
            continue
        if scenario and record.get('scenario') != scenario:
            continue
        if model and record.get('modelName') != model:
            continue
        if _record_has_year_data(record):
            popularity[variable] = popularity.get(variable, 0) + 1

    strict_scoped_vars = _filter_vars_by_intent(scoped_vars, intent, strict=True)
    if strict_scoped_vars:
        scoped_vars = set(strict_scoped_vars)
    elif intent != "general":
        # A specific semantic request with no matching family is uncertainty,
        # not permission to recommend an unrelated popular variable. Returning
        # no candidates lets the caller ask for clarification truthfully.
        return []
    else:
        scoped_vars = set(_filter_vars_by_intent(scoped_vars, intent, strict=False))

    candidates = _rank_variable_candidates(
        question,
        variable_dict,
        scoped_vars,
        ranked_vars=ranked_vars,
        significant_words=significant_words,
        ignored_values=ignored_values,
        limit=max(limit, 6),
    )
    if not candidates:
        candidates = list(scoped_vars)
    return _rank_scored_candidates(
        candidates,
        question,
        intent,
        significant_words=significant_words,
        popularity=popularity,
        limit=limit,
    )


def _unmatched_variable_qualifiers(
    question: str,
    candidates: list[str],
    *,
    region: str | None = None,
    scenario: str | None = None,
    model: str | None = None,
    ignored_values: list[str] | None = None,
) -> list[str]:
    """Return meaningful query modifiers absent from all candidate names.

    This is a confidence explanation, not a domain alias table.  It allows the
    chatbot to say that a modifier was not represented in the runtime variable
    catalogue instead of presenting merely related variables as exact matches.
    """
    if not candidates:
        return []
    terms = set(_query_terms(question))
    candidate_terms = set().union(*(_tokenize_text(candidate) for candidate in candidates))
    for scoped_value in (region, scenario, model, *(ignored_values or [])):
        terms -= _tokenize_text(str(scoped_value or ""))
    if region:
        terms -= _tokenize_text(format_region_label(region))
    terms = {term for term in terms if not term.isdigit()}
    # These describe the requested measurement operation rather than a
    # taxonomy qualifier. Their exact spelling is often absent from IAMC-style
    # variable paths even when the candidate is compatible.
    measurement_terms = {
        "produce", "produced", "production", "generate", "generated", "generation",
        "supply", "demand", "consumption", "output", "total", "amount", "level",
        "capacity", "emission", "emissions", "energy", "electricity", "power",
    }
    terms -= measurement_terms
    return sorted(term for term in terms if term not in candidate_terms)


def _suggest_recovery_variables(
    question: str,
    ts_data: list,
    significant_words: list | None = None,
    region: str | None = None,
    scenario: str | None = None,
    model: str | None = None,
    limit: int = 3,
) -> list[str]:
    """
    Suggest fallback variables after a user-selected choice returns no data.
    Priority: same region+scenario, then same region, then same scenario, then model/global.
    """
    intent = _infer_variable_intent(question, significant_words)

    def _pool(rg: bool, sc: bool) -> set[str]:
        return {
            str(r.get('variable', '')).strip()
            for r in ts_data
            if r and r.get('variable') and _record_has_year_data(r)
            and (not model or r.get('modelName') == model)
            and (not rg or (region and regions_equivalent(r.get('region'), region)))
            and (not sc or (scenario and r.get('scenario') == scenario))
        }

    pools = []
    if region and scenario:
        pools.append(_pool(True, True))
    if region:
        pools.append(_pool(True, False))
    if scenario:
        pools.append(_pool(False, True))
    pools.append(_pool(False, False))

    excluded = {str(v).strip() for v in re.findall(r"`([^`]+)`", question or "")}

    for pool in pools:
        if not pool:
            continue
        popularity = {}
        for record in ts_data:
            if not record or not record.get('variable'):
                continue
            variable = str(record.get('variable', '')).strip()
            if variable not in pool:
                continue
            if model and record.get('modelName') != model:
                continue
            if region and not regions_equivalent(record.get('region'), region) and scenario and record.get('scenario') != scenario:
                continue
            if _record_has_year_data(record):
                popularity[variable] = popularity.get(variable, 0) + 1
        filtered_pool = _filter_vars_by_intent(pool, intent, strict=True)
        if not filtered_pool:
            filtered_pool = _filter_vars_by_intent(pool, intent, strict=False)
            if not filtered_pool:
                continue
        ranked = _rank_variable_candidates(
            question,
            variable_dict,
            set(filtered_pool),
            ranked_vars=None,
            significant_words=significant_words,
            limit=max(limit, 6),
        )
        ranked = [candidate for candidate in ranked if candidate not in excluded]
        if not ranked:
            ranked = [candidate for candidate in filtered_pool if candidate not in excluded]
        if ranked:
            return _rank_scored_candidates(
                ranked,
                question,
                intent,
                significant_words=significant_words,
                popularity=popularity,
                limit=limit,
            )

    return []


def _starter_variable_candidates(
    ts_data: list,
    region: str | None = None,
    scenario: str | None = None,
    model: str | None = None,
    limit: int = 3,
) -> list[str]:
    region_aliases = {
        "EU": ["EU", "EU28", "EU-27", "EU27", "EU-12", "EU-15"],
        "EU28": ["EU28", "EU", "EU-27", "EU27", "EU-12", "EU-15"],
        "World": ["World", "Global"],
        "IND": ["IND", "India"],
        "CHN": ["CHN", "China"],
        "USA": ["USA", "United States"],
    }
    preferred_patterns = [
        "Emissions|CO2",
        "Gross Emissions|CO2",
        "Final Energy",
        "Primary Energy",
        "Secondary Energy|Electricity",
        "Capacity|Electricity",
    ]
    fallback_family_prefixes = [
        "Emissions|CO2",
        "Gross Emissions|CO2",
        "Final Energy",
        "Primary Energy",
        "Secondary Energy|Electricity",
        "Capacity|Electricity",
        "Secondary Energy",
    ]

    def _collect_available(region_scope: list[str] | None = None) -> list[str]:
        collected = []
        for record in ts_data:
            if not record or not record.get("variable"):
                continue
            if region_scope and not any(
                regions_equivalent(record.get("region"), candidate)
                for candidate in region_scope
            ):
                continue
            if scenario and record.get("scenario") != scenario:
                continue
            if model and record.get("modelName") != model:
                continue
            collected.append(str(record.get("variable", "")).strip())
        return collected

    available = []
    if region:
        candidate_regions = region_aliases.get(region, [region])
        available = _collect_available(candidate_regions)
    if not available:
        available = _collect_available()

    deduped = []
    for pattern in preferred_patterns:
        matches = sorted({
            variable for variable in available
            if variable == pattern or variable.startswith(pattern + "|")
        }, key=lambda var: (var.count("|"), len(var), var))
        for match in matches:
            if match not in deduped:
                deduped.append(match)
            if len(deduped) >= limit:
                return deduped[:limit]

    broad_family_matches = []
    for prefix in fallback_family_prefixes:
        matches = sorted({
            variable for variable in available
            if variable == prefix or variable.startswith(prefix + "|")
        }, key=lambda var: (var.count("|"), len(var), var))
        broad_family_matches.extend(matches)

    for match in broad_family_matches:
        if match not in deduped:
            deduped.append(match)
        if len(deduped) >= limit:
            return deduped[:limit]

    return deduped[:limit]


def _choice_prompt(prefix: str, kind: str, options: list[str]) -> str:
    """
    Build a concise numbered-choice prompt for variables/regions/scenarios.
    """
    clean_options = [opt for opt in options if opt]
    if not clean_options:
        return f"{prefix} Please provide the {kind} you want."

    rendered_options = []
    for idx, opt in enumerate(clean_options, start=1):
        label = _describe_choice_option(kind, opt)
        if label:
            rendered_options.append(f"{idx}. `{opt}` ({label})")
        else:
            rendered_options.append(f"{idx}. `{opt}`")
    option_text = " ".join(rendered_options)
    return (
        f"{prefix} Choose the {kind}: {option_text} "
        f"Reply with a number (1-{len(clean_options)}), or `yes` for option 1."
    )


def _compact_recovery_prompt(
    prefix: str,
    variable_options: list[str] | None = None,
    region_options: list[str] | None = None,
    scenario_options: list[str] | None = None,
) -> str:
    def _no_data_reason() -> str:
        if scenario_options:
            return "Reason: the exact scenario combination is not available for this data slice."
        if region_options:
            return "Reason: the exact region combination is not available for this data slice."
        if variable_options:
            return "Reason: the requested variable is unavailable in the current scope."
        return "Reason: the exact variable, region, scenario, or model combination is unavailable."

    def _standard_no_data_prefix(text: str) -> str:
        clean = str(text or "").strip()
        patterns = [
            (
                r"No data found for \*\*(?P<variable>[^*]+)\*\* in region `(?P<region>[^`]+)` under scenario `(?P<scenario>[^`]+)`\.?",
                lambda m: (
                    f"I could not find data for `{m.group('variable')}` in `{m.group('region')}` "
                    f"under `{m.group('scenario')}`."
                ),
            ),
            (
                r"No data found for `(?P<variable>[^`]+)` in region `(?P<region>[^`]+)` under scenario `(?P<scenario>[^`]+)`\.?",
                lambda m: (
                    f"I could not find data for `{m.group('variable')}` in `{m.group('region')}` "
                    f"under `{m.group('scenario')}`."
                ),
            ),
            (
                r"No data found for \*\*(?P<variable>[^*]+)\*\* in region `(?P<region>[^`]+)`\.?",
                lambda m: f"I could not find data for `{m.group('variable')}` in `{m.group('region')}`.",
            ),
            (
                r"No data found for \*\*(?P<variable>[^*]+)\*\* in model `(?P<model>[^`]+)`\.?",
                lambda m: f"I could not find data for `{m.group('variable')}` using model `{m.group('model')}`.",
            ),
        ]
        for pattern, renderer in patterns:
            match = re.search(pattern, clean)
            if match:
                return renderer(match)
        return clean

    option_rows: list[tuple[str, str]] = []
    for kind, options in (
        ("variable", variable_options or []),
        ("region", region_options or []),
        ("scenario", scenario_options or []),
    ):
        for option in options[:3]:
            if option:
                option_rows.append((kind, str(option)))
            if len(option_rows) >= 3:
                break
        if len(option_rows) >= 3:
            break

    lines = [_standard_no_data_prefix(prefix)]
    if lines[0].lower().startswith(("i could not find data", "no data found")):
        lines.append(_no_data_reason())
    if option_rows:
        lines.append("Closest valid options:")
        for idx, (kind, option) in enumerate(option_rows, start=1):
            label = _describe_choice_option(kind, option)
            reason = f" ({label})" if label else ""
            lines.append(f"{idx}. {kind} `{option}`{reason}")
        lines.append(f"Reply with `1`, `2`, or `3`, or type the {option_rows[0][0]} you want.")
    else:
        lines.append("Reply with a different variable, region, or scenario to continue.")
    return "\n\n".join(lines)


def _matrix_recovery_prompt(
    metadata: Any | None,
    prefix: str,
    variable: str | None = None,
    region: str | None = None,
    scenario: str | None = None,
    model: str | None = None,
) -> str | None:
    if not metadata:
        return None

    def _same_family_variable_options() -> list[str]:
        if not variable:
            return []
        base = str(variable or "").strip()
        base_lower = base.lower()
        all_variables = sorted(getattr(metadata, "all_variables", set()) or [])
        if not all_variables:
            return []

        def in_scope(candidate: str) -> bool:
            # Only keep candidates that actually have data in the requested
            # region (and scenario), so a selected option never dead-ends.
            if region:
                candidate_regions = getattr(metadata, "variable_regions", {}).get(candidate, set())
                matching_regions = [
                    value for value in candidate_regions
                    if regions_equivalent(value, region)
                ]
                if not matching_regions:
                    return False
                if scenario:
                    region_map = getattr(metadata, "availability_matrix", {}).get(candidate, {})
                    if not any(scenario in region_map.get(value, {}) for value in matching_regions):
                        return False
            elif scenario:
                if scenario not in getattr(metadata, "variable_scenarios", {}).get(candidate, set()):
                    return False
            return True

        def valid(candidate: str) -> bool:
            cand_lower = candidate.lower()
            if candidate == base:
                return False
            if not in_scope(candidate):
                return False
            if "solar" in base_lower:
                return "solar" in cand_lower and "investment" not in cand_lower and "additions" not in cand_lower
            if "wind" in base_lower:
                return "wind" in cand_lower and "investment" not in cand_lower and "additions" not in cand_lower
            if base_lower.startswith("gdp"):
                return cand_lower.startswith("gdp")
            if base_lower.startswith("emissions|co2"):
                return cand_lower.startswith("emissions|co2")
            family = base.split("|", 1)[0].lower()
            return bool(family and cand_lower.startswith(family))

        return [candidate for candidate in all_variables if valid(candidate)][:3]

    options = metadata.suggest_valid_options(
        variable=variable,
        region=region,
        scenario=scenario,
        model=model,
        limit=3,
    )
    variable_options = [opt for opt in options.get("variables", []) if opt != variable]
    if variable and not variable_options:
        variable_options = _same_family_variable_options()
    elif not variable_options:
        variable_options = metadata.suggest_valid_options(
            region=region,
            scenario=scenario,
            model=model,
            limit=3,
        ).get("variables", [])
        variable_options = [opt for opt in variable_options if opt != variable]

    region_options = [
        opt for opt in options.get("regions", [])
        if not region or not regions_equivalent(opt, region)
    ]
    if hasattr(metadata, "suggest_scenarios_by_scope"):
        scenario_options = metadata.suggest_scenarios_by_scope(
            variable=variable,
            region=region,
            model=model,
            exclude=scenario,
            limit=3,
        )
    else:
        scenario_options = [opt for opt in options.get("scenarios", []) if opt != scenario]
    if not (variable_options or region_options or scenario_options):
        return None

    return _compact_recovery_prompt(
        prefix,
        variable_options=variable_options,
        region_options=region_options,
        scenario_options=scenario_options,
    )


def _scenario_family_recovery_prompt(
    ts_data: list,
    *,
    variable: str,
    region: str | None,
    scenario: str | None,
    model: str | None = None,
) -> str | None:
    """Offer only one-dimension changes that lead to a real data slice.

    Canonical scenario families such as ``Net Zero`` expand to several runtime
    codes.  When the requested region has no member of that family, a generic
    availability fallback can recommend a scenario that exists only somewhere
    else, which still fails if selected.  Build the two honest recovery sets
    directly from the records instead:

    * scenarios that contain the requested variable in the requested region;
    * regions that contain the requested variable in the requested family.
    """
    family = str(scenario or "").strip()
    requested_region = str(region or "").strip()
    requested_variable = str(variable or "").strip()
    requested_model = str(model or "").strip()
    if not requested_variable or not _scenario_is_family_label(family):
        return None

    records = [
        record for record in (ts_data or [])
        if record
        and str(record.get("variable", "")).strip() == requested_variable
        and (not requested_model or str(record.get("modelName", "")).strip() == requested_model)
        and _record_has_year_data(record)
    ]
    if not records:
        return None

    def _dedupe_labels(values: set[str]) -> list[str]:
        ordered = sorted(
            values,
            key=lambda value: (
                value.casefold(),
                0 if value[:1].isupper() else 1,
                value,
            ),
        )
        deduped: list[str] = []
        seen: set[str] = set()
        for value in ordered:
            key = value.casefold()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(value)
        return deduped

    compatible_regions = dedupe_equivalent_regions(_dedupe_labels({
        str(record.get("region", "")).strip()
        for record in records
        if str(record.get("region", "")).strip()
        and not regions_equivalent(record.get("region"), requested_region)
        and _scenario_match_ok(record.get("scenario"), family)
    }))

    compatible_scenarios = _dedupe_labels({
        str(record.get("scenario", "")).strip()
        for record in records
        if requested_region
        and regions_equivalent(record.get("region"), requested_region)
        and str(record.get("scenario", "")).strip()
        and not _scenario_match_ok(record.get("scenario"), family)
    })

    if not compatible_regions and not compatible_scenarios:
        return None

    region_label = format_region_label(requested_region) if requested_region else "the requested region"
    model_note = f" for model `{requested_model}`" if requested_model else ""
    return _compact_recovery_prompt(
        (
            f"I could not find data for `{requested_variable}` in region `{region_label}` "
            f"under the requested scenario family `{family}`{model_note}."
        ),
        region_options=compatible_regions[:3],
        scenario_options=compatible_scenarios[:3],
    )


def _confirmation_prompt(prefix: str, best: str, alternatives: list[str] | None = None) -> str:
    """
    Build a concise clarification prompt with numbered options.
    """
    options = [best]
    if alternatives:
        extras = [a for a in alternatives if a and a != best][:2]
        options.extend(extras)
    return _choice_prompt(prefix, "variable", options)


def _renewable_share_prompt(candidates: list[str]) -> str:
    options = [candidate for candidate in candidates if candidate][:3]
    if not options:
        return (
            "I could not find an explicit renewable share variable in the loaded data. "
            "Try `list variables` to see the available renewable and energy variables."
        )
    return _choice_prompt(
        (
            "I could not find an explicit renewable share variable in the loaded data. "
            "These are the closest renewable-energy variables I found."
        ),
        "variable",
        options,
    )


def _format_model_info_answer(model_name: str, record: dict, asks_assumptions: bool = False) -> str:
    from model_profiles import clean_metadata_text
    desc = clean_metadata_text(record.get('description'))
    asum = clean_metadata_text(record.get('assumptions'))
    source = clean_metadata_text(record.get('source'))
    profile = find_model_profile(model_name)
    use_profile = bool(profile and not has_strong_model_metadata(record))

    parts = [f"### {model_name}"]
    if desc:
        parts.append(f"Description:\n{desc}")
    elif profile and clean_metadata_text(profile.get('description')):
        parts.append(f"Description:\n{clean_metadata_text(profile.get('description'))}")
    if asum:
        parts.append(f"Assumptions:\n{asum}")
    elif asks_assumptions:
        assumption_note = str((profile or {}).get("assumptions_note", "") or "").strip()
        if assumption_note:
            parts.append(f"Assumptions:\n{assumption_note}")
        else:
            parts.append("Assumptions:\nNo explicit assumptions field is available in the model metadata.")
    if use_profile:
        sectors = [str(item) for item in profile.get("sectors", []) if item]
        uses = [str(item) for item in profile.get("typical_use_cases", []) if item]
        limitations = [str(item) for item in profile.get("limitations", []) if item]
        if sectors:
            parts.append("Model scope:\n- " + "\n- ".join(sectors))
        if uses:
            parts.append("Useful for:\n- " + "\n- ".join(uses))
        if limitations:
            parts.append("Interpretation notes:\n- " + "\n- ".join(limitations))
    if source:
        parts.append(f"Source:\n{source}")
    model_url = _iamparis_model_url(record) or _iamparis_model_url(profile or {})
    if model_url:
        parts.append(f"Related model documentation:\n- [IAM PARIS model page]({model_url})")
    else:
        parts.append("Related model documentation:\n- [IAM PARIS Models](https://iamparis.eu/models)")

    if len(parts) == 2 and parts[-1].startswith("Related model documentation"):
        return f"I found the model `{model_name}`, but no description was provided in metadata."

    return "\n\n".join(parts)


def data_query(
    question: str,
    model_data: list,
    ts_data: list,
    history: list | None = None,
    forced_entities: dict | None = None,
    metadata: Any | None = None,
    allow_plots: bool = True,
) -> str:
    entities = dict(forced_entities or {})
    if not allow_plots:
        entities["action"] = "query"
        if _looks_like_plot_request(question):
            entry = _matched_workspace(question, _workspace_entries(ts_data)) or _workspace_catalog().get(
                str(entities.get("workspace_code") or ""), {}
            )
            url = entry.get("explorer_url") or "https://iamparis.eu/results"
            return f"Please use the [IAM PARIS data explorer]({url}) to plot these results."
        if any(len(entities.get(key) or []) > 1 for key in ("variables", "regions", "scenarios", "models")):
            return _comparison_tables(question, ts_data, entities)
    answer = _data_query(question, model_data, ts_data, history, entities, metadata, allow_plots)
    if (
        re.search(r"\boil\b", str(question or ""), re.IGNORECASE)
        and "Final Energy|Liquids" in answer
        and not re.search(r"Final Energy\|[^`\n]*Oil", answer, re.IGNORECASE)
    ):
        answer = (
            "Interpretation: the available liquid-fuel demand indicator is broader than oil "
            "and may include other liquid fuels. It is not an oil-only estimate.\n\n" + answer
        )
    return answer


def _comparison_tables(question: str, records: list, entities: dict) -> str:
    """Render a bounded comparison without passing through the plotting resolver."""
    entries = _workspace_entries(records)
    named = _matched_workspace(question, entries)
    workspace = str((named or {}).get("code") or entities.get("workspace_code") or "")
    if not workspace and len(entries) > 1:
        return "Please choose a study before comparing numeric results. Ask `list workspaces` to see the available studies."
    if not workspace and len(entries) == 1:
        workspace = entries[0]["code"]
    selections = {}
    for key in ("variable", "region", "scenario", "model"):
        selections[key] = list(entities.get(key + "s") or ([entities[key]] if entities.get(key) else []))
    if not selections["variable"] or not selections["region"]:
        return "Please provide the variable and region for the numeric comparison."
    scoped = [row for row in records if not workspace or row.get("workspace_code") == workspace]
    for key, choices in selections.items():
        if not choices:
            continue
        field = "modelName" if key == "model" else key
        if key == "region":
            scoped = [row for row in scoped if any(regions_equivalent(row.get(field), value) for value in choices)]
        elif key == "scenario":
            scoped = [row for row in scoped if any(_scenario_match_ok(row.get(field), value) for value in choices)]
        else:
            scoped = [row for row in scoped if row.get(field) in choices]
    text_start, text_end = extract_year_range(question)
    start = entities.get("start_year", text_start)
    end = entities.get("end_year", text_end)

    def _row_values(row: dict) -> dict:
        values = {
            str(key): value for key, value in row.items()
            if str(key).isdigit() and len(str(key)) == 4
        }
        nested = row.get("years")
        if isinstance(nested, dict):
            values.update({
                str(key): value for key, value in nested.items()
                if str(key).isdigit() and len(str(key)) == 4
            })
        return values

    # Each variable/region table chooses its latest year across all series.
    # Missing-member detection must use that same year, not each row's latest.
    table_years = {}
    for variable in selections["variable"]:
        for region in selections["region"]:
            years = {
                year for row in scoped
                if row.get("variable") == variable and regions_equivalent(row.get("region"), region)
                for year, value in _row_values(row).items()
                if is_finite_numeric_value(value)
            }
            table_years[(variable, region)] = select_years(sorted(years, key=int), start, end)

    def _has_value_in_period(row: dict) -> bool:
        values = _row_values(row)
        years = {
            year for (variable, region), selected in table_years.items()
            if row.get("variable") == variable and regions_equivalent(row.get("region"), region)
            for year in selected
        }
        return any(is_finite_numeric_value(values.get(year)) for year in years)

    visible_rows = [row for row in scoped if _has_value_in_period(row)]
    missing = []
    for key, choices in selections.items():
        if not choices:
            continue
        field = "modelName" if key == "model" else key
        if key == "region":
            present = lambda value: any(
                regions_equivalent(row.get(field), value) for row in visible_rows
            )
        elif key == "scenario":
            present = lambda value: any(
                _scenario_match_ok(row.get(field), value) for row in visible_rows
            )
        else:
            present = lambda value: any(
                str(row.get(field) or "") == str(value) for row in visible_rows
            )
        absent = [value for value in choices if not present(value)]
        if absent:
            missing.append((key, absent))

    consume_resolved_scope()
    sections = []
    section_scopes = []
    for variable in selections["variable"]:
        for region in selections["region"]:
            rows = [row for row in scoped if row.get("variable") == variable and regions_equivalent(row.get("region"), region)]
            sections.append(format_time_series_data(rows, variable, region, start, end, workspace))
            scope = consume_resolved_scope()
            if scope:
                section_scopes.append(scope)
            available = [row for row in rows if _has_value_in_period(row)]
            absent_pairs = []
            for scenario in selections["scenario"] or [None]:
                for model in selections["model"] or [None]:
                    if not any(
                        (scenario is None or _scenario_match_ok(row.get("scenario"), scenario))
                        and (model is None or row.get("modelName") == model)
                        for row in available
                    ):
                        labels = []
                        if scenario is not None:
                            labels.append(f"scenario: `{scenario}`")
                        if model is not None:
                            labels.append(f"model: `{model}`")
                        if labels:
                            absent_pairs.append(", ".join(labels))
            if absent_pairs:
                sections.append(
                    f"For `{variable}` in `{region}`, no data is available for the requested period "
                    f"for {'; '.join(absent_pairs)}."
                )

    if section_scopes:
        def _scope_values(singular: str, plural: str) -> list:
            values = []
            for scope in section_scopes:
                candidates = list(scope.get(plural) or [])
                if not candidates and scope.get(singular):
                    candidates = [scope[singular]]
                for value in candidates:
                    if value not in values:
                        values.append(value)
            return values

        variables = _scope_values("variable", "variables")
        regions = _scope_values("region", "regions")
        scenarios = _scope_values("scenario", "scenarios")
        models = _scope_values("model", "result_models")
        units = _scope_values("unit", "units")
        aggregate = {
            "variables": variables,
            "regions": regions,
            "scenarios": scenarios,
            "result_models": models,
            "unit": units[0] if len(units) == 1 else "multiple",
            "start_year": start,
            "end_year": end,
            "action": "query",
            "workspace_code": workspace,
            "comparison": entities.get("comparison") or "",
            "all_scenarios": len(scenarios) > 1,
        }
        for singular, values in (
            ("variable", variables), ("region", regions),
            ("scenario", scenarios), ("model", models),
        ):
            if len(values) == 1:
                aggregate[singular] = values[0]
        record_resolved_scope(**aggregate)
    warning = ""
    if missing:
        details = "; ".join(
            f"{key}: " + ", ".join(f"`{value}`" for value in values)
            for key, values in missing
        )
        warning = (
            "Some requested comparison members have no data in this study for the requested period "
            f"({details}). The table below contains only the available members.\n\n"
        )
    return warning + "\n\n".join(sections)


def _data_query(
    question: str,
    model_data: list,
    ts_data: list,
    history: list | None = None,
    forced_entities: dict | None = None,
    metadata: Any | None = None,
    allow_plots: bool = True,
) -> str:
    """Process a user query about IAM data, optionally returning results or plots."""
    if not question or not isinstance(question, str):
        return "Please provide a valid question."
    q = question.lower()
    forced_entities = dict(forced_entities or {})
    forced_variable = str(forced_entities.get("variable", "") or "").strip()
    forced_region = str(forced_entities.get("region", "") or "").strip()
    forced_scenario = str(forced_entities.get("scenario", "") or "").strip()
    forced_model = str(forced_entities.get("model", "") or "").strip()
    forced_workspace = str(forced_entities.get("workspace_code", "") or "").strip()
    requested_forced_model = forced_model
    forced_start_year = forced_entities.get("start_year")
    forced_end_year = forced_entities.get("end_year")

    def _resolved_year_range() -> tuple[int | None, int | None]:
        """Prefer structured caller scope, falling back to the question text."""
        text_start, text_end = _extract_year_range(question)
        start = forced_start_year if forced_start_year is not None else text_start
        end = forced_end_year if forced_end_year is not None else text_end
        # An open bound stays open. Mirroring one side onto the other turned
        # every open-ended request into a single year: "after 2040" answered for
        # 2041 alone and "until 2080" for 2080 alone, quietly answering a
        # different question than the one asked. `select_years` already treats
        # `None` as unbounded on either side.
        #
        # The mirroring is still right when the question names one year and the
        # phrasing is not open-ended ("in 2050"), but `extract_year_range`
        # already returns both bounds in that case, so nothing to do here.
        if start is not None and end is not None and start > end:
            start, end = end, start
        return start, end

    def _interpretation_notice(variable: str | None) -> str:
        tokens = _token_set(question)
        specific_gas_terms = {
            "co2", "carbon", "dioxide", "ch4", "methane", "n2o", "nitrous",
            "ghg", "greenhouse", "kyoto", "afolu",
        }
        if (
            str(variable or "") == "Emissions|CO2"
            and tokens & {"emission", "emissions", "emisions", "emisson"}
            and not tokens & specific_gas_terms
        ):
            return (
                "Interpretation: I treated bare **emissions** as `Emissions|CO2` "
                "(carbon-dioxide emissions).\n\n"
            )
        return ""
    # Keep all discovery and result selection within a named study when the
    # user selected one earlier in the conversation, or names it directly.
    # Workspace codes remain internal; matching uses the public study title.
    initial_workspace_entries = _workspace_entries(ts_data)
    named_workspace = _matched_workspace(question, initial_workspace_entries)
    # An explicitly named study replaces conversation state from an earlier
    # turn. Otherwise a request naming study B can return records from study A.
    workspace_code = str((named_workspace or {}).get("code") or "") or forced_workspace
    if workspace_code:
        ts_data = [
            record for record in ts_data
            if str((record or {}).get("workspace_code", "")).strip() == workspace_code
        ]
        metadata = None

    # Numeric records from different studies are not interchangeable.  When
    # several studies are loaded, do not group identically named
    # model/scenario series across them or pick whichever record came first.
    # Ask the user to establish a study scope before returning values.
    if (
        not workspace_code
        and len(initial_workspace_entries) > 1
        and _looks_like_data_request(question)
        and bool(re.search(
            r"\b(?:this|that|the)\s+(?:study|workspace)\b",
            q,
        ))
        and not any(
            _looks_like_category_list_request(question, category)
            for category in ("models", "variables", "regions", "scenarios", "workspaces")
        )
    ):
        return _workspace_choice_response(initial_workspace_entries)

    # A scenario name typed verbatim (e.g. "BAU", "PR_Baseline") is ground truth
    # and overrides an extractor result that collapsed it to a generic family
    # (e.g. "Baseline"), which would otherwise cause a false "no data".
    # Only correct an extractor-supplied scenario that drifted off the verbatim
    # name; when no scenario was forced, leave the in-function matcher to resolve
    # it (overriding there would change no-data recovery for unrelated queries).
    if forced_scenario:
        _available_scenarios = {str(r.get("scenario", "")).strip() for r in ts_data if r and r.get("scenario")}
        _typed_scenarios = explicit_scenarios_from_query(question, _available_scenarios)
        if _typed_scenarios and forced_scenario not in _typed_scenarios:
            forced_scenario = _typed_scenarios[0]
    # Only treat this as a hard choice once a variable has been explicitly selected.
    # Region/scenario/model alone should still allow the normal guided clarification flow.
    forced_choice = bool(forced_variable)
    model_names = sorted({str(m.get("modelName", "")).strip() for m in model_data if m and m.get("modelName")})
    # The filters below compare against the model name stored on the timeseries
    # records, which can differ in case/format from what the entity extractor
    # emits (e.g. extractor "GCAM" vs record "gcam"). Canonicalize a forced model
    # to the actual record name so an explicit model filter is not silently
    # dropped as "no data" when the model in fact has data.
    ts_model_names = sorted({str(r.get("modelName", "")).strip() for r in ts_data if r and r.get("modelName")})
    if forced_model and forced_model not in ts_model_names:
        canonical_forced_model = match_model_name(forced_model, ts_model_names)
        if canonical_forced_model:
            forced_model = canonical_forced_model
    # Recognize an explicit "from <model>" clause the extractor missed (e.g.
    # "CO2 emissions for EU from E3ME"). Matching the isolated token avoids the
    # surrounding words diluting the model match.
    _from_model = None
    if not forced_model:
        _from_model = re.search(r"\bfrom\s+([A-Za-z0-9][\w\-\.]*)\s*$", question, re.IGNORECASE)
        if _from_model:
            _resolved_from = match_model_name(_from_model.group(1).strip(), ts_model_names)
            if _resolved_from:
                forced_model = _resolved_from

    # Resolve exact user wording before deciding whether a numeric slice spans
    # studies. Entity extraction can collapse a verbatim scenario (for example
    # BAU) to a family label, and explicit ``from <model>`` clauses may not have
    # been included in forced entities. Scoping earlier would select the wrong
    # study or ask an unnecessary study question.
    if not workspace_code and forced_variable:
        explicit_model_without_records = bool(_from_model and not forced_model)
        matching_records = [] if explicit_model_without_records else [
            record for record in ts_data
            if str(record.get("variable") or "") == forced_variable
            and (not forced_region or regions_equivalent(record.get("region"), forced_region))
            and (not forced_scenario or _scenario_match_ok(record.get("scenario"), forced_scenario))
            and (not forced_model or str(record.get("modelName") or "") == forced_model)
        ]
        matching_codes = sorted({
            str(record.get("workspace_code") or "").strip()
            for record in matching_records if str(record.get("workspace_code") or "").strip()
        })
        if len(matching_codes) > 1:
            choices = [
                entry for entry in initial_workspace_entries
                if str(entry.get("code") or "") in matching_codes
            ]
            return _workspace_choice_response(choices)
        if len(matching_codes) == 1:
            workspace_code = matching_codes[0]
            ts_data = [
                record for record in ts_data
                if str(record.get("workspace_code") or "").strip() == workspace_code
            ]
            metadata = None

    def _extract_model_hint(query: str) -> str:
        return extract_model_hint(query)

    def _resolve_model_candidates(query: str) -> list[str]:
        return resolve_model_candidates(query, model_names)

    def _match_model_name(query: str) -> str:
        return match_model_name(query, model_names)

    def _match_scenario_name(query: str) -> str:
        scenarios = sorted({str(r.get('scenario', '')).strip() for r in ts_data if r and r.get('scenario')})
        if not scenarios:
            return ""
        ql = query.lower()
        # Explicit "under X" or "scenario X"
        m = re.search(r"(?:under|scenario)\s+([\w\-\.]+)", ql)
        if m:
            token = m.group(1)
            for s in scenarios:
                if token.lower() in s.lower():
                    return s
        # SSP / RCP tokens
        for token in re.findall(r"(ssp\d|rcp\d(?:\.\d)?)", ql):
            for s in scenarios:
                if token.lower() in s.lower():
                    return s
        return ""

    def _extract_year_range(text: str) -> tuple[Optional[int], Optional[int]]:
        return extract_year_range(text)

    def _is_data_request(text: str) -> bool:
        return _looks_like_data_request(text)

    # Resolve the region before variable ranking. A natural-language location
    # may differ from its stored runtime code (for example an alias resolving
    # to an aggregate code), and must be treated as scope rather than variable
    # evidence throughout every route below.
    region_candidates = sorted({
        str(record.get("region", "")).strip()
        for record in ts_data
        if record and str(record.get("region", "")).strip()
    })
    query_region_match = forced_region or extract_region_from_query(
        question, region_dict, region_candidates
    )
    query_region_mentions = region_mentions_from_query(
        question,
        region_dict,
        region_candidates,
        resolved_region=query_region_match or None,
    )

    # Phrases such as "pathways by country" describe a geographic dimension,
    # not a concrete location. Returning a blended cross-region table hides
    # which geography each series belongs to and makes follow-ups unreliable,
    # so turn the missing scope into a typed clarification instead.
    generic_geography_scope = bool(re.search(
        r"\b(?:by|for|in|across)\s+(?:(?:each|a|the)\s+)?(?:country|countries|region|regions)\b",
        q,
    ))
    geography_is_catalogue_request = any(
        _looks_like_category_list_request(question, category)
        for category in ("regions", "variables", "scenarios", "models")
    )
    if (
        generic_geography_scope
        and not query_region_match
        and not geography_is_catalogue_request
        and _is_data_request(q)
    ):
        preferred_regions = [
            region for region in ("CHN", "IND", "USA", "EU", "World")
            if region in region_candidates
        ]
        if not preferred_regions:
            preferred_regions = region_candidates[:3]
        return _choice_prompt(
            "I need a specific country or region before I can return a grounded pathway.",
            "region",
            preferred_regions[:3],
        )

    def _show_all_requested(category: str) -> bool:
        singular = category.rstrip("s")
        return bool(re.search(rf"\b(?:show|list|get)\s+all\s+{singular}s?\b", q))

    def _show_all_hint(category: str, total: int, shown: int) -> str:
        if total <= shown:
            return ""
        return f"\n\nShowing {shown} of {total}. Say `show all {category}` if you need the full list."

    def _years_in_records(records: list[dict]) -> list[int]:
        years: set[int] = set()
        for record in records:
            if not record:
                continue
            nested = record.get("years")
            if isinstance(nested, dict):
                years.update(int(year) for year in nested if str(year).isdigit())
            years.update(int(key) for key in record if str(key).isdigit())
        return sorted(years)

    def _model_family_scope() -> tuple[str, list[str]]:
        request_name = requested_forced_model or forced_model
        lookup_text = request_name or question
        members = resolve_model_family_members(lookup_text, ts_model_names)
        if not members and forced_model in ts_model_names:
            members = [forced_model]
        display = request_name
        if not display and members:
            display = members[0]
        return display, members

    # Exact catalogue facts must be computed from the loaded records, never
    # delegated to free-form QA.  These run before discovery/listing so words
    # such as "database" cannot accidentally become scenario qualifiers.
    count_match = re.search(
        r"\b(?:how\s+many|number\s+of)\s+"
        r"(models?|variables?|indicators?|regions?|countries|scenarios?|pathways?|workspaces?)\b",
        q,
    )
    if count_match:
        raw_category = count_match.group(1)
        category = (
            "variables" if raw_category.startswith(("variable", "indicator"))
            else "regions" if raw_category.startswith(("region", "countr"))
            else "scenarios" if raw_category.startswith(("scenario", "pathway"))
            else "workspaces" if raw_category.startswith("workspace")
            else "models"
        )
        _display_model, _family_members = _model_family_scope()
        scoped_records = [
            record for record in ts_data
            if record
            and (not query_region_match or regions_equivalent(record.get("region"), query_region_match))
            and (not _family_members or str(record.get("modelName", "")).strip() in set(_family_members))
        ]
        field = {
            "variables": "variable", "regions": "region",
            "scenarios": "scenario", "workspaces": "workspace_code",
            "models": "modelName",
        }[category]
        if category == "models" and not query_region_match and not _family_members:
            values = {
                str(model.get("modelName", "")).strip()
                for model in model_data if model and model.get("modelName")
            }
        else:
            values = {
                str(record.get(field, "")).strip()
                for record in scoped_records if record.get(field)
            }
        if category == "regions":
            values = set(dedupe_equivalent_regions(sorted(values)))
        scope_parts = []
        if query_region_match:
            scope_parts.append(f"region `{format_region_label(query_region_match)}`")
        if _display_model:
            scope_parts.append(f"model `{_display_model}`")
        scope_text = f" for {' and '.join(scope_parts)}" if scope_parts else " in the loaded database"
        return f"There are **{len(values)} distinct {category}**{scope_text}."

    latest_start, latest_end = _resolved_year_range()
    global_latest_request = (
        is_latest_year_filter(latest_start, latest_end)
        and not forced_variable
        and not re.search(
            r"\b(?:emissions?|co2|ch4|n2o|gdp|population|energy|capacity|"
            r"electricity|price|production|demand|supply|investment)\b",
            q,
        )
    )
    if global_latest_request:
        _display_model, _family_members = _model_family_scope()
        scoped_records = [
            record for record in ts_data
            if record
            and (not query_region_match or regions_equivalent(record.get("region"), query_region_match))
            and (not _family_members or str(record.get("modelName", "")).strip() in set(_family_members))
        ]
        years = _years_in_records(scoped_records)
        if not years:
            return "I could not find any projection years in the loaded IAM PARIS data for that scope."
        scope_parts = []
        if query_region_match:
            scope_parts.append(f"region `{format_region_label(query_region_match)}`")
        if _display_model:
            scope_parts.append(f"model `{_display_model}`")
        scope_text = f" for {' and '.join(scope_parts)}" if scope_parts else ""
        return (
            f"The latest available projection year{scope_text} is **{years[-1]}**. "
            f"The loaded records span {years[0]}–{years[-1]} ({len(years)} distinct years)."
        )

    region_summary_model_members = resolve_model_family_members(
        requested_forced_model or forced_model or question,
        ts_model_names,
    )
    region_summary_is_model_scoped = bool(
        _model_scoped_category(question)
        and (requested_forced_model or forced_model or region_summary_model_members)
    )
    if query_region_match and not region_summary_is_model_scoped and re.search(
        r"\bwhat\s+data\s+(?:do|does|can|could|is|are|'s|s)\b[^?]*\b(?:for|in)\b",
        q,
    ):
        if metadata:
            region_info = metadata.get_available_for_region(query_region_match)
            variables = region_info.get("variables", [])
            scenarios = region_info.get("scenarios", [])
            scoped_models = region_info.get("models", [])
            years = [int(year) for year in region_info.get("years", []) if str(year).isdigit()]
        else:
            region_records = [
                record for record in ts_data
                if record and regions_equivalent(record.get("region"), query_region_match)
            ]
            variables = _display_variable_catalog(
                record.get("variable") for record in region_records if record.get("variable")
            )
            scenarios = sorted({str(record.get("scenario", "")).strip() for record in region_records if record.get("scenario")})
            scoped_models = sorted({
                str(record.get("modelName", "")).strip()
                for record in region_records
                if record and is_presentable_model_label(record.get("modelName"))
            })
            years = _years_in_records(region_records)
        variables = _display_variable_catalog(variables)
        if not variables:
            return (
                f"I found region `{format_region_label(query_region_match)}`, but no timeseries data is loaded for it. "
                "Ask `list regions` to see regions with available results."
            )
        year_text = f"{years[0]}–{years[-1]}" if years else "no year values"
        sample = ", ".join(f"`{value}`" for value in variables[:8])
        more = f" and {len(variables) - 8} more" if len(variables) > 8 else ""
        return (
            f"### Data available for {format_region_label(query_region_match)}\n\n"
            f"- **Variables:** {len(variables)} ({sample}{more})\n"
            f"- **Scenarios:** {len(scenarios)}\n"
            f"- **Models:** {len(scoped_models)}\n"
            f"- **Years:** {year_text}\n\n"
            "Ask for one variable, for example `CO2 emissions`, to retrieve values or make a plot."
        )

    # Scenarios/variables/regions scoped to a specific model, e.g.
    # "what scenarios does GCAM have" or (with a model carried in context)
    # "what scenarios does it have". Must run before the generic discovery and
    # category-list paths, which would otherwise return an unscoped overview.
    scoped_category = _model_scoped_category(question)
    if scoped_category:
        # Resolve against the timeseries model names so the listing matches the
        # record casing/format (e.g. "gcam"), not the display catalogue ("GCAM").
        requested_scoped_model = forced_model or requested_forced_model
        family_lookup = requested_forced_model or forced_model or question
        scoped_members = resolve_model_family_members(family_lookup, ts_model_names)
        runtime_candidates = resolve_model_candidates(question, ts_model_names)
        # A model may exist in the descriptive catalogue while carrying no
        # loaded timeseries. Resolve its name from the model catalogue too so a
        # scoped request returns an honest empty scope instead of falling
        # through to an unscoped list of every variable in the database.
        catalogue_candidates = resolve_model_candidates(question, model_names)
        scoped_model = (
            requested_scoped_model
            or (catalogue_candidates[0] if catalogue_candidates else "")
            or (runtime_candidates[0] if runtime_candidates else "")
        )
        if scoped_model and not scoped_members:
            scoped_members = resolve_model_family_members(scoped_model, ts_model_names)
        if not scoped_members and scoped_model in ts_model_names:
            scoped_members = [scoped_model]
        if scoped_model:
            return _list_model_category(
                scoped_category,
                scoped_model,
                ts_data,
                show_all=_show_all_requested(scoped_category),
                model_members=scoped_members,
            )

    # Route variable-discovery phrasing to the variable list path.
    if not forced_choice and _looks_like_category_list_request(question, "variables") and _looks_like_plot_request(question):
        listing_records = [
            record for record in ts_data
            if record and (not query_region_match or regions_equivalent(record.get("region"), query_region_match))
        ]
        vars = _display_variable_catalog(r.get("variable") for r in listing_records if r.get("variable"))
        if not vars:
            return "I don't see any variables in the loaded dataset. Try reloading or check the IAM PARIS results website."

        show_all = _show_all_requested("variables")
        sample = vars if show_all else (vars[:12] if len(vars) > 8 else vars)
        more = "" if len(vars) <= len(sample) else f" and {len(vars)-len(sample)} more"
        sample_str = "\n- ".join(sample)
        example_region = query_region_match or "Greece"
        return (f"I can work with these variables:\n- {sample_str}{more}\n\n"
                f"Try queries like 'Capacity|Electricity|Solar|Utility for {example_region}' "
                f"or 'plot [variable name] in {example_region}'."
                + ("" if show_all else _show_all_hint("variables", len(vars), len(sample))))

    # Discovery mode: the user wants help understanding what is available
    discovery_phrases = [
        r"\bi don'?t know\b",
        r"\bi'?m not sure\b",
        r"\bnot sure what\b",
        r"\bexplore data\b",
    ]
    if _looks_like_discovery_request(question) or any(re.search(pattern, q) for pattern in discovery_phrases):
        models = sorted({
            str(m.get('modelName', '')).strip()
            for m in model_data
            if m and is_presentable_model_label(m.get('modelName'))
        })
        variables = _display_variable_catalog(r.get("variable") for r in ts_data if r and r.get("variable"))
        scenarios = sorted({str(r.get('scenario', '')).strip() for r in ts_data if r and r.get('scenario')})
        regions = dedupe_equivalent_regions(sorted({
            str(r.get('region', '')).strip() for r in ts_data if r and r.get('region')
        }))
        workspaces = sorted({str(r.get('workspace_code', '')).strip() for r in ts_data if r and r.get('workspace_code')})

        def _pick_examples(items: list[str], keywords: list[str], limit: int = 4) -> list[str]:
            if not items:
                return []
            if keywords:
                matched = [item for item in items if any(k in item.lower() for k in keywords)]
                if matched:
                    return matched[:limit]
            return items[:limit]

        q_lower = q.lower()
        if any(k in q_lower for k in ["energy", "power", "electricity", "solar", "wind", "hydro", "nuclear"]):
            sample_vars = _pick_examples(variables, ["energy", "electricity", "capacity", "power", "solar", "wind", "hydro", "nuclear"])
        elif any(k in q_lower for k in ["emission", "co2", "carbon", "ch4", "kyoto"]):
            sample_vars = _pick_examples(variables, ["emission", "co2", "carbon", "ch4", "kyoto"])
        elif any(k in q_lower for k in ["demand", "supply", "industry", "transport", "buildings"]):
            sample_vars = _pick_examples(variables, ["demand", "supply", "industry", "transport", "buildings"])
        elif any(k in q_lower for k in ["region", "country", "where", "china", "india", "europe", "usa", "world"]):
            sample_vars = _pick_examples(variables, [])
        else:
            sample_vars = _pick_examples(variables, [])

        sample_models = models[:4]
        sample_scenarios = scenarios[:4]
        sample_regions = regions[:4]

        response = "### What I can help you with\n\n"
        response += f"- **Models:** {len(models)} available. Examples: {', '.join(sample_models)}.\n"
        response += f"- **Variables:** {len(variables)} available. Examples: {', '.join(sample_vars)}.\n"
        response += f"- **Scenarios:** {len(scenarios)} available. Examples: {', '.join(sample_scenarios)}.\n"
        response += f"- **Regions:** {len(regions)} available. Examples: {', '.join(sample_regions)}.\n"
        response += f"- **Workspaces:** {len(workspaces)} available. Ask `list workspaces`.\n\n"
        response += "Try one of these next:\n"
        response += "- `list variables`\n"
        response += "- `list regions`\n"
        response += "- `list scenarios`\n"
        response += "- `Show me CO2 emissions for EU`\n"
        response += "- `Plot solar vs wind capacity for EU`\n"
        return response

    # Reject an explicitly named but unavailable location before either the
    # plotter or variable recommender can reinterpret it as taxonomy wording.
    if (
        not forced_region
        and not query_region_match
        and (
            _looks_like_data_request(question)
            or _looks_like_plot_request(question)
            or _looks_like_comparison_request(question)
        )
    ):
        unknown_place = unknown_named_region(question, region_candidates, model_names)
        if unknown_place:
            examples = sorted(region_candidates, key=lambda value: (len(value), value.casefold()))[:5]
            labels = ", ".join(f"`{format_region_label(region)}`" for region in examples)
            suffix = f" Try one of the available regions, e.g. {labels}," if labels else ""
            return (
                f"I couldn't find `{unknown_place}` as a region in the IAM PARIS data, "
                f"so I can't return results for it.{suffix} "
                "or ask `list regions` to see all options."
            )

    # -------------------------------
    # Handle COMPARISON QUERIES (route to plotter)
    # -------------------------------
    if allow_plots and _looks_like_comparison_request(question):
        return simple_plot_query(question, model_data, ts_data)

    # -------------------------------
    # Handle PLOTTING QUERIES
    # -------------------------------
    if allow_plots and not forced_choice and (
        _looks_like_plot_request(question)
        or (_looks_like_data_request(question) and any(word in q for word in ['show', 'display']))
    ):
        q_lower = q.lower()
        variable_hints = [
            "co2", "emission", "emissions", "capacity", "electricity", "energy", "solar",
            "wind", "oil", "gas", "nuclear", "hydro", "investment", "demand", "supply",
            "generation", "temperature", "transport", "industry", "buildings", "power"
        ]
        has_variable_hint = any(h in q_lower for h in variable_hints) or "|" in question
        is_explicit_plot = any(word in q for word in ['plot', 'graph', 'visualize'])

        # Before the plotting resolver's broad fallback, require at least one
        # catalogue variable from the requested semantic family. This prevents
        # a specific but unavailable concept from drifting to an unrelated plot.
        if not has_variable_hint and not is_explicit_plot:
            pass
        else:
            # Try universal natural language resolution first
            natural_variable = resolve_natural_language_variable_universal(question, variable_dict)
            if natural_variable:
                # Check if this variable exists in our data
                available_vars = {str(r.get('variable', '')) for r in ts_data if r and r.get('variable')}
                if natural_variable in available_vars and sanitize_variable_for_query(
                    natural_variable, question
                ):
                    # Use simple_plot_query with resolved variable
                    return simple_plot_query(question, model_data, ts_data)

            variable_intent = _infer_variable_intent(question)
            if variable_intent != "general":
                guarded_candidates = _suggest_variable_candidates(
                    question,
                    variable_dict,
                    ts_data,
                    region=query_region_match or None,
                    ignored_values=query_region_mentions,
                    limit=3,
                )
                if not guarded_candidates:
                    return "I could not confidently match that wording to a loaded variable. Which variable should I use?"
                if len(guarded_candidates) > 1:
                    unresolved_qualifiers = _unmatched_variable_qualifiers(
                        question,
                        guarded_candidates,
                        region=query_region_match,
                        scenario=forced_scenario,
                        model=forced_model,
                        ignored_values=query_region_mentions,
                    )
                    if unresolved_qualifiers:
                        rendered = ", ".join(f"`{term}`" for term in unresolved_qualifiers)
                        prefix = (
                            f"I could not match the qualifier(s) {rendered} exactly in the loaded "
                            "variable catalogue. These are the nearest variables in the requested semantic family."
                        )
                    else:
                        prefix = "I found several loaded variables in that semantic family."
                    return _choice_prompt(
                        prefix,
                        "variable",
                        guarded_candidates,
                    )
                guarded_entities = {"variable": guarded_candidates[0], "action": "plot"}
                guarded_region = query_region_match
                if guarded_region:
                    guarded_entities["region"] = guarded_region
                guarded_start, guarded_end = extract_year_range(question)
                if guarded_start is not None:
                    guarded_entities["start_year"] = guarded_start
                    guarded_entities["end_year"] = guarded_end
                return simple_plot_query_with_entities(
                    question, model_data, ts_data, guarded_entities
                )

            # Fallback to existing plotting logic
            return simple_plot_query(question, model_data, ts_data)

    # -------------------------------
    # LIST AVAILABLE MODELS
    # -------------------------------
    if not forced_choice and _looks_like_category_list_request(question, "models"):
        if query_region_match:
            models = sorted({
                str(record.get("modelName", "")).strip()
                for record in ts_data
                if record
                and regions_equivalent(record.get("region"), query_region_match)
                and is_presentable_model_label(record.get("modelName"))
            })
        else:
            models = sorted({
                str(r.get('modelName', '')).strip()
                for r in model_data
                if r and is_presentable_model_label(r.get('modelName'))
            })
        if not models:
            return "I couldn't find any models in the data right now. Try `help` or refresh the data."

        show_all = _show_all_requested("models")
        if len(models) <= 6 or show_all:
            model_str = ", ".join(models[:-1]) + (" and " + models[-1] if len(models) > 1 else models[0])
            return f"I found these models in the IAM PARIS dataset: {model_str}. Which one would you like to know more about?"

        sample = ", ".join(models[:8])
        return (f"There are {len(models)} models available. "
                f"Examples: {sample}. "
                "You can ask for details about a specific model using `info [model name]`, "
                "or say `list variables` to see the kinds of outputs available.")

    # -------------------------------
    # LIST AVAILABLE VARIABLES
    # -------------------------------
    if not forced_choice and _looks_like_category_list_request(question, "variables"):
        listing_records = [
            record for record in ts_data
            if record and (not query_region_match or regions_equivalent(record.get("region"), query_region_match))
        ]
        vars = _display_variable_catalog(r.get("variable") for r in listing_records if r.get("variable"))
        if not vars:
            return "I don't see any variables in the loaded dataset. Try reloading or check the IAM PARIS results website."

        # Filter for energy-related variables if "energy" is mentioned
        if 'energy' in q:
            energy_vars = [v for v in vars if any(term in v.lower() for term in ['energy', 'electricity', 'capacity', 'power', 'generation', 'solar', 'wind', 'hydro', 'gas', 'nuclear', 'biomass'])]
            if energy_vars:
                vars = energy_vars[:15]  # Show more energy variables
            else:
                vars = vars[:8]

        show_all = _show_all_requested("variables")
        sample = vars if show_all else (vars[:12] if len(vars) > 8 else vars)
        more = "" if len(vars) <= len(sample) else f" and {len(vars)-len(sample)} more"
        sample_str = "\n- ".join(sample)
        scope_text = (
            f" for region `{format_region_label(query_region_match)}`"
            if query_region_match else ""
        )
        example_region = query_region_match or "Greece"
        return (f"I can work with these variables{scope_text}:\n- {sample_str}{more}\n\n"
                f"Try queries like 'Capacity|Electricity|Solar|Utility for {example_region}' "
                f"or 'plot [variable name] in {example_region}'."
                + ("" if show_all else _show_all_hint("variables", len(vars), len(sample))))

    # -------------------------------
    # LIST AVAILABLE SCENARIOS
    # -------------------------------
    if not forced_choice and _looks_like_category_list_request(question, "scenarios"):
        scenarios = sorted({
            r.get('scenario', '') for r in ts_data
            if r and r.get('scenario')
            and (not query_region_match or regions_equivalent(r.get("region"), query_region_match))
        })
        if not scenarios:
            return "No scenarios are loaded in the current dataset. Try a different query or check IAM PARIS results."

        # If the user named a scenario qualifier (e.g. "net zero"), narrow the
        # list to matching scenarios. Scenario names are coded, so expand common
        # phrases to their codes (net zero -> NZE) before substring matching.
        ql = question.lower()
        _scenario_synonyms = {
            "net zero": ["nze", "nz"], "net-zero": ["nze", "nz"], "netzero": ["nze", "nz"],
            "current policies": ["curpol", "cp"], "current policy": ["curpol"],
            "business as usual": ["bau"],
        }
        qualifier_terms: list[str] = []
        for phrase, codes in _scenario_synonyms.items():
            if phrase in ql:
                qualifier_terms.extend(codes)
        _stop = {
            "scenario", "scenarios", "pathway", "pathways", "what", "which", "list",
            "show", "available", "are", "is", "the", "for", "me", "all", "of", "do",
            "you", "have", "there", "any", "provide", "tell", "about", "under", "and",
        }
        for tok in re.findall(r"[a-z0-9.]+", ql):
            if tok not in _stop and len(tok) > 1:
                qualifier_terms.append(tok)
        qualifier_label = ""
        if qualifier_terms:
            matched = [s for s in scenarios if any(t in s.lower() for t in qualifier_terms)]
            if matched and len(matched) < len(scenarios):
                scenarios = matched
                qualifier_label = " matching your request"

        show_all = _show_all_requested("scenarios")
        sample = scenarios if show_all else scenarios[:8]
        more = "" if len(scenarios) <= 8 else f" and {len(scenarios)-8} more"
        sample_str = ", ".join(sample[:-1]) + (" and " + sample[-1] if len(sample) > 1 else sample[0])
        return (
            f"I found scenarios{qualifier_label}"
            f"{' for ' + format_region_label(query_region_match) if query_region_match else ''} "
            f"like {sample_str}{'' if show_all else more}. "
            "You can plot variables for any of these scenarios."
            + ("" if show_all else _show_all_hint("scenarios", len(scenarios), len(sample)))
        )

    # -------------------------------
    # LIST AVAILABLE REGIONS
    # -------------------------------
    if not forced_choice and _looks_like_category_list_request(question, "regions"):
        regions = dedupe_equivalent_regions(sorted({
            str(r.get('region', '')).strip() for r in ts_data if r and r.get('region')
        }))
        if not regions:
            return "No regions are loaded in the current dataset. Try a different query or check IAM PARIS results."

        show_all = _show_all_requested("regions")
        sample = regions if show_all else regions[:10]
        more = "" if len(regions) <= 10 else f" and {len(regions)-10} more"
        sample_str = ", ".join(sample[:-1]) + (" and " + sample[-1] if len(sample) > 1 else sample[0])
        return (
            f"I found regions like {sample_str}{'' if show_all else more}. "
            "You can plot variables for any of these regions."
            + ("" if show_all else _show_all_hint("regions", len(regions), len(sample)))
        )

    workspace_entries = _workspace_entries(ts_data)
    matched_workspace = _matched_workspace(question, workspace_entries)
    workspace_overview_request = bool(re.search(
        r"\b(?:about|overview|summary|what(?:\s+is|\s+results?|\s+data)?|show|tell\s+me|"
        r"available|workspace|study)\b",
        q,
    ))
    # A named study without a data dimension is best introduced through its
    # own variables/scenarios, rather than through the global variable matcher.
    if not forced_choice and matched_workspace and workspace_overview_request and not re.search(
        r"\b(?:plot|chart|graph|table|value|values|timeseries|time\s+series|"
        r"emissions?|capacity|price|gdp)\b",
        q,
    ):
        record_resolved_scope(workspace_code=str(matched_workspace["code"]), action="query")
        return _workspace_summary(matched_workspace)

    results_overview_request = bool(re.fullmatch(
        r"\s*(?:(?:what|which)\s+)?(?:results?|stud(?:y|ies)|workspaces?)\s+"
        r"(?:are\s+)?(?:available|loaded|included)(?:\s+(?:here|now))?[?.!]?\s*",
        q,
    ))
    if not forced_choice and (
        _looks_like_category_list_request(question, "workspaces")
        or results_overview_request
    ):
        if not workspace_entries:
            return "No workspaces are loaded in the current dataset. Try a different query or check IAM PARIS results."

        show_all = _show_all_requested("workspaces")
        sample = workspace_entries if show_all else workspace_entries[:12]
        more = "" if len(workspace_entries) <= len(sample) else f" and {len(workspace_entries)-len(sample)} more"
        return (
            "I found these workspaces (studies): " + "; ".join(str(entry["title"]) for entry in sample) + more + ". "
            "Ask about a study to see its variables, scenarios, and direct IAM PARIS links."
            + ("" if show_all else _show_all_hint("workspaces", len(workspace_entries), len(sample)))
        )

    # -------------------------------
    # LIST ALL MODELS, RESULTS, AND WORKSPACES
    # -------------------------------
    if re.search(r"\b(list|get)\b.*\b(all|available)\b.*\b(models?|results?|workspaces?)\b", q) or \
       re.search(r"\b(models?|results?|workspaces?)\b.*\b(list|get|available)\b", q) or \
       re.search(r"\b(list|get)\b.*\b(models?|results?|workspaces?)\b.*\b(and|,)\b.*\b(models?|results?|workspaces?)\b", q):
        models = sorted({
            str(r.get('modelName', '')).strip()
            for r in model_data
            if r and is_presentable_model_label(r.get('modelName'))
        })
        variables = _display_variable_catalog(r.get("variable") for r in ts_data if r and r.get("variable"))
        scenarios = sorted({r.get('scenario', '') for r in ts_data if r and r.get('scenario')})
        workspaces = _workspace_entries(ts_data)

        response = "### Available Models, Results, and Workspaces\n\n"
        model_sample = models[:10]
        response += f"**Models ({len(models)}):**\n" + ", ".join(model_sample)
        response += (f" and {len(models)-10} more" if len(models) > 10 else "") + "\n\n"
        response += f"**Results - Variables ({len(variables)}):**\n" + ", ".join(variables[:10]) + (f" and {len(variables)-10} more" if len(variables) > 10 else "") + "\n\n"
        response += f"**Results - Scenarios ({len(scenarios)}):**\n" + ", ".join(scenarios[:10]) + (f" and {len(scenarios)-10} more" if len(scenarios) > 10 else "") + "\n\n"
        response += f"**Studies ({len(workspaces)}):**\n" + ", ".join(str(entry["title"]) for entry in workspaces) + "\n\n"
        response += "Ask about a specific study to see up to 10 of its variables and scenarios, plus direct IAM PARIS links."
        return response

    # -------------------------------
    # MODEL INFO REQUESTS
    # -------------------------------
    explicit_model_question = bool(re.search(r"\bwhat\s+is\b", q) or re.search(r"\bwho\s+is\b", q))
    profile_model_question = _looks_like_model_info_request(question) or explicit_model_question
    profile_match = find_model_profile(question) if profile_model_question else None
    hinted_model = _extract_model_hint(question)
    direct_model_match = _match_model_name(question)
    candidate_model_matches = [direct_model_match] if direct_model_match else _resolve_model_candidates(hinted_model or question)
    model_info_request = (
        (_looks_like_model_info_request(question) and (candidate_model_matches or 'model' in q))
        or ('model' in q)
        or (explicit_model_question and candidate_model_matches)
    )
    # A rendered conversational patch may include ``for model <name>`` as part
    # of a complete data scope. Once a variable was explicitly forced, that is
    # a model filter—not a request for the model's description.
    if model_info_request and not forced_choice:
        if not model_names:
            if profile_match:
                return format_model_profile_answer(
                    profile_match,
                    requested_name=str(profile_match.get("name", "")),
                    asks_assumptions=bool(re.search(r"\bassumption\b|\bassumptions\b", q)),
                )
            return "I couldn't find any model metadata. Try reloading the models data."

        hint = hinted_model
        substring_matches = candidate_model_matches

        if not substring_matches:
            # Fallback: check model names present in results data
            ts_model_names = sorted({r.get('modelName', '') for r in ts_data if r and r.get('modelName')})
            if ts_model_names:
                stopwords = {
                    'tell', 'me', 'about', 'the', 'a', 'an', 'model', 'models', 'info',
                    'details', 'describe', 'of', 'for', 'on', 'please'
                }
                query_lower = q
                tokens = [t for t in re.split(r"\W+", query_lower) if t and t not in stopwords and len(t) >= 3]
                ts_matches = [m for m in ts_model_names if any(t in m.lower() for t in tokens)]
                if ts_matches:
                    if profile_match:
                        return format_model_profile_answer(
                            profile_match,
                            requested_name=str(profile_match.get("name", "")),
                            asks_assumptions=bool(re.search(r"\bassumption\b|\bassumptions\b", q)),
                        )
                    sample = ", ".join(ts_matches[:5])
                    return (
                        "I found matching model names in the results data, but no metadata description is available. "
                        f"Examples: {sample}. If you want, ask for plots or values for one of these models."
                    )

            if profile_match:
                return format_model_profile_answer(
                    profile_match,
                    requested_name=str(profile_match.get("name", "")),
                    asks_assumptions=bool(re.search(r"\bassumption\b|\bassumptions\b", q)),
                )
            return "I couldn't match that to a known model. Try `list models` to see available options."

        if len(substring_matches) > 1:
            if profile_match:
                matched_profile_names = {
                    str((find_model_profile(match) or {}).get("name", ""))
                    for match in substring_matches
                }
                if matched_profile_names == {str(profile_match.get("name", ""))}:
                    return format_model_profile_answer(
                        profile_match,
                        requested_name=str(profile_match.get("name", "")),
                        asks_assumptions=bool(re.search(r"\bassumption\b|\bassumptions\b", q)),
                    )
            sample = ", ".join(substring_matches[:5])
            return f"I found multiple model matches: {sample}. Which one do you want details for?"

        model_name = substring_matches[0]
        records = [r for r in model_data if r and r.get('modelName') == model_name]
        rec = records[0] if records else {}

        asks_assumptions = bool(re.search(r"\bassumption\b|\bassumptions\b", q))
        # A qualitative coverage question must remain grounded in the model
        # profile.  In particular, do not let its topic (for example "energy")
        # fall through to the variable resolver and suggest unrelated results
        # variables.  The focused profile formatter says when coverage is not
        # stated and supplies the model-specific IAM PARIS page when an ID or
        # URL is available in the catalogue record.
        coverage_question = bool(re.search(
            r"\bdoes\s+[^?]+\s+(?:cover(?:s|ed|ing)?|include(?:s|d|ing)?|"
            r"represent(?:s|ed|ing)?|simulate(?:s|d|ing)?|capture(?:s|d|ing)?|account\s+for)\b",
            q,
        ))
        profile = find_model_profile(model_name)
        if coverage_question and (profile or rec):
            profile_with_catalogue_link = dict(profile or {})
            for key in (
                "description", "developer", "sectors", "typical_use_cases",
                "methodology_note", "technology_note", "assumptions_note",
                "iamparis_model_url", "model_url", "modelUrl", "url", "URL",
                "route", "path", "slug", "href", "model_id", "modelId",
                "modelID", "id", "ID", "pk", "modelPk", "model_pk",
            ):
                if rec.get(key) not in (None, ""):
                    profile_with_catalogue_link[key] = rec[key]
            return format_model_profile_answer(
                profile_with_catalogue_link,
                requested_name=model_name,
                asks_assumptions=asks_assumptions,
                query=question,
            )
        return _format_model_info_answer(model_name, rec, asks_assumptions=asks_assumptions)

    # -------------------------------
    # SPECIFIC VARIABLE QUERIES - Enhanced matching with universal resolver
    # -------------------------------
    # Try universal natural language resolution first (with confidence info)
    variable_match = None
    var_score = None
    matched_words = []
    significant_words = []
    ranked_vars = resolve_natural_language_variable_ranked(question, variable_dict, top_k=5)
    resolved = resolve_natural_language_variable_with_score(question, variable_dict)
    available_vars = {str(r.get('variable', '')).strip() for r in ts_data if r and r.get('variable')}
    preferred_variable = _preferred_available_variable(question, available_vars)
    explicit_variable_name = _explicit_variable_in_query(question, available_vars)
    # A natural-language phrase with explicit qualifiers ("primary energy from
    # coal", "solar PV capacity") is more specific than a base variable name
    # that merely appears as a substring of that phrase.
    prefer_semantic = bool(
        preferred_variable
        and explicit_variable_name
        and preferred_variable.count("|") > explicit_variable_name.count("|")
        and "|" not in question
    )
    if explicit_variable_name and not forced_variable and not prefer_semantic:
        variable_match = explicit_variable_name
        var_score = 999
        matched_words = []
        significant_words = []
    elif preferred_variable and not forced_variable:
        variable_match = preferred_variable
        var_score = 999
        matched_words = []
        significant_words = []
    elif resolved and not forced_variable:
        variable_match, var_score, matched_words, significant_words = resolved
        if isinstance(variable_match, str):
            variable_match = variable_match.strip()
    variable_intent = _infer_variable_intent(question, significant_words)
    region_match = None

    # Always try to extract region, regardless of variable match success
    extracted = extract_variable_and_region_from_query(question, variable_dict, region_dict, region_candidates)
    region_match = query_region_match or extracted['region']
    if re.search(r"\b(world|global)\b", question.lower()):
        region_match = "World"
    if forced_variable:
        # The entity extractor can drift from a typed variable to a superstring
        # (e.g. "Secondary Energy|Electricity" -> "Price|Secondary Energy|
        # Electricity"). When the user typed an exact structured variable name
        # verbatim that the extractor did not choose, trust what they typed.
        if (
            explicit_variable_name
            and explicit_variable_name != forced_variable
            and forced_variable.lower() not in question.lower()
        ):
            forced_variable = explicit_variable_name
        variable_match = forced_variable
        var_score = 999
    if forced_region:
        region_match = forced_region

    # Unknown named region: if the user explicitly named a place after "for"/"in"
    # that resolves to no known region (and is not a model/scenario), do not
    # silently fall back to unfiltered (global) data — say so and suggest valid
    # regions. This guards against inventing results for any nonexistent place.
    if not region_match and not forced_region:
        _place = unknown_named_region(
            question,
            region_candidates,
            model_names,
            scenario_names={
                str(record.get("scenario", "")).strip()
                for record in ts_data
                if record and str(record.get("scenario", "")).strip()
            },
            variable_names=available_vars,
        )
        if _place:
            _examples = sorted(
                region_candidates,
                key=lambda value: (len(value), value.casefold()),
            )[:5]
            _labels = ", ".join(f"`{format_region_label(r)}`" for r in _examples[:5])
            return (
                f"I couldn't find `{_place}` as a region in the IAM PARIS data, "
                f"so I can't return results for it.\n\n"
                f"Try one of the available regions, e.g. {_labels}, "
                f"or ask `list regions` to see all options."
            )

    availability_question = bool(
        re.search(r"\b(?:does|do|is|are|can)\b", q)
        and re.search(
            r"\b(?:report|reports|reported|have|has|include|includes|"
            r"provide|provides|contain|contains|available)\b",
            q,
        )
    )
    if variable_match and availability_question:
        requested_model_label = requested_forced_model or forced_model
        model_members = resolve_model_family_members(
            requested_model_label or question,
            ts_model_names,
        )
        if not model_members and forced_model in ts_model_names:
            model_members = [forced_model]
        model_set = set(model_members)
        # A named model that has no runtime timeseries member is an empty model
        # scope, not an instruction to drop the model filter.  Treating an empty
        # set as "all models" produced false claims such as "WITCH reports..."
        # while the supporting runtime labels contained no WITCH model at all.
        if requested_model_label and not model_set:
            alternatives = sorted({
                str(record.get("modelName", "")).strip()
                for record in ts_data
                if record
                and str(record.get("variable", "")).strip() == variable_match
                and (not region_match or regions_equivalent(record.get("region"), region_match))
                and is_presentable_model_label(record.get("modelName"))
            })
            region_text = (
                f" in region `{format_region_label(region_match)}`"
                if region_match else ""
            )
            alternative_text = (
                f" Models that do have this slice include: {', '.join(alternatives[:5])}."
                if alternatives else ""
            )
            return (
                f"No. I could not find loaded timeseries data for model "
                f"`{requested_model_label}` reporting `{variable_match}`{region_text}."
                + alternative_text
            )
        named_subject = re.search(
            r"\b(?:does|do)\s+([A-Za-z0-9][\w.-]*)\s+"
            r"(?:report|reports|have|has|include|includes|provide|provides|contain|contains)\b",
            question,
            re.IGNORECASE,
        )
        if (
            named_subject
            and named_subject.group(1).casefold() not in {"the", "data", "dataset", "database", "results"}
            and not model_set
            and not requested_model_label
        ):
            return (
                f"I could not match model `{named_subject.group(1)}` to a model with loaded timeseries data. "
                "Ask `list models` to see the available model names."
            )
        matching_records = [
            record for record in ts_data
            if record
            and str(record.get("variable", "")).strip() == variable_match
            and (not region_match or regions_equivalent(record.get("region"), region_match))
            and (not model_set or str(record.get("modelName", "")).strip() in model_set)
        ]
        model_scope_named = bool(requested_model_label or model_members)
        if matching_records:
            matched_models = sorted({
                str(record.get("modelName", "")).strip()
                for record in matching_records
                if is_presentable_model_label(record.get("modelName"))
            })
            matched_scenarios = sorted({
                str(record.get("scenario", "")).strip()
                for record in matching_records if record.get("scenario")
            })
            matched_years = _years_in_records(matching_records)
            model_text = (
                f"Model `{requested_model_label or ', '.join(matched_models)}`"
                if model_scope_named else "The loaded IAM PARIS results"
            )
            region_text = f" for region `{format_region_label(region_match)}`" if region_match else ""
            details = []
            if matched_models:
                details.append(f"runtime model label(s): {', '.join(matched_models)}")
            if matched_scenarios:
                details.append(f"{len(matched_scenarios)} scenario(s)")
            if matched_years:
                details.append(f"years {matched_years[0]}–{matched_years[-1]}")
            detail_text = f" ({'; '.join(details)})" if details else ""
            return f"Yes. {model_text} reports `{variable_match}`{region_text}{detail_text}."

        model_text = f"model `{requested_model_label}`" if requested_model_label else "that model scope"
        region_text = f" in region `{format_region_label(region_match)}`" if region_match else ""
        alternatives = sorted({
            str(record.get("modelName", "")).strip()
            for record in ts_data
            if record
            and str(record.get("variable", "")).strip() == variable_match
            and (not region_match or regions_equivalent(record.get("region"), region_match))
            and is_presentable_model_label(record.get("modelName"))
        })
        alternative_text = (
            f" Models that do have this slice include: {', '.join(alternatives[:5])}."
            if alternatives else ""
        )
        return (
            f"No. I could not find `{variable_match}`{region_text} for {model_text} "
            "in the loaded IAM PARIS timeseries data."
            + alternative_text
        )

    if forced_choice:
        model_match = forced_model or _match_model_name(question)
        scenario_match = forced_scenario or _match_scenario_name(question)
        if not variable_match:
            return "I need one more detail. Which variable should I use?"

        if (
            metadata
            and region_match
            and scenario_match
            and not metadata.combination_exists(
                variable_match,
                region=region_match,
                scenario=None if _scenario_is_family_label(scenario_match) else scenario_match,
                model=model_match or None,
            )
        ):
            family_prompt = _scenario_family_recovery_prompt(
                ts_data,
                variable=variable_match,
                region=region_match,
                scenario=scenario_match,
                model=model_match or None,
            )
            if family_prompt:
                return family_prompt
            matrix_prompt = _matrix_recovery_prompt(
                metadata,
                (
                    f"No data found for **{variable_match}** in region `{region_match}` "
                    f"under scenario `{scenario_match}`."
                ),
                variable=variable_match,
                region=region_match,
                scenario=scenario_match,
                model=model_match or None,
            )
            if matrix_prompt:
                return matrix_prompt

        def _filter(use_model: bool) -> list:
            out = []
            for r in ts_data:
                if r is None:
                    continue
                if str(r.get('variable', '')) != variable_match:
                    continue
                if use_model and model_match and r.get('modelName') != model_match:
                    continue
                if scenario_match and not _scenario_match_ok(r.get('scenario'), scenario_match):
                    continue
                if region_match and not regions_equivalent(r.get('region'), region_match):
                    continue
                out.append(r)
            return out

        filtered_data = _filter(use_model=True)
        # Self-heal: a fuzzy model match can mis-read a region word (e.g.
        # "China" -> model "China-MORE") and wrongly empty the result. If
        # dropping the (non-forced) model filter recovers data, the model match
        # was spurious.
        model_relaxed_notice = ""
        if not filtered_data and model_match and not forced_model:
            recovered = _filter(use_model=False)
            if recovered:
                model_match = ""
                filtered_data = recovered
        # An explicit model is a hard data-integrity boundary.  Never substitute
        # rows from other models merely because the requested slice is empty.
        elif not filtered_data and model_match and forced_model:
            alternatives = sorted({
                str(record.get("modelName", "")).strip()
                for record in _filter(use_model=False)
                if is_presentable_model_label(record.get("modelName"))
            })
            requested_label = requested_forced_model or model_match
            region_text = (
                f" in `{format_region_label(region_match)}`" if region_match else ""
            )
            scenario_text = (
                f" under `{scenario_match}`" if scenario_match else ""
            )
            alternative_text = (
                f" Models that do have this exact slice include: {', '.join(alternatives[:5])}."
                if alternatives else ""
            )
            return (
                f"I could not find `{variable_match}`{region_text}{scenario_text} "
                f"for the explicitly requested model `{requested_label}`. "
                "I did not substitute results from a different model."
                + alternative_text
            )

        if filtered_data:
            has_year_data = any(_record_has_year_data(record) for record in filtered_data)
            if has_year_data:
                start_year, end_year = _resolved_year_range()
                return (
                    _interpretation_notice(variable_match)
                    + model_relaxed_notice
                    + format_time_series_data(
                        filtered_data, variable_match, region_match, start_year, end_year,
                        workspace_code=workspace_code,
                    )
                )

            same_scope_variables = sorted({
                str(r.get('variable', '')).strip()
                for r in ts_data
                if r and r.get('variable')
                and (not model_match or r.get('modelName') == model_match)
                and (not scenario_match or _scenario_match_ok(r.get('scenario'), scenario_match))
                and (not region_match or regions_equivalent(r.get('region'), region_match))
                and _record_has_year_data(r)
            })
            same_scope_variables = _filter_vars_by_intent(same_scope_variables, variable_intent)
            if same_scope_variables:
                return _choice_prompt(
                    (
                        f"I found records for `{variable_match}` in region "
                        f"`{format_region_label(region_match) if region_match else region_match}` "
                        f"under scenario `{scenario_match or 'any'}`, but they do not include year values."
                    ),
                    "variable",
                    same_scope_variables[:3],
                )
            return (
                f"I found records for `{variable_match}`, but they do not include year values. "
                "Try a different variable or ask for `list variables`."
            )

        family_prompt = _scenario_family_recovery_prompt(
            ts_data,
            variable=variable_match,
            region=region_match,
            scenario=scenario_match,
            model=model_match or None,
        )
        if family_prompt:
            return family_prompt

        matrix_prompt = _matrix_recovery_prompt(
            metadata,
            f"No data found for **{variable_match}** in region `{region_match}` under scenario `{scenario_match}`.",
            variable=variable_match,
            region=region_match,
            scenario=scenario_match,
            model=model_match or None,
        )
        if matrix_prompt:
            return matrix_prompt

        similar_vars = find_similar_available_variables(
            variable_match,
            {str(r.get('variable', '')).strip() for r in ts_data if r and r.get('variable')},
            query_terms={w for w in significant_words if w in {"co2", "emission", "emissions", "capacity", "energy", "demand", "supply"}},
            intent=variable_intent,
            question=question,
            significant_words=significant_words,
        )
        if similar_vars:
            return _confirmation_prompt(
                f"No data found for **{variable_match}** in region `{region_match}` under scenario `{scenario_match}`.",
                similar_vars[0],
                similar_vars[1:],
            )
        recovery_vars = _suggest_recovery_variables(
            question,
            ts_data,
            significant_words=significant_words,
            region=region_match,
            scenario=scenario_match,
            model=model_match,
            limit=3,
        )
        if recovery_vars:
            return _choice_prompt(
                f"No data found for **{variable_match}** in region `{region_match}` under scenario `{scenario_match}`.",
                "variable",
                recovery_vars,
            )
        return (
            f"No data found for **{variable_match}** in region `{region_match}` under scenario `{scenario_match}`. "
            "Try a different variable or ask for `list variables`."
        )

    # If the user mostly specified a region but not a variable, ask for the variable first.
    # This prevents the resolver from guessing an unrelated variable from the wording.
    q_lower = question.lower()
    broad_electricity_request = (
        "electricity" in q_lower
        and not any(token in q_lower for token in ["solar", "wind", "hydro", "nuclear", "oil", "gas", "coal", "hydrogen", "bioenergy", "biomass"])
    )

    def _is_over_specific_electricity_match(variable_name: str | None) -> bool:
        if not broad_electricity_request or not variable_name:
            return False
        lower = str(variable_name).lower()
        return any(
            token in lower
            for token in ["|solar", "|wind", "|hydro", "|nuclear", "|oil", "|gas", "|coal", "|hydrogen", "|bioenergy", "|biomass"]
        )

    variable_hints = [
        "co2", "emission", "emissions", "capacity", "electricity", "energy", "solar",
        "wind", "oil", "gas", "nuclear", "hydro", "investment", "demand", "supply",
        "generation", "temperature", "transport", "industry", "buildings", "power"
    ]
    asks_for_data = _is_data_request(q_lower)
    asks_for_lists = re.search(r"\b(list|available|what)\b.*\b(models?|scenarios?|variables?|regions?|workspaces?)\b", q_lower)
    asks_for_model_info = any(w in q_lower for w in ('info', 'details', 'describe', 'about', 'tell me about'))
    if not forced_choice and region_match and asks_for_data and not asks_for_lists and not asks_for_model_info and not any(h in q_lower for h in variable_hints):
        if _has_meaningful_query_signal(
            question,
            significant_words=significant_words,
            region=region_match,
            ignored_values=query_region_mentions,
            available_variables=available_vars,
        ):
            candidates = _suggest_variable_candidates(
                question,
                variable_dict,
                ts_data,
                ranked_vars=ranked_vars,
                significant_words=significant_words,
                region=region_match,
                ignored_values=query_region_mentions,
                limit=3,
            )
            if candidates:
                return _confirmation_prompt(
                    f"Based on your wording, I think you may mean one of these for `{format_region_label(region_match)}`.",
                    candidates[0],
                    candidates[1:]
                )
        return f"I found the region `{format_region_label(region_match)}`. Which variable should I use?"

    # Confidence threshold: if score is low or ambiguous and query isn't explicit, force clarification
    explicit_variable = "|" in question or extracted['variable'].get('match_type') in ("exact", "templated")
    min_conf = 6
    if any(w in significant_words for w in ["capacity", "investment", "investments", "invest"]):
        min_conf = 4
    # A canonical/preferred match (sentinel score 999 from
    # `_preferred_available_variable`) is already disambiguated, so it must not be
    # discarded by the fuzzy-ambiguity check below. Otherwise a confident alias
    # like "co2 emissions" -> `Emissions|CO2` gets nulled and the YAML fuzzy
    # fallback can pick a wrong variable (e.g. `Emissions|OC`).
    if variable_match and var_score is not None and var_score < 999 and not explicit_variable:
        top1 = ranked_vars[0][1] if ranked_vars else None
        top2 = ranked_vars[1][1] if ranked_vars and len(ranked_vars) > 1 else None
        ambiguous = top1 is not None and top2 is not None and (top1 - top2) < 3
        if var_score < min_conf or ambiguous:
            variable_match = None

    if variable_match and not explicit_variable and not forced_variable:
        if not _variable_matches_query_signal(
            variable_match,
            question,
            variable_intent,
            significant_words=significant_words,
        ):
            variable_match = None
        elif _is_over_specific_electricity_match(variable_match):
            variable_match = None

    available_vars = {str(r.get('variable', '')).strip() for r in ts_data if r and r.get('variable')}

    # If universal resolver didn't find variable, try YAML-based matching as fallback
    if not variable_match and not forced_variable:
        if extracted['variable']['match_type'] not in [None, 'ambiguous']:
            variable_match = extracted['variable']['matched_variable']
            if variable_match and not _variable_matches_query_signal(
                variable_match,
                question,
                variable_intent,
                significant_words=significant_words,
            ):
                variable_match = None
            elif _is_over_specific_electricity_match(variable_match):
                variable_match = None

    # Prefer variables that exist in loaded results; if not available, ask instead of guessing
    if variable_match and variable_match not in available_vars and not forced_variable:
        if broad_electricity_request and not explicit_variable:
            electricity_candidates = _broad_electricity_candidates(available_vars)
            if electricity_candidates:
                return _choice_prompt(
                    "For this broad electricity query, which variable should I use?",
                    "variable",
                    electricity_candidates[:3],
                )
        similar_vars = find_similar_available_variables(
            variable_match,
            available_vars,
            query_terms=set(significant_words),
            intent=variable_intent,
            question=question,
            significant_words=significant_words,
        )
        if similar_vars:
            return _confirmation_prompt(
                f"I matched `{variable_match}`, but that exact variable is not in the loaded data.",
                similar_vars[0],
                similar_vars[1:]
            )
        variable_match = None

    # If still no match, try direct data search
    if not variable_match and not forced_variable:
        q_lower = question.lower()
        def _pick_best(candidates: list[str]) -> str | None:
            ranked = _rank_scored_candidates(
                candidates,
                question,
                variable_intent,
                significant_words=significant_words,
                limit=1,
            )
            return ranked[0] if ranked else None

        preferred_family = _preferred_family_matches(question, available_vars)
        if preferred_family:
            variable_match = _pick_best(preferred_family)
            if _is_over_specific_electricity_match(variable_match):
                variable_match = None

        # Direct search for methane / CH4 variables
        if not variable_match and any(word in q_lower for word in ['methane', 'ch4']):
            methane_vars = [v for v in available_vars if 'methane' in v.lower() or 'ch4' in v.lower()]
            if methane_vars:
                variable_match = _pick_best(methane_vars)

        # Direct search for solar-related variables
        elif not variable_match and any(word in q_lower for word in ['solar', 'pv', 'photovoltaic']):
            solar_vars = [
                v for v in available_vars
                if 'solar' in v.lower()
                and "investment" not in v.lower()
                and "additions" not in v.lower()
            ]
            if solar_vars:
                variable_match = _pick_best(solar_vars)

        # Direct search for wind-related variables
        elif not variable_match and any(word in q_lower for word in ['wind']):
            wind_vars = [
                v for v in available_vars
                if 'wind' in v.lower()
                and "investment" not in v.lower()
                and "additions" not in v.lower()
            ]
            if wind_vars:
                variable_match = _pick_best(wind_vars)

        # Direct search for electricity/generation variables
        elif not variable_match and any(word in q_lower for word in ['electricity', 'generation', 'power']):
            elec_vars = [v for v in available_vars if 'electricity' in v.lower() and 'secondary' in v.lower()]
            if elec_vars:
                variable_match = _pick_best(elec_vars)
                if _is_over_specific_electricity_match(variable_match):
                    variable_match = None

        # Direct search for emissions variables
        elif not variable_match and any(word in q_lower for word in ['emission', 'co2', 'carbon']):
            emission_vars = [v for v in available_vars if 'co2' in v.lower() and 'energy' in v.lower()]
            if emission_vars:
                variable_match = _pick_best(emission_vars)

        # NEW: Direct search for investment variables
        elif not variable_match and any(word in q_lower for word in ['investment', 'investments', 'invest']):
            investment_vars = [v for v in available_vars if 'investment' in v.lower()]
            if investment_vars:
                variable_match = _pick_best(investment_vars)

    # If the user asked for data but no variable could be resolved, guide them to choose one
    if not variable_match and not forced_choice:
        q_lower = question.lower()
        broad_policy_or_ndc = re.fullmatch(
            r"\s*(policy|policies|ndc|ndcs)\s*",
            q_lower,
            flags=re.IGNORECASE,
        )
        if broad_policy_or_ndc:
            if broad_policy_or_ndc.group(1).lower() in {"ndc", "ndcs"}:
                return (
                    "I need one more detail. Are you looking for NDC scenario results, "
                    "NDC ASPECTS workspaces, or a specific variable and region? Try "
                    "`NDC emissions for EU` or `global impacts of NDCs`."
                )
            return (
                "I need one more detail. Are you looking for policy results, NDC ASPECTS workspaces, "
                "or a scenario such as `Current Policies`? Try `current policy emissions for EU` "
                "or `global impacts of NDCs`."
            )
        if ("renewable" in q_lower or "renewables" in q_lower) and ("share" in q_lower or "shares" in q_lower):
            renewable_candidates = _rank_scored_candidates(
                [
                    v for v in available_vars
                    if any(token in v.lower() for token in ["renewable", "renewables", "solar", "wind", "hydro", "bio", "geothermal"])
                    and "investment share" not in v.lower()
                ],
                question,
                variable_intent,
                significant_words=significant_words,
                limit=3,
            )
            if renewable_candidates:
                return _renewable_share_prompt(renewable_candidates)
        if asks_for_data and not asks_for_lists and not asks_for_model_info:
            has_signal = _has_meaningful_query_signal(
                question,
                significant_words=significant_words,
                region=region_match,
                scenario=forced_scenario or _match_scenario_name(question),
                model=forced_model or _match_model_name(question),
                ignored_values=query_region_mentions,
                available_variables=available_vars,
            )
            if not any(h in q_lower for h in variable_hints) and not has_signal:
                if region_match:
                    return f"I found the region `{format_region_label(region_match)}`. Which variable should I use?"
                return "I need one more detail. Which variable should I use?"
            candidates = _suggest_variable_candidates(
                question,
                variable_dict,
                ts_data,
                ranked_vars=ranked_vars,
                significant_words=significant_words,
                region=region_match,
                scenario=forced_scenario or _match_scenario_name(question),
                model=forced_model or _match_model_name(question),
                ignored_values=query_region_mentions,
                limit=3,
            )
            if candidates:
                if ("renewable" in q_lower or "renewables" in q_lower) and ("share" in q_lower or "shares" in q_lower):
                    non_investment_candidates = [c for c in candidates if "investment share" not in c.lower()]
                    if non_investment_candidates:
                        return _renewable_share_prompt(non_investment_candidates)
                return _confirmation_prompt(
                    "Based on your wording, I think one of these is the closest match.",
                    candidates[0],
                    candidates[1:]
                )
            return "I need one more detail. Which variable should I use?"

    # If query is vague and variable match confidence is low, ask for clarification
    q_lower = question.lower()
    broad_electricity_request = (
        any(term in q_lower for term in ["electricity", "power"])
        and not any(
            token in q_lower
            for token in [
                "solar", "wind", "hydro", "nuclear", "oil", "gas", "coal", "hydrogen", "bioenergy", "biomass",
                "capacity", "generation", "demand", "supply", "emission", "emissions", "co2",
                "price", "cost", "investment", "share"
            ]
        )
    )
    variable_hints = [
        "co2", "emission", "emissions", "capacity", "electricity", "energy", "solar",
        "wind", "oil", "gas", "nuclear", "hydro", "investment", "demand", "supply",
        "generation", "temperature", "transport", "industry", "buildings", "power"
    ]
    has_variable_hint = any(h in q_lower for h in variable_hints) or "|" in question
    asks_for_data = _is_data_request(q_lower)
    if not forced_choice and variable_match and asks_for_data and not has_variable_hint:
        candidates = _suggest_variable_candidates(
            question,
            variable_dict,
            ts_data,
            ranked_vars=ranked_vars,
            significant_words=significant_words,
            region=region_match,
            ignored_values=query_region_mentions,
            limit=3,
        )
        if candidates:
            return _confirmation_prompt("I need one more detail.", candidates[0], candidates[1:])
        return "I need one more detail. Which variable should I use?"

    # If we found a variable match, filter and return data
    if variable_match and not forced_choice and _is_capacity_additions_mismatch(question, variable_match):
        variable_match = None

    if variable_match:
        model_match = forced_model or _match_model_name(question)
        scenario_match = forced_scenario or _match_scenario_name(question)
        start_year, end_year = _resolved_year_range()

        if (
            metadata
            and region_match
            and scenario_match
            and not metadata.combination_exists(
                variable_match,
                region=region_match,
                scenario=None if _scenario_is_family_label(scenario_match) else scenario_match,
                model=model_match or None,
            )
        ):
            matrix_prompt = _matrix_recovery_prompt(
                metadata,
                (
                    f"No data found for **{variable_match}** in region `{region_match}` "
                    f"under scenario `{scenario_match}`."
                ),
                variable=variable_match,
                region=region_match,
                scenario=scenario_match,
                model=model_match or None,
            )
            if matrix_prompt:
                return matrix_prompt

        if broad_electricity_request and not forced_choice and not explicit_variable:
            scoped_available_vars = {
                str(r.get('variable', '')).strip()
                for r in ts_data
                if r and r.get('variable')
                and (not region_match or regions_equivalent(r.get('region'), region_match))
                and (not scenario_match or _scenario_match_ok(r.get('scenario'), scenario_match))
                and (not model_match or r.get('modelName') == model_match)
                and _record_has_year_data(r)
            }
            candidates = _broad_electricity_candidates(scoped_available_vars or available_vars)
            if len(candidates) >= 2:
                prefix = "For this electricity query, which variable should I use?"
                if region_match:
                    prefix = f"I found the region `{format_region_label(region_match)}`. For this electricity query, which variable should I use?"
                return _choice_prompt(prefix, "variable", candidates[:3])

        # If region/workspace not specified, suggest options to complete the query
        if not region_match:
            suggest_region_workspace = not _history_has_region_or_workspace(history, region_dict, ts_data)
            regions_for_var = sorted({
                str(r.get('region', '')).strip()
                for r in ts_data
                if r and r.get('variable') == variable_match and r.get('region')
                and (not model_match or r.get('modelName') == model_match)
                and (not scenario_match or _scenario_match_ok(r.get('scenario'), scenario_match))
            })
            workspaces_for_var = sorted({
                str(r.get('workspace_code', '')).strip()
                for r in ts_data
                if r and r.get('variable') == variable_match and r.get('workspace_code')
                and (not model_match or r.get('modelName') == model_match)
                and (not scenario_match or _scenario_match_ok(r.get('scenario'), scenario_match))
            })
            scenarios_for_var = sorted({
                str(r.get('scenario', '')).strip()
                for r in ts_data
                if r and r.get('variable') == variable_match and r.get('scenario')
                and (not model_match or r.get('modelName') == model_match)
            })
            if regions_for_var and len(regions_for_var) > 1:
                region_options = [format_region_label(r) for r in regions_for_var[:3]]
                return _choice_prompt(
                    f"I found variable `{variable_match}`, but I still need the region.",
                    "region",
                    region_options,
                )
        # If region is specified but scenario is missing, prefer scenario choice.
        if region_match and not scenario_match:
            scenarios_for_var = sorted({
                str(r.get('scenario', '')).strip()
                for r in ts_data
                if r and r.get('variable') == variable_match and r.get('scenario')
                and (not model_match or r.get('modelName') == model_match)
                and (not region_match or regions_equivalent(r.get('region'), region_match))
            })
            if scenarios_for_var and len(scenarios_for_var) > 1:
                variable_label = _describe_choice_option("variable", variable_match)
                variable_note = f" `{variable_match}`"
                if variable_label:
                    variable_note += f" ({variable_label})"
                return _choice_prompt(
                    f"I found the variable{variable_note} in `{format_region_label(region_match)}`. I still need the scenario.",
                    "scenario",
                    scenarios_for_var[:3],
                )
        # Validate that this variable actually exists in our loaded data
        if variable_match not in available_vars:
            # For comparison queries, try to find a general capacity variable that exists
            if 'compare' in question.lower() and 'capacity' in question.lower():
                general_capacity_vars = [v for v in available_vars if 'capacity' in v.lower() and 'electricity' in v.lower()]
                if general_capacity_vars:
                    variable_match = general_capacity_vars[0]  # Use the first general capacity variable
                    logger.debug("Using general capacity variable for comparison: %s", variable_match)
                else:
                    # Try to find any capacity variable
                    any_capacity_vars = [v for v in available_vars if 'capacity' in v.lower()]
                    if any_capacity_vars:
                        variable_match = any_capacity_vars[0]
                        logger.debug("Using any capacity variable for comparison: %s", variable_match)

            # If still not found, try to find similar variables
            if variable_match not in available_vars:
                if broad_electricity_request and not forced_choice and not explicit_variable:
                    electricity_candidates = _broad_electricity_candidates(available_vars)
                    if electricity_candidates:
                        return _choice_prompt(
                            "For this broad electricity query, which variable should I use?",
                            "variable",
                            electricity_candidates[:3],
                        )
                key_terms = {"methane", "ch4", "demand", "electricity", "emission", "emissions", "co2", "capacity",
                             "solar", "wind", "oil", "gas", "transport", "industry", "buildings", "final", "primary"}
                query_terms = {w for w in significant_words if w in key_terms}
                similar_vars = find_similar_available_variables(
                    variable_match,
                    available_vars,
                    query_terms=query_terms,
                    intent=variable_intent,
                    question=question,
                    significant_words=significant_words,
                )
                if similar_vars:
                    # For comparison queries, automatically use the first similar variable
                    if 'compare' in question.lower():
                        variable_match = similar_vars[0]
                        logger.debug("Auto-selected similar variable for comparison: %s", variable_match)
                    else:
                        return _confirmation_prompt(
                            f"Variable `{variable_match}` was not found.",
                            similar_vars[0],
                            similar_vars[1:]
                        )
                else:
                    return f"Variable '{variable_match}' not found in loaded data. Available variables include: {', '.join(list(available_vars)[:5])}..."

        # Filter time series data
        filtered_data = []
        for record in ts_data:
            if str(record.get('variable', '')) == variable_match:
                if model_match and record.get('modelName') != model_match:
                    continue
                if scenario_match and not _scenario_match_ok(record.get('scenario'), scenario_match):
                    continue
                if region_match and regions_equivalent(record.get('region'), region_match):
                    filtered_data.append(record)
                elif not region_match:  # No region specified, include all
                    filtered_data.append(record)

        if filtered_data:
            has_year_data = any(_record_has_year_data(record) for record in filtered_data)
            if not has_year_data:
                same_scope_variables = sorted({
                    str(r.get('variable', '')).strip()
                    for r in ts_data
                    if r and r.get('variable')
                    and (not model_match or r.get('modelName') == model_match)
                    and (not scenario_match or _scenario_match_ok(r.get('scenario'), scenario_match))
                    and (not region_match or regions_equivalent(r.get('region'), region_match))
                    and _record_has_year_data(r)
                })
                same_scope_variables = _filter_vars_by_intent(same_scope_variables, variable_intent)
                if same_scope_variables:
                    return _choice_prompt(
                        (
                            f"I found records for `{variable_match}` in region "
                            f"`{format_region_label(region_match) if region_match else region_match}` "
                            f"under scenario `{scenario_match or 'any'}`, but they do not include year values."
                        ),
                        "variable",
                        same_scope_variables[:3],
                    )
                return (
                    f"I found records for `{variable_match}`, but they do not include year values. "
                    "Try a different variable or ask for `list variables`."
                )
            # Format and return the data
            return _interpretation_notice(variable_match) + format_time_series_data(
                filtered_data, variable_match, region_match, start_year, end_year,
                workspace_code=workspace_code,
            )
        else:
            family_prompt = _scenario_family_recovery_prompt(
                ts_data,
                variable=variable_match,
                region=region_match,
                scenario=scenario_match,
                model=model_match or None,
            )
            if family_prompt:
                return family_prompt

            matrix_prompt = _matrix_recovery_prompt(
                metadata,
                f"No data found for **{variable_match}** in region `{region_match}` under scenario `{scenario_match}`.",
                variable=variable_match,
                region=region_match,
                scenario=scenario_match,
                model=model_match or None,
            )
            if matrix_prompt:
                return matrix_prompt

            from collections import Counter

            scoped_regions = sorted({
                str(r.get('region', '')).strip()
                for r in ts_data
                if r and r.get('variable') == variable_match and r.get('region')
                and (not model_match or r.get('modelName') == model_match)
                and (not scenario_match or _scenario_match_ok(r.get('scenario'), scenario_match))
            })
            scoped_scenarios = sorted({
                str(r.get('scenario', '')).strip()
                for r in ts_data
                if r and r.get('variable') == variable_match and r.get('scenario')
                and (not model_match or r.get('modelName') == model_match)
                and (not region_match or regions_equivalent(r.get('region'), region_match))
            })
            key_terms = {
                "methane", "ch4", "demand", "electricity", "emission", "emissions", "co2", "capacity",
                "solar", "wind", "oil", "gas", "transport", "industry", "buildings", "final", "primary",
                "energy", "hydrogen", "nuclear", "hydro", "investment"
            }
            variable_terms = {w for w in significant_words if w in key_terms}

            # If user already provided region and scenario, suggest a closer variable instead of
            # asking again for region/scenario.
            if region_match and scenario_match and not scoped_regions and not scoped_scenarios:
                vars_for_region_scenario = sorted({
                    str(r.get('variable', '')).strip()
                    for r in ts_data
                    if r
                    and r.get('variable')
                    and regions_equivalent(r.get('region'), region_match)
                    and _scenario_match_ok(r.get('scenario'), scenario_match)
                    and (not model_match or r.get('modelName') == model_match)
                })
                similar_vars = find_similar_available_variables(
                    variable_match,
                    set(vars_for_region_scenario) if vars_for_region_scenario else available_vars,
                    query_terms=variable_terms or None,
                    intent=variable_intent,
                    question=question,
                    significant_words=significant_words,
                )
                if similar_vars:
                    return _compact_recovery_prompt(
                        (
                            f"No data found for `{variable_match}` in region "
                            f"`{region_match}` under scenario `{scenario_match}`."
                        ),
                        variable_options=similar_vars[:3],
                    )
                return (
                    f"No data found for `{variable_match}` in region `{region_match}` under scenario "
                    f"`{scenario_match}`. Try `list variables` to pick one that exists for this slice."
                )

            def _top_values(key: str, limit: int = 3, filter_region: bool = False) -> list:
                records = [
                    r for r in ts_data
                    if r and r.get('variable') == variable_match
                    and (not model_match or r.get('modelName') == model_match)
                    and (not scenario_match or _scenario_match_ok(r.get('scenario'), scenario_match))
                    and (
                        not filter_region
                        or (region_match and regions_equivalent(r.get('region'), region_match))
                    )
                ]
                counts = Counter([str(r.get(key, '')).strip() for r in records if r and r.get(key)])
                return [k for k, _ in counts.most_common(limit)]

            if model_match and not scoped_regions and not scoped_scenarios:
                # Fallback: show availability across all models
                all_regions = _top_values("region", limit=3)
                all_scenarios = _top_values("scenario", limit=3)
                region_suggestion = ", ".join(format_region_label(r) for r in all_regions) if all_regions else "none"
                scenario_suggestion = ", ".join(all_scenarios) if all_scenarios else "none"
                ask_target = "region or scenario"
                if region_suggestion == "none" and scenario_suggestion == "none":
                    pool = {
                        str(r.get("variable", "")).strip()
                        for r in ts_data
                        if r and r.get("variable") and _record_has_year_data(r)
                        and (not model_match or r.get("modelName") == model_match)
                    }
                    similar_vars = find_similar_available_variables(
                        variable_match,
                        pool or available_vars,
                        query_terms=variable_terms or None,
                        intent=variable_intent,
                        question=question,
                        significant_words=significant_words,
                    )
                    if similar_vars:
                        return _compact_recovery_prompt(
                            f"No data found for **{variable_match}** in model `{model_match}`.",
                            variable_options=similar_vars[:3],
                            region_options=all_regions[:3],
                            scenario_options=all_scenarios[:3],
                        )
                    ask_target = "variable"
                return (
                    f"No data found for **{variable_match}** in model `{model_match}`.\n\n"
                    f"Across all models, top regions: {region_suggestion}\n"
                    f"Across all models, top scenarios: {scenario_suggestion}\n\n"
                    f"Tell me which {ask_target} you want."
                )

            # Best-effort recommendations when a specific region/scenario is missing
            if region_match and not any(
                regions_equivalent(region_match, value) for value in scoped_regions
            ):
                # N6: prefer the requested sub-region's aggregate (e.g. EU for Germany)
                # over alphabetically/first-by-count regions.
                aggregate_regions = _aggregate_region_candidates(region_match, scoped_regions)
                close_regions = get_close_matches(region_match, scoped_regions, n=3, cutoff=0.6)
                ranked = aggregate_regions + [r for r in close_regions if r not in aggregate_regions]
                region_candidates = ranked or _top_values("region", limit=3)
                scenario_candidates = _top_values("scenario", limit=3)
                region_suggestion = ", ".join(format_region_label(r) for r in region_candidates) if region_candidates else "none"
                scenario_suggestion = ", ".join(scenario_candidates) if scenario_candidates else "none"
                ask_target = "region or scenario"
                if region_suggestion == "none" and scenario_suggestion == "none":
                    pool = {
                        str(r.get("variable", "")).strip()
                        for r in ts_data
                        if r and r.get("variable") and _record_has_year_data(r)
                        and (not model_match or r.get("modelName") == model_match)
                        and (not region_match or regions_equivalent(r.get("region"), region_match))
                    }
                    similar_vars = find_similar_available_variables(
                        variable_match,
                        pool or available_vars,
                        query_terms=variable_terms or None,
                        intent=variable_intent,
                        question=question,
                        significant_words=significant_words,
                    )
                    if similar_vars:
                        return _compact_recovery_prompt(
                            f"No data found for **{variable_match}** in region `{region_match}`.",
                            variable_options=similar_vars[:3],
                            region_options=region_candidates[:3],
                            scenario_options=scenario_candidates[:3],
                        )
                    ask_target = "variable"
                if ask_target == "variable":
                    return (
                        f"No data found for **{variable_match}** in region `{region_match}`.\n\n"
                        f"Recommended regions: {region_suggestion}\n"
                        f"Recommended scenarios: {scenario_suggestion}\n\n"
                        f"Tell me which {ask_target} you want."
                    )
                # Prefer asking for region first when region is missing or mismatched.
                if region_candidates:
                    return _choice_prompt(
                        f"No data found for **{variable_match}** in region `{region_match}`.",
                        "region",
                        [format_region_label(r) for r in region_candidates],
                    )
                if scenario_candidates:
                    return _choice_prompt(
                        f"No data found for **{variable_match}** in region `{region_match}`.",
                        "scenario",
                        scenario_candidates,
                    )
                return (
                    f"No data found for **{variable_match}** in region `{region_match}`.\n\n"
                    f"Recommended regions: {region_suggestion}\n"
                    f"Recommended scenarios: {scenario_suggestion}\n\n"
                    f"Tell me which {ask_target} you want."
                )

            region_suggestion = ", ".join(format_region_label(r) for r in _top_values("region", limit=3)) if scoped_regions else "none"
            scenario_suggestion = ", ".join(_top_values("scenario", limit=3, filter_region=bool(region_match))) if scoped_scenarios else "none"
            model_note = f" for model `{model_match}`" if model_match else ""
            ask_target = "region or scenario"
            if region_suggestion == "none" and scenario_suggestion == "none":
                pool = {
                    str(r.get("variable", "")).strip()
                    for r in ts_data
                    if r and r.get("variable") and _record_has_year_data(r)
                    and (not model_match or r.get("modelName") == model_match)
                    and (not region_match or regions_equivalent(r.get("region"), region_match))
                    and (not scenario_match or _scenario_match_ok(r.get("scenario"), scenario_match))
                }
                similar_vars = find_similar_available_variables(
                    variable_match,
                    pool or available_vars,
                    query_terms=variable_terms or None,
                    intent=variable_intent,
                    question=question,
                    significant_words=significant_words,
                )
                if similar_vars:
                    return _compact_recovery_prompt(
                        f"No data found for **{variable_match}**{model_note}.",
                        variable_options=similar_vars[:3],
                        region_options=_top_values("region", limit=3),
                        scenario_options=_top_values("scenario", limit=3, filter_region=bool(region_match)),
                    )
                ask_target = "variable"
            if ask_target == "variable":
                return (
                    f"No data found for **{variable_match}**{model_note}.\n\n"
                    f"Recommended regions: {region_suggestion}\n"
                    f"Recommended scenarios: {scenario_suggestion}\n\n"
                    f"Tell me which {ask_target} you want."
                )
            if region_suggestion != "none":
                region_options = [format_region_label(r) for r in _top_values("region", limit=3)]
                return _choice_prompt(
                    f"No data found for **{variable_match}**{model_note}.",
                    "region",
                    region_options,
                )
            if scenario_suggestion != "none":
                scenario_options = _top_values("scenario", limit=3, filter_region=bool(region_match))
                return _choice_prompt(
                    f"No data found for **{variable_match}**{model_note}.",
                    "scenario",
                    scenario_options,
                )
            return (
                f"No data found for **{variable_match}**{model_note}.\n\n"
                f"Recommended regions: {region_suggestion}\n"
                f"Recommended scenarios: {scenario_suggestion}\n\n"
                f"Tell me which {ask_target} you want."
            )

    # -------------------------------
    # MODEL INFO REQUESTS
    # -------------------------------
    if any(w in q for w in ('info', 'details', 'describe', 'about', 'tell me about')):
        model_names = sorted({
            str(r.get('modelName', '')).strip()
            for r in model_data
            if r and is_presentable_model_label(r.get('modelName'))
        })
        profile_match = find_model_profile(question)
        if not model_names:
            if profile_match:
                return format_model_profile_answer(profile_match, requested_name=str(profile_match.get("name", "")))
            return "I couldn't find any model metadata. Try reloading the models data."

        query_lower = q

        # Prefer exact substring matches (case-insensitive)
        substring_matches = [m for m in model_names if m.lower() in query_lower or query_lower in m.lower()]
        if not substring_matches:
            # Try matching by token (e.g., 'remind' -> 'REMIND-MAgPIE 3.0').
            # Drop filler words so a stopword like "me" in "tell me about" does
            # not match models such as E3ME/MEDEAS as a substring.
            stopwords = {
                'tell', 'me', 'about', 'the', 'a', 'an', 'model', 'models', 'info',
                'information', 'details', 'describe', 'of', 'for', 'on', 'please',
                'what', 'is', 'are', 'explain', 'give', 'show',
            }
            tokens = [t for t in re.split(r"\W+", query_lower) if t and t not in stopwords and len(t) >= 3]
            for m in model_names:
                m_lower = m.lower()
                if any(t in m_lower for t in tokens):
                    substring_matches.append(m)

        # Fuzzy fallback
        if not substring_matches:
            substring_matches = get_close_matches(query_lower, model_names, n=3, cutoff=0.5)

        if not substring_matches:
            if profile_match:
                return format_model_profile_answer(profile_match, requested_name=str(profile_match.get("name", "")))
            return "I couldn't match that to a known model. Try `list models` to see available options."

        # When the profile the user named (e.g. "MESSAGEix-GLOBIOM") merely
        # *contains* an unrelated ts model as a substring (e.g. "GLOBIO"), trust
        # the profile instead of the incidental fragment.
        if profile_match:
            profile_name = str(profile_match.get("name", ""))
            if profile_name and all(m.lower() in profile_name.lower() for m in substring_matches):
                return format_model_profile_answer(profile_match, requested_name=profile_name)

        if len(substring_matches) > 1:
            if profile_match:
                matched_profile_names = {
                    str((find_model_profile(match) or {}).get("name", ""))
                    for match in substring_matches
                }
                if matched_profile_names == {str(profile_match.get("name", ""))}:
                    return format_model_profile_answer(
                        profile_match,
                        requested_name=str(profile_match.get("name", "")),
                    )
            sample = ", ".join(substring_matches[:5])
            return f"I found multiple model matches: {sample}. Which one do you want details for?"

        model_name = substring_matches[0]
        records = [r for r in model_data if r and r.get('modelName') == model_name]
        rec = records[0] if records else {}

        return _format_model_info_answer(model_name, rec, asks_assumptions=False)

    # -------------------------------
    # HELP COMMAND
    # -------------------------------
    if 'help' in q:
        return (
            "Tell me what you want to do and I'll help. Examples:\n"
            "- Ask about models: `list models` or `info GCAM`\n"
            "- Explore variables: `list variables` or `plot CO2 emissions`\n"
            "- Visualize results: `plot emissions for GCAM`\n"
            "- To make plots, you can ask questions like:\n"
            "  * `plot [variable name]`\n"
            "  * `show me a plot of [variable name]`\n"
            "  * `graph [variable name] for [model name]`\n"
            "  * `visualize [variable name]`\n"
            "If you want more conversational guidance, just say 'suggest' or ask a question in plain language."
        )

    # -------------------------------
    # FALLBACK
    # -------------------------------
    if _is_data_request(q) and not forced_choice:
        broad_policy_or_ndc = bool(re.fullmatch(r"\s*(policy|policies|ndc|ndcs)\s*", q, flags=re.IGNORECASE))
        if broad_policy_or_ndc:
            return (
                "I need one more detail. Are you looking for policy results, NDC ASPECTS workspaces, "
                "or a scenario such as `Current Policies`? Try `current policy emissions for EU` "
                "or `global impacts of NDCs`."
            )
        has_signal = _has_meaningful_query_signal(
            question,
            significant_words=significant_words,
            region=region_match,
            ignored_values=query_region_mentions,
            available_variables=available_vars,
        )
        if not any(h in q for h in variable_hints) and not has_signal:
            if region_match:
                return f"I found the region `{format_region_label(region_match)}`. Which variable should I use?"
            return "I need one more detail. Which variable should I use?"
        candidates = _suggest_variable_candidates(
            question,
            variable_dict,
            ts_data,
            ranked_vars=ranked_vars,
            significant_words=significant_words,
            region=region_match,
            ignored_values=query_region_mentions,
            limit=3,
        )
        if candidates:
            return _confirmation_prompt(
                "Based on your wording, I think one of these is the closest match.",
                candidates[0],
                candidates[1:]
            )
        return "I need one more detail. Which variable should I use?"
    return ""


def find_similar_available_variables(
    requested_var: str,
    available_vars: set,
    query_terms: set | None = None,
    intent: str | None = None,
    question: str | None = None,
    significant_words: list | None = None,
) -> list:
    """Find similar variables that exist in the loaded data."""
    available_list = list(available_vars)
    if intent:
        filtered = _filter_vars_by_intent(available_list, intent, strict=True)
        if not filtered:
            filtered = _filter_vars_by_intent(available_list, intent, strict=False)
        available_list = filtered
    requested_norm = re.sub(r"\s+", "", str(requested_var or "").lower())

    # Try exact substring matches first
    substring_matches = [
        v for v in available_list
        if (requested_var.lower() in v.lower() or v.lower() in requested_var.lower())
        and re.sub(r"\s+", "", v.lower()) != requested_norm
    ]
    if query_terms:
        substring_matches = [v for v in substring_matches if any(t in v.lower() for t in query_terms)]
    if substring_matches:
        reference_question = question or requested_var
        return _rank_scored_candidates(
            substring_matches,
            reference_question,
            intent or "general",
            significant_words=significant_words,
            limit=3,
        )

    # Fall back to fuzzy matching
    fuzzy_matches = get_close_matches(requested_var, available_list, n=3, cutoff=0.4)
    fuzzy_matches = [v for v in fuzzy_matches if re.sub(r"\s+", "", v.lower()) != requested_norm]
    if query_terms:
        fuzzy_matches = [v for v in fuzzy_matches if any(t in v.lower() for t in query_terms)]
    if not fuzzy_matches:
        return []
    reference_question = question or requested_var
    return _rank_scored_candidates(
        fuzzy_matches,
        reference_question,
        intent or "general",
        significant_words=significant_words,
        limit=3,
    )

def _normalize_display_unit(unit: object, variable: str = "") -> str:
    """Normalize harmless spelling variants without changing unit meaning."""
    value = re.sub(r"\s+", " ", str(unit or "").strip())
    value = re.sub(r"/\s*(?:y|year)(?=\b)", "/yr", value, flags=re.IGNORECASE)
    if str(variable or "").casefold().startswith("price|carbon"):
        # Source records spell the same denominator as /tCO2, /t CO2 and /t.
        # Keep the currency/year prefix intact and normalize only that suffix.
        value = re.sub(r"/\s*t(?:\s*co2)?\s*$", "/t CO2", value, flags=re.IGNORECASE)
    return value or "N/A"


def _unit_dimension(unit: object) -> str:
    """Return a broad physical dimension used to catch mixed-unit outliers.

    This deliberately classifies only well-known unit tokens. Unknown or
    compound units are left untouched rather than guessed.
    """
    value = _normalize_display_unit(unit).casefold()
    compact = re.sub(r"\s+", "", value)
    if value == "n/a":
        return "unknown"
    if re.search(r"(?:^|[^a-z])(?:ej|pj|tj|gj|mj|kj)(?:[^a-z]|$)", value):
        return "energy"
    if re.search(r"(?:^|[^a-z])(?:kwh|mwh|gwh|twh)(?:[^a-z]|$)", value):
        return "energy"
    if re.search(r"(?:^|[^a-z])(?:kg|kt|mt|gt|tonne|tonnes|ton|tons)(?:[^a-z]|$)", value):
        return "mass"
    if re.search(r"(?:^|[^a-z])(?:kw|mw|gw|tw)(?:[^a-z]|$)", value):
        return "power"
    if any(token in compact for token in ("usd", "eur", "gbp", "jpy", "€", "$", "£")):
        return "currency"
    if re.search(r"(?:^|[^a-z])(?:person|persons|people|capita)(?:[^a-z]|$)", value):
        return "population"
    return "unknown"


def format_time_series_data(
    data_records: list,
    variable: str,
    region: str = "",
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    workspace_code: str = "",
) -> str:
    """Format time series data into a readable table."""
    if not data_records:
        return f"No data found for variable '{variable}'{' in ' + region if region else ''}."

    record_workspaces = sorted({
        str(record.get("workspace_code") or "").strip()
        for record in data_records if str(record.get("workspace_code") or "").strip()
    })
    if not workspace_code and len(record_workspaces) > 1:
        entries = [
            entry for entry in _workspace_entries(data_records)
            if str(entry.get("code") or "") in record_workspaces
        ]
        return _workspace_choice_response(entries)
    if not workspace_code and len(record_workspaces) == 1:
        workspace_code = record_workspaces[0]

    # Group by scenario and model. Groups are validated against the requested
    # year window below before any heading, scope metadata, or row is rendered.
    scenario_groups = {}
    for record in data_records:
        key = (str(record.get('scenario', 'Unknown')), str(record.get('modelName', 'Unknown')),
               _normalize_display_unit(record.get('unit'), variable))
        if key not in scenario_groups:
            scenario_groups[key] = []
        scenario_groups[key].append(record)

    def _year_scope_text(years: list[str]) -> str:
        if is_latest_year_filter(start_year, end_year):
            return "latest available"
        if start_year is not None or end_year is not None:
            if start_year == end_year and start_year is not None:
                return str(start_year)
            start_label = str(start_year) if start_year is not None else "first available"
            end_label = str(end_year) if end_year is not None else "latest available"
            return f"{start_label}-{end_label}"
        if not years:
            return "available years"
        return f"{years[0]}-{years[-1]}" if len(years) > 1 else years[0]

    all_years: list[str] = []
    for record in data_records:
        all_years.extend([
            str(key) for key, value in record.items()
            if str(key).isdigit() and is_finite_numeric_value(value)
        ])
        nested_years = record.get("years")
        if isinstance(nested_years, dict):
            all_years.extend([
                str(key) for key, value in nested_years.items()
                if str(key).isdigit() and is_finite_numeric_value(value)
            ])
    selected_years = select_years(
        sorted({str(year) for year in all_years}, key=lambda y: int(y)),
        start_year,
        end_year,
    )
    def _value_for_year(records: list, year: str):
        for record in records:
            if year in record and is_finite_numeric_value(record[year]):
                return record[year]
            nested_years = record.get("years")
            if (
                isinstance(nested_years, dict)
                and year in nested_years
                and is_finite_numeric_value(nested_years[year])
            ):
                return nested_years[year]
        return None

    visible_groups = {}
    group_units = {}
    for key, records in scenario_groups.items():
        values = {
            year: _value_for_year(records, str(year))
            for year in selected_years
        }
        values = {year: value for year, value in values.items() if value is not None}
        if not values:
            continue
        visible_groups[key] = (records, values)
        raw_unit = next((record.get("unit") for record in records if record.get("unit")), "N/A")
        group_units[key] = _normalize_display_unit(raw_unit, variable)

    if not visible_groups:
        place = f" in {region}" if region else ""
        period = _year_scope_text(selected_years)
        return f"No data found for variable '{variable}'{place} for the requested period ({period})."

    # When one physical dimension clearly dominates, do not silently mix a
    # conflicting outlier into the same table. Unknown units are retained; a
    # tie is also retained because there is no principled dominant dimension.
    dimension_counts = {}
    for key in visible_groups:
        dimension = _unit_dimension(group_units[key])
        if dimension != "unknown":
            dimension_counts[dimension] = dimension_counts.get(dimension, 0) + 1
    excluded_groups = []
    if len(dimension_counts) > 1:
        ranked_dimensions = sorted(dimension_counts.items(), key=lambda item: (-item[1], item[0]))
        dominant_dimension, dominant_count = ranked_dimensions[0]
        if dominant_count > ranked_dimensions[1][1]:
            for key in list(visible_groups):
                dimension = _unit_dimension(group_units[key])
                if dimension not in {"unknown", dominant_dimension}:
                    scenario, model, _unit = key
                    excluded_groups.append((display_model_label(model), scenario, group_units[key]))
                    visible_groups.pop(key)
                    group_units.pop(key, None)

    units = sorted(set(group_units.values()))
    scenarios = sorted({scenario for scenario, _model, _unit in visible_groups})
    raw_models = sorted({model for _scenario, model, _unit in visible_groups})
    models = sorted({display_model_label(model) for model in raw_models})

    def _format_value(value: object) -> str:
        if isinstance(value, (int, float)):
            if abs(value) >= 1e6:
                return f"{value/1e6:.1f}M"
            if abs(value) >= 1e3:
                return f"{value/1e3:.1f}K"
            return f"{value:.2f}"
        return str(value)

    def _markdown_cell(value: object) -> str:
        return str(value).replace("\n", " ").replace("|", "\\|")

    def _series_sort_key(key: tuple[str, str, str]) -> tuple:
        scenario, model, unit = key
        return (model.casefold(), scenario.casefold(), model, scenario, unit)

    ordered_group_keys = sorted(visible_groups, key=_series_sort_key)
    use_compact_matrix = len(ordered_group_keys) > DATA_COMPACT_GROUP_THRESHOLD

    response = f"### {variable}"
    if region:
        response += f" in {region}"
    response += "\n\n"
    scenario_scope = scenarios[0] if len(scenarios) == 1 else "multiple"
    model_scope = display_model_label(raw_models[0]) if len(raw_models) == 1 else "multiple"
    # The formatter has the authoritative unit after removing empty or
    # physically incompatible groups. Record that resolved value alongside
    # the other scope dimensions so the manager does not retain an extractor
    # guess from before the data was filtered (for example ``EJ/yr`` for
    # Population or ``Mt CO2/yr`` for electricity output).
    unit_scope = units[0] if len(units) == 1 else ("multiple" if units else "N/A")
    record_resolved_scope(
        variable=variable,
        region=region,
        scenario=scenario_scope,
        scenarios=scenarios,
        result_models=models,
        unit=unit_scope,
        start_year=int(selected_years[0]) if selected_years else start_year,
        end_year=int(selected_years[-1]) if selected_years else end_year,
        all_scenarios=len(scenarios) > 1,
        action="query",
        workspace_code=workspace_code,
    )
    response += (
        f"Scope: scenario `{scenario_scope}`, model `{model_scope}`, "
        f"years `{_year_scope_text(selected_years)}`\n"
    )
    response += f"Unit: `{unit_scope}`\n\n"
    if excluded_groups:
        excluded = ", ".join(
            f"`{model} - {scenario}` ({unit})"
            for model, scenario, unit in excluded_groups
        )
        response += (
            "Note: excluded incompatible unit group(s) from this table: "
            f"{excluded}.\n\n"
        )
    response += "Answer:\n"

    if use_compact_matrix:
        total_series = len(ordered_group_keys)
        display_limit = max(1, DATA_GROUP_DISPLAY_LIMIT)
        displayed_keys = ordered_group_keys[:display_limit]
        displayed_count = len(displayed_keys)
        model_label = "model" if len(raw_models) == 1 else "models"
        scenario_label = "scenario" if len(scenarios) == 1 else "scenarios"
        response += (
            f"Summary: {total_series} series across {len(raw_models)} {model_label} "
            f"and {len(scenarios)} {scenario_label}.\n"
        )
        if displayed_count < total_series:
            remaining = total_series - displayed_count
            response += (
                f"Showing series 1-{displayed_count} of {total_series}, ordered by model "
                "and scenario. "
                f"Narrow by model or scenario to view the {remaining} remaining "
                "series.\n"
            )
        else:
            response += "Series are ordered by model and scenario.\n"
        response += "\n"

        year_columns = [year for year in selected_years if str(year).isdigit()]
        header_cells = ["Model", "Scenario", *year_columns, "Unit"]
        response += "| " + " | ".join(header_cells) + " |\n"
        response += "|" + "|".join("---" for _ in header_cells) + "|\n"
        for scenario, model, series_unit in displayed_keys:
            _records, values = visible_groups[(scenario, model, series_unit)]
            value_cells = [
                _format_value(values[year]) if year in values else "—"
                for year in year_columns
            ]
            row_cells = [
                _markdown_cell(display_model_label(model)),
                _markdown_cell(scenario),
                *(_markdown_cell(value) for value in value_cells),
                _markdown_cell(group_units[(scenario, model, series_unit)]),
            ]
            response += "| " + " | ".join(row_cells) + " |\n"
        response += "\n"
    else:
        # Preserve the familiar detailed layout for small scopes. Iterating in
        # insertion order keeps this path backward compatible with prior output.
        for (scenario, model, series_unit), (_records, values) in visible_groups.items():
            response += f"**{display_model_label(model)} - {scenario}**\n"

            response += "| Year | Value | Unit |\n|------|-------|------|\n"
            unit = group_units[(scenario, model, series_unit)]

            for year, value in values.items():
                if not str(year).isdigit():
                    continue
                response += f"| {year} | {_format_value(value)} | {unit} |\n"

            response += "\n"

    workspace_entry = _workspace_catalog().get(str(workspace_code or "").strip(), {})
    explorer_url = str(workspace_entry.get("explorer_url") or "https://iamparis.eu/results")
    response += (
        "For interactive charts and additional filters, use the "
        f"[IAM PARIS data explorer]({explorer_url}).\n"
    )

    return response
