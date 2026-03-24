import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime
import logging
import base64
from io import BytesIO
import requests.exceptions

from simple_plotter import simple_plot_query
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
    format_region_label,
)
from utils.yaml_loader import load_all_yaml_files
from difflib import get_close_matches
import pickle
import os

# Create cached versions of YAML loading
def get_cached_yaml_definitions():
    # Try file cache
    cache_file = "cache/yaml_dicts.pkl"
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
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


def _normalize_free_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _token_set(text: str) -> set[str]:
    return {tok for tok in re.findall(r"[a-z0-9]+", _normalize_free_text(text)) if tok}


def _looks_like_plot_request(text: str) -> bool:
    tokens = _token_set(text)
    return bool(tokens & {"plot", "graph", "chart", "visualize", "visualise"})


def _looks_like_comparison_request(text: str) -> bool:
    q = _normalize_free_text(text)
    tokens = _token_set(text)
    return bool(
        {"compare", "comparison", "versus", "vs"} & tokens
        or re.search(r"\bcompare\b", q)
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
        "trajectory", "renewable", "renewables",
    }
    if tokens & data_terms:
        return True
    q = _normalize_free_text(text)
    return bool(re.search(r"\btime\s+series\b", q) or re.search(r"\bunder\s+different\s+scenarios\b", q))


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
    if re.search(r"\bwhat\s+kinds?\s+of\s+data\b", q):
        return True
    if re.search(r"\bhelp\s+me\s+find\s+data\b", q):
        return True
    return False


def _looks_like_model_info_request(text: str) -> bool:
    q = _normalize_free_text(text)
    tokens = _token_set(text)
    if tokens & {
        "info", "information", "details", "describe", "about", "explain",
        "assumption", "assumptions", "methodology", "structure", "works",
    }:
        return True
    return bool(
        re.search(r"\btell\s+me\s+about\b", q)
        or re.search(r"\bhow\s+does\b", q)
        or re.search(r"\bwhat\s+are\s+the\s+assumptions\b", q)
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
    if tokens & valid_names and tokens & explicit_list_terms:
        if tokens & {"price", "trajectory", "trend", "plot", "graph", "chart", "compare", "growth", "share", "emissions", "capacity", "gdp"}:
            return False
        return True
    return bool(
        re.search(rf"\bwhat\s+(?:{category_pattern})\s+(?:are\s+)?(?:available|included)\b", q)
        or re.search(
            rf"\b(?:what|which)\s+(?:{category_pattern})\s+can\s+you\s+"
            r"(?:plot|graph|chart|visuali[sz]e|show|display)\b",
            q,
        )
        or re.search(rf"\bwhich\s+(?:{category_pattern})\b", q)
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
        if len(deduped) >= limit:
            break
    return deduped


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
) -> bool:
    """
    Decide whether the user's free-text query contains enough semantic signal
    to offer ranked variable choices instead of asking a generic follow-up.
    """
    profile = _query_profile(question, significant_words)
    terms = set(profile["terms"])

    for value in [region, scenario, model]:
        if value:
            terms -= _tokenize_text(str(value))

    weak_terms = {
        "data", "show", "give", "plot", "graph", "chart", "value", "values",
        "time", "series", "timeseries", "query", "information", "info"
    }
    terms = {term for term in terms if term not in weak_terms}

    if profile["metric_terms"] or profile["sector_terms"] or profile["energy_terms"]:
        return True
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
        if "capacity" in ql:
            candidates.extend(
                v for v in available_vars
                if "solar" in v.lower() and "capacity|electricity" in v.lower()
            )
        if any(token in ql for token in ["energy", "electricity", "power", "generation"]):
            candidates.extend(
                v for v in available_vars
                if "solar" in v.lower()
                and (
                    "secondary energy|electricity" in v.lower()
                    or "generation|electricity" in v.lower()
                    or "capacity|electricity" in v.lower()
                )
                and "investment" not in v.lower()
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
        and (not region or r.get('region') == region)
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
        if region and record.get('region') != region:
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
    else:
        scoped_vars = set(_filter_vars_by_intent(scoped_vars, intent, strict=False))

    candidates = _rank_variable_candidates(
        question,
        variable_dict,
        scoped_vars,
        ranked_vars=ranked_vars,
        significant_words=significant_words,
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
            and (not rg or (region and r.get('region') == region))
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
            if region and record.get('region') != region and scenario and record.get('scenario') != scenario:
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
            if region_scope and str(record.get("region", "")).strip() not in region_scope:
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
    lines = [prefix]
    if variable_options:
        rendered = []
        for option in variable_options[:3]:
            label = _describe_choice_option("variable", option)
            rendered.append(f"`{option}`" + (f" ({label})" if label else ""))
        lines.append("Closest variables: " + ", ".join(rendered))
    if region_options:
        rendered = []
        for option in region_options[:3]:
            label = _describe_choice_option("region", option)
            rendered.append(f"`{option}`" + (f" ({label})" if label and label != option else ""))
        lines.append("Closest regions: " + ", ".join(rendered))
    if scenario_options:
        rendered = []
        for option in scenario_options[:3]:
            label = _describe_choice_option("scenario", option)
            rendered.append(f"`{option}`" + (f" ({label})" if label else ""))
        lines.append("Closest scenarios: " + ", ".join(rendered))
    lines.append("Reply with the variable, region, or scenario you want to use next.")
    return "\n\n".join(lines)


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


def data_query(
    question: str,
    model_data: list,
    ts_data: list,
    history: list | None = None,
    forced_entities: dict | None = None,
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
    # Only treat this as a hard choice once a variable has been explicitly selected.
    # Region/scenario/model alone should still allow the normal guided clarification flow.
    forced_choice = bool(forced_variable)
    model_names = sorted({str(m.get("modelName", "")).strip() for m in model_data if m and m.get("modelName")})

    def _norm_model(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", (text or "").lower())

    def _build_model_alias_map(names: list[str]) -> dict[str, set[str]]:
        alias_map: dict[str, set[str]] = {}
        for name in names:
            raw = str(name or "").strip()
            if not raw:
                continue
            low = raw.lower()
            norm = _norm_model(raw)
            words = [w for w in re.split(r"[^a-z0-9]+", low) if w]
            aliases = {low, norm}
            if words:
                aliases.add(words[0])
                if len(words) >= 2:
                    aliases.add(" ".join(words[:2]))
                    aliases.add("".join(words[:2]))
                alpha_prefix = re.match(r"[a-z]+", words[0] or "")
                if alpha_prefix and len(alpha_prefix.group(0)) >= 4:
                    aliases.add(alpha_prefix.group(0))
            for alias in aliases:
                if not alias:
                    continue
                alias_map.setdefault(alias, set()).add(raw)
        return alias_map

    model_alias_map = _build_model_alias_map(model_names)

    def _extract_model_hint(query: str) -> str:
        ql = (query or "").lower()
        # model GCAM / using GCAM / with GCAM / for GCAM
        m = re.search(
            r"\b(?:model|using|with|for)\s+([a-z0-9][a-z0-9\-\._ ]{1,50})",
            ql,
        )
        if not m:
            return ""
        raw = m.group(1).strip()
        # Stop at next dimension marker.
        raw = re.split(r"\b(?:under|scenario|region|workspace|from|between|during|in)\b", raw)[0].strip()
        return raw

    def _resolve_model_candidates(query: str) -> list[str]:
        if not model_names:
            return []
        query_lower = (query or "").lower()
        query_norm = _norm_model(query)

        # 1) Exact full-name containment.
        exact_hits = [
            name
            for name in model_names
            if re.search(r"(?<!\w)" + re.escape(name.lower()) + r"(?!\w)", query_lower)
        ]
        if exact_hits:
            return exact_hits

        # 2) Alias-based lookup using query n-grams.
        tokens = [t for t in re.split(r"[^a-z0-9]+", query_lower) if t]
        spans = set(tokens)
        for i in range(len(tokens)):
            if i + 1 < len(tokens):
                spans.add(tokens[i] + " " + tokens[i + 1])
                spans.add(tokens[i] + tokens[i + 1])
            if i + 2 < len(tokens):
                spans.add(tokens[i] + " " + tokens[i + 1] + " " + tokens[i + 2])
                spans.add(tokens[i] + tokens[i + 1] + tokens[i + 2])
        if query_norm:
            spans.add(query_norm)

        alias_hits: set[str] = set()
        for span in spans:
            alias_hits.update(model_alias_map.get(span, set()))
        if alias_hits:
            # Prefer names that exactly match one token/alias and shorter canonical names.
            def _rank(name: str) -> tuple[int, int, str]:
                low = name.lower()
                exact_alias = 1 if low in spans or _norm_model(name) in spans else 0
                return (exact_alias, -len(name), name)
            return sorted(alias_hits, key=_rank, reverse=True)

        # 3) Fuzzy fallback on normalized model names.
        from difflib import get_close_matches
        if len(query_norm) < 4 or len(query_norm) > 24:
            return []
        norm_to_name = {_norm_model(n): n for n in model_names}
        fuzzy = get_close_matches(query_norm, list(norm_to_name.keys()), n=3, cutoff=0.84)
        return [norm_to_name[f] for f in fuzzy if f in norm_to_name]

    def _match_model_name(query: str) -> str:
        query_lower = (query or "").lower()
        # Exact full-name containment without additional gating.
        direct = [
            name for name in model_names
            if re.search(r"(?<!\w)" + re.escape(name.lower()) + r"(?!\w)", query_lower)
        ]
        if direct:
            return direct[0]

        hint = _extract_model_hint(query)
        candidates = _resolve_model_candidates(hint) if hint else []
        if not candidates and not hint:
            # Fast path for queries like "CO2 for GCAM": accept exact standalone token
            for name in model_names:
                nlow = name.lower()
                if re.fullmatch(r"[a-z0-9\-_\.]+", nlow) and re.search(r"(?<!\w)" + re.escape(nlow) + r"(?!\w)", query_lower):
                    return name
        return candidates[0] if candidates else ""

    def _match_scenario_name(query: str) -> str:
        scenarios = sorted({str(r.get('scenario', '')).strip() for r in ts_data if r and r.get('scenario')})
        if not scenarios:
            return ""
        ql = query.lower()
        # Explicit "under X" or "scenario X"
        m = re.search(r"(?:under|scenario)\s+([\\w\\-\\.]+)", ql)
        if m:
            token = m.group(1)
            for s in scenarios:
                if token.lower() in s.lower():
                    return s
        # SSP / RCP tokens
        for token in re.findall(r"(ssp\\d|rcp\\d(?:\\.\\d)?)", ql):
            for s in scenarios:
                if token.lower() in s.lower():
                    return s
        return ""

    def _extract_year_range(text: str) -> tuple[Optional[int], Optional[int]]:
        m = re.search(r"\b(19\d{2}|20\d{2})\s*(?:-|to|–|—)\s*(19\d{2}|20\d{2})\b", text)
        if m:
            return int(m.group(1)), int(m.group(2))
        m = re.search(r"\b(19\d{2}|20\d{2})\b", text)
        if m:
            y = int(m.group(1))
            return y, y
        return None, None

    def _is_data_request(text: str) -> bool:
        return _looks_like_data_request(text)

    # Route variable-discovery phrasing to the variable list path.
    if _looks_like_category_list_request(question, "variables") and _looks_like_plot_request(question):
        vars = sorted({str(r.get('variable', '')) for r in ts_data if r and r.get('variable')})
        if not vars:
            return "I don't see any variables in the loaded dataset. Try reloading or check the IAM PARIS results website."

        sample = vars[:12] if len(vars) > 8 else vars
        more = "" if len(vars) <= len(sample) else f" and {len(vars)-len(sample)} more"
        sample_str = "\n- ".join(sample)
        return (f"I can work with these variables:\n- {sample_str}{more}\n\n"
                "Try queries like 'Capacity|Electricity|Solar|Utility for Greece' or 'plot [variable name] in Greece'.")

    # Discovery mode: the user wants help understanding what is available
    discovery_phrases = [
        r"\bi don'?t know\b",
        r"\bi'?m not sure\b",
        r"\bnot sure what\b",
        r"\bexplore data\b",
    ]
    if _looks_like_discovery_request(question) or any(re.search(pattern, q) for pattern in discovery_phrases):
        models = sorted({str(m.get('modelName', '')).strip() for m in model_data if m and m.get('modelName')})
        variables = sorted({str(r.get('variable', '')).strip() for r in ts_data if r and r.get('variable')})
        scenarios = sorted({str(r.get('scenario', '')).strip() for r in ts_data if r and r.get('scenario')})
        regions = sorted({str(r.get('region', '')).strip() for r in ts_data if r and r.get('region')})
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

    # -------------------------------
    # Handle COMPARISON QUERIES (route to plotter)
    # -------------------------------
    if _looks_like_comparison_request(question):
        return simple_plot_query(question, model_data, ts_data)

    # -------------------------------
    # Handle PLOTTING QUERIES
    # -------------------------------
    if _looks_like_plot_request(question) or (_looks_like_data_request(question) and any(word in q for word in ['show', 'display'])):
        q_lower = q.lower()
        variable_hints = [
            "co2", "emission", "emissions", "capacity", "electricity", "energy", "solar",
            "wind", "oil", "gas", "nuclear", "hydro", "investment", "demand", "supply",
            "generation", "temperature", "transport", "industry", "buildings", "power"
        ]
        has_variable_hint = any(h in q_lower for h in variable_hints) or "|" in question
        is_explicit_plot = any(word in q for word in ['plot', 'graph', 'visualize'])

        # If the user only said "show" without a clear variable, don't guess—fall through to clarification
        if not has_variable_hint and not is_explicit_plot:
            pass
        else:
            # Try universal natural language resolution first
            natural_variable = resolve_natural_language_variable_universal(question, variable_dict)
            if natural_variable:
                # Check if this variable exists in our data
                available_vars = {str(r.get('variable', '')) for r in ts_data if r and r.get('variable')}
                if natural_variable in available_vars:
                    # Use simple_plot_query with resolved variable
                    return simple_plot_query(question, model_data, ts_data)

            # Fallback to existing plotting logic
            return simple_plot_query(question, model_data, ts_data)

    # -------------------------------
    # LIST AVAILABLE MODELS
    # -------------------------------
    if _looks_like_category_list_request(question, "models"):
        models = sorted({r.get('modelName', '') for r in model_data if r and r.get('modelName')})
        if not models:
            return "I couldn't find any models in the data right now. Try `help` or refresh the data."

        if len(models) <= 6:
            model_str = ", ".join(models[:-1]) + (" and " + models[-1] if len(models) > 1 else models[0])
            return f"I found these models in the IAM PARIS dataset: {model_str}. Which one would you like to know more about?"

        return (f"There are {len(models)} models available. "
                "You can ask for details about a specific model using `info [model name]`, "
                "or say `list variables` to see the kinds of outputs available.")

    # -------------------------------
    # LIST AVAILABLE VARIABLES
    # -------------------------------
    if _looks_like_category_list_request(question, "variables"):
        vars = sorted({str(r.get('variable', '')) for r in ts_data if r and r.get('variable')})
        if not vars:
            return "I don't see any variables in the loaded dataset. Try reloading or check the IAM PARIS results website."

        # Filter for energy-related variables if "energy" is mentioned
        if 'energy' in q:
            energy_vars = [v for v in vars if any(term in v.lower() for term in ['energy', 'electricity', 'capacity', 'power', 'generation', 'solar', 'wind', 'hydro', 'gas', 'nuclear', 'biomass'])]
            if energy_vars:
                vars = energy_vars[:15]  # Show more energy variables
            else:
                vars = vars[:8]

        sample = vars[:12] if len(vars) > 8 else vars
        more = "" if len(vars) <= len(sample) else f" and {len(vars)-len(sample)} more"
        sample_str = "\n- ".join(sample)
        return (f"I can work with these variables:\n- {sample_str}{more}\n\n"
                "Try queries like 'Capacity|Electricity|Solar|Utility for Greece' or 'plot [variable name] in Greece'.")

    # -------------------------------
    # LIST AVAILABLE SCENARIOS
    # -------------------------------
    if _looks_like_category_list_request(question, "scenarios"):
        scenarios = sorted({r.get('scenario', '') for r in ts_data if r and r.get('scenario')})
        if not scenarios:
            return "No scenarios are loaded in the current dataset. Try a different query or check IAM PARIS results."

        sample = scenarios[:8]
        more = "" if len(scenarios) <= 8 else f" and {len(scenarios)-8} more"
        sample_str = ", ".join(sample[:-1]) + (" and " + sample[-1] if len(sample) > 1 else sample[0])
        return f"I found scenarios like {sample_str}{more}. You can plot variables for any of these scenarios."

    # -------------------------------
    # LIST AVAILABLE REGIONS
    # -------------------------------
    if _looks_like_category_list_request(question, "regions"):
        regions = sorted({str(r.get('region', '')).strip() for r in ts_data if r and r.get('region')})
        if not regions:
            return "No regions are loaded in the current dataset. Try a different query or check IAM PARIS results."

        sample = regions[:10]
        more = "" if len(regions) <= 10 else f" and {len(regions)-10} more"
        sample_str = ", ".join(sample[:-1]) + (" and " + sample[-1] if len(sample) > 1 else sample[0])
        return f"I found regions like {sample_str}{more}. You can plot variables for any of these regions."

    # -------------------------------
    # LIST ALL MODELS, RESULTS, AND WORKSPACES
    # -------------------------------
    if re.search(r"\b(list|get)\b.*\b(all|available)\b.*\b(models?|results?|workspaces?)\b", q) or \
       re.search(r"\b(models?|results?|workspaces?)\b.*\b(list|get|available)\b", q) or \
       re.search(r"\b(list|get)\b.*\b(models?|results?|workspaces?)\b.*\b(and|,)\b.*\b(models?|results?|workspaces?)\b", q):
        models = sorted({r.get('modelName', '') for r in model_data if r and r.get('modelName')})
        variables = sorted({str(r.get('variable', '')) for r in ts_data if r and r.get('variable')})
        scenarios = sorted({r.get('scenario', '') for r in ts_data if r and r.get('scenario')})
        workspaces = get_available_workspaces(ts_data)

        response = "### Available Models, Results, and Workspaces\n\n"
        response += f"**Models ({len(models)}):**\n" + ", ".join(models) + "\n\n"
        response += f"**Results - Variables ({len(variables)}):**\n" + ", ".join(variables[:10]) + (f" and {len(variables)-10} more" if len(variables) > 10 else "") + "\n\n"
        response += f"**Results - Scenarios ({len(scenarios)}):**\n" + ", ".join(scenarios[:10]) + (f" and {len(scenarios)-10} more" if len(scenarios) > 10 else "") + "\n\n"
        response += f"**Workspaces ({len(workspaces)}):**\n" + ", ".join(workspaces) + "\n\n"
        response += "For more details on any item, ask specific questions like 'list models' or 'plot [variable]'."
        return response

    # -------------------------------
    # MODEL INFO REQUESTS
    # -------------------------------
    explicit_model_question = bool(re.search(r"\bwhat\s+is\b", q) or re.search(r"\bwho\s+is\b", q))
    hinted_model = _extract_model_hint(question)
    candidate_model_matches = _resolve_model_candidates(hinted_model or question)
    if (_looks_like_model_info_request(question) and (candidate_model_matches or 'model' in q)) or ('model' in q) or (explicit_model_question and candidate_model_matches):
        if not model_names:
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
                    sample = ", ".join(ts_matches[:5])
                    return (
                        "I found matching model names in the results data, but no metadata description is available. "
                        f"Examples: {sample}. If you want, ask for plots or values for one of these models."
                    )

            return "I couldn't match that to a known model. Try `list models` to see available options."

        if len(substring_matches) > 1:
            sample = ", ".join(substring_matches[:5])
            return f"I found multiple model matches: {sample}. Which one do you want details for?"

        model_name = substring_matches[0]
        records = [r for r in model_data if r and r.get('modelName') == model_name]
        rec = records[0] if records else {}

        desc = str(rec.get('description', '') or '').strip()
        asum = str(rec.get('assumptions', '') or '').strip()
        source = str(rec.get('source', '') or '').strip()
        asks_assumptions = bool(re.search(r"\bassumption\b|\bassumptions\b", q))

        parts = [f"### {model_name}"]
        if asks_assumptions and asum:
            parts.append(f"**Assumptions:** {asum}")
            if desc:
                parts.append(f"**Model description:** {desc}")
        elif asks_assumptions:
            if desc:
                parts.append(desc)
            parts.append("No explicit assumptions field is available in the model metadata.")
        elif desc:
            parts.append(desc)
        if asum and not asks_assumptions:
            parts.append(f"**Assumptions:** {asum}")
        if source:
            parts.append(f"**Source:** {source}")

        if len(parts) == 1:
            return f"I found the model `{model_name}`, but no description was provided in metadata."

        return "\n\n".join(parts)

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
    if resolved and not forced_variable:
        variable_match, var_score, matched_words, significant_words = resolved
        if isinstance(variable_match, str):
            variable_match = variable_match.strip()
    variable_intent = _infer_variable_intent(question, significant_words)
    region_match = None

    # Always try to extract region, regardless of variable match success
    region_candidates = sorted({str(r.get('region', '')).strip() for r in ts_data if r and r.get('region')})
    extracted = extract_variable_and_region_from_query(question, variable_dict, region_dict, region_candidates)
    region_match = extracted['region']
    if re.search(r"\b(world|global)\b", question.lower()):
        region_match = "World"
    if forced_variable:
        variable_match = forced_variable
        var_score = 999
    if forced_region:
        region_match = forced_region

    if forced_choice:
        model_match = forced_model or _match_model_name(question)
        scenario_match = forced_scenario or _match_scenario_name(question)
        if not variable_match:
            return "I need one more detail. Which variable should I use?"

        filtered_data = []
        for r in ts_data:
            if r is None:
                continue
            if str(r.get('variable', '')) != variable_match:
                continue
            if model_match and r.get('modelName') != model_match:
                continue
            if scenario_match and r.get('scenario') != scenario_match:
                continue
            if region_match and r.get('region') != region_match:
                continue
            filtered_data.append(r)

        if filtered_data:
            has_year_data = any(_record_has_year_data(record) for record in filtered_data)
            if has_year_data:
                start_year, end_year = _extract_year_range(question)
                return format_time_series_data(filtered_data, variable_match, region_match, start_year, end_year)

            same_scope_variables = sorted({
                str(r.get('variable', '')).strip()
                for r in ts_data
                if r and r.get('variable')
                and (not model_match or r.get('modelName') == model_match)
                and (not scenario_match or r.get('scenario') == scenario_match)
                and (not region_match or r.get('region') == region_match)
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
        ):
            candidates = _suggest_variable_candidates(
                question,
                variable_dict,
                ts_data,
                ranked_vars=ranked_vars,
                significant_words=significant_words,
                region=region_match,
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
    if variable_match and var_score is not None and not explicit_variable:
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
            solar_vars = [v for v in available_vars if 'solar' in v.lower()]
            if solar_vars:
                variable_match = _pick_best(solar_vars)

        # Direct search for wind-related variables
        elif not variable_match and any(word in q_lower for word in ['wind']):
            wind_vars = [v for v in available_vars if 'wind' in v.lower()]
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
        start_year, end_year = _extract_year_range(question)

        if broad_electricity_request and not forced_choice and not explicit_variable:
            scoped_available_vars = {
                str(r.get('variable', '')).strip()
                for r in ts_data
                if r and r.get('variable')
                and (not region_match or r.get('region') == region_match)
                and (not scenario_match or r.get('scenario') == scenario_match)
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
                and (not scenario_match or r.get('scenario') == scenario_match)
            })
            workspaces_for_var = sorted({
                str(r.get('workspace_code', '')).strip()
                for r in ts_data
                if r and r.get('variable') == variable_match and r.get('workspace_code')
                and (not model_match or r.get('modelName') == model_match)
                and (not scenario_match or r.get('scenario') == scenario_match)
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
                    "I found the variable, but I still need the region.",
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
                and (not region_match or r.get('region') == region_match)
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
                if scenario_match and record.get('scenario') != scenario_match:
                    continue
                if region_match and record.get('region') == region_match:
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
                    and (not scenario_match or r.get('scenario') == scenario_match)
                    and (not region_match or r.get('region') == region_match)
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
            return format_time_series_data(filtered_data, variable_match, region_match, start_year, end_year)
        else:
            from collections import Counter

            scoped_regions = sorted({
                str(r.get('region', '')).strip()
                for r in ts_data
                if r and r.get('variable') == variable_match and r.get('region')
                and (not model_match or r.get('modelName') == model_match)
                and (not scenario_match or r.get('scenario') == scenario_match)
            })
            scoped_scenarios = sorted({
                str(r.get('scenario', '')).strip()
                for r in ts_data
                if r and r.get('variable') == variable_match and r.get('scenario')
                and (not model_match or r.get('modelName') == model_match)
                and (not region_match or r.get('region') == region_match)
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
                    and r.get('region') == region_match
                    and r.get('scenario') == scenario_match
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
                    and (not scenario_match or r.get('scenario') == scenario_match)
                    and (not filter_region or (region_match and r.get('region') == region_match))
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
            if region_match and region_match not in scoped_regions:
                close_regions = get_close_matches(region_match, scoped_regions, n=3, cutoff=0.6)
                region_candidates = close_regions or _top_values("region", limit=3)
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
                        and (not region_match or r.get("region") == region_match)
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
                    and (not region_match or r.get("region") == region_match)
                    and (not scenario_match or r.get("scenario") == scenario_match)
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
        model_names = sorted({r.get('modelName', '') for r in model_data if r and r.get('modelName')})
        if not model_names:
            return "I couldn't find any model metadata. Try reloading the models data."

        query_lower = q

        # Prefer exact substring matches (case-insensitive)
        substring_matches = [m for m in model_names if m.lower() in query_lower or query_lower in m.lower()]
        if not substring_matches:
            # Try matching by token (e.g., 'remind' -> 'REMIND-MAgPIE 3.0')
            tokens = [t for t in re.split(r"\W+", query_lower) if t]
            for m in model_names:
                m_lower = m.lower()
                if any(t in m_lower for t in tokens):
                    substring_matches.append(m)

        # Fuzzy fallback
        if not substring_matches:
            substring_matches = get_close_matches(query_lower, model_names, n=3, cutoff=0.5)

        if not substring_matches:
            return "I couldn't match that to a known model. Try `list models` to see available options."

        if len(substring_matches) > 1:
            sample = ", ".join(substring_matches[:5])
            return f"I found multiple model matches: {sample}. Which one do you want details for?"

        model_name = substring_matches[0]
        records = [r for r in model_data if r and r.get('modelName') == model_name]
        rec = records[0] if records else {}

        desc = str(rec.get('description', '') or '').strip()
        asum = str(rec.get('assumptions', '') or '').strip()
        source = str(rec.get('source', '') or '').strip()

        parts = [f"### {model_name}"]
        if desc:
            parts.append(desc)
        if asum:
            parts.append(f"**Assumptions:** {asum}")
        if source:
            parts.append(f"**Source:** {source}")

        if len(parts) == 1:
            return f"I found the model `{model_name}`, but no description was provided in metadata."

        return "\n\n".join(parts)

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
        has_signal = _has_meaningful_query_signal(
            question,
            significant_words=significant_words,
            region=region_match,
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

def format_time_series_data(
    data_records: list,
    variable: str,
    region: str = "",
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
) -> str:
    """Format time series data into a readable table."""
    if not data_records:
        return f"No data found for variable '{variable}'{' in ' + region if region else ''}."

    # Group by scenario and model
    scenario_groups = {}
    for record in data_records:
        key = (str(record.get('scenario', 'Unknown')), str(record.get('modelName', 'Unknown')))
        if key not in scenario_groups:
            scenario_groups[key] = []
        scenario_groups[key].append(record)

    response = f"### {variable}"
    if region:
        response += f" in {region}"
    response += "\n\n"

    for (scenario, model), records in scenario_groups.items():
        response += f"**{model} - {scenario}**\n"

        # Get year columns
        years = []
        for record in records:
            years.extend([k for k in record.keys() if str(k).isdigit()])
            nested_years = record.get("years")
            if isinstance(nested_years, dict):
                years.extend([k for k in nested_years.keys() if str(k).isdigit()])
        years = sorted({str(year) for year in years}, key=lambda y: int(y))

        if not years:
            response += "No year data available\n\n"
            continue

        # Create table header
        response += "| Year | Value | Unit |\n|------|-------|------|\n"

        # Get data for each year
        unit = records[0].get('unit', 'N/A') if records else 'N/A'

        for year in years:
            try:
                year_int = int(year)
            except Exception:
                continue
            if start_year and year_int < start_year:
                continue
            if end_year and year_int > end_year:
                continue
            # Find value for this year (take first record that has it)
            value = None
            for record in records:
                if str(year) in record and record[str(year)] is not None:
                    value = record[str(year)]
                    break
                nested_years = record.get("years")
                if isinstance(nested_years, dict) and str(year) in nested_years and nested_years[str(year)] is not None:
                    value = nested_years[str(year)]
                    break

            if value is not None:
                # Format large numbers
                if isinstance(value, (int, float)):
                    if abs(value) >= 1e6:
                        formatted_value = f"{value/1e6:.1f}M"
                    elif abs(value) >= 1e3:
                        formatted_value = f"{value/1e3:.1f}K"
                    else:
                        formatted_value = f"{value:.2f}"
                else:
                    formatted_value = str(value)
                response += f"| {year} | {formatted_value} | {unit} |\n"

        response += "\n"

    return response
