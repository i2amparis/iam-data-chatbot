import re
import warnings
import logging
import functools
import threading
import pandas as pd
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# matplotlib's pyplot keeps global figure state that is NOT thread-safe. FastAPI
# runs sync endpoints in a worker threadpool, so concurrent plot requests could
# interleave plt.figure()/savefig()/close() and corrupt each other's output.
# Serialize all plot entrypoints behind a reentrant lock (entrypoints can call
# one another, hence RLock).
_PLOT_LOCK = threading.RLock()


def _serialized_plot(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with _PLOT_LOCK:
            return func(*args, **kwargs)
    return wrapper
from typing import List, Dict, Any, Optional, Tuple
from utils_query import (
    match_variable_from_yaml,
    extract_region_from_query,
    find_closest_variable_name,
    resolve_natural_language_variable_universal,
    resolve_natural_language_variable_candidates,
    resolve_natural_language_variable_with_score,
    resolve_natural_language_variable_ranked,
    format_region_label,
)
from model_aliases import match_model_name
from year_filters import extract_year_range, select_years


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


def _pretty_variable_name(variable: str) -> str:
    value = str(variable or "").strip()
    lower = value.lower()

    if "capacity|electricity|solar" in lower:
        return "Solar Capacity"
    if "capacity|electricity|wind" in lower:
        return "Wind Capacity"
    if "secondary energy|electricity|solar" in lower:
        return "Solar Electricity"
    if "secondary energy|electricity|wind" in lower:
        return "Wind Electricity"
    if "secondary energy|electricity" in lower:
        return "Electricity Generation"
    if "emissions|co2" in lower or lower.startswith("gross emissions|co2"):
        return "CO2 Emissions"
    if "final energy" in lower and "oil" in lower:
        return "Oil Final Energy Demand"
    if "primary energy" in lower and "oil" in lower:
        return "Oil Primary Energy"

    parts = [part.strip() for part in value.split("|") if part.strip()]
    if not parts:
        return value
    if parts[0].lower() == "capacity" and len(parts) >= 3:
        return f"{parts[-1]} Capacity"
    if parts[0].lower() == "secondary energy" and len(parts) >= 3 and parts[1].lower() == "electricity":
        return f"{parts[-1]} Electricity"
    if parts[0].lower().startswith("emissions") and len(parts) >= 2:
        return f"{parts[1]} Emissions"
    return parts[-1]


def _year_range_text(start_year: int | None = None, end_year: int | None = None) -> str:
    if start_year or end_year:
        return f" ({start_year or '?'}-{end_year or '?'})"
    return ""


def _format_compact_options(label: str, values: list[str]) -> str:
    if not values:
        return ""
    formatted = ", ".join(f"`{str(value).strip()}`" for value in values[:3] if str(value).strip())
    if not formatted:
        return ""
    return f"\n- Closest {label}: {formatted}"


def _compact_plot_recovery_prompt(
    message: str,
    variable_options: list[str] | None = None,
    region_options: list[str] | None = None,
    scenario_options: list[str] | None = None,
) -> str:
    parts = [message.strip()]
    parts.append(_format_compact_options("variables", variable_options or []))
    parts.append(_format_compact_options("regions", region_options or []))
    parts.append(_format_compact_options("scenarios", scenario_options or []))
    parts.append("\nReply with the option you want to use.")
    return "".join(part for part in parts if part)


def _matrix_plot_recovery_prompt(
    metadata: Any | None,
    message: str,
    variable: str | None = None,
    region: str | None = None,
    scenario: str | None = None,
    model: str | None = None,
) -> str | None:
    if not metadata:
        return None

    def _same_family_variable_options(limit: int = 3) -> list[str]:
        if not variable:
            return []
        base = str(variable or "").strip()
        base_lower = base.lower()
        all_variables = sorted(getattr(metadata, "all_variables", set()) or [])
        if not all_variables:
            return []

        def valid(candidate: str) -> bool:
            candidate_lower = candidate.lower()
            if candidate == base:
                return False
            if "solar" in base_lower:
                return "solar" in candidate_lower and "investment" not in candidate_lower and "additions" not in candidate_lower
            if "wind" in base_lower:
                return "wind" in candidate_lower and "investment" not in candidate_lower and "additions" not in candidate_lower
            if base_lower.startswith("gdp"):
                return candidate_lower.startswith("gdp")
            if base_lower.startswith("emissions|co2"):
                return candidate_lower.startswith("emissions|co2")
            family = base.split("|", 1)[0].lower()
            return bool(family and candidate_lower.startswith(family))

        return [candidate for candidate in all_variables if valid(candidate)][:limit]

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

    region_options = [opt for opt in options.get("regions", []) if opt != region]
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
    return _compact_plot_recovery_prompt(
        message,
        variable_options=variable_options,
        region_options=region_options,
        scenario_options=scenario_options,
    )


def _plot_subject(variable: str, region: str | None = None) -> str:
    label = _pretty_variable_name(variable)
    if region:
        return f"{label} in {region}"
    return label


def _preferred_plot_family_matches(question: str, available_vars: set[str]) -> list[str]:
    """
    Keep plot-side variable resolution aligned with data-side resolution for
    common plain-language requests like "solar energy" and "oil demand".
    """
    ql = str(question or "").lower()
    candidates: list[str] = []

    solar_terms = ("solar", "pv", "photovoltaic", "photovoltaics")
    if any(term in ql for term in solar_terms):
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
            and any(token in v.lower() for token in ["final energy", "primary energy", "secondary energy", "demand"])
            and "investment" not in v.lower()
        )

    if "electricity" in ql and not any(
        token in ql for token in [
            "solar", "wind", "hydro", "nuclear", "oil", "gas", "coal", "hydrogen", "bioenergy", "biomass",
            "capacity", "generation", "demand", "supply", "emission", "emissions", "co2",
            "price", "cost", "investment", "share",
        ]
    ):
        candidates.extend(
            v for v in available_vars
            if v in {
                "Secondary Energy|Electricity",
                "Final Energy|Electricity",
                "Capacity|Electricity",
            }
        )

    deduped: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in deduped:
            deduped.append(candidate)

    def _score(candidate: str) -> tuple[int, int, int, str]:
        lower = candidate.lower()
        exact_capacity = lower == "capacity|electricity|solar"
        exact_solar_electricity = lower == "secondary energy|electricity|solar"
        exact_oil_final = lower == "final energy|oil"
        exact_oil_primary = lower == "primary energy|oil"
        exact_electricity = lower == "secondary energy|electricity"
        broad_energy = (
            "secondary energy|electricity|solar" in lower
            or "capacity|electricity|solar" in lower
            or "final energy|oil" in lower
            or "primary energy|oil" in lower
            or lower == "secondary energy|electricity"
            or lower == "final energy|electricity"
            or lower == "capacity|electricity"
        )
        return (
            0 if (exact_solar_electricity or exact_capacity or exact_oil_final or exact_oil_primary or exact_electricity) else 1,
            0 if broad_energy else 1,
            lower.count("|"),
            len(lower),
        )

    return sorted(deduped, key=_score)


def _wrap_plot_markdown(
    plot_str: str,
    variable: str,
    region: str | None = None,
    scenario: str | None = None,
    scenarios_in_data: list | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    prefix: str = "Showing",
) -> str:
    subject = _plot_subject(variable, region)
    years = _year_range_text(start_year, end_year)
    if scenario:
        caption = f"{prefix} {subject} for scenario `{scenario}`{years}."
    elif scenarios_in_data and len([s for s in scenarios_in_data if s]) > 1:
        caption = f"{prefix} {subject} across available scenarios{years}."
    else:
        caption = f"{prefix} {subject}{years}."
    return caption + "\n" + plot_str


def save_plot_to_base64() -> str:
    """Return the current matplotlib figure as an inline markdown image."""
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()
    return f"![Plot](data:image/png;base64,{img_base64})"


from utils.yaml_loader import load_all_yaml_files
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# Global metadata instance (lazy loaded)
_metadata = None
_metadata_signature = None
logger = logging.getLogger(__name__)

# Suppress tight_layout warnings globally for cleaner output
warnings.filterwarnings(
    "ignore",
    message="Tight layout not applied.*",
    category=UserWarning
)

def get_metadata(ts_data: List[Dict] = None, models: List[Dict] = None):
    """Get or create DataMetadata instance."""
    global _metadata, _metadata_signature
    signature = None
    if ts_data is not None:
        sample = []
        for record in list(ts_data[:3]) + list(ts_data[-3:] if len(ts_data) > 3 else []):
            if not record:
                continue
            sample.append((
                record.get("variable"),
                record.get("region"),
                record.get("scenario"),
                record.get("modelName"),
            ))
        signature = (len(ts_data), tuple(sample))
    if ts_data is not None and (_metadata is None or signature != _metadata_signature):
        from data_metadata import build_metadata_with_cache
        _metadata = build_metadata_with_cache(ts_data, models)
        _metadata_signature = signature
    return _metadata


def generate_llm_suggestion(query: str, variable: str, region: str, 
                            available_regions: List[str], available_scenarios: List[str],
                            api_key: str) -> str:
    """
    Use LLM to generate helpful suggestions when data is not found.
    
    Args:
        query: Original user query
        variable: Requested variable
        region: Requested region
        available_regions: List of available regions for the variable
        available_scenarios: List of available scenarios for the variable
        api_key: OpenAI API key
        
    Returns:
        Helpful suggestion message
    """
    llm = ChatOpenAI(
        model_name="gpt-4-turbo",
        temperature=0.7,
        timeout=30,
        max_retries=1,
        api_key=api_key
    )
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("""You are a helpful assistant for the IAM PARIS climate data platform.

A user requested data that is not available. Generate a helpful, friendly response that:

1. Explains that the specific data combination is not available
2. Suggests similar alternatives from the available data
3. Offers to help the user find what they're looking for

Be concise and helpful. Use Markdown formatting.

## Context:
- Requested variable: {variable}
- Requested region: {region}
- Available regions for this variable: {available_regions}
- Available scenarios for this variable: {available_scenarios}

Generate a helpful response:"""),
        HumanMessagePromptTemplate.from_template("User query: {query}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({
        "query": query,
        "variable": variable,
        "region": region,
        "available_regions": ", ".join(available_regions[:20]) if available_regions else "None",
        "available_scenarios": ", ".join(available_scenarios[:10]) if available_scenarios else "None"
    })
    
    return response.content


def detect_multi_variable_comparison(query: str) -> List[str]:
    """
    Detect if query is asking to compare multiple variables.
    
    Args:
        query: User query string
        
    Returns:
        List of variable keywords found in comparison context
    """
    query_lower = query.lower()
    
    # Patterns that indicate multi-variable comparison
    comparison_patterns = [
        r'compare\s+(\w+)\s+(?:and|vs|versus|with)\s+(\w+)',
        r'(\w+)\s+(?:and|vs|versus)\s+(\w+)\s+(?:capacity|generation|energy|emissions)',
        r'(\w+)\s+vs\s+(\w+)',  # Simple "X vs Y" pattern
        r'both\s+(\w+)\s+and\s+(\w+)',
        r'(\w+)\s+or\s+(\w+)',
    ]
    
    for pattern in comparison_patterns:
        match = re.search(pattern, query_lower)
        if match:
            return [match.group(1), match.group(2)]
    
    return []


def detect_region_comparison(question: str, metadata) -> List[str]:
    """
    Detect region comparison like 'USA vs EU' and return matched regions.
    """
    if not metadata:
        return []
    ql = question.lower()

    def _match_region(text: str) -> str | None:
        t = text.lower()
        if re.search(r"\busa\b|\bunited\s+states\b|\bu\.s\.\b|\bus\b", t):
            return "USA"
        if re.search(r"\beu\b|\beurope\b|\beuropean\b", t):
            return "EU"
        if re.search(r"\bchina\b|\bchn\b", t):
            return "CHN"
        if re.search(r"\bindia\b|\bind\b", t):
            return "IND"
        # Prefer exact region names if present in text
        all_regions = sorted({reg for regs in metadata.variable_regions.values() for reg in regs})
        candidates = []
        for r in all_regions:
            if r and r.lower() in t:
                candidates.append(r)
        if candidates:
            return max(candidates, key=len)
        return metadata._find_best_region_match(text)

    if " vs " in ql or " versus " in ql:
        splitter = " vs " if " vs " in ql else " versus "
        left, right = ql.split(splitter, 1)
        r1 = _match_region(left)
        r2 = _match_region(right)
        if r1 and r2 and r1 != r2:
            return [r1, r2]
    # Handle "between X and Y", "for X and Y", or "in X and Y"
    if " and " in ql and (" between " in ql or " for " in ql or " in " in ql):
        if " between " in ql:
            anchor = " between "
        elif " for " in ql:
            anchor = " for "
        else:
            anchor = " in "
        tail = ql.rsplit(anchor, 1)[-1]
        parts = [p.strip() for p in tail.split(" and ") if p.strip()]
        if len(parts) >= 2:
            r1 = _match_region(parts[0])
            r2 = _match_region(parts[1])
            if r1 and r2 and r1 != r2:
                return [r1, r2]
    return []


@_serialized_plot
def plot_variable_across_regions(question: str, model_data: List[Dict], ts_data: List[Dict],
                                 variable: str, regions: List[str],
                                 scenario: str = None, start_year: int = None,
                                 end_year: int = None) -> str:
    """
    Plot a single variable across multiple regions.
    """
    if not variable or not regions:
        return "Could not identify enough regions to compare."
    metadata = get_metadata(ts_data, model_data)

    # Collect data for each region
    all_data = {}
    unit = None
    for region in regions:
        filtered = []
        for r in ts_data:
            if r is None:
                continue
            if str(r.get('variable', '')) != variable:
                continue
            if scenario and r.get('scenario') != scenario:
                continue
            if region and str(r.get('region', '')).lower() != region.lower():
                continue
            filtered.append(r)
        if filtered:
            all_data[region] = filtered
            if unit is None:
                unit = filtered[0].get('unit', '')

    if not all_data:
        region_scope = regions[0] if len(regions) == 1 else None
        recovery = _matrix_plot_recovery_prompt(
            metadata,
            f"No data found for **{variable}** in the requested regions.",
            variable=variable,
            region=region_scope,
            scenario=scenario,
        )
        return recovery or f"No data found for **{variable}** in the requested regions."

    plt.figure(figsize=(12, 7))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'D', 'v', '<']
    scenarios_in_data = set()

    for idx, (region, data) in enumerate(all_data.items()):
        df = pd.DataFrame(data)
        if 'years' in df.columns:
            years_df = df['years'].apply(pd.Series)
            df = df.drop('years', axis=1).join(years_df)

        year_cols = [col for col in df.columns if str(col).isdigit()]
        if start_year or end_year:
            filtered_year_cols = []
            for col in year_cols:
                year_int = int(col)
                if start_year and year_int < start_year:
                    continue
                if end_year and year_int > end_year:
                    continue
                filtered_year_cols.append(col)
            if filtered_year_cols:
                year_cols = filtered_year_cols

        if len(df) > 1:
            df = df.groupby('scenario').first().reset_index()

        # Plot first row per region (scenario already filtered above)
        row = df.iloc[0]
        if 'scenario' in row:
            scenarios_in_data.add(str(row.get('scenario', '')).strip())
        values = [row.get(str(year), 0) for year in sorted(year_cols, key=int)]
        plt.plot(sorted(year_cols, key=int), values,
                 label=region,
                 color=colors[idx % len(colors)],
                 marker=markers[idx % len(markers)],
                 linewidth=2)

    title = f"{_pretty_variable_name(variable)}: " + " vs ".join(format_region_label(r) for r in regions)
    plt.title(title, fontsize=12, fontweight='bold')
    plt.xlabel("Year", fontsize=10)
    if unit:
        plt.ylabel(f"Value ({unit})", fontsize=10)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plot_str = save_plot_to_base64()
    return _wrap_plot_markdown(plot_str, variable, None, scenario, list(scenarios_in_data), start_year, end_year, prefix="Showing")


@_serialized_plot
def plot_multiple_variables(question: str, model_data: List[Dict], ts_data: List[Dict],
                            variables: List[str], region: str = None, 
                            scenario: str = None, start_year: int = None, 
                            end_year: int = None) -> str:
    """
    Generate a plot comparing multiple variables.
    
    Args:
        question: Original user query
        model_data: List of model metadata
        ts_data: List of time series data
        variables: List of variable names to compare (can be keywords or exact names)
        region: Optional region filter
        scenario: Optional scenario filter
        start_year: Optional start year for filtering
        end_year: Optional end year for filtering
        
    Returns:
        Base64 encoded PNG image or error message
    """
    metadata = get_metadata(ts_data, model_data)
    
    # Check if variables are already exact names (from LLM extraction)
    # or if they need to be resolved (from regex detection)
    resolved_variables = []
    for var in variables:
        # Check if it's an exact variable name
        exact_match = False
        for r in ts_data:
            if r and r.get('variable') == var:
                resolved_variables.append(var)
                exact_match = True
                logger.debug("Using exact variable: '%s'", var)
                break
        
        # If not exact, try to resolve using metadata
        if not exact_match and metadata:
            suggestions = metadata.suggest_variables(var, limit=3)
            if suggestions:
                resolved_variables.append(suggestions[0][0])
                logger.debug("Resolved '%s' to '%s'", var, suggestions[0][0])
    
    if len(resolved_variables) < 2:
        return f"Could not identify enough variables to compare. Found: {resolved_variables}"
    
    # Extract region from query if not provided
    if region is None:
        # Check for common region names in query first
        region_keywords = ['world', 'europe', 'eu', 'usa', 'china', 'india', 'africa', 'asia', 'greece', 'germany', 'brazil']
        question_lower = question.lower()
        for kw in region_keywords:
            if kw in question_lower:
                if metadata:
                    matched = metadata._find_best_region_match(kw)
                    if matched:
                        region = matched
                        break
                else:
                    region = kw.title()
                    break
        
        # Fallback: Use metadata to find region in query
        if region is None and metadata:
            region = metadata._find_best_region_match(question)
    
    logger.debug("Using region: %s", region)
    
    # Collect data for each variable
    all_data = {}
    units = {}
    
    for variable in resolved_variables:
        filtered_data = []
        for r in ts_data:
            if r is None:
                continue
            if str(r.get('variable', '')) != variable:
                continue
            if scenario and r.get('scenario') != scenario:
                continue
            if region:
                r_region = str(r.get('region', ''))
                if r_region.lower() != region.lower():
                    continue
            filtered_data.append(r)
        
        if filtered_data:
            all_data[variable] = filtered_data
            units[variable] = filtered_data[0].get('unit', '')
    
    if not all_data:
        return f"No data found for the requested variables in region '{region}'."
    
    # Create comparison plot
    plt.figure(figsize=(12, 7))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'D', 'v', '<']
    
    scenarios_in_data = set()
    for idx, (variable, data) in enumerate(all_data.items()):
        df = pd.DataFrame(data)
        
        # Handle years column
        if 'years' in df.columns:
            years_df = df['years'].apply(pd.Series)
            df = df.drop('years', axis=1).join(years_df)
        
        # Get year columns
        year_cols = [col for col in df.columns if str(col).isdigit()]
        if not year_cols:
            continue
        
        # Filter to specific year range if requested
        if start_year or end_year:
            filtered_year_cols = []
            for col in year_cols:
                year_int = int(col)
                if start_year and year_int < start_year:
                    continue
                if end_year and year_int > end_year:
                    continue
                filtered_year_cols.append(col)
            if filtered_year_cols:
                year_cols = filtered_year_cols
        
        # Aggregate if multiple rows (take mean)
        if len(df) > 1:
            # Group by scenario and take first of each
            df = df.groupby('scenario').first().reset_index()
        
        # Plot each row
        for _, row in df.iterrows():
            label = variable
            if len(df) > 1:
                label = f"{variable} ({row.get('scenario', '')})"
            if 'scenario' in row:
                scenarios_in_data.add(str(row.get('scenario', '')).strip())
            
            values = [row.get(str(year), 0) for year in sorted(year_cols, key=int)]
            plt.plot(sorted(year_cols, key=int), values, 
                    label=label, 
                    color=colors[idx % len(colors)],
                    marker=markers[idx % len(markers)],
                    linewidth=2)
    
    # Build title
    title = f"Comparison: {' vs '.join([_pretty_variable_name(v) for v in all_data.keys()])}"
    if region:
        title += f" for {format_region_label(region)}"
    
    plt.title(title, fontsize=12, fontweight='bold')
    plt.xlabel("Year", fontsize=10)
    
    # Use first unit as Y-axis label (assuming same units)
    if units:
        first_unit = list(units.values())[0]
        plt.ylabel(f"Value ({first_unit})", fontsize=10)
    
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save to base64
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    plot_str = f"![Plot](data:image/png;base64,{img_base64})"
    caption_var = " vs ".join([_pretty_variable_name(v) for v in all_data.keys()])
    return _wrap_plot_markdown(plot_str, caption_var, region, scenario, list(scenarios_in_data), start_year, end_year, prefix="Showing comparison of")


@_serialized_plot
def plot_model_comparison(question: str, model_data: List[Dict], ts_data: List[Dict],
                          variable: str, models: List[str], region: str = None,
                          scenario: str = None, start_year: int = None, 
                          end_year: int = None) -> str:
    """
    Generate a plot comparing the same variable across different models.
    
    Args:
        question: Original user query
        model_data: List of model metadata
        ts_data: List of time series data
        variable: Variable name to compare
        models: List of model names to compare
        region: Optional region filter
        scenario: Optional scenario filter
        start_year: Optional start year for filtering
        end_year: Optional end year for filtering
        
    Returns:
        Base64 encoded PNG image or error message
    """
    metadata = get_metadata(ts_data, model_data)
    
    # Resolve variable name if needed
    resolved_variable = None
    for r in ts_data:
        if r and r.get('variable') == variable:
            resolved_variable = variable
            break
    
    if not resolved_variable and metadata:
        suggestions = metadata.suggest_variables(variable, limit=3)
        if suggestions:
            resolved_variable = suggestions[0][0]
            logger.debug("Resolved variable '%s' to '%s'", variable, resolved_variable)
    
    if not resolved_variable:
        return f"Could not identify variable '{variable}'."
    
    # Resolve model names (fuzzy match)
    # Note: ts_data uses 'modelName' field, not 'model'
    resolved_models = []
    available_models = sorted({str(r.get('modelName', '') or r.get('model', '')) for r in ts_data if r and (r.get('modelName') or r.get('model'))})
    logger.debug("ts_data length: %s", len(ts_data))
    logger.debug("Available models in ts_data: %s...", available_models[:20])
    logger.debug("Looking for models: %s", models)
    
    # If ts_data is empty, try to get models from model_data
    if not available_models and model_data:
        available_models = sorted({str(m.get('modelName', '')) for m in model_data if m and m.get('modelName')})
        logger.debug("Using model_data instead. Available models: %s...", available_models[:20])
    
    for model_name in models:
        # Try exact match first
        if model_name in available_models:
            resolved_models.append(model_name)
            logger.debug("Using exact model: '%s'", model_name)
            continue
        
        # Try case-insensitive match
        for avail in available_models:
            if avail.lower() == model_name.lower():
                resolved_models.append(avail)
                logger.debug("Matched model '%s' to '%s'", model_name, avail)
                break
        else:
            # Try partial match
            for avail in available_models:
                if model_name.lower() in avail.lower() or avail.lower() in model_name.lower():
                    resolved_models.append(avail)
                    logger.debug("Partial matched model '%s' to '%s'", model_name, avail)
                    break
    
    if len(resolved_models) < 2:
        return f"Could not identify enough models to compare. Found: {resolved_models}. Available models include: {', '.join(available_models[:10])}..."
    
    # Extract region from query if not provided
    if region is None and metadata:
        region = metadata._find_best_region_match(question)
    
    logger.debug("Model comparison - variable: %s, models: %s, region: %s", resolved_variable, resolved_models, region)
    
    # Collect data for each model
    all_data = {}
    units = {}
    
    for model_name in resolved_models:
        filtered_data = []
        for r in ts_data:
            if r is None:
                continue
            if str(r.get('variable', '')) != resolved_variable:
                continue
            # Use modelName field (the actual field name in ts_data)
            r_model = str(r.get('modelName', '') or r.get('model', ''))
            if r_model != model_name:
                continue
            if scenario and r.get('scenario') != scenario:
                continue
            if region:
                r_region = str(r.get('region', ''))
                if r_region.lower() != region.lower():
                    continue
            filtered_data.append(r)
        
        if filtered_data:
            all_data[model_name] = filtered_data
            units[model_name] = filtered_data[0].get('unit', '')
    
    if not all_data:
        return f"No data found for '{resolved_variable}' across models {resolved_models} in region '{region}'."
    
    # Create comparison plot
    plt.figure(figsize=(12, 7))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'D', 'v', '<']
    
    scenarios_in_data = set()
    for idx, (model_name, data) in enumerate(all_data.items()):
        df = pd.DataFrame(data)
        
        # Handle years column
        if 'years' in df.columns:
            years_df = df['years'].apply(pd.Series)
            df = df.drop('years', axis=1).join(years_df)
        
        # Get year columns
        year_cols = [col for col in df.columns if str(col).isdigit()]
        if not year_cols:
            continue
        
        # Filter to specific year range if requested
        if start_year or end_year:
            filtered_year_cols = []
            for col in year_cols:
                year_int = int(col)
                if start_year and year_int < start_year:
                    continue
                if end_year and year_int > end_year:
                    continue
                filtered_year_cols.append(col)
            if filtered_year_cols:
                year_cols = filtered_year_cols
        
        # Aggregate if multiple rows (take mean by scenario)
        if len(df) > 1:
            # Group by scenario and take first of each
            df = df.groupby('scenario').first().reset_index()
        
        # Plot each row
        for _, row in df.iterrows():
            label = model_name
            if len(df) > 1:
                label = f"{model_name} ({row.get('scenario', '')})"
            if 'scenario' in row:
                scenarios_in_data.add(str(row.get('scenario', '')).strip())
            
            values = [row.get(str(year), 0) for year in sorted(year_cols, key=int)]
            plt.plot(sorted(year_cols, key=int), values, 
                    label=label, 
                    color=colors[idx % len(colors)],
                    marker=markers[idx % len(markers)],
                    linewidth=2)
    
    # Build title
    title = f"Model Comparison: {_pretty_variable_name(resolved_variable)}"
    if region:
        title += f" for {format_region_label(region)}"
    if start_year or end_year:
        title += f" ({start_year or '?'}-{end_year or '?'})"
    
    plt.title(title, fontsize=12, fontweight='bold')
    plt.xlabel("Year", fontsize=10)
    
    # Use first unit as Y-axis label
    if units:
        first_unit = list(units.values())[0]
        plt.ylabel(f"{_pretty_variable_name(resolved_variable)} ({first_unit})", fontsize=10)
    
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save to base64
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    plot_str = f"![Plot](data:image/png;base64,{img_base64})"
    return _wrap_plot_markdown(plot_str, f"model comparison of {_pretty_variable_name(resolved_variable)}", region, scenario, list(scenarios_in_data), start_year, end_year, prefix="Showing")


@_serialized_plot
def simple_plot_query_with_entities(question: str, model_data: List[Dict], ts_data: List[Dict],
                                    entities: Dict[str, Any], region: str = None) -> str:
    """
    Generate a plot using pre-extracted entities for better accuracy.
    
    Args:
        question: Original user query
        model_data: List of model metadata
        ts_data: List of time series data
        entities: Pre-extracted entities from QueryEntityExtractor
        region: Optional region override
    
    Returns:
        Base64 encoded PNG image or error message
    """
    # Check for multi-variable comparison from LLM-extracted entities
    variables_list = entities.get('variables')
    models_list = entities.get('models')
    comparison_type = entities.get('comparison')
    
    # Check for model comparison
    if models_list and len(models_list) >= 2:
        logger.debug("LLM detected multi-model comparison: %s", models_list)
        variable = entities.get('variable')
        scenario = entities.get('scenario')
        region_from_entities = entities.get('region')
        start_year = entities.get('start_year')
        end_year = entities.get('end_year')
        return plot_model_comparison(question, model_data, ts_data, variable, models_list,
                                    region or region_from_entities, scenario,
                                    start_year, end_year)
    
    metadata = get_metadata(ts_data, model_data)
    region_compare = detect_region_comparison(question, metadata)
    available_vars = {str(r.get('variable', '')).strip() for r in ts_data if r and r.get('variable')}
    ranked_vars = []
    significant_words = []
    if variables_list and len(variables_list) >= 2:
        region_keywords = {
            "usa", "us", "united", "states", "eu", "europe", "china", "chn", "india", "ind",
            "asia", "africa", "world", "global", "oecd", "latin", "america", "european"
        }
        if any(v.lower() in region_keywords for v in variables_list):
            variables_list = []
        logger.debug("LLM detected multi-variable comparison: %s", variables_list)
        scenario = entities.get('scenario')
        region_from_entities = entities.get('region')
        start_year = entities.get('start_year')
        end_year = entities.get('end_year')
        return plot_multiple_variables(question, model_data, ts_data, variables_list, 
                                       region or region_from_entities, scenario,
                                       start_year, end_year)
    
    # Fallback: Check for multi-variable comparison using regex patterns
    comparison_vars = detect_multi_variable_comparison(question)
    if region_compare and comparison_vars:
        region_keywords = {
            "usa", "us", "united", "states", "eu", "europe", "china", "chn", "india", "ind",
            "asia", "africa", "world", "global", "oecd", "latin", "america", "european"
        }
        if any(v.lower() in region_keywords for v in comparison_vars):
            comparison_vars = []
    if len(comparison_vars) >= 2:
        logger.debug("Regex detected multi-variable comparison: %s", comparison_vars)
        scenario = entities.get('scenario')
        start_year = entities.get('start_year')
        end_year = entities.get('end_year')
        return plot_multiple_variables(question, model_data, ts_data, comparison_vars, region, scenario,
                                       start_year, end_year)
    
    # Use extracted entities directly
    variable = entities.get('variable')
    if isinstance(variable, str):
        variable = variable.strip()
    scenario = entities.get('scenario')
    comparison = entities.get('comparison')
    scenarios_list = entities.get('scenarios')
    if isinstance(scenarios_list, list):
        scenarios_list = [str(item).strip() for item in scenarios_list if str(item or "").strip()]
    else:
        scenarios_list = []
    if scenarios_list:
        scenario = None
        comparison = "scenario"
    model = entities.get('model')
    start_year = entities.get('start_year')
    end_year = entities.get('end_year')
    unit = entities.get('unit')

    if variable:
        v = str(variable).lower()
        ql = question.lower()
        if "emission" in ql and "emission" not in v:
            variable = None
        elif "co2" in ql and "co2" not in v:
            variable = None
        if variable and "solar" in ql and "solar" not in v:
            variable = None
        if variable and "wind" in ql and "wind" not in v:
            variable = None
        if variable and "capacity" in ql and "capacity" not in v:
            variable = None
        if variable and _is_capacity_additions_mismatch(question, variable):
            variable = None
    
    # Use region from entities if not overridden
    if region is None:
        region = entities.get('region')

    if not variable:
        preferred_family = _preferred_plot_family_matches(question, available_vars)
        if preferred_family:
            variable = preferred_family[0]
    
    # If no variable extracted, fall back to keyword extraction
    if not variable:
        from pathlib import Path
        variable_path = Path('definitions/variable').resolve()
        variable_dict = load_all_yaml_files(str(variable_path))
        
        ranked_vars = resolve_natural_language_variable_ranked(question, variable_dict, top_k=5)
        natural_variable, var_score, _, significant_words = resolve_natural_language_variable_with_score(question, variable_dict)
        if natural_variable and isinstance(natural_variable, str):
            natural_variable = natural_variable.strip()
            if natural_variable in available_vars:
                var_lower = natural_variable.lower()
                if any(t in significant_words for t in ["emission", "emissions"]) and "emission" not in var_lower:
                    natural_variable = None
                elif "co2" in significant_words and "co2" not in var_lower:
                    natural_variable = None
                if "capacity" in significant_words and "capacity" not in var_lower:
                    natural_variable = None
                if "solar" in significant_words and "solar" not in var_lower:
                    natural_variable = None
                if "wind" in significant_words and "wind" not in var_lower:
                    natural_variable = None
                if natural_variable and _is_capacity_additions_mismatch(question, natural_variable):
                    natural_variable = None
                explicit_variable = "|" in question
                min_conf = 6
                if any(w in significant_words for w in ["capacity", "investment", "investments", "invest"]):
                    min_conf = 4
                if var_score is not None and not explicit_variable:
                    top1 = ranked_vars[0][1] if ranked_vars else None
                    top2 = ranked_vars[1][1] if ranked_vars and len(ranked_vars) > 1 else None
                    ambiguous = top1 is not None and top2 is not None and (top1 - top2) < 3
                    if var_score < min_conf or ambiguous:
                        natural_variable = None
                if natural_variable:
                    variable = natural_variable

        if not variable:
            candidates = []
            preferred_family = _preferred_plot_family_matches(question, available_vars)
            if preferred_family:
                candidates = preferred_family[:3]
            if ranked_vars:
                ranked_candidates = [name for name, _, _, _ in ranked_vars if name in available_vars][:3]
                for candidate in ranked_candidates:
                    if candidate not in candidates:
                        candidates.append(candidate)
                candidates = candidates[:3]
            if not candidates:
                candidates = resolve_natural_language_variable_candidates(question, variable_dict, top_k=3)
            if candidates:
                sample = ", ".join(candidates)
                return (
                    "Which variable should I use?\n"
                    f"Recommended variables: {sample}\n"
                    "Reply with the variable you want."
                )
    
    if not variable:
        # Final attempt: use metadata to suggest similar variables
        metadata = get_metadata(ts_data, model_data)
        if metadata:
            similar = metadata._suggest_similar_variables(question)
            if similar:
                return f"Could not identify a variable to plot. Did you mean: {', '.join(similar[:3])}?"
        return "Could not identify a variable to plot. Please specify a variable like 'solar capacity' or 'CO2 emissions'."

    metadata = get_metadata(ts_data, model_data)
    region_compare = detect_region_comparison(question, metadata)
    if region_compare and variable:
        return plot_variable_across_regions(question, model_data, ts_data, variable, region_compare, scenario, start_year, end_year)
    
    # Guard: if variable doesn't exist in loaded data, ask for a valid one
    available_vars = {str(r.get('variable', '')).strip() for r in ts_data if r and r.get('variable')}
    if variable and variable not in available_vars:
        candidates = []
        if ranked_vars:
            key_terms = {"methane", "ch4", "demand", "electricity", "emission", "emissions", "co2", "capacity",
                         "solar", "wind", "oil", "gas", "transport", "industry", "buildings", "final", "primary"}
            query_terms = {w for w in significant_words if w in key_terms}
            ranked_names = [name for name, _, _, _ in ranked_vars if name in available_vars]
            if query_terms:
                filtered = [n for n in ranked_names if any(t in n.lower() for t in query_terms)]
                candidates = filtered[:3]
            else:
                candidates = ranked_names[:3]
        if not candidates:
            try:
                variable_dict
            except NameError:
                from pathlib import Path
                variable_path = Path('definitions/variable').resolve()
                variable_dict = load_all_yaml_files(str(variable_path))
            candidates = resolve_natural_language_variable_candidates(question, variable_dict, top_k=3)
        if candidates:
            sample = ", ".join(candidates)
            return (
                f"Variable '{variable}' not found in loaded data.\n"
                "Which variable should I use?\n"
                f"Recommended variables: {sample}\n"
                "Reply with the variable you want."
            )
        return f"Variable '{variable}' not found in loaded data. Try `list variables`."

    # Filter data using extracted entities
    filtered_data = []
    for r in ts_data:
        if r is None:
            continue
        if str(r.get('variable', '')) != variable:
            continue
        if model and r.get('model') != model:
            continue
        row_scenario = str(r.get('scenario', '') or '')
        if scenarios_list:
            if row_scenario not in scenarios_list:
                continue
        elif scenario and row_scenario != scenario:
            continue
        # Case-insensitive region matching
        if region:
            r_region = str(r.get('region', ''))
            if r_region.lower() != region.lower():
                continue
        filtered_data.append(r)
    
    if not filtered_data:
        available_regions = sorted(set(str(r.get('region', '')) for r in ts_data if r and r.get('region') and r.get('variable') == variable))
        available_scenarios = sorted(set(str(r.get('scenario', '')) for r in ts_data if r and r.get('scenario') and r.get('variable') == variable))

        suggestions = []
        if available_regions:
            suggestions.append(f"Recommended regions: {', '.join(format_region_label(r) for r in available_regions[:3])}")
        if available_scenarios:
            suggestions.append(f"Recommended scenarios: {', '.join(available_scenarios[:3])}")

        suggestion_text = "\n".join(suggestions) if suggestions else "Try `list variables` to see available options."
        return f"No data found for variable '{variable}'.\n{suggestion_text}"
    
    # Get unit from data if not in entities
    if not unit:
        unit = filtered_data[0].get('unit', '')
    
    # Prepare data for plotting
    df = pd.DataFrame(filtered_data)
    if 'years' in df.columns:
        years_df = df['years'].apply(pd.Series)
        df = df.drop('years', axis=1).join(years_df)
    
    # Get year columns
    year_cols = [col for col in df.columns if str(col).isdigit()]
    if not year_cols:
        return "No time series data available for plotting."
    
    # Filter to specific year range if requested
    if start_year or end_year:
        filtered_year_cols = []
        for col in year_cols:
            year_int = int(col)
            if start_year and year_int < start_year:
                continue
            if end_year and year_int > end_year:
                continue
            filtered_year_cols.append(col)
        if filtered_year_cols:
            year_cols = filtered_year_cols
            logger.debug(
                "Filtered years to range %s-%s: %s...%s",
                start_year,
                end_year,
                year_cols[:5],
                year_cols[-5:] if len(year_cols) > 5 else "",
            )
        else:
            logger.debug("No years in range %s-%s, using all years", start_year, end_year)
    
    # Create plot
    plt.figure(figsize=(12, 7))
    
    # Determine how to group data based on comparison type or data variety
    scenarios_in_data = df['scenario'].unique()
    regions_in_data = df['region'].unique() if 'region' in df.columns else ['All']
    models_in_data = df['model'].unique() if 'model' in df.columns else ['All']
    
    # Choose grouping strategy
    if comparison == 'scenario' or (len(scenarios_in_data) > 1 and len(regions_in_data) == 1):
        # Group by scenario
        for scenario_name in scenarios_in_data:
            scenario_data = df[df['scenario'] == scenario_name]
            if not scenario_data.empty:
                row = scenario_data.iloc[0]
                label = f"{scenario_name}"
                values = [row.get(str(year), 0) for year in sorted(year_cols, key=int)]
                plt.plot(sorted(year_cols, key=int), values, label=label, marker='o', linewidth=2)
    
    elif comparison == 'region' or (len(regions_in_data) > 1 and len(scenarios_in_data) == 1):
        # Group by region
        for region_name in regions_in_data:
            region_data = df[df['region'] == region_name]
            if not region_data.empty:
                row = region_data.iloc[0]
                label = f"{region_name}"
                values = [row.get(str(year), 0) for year in sorted(year_cols, key=int)]
                plt.plot(sorted(year_cols, key=int), values, label=label, marker='o', linewidth=2)
    
    elif comparison == 'model' or (len(models_in_data) > 1):
        # Group by model
        for model_name in models_in_data:
            model_data_row = df[df['model'] == model_name]
            if not model_data_row.empty:
                row = model_data_row.iloc[0]
                label = f"{model_name}"
                values = [row.get(str(year), 0) for year in sorted(year_cols, key=int)]
                plt.plot(sorted(year_cols, key=int), values, label=label, marker='o', linewidth=2)
    
    else:
        # Default: plot each row
        for idx, row in df.iterrows():
            label_parts = []
            if 'model' in row:
                label_parts.append(str(row['model']))
            if 'scenario' in row:
                label_parts.append(str(row['scenario']))
            if 'region' in row:
                label_parts.append(str(row['region']))
            label = " - ".join(label_parts) if label_parts else f"Series {idx+1}"
            
            values = [row.get(str(year), 0) for year in sorted(year_cols, key=int)]
            plt.plot(sorted(year_cols, key=int), values, label=label, marker='o')
    
    # Build title with context
    title_parts = [_pretty_variable_name(variable)]
    if scenario and len(scenarios_in_data) == 1:
        title_parts.append(f"({scenario})")
    if region and len(regions_in_data) == 1:
        title_parts.append(f"- {format_region_label(region)}")
    if start_year or end_year:
        year_range = f"({start_year or '?'}-{end_year or '?'})"
        title_parts.append(year_range)
    
    plt.title(" ".join(title_parts), fontsize=12, fontweight='bold')
    plt.xlabel("Year", fontsize=10)
    
    # Use unit in Y-axis label
    ylabel = _pretty_variable_name(variable)
    if unit:
        ylabel = f"{_pretty_variable_name(variable)} ({unit})"
    plt.ylabel(ylabel, fontsize=10)
    
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save to base64
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    plot_str = f"![Plot](data:image/png;base64,{img_base64})"
    return _wrap_plot_markdown(plot_str, variable, region, scenario, list(scenarios_in_data), start_year, end_year)


@_serialized_plot
def simple_plot_query(question: str, model_data: List[Dict], ts_data: List[Dict], region: str = None) -> str:
    """
    Generate a plot based on natural language query.
    
    Args:
        question: User's natural language query
        model_data: List of model metadata
        ts_data: List of time series data
        region: Optional region filter
    
    
    Returns:
        Base64 encoded PNG image or error message
    """
    from pathlib import Path
    from utils_query import extract_region_from_query, find_closest_variable_name, resolve_natural_language_variable_universal

    def _extract_year_range(text: str) -> tuple[Optional[int], Optional[int]]:
        return extract_year_range(text)
    
    metadata = get_metadata(ts_data, model_data)
    region_compare = detect_region_comparison(question, metadata)

    # Check for multi-variable comparison first
    comparison_vars = detect_multi_variable_comparison(question)
    if region_compare and comparison_vars:
        region_keywords = {
            "usa", "us", "united", "states", "eu", "europe", "china", "chn", "india", "ind",
            "asia", "africa", "world", "global", "oecd", "latin", "america", "european"
        }
        if any(v.lower() in region_keywords for v in comparison_vars):
            comparison_vars = []
    if len(comparison_vars) >= 2:
        logger.debug("Detected multi-variable comparison: %s", comparison_vars)
        start_year, end_year = _extract_year_range(question.lower())
        return plot_multiple_variables(question, model_data, ts_data, comparison_vars, region, None, start_year, end_year)
    
    # Extract region from query if not provided
    if region is None:
        region_path = Path('definitions/region').resolve()
        region_dict = load_all_yaml_files(str(region_path))
        region_candidates = sorted({str(r.get('region', '')).strip() for r in ts_data if r and r.get('region')})
        region = extract_region_from_query(question, region_dict, region_candidates)
        if re.search(r"\b(world|global)\b", question.lower()):
            region = "World"
    
    # Load variable definitions
    variable_path = Path('definitions/variable').resolve()
    variable_dict = load_all_yaml_files(str(variable_path))
    available_vars = {str(r.get('variable', '')).strip() for r in ts_data if r and r.get('variable')}
    
    # Try to match variable from query
    variable = None
    
    # First try natural language resolution
    ranked_vars = resolve_natural_language_variable_ranked(question, variable_dict, top_k=5)
    natural_variable, var_score, _, significant_words = resolve_natural_language_variable_with_score(question, variable_dict)
    if natural_variable and isinstance(natural_variable, str):
        natural_variable = natural_variable.strip()
        if natural_variable in available_vars:
            var_lower = natural_variable.lower()
            if any(t in significant_words for t in ["emission", "emissions"]) and "emission" not in var_lower:
                natural_variable = None
            elif "co2" in significant_words and "co2" not in var_lower:
                natural_variable = None
            if "capacity" in significant_words and "capacity" not in var_lower:
                natural_variable = None
            if "solar" in significant_words and "solar" not in var_lower:
                natural_variable = None
            if "wind" in significant_words and "wind" not in var_lower:
                natural_variable = None
            explicit_variable = "|" in question
            min_conf = 6
            if any(w in significant_words for w in ["capacity", "investment", "investments", "invest"]):
                min_conf = 4
            if var_score is not None and not explicit_variable:
                top1 = ranked_vars[0][1] if ranked_vars else None
                top2 = ranked_vars[1][1] if ranked_vars and len(ranked_vars) > 1 else None
                ambiguous = top1 is not None and top2 is not None and (top1 - top2) < 3
                if var_score < min_conf or ambiguous:
                    natural_variable = None
            if natural_variable:
                variable = natural_variable
    
    # Fall back to keyword matching
    if not variable:
        preferred_family = _preferred_plot_family_matches(question, available_vars)
        if preferred_family:
            variable = preferred_family[0]

    if not variable:
        variable = match_variable_from_yaml(question, variable_dict)
        if isinstance(variable, dict):
            variable = variable.get('matched_variable') or ""

    # If matched variable isn't in available data, reset and continue
    if variable:
        if variable not in available_vars:
            variable = ""
    
    # If still no variable and the user gave an explicit variable format, try a closest match
    if not variable and "|" in question:
        available_vars = sorted(set(str(r.get('variable', '')) for r in ts_data if r and r.get('variable')))
        variable = find_closest_variable_name(question, available_vars)
    
    # If region comparison detected and variable resolved, plot across regions
    if region_compare and variable:
        start_year, end_year = _extract_year_range(question.lower())
        return plot_variable_across_regions(question, model_data, ts_data, variable, region_compare, None, start_year, end_year)

    # Try metadata-based variable matching if still no match
    if not variable:
        candidates = []
        preferred_family = _preferred_plot_family_matches(question, available_vars)
        if preferred_family:
            candidates = preferred_family[:3]
        if ranked_vars:
            ranked_candidates = [name for name, _, _, _ in ranked_vars if name in available_vars][:3]
            for candidate in ranked_candidates:
                if candidate not in candidates:
                    candidates.append(candidate)
            candidates = candidates[:3]
        if not candidates:
            candidates = resolve_natural_language_variable_candidates(question, variable_dict, top_k=3)
        if candidates:
            sample = ", ".join(candidates)
            return (
                "Which variable should I use?\n"
                f"Recommended variables: {sample}\n"
                "Reply with the variable you want."
            )
    
    if not variable:
        # Final attempt: use metadata to suggest similar variables
        metadata = get_metadata(ts_data, model_data)
        if metadata:
            similar = metadata._suggest_similar_variables(question)
            if similar:
                return f"Could not identify a variable to plot. Did you mean: {', '.join(similar[:3])}?"
        return "Could not identify a variable to plot. Please specify a variable like 'solar capacity' or 'CO2 emissions'."
    
    # Extract model from query if mentioned
    model_match = None
    model_names = sorted({m.get('modelName', '') for m in model_data if m and m.get('modelName')})
    if model_names:
        model_match = match_model_name(question, model_names)

    # Extract scenario from query if mentioned
    scenario = None
    question_lower = question.lower()
    scenarios = sorted({str(r.get('scenario', '')).strip() for r in ts_data if r and r.get('scenario')})
    m = re.search(r"(?:under|scenario)\s+([\w\-\.]+)", question_lower)
    if m:
        token = m.group(1)
        for s in scenarios:
            if token.lower() in s.lower():
                scenario = s
                break
    if not scenario:
        for token in re.findall(r"(ssp\d|rcp\d(?:\.\d)?)", question_lower):
            for s in scenarios:
                if token.lower() in s.lower():
                    scenario = s
                    break
            if scenario:
                break

    # Extract year range if mentioned
    start_year, end_year = _extract_year_range(question_lower)

    if (
        metadata
        and variable
        and region
        and scenario
        and not metadata.combination_exists(
            variable,
            region=region,
            scenario=scenario,
            model=model_match or None,
        )
    ):
        matrix_prompt = _matrix_plot_recovery_prompt(
            metadata,
            f"No data found for **{variable}** in region `{region}` under scenario `{scenario}`.",
            variable=variable,
            region=region,
            scenario=scenario,
            model=model_match or None,
        )
        if matrix_prompt:
            return matrix_prompt
    
    # Filter data
    filtered_data = []
    for r in ts_data:
        if r is None:
            continue
        if str(r.get('variable', '')) != variable:
            continue
        if model_match and r.get('modelName') != model_match:
            continue
        if scenario and r.get('scenario') != scenario:
            continue
        if region and r.get('region') != region:
            continue
        filtered_data.append(r)
    
    if not filtered_data:
        from collections import Counter
        from difflib import get_close_matches

        matrix_prompt = _matrix_plot_recovery_prompt(
            metadata,
            f"No data found for **{variable}** in region `{region}` under scenario `{scenario}`.",
            variable=variable,
            region=region,
            scenario=scenario,
            model=model_match or None,
        )
        if matrix_prompt:
            return matrix_prompt

        scoped_regions = sorted({
            str(r.get('region', '')) for r in ts_data
            if r and r.get('region') and r.get('variable') == variable
            and (not model_match or r.get('modelName') == model_match)
            and (not scenario or r.get('scenario') == scenario)
        })
        scoped_scenarios = sorted({
            str(r.get('scenario', '')) for r in ts_data
            if r and r.get('scenario') and r.get('variable') == variable
            and (not model_match or r.get('modelName') == model_match)
            and (not region or r.get('region') == region)
        })

        def _top_values(key: str, limit: int = 3, filter_region: bool = False) -> list:
            records = [
                r for r in ts_data
                if r and r.get('variable') == variable
                and (not model_match or r.get('modelName') == model_match)
                and (not scenario or r.get('scenario') == scenario)
                and (not filter_region or (region and r.get('region') == region))
            ]
            counts = Counter([str(r.get(key, '')).strip() for r in records if r and r.get(key)])
            return [k for k, _ in counts.most_common(limit)]

        if model_match and not scoped_regions and not scoped_scenarios:
            all_regions = _top_values("region", limit=3)
            all_scenarios = _top_values("scenario", limit=3)
            region_suggestion = ", ".join(all_regions) if all_regions else "none"
            scenario_suggestion = ", ".join(all_scenarios) if all_scenarios else "none"
            return (
                f"No data found for **{variable}** in model `{model_match}`.\n\n"
                f"Across all models, recommended regions: {region_suggestion}\n"
                f"Across all models, recommended scenarios: {scenario_suggestion}\n\n"
                "Tell me which region or scenario you want."
            )

        if region and region not in scoped_regions:
            close_regions = get_close_matches(region, scoped_regions, n=3, cutoff=0.6)
            region_candidates = close_regions or _top_values("region", limit=3)
            scenario_candidates = _top_values("scenario", limit=3)
            region_suggestion = ", ".join(format_region_label(r) for r in region_candidates) if region_candidates else "none"
            scenario_suggestion = ", ".join(scenario_candidates) if scenario_candidates else "none"
            return (
                f"No data found for **{variable}** in region `{region}`.\n\n"
                f"Recommended regions: {region_suggestion}\n"
                f"Recommended scenarios: {scenario_suggestion}\n\n"
                "Tell me which region or scenario you want."
            )

        region_suggestion = ", ".join(format_region_label(r) for r in _top_values("region", limit=3)) if scoped_regions else "none"
        scenario_suggestion = ", ".join(_top_values("scenario", limit=3, filter_region=bool(region))) if scoped_scenarios else "none"
        model_note = f" for model `{model_match}`" if model_match else ""
        return (
            f"No data found for **{variable}**{model_note}.\n\n"
            f"Recommended regions: {region_suggestion}\n"
            f"Recommended scenarios: {scenario_suggestion}\n\n"
            "Tell me which region or scenario you want."
        )
    
    # Get unit from data
    unit = filtered_data[0].get('unit', '')
    
    # Prepare data for plotting
    df = pd.DataFrame(filtered_data)
    if 'years' in df.columns:
        years_df = df['years'].apply(pd.Series)
        df = df.drop('years', axis=1).join(years_df)
    
    # Get year columns
    year_cols = [col for col in df.columns if str(col).isdigit()]
    if start_year or end_year:
        filtered_years = select_years([str(year) for year in year_cols], start_year, end_year)
        if filtered_years:
            year_cols = filtered_years
    if not year_cols:
        return "No time series data available for plotting."
    
    # Create plot
    plt.figure(figsize=(12, 7))
    
    # Determine grouping strategy
    scenarios_in_data = df['scenario'].unique() if 'scenario' in df.columns else ['All']
    regions_in_data = df['region'].unique() if 'region' in df.columns else ['All']
    models_in_data = df['model'].unique() if 'model' in df.columns else ['All']
    
    # Choose grouping based on data variety
    if len(scenarios_in_data) > 1 and len(regions_in_data) == 1:
        # Group by scenario
        for scenario_name in scenarios_in_data:
            scenario_data = df[df['scenario'] == scenario_name]
            if not scenario_data.empty:
                row = scenario_data.iloc[0]
                label = f"{scenario_name}"
                values = [row.get(str(year), 0) for year in sorted(year_cols, key=int)]
                plt.plot(sorted(year_cols, key=int), values, label=label, marker='o', linewidth=2)
    
    elif len(regions_in_data) > 1 and len(scenarios_in_data) == 1:
        # Group by region
        for region_name in regions_in_data:
            region_data = df[df['region'] == region_name]
            if not region_data.empty:
                row = region_data.iloc[0]
                label = f"{region_name}"
                values = [row.get(str(year), 0) for year in sorted(year_cols, key=int)]
                plt.plot(sorted(year_cols, key=int), values, label=label, marker='o', linewidth=2)
    
    elif len(models_in_data) > 1:
        # Group by model
        for model_name in models_in_data:
            model_data_row = df[df['model'] == model_name]
            if not model_data_row.empty:
                row = model_data_row.iloc[0]
                label = f"{model_name}"
                values = [row.get(str(year), 0) for year in sorted(year_cols, key=int)]
                plt.plot(sorted(year_cols, key=int), values, label=label, marker='o', linewidth=2)
    
    else:
        # Default: plot each row
        for idx, row in df.iterrows():
            label_parts = []
            if 'model' in row:
                label_parts.append(str(row['model']))
            if 'scenario' in row:
                label_parts.append(str(row['scenario']))
            if 'region' in row:
                label_parts.append(str(row['region']))
            label = " - ".join(label_parts) if label_parts else f"Series {idx+1}"
            
            values = [row.get(str(year), 0) for year in sorted(year_cols, key=int)]
            plt.plot(sorted(year_cols, key=int), values, label=label, marker='o')
    
    # Build title
    title_parts = [_pretty_variable_name(variable)]
    if scenario and len(scenarios_in_data) == 1:
        title_parts.append(f"({scenario})")
    if region and len(regions_in_data) == 1:
        title_parts.append(f"- {format_region_label(region)}")
    
    plt.title(" ".join(title_parts), fontsize=12, fontweight='bold')
    plt.xlabel("Year", fontsize=10)
    
    # Use unit in Y-axis label
    ylabel = _pretty_variable_name(variable)
    if unit:
        ylabel = f"{_pretty_variable_name(variable)} ({unit})"
    plt.ylabel(ylabel, fontsize=10)
    
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save to base64
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    plot_str = f"![Plot](data:image/png;base64,{img_base64})"
    return _wrap_plot_markdown(plot_str, variable, region, scenario, list(scenarios_in_data), start_year, end_year)
