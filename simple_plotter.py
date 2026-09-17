import re
from llm_config import QA_MODEL
from resolved_scope import record_resolved_scope
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
from model_aliases import (
    display_model_label,
    is_presentable_model_label,
    match_model_name,
)
from canonical_aliases import (
    dedupe_equivalent_regions,
    explicit_scenarios_from_query,
    preferred_variable_from_query,
    rank_catalogue_variable_matches,
    regions_equivalent,
    scenario_family_members,
)
from year_filters import extract_year_range, is_latest_year_filter


def _explicit_variable_in_query(question: str, available_vars) -> str | None:
    """Return a structured (pipe-delimited) variable name the user typed
    verbatim, so the plot path can prefer it over an extractor result that
    drifted to a sibling/superstring. Longest verbatim match wins."""
    q = (question or "").lower()
    matches = [v for v in available_vars if v and "|" in v and v.lower() in q]
    if not matches:
        return None
    return max(matches, key=len)


def _canonical_ts_model(model: str | None, ts_data) -> str | None:
    """Map an extractor/display model name to the name used on the timeseries
    records (e.g. "GCAM" -> "gcam") so a model filter is not silently emptied."""
    if not model:
        return model
    ts_model_names = sorted({
        str(r.get("modelName", "")).strip()
        for r in ts_data
        if r and is_presentable_model_label(r.get("modelName"))
    })
    if model in ts_model_names:
        return model
    canonical = match_model_name(model, ts_model_names)
    return canonical or model


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

    # A comparison caption arrives already joined ("A vs B") and gets prettified
    # a second time on its way into the answer. The substring rules below match
    # anywhere in the string, so "Oil Primary Energy vs Gas" collapsed to "Oil
    # Primary Energy" and the second variable vanished from the caption even
    # though it was plotted. Prettify each side instead.
    if " vs " in value:
        return " vs ".join(
            _pretty_variable_name(part) for part in value.split(" vs ") if part.strip()
        )

    lower = value.lower()

    if lower == "price|carbon":
        return "Carbon price"
    if lower == "gdp|mer":
        return "GDP (MER)"
    if lower == "secondary energy|electricity|nuclear":
        return "Nuclear electricity"
    if lower == "capacity|electricity|nuclear":
        return "Nuclear capacity"
    if lower.startswith("capacity|electricity|solar|pv"):
        return "Solar PV Capacity"
    if lower.startswith("secondary energy|electricity|solar|pv"):
        return "Solar PV Electricity"
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
    if is_latest_year_filter(start_year, end_year):
        return " (latest available year)"
    if start_year is not None and end_year is not None:
        if int(start_year) == int(end_year):
            return f" ({start_year})"
        return f" ({start_year}–{end_year})"
    if start_year is not None:
        return f" (from {start_year})"
    if end_year is not None:
        return f" (through {end_year})"
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

    if "primary energy" in ql and re.search(r"\bcoal\b", ql):
        candidates.extend(
            v for v in available_vars
            if v.lower() == "primary energy|coal"
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

    generation_requested = any(
        token in ql for token in ("electricity generation", "power generation", "generation from")
    )
    pv_requested = bool(re.search(r"\b(?:solar\s+pv|photovoltaic|pv)\b", ql))
    coal_requested = bool(re.search(r"\bcoal\b", ql))

    def _constraint_score(variable: str):
        lower = variable.lower()
        penalty = 0
        if generation_requested and "capacity" in lower:
            penalty += 100
        if pv_requested and not ("solar" in lower and "pv" in lower):
            penalty += 50
        if coal_requested and "coal" not in lower:
            penalty += 50
        return (penalty, *_score(variable))

    return sorted(deduped, key=_constraint_score)


def _refine_explicit_technology_leaf(
    question: str,
    variable: str,
    available_vars: set[str],
) -> str:
    """Keep an explicitly requested technology leaf from being broadened.

    The entity extractor can return the valid aggregate
    ``Capacity|Electricity|Solar`` even when the user explicitly wrote
    ``solar PV``.  Trust the aggregate for generic solar requests, but when the
    exact immediate ``|PV`` descendant exists, retain the leaf the user named.
    """
    value = str(variable or "").strip()
    if not value or not re.search(
        r"\b(?:solar[\s\-_/]+pv|photovoltaic|pv)\b",
        str(question or ""),
        flags=re.IGNORECASE,
    ):
        return value
    if "solar" not in value.casefold() or re.search(
        r"(?:^|\|)pv(?:\||$)", value, flags=re.IGNORECASE
    ):
        return value
    exact_descendant = f"{value}|PV"
    return next(
        (
            candidate
            for candidate in available_vars
            if candidate.casefold() == exact_descendant.casefold()
        ),
        value,
    )


# Past roughly this many lines a time-series chart stops being readable: the
# legend grows until it covers the series it describes. Charts that would exceed
# it are trimmed to a representative subset and the caption says what was left
# out (see `_wrap_plot_markdown`).
MAX_PLOT_SERIES = 8

# Scenario families users ask about most. When a chart has to be trimmed these
# are kept first, so the subset still answers the usual question.
_SCENARIO_PRIORITY_HINTS = (
    "baseline", "curpol", "current polic", "ndc", "net zero", "netzero", "nze",
)


def _scenario_priority(value: object) -> int:
    """Rank a scenario name; lower sorts first when trimming a busy chart."""
    text = str(value or "").casefold()
    for index, hint in enumerate(_SCENARIO_PRIORITY_HINTS):
        if hint in text:
            return index
    return len(_SCENARIO_PRIORITY_HINTS)


def _representative_rows(
    df,
    max_series: int = MAX_PLOT_SERIES,
    balance_column: str | None = None,
):
    """Trim a plot frame to a readable, representative set of rows.

    Rows are taken round-robin across models so a trimmed chart still shows the
    spread between models rather than every scenario of whichever model happened
    to sort first. Within each model the priority scenarios come first.

    Returns ``(trimmed_df, omitted_count)``; the frame is returned untouched when
    it is already small enough.
    """
    total = len(df)
    if total <= max_series:
        return df, 0

    # Comparisons need to retain both sides. Alternating between the requested
    # variables/regions/models prevents a busy first member from consuming the
    # entire chart before the other comparison members are reached.
    if balance_column and balance_column in df.columns:
        model_column = next(
            (name for name in ("_plot_model", "model", "modelName") if name in df.columns),
            None,
        )
        scenario_column = "scenario" if "scenario" in df.columns else None
        queues: dict[str, list] = {}
        for index, row in df.iterrows():
            member = _clean_text(row.get(balance_column, ""))
            scenario_value = _clean_text(row.get(scenario_column, "")) if scenario_column else ""
            model_value = _clean_text(row.get(model_column, "")) if model_column else ""
            queues.setdefault(member, []).append(
                (_scenario_priority(scenario_value), scenario_value.casefold(), model_value.casefold(), index)
            )
        for entries in queues.values():
            entries.sort()

        keep: list = []
        members = sorted(queues, key=str.casefold)
        while len(keep) < max_series and any(queues[member] for member in members):
            for member in members:
                if queues[member]:
                    keep.append(queues[member].pop(0)[-1])
                if len(keep) >= max_series:
                    break
        return df.loc[keep], total - len(keep)

    model_column = next(
        (name for name in ("_plot_model", "model", "modelName") if name in df.columns),
        None,
    )
    scenario_column = "scenario" if "scenario" in df.columns else None

    # Scenario is the axis that carries the message in IAM data -- a chart of
    # eight baselines answers nothing. Group by scenario and take them in turn,
    # so the trimmed chart spans policy outcomes rather than one family of them.
    grouped: dict[tuple, list] = {}
    for index, row in df.iterrows():
        scenario_key = _clean_text(row.get(scenario_column, "")) if scenario_column else ""
        model_key = _clean_text(row.get(model_column, "")) if model_column else ""
        priority = _scenario_priority(scenario_key) if scenario_column else 0
        grouped.setdefault((priority, scenario_key), []).append((model_key, index))
    for entries in grouped.values():
        entries.sort(key=lambda entry: entry[0])

    keep: list = []
    seen_models: set[str] = set()
    order = sorted(grouped)
    while len(keep) < max_series and any(grouped[key] for key in order):
        for key in order:
            entries = grouped[key]
            if not entries:
                continue
            # Prefer a model not yet on the chart so the subset spans models too.
            position = next(
                (i for i, (model_key, _) in enumerate(entries) if model_key not in seen_models),
                0,
            )
            model_key, index = entries.pop(position)
            seen_models.add(model_key)
            keep.append(index)
            if len(keep) >= max_series:
                break
    return df.loc[keep], total - len(keep)


def _comparison_source_key(record: Any) -> tuple[str, str] | None:
    """Return a conservative model/scenario identity for comparison pairing."""
    model = _clean_text(_row_model(record))
    scenario = _clean_text(record.get("scenario")) if hasattr(record, "get") else ""
    if not model or not scenario:
        return None
    return (model.casefold(), scenario.casefold())


def _common_comparison_source_keys(
    member_records: Dict[str, List[Dict]],
) -> set[tuple[str, str]]:
    """Model/scenario keys represented for every requested comparison member."""
    if not member_records or any(not rows for rows in member_records.values()):
        return set()
    key_sets = [
        {
            key
            for record in records
            if (key := _comparison_source_key(record)) is not None
        }
        for records in member_records.values()
    ]
    if not key_sets or any(not keys for keys in key_sets):
        return set()
    return set.intersection(*key_sets)


def _paired_region_rows(
    frame: pd.DataFrame,
    regions: List[str],
    max_series: int = MAX_PLOT_SERIES,
) -> tuple[pd.DataFrame, int, set[tuple[str, str]]]:
    """Select complete, like-for-like source groups across compared regions.

    One record per region and common model/scenario key is retained. Complete
    groups are selected in scenario-priority order, so the series cap cannot
    end with an unrelated source on only one side of the comparison.
    """
    total = len(frame)
    if frame.empty or "_comparison_region" not in frame.columns or not regions:
        return frame, 0, set()

    rows_by_region: dict[str, dict[tuple[str, str], list]] = {
        str(region): {} for region in regions
    }
    for index, row in frame.iterrows():
        region = _clean_text(row.get("_comparison_region"))
        key = _comparison_source_key(row)
        if region not in rows_by_region or key is None:
            continue
        rows_by_region[region].setdefault(key, []).append(index)

    key_sets = [set(rows_by_region[region]) for region in rows_by_region]
    common_keys = set.intersection(*key_sets) if key_sets and all(key_sets) else set()
    if not common_keys:
        return frame, 0, set()

    ordered_keys = sorted(
        common_keys,
        key=lambda key: (_scenario_priority(key[1]), key[1], key[0]),
    )
    group_width = len(rows_by_region)
    group_limit = max(1, max_series // group_width)
    keep: list = []
    for key in ordered_keys[:group_limit]:
        for region in rows_by_region:
            keep.append(rows_by_region[region][key][0])
    return frame.loc[keep], total - len(keep), common_keys


def _finalize_plot_layout(series_count: int) -> None:
    """Draw the legend so it never covers the plotted data, then lay out.

    ``loc='best'`` has no good position once there are many entries: matplotlib
    puts the box on top of the series it labels and, past ~20 entries, it runs
    off the canvas. Beyond a handful of series the legend moves outside the axes
    and the axes shrink to make room -- `tight_layout` alone does not reserve
    space for artists placed outside the axes.
    """
    if series_count <= 6:
        plt.legend(loc="best", fontsize=9)
        plt.tight_layout()
        return

    # Lay the axes out normally first; `tight_layout(rect=...)` re-flows the
    # title into the reserved strip and clips it, so the space for the legend is
    # taken afterwards with `subplots_adjust`.
    axes = plt.gca()
    ticks = axes.get_xticks()
    if len(ticks) > 10:
        axes.set_xticks(ticks[:: max(1, len(ticks) // 8)])
    plt.setp(axes.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()

    # Series labels here are long ("Primary Energy|Coal (Unharmonised baseline)"),
    # so a second column gets clipped at the figure edge long before a single
    # column runs out of vertical room. Stay single-column as far as it fits.
    columns = 1 if series_count <= 22 else 2
    plt.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=7,
        ncol=columns,
        borderaxespad=0.0,
    )
    plt.subplots_adjust(right=0.68 if columns == 1 else 0.52)


def _wrap_plot_markdown(
    plot_str: str,
    variable: str,
    region: str | None = None,
    scenario: str | None = None,
    scenarios_in_data: list | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    prefix: str = "Showing",
    scope_variable: str | None = None,
    models_in_data: list | None = None,
    regions_in_data: list | None = None,
    all_scenarios: bool | None = None,
    omitted_series: int = 0,
    scope_variables: list | None = None,
    scope_models: list | None = None,
    comparison_dimension: str | None = None,
    displayed_series: list | None = None,
    chart_type: str = "line",
    unit: str | None = None,
    comparison_pairing: str | None = None,
) -> str:
    # Report the resolved scope structurally so the manager does not have to
    # re-parse the caption. `scope_variable` overrides caption-style variables
    # (e.g. "X vs Y") with the real variable name.
    concrete_scenarios = [str(value) for value in (scenarios_in_data or []) if value]
    resolved_all_scenarios = (
        bool(not scenario and len(concrete_scenarios) > 1)
        if all_scenarios is None
        else bool(all_scenarios and not scenario and len(concrete_scenarios) > 1)
    )
    resolved_variable = scope_variable if scope_variable is not None else variable
    concrete_variables = [
        str(value) for value in (scope_variables or []) if str(value or "").strip()
    ]
    if not concrete_variables and resolved_variable:
        concrete_variables = [str(resolved_variable)]
    concrete_models = [
        display_model_label(value)
        for value in (scope_models or [])
        if str(value or "").strip()
    ]
    result_models = sorted({
        display_model_label(value)
        for value in (models_in_data or [])
        if str(value or "").strip()
    })
    concrete_displayed_series = [
        str(value) for value in (displayed_series or []) if str(value or "").strip()
    ]
    concrete_regions = dedupe_equivalent_regions(
        str(value) for value in (regions_in_data or []) if str(value or "").strip()
    )
    if (
        region
        and concrete_regions
        and all(regions_equivalent(value, region) for value in concrete_regions)
    ):
        concrete_regions = [str(region)]
    resolved_comparison_dimension = comparison_dimension
    if comparison_dimension == "region" and len(concrete_regions) < 2:
        resolved_comparison_dimension = None
    record_resolved_scope(
        variable=resolved_variable,
        variables=concrete_variables,
        region=region,
        regions=concrete_regions,
        scenario=scenario,
        scenarios=concrete_scenarios,
        models=concrete_models,
        result_models=result_models,
        all_scenarios=resolved_all_scenarios,
        start_year=start_year,
        end_year=end_year,
        comparison=resolved_comparison_dimension,
        comparison_dimension=resolved_comparison_dimension,
        displayed_series=concrete_displayed_series,
        displayed_series_count=len(concrete_displayed_series),
        omitted_series=int(omitted_series or 0),
        chart_type=chart_type,
        unit=unit,
        comparison_pairing=comparison_pairing,
        action="plot",
    )
    compared_regions = concrete_regions
    if len(compared_regions) > 1:
        formatted_regions = [format_region_label(value) for value in compared_regions]
        if comparison_pairing == "unpaired source scopes":
            subject = (
                f"{_pretty_variable_name(variable)} using region-specific sources for "
                + " and ".join(formatted_regions)
            )
        else:
            subject = f"{_pretty_variable_name(variable)}: " + " vs ".join(formatted_regions)
    else:
        subject = _plot_subject(variable, region)
    years = _year_range_text(start_year, end_year)
    if scenario:
        caption = f"{prefix} {subject} for scenario `{scenario}`{years}."
    elif len(concrete_scenarios) > 1 and resolved_all_scenarios:
        caption = f"{prefix} {subject} across available scenarios{years}."
    elif len(concrete_scenarios) > 1:
        quoted_scenarios = [f"`{value}`" for value in concrete_scenarios]
        selected_scenarios = (
            f"{quoted_scenarios[0]} and {quoted_scenarios[1]}"
            if len(quoted_scenarios) == 2
            else f"{', '.join(quoted_scenarios[:-1])}, and {quoted_scenarios[-1]}"
        )
        caption = f"{prefix} {subject} for selected scenarios {selected_scenarios}{years}."
    elif len(concrete_scenarios) == 1:
        concrete_scenario = concrete_scenarios[0]
        caption = f"{prefix} {subject} for scenario `{concrete_scenario}`{years}."
    else:
        caption = f"{prefix} {subject}{years}."
    # Keep the friendly caption while also exposing the exact runtime taxonomy
    # value used for a single-variable plot. This makes alias/fuzzy resolution
    # auditable without coupling the presentation layer to any variable name.
    if scope_variable is None and "|" in str(variable or ""):
        caption += f" Resolved variable: `{variable}`."
    # A chart that carries every model/scenario pair becomes unreadable, so the
    # renderer trims it. Say so, otherwise the plot silently misrepresents how
    # much data exists.
    if omitted_series > 0:
        displayed_count = len(concrete_displayed_series) or MAX_PLOT_SERIES
        caption += (
            f" Showing {displayed_count} of {displayed_count + omitted_series} available series"
            " for readability; name a scenario or model to see a specific one."
        )
    return caption + "\n" + plot_str


def _model_names_from_records(records) -> list[str]:
    """Return selectable model labels for no-data recovery suggestions."""
    return sorted({
        str(record.get("modelName") or record.get("model") or "").strip()
        for record in (records or [])
        if isinstance(record, dict)
        and is_presentable_model_label(
            record.get("modelName") or record.get("model")
        )
    })


def _clean_text(value: Any) -> str:
    """Convert a scalar dataframe value to text without leaking ``nan`` labels."""
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _row_model(row: Any) -> str:
    """Return the canonical model label from a record or pandas row."""
    getter = row.get if hasattr(row, "get") else lambda _key, _default=None: _default
    for key in ("modelName", "model", "_plot_model"):
        value = _clean_text(getter(key))
        if value:
            return value
    return ""


def _distinct_units(records) -> list[str]:
    """Return meaningful units in stable order, ignoring blank metadata."""
    units: list[str] = []
    for record in records or []:
        if not hasattr(record, "get"):
            continue
        unit = str(record.get("unit") or "").strip()
        if unit and unit not in units:
            units.append(unit)
    return units


def _unit_key(unit: str, variable: str | None = None) -> str:
    """Normalise harmless spelling variants without merging different dimensions."""
    value = re.sub(r"\s+", " ", str(unit or "").strip().casefold())
    value = value.replace(" per year", "/yr")
    value = re.sub(r"/\s*(?:year|y|a)(?=$|\b)", "/yr", value)
    value = re.sub(r"\s+", "", value)
    variable_key = str(variable or "").strip().casefold()
    # Carbon-price records in the source catalogue use all three spellings
    # below for the same price-per-tonne quantity.  This alias is deliberately
    # variable-scoped; a bare ``/t`` is not assumed to mean CO2 elsewhere.
    if variable_key == "carbon price" or variable_key.startswith("price|carbon"):
        value = re.sub(r"/t(?:co2)?$", "/tco2", value)
    return value


def _record_unit_key(record: Any, variable: str | None = None) -> str:
    record_variable = (
        _clean_text(record.get("variable"))
        if hasattr(record, "get")
        else ""
    )
    return _unit_key(
        record.get("unit") if hasattr(record, "get") else "",
        record_variable or variable,
    )


def _dominant_unit_subset(
    records,
    subject: str,
    *,
    variable: str | None = None,
    requested_unit: str | None = None,
    coverage_dimension: str | None = None,
    required_values: list | tuple | set | None = None,
) -> tuple[list, str, int, str, str | None]:
    """Choose one coherent unit group without mixing physical dimensions.

    Equivalent spellings remain together.  When a single-variable slice has
    genuinely inconsistent unit metadata, the largest coherent group is used
    and the omission is disclosed.  A requested region/model comparison is
    only allowed when the chosen group still covers every compared member.
    """
    concrete_records = [record for record in (records or []) if hasattr(record, "get")]
    groups: dict[str, list] = {}
    blank_records: list = []
    for record in concrete_records:
        raw_unit = _clean_text(record.get("unit"))
        if not raw_unit:
            blank_records.append(record)
            continue
        groups.setdefault(_record_unit_key(record, variable), []).append(record)

    if not groups:
        return concrete_records, "", 0, "", None
    if len(groups) == 1:
        selected = concrete_records
        first_group = next(iter(groups.values()))
        selected_unit = _clean_text(first_group[0].get("unit"))
        return selected, selected_unit, 0, "", None

    required = {
        _clean_text(value).casefold()
        for value in (required_values or [])
        if _clean_text(value)
    }

    def _coverage(group_records: list) -> set[str]:
        if coverage_dimension == "model":
            return {_row_model(record).casefold() for record in group_records if _row_model(record)}
        if coverage_dimension:
            return {
                _clean_text(record.get(coverage_dimension)).casefold()
                for record in group_records
                if _clean_text(record.get(coverage_dimension))
            }
        return set()

    requested_key = _unit_key(requested_unit, variable) if requested_unit else ""
    selected_key = requested_key if requested_key in groups else ""
    if not selected_key:
        variable_key = str(variable or "").casefold()

        def _semantic_preference(unit_key: str) -> int:
            is_annual_rate = "/yr" in unit_key
            if "capacity" in variable_key and "addition" in variable_key:
                return int(is_annual_rate)
            if "capacity" in variable_key:
                return int(not is_annual_rate)
            return 0

        ordered_keys = list(groups)
        selected_key = max(
            ordered_keys,
            key=lambda key: (
                len(_coverage(groups[key]) & required) if required else 0,
                len(groups[key]),
                _semantic_preference(key),
                -ordered_keys.index(key),
            ),
        )

        # A unit such as ``billion US$2010/yr OR local currency`` does not
        # guarantee comparable values across models, even though it may be the
        # largest metadata bucket. Prefer a concrete unit when it retains a
        # substantial share of the records. The 60% threshold prevents a tiny
        # clean-looking minority from replacing the representative group.
        def _is_ambiguous_unit_group(key: str) -> bool:
            return any(
                re.search(r"\b(?:or|and/or)\b", _clean_text(record.get("unit")), re.IGNORECASE)
                for record in groups[key]
            )

        if _is_ambiguous_unit_group(selected_key):
            selected_size = len(groups[selected_key])
            concrete_candidates = [
                key for key in ordered_keys
                if not _is_ambiguous_unit_group(key)
                and len(groups[key]) >= 3
                and len(groups[key]) * 5 >= selected_size * 3
                and (
                    not required
                    or required.issubset(_coverage(groups[key]))
                )
            ]
            if concrete_candidates:
                selected_key = max(
                    concrete_candidates,
                    key=lambda key: (
                        len(_coverage(groups[key]) & required) if required else 0,
                        len(groups[key]),
                        _semantic_preference(key),
                        -ordered_keys.index(key),
                    ),
                )

    if required and not required.issubset(_coverage(groups[selected_key])):
        error = _incompatible_units_message(concrete_records, subject)
        return [], "", 0, "", error or (
            f"I can't combine {subject} on one axis because no compatible unit "
            "covers every requested series. Plot the unit groups separately."
        )

    selected = list(groups[selected_key])
    selected_unit = _clean_text(selected[0].get("unit")) if selected else ""
    omitted_count = len(concrete_records) - len(selected)
    omitted_units: list[str] = []
    for key, group_records in groups.items():
        if key == selected_key:
            continue
        for record in group_records:
            label = _clean_text(record.get("unit"))
            if label and label not in omitted_units:
                omitted_units.append(label)
    if blank_records:
        omitted_units.append("unspecified unit")
    unit_list = ", ".join(f"`{label}`" for label in omitted_units)
    notice = (
        f"Note: using the dominant compatible unit `{selected_unit}` for this plot; "
        f"omitted {omitted_count} series "
        f"reported in {unit_list}. Plot those unit groups separately to inspect them.\n\n"
    )
    return selected, selected_unit, omitted_count, notice, None


def _incompatible_units_message(records, subject: str) -> str | None:
    """Refuse a shared axis when its series use genuinely different units."""
    units = _distinct_units(records)
    unit_keys = {
        _record_unit_key(record)
        for record in (records or [])
        if hasattr(record, "get") and _clean_text(record.get("unit"))
    }
    if len(unit_keys) <= 1:
        return None
    unit_text = ", ".join(f"`{unit}`" for unit in units)
    return (
        f"I can't combine {subject} on one axis because the loaded series use "
        f"incompatible units: {unit_text}. Plot each variable or unit separately, "
        "or narrow the model/scenario scope."
    )


def _expand_years(records) -> pd.DataFrame:
    """Turn record ``years`` mappings into plotting columns without aggregation."""
    frame = pd.DataFrame(records)
    if "years" in frame.columns:
        years_frame = frame["years"].apply(
            lambda value: value if isinstance(value, dict) else {}
        ).apply(pd.Series)
        frame = frame.drop("years", axis=1).join(years_frame)
    if not frame.empty:
        # Keep the raw ``modelName`` column for source pairing and filtering,
        # but use a presentation-only model column for legends, captions, and
        # structured displayed-series scope.
        frame["_plot_model"] = frame.apply(
            lambda row: display_model_label(_row_model(row)), axis=1,
        )
    return frame


def _canonicalize_scoped_region(
    frame: pd.DataFrame,
    requested_region: str | None,
) -> pd.DataFrame:
    """Collapse equivalent raw labels inside one requested region scope.

    Several source workspaces encode one geography differently (for example
    ``India`` and ``IND`` or ``World`` and ``WORLD``).  Filtering correctly
    retains all of those records, but leaving the raw labels in the plotting
    dataframe makes a single-region chart look like a region comparison.  Keep
    every record and rewrite only labels proven equivalent to the resolved
    region the user requested.
    """
    canonical = str(requested_region or "").strip()
    if not canonical or frame.empty or "region" not in frame.columns:
        return frame
    equivalent = frame["region"].map(
        lambda value: regions_equivalent(value, canonical)
    )
    if not bool(equivalent.any()):
        return frame
    normalized = frame.copy()
    normalized.loc[equivalent, "region"] = canonical
    return normalized


def _selected_year_columns(
    frame: pd.DataFrame,
    start_year: int | None,
    end_year: int | None,
) -> list:
    columns = [column for column in frame.columns if str(column).isdigit()]
    if is_latest_year_filter(start_year, end_year):
        return [max(columns, key=lambda value: int(value))] if columns else []
    selected = [
        column for column in columns
        if (start_year is None or int(column) >= int(start_year))
        and (end_year is None or int(column) <= int(end_year))
    ]
    if start_year is not None or end_year is not None:
        return sorted(selected, key=lambda value: int(value))
    return sorted(columns, key=lambda value: int(value))


def _normalise_chart_type(chart_type: object, year_columns: list) -> str:
    """Return the chart type the renderer will actually draw."""
    requested = str(chart_type or "line").strip().casefold().replace("_", " ")
    if requested in {"bar", "bars", "bar chart", "column", "column chart"} and len(year_columns) == 1:
        return "bar"
    if requested in {"scatter", "scatter plot", "scatter chart"}:
        return "scatter"
    if requested in {"area", "area plot", "area chart"}:
        return "area"
    return "line"


def _chart_type_from_question(question: str) -> str | None:
    tokens = set(re.findall(r"[a-z0-9]+", str(question or "").casefold()))
    for chart_type, aliases in (
        ("bar", {"bar", "column"}),
        ("scatter", {"scatter"}),
        ("area", {"area"}),
        ("line", {"line"}),
    ):
        if tokens & aliases:
            return chart_type
    return None


def _resolve_latest_plot_scope(
    year_columns: list,
    start_year: int | None,
    end_year: int | None,
    chart_type: object = None,
) -> tuple[int | None, int | None, object]:
    """Resolve the latest-year sentinel to the concrete displayed year."""
    if not is_latest_year_filter(start_year, end_year) or not year_columns:
        return start_year, end_year, chart_type
    latest_year = max(int(column) for column in year_columns)
    # A bar conveys a one-year cross-series comparison more clearly than a
    # collection of disconnected one-point lines. Honour an explicit type.
    return latest_year, latest_year, chart_type or "bar"


def _unique_label(base_label: str, counts: Dict[str, int]) -> str:
    counts[base_label] = counts.get(base_label, 0) + 1
    occurrence = counts[base_label]
    return base_label if occurrence == 1 else f"{base_label} ({occurrence})"


def _draw_plot_rows(
    frame: pd.DataFrame,
    year_columns: list,
    label_builder,
    *,
    chart_type: object = None,
    style_builder=None,
) -> Tuple[list[str], str]:
    """Render one chart series per row and return its exact display labels."""
    actual_chart_type = _normalise_chart_type(chart_type, year_columns)
    labels: list[str] = []
    label_counts: Dict[str, int] = {}
    for position, (_, row) in enumerate(frame.iterrows()):
        label = _unique_label(str(label_builder(row)), label_counts)
        labels.append(label)
        values = [row.get(column, float("nan")) for column in year_columns]
        style = dict(style_builder(row, position) if style_builder else {})
        if actual_chart_type == "bar":
            plt.bar(position, values[0], label=label, **style)
        elif actual_chart_type == "scatter":
            numeric_years = [int(column) for column in year_columns]
            plt.scatter(numeric_years, values, label=label, **style)
        elif actual_chart_type == "area":
            numeric_years = [int(column) for column in year_columns]
            plt.fill_between(
                numeric_years,
                [0] * len(numeric_years),
                values,
                label=label,
                alpha=0.3,
                **style,
            )
        else:
            numeric_years = [int(column) for column in year_columns]
            plt.plot(numeric_years, values, label=label, marker="o", linewidth=2, **style)
    if actual_chart_type == "bar":
        plt.xticks(range(len(labels)), labels, rotation=35, ha="right")
    return labels, actual_chart_type


def _common_scope_examples(
    member_records: Dict[str, List[Dict]],
    dimensions: Tuple[str, ...],
    *,
    limit: int = 3,
) -> list[dict[str, str]]:
    """Find concrete scopes that exist for every requested comparison member."""
    if not member_records or any(not rows for rows in member_records.values()):
        return []

    def value(record: Dict, dimension: str) -> str:
        if dimension == "model":
            return _row_model(record)
        if dimension == "unit":
            return _record_unit_key(record)
        return str(record.get(dimension) or "").strip()

    tuple_sets = []
    for rows in member_records.values():
        tuple_sets.append({tuple(value(row, dimension) for dimension in dimensions) for row in rows})
    common = set.intersection(*tuple_sets) if tuple_sets else set()
    examples = []
    for values in sorted(common, key=lambda item: tuple(part.casefold() for part in item))[:limit]:
        examples.append(dict(zip(dimensions, values)))
    return examples


def _format_common_scope_suggestions(examples: list[dict[str, str]]) -> str:
    if not examples:
        return (
            "No common scope is available for every requested comparison member. "
            "Plot them separately or change the region, scenario, or model."
        )
    lines = ["Common scopes available to every requested comparison member:"]
    for example in examples:
        parts = [
            f"{dimension} `{display_model_label(value) if dimension == 'model' else value}`"
            for dimension, value in example.items()
            if value
        ]
        lines.append(f"- {'; '.join(parts)}")
    lines.append("Reply with one of these scopes to retry the comparison.")
    return "\n".join(lines)


def _scenario_filter_members(scenario: str | None, ts_data) -> set[str]:
    """Resolve a canonical family to concrete runtime scenario values."""
    if not scenario:
        return set()
    available = {
        str(record.get("scenario") or "").strip()
        for record in (ts_data or [])
        if isinstance(record, dict) and str(record.get("scenario") or "").strip()
    }
    family = scenario_family_members(str(scenario), available)
    return set(family or [str(scenario)])


def _catalogue_variable_suggestions(
    question: str,
    available_vars,
    *,
    ignored_values=(),
    limit: int = 3,
) -> list[str]:
    """Return only runtime candidates with meaningful wording coverage."""
    ranked = rank_catalogue_variable_matches(
        question,
        available_vars,
        ignored_values=ignored_values,
    )
    supported = [
        item["variable"] for item in ranked
        if item["auto_accept"]
        or len(item["matched_terms"]) >= 2
        or item["query_coverage"] >= 0.5
    ]
    return supported[:limit]


def save_plot_to_base64() -> str:
    """Return the current matplotlib figure as an inline markdown image."""
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()
    return f"![Plot](data:image/png;base64,{img_base64})"


from utils.yaml_loader import load_all_yaml_files
from llm_factory import get_chat_openai as ChatOpenAI
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
        model_name=QA_MODEL,
        temperature=0.7,
        timeout=30,
        max_retries=1
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
    
    # The family noun ("primary energy", "capacity", ...) is captured so it can
    # be carried onto each carrier. Resolving a bare "coal" against the
    # catalogue picks `Price|Coal`; "coal primary energy" resolves to
    # `Primary Energy|Coal`, which is what the question asked for.
    family_pattern = (
        r'((?:primary|final|secondary)\s+energy'
        r'|capacity|generation|energy|emissions|production|demand|consumption)'
    )
    comparison_patterns = [
        (r'compare\s+(\w+)\s+(?:and|vs|versus|with)\s+(\w+)', None),
        (rf'(\w+)\s+(?:and|vs|versus)\s+(\w+)\s+{family_pattern}', 3),
        (r'(\w+)\s+vs\s+(\w+)', None),  # Simple "X vs Y" pattern
        (r'both\s+(\w+)\s+and\s+(\w+)', None),
        (r'(\w+)\s+or\s+(\w+)', None),
    ]

    for pattern, family_group in comparison_patterns:
        match = re.search(pattern, query_lower)
        if not match:
            continue
        first, second = match.group(1), match.group(2)
        family = match.group(family_group).strip() if family_group else ""
        if family:
            return [f"{first} {family}", f"{second} {family}"]
        return [first, second]

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
        if r1 and r2 and not regions_equivalent(r1, r2):
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
            if r1 and r2 and not regions_equivalent(r1, r2):
                return [r1, r2]
    return []


@_serialized_plot
def plot_variable_across_regions(question: str, model_data: List[Dict], ts_data: List[Dict],
                                 variable: str, regions: List[str],
                                 scenario: str = None, start_year: int = None,
                                 end_year: int = None,
                                 scenarios: Optional[List[str]] = None,
                                 all_scenarios: bool | None = None,
                                 chart_type: str | None = None) -> str:
    """
    Plot a single variable across multiple regions.
    """
    if not variable or not regions:
        return "Could not identify enough regions to compare."
    metadata = get_metadata(ts_data, model_data)
    typed_scenarios = explicit_scenarios_from_query(
        question,
        {str(record.get("scenario") or "").strip() for record in ts_data if record},
    )
    if len(typed_scenarios) == 1:
        # The rendered comparison query is the final scope contract.  Recover
        # its exact runtime scenario if an upstream singular/plural conversion
        # dropped or generalized the structured field.
        scenario = typed_scenarios[0]
    requested_scenarios = []
    for value in scenarios or []:
        normalized = str(value or "").strip()
        if normalized and normalized not in requested_scenarios:
            requested_scenarios.append(normalized)
    scenario_members = set(requested_scenarios) or _scenario_filter_members(scenario, ts_data)

    # Collect every concrete record. Never collapse rows with
    # ``groupby(...).first()``: a scenario can be reported by several models.
    all_data: Dict[str, List[Dict]] = {}
    catalogue_by_region: Dict[str, List[Dict]] = {}
    for region in regions:
        catalogue_by_region[region] = [
            record for record in ts_data
            if record
            and str(record.get("variable") or "") == variable
            and regions_equivalent(record.get("region"), region)
        ]
        filtered = []
        for r in catalogue_by_region[region]:
            if r is None:
                continue
            if scenario_members and str(r.get('scenario') or '') not in scenario_members:
                continue
            filtered.append(r)
        if filtered:
            all_data[region] = filtered

    missing_regions = [region for region in regions if region not in all_data]
    if missing_regions:
        missing_text = ", ".join(f"`{region}`" for region in missing_regions)
        examples = _common_scope_examples(
            catalogue_by_region,
            ("scenario", "model", "unit"),
        )
        return (
            f"I can't plot the complete region comparison for **{variable}**: "
            f"no data matched the requested scope for {missing_text}.\n\n"
            f"{_format_common_scope_suggestions(examples)}"
        )

    # A region comparison is meaningful only when the selected lines share a
    # source identity. Prefer model/scenario keys present in every requested
    # region; region-only sources are retained only when no paired key exists,
    # in which case the chart is explicitly presented as an unpaired view.
    common_source_keys = _common_comparison_source_keys(all_data)
    paired_source_comparison = bool(common_source_keys)
    source_omitted_series = 0
    if paired_source_comparison:
        original_source_count = sum(len(records) for records in all_data.values())
        all_data = {
            compared_region: [
                record
                for record in records
                if _comparison_source_key(record) in common_source_keys
            ]
            for compared_region, records in all_data.items()
        }
        source_omitted_series = original_source_count - sum(
            len(records) for records in all_data.values()
        )

    contributing_records = [record for rows in all_data.values() for record in rows]
    (
        compatible_records,
        unit,
        unit_omitted_series,
        unit_notice,
        unit_error,
    ) = _dominant_unit_subset(
        contributing_records,
        "these regions",
        variable=variable,
        coverage_dimension="region",
        required_values=list(all_data),
    )
    if unit_error:
        return unit_error
    compatible_ids = {id(record) for record in compatible_records}
    all_data = {
        compared_region: [record for record in records if id(record) in compatible_ids]
        for compared_region, records in all_data.items()
    }

    frames = []
    for compared_region, records in all_data.items():
        frame = _expand_years(records)
        frame["_comparison_region"] = compared_region
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    year_cols = _selected_year_columns(combined, start_year, end_year)
    if not year_cols:
        return "No time series data is available in the requested year range."
    start_year, end_year, chart_type = _resolve_latest_plot_scope(
        year_cols, start_year, end_year, chart_type,
    )

    scenarios_in_data = {
        _clean_text(value) for value in combined.get("scenario", []) if _clean_text(value)
    }
    models_in_scope = {
        _clean_text(value) for value in combined.get("_plot_model", []) if _clean_text(value)
    }
    if paired_source_comparison:
        plotted_df, omitted_series, rendered_common_keys = _paired_region_rows(
            combined,
            list(all_data),
        )
        if not rendered_common_keys:
            # Unit compatibility can remove one half of every originally common
            # key. In that rare case keep the available records, but disclose
            # that the final rendered sources are not paired.
            paired_source_comparison = False
            plotted_df, omitted_series = _representative_rows(
                combined,
                balance_column="_comparison_region",
            )
    else:
        plotted_df, omitted_series = _representative_rows(
            combined,
            balance_column="_comparison_region",
        )
    omitted_series += unit_omitted_series + source_omitted_series

    plt.figure(figsize=(12, 7))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    region_colors = {
        compared_region: colors[index % len(colors)]
        for index, compared_region in enumerate(regions)
    }

    def region_label(row) -> str:
        compared_region = _clean_text(row.get("_comparison_region"))
        model_name = _clean_text(row.get("_plot_model"))
        row_scenario = _clean_text(row.get("scenario"))
        if not paired_source_comparison:
            return " — ".join(
                part for part in (compared_region, model_name, row_scenario) if part
            )
        if len(models_in_scope) > 1:
            parts = [compared_region, model_name]
            if len(scenarios_in_data) > 1:
                parts.append(row_scenario)
            return " — ".join(part for part in parts if part)
        if len(scenarios_in_data) > 1 and row_scenario:
            return f"{compared_region} ({row_scenario})"
        return compared_region

    displayed_series, actual_chart_type = _draw_plot_rows(
        plotted_df,
        year_cols,
        region_label,
        chart_type=chart_type,
        style_builder=lambda row, _position: {
            "color": region_colors.get(_clean_text(row.get("_comparison_region")))
        },
    )

    if paired_source_comparison:
        title = f"{_pretty_variable_name(variable)}: " + " vs ".join(
            format_region_label(region) for region in regions
        )
    else:
        title = f"{_pretty_variable_name(variable)}: region-specific sources — " + " / ".join(
            format_region_label(region) for region in regions
        )
    title += _year_range_text(start_year, end_year)
    plt.title(title, fontsize=12, fontweight='bold')
    plt.xlabel("Series" if actual_chart_type == "bar" else "Year", fontsize=10)
    if unit:
        plt.ylabel(f"{_pretty_variable_name(variable)} ({unit})", fontsize=10)
    plt.grid(alpha=0.3)
    _finalize_plot_layout(len(displayed_series))

    plot_str = save_plot_to_base64()
    ordered_scenarios = [value for value in requested_scenarios if value in scenarios_in_data]
    ordered_scenarios.extend(sorted(
        scenarios_in_data.difference(ordered_scenarios), key=str.casefold,
    ))
    displayed_models = sorted({
        _clean_text(value) for value in plotted_df.get("_plot_model", []) if _clean_text(value)
    })
    if paired_source_comparison:
        source_notice = (
            "Note: this region comparison uses model/scenario sources available "
            "in every requested region. Region-only source series were omitted.\n\n"
            if source_omitted_series
            else ""
        )
        comparison_pairing = "shared model/scenario"
    else:
        source_notice = (
            "Note: no shared model-and-scenario source is available across all "
            "requested regions. Each line is labeled with its own source; this is "
            "a side-by-side view, not a like-for-like paired comparison.\n\n"
        )
        comparison_pairing = "unpaired source scopes"
    return unit_notice + source_notice + _wrap_plot_markdown(
        plot_str, variable, None, scenario, ordered_scenarios, start_year, end_year,
        prefix="Showing", models_in_data=displayed_models,
        regions_in_data=list(all_data.keys()),
        all_scenarios=all_scenarios,
        omitted_series=omitted_series,
        scope_variables=[variable],
        comparison_dimension="region",
        displayed_series=displayed_series,
        chart_type=actual_chart_type,
        unit=unit,
        comparison_pairing=comparison_pairing,
    )


@_serialized_plot
def plot_multiple_variables(question: str, model_data: List[Dict], ts_data: List[Dict],
                            variables: List[str], region: str = None, 
                            scenario: str = None, start_year: int = None, 
                            end_year: int = None,
                            chart_type: str | None = None) -> str:
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
    scenario_members = _scenario_filter_members(scenario, ts_data)
    available_vars = {
        str(record.get("variable") or "").strip()
        for record in ts_data
        if record and str(record.get("variable") or "").strip()
    }
    
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
        
        # The alias catalogue understands the family wording ("coal primary
        # energy" -> `Primary Energy|Coal`); `suggest_variables` only sees the
        # bare token and answered `Price|Coal`. Try the alias layer first.
        if not exact_match:
            alias_match = preferred_variable_from_query(var, available_vars)
            if alias_match:
                resolved_variables.append(alias_match)
                exact_match = True
                logger.debug("Alias-resolved '%s' to '%s'", var, alias_match)

        # If still unresolved, fall back to catalogue keyword suggestions.
        if not exact_match and metadata:
            suggestions = metadata.suggest_variables(var, limit=3)
            if suggestions:
                resolved_variables.append(suggestions[0][0])
                logger.debug("Resolved '%s' to '%s'", var, suggestions[0][0])
    
    resolved_variables = [
        _refine_explicit_technology_leaf(question, variable, available_vars)
        for variable in resolved_variables
    ]
    resolved_variables = list(dict.fromkeys(resolved_variables))
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
    all_data: Dict[str, List[Dict]] = {}
    catalogue_by_variable: Dict[str, List[Dict]] = {}
    
    for variable in resolved_variables:
        catalogue_by_variable[variable] = [
            record for record in ts_data
            if record and str(record.get("variable") or "") == variable
        ]
        filtered_data = []
        for r in catalogue_by_variable[variable]:
            if r is None:
                continue
            if scenario_members and str(r.get('scenario') or '') not in scenario_members:
                continue
            if region:
                if not regions_equivalent(r.get('region'), region):
                    continue
            filtered_data.append(r)
        
        if filtered_data:
            all_data[variable] = filtered_data

    missing_variables = [variable for variable in resolved_variables if variable not in all_data]
    if missing_variables:
        missing_text = ", ".join(f"`{variable}`" for variable in missing_variables)
        requested_scope = []
        if region:
            requested_scope.append(f"region `{region}`")
        if scenario:
            requested_scope.append(f"scenario `{scenario}`")
        scope_text = f" in {' and '.join(requested_scope)}" if requested_scope else ""
        examples = _common_scope_examples(
            catalogue_by_variable,
            ("region", "scenario", "model", "unit"),
        )
        return (
            f"I can't plot the complete variable comparison{scope_text}: no data "
            f"matched for {missing_text}.\n\n{_format_common_scope_suggestions(examples)}"
        )

    unit_omitted_series = 0
    unit_notices: list[str] = []
    for compared_variable, records in list(all_data.items()):
        (
            compatible_records,
            _selected_unit,
            omitted_count,
            unit_notice,
            _unit_error,
        ) = _dominant_unit_subset(
            records,
            f"the {_pretty_variable_name(compared_variable)} series",
            variable=compared_variable,
        )
        all_data[compared_variable] = compatible_records
        unit_omitted_series += omitted_count
        if unit_notice:
            unit_notices.append(unit_notice)

    contributing_records = [record for rows in all_data.values() for record in rows]
    unit_error = _incompatible_units_message(contributing_records, "these variables")
    if unit_error:
        return unit_error
    units = _distinct_units(contributing_records)
    unit = units[0] if units else ""

    frames = []
    for compared_variable, records in all_data.items():
        frame = _expand_years(records)
        frame["_comparison_variable"] = compared_variable
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined = _canonicalize_scoped_region(combined, region)
    year_cols = _selected_year_columns(combined, start_year, end_year)
    if not year_cols:
        return "No time series data is available in the requested year range."
    start_year, end_year, chart_type = _resolve_latest_plot_scope(
        year_cols, start_year, end_year, chart_type,
    )

    scenarios_in_data = {
        _clean_text(value) for value in combined.get("scenario", []) if _clean_text(value)
    }
    regions_in_data = {
        _clean_text(value) for value in combined.get("region", []) if _clean_text(value)
    }
    models_in_scope = {
        _clean_text(value) for value in combined.get("_plot_model", []) if _clean_text(value)
    }
    plotted_df, omitted_series = _representative_rows(
        combined,
        balance_column="_comparison_variable",
    )
    omitted_series += unit_omitted_series
    
    # Create comparison plot
    plt.figure(figsize=(12, 7))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    variable_colors = {
        compared_variable: colors[index % len(colors)]
        for index, compared_variable in enumerate(resolved_variables)
    }

    def variable_label(row) -> str:
        compared_variable = _clean_text(row.get("_comparison_variable"))
        parts = [_pretty_variable_name(compared_variable)]
        if len(models_in_scope) > 1:
            parts.append(_clean_text(row.get("_plot_model")))
        if len(scenarios_in_data) > 1:
            parts.append(_clean_text(row.get("scenario")))
        if not region and len(regions_in_data) > 1:
            parts.append(_clean_text(row.get("region")))
        return " — ".join(part for part in parts if part)

    displayed_series, actual_chart_type = _draw_plot_rows(
        plotted_df,
        year_cols,
        variable_label,
        chart_type=chart_type,
        style_builder=lambda row, _position: {
            "color": variable_colors.get(_clean_text(row.get("_comparison_variable")))
        },
    )
    
    # Build title
    title = f"Comparison: {' vs '.join([_pretty_variable_name(v) for v in all_data.keys()])}"
    if region:
        title += f" for {format_region_label(region)}"
    title += _year_range_text(start_year, end_year)
    
    plt.title(title, fontsize=12, fontweight='bold')
    plt.xlabel("Series" if actual_chart_type == "bar" else "Year", fontsize=10)
    
    if unit:
        plt.ylabel(f"Value ({unit})", fontsize=10)
    
    plt.grid(True, alpha=0.3)
    _finalize_plot_layout(len(displayed_series))

    plot_str = save_plot_to_base64()
    caption_var = " vs ".join([_pretty_variable_name(v) for v in all_data.keys()])
    primary_variable = next(iter(all_data.keys()), "")
    displayed_models = sorted({
        _clean_text(value) for value in plotted_df.get("_plot_model", []) if _clean_text(value)
    })
    return "".join(unit_notices) + _wrap_plot_markdown(
        plot_str, caption_var, region, scenario, sorted(scenarios_in_data), start_year, end_year,
        prefix="Showing comparison of", scope_variable=str(primary_variable),
        models_in_data=displayed_models,
        omitted_series=omitted_series,
        scope_variables=list(all_data.keys()),
        comparison_dimension="variable",
        displayed_series=displayed_series,
        chart_type=actual_chart_type,
        unit=unit,
    )


@_serialized_plot
def plot_model_comparison(question: str, model_data: List[Dict], ts_data: List[Dict],
                          variable: str, models: List[str], region: str = None,
                          scenario: str = None, start_year: int = None, 
                          end_year: int = None,
                          scenarios: Optional[List[str]] = None,
                          all_scenarios: bool | None = None,
                          chart_type: str | None = None) -> str:
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
    scenario_members = {
        str(value).strip() for value in (scenarios or []) if str(value or "").strip()
    } or _scenario_filter_members(scenario, ts_data)
    
    # Resolve variable name if needed
    resolved_variable = None
    for r in ts_data:
        if r and r.get('variable') == variable:
            resolved_variable = variable
            break
    
    if not resolved_variable and metadata and variable:
        suggestions = metadata.suggest_variables(variable, limit=3)
        if suggestions:
            resolved_variable = suggestions[0][0]
            logger.debug("Resolved variable '%s' to '%s'", variable, resolved_variable)
    
    if not resolved_variable:
        return f"Could not identify variable '{variable}'."
    
    # Resolve model names (fuzzy match)
    # Note: ts_data uses 'modelName' field, not 'model'
    resolved_models = []
    available_models = sorted({
        str(r.get('modelName', '') or r.get('model', '')).strip()
        for r in ts_data
        if r and is_presentable_model_label(r.get('modelName') or r.get('model'))
    })
    logger.debug("ts_data length: %s", len(ts_data))
    logger.debug("Available models in ts_data: %s...", available_models[:20])
    logger.debug("Looking for models: %s", models)
    
    # If ts_data is empty, try to get models from model_data
    if not available_models and model_data:
        available_models = sorted({
            str(m.get('modelName', '')).strip()
            for m in model_data
            if m and is_presentable_model_label(m.get('modelName'))
        })
        logger.debug("Using model_data instead. Available models: %s...", available_models[:20])
    
    unmatched_models: List[str] = []
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
            else:
                unmatched_models.append(str(model_name))

    if not resolved_models:
        # Every requested model failed to resolve against the data. When the
        # user named specific models, that means those models carry no
        # timeseries — say so instead of dumping the catalogue.
        if unmatched_models:
            named = ", ".join(f"`{name}`" for name in dict.fromkeys(unmatched_models))
            return (
                f"I can't plot a comparison: {named} "
                f"{'has' if len(unmatched_models) == 1 else 'have'} no timeseries data "
                f"in the IAM PARIS dataset for this variable."
            )
        return f"Could not identify enough models to compare. Found: {resolved_models}. Available models include: {', '.join(available_models[:10])}..."
    
    # Extract region from query if not provided
    if region is None and metadata:
        region = metadata._find_best_region_match(question)
    
    logger.debug("Model comparison - variable: %s, models: %s, region: %s", resolved_variable, resolved_models, region)
    
    # Collect data for each model
    all_data: Dict[str, List[Dict]] = {}
    catalogue_by_model: Dict[str, List[Dict]] = {}
    
    for model_name in resolved_models:
        catalogue_by_model[model_name] = [
            record for record in ts_data
            if record
            and str(record.get("variable") or "") == resolved_variable
            and _row_model(record) == model_name
        ]
        filtered_data = []
        for r in catalogue_by_model[model_name]:
            if r is None:
                continue
            if scenario_members and str(r.get('scenario') or '') not in scenario_members:
                continue
            if region:
                if not regions_equivalent(r.get('region'), region):
                    continue
            filtered_data.append(r)
        
        if filtered_data:
            all_data[model_name] = filtered_data
    for model_name in unmatched_models:
        catalogue_by_model.setdefault(model_name, [])
    
    if not all_data:
        examples = _common_scope_examples(
            catalogue_by_model,
            ("region", "scenario", "unit"),
        )
        return (
            f"No data found for **{resolved_variable}** across the requested models "
            f"in region `{region}`.\n\n{_format_common_scope_suggestions(examples)}"
        )

    # A requested model that resolved to nothing, or resolved but has no rows
    # in this slice, must not turn the whole comparison into a dead end: plot
    # what exists and say which models are missing.
    skipped_models = unmatched_models + [
        model_name for model_name in resolved_models if model_name not in all_data
    ]
    comparison_notice = ""
    if skipped_models:
        skipped_text = ", ".join(f"`{name}`" for name in dict.fromkeys(skipped_models))
        shown_text = ", ".join(f"`{name}`" for name in all_data)
        examples = _common_scope_examples(
            catalogue_by_model,
            ("region", "scenario", "unit"),
        )
        comparison_notice = (
            f"Note: no timeseries data for model(s) {skipped_text} in this "
            f"slice; plotting {shown_text}.\n\n"
            f"{_format_common_scope_suggestions(examples)}\n\n"
        )

    contributing_records = [record for rows in all_data.values() for record in rows]
    (
        compatible_records,
        unit,
        unit_omitted_series,
        unit_notice,
        unit_error,
    ) = _dominant_unit_subset(
        contributing_records,
        "these models",
        variable=resolved_variable,
        coverage_dimension="model",
        required_values=list(all_data),
    )
    if unit_error:
        return unit_error
    compatible_ids = {id(record) for record in compatible_records}
    all_data = {
        compared_model: [record for record in records if id(record) in compatible_ids]
        for compared_model, records in all_data.items()
    }
    frames = []
    for compared_model, records in all_data.items():
        frame = _expand_years(records)
        frame["_comparison_model"] = compared_model
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined = _canonicalize_scoped_region(combined, region)
    year_cols = _selected_year_columns(combined, start_year, end_year)
    if not year_cols:
        return "No time series data is available in the requested year range."
    start_year, end_year, chart_type = _resolve_latest_plot_scope(
        year_cols, start_year, end_year, chart_type,
    )

    scenarios_in_data = {
        _clean_text(value) for value in combined.get("scenario", []) if _clean_text(value)
    }
    regions_in_data = {
        _clean_text(value) for value in combined.get("region", []) if _clean_text(value)
    }
    plotted_df, omitted_series = _representative_rows(
        combined,
        balance_column="_comparison_model",
    )
    omitted_series += unit_omitted_series

    # Create comparison plot
    plt.figure(figsize=(12, 7))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    model_colors = {
        compared_model: colors[index % len(colors)]
        for index, compared_model in enumerate(all_data)
    }

    def model_label(row) -> str:
        compared_model = _clean_text(row.get("_comparison_model"))
        parts = [display_model_label(compared_model)]
        if len(scenarios_in_data) > 1:
            parts.append(_clean_text(row.get("scenario")))
        if not region and len(regions_in_data) > 1:
            parts.append(_clean_text(row.get("region")))
        return " — ".join(part for part in parts if part)

    displayed_series, actual_chart_type = _draw_plot_rows(
        plotted_df,
        year_cols,
        model_label,
        chart_type=chart_type,
        style_builder=lambda row, _position: {
            "color": model_colors.get(_clean_text(row.get("_comparison_model")))
        },
    )
    
    # Build title
    title = f"Model Comparison: {_pretty_variable_name(resolved_variable)}"
    if region:
        title += f" for {format_region_label(region)}"
    title += _year_range_text(start_year, end_year)
    
    plt.title(title, fontsize=12, fontweight='bold')
    plt.xlabel("Series" if actual_chart_type == "bar" else "Year", fontsize=10)
    
    if unit:
        plt.ylabel(f"{_pretty_variable_name(resolved_variable)} ({unit})", fontsize=10)
    
    plt.grid(True, alpha=0.3)
    _finalize_plot_layout(len(displayed_series))

    plot_str = save_plot_to_base64()
    displayed_models = sorted({
        _clean_text(value) for value in plotted_df.get("_comparison_model", []) if _clean_text(value)
    })
    return comparison_notice + unit_notice + _wrap_plot_markdown(
        plot_str, f"model comparison of {_pretty_variable_name(resolved_variable)}",
        region, scenario, sorted(scenarios_in_data), start_year, end_year,
        prefix="Showing", scope_variable=str(resolved_variable),
        models_in_data=displayed_models,
        all_scenarios=all_scenarios,
        omitted_series=omitted_series,
        scope_variables=[resolved_variable],
        scope_models=list(dict.fromkeys(resolved_models)),
        comparison_dimension="model",
        displayed_series=displayed_series,
        chart_type=actual_chart_type,
        unit=unit,
    )


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
    regions_list = entities.get('regions')
    comparison_type = entities.get('comparison')
    requested_chart_type = entities.get('chart_type') or entities.get('plot_type')
    
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
                                    start_year, end_year,
                                    scenarios=entities.get("scenarios"),
                                    all_scenarios=bool(
                                        entities.get("all_scenarios") is True
                                        or (not scenario and not entities.get("scenarios"))
                                    ),
                                    chart_type=requested_chart_type)
    
    metadata = get_metadata(ts_data, model_data)
    if isinstance(regions_list, list) and len(regions_list) >= 2:
        comparison_variable = str(entities.get("variable") or "").strip()
        if comparison_variable:
            comparison_scenario = str(entities.get("scenario") or "").strip() or None
            scoped_scenarios = [
                str(value).strip() for value in (entities.get("scenarios") or [])
                if str(value or "").strip()
            ]
            if comparison_scenario is None and len(scoped_scenarios) == 1:
                comparison_scenario = scoped_scenarios[0]
            return plot_variable_across_regions(
                question,
                model_data,
                ts_data,
                comparison_variable,
                [str(value) for value in regions_list if str(value).strip()],
                comparison_scenario,
                entities.get("start_year"),
                entities.get("end_year"),
                scenarios=scoped_scenarios,
                all_scenarios=bool(
                    entities.get("all_scenarios") is True
                    or (not comparison_scenario and not scoped_scenarios)
                ),
                chart_type=requested_chart_type,
            )
    region_compare = detect_region_comparison(question, metadata)
    available_vars = {str(r.get('variable', '')).strip() for r in ts_data if r and r.get('variable')}
    ranked_vars = []
    significant_words = []
    structured_scenarios = [
        str(value).strip() for value in (entities.get("scenarios") or [])
        if str(value or "").strip()
    ]
    structured_scenario_comparison = bool(
        comparison_type == "scenario"
        and len(structured_scenarios) >= 2
        and str(entities.get("variable") or "").strip()
    )
    # Some extractors expose only their highest-ranked variable in the
    # structured list.  A one-item list is not evidence that a natural-language
    # comparison is singular, so recover both sides from the query before
    # routing.  Two or more structured variables remain authoritative.
    if (
        not structured_scenario_comparison
        and (not isinstance(variables_list, list) or len(variables_list) < 2)
    ):
        detected_variables = detect_multi_variable_comparison(question)
        if len(detected_variables) >= 2:
            variables_list = detected_variables
        elif not isinstance(variables_list, list):
            variables_list = []
    if variables_list and len(variables_list) >= 2 and not structured_scenario_comparison:
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
                                       start_year, end_year,
                                       chart_type=requested_chart_type)
    
    # Fallback: Check for multi-variable comparison using regex patterns
    # Structured comparison state is more authoritative than regex noun
    # splitting.  In a follow-up such as "compare it with <scenario>", the
    # pronoun and scenario label are not variable names.
    comparison_vars = (
        []
        if structured_scenario_comparison
        else detect_multi_variable_comparison(question)
    )
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
                                       start_year, end_year,
                                       chart_type=requested_chart_type)
    
    # Reject an explicitly named place that is not a known region, instead of
    # plotting unfiltered global data for a nonexistent location.
    if not (entities.get('region') or region):
        from data_utils import unknown_named_region as _unknown_region
        _rc = sorted({str(r.get('region', '')).strip() for r in ts_data if r and r.get('region')})
        _mn = sorted({
            str(m.get('modelName', '')).strip()
            for m in model_data
            if m and is_presentable_model_label(m.get('modelName'))
        })
        _bad = _unknown_region(question, _rc, _mn)
        if _bad:
            return (f"I couldn't find `{_bad}` as a region in the IAM PARIS data, so I can't plot it. "
                    f"Try a region like `World`, `EU` or `CHN`, or ask `list regions`.")

    # Use extracted entities directly
    variable = entities.get('variable')
    if isinstance(variable, str):
        variable = variable.strip()
    # Prefer a structured variable name typed verbatim over an extractor result
    # that drifted to a sibling/superstring (e.g. "Final Energy|Industry"
    # widened to "Final Energy (excl. feedstocks)|Industry").
    typed_variable = _explicit_variable_in_query(question, available_vars)
    if typed_variable and typed_variable != variable:
        if not variable or (isinstance(variable, str) and variable.lower() not in question.lower()):
            variable = typed_variable
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
    # Scenario names typed verbatim (e.g. "PR_Baseline and PR_NDC_CP") are ground
    # truth and override an extractor result that collapsed them to a generic
    # family such as "Baseline"; two or more become a scenario comparison.
    typed_scenarios = explicit_scenarios_from_query(
        question,
        {str(r.get('scenario', '')).strip() for r in ts_data if r and r.get('scenario')},
    )
    if len(typed_scenarios) >= 2:
        scenarios_list = typed_scenarios
        scenario = None
        comparison = "scenario"
    elif (
        len(typed_scenarios) == 1
        and not scenarios_list
        and scenario != typed_scenarios[0]
    ):
        # A structured comparison may contain the carried scenario plus the
        # one scenario named in the follow-up.  The single textual mention is
        # only one side of that pair, so it must not collapse the bounded list.
        scenario = typed_scenarios[0]
        scenarios_list = []
    if scenario:
        available_scenarios = {
            str(record.get("scenario", "")).strip()
            for record in ts_data
            if record and str(record.get("scenario", "")).strip()
        }
        family_members = scenario_family_members(str(scenario), available_scenarios)
        if len(family_members) == 1:
            # Preserve the concrete runtime member in the caption and resolved
            # scope when a family maps to exactly one available scenario.
            scenario = family_members[0]
            scenarios_list = []
        elif family_members:
            scenarios_list = family_members
            scenario = None
            comparison = "scenario"
    model = _canonical_ts_model(entities.get('model'), ts_data)
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
        if variable and re.search(r"\bcoal\b", ql) and "coal" not in v:
            variable = None
        if variable and any(t in ql for t in ("electricity generation", "power generation", "generation from")) and "capacity" in v:
            variable = None
        if variable and re.search(r"\b(?:solar\s+pv|photovoltaic|pv)\b", ql) and not ("solar" in v and "pv" in v):
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
            for candidate in _catalogue_variable_suggestions(
                question,
                available_vars,
                ignored_values=(region, scenario, model, format_region_label(region) if region else ""),
            ):
                if candidate not in candidates:
                    candidates.append(candidate)
            candidates = candidates[:3]
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
            supported = set(_catalogue_variable_suggestions(
                question,
                available_vars,
                ignored_values=(region, scenario, model, format_region_label(region) if region else ""),
            ))
            similar = [candidate for candidate in similar if candidate in supported]
            if similar:
                return f"Could not identify a variable to plot. Did you mean: {', '.join(similar[:3])}?"
        return "I could not confidently match that wording to a loaded variable. Which variable should I use?"

    metadata = get_metadata(ts_data, model_data)
    region_compare = detect_region_comparison(question, metadata)
    if region_compare and variable:
        return plot_variable_across_regions(
            question, model_data, ts_data, variable, region_compare, scenario,
            start_year, end_year, chart_type=requested_chart_type,
        )
    
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
    def _filter_records(use_model: bool) -> list:
        out = []
        for r in ts_data:
            if r is None:
                continue
            if str(r.get('variable', '')) != variable:
                continue
            if use_model and model and str(r.get('modelName', '') or r.get('model', '')) != model:
                continue
            row_scenario = str(r.get('scenario', '') or '')
            if scenarios_list:
                if row_scenario not in scenarios_list:
                    continue
            elif scenario and row_scenario != scenario:
                continue
            # Case-insensitive region matching
            if region:
                if not regions_equivalent(r.get('region'), region):
                    continue
            out.append(r)
        return out

    filtered_data = _filter_records(use_model=True)
    # A requested model is part of the answer's scope contract.  Never relax
    # it and silently plot other models: that produces a valid-looking chart
    # for data the user did not request.  Instead, keep the miss explicit and
    # offer models that really do cover the same variable/scope.
    if not filtered_data and model:
        same_slice_records = _filter_records(use_model=False)
        alternative_models = _model_names_from_records(same_slice_records)
        model_variable_records = [
            record for record in ts_data
            if record
            and str(record.get("variable") or "") == variable
            and _row_model(record) == model
        ]
        available_regions = dedupe_equivalent_regions(sorted({
            str(record.get("region") or "").strip()
            for record in model_variable_records
            if str(record.get("region") or "").strip()
        }))
        available_scenarios = sorted({
            str(record.get("scenario") or "").strip()
            for record in model_variable_records
            if str(record.get("scenario") or "").strip()
        })
        requested_scope = []
        if region:
            requested_scope.append(f"region `{format_region_label(region)}`")
        if scenario:
            requested_scope.append(f"scenario `{scenario}`")
        elif scenarios_list:
            requested_scope.append(
                "scenarios " + ", ".join(f"`{value}`" for value in scenarios_list)
            )
        scope_text = f" for {' and '.join(requested_scope)}" if requested_scope else ""
        suggestions = []
        if alternative_models:
            suggestions.append(
                "Models with data for the requested slice: "
                + ", ".join(f"`{name}`" for name in alternative_models[:5])
            )
        if available_regions:
            suggestions.append(
                f"Regions available for `{model}`: "
                + ", ".join(format_region_label(value) for value in available_regions[:5])
            )
        if available_scenarios:
            suggestions.append(
                f"Scenarios available for `{model}`: "
                + ", ".join(f"`{value}`" for value in available_scenarios[:5])
            )
        suggestion_text = (
            "\n".join(suggestions)
            if suggestions
            else "Try another model, region, or scenario from the loaded data."
        )
        return (
            f"No data found for **{variable}** in model `{model}`{scope_text}.\n\n"
            f"{suggestion_text}\n\n"
            "Tell me which alternative scope you want to plot."
        )

    if not filtered_data:
        # A canonical scenario family (notably ``Net Zero``) may be valid in
        # the catalogue but unavailable for this exact region. Reuse the
        # data-side scoped recovery so every suggested scenario works in the
        # requested region and every suggested region works for the requested
        # family. Keep this local import to avoid the module-level circular
        # dependency (data_utils imports this plotter).
        requested_family = str(entities.get("scenario") or "").strip()
        if requested_family:
            from data_utils import _scenario_family_recovery_prompt

            family_prompt = _scenario_family_recovery_prompt(
                ts_data,
                variable=variable,
                region=region,
                scenario=requested_family,
                model=model or None,
            )
            if family_prompt:
                return family_prompt

        available_regions = dedupe_equivalent_regions(sorted(set(
            str(r.get('region', ''))
            for r in ts_data
            if r and r.get('region') and r.get('variable') == variable
        )))
        available_scenarios = sorted(set(str(r.get('scenario', '')) for r in ts_data if r and r.get('scenario') and r.get('variable') == variable))

        suggestions = []
        if available_regions:
            suggestions.append(f"Recommended regions: {', '.join(format_region_label(r) for r in available_regions[:3])}")
        if available_scenarios:
            suggestions.append(f"Recommended scenarios: {', '.join(available_scenarios[:3])}")

        suggestion_text = "\n".join(suggestions) if suggestions else "Try `list variables` to see available options."
        return f"No data found for variable '{variable}'.\n{suggestion_text}"
    
    (
        filtered_data,
        resolved_unit,
        unit_omitted_series,
        unit_notice,
        unit_error,
    ) = _dominant_unit_subset(
        filtered_data,
        "these series",
        variable=variable,
    )
    if unit_error:
        return unit_error
    unit = resolved_unit or str(unit or "").strip()

    # Prepare data for plotting without aggregating model/scenario rows.
    df = _expand_years(filtered_data)
    df = _canonicalize_scoped_region(df, region)
    year_cols = _selected_year_columns(df, start_year, end_year)
    if not year_cols:
        return "No time series data available for plotting."
    start_year, end_year, requested_chart_type = _resolve_latest_plot_scope(
        year_cols, start_year, end_year, requested_chart_type,
    )
    
    # Create plot
    plt.figure(figsize=(12, 7))
    
    # Determine how to group data based on comparison type or data variety
    scenarios_in_data = df['scenario'].unique() if 'scenario' in df.columns else []
    regions_in_data = df['region'].unique() if 'region' in df.columns else ['All']
    # A plotted line is one concrete record. Grouping only by scenario, region,
    # or model and taking ``iloc[0]`` silently discarded the other dimensions
    # whenever more than one varied. Build composite labels from every varying
    # dimension and render every series instead.
    dimension_columns = {
        'scenario': 'scenario',
        'region': 'region',
        'model': '_plot_model',
    }
    comparison_dimension = str(comparison or '').lower()
    varying_dimensions = [
        column
        for column in ('scenario', 'region', '_plot_model')
        if column in df.columns
        and len({_clean_text(value) for value in df[column] if _clean_text(value)}) > 1
    ]
    preferred_column = dimension_columns.get(comparison_dimension)
    if preferred_column in varying_dimensions:
        varying_dimensions.remove(preferred_column)
        varying_dimensions.insert(0, preferred_column)

    label_dimensions = varying_dimensions or [
        column for column in ('_plot_model', 'scenario', 'region') if column in df.columns
    ]
    sort_dimensions = varying_dimensions or label_dimensions
    plotted_df = (
        df.sort_values(sort_dimensions, kind='stable', na_position='last')
        if sort_dimensions else df
    )
    plotted_df, omitted_series = _representative_rows(plotted_df)
    omitted_series += unit_omitted_series
    def direct_label(row) -> str:
        label_parts = [
            _clean_text(row.get(column))
            for column in label_dimensions
            if _clean_text(row.get(column))
        ]
        return " - ".join(label_parts) if label_parts else "Series"

    displayed_series, actual_chart_type = _draw_plot_rows(
        plotted_df,
        year_cols,
        direct_label,
        chart_type=requested_chart_type,
    )
    
    # Build title with context
    title_parts = [_pretty_variable_name(variable)]
    if scenario and len(scenarios_in_data) == 1:
        title_parts.append(f"({scenario})")
    if region and len(regions_in_data) == 1:
        title_parts.append(f"- {format_region_label(region)}")
    if start_year is not None or end_year is not None:
        title_parts.append(_year_range_text(start_year, end_year).strip())
    
    plt.title(" ".join(title_parts), fontsize=12, fontweight='bold')
    plt.xlabel("Series" if actual_chart_type == "bar" else "Year", fontsize=10)
    
    # Use unit in Y-axis label
    ylabel = _pretty_variable_name(variable)
    if unit:
        ylabel = f"{_pretty_variable_name(variable)} ({unit})"
    plt.ylabel(ylabel, fontsize=10)
    
    plt.grid(True, alpha=0.3)
    _finalize_plot_layout(len(displayed_series))

    plot_str = save_plot_to_base64()
    displayed_models = sorted({
        _clean_text(value) for value in plotted_df.get("_plot_model", []) if _clean_text(value)
    })
    return unit_notice + _wrap_plot_markdown(
        plot_str, variable, region, scenario, list(scenarios_in_data), start_year, end_year,
        models_in_data=displayed_models,
        regions_in_data=list(regions_in_data),
        all_scenarios=bool(
            entities.get("all_scenarios") is True
            or (not scenario and not scenarios_list)
        ),
        omitted_series=omitted_series,
        scope_variables=[variable],
        scope_models=[model] if model else None,
        comparison_dimension=str(comparison or "").strip() or None,
        displayed_series=displayed_series,
        chart_type=actual_chart_type,
        unit=unit,
    )


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

    legacy_requested_chart_type = _chart_type_from_question(question)
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
        return plot_multiple_variables(
            question, model_data, ts_data, comparison_vars, region, None,
            start_year, end_year, chart_type=legacy_requested_chart_type,
        )
    
    # Extract region from query if not provided
    if region is None:
        region_path = Path('definitions/region').resolve()
        region_dict = load_all_yaml_files(str(region_path))
        region_candidates = sorted({str(r.get('region', '')).strip() for r in ts_data if r and r.get('region')})
        region = extract_region_from_query(question, region_dict, region_candidates)
        if re.search(r"\b(world|global)\b", question.lower()):
            region = "World"
        if not region:
            from data_utils import unknown_named_region as _unknown_region
            _mn = sorted({
                str(m.get('modelName', '')).strip()
                for m in model_data
                if m and is_presentable_model_label(m.get('modelName'))
            })
            _bad = _unknown_region(question, region_candidates, _mn)
            if _bad:
                return (f"I couldn't find `{_bad}` as a region in the IAM PARIS data, so I can't plot it. "
                        f"Try a region like `World`, `EU` or `CHN`, or ask `list regions`.")

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
        else:
            evidence = next(
                (
                    item for item in rank_catalogue_variable_matches(
                        question,
                        available_vars,
                        ignored_values=(region, format_region_label(region) if region else ""),
                    )
                    if item["variable"] == variable
                ),
                None,
            )
            preferred_alias = preferred_variable_from_query(question, available_vars)
            if preferred_alias != variable and not (evidence and evidence["auto_accept"]):
                variable = ""
    
    # If still no variable and the user gave an explicit variable format, try a closest match
    if not variable and "|" in question:
        available_vars = sorted(set(str(r.get('variable', '')) for r in ts_data if r and r.get('variable')))
        variable = find_closest_variable_name(question, available_vars)
    
    # If region comparison detected and variable resolved, plot across regions
    if region_compare and variable:
        start_year, end_year = _extract_year_range(question.lower())
        return plot_variable_across_regions(
            question, model_data, ts_data, variable, region_compare, None,
            start_year, end_year, chart_type=legacy_requested_chart_type,
        )

    # Try metadata-based variable matching if still no match
    if not variable:
        candidates = []
        preferred_family = _preferred_plot_family_matches(question, available_vars)
        if preferred_family:
            candidates = preferred_family[:3]
        for candidate in _catalogue_variable_suggestions(
            question,
            available_vars,
            ignored_values=(region, format_region_label(region) if region else ""),
        ):
            if candidate not in candidates:
                candidates.append(candidate)
        candidates = candidates[:3]
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
            supported = set(_catalogue_variable_suggestions(
                question,
                available_vars,
                ignored_values=(region, format_region_label(region) if region else ""),
            ))
            similar = [candidate for candidate in similar if candidate in supported]
            if similar:
                return f"Could not identify a variable to plot. Did you mean: {', '.join(similar[:3])}?"
        return "I could not confidently match that wording to a loaded variable. Which variable should I use?"
    
    # Extract model from query if mentioned
    model_match = None
    model_names = sorted({
        str(m.get('modelName', '')).strip()
        for m in model_data
        if m and is_presentable_model_label(m.get('modelName'))
    })
    if model_names:
        model_match = match_model_name(question, model_names)
        # Canonicalize to the timeseries record name (e.g. "GCAM" -> "gcam") so
        # the model filter below is not silently emptied.
        model_match = _canonical_ts_model(model_match, ts_data)

    # Extract scenario from query if mentioned
    scenario = None
    question_lower = question.lower()
    scenarios = sorted({str(r.get('scenario', '')).strip() for r in ts_data if r and r.get('scenario')})
    typed_scenarios = explicit_scenarios_from_query(question, scenarios)
    if len(typed_scenarios) == 1:
        scenario = typed_scenarios[0]
    m = re.search(r"(?:under|scenario)\s+([\w\-\.]+)", question_lower)
    if not scenario and m:
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
    scenario_members = (
        set(typed_scenarios)
        if len(typed_scenarios) >= 2
        else _scenario_filter_members(scenario, ts_data)
    )

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
            scenario=(
                scenario
                if scenario and scenario_members == {scenario}
                else None
            ),
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
        if scenario_members and str(r.get('scenario') or '') not in scenario_members:
            continue
        if region and not regions_equivalent(r.get('region'), region):
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
            and (not region or regions_equivalent(r.get('region'), region))
        })

        def _top_values(key: str, limit: int = 3, filter_region: bool = False) -> list:
            records = [
                r for r in ts_data
                if r and r.get('variable') == variable
                and (not model_match or r.get('modelName') == model_match)
                and (not scenario or r.get('scenario') == scenario)
                and (not filter_region or (region and regions_equivalent(r.get('region'), region)))
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

        if region and not any(regions_equivalent(region, value) for value in scoped_regions):
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
    
    (
        filtered_data,
        unit,
        unit_omitted_series,
        unit_notice,
        unit_error,
    ) = _dominant_unit_subset(
        filtered_data,
        "these series",
        variable=variable,
    )
    if unit_error:
        return unit_error

    # Prepare every concrete model/scenario/region record for plotting.
    df = _expand_years(filtered_data)
    df = _canonicalize_scoped_region(df, region)
    year_cols = _selected_year_columns(df, start_year, end_year)
    if not year_cols:
        return "No time series data available for plotting."
    start_year, end_year, legacy_chart_type = _resolve_latest_plot_scope(
        year_cols, start_year, end_year, legacy_requested_chart_type,
    )
    
    # Create plot
    plt.figure(figsize=(12, 7))
    
    scenarios_in_data = [
        _clean_text(value) for value in df.get('scenario', []) if _clean_text(value)
    ]
    regions_in_data = [
        _clean_text(value) for value in df.get('region', []) if _clean_text(value)
    ]
    varying_dimensions = [
        column for column in ('scenario', 'region', '_plot_model')
        if column in df.columns
        and len({_clean_text(value) for value in df[column] if _clean_text(value)}) > 1
    ]
    label_dimensions = varying_dimensions or [
        column for column in ('_plot_model', 'scenario', 'region') if column in df.columns
    ]
    plotted_df, omitted_series = _representative_rows(df)
    omitted_series += unit_omitted_series

    def direct_label(row) -> str:
        parts = [
            _clean_text(row.get(column))
            for column in label_dimensions
            if _clean_text(row.get(column))
        ]
        return " - ".join(parts) if parts else "Series"

    displayed_series, actual_chart_type = _draw_plot_rows(
        plotted_df,
        year_cols,
        direct_label,
        chart_type=legacy_chart_type,
    )
    
    # Build title
    title_parts = [_pretty_variable_name(variable)]
    if scenario and len(scenarios_in_data) == 1:
        title_parts.append(f"({scenario})")
    if region and len(regions_in_data) == 1:
        title_parts.append(f"- {format_region_label(region)}")
    if start_year is not None or end_year is not None:
        title_parts.append(_year_range_text(start_year, end_year).strip())
    
    plt.title(" ".join(title_parts), fontsize=12, fontweight='bold')
    plt.xlabel("Year", fontsize=10)
    
    # Use unit in Y-axis label
    ylabel = _pretty_variable_name(variable)
    if unit:
        ylabel = f"{_pretty_variable_name(variable)} ({unit})"
    plt.ylabel(ylabel, fontsize=10)

    plt.grid(True, alpha=0.3)
    _finalize_plot_layout(len(displayed_series))

    plot_str = save_plot_to_base64()
    displayed_models = sorted({
        _clean_text(value) for value in plotted_df.get("_plot_model", []) if _clean_text(value)
    })
    return unit_notice + _wrap_plot_markdown(
        plot_str, variable, region, scenario, list(dict.fromkeys(scenarios_in_data)), start_year, end_year,
        models_in_data=displayed_models,
        omitted_series=omitted_series,
        scope_variables=[variable],
        scope_models=[model_match] if model_match else None,
        displayed_series=displayed_series,
        chart_type=actual_chart_type,
        unit=unit,
    )
