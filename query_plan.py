"""Data-driven query planning before entity extraction.

This module deliberately contains no IAM variable, model, scenario, region or
URL literals.  It identifies generic conversational operations and lets the
runtime catalogues resolve their values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Iterable

from year_filters import YearFilter, extract_year_filter


_WORDS = {
    "region": ("region", "regions", "geography", "geographies", "country", "countries"),
    "scenario": ("scenario", "scenarios", "pathway", "pathways"),
    "model": ("model", "models"),
    "variable": ("variable", "variables", "metric", "metrics", "indicator", "indicators"),
}


def _contains_word(text: str, words: Iterable[str]) -> bool:
    return any(re.search(r"\b" + re.escape(word) + r"\b", text) for word in words)


def _known_mentions(query: str, values: Iterable[str]) -> list[str]:
    q = query.casefold()
    matches = []
    for value in sorted({str(v).strip() for v in values if str(v).strip()}, key=len, reverse=True):
        normalized = value.casefold()
        if re.search(r"(?<![\w-])" + re.escape(normalized) + r"(?![\w-])", q):
            if not any(normalized in found.casefold() for found in matches):
                matches.append(value)
    return matches


def _replacement_value(text: str) -> str:
    """Remove a trailing scope-preservation instruction from a patch value.

    For example, ``Future_Path and keep the same year`` names one replacement
    value followed by a separate operation. Resolution of ``Future_Path`` is
    still delegated to the runtime catalogue.
    """
    value = str(text or "").strip()
    return re.split(
        r"\s+(?:and|while|but)\s+(?=(?:keep|preserve|retain|leave)\b)",
        value,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()


@dataclass(frozen=True)
class ScopePatch:
    """A typed, dimension-preserving mutation of an existing data scope."""

    replacement_dimension: str | None = None
    replacement_value: str | None = None
    output_mode: str | None = None
    year_filter: YearFilter = field(default_factory=YearFilter)

    @property
    def has_changes(self) -> bool:
        return bool(
            (self.replacement_dimension and self.replacement_value)
            or self.output_mode
            or self.year_filter.explicit
        )

    def apply(self, scope: dict[str, Any]) -> dict[str, Any]:
        updated = dict(scope or {})
        dimension = str(self.replacement_dimension or "").strip()
        value = self.replacement_value
        if dimension and value not in (None, ""):
            updated[dimension] = value
            plural = {
                "variable": "variables",
                "region": "regions",
                "scenario": "scenarios",
                "model": "models",
            }.get(dimension)
            if plural:
                updated.pop(plural, None)
            if dimension == "scenario":
                updated["all_scenarios"] = False
        updated = self.year_filter.apply(updated)
        if self.output_mode:
            updated["action"] = "plot" if self.output_mode == "plot" else "query"
        return updated


@dataclass(frozen=True)
class QueryPlan:
    intent: str = "answer"
    output_mode: str | None = None
    availability_targets: tuple[str, ...] = ()
    mentioned_models: tuple[str, ...] = ()
    followup: bool = False
    replacement_dimension: str | None = None
    replacement_value: str | None = None
    start_year: int | None = None
    end_year: int | None = None
    year_filter: YearFilter = field(default_factory=YearFilter)
    scope_patch: ScopePatch = field(default_factory=ScopePatch)
    preserve_scope: bool = False
    diagnostics: dict = field(default_factory=dict)


def build_query_plan(
    query: str,
    *,
    available_models: Iterable[str] = (),
) -> QueryPlan:
    q = str(query or "").strip()
    lower = q.casefold()
    models = _known_mentions(q, available_models)

    comparison_language = bool(re.search(
        r"\b(?:compar(?:e|ed|es|ing|ison)|contrast(?:ed|ing|s)?|"
        r"differ(?:s|ed|ent|ently|ence|ences)?|versus|vs\.?|between)\b",
        lower,
    ))
    model_comparison = comparison_language and len(models) >= 2

    availability_language = bool(
        re.search(
            r"\b(?:available|availability|can\s+i|can\s+be|report|reports|"
            r"provide|provides|provided|cover|covers|which|what)\b",
            lower,
        )
    )
    mentioned_dimensions = tuple(
        key for key, words in _WORDS.items() if _contains_word(lower, words)
    )
    projection = re.search(
        r"\b(?:which|what|list|show)\s+(.{0,80}?)"
        r"(?=\s+(?:are|is|do|does|can|report|reports|provide|provides|cover|covers)\b|[?.!]|$)",
        lower,
    )
    projected_dimensions = tuple(
        key for key, words in _WORDS.items()
        if projection and _contains_word(projection.group(1), words)
    )
    targets = projected_dimensions or mentioned_dimensions

    negative_plot = bool(
        re.search(
            r"(?:\b(?:instead\s+of|rather\s+than|without|no)\b|\b(?:do|does|did)\s+not\b)"
            r"[^.?!]*\b(?:plot|chart|graph|visuali[sz]ation)\b",
            lower,
        )
    )
    asks_plot = bool(re.search(r"\b(?:plot|chart|graph|visuali[sz]e|draw)\b", lower))
    asks_table = bool(re.search(r"\b(?:table|tabular|value|values|number|numbers|text)\b", lower))
    output_mode = "table" if negative_plot or (asks_table and not asks_plot) else ("plot" if asks_plot else None)

    year_filter = extract_year_filter(q)
    start_year = year_filter.start_year
    end_year = year_filter.end_year

    followup_markers = bool(
        re.search(
            r"\b(?:same|keep|switch|change|instead|only|now|everything\s+else|"
            r"that|this|it|there|them|these|those|comparison)\b",
            lower,
        )
    )
    # A compact year-only phrase is a scope patch even without a pronoun:
    # ``after 2030`` and ``until 2050`` should mutate the previous result.
    year_only_followup = bool(
        year_filter.explicit
        and re.fullmatch(
            r"\s*(?:(?:show|plot|chart|graph)(?:\s+me)?\s+)?"
            r"(?:after|before|by|until|up\s+to|through|from|since|in|at|around|only|just)"
            r"\s+(?:19|20|21|22)\d{2}"
            r"(?:\s*(?:-|–|—|to|until|and)\s*(?:19|20|21|22)\d{2})?\s*[?.!]?\s*",
            q,
            flags=re.IGNORECASE,
        )
    )
    followup_markers = followup_markers or year_only_followup
    replacement_dimension = None
    replacement_value = None
    switch = re.search(
        r"\b(?:switch|change|replace)(?:\s+(?:only|just))?(?:\s+the)?\s+"
        r"(region|geography|country|scenario|pathway|model|metric|variable)\s+(?:to|with)\s+([^,.?!]+)",
        q,
        re.IGNORECASE,
    )
    if switch:
        raw_dimension = switch.group(1).casefold()
        replacement_dimension = {
            "geography": "region", "country": "region", "pathway": "scenario",
            "metric": "variable",
        }.get(raw_dimension, raw_dimension)
        replacement_value = _replacement_value(switch.group(2))

    # Natural patch phrasing often puts the dimension after its value, e.g.
    # "use current-policy scenarios instead". The runtime catalogue resolves
    # the captured value later, so this stays domain-independent.
    if not replacement_dimension:
        use_instead = re.search(
            r"\b(?:use|select|choose)\s+(?:the\s+)?(.+?)\s+"
            r"(regions?|geograph(?:y|ies)|countries|scenarios?|pathways?|models?|"
            r"metrics?|variables?|indicators?)\s+instead\b",
            q,
            re.IGNORECASE,
        )
        if use_instead:
            raw_dimension = use_instead.group(2).casefold()
            if raw_dimension.startswith(("region", "geograph", "countr")):
                replacement_dimension = "region"
            elif raw_dimension.startswith(("scenario", "pathway")):
                replacement_dimension = "scenario"
            elif raw_dimension.startswith("model"):
                replacement_dimension = "model"
            else:
                replacement_dimension = "variable"
            replacement_value = _replacement_value(use_instead.group(1))

    # The same value-before-dimension shape is also common without "instead":
    # "switch to <value> scenarios".  The value is still resolved against the
    # runtime catalogue by the manager.
    if not replacement_dimension:
        switch_to = re.search(
            r"\b(?:switch|change|move)\s+to\s+(?:the\s+)?(.+?)\s+"
            r"(regions?|geograph(?:y|ies)|countries|scenarios?|pathways?|models?|"
            r"metrics?|variables?|indicators?)(?:\b|[.?!,])",
            q,
            re.IGNORECASE,
        )
        if switch_to:
            raw_dimension = switch_to.group(2).casefold()
            if raw_dimension.startswith(("region", "geograph", "countr")):
                replacement_dimension = "region"
            elif raw_dimension.startswith(("scenario", "pathway")):
                replacement_dimension = "scenario"
            elif raw_dimension.startswith("model"):
                replacement_dimension = "model"
            else:
                replacement_dimension = "variable"
            replacement_value = _replacement_value(switch_to.group(1))

    # A user may omit the dimension entirely when preserving the other scope:
    # "use <value>, but keep everything else".  Mark it as an automatic
    # replacement; the manager classifies <value> from the live catalogues.
    if not replacement_dimension:
        preserve_value = re.search(
            r"\b(?:use|select|choose)\s+(?:the\s+)?(.+?)\s+"
            r"(?:and|but|while)\s+(?:keep|preserve|retain|leave)\b",
            q,
            re.IGNORECASE,
        )
        if preserve_value:
            replacement_dimension = "auto"
            replacement_value = _replacement_value(preserve_value.group(1))

    # Restricting an existing result to one named model is a scope mutation,
    # not a request to enumerate the model catalogue. The actual model value
    # is still resolved against the runtime catalogue by the manager.
    if not replacement_dimension:
        only_model = re.search(
            r"\b(?:show|use|select|keep|retain)\s+only\s+(?:the\s+)?model\s+"
            r"(.+?)(?=\s*[?!]\s*$|\s*\.\s*$|$)",
            q,
            re.IGNORECASE,
        )
        if only_model:
            replacement_dimension = "model"
            replacement_value = _replacement_value(only_model.group(1))

    # The word ``model`` is often omitted when the value itself is an exact
    # runtime catalogue label ("show only Alpha 2.0").  Require the known
    # value immediately after the restrictive phrase so an ordinary request
    # such as "show only emissions for Alpha" is not misclassified.
    if not replacement_dimension and len(models) == 1:
        model_value = models[0]
        if re.search(
            r"\b(?:show|use|select|keep|retain)\s+only\s+(?:the\s+)?"
            + re.escape(model_value.casefold())
            + r"(?=$|[,.?!]|\s+(?:and|while|but)\b)",
            lower,
        ):
            replacement_dimension = "model"
            replacement_value = model_value

    # ``all available scenarios`` and ``every reported region`` describe the
    # requested plot scope; they are not requests for a catalogue listing when
    # the user explicitly asked to plot/chart/graph the data.  Let extraction
    # and the plotter handle (or safely clarify) that scope instead of returning
    # an unrelated availability table.
    intent = "model_comparison" if model_comparison else (
        "availability"
        if availability_language and targets and output_mode != "plot"
        else "answer"
    )
    scope_patch = ScopePatch(
        replacement_dimension=replacement_dimension,
        replacement_value=replacement_value,
        output_mode=output_mode,
        year_filter=year_filter,
    )
    return QueryPlan(
        intent=intent,
        output_mode=output_mode,
        availability_targets=targets,
        mentioned_models=tuple(models),
        followup=followup_markers,
        replacement_dimension=replacement_dimension,
        replacement_value=replacement_value,
        start_year=start_year,
        end_year=end_year,
        year_filter=year_filter,
        scope_patch=scope_patch,
        preserve_scope=followup_markers,
        diagnostics={"comparison_language": comparison_language, "availability_language": availability_language},
    )


def render_scope_query(scope: dict, plan: QueryPlan) -> str:
    """Render one normalized query from a scope mutation.

    Values come exclusively from runtime extraction/catalogues or the user's
    replacement phrase; this function has no domain-specific values.
    """
    merged = plan.scope_patch.apply(dict(scope or {}))

    lead = "plot" if merged.get("action") == "plot" else "show"
    parts = [lead]
    if merged.get("variable"):
        parts.append(str(merged["variable"]))
    if merged.get("region"):
        parts.append(f"for {merged['region']}")
    scenario = merged.get("scenario")
    if not scenario:
        scenarios = list(merged.get("scenarios") or [])
        if len(scenarios) == 1:
            scenario = scenarios[0]
    if scenario:
        parts.append(f"under {scenario}")
    if plan.year_filter.explicit:
        rendered_year = plan.year_filter.render()
    else:
        carried_year = YearFilter(
            merged.get("start_year"),
            merged.get("end_year"),
            explicit=(
                merged.get("start_year") is not None
                or merged.get("end_year") is not None
            ),
        )
        rendered_year = carried_year.render()
    if rendered_year:
        parts.append(rendered_year)
    if merged.get("model"):
        parts.append(f"for model {merged['model']}")
    return " ".join(parts)
