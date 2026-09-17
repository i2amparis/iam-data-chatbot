"""Structured resolved-scope channel between answer formatters and the manager.

The data/plot pipelines record the scope they actually resolved (variable,
region, scenario, model) at the point where they format the final answer.
The manager consumes it to persist follow-up context, instead of re-parsing
the rendered markdown with regexes.

Thread-local so concurrent API requests (one per worker thread) don't mix.
"""

from collections.abc import Iterator, MutableMapping
from dataclasses import dataclass, field
import threading
import re
from typing import Any, Mapping, Optional

_state = threading.local()


def has_numeric_result_table(text: str) -> bool:
    """Recognize populated detailed and compact result tables, not headers alone."""
    columns = None
    for line in str(text or "").splitlines():
        if not line.strip().startswith("|"):
            columns = None
            continue
        cells = [cell.strip() for cell in re.split(r"(?<!\\)\|", line.strip())[1:-1]]
        if cells == ["Year", "Value", "Unit"]:
            columns = [1]
            continue
        if cells[:2] == ["Model", "Scenario"] and cells[-1:] == ["Unit"]:
            columns = [i for i, cell in enumerate(cells) if re.fullmatch(r"\d{4}", cell)]
            continue
        if columns and any(
            i < len(cells) and re.fullmatch(r"[+-]?\d[\d,.]*(?:[eE][+-]?\d+|[KM])?", cells[i])
            for i in columns
        ):
            return True
    return False


_SCOPE_DIMENSIONS = (
    "variable", "variables", "region", "regions", "scenario", "scenarios",
    "model", "models", "unit", "start_year", "end_year", "action", "chart_type",
    "observed_start_year", "observed_end_year",
    "comparison", "all_scenarios", "unmatched_region",
    "comparison_dimension",
    "workspace_code",
)


def clean_scope(scope: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    """Copy a scope while retaining meaningful false/zero values."""
    cleaned: dict[str, Any] = {}
    for key, value in dict(scope or {}).items():
        if value is None or value == "" or value == [] or value == {}:
            continue
        cleaned[key] = list(value) if isinstance(value, (tuple, set)) else value
    return cleaned


@dataclass(frozen=True)
class ClarificationOption:
    kind: str
    value: str

    def as_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "value": self.value}


@dataclass
class PendingClarification(MutableMapping[str, Any]):
    """Typed pending clarification with a legacy mapping interface.

    Existing manager code mutates clarification dictionaries in-place.  This
    wrapper preserves that interface while giving API/state code typed access
    to the base scope, missing dimension and options.
    """

    data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        value: Optional[Mapping[str, Any]],
    ) -> Optional["PendingClarification"]:
        if not value:
            return None
        if isinstance(value, cls):
            return value
        return cls(dict(value))

    @property
    def base_scope(self) -> dict[str, Any]:
        entities = clean_scope(self.data.get("entities") or {})
        return {
            key: value for key, value in entities.items()
            if key in _SCOPE_DIMENSIONS or key == "entity_confidence"
        }

    @property
    def missing_dimension(self) -> str:
        return str(self.data.get("suggested_kind") or "").strip().casefold()

    @property
    def options(self) -> tuple[ClarificationOption, ...]:
        values = list(self.data.get("suggested_options") or [])
        kinds = list(self.data.get("suggested_option_kinds") or [])
        fallback = self.missing_dimension or "variable"
        return tuple(
            ClarificationOption(
                str(kinds[index] if index < len(kinds) else fallback).strip().casefold()
                or fallback,
                str(value).strip(),
            )
            for index, value in enumerate(values)
            if str(value or "").strip()
        )

    def api_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "clarification_id": str(self.data.get("clarification_id") or ""),
            "missing_dimension": self.missing_dimension,
            "base_scope": self.base_scope,
            "options": [option.as_dict() for option in self.options],
        }
        return {
            key: value for key, value in payload.items()
            if value not in (None, "", [], {})
        }

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.data[key] = value

    def __delitem__(self, key: str) -> None:
        del self.data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)


@dataclass
class ConversationState:
    """Authoritative per-session conversational data scope."""

    active_scope: dict[str, Any] = field(default_factory=dict)
    previous_successful_scopes: list[dict[str, Any]] = field(default_factory=list)
    attempted_scope: dict[str, Any] = field(default_factory=dict)
    pending_clarification: Optional[PendingClarification] = None
    response_scope_override: Optional[dict[str, Any]] = None
    revision: int = 0
    max_successful_scopes: int = 8

    @property
    def previous_scope(self) -> dict[str, Any]:
        return (
            self.previous_successful_scopes[-1]
            if self.previous_successful_scopes
            else {}
        )

    @previous_scope.setter
    def previous_scope(self, value: Optional[Mapping[str, Any]]) -> None:
        scope = clean_scope(value)
        if scope:
            if self.previous_successful_scopes:
                self.previous_successful_scopes[-1] = scope
            else:
                self.previous_successful_scopes.append(scope)
        else:
            self.previous_successful_scopes.clear()

    def set_active(self, scope: Optional[Mapping[str, Any]]) -> None:
        self.active_scope = clean_scope(scope)
        self.response_scope_override = None

    def record_success(self, scope: Optional[Mapping[str, Any]]) -> None:
        resolved = clean_scope(scope)
        if not resolved:
            return
        prior = clean_scope(self.active_scope)
        if prior and prior != resolved:
            self.previous_successful_scopes.append(prior)
            if len(self.previous_successful_scopes) > self.max_successful_scopes:
                del self.previous_successful_scopes[:-self.max_successful_scopes]
        self.active_scope = resolved
        self.attempted_scope = {}
        self.response_scope_override = None
        self.revision += 1

    def record_attempt(self, scope: Optional[Mapping[str, Any]]) -> None:
        self.attempted_scope = clean_scope(scope)
        self.revision += 1

    def set_pending(self, value: Optional[Mapping[str, Any]]) -> None:
        self.pending_clarification = PendingClarification.from_mapping(value)
        self.revision += 1

    def response_entities(self) -> dict[str, Any]:
        """Use a pending base scope only when no successful scope exists."""
        if self.response_scope_override is not None:
            return clean_scope(self.response_scope_override)
        active = clean_scope(self.active_scope)
        if active:
            return active
        if self.pending_clarification:
            return self.pending_clarification.base_scope
        return clean_scope(self.attempted_scope)


def record_resolved_scope(**scope: object) -> None:
    """Record the scope of the answer being formatted.

    Keep native values (notably years, lists and booleans) so conversational
    follow-ups can mutate the resolved scope without parsing rendered markdown.
    Empty scalar values and aggregate dimension markers such as a ``multiple``
    model or scenario are dropped. ``unit=multiple`` is meaningful, however:
    it explicitly clears any earlier single-unit extractor guess when the
    rendered table contains more than one compatible unit label.
    """
    cleaned = {}
    for key, value in scope.items():
        if value is None:
            continue
        if isinstance(value, str):
            text = value.strip()
            if text and (text.lower() != "multiple" or key == "unit"):
                cleaned[key] = text
        elif isinstance(value, (list, tuple, set)):
            values = [item for item in value if item not in (None, "")]
            if values:
                cleaned[key] = values
        else:
            cleaned[key] = value
    _state.scope = cleaned


def consume_resolved_scope() -> dict:
    """Return and clear the last recorded scope (empty dict when none)."""
    scope = dict(getattr(_state, "scope", {}) or {})
    _state.scope = {}
    return scope
