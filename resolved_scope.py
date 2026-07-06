"""Structured resolved-scope channel between answer formatters and the manager.

The data/plot pipelines record the scope they actually resolved (variable,
region, scenario, model) at the point where they format the final answer.
The manager consumes it to persist follow-up context, instead of re-parsing
the rendered markdown with regexes.

Thread-local so concurrent API requests (one per worker thread) don't mix.
"""

import threading

_state = threading.local()


def record_resolved_scope(**scope: object) -> None:
    """Record the scope of the answer being formatted. Empty values and the
    aggregate marker "multiple" are dropped."""
    cleaned = {}
    for key, value in scope.items():
        text = str(value or "").strip()
        if text and text.lower() != "multiple":
            cleaned[key] = text
    _state.scope = cleaned


def consume_resolved_scope() -> dict:
    """Return and clear the last recorded scope (empty dict when none)."""
    scope = dict(getattr(_state, "scope", {}) or {})
    _state.scope = {}
    return scope
