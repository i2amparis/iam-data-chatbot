"""Memoize lookups derived from the loaded timeseries records.

The runtime holds ~600k records and several request paths rebuilt the same
derived structures (workspace groups, model-name sets, per-variable indexes)
from scratch on every query, costing seconds per request. Results are cached
by source-list identity; source records must remain immutable after indexing.

Only large lists are cached: request-scoped filtered lists are small and
short-lived, and are cheap to recompute.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, Callable

MIN_CACHED_RECORDS = 5000
# Entries kept per lookup name: the full record list plus a few frequently
# used study-filtered lists, without letting them evict each other.
ENTRIES_PER_NAME = 8

_cache: dict[str, "OrderedDict[int, tuple[list, Any]]"] = {}
_lock = threading.Lock()


def _fingerprint(records: list) -> int:
    """Use list identity; sampled elements can miss changes in the middle."""
    return id(records)


def cached_on_records(records: list, name: str, builder: Callable[[list], Any]) -> Any:
    """Return ``builder(records)``, reusing the result for the same records.

    Callers must treat the returned value as read-only.
    """
    if not isinstance(records, list) or len(records) < MIN_CACHED_RECORDS:
        return builder(records)
    fingerprint = _fingerprint(records)
    with _lock:
        entries = _cache.get(name)
        if entries is not None and fingerprint in entries:
            entries.move_to_end(fingerprint)
            source, value = entries[fingerprint]
            if source is records:
                return value
    value = builder(records)
    with _lock:
        entries = _cache.setdefault(name, OrderedDict())
        # Retain the source object alongside the result. This prevents Python
        # from reusing its id while the entry exists and verifies exact identity.
        entries[fingerprint] = (records, value)
        while len(entries) > ENTRIES_PER_NAME:
            entries.popitem(last=False)
    return value


def distinct_values(records: list, field: str) -> frozenset:
    """Distinct non-empty stripped string values of ``field``."""
    def build(items: list) -> frozenset:
        return frozenset(
            str(record.get(field, "")).strip()
            for record in items
            if isinstance(record, dict) and str(record.get(field, "") or "").strip()
        )
    return cached_on_records(records, f"distinct:{field}", build)


def records_by_field(records: list, field: str) -> dict:
    """Index records by the stripped string value of ``field``."""
    def build(items: list) -> dict:
        index: dict = {}
        for record in items:
            if isinstance(record, dict):
                index.setdefault(str(record.get(field) or "").strip(), []).append(record)
        return index
    return cached_on_records(records, f"by:{field}", build)


def clear() -> None:
    with _lock:
        _cache.clear()
