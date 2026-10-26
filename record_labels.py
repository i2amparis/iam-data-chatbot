"""Normalize label fields of loaded timeseries records once, at load time.

The source data spells the same label in several ways: trailing spaces
(``"Emissions|CO2|Energy|Supply|Electricity "``), and case-only variants
(``World``/``WORLD``, ``e3me``/``E3ME``, ``EJ/yr``/``EJ/Yr``). Each spelling
became a separate variable, region, model or unit, so lists showed duplicates
("MUSE, muse") and exact filters silently missed records.

Whitespace is collapsed in every label field. Case-only variants are merged
into the most common spelling for variables, regions, models and units
(for regions "_" and spaces, and for variables "&" and "and", also count
as the same spelling).
Scenario names are left as published (only whitespace is fixed): they are
study identifiers shown on the IAM PARIS website.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Iterable

WHITESPACE_FIELDS = ("variable", "region", "scenario", "modelName", "model", "unit", "workspace_code")
CASE_MERGED_FIELDS = ("variable", "region", "modelName", "unit")


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _canonical_spellings(records: list, field: str) -> dict[str, str]:
    """Map each spelling to the most common spelling of its case-folded form."""
    counts: Counter = Counter(
        record[field] for record in records
        if isinstance(record, dict) and isinstance(record.get(field), str) and record[field]
    )
    groups: dict[str, list[str]] = defaultdict(list)
    for spelling in counts:
        key = spelling.casefold()
        if field == "region":
            # "South_Korea" and "South Korea" name the same region.
            key = re.sub(r"[_\s]+", " ", key)
        elif field == "variable":
            # "Residential & Commercial", "Residential&Commercial" and
            # "Residential and Commercial" are one variable.
            key = re.sub(r"\s+", " ", key.replace("&", " and ")).strip()
        groups[key].append(spelling)
    mapping: dict[str, str] = {}
    for spellings in groups.values():
        if len(spellings) < 2:
            continue
        # Most records wins; ties prefer the spelling with more capitals
        # ("GEMINI-E3" over "Gemini-E3"), then a stable order.
        canonical = max(spellings, key=lambda value: (counts[value], sum(ch.isupper() for ch in value), value))
        for spelling in spellings:
            if spelling != canonical:
                mapping[spelling] = canonical
    return mapping


def normalize_record_labels(records: Iterable[dict]) -> dict[str, int]:
    """Normalize label fields in place; return the number of changed values per field."""
    records = [record for record in records if isinstance(record, dict)]
    changed: Counter = Counter()
    for record in records:
        for field in WHITESPACE_FIELDS:
            value = record.get(field)
            if isinstance(value, str):
                cleaned = _clean(value)
                if field == "variable":
                    # "Biodiversity| Conservation Costs" -> "Biodiversity|Conservation Costs"
                    cleaned = re.sub(r"\s*\|\s*", "|", cleaned)
                if cleaned != value:
                    record[field] = cleaned
                    changed[field] += 1
    for field in CASE_MERGED_FIELDS:
        mapping = _canonical_spellings(records, field)
        if not mapping:
            continue
        for record in records:
            value = record.get(field)
            if value in mapping:
                record[field] = mapping[value]
                changed[field] += 1
    return dict(changed)
