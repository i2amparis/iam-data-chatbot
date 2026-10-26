"""Expanded variable definitions from the IAMC nomenclature YAML.

Definitions are written as templates, e.g. ``Emissions|{Level-1 Species}``
with ``description: Emissions of {Level-1 Species}`` and tag files listing
``CO2: {description: carbon dioxide (CO2), unit: Mt CO2/yr}``. Expanding them
gives an exact definition for each real variable (``Emissions|CO2``). The
result is an in-memory lookup used for direct definition answers; it is not
embedded, so the retrieval index and startup time do not grow.
"""

from __future__ import annotations

import itertools
import os
import re
from functools import lru_cache
from typing import Any

from utils.yaml_loader import load_all_yaml_files

DEFINITIONS_DIR = "definitions/variable"
_TAG = re.compile(r"\{([^{}]+)\}")
# A template with several tags expands combinatorially; beyond this many
# combinations it is skipped rather than slowing startup.
MAX_EXPANSIONS_PER_TEMPLATE = 5000


def _entries(items: Any):
    for item in items if isinstance(items, list) else []:
        if isinstance(item, str):
            yield item.strip(), {}
        elif isinstance(item, dict):
            for name, attributes in item.items():
                yield str(name).strip(), attributes if isinstance(attributes, dict) else {}


def _load_tags(files: dict) -> dict[str, list[tuple[str, dict]]]:
    tags: dict[str, list[tuple[str, dict]]] = {}
    for path, items in files.items():
        if not os.path.basename(str(path)).startswith("tag_"):
            continue
        for item in items if isinstance(items, list) else []:
            if isinstance(item, dict):
                for tag, values in item.items():
                    if isinstance(values, list):
                        tags.setdefault(str(tag).strip(), []).extend(_entries(values))
    return tags


def _substitute(text: str, choice: dict[str, tuple[str, dict]], field: str) -> str:
    def replace(match: re.Match) -> str:
        tag = match.group(1)
        if tag not in choice:
            return match.group(0)
        name, attributes = choice[tag]
        if field == "name":
            return name
        if field == "unit":
            return str(attributes.get("unit") or name)
        return str(attributes.get("description") or name)

    return _TAG.sub(replace, text)


@lru_cache(maxsize=1)
def variable_definitions(directory: str = DEFINITIONS_DIR) -> dict[str, dict[str, str]]:
    """``{variable: {description, unit, notes}}`` with templates expanded."""
    files = load_all_yaml_files(directory)
    tags = _load_tags(files)
    definitions: dict[str, dict[str, str]] = {}
    for path, items in files.items():
        if os.path.basename(str(path)).startswith("tag_"):
            continue
        for name, attributes in _entries(items):
            description = " ".join(str(attributes.get("description") or "").split())
            if not name or not description:
                continue
            unit = str(attributes.get("unit") or "")
            notes = " ".join(str(attributes.get("notes") or "").split())
            names = _TAG.findall(name)
            if not names:
                definitions.setdefault(name, {"description": description, "unit": unit, "notes": notes})
                continue
            if any(tag not in tags for tag in names):
                continue
            pools = [tags[tag] for tag in names]
            total = 1
            for pool in pools:
                total *= max(len(pool), 1)
            if total > MAX_EXPANSIONS_PER_TEMPLATE:
                continue
            for combination in itertools.product(*pools):
                choice = dict(zip(names, combination))
                expanded = _substitute(name, choice, "name")
                expanded_notes = _substitute(notes, choice, "description")
                # A template note naming one specific variable (e.g. the CO2
                # balance) does not apply to the other expansions (CH4, N2O).
                quoted = re.findall(r"'([^']*\|[^']*)'", expanded_notes)
                if quoted and not any(value.startswith(expanded) or expanded.startswith(value) for value in quoted):
                    expanded_notes = ""
                definitions.setdefault(expanded, {
                    "description": _substitute(description, choice, "description"),
                    "unit": _substitute(unit, choice, "unit"),
                    "notes": expanded_notes,
                })
    return definitions


def definition_answer(variable: str) -> str:
    """Markdown definition of one variable, or ``""`` when none is known."""
    definition = variable_definitions().get(str(variable or "").strip())
    if not definition:
        return ""
    lines = [f"### {variable}", "", definition["description"].rstrip(".") + "."]
    if definition.get("unit"):
        lines.append(f"\nUnit: `{definition['unit']}`")
    if definition.get("notes"):
        lines.append(f"\nNotes: {definition['notes']}")
    lines.append(
        f"\nSource: IAMC variable nomenclature (definitions/variable). "
        f"Ask `{variable} for World in 2050` to see its values."
    )
    return "\n".join(lines)
