"""Curated explanations for common "why/how" questions (config/explanations.json).

A small local model answers conceptual questions from retrieved context.
When nothing in the index explains the topic directly it stitches an answer
from loosely related variable or glossary text, which is how "why do carbon
prices differ" came back as a list of carbon-price variable names. Each
explanation is indexed, and one whose patterns match the question is placed
first in the context, so the model starts from a correct, sourced text.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

EXPLANATIONS_FILE = Path(__file__).resolve().parent / "config" / "explanations.json"


@lru_cache(maxsize=1)
def load_explanations() -> tuple[dict[str, Any], ...]:
    data = json.loads(EXPLANATIONS_FILE.read_text(encoding="utf-8"))
    entries = []
    for entry in data.get("explanations", []):
        groups = [[re.compile(pattern, re.IGNORECASE) for pattern in group] for group in entry.get("match", [])]
        entries.append({**entry, "_patterns": groups})
    return tuple(entries)


def _text(entry: dict[str, Any]) -> str:
    return f"Explanation — {entry['topic']}: {entry['text']} (Source: {entry.get('source', 'n/a')})"


def explanation_documents() -> list:
    from langchain.schema import Document

    return [
        Document(page_content=_text(entry), metadata={"type": "explanation", "topic": entry["topic"]})
        for entry in load_explanations()
    ]


def matching_explanations(question: str, limit: int = 2) -> list:
    """Explanation documents whose every match group is found in *question*.

    Entries with more match groups are more specific and come first.
    """
    from langchain.schema import Document

    text = str(question or "")
    matched = [
        entry for entry in load_explanations()
        if entry["_patterns"]
        and all(any(pattern.search(text) for pattern in group) for group in entry["_patterns"])
    ]
    matched.sort(key=lambda entry: len(entry["_patterns"]), reverse=True)
    return [
        Document(page_content=_text(entry), metadata={"type": "explanation", "topic": entry["topic"]})
        for entry in matched[:limit]
    ]
