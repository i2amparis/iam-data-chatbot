"""Curated glossary for conceptual questions (config/glossary.json).

The general-QA model is small; definitions it produces from memory were often
wrong. The glossary is indexed for retrieval, and a direct definition question
that names a glossary term is answered from it verbatim.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

GLOSSARY_PATH = Path("config/glossary.json")


@lru_cache(maxsize=1)
def load_glossary(path: str = str(GLOSSARY_PATH)) -> tuple[dict, ...]:
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return ()
    terms = []
    for entry in data.get("terms", []) if isinstance(data, dict) else []:
        if isinstance(entry, dict) and entry.get("term") and entry.get("definition"):
            terms.append(entry)
    return tuple(terms)


def glossary_documents() -> list:
    from langchain.schema import Document

    return [
        Document(
            page_content=f"Glossary — {entry['term']}: {entry['definition']} (Source: {entry.get('source', 'n/a')})",
            metadata={"type": "glossary", "term": entry["term"]},
        )
        for entry in load_glossary()
    ]


def _alias_pattern(alias: str) -> re.Pattern:
    words = [re.escape(word) for word in re.findall(r"[a-z0-9.]+", alias.casefold())]
    return re.compile(r"(?<![a-z0-9])" + r"[\s\-]+".join(words) + r"(?![a-z0-9])")


def find_terms(text: str) -> list[dict[str, Any]]:
    """Glossary entries named in *text*, longest alias first, in text order."""
    lowered = str(text or "").casefold()
    hits: list[tuple[int, int, dict]] = []
    for entry in load_glossary():
        best = None
        for alias in [entry["term"], *entry.get("aliases", [])]:
            match = _alias_pattern(alias).search(lowered)
            if match and (best is None or match.end() - match.start() > best[1] - best[0]):
                best = (match.start(), match.end())
        if best:
            hits.append((best[0], best[1], entry))
    # A term inside a longer matched phrase ("scenario" in "NDC scenario")
    # belongs to that phrase, not to a separate glossary entry.
    hits = [
        hit for hit in hits
        if not any(
            other is not hit and other[0] <= hit[0] and hit[1] <= other[1]
            and (other[1] - other[0]) > (hit[1] - hit[0])
            for other in hits
        )
    ]
    hits.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    found: list[dict] = []
    for _start, _end, entry in hits:
        if entry not in found:
            found.append(entry)
    return found


# Words that may surround a glossary term in a pure definition question.
_DEFINITION_FILLER = frozenset("""
what whats is are does do mean means meaning of the a an by define definition
defined explain explanation tell me about term concept difference differences
between and vs versus how differ differs different from compare comparison
in iam iams s you please exactly with regard to called stand stands for
""".split())

_DEFINITION_CUE = re.compile(
    r"^\s*(?:what(?:['’]s|\s+is|\s+are|\s+does|\s+do)\b|define\b|explain\b|"
    r"meaning\s+of\b|tell\s+me\s+about\b|what\s+is\s+meant\s+by\b|"
    r"(?:what(?:['’]s|\s+is|\s+are)\s+)?(?:the\s+)?differences?\s+between\b|"
    r"how\s+(?:is|are|does|do)\b.+\bdiffer\b|"
    r"are\s+(?:the\s+|these\s+|your\s+)?(?:results|projections|scenarios|numbers)\b)",
    re.IGNORECASE,
)


def definition_terms(question: str) -> list[dict[str, Any]]:
    """Glossary entries when *question* only asks what they mean.

    Returns ``[]`` unless the question is phrased as a definition or a
    difference question and contains nothing but glossary terms and
    question words, so "what is the carbon price in 2050" or "what are the
    scenarios for the EU" stay data questions.
    """
    text = str(question or "").strip()
    if not _DEFINITION_CUE.search(text):
        return []
    terms = find_terms(text)
    if not terms:
        return []
    residual = text.casefold()
    aliases = sorted(
        {alias.casefold() for entry in terms for alias in [entry["term"], *entry.get("aliases", [])]},
        key=len,
        reverse=True,
    )
    for alias in aliases:
        residual = _alias_pattern(alias).sub(" ", residual)
    residual = re.sub(r"\([^)]*\)", " ", residual).replace("'", "").replace("’", "")
    leftover = [word for word in re.findall(r"[a-z0-9']+", residual) if word.strip("'") not in _DEFINITION_FILLER]
    return terms if not leftover else []


def definition_answer(terms: list[dict[str, Any]]) -> str:
    """Markdown answer quoting the curated definitions with their sources."""
    blocks = []
    for entry in terms:
        source = f"\n\nSource: {entry['source']}" if entry.get("source") else ""
        blocks.append(f"### {entry['term']}\n\n{entry['definition']}{source}")
    answer = "\n\n".join(blocks)
    if len(terms) >= 2:
        answer += (
            "\n\nTo see how these differ in the data, ask for a variable under each scenario, "
            "for example `CO2 emissions for World in 2050`, then `compare with` another scenario."
        )
    return answer
