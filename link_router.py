import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from runtime_context import load_link_catalog
from link_catalog import DEFAULT_OUTPUT as DEFAULT_LINK_CATALOG


@dataclass(frozen=True)
class RelevantLink:
    title: str
    url: str
    reason: str = ""
    confidence: float = 0.0
    search_hint: str = ""
    category: str = ""
    verified_direct_url: bool = False
    fallback_instruction: str = ""


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", _normalize(text)) if len(token) >= 2}


def _query_terms(query: str, entities: dict[str, Any] | None = None, variable_intent: str = "") -> str:
    pieces = [query, variable_intent]
    for value in (entities or {}).values():
        if isinstance(value, str):
            pieces.append(value)
        elif isinstance(value, list):
            pieces.extend(str(item) for item in value)
    return " ".join(pieces)


def _category_boosts(query: str, agent_name: str, entities: dict[str, Any]) -> dict[str, float]:
    q = _normalize(query)
    boosts = {
        "models": 0.0,
        "results": 0.0,
        "application_library": 0.0,
        "data_stories": 0.0,
        "analysis": 0.0,
        "contact": 0.0,
        "main": 0.0,
    }

    if agent_name == "model_explanation" or entities.get("model"):
        boosts["models"] += 18.0
    if agent_name in {"data_query", "data_plotting"}:
        boosts["results"] += 10.0
    if agent_name == "modelling_suggestions":
        boosts["results"] += 8.0
        boosts["data_stories"] += 4.0

    if any(term in q for term in ["model comparison", "compare models", "sdg"]):
        boosts["models"] += 16.0
    if any(term in q for term in ["result", "scenario", "workspace", "pathway", "policy question"]):
        boosts["results"] += 8.0
    if any(term in q for term in ["tool", "dashboard", "map", "raw data", "online model", "application", "library"]):
        boosts["application_library"] += 16.0
    if any(term in q for term in ["data story", "explainer", "catalogue", "catalog", "barrier", "technology inventory"]):
        boosts["data_stories"] += 14.0
    if "analysis" in q:
        boosts["analysis"] += 10.0
    if any(term in q for term in ["contact", "team", "support"]):
        boosts["contact"] += 28.0

    return boosts


def _reason_for(item: dict[str, Any], matched_keywords: list[str], category_boost: float) -> str:
    if matched_keywords:
        return "Matched: " + ", ".join(matched_keywords[:3])
    if category_boost:
        category = str(item.get("category", "")).replace("_", " ")
        return f"Relevant {category} page for this question."
    return "General IAM PARIS reference."


def _score_item(
    item: dict[str, Any],
    query_text: str,
    query_tokens: set[str],
    category_boosts: dict[str, float],
) -> tuple[float, list[str]]:
    title = str(item.get("title", ""))
    title_norm = _normalize(title)
    url = str(item.get("url", ""))
    keywords = [str(keyword) for keyword in item.get("keywords", []) if keyword]
    haystack = " ".join([title, url, item.get("project", ""), item.get("workspace", ""), *keywords])
    haystack_norm = _normalize(haystack)
    haystack_tokens = _tokens(haystack)

    score = 0.0
    matched_keywords: list[str] = []

    generic_titles = {"home", "models", "results", "application library", "analysis", "contact"}
    if title_norm and title_norm in query_text and title_norm not in generic_titles:
        score += 28.0
        matched_keywords.append(title)

    for keyword in keywords:
        keyword_norm = _normalize(keyword)
        if not keyword_norm:
            continue
        if keyword_norm in query_text:
            score += 12.0
            matched_keywords.append(keyword)
        else:
            overlap = _tokens(keyword) & query_tokens
            if overlap:
                score += min(len(overlap), 4) * 2.0
                if len(matched_keywords) < 3:
                    matched_keywords.append(keyword)

    title_overlap = _tokens(title) & query_tokens
    if title_overlap:
        score += min(len(title_overlap), 4) * 4.0

    generic_terms = {"iam", "paris", "data", "model", "models", "result", "results"}
    meaningful_overlap = (haystack_tokens & query_tokens) - generic_terms
    score += min(len(meaningful_overlap), 8) * 1.5

    category = str(item.get("category", ""))
    score += category_boosts.get(category, 0.0)

    project = str(item.get("project", "")).lower()
    if "ndc" in query_text:
        if "ndc aspects" in project:
            score += 16.0
        elif category == "results" and project:
            score -= 8.0
    if any(term in query_text for term in ["fit for 55", "fit-for-55", "glasgow", "cost of capital", "behavioural", "net zero"]):
        if "iam compact" in project:
            score += 12.0

    if item.get("verified_direct_url"):
        score += 1.0
    if item.get("search_hint") and url.endswith("/application_library"):
        score -= 1.0

    return score, matched_keywords


def suggest_links(
    query: str,
    catalog: list[dict[str, Any]] | None = None,
    *,
    agent_name: str = "",
    entities: dict[str, Any] | None = None,
    variable_intent: str = "",
    limit: int = 3,
) -> list[dict[str, Any]]:
    if catalog is None:
        catalog = load_link_catalog(DEFAULT_LINK_CATALOG)
    if not catalog or limit <= 0:
        return []

    entities = entities or {}
    query_text = _normalize(_query_terms(query, entities, variable_intent))
    query_tokens = _tokens(query_text)
    boosts = _category_boosts(query, agent_name, entities)

    scored: list[tuple[float, dict[str, Any], list[str]]] = []
    for item in catalog:
        score, matched_keywords = _score_item(item, query_text, query_tokens, boosts)
        if score <= 0:
            continue
        scored.append((score, item, matched_keywords))

    scored.sort(
        key=lambda row: (
            -row[0],
            str(row[1].get("category", "")),
            str(row[1].get("title", "")),
            str(row[1].get("url", "")),
        )
    )

    # Minimum-relevance gate: absolute floor filters unrelated queries (junk
    # token overlap tops out well below 10), the relative floor drops weak
    # tail links when one link clearly dominates.
    min_score = 10.0
    top_score = scored[0][0] if scored else 0.0

    selected: list[RelevantLink] = []
    seen: set[str] = set()
    for score, item, matched_keywords in scored:
        if score < min_score or score < top_score * 0.2:
            continue
        # Category boosts alone (e.g. +10 for any results page on a data query)
        # must not qualify a link: require query-specific evidence beyond the
        # boost and the flat verified-URL/search-hint bonuses (±1 point).
        category_boost = boosts.get(str(item.get("category", "")), 0.0)
        if score - category_boost <= 1.5:
            continue
        # Dedup by URL so the same page never appears twice under different
        # titles; fall back to title+hint for items without a URL.
        url = str(item.get("url", ""))
        key = url or f"{item.get('title', '')}|{item.get('search_hint', '')}"
        if key in seen:
            continue
        seen.add(key)
        selected.append(
            RelevantLink(
                title=str(item.get("title", "")),
                url=str(item.get("url", "")),
                reason=_reason_for(item, matched_keywords, boosts.get(str(item.get("category", "")), 0.0)),
                confidence=round(min(score / 50.0, 1.0), 3),
                search_hint=str(item.get("search_hint", "")),
                category=str(item.get("category", "")),
                verified_direct_url=bool(item.get("verified_direct_url")),
                fallback_instruction=str(item.get("fallback_instruction", "")),
            )
        )
        if len(selected) >= limit:
            break

    model_name = str((entities or {}).get("model", "") or "").strip()
    if model_name and agent_name == "model_explanation":
        model_norm = _normalize(model_name)
        # Drop unrelated specific-model entries (e.g. "PowerPlan"/"GEMINI-E3 EU"
        # for a GCAM query). Individual model entries point at the generic
        # /models page with the model name as the title; keep the requested
        # model, the generic "Models" hub, and model tools (e.g. /models/sdg).
        selected = [
            link for link in selected
            if not (
                _normalize(getattr(link, "category", "")) == "models"
                and getattr(link, "url", "") == "https://iamparis.eu/models"
                and _normalize(link.title) not in {"models", model_norm}
            )
        ]
        has_exact_model_link = any(
            _normalize(link.title) == model_norm
            or _normalize(link.search_hint) == model_norm
            for link in selected
        )
        if not has_exact_model_link:
            generic_models = next(
                (
                    item for item in catalog
                    if str(item.get("category", "")) == "models"
                    and str(item.get("url", "")) == "https://iamparis.eu/models"
                    and _normalize(str(item.get("title", ""))) == "models"
                ),
                None,
            )
            if generic_models:
                selected = [
                    link for link in selected
                    if not (
                        _normalize(link.title) == "models"
                        and link.url == "https://iamparis.eu/models"
                    )
                ]
                selected.insert(
                    0,
                    RelevantLink(
                        title="Models",
                        url="https://iamparis.eu/models",
                        reason=f"Open the IAM PARIS Models page and search for `{model_name}`.",
                        confidence=1.0,
                        search_hint=model_name,
                        category="models",
                        verified_direct_url=True,
                        fallback_instruction=f"Open the Models page and search for: {model_name}",
                    ),
                )
                selected = selected[:limit]

    # Only pad with the generic results page for data-centric answers; a general
    # question with no real match is better served by no link than a wrong one.
    if not selected and agent_name in {"data_query", "data_plotting", "modelling_suggestions"}:
        fallback = next((item for item in catalog if item.get("url") == "https://iamparis.eu/results"), None)
        if fallback:
            selected.append(
                RelevantLink(
                    title=str(fallback.get("title", "IAM PARIS Results")),
                    url=str(fallback.get("url")),
                    reason="General IAM PARIS results page.",
                    confidence=0.1,
                    search_hint=str(fallback.get("search_hint", "")),
                    category=str(fallback.get("category", "")),
                    verified_direct_url=bool(fallback.get("verified_direct_url")),
                    fallback_instruction=str(fallback.get("fallback_instruction", "")),
                )
            )

    return [asdict(link) for link in selected]


def format_relevant_links(links: list[dict[str, Any]]) -> str:
    if not links:
        return ""
    lines = ["Relevant IAM PARIS links:"]
    for link in links:
        suffix = f" Search for: {link['search_hint']}." if link.get("search_hint") else ""
        reason = f" - {link['reason']}" if link.get("reason") else ""
        lines.append(f"- [{link['title']}]({link['url']}){reason}{suffix}")
    return "\n".join(lines)


def load_default_catalog(path: Path = DEFAULT_LINK_CATALOG) -> list[dict[str, Any]]:
    return load_link_catalog(path)
