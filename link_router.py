import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from runtime_context import load_link_catalog
from link_catalog import DEFAULT_OUTPUT as DEFAULT_LINK_CATALOG
from model_aliases import is_presentable_model_label, is_unlabelled_model_display


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


# --- Scoring weights ---------------------------------------------------------
# All link-ranking magic numbers live here so they can be tuned in one place.
EXACT_TITLE_SCORE = 28.0            # full link title appears in the query
KEYWORD_PHRASE_SCORE = 12.0         # catalog keyword phrase appears in the query
KEYWORD_TOKEN_SCORE = 2.0           # per overlapping keyword token (max 4)
TITLE_TOKEN_SCORE = 4.0             # per overlapping title token (max 4)
TITLE_COVERAGE_BONUS = 24.0         # most meaningful title terms appear in query
MEANINGFUL_TOKEN_SCORE = 1.5        # per non-generic overlapping token (max 8)
VERIFIED_URL_BONUS = 1.0
SEARCH_HINT_PENALTY = 1.0           # generic application-library entry with a hint
NDC_PROJECT_BONUS = 16.0
NDC_OTHER_RESULTS_PENALTY = 8.0
IAM_COMPACT_PROJECT_BONUS = 12.0
MIN_LINK_SCORE = 10.0               # absolute relevance floor
RELATIVE_SCORE_FLOOR = 0.2          # fraction of the top score a link must reach
CATEGORY_EVIDENCE_MARGIN = 1.5      # query evidence required beyond category boost
THEMATIC_FIELD_COVERAGE = 0.5       # coherent share of one metadata identity field
MAX_ALIAS_DOCUMENT_FREQUENCY = 0.10 # reject aliases common across the live catalogue
MIN_CATALOG_FOR_FREQUENCY_GATE = 20 # small fixture/catalogue sets lack useful IDF


# Structured extraction contains both user-requested entities and runtime
# metadata (candidate lists, available values, result scope, and so on).  Only
# these stable request dimensions are useful as link-ranking evidence.  The raw
# query remains the primary signal; this small schema allow-list prevents an
# unrelated candidate/scenario from becoming a link keyword by accident.
_LINK_QUERY_ENTITY_KEYS = ("variable", "region", "scenario", "model", "models")

# Generic language used to ask for a destination is useful for detecting
# navigation syntax but carries no information about *which* catalogue item the
# user wants.  Keeping it out of destination matching makes the target entirely
# catalogue-driven.
_NAVIGATION_FILLER_STEMS = {
    "a", "about", "access", "also", "and", "browse", "can", "catalog",
    "catalogue", "directory", "doc", "documentation", "for", "find",
    "give", "go", "how", "i", "iam", "in", "link", "me", "name", "navigate", "of",
    "on", "open", "page", "read", "route", "send", "show", "site",
    "specifically", "take", "the", "this", "to", "url", "use", "user",
    "valid", "view", "visit", "want", "website", "where", "which", "with", "paris",
}

# These words describe catalogue/result plumbing rather than subject matter.
# A specific data link needs stronger evidence than an overlap on one of them.
_THEMATIC_GENERIC_STEMS = {
    "a", "about", "analysis", "and", "application", "catalog", "catalogue",
    "data", "for", "iam", "in", "library", "model", "of", "page", "paris",
    "policy", "project", "question", "result", "scenario", "section", "the",
    "to", "use", "workspace",
}

# Ordinary data dimensions can legitimately appear in a workspace's keywords,
# but they do not identify that workspace on their own.  A query such as
# ``final energy demand for India`` must not therefore acquire a Buildings
# Transformation link merely because that row contains the keyword
# ``energy demand``.
_BROAD_DATA_TOPIC_STEMS = {
    "capaciti", "carbon", "demand", "electriciti", "emission", "energi",
    "final", "generat", "gdp", "populat", "price", "primari",
    "secondari", "suppli",
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", _normalize(text)) if len(token) >= 2}


def _stem_token(token: str) -> str:
    """Small language-agnostic-enough normalizer for ranking, not retrieval.

    It only collapses common English inflections so catalogue nouns can match
    user verb/adjective forms (for example relocation/relocating). Domain terms
    and destinations remain entirely catalogue-driven.
    """
    value = str(token or "").lower()
    if value.endswith("ies") and len(value) > 5:
        value = value[:-3] + "y"
    if value.endswith("ations") and len(value) > 8:
        value = value[:-6] + "ate"
    elif value.endswith("ation") and len(value) > 7:
        value = value[:-5] + "ate"
    elif value.endswith("ison") and len(value) > 7:
        # Align noun/verb pairs such as comparison/compare without knowing
        # anything about the catalogue subject being compared.
        value = value[:-4]
    for suffix in ("ments", "ment", "ingly", "edly", "ing", "ed", "es", "s"):
        if value.endswith(suffix) and len(value) - len(suffix) >= 4:
            value = value[:-len(suffix)]
            break
    # Align productive noun/adjective variants without knowing the subject
    # vocabulary (for example ``industry`` / ``industrial``).
    if value.endswith("ial") and len(value) > 6:
        value = value[:-2]
    elif value.endswith("y") and len(value) > 4:
        value = value[:-1] + "i"
    # Forms ending in silent-e and their gerunds should share a root.
    if value.endswith("e") and len(value) > 5:
        value = value[:-1]
    return value


def _stemmed_tokens(text: str) -> set[str]:
    return {_stem_token(token) for token in _tokens(text)}


def _category_tokens(value: str) -> set[str]:
    """Return comparable tokens for a runtime link-catalog category/title."""
    return _stemmed_tokens(str(value or "").replace("_", " ").replace("-", " "))


def catalog_category_root(
    catalog: list[dict[str, Any]],
    category: str,
) -> dict[str, Any] | None:
    """Find a category root from catalog metadata, without knowing its URL.

    Root entries are ordinary catalogue routes whose title represents the
    category itself and which do not require a search hint.  This lets a
    navigation request resolve a generic destination such as a model directory
    without weakening normal link-relevance scoring.
    """
    wanted = _normalize(str(category or "")).replace("_", " ")
    wanted_tokens = _category_tokens(wanted)
    if not wanted_tokens:
        return None

    candidates: list[dict[str, Any]] = []
    for item in catalog:
        item_category = _normalize(str(item.get("category", ""))).replace("_", " ")
        if item_category != wanted or item.get("dead") or item.get("search_hint"):
            continue
        item_type = _normalize(str(item.get("item_type", "")))
        # Older runtime catalogues did not persist item_type.  Category, root-
        # shaped title, absence of a search hint, and live status still provide
        # enough structured evidence; explicitly non-route records stay out.
        if item_type and item_type != "route":
            continue
        title_tokens = _category_tokens(str(item.get("title", "")))
        if wanted_tokens.issubset(title_tokens):
            candidates.append(item)

    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda item: (
            len(_category_tokens(str(item.get("title", ""))) - wanted_tokens),
            not bool(item.get("verified_direct_url")),
            str(item.get("title", "")),
            str(item.get("url", "")),
        ),
    )[0]


def catalog_link_matches_entry(
    link: dict[str, Any],
    entry: dict[str, Any],
) -> bool:
    """Match a rendered link to a catalogue entry without assuming a URL path.

    URL equality is preferred and tolerates trailing slashes.  The structured
    category/title pair is a safe fallback for catalogues that omit a URL from
    one representation of the same root route.
    """
    link_url = str(link.get("url", "")).strip().rstrip("/").casefold()
    entry_url = str(entry.get("url", "")).strip().rstrip("/").casefold()
    if link_url and entry_url and link_url == entry_url:
        # Application-library children legitimately share the root URL and use
        # ``search_hint`` to identify the requested catalogue item.  Treating
        # every shared URL as the root erased those children during navigation
        # ranking (for example Climate Policy Radar became Policy Catalogue).
        link_hint = str(link.get("search_hint", "") or "").strip().casefold()
        entry_hint = str(entry.get("search_hint", "") or "").strip().casefold()
        link_title = _normalize(str(link.get("title", "")))
        entry_title = _normalize(str(entry.get("title", "")))
        if (link_hint or entry_hint) and link_title != entry_title:
            return False
        return True

    link_category = _normalize(str(link.get("category", ""))).replace("_", " ")
    entry_category = _normalize(str(entry.get("category", ""))).replace("_", " ")
    link_title = _normalize(str(link.get("title", "")))
    entry_title = _normalize(str(entry.get("title", "")))
    return bool(
        link_category
        and link_category == entry_category
        and link_title
        and link_title == entry_title
    )


def infer_navigation_category(
    query: str,
    catalog: list[dict[str, Any]] | None,
) -> str:
    """Infer a requested root category from the live link catalogue.

    Only categories with an identifiable root route are eligible.  Category
    names and destinations therefore remain data-driven; this function merely
    compares their normalized tokens with the user's navigation request.
    """
    if not catalog:
        return ""
    query_tokens = _stemmed_tokens(query)
    if not query_tokens:
        return ""
    _is_nav, has_surface = _navigation_syntax(query)

    ranked: list[tuple[float, str]] = []
    categories = sorted({
        str(item.get("category", "")).strip()
        for item in catalog
        if str(item.get("category", "")).strip()
    })
    for category in categories:
        root = catalog_category_root(catalog, category)
        if not root:
            continue
        category_tokens = _category_tokens(category)
        title_tokens = _category_tokens(str(root.get("title", "")))
        destination_tokens = category_tokens | title_tokens
        overlap = destination_tokens & query_tokens
        if overlap and category_tokens.issubset(query_tokens):
            coverage = len(overlap) / max(len(destination_tokens), 1)
            ranked.append((coverage + len(overlap), category))
            continue

        # A user can name a detail destination without saying its parent
        # category (for example a named explorer). Derive the category from a
        # strong live item-identity match instead of requiring root vocabulary.
        detail_evidence = max(
            (
                _catalog_navigation_evidence(query, item, has_surface=has_surface)
                for item in catalog
                if not item.get("dead")
                and _normalize(str(item.get("category", ""))).replace("_", " ")
                == _normalize(category).replace("_", " ")
                and not catalog_link_matches_entry(item, root)
            ),
            default=0.0,
        )
        if detail_evidence >= 12.0:
            ranked.append((detail_evidence, category))

    if not ranked:
        return ""
    ranked.sort(key=lambda row: (-row[0], row[1]))
    return ranked[0][1]


def _navigation_syntax(query: str) -> tuple[bool, bool]:
    """Return ``(is_navigation, has_explicit_destination_surface)``.

    This deliberately recognizes only language shape.  A positive result is
    not enough to route a request: :func:`has_catalog_navigation_target` also
    requires destination evidence from the runtime catalogue.
    """
    q = _normalize(query)
    if not q:
        return False, False

    has_action = bool(re.search(
        r"\b(?:access|browse|find|give|go|link|navigate|open|read|send|show|take|view|visit)\b",
        q,
    ))
    has_unambiguous_navigation_action = bool(re.search(
        r"\b(?:access|browse|go\s+to|navigate|open|"
        r"link\s+(?:me\s+)?to|take\s+me\s+(?:directly\s+)?to|visit)\b",
        q,
    ))
    has_surface = bool(re.search(
        r"\b(?:application|catalog(?:ue)?|directory|docs?|documentation|library|link|page|portal|route|site|url|website|workspace)\b",
        q,
    ))
    indirect_request = bool(re.search(
        r"\b(?:can\s+i\s+find|where\s+(?:can|could|do|should|would)\s+i|"
        r"where\s+is|how\s+(?:can|do)\s+i)\b",
        q,
    ))
    return bool(
        has_unambiguous_navigation_action
        or (has_action and has_surface)
        or indirect_request
    ), has_surface


def _meaningful_stems(text: str, excluded: set[str]) -> set[str]:
    excluded_stems = {_stem_token(token) for token in excluded}
    return {
        stem
        for stem in _stemmed_tokens(text)
        if stem and stem not in excluded_stems
    }


def _catalog_navigation_evidence(
    query: str,
    item: dict[str, Any],
    *,
    has_surface: bool,
) -> float:
    """Score destination identity evidence for one live catalogue entry."""
    query_norm = _normalize(query)
    query_stems = _meaningful_stems(query, _NAVIGATION_FILLER_STEMS)
    if not query_stems:
        return 0.0

    title = str(item.get("title", ""))
    title_norm = _normalize(title)
    title_stems = _meaningful_stems(title, _NAVIGATION_FILLER_STEMS)
    # A literal catalogue title is the strongest identity evidence. Evaluate
    # it before unordered token coverage so concise detail names are not
    # eclipsed by a longer category-root title in the same request.
    if title_norm and title_stems and re.search(
        r"(?<!\w)" + re.escape(title_norm) + r"(?!\w)",
        query_norm,
    ):
        return 20.0 + len(title_stems)
    # Prefer a multi-concept destination whose complete title is expressed in
    # the request, even when the words are reordered or inflected.  This keeps
    # a generic category root from eclipsing a more precise child route solely
    # because the root's one-word title appears verbatim in the query.
    if len(title_stems) >= 2 and title_stems.issubset(query_stems):
        return 10.0 + len(title_stems) * 2.0
    title_overlap = title_stems & query_stems
    if (
        len(title_stems) >= 3
        and len(title_overlap) >= 3
        and len(title_overlap) / len(title_stems) >= 0.6
    ):
        # Users commonly omit one generic modifier from a long destination
        # title ("barriers and enablers analysis" vs "Barriers, Enablers &
        # Policy Analysis").  Three coherent title concepts are still strong
        # enough to beat a generic root that shares only "analysis".
        return 10.0 + len(title_overlap)

    best = 0.0
    identity_fields = [
        title,
        str(item.get("project", "")),
        str(item.get("workspace", "")).replace("-", " "),
        str(item.get("search_hint", "")),
        *(str(keyword) for keyword in item.get("keywords", []) if keyword),
    ]
    for index, field in enumerate(identity_fields):
        field_stems = _meaningful_stems(field, _NAVIGATION_FILLER_STEMS)
        if not field_stems:
            continue
        overlap = field_stems & query_stems
        if not overlap:
            continue
        if len(field_stems) >= 2 and field_stems.issubset(query_stems):
            # A complete multi-concept catalogue keyword can identify a detail
            # destination just as reliably as its title (for example when the
            # public title is long but the catalogue stores a concise alias).
            best = max(best, 10.0 + len(field_stems) * 2.0)
            continue
        coverage = len(overlap) / len(field_stems)
        # A complete one-word title is a valid destination when the
        # user explicitly asked for a page/catalogue.  Single keyword hits are
        # intentionally insufficient because they are often just broad topics.
        is_title = index == 0
        if len(overlap) == 1 and not (
            has_surface and is_title and coverage == 1.0
        ):
            continue
        if len(overlap) >= 2 or coverage >= 0.75:
            best = max(best, len(overlap) * 2.0 + coverage)
    return best


def _catalog_identity_stems(item: dict[str, Any]) -> set[str]:
    """Return subject stems that identify one catalogue entry."""
    identity = " ".join([
        str(item.get("title", "")),
        str(item.get("project", "")),
        str(item.get("workspace", "")).replace("-", " "),
        str(item.get("search_hint", "")),
        *(str(keyword) for keyword in item.get("keywords", []) if keyword),
    ])
    return _meaningful_stems(identity, _THEMATIC_GENERIC_STEMS)


def _distinctive_child_evidence(
    query: str,
    item: dict[str, Any],
    catalog: list[dict[str, Any]],
    category: str,
    root: dict[str, Any] | None,
) -> float:
    """Recognize a unique subject child when its category is explicit.

    Requests such as ``buildings results`` name the generic category plus one
    subject, while the catalogue child is titled ``Buildings Transformation``.
    Requiring that subject to identify exactly one live child within the named
    category keeps this useful shortcut deterministic and prevents an arbitrary
    workspace from winning on a common word.
    """
    if root and catalog_link_matches_entry(item, root):
        return 0.0

    query_subjects = _meaningful_stems(query, _THEMATIC_GENERIC_STEMS)
    # The category word establishes the parent destination; it is not the
    # distinctive child subject. Without removing it, a request such as
    # "open the reports explorer" could select the only child whose metadata
    # happens to repeat "report" instead of correctly retaining the root.
    query_subjects -= _category_tokens(category)
    if root:
        query_subjects -= _category_tokens(str(root.get("title", "")))
    overlap = query_subjects & _catalog_identity_stems(item)
    if not overlap:
        return 0.0

    normalized_category = _normalize(category).replace("_", " ")
    matching_children = 0
    for candidate in catalog:
        candidate_category = _normalize(str(candidate.get("category", ""))).replace("_", " ")
        if candidate.get("dead") or candidate_category != normalized_category:
            continue
        if root and catalog_link_matches_entry(candidate, root):
            continue
        # Use the complete subject signature expressed by this candidate. This
        # handles both concise requests ("buildings results") and a pair of
        # mutually reinforcing aliases ("AFOLU land use results").
        if overlap <= _catalog_identity_stems(candidate):
            matching_children += 1
            if matching_children > 1:
                return 0.0
    # Twelve points is the established strong-detail threshold below. It is
    # deliberately earned only by a unique child in an explicitly named
    # category, never by an isolated global keyword hit.
    return 12.0 + min(len(overlap) - 1, 3) if matching_children == 1 else 0.0


def _model_catalogue_intent(query: str) -> bool:
    """Whether the user is asking to enumerate the available model catalogue."""
    q = _normalize(query)
    return bool(
        re.search(r"\b(?:list|enumerate|browse|show)\s+(?:all\s+|available\s+)?models?\b", q)
        or re.search(
            r"\b(?:which|what|how\s+many)\s+models?\b[^.?!]*"
            r"\b(?:available|exist|included|listed|in\s+the\s+(?:data|database|catalog(?:ue)?))\b",
            q,
        )
    )


def _catalog_navigation_target(
    query: str,
    catalog: list[dict[str, Any]],
    category: str,
) -> tuple[dict[str, Any] | None, float]:
    """Return the best catalogue destination for an explicit navigation ask.

    The target is derived exclusively from live item metadata.  Restricting the
    candidates to the inferred category prevents an incidental word in another
    section from competing with the requested destination.
    """
    # The caller has already classified the request and supplied a catalogue
    # category. Re-running the full navigation gate here used to reject valid
    # phrasings such as "link me to ..." and "show me ... results" after the
    # manager had successfully routed them.
    _is_navigation, has_surface = _navigation_syntax(query)
    normalized_category = _normalize(category).replace("_", " ")
    if not normalized_category:
        return None, 0.0

    query_stems = _meaningful_stems(query, _NAVIGATION_FILLER_STEMS)
    raw_query_stems = _stemmed_tokens(query)
    root = catalog_category_root(catalog, category)
    category_identity = _category_tokens(category)
    if root:
        category_identity |= _category_tokens(str(root.get("title", "")))
    category_is_explicit = bool(category_identity and category_identity <= raw_query_stems)
    ranked: list[tuple[float, int, int, str, str, dict[str, Any]]] = []
    for item in catalog:
        item_category = _normalize(str(item.get("category", ""))).replace("_", " ")
        if item.get("dead") or item_category != normalized_category:
            continue
        evidence = _catalog_navigation_evidence(
            query,
            item,
            has_surface=has_surface,
        )
        if category_is_explicit:
            evidence = max(
                evidence,
                _distinctive_child_evidence(query, item, catalog, category, root),
            )
        if evidence < 4.0:
            continue
        # Prefer actual routes when evidence ties, then verified destinations;
        # both attributes come from the runtime catalogue.
        is_route = int(_normalize(str(item.get("item_type", ""))) == "route")
        is_verified = int(bool(item.get("verified_direct_url")))
        title_overlap = len(
            _meaningful_stems(str(item.get("title", "")), _NAVIGATION_FILLER_STEMS)
            & query_stems
        )
        ranked.append((
            evidence,
            title_overlap,
            is_route + is_verified,
            str(item.get("title", "")),
            str(item.get("url", "")),
            item,
        ))

    if not ranked:
        return None, 0.0
    # Category text is used to constrain the candidate set, not as evidence
    # that every item in that category is the requested destination. When a
    # detail entry has strong identity evidence, do not let the shared root
    # win merely because its generic title is also present in the request.
    if root:
        strongest_detail = max(
            (
                row[0] for row in ranked
                if not catalog_link_matches_entry(row[5], root)
            ),
            default=0.0,
        )
        if strongest_detail >= 12.0:
            ranked = [
                row for row in ranked
                if not catalog_link_matches_entry(row[5], root)
            ]
    ranked.sort(key=lambda row: (-row[0], -row[1], -row[2], row[3], row[4]))
    return ranked[0][5], ranked[0][0]


def has_catalog_navigation_target(
    query: str,
    catalog: list[dict[str, Any]] | None = None,
) -> bool:
    """Whether a navigation-shaped query names a runtime catalogue target.

    The function contains no destination names or URLs.  It first recognizes
    generic navigation syntax, then requires a sufficiently strong match to a
    non-dead entry's title/category/keywords/project metadata.  Callers can use
    this single predicate before routing to a catalogue-grounded answer.
    """
    is_navigation, has_surface = _navigation_syntax(query)
    if not is_navigation:
        return False
    if catalog is None:
        catalog = load_link_catalog(DEFAULT_LINK_CATALOG)
    if not catalog:
        return False
    return any(
        not item.get("dead")
        and _catalog_navigation_evidence(query, item, has_surface=has_surface) >= 4.0
        for item in catalog
    )


# Function words and request verbs carry no topical signal, but they still match
# inside long catalogue titles -- "Comparison of Fit-for-55 Policy and Cost
# Optimal Scenarios for the EU" matches "for" twice and out-scored the entry
# literally titled "GCAM" on a query naming GCAM. Excluded from keyword scoring
# only; navigation-intent detection keeps the full query.
_LINK_SCORING_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "and", "or", "the", "to", "for", "of", "in", "on", "at", "by",
    "with", "from", "me", "my", "i", "you", "it", "is", "are", "was", "were",
    "can", "could", "would", "should", "do", "does", "please", "show", "give",
    "find", "link", "linked", "links", "take", "go", "get", "see", "open",
    "visit", "want", "need", "looking", "look", "tell", "about", "where",
    "what", "which", "how", "there", "here", "this", "that",
})


def _query_terms(query: str, entities: dict[str, Any] | None = None, variable_intent: str = "") -> str:
    pieces = [query, variable_intent]
    for key in _LINK_QUERY_ENTITY_KEYS:
        value = (entities or {}).get(key)
        if isinstance(value, str):
            pieces.append(value)
        elif isinstance(value, (list, tuple, set)):
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
    generic_keyword_tokens = {
        "iam", "paris", "analysis", "data", "model", "models", "result", "results",
        "page", "pages", "application", "library",
    }
    if title_norm and title_norm in query_text and title_norm not in generic_titles:
        score += EXACT_TITLE_SCORE
        matched_keywords.append(title)

    for keyword in keywords:
        keyword_norm = _normalize(keyword)
        if not keyword_norm:
            continue
        keyword_tokens = _tokens(keyword)
        generic_keyword = bool(keyword_tokens and keyword_tokens <= generic_keyword_tokens)
        if keyword_norm in query_text and not generic_keyword:
            score += KEYWORD_PHRASE_SCORE
            matched_keywords.append(keyword)
        else:
            overlap = (keyword_tokens - generic_keyword_tokens) & query_tokens
            if overlap:
                score += min(len(overlap), 4) * KEYWORD_TOKEN_SCORE
                if len(matched_keywords) < 3:
                    matched_keywords.append(keyword)

    generic_terms = {"iam", "paris", "data", "model", "models", "result", "results"}
    title_tokens = _tokens(title) - generic_terms
    query_stems = {_stem_token(token) for token in query_tokens}
    title_stems = {_stem_token(token) for token in title_tokens}
    title_overlap = title_stems & query_stems
    if title_overlap:
        score += min(len(title_overlap), 4) * TITLE_TOKEN_SCORE

    if title_stems and len(title_stems) >= 2:
        coverage = len(title_overlap) / len(title_stems)
        if coverage >= 0.6:
            score += TITLE_COVERAGE_BONUS * coverage

    meaningful_overlap = (
        {_stem_token(token) for token in haystack_tokens - generic_terms}
        & query_stems
    )
    score += min(len(meaningful_overlap), 8) * MEANINGFUL_TOKEN_SCORE

    category = str(item.get("category", ""))
    score += category_boosts.get(category, 0.0)

    project = str(item.get("project", "")).lower()
    if "ndc" in query_text:
        if "ndc aspects" in project:
            score += NDC_PROJECT_BONUS
        elif category == "results" and project:
            score -= NDC_OTHER_RESULTS_PENALTY
    if any(term in query_text for term in ["fit for 55", "fit-for-55", "glasgow", "cost of capital", "behavioural", "net zero"]):
        if "iam compact" in project:
            score += IAM_COMPACT_PROJECT_BONUS

    if item.get("verified_direct_url"):
        score += VERIFIED_URL_BONUS
    if item.get("search_hint") and url.endswith("/application_library"):
        score -= SEARCH_HINT_PENALTY

    return score, matched_keywords


def _catalog_stem_is_distinctive(
    stem: str,
    catalog: list[dict[str, Any]] | None,
) -> bool:
    """Estimate thematic distinctiveness from the currently loaded catalogue."""
    if not catalog or len(catalog) < MIN_CATALOG_FOR_FREQUENCY_GATE:
        return True
    matching_documents = 0
    for candidate in catalog:
        identity = " ".join([
            str(candidate.get("title", "")),
            str(candidate.get("project", "")),
            str(candidate.get("workspace", "")).replace("-", " "),
            str(candidate.get("search_hint", "")),
            *(str(keyword) for keyword in candidate.get("keywords", []) if keyword),
        ])
        if stem in _meaningful_stems(identity, _THEMATIC_GENERIC_STEMS):
            matching_documents += 1
    return matching_documents / len(catalog) <= MAX_ALIAS_DOCUMENT_FREQUENCY


def _has_strong_thematic_evidence(
    item: dict[str, Any],
    query_text: str,
    catalog: list[dict[str, Any]] | None = None,
) -> bool:
    """Require subject-level evidence before suggesting a specific data link.

    Category boosts answer *where* a data response belongs, not *which* project
    or workspace it discusses.  This gate accepts an exact catalogue title, an
    exact multi-word keyword, or at least two meaningful overlapping concepts.
    All subject vocabulary comes from the runtime item rather than a list of
    known projects/topics.
    """
    query_norm = _normalize(query_text)
    # Scenario/variable identifiers often contain underscore-delimited
    # abbreviations.  Their fragments are retrieval keys, not evidence that the
    # user asked for a similarly named project page.  Strip the whole identifier
    # before thematic matching while leaving ordinary prose untouched.
    thematic_query = re.sub(
        r"\b[a-z0-9]+(?:_[a-z0-9]+)+\b",
        " ",
        query_norm,
        flags=re.IGNORECASE,
    )
    query_stems = _meaningful_stems(thematic_query, _THEMATIC_GENERIC_STEMS)
    if not query_stems:
        return False

    title = str(item.get("title", ""))
    title_norm = _normalize(title)
    title_stems = _meaningful_stems(title, _THEMATIC_GENERIC_STEMS)
    if title_norm and title_norm in query_norm and title_stems:
        return True

    title_overlap = title_stems & query_stems
    if title_stems and title_overlap:
        # A title is the catalogue item's strongest identity field.  Requiring
        # two concepts plus coherent coverage prevents a single ubiquitous
        # domain term from promoting a narrow two-word workspace. Exact title
        # phrases, including one-word roots, were already accepted above.
        if (
            len(title_overlap) >= 2
            and len(title_overlap) / len(title_stems) >= THEMATIC_FIELD_COVERAGE
            and any(_catalog_stem_is_distinctive(stem, catalog) for stem in title_overlap)
        ):
            return True

    fields = [
        (str(item.get("project", "")), False),
        (str(item.get("workspace", "")).replace("-", " "), False),
        (str(item.get("search_hint", "")), False),
        *(
            (str(keyword), True)
            for keyword in item.get("keywords", [])
            if keyword
        ),
    ]
    for field, is_keyword_alias in fields:
        field_norm = _normalize(field)
        field_stems = _meaningful_stems(field, _THEMATIC_GENERIC_STEMS)
        if not field_stems:
            continue
        overlap = field_stems & query_stems
        # A one-concept runtime keyword is an explicit catalogue alias, unlike
        # one incidental word from a multi-concept title/project. Accept its
        # morphological stem when the query names that topic directly.
        if (
            is_keyword_alias
            and len(field_stems) == 1
            and min(len(stem) for stem in field_stems) >= 5
            and field_stems <= query_stems
        ):
            # A one-word alias is only discriminative when it is uncommon in
            # the loaded catalogue. This prevents broad corpus vocabulary from
            # selecting one arbitrary project while retaining rare topics.
            if catalog and len(catalog) >= MIN_CATALOG_FOR_FREQUENCY_GATE:
                alias_stem = next(iter(field_stems))
                if not _catalog_stem_is_distinctive(alias_stem, catalog):
                    continue
            return True
        if len(field_stems) >= 2 and field_norm and field_norm in query_norm:
            if (
                is_keyword_alias
                and field_stems <= _BROAD_DATA_TOPIC_STEMS
                and not title_overlap
            ):
                continue
            return True
        # Two broad scope tokens (for example a region plus ``energy``) are not
        # enough to identify a long project keyword.  Require the overlap to
        # describe a coherent share of this *single* runtime metadata field.
        # Exact phrases above and concise two-concept aliases remain eligible,
        # while most of a longer destination identity must actually be present.
        if (
            len(overlap) >= 2
            and len(overlap) / len(field_stems) >= THEMATIC_FIELD_COVERAGE
            and any(_catalog_stem_is_distinctive(stem, catalog) for stem in overlap)
        ):
            return True
    # Do not combine isolated matches across unrelated metadata fields.  For
    # example, one broad variable token in a keyword plus one scenario acronym
    # in a project name is not evidence for that project's workspace.
    return False


def _data_export_intent(query: str) -> bool:
    """A request to download/export the data, with no named page or topic."""
    q = _normalize(query)
    return bool(
        re.search(r"\b(?:download|export)\b", q)
        and re.search(r"\b(?:data|dataset|datasets|results|timeseries|csv)\b", q)
    )


def _homepage_intent(query: str) -> bool:
    """A request for the site's home/landing page."""
    q = _normalize(query)
    return bool(re.search(r"\b(?:homepage|home\s*page|landing\s*page|main\s*page|main\s*site)\b", q))


def _project_publications_intent(query: str) -> bool:
    """A request for project publications, studies, or published outputs."""
    q = _normalize(query)
    return bool(
        re.search(r"\b(?:publication|publications|published\s+outputs?|project\s+outputs?)\b", q)
        and re.search(r"\b(?:project|research|result|results|output|outputs|publication|publications)\b", q)
    )


def _platform_guide_intent(query: str) -> bool:
    """A request for help using the IAM PARIS platform itself."""
    q = _normalize(query)
    return bool(
        re.search(r"\b(?:user\s+guide|platform\s+guide|tutorial|how\s+to\s+use)\b", q)
        and re.search(r"\b(?:platform|iam\s+paris|site|user\s+guide|tutorial)\b", q)
    )


def _site_home_entry(catalog: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The catalogue's home/landing route, identified by its title (data-driven)."""
    for item in catalog or []:
        if item.get("dead") or str(item.get("search_hint", "")).strip():
            continue
        if _normalize(str(item.get("title", ""))) == "home":
            return item
    return None


def suggest_links(
    query: str,
    catalog: list[dict[str, Any]] | None = None,
    *,
    agent_name: str = "",
    entities: dict[str, Any] | None = None,
    variable_intent: str = "",
    navigation_category: str = "",
    limit: int = 3,
) -> list[dict[str, Any]]:
    if catalog is None:
        catalog = load_link_catalog(DEFAULT_LINK_CATALOG)
    if not catalog or limit <= 0:
        return []

    # Numeric source identifiers are retained in the underlying catalogue but
    # are not meaningful model destinations.  Exclude only those model-detail
    # rows from presentation; the generic Models directory remains available.
    catalog = [
        item for item in catalog
        if not (
            _normalize(str(item.get("category", ""))) == "models"
            and _normalize(str(item.get("item_type", ""))) == "model"
            and (
                not is_presentable_model_label(
                    item.get("search_hint") or item.get("title")
                )
                or is_unlabelled_model_display(
                    item.get("search_hint") or item.get("title")
                )
            )
        )
    ]

    entities = entities or {}
    query_text = _normalize(_query_terms(query, entities, variable_intent))
    query_tokens = _tokens(query_text) - _LINK_SCORING_STOPWORDS
    boosts = _category_boosts(query, agent_name, entities)

    scored: list[tuple[float, dict[str, Any], list[str]]] = []
    for item in catalog:
        # Skip URLs confirmed broken by validate_links.py --mark-dead.
        if item.get("dead"):
            continue
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
    min_score = MIN_LINK_SCORE
    top_score = scored[0][0] if scored else 0.0

    selected: list[RelevantLink] = []
    seen: set[str] = set()
    for score, item, matched_keywords in scored:
        if score < min_score or score < top_score * RELATIVE_SCORE_FLOOR:
            continue
        # Category boosts alone (e.g. +10 for any results page on a data query)
        # must not qualify a link: require query-specific evidence beyond the
        # boost and the flat verified-URL/search-hint bonuses (±1 point).
        category_boost = boosts.get(str(item.get("category", "")), 0.0)
        entity_backed_category = bool(
            agent_name == "model_explanation"
            and str(item.get("category", "")) == "models"
            and ((entities or {}).get("model") or (entities or {}).get("models"))
        )
        if score - category_boost <= CATEGORY_EVIDENCE_MARGIN and not entity_backed_category:
            continue
        # A broad data route receives a category boost for every results entry.
        # Do not let that boost plus one incidental token (for example a region)
        # promote a thematically unrelated project/workspace.
        if agent_name in {"data_query", "data_plotting"} and not _has_strong_thematic_evidence(
            item,
            query_text,
            catalog,
        ):
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

    requested_models = []
    seen_model_names: set[str] = set()
    for value in [
        (entities or {}).get("model"),
        *((entities or {}).get("models") or []),
    ]:
        name = str(value or "").strip()
        normalized = _normalize(name)
        if (
            not name
            or not is_presentable_model_label(name)
            or is_unlabelled_model_display(name)
            or normalized in seen_model_names
        ):
            continue
        seen_model_names.add(normalized)
        requested_models.append(name)
    if not requested_models:
        selected = [
            link for link in selected
            if not (_normalize(link.category) == "models" and bool(link.search_hint))
        ]
    if requested_models:
        requested_norms = {_normalize(value) for value in requested_models}

        def _specific_model_entry(link: RelevantLink) -> bool:
            return _normalize(link.category) == "models" and bool(link.search_hint)

        def _matches_requested_model(link: RelevantLink) -> bool:
            candidates = {_normalize(link.title), _normalize(link.search_hint)} - {""}

            def _identifier_tokens(value: str) -> set[str]:
                return set(re.findall(r"[a-z]+|\d+(?:\.\d+)*", value))

            return any(
                requested == candidate
                or (
                    bool(_identifier_tokens(candidate))
                    and _identifier_tokens(candidate) <= _identifier_tokens(requested)
                )
                for requested in requested_norms
                for candidate in candidates
            )

        # A model-specific catalogue entry is valid only when its title/search
        # hint matches one of the structured model entities. Generic model tools
        # have no search hint and remain eligible.
        selected = [
            link for link in selected
            if not _specific_model_entry(link) or _matches_requested_model(link)
        ]
        matched_norms = {
            requested
            for requested in requested_norms
            if any(
                requested in {_normalize(link.title), _normalize(link.search_hint)}
                for link in selected
            )
        }
        if matched_norms != requested_norms:
            generic_models = next(
                (
                    item for item in catalog
                    if _normalize(str(item.get("category", ""))) == "models"
                    and _normalize(str(item.get("title", ""))) == "models"
                    and not str(item.get("search_hint", "") or "").strip()
                ),
                None,
            )
            if generic_models:
                search_hint = ", ".join(requested_models)
                selected = [link for link in selected if _normalize(link.title) != "models"]
                selected.insert(
                    0,
                    RelevantLink(
                        title=str(generic_models.get("title", "Models")),
                        url=str(generic_models.get("url", "")),
                        reason=f"Open the models catalogue and search for `{search_hint}`.",
                        confidence=1.0,
                        search_hint=search_hint,
                        category=str(generic_models.get("category", "models")),
                        verified_direct_url=bool(generic_models.get("verified_direct_url")),
                        fallback_instruction=f"Open the models catalogue and search for: {search_hint}",
                    ),
                )
                selected = selected[:limit]

    # A model-listing answer belongs with the model catalogue even though it is
    # served by the data-query agent. General data fallback otherwise supplied
    # only Results, which is less useful than the live Models directory.
    if not requested_models and _model_catalogue_intent(query):
        models_root = catalog_category_root(catalog, "models")
        if models_root:
            selected = [
                RelevantLink(
                    title=str(models_root.get("title") or "Models"),
                    url=str(models_root.get("url") or ""),
                    reason="Catalogued directory for the available IAM PARIS models.",
                    confidence=0.8,
                    search_hint=str(models_root.get("search_hint", "")),
                    category=str(models_root.get("category", "models")),
                    verified_direct_url=bool(models_root.get("verified_direct_url")),
                    fallback_instruction=str(models_root.get("fallback_instruction", "")),
                )
            ]

    # Explicit navigation is resolved independently from general relevance
    # scoring.  General scoring can overvalue incidental stopwords in detailed
    # workspaces; destination evidence instead compares the query with live
    # title/category/keyword metadata within the inferred category.
    normalized_navigation_category = _normalize(navigation_category).replace("_", " ")
    if normalized_navigation_category:
        target, evidence = _catalog_navigation_target(
            query,
            catalog,
            navigation_category,
        )
        if target is None:
            target = catalog_category_root(catalog, navigation_category)
            evidence = 0.0
        if target:
            display_target = target
            target_search_hint = str(target.get("search_hint", "")).strip()
            target_fallback = str(target.get("fallback_instruction", "")).strip()

            # Model rows whose URL is the shared directory are search targets,
            # not direct model pages. Present the verified Models hub and keep
            # the requested model as an explicit search instruction instead of
            # implying that the directory URL opens a GCAM-specific document.
            if (
                _normalize(str(target.get("category", ""))) == "models"
                and _normalize(str(target.get("item_type", ""))) == "model"
                and target_search_hint
                and not bool(target.get("verified_direct_url"))
            ):
                models_root = catalog_category_root(catalog, "models")
                if models_root:
                    display_target = models_root
                    if not target_fallback:
                        target_fallback = (
                            "Open the Models directory and search for: "
                            f"{target_search_hint}"
                        )

            target_link = RelevantLink(
                title=str(display_target.get("title", "")),
                url=str(display_target.get("url", "")),
                reason="Catalogued destination matching the navigation request.",
                confidence=round(min(0.55 + evidence / 20.0, 1.0), 3),
                search_hint=target_search_hint,
                category=str(display_target.get("category", "")),
                verified_direct_url=bool(display_target.get("verified_direct_url")),
                fallback_instruction=target_fallback,
            )
            # A navigation answer should not be diluted with links that merely
            # shared broad vocabulary.  If callers want multiple destinations,
            # they can issue separate catalogue targets explicitly.
            selected = [target_link]

    # A generic download/export request must resolve before weak relevance
    # matches (for example a scenario-metadata story) can displace the actual
    # results/data hub. A named navigation category still wins, allowing a
    # request for a particular workspace's downloadable data to stay specific.
    if _data_export_intent(query) and not normalized_navigation_category:
        export_root = catalog_category_root(catalog, "results")
        if export_root:
            selected = [
                RelevantLink(
                    title=str(export_root.get("title") or export_root.get("category") or "Results"),
                    url=str(export_root.get("url") or ""),
                    reason="The results section provides access to the underlying data.",
                    confidence=0.55,
                    search_hint=str(export_root.get("search_hint", "")),
                    category=str(export_root.get("category", "")),
                    verified_direct_url=bool(export_root.get("verified_direct_url")),
                    fallback_instruction=str(export_root.get("fallback_instruction", "")),
                )
            ]

    # The Results route is the verified catalogue hub for project studies and
    # outputs.  This keeps a publication-navigation question grounded instead
    # of letting a location extractor or free-form QA invent a destination.
    if _project_publications_intent(query) and not normalized_navigation_category:
        results_root = catalog_category_root(catalog, "results")
        if results_root:
            selected = [
                RelevantLink(
                    title=str(results_root.get("title") or "Results"),
                    url=str(results_root.get("url") or ""),
                    reason="The Results section lists project studies and research outputs.",
                    confidence=0.65,
                    search_hint=str(results_root.get("search_hint", "")),
                    category=str(results_root.get("category", "results")),
                    verified_direct_url=bool(results_root.get("verified_direct_url")),
                    fallback_instruction=str(results_root.get("fallback_instruction", "")),
                )
            ]

    # The live catalogue currently has no verified dedicated platform-guide
    # route.  Prefer the verified Contact page over an unrelated weak match,
    # and disclose why it is being offered in the link reason.
    if _platform_guide_intent(query):
        contact_root = catalog_category_root(catalog, "contact")
        if contact_root:
            selected = [
                RelevantLink(
                    title=str(contact_root.get("title") or "Contact"),
                    url=str(contact_root.get("url") or ""),
                    reason=(
                        "No dedicated user-guide destination is present in the loaded "
                        "catalogue; use the IAM PARIS contact page for platform help."
                    ),
                    confidence=0.55,
                    search_hint=str(contact_root.get("search_hint", "")),
                    category=str(contact_root.get("category", "contact")),
                    verified_direct_url=bool(contact_root.get("verified_direct_url")),
                    fallback_instruction=str(contact_root.get("fallback_instruction", "")),
                )
            ]

    # A "homepage" request names no topic, so relevance scoring misses the home
    # route (its title is "Home", not "homepage"). Resolve it from the catalogue
    # entry titled Home.
    if not selected and _homepage_intent(query):
        home = _site_home_entry(catalog)
        if home:
            selected = [
                RelevantLink(
                    title=str(home.get("title") or "Home"),
                    url=str(home.get("url") or ""),
                    reason="IAM PARIS home page.",
                    confidence=0.6,
                    search_hint=str(home.get("search_hint", "")),
                    category=str(home.get("category", "")),
                    verified_direct_url=bool(home.get("verified_direct_url")),
                    fallback_instruction=str(home.get("fallback_instruction", "")),
                )
            ]

    # Only pad with the generic results page for data-centric answers; a general
    # question with no real match is better served by no link than a wrong one.
    if not selected and agent_name in {"data_query", "data_plotting", "modelling_suggestions"}:
        fallback = catalog_category_root(catalog, "results")
        if fallback:
            selected.append(
                RelevantLink(
                    title=str(fallback.get("title") or fallback.get("category") or "Results"),
                    url=str(fallback.get("url") or ""),
                    reason="Catalogued results root for data follow-ups.",
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
