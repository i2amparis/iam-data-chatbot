import re
from difflib import get_close_matches


CURATED_MODEL_ALIASES: dict[str, set[str]] = {
    "gcam": {"gcam"},
    "gcampr": {"gcam-pr", "gcam pr", "gcampr", "gcam-princeton", "gcam princeton"},
    "prometheus": {"prometheus"},
    "leap": {"leap"},
    "remind": {"remind"},
    "message": {"message", "messageix", "message-ix", "message ix"},
    "witch": {"witch"},
    # GEM-E3 and GEMINI-E3 are different models.  Their punctuation-normalised
    # names are similar enough that fuzzy matching used to merge them into one
    # runtime family, silently serving GEMINI-E3 rows for an exact GEM-E3
    # request.  Keep their aliases intentionally disjoint.
    "geme3": {"gem-e3", "gem e3", "geme3"},
    "geminie3": {"gemini-e3", "gemini e3", "geminie3", "gemini_e3"},
    "poles": {"poles"},
}


UNLABELLED_MODEL_LABEL = "Unlabelled source model"


def normalize_model_name(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def is_presentable_model_label(value: object) -> bool:
    """Return whether a catalogue value is suitable to show as a model name.

    Some source records contain numeric identifiers in ``modelName`` (for
    example ``42``).  They remain valid source/provenance values, but are not
    meaningful model labels for discovery, suggestions, or name resolution.
    Keep this deliberately narrow so legitimate names containing digits (such
    as ``GCAM 7.0`` or ``GEM-E3``) are unaffected.
    """
    label = str(value or "").strip()
    return bool(label) and not bool(re.fullmatch(r"[+-]?\d+(?:[.,]\d+)*", label))


def display_model_label(value: object) -> str:
    """Return a safe user-facing label without exposing numeric source IDs."""
    label = str(value or "").strip()
    if label and not is_presentable_model_label(label):
        return UNLABELLED_MODEL_LABEL
    return label


def is_unlabelled_model_display(value: object) -> bool:
    """Whether *value* is the neutral presentation label for source-only IDs."""
    return str(value or "").strip().casefold() == UNLABELLED_MODEL_LABEL.casefold()


def requests_exact_gem_e3(text: str) -> bool:
    """Detect GEM-E3 without matching the distinct GEMINI-E3 name."""
    return bool(re.search(r"(?<![\w-])gem(?:[\s_-]*)e3(?![\w-])", str(text or ""), re.IGNORECASE))


def model_family_key(text: str) -> str:
    """Canonical identity for versioned catalogue labels and their base name.

    Only a trailing numeric version is removed; meaningful qualifiers that
    distinguish related models remain part of the key.
    """
    value = re.sub(
        r"(?:\s+v?|[._-]+v)\d+(?:[._-]\d+)*(?:[a-z])?\s*$",
        "",
        str(text or "").strip(),
        flags=re.IGNORECASE,
    ).strip()
    return normalize_model_name(value or text)


def curated_model_family_key(text: str) -> str:
    """Return a curated family identity while retaining meaningful qualifiers."""
    base = model_family_key(text)
    for family, aliases in CURATED_MODEL_ALIASES.items():
        alias_keys = {model_family_key(alias) for alias in aliases}
        if base == family or base in alias_keys:
            return family
    return base


def _allow_direct_full_name_match(name: str, query: str) -> bool:
    low = name.lower()
    ql = (query or "").lower().strip()
    if low == ql:
        return True
    return bool(re.search(r"[^a-z0-9]", low))


def extract_model_hint(query: str) -> str:
    ql = (query or "").lower()
    # Model-specific prepositions come first. "for" is ambiguous (usually a
    # region, e.g. "for China") so it is tried last, only when no stronger
    # model idiom is present. This keeps "... for China from PROMETHEUS" from
    # resolving the region token as the model.
    for prefix in ("model", "using", "with", "from", "for"):
        match = re.search(
            r"\b" + prefix + r"\s+([a-z0-9][a-z0-9\-\._ ]{1,50})",
            ql,
        )
        if match:
            raw = match.group(1).strip()
            return re.split(r"\b(?:under|scenario|region|workspace|from|between|during|in|with|using|model|for)\b", raw)[0].strip()
    return ""


def build_model_alias_map(model_names: list[str]) -> dict[str, set[str]]:
    alias_map: dict[str, set[str]] = {}
    for name in model_names:
        raw = str(name or "").strip()
        if not is_presentable_model_label(raw):
            continue
        low = raw.lower()
        norm = normalize_model_name(raw)
        words = [word for word in re.split(r"[^a-z0-9]+", low) if word]
        aliases = {low, norm}

        if words:
            if not (words[0] == "gcam" and norm.startswith("gcampr")):
                aliases.add(words[0])
            if len(words) >= 2:
                aliases.add(" ".join(words[:2]))
                aliases.add("".join(words[:2]))
            alpha_prefix = re.match(r"[a-z]+", words[0] or "")
            if alpha_prefix and len(alpha_prefix.group(0)) >= 4:
                aliases.add(alpha_prefix.group(0))

        for family, family_aliases in CURATED_MODEL_ALIASES.items():
            family_keys = {
                model_family_key(alias) for alias in family_aliases
            } | {family}
            if family == "gcam":
                family_matches = norm == "gcam"
            else:
                record_family = model_family_key(raw)
                family_matches = any(
                    record_family == key or record_family.startswith(key)
                    for key in family_keys
                    if len(key) >= 4
                )
            if family_matches:
                aliases.update(family_aliases)

        for alias in aliases:
            normalized_alias = alias.strip().lower()
            if normalized_alias:
                alias_map.setdefault(normalized_alias, set()).add(raw)
                alias_map.setdefault(normalize_model_name(normalized_alias), set()).add(raw)
    return alias_map


def resolve_model_candidates(query: str, model_names: list[str]) -> list[str]:
    names = sorted({
        str(name or "").strip()
        for name in model_names
        if is_presentable_model_label(name)
    })
    if not names:
        return []

    # An exact GEM-E3 mention is an identity constraint, not a fuzzy hint.  If
    # that model has no runtime rows, return no candidate rather than borrowing
    # the similarly named GEMINI-E3 family.
    exact_gem_e3_request = requests_exact_gem_e3(query)
    runtime_has_gem_e3 = any(
        curated_model_family_key(name) == "geme3" for name in names
    )
    if exact_gem_e3_request:
        # Remove only the confusable family. Other explicitly named models in a
        # comparison (for example ``GEM-E3 vs GCAM``) must remain resolvable.
        names = [
            name for name in names
            if curated_model_family_key(name) != "geminie3"
        ]

    query_lower = (query or "").lower()
    query_norm = normalize_model_name(query)

    exact_hits = [
        name
        for name in names
        if _allow_direct_full_name_match(name, query_lower)
        and re.search(r"(?<![\w-])" + re.escape(name.lower()) + r"(?![\w-])", query_lower)
        and not (name.lower() == "gcam" and re.search(r"\bgcam\s*[- ]?\s*pr\b", query_lower))
    ]
    if exact_hits:
        return exact_hits

    tokens = [token for token in re.split(r"[^a-z0-9]+", query_lower) if token]
    spans = set(tokens)
    for index in range(len(tokens)):
        if index + 1 < len(tokens):
            spans.add(tokens[index] + " " + tokens[index + 1])
            spans.add(tokens[index] + tokens[index + 1])
        if index + 2 < len(tokens):
            spans.add(tokens[index] + " " + tokens[index + 1] + " " + tokens[index + 2])
            spans.add(tokens[index] + tokens[index + 1] + tokens[index + 2])
    if query_norm:
        spans.add(query_norm)

    alias_map = build_model_alias_map(names)
    alias_hits: dict[str, int] = {}
    for span in spans:
        for name in alias_map.get(span, set()):
            alias_hits[name] = max(alias_hits.get(name, 0), len(normalize_model_name(span)))
        normalized_span = normalize_model_name(span)
        for name in alias_map.get(normalized_span, set()):
            alias_hits[name] = max(alias_hits.get(name, 0), len(normalized_span))
    if alias_hits:
        def rank(name: str) -> tuple[int, int, str]:
            low = name.lower()
            exact_alias = 1 if low in spans or normalize_model_name(name) in spans else 0
            return alias_hits.get(name, 0), exact_alias, -len(name), name

        return sorted(alias_hits, key=rank, reverse=True)

    if exact_gem_e3_request and not runtime_has_gem_e3:
        return []

    if len(query_norm) < 4 or len(query_norm) > 24:
        return []
    norm_to_name = {normalize_model_name(name): name for name in names}
    fuzzy = get_close_matches(query_norm, list(norm_to_name.keys()), n=3, cutoff=0.84)
    return [norm_to_name[item] for item in fuzzy if item in norm_to_name]


def match_model_name(query: str, model_names: list[str]) -> str:
    names = sorted({
        str(name or "").strip()
        for name in model_names
        if is_presentable_model_label(name)
    })
    query_lower = (query or "").lower()

    direct = [
        name
        for name in names
        if _allow_direct_full_name_match(name, query_lower)
        and re.search(r"(?<![\w-])" + re.escape(name.lower()) + r"(?![\w-])", query_lower)
    ]
    if direct:
        return direct[0]

    hint = extract_model_hint(query)
    candidates = resolve_model_candidates(hint, names) if hint else []
    if not candidates and len(query_lower.split()) <= 3:
        # No preposition-anchored hint resolved and the input is a bare model
        # mention (e.g. the LLM's "IMAGE"). Try alias/family matching so it
        # resolves to its versioned catalogue label ("IMAGE" -> "IMAGE 3.2")
        # instead of falling through to a loose fuzzy match that can collide
        # with an unrelated name (IMAGE -> MANAGE). Kept short-input-only so a
        # full sentence cannot promote a weak first-word alias to a model.
        candidates = resolve_model_candidates(query, names)
    if not candidates and not hint:
        for name in names:
            low = name.lower()
            if re.fullmatch(r"[a-z0-9\-_\.]+", low) and re.search(r"(?<!\w)" + re.escape(low) + r"(?!\w)", query_lower):
                return name
    return candidates[0] if candidates else ""


def resolve_model_family_members(query_or_name: str, model_names: list[str]) -> list[str]:
    """Resolve a model mention to every runtime label in that model family.

    A base request such as ``GEM-E3`` or ``POLES`` includes versioned/runtime
    aliases, while a fully specified version remains exact.  Related models
    with meaningful qualifiers (notably GCAM versus GCAM-PR) stay distinct.
    """
    names = sorted({
        str(name or "").strip()
        for name in model_names
        if is_presentable_model_label(name)
    })
    if not names:
        return []
    query = str(query_or_name or "").strip()
    query_lower = query.casefold()
    explicit_versioned = [
        name for name in names
        if re.search(r"\d+(?:\.\d+)+\s*$", name)
        and re.search(
            r"(?<![\w-])" + re.escape(name.casefold()) + r"(?![\w-])",
            query_lower,
        )
    ]
    if explicit_versioned:
        return explicit_versioned

    candidates = resolve_model_candidates(query, names)
    if requests_exact_gem_e3(query):
        target_family = "geme3"
    elif candidates:
        target_family = curated_model_family_key(candidates[0])
    else:
        query_norm = normalize_model_name(query)
        target_family = ""
        for family, aliases in CURATED_MODEL_ALIASES.items():
            alias_keys = {normalize_model_name(alias) for alias in aliases}
            if any(alias and alias in query_norm for alias in alias_keys):
                target_family = family
                break
        if not target_family:
            return []

    return [
        name for name in names
        if curated_model_family_key(name) == target_family
    ]
