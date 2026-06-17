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
}


def normalize_model_name(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _allow_direct_full_name_match(name: str, query: str) -> bool:
    low = name.lower()
    ql = (query or "").lower().strip()
    if low == ql:
        return True
    return bool(re.search(r"[^a-z0-9]", low))


def extract_model_hint(query: str) -> str:
    ql = (query or "").lower()
    for prefix in ("model", "using", "with", "for"):
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
        if not raw:
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
            if family == "gcam":
                family_matches = norm == "gcam"
            else:
                family_matches = norm.startswith(family) or family in norm
            if family_matches:
                aliases.update(family_aliases)

        for alias in aliases:
            normalized_alias = alias.strip().lower()
            if normalized_alias:
                alias_map.setdefault(normalized_alias, set()).add(raw)
                alias_map.setdefault(normalize_model_name(normalized_alias), set()).add(raw)
    return alias_map


def resolve_model_candidates(query: str, model_names: list[str]) -> list[str]:
    names = sorted({str(name or "").strip() for name in model_names if str(name or "").strip()})
    if not names:
        return []

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

    if len(query_norm) < 4 or len(query_norm) > 24:
        return []
    norm_to_name = {normalize_model_name(name): name for name in names}
    fuzzy = get_close_matches(query_norm, list(norm_to_name.keys()), n=3, cutoff=0.84)
    return [norm_to_name[item] for item in fuzzy if item in norm_to_name]


def match_model_name(query: str, model_names: list[str]) -> str:
    names = sorted({str(name or "").strip() for name in model_names if str(name or "").strip()})
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
    if not candidates and not hint:
        for name in names:
            low = name.lower()
            if re.fullmatch(r"[a-z0-9\-_\.]+", low) and re.search(r"(?<!\w)" + re.escape(low) + r"(?!\w)", query_lower):
                return name
    return candidates[0] if candidates else ""
