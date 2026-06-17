import re


PHRASE_SYNONYMS: dict[str, list[str]] = {
    "carbon dioxide": ["co2"],
    "greenhouse gas": ["ghg", "emissions"],
    "greenhouse gases": ["ghg", "emissions"],
    "solar pv": ["solar"],
    "photovoltaic": ["solar"],
    "photovoltaics": ["solar"],
    "wind power": ["wind"],
    "wind turbine": ["wind"],
    "wind turbines": ["wind"],
    "gross domestic product": ["gdp"],
    "time series": ["data", "query", "timeseries"],
    "data values": ["data", "query"],
}


TOKEN_SYNONYMS: dict[str, list[str]] = {
    "emission": ["emissions"],
    "emissions": ["emissions"],
    "electricity": ["electricity"],
    "power": ["electricity"],
    "pathway": ["scenario"],
    "pathways": ["scenario"],
    "country": ["region"],
    "countries": ["region"],
    "geography": ["region"],
    "geographies": ["region"],
    "location": ["region"],
    "locations": ["region"],
    "chart": ["plot"],
    "graph": ["plot"],
    "visualize": ["plot"],
    "visualise": ["plot"],
    "compare": ["comparison"],
    "versus": ["comparison"],
    "vs": ["comparison"],
    "data": ["data", "query"],
    "value": ["data", "query"],
    "values": ["data", "query"],
}


def normalize_query_text(text: str, expand_synonyms: bool = True) -> str:
    value = str(text or "").lower()
    value = value.replace("–", "-").replace("—", "-").replace("_", " ")
    value = re.sub(r"[-/]+", " ", value)
    value = re.sub(r"[^a-z0-9|.\s]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()

    if not expand_synonyms:
        return value

    additions: list[str] = []
    padded = f" {value} "
    for phrase, canonical_terms in PHRASE_SYNONYMS.items():
        if f" {phrase} " in padded:
            additions.extend(canonical_terms)

    for token in re.findall(r"[a-z0-9]+", value):
        additions.extend(TOKEN_SYNONYMS.get(token, []))
        if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
            additions.append(token[:-1])

    if additions:
        value = f"{value} {' '.join(additions)}"
    return re.sub(r"\s+", " ", value).strip()


def query_tokens(text: str) -> set[str]:
    normalized = normalize_query_text(text)
    return {token for token in re.findall(r"[a-z0-9]+", normalized) if token}
