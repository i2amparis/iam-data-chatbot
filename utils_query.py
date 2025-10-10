import re
from typing import List, Tuple, Dict, Any
from langchain.schema import Document
from difflib import get_close_matches

# --------------------------
# Extract Example Data
# --------------------------

def extract_examples_from_data(models, ts):
    """Extract examples of models, scenarios, and variables from the dataset."""
    model_names = list({m.get('modelName', '') for m in models if m and m.get('modelName')})
    scenario_names = list({m.get('scenario', '') for m in ts if m and m.get('scenario')})
    variable_names = list({m.get('variable', '') for m in ts if m and m.get('variable')})
    return {
        'scenarios': sorted(scenario_names)[:10],
        'models': sorted(model_names)[:10],
        'variables': sorted(variable_names)[:10]
    }

# --------------------------
# Match User Query to YAML Definitions
# --------------------------

def match_variable_to_definition(query: str, definition_docs: List[Document], top_k: int = 5):
    """Try to semantically match a variable name to known variable definitions (from langchain docs)."""
    query = query.lower()
    matches = []

    for doc in definition_docs:
        name = doc.metadata.get('name', '').lower()
        description = doc.page_content.lower()
        aliases = [a.lower() for a in doc.metadata.get('aliases', [])]

        if query in name or query in description or any(query in alias for alias in aliases):
            matches.append((doc.metadata.get('name', ''), doc.page_content.strip()))

    if not matches:
        all_names = [doc.metadata.get('name', '') for doc in definition_docs]
        fuzzy_match = get_close_matches(query, all_names, n=top_k, cutoff=0.5)
        for doc in definition_docs:
            if doc.metadata.get('name', '') in fuzzy_match:
                matches.append((doc.metadata.get('name', ''), doc.page_content.strip()))

    return matches[:top_k]

# --------------------------
# Fuzzy Variable Name Helper
# --------------------------

def find_closest_variable_name(user_query: str, variable_names: List[str]) -> str:
    """Use fuzzy matching to find the closest known variable name."""
    matches = get_close_matches(user_query, variable_names, n=1, cutoff=0.4)
    return matches[0] if matches else ""

# --------------------------
# Region Extraction
# --------------------------

def extract_region_from_query(query: str, region_list: List[str]) -> str:
    """Attempt to extract a known region name or code from the query."""
    query = query.lower()
    for region in region_list:
        if region.lower() in query:
            return region
    return ""

# --------------------------
# Display Examples
# --------------------------

def display_examples(examples: Dict[str, List[str]]) -> str:
    """Format example data into a markdown string."""
    out = "### Available Examples\n"
    out += "\n**Models:**\n" + ", ".join(examples["models"])
    out += "\n\n**Scenarios:**\n" + ", ".join(examples["scenarios"])
    out += "\n\n**Variables:**\n" + ", ".join(examples["variables"])
    return out

# --------------------------
# Getters from Raw Data
# --------------------------

def get_available_scenarios(ts: list) -> list:
    """Extract sorted scenario names from timeseries records."""
    return sorted({t.get("scenario", "").strip() for t in ts if t and t.get("scenario")})

def get_available_models(models: list) -> list:
    """Extract sorted model names from model records."""
    return sorted({m.get("modelName", "").strip() for m in models if m and m.get("modelName")})

def get_available_variables(ts: list) -> list:
    """Extract sorted variable names from timeseries records."""
    return sorted({r.get("variable", "").strip() for r in ts if r and "variable" in r})

# --------------------------
# Getters from YAML Files
# --------------------------

def get_available_variables_from_yaml(variable_dict: dict) -> list:
    """
    Extract sorted variable names from the loaded YAML variable dictionary.
    """
    variable_names = []
    for file_data in variable_dict.values():
        for item in file_data:
            name = item.get("name") or item.get("variable")
            if name:
                variable_names.append(name.strip())
    return sorted(variable_names)

def match_variable_from_yaml(query: str, variable_dict: dict) -> dict:
    """
    Try to match a variable name to YAML definitions.
    Returns:
      - match_type: 'exact', 'ambiguous', 'fuzzy', or None
      - matched_variable: best match if any
      - matches: list of possible matches for ambiguous case
    """
    query_lower = query.lower()
    exact_matches = []
    fuzzy_matches = []

    for file_data in variable_dict.values():
        for item in file_data:
            name = item.get("name") or item.get("variable")
            if not name:
                continue
            name_lower = name.lower()
            description = item.get("description", "").lower()

            if name_lower in query_lower:
                exact_matches.append(name)
            elif any(word in name_lower for word in query_lower.split()) or \
                 any(word in description for word in query_lower.split()):
                fuzzy_matches.append(name)

    if len(exact_matches) == 1:
        return {"match_type": "exact", "matched_variable": exact_matches[0], "matches": []}
    elif len(exact_matches) > 1:
        return {"match_type": "ambiguous", "matched_variable": "", "matches": exact_matches}
    elif fuzzy_matches:
        return {"match_type": "fuzzy", "matched_variable": fuzzy_matches[0], "matches": []}
    else:
        return {"match_type": None, "matched_variable": "", "matches": []}
