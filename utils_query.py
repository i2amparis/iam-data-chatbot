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
            if isinstance(item, dict):
                for name, details in item.items():
                    if isinstance(details, dict):
                        variable_names.append(name.strip())
    return sorted(variable_names)

def get_available_workspaces(ts: list) -> list:
    """Extract sorted workspace codes from timeseries records."""
    return sorted({t.get("workspace_code", "").strip() for t in ts if t and t.get("workspace_code")})

def match_variable_from_yaml(query: str, variable_dict: dict) -> dict:
    """
    Try to match a variable name to YAML definitions, including templated variables.
    Returns:
      - match_type: 'exact', 'ambiguous', 'fuzzy', 'templated', or None
      - matched_variable: best match if any
      - matches: list of possible matches for ambiguous case
    """
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Matching variable for query: '{query}'")

    query_lower = query.lower()
    exact_matches = []
    fuzzy_matches = []
    templated_matches = []

    # Build template map for all templated variables
    template_map = {}
    for file_data in variable_dict.values():
        for item in file_data:
            if isinstance(item, dict):
                for key, value in item.items():
                    if isinstance(value, list) and value and isinstance(value[0], dict):
                        # This is a template definition like Electricity Source
                        template_name = key
                        values = [list(d.keys())[0] for d in value]
                        template_map[template_name] = values
    logger.info(f"Template map: {template_map}")

    # Collect all possible variable names for fuzzy matching
    all_variable_names = []
    for file_data in variable_dict.values():
        for item in file_data:
            if isinstance(item, dict):
                for name, details in item.items():
                    if not isinstance(details, dict):
                        continue
                    all_variable_names.append(name)
                    # Add expanded templated names
                    if '{' in name and '}' in name:
                        # Find the template name inside {}
                        import re
                        template_match = re.search(r'\{([^}]+)\}', name)
                        if template_match:
                            template_name = template_match.group(1)
                            if template_name in template_map:
                                for value in template_map[template_name]:
                                    expanded_name = name.replace('{' + template_name + '}', value)
                                    all_variable_names.append(expanded_name)
    logger.info(f"All variable names for fuzzy: {all_variable_names[:10]}...")  # Log first 10

    for file_data in variable_dict.values():
        for item in file_data:
            if isinstance(item, dict):
                for name, details in item.items():
                    if not isinstance(details, dict):
                        continue
                    name_lower = name.lower()
                    description = details.get("description", "").lower()

                    # Check for exact match
                    if name_lower == query_lower:
                        exact_matches.append(name)
                        logger.info(f"Exact match: {name}")
                    # Check for templated variables like Capacity|Electricity|{Electricity Source}
                    elif '{' in name and '}' in name:
                        # Find the template name inside {}
                        template_match = re.search(r'\{([^}]+)\}', name)
                        if template_match:
                            template_name = template_match.group(1)
                            if template_name in template_map:
                                for value in template_map[template_name]:
                                    expanded_name = name.replace('{' + template_name + '}', value)
                                    expanded_lower = expanded_name.lower()
                                    if expanded_lower == query_lower:
                                        templated_matches.append(expanded_name)
                                        logger.info(f"Templated exact match: {expanded_name}")
                                    elif query_lower in expanded_lower or expanded_lower in query_lower:
                                        fuzzy_matches.append(expanded_name)
                                        logger.info(f"Templated fuzzy match: {expanded_name}")

    # Fuzzy matching using similarity and word presence
    if not exact_matches and not templated_matches:
        # First check for exact match with variable names (case-insensitive)
        exact_variable_matches = [n for n in all_variable_names if n.lower() == query_lower]
        if exact_variable_matches:
            fuzzy_matches = exact_variable_matches
            logger.info(f"Exact variable name match: {fuzzy_matches}")
        else:
            # Check if the query contains a full variable name (pipe-separated)
            pipe_count = query_lower.count('|')
            if pipe_count >= 2:  # Likely a full variable name
                # Look for exact match of the full variable name
                full_variable_matches = [n for n in all_variable_names if n.lower() == query_lower]
                if full_variable_matches:
                    fuzzy_matches = full_variable_matches
                    logger.info(f"Full variable name match: {fuzzy_matches}")
                else:
                    # Try partial matches but prefer longer ones
                    partial_matches = [n for n in all_variable_names if all(part.strip().lower() in n.lower() for part in query_lower.split('|') if part.strip())]
                    if partial_matches:
                        # Sort by how well they match (prefer exact substring matches)
                        partial_matches.sort(key=lambda x: len(x) if query_lower in x.lower() else 0, reverse=True)
                        fuzzy_matches = partial_matches[:3]
                        logger.info(f"Partial variable name matches: {fuzzy_matches}")
                    else:
                        # Fall back to substring matching with better prioritization
                        substring_matches = [n for n in all_variable_names if query_lower in n.lower()]
                        if substring_matches:
                            # Prioritize matches where the query is a significant portion of the variable name
                            substring_matches.sort(key=lambda x: (len(query_lower) / len(x), len(x)), reverse=True)
                            fuzzy_matches = substring_matches[:3]
                            logger.info(f"Substring matches (prioritized): {fuzzy_matches}")
            else:
                # For non-pipe queries, use substring matching with better prioritization
                substring_matches = [n for n in all_variable_names if query_lower in n.lower() or n.lower() in query_lower]
                if substring_matches:
                    # Prioritize longer matches and exact substrings
                    substring_matches.sort(key=lambda x: (query_lower in x.lower(), len(x)), reverse=True)
                    fuzzy_matches = substring_matches[:3]
                    logger.info(f"General substring matches (prioritized): {fuzzy_matches}")
                else:
                    # Fall back to fuzzy similarity matching
                    fuzzy_matched_names = get_close_matches(query_lower, [n.lower() for n in all_variable_names], n=5, cutoff=0.6)
                    fuzzy_matches = [n for n in all_variable_names if n.lower() in fuzzy_matched_names]
                    logger.info(f"Fuzzy matches from similarity: {fuzzy_matches}")

                    # Additional check: if significant non-stop words in query are in name or description
                    if not fuzzy_matches:
                        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'will', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'can', 'could', 'should', 'would', 'may', 'might', 'must', 'shall', 'increase', 'decrease', 'future', 'past', 'now', 'then', 'here', 'there', 'this', 'that', 'these', 'those', 'plot', 'show', 'display', 'graph', 'chart'}
                        query_words = set(re.findall(r'\b\w+\b', query_lower))
                        significant_words = [w for w in query_words if len(w) > 2 and w not in stop_words]  # Ignore short words and stop words
                        logger.info(f"Significant words: {significant_words}")
                        if significant_words:  # Only proceed if there are significant words
                            for file_data in variable_dict.values():
                                for item in file_data:
                                    if isinstance(item, dict):
                                        for name, details in item.items():
                                            if not isinstance(details, dict):
                                                continue
                                            name_lower = name.lower()
                                            description = details.get("description", "").lower()
                                            combined_text = name_lower + " " + description
                                            if any(word in combined_text for word in significant_words):
                                                fuzzy_matches.append(name)
                                                logger.info(f"Word presence match: {name}")
                                                break  # Take the first match
                            fuzzy_matches = fuzzy_matches[:1]  # Limit to one match


    result = {}
    if templated_matches:
        result = {"match_type": "templated", "matched_variable": templated_matches[0], "matches": []}
    elif len(exact_matches) == 1:
        result = {"match_type": "exact", "matched_variable": exact_matches[0], "matches": []}
    elif len(exact_matches) > 1:
        # Provide more context for ambiguous matches
        context_matches = []
        for var in exact_matches[:5]:  # Limit to 5 for readability
            # Try to find description from YAML
            description = ""
            for file_data in variable_dict.values():
                for item in file_data:
                    if isinstance(item, dict):
                        for name, details in item.items():
                            if name == var and isinstance(details, dict):
                                description = details.get("description", "")
                                break
                        if description:
                            break
                if description:
                    break
            context_matches.append({
                "variable": var,
                "description": description[:100] + "..." if len(description) > 100 else description
            })
        result = {"match_type": "ambiguous", "matched_variable": "", "matches": context_matches}
    elif fuzzy_matches:
        result = {"match_type": "fuzzy", "matched_variable": fuzzy_matches[0], "matches": []}
    else:
        result = {"match_type": None, "matched_variable": "", "matches": []}
    logger.info(f"Final match result: {result}")
    return result

def extract_region_from_query(query: str, region_dict: dict) -> str:
    """
    Extract region from query using region definitions.
    Returns the matched region name or empty string.
    """
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Extracting region for query: '{query}'")

    query_lower = query.lower()

    # First check for exact matches in region names
    for file_data in region_dict.values():
        for region_group in file_data:
            for region_name, region_info in region_group.items():
                if isinstance(region_info, dict):
                    # Check region name
                    if region_name.lower() in query_lower:
                        logger.info(f"Exact region match: {region_name}")
                        return region_name
                    # Check countries
                    countries = region_info.get("countries", [])
                    for country in countries:
                        if country.lower() in query_lower:
                            logger.info(f"Country match: {country} -> {region_name}")
                            return region_name
                elif isinstance(region_info, list):
                    # Handle list format
                    for item in region_info:
                        if isinstance(item, str) and item.lower() in query_lower:
                            logger.info(f"List item match: {item} -> {region_name}")
                            return region_name

    # Fuzzy matching for common misspellings or partial matches
    from difflib import get_close_matches
    all_regions = []
    for file_data in region_dict.values():
        for region_group in file_data:
            for region_name in region_group.keys():
                all_regions.append(region_name)
    logger.info(f"All regions for fuzzy: {all_regions[:10]}...")  # Log first 10

    matches = get_close_matches(query_lower, [r.lower() for r in all_regions], n=1, cutoff=0.6)
    if matches:
        # Find the original case region name
        for region in all_regions:
            if region.lower() == matches[0]:
                logger.info(f"Fuzzy region match: {region}")
                return region

    logger.info("No region match found")
    return ""

def extract_variable_and_region_from_query(query: str, variable_dict: dict, region_dict: dict) -> dict:
    """
    Extract both variable and region from a natural language query.
    Returns:
      - variable: matched variable info
      - region: matched region name
    """
    variable_match = match_variable_from_yaml(query, variable_dict)
    region_match = extract_region_from_query(query, region_dict)

    return {
        "variable": variable_match,
        "region": region_match
    }
