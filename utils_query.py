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

    # Enhanced fuzzy matching with better solar/PV capacity recognition
    if not exact_matches and not templated_matches:
        # Special handling for solar/PV capacity queries
        solar_keywords = ['pv', 'solar', 'photovoltaic', 'capacity']
        if any(keyword in query_lower for keyword in solar_keywords):
            # Prioritize solar-related variables
            solar_variables = [n for n in all_variable_names if any(solar_term in n.lower() for solar_term in ['solar', 'pv', 'photovoltaic', 'capacity'])]
            if solar_variables:
                # Look for exact solar capacity matches first
                solar_capacity_matches = [n for n in solar_variables if 'capacity' in n.lower() and ('solar' in n.lower() or 'pv' in n.lower())]
                if solar_capacity_matches:
                    # Prioritize Capacity|Electricity|Solar|Utility specifically
                    utility_matches = [n for n in solar_capacity_matches if 'utility' in n.lower()]
                    if utility_matches:
                        fuzzy_matches = utility_matches[:1]  # Take the first utility match
                        logger.info(f"Solar utility capacity prioritized match: {fuzzy_matches}")
                    else:
                        fuzzy_matches = solar_capacity_matches[:3]
                        logger.info(f"Solar capacity prioritized matches: {fuzzy_matches}")
                else:
                    fuzzy_matches = solar_variables[:3]
                    logger.info(f"Solar-related matches: {fuzzy_matches}")

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

    # Enhanced region extraction with better country recognition
    # First check for exact matches in region names and countries
    for file_data in region_dict.values():
        for region_group in file_data:
            for region_name, region_info in region_group.items():
                if isinstance(region_info, dict):
                    # Check region name
                    if region_name.lower() in query_lower:
                        logger.info(f"Exact region match: {region_name}")
                        return region_name
                    # Check countries with better matching
                    countries = region_info.get("countries", [])
                    for country in countries:
                        # Check for exact country match or common variations
                        country_lower = country.lower()
                        if country_lower in query_lower or query_lower in country_lower:
                            logger.info(f"Country match: {country} -> {region_name}")
                            return region_name
                        # Special handling for Greece (common in energy systems)
                        if 'greece' in query_lower and country_lower == 'greece':
                            logger.info(f"Greece country match -> {region_name}")
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

def build_semantic_index(variable_dict: dict) -> dict:
    """
    Build a semantic index of all variables from YAML definitions.
    Returns a dict mapping semantic keywords to variable names.
    """
    semantic_index = {}

    for file_data in variable_dict.values():
        for item in file_data:
            if isinstance(item, dict):
                for var_name, var_info in item.items():
                    if not isinstance(var_info, dict):
                        continue

                    # Extract semantic information
                    description = (var_info.get('description') or '').lower()
                    unit = (var_info.get('unit') or '').lower()

                    # Create semantic keywords from description and variable name
                    var_words = set(var_name.lower().replace('|', ' ').replace('{', ' ').replace('}', ' ').split())
                    desc_words = set(description.split())
                    unit_words = set(unit.replace('/', ' ').replace('(', ' ').replace(')', ' ').split())

                    # Combine all semantic keywords
                    semantic_keywords = var_words | desc_words | unit_words

                    # Add common synonyms and related terms
                    enhanced_keywords = set()
                    for keyword in semantic_keywords:
                        enhanced_keywords.add(keyword)
                        # Add related terms
                        if keyword in ['investment', 'investments']:
                            enhanced_keywords.update(['funding', 'capital', 'spending', 'invest'])
                        elif keyword in ['capacity']:
                            enhanced_keywords.update(['installed', 'generation', 'power'])
                        elif keyword in ['emission', 'emissions']:
                            enhanced_keywords.update(['co2', 'carbon', 'greenhouse', 'gas'])
                        elif keyword in ['energy']:
                            enhanced_keywords.update(['power', 'electricity', 'electric'])
                        elif keyword in ['future', 'annual', 'yearly']:
                            enhanced_keywords.update(['long-term', 'projection', 'forecast'])

                    # Store in index
                    for keyword in enhanced_keywords:
                        if keyword not in semantic_index:
                            semantic_index[keyword] = []
                        semantic_index[keyword].append({
                            'variable': var_name,
                            'description': description,
                            'unit': unit,
                            'is_template': '{' in var_name and '}' in var_name
                        })

    return semantic_index


def resolve_natural_language_variable_universal(query: str, variable_dict: dict) -> str:
    """
    Universal resolver that works for all variables in YAML definitions.
    Uses semantic indexing and scoring to find the best match.
    """
    query_lower = query.lower()

    # Build semantic index (cache this for performance)
    if not hasattr(resolve_natural_language_variable_universal, '_semantic_index'):
        resolve_natural_language_variable_universal._semantic_index = build_semantic_index(variable_dict)

    semantic_index = resolve_natural_language_variable_universal._semantic_index

    # Extract significant words from query
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'plot', 'show', 'graph', 'display', 'visualize', 'give', 'me', 'a', 'please'}
    query_words = set(re.findall(r'\b\w+\b', query_lower))
    significant_words = [w for w in query_words if len(w) > 2 and w not in stop_words and w is not None]

    # Score all variables
    variable_scores = {}

    for word in significant_words:
        if word in semantic_index:
            for var_info in semantic_index[word]:
                var_name = var_info['variable']
                if var_name not in variable_scores:
                    variable_scores[var_name] = {
                        'score': 0,
                        'info': var_info,
                        'matched_words': []
                    }

                # Enhanced scoring with priority for investment variables
                if 'investment' in var_name.lower() and 'investment' in word:
                    variable_scores[var_name]['score'] += 5  # High priority for investment matches
                elif word in var_name.lower():
                    variable_scores[var_name]['score'] += 3  # Variable name match
                elif word in var_info['description']:
                    variable_scores[var_name]['score'] += 2  # Description match
                else:
                    variable_scores[var_name]['score'] += 1  # Related term match

                # Bonus points for multi-word matches
                if len(significant_words) > 1:
                    # Check if multiple keywords match this variable
                    keyword_matches = sum(1 for kw in significant_words if kw in var_name.lower() or kw in var_info['description'])
                    if keyword_matches > 1:
                        variable_scores[var_name]['score'] += keyword_matches

                # Special handling for investment queries
                if 'investment' in significant_words and 'investment' in var_name.lower():
                    if 'future' in significant_words and ('annual' in var_name.lower() or 'yearly' in var_name.lower()):
                        variable_scores[var_name]['score'] += 3  # Boost annual investments for "future" queries
                    if 'biomass' in significant_words and 'biomass' in var_name.lower():
                        variable_scores[var_name]['score'] += 4  # Boost biomass investments
                    if 'solar' in significant_words and 'solar' in var_name.lower():
                        variable_scores[var_name]['score'] += 4  # Boost solar investments
                    if 'wind' in significant_words and 'wind' in var_name.lower():
                        variable_scores[var_name]['score'] += 4  # Boost wind investments

                    # Boost any investment variable for investment queries
                    variable_scores[var_name]['score'] += 2  # General investment boost

                variable_scores[var_name]['matched_words'].append(word)

    if not variable_scores:
        return None

    # Find best match
    best_variable = max(variable_scores.items(), key=lambda x: x[1]['score'])
    best_var_name = best_variable[0]
    best_score = best_variable[1]['score']

    # Minimum confidence threshold - lower for investment queries
    min_threshold = 1 if any(word in significant_words for word in ['investment', 'investments', 'invest']) else 2
    if best_score < min_threshold:
        # For investment queries, try to find any investment variable as fallback
        if any(word in significant_words for word in ['investment', 'investments', 'invest']):
            investment_vars = [name for name in variable_scores.keys() if 'investment' in name.lower()]
            if investment_vars:
                # Sort by score and pick the highest
                best_investment = max(investment_vars, key=lambda x: variable_scores[x]['score'])
                return best_investment
        return None

    return best_variable

    # Resolve templates if needed
    if best_variable[1]['info']['is_template']:
        best_var_name = resolve_template(best_var_name, significant_words, variable_dict)

    return best_var_name


def resolve_template(template_var: str, query_words: list, variable_dict: dict) -> str:
    """
    Resolve templated variables like Capacity|Electricity|{Electricity Source}
    """
    if '{' not in template_var or '}' not in template_var:
        return template_var

    # Extract template name
    import re
    template_match = re.search(r'\{([^}]+)\}', template_var)
    if not template_match:
        return template_var

    template_name = template_match.group(1)

    # Find possible values for this template
    template_values = find_template_values(template_name, variable_dict)

    # Match query words to template values
    for word in query_words:
        if word is None:
            continue
        for value_info in template_values:
            value_name = value_info['name']
            if word.lower() == value_name.lower() or word.lower() in value_info.get('aliases', []):
                return template_var.replace(f'{{{template_name}}}', value_name)

    # Return first available value as default
    if template_values:
        return template_var.replace(f'{{{template_name}}}', template_values[0]['name'])

    return template_var


def find_template_values(template_name: str, variable_dict: dict) -> list:
    """
    Find all possible values for a template like 'Electricity Source'
    """
    values = []

    for file_data in variable_dict.values():
        for item in file_data:
            if isinstance(item, dict):
                for var_name, var_info in item.items():
                    if var_name == template_name and isinstance(var_info, list):
                        for value_item in var_info:
                            if isinstance(value_item, dict):
                                for value_name, value_details in value_item.items():
                                    values.append({
                                        'name': value_name,
                                        'aliases': value_details.get('aliases', []) if isinstance(value_details, dict) else []
                                    })

    return values


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