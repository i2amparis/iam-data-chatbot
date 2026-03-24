import re
import logging
from typing import List, Tuple, Dict, Any
from langchain.schema import Document
from difflib import get_close_matches

logger = logging.getLogger(__name__)

# --------------------------
# Extract Example Data
# --------------------------

def extract_examples_from_data(models, ts):
    """Extract examples of models, scenarios, and variables from the dataset."""
    model_names = list({m.get('modelName', '') for m in models if m and m.get('modelName')})
    scenario_names = list({m.get('scenario', '') for m in ts if m and m.get('scenario')})
    variable_names = list({str(m.get('variable', '')) for m in ts if m and m.get('variable')})
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
    return sorted({str(r.get("variable", "")).strip() for r in ts if r and "variable" in r})


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
    logger.debug(f"Matching variable for query: '{query}'")

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
    logger.debug(f"Template map: {template_map}")

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
                                    logger.debug(f"Added expanded template: {expanded_name}")
    logger.debug(f"All variable names for fuzzy: {all_variable_names[:10]}...")  # Log first 10

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
                        logger.debug(f"Exact match: {name}")
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
                                        logger.debug(f"Templated exact match: {expanded_name}")
                                    elif query_lower in expanded_lower or expanded_lower in query_lower:
                                        fuzzy_matches.append(expanded_name)
                                        logger.debug(f"Templated fuzzy match: {expanded_name}")

    # Enhanced fuzzy matching with better technology recognition
    if not exact_matches and not templated_matches:
        # Special handling for technology-specific queries
        tech_keywords = {
            'solar': ['pv', 'solar', 'photovoltaic'],
            'wind': ['wind', 'onshore', 'offshore'],
            'nuclear': ['nuclear'],
            'hydro': ['hydro', 'hydropower'],
            'biomass': ['biomass'],
            'gas': ['gas', 'natural gas'],
            'coal': ['coal'],
            'ccs': ['ccs', 'carbon capture'],
            'battery': ['battery', 'storage'],
            'hydrogen': ['hydrogen', 'electrolysis']
        }

        for tech, keywords in tech_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                # Find variables related to this technology
                tech_variables = [n for n in all_variable_names if any(kw in n.lower() for kw in keywords)]
                if tech_variables:
                    # Prioritize capacity variables for this technology
                    capacity_matches = [n for n in tech_variables if 'capacity' in n.lower()]
                    if capacity_matches:
                        fuzzy_matches = capacity_matches[:3]
                        logger.debug(f"{tech.title()} capacity prioritized matches: {fuzzy_matches}")
                    else:
                        fuzzy_matches = tech_variables[:3]
                        logger.debug(f"{tech.title()}-related matches: {fuzzy_matches}")
                    break  # Stop after finding first technology match

        # First check for exact match with variable names (case-insensitive)
        exact_variable_matches = [n for n in all_variable_names if n.lower() == query_lower]
        if exact_variable_matches:
            fuzzy_matches = exact_variable_matches
            logger.debug(f"Exact variable name match: {fuzzy_matches}")
        else:
            # Check if the query contains a full variable name (pipe-separated)
            pipe_count = query_lower.count('|')
            if pipe_count >= 2:  # Likely a full variable name
                # Look for exact match of the full variable name
                full_variable_matches = [n for n in all_variable_names if n.lower() == query_lower]
                if full_variable_matches:
                    fuzzy_matches = full_variable_matches
                    logger.debug(f"Full variable name match: {fuzzy_matches}")
                else:
                    # Try partial matches but prefer longer ones
                    partial_matches = [n for n in all_variable_names if all(part.strip().lower() in n.lower() for part in query_lower.split('|') if part.strip())]
                    if partial_matches:
                        # Sort by how well they match (prefer exact substring matches)
                        partial_matches.sort(key=lambda x: len(x) if query_lower in x.lower() else 0, reverse=True)
                        fuzzy_matches = partial_matches[:3]
                        logger.debug(f"Partial variable name matches: {fuzzy_matches}")
                    else:
                        # Fall back to substring matching with better prioritization
                        substring_matches = [n for n in all_variable_names if query_lower in n.lower()]
                        if substring_matches:
                            # Prioritize matches where the query is a significant portion of the variable name
                            substring_matches.sort(key=lambda x: (len(query_lower) / len(x), len(x)), reverse=True)
                            fuzzy_matches = substring_matches[:3]
                            logger.debug(f"Substring matches (prioritized): {fuzzy_matches}")
            else:
                # For non-pipe queries, use substring matching with better prioritization
                substring_matches = [n for n in all_variable_names if query_lower in n.lower() or n.lower() in query_lower]
                if substring_matches:
                    # Prioritize longer matches and exact substrings
                    substring_matches.sort(key=lambda x: (query_lower in x.lower(), len(x)), reverse=True)
                    fuzzy_matches = substring_matches[:3]
                    logger.debug(f"General substring matches (prioritized): {fuzzy_matches}")
                else:
                    # Fall back to fuzzy similarity matching
                    fuzzy_matched_names = get_close_matches(query_lower, [n.lower() for n in all_variable_names], n=5, cutoff=0.6)
                    fuzzy_matches = [n for n in all_variable_names if n.lower() in fuzzy_matched_names]
                    logger.debug(f"Fuzzy matches from similarity: {fuzzy_matches}")

                    # Additional check: if significant non-stop words in query are in name or description
                    if not fuzzy_matches:
                        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'will', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'can', 'could', 'should', 'would', 'may', 'might', 'must', 'shall', 'increase', 'decrease', 'future', 'past', 'now', 'then', 'here', 'there', 'this', 'that', 'these', 'those', 'plot', 'show', 'display', 'graph', 'chart'}
                        query_words = set(re.findall(r'\b\w+\b', query_lower))
                        significant_words = [w for w in query_words if len(w) > 2 and w not in stop_words]  # Ignore short words and stop words
                        logger.debug(f"Significant words: {significant_words}")
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
                                                logger.debug(f"Word presence match: {name}")
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
    logger.debug(f"Final match result: {result}")
    return result

def extract_region_from_query(query: str, region_dict: dict, region_candidates: List[str] | None = None) -> str:
    """
    Extract region from query using region definitions.
    Returns the matched region name or empty string.
    """
    import logging
    logger = logging.getLogger(__name__)
    logger.debug(f"Extracting region for query: '{query}'")

    query_lower = query.lower()

    # Optional fast-path: match explicit region codes from candidates (e.g., CHN, USA)
    if region_candidates:
        candidate_set = {str(r).lower(): str(r) for r in region_candidates if r}
        # Map common country/region names to dataset region codes
        name_to_code = {
            "china": "CHN",
            "india": "IND",
            "united states": "USA",
            "u.s.": "USA",
            "u.s.a.": "USA",
            "usa": "USA",
            "european union": "EU",
            "europe": "EU",
            "eu": "EU",
            "european union 27": "EU-27",
            "eu-27": "EU-27",
            "eu27": "EU27",
            "european union 28": "EU28",
            "eu-28": "EU28",
            "eu28": "EU28",
            "united kingdom": "GBR",
            "uk": "GBR",
            "russia": "RUS",
        }
        for name in sorted(name_to_code.keys(), key=len, reverse=True):
            if re.search(r"\b" + re.escape(name) + r"\b", query_lower):
                code = name_to_code[name]
                code_key = code.lower()
                if code_key in candidate_set:
                    logger.debug(f"Region alias match: {name} -> {candidate_set[code_key]}")
                    return candidate_set[code_key]
        for token in re.findall(r"[A-Za-z0-9-]{2,8}", query):
            token_lower = token.lower()
            if token_lower in candidate_set:
                logger.debug(f"Region candidate match: {candidate_set[token_lower]}")
                return candidate_set[token_lower]
        # Fuzzy match for misspelled region codes
        from difflib import get_close_matches
        code_matches = get_close_matches(query_lower, list(candidate_set.keys()), n=1, cutoff=0.8)
        if code_matches:
            logger.debug(f"Region code fuzzy match: {code_matches[0]} -> {candidate_set[code_matches[0]]}")
            return candidate_set[code_matches[0]]

        # Try to match country names (including misspellings) to ISO and then to dataset codes
        try:
            import pycountry  # type: ignore
        except Exception:
            pycountry = None
        if pycountry:
            # Try direct lookup from free text
            try:
                match = pycountry.countries.search_fuzzy(query)
                if match:
                    c = match[0]
                    for code in [getattr(c, "alpha_3", None), getattr(c, "alpha_2", None)]:
                        if code and code.lower() in candidate_set:
                            logger.debug(f"Country fuzzy match: {c.name} -> {candidate_set[code.lower()]}")
                            return candidate_set[code.lower()]
            except Exception:
                pass

    # Map common ISO2/ISO3 codes to country names for region lookup
    iso_code_map = {
        # ISO3
        "CHN": "China", "USA": "United States", "IND": "India", "RUS": "Russian Federation",
        "DEU": "Germany", "FRA": "France", "GBR": "United Kingdom", "JPN": "Japan",
        "KOR": "South Korea", "BRA": "Brazil", "ZAF": "South Africa", "AUS": "Australia",
        "CAN": "Canada", "MEX": "Mexico", "IDN": "Indonesia", "TUR": "Turkey",
        "SAU": "Saudi Arabia", "ARG": "Argentina", "ITA": "Italy", "ESP": "Spain",
        "GRC": "Greece", "NLD": "Netherlands", "CHE": "Switzerland", "SWE": "Sweden",
        "NOR": "Norway", "POL": "Poland", "UKR": "Ukraine", "IRN": "Iran",
        "IRQ": "Iraq", "EGY": "Egypt", "NGA": "Nigeria", "PAK": "Pakistan",
        "VNM": "Viet Nam", "THA": "Thailand",
        # ISO2
        "CN": "China", "US": "United States", "IN": "India", "RU": "Russian Federation",
        "DE": "Germany", "FR": "France", "GB": "United Kingdom", "JP": "Japan",
        "KR": "South Korea", "BR": "Brazil", "ZA": "South Africa", "AU": "Australia",
        "CA": "Canada", "MX": "Mexico", "ID": "Indonesia", "TR": "Turkey",
        "SA": "Saudi Arabia", "AR": "Argentina", "IT": "Italy", "ES": "Spain",
        "GR": "Greece", "NL": "Netherlands", "CH": "Switzerland", "SE": "Sweden",
        "NO": "Norway", "PL": "Poland", "UA": "Ukraine", "IR": "Iran",
        "IQ": "Iraq", "EG": "Egypt", "NG": "Nigeria", "PK": "Pakistan",
        "VN": "Viet Nam", "TH": "Thailand",
    }
    for token in re.findall(r"[A-Za-z]{2,4}", query):
        mapped = iso_code_map.get(token.upper())
        if mapped:
            # Try to resolve mapped country name to a region
            mapped_lower = mapped.lower()
            for file_data in region_dict.values():
                for region_group in file_data:
                    for region_name, region_info in region_group.items():
                        if isinstance(region_info, dict):
                            countries = region_info.get("countries", [])
                            if any(mapped_lower == c.lower() for c in countries):
                                logger.debug(f"ISO code match: {token.upper()} -> {mapped} -> {region_name}")
                                return region_name

    # Enhanced region extraction with better country recognition
    # First check for exact matches in region names and countries
    for file_data in region_dict.values():
        for region_group in file_data:
            for region_name, region_info in region_group.items():
                if isinstance(region_info, dict):
                    # Check region name
                    if region_name.lower() in query_lower:
                        logger.debug(f"Exact region match: {region_name}")
                        return region_name
                    # Check countries with better matching
                    countries = region_info.get("countries", [])
                    for country in countries:
                        # Check for exact country match or common variations
                        country_lower = country.lower()
                        if country_lower in query_lower or query_lower in country_lower:
                            logger.debug(f"Country match: {country} -> {region_name}")
                            return region_name
                        # Special handling for Greece (common in energy systems)
                        if 'greece' in query_lower and country_lower == 'greece':
                            logger.debug(f"Greece country match -> {region_name}")
                            return region_name
                        # Handle "Greek" as variation of Greece
                        if 'greek' in query_lower and country_lower == 'greece':
                            logger.debug(f"Greek country match -> {region_name}")
                            return region_name
                elif isinstance(region_info, list):
                    # Handle list format
                    for item in region_info:
                        if isinstance(item, str) and item.lower() in query_lower:
                            logger.debug(f"List item match: {item} -> {region_name}")
                            return region_name

    # Fuzzy matching for common misspellings or partial matches
    from difflib import get_close_matches
    all_regions = []
    for file_data in region_dict.values():
        for region_group in file_data:
            for region_name in region_group.keys():
                all_regions.append(region_name)
    logger.debug(f"All regions for fuzzy: {all_regions[:10]}...")  # Log first 10

    matches = get_close_matches(query_lower, [r.lower() for r in all_regions], n=1, cutoff=0.6)
    if matches:
        # Find the original case region name
        for region in all_regions:
            if region.lower() == matches[0]:
                logger.debug(f"Fuzzy region match: {region}")
                return region

    logger.debug("No region match found")
    return ""


def format_region_label(region: str) -> str:
    """
    Format region codes with human-readable names when possible.
    Examples: AGO -> AGO (Angola), ARE -> ARE (United Arab Emirates).
    """
    if not region:
        return region
    code = str(region).strip()

    # Common dataset aliases
    alias = {
        "EU": "European Union",
        "EU-27": "European Union (EU-27)",
        "EU27": "European Union (EU27)",
        "USA": "United States",
        "CHN": "China",
        "IND": "India",
        "GBR": "United Kingdom",
        "RUS": "Russian Federation",
        "KOR": "South Korea",
    }
    if code in alias:
        return f"{code} ({alias[code]})"

    # Try ISO country lookup if available
    try:
        import pycountry  # type: ignore
    except Exception:
        return code

    try:
        if len(code) == 3 and code.isalpha():
            country = pycountry.countries.get(alpha_3=code.upper())
            if country:
                return f"{code} ({country.name})"
        if len(code) == 2 and code.isalpha():
            country = pycountry.countries.get(alpha_2=code.upper())
            if country:
                return f"{code} ({country.name})"
    except Exception:
        return code
    return code

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
                            enhanced_keywords.update(['installed', 'generation', 'power', 'pv', 'solar', 'photovoltaic'])
                        elif keyword in ['solar', 'pv', 'photovoltaic']:
                            enhanced_keywords.update(['capacity', 'pv', 'photovoltaic', 'solar'])
                        elif keyword in ['emission', 'emissions']:
                            enhanced_keywords.update(['co2', 'carbon', 'greenhouse', 'gas'])
                        elif keyword in ['energy']:
                            enhanced_keywords.update(['power', 'electricity', 'electric'])
                        elif keyword in ['future', 'annual', 'yearly']:
                            enhanced_keywords.update(['long-term', 'projection', 'forecast'])

                    # Store in index - include both original and expanded templates
                    for keyword in enhanced_keywords:
                        if keyword not in semantic_index:
                            semantic_index[keyword] = []
                        semantic_index[keyword].append({
                            'variable': var_name,
                            'description': description,
                            'unit': unit,
                            'is_template': '{' in var_name and '}' in var_name
                        })

                    # Also add expanded templated variables to semantic index
                    if '{' in var_name and '}' in var_name:
                        import re
                        template_match = re.search(r'\{([^}]+)\}', var_name)
                        if template_match:
                            template_name = template_match.group(1)
                            # Find template values
                            template_values = find_template_values(template_name, variable_dict)
                            for value_info in template_values:
                                value_name = value_info['name']
                                expanded_name = var_name.replace('{' + template_name + '}', value_name)

                                # Add expanded variable to semantic index with same keywords
                                for keyword in enhanced_keywords:
                                    if keyword not in semantic_index:
                                        semantic_index[keyword] = []
                                    semantic_index[keyword].append({
                                        'variable': expanded_name,
                                        'description': description,
                                        'unit': unit,
                                        'is_template': False  # Expanded, so not a template anymore
                                    })

    return semantic_index


def _score_variables(query: str, variable_dict: dict) -> tuple[dict, list]:
    query_lower = query.lower()

    # Build semantic index (cache this for performance)
    if not hasattr(resolve_natural_language_variable_universal, '_semantic_index'):
        resolve_natural_language_variable_universal._semantic_index = build_semantic_index(variable_dict)

    semantic_index = resolve_natural_language_variable_universal._semantic_index

    # Extract significant words from query
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'plot', 'show', 'graph', 'display', 'visualize', 'give', 'me', 'a', 'please', 'will', 'be', 'increase', 'future', 'can', 'you', 'tell', 'me', 'about', 'what', 'is', 'are', 'do', 'does', 'have', 'has', 'had', 'greece', 'greek', 'under', 'different', 'scenario', 'scenarios'}
    query_words = set(re.findall(r'\b\w+\b', query_lower))
    significant_words = [w for w in query_words if len(w) > 2 and w not in stop_words and w is not None]

    # Add back short but important words that were filtered out
    important_short_words = ['pv', 'co2', 'co', 'ch4', 'eu', 'us']
    for word in important_short_words:
        if word in query_lower and word not in significant_words:
            significant_words.append(word)

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

                # Enhanced scoring with priority for specific technology matches
                if 'nuclear' in significant_words and 'nuclear' in var_name.lower():
                    variable_scores[var_name]['score'] += 8  # High priority for nuclear matches
                elif 'pv' in significant_words and 'solar' in var_name.lower() and 'capacity' in var_name.lower():
                    variable_scores[var_name]['score'] += 10  # Very high priority for solar capacity when PV mentioned
                elif 'photovoltaic' in significant_words and 'solar' in var_name.lower() and 'capacity' in var_name.lower():
                    variable_scores[var_name]['score'] += 10  # Very high priority for solar capacity when photovoltaic mentioned
                elif 'capacity' in significant_words and 'capacity' in var_name.lower():
                    variable_scores[var_name]['score'] += 5  # High priority for capacity matches
                    if 'nuclear' in significant_words and 'nuclear' in var_name.lower():
                        variable_scores[var_name]['score'] += 5  # Extra boost for nuclear capacity
                    if 'pv' in significant_words and 'solar' in var_name.lower():
                        variable_scores[var_name]['score'] += 5  # Extra boost for PV + solar + capacity
                    if 'photovoltaic' in significant_words and 'solar' in var_name.lower():
                        variable_scores[var_name]['score'] += 5  # Extra boost for photovoltaic + solar + capacity
                    if 'solar' in significant_words and 'solar' in var_name.lower():
                        variable_scores[var_name]['score'] += 5  # Extra boost for solar + capacity
                    if 'wind' in significant_words and 'wind' in var_name.lower():
                        variable_scores[var_name]['score'] += 5  # Extra boost for wind capacity
                    if 'hydro' in significant_words and 'hydro' in var_name.lower():
                        variable_scores[var_name]['score'] += 5  # Extra boost for hydro capacity
                elif 'pv' in significant_words and ('solar' in var_name.lower() or 'pv' in var_name.lower()):
                    variable_scores[var_name]['score'] += 5  # High priority for PV/solar matches
                elif 'photovoltaic' in significant_words and ('solar' in var_name.lower() or 'pv' in var_name.lower()):
                    variable_scores[var_name]['score'] += 5  # High priority for photovoltaic matches
                elif 'solar' in significant_words and ('solar' in var_name.lower() or 'pv' in var_name.lower()):
                    variable_scores[var_name]['score'] += 5  # High priority for solar matches
                elif 'nuclear' in significant_words and 'nuclear' in var_name.lower():
                    variable_scores[var_name]['score'] += 5  # High priority for nuclear matches
                elif 'wind' in significant_words and 'wind' in var_name.lower():
                    variable_scores[var_name]['score'] += 5  # High priority for wind matches
                elif 'hydro' in significant_words and 'hydro' in var_name.lower():
                    variable_scores[var_name]['score'] += 5  # High priority for hydro matches
                elif 'methane' in significant_words and ('ch4' in var_name.lower() or 'methane' in var_name.lower()):
                    variable_scores[var_name]['score'] += 8  # Strong boost for methane/CH4 matches
                elif 'ch4' in significant_words and 'ch4' in var_name.lower():
                    variable_scores[var_name]['score'] += 8  # Strong boost for CH4
                elif 'demand' in significant_words and 'demand' in var_name.lower():
                    variable_scores[var_name]['score'] += 5  # Boost demand matches
                    if 'electricity' in significant_words and 'electricity' in var_name.lower():
                        variable_scores[var_name]['score'] += 5  # Boost electricity demand
                    if 'final' in significant_words and 'final' in var_name.lower():
                        variable_scores[var_name]['score'] += 3  # Boost final energy demand
                elif 'electricity' in significant_words and 'electricity' in var_name.lower():
                    variable_scores[var_name]['score'] += 4  # Boost electricity-related variables
                elif 'investment' in var_name.lower() and 'investment' in word:
                    variable_scores[var_name]['score'] += 4  # Lower priority for investment matches
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

                # Special handling for capacity and technology queries
                if 'capacity' in significant_words and 'capacity' in var_name.lower():
                    if 'nuclear' in significant_words and 'nuclear' in var_name.lower():
                        variable_scores[var_name]['score'] += 8  # Strong boost for nuclear capacity
                    if 'solar' in significant_words and 'solar' in var_name.lower():
                        variable_scores[var_name]['score'] += 5  # Boost solar capacity
                    if 'pv' in significant_words and ('solar' in var_name.lower() or 'pv' in var_name.lower()):
                        variable_scores[var_name]['score'] += 5  # Boost PV capacity
                    if 'photovoltaic' in significant_words and ('solar' in var_name.lower() or 'pv' in var_name.lower()):
                        variable_scores[var_name]['score'] += 5  # Boost photovoltaic capacity
                    if 'wind' in significant_words and 'wind' in var_name.lower():
                        variable_scores[var_name]['score'] += 5  # Boost wind capacity
                    if 'hydro' in significant_words and 'hydro' in var_name.lower():
                        variable_scores[var_name]['score'] += 5  # Boost hydro capacity
                    if 'electricity' in significant_words and 'electricity' in var_name.lower():
                        variable_scores[var_name]['score'] += 3  # Boost electricity capacity

                    # Boost any capacity variable for capacity queries
                    variable_scores[var_name]['score'] += 2  # General capacity boost

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

                # Prefer broad CO2 emission variables for generic emissions queries
                if any(word in significant_words for word in ['co2', 'emission', 'emissions']):
                    lower_name = var_name.lower()
                    if lower_name.startswith('emissions|co2') or lower_name.startswith('gross emissions|co2'):
                        variable_scores[var_name]['score'] += 8
                    elif 'co2' in lower_name:
                        variable_scores[var_name]['score'] += 4
                    if 'afolu' in lower_name and not any(word in significant_words for word in ['afolu', 'land', 'agriculture']):
                        variable_scores[var_name]['score'] -= 6
                    if any(term in lower_name for term in ['voc', 'bc', 'nh3', 'nox', 'sulfur']) and 'co2' not in lower_name:
                        variable_scores[var_name]['score'] -= 4

                variable_scores[var_name]['matched_words'].append(word)

    return variable_scores, significant_words


def resolve_natural_language_variable_with_score(
    query: str,
    variable_dict: dict
) -> tuple[str | None, int | None, list, list]:
    """
    Universal resolver that works for all variables in YAML definitions.
    Returns (best_var_name, best_score, matched_words, significant_words).
    """
    query_lower = query.lower()
    variable_scores, significant_words = _score_variables(query, variable_dict)

    if not variable_scores:
        return None, None, [], significant_words

    # Find best match
    best_variable = max(variable_scores.items(), key=lambda x: x[1]['score'])
    best_var_name = best_variable[0]
    best_score = best_variable[1]['score']

    # Debug logging - show all scored variables
    logger.debug("All variable scores:")
    for var_name, info in sorted(variable_scores.items(), key=lambda x: x[1]['score'], reverse=True)[:5]:
        logger.debug("  %s: %s points", var_name, info["score"])
    logger.debug("Best match: %s with score %s", best_var_name, best_score)
    logger.debug("Significant words: %s", significant_words)
    logger.debug("Query: %s", query)

    # For comparison queries, try to find a more general capacity variable
    if 'compare' in query_lower and 'capacity' in significant_words:
        # Look for general capacity variables that might exist
        general_capacity_vars = [name for name in variable_scores.keys() if 'capacity' in name.lower() and 'electricity' in name.lower()]
        if general_capacity_vars:
            # Sort by score and pick the highest
            best_general = max(general_capacity_vars, key=lambda x: variable_scores[x]['score'])
            logger.debug("Fallback to general capacity variable: %s", best_general)
            return best_general, variable_scores[best_general]["score"], variable_scores[best_general]["matched_words"], significant_words

    # For queries that result in variables not found in data, try to find similar available variables
    # This is a general fallback mechanism, not specific to PV capacity
    if best_var_name:
        # Check if the resolved variable actually exists in available data
        # We need to pass the available variables from the data context
        # For now, we'll implement a more general approach in the calling function
        # Return the best match anyway - let the calling function handle data availability
        pass

    # Minimum confidence threshold - lower for capacity and investment queries
    min_threshold = 1 if any(word in significant_words for word in ['capacity', 'investment', 'investments', 'invest']) else 2
    if best_score < min_threshold:
        # For capacity queries, try to find any capacity variable as fallback
        if any(word in significant_words for word in ['capacity']):
            capacity_vars = [name for name in variable_scores.keys() if 'capacity' in name.lower()]
            if capacity_vars:
                # Sort by score and pick the highest
                best_capacity = max(capacity_vars, key=lambda x: variable_scores[x]['score'])
                logger.debug("Fallback to capacity variable: %s", best_capacity)
                return best_capacity, variable_scores[best_capacity]["score"], variable_scores[best_capacity]["matched_words"], significant_words
        # For investment queries, try to find any investment variable as fallback
        if any(word in significant_words for word in ['investment', 'investments', 'invest']):
            investment_vars = [name for name in variable_scores.keys() if 'investment' in name.lower()]
            if investment_vars:
                # Sort by score and pick the highest
                best_investment = max(investment_vars, key=lambda x: variable_scores[x]['score'])
                return best_investment, variable_scores[best_investment]["score"], variable_scores[best_investment]["matched_words"], significant_words
        return None, None, [], significant_words

    # Additional fallback for comparison queries
    if 'compare' in query_lower and best_score < 3:
        # For comparison queries with low confidence, try general capacity variables
        general_capacity_vars = [name for name in variable_scores.keys() if 'capacity' in name.lower() and 'electricity' in name.lower()]
        if general_capacity_vars:
            best_general = max(general_capacity_vars, key=lambda x: variable_scores[x]['score'])
            logger.debug("Comparison query fallback to general capacity: %s", best_general)
            return best_general, variable_scores[best_general]["score"], variable_scores[best_general]["matched_words"], significant_words

    return best_var_name, best_score, best_variable[1]["matched_words"], significant_words


def resolve_natural_language_variable_candidates(
    query: str,
    variable_dict: dict,
    top_k: int = 3
) -> list[str]:
    """
    Return top candidate variables ranked by semantic score.
    """
    variable_scores, significant_words = _score_variables(query, variable_dict)
    if not variable_scores:
        return []
    ranked = sorted(variable_scores.items(), key=lambda x: x[1]["score"], reverse=True)

    key_terms = {"methane", "ch4", "demand", "electricity", "emission", "emissions", "co2", "capacity",
                 "solar", "wind", "oil", "gas", "transport", "industry", "buildings", "final", "primary"}
    query_terms = {w for w in significant_words if w in key_terms}

    if query_terms:
        matched = [name for name, _ in ranked if any(t in name.lower() for t in query_terms)]
        if matched:
            if any(t in query_terms for t in {"co2", "emission", "emissions"}):
                preferred = [
                    name for name in matched
                    if name.lower().startswith("emissions|co2")
                    or name.lower().startswith("gross emissions|co2")
                    or name.lower() == "emissions|co2"
                ]
                if preferred:
                    remainder = [name for name in matched if name not in preferred]
                    return (preferred + remainder)[:top_k]
            return matched[:top_k]

    return [name for name, _ in ranked[:top_k]]


def resolve_natural_language_variable_ranked(
    query: str,
    variable_dict: dict,
    top_k: int = 5
) -> list[tuple[str, int, list, list]]:
    """
    Return ranked variables with scores and matched words.
    Each item: (variable_name, score, matched_words, significant_words)
    """
    variable_scores, significant_words = _score_variables(query, variable_dict)
    if not variable_scores:
        return []
    ranked = sorted(variable_scores.items(), key=lambda x: x[1]["score"], reverse=True)
    out = []
    for name, info in ranked[:top_k]:
        out.append((name, info["score"], info.get("matched_words", []), significant_words))
    return out


def resolve_natural_language_variable_universal(query: str, variable_dict: dict) -> str:
    """
    Backward-compatible wrapper that returns only the best variable name.
    """
    best_var_name, _, _, _ = resolve_natural_language_variable_with_score(query, variable_dict)
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


def extract_variable_and_region_from_query(
    query: str,
    variable_dict: dict,
    region_dict: dict,
    region_candidates: List[str] | None = None
) -> dict:
    """
    Extract both variable and region from a natural language query.
    Returns:
      - variable: matched variable info
      - region: matched region name
    """
    variable_match = match_variable_from_yaml(query, variable_dict)
    region_match = extract_region_from_query(query, region_dict, region_candidates)

    return {
        "variable": variable_match,
        "region": region_match
    }
