"""
Query Entity Extractor - Extracts structured entities from user queries.

This module uses LLM to extract all data dimensions from user queries:
- Variables (e.g., "CO2 emissions", "solar capacity")
- Regions (e.g., "Greece", "Europe", "World")
- Scenarios (e.g., "SSP2-45", "NetZero")
- Models (e.g., "REMIND", "GCAM")
- Years (e.g., "2050", "2020-2100")
- Units (e.g., "GW", "Mt CO2/yr")
"""

import logging
from typing import Dict, Any, List, Optional, Set
import re
import json

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from canonical_aliases import REGION_ALIASES, canonical_scenario_from_query, preferred_variable_from_query
from model_aliases import build_model_alias_map, match_model_name, normalize_model_name
from utils.yaml_loader import load_all_yaml_files
from utils_query import extract_region_from_query, resolve_natural_language_variable_candidates
from year_filters import extract_year_range


class QueryEntityExtractor:
    """Extracts structured entities from user queries using LLM."""
    
    def __init__(self, models: List[Dict], ts_data: List[Dict], api_key: str):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.models = models
        self.ts_data = ts_data
        self.api_key = api_key
        
        # Build lookup sets from data
        self._build_lookups()
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model_name="gpt-4-turbo",
            temperature=0,
            timeout=30,
            max_retries=1,
            api_key=api_key
        )
        
        # Create extraction prompt
        self._create_prompt()
    
    def _build_lookups(self):
        """Build lookup sets from the data."""
        # Extract unique values for each dimension
        self.available_models = sorted({
            str(m.get('modelName', '')) 
            for m in self.models 
            if m and m.get('modelName')
        })
        
        self.available_scenarios = sorted({
            str(r.get('scenario', '')) 
            for r in self.ts_data 
            if r and r.get('scenario')
        })
        
        self.available_variables = sorted({
            str(r.get('variable', '')) 
            for r in self.ts_data 
            if r and r.get('variable')
        })
        
        self.available_regions = sorted({
            str(r.get('region', '')) 
            for r in self.ts_data 
            if r and r.get('region')
        })
        self.variable_dict = load_all_yaml_files('definitions/variable')
        self.region_dict = load_all_yaml_files('definitions/region')
        
        # Extract years from data
        self.available_years = set()
        for r in self.ts_data:
            if r and 'years' in r:
                self.available_years.update(r['years'].keys())
        self.available_years = sorted(self.available_years)
        
        # Build variable -> unit mapping
        self.variable_units = {}
        for r in self.ts_data:
            if r and r.get('variable') and r.get('unit'):
                var = r['variable']
                unit = r['unit']
                if var not in self.variable_units:
                    self.variable_units[var] = unit
        
        # Build variable -> regions mapping
        self.variable_regions = {}
        for r in self.ts_data:
            if r and r.get('variable') and r.get('region'):
                var = r['variable']
                reg = r['region']
                if var not in self.variable_regions:
                    self.variable_regions[var] = set()
                self.variable_regions[var].add(reg)
        
        # Build variable -> scenarios mapping
        self.variable_scenarios = {}
        for r in self.ts_data:
            if r and r.get('variable') and r.get('scenario'):
                var = r['variable']
                scen = r['scenario']
                if var not in self.variable_scenarios:
                    self.variable_scenarios[var] = set()
                self.variable_scenarios[var].add(scen)
        
        self.logger.debug(f"Built lookups: {len(self.available_models)} models, "
                        f"{len(self.available_scenarios)} scenarios, "
                        f"{len(self.available_variables)} variables, "
                        f"{len(self.available_regions)} regions")
        self.model_alias_map = self._build_model_alias_map(self.available_models)

    def _normalize_model(self, text: str) -> str:
        return normalize_model_name(text)

    def _build_model_alias_map(self, models: List[str]) -> Dict[str, Set[str]]:
        return build_model_alias_map(models)

    def _match_model_alias(self, query_or_model: str) -> Optional[str]:
        if re.search(r"\bgcam\s*[- ]?\s*pr\b", str(query_or_model or ""), re.IGNORECASE):
            gcam_pr_models = [model for model in self.available_models if "gcam-pr" in model.lower()]
            if gcam_pr_models:
                return next((model for model in sorted(gcam_pr_models, reverse=True) if "7.0" in model), sorted(gcam_pr_models, reverse=True)[0])
        return match_model_name(query_or_model, self.available_models) or None
    
    def _create_prompt(self):
        """Create the LLM extraction prompt."""
        
        # Prioritize common variables that users are likely to query
        priority_vars = [
            "Emissions|CO2",
            "Emissions|CO2|Energy",
            "Emissions|CO2|Energy and Industrial Processes",
            "Capacity|Electricity|Solar",
            "Capacity|Electricity|Wind",
            "Secondary Energy|Electricity|Solar",
            "Secondary Energy|Electricity|Wind",
            "Primary Energy",
            "Final Energy",
            "Secondary Energy|Electricity",
        ]
        
        # Find which priority vars exist in data
        available_priority = [v for v in priority_vars if v in self.available_variables]
        
        # Sample variables for the prompt (priority first, then others)
        other_vars = [v for v in self.available_variables if v not in available_priority][:40]
        var_samples = available_priority + other_vars
        var_list = "\n".join(f"- {v}" for v in var_samples)
        if len(self.available_variables) > len(var_samples):
            var_list += f"\n... and {len(self.available_variables) - len(var_samples)} more"
        
        # Sample regions
        region_list = ", ".join(self.available_regions[:30])
        if len(self.available_regions) > 30:
            region_list += f" ... and {len(self.available_regions) - 30} more"
        
        # Sample scenarios
        scenario_list = ", ".join(self.available_scenarios[:20])
        if len(self.available_scenarios) > 20:
            scenario_list += f" ... and {len(self.available_scenarios) - 20} more"
        
        # Sample models
        model_list = ", ".join(self.available_models[:20])
        if len(self.available_models) > 20:
            model_list += f" ... and {len(self.available_models) - 20} more"
        
        system_template = f"""You are an entity extractor for IAM PARIS climate data queries.

Extract the following entities from user queries and return as JSON:

{{{{
    "action": "plot" or "query",
    "variable": "exact variable name from list below or null",
    "variables": ["list of variables for comparison queries"] or null,
    "region": "region name or null", 
    "scenario": "scenario name or null",
    "model": "model name or null",
    "models": ["list of models for comparison queries"] or null,
    "start_year": year or null,
    "end_year": year or null,
    "comparison": "model" or "scenario" or "region" or "variable" or null
}}}}

## Available Data:

### Variables (sample):
{var_list}

### Regions:
{region_list}

### Scenarios:
{scenario_list}

### Models:
{model_list}

## Extraction Rules:

1. **action**: 
   - "plot" if user wants a graph/chart/visualization
   - "query" for questions about data

2. **variable**: 
   - Match to exact variable name from the list
   - For "solar" use "Capacity|Electricity|Solar"
   - For "wind" use "Capacity|Electricity|Wind"
   - For "CO2 emissions" use "Emissions|CO2"
   - For "energy" look for variables containing "Energy"
   - Return null if no variable mentioned

3. **variables** (for multi-variable comparison):
   - Use when user wants to COMPARE multiple variables
   - Examples: "compare solar and wind", "solar vs wind", "difference between CO2 and CH4"
   - Return list of exact variable names: ["Capacity|Electricity|Solar", "Capacity|Electricity|Wind"]
   - Return null for single-variable queries

4. **region**:
   - Match country/region names
   - Common: Greece, Germany, Europe, World, EU

5. **scenario**:
   - Match scenario names like SSP2-45, NetZero, Current Policies

6. **model**:
   - Match model names like REMIND, GCAM, MESSAGE
   - For single model queries

7. **models** (for multi-model comparison):
   - Use when user wants to COMPARE multiple models
   - Examples: "compare GCAM and REMIND", "GCAM vs MESSAGE", "difference between models"
   - Return list of exact model names: ["GCAM", "REMIND-MAgPIE"]
   - Return null for single-model queries

8. **years**:
   - Extract mentioned years or year ranges
   - For single year: "2050" -> start_year: 2050, end_year: 2050
   - For range: "2020 to 2050" or "from 2020 until 2050" -> start_year: 2020, end_year: 2050
   - For open-ended: "after 2030" -> start_year: 2030, end_year: null
   - For open-ended: "before 2050" -> start_year: null, end_year: 2050
   - Return null if no years mentioned

9. **comparison**:
   - "variable" if comparing different variables (e.g., "compare solar and wind")
   - "model" if comparing different models (e.g., "compare GCAM and REMIND")
   - "scenario" if comparing different scenarios  
   - "region" if comparing different regions
   - null otherwise

Return ONLY valid JSON, no other text."""

        human_template = "Query: {query}"
        
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
    
    def extract(self, query: str) -> Dict[str, Any]:
        """
        Extract entities from a user query.
        
        Returns:
            Dict with keys: action, variable, region, scenario, model, years, comparison
        """
        deterministic = self._fallback_extraction(query)
        if self._deterministic_result_is_sufficient(deterministic):
            deterministic["extraction_method"] = "deterministic"
            return deterministic

        try:
            # Use LLM to extract entities
            chain = self.prompt | self.llm
            response = chain.invoke({"query": query})
            
            # Parse JSON response
            content = response.content.strip()
            
            # Remove markdown code blocks if present
            if content.startswith("```"):
                content = re.sub(r'^```(?:json)?\s*', '', content)
                content = re.sub(r'\s*```$', '', content)
            
            result = json.loads(content)
            
            # Validate and enhance the result
            result = self._validate_result(result, query)
            result = self._finalize_confidence(result)
            result["extraction_method"] = "llm"
            
            self.logger.debug(f"Extracted entities: {result}")
            return result
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parse error: {e}")
            deterministic["extraction_method"] = "fallback"
            return deterministic
        except Exception as e:
            self.logger.error(f"Extraction error: {e}")
            deterministic["extraction_method"] = "fallback"
            return deterministic

    def _deterministic_result_is_sufficient(self, result: Dict[str, Any]) -> bool:
        confidence = result.get("entity_confidence") or {}
        strong_fields = {
            field for field in ("variable", "region", "scenario", "model", "years")
            if result.get(field) and confidence.get(field, 0) >= 0.7
        }
        if result.get("start_year") is not None or result.get("end_year") is not None:
            if confidence.get("years", 0) >= 0.7:
                strong_fields.add("years")
        if result.get("action") == "plot" and (result.get("variable") or result.get("region")):
            return True
        return bool(strong_fields)
    
    def _validate_result(self, result: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Validate and enhance the extraction result."""
        entity_confidence = dict(result.get("entity_confidence") or {})
        result["entity_confidence"] = entity_confidence
        
        # Validate variable
        if result.get('variable'):
            var = result['variable']
            # Check if exact match
            if var in self.available_variables:
                entity_confidence.setdefault("variable", 0.95)
            else:
                # Try fuzzy match
                matched = self._fuzzy_match(var, self.available_variables)
                if matched:
                    result['variable'] = matched
                    result['variable_matched'] = True
                    entity_confidence["variable"] = 0.75
                else:
                    entity_confidence["variable"] = 0.35
        
        # Validate region
        if result.get('region'):
            region = result['region']
            if region in self.available_regions:
                entity_confidence.setdefault("region", 0.95)
            else:
                # Try alias/region definition mapping before fuzzy
                mapped = extract_region_from_query(query, self.region_dict, self.available_regions)
                if mapped:
                    result['region'] = mapped
                    result['region_matched'] = True
                    entity_confidence["region"] = 0.85
                else:
                    matched = self._fuzzy_match(region, self.available_regions)
                    if matched:
                        result['region'] = matched
                        result['region_matched'] = True
                        entity_confidence["region"] = 0.75
                    else:
                        entity_confidence["region"] = 0.35
        
        # Validate scenario
        if result.get('scenario'):
            scenario = result['scenario']
            if scenario in self.available_scenarios:
                entity_confidence.setdefault("scenario", 0.95)
            else:
                matched = self._fuzzy_match(scenario, self.available_scenarios)
                if matched:
                    result['scenario'] = matched
                    result['scenario_matched'] = True
                    entity_confidence["scenario"] = 0.75
                else:
                    entity_confidence["scenario"] = 0.35
        
        # Validate model
        strong_model_alias_query = re.search(r"\b(message\s*ix|messageix|message-ix)\b", query, re.IGNORECASE)
        if strong_model_alias_query and not result.get("model"):
            result["model"] = "MESSAGEix-GLOBIOM 2.0"
            result["model_matched"] = True
            entity_confidence["model"] = 0.75

        if result.get('model'):
            model = result['model']
            query_alias_match = self._match_model_alias(query)
            if strong_model_alias_query and query_alias_match:
                result['model'] = query_alias_match
                result['model_matched'] = True
                entity_confidence["model"] = 0.8
            elif strong_model_alias_query:
                result['model'] = "MESSAGEix-GLOBIOM 2.0"
                result['model_matched'] = True
                entity_confidence["model"] = 0.75
            elif model in self.available_models:
                entity_confidence.setdefault("model", 0.95)
            else:
                matched = self._match_model_alias(model) or query_alias_match
                if not matched:
                    matched = self._fuzzy_match(model, self.available_models)
                if matched:
                    result['model'] = matched
                    result['model_matched'] = True
                    entity_confidence["model"] = 0.8
                else:
                    entity_confidence["model"] = 0.35
        
        # Add unit if variable is found
        if result.get('variable') and result['variable'] in self.variable_units:
            result['unit'] = self.variable_units[result['variable']]
        
        return result

    def _finalize_confidence(self, result: Dict[str, Any]) -> Dict[str, Any]:
        entity_confidence = dict(result.get("entity_confidence") or {})
        if result.get("action"):
            entity_confidence.setdefault("action", 0.9)
        if result.get("comparison"):
            entity_confidence.setdefault("comparison", 0.8)
        if result.get("start_year") or result.get("end_year"):
            entity_confidence.setdefault("years", 0.9)

        scored = [
            score for field, score in entity_confidence.items()
            if field != "action" and isinstance(score, (int, float))
        ]
        if not scored and isinstance(entity_confidence.get("action"), (int, float)):
            scored = [entity_confidence["action"]]
        result["entity_confidence"] = entity_confidence
        result["confidence"] = round(sum(scored) / len(scored), 3) if scored else 0.0
        return result
    
    def _fuzzy_match(self, value: str, options: List[str]) -> Optional[str]:
        """Find fuzzy match for a value in options."""
        from difflib import get_close_matches
        
        value_lower = value.lower()
        
        # Try exact case-insensitive match first
        for opt in options:
            if opt.lower() == value_lower:
                return opt
        
        # Try substring match
        for opt in options:
            if value_lower in opt.lower() or opt.lower() in value_lower:
                return opt
        
        # Try fuzzy match
        matches = get_close_matches(value_lower, [o.lower() for o in options], n=1, cutoff=0.6)
        if matches:
            # Find original case version
            for opt in options:
                if opt.lower() == matches[0]:
                    return opt
        
        return None

    def _fuzzy_variable_from_tokens(self, q: str) -> Optional[str]:
        """N8: resolve a variable when the query contains a misspelled keyword by
        fuzzy-matching query tokens against variable-name tokens."""
        from difflib import get_close_matches

        query_terms = [tok for tok in re.findall(r"[a-z0-9]+", q.lower()) if len(tok) > 3]
        if not query_terms:
            return None
        scored = []
        for var in self.available_variables:
            var_terms = {tok for tok in re.findall(r"[a-z0-9]+", var.lower()) if len(tok) > 3}
            if not var_terms:
                continue
            hits = 0
            for term in query_terms:
                if get_close_matches(term, var_terms, n=1, cutoff=0.82):
                    hits += 1
            if hits:
                scored.append((hits, -len(var), var))
        if not scored:
            return None
        scored.sort(reverse=True)
        return scored[0][2]

    def _fuzzy_region_from_tokens(self, q: str) -> Optional[str]:
        """N8: resolve a region from a misspelled token (e.g. "europ" -> "EU")."""
        from difflib import get_close_matches

        query_terms = [tok for tok in re.findall(r"[a-z]+", q.lower()) if len(tok) > 3]
        if not query_terms:
            return None
        # Match against alias keys first (so "europ" maps via the canonical alias),
        # then against the raw available region names.
        for phrases, canonical in REGION_ALIASES:
            for phrase in phrases:
                if " " in phrase:
                    continue
                for term in query_terms:
                    if get_close_matches(term, [phrase], n=1, cutoff=0.82):
                        if not self.available_regions or canonical in self.available_regions:
                            return canonical
                        match = self._fuzzy_match(canonical, self.available_regions)
                        if match:
                            return match
        region_names_lower = {r.lower(): r for r in self.available_regions}
        for term in query_terms:
            matches = get_close_matches(term, list(region_names_lower.keys()), n=1, cutoff=0.85)
            if matches:
                return region_names_lower[matches[0]]
        return None

    def _query_allows_model_match(self, query: str) -> bool:
        q = str(query or "").lower()
        if not q.strip():
            return False
        if re.search(r"\b(model|models|using|use|with|about|explain|assumptions?|information|info)\b", q):
            return True
        return bool(re.search(r"\b(gcam|remind|message\s*ix|messageix|witch|prometheus|leap|gemini|gem-e3|e3me)\b", q))
    
    def _fallback_extraction(self, query: str) -> Dict[str, Any]:
        """Fallback keyword-based extraction when LLM fails."""
        result = {
            'action': 'query',
            'variable': None,
            'variables': None,
            'region': None,
            'scenario': None,
            'model': None,
            'models': None,
            'start_year': None,
            'end_year': None,
            'comparison': None,
            'entity_confidence': {'action': 0.75}
        }
        
        q = query.lower()
        
        tokens = {tok for tok in re.findall(r"[a-z0-9]+", q) if tok}
        explicit_data_query = bool(
            tokens & {
                'data', 'value', 'values', 'timeseries', 'time', 'series',
                'show', 'display', 'give', 'provide', 'retrieve', 'fetch'
            }
        ) or bool(re.search(r'\btime\s+series\b', q))
        # Detect action
        if any(word in tokens for word in ['plot', 'graph', 'chart', 'visualize', 'visualise']):
            result['action'] = 'plot'
            result['entity_confidence']['action'] = 0.9

        preferred_variable = self._preferred_variable_from_query(query)
        if preferred_variable:
            result['variable'] = preferred_variable
            result['entity_confidence']['variable'] = 0.9
        
        # Try to match variables with the shared candidate resolver first.
        if not result['variable']:
            variable_candidates = resolve_natural_language_variable_candidates(query, self.variable_dict, top_k=3)
            for candidate in variable_candidates:
                if candidate in self.available_variables:
                    result['variable'] = candidate
                    result['entity_confidence']['variable'] = 0.7
                    break
        if not result['variable']:
            query_terms = {tok for tok in tokens if len(tok) > 2}
            scored = []
            for var in self.available_variables:
                var_terms = {tok for tok in re.findall(r"[a-z0-9]+", var.lower()) if len(tok) > 2}
                overlap = len(query_terms & var_terms)
                if overlap:
                    scored.append((overlap, var))
            if scored:
                scored.sort(key=lambda item: (-item[0], item[1]))
                result['variable'] = scored[0][1]
                result['entity_confidence']['variable'] = 0.55

        # N8: typo tolerance. If no variable resolved, fuzzy-match query tokens
        # against variable-name tokens (e.g. "emisions" -> "emissions").
        if not result['variable']:
            fuzzy_var = self._fuzzy_variable_from_tokens(q)
            if fuzzy_var:
                result['variable'] = fuzzy_var
                result['entity_confidence']['variable'] = 0.5

        # Guard: an energy question ("final/primary/secondary energy") must not
        # resolve to an emissions variable just because the variable name
        # contains "Energy" (e.g. "final energy demand" -> Emissions|CO2|Energy|Demand).
        if (
            result['variable']
            and 'emission' in str(result['variable']).lower()
            and 'emission' not in q
            and 'co2' not in q
        ):
            for fam in ('final energy', 'primary energy', 'secondary energy'):
                if fam in q:
                    base = next((v for v in self.available_variables if v.lower() == fam), None)
                    candidates = [v for v in self.available_variables if v.lower().startswith(fam)]
                    if base:
                        result['variable'] = base
                    elif candidates:
                        result['variable'] = min(candidates, key=len)
                    else:
                        break
                    result['entity_confidence']['variable'] = 0.8
                    break

        # Try to match regions (use shared extractor with alias support)
        region_match = extract_region_from_query(query, self.region_dict, self.available_regions)
        if region_match:
            result['region'] = region_match
            result['entity_confidence']['region'] = 0.85
        else:
            # N8: typo tolerance for region tokens (e.g. "europ" -> "Europe"/"EU").
            fuzzy_region = self._fuzzy_region_from_tokens(q)
            if fuzzy_region:
                result['region'] = fuzzy_region
                result['entity_confidence']['region'] = 0.6

        # Try to match scenarios
        scenario_match = canonical_scenario_from_query(query, self.available_scenarios)
        if scenario_match:
            result['scenario'] = scenario_match
            result['entity_confidence']['scenario'] = 0.9
        for scenario in self.available_scenarios:
            if not result['scenario'] and scenario.lower() in q:
                result['scenario'] = scenario
                result['entity_confidence']['scenario'] = 0.95
                break

        # Try to match model names/aliases.
        model_match = self._match_model_alias(query) if self._query_allows_model_match(query) else None
        if not model_match and re.search(r"\b(message\s*ix|messageix|message-ix)\b", query, re.IGNORECASE):
            model_match = "MESSAGEix-GLOBIOM 2.0"
        if model_match:
            result['model'] = model_match
            result['entity_confidence']['model'] = 0.8
        
        # Extract years and year ranges
        start_year, end_year = extract_year_range(query)
        if start_year is not None or end_year is not None:
            result['start_year'] = start_year
            result['end_year'] = end_year
            result['entity_confidence']['years'] = 0.9

        return self._finalize_confidence(result)

    def _preferred_variable_from_query(self, query: str) -> Optional[str]:
        return preferred_variable_from_query(query, self.available_variables)
    
    def get_available_for_variable(self, variable: str) -> Dict[str, Any]:
        """Get available regions, scenarios, models, and unit for a variable."""
        return {
            'regions': sorted(self.variable_regions.get(variable, [])),
            'scenarios': sorted(self.variable_scenarios.get(variable, [])),
            'unit': self.variable_units.get(variable, 'Unknown'),
            'models': self.available_models  # Models are typically available for all
        }
    
    def suggest_combinations(self, variable: str = None, region: str = None, 
                            scenario: str = None) -> List[Dict[str, str]]:
        """Suggest valid combinations based on partial specifications."""
        suggestions = []
        
        if variable:
            var_info = self.get_available_for_variable(variable)
            
            if not region and var_info['regions']:
                for reg in var_info['regions'][:5]:
                    suggestions.append({
                        'variable': variable,
                        'region': reg,
                        'unit': var_info['unit']
                    })
            
            if not scenario and var_info['scenarios']:
                for scen in var_info['scenarios'][:5]:
                    suggestions.append({
                        'variable': variable,
                        'scenario': scen,
                        'unit': var_info['unit']
                    })
        
        return suggestions
