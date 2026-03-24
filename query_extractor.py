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
from utils.yaml_loader import load_all_yaml_files
from utils_query import extract_region_from_query, resolve_natural_language_variable_candidates


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
        return re.sub(r"[^a-z0-9]+", "", (text or "").lower())

    def _build_model_alias_map(self, models: List[str]) -> Dict[str, Set[str]]:
        alias_map: Dict[str, Set[str]] = {}
        for model in models:
            raw = str(model or "").strip()
            if not raw:
                continue
            low = raw.lower()
            tokens = [t for t in re.split(r"[^a-z0-9]+", low) if t]
            aliases = {low, self._normalize_model(raw)}
            if tokens:
                aliases.add(tokens[0])
                if len(tokens) >= 2:
                    aliases.add(" ".join(tokens[:2]))
                    aliases.add("".join(tokens[:2]))
            for alias in aliases:
                if alias:
                    alias_map.setdefault(alias, set()).add(raw)
        return alias_map

    def _match_model_alias(self, query_or_model: str) -> Optional[str]:
        text = str(query_or_model or "").lower()
        if not text:
            return None

        # Exact model-name containment first.
        for model in self.available_models:
            if re.search(r"(?<!\w)" + re.escape(model.lower()) + r"(?!\w)", text):
                return model

        tokens = [t for t in re.split(r"[^a-z0-9]+", text) if t]
        spans: Set[str] = set(tokens)
        for i in range(len(tokens)):
            if i + 1 < len(tokens):
                spans.add(tokens[i] + " " + tokens[i + 1])
                spans.add(tokens[i] + tokens[i + 1])
        spans.add(self._normalize_model(text))

        candidates: Set[str] = set()
        for span in spans:
            candidates.update(self.model_alias_map.get(span, set()))
        if not candidates:
            return None

        def _rank(name: str) -> tuple[int, int, str]:
            low = name.lower()
            exact = 1 if low in spans or self._normalize_model(name) in spans else 0
            return (exact, -len(name), name)

        return sorted(candidates, key=_rank, reverse=True)[0]
    
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
            
            self.logger.debug(f"Extracted entities: {result}")
            return result
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parse error: {e}")
            return self._fallback_extraction(query)
        except Exception as e:
            self.logger.error(f"Extraction error: {e}")
            return self._fallback_extraction(query)
    
    def _validate_result(self, result: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Validate and enhance the extraction result."""
        
        # Validate variable
        if result.get('variable'):
            var = result['variable']
            # Check if exact match
            if var in self.available_variables:
                pass  # Valid
            else:
                # Try fuzzy match
                matched = self._fuzzy_match(var, self.available_variables)
                if matched:
                    result['variable'] = matched
                    result['variable_matched'] = True
        
        # Validate region
        if result.get('region'):
            region = result['region']
            if region not in self.available_regions:
                # Try alias/region definition mapping before fuzzy
                mapped = extract_region_from_query(query, self.region_dict, self.available_regions)
                if mapped:
                    result['region'] = mapped
                    result['region_matched'] = True
                else:
                    matched = self._fuzzy_match(region, self.available_regions)
                    if matched:
                        result['region'] = matched
                        result['region_matched'] = True
        
        # Validate scenario
        if result.get('scenario'):
            scenario = result['scenario']
            if scenario not in self.available_scenarios:
                matched = self._fuzzy_match(scenario, self.available_scenarios)
                if matched:
                    result['scenario'] = matched
                    result['scenario_matched'] = True
        
        # Validate model
        if result.get('model'):
            model = result['model']
            if model not in self.available_models:
                matched = self._match_model_alias(model) or self._match_model_alias(query)
                if not matched:
                    matched = self._fuzzy_match(model, self.available_models)
                if matched:
                    result['model'] = matched
                    result['model_matched'] = True
        
        # Add unit if variable is found
        if result.get('variable') and result['variable'] in self.variable_units:
            result['unit'] = self.variable_units[result['variable']]
        
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
            'comparison': None
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
        
        # Try to match variables with the shared candidate resolver first.
        variable_candidates = resolve_natural_language_variable_candidates(query, self.variable_dict, top_k=3)
        for candidate in variable_candidates:
            if candidate in self.available_variables:
                result['variable'] = candidate
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
        
        # Try to match regions (use shared extractor with alias support)
        region_match = extract_region_from_query(query, self.region_dict, self.available_regions)
        if region_match:
            result['region'] = region_match
        
        # Try to match scenarios
        for scenario in self.available_scenarios:
            if scenario.lower() in q:
                result['scenario'] = scenario
                break

        # Try to match model names/aliases.
        model_match = self._match_model_alias(query)
        if model_match:
            result['model'] = model_match
        
        # Extract years and year ranges
        year_match = re.search(r'\b(20\d{2})\s*(?:to|-|until|through)\s*(20\d{2})\b', q)
        if year_match:
            result['start_year'] = int(year_match.group(1))
            result['end_year'] = int(year_match.group(2))
        else:
            # Check for "from X to Y" pattern
            year_match = re.search(r'from\s+(20\d{2})\s*(?:to|-|until|through)\s*(20\d{2})\b', q)
            if year_match:
                result['start_year'] = int(year_match.group(1))
                result['end_year'] = int(year_match.group(2))
            else:
                # Check for "after X" pattern
                year_match = re.search(r'after\s+(20\d{2})\b', q)
                if year_match:
                    result['start_year'] = int(year_match.group(1))
                else:
                    # Check for "before X" pattern
                    year_match = re.search(r'before\s+(20\d{2})\b', q)
                    if year_match:
                        result['end_year'] = int(year_match.group(1))
                    else:
                        # Single year
                        years = re.findall(r'\b(20\d{2})\b', q)
                        if years:
                            result['start_year'] = int(years[0])
                            result['end_year'] = int(years[0])
        
        return result
    
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
