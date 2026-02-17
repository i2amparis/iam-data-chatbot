"""
Data Metadata Builder for IAM PARIS Climate Policy Assistant.

This module builds metadata structures that know:
- Which variables have data for which regions
- Which scenarios are available per variable
- Valid model/scenario/region combinations
- Units per variable

This helps validate queries and provide helpful suggestions.
"""

import os
import pickle
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class DataMetadata:
    """
    Builds and maintains metadata about available data combinations.
    
    Attributes:
        variable_regions: Maps variable -> set of available regions
        variable_scenarios: Maps variable -> set of available scenarios
        variable_models: Maps variable -> set of available models
        variable_units: Maps variable -> unit string
        region_variables: Maps region -> set of available variables
        scenario_variables: Maps scenario -> set of available variables
    """
    
    def __init__(self, ts_data: List[dict], models: List[dict] = None):
        """
        Initialize metadata from time series data.
        
        Args:
            ts_data: List of time series records from IAM PARIS API
            models: List of model records (optional, for model metadata)
        """
        self.ts_data = ts_data
        self.models = models or []
        
        # Build metadata structures
        self.variable_regions: Dict[str, Set[str]] = defaultdict(set)
        self.variable_scenarios: Dict[str, Set[str]] = defaultdict(set)
        self.variable_models: Dict[str, Set[str]] = defaultdict(set)
        self.variable_units: Dict[str, str] = {}
        
        self.region_variables: Dict[str, Set[str]] = defaultdict(set)
        self.scenario_variables: Dict[str, Set[str]] = defaultdict(set)
        self.model_variables: Dict[str, Set[str]] = defaultdict(set)
        
        # All unique values
        self.all_variables: Set[str] = set()
        self.all_regions: Set[str] = set()
        self.all_scenarios: Set[str] = set()
        self.all_model_names: Set[str] = set()
        
        # Variable categories (for better suggestions)
        self.variable_categories: Dict[str, List[str]] = defaultdict(list)
        
        # Build the metadata
        self._build_metadata()
        self._categorize_variables()
    
    def _build_metadata(self):
        """Build all metadata structures from time series data."""
        logger.info("Building data metadata...")
        
        for record in self.ts_data:
            if record is None:
                continue
                
            variable = record.get('variable', '')
            region = record.get('region', '')
            scenario = record.get('scenario', '')
            model = record.get('model', '')
            unit = record.get('unit', '')
            
            if variable:
                self.all_variables.add(variable)
                
                if region:
                    self.variable_regions[variable].add(region)
                    self.all_regions.add(region)
                    self.region_variables[region].add(variable)
                
                if scenario:
                    self.variable_scenarios[variable].add(scenario)
                    self.all_scenarios.add(scenario)
                    self.scenario_variables[scenario].add(variable)
                
                if model:
                    self.variable_models[variable].add(model)
                    self.all_model_names.add(model)
                    self.model_variables[model].add(variable)
                
                if unit and variable not in self.variable_units:
                    self.variable_units[variable] = unit
        
        # Add model names from models list
        for model in self.models:
            if model and model.get('modelName'):
                self.all_model_names.add(model['modelName'])
        
        logger.info(f"Metadata built: {len(self.all_variables)} variables, "
                   f"{len(self.all_regions)} regions, "
                   f"{len(self.all_scenarios)} scenarios, "
                   f"{len(self.all_model_names)} models")
    
    def _categorize_variables(self):
        """Categorize variables by topic for better suggestions."""
        categories = {
            'Emissions': ['Emissions|CO2', 'Emissions|CH4', 'Emissions|N2O', 'Emissions|GHG'],
            'Energy Supply': ['Energy Supply', 'Electricity', 'Capacity'],
            'Energy Demand': ['Final Energy', 'Primary Energy', 'Energy Demand'],
            'Renewables': ['Solar', 'Wind', 'Hydro', 'Biomass', 'Geothermal'],
            'Transport': ['Transport', 'Vehicle', 'Travel'],
            'Industry': ['Industrial', 'Industry', 'Production'],
            'Buildings': ['Building', 'Residential', 'Commercial'],
            'Agriculture': ['Agricultural', 'Agriculture', 'Crop', 'Livestock'],
            'Land Use': ['Land Use', 'Forestry', 'AFOLU'],
            'Economic': ['GDP', 'Policy Cost', 'Investment', 'Price'],
            'Population': ['Population', 'Urban', 'Household'],
            'Climate': ['Temperature', 'Forcing', 'Carbon']
        }
        
        for variable in self.all_variables:
            for category, keywords in categories.items():
                if any(kw.lower() in variable.lower() for kw in keywords):
                    self.variable_categories[category].append(variable)
                    break
    
    def get_available_for_variable(self, variable: str) -> Dict[str, any]:
        """
        Get all available metadata for a specific variable.
        
        Args:
            variable: The variable name to look up
            
        Returns:
            Dict with regions, scenarios, models, and unit for the variable
        """
        # Find best match if not exact
        matched_var = self._find_best_variable_match(variable)
        
        if not matched_var:
            return {
                'variable': None,
                'regions': [],
                'scenarios': [],
                'models': [],
                'unit': None,
                'suggestions': self._suggest_similar_variables(variable)
            }
        
        return {
            'variable': matched_var,
            'regions': sorted(self.variable_regions.get(matched_var, [])),
            'scenarios': sorted(self.variable_scenarios.get(matched_var, [])),
            'models': sorted(self.variable_models.get(matched_var, [])),
            'unit': self.variable_units.get(matched_var),
            'suggestions': []
        }
    
    def get_available_for_region(self, region: str) -> Dict[str, any]:
        """
        Get all variables available for a specific region.
        
        Args:
            region: The region name to look up
            
        Returns:
            Dict with variables and scenarios available for the region
        """
        # Case-insensitive region matching
        matched_region = self._find_best_region_match(region)
        
        if not matched_region:
            return {
                'region': None,
                'variables': [],
                'scenarios': [],
                'suggestions': self._suggest_similar_regions(region)
            }
        
        return {
            'region': matched_region,
            'variables': sorted(self.region_variables.get(matched_region, [])),
            'scenarios': sorted(self.all_scenarios),
            'suggestions': []
        }
    
    def validate_combination(self, variable: str, region: str = None, 
                            scenario: str = None, model: str = None) -> Dict[str, any]:
        """
        Validate if a combination of dimensions has data available.
        
        Args:
            variable: Variable name
            region: Region name (optional)
            scenario: Scenario name (optional)
            model: Model name (optional)
            
        Returns:
            Dict with validation result and suggestions if invalid
        """
        result = {
            'valid': True,
            'issues': [],
            'suggestions': []
        }
        
        # Check variable
        var_info = self.get_available_for_variable(variable)
        if not var_info['variable']:
            result['valid'] = False
            result['issues'].append(f"Variable '{variable}' not found")
            result['suggestions'].extend(var_info['suggestions'])
            return result
        
        matched_var = var_info['variable']
        
        # Check region
        if region:
            matched_region = self._find_best_region_match(region)
            if not matched_region:
                result['valid'] = False
                result['issues'].append(f"Region '{region}' not found")
                result['suggestions'].extend(self._suggest_similar_regions(region))
            elif matched_region not in self.variable_regions.get(matched_var, []):
                result['valid'] = False
                available_regions = sorted(self.variable_regions.get(matched_var, []))
                result['issues'].append(
                    f"No data for '{matched_var}' in region '{matched_region}'"
                )
                result['suggestions'].append(
                    f"Available regions for '{matched_var}': {', '.join(available_regions[:10])}"
                    + (f" ... and {len(available_regions) - 10} more" if len(available_regions) > 10 else "")
                )
        
        # Check scenario
        if scenario:
            matched_scenario = self._find_best_scenario_match(scenario)
            if matched_scenario and matched_scenario not in self.variable_scenarios.get(matched_var, []):
                available_scenarios = sorted(self.variable_scenarios.get(matched_var, []))
                result['issues'].append(
                    f"No data for '{matched_var}' in scenario '{matched_scenario}'"
                )
                result['suggestions'].append(
                    f"Available scenarios: {', '.join(available_scenarios[:5])}"
                )
        
        return result
    
    def suggest_variables(self, query: str, limit: int = 10) -> List[Tuple[str, int]]:
        """
        Suggest variables based on a query string.
        
        Args:
            query: User's query string
            limit: Maximum number of suggestions
            
        Returns:
            List of (variable_name, relevance_score) tuples
        """
        query_lower = query.lower()
        scores = {}
        
        # Common term mappings - map user terms to preferred variable patterns
        term_mappings = {
            'co2': ['emissions|co2', 'carbon dioxide'],
            'emissions': ['emissions'],
            'solar': ['solar', 'pv', 'photovoltaic'],
            'wind': ['wind'],
            'capacity': ['capacity'],
            'generation': ['generation', 'electricity'],
            'energy': ['energy'],
            'population': ['population'],
            'gdp': ['gdp|', 'gdp '],  # Prefer variables starting with GDP
            'investment': ['investment'],
            'transport': ['transport', 'transportation'],
            'industry': ['industrial', 'industry'],
            'building': ['building', 'residential', 'commercial'],
        }
        
        # Priority variables - common variables that should be preferred
        priority_patterns = [
            ('gdp', ['GDP|PPP', 'GDP|MER', 'GDP']),
            ('population', ['Population']),
            ('co2', ['Emissions|CO2']),
            ('solar', ['Capacity|Electricity|Solar', 'Electricity|Solar']),
            ('wind', ['Capacity|Electricity|Wind', 'Electricity|Wind']),
        ]
        
        # Expand query terms
        search_terms = set()
        for word in query_lower.split():
            search_terms.add(word)
            if word in term_mappings:
                search_terms.update(term_mappings[word])
        
        # Score each variable
        for variable in self.all_variables:
            var_lower = variable.lower()
            score = 0
            
            for term in search_terms:
                if term in var_lower:
                    # Exact term match
                    score += 10
                    # Bonus for exact word match
                    if f'|{term}|' in f'|{var_lower}|' or var_lower.startswith(f'{term}|'):
                        score += 5
                    # Bonus for end of variable name (more specific)
                    if var_lower.endswith(term):
                        score += 3
            
            # Bonus for priority variables
            for key, patterns in priority_patterns:
                if key in query_lower:
                    for pattern in patterns:
                        if var_lower.startswith(pattern.lower()):
                            score += 20  # Strong bonus for priority match
                        elif pattern.lower() in var_lower:
                            score += 10
            
            if score > 0:
                scores[variable] = score
        
        # Sort by score and return top matches
        sorted_vars = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_vars[:limit]
    
    def _find_best_variable_match(self, variable: str) -> Optional[str]:
        """Find the best matching variable name."""
        if variable in self.all_variables:
            return variable
        
        # Case-insensitive match
        var_lower = variable.lower()
        for v in self.all_variables:
            if v.lower() == var_lower:
                return v
        
        # Partial match
        for v in self.all_variables:
            if var_lower in v.lower() or v.lower() in var_lower:
                return v
        
        return None
    
    def _find_best_region_match(self, region: str) -> Optional[str]:
        """Find the best matching region name."""
        if region in self.all_regions:
            return region
        
        # Case-insensitive match
        region_lower = region.lower()
        for r in self.all_regions:
            if r.lower() == region_lower:
                return r
        
        # Common region name mappings
        region_mappings = {
            'world': ['World', 'GLOBAL', 'Global'],
            'eu': ['EU', 'European Union', 'EUR', 'Europe'],
            'europe': ['EU', 'European Union', 'EUR', 'Europe'],
            'usa': ['USA', 'United States', 'US'],
            'china': ['China', 'CHN'],
            'india': ['India', 'IND'],
            'greece': ['Greece', 'GR', 'GRC'],
        }
        
        if region_lower in region_mappings:
            for mapped in region_mappings[region_lower]:
                if mapped in self.all_regions:
                    return mapped
        
        # Partial match
        for r in self.all_regions:
            if region_lower in r.lower() or r.lower() in region_lower:
                return r
        
        return None
    
    def _find_best_scenario_match(self, scenario: str) -> Optional[str]:
        """Find the best matching scenario name."""
        if scenario in self.all_scenarios:
            return scenario
        
        # Case-insensitive match
        scenario_lower = scenario.lower()
        for s in self.all_scenarios:
            if s.lower() == scenario_lower:
                return s
        
        return None
    
    def _suggest_similar_variables(self, variable: str) -> List[str]:
        """Suggest similar variable names."""
        suggestions = self.suggest_variables(variable, limit=5)
        return [v for v, _ in suggestions]
    
    def _suggest_similar_regions(self, region: str) -> List[str]:
        """Suggest similar region names."""
        region_lower = region.lower()
        suggestions = []
        
        for r in self.all_regions:
            if region_lower in r.lower() or r.lower() in region_lower:
                suggestions.append(r)
        
        return sorted(suggestions)[:5]
    
    def get_summary(self) -> Dict[str, int]:
        """Get a summary of available data."""
        return {
            'total_variables': len(self.all_variables),
            'total_regions': len(self.all_regions),
            'total_scenarios': len(self.all_scenarios),
            'total_models': len(self.all_model_names),
            'categories': len(self.variable_categories)
        }
    
    def get_category_variables(self, category: str) -> List[str]:
        """Get all variables in a category."""
        return sorted(self.variable_categories.get(category, []))


def build_metadata_with_cache(ts_data: List[dict], models: List[dict] = None, 
                               cache_file: str = 'cache/data_metadata.pkl') -> DataMetadata:
    """
    Build or load DataMetadata with caching.
    
    Args:
        ts_data: Time series data records
        models: Model records (optional)
        cache_file: Path to cache file
        
    Returns:
        DataMetadata instance
    """
    # Check cache
    if os.path.exists(cache_file):
        logger.info(f"Loading metadata from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    # Build new metadata
    logger.info("Building new data metadata...")
    metadata = DataMetadata(ts_data, models)
    
    # Save to cache
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(metadata, f)
    logger.info(f"Metadata cached to: {cache_file}")
    
    return metadata
