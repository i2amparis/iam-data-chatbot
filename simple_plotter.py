import re
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from typing import List, Dict, Any, Optional, Tuple
from utils_query import match_variable_from_yaml, extract_region_from_query, find_closest_variable_name, resolve_natural_language_variable_universal
from utils.yaml_loader import load_all_yaml_files
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# Global metadata instance (lazy loaded)
_metadata = None

def get_metadata(ts_data: List[Dict] = None, models: List[Dict] = None):
    """Get or create DataMetadata instance."""
    global _metadata
    if _metadata is None and ts_data is not None:
        from data_metadata import build_metadata_with_cache
        _metadata = build_metadata_with_cache(ts_data, models)
    return _metadata


def generate_llm_suggestion(query: str, variable: str, region: str, 
                            available_regions: List[str], available_scenarios: List[str],
                            api_key: str) -> str:
    """
    Use LLM to generate helpful suggestions when data is not found.
    
    Args:
        query: Original user query
        variable: Requested variable
        region: Requested region
        available_regions: List of available regions for the variable
        available_scenarios: List of available scenarios for the variable
        api_key: OpenAI API key
        
    Returns:
        Helpful suggestion message
    """
    llm = ChatOpenAI(
        model_name="gpt-4-turbo",
        temperature=0.7,
        api_key=api_key
    )
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("""You are a helpful assistant for the IAM PARIS climate data platform.

A user requested data that is not available. Generate a helpful, friendly response that:

1. Explains that the specific data combination is not available
2. Suggests similar alternatives from the available data
3. Offers to help the user find what they're looking for

Be concise and helpful. Use Markdown formatting.

## Context:
- Requested variable: {variable}
- Requested region: {region}
- Available regions for this variable: {available_regions}
- Available scenarios for this variable: {available_scenarios}

Generate a helpful response:"""),
        HumanMessagePromptTemplate.from_template("User query: {query}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({
        "query": query,
        "variable": variable,
        "region": region,
        "available_regions": ", ".join(available_regions[:20]) if available_regions else "None",
        "available_scenarios": ", ".join(available_scenarios[:10]) if available_scenarios else "None"
    })
    
    return response.content


def detect_multi_variable_comparison(query: str) -> List[str]:
    """
    Detect if query is asking to compare multiple variables.
    
    Args:
        query: User query string
        
    Returns:
        List of variable keywords found in comparison context
    """
    query_lower = query.lower()
    
    # Patterns that indicate multi-variable comparison
    comparison_patterns = [
        r'compare\s+(\w+)\s+(?:and|vs|versus|with)\s+(\w+)',
        r'(\w+)\s+(?:and|vs|versus)\s+(\w+)\s+(?:capacity|generation|energy|emissions)',
        r'(\w+)\s+vs\s+(\w+)',  # Simple "X vs Y" pattern
        r'both\s+(\w+)\s+and\s+(\w+)',
        r'(\w+)\s+or\s+(\w+)',
    ]
    
    for pattern in comparison_patterns:
        match = re.search(pattern, query_lower)
        if match:
            return [match.group(1), match.group(2)]
    
    return []


def plot_multiple_variables(question: str, model_data: List[Dict], ts_data: List[Dict], 
                            variables: List[str], region: str = None, 
                            scenario: str = None, start_year: int = None, 
                            end_year: int = None) -> str:
    """
    Generate a plot comparing multiple variables.
    
    Args:
        question: Original user query
        model_data: List of model metadata
        ts_data: List of time series data
        variables: List of variable names to compare (can be keywords or exact names)
        region: Optional region filter
        scenario: Optional scenario filter
        start_year: Optional start year for filtering
        end_year: Optional end year for filtering
        
    Returns:
        Base64 encoded PNG image or error message
    """
    metadata = get_metadata(ts_data, model_data)
    
    # Check if variables are already exact names (from LLM extraction)
    # or if they need to be resolved (from regex detection)
    resolved_variables = []
    for var in variables:
        # Check if it's an exact variable name
        exact_match = False
        for r in ts_data:
            if r and r.get('variable') == var:
                resolved_variables.append(var)
                exact_match = True
                print(f"DEBUG: Using exact variable: '{var}'")
                break
        
        # If not exact, try to resolve using metadata
        if not exact_match and metadata:
            suggestions = metadata.suggest_variables(var, limit=3)
            if suggestions:
                resolved_variables.append(suggestions[0][0])
                print(f"DEBUG: Resolved '{var}' to '{suggestions[0][0]}'")
    
    if len(resolved_variables) < 2:
        return f"Could not identify enough variables to compare. Found: {resolved_variables}"
    
    # Extract region from query if not provided
    if region is None:
        # Check for common region names in query first
        region_keywords = ['world', 'europe', 'eu', 'usa', 'china', 'india', 'africa', 'asia', 'greece', 'germany', 'brazil']
        question_lower = question.lower()
        for kw in region_keywords:
            if kw in question_lower:
                if metadata:
                    matched = metadata._find_best_region_match(kw)
                    if matched:
                        region = matched
                        break
                else:
                    region = kw.title()
                    break
        
        # Fallback: Use metadata to find region in query
        if region is None and metadata:
            region = metadata._find_best_region_match(question)
    
    print(f"DEBUG: Using region: {region}")
    
    # Collect data for each variable
    all_data = {}
    units = {}
    
    for variable in resolved_variables:
        filtered_data = []
        for r in ts_data:
            if r is None:
                continue
            if str(r.get('variable', '')) != variable:
                continue
            if scenario and r.get('scenario') != scenario:
                continue
            if region:
                r_region = str(r.get('region', ''))
                if r_region.lower() != region.lower():
                    continue
            filtered_data.append(r)
        
        if filtered_data:
            all_data[variable] = filtered_data
            units[variable] = filtered_data[0].get('unit', '')
    
    if not all_data:
        return f"No data found for the requested variables in region '{region}'."
    
    # Create comparison plot
    plt.figure(figsize=(12, 7))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'D', 'v', '<']
    
    for idx, (variable, data) in enumerate(all_data.items()):
        df = pd.DataFrame(data)
        
        # Handle years column
        if 'years' in df.columns:
            years_df = df['years'].apply(pd.Series)
            df = df.drop('years', axis=1).join(years_df)
        
        # Get year columns
        year_cols = [col for col in df.columns if str(col).isdigit()]
        if not year_cols:
            continue
        
        # Filter to specific year range if requested
        if start_year or end_year:
            filtered_year_cols = []
            for col in year_cols:
                year_int = int(col)
                if start_year and year_int < start_year:
                    continue
                if end_year and year_int > end_year:
                    continue
                filtered_year_cols.append(col)
            if filtered_year_cols:
                year_cols = filtered_year_cols
        
        # Aggregate if multiple rows (take mean)
        if len(df) > 1:
            # Group by scenario and take first of each
            df = df.groupby('scenario').first().reset_index()
        
        # Plot each row
        for _, row in df.iterrows():
            label = variable
            if len(df) > 1:
                label = f"{variable} ({row.get('scenario', '')})"
            
            values = [row.get(str(year), 0) for year in sorted(year_cols, key=int)]
            plt.plot(sorted(year_cols, key=int), values, 
                    label=label, 
                    color=colors[idx % len(colors)],
                    marker=markers[idx % len(markers)],
                    linewidth=2)
    
    # Build title
    title = f"Comparison: {' vs '.join([v.split('|')[-1] if '|' in v else v for v in all_data.keys()])}"
    if region:
        title += f" for {region}"
    
    plt.title(title, fontsize=12, fontweight='bold')
    plt.xlabel("Year", fontsize=10)
    
    # Use first unit as Y-axis label (assuming same units)
    if units:
        first_unit = list(units.values())[0]
        plt.ylabel(f"Value ({first_unit})", fontsize=10)
    
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save to base64
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return f"![Plot](data:image/png;base64,{img_base64})"


def plot_model_comparison(question: str, model_data: List[Dict], ts_data: List[Dict],
                          variable: str, models: List[str], region: str = None,
                          scenario: str = None, start_year: int = None, 
                          end_year: int = None) -> str:
    """
    Generate a plot comparing the same variable across different models.
    
    Args:
        question: Original user query
        model_data: List of model metadata
        ts_data: List of time series data
        variable: Variable name to compare
        models: List of model names to compare
        region: Optional region filter
        scenario: Optional scenario filter
        start_year: Optional start year for filtering
        end_year: Optional end year for filtering
        
    Returns:
        Base64 encoded PNG image or error message
    """
    metadata = get_metadata(ts_data, model_data)
    
    # Resolve variable name if needed
    resolved_variable = None
    for r in ts_data:
        if r and r.get('variable') == variable:
            resolved_variable = variable
            break
    
    if not resolved_variable and metadata:
        suggestions = metadata.suggest_variables(variable, limit=3)
        if suggestions:
            resolved_variable = suggestions[0][0]
            print(f"DEBUG: Resolved variable '{variable}' to '{resolved_variable}'")
    
    if not resolved_variable:
        return f"Could not identify variable '{variable}'."
    
    # Resolve model names (fuzzy match)
    # Note: ts_data uses 'modelName' field, not 'model'
    resolved_models = []
    available_models = sorted({str(r.get('modelName', '') or r.get('model', '')) for r in ts_data if r and (r.get('modelName') or r.get('model'))})
    print(f"DEBUG: ts_data length: {len(ts_data)}")
    print(f"DEBUG: Available models in ts_data: {available_models[:20]}...")
    print(f"DEBUG: Looking for models: {models}")
    
    # If ts_data is empty, try to get models from model_data
    if not available_models and model_data:
        available_models = sorted({str(m.get('modelName', '')) for m in model_data if m and m.get('modelName')})
        print(f"DEBUG: Using model_data instead. Available models: {available_models[:20]}...")
    
    for model_name in models:
        # Try exact match first
        if model_name in available_models:
            resolved_models.append(model_name)
            print(f"DEBUG: Using exact model: '{model_name}'")
            continue
        
        # Try case-insensitive match
        for avail in available_models:
            if avail.lower() == model_name.lower():
                resolved_models.append(avail)
                print(f"DEBUG: Matched model '{model_name}' to '{avail}'")
                break
        else:
            # Try partial match
            for avail in available_models:
                if model_name.lower() in avail.lower() or avail.lower() in model_name.lower():
                    resolved_models.append(avail)
                    print(f"DEBUG: Partial matched model '{model_name}' to '{avail}'")
                    break
    
    if len(resolved_models) < 2:
        return f"Could not identify enough models to compare. Found: {resolved_models}. Available models include: {', '.join(available_models[:10])}..."
    
    # Extract region from query if not provided
    if region is None and metadata:
        region = metadata._find_best_region_match(question)
    
    print(f"DEBUG: Model comparison - variable: {resolved_variable}, models: {resolved_models}, region: {region}")
    
    # Collect data for each model
    all_data = {}
    units = {}
    
    for model_name in resolved_models:
        filtered_data = []
        for r in ts_data:
            if r is None:
                continue
            if str(r.get('variable', '')) != resolved_variable:
                continue
            # Use modelName field (the actual field name in ts_data)
            r_model = str(r.get('modelName', '') or r.get('model', ''))
            if r_model != model_name:
                continue
            if scenario and r.get('scenario') != scenario:
                continue
            if region:
                r_region = str(r.get('region', ''))
                if r_region.lower() != region.lower():
                    continue
            filtered_data.append(r)
        
        if filtered_data:
            all_data[model_name] = filtered_data
            units[model_name] = filtered_data[0].get('unit', '')
    
    if not all_data:
        return f"No data found for '{resolved_variable}' across models {resolved_models} in region '{region}'."
    
    # Create comparison plot
    plt.figure(figsize=(12, 7))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'D', 'v', '<']
    
    for idx, (model_name, data) in enumerate(all_data.items()):
        df = pd.DataFrame(data)
        
        # Handle years column
        if 'years' in df.columns:
            years_df = df['years'].apply(pd.Series)
            df = df.drop('years', axis=1).join(years_df)
        
        # Get year columns
        year_cols = [col for col in df.columns if str(col).isdigit()]
        if not year_cols:
            continue
        
        # Filter to specific year range if requested
        if start_year or end_year:
            filtered_year_cols = []
            for col in year_cols:
                year_int = int(col)
                if start_year and year_int < start_year:
                    continue
                if end_year and year_int > end_year:
                    continue
                filtered_year_cols.append(col)
            if filtered_year_cols:
                year_cols = filtered_year_cols
        
        # Aggregate if multiple rows (take mean by scenario)
        if len(df) > 1:
            # Group by scenario and take first of each
            df = df.groupby('scenario').first().reset_index()
        
        # Plot each row
        for _, row in df.iterrows():
            label = model_name
            if len(df) > 1:
                label = f"{model_name} ({row.get('scenario', '')})"
            
            values = [row.get(str(year), 0) for year in sorted(year_cols, key=int)]
            plt.plot(sorted(year_cols, key=int), values, 
                    label=label, 
                    color=colors[idx % len(colors)],
                    marker=markers[idx % len(markers)],
                    linewidth=2)
    
    # Build title
    title = f"Model Comparison: {resolved_variable}"
    if region:
        title += f" for {region}"
    if start_year or end_year:
        title += f" ({start_year or '?'}-{end_year or '?'})"
    
    plt.title(title, fontsize=12, fontweight='bold')
    plt.xlabel("Year", fontsize=10)
    
    # Use first unit as Y-axis label
    if units:
        first_unit = list(units.values())[0]
        plt.ylabel(f"{resolved_variable} ({first_unit})", fontsize=10)
    
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save to base64
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return f"![Plot](data:image/png;base64,{img_base64})"


def simple_plot_query_with_entities(question: str, model_data: List[Dict], ts_data: List[Dict], 
                                     entities: Dict[str, Any], region: str = None) -> str:
    """
    Generate a plot using pre-extracted entities for better accuracy.
    
    Args:
        question: Original user query
        model_data: List of model metadata
        ts_data: List of time series data
        entities: Pre-extracted entities from QueryEntityExtractor
        region: Optional region override
    
    Returns:
        Base64 encoded PNG image or error message
    """
    # Check for multi-variable comparison from LLM-extracted entities
    variables_list = entities.get('variables')
    models_list = entities.get('models')
    comparison_type = entities.get('comparison')
    
    # Check for model comparison
    if models_list and len(models_list) >= 2:
        print(f"DEBUG: LLM detected multi-model comparison: {models_list}")
        variable = entities.get('variable')
        scenario = entities.get('scenario')
        region_from_entities = entities.get('region')
        start_year = entities.get('start_year')
        end_year = entities.get('end_year')
        return plot_model_comparison(question, model_data, ts_data, variable, models_list,
                                    region or region_from_entities, scenario,
                                    start_year, end_year)
    
    if variables_list and len(variables_list) >= 2:
        print(f"DEBUG: LLM detected multi-variable comparison: {variables_list}")
        scenario = entities.get('scenario')
        region_from_entities = entities.get('region')
        start_year = entities.get('start_year')
        end_year = entities.get('end_year')
        return plot_multiple_variables(question, model_data, ts_data, variables_list, 
                                       region or region_from_entities, scenario,
                                       start_year, end_year)
    
    # Fallback: Check for multi-variable comparison using regex patterns
    comparison_vars = detect_multi_variable_comparison(question)
    if len(comparison_vars) >= 2:
        print(f"DEBUG: Regex detected multi-variable comparison: {comparison_vars}")
        scenario = entities.get('scenario')
        start_year = entities.get('start_year')
        end_year = entities.get('end_year')
        return plot_multiple_variables(question, model_data, ts_data, comparison_vars, region, scenario,
                                       start_year, end_year)
    
    # Use extracted entities directly
    variable = entities.get('variable')
    scenario = entities.get('scenario')
    model = entities.get('model')
    start_year = entities.get('start_year')
    end_year = entities.get('end_year')
    unit = entities.get('unit')
    comparison = entities.get('comparison')
    
    # Use region from entities if not overridden
    if region is None:
        region = entities.get('region')
    
    # If no variable extracted, fall back to keyword extraction
    if not variable:
        from pathlib import Path
        variable_path = Path('definitions/variable').resolve()
        variable_dict = load_all_yaml_files(str(variable_path))
        
        natural_variable = resolve_natural_language_variable_universal(question, variable_dict)
        if natural_variable and isinstance(natural_variable, str):
            available_vars = {str(r.get('variable', '')) for r in ts_data if r and r.get('variable') is not None}
            if natural_variable in available_vars:
                variable = natural_variable
    
    # Try metadata-based variable matching if still no match
    if not variable:
        metadata = get_metadata(ts_data, model_data)
        if metadata:
            suggestions = metadata.suggest_variables(question, limit=5)
            if suggestions:
                # Use the highest-scored variable
                variable = suggestions[0][0]
                print(f"DEBUG: Metadata suggested variable: {variable} (score: {suggestions[0][1]})")
    
    if not variable:
        # Final attempt: use metadata to suggest similar variables
        metadata = get_metadata(ts_data, model_data)
        if metadata:
            similar = metadata._suggest_similar_variables(question)
            if similar:
                return f"Could not identify a variable to plot. Did you mean: {', '.join(similar[:3])}?"
        return "Could not identify a variable to plot. Please specify a variable like 'solar capacity' or 'CO2 emissions'."
    
    # Filter data using extracted entities
    filtered_data = []
    for r in ts_data:
        if r is None:
            continue
        if str(r.get('variable', '')) != variable:
            continue
        if model and r.get('model') != model:
            continue
        if scenario and r.get('scenario') != scenario:
            continue
        # Case-insensitive region matching
        if region:
            r_region = str(r.get('region', ''))
            if r_region.lower() != region.lower():
                continue
        filtered_data.append(r)
    
    if not filtered_data:
        # Provide helpful suggestions using LLM
        available_regions = sorted(set(str(r.get('region', '')) for r in ts_data if r and r.get('region') and r.get('variable') == variable))
        available_scenarios = sorted(set(str(r.get('scenario', '')) for r in ts_data if r and r.get('scenario') and r.get('variable') == variable))
        
        # Use LLM to generate helpful suggestions
        try:
            import os
            api_key = os.environ.get('OPENAI_API_KEY')
            if api_key:
                return generate_llm_suggestion(
                    query=question,
                    variable=variable,
                    region=region or "not specified",
                    available_regions=available_regions,
                    available_scenarios=available_scenarios,
                    api_key=api_key
                )
        except Exception as e:
            print(f"DEBUG: LLM suggestion failed: {e}")
        
        # Fallback to basic suggestions
        suggestions = []
        if available_regions:
            suggestions.append(f"Available regions for '{variable}': {', '.join(available_regions[:5])}")
        if available_scenarios:
            suggestions.append(f"Available scenarios: {', '.join(available_scenarios[:5])}")
        
        suggestion_text = " | ".join(suggestions) if suggestions else "Try 'list variables' to see available options."
        return f"No data found for variable '{variable}'. {suggestion_text}"
    
    # Get unit from data if not in entities
    if not unit:
        unit = filtered_data[0].get('unit', '')
    
    # Prepare data for plotting
    df = pd.DataFrame(filtered_data)
    if 'years' in df.columns:
        years_df = df['years'].apply(pd.Series)
        df = df.drop('years', axis=1).join(years_df)
    
    # Get year columns
    year_cols = [col for col in df.columns if str(col).isdigit()]
    if not year_cols:
        return "No time series data available for plotting."
    
    # Filter to specific year range if requested
    if start_year or end_year:
        filtered_year_cols = []
        for col in year_cols:
            year_int = int(col)
            if start_year and year_int < start_year:
                continue
            if end_year and year_int > end_year:
                continue
            filtered_year_cols.append(col)
        if filtered_year_cols:
            year_cols = filtered_year_cols
            print(f"DEBUG: Filtered years to range {start_year}-{end_year}: {year_cols[:5]}...{year_cols[-5:] if len(year_cols) > 5 else ''}")
        else:
            print(f"DEBUG: No years in range {start_year}-{end_year}, using all years")
    
    # Create plot
    plt.figure(figsize=(12, 7))
    
    # Determine how to group data based on comparison type or data variety
    scenarios_in_data = df['scenario'].unique()
    regions_in_data = df['region'].unique() if 'region' in df.columns else ['All']
    models_in_data = df['model'].unique() if 'model' in df.columns else ['All']
    
    # Choose grouping strategy
    if comparison == 'scenario' or (len(scenarios_in_data) > 1 and len(regions_in_data) == 1):
        # Group by scenario
        for scenario_name in scenarios_in_data:
            scenario_data = df[df['scenario'] == scenario_name]
            if not scenario_data.empty:
                row = scenario_data.iloc[0]
                label = f"{scenario_name}"
                values = [row.get(str(year), 0) for year in sorted(year_cols, key=int)]
                plt.plot(sorted(year_cols, key=int), values, label=label, marker='o', linewidth=2)
    
    elif comparison == 'region' or (len(regions_in_data) > 1 and len(scenarios_in_data) == 1):
        # Group by region
        for region_name in regions_in_data:
            region_data = df[df['region'] == region_name]
            if not region_data.empty:
                row = region_data.iloc[0]
                label = f"{region_name}"
                values = [row.get(str(year), 0) for year in sorted(year_cols, key=int)]
                plt.plot(sorted(year_cols, key=int), values, label=label, marker='o', linewidth=2)
    
    elif comparison == 'model' or (len(models_in_data) > 1):
        # Group by model
        for model_name in models_in_data:
            model_data_row = df[df['model'] == model_name]
            if not model_data_row.empty:
                row = model_data_row.iloc[0]
                label = f"{model_name}"
                values = [row.get(str(year), 0) for year in sorted(year_cols, key=int)]
                plt.plot(sorted(year_cols, key=int), values, label=label, marker='o', linewidth=2)
    
    else:
        # Default: plot each row
        for idx, row in df.iterrows():
            label_parts = []
            if 'model' in row:
                label_parts.append(str(row['model']))
            if 'scenario' in row:
                label_parts.append(str(row['scenario']))
            if 'region' in row:
                label_parts.append(str(row['region']))
            label = " - ".join(label_parts) if label_parts else f"Series {idx+1}"
            
            values = [row.get(str(year), 0) for year in sorted(year_cols, key=int)]
            plt.plot(sorted(year_cols, key=int), values, label=label, marker='o')
    
    # Build title with context
    title_parts = [variable]
    if scenario and len(scenarios_in_data) == 1:
        title_parts.append(f"({scenario})")
    if region and len(regions_in_data) == 1:
        title_parts.append(f"- {region}")
    if start_year or end_year:
        year_range = f"({start_year or '?'}-{end_year or '?'})"
        title_parts.append(year_range)
    
    plt.title(" ".join(title_parts), fontsize=12, fontweight='bold')
    plt.xlabel("Year", fontsize=10)
    
    # Use unit in Y-axis label
    ylabel = variable
    if unit:
        ylabel = f"{variable} ({unit})"
    plt.ylabel(ylabel, fontsize=10)
    
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save to base64
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return f"![Plot](data:image/png;base64,{img_base64})"


def simple_plot_query(question: str, model_data: List[Dict], ts_data: List[Dict], region: str = None) -> str:
    """
    Generate a plot based on natural language query.
    
    Args:
        question: User's natural language query
        model_data: List of model metadata
        ts_data: List of time series data
        region: Optional region filter
    
    
    Returns:
        Base64 encoded PNG image or error message
    """
    from pathlib import Path
    from utils_query import extract_region_from_query, find_closest_variable_name, resolve_natural_language_variable_universal
    
    # Check for multi-variable comparison first
    comparison_vars = detect_multi_variable_comparison(question)
    if len(comparison_vars) >= 2:
        print(f"DEBUG: Detected multi-variable comparison: {comparison_vars}")
        return plot_multiple_variables(question, model_data, ts_data, comparison_vars, region)
    
    # Extract region from query if not provided
    if region is None:
        region = extract_region_from_query(question)
    
    # Load variable definitions
    variable_path = Path('definitions/variable').resolve()
    variable_dict = load_all_yaml_files(str(variable_path))
    
    # Try to match variable from query
    variable = None
    
    # First try natural language resolution
    natural_variable = resolve_natural_language_variable_universal(question, variable_dict)
    if natural_variable and isinstance(natural_variable, str):
        available_vars = {str(r.get('variable', '')) for r in ts_data if r and r.get('variable') is not None}
        if natural_variable in available_vars:
            variable = natural_variable
    
    # Fall back to keyword matching
    if not variable:
        variable = match_variable_from_yaml(question, variable_dict)
    
    # Try to find closest match
    if not variable:
        available_vars = sorted(set(str(r.get('variable', '')) for r in ts_data if r and r.get('variable')))
        variable = find_closest_variable_name(question, available_vars)
    
    # Try metadata-based variable matching if still no match
    if not variable:
        metadata = get_metadata(ts_data, model_data)
        if metadata:
            suggestions = metadata.suggest_variables(question, limit=5)
            if suggestions:
                variable = suggestions[0][0]
                print(f"DEBUG: Metadata suggested variable: {variable} (score: {suggestions[0][1]})")
    
    if not variable:
        # Final attempt: use metadata to suggest similar variables
        metadata = get_metadata(ts_data, model_data)
        if metadata:
            similar = metadata._suggest_similar_variables(question)
            if similar:
                return f"Could not identify a variable to plot. Did you mean: {', '.join(similar[:3])}?"
        return "Could not identify a variable to plot. Please specify a variable like 'solar capacity' or 'CO2 emissions'."
    
    # Extract scenario from query if mentioned
    scenario = None
    scenario_keywords = ['ssp1', 'ssp2', 'ssp3', 'ssp4', 'ssp5', 'rcp', 'scenario']
    question_lower = question.lower()
    for kw in scenario_keywords:
        if kw in question_lower:
            # Try to find matching scenario in data
            for r in ts_data:
                if r and r.get('scenario') and kw in str(r.get('scenario', '')).lower():
                    scenario = r.get('scenario')
                    break
            if scenario:
                break
    
    # Filter data
    filtered_data = []
    for r in ts_data:
        if r is None:
            continue
        if str(r.get('variable', '')) != variable:
            continue
        if scenario and r.get('scenario') != scenario:
            continue
        if region and r.get('region') != region:
            continue
        filtered_data.append(r)
    
    if not filtered_data:
        # Provide helpful suggestions using LLM
        available_regions = sorted(set(str(r.get('region', '')) for r in ts_data if r and r.get('region') and r.get('variable') == variable))
        available_scenarios = sorted(set(str(r.get('scenario', '')) for r in ts_data if r and r.get('scenario') and r.get('variable') == variable))
        
        # Use LLM to generate helpful suggestions
        try:
            import os
            api_key = os.environ.get('OPENAI_API_KEY')
            if api_key:
                return generate_llm_suggestion(
                    query=question,
                    variable=variable,
                    region=region or "not specified",
                    available_regions=available_regions,
                    available_scenarios=available_scenarios,
                    api_key=api_key
                )
        except Exception as e:
            print(f"DEBUG: LLM suggestion failed: {e}")
        
        # Fallback to basic suggestions
        suggestions = []
        if available_regions:
            suggestions.append(f"Available regions for '{variable}': {', '.join(available_regions[:5])}")
        if available_scenarios:
            suggestions.append(f"Available scenarios: {', '.join(available_scenarios[:5])}")
        
        suggestion_text = " | ".join(suggestions) if suggestions else "Try 'list variables' to see available options."
        return f"No data found for variable '{variable}'. {suggestion_text}"
    
    # Get unit from data
    unit = filtered_data[0].get('unit', '')
    
    # Prepare data for plotting
    df = pd.DataFrame(filtered_data)
    if 'years' in df.columns:
        years_df = df['years'].apply(pd.Series)
        df = df.drop('years', axis=1).join(years_df)
    
    # Get year columns
    year_cols = [col for col in df.columns if str(col).isdigit()]
    if not year_cols:
        return "No time series data available for plotting."
    
    # Create plot
    plt.figure(figsize=(12, 7))
    
    # Determine grouping strategy
    scenarios_in_data = df['scenario'].unique() if 'scenario' in df.columns else ['All']
    regions_in_data = df['region'].unique() if 'region' in df.columns else ['All']
    models_in_data = df['model'].unique() if 'model' in df.columns else ['All']
    
    # Choose grouping based on data variety
    if len(scenarios_in_data) > 1 and len(regions_in_data) == 1:
        # Group by scenario
        for scenario_name in scenarios_in_data:
            scenario_data = df[df['scenario'] == scenario_name]
            if not scenario_data.empty:
                row = scenario_data.iloc[0]
                label = f"{scenario_name}"
                values = [row.get(str(year), 0) for year in sorted(year_cols, key=int)]
                plt.plot(sorted(year_cols, key=int), values, label=label, marker='o', linewidth=2)
    
    elif len(regions_in_data) > 1 and len(scenarios_in_data) == 1:
        # Group by region
        for region_name in regions_in_data:
            region_data = df[df['region'] == region_name]
            if not region_data.empty:
                row = region_data.iloc[0]
                label = f"{region_name}"
                values = [row.get(str(year), 0) for year in sorted(year_cols, key=int)]
                plt.plot(sorted(year_cols, key=int), values, label=label, marker='o', linewidth=2)
    
    elif len(models_in_data) > 1:
        # Group by model
        for model_name in models_in_data:
            model_data_row = df[df['model'] == model_name]
            if not model_data_row.empty:
                row = model_data_row.iloc[0]
                label = f"{model_name}"
                values = [row.get(str(year), 0) for year in sorted(year_cols, key=int)]
                plt.plot(sorted(year_cols, key=int), values, label=label, marker='o', linewidth=2)
    
    else:
        # Default: plot each row
        for idx, row in df.iterrows():
            label_parts = []
            if 'model' in row:
                label_parts.append(str(row['model']))
            if 'scenario' in row:
                label_parts.append(str(row['scenario']))
            if 'region' in row:
                label_parts.append(str(row['region']))
            label = " - ".join(label_parts) if label_parts else f"Series {idx+1}"
            
            values = [row.get(str(year), 0) for year in sorted(year_cols, key=int)]
            plt.plot(sorted(year_cols, key=int), values, label=label, marker='o')
    
    # Build title
    title_parts = [variable]
    if scenario and len(scenarios_in_data) == 1:
        title_parts.append(f"({scenario})")
    if region and len(regions_in_data) == 1:
        title_parts.append(f"- {region}")
    
    plt.title(" ".join(title_parts), fontsize=12, fontweight='bold')
    plt.xlabel("Year", fontsize=10)
    
    # Use unit in Y-axis label
    ylabel = variable
    if unit:
        ylabel = f"{variable} ({unit})"
    plt.ylabel(ylabel, fontsize=10)
    
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save to base64
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return f"![Plot](data:image/png;base64,{img_base64})"
