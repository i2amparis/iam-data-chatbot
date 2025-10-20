import re
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from typing import List, Dict, Any
from utils_query import match_variable_from_yaml, extract_region_from_query, find_closest_variable_name
from utils.yaml_loader import load_all_yaml_files

def simple_plot_query(question: str, model_data: List[Dict], ts_data: List[Dict], region: str = None) -> str:
    """
    Generate a simple plot based on the user query.
    Filters data by variable, model, scenario, region if specified.
    Returns base64 encoded PNG image or error message.
    """
    # Load variable definitions
    from pathlib import Path
    variable_path = Path('definitions/variable').resolve()
    variable_dict = load_all_yaml_files(str(variable_path))

    q = question.lower()

    # Extract filters from query
    variable_match = match_variable_from_yaml(q, variable_dict)
    if variable_match['match_type'] in ['exact', 'fuzzy']:
        variable = variable_match['matched_variable']
    else:
        # Fallback to closest match from ts_data
        ts_vars = list({r.get('variable', '') for r in ts_data if r and r.get('variable')})
        variable = find_closest_variable_name(q, ts_vars)
        if not variable:
            return "Could not identify a variable to plot. Try specifying a variable like 'CO2 emissions' or 'energy consumption'."

    # Extract model
    model = None
    for m in model_data:
        if m and m.get('modelName', '').lower() in q:
            model = m['modelName']
            break

    # Extract scenario
    scenario = None
    for r in ts_data:
        if r and r.get('scenario', '').lower() in q.replace('_', ' '):
            scenario = r['scenario']
            break

    # Use provided region if available, otherwise extract from query
    if region is None:
        # Extract region from actual data instead of region definitions
        regions_in_data = {r.get('region', '') for r in ts_data if r and r.get('region')}
        for reg in regions_in_data:
            if reg.lower() in q.lower():
                region = reg
                break

    # Filter data
    filtered_data = []
    for r in ts_data:
        if r is None:
            continue
        if r.get('variable') != variable:
            continue
        if model and r.get('model') != model:
            continue
        if scenario and r.get('scenario') != scenario:
            continue
        if region and r.get('region') != region:
            continue
        filtered_data.append(r)

    if not filtered_data:
        available_regions = sorted(set(r.get('region', '') for r in ts_data if r and r.get('region')))
        available_scenarios = sorted(set(r.get('scenario', '') for r in ts_data if r and r.get('scenario')))
        available_models = sorted(set(r.get('model', '') for r in ts_data if r and r.get('model')))

        suggestions = []
        if region and region not in available_regions:
            suggestions.append(f"Try a different region. Available regions include: {', '.join(available_regions[:5])}...")
        if scenario and scenario not in available_scenarios:
            suggestions.append(f"Try a different scenario. Available scenarios include: {', '.join(available_scenarios[:3])}...")
        if model and model not in available_models:
            suggestions.append(f"Try a different model. Available models include: {', '.join(available_models[:3])}...")

        if suggestions:
            suggestion_text = " ".join(suggestions)
        else:
            suggestion_text = "Try using 'list variables' to see available options, or try a different variable name."

        return f"No data found for variable '{variable}' with the specified filters (model: {model}, scenario: {scenario}, region: {region}). {suggestion_text}"

    # Prepare data for plotting
    df = pd.DataFrame(filtered_data)
    if 'years' in df.columns:
        # Handle nested years
        years_df = df['years'].apply(pd.Series)
        df = df.drop('years', axis=1).join(years_df)
    
    # Get year columns
    year_cols = [col for col in df.columns if str(col).isdigit()]
    if not year_cols:
        return "No time series data available for plotting."

    # Extract specific year from query if mentioned
    specific_year = None
    year_match = re.search(r'\b(20\d{2})\b', q)
    if year_match:
        specific_year = year_match.group(1)
        if specific_year in year_cols:
            year_cols = [specific_year]

    # Plot
    plt.figure(figsize=(10, 6))

    # Group data by scenario for better visualization when multiple scenarios exist
    if len(df['scenario'].unique()) > 1:
        # Multiple scenarios - plot each scenario as a separate line
        for scenario_name in df['scenario'].unique():
            scenario_data = df[df['scenario'] == scenario_name]
            if not scenario_data.empty:
                # Use the first row for this scenario (assuming all rows for same scenario have same data)
                row = scenario_data.iloc[0]
                label = f"{scenario_name}"
                values = [row.get(str(year), 0) for year in sorted(year_cols, key=int)]
                plt.plot(sorted(year_cols, key=int), values, label=label, marker='o', linewidth=2)
    else:
        # Single scenario - plot normally
        for idx, row in df.iterrows():
            label = f"{row.get('model', 'Unknown')} - {row.get('scenario', 'Unknown')} - {row.get('region', 'Global')}"
            values = [row.get(str(year), 0) for year in sorted(year_cols, key=int)]
            plt.plot(sorted(year_cols, key=int), values, label=label, marker='o')

    # Update title to reflect filters applied
    title_suffix = ""
    if scenario:
        title_suffix += f" - {scenario}"
    if region:
        title_suffix += f" ({region})"
    if specific_year:
        title_suffix += f" - {specific_year}"

    plt.title(f"{variable} Projections{title_suffix}")
    plt.xlabel("Year")
    plt.ylabel(variable)
    plt.legend()
    plt.grid(True)

    # Save to base64
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return f"![Plot](data:image/png;base64,{img_base64})"
