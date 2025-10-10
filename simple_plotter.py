import re
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from typing import List, Dict, Any
from utils_query import match_variable_from_yaml, extract_region_from_query, find_closest_variable_name
from utils.yaml_loader import load_all_yaml_files

def simple_plot_query(question: str, model_data: List[Dict], ts_data: List[Dict]) -> str:
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

    # Extract region
    region = extract_region_from_query(q, [r.get('region', '') for r in ts_data if r])

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
        return f"No data found for variable '{variable}' with the specified filters (model: {model}, scenario: {scenario}, region: {region}). Try adjusting your query."

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

    # Plot
    plt.figure(figsize=(10, 6))
    for idx, row in df.iterrows():
        label = f"{row.get('model', 'Unknown')} - {row.get('scenario', 'Unknown')} - {row.get('region', 'Global')}"
        values = [row.get(str(year), 0) for year in sorted(year_cols, key=int)]
        plt.plot(sorted(year_cols, key=int), values, label=label, marker='o')

    plt.title(f"{variable} Projections")
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
