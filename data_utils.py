import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime
import logging
import base64
from io import BytesIO

from simple_plotter import simple_plot_query
from utils_query import match_variable_from_yaml, extract_examples_from_data, get_available_workspaces, extract_variable_and_region_from_query
from utils.yaml_loader import load_all_yaml_files


# Load variable and region definitions from YAML files
variable_dict = load_all_yaml_files('definitions/variable')
region_dict = load_all_yaml_files('definitions/region')


def data_query(question: str, model_data: list, ts_data: list) -> str:
    """Process a user query about IAM data, optionally returning results or plots."""
    q = question.lower()

    # -------------------------------
    # Handle PLOTTING QUERIES
    # -------------------------------
    if any(word in q for word in ['plot', 'show', 'graph', 'visualize']):
        # Use data-based matching as primary method (more reliable than YAML)
        return simple_plot_query(question, model_data, ts_data)

    # -------------------------------
    # LIST AVAILABLE MODELS
    # -------------------------------
    if re.search(r"\b(list|available|what)\b.*\bmodels?\b", q) or re.search(r"\bmodels?\b.*\b(available|list|what|do you have)\b", q) or re.search(r"\bwhat.*models?\b", q):
        models = sorted({r.get('modelName', '') for r in model_data if r and r.get('modelName')})
        if not models:
            return "I couldn't find any models in the data right now. Try `help` or refresh the data."

        if len(models) <= 6:
            model_str = ", ".join(models[:-1]) + (" and " + models[-1] if len(models) > 1 else models[0])
            return f"I found these models in the IAM PARIS dataset: {model_str}. Which one would you like to know more about?"

        return (f"There are {len(models)} models available. "
                "You can ask for details about a specific model using `info [model name]`, "
                "or say `list variables` to see the kinds of outputs available.")

    # -------------------------------
    # LIST AVAILABLE VARIABLES
    # -------------------------------
    if 'list variables' in q or 'show me variables' in q:
        vars = sorted({r.get('variable', '') for r in ts_data if r and r.get('variable')})
        if not vars:
            return "I don't see any variables in the loaded dataset. Try reloading or check the IAM PARIS results website."

        # Filter for energy-related variables if "energy" is mentioned
        if 'energy' in q:
            energy_vars = [v for v in vars if any(term in v.lower() for term in ['energy', 'electricity', 'capacity', 'power', 'generation', 'solar', 'wind', 'hydro', 'gas', 'nuclear', 'biomass'])]
            if energy_vars:
                vars = energy_vars[:15]  # Show more energy variables
            else:
                vars = vars[:8]

        sample = vars[:12] if len(vars) > 8 else vars
        more = "" if len(vars) <= len(sample) else f" and {len(vars)-len(sample)} more"
        sample_str = "\n- ".join(sample)
        return (f"I can work with these variables:\n- {sample_str}{more}\n\n"
                "Try queries like 'Capacity|Electricity|Solar|Utility for Greece' or 'plot [variable name] in Greece'.")

    # -------------------------------
    # LIST AVAILABLE SCENARIOS
    # -------------------------------
    if re.search(r"\b(list|available|what)\b.*\bscenarios?\b", q) or re.search(r"\bscenarios?\b.*\b(available|included|list|what|are there)\b", q) or re.search(r"\bwhat.*scenarios?\b", q):
        scenarios = sorted({r.get('scenario', '') for r in ts_data if r and r.get('scenario')})
        if not scenarios:
            return "No scenarios are loaded in the current dataset. Try a different query or check IAM PARIS results."

        sample = scenarios[:8]
        more = "" if len(scenarios) <= 8 else f" and {len(scenarios)-8} more"
        sample_str = ", ".join(sample[:-1]) + (" and " + sample[-1] if len(sample) > 1 else sample[0])
        return f"I found scenarios like {sample_str}{more}. You can plot variables for any of these scenarios."

    # -------------------------------
    # LIST ALL MODELS, RESULTS, AND WORKSPACES
    # -------------------------------
    if re.search(r"\b(list|get)\b.*\b(all|available)\b.*\b(models?|results?|workspaces?)\b", q) or \
       re.search(r"\b(models?|results?|workspaces?)\b.*\b(list|get|available)\b", q) or \
       re.search(r"\b(list|get)\b.*\b(models?|results?|workspaces?)\b.*\b(and|,)\b.*\b(models?|results?|workspaces?)\b", q):
        models = sorted({r.get('modelName', '') for r in model_data if r and r.get('modelName')})
        variables = sorted({r.get('variable', '') for r in ts_data if r and r.get('variable')})
        scenarios = sorted({r.get('scenario', '') for r in ts_data if r and r.get('scenario')})
        workspaces = get_available_workspaces(ts_data)

        response = "### Available Models, Results, and Workspaces\n\n"
        response += f"**Models ({len(models)}):**\n" + ", ".join(models) + "\n\n"
        response += f"**Results - Variables ({len(variables)}):**\n" + ", ".join(variables[:10]) + (f" and {len(variables)-10} more" if len(variables) > 10 else "") + "\n\n"
        response += f"**Results - Scenarios ({len(scenarios)}):**\n" + ", ".join(scenarios[:10]) + (f" and {len(scenarios)-10} more" if len(scenarios) > 10 else "") + "\n\n"
        response += f"**Workspaces ({len(workspaces)}):**\n" + ", ".join(workspaces) + "\n\n"
        response += "For more details on any item, ask specific questions like 'list models' or 'plot [variable]'."
        return response

    # -------------------------------
    # SPECIFIC VARIABLE QUERIES - Enhanced matching with direct data search
    # -------------------------------
    # First try to extract using utils functions (YAML-based)
    extracted = extract_variable_and_region_from_query(question, variable_dict, region_dict)

    variable_match = None
    region_match = extracted['region']

    # Enhanced variable matching: try YAML first, then direct data search
    if extracted['variable']['match_type'] not in [None, 'ambiguous']:
        variable_match = extracted['variable']['matched_variable']
        # Validate that this variable actually exists in our loaded data
        available_vars = {r.get('variable', '') for r in ts_data if r and r.get('variable')}
        if variable_match not in available_vars:
            variable_match = None  # Reset if not found

    # If YAML matching failed or variable not in data, try direct data search
    if not variable_match:
        q_lower = question.lower()
        available_vars = {r.get('variable', '') for r in ts_data if r and r.get('variable')}

        # Direct search for solar-related variables
        if any(word in q_lower for word in ['solar', 'pv', 'photovoltaic']):
            solar_vars = [v for v in available_vars if 'solar' in v.lower()]
            if solar_vars:
                # Prefer utility solar capacity
                utility_solar = [v for v in solar_vars if 'utility' in v.lower()]
                variable_match = utility_solar[0] if utility_solar else solar_vars[0]

        # Direct search for wind-related variables
        elif any(word in q_lower for word in ['wind']):
            wind_vars = [v for v in available_vars if 'wind' in v.lower()]
            if wind_vars:
                onshore_wind = [v for v in wind_vars if 'onshore' in v.lower()]
                variable_match = onshore_wind[0] if onshore_wind else wind_vars[0]

        # Direct search for electricity/generation variables
        elif any(word in q_lower for word in ['electricity', 'generation', 'power']):
            elec_vars = [v for v in available_vars if 'electricity' in v.lower() and 'secondary' in v.lower()]
            if elec_vars:
                variable_match = elec_vars[0]

        # Direct search for emissions variables
        elif any(word in q_lower for word in ['emission', 'co2', 'carbon']):
            emission_vars = [v for v in available_vars if 'co2' in v.lower() and 'energy' in v.lower()]
            if emission_vars:
                variable_match = emission_vars[0]

    # If we found a variable match, filter and return data
    if variable_match:
        # Filter time series data
        filtered_data = []
        for record in ts_data:
            if record.get('variable') == variable_match:
                if region_match and record.get('region') == region_match:
                    filtered_data.append(record)
                elif not region_match:  # No region specified, include all
                    filtered_data.append(record)

        if filtered_data:
            # Format and return the data
            return format_time_series_data(filtered_data, variable_match, region_match)

    # -------------------------------
    # MODEL INFO REQUESTS
    # -------------------------------
    if any(w in q for w in ('info', 'details', 'describe', 'about', 'tell me about')):
        return "Model information feature is under development. Please ask a specific question about models."

    # -------------------------------
    # HELP COMMAND
    # -------------------------------
    if 'help' in q:
        return (
            "Tell me what you want to do and I'll help. Examples:\n"
            "- Ask about models: `list models` or `info GCAM`\n"
            "- Explore variables: `list variables` or `plot CO2 emissions`\n"
            "- Visualize results: `plot emissions for GCAM`\n"
            "- To make plots, you can ask questions like:\n"
            "  * `plot [variable name]`\n"
            "  * `show me a plot of [variable name]`\n"
            "  * `graph [variable name] for [model name]`\n"
            "  * `visualize [variable name]`\n"
            "If you want more conversational guidance, just say 'suggest' or ask a question in plain language."
        )

    # -------------------------------
    # FALLBACK
    # -------------------------------
    return ""


def format_time_series_data(data_records: list, variable: str, region: str = "") -> str:
    """Format time series data into a readable table."""
    if not data_records:
        return f"No data found for variable '{variable}'{' in ' + region if region else ''}."

    # Group by scenario and model
    scenario_groups = {}
    for record in data_records:
        key = (record.get('scenario', 'Unknown'), record.get('modelName', 'Unknown'))
        if key not in scenario_groups:
            scenario_groups[key] = []
        scenario_groups[key].append(record)

    response = f"### {variable}"
    if region:
        response += f" in {region}"
    response += "\n\n"

    for (scenario, model), records in scenario_groups.items():
        response += f"**{model} - {scenario}**\n"

        # Get year columns
        years = []
        for record in records:
            years.extend([k for k in record.keys() if str(k).isdigit()])
        years = sorted(list(set(years)))

        if not years:
            response += "No year data available\n\n"
            continue

        # Create table header
        response += "| Year | Value | Unit |\n|------|-------|------|\n"

        # Get data for each year
        unit = records[0].get('unit', 'N/A') if records else 'N/A'

        for year in years:
            # Find value for this year (take first record that has it)
            value = None
            for record in records:
                if str(year) in record and record[str(year)] is not None:
                    value = record[str(year)]
                    break

            if value is not None:
                # Format large numbers
                if isinstance(value, (int, float)):
                    if abs(value) >= 1e6:
                        formatted_value = f"{value/1e6:.1f}M"
                    elif abs(value) >= 1e3:
                        formatted_value = f"{value/1e3:.1f}K"
                    else:
                        formatted_value = f"{value:.2f}"
                else:
                    formatted_value = str(value)
                response += f"| {year} | {formatted_value} | {unit} |\n"

        response += "\n"

    return response
