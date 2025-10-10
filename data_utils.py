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
from utils_query import match_variable_from_yaml, extract_examples_from_data
from utils.yaml_loader import load_all_yaml_files


# Load variable definitions from YAML files
variable_dict = load_all_yaml_files('definitions/variable')


def data_query(question: str, model_data: list, ts_data: list) -> str:
    """Process a user query about IAM data, optionally returning results or plots."""
    q = question.lower()

    # -------------------------------
    # Handle PLOTTING QUERIES
    # -------------------------------
    if any(word in q for word in ['plot', 'show', 'graph', 'visualize']):
        match = match_variable_from_yaml(question, variable_dict)

        if match["match_type"] in ["exact", "fuzzy"]:
            matched_var = match["matched_variable"]
            return simple_plot_query(matched_var, model_data, ts_data)

        elif match["match_type"] == "ambiguous":
            options = ", ".join(match["matches"][:5])
            return f"Your query matched multiple variables. Please clarify one of the following: {options}"

        else:
            return "Could not identify a variable to plot. Try specifying a variable like 'CO2 emissions' or 'energy consumption'."

    # -------------------------------
    # LIST AVAILABLE MODELS
    # -------------------------------
    if re.search(r"\b(list|available)\b.*\bmodels?\b", q) or re.search(r"\bmodels?\b.*\b(available|list)\b", q):
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
    if 'list variables' in q:
        vars = sorted({r.get('variable', '') for r in ts_data if r and r.get('variable')})
        if not vars:
            return "I don't see any variables in the loaded dataset. Try reloading or check the IAM PARIS results website."

        sample = vars[:8]
        more = "" if len(vars) <= 8 else f" and {len(vars)-8} more"
        sample_str = ", ".join(sample[:-1]) + (" and " + sample[-1] if len(sample) > 1 else sample[0])
        return (f"I can work with variables like {sample_str}{more}. "
                "If you want the complete list, say `list variables` again or specify a variable to plot (e.g. `plot CO2 emissions`).")

    # -------------------------------
    # LIST AVAILABLE SCENARIOS
    # -------------------------------
    if re.search(r"\b(list|available)\b.*\bscenarios?\b", q) or re.search(r"\bscenarios?\b.*\b(available|included|list)\b", q):
        scenarios = sorted({r.get('scenario', '') for r in ts_data if r and r.get('scenario')})
        if not scenarios:
            return "No scenarios are loaded in the current dataset. Try a different query or check IAM PARIS results."

        sample = scenarios[:8]
        more = "" if len(scenarios) <= 8 else f" and {len(scenarios)-8} more"
        sample_str = ", ".join(sample[:-1]) + (" and " + sample[-1] if len(sample) > 1 else sample[0])
        return f"I found scenarios like {sample_str}{more}. You can plot variables for any of these scenarios."

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
