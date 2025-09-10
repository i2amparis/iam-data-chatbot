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

def data_query(question: str, model_data: list, ts_data: list) -> str:
    """Direct data lookup without using the LLM"""
    q = question.lower()

    # Handle plot requests using the simplified plotter
    if any(word in q for word in ['plot', 'show', 'graph', 'visualize']):
        return simple_plot_query(question, model_data, ts_data)

    # List models (conversational)
    if re.search(r"\b(list|available)\b.*\bmodels?\b", q) or re.search(r"\bmodels?\b.*\b(available|list)\b", q):
        models = sorted({r.get('modelName', '') for r in model_data if r.get('modelName')})
        if not models:
            return "I couldn't find any models in the data right now. Try `help` or refresh the data."
        # Build a natural sentence rather than a dry bullet list
        if len(models) <= 6:
            model_str = ", ".join(models[:-1]) + (" and " + models[-1] if len(models) > 1 else models[0])
            return f"I found these models in the IAM PARIS dataset: {model_str}. Which one would you like to know more about?"
        # For many models, give a short hint and invite follow-up
        return (f"There are {len(models)} models available. "
                "You can ask for details about a specific model using `info [model name]`, "
                "or say `list variables` to see the kinds of outputs available.")

    # List variables (conversational)
    if 'list variables' in q:
        vars = sorted({r.get('variable', '') for r in ts_data if r.get('variable')})
        if not vars:
            return "I don't see any variables in the loaded dataset. Try reloading or check the IAM PARIS results website."
        # Present a friendly sample and hint for full list
        sample = vars[:8]
        more = "" if len(vars) <= 8 else f" and {len(vars)-8} more"
        sample_str = ", ".join(sample[:-1]) + (" and " + sample[-1] if len(sample) > 1 else sample[0])
        return (f"I can work with variables like {sample_str}{more}. "
                "If you want the complete list, say `list variables` again or specify a variable to plot (e.g. `plot CO2 emissions`).")

    # List scenarios (conversational)
    if re.search(r"\b(list|available)\b.*\bscenarios?\b", q) or re.search(r"\bscenarios?\b.*\b(available|included|list)\b", q):
        scenarios = sorted({r.get('scenario', '') for r in ts_data if r.get('scenario')})
        if not scenarios:
            return "No scenarios are loaded in the current dataset. Try a different query or check IAM PARIS results."
        return ("I see several scenarios in the data. If you tell me which one interests you I can compare results or plot variables "
                "for that scenario. Try `list scenarios` to get the exact names.")

    # Model info or general models query — now conversational and data-driven
    if any(w in q for w in ('info', 'details', 'describe', 'about', 'tell me about')):
        return "Model information feature is under development. Please ask a specific question about models."

    # What data can you plot queries
    if re.search(r"\b(what|which)\b.*\b(data|variables?|plots?|graphs?|charts?)\b.*\b(can|could|do)\b.*\b(plot|show|graph|visualize|display)\b", q) or \
       re.search(r"\b(can|could|do)\b.*\b(plot|show|graph|visualize|display)\b.*\b(what|which)\b.*\b(data|variables?|plots?|graphs?|charts?)\b", q) or \
       re.search(r"\b(what|which|how)\b.*\b(question|questions|query|queries|ask|phrase|word)\b.*\b(can|could|do|should|to)\b.*\b(make|create|generate|get)\b.*\b(plot|plots?|graph|graphs?|chart|charts?|visualize)\b", q):
        # Check what data is available
        vars = sorted({r.get('variable', '') for r in ts_data if r.get('variable')})
        models = sorted({r.get('modelName', '') for r in model_data if r.get('modelName')})
        scenarios = sorted({r.get('scenario', '') for r in ts_data if r.get('scenario')})

        # Get year columns to check for time series data
        if ts_data:
            df = pd.DataFrame(ts_data)
            year_cols = [col for col in df.columns if str(col).isdigit()]

            # If no digit columns, check for "years" column with nested data
            if not year_cols and 'years' in df.columns:
                try:
                    # Sample the years column to understand its structure
                    sample_years = df['years'].dropna().head(3)
                    if len(sample_years) > 0:
                        first_sample = sample_years.iloc[0]
                        if isinstance(first_sample, dict):
                            # Expand nested years data
                            years_expanded = df['years'].apply(pd.Series)
                            year_cols_from_nested = [col for col in years_expanded.columns if str(col).isdigit()]
                            if year_cols_from_nested:
                                year_cols = year_cols_from_nested
                except Exception as e:
                    logging.warning(f"Error processing 'years' column: {e}")

            response = "## What I Can Plot\n\n"

            if year_cols:
                response += f"I can create time series plots showing how variables change over time from {min(year_cols)} to {max(year_cols)}.\n\n"
            else:
                response += "I can create plots, but the current dataset may not have complete time series data.\n\n"

            if vars:
                # Group variables by category for better presentation
                emission_vars = [v for v in vars if any(term in v.lower() for term in ['emission', 'co2', 'ghg', 'carbon'])]
                energy_vars = [v for v in vars if any(term in v.lower() for term in ['energy', 'electricity', 'power'])]
                economic_vars = [v for v in vars if any(term in v.lower() for term in ['gdp', 'price', 'cost', 'economic'])]
                other_vars = [v for v in vars if v not in emission_vars + energy_vars + economic_vars]

                response += "**Available Variables:**\n"
                if emission_vars:
                    response += f"- **Emissions**: {', '.join(emission_vars[:5])}{'...' if len(emission_vars) > 5 else ''}\n"
                if energy_vars:
                    response += f"- **Energy**: {', '.join(energy_vars[:5])}{'...' if len(energy_vars) > 5 else ''}\n"
                if economic_vars:
                    response += f"- **Economic**: {', '.join(economic_vars[:5])}{'...' if len(economic_vars) > 5 else ''}\n"
                if other_vars:
                    response += f"- **Other**: {', '.join(other_vars[:5])}{'...' if len(other_vars) > 5 else ''}\n"

                response += f"\n**Total variables available:** {len(vars)}\n"
            else:
                response += "**Variables:** None currently loaded\n"

            if models:
                response += f"**Models:** {len(models)} available ({', '.join(models[:3])}{'...' if len(models) > 3 else ''})\n"
            else:
                response += "**Models:** None currently loaded\n"

            if scenarios:
                response += f"**Scenarios:** {len(scenarios)} available ({', '.join(scenarios[:3])}{'...' if len(scenarios) > 3 else ''})\n"
            else:
                response += "**Scenarios:** None currently loaded\n"

            response += "\n**Examples of what you can ask:**\n"
            response += "- `plot CO2 emissions`\n"
            response += "- `plot energy consumption for GCAM`\n"
            response += "- `show GDP projections`\n"
            response += "- `visualize renewable energy adoption`\n\n"

            response += "**Tips:**\n"
            response += "- Use `list variables` for the complete variable list\n"
            response += "- Use `list models` to see all available models\n"
            response += "- Specify a model name to focus on that model's data\n"
            response += "- Include 'emissions' in your query to filter for emission-related variables\n\n"

            response += "_Data source: [IAM PARIS Results](https://iamparis.eu/results)_"

            return response
        else:
            return """## No Data Available for Plotting

I don't have any time series data loaded right now. This could be because:

- The data hasn't been loaded yet
- There's a connection issue with the data source
- The dataset is empty

Try:
1. Refreshing the data connection
2. Checking the [IAM PARIS Results](https://iamparis.eu/results) website directly
3. Using `help` to see other available commands"""

    # Help command
    if 'help' in q:
        return ("Tell me what you want to do and I'll help. Examples:\n"
                "- Ask about models: `list models` or `info GCAM`\n"
                "- Explore variables: `list variables` or `plot CO2 emissions`\n"
                "- Visualize results: `plot emissions for GCAM`\n"
                "- To make plots, you can ask questions like:\n"
                "  * `plot [variable name]`\n"
                "  * `show me a plot of [variable name]`\n"
                "  * `graph [variable name] for [model name]`\n"
                "  * `visualize [variable name]`\n"
                "If you want more conversational guidance, just say 'suggest' or ask a question in plain language.")

    return ""
