import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime
import logging
import base64
from io import BytesIO

def data_query(question: str, model_data: list, ts_data: list) -> str:
    """Direct data lookup without using the LLM"""
    q = question.lower()

    # Handle plot requests
    if any(word in q for word in ['plot', 'show', 'graph', 'visualize']):
        try:
            # Debug information
            logging.info(f"Total records in ts_data: {len(ts_data)}")

            # Create DataFrame
            df = pd.DataFrame(ts_data)
            logging.info(f"DataFrame columns: {df.columns}")

            # Get year columns first
            year_cols = [col for col in df.columns if str(col).isdigit()]
            logging.info(f"Year columns found: {year_cols}")

            # If user query is ambiguous about time series, ask for clarification
            if not year_cols and ('per year' in q or 'time series' in q or 'yearly' in q):
                return ("I see you want data per year or time series plots, but the current dataset doesn't have year-based data loaded. "
                        "Please specify which variable or model you want to see over time, or try `list variables` to see available data.")

            if not year_cols:
                # No time series data available - provide helpful guidance
                vars = sorted({r.get('variable', '') for r in ts_data if r.get('variable')})
                models = sorted({r.get('modelName', '') for r in model_data if r.get('modelName')})
                scenarios = sorted({r.get('scenario', '') for r in ts_data if r.get('scenario')})

                response = "## Plotting Information\n\n"
                response += "I can help you create plots, but the current dataset doesn't contain time series data with year columns.\n\n"

                if vars:
                    response += "**Available Variables for Future Plotting:**\n"
                    for var in vars[:10]:  # Show first 10 variables
                        response += f"- {var}\n"
                    if len(vars) > 10:
                        response += f"- ... and {len(vars) - 10} more\n"
                    response += "\n"

                if models:
                    response += f"**Available Models:** {', '.join(models[:5])}{'...' if len(models) > 5 else ''}\n\n"

                if scenarios:
                    response += f"**Available Scenarios:** {', '.join(scenarios[:5])}{'...' if len(scenarios) > 5 else ''}\n\n"

                response += "**To create plots in the future, you can ask:**\n"
                response += "- `plot [variable name]` (once time series data is available)\n"
                response += "- `show me [variable name] over time`\n"
                response += "- `graph [variable name] for [model name]`\n\n"

                response += "**Current Data Exploration Options:**\n"
                response += "- Use `list variables` to see all available variables\n"
                response += "- Use `list models` to see available models\n"
                response += "- Use `list scenarios` to explore scenarios\n"
                response += "- Visit [IAM PARIS Results](https://iamparis.eu/results) for the latest data\n\n"

                response += "_Note: Plotting requires time series data with year columns, which isn't currently available in the loaded dataset._"

                return response

            # Filter for emissions variables if requested
            emissions_vars = []
            if any(word in q for word in ['emission', 'co2', 'ghg']):
                emissions_vars = [v for v in df['variable'].unique()
                                if 'emission' in str(v).lower() or 'co2' in str(v).lower()]
                logging.info(f"Found emission variables: {emissions_vars}")

                if emissions_vars:
                    df = df[df['variable'].isin(emissions_vars)]
                else:
                    return """## No Emission Data Found

I couldn't find any emission-related variables in the current dataset. Available variables are:
""" + "\n".join([f"- {v}" for v in sorted(df['variable'].unique())]) + """

Try:
- Using `list variables` to see all available variables
- Plotting a different variable
- Checking the data source at [IAM PARIS Results](https://iamparis.eu/results)
"""

            # Create plot
            plt.figure(figsize=(12, 6))

            # Plot data for each model/variable combination
            for (model_name, var), group in df.groupby(['modelName', 'variable']):
                years = sorted(year_cols)
                values = [float(group[year].mean()) for year in years if not pd.isna(group[year].mean())]
                if values:  # Only plot if we have values
                    plt.plot(years[:len(values)], values, marker='o',
                            label=f"{model_name}: {var}", linewidth=2)

            if plt.gca().get_lines():  # Check if any lines were plotted
                plt.title("Emissions Time Series")
                plt.xlabel("Year")
                plt.ylabel(f"Value ({df['unit'].iloc[0]})" if 'unit' in df else "Value")
                plt.grid(True, alpha=0.3)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()

                # Convert plot to base64 for embedding in chat
                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                plt.close()

                # Also save to file as backup
                os.makedirs("plots", exist_ok=True)
                filename = f"plots/timeseries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                with open(filename, 'wb') as f:
                    f.write(base64.b64decode(image_base64))

                return f"""## Time Series Plot Generated

I've created a plot showing emissions over time across different models:

![Emissions Time Series](data:image/png;base64,{image_base64})

**Plot Details:**
- Data from {min(year_cols)} to {max(year_cols)}
- {len(df['modelName'].unique())} different models
- {len(emissions_vars)} emission-related variables

**File saved as:** `{filename}`

Want to:
- Focus on a specific model? Try `plot emissions for [model name]`
- Look at different variables? Try `list variables`
- Compare specific years? Include a year in your query

_Data source: [IAM PARIS Results](https://iamparis.eu/results)_"""
            else:
                return """## No Plottable Data Found

While I found some data, I couldn't create a meaningful plot. This might be because:
- The values are missing or invalid
- The time series is incomplete
- The data format isn't suitable for plotting

Try:
1. Using a different variable
2. Specifying a particular model
3. Checking the raw data with `list variables`"""

        except Exception as e:
            logging.error(f"Error creating plot: {str(e)}")
            return f"""## Error Creating Plot

I encountered an error: `{str(e)}`

Troubleshooting steps:
1. Check if the variable exists using `list variables`
2. Try specifying a model with `plot [model_name] emissions`
3. Make sure the data contains time series information

_If the problem persists, there might be an issue with the data source._"""

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
