"""
Optimized plotting module for IAM PARIS chatbot
Enhanced performance with caching and improved natural language support
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime
import logging
import base64
from io import BytesIO
from difflib import get_close_matches
import hashlib
import json
from functools import lru_cache


class SimplePlotter:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.plot_cache = {}  # Cache for plot results
        self.data_cache = {}  # Cache for processed data
        self.cache_dir = "plot_cache"
        self.precomputed_plots = {}  # Pre-computed common plots
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_key(self, data_hash: str, variable: str = None, model: str = None, plot_type: str = 'line') -> str:
        """Generate a unique cache key for plot requests"""
        key_data = f"{data_hash}_{variable or 'all'}_{model or 'all'}_{plot_type}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_data_hash(self, data: List[Dict]) -> str:
        """Generate hash for data to detect changes"""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(data_str.encode()).hexdigest()

    def _load_cached_plot(self, cache_key: str) -> Optional[str]:
        """Load plot from cache if available and not expired"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                # Check if cache is still valid (24 hours)
                cache_time = datetime.fromisoformat(cached_data['timestamp'])
                if (datetime.now() - cache_time).total_seconds() < 86400:  # 24 hours
                    self.logger.info(f"Loading cached plot for key: {cache_key}")
                    return cached_data['result']
            except Exception as e:
                self.logger.warning(f"Error loading cached plot: {e}")
        return None

    def _save_plot_to_cache(self, cache_key: str, result: str):
        """Save plot result to cache"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'result': result
        }
        try:
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
            self.logger.info(f"Cached plot result for key: {cache_key}")
        except Exception as e:
            self.logger.warning(f"Error saving plot to cache: {e}")

    def find_variable(self, query: str, available_vars: List[str]) -> Optional[str]:
        """Find the best matching variable from user's query"""
        query_lower = query.lower()

        # Direct matches first
        for var in available_vars:
            if var.lower() in query_lower:
                return var

        # Fuzzy matching for close matches
        close_matches = get_close_matches(query_lower, [v.lower() for v in available_vars], n=1, cutoff=0.6)
        if close_matches:
            # Find the original case variable
            for var in available_vars:
                if var.lower() == close_matches[0]:
                    return var

        return None

    def find_model(self, query: str, available_models: List[str]) -> Optional[str]:
        """Find the best matching model from user's query"""
        query_lower = query.lower()

        # Direct matches first
        for model in available_models:
            if model.lower() in query_lower:
                return model

        # Fuzzy matching
        close_matches = get_close_matches(query_lower, [m.lower() for m in available_models], n=1, cutoff=0.6)
        if close_matches:
            for model in available_models:
                if model.lower() == close_matches[0]:
                    return model

        return None

    def parse_plot_request(self, query: str, available_vars: List[str], available_models: List[str]) -> Dict[str, Any]:
        """Parse natural language plot requests"""
        query_lower = query.lower()

        # Initialize result
        result = {
            'action': 'plot',
            'variable': None,
            'model': None,
            'plot_type': 'line',  # default
            'time_series': True
        }

        # Find variable
        result['variable'] = self.find_variable(query, available_vars)

        # Find model
        result['model'] = self.find_model(query, available_models)

        # Determine plot type
        if any(word in query_lower for word in ['bar', 'bars', 'histogram']):
            result['plot_type'] = 'bar'
        elif any(word in query_lower for word in ['scatter', 'points']):
            result['plot_type'] = 'scatter'

        return result

    def create_simple_plot(self, data: List[Dict], variable: str = None, model: str = None,
                          plot_type: str = 'line') -> str:
        """Create a simple plot from the data with caching support"""

        if not data:
            return "No data available to plot."

        # Check cache first
        data_hash = self._get_data_hash(data)
        cache_key = self._get_cache_key(data_hash, variable, model, plot_type)
        cached_result = self._load_cached_plot(cache_key)
        if cached_result:
            return cached_result

        # Create DataFrame with optimized dtypes
        df = pd.DataFrame(data)

        # Debug: Log data structure
        self.logger.info(f"DataFrame shape: {df.shape}")
        self.logger.info(f"DataFrame columns: {list(df.columns)[:10]}...")  # First 10 columns

        # Pre-filter data to reduce processing time
        filter_conditions = []

        # Filter by variable if specified - use more efficient string matching
        if variable:
            if 'variable' in df.columns:
                # Use case-insensitive regex for better performance
                var_pattern = re.compile(re.escape(variable), re.IGNORECASE)
                df = df[df['variable'].str.contains(var_pattern, na=False)]
                self.logger.info(f"Filtered by variable '{variable}': {len(df)} records remaining")
            else:
                return f"Variable '{variable}' not found in data. Available variables: {list(set(df.get('variable', [])))}"

        # Filter by model if specified - use more efficient string matching
        if model:
            if 'modelName' in df.columns:
                # Use case-insensitive regex for better performance
                model_pattern = re.compile(re.escape(model), re.IGNORECASE)
                df = df[df['modelName'].str.contains(model_pattern, na=False)]
                self.logger.info(f"Filtered by model '{model}': {len(df)} records remaining")
            else:
                return f"Model '{model}' not found in data. Available models: {list(set(df.get('modelName', [])))}"

        if df.empty:
            return "No data matches your criteria."

        # Get year columns - try multiple patterns
        year_cols = [col for col in df.columns if str(col).isdigit()]
        self.logger.info(f"Found {len(year_cols)} year columns: {year_cols[:5]}...")

        # If no digit columns, try other patterns that might represent years
        if not year_cols:
            # Try columns that look like years (e.g., "2020", "2030", etc.)
            potential_year_cols = []
            for col in df.columns:
                col_str = str(col).strip()
                if len(col_str) == 4 and col_str.isdigit():
                    try:
                        year = int(col_str)
                        if 1900 <= year <= 2100:  # Reasonable year range
                            potential_year_cols.append(col)
                    except ValueError:
                        pass
            year_cols = potential_year_cols
            self.logger.info(f"Found {len(year_cols)} potential year columns with alternative method: {year_cols[:5]}...")

        # If still no year columns, check for "years" column with nested data
        if not year_cols and 'years' in df.columns:
            self.logger.info("Checking 'years' column for time series data...")
            # Sample a few rows to understand the structure
            sample_years = df['years'].dropna().head(3)
            if len(sample_years) > 0:
                first_sample = sample_years.iloc[0]
                self.logger.info(f"Sample 'years' data: {type(first_sample)} - {str(first_sample)[:200]}...")

                # Try to extract year-value pairs from the 'years' column
                try:
                    # If it's a dict-like structure, expand it
                    if isinstance(first_sample, dict):
                        # Convert nested years data to separate columns
                        years_expanded = df['years'].apply(pd.Series)
                        # Find numeric columns that look like years
                        year_cols_from_nested = [col for col in years_expanded.columns if str(col).isdigit()]
                        if year_cols_from_nested:
                            self.logger.info(f"Found {len(year_cols_from_nested)} year columns in nested 'years' data")
                            # Add these as new columns to the dataframe
                            for year_col in year_cols_from_nested:
                                df[year_col] = years_expanded[year_col]
                            year_cols = year_cols_from_nested
                        else:
                            self.logger.info("No numeric year columns found in nested 'years' data")
                    elif isinstance(first_sample, list):
                        # Handle list format if present
                        self.logger.info("'years' column contains list data - may need different parsing")
                    else:
                        self.logger.info(f"'years' column contains {type(first_sample)} data")
                except Exception as e:
                    self.logger.warning(f"Error processing 'years' column: {e}")

        if not year_cols:
            # Provide helpful error message with data structure info
            available_cols = list(df.columns)[:20]  # Show first 20 columns
            sample_values = {}
            for col in available_cols[:5]:  # Sample first 5 columns
                sample_val = df[col].iloc[0] if len(df) > 0 else "N/A"
                sample_values[col] = str(sample_val)[:50]  # Truncate long values

            error_msg = "## Unable to Create Plot\n\n"
            error_msg += "The data doesn't contain time series information (year columns) needed for plotting.\n\n"
            error_msg += "**Available data columns:**\n"
            for col in available_cols:
                error_msg += f"- `{col}`\n"
            error_msg += "\n**Sample values:**\n"
            for col, val in sample_values.items():
                error_msg += f"- `{col}`: {val}\n"
            error_msg += "\n**Suggestions:**\n"
            error_msg += "- Try asking for data summaries instead of plots\n"
            error_msg += "- Check if the data source has time series data available\n"
            error_msg += "- Ask about specific models or variables without requesting plots\n\n"
            error_msg += "_Data source: [IAM PARIS Results](https://iamparis.eu/results)_"

            return error_msg

        # Create plot
        plt.figure(figsize=(10, 6))
        plt.style.use('default')

        if plot_type == 'bar':
            # Bar plot for a specific year (most recent)
            latest_year = max(year_cols)
            if model:
                # Single model bar
                model_data = df[df['modelName'].str.contains(model, case=False, na=False)]
                if not model_data.empty:
                    vars_data = model_data.groupby('variable')[latest_year].mean()
                    vars_data.plot(kind='bar', color='skyblue')
                    plt.title(f"{model} - {latest_year}")
                    plt.ylabel("Value")
                    plt.xticks(rotation=45)
            else:
                # Multiple models bar
                model_data = df.groupby('modelName')[latest_year].mean()
                model_data.plot(kind='bar', color='lightcoral')
                plt.title(f"All Models - {latest_year}")
                plt.ylabel("Value")
                plt.xticks(rotation=45)

        else:  # line plot (default)
            if model:
                # Single model time series
                model_data = df[df['modelName'].str.contains(model, case=False, na=False)]
                if not model_data.empty:
                    for var_name, var_data in model_data.groupby('variable'):
                        years = sorted(year_cols, key=int)
                        values = [var_data[year].mean() for year in years if not pd.isna(var_data[year].mean())]
                        if values:
                            plt.plot(years[:len(values)], values, marker='o', label=f"{model}: {var_name}")
            else:
                # Multiple models time series
                for model_name, model_data in df.groupby('modelName'):
                    years = sorted(year_cols, key=int)
                    values = [model_data[year].mean() for year in years if not pd.isna(model_data[year].mean())]
                    if values:
                        plt.plot(years[:len(values)], values, marker='o', label=model_name)

            plt.title("Time Series Comparison")
            plt.xlabel("Year")
            plt.ylabel("Value")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # Convert to base64 in-memory (no file I/O)
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        buffer.close()  # Free memory

        # Optional: Save file only if explicitly requested or for debugging
        save_file = os.getenv('SAVE_PLOT_FILES', 'false').lower() == 'true'
        if save_file:
            os.makedirs("plots", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"plots/simple_plot_{timestamp}.png"
            with open(filename, 'wb') as f:
                f.write(base64.b64decode(image_base64))
            result = f"![Plot](data:image/png;base64,{image_base64})\n\n**Plot saved as:** `{filename}`"
        else:
            result = f"![Plot](data:image/png;base64,{image_base64})"

        # Save to cache
        self._save_plot_to_cache(cache_key, result)

        return result

    def precompute_common_plots(self, ts_data: List[Dict], model_data: List[Dict]):
        """Pre-compute common plots for faster access"""
        if not ts_data:
            return

        self.logger.info("Pre-computing common plots...")

        # Get available variables and models
        available_vars = list(set(r.get('variable', '') for r in ts_data if r.get('variable')))
        available_models = list(set(r.get('modelName', '') for r in model_data if r.get('modelName')))

        # Common plot configurations to pre-compute
        common_configs = [
            # Overall time series for all models
            {'variable': None, 'model': None, 'plot_type': 'line'},
            # Bar chart for all models
            {'variable': None, 'model': None, 'plot_type': 'bar'},
        ]

        # Add common variables (first few most common ones)
        common_variables = available_vars[:5] if len(available_vars) > 5 else available_vars
        for var in common_variables:
            common_configs.extend([
                {'variable': var, 'model': None, 'plot_type': 'line'},
                {'variable': var, 'model': None, 'plot_type': 'bar'},
            ])

        # Add common models (first few most common ones)
        common_models = available_models[:3] if len(available_models) > 3 else available_models
        for model in common_models:
            common_configs.append({'variable': None, 'model': model, 'plot_type': 'line'})

        # Pre-compute and cache these plots
        for config in common_configs:
            try:
                plot_key = f"{config['variable'] or 'all'}_{config['model'] or 'all'}_{config['plot_type']}"
                if plot_key not in self.precomputed_plots:
                    self.logger.info(f"Pre-computing plot: {plot_key}")
                    plot_result = self.create_simple_plot(
                        ts_data,
                        variable=config['variable'],
                        model=config['model'],
                        plot_type=config['plot_type']
                    )
                    self.precomputed_plots[plot_key] = plot_result
            except Exception as e:
                self.logger.warning(f"Failed to pre-compute plot {config}: {e}")

        self.logger.info(f"Pre-computed {len(self.precomputed_plots)} common plots")

    def handle_plot_request(self, query: str, model_data: List[Dict], ts_data: List[Dict]) -> str:
        """Main handler for plot requests"""

        # Get available variables and models
        available_vars = list(set(r.get('variable', '') for r in ts_data if r.get('variable')))
        available_models = list(set(r.get('modelName', '') for r in model_data if r.get('modelName')))

        # Parse the request
        parsed = self.parse_plot_request(query, available_vars, available_models)

        if not parsed['variable'] and not parsed['model']:
            return ("I couldn't understand what you want to plot. Try mentioning a variable like 'emissions' or 'GDP', "
                   "or a model name. For example: 'plot CO2 emissions' or 'show me GCAM results'.")

        # Check pre-computed plots first
        plot_key = f"{parsed['variable'] or 'all'}_{parsed['model'] or 'all'}_{parsed['plot_type']}"
        if plot_key in self.precomputed_plots:
            self.logger.info(f"Using pre-computed plot: {plot_key}")
            return self.precomputed_plots[plot_key]

        # Create the plot
        return self.create_simple_plot(
            ts_data,
            variable=parsed['variable'],
            model=parsed['model'],
            plot_type=parsed['plot_type']
        )


# Global instance for easy use
simple_plotter = SimplePlotter()


def simple_plot_query(query: str, model_data: List[Dict], ts_data: List[Dict]) -> str:
    """Simplified plot query handler"""
    return simple_plotter.handle_plot_request(query, model_data, ts_data)
