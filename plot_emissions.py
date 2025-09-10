import os
import pandas as pd
import webbrowser
import re
from data_utils import data_query

# Load cached models data
models_cache_files = [f for f in os.listdir('cache') if f.startswith('models_') and f.endswith('.json')]
if models_cache_files:
    # Load the first (or latest) models cache file
    models_df = pd.read_json(f'cache/{models_cache_files[0]}')
    models = models_df.to_dict('records')
else:
    print("No cached models data found.")
    models = []

# Load cached time series data
ts_file = 'cache/results_604172216409046961.json'
if os.path.exists(ts_file):
    ts_df = pd.read_json(ts_file)
    ts = ts_df.to_dict('records')
else:
    print("Specified time series data file not found.")
    ts = []

# Ensure plots directory exists
os.makedirs("plots", exist_ok=True)

# Call data_query to plot emissions
query = "plot emissions"
result = data_query(query, models, ts)

print(result)

# Extract and open the plot file if it was generated
plot_match = re.search(r'`([^`]+)`', result)
if plot_match:
    plot_file = plot_match.group(1)
    if os.path.exists(plot_file):
        webbrowser.open(plot_file)
        print(f"Opened plot: {plot_file}")
    else:
        print(f"Plot file not found: {plot_file}")
