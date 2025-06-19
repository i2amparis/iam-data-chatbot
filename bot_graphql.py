#!/usr/bin/env python3
"""
Climate and Energy Dataset Explorer Chatbot

Usage:
    python bot_graphql.py

This script uses OpenAI to summarize model descriptions, Directus REST for model metadata,
and the IAM Paris REST API for time-series results.
"""
import os
import re
import openai
import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt

# === Configuration ===
API_KEY = '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'
# REST endpoint for model metadata
REST_MODELS_URL = 'https://cms.iamparis.eu/items/models'
# REST endpoint for time-series results
RESULTS_URL = 'https://api.iamparis.eu/results'

# === Initialize OpenAI client ===
client = openai.OpenAI(
    api_key=API_KEY
)

# === Model metadata via REST ===
def list_models_rest() -> list[str]:
    """Fetch all model names via Directus REST."""
    try:
        resp = requests.get(REST_MODELS_URL, params={'fields': 'modelName', 'limit': -1})
        resp.raise_for_status()
        items = resp.json().get('data', [])
        return [item.get('modelName') for item in items if 'modelName' in item]
    except Exception as e:
        print(f"Error listing models via REST: {e}")
        return []


def fetch_model_info_rest(name: str) -> dict:
    """Fetch a single model's metadata via Directus REST."""
    try:
        params = {
            'filter[modelName][_eq]': name,
            'fields': 'modelName,description,overview,long_name,institute,model_type'
        }
        resp = requests.get(REST_MODELS_URL, params=params)
        resp.raise_for_status()
        items = resp.json().get('data', [])
        return items[0] if items else {}
    except Exception as e:
        print(f"Error fetching model info via REST: {e}")
        return {}

# === Time-series results via REST ===
def fetch_results(filters: dict) -> pd.DataFrame:
    """Fetch time-series data via IAM Paris REST API."""
    try:
        resp = requests.post(RESULTS_URL, json=filters)
        resp.raise_for_status()
        data = resp.json()
        if 'data' not in data or not data['data']:
            return pd.DataFrame()
        df = pd.DataFrame(data['data'])
        # Normalize nested years column
        if 'years' in df.columns:
            yrs = pd.json_normalize(df.pop('years'))
            df = pd.concat([df, yrs], axis=1)
        df.rename(columns={
            'modelName': 'Model',
            'unit': 'Unit',
            'variable': 'Variable',
            'region': 'Region',
            'scenario': 'Scenario'
        }, inplace=True)
        return df
    except Exception as e:
        print(f"Error fetching time-series data: {e}")
        return pd.DataFrame()

# === Formatting helpers ===
def format_trend(row: pd.Series, max_points: int = 3) -> str:
    years = sorted(int(c) for c in row.index if c.isdigit())
    pts = [(y, row[str(y)]) for y in years if pd.notna(row[str(y)])][:max_points]
    return ', '.join(f'in {y}: {val:.2f} {row.Unit}' for y, val in pts)

def format_text(df: pd.DataFrame) -> str:
    return ' '.join(
        f"Model '{r.Model}' projects {r.Variable} in {r.Region} under {r.Scenario}, {format_trend(r)}."
        for _, r in df.iterrows()
    )

# === Plotting ===
def plot_data(df: pd.DataFrame, year: int = None) -> None:
    if year and str(year) in df.columns:
        plt.figure(figsize=(8,5))
        plt.bar(df.Model, df[str(year)])
        plt.title(f"{df.Variable.iloc[0]} in {year}")
        plt.ylabel(df.Unit.iloc[0])
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        yrs = sorted(int(c) for c in df.columns if c.isdigit())
        plt.figure(figsize=(8,5))
        for _, r in df.iterrows():
            plt.plot(yrs, [r[str(y)] for y in yrs], label=r.Model)
        plt.title(f"{df.Variable.iloc[0]} over Time")
        plt.ylabel(df.Unit.iloc[0])
        plt.legend()
        plt.tight_layout()
        plt.show()

# === Chatbot logic ===
def chatbot_response(q: str) -> str:
    ql = q.lower()
    # List available models
    if any(kw in ql for kw in ['which models', 'list models', 'show models']):
        return "Available models: " + ", ".join(list_models_rest())

    # Model metadata
    for name in list_models_rest():
        if name.lower() in ql:
            info = fetch_model_info_rest(name)
            return (
                f"**{info.get('modelName','')}**: {info.get('description','')}\n"
                f"Overview: {info.get('overview','')}\n"
                f"Institution: {info.get('institute','')} | Type: {info.get('model_type','')}"
            )

    # Build filters for results
    filters = {
        'modelName': [], 'scenario': [], 'study': ['paris-reinforce'],
        'variable': [], 'workspace_code': ['eu-headed']
    }
    if 'emission' in ql or 'co2' in ql:
        filters['variable'] = ['Emissions|CO2|Energy|Supply|Other Sector']
    elif 'food' in ql:
        filters['variable'] = ['Crops|Food']
    else:
        filters['variable'] = ['Final Energy']
    if 'usa' in ql:
        filters['workspace_code'] = ['us-headed']
    if 'world' in ql:
        filters['workspace_code'] = ['world-headed']
    if 'pr_wwh_cp' in ql:
        filters['scenario'] = ['PR_WWH_CP']

    df = fetch_results(filters)
    if df.empty:
        return "No data found for your query."
    if any(x in ql for x in ['plot', 'chart', 'graph']):
        m = re.search(r'\b(20\d{2})\b', q)
        year = int(m.group(1)) if m else None
        plot_data(df, year)
        return "Displayed chart."
    return format_text(df)

# === CLI ===
if __name__ == '__main__':
    print('Welcome to the Climate and Energy Dataset Explorer!')
    while True:
        inp = input('You: ')
        if inp.lower() in ['exit', 'quit']:
            print('Goodbye!')
            break
        print(chatbot_response(inp))
