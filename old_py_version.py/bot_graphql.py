#!/usr/bin/env python3
"""
Climate and Energy Dataset Explorer Chatbot

Usage:
    python bot_graphql.py

This script uses OpenAI to maintain conversational context, summarize model descriptions via REST,
and fetch time-series results via IAM Paris REST API. Sensitive keys and URLs are read from config.py.
"""
import os
import re
import openai
import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt
from config import API_KEY, REST_MODELS_URL, RESULTS_URL

# === Validate config ===
if not API_KEY or not REST_MODELS_URL or not RESULTS_URL:
    raise RuntimeError("Configuration error: please set API_KEY, REST_MODELS_URL and RESULTS_URL in config.py")

# === Initialize OpenAI client ===
client = openai.OpenAI(api_key=API_KEY)

# === Conversation memory ===
conversation = [
    {"role": "system", "content": (
        "You are a friendly, engaging assistant that helps users explore climate and energy data. "
        "Use natural, conversational summaries and always include a relevant follow-up question."
    )}
]

# === Model metadata via REST ===
def list_models_rest() -> list[str]:
    """Return all model names from REST endpoint."""
    try:
        resp = requests.get(REST_MODELS_URL, params={'fields': 'modelName', 'limit': -1})
        resp.raise_for_status()
        return [item['modelName'] for item in resp.json().get('data', []) if 'modelName' in item]
    except Exception:
        return []


def fetch_model_info_rest(name: str) -> dict:
    """Fetch a single model's metadata using REST."""
    try:
        params = {'filter[modelName][_eq]': name,
                  'fields': 'modelName,description,overview,institute,model_type'}
        resp = requests.get(REST_MODELS_URL, params=params)
        resp.raise_for_status()
        items = resp.json().get('data', [])
        return items[0] if items else {}
    except Exception:
        return {}

# === Time-series data via REST ===
def fetch_results(filters: dict) -> pd.DataFrame:
    resp = requests.post(RESULTS_URL, json=filters)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json().get('data', []))
    if 'years' in df.columns:
        yrs = pd.json_normalize(df.pop('years'))
        df = pd.concat([df, yrs], axis=1)
    df.rename(columns={
        'region': 'Region', 'scenario': 'Scenario', 'modelName': 'Model',
        'unit': 'Unit', 'variable': 'Variable'
    }, inplace=True)
    return df

# === Formatting helpers ===
def format_trend(row, max_points: int = 3) -> str:
    years = sorted(int(c) for c in row.index if c.isdigit())
    pts = [(y, row[str(y)]) for y in years if pd.notna(row[str(y)])][:max_points]
    return ', '.join(f"in {y}: {val:.2f} {row.Unit}" for y, val in pts)


def format_text(row) -> str:
    return (f"Model '{row.Model}' projects {row.Variable} in {row.Region} under {row.Scenario}, "
            f"{format_trend(row)}.")

# === Plotting ===
def plot_data(df: pd.DataFrame, year: int = None):
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
def chatbot_response(user_input: str) -> str:
    ql = user_input.lower()
    conversation.append({"role": "user", "content": user_input})

    # Exit
    if ql in ['exit', 'quit']:
        return 'Goodbye!'

    # List models
    if any(kw in ql for kw in ['which models', 'list models', 'show models']):
        models = list_models_rest()
        reply = "Available models: " + ", ".join(models)
        conversation.append({"role": "assistant", "content": reply})
        return reply

    # Specific model info
    for name in list_models_rest():
        if name.lower() in ql:
            info = fetch_model_info_rest(name)
            desc = info.get('description', 'No description available').strip()
            overview = info.get('overview', '').strip()
            inst = info.get('institute', '').strip()
            mtype = info.get('model_type', '').strip()
            reply = (
                f"**{name}**: {desc}\n"
                f"Overview: {overview}\n"
                f"Institution: {inst} | Type: {mtype}\n"
                "Let me know if you'd like details on another model or data point!"
            )
            conversation.append({"role": "assistant", "content": reply})
            return reply

    # Data query filters
    filters = {'modelName': [], 'scenario': [], 'study': ['paris-reinforce'], 'variable': [], 'workspace_code': ['eu-headed']}
    if 'co2' in ql or 'emission' in ql:
        filters['variable'] = ['Emissions|CO2|Energy|Supply|Other Sector']
    elif 'food' in ql:
        filters['variable'] = ['Crops|Food']
    else:
        filters['variable'] = ['Final Energy']
    if 'usa' in ql: filters['workspace_code'] = ['us-headed']
    if 'world' in ql: filters['workspace_code'] = ['world-headed']
    if 'pr_wwh_cp' in ql: filters['scenario'] = ['PR_WWH_CP']

    df = fetch_results(filters)
    if df.empty:
        reply = "I couldn't find data for that query—anything else I can help with?"
        conversation.append({"role": "assistant", "content": reply})
        return reply

    # Plot request
    if any(k in ql for k in ['plot', 'chart', 'graph']):
        m = re.search(r'\b(20\d{2})\b', ql)
        year = int(m.group(1)) if m else None
        plot_data(df, year)
        reply = "Here's the chart—let me know if you want more details or another plot!"
        conversation.append({"role": "assistant", "content": reply})
        return reply

    # Year-specific values
    m_year = re.search(r'\b(20\d{2})\b', ql)
    if m_year and m_year.group(1) in df.columns:
        yr = m_year.group(1)
        lines = [f"{r.Model}: {r[yr]:.2f} {r.Unit}" for _, r in df.iterrows()]
        reply = f"Projections for {yr}:\n" + "\n".join(lines) + "\nAnything else?"
        conversation.append({"role": "assistant", "content": reply})
        return reply

    # General summary via GPT
    rows = [format_text(r) for _, r in df.iterrows()]
    prompt = (
        "Summarize these climate and energy projections conversationally and ask a relevant follow-up question to continue the dialogue:\n\n"
        "Data:\n" + "\n".join(rows) + "\n\nSummary:")
    resp = client.chat.completions.create(
        model="gpt-4",
        messages=conversation + [{"role": "user", "content": prompt}],
        temperature=0.7
    )
    reply = resp.choices[0].message.content.strip()
    conversation.append({"role": "assistant", "content": reply})
    return reply

# === CLI ===
if __name__ == '__main__':
    print("Welcome to the Climate and Energy Dataset Explorer powered by Holistic SA! (type 'exit' or 'quit' to close)")
    while True:
        text = input('You: ')
        if text.lower() in ['exit', 'quit']:
            print('Goodbye!')
            break
        print(chatbot_response(text))
