#!/usr/bin/env python3
"""
Climate and Energy Dataset Explorer Chatbot

Usage:
    python bot_graphql.py

This script directly embeds your OpenAI API key and uses Directus GraphQL for model lookups.
"""
import openai
import pandas as pd
import requests
import faiss
import numpy as np
import matplotlib.pyplot as plt
import os

# === OpenAI API client (direct key) ===
client = openai.OpenAI(
    api_key='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'
)

# Directus GraphQL endpoint
GRAPHQL_ENDPOINT = os.getenv('DIRECTUS_GRAPHQL_ENDPOINT', "https://cms.iamparis.eu/graphql")

# Fetch model info from Directus via GraphQL
def fetch_model_graphql(model_name: str) -> dict:
    query = '''
    query GetModelByName($modelName: String!) {
      models(filter: { modelName: { _eq: $modelName } }) {
        id
        modelName
        description
        created_on
        updated_on
      }
    }
    '''
    payload = {"query": query, "variables": {"modelName": model_name}}
    resp = requests.post(GRAPHQL_ENDPOINT, json=payload)
    resp.raise_for_status()
    data = resp.json()
    if errors := data.get('errors'):
        raise RuntimeError(f"GraphQL errors: {errors}")
    items = data['data'].get('models', [])
    return items[0] if items else {}

# Global conversation memory
conversation = [
    {
        "role": "system",
        "content": (
            "You are a friendly AI assistant helping users understand climate and energy data. "
            "Based on the data provided, answer clearly and ask a relevant follow-up question to continue the conversation. "
            "If the user asks for a plot, determine if it's a line plot over time or a bar chart by model in a specific year."
        )
    }
]

# Load and index local model descriptions (fallback)
def load_model_descriptions(folder="models"):
    descriptions = {}
    if not os.path.exists(folder):
        return descriptions
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            model_name = filename.replace(".txt", "").upper()
            with open(os.path.join(folder, filename), "r", encoding="utf-8-sig", errors="replace") as f:
                descriptions[model_name] = f.read()
    return descriptions

MODEL_DESCRIPTIONS = load_model_descriptions()

def get_requested_model_name(query: str) -> str:
    for model in MODEL_DESCRIPTIONS:
        if model.lower() in query.lower():
            return model
    return None

# Fetch data from API based on dynamic filters
def fetch_results_from_api(filters: dict) -> pd.DataFrame:
    url = "https://api.iamparis.eu/results"
    try:
        response = requests.post(url, json=filters)
        response.raise_for_status()
        data = response.json()

        if 'data' not in data or not data['data']:
            return pd.DataFrame()

        df_api = pd.DataFrame(data['data'])
        if 'years' in df_api.columns:
            years_df = pd.json_normalize(df_api['years'])
            df_api = pd.concat([df_api.drop(columns=['years']), years_df], axis=1)

        df_api.rename(columns={
            "region": "Region",
            "scenario": "Scenario",
            "modelName": "Model",
            "unit": "Unit",
            "variable": "Variable"
        }, inplace=True)

        return df_api

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

# Format each row into a text entry
def format_text(row: pd.Series) -> str:
    year_columns = [col for col in row.index if str(col).isdigit()]
    values = [
        f"{year}: {row[year]}"
        for year in year_columns
        if pd.notna(row[year]) and year in ("2030", "2040", "2050")
    ][:3]
    trend_text = ", ".join(values) if values else "No data available"

    var_parts = row['Variable'].split('|') if isinstance(row['Variable'], str) else []
    return (
        f"Scenario: {row.get('Scenario','')} | Region: {row.get('Region','')} | "
        f"Variable: {' > '.join(var_parts)} ({row.get('Unit','')}) | "
        f"Model: {row.get('Model','')} | Trend: {trend_text}."
    )

# Create embeddings
def get_embedding(text: str) -> np.ndarray:
    try:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        vector = np.array(response.data[0].embedding)
        return vector / np.linalg.norm(vector)
    except Exception as e:
        print(f"Error generating embedding: {str(e)}")
        return None

# Parse user query into API filters
def parse_query(query: str) -> dict:
    query = query.lower()
    filters = {
        "modelName": [],
        "scenario": [],
        "study": ["paris-reinforce"],
        "variable": [],
        "workspace_code": ["eu-headed"]
    }
    if "usa" in query or "us " in query:
        filters["workspace_code"] = ["us-headed"]
    if "world" in query:
        filters["workspace_code"] = ["world-headed"]
    if not filters['variable']:
        filters['variable'].append("Final Energy")
    return filters

# Plotting logic
def plot_data(df: pd.DataFrame, user_query: str = None) -> None:
    year_columns = [col for col in df.columns if str(col).isdigit()]
    if not year_columns:
        return

    requested_year = None
    if user_query:
        for y in year_columns:
            if str(y) in user_query:
                requested_year = y
                break

    plt.figure(figsize=(10, 6))
    if requested_year:
        plt.bar(df["Model"], df[requested_year])
        plt.xlabel("Model")
