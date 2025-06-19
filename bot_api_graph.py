
import openai
import pandas as pd
import requests
import faiss
import numpy as np
import matplotlib.pyplot as plt
import os

# OpenAI API client
client = openai.OpenAI(
    api_key='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'
)

#load and index files
def load_model_descriptions(folder="models"):
    descriptions = {}
    if not os.path.exists(folder):
        print(f"Warning: folder '{folder}' not found. No model descriptions loaded.")
        return descriptions
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            model_name = filename.replace(".txt", "").upper()
            file_path = os.path.join(folder, filename)
            try:
                with open(file_path, "r", encoding="utf-8-sig", errors="replace") as f:
                    descriptions[model_name] = f.read()
            except Exception as e:
                print(f"⚠️ Failed to load {filename}: {e}")
    return descriptions


MODEL_DESCRIPTIONS= load_model_descriptions()

#detect model query
def get_requested_model_name(query):
    for model in MODEL_DESCRIPTIONS:
        if model.lower() in query.lower():
            return model
    return None




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

# Fetch data from API based on dynamic filters
def fetch_results_from_api(filters):
    url = "https://api.iamparis.eu/results"
    try:
        response = requests.post(url, json=filters)
        response.raise_for_status()
        data = response.json()

        if 'data' not in data or not data['data']:
            print("No results found in the API.")
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
def format_text(row):
    year_columns = [col for col in row.index if str(col).isdigit()]
    # 👇 Only keep 3 year values max (or filter to just 2030)
    values = [
        f"{year}: {row[year]}"
        for year in year_columns
        if pd.notna(row[year]) and (year == "2030" or year == "2040" or year == "2050")
    ][:3]
    trend_text = ", ".join(values) if values else "No data available"

    var_parts = row['Variable'].split('|') if isinstance(row['Variable'], str) else []
    var_category = var_parts[0] if len(var_parts) > 0 else ""
    var_subcategory = var_parts[1] if len(var_parts) > 1 else ""
    var_specification = var_parts[2] if len(var_parts) > 2 else ""

    return (
        f"Scenario: {row.get('Scenario', '')} | Region: {row.get('Region', '')} | "
        f"Category: {var_category} | Subcategory: {var_subcategory} | Specification: {var_specification} "
        f"({row.get('Unit', '')}) | Model: {row.get('Model', '')} | Trend: {trend_text}."
    )

# Create embeddings
def get_embedding(text):
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
def parse_query(query):
    query = query.lower()
    filters = {
        "modelName": [],
        "scenario": [],
        "study": ["paris-reinforce"],
        "variable": [],
        "workspace_code": ["eu-headed"]
    }

    # Workspace detection
    if "usa" in query or "us" in query:
        filters["workspace_code"] = ["us-headed"]
    if "world" in query:
        filters["workspace_code"] = ["world-headed"]

    # Smart variable detection
    if any(word in query for word in ["energy in food", "food energy"]):
        filters["variable"].append("Crops|Food")
    elif "food" in query:
        filters["variable"].append("Crops|Food")
    elif "feed" in query:
        filters["variable"].append("Crops|Feed")
    elif any(word in query for word in ["energy", "energy demand", "energy use", "final energy"]):
        filters["variable"].append("Final Energy")
    elif any(word in query for word in ["emissions", "co2", "carbon"]):
        filters["variable"].append("Emissions|CO2|Energy|Supply|Other Sector")

    # Scenario detection
    if "pr_wwh_cp" in query:
        filters["scenario"].append("PR_WWH_CP")
    if "ref" in query:
        filters["scenario"].append("REF")

    # If no variable is detected
    if not filters["variable"]:
        if "emissions" in query or "co2" in query or "carbon" in query:
            filters["variable"].append("Emissions|CO2|Energy|Supply|Other Sector")
        elif "food" in query or "crops" in query:
            filters["variable"].append("Crops|Food")
        else:
            filters["variable"].append("Final Energy")

    return filters

# Main chatbot logic
def chatbot_response(user_query, temperature=0.2):
    model_name = get_requested_model_name(user_query)

    # ✅ Handle model listing
    if "which models" in user_query.lower() or "what models" in user_query.lower():
        available_models = ", ".join(sorted(MODEL_DESCRIPTIONS.keys()))
        return f"I use the following models: {available_models}"

    # ✅ Handle specific model description
    if model_name:
        summary_prompt = (
            f"Please summarize the following description of the {model_name} model:\n\n"
            f"{MODEL_DESCRIPTIONS[model_name]}"
        )
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": summary_prompt}],
                temperature=0.4
            )
            return f"📘 Summary for **{model_name}**:\n\n{response.choices[0].message.content}"
        except Exception as e:
            return f"Found description for model {model_name}, but failed to summarize it: {str(e)}"

    # ✅ Plotting logic
    def plot_data(df, user_query=None):
        year_columns = [col for col in df.columns if str(col).isdigit()]
        if not year_columns:
            print('No year columns found')
            return

        requested_year = None
        if user_query:
            for y in year_columns:
                if str(y) in user_query:
                    requested_year = y
                    break

        plt.figure(figsize=(10, 6))

        if requested_year:
            models = df["Model"]
            values = df[requested_year]
            plt.bar(models, values)
            plt.xlabel("Model")
            plt.ylabel(df.iloc[0].get("Unit", "Value"))
            plt.title(f"Energy Demand in {requested_year} by Model")
            plt.xticks(rotation=45)
        else:
            for _, row in df.iterrows():
                values = row[year_columns].dropna()
                plt.plot(values.index.astype(int), values.values, label=row["Model"])

            plt.xlabel("Year")
            plt.ylabel(df.iloc[0].get("Unit", "Value"))
            plt.title(f"{df.iloc[0].get('Variable', 'Data')} over Time")
            plt.legend()

        plt.grid(True)
        plt.tight_layout()
        plt.show()

    wants_plot = any(kw in user_query.lower() for kw in ["plot", "graph", "chart", "visualize"])
    filters = parse_query(user_query)
    df = fetch_results_from_api(filters)

    if df.empty:
        return "I don’t know. I couldn’t find the info you’re looking for."

    if wants_plot:
        try:
            plot_data(df, user_query)
            return "Here is the graph generated from the data."
        except Exception as e:
            return f"Data fetched, but unable to generate the graph: {str(e)}"

    df["Text"] = df.apply(format_text, axis=1)
    context = df["Text"].tolist()[:5]

    user_prompt = (
        "You are a friendly AI assistant helping users understand climate and energy data.\n"
        "This is a sample of the data (limited to a few models).\n"
        "If the user asks for a plot, determine if it's a line plot over time or a bar chart by model in a specific year.\n"
        "Respond clearly, and if needed, ask a follow-up question.\n\n"
        "Data:\n" + "\n".join(context) + "\n\n"
        f"User Question: {user_query}\n\n"
        "Answer:"
    )

    conversation.append({"role": "user", "content": user_prompt})

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=conversation,
            temperature=temperature
        )
        reply = response.choices[0].message.content
        conversation.append({"role": "assistant", "content": reply})
        return reply
    except Exception as e:
        return f"Oops, something went wrong: {str(e)}"


# CLI interface
if __name__ == "__main__":
    print("Welcome to the Climate and Energy Dataset Explorer powered by Holistic SA!")
    print("Type your question and press Enter. Type 'exit' to quit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting. Goodbye!")
            break
        try:
            response = chatbot_response(user_input)
            print(f"Bot: {response}")
        except Exception as e:
            print(f"Error: {e}")