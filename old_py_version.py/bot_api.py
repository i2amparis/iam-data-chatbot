import openai
import pandas as pd
import requests
import faiss
import numpy as np

# OpenAI API client (replace with your real API key)
client = openai.OpenAI(
    api_key='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'
)

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
    values = [f"{year}: {row[year]}" for year in year_columns if pd.notna(row[year])]
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

    workspace_set = False
    variable_set = False

    # Workspace detection
    if "usa" in query or "us" in query:
        filters["workspace_code"] = ["us-headed"]
        workspace_set = True
    if "world" in query:
        filters["workspace_code"] = ["world-headed"]
        workspace_set = True

    # Smart variable detection
    if any(word in query for word in ["energy in food", "food energy"]):
        filters["variable"].append("Crops|Food")
        variable_set = True
    elif "food" in query:
        filters["variable"].append("Crops|Food")
        variable_set = True
    elif "feed" in query:
        filters["variable"].append("Crops|Feed")
        variable_set = True
    elif any(word in query for word in ["energy", "energy demand", "energy use", "final energy"]):
        filters["variable"].append("Final Energy")
        variable_set = True
    elif any(word in query for word in ["emissions", "co2", "carbon"]):
        filters["variable"].append("Emissions|CO2|Energy|Supply|Other Sector")
        variable_set = True

    # Scenario detection
    if "pr_wwh_cp" in query:
        filters["scenario"].append("PR_WWH_CP")
    if "ref" in query:
        filters["scenario"].append("REF")

    # 🚀 If no variable is detected, default intelligently
    if not variable_set:
        if "emissions" in query or "co2" in query or "carbon" in query:
            filters["variable"].append("Emissions|CO2|Energy|Supply|Other Sector")
        elif "food" in query or "crops" in query:
            filters["variable"].append("Crops|Food")
        else:
            filters["variable"].append("Final Energy")  # General fallback

    return filters



# Search for best matching text entries
def search_faiss(query, top_k=10):
    filters = parse_query(query)
    df = fetch_results_from_api(filters)

    if df.empty:
        print("No data fetched for this query.")
        return []
    
    df["Text"] = df.apply(format_text, axis=1)

    embeddings = []
    for i, text in enumerate(df["Text"]):
        emb = get_embedding(text)
        if emb is None:
            print(f"Skipping row {i} due to embedding failure")
            continue
        embeddings.append(emb)
    
    if not embeddings:
        print("No embeddings generated.")
        return []

    embeddings = np.array(embeddings)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    query_embedding = get_embedding(query)
    if query_embedding is None:
        print("Error: Unable to generate query embedding")
        return []
    query_embedding = np.array([query_embedding])

    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i in indices[0]:
        if i >= len(df):
            continue
        row = df.iloc[i]
        results.append(row["Text"])
    
    return results

# Generate chatbot response using GPT-4
def chatbot_response(user_query, temperature=0.2):
    context = search_faiss(user_query)
    if not context:
        return "I don’t know. I couldn’t find the info you’re looking for."

    query_parts = parse_query(user_query)
    extra_instruction = ""
    if "variable" in query_parts and "Crops|Food" in query_parts["variable"]:
        extra_instruction = (
            "The query might mean the energy in food crops. If so, use the exact number for 'Crops|Food' "
            "(million tons of dry matter per year) from the data. Keep it simple—say what the number is, "
            "what it’s for, and add an easy example like feeding people or powering something. "
            "Don’t use fancy energy units or math. Mention the number is for crops and might change with different crops."
        )

    prompt = (
    "You are a friendly AI assistant helping users understand climate and energy data.\n"
    "Based on the provided information, give a clear, simple, human-style answer.\n"
    "If the data includes multiple values (like for different models), summarize them: "
    "mention the average or typical range, and avoid listing all values unless necessary.\n"
    "Focus on clarity, not on dumping raw data.\n\n"
    f"Data:\n{context}\n\n"
    f"User Question: {user_query}\n\n"
    "Answer:"
)


    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {'role': 'system', 'content': 'You are a friendly AI helper'},
                {'role': 'user', 'content': prompt}
            ],
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Oops, something went wrong: {str(e)}"

# Command line interface
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