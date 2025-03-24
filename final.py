import openai
import pandas as pd
import faiss
import numpy as np

# Load dataset
file_path = "./Datasets/test_data_small.xlsx"  # Update with your actual file path
df = pd.read_excel(file_path)
df.columns = [int(col) if col.isdigit() else col for col in df.columns.astype(str)]

# Function to format text for embedding with variable hierarchy
def format_text(row):
    year_columns = [col for col in df.columns if isinstance(col, int) and col >= 2005]
    values = [f"{year}: {row[year]}" for year in year_columns if pd.notna(row[year])]
    trend_text = ", ".join(values) if values else "No data available"
    
    # Split variable into hierarchical components
    var_parts = row['Variable'].split('|')
    var_depth = len(var_parts)
    var_category = var_parts[0] if var_depth > 0 else ""
    var_subcategory = var_parts[1] if var_depth > 1 else ""
    var_specification = var_parts[2] if var_depth > 2 else ""
    
    return (
        f"Scenario: {row['Scenario']} | Region: {row['Region']} | "
        f"Category: {var_category} | Subcategory: {var_subcategory} | Specification: {var_specification} "
        f"({row['Unit']}) | Model: {row['Model']} | Trend: {trend_text}."
    )

# Create text entries for embedding
df["Text"] = df.apply(format_text, axis=1)

# OpenAI API client (replace with your actual API key)
client = openai.OpenAI(api_key='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')

# Function to generate embeddings
def get_embedding(text):
    try:
        response = client.embeddings.create(
            input=text, model="text-embedding-3-small"
        )
        vector = np.array(response.data[0].embedding)
        return vector / np.linalg.norm(vector)  # Normalize the embedding
    except Exception as e:
        print(f"Error generating embedding: {str(e)}")
        return None

# Compute embeddings with error handling
embeddings = []
for i, text in enumerate(df["Text"]):
    emb = get_embedding(text)
    if emb is None:
        print(f"Skipping row {i} due to embedding failure")
        continue
    embeddings.append(emb)
embeddings = np.array(embeddings)

# Store embeddings in FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
faiss.write_index(index, "embeddings.index")

# Save metadata
df[["Text"]].to_csv("metadata.csv", index=False)

print("Embeddings stored successfully!")

# Simple query parser to interpret user input
def parse_query(query):
    query = query.lower()
    parts = {}
    if "usa" in query or "us" in query:
        parts["Region"] = "USA"
    if "world" in query:
        parts["Region"] = "World"
    if "energy in food" in query:
        parts["Variable"] = "Crops|Food"  # Assume it means energy content of food crops
        parts["Special"] = "energy_content"  # Flag for special handling
    elif "energy" in query:
        parts["Variable"] = "Crops|Energy"
    elif "food" in query:
        parts["Variable"] = "Crops|Food"
    elif "feed" in query:
        parts["Variable"] = "Crops|Feed"
    if "202.conditionals" in query:
        parts["Year"] = 2025
    return parts

# Function to search FAISS and retrieve relevant text
def search_faiss(query, top_k=10):
    query_parts = parse_query(query)
    query_embedding = get_embedding(query)
    if query_embedding is None:
        print("Error: Unable to generate query embedding")
        return []
    
    query_embedding = np.array([query_embedding])
    distances, indices = index.search(query_embedding, top_k)

    # Filter and rank results based on parsed query
    filtered_results = []
    for i in indices[0]:
        if i >= len(df):
            continue
        row = df.iloc[i]
        score = 0
        if "Region" in query_parts and query_parts["Region"] == row["Region"]:
            score += 1
        if "Variable" in query_parts and query_parts["Variable"] in row["Variable"]:
            score += 2
        if "Year" in query_parts and str(query_parts["Year"]) in row["Text"]:
            score += 1  # Boost if the year matches in the trend text
        filtered_results.append((score, row["Text"]))
    
    # Sort by score and return top texts
    filtered_results.sort(reverse=True, key=lambda x: x[0])
    return [text for score, text in filtered_results[:top_k]]
# Function to generate chatbot responses using GPT-4
def chatbot_response(user_query, temperature=0.2):
    context = search_faiss(user_query)
    if not context:
        return "I don’t know. I couldn’t find the info you’re looking for."
    
    query_parts = parse_query(user_query)
    extra_instruction = ""
    if "Special" in query_parts and query_parts["Special"] == "energy_content":
        extra_instruction = (
            "The query might mean the energy in food crops. If so, use the exact number for 'Crops|Food' "
            "(million tons of dry matter per year) from the data. Keep it simple—say what the number is, "
            "what it’s for, and add an easy example like feeding people or powering something. Don’t use "
            "fancy energy units or math. Mention the number is for crops and might change with different crops."
        )
    
    prompt = (
        f"You are a friendly AI helper explaining things simply.\n"
        f"Use only the info below to answer.\n"
        f"The ‘Variable’ column splits into parts with '|', like Category|Subcategory.\n"
        f"Match the question to the right part (e.g., 'Energy' might mean 'Crops|Energy').\n"
        f"Give simple answers for people who don’t know stats. Use the exact numbers from the data, "
        f"but don’t do complicated math or use big words.\n"
        f"If the question isn’t clear (like 'Energy in Food'), guess it means energy from food crops.\n"
        f"If there’s no answer, say 'I don’t know'.\n"
        f"{extra_instruction}\n\n"
        f"Info:\n{context}\n\n"
        f"Question: {user_query}\n"
        f"Give a short, easy answer with the exact number from the data:"
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
    
# Test the chatbot
# test_query = "How much crops for feed we will need globally in 2050? Are there different results per model and per scenario?"
# print(chatbot_response(test_query))

if __name__ == "__main__":
    print("Welcome to the Climate and Energy Dataset Explorer!")
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