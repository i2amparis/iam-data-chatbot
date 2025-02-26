import openai
import pandas as pd
import faiss
import numpy as np

# Load dataset
file_path = "test_data_small.xlsx"  # Update with your actual file path
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
    if "energy" in query:
        parts["Variable"] = "Crops|Energy"
    if "food" in query:
        parts["Variable"] = "Crops|Food"
    if "feed" in query:
        parts["Variable"] = "Crops|Feed"
    if "2025" in query:
        parts["Year"] = 2025
    # Add more mappings as needed (e.g., "other" -> "Crops|Other")
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
            score += 2  # Higher weight for variable match
        filtered_results.append((score, row["Text"]))
    
    # Sort by score and return top texts
    filtered_results.sort(reverse=True, key=lambda x: x[0])
    return [text for score, text in filtered_results[:top_k]]

# Function to generate chatbot responses using GPT-4
def chatbot_response(user_query, temperature=0.2):
    context = search_faiss(user_query)
    if not context:
        return "I don't know. No relevant information found."
    
    prompt = (
        f"You are an AI assistant providing insights from a dataset.\n"
        f"Use only the context below to answer.\n"
        f"The ‘Variable’ column uses a semi-hierarchical structure with '|' separators, e.g., Category|Subcategory|Specification.\n"
        f"Match the query to the most relevant variable component (e.g., 'Energy' might mean 'Crops|Energy').\n"
        f"If the query is ambiguous (e.g., 'Energy in Food'), clarify the interpretation or suggest possibilities.\n"
        f"If no relevant information exists, say 'I don’t know'.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {user_query}\n"
        f"Based on the above context, provide a detailed response:"
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {'role': 'system', 'content': 'You are an AI assistant providing insights'},
                {'role': 'user', 'content': prompt}
            ],
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Test the chatbot
test_query = "in usa Energy in Food how much energy will be needed in 2025?"
print(chatbot_response(test_query))