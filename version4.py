import openai
import pandas as pd
import faiss
import numpy as np

# Load dataset
file_path = "2023_PR_IC_WWH_subset.xlsx"  # Update if needed
df = pd.read_excel(file_path)

# Function to format text for embedding
def format_text(row):
    values = [f"{year}: {row[year]}" for year in range(2005, 2101, 5) if year in row and pd.notna(row[year])]
    trend_text = ", ".join(values)
    return (
        f"In the {row['Scenario']} scenario for the {row['Region']} region, "
        f"the variable {row['Variable']} ({row['Unit']}) is modeled using {row['Model']}. "
        f"Trend data: {trend_text}."
    )

# Create text entries for embedding
df["Text"] = df.apply(format_text, axis=1)

# OpenAI API client
client = openai.OpenAI(api_key='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')  # Initialize OpenAI client

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

# Compute embeddings for all rows
embeddings = np.array([get_embedding(text) for text in df["Text"] if text is not None])

# Store embeddings in FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
faiss.write_index(index, "embeddings.index")

# Save metadata
df[["Text"]].to_csv("metadata.csv", index=False)

print("Embeddings stored successfully!")

# Function to search FAISS and retrieve relevant text
def search_faiss(query, top_k=10):
    query_embedding = get_embedding(query)
    if query_embedding is None:
        print("Error: Unable to generate query embedding")
        return []
    query_embedding = np.array([query_embedding])
    distances, indices = index.search(query_embedding, top_k)
    return [df.iloc[i]["Text"] for i in indices[0] if i < len(df)]

# Function to generate chatbot responses using GPT-4
def chatbot_response(user_query, temperature=0.4):
    context = search_faiss(user_query)
    if not context:
        return "I don't know. No relevant information found."
    prompt = (
        f"You are an AI assistant providing insights from a dataset.\n"
        f"Use only the context below to answer.\n"
        f"If no relevant information exists, say 'I don’t know'.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {user_query}\n"
        f"Based on the above context, provide a detailed response:"
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4", messages=[
                {'role': 'system', 'content': 'You are a knowledgeable assistant'},
                {'role': 'user', 'content': prompt}
            ], temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Test the chatbot
print(chatbot_response("What is the trend in Agricultural Demand Crops for 2040?"))
