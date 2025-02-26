import pandas as pd
import numpy as np
import openai
import faiss
import time
import os

client = openai.OpenAI(api_key='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')


# Constants
EMBEDDING_MODEL = 'text-embedding-3-small'
LLM_MODEL = 'gpt-4-0125-preview'
BATCH_SIZE = 50
MAX_RETRIES = 3

# Load Data
file_name = '2023_PR_IC_WWH_subset.xlsx'
try:
    df = pd.read_excel(file_name)
except FileNotFoundError:
    print(f"Error: File '{file_name}' not found. Ensure it is in the correct directory.")
    exit(1)

# Check for empty dataset
if df.empty:
    print("Error: The dataset is empty.")
    exit(1)

text_columns = df.select_dtypes(include='object').columns
df['combined_text'] = df[text_columns].astype(str).agg(' '.join, axis=1)
documents = df["combined_text"].tolist()

# OpenAI Embedding Function with Improved Error Handling
def get_embedding(text):
    for attempt in range(MAX_RETRIES):
        try:
            response = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
            return np.array(response.data[0].embedding, dtype=np.float32)
        except Exception as e:
            print(f"Error: {e}, retrying ({attempt+1}/{MAX_RETRIES})...")
            time.sleep(2)
    raise RuntimeError("Failed to generate embedding after multiple attempts.")

# Generate Document Embeddings
document_embeddings = []
for i in range(0, len(documents), BATCH_SIZE):
    batch = documents[i:i + BATCH_SIZE]
    embeddings = [get_embedding(doc) for doc in batch]
    document_embeddings.extend(e for e in embeddings if e is not None)
    print(f"Processed {min(i + BATCH_SIZE, len(documents))}/{len(documents)}")

# Check for empty embeddings
if len(document_embeddings) == 0:
    print("Error: No embeddings generated. Exiting.")
    exit(1)

document_embeddings = np.array(document_embeddings, dtype=np.float32)

# FAISS Indexing
dimension = document_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(document_embeddings)
faiss.write_index(index, 'faiss_index.idx')

# Save Metadata
df.to_pickle('metadata.pkl')
print('Setup complete')

# FAISS Search Function with Error Handling
def search_faiss(query, top_k=10):
    query_embedding = get_embedding(query)
    if query_embedding is None:
        return ["Error: Unable to generate embedding for query."]
    
    query_embedding = query_embedding.reshape(1, -1)
    
    try:
        distances, indices = index.search(query_embedding, top_k)
        return [documents[i] for i in indices[0] if i < len(documents)]
    except Exception as e:
        return [f"Error in FAISS search: {str(e)}"]

# Chatbot Response Function with Improved Handling
def chatbot_response(user_query, temperature=0.4):
    context = search_faiss(user_query)
    
    if "Error" in context[0]:  
        return context[0]  # Return error message from search_faiss
    
    prompt = f"""You are an AI assistant providing insights from a database.
    Use only the context provided to answer.
    If no relevant information exists, say 'I don’t know'.
    
    Context:
    {context}
    
    Question: {user_query}
    Answer:"""

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{'role': 'system', 'content': 'You are a knowledgeable assistant'},
                      {'role': 'user', 'content': prompt}],
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}"
