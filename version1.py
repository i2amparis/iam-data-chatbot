import pandas as pd
import numpy as np
import openai
import faiss
import time
import os

EMBEDDING_MODEL = 'text-embedding-ada-002'
LLM_MODEL = 'gpt-4-0125-preview'
BATCH_SIZE = 50
MAX_RETRIES = 3

# Load FAISS Index
index = faiss.read_index('faiss_index.idx')

# Load Metadata
df = pd.read_pickle('metadata.pkl')
documents = df['combined_text'].tolist()

# Ensure FAISS index and documents match
if index.ntotal != len(documents):
    print("Warning: FAISS index and document metadata mismatch! Check if they are correctly aligned.")

# OpenAI Client
client = openai.OpenAI(api_key='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')

# OpenAI Embedding Function
def get_embedding(text):
    for _ in range(MAX_RETRIES):
        try:
            response = client.embeddings.create(
                input=text, model=EMBEDDING_MODEL
            )
            return np.array(response.data[0].embedding, dtype=np.float32)
        except Exception as e:
            print(f"Error: {e}, retrying...")
            time.sleep(2)
    return None

# FAISS Search Function
def search_faiss(query, top_k=10):
    query_embedding = get_embedding(query)
    if query_embedding is None:
        print("Error: Unable to generate query embedding")
        return []
    
    query_embedding = query_embedding.reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)
    
    print(f"FAISS Distances: {distances}")  # Print distances
    print(f"FAISS Indices: {indices}")  # Print indices

    return [documents[i] for i in indices[0] if i < len(documents)]


# Chatbot Response Function
def chatbot_response(user_query, temperature=0.4):
    context = search_faiss(user_query)
    if not context:
        return "I don't know. No relevant information found."
    
    print(f"Retrieved Context: {context}")
    
    prompt = f"""
    You are an AI assistant providing insights from a database.
    Use only the context below to answer.
    If no relevant information exists, say 'I don’t know'.
    
    Context:
    {context}
    
    Question: {user_query}
    Based on the above context, provide a detailed response:
    """
    
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {'role': 'system', 'content': 'You are a knowledgeable assistant'},
                {'role': 'user', 'content': prompt}
            ],
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Test the Chatbot
print(chatbot_response('In Agricultural Demand Crops what will be the trends in 2040'))