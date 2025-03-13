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

# Load Data
df = pd.read_excel('2023_PR_IC_WWH_subset.xlsx')
text_columns = df.select_dtypes(include='object').columns
year_columns = [col for col in df.columns if str(col).isdigit()]
df['combined_text'] = df[text_columns].astype(str).agg('|'.join, axis=1)
documents = df["combined_text"].tolist()

# OpenAI Embedding Function
client = openai.OpenAI(api_key='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')  # Initialize OpenAI client

def get_embedding(text):
    for attempt in range(MAX_RETRIES):
        try:
            response = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
            return np.array(response.data[0].embedding, dtype=np.float32)
        except Exception as e:
            print(f"Attempt {attempt+1} - Error generating embedding: {e}")
            time.sleep(2)
    print("Final embedding failure for:", text)
    return None




# FAISS Index
index= faiss.read_index('faiss_index.idx')

# load metadata
df = pd.read_pickle('metadata.pkl')
documents = df['combined_text'].tolist()



# FAISS Search Function
def search_faiss(query, top_k=5):
    query_embedding = get_embedding(query)
    if query_embedding is None:
        print("Embedding generation failed for query.")
        return []
    query_embedding = query_embedding.reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)
    return [documents[i] for i in indices[0] if i < len(documents)]


# Chatbot Response Function
def chatbot_response(user_query, temperature=0.7):
    context = search_faiss(user_query)
    print("Retrieved Context:", context)

    prompt = f"""Answer the question using only the context below.
    
    Context:
    {context}
    
    Question: {user_query}
    Answer:"""

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

# Test
print(chatbot_response('What will be the trends in 2040 for Energy?'))