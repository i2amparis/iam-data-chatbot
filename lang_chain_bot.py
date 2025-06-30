import os
import sys
import requests
import pandas as pd
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Prevent script from shadowing the LangChain package
script_name = os.path.basename(__file__)
if script_name.lower().startswith("langchain"):
    print(f"❌ Rename '{script_name}' to avoid conflicting with LangChain packages.")
    sys.exit(1)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPEN_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY environment variable.")
REST_MODELS_URL = os.getenv(
    "REST_MODELS_URL",
    "https://cms.iamparis.eu/items/models"
)
RESULTS_URL = os.getenv(
    "RESULTS_URL",
    "https://api.iamparis.eu/results"
)

# 1) Data fetching functions

def fetch_model_metadata() -> list[dict]:
    resp = requests.get(
        REST_MODELS_URL,
        params={"fields": "modelName,description"}
    )
    resp.raise_for_status()
    return resp.json().get("data", [])


def fetch_time_series(filters: dict) -> list[dict]:
    resp = requests.post(RESULTS_URL, json=filters)
    resp.raise_for_status()
    return resp.json().get("data", [])

# 2) Convert raw records to LangChain Documents

def records_to_documents(records: list[dict], text_key: str) -> list[Document]:
    docs: list[Document] = []
    for rec in records:
        text = rec.get(text_key, "")
        if not text:
            continue
        metadata = {k: v for k, v in rec.items() if k != text_key}
        docs.append(Document(page_content=text, metadata=metadata))
    return docs

# 3) Load and combine API data
model_data = fetch_model_metadata()
time_series_data = fetch_time_series({
    "study": ["paris-reinforce"],
    "workspace_code": ["eu-headed"]
})
combined = model_data + time_series_data
# Ensure every record has `description`
for record in combined:
    record.setdefault("description", record.get("modelName", ""))

# 4) Split into text chunks
docs = records_to_documents(combined, text_key="description")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(docs)

# 5) Build or load FAISS index with embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY
)
index_dir = "faiss_index"
if os.path.exists(index_dir):
    vectorstore = FAISS.load_local(
    index_dir,
    embeddings,
    allow_dangerous_deserialization=True
)
else:
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_dir)

# 6) Set up the conversational retrieval chain
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)
chat_chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(
        temperature=0,
        api_key=OPENAI_API_KEY
    ),
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    memory=memory
)

# 7) Chat function

def chat(user_input: str) -> str:
    return chat_chain.run(user_input)

# 8) Command-line interface
if __name__ == '__main__':
    print("Chatbot ready! (type 'exit' or 'quit' to stop)")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        response = chat(user_input)
        print("Bot:", response)
