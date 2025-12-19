import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from typing import List, Tuple
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests.exceptions
import logging
logger = logging.getLogger(__name__)
from main import IAMParisBot, docs_from_records, build_faiss_index
from utils.yaml_loader import load_all_yaml_files, yaml_to_documents
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from manager import MultiAgentManager




#Add caching
def load_definitions():
    # Try file cache
    cache_file = "cache/yaml_definitions.pkl"
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    # Load and parse YAML files
    region_path = Path('definitions/region').resolve()
    variable_path = Path('definitions/variable').resolve()
    region_yaml = load_all_yaml_files(str(region_path))
    variable_yaml = load_all_yaml_files(str(variable_path))
    result = yaml_to_documents(region_yaml), yaml_to_documents(variable_yaml)
    
    # Save to cache
    os.makedirs("cache", exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(result, f)
    
    return result


# Pydantic Models
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    history: List[Tuple[str, str]] = []

# FastAPI Setup
app = FastAPI(
    title='IAM Paris Data Chatbot API',
    description='Multi-agent conversational AI for IAM Paris climate data',
    version='1.0.0'
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post('/query', response_model=QueryResponse)
def query_chatbot(req: QueryRequest):
    """
    Process a user query through the multi-agent system.

    - **query**: The user's question or request
    - Returns structured response with answer and conversation history
    """
    try:
        # Initialize bot (non-streaming for API)
        bot = IAMParisBot(streaming=False)

        # Load models and timeseries data (energy-systems workspace only)
        try:
            models = bot.fetch_json(bot.env['REST_MODELS_URL'], params={'limit': -1}, cache=False)
            ts_payload = {'workspace_code': ['energy-systems'], 'limit': -1}
            ts = bot.fetch_json(bot.env['REST_API_FULL'], payload=ts_payload, cache=False)
        except RuntimeError as e:
            logger.error(f"Failed to fetch data: {e}")
            raise HTTPException(status_code=503, detail=f"Data service unavailable: {str(e)}")


        # Prepare documents and index
        all_docs = docs_from_records(models + ts) + region_docs + variable_docs
        chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80).split_documents(all_docs)
        embeddings = OpenAIEmbeddings(model='text-embedding-3-small', api_key=bot.env['OPENAI_API_KEY'])
        faiss_index = build_faiss_index(chunks, embeddings)

        # Setup shared resources
        shared_resources = {
            'models': models,
            'ts': ts,
            'vector_store': faiss_index,
            'env': bot.env,
            'bot': bot
        }

        # Initialize multi-agent manager
        manager = MultiAgentManager(shared_resources, streaming=False)
        chat_history: List[Tuple[str, str]] = []

        # Route query through appropriate agent
        response = manager.route_query(req.query, chat_history)
        chat_history.append((req.query, response))

        return QueryResponse(answer=response, history=chat_history)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get('/')
def root():
    """Root endpoint with API information"""
    return {
        "message": "IAM Paris Data Chatbot API",
    
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)