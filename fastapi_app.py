import os
import sys
import pickle
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from typing import List, Tuple, Optional
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

# Configuration
INITIALIZATION_TIMEOUT = 300  # 5 minutes timeout for cache building
API_REQUEST_TIMEOUT = 120    # 2 minutes timeout for individual API calls

# Global variables for cached data
_cached_resources = None
_initialization_status = "not_started"  # not_started, initializing, ready, error
_initialization_error = None
_initialization_start_time = None

def load_definitions():
    """Load YAML definitions with caching."""
    cache_file = "cache/yaml_definitions.pkl"
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    region_path = Path('definitions/region').resolve()
    variable_path = Path('definitions/variable').resolve()
    region_yaml = load_all_yaml_files(str(region_path))
    variable_yaml = load_all_yaml_files(str(variable_path))
    result = yaml_to_documents(region_yaml), yaml_to_documents(variable_yaml)
    
    os.makedirs("cache", exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(result, f)
    
    return result


def _check_timeout(operation: str):
    """Check if initialization has exceeded timeout."""
    if _initialization_start_time is None:
        return
    elapsed = time.time() - _initialization_start_time
    if elapsed > INITIALIZATION_TIMEOUT:
        raise TimeoutError(f"Initialization exceeded {INITIALIZATION_TIMEOUT}s during {operation}")


def initialize_resources():
    """Initialize all resources once at startup with timeout protection."""
    global _cached_resources, _initialization_status, _initialization_error, _initialization_start_time
    
    if _cached_resources is not None:
        return _cached_resources
    
    _initialization_status = "initializing"
    _initialization_start_time = time.time()
    logger.info("=" * 50)
    logger.info("Starting resource initialization...")
    logger.info(f"Timeout configured: {INITIALIZATION_TIMEOUT}s")
    logger.info("=" * 50)
    
    try:
        # Ensure cache directory exists with proper permissions
        logger.info("Creating cache directories...")
        os.makedirs("cache", exist_ok=True)
        os.makedirs("cache/faiss_index", exist_ok=True)
        
        # Initialize bot
        logger.info("Initializing bot...")
        bot = IAMParisBot(streaming=False)
        
        # Load data with caching and timeout check
        logger.info("Fetching models data...")
        _check_timeout("models fetch")
        models = bot.fetch_json(bot.env['REST_MODELS_URL'], params={'limit': -1}, cache=True)
        logger.info(f"Loaded {len(models)} models")
        
        logger.info("Fetching timeseries data (this may take a minute)...")
        _check_timeout("timeseries fetch")
        all_workspaces = [
            "afolu", "buildings-transf", "covid-rec", "decarb-potentials", "decipher_1",
            "energy-systems", "eu-headed", "index-decomp", "industrial-transf", "ndcs-impacts",
            "net-zero", "post-glasgow", "power-people", "study-1", "study-2", "study-3",
            "study-4", "study-6", "study-7", "transp-transf", "world-headed"
        ]
        ts_payload = {'workspace_code': all_workspaces}
        ts = bot.fetch_json(bot.env['REST_API_FULL'], payload=ts_payload, cache=True)
        logger.info(f"Loaded {len(ts)} timeseries records")
        
        # Build FAISS index
        logger.info("Loading YAML definitions...")
        _check_timeout("YAML definitions")
        region_docs, variable_docs = load_definitions()
        logger.info(f"Loaded {len(region_docs)} region docs, {len(variable_docs)} variable docs")
        
        logger.info("Building document chunks...")
        _check_timeout("document chunking")
        # NOTE: timeseries (ts) data is NOT embedded - it's numeric data, not semantic text
        # ts is still available in _cached_resources for querying but not in vector store
        all_docs = docs_from_records(models) + region_docs + variable_docs
        chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80).split_documents(all_docs)
        logger.info(f"Created {len(chunks)} document chunks")
        
        logger.info("Building FAISS vector index...")
        _check_timeout("FAISS index building")
        embeddings = OpenAIEmbeddings(model='text-embedding-3-small', api_key=bot.env['OPENAI_API_KEY'])
        faiss_index = build_faiss_index(chunks, embeddings)
        logger.info("FAISS index built successfully")
        
        # Cache resources
        _cached_resources = {
            'models': models,
            'ts': ts,
            'vector_store': faiss_index,
            'env': bot.env,
            'bot': bot
        }
        
        _initialization_status = "ready"
        elapsed = time.time() - _initialization_start_time
        logger.info("=" * 50)
        logger.info(f"Resources initialized in {elapsed:.1f} seconds")
        logger.info("=" * 50)
        
        return _cached_resources
        
    except TimeoutError as e:
        _initialization_status = "error"
        _initialization_error = f"Timeout: {str(e)}"
        logger.error(f"Initialization timed out: {e}")
        raise
    except requests.exceptions.Timeout as e:
        _initialization_status = "error"
        _initialization_error = f"API request timeout: {str(e)}"
        logger.error(f"API request timed out: {e}")
        raise
    except requests.exceptions.ConnectionError as e:
        _initialization_status = "error"
        _initialization_error = f"Connection error: {str(e)}"
        logger.error(f"Connection error during initialization: {e}")
        raise
    except Exception as e:
        _initialization_status = "error"
        _initialization_error = str(e)
        logger.error(f"Initialization failed: {e}")
        raise


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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize resources when the server starts."""
    initialize_resources()


@app.post('/query', response_model=QueryResponse)
def query_chatbot(req: QueryRequest):
    """
    Process a user query through the multi-agent system.
    Uses cached resources for fast response.
    """
    # Check if resources are ready
    if _initialization_status == "initializing":
        raise HTTPException(
            status_code=503, 
            detail="Service is initializing. Please try again in a moment."
        )
    
    if _initialization_status == "error":
        raise HTTPException(
            status_code=503,
            detail=f"Service unavailable due to initialization error: {_initialization_error}"
        )
    
    if _cached_resources is None:
        raise HTTPException(
            status_code=503,
            detail="Service not ready. Resources not loaded."
        )
    
    try:
        # Get cached resources
        resources = _cached_resources
        
        # Create a new manager for each request (lightweight)
        manager = MultiAgentManager(resources, streaming=False)
        chat_history: List[Tuple[str, str]] = []
        
        # Route query
        response = manager.route_query(req.query, chat_history)
        chat_history.append((req.query, response))
        
        return QueryResponse(answer=response, history=chat_history)
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get('/')
def root():
    """Root endpoint with API information"""
    return {
        "message": "IAM Paris Data Chatbot API",
        "status": "ready" if _cached_resources else "initializing"
    }


@app.get('/health')
def health_check():
    """Health check endpoint with initialization status."""
    elapsed = None
    if _initialization_start_time:
        elapsed = time.time() - _initialization_start_time
    
    return {
        "status": _initialization_status,
        "resources_loaded": _cached_resources is not None,
        "error": _initialization_error,
        "elapsed_seconds": round(elapsed, 1) if elapsed else None,
        "timeout_limit": INITIALIZATION_TIMEOUT
    }


@app.get('/status')
def status_check():
    """Detailed status endpoint for monitoring cache readiness."""
    elapsed = None
    if _initialization_start_time:
        elapsed = time.time() - _initialization_start_time
    
    # Count cached items
    cache_status = {
        "models_count": len(_cached_resources.get('models', [])) if _cached_resources else 0,
        "timeseries_count": len(_cached_resources.get('ts', [])) if _cached_resources else 0,
        "vector_store_ready": _cached_resources.get('vector_store') is not None if _cached_resources else False
    }
    
    return {
        "initialization": {
            "status": _initialization_status,
            "error": _initialization_error,
            "elapsed_seconds": round(elapsed, 1) if elapsed else None,
            "timeout_limit": INITIALIZATION_TIMEOUT
        },
        "cache": cache_status,
        "ready": _initialization_status == "ready" and _cached_resources is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
