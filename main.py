import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import glob
import requests.exceptions
import time
import hashlib
import re
from pathlib import Path
import argparse
import requests
import subprocess
import pandas as pd
from dotenv import load_dotenv
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime
import logging
import base64

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import ConversationalRetrievalChain
from langchain.memory.buffer import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from data_utils import data_query
from utils.yaml_loader import load_all_yaml_files, yaml_to_documents
from manager import MultiAgentManager
from utils_query import (
    get_available_models,
    get_available_scenarios,
    get_available_variables_from_yaml)

import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)




def load_definitions():
    #try file cache
    cache_file ='cache/yaml_definitions.pkl'
    if os.path.exists(cache_file):
        print('loading yaml definitions from file cache..')
        with open(cache_file,'rb') as f:
            return pickle.load(f)
        
    #print and parse yaml files
    print('loading and parsing yaml files..')
    region_path = Path('definitions/region').resolve()
    variable_path = Path('definitions/variable').resolve()
    region_yaml = load_all_yaml_files(str(region_path))
    variable_yaml = load_all_yaml_files(str(variable_path))
    result = yaml_to_documents(region_yaml), yaml_to_documents(variable_yaml)

    #save to cache
    os.makedirs('cache',exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(result,f)
    
    return result

def setup_logging(debug: bool = False):
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if debug else logging.INFO)
    root_logger.handlers.clear()

    file_handler = logging.FileHandler('chatbot.log')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(file_handler)

    if debug:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        root_logger.addHandler(console_handler)




def docs_from_records(records: list) -> List[Document]:
    docs = []
    for rec in records:
        if rec is None:
            continue
        # Handle case where description/modelName might be float (nan) instead of string
        desc_val = rec.get("description") or rec.get("modelName") or ""
        desc = str(desc_val).strip() if desc_val else ""
        asum_val = rec.get("assumptions") or ""
        asum = str(asum_val).strip() if asum_val else ""
        if not desc and not asum:
            continue
        content = desc + (f"\n\nAssumptions: {asum}" if asum else "")
        doc = Document(
            page_content=content,
            metadata={
                "modelName": rec.get("modelName", ""),
                "variable": rec.get("variable", ""),
                "unit": rec.get("unit", ""),
                "study": rec.get("study", ""),
                "scenario": rec.get("scenario", ""),
                "type": "model" if "modelName" in rec else "timeseries"
            }
        )
        docs.append(doc)
    return docs


def load_best_cached_results(current_records: list | None = None) -> tuple[list, str]:
    """
    Merge all cached results files and prefer the richest deduplicated dataset.
    This helps the live app use the fullest local cache for query clarification.
    """
    cache_files = sorted(glob.glob("cache/results*.json"))
    if not cache_files:
        return current_records or [], "current"

    best_records = list(current_records or [])
    best_source = "current"
    seen = set()
    merged = []

    def _record_key(record: dict) -> tuple[str, str, str, str, str, str]:
        return (
            str(record.get("resultId", "")),
            str(record.get("workspace_code", "")),
            str(record.get("modelName", "")),
            str(record.get("scenario", "")),
            str(record.get("region", "")),
            str(record.get("variable", "")),
        )

    for record in best_records:
        if record is None:
            continue
        key = _record_key(record)
        if key in seen:
            continue
        seen.add(key)
        merged.append(record)

    for cache_file in cache_files:
        try:
            records = pd.read_json(cache_file).to_dict("records")
        except Exception:
            continue
        for record in records:
            if record is None:
                continue
            key = _record_key(record)
            if key in seen:
                continue
            seen.add(key)
            merged.append(record)

    if len(merged) > len(best_records):
        best_records = merged
        best_source = "merged-cache"

    merged_cache_file = "cache/results_merged.json"
    if best_source == "merged-cache":
        try:
            pd.DataFrame(best_records).to_json(merged_cache_file)
        except Exception:
            pass

    return best_records, best_source

def build_faiss_index(docs:list, embeddings) ->FAISS:
    #try file cache
    index_dir = 'cache/faiss_index'
    index_file = os.path.join(index_dir, 'index.faiss')
    
    if os.path.exists(index_file):
        print('Loading FAISS index from file cache ..')
        return FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    
    # Create FAISS index if cache doesn't exist
    print('Creating FAISS index...')
    faiss_index = FAISS.from_documents(docs, embeddings)
    
    # Save to cache using FAISS native save method
    os.makedirs(index_dir, exist_ok=True)
    faiss_index.save_local(index_dir)
    
    return faiss_index

#clear cache
def clear_cache():
    """Clear all cached data."""
    import shutil
    if os.path.exists("cache"):
        shutil.rmtree("cache")
    
    print("Cache cleared")



def _slugify_filename(text: str, fallback: str = "plot") -> str:
    cleaned = (text or "").strip().lower()
    cleaned = re.sub(r"^(plot|show|graph|chart|visualize|display|please)\s+", "", cleaned)
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", cleaned).strip("_")
    return slug or fallback


def save_plot_from_base64(base64_string: str, output_dir: str = "plots", label: str | None = None) -> str:
    """
    Save a base64 PNG plot to disk and return the file path.
    """
    try:
        if "data:image/png;base64," in base64_string:
            base64_data = base64_string.split("data:image/png;base64,")[1]
        else:
            base64_data = base64_string
        image_bytes = base64.b64decode(base64_data)
        os.makedirs(output_dir, exist_ok=True)
        if label:
            digest = hashlib.sha1(image_bytes).hexdigest()[:10]
            file_name = f"plot_{_slugify_filename(label)}_{digest}.png"
        else:
            ts = int(time.time())
            file_name = f"plot_{ts}.png"
        file_path = os.path.join(output_dir, file_name)
        with open(file_path, "wb") as f:
            f.write(image_bytes)
        return file_path
    except Exception as e:
        print(f"Error saving plot: {e}")
        return ""


def open_plot_file(file_path: str) -> None:
    try:
        if file_path and os.path.exists(file_path):
            subprocess.Popen(["open", file_path])
    except Exception as e:
        print(f"Error opening plot: {e}")


def _extract_plot_markdown(response: str) -> tuple[str, str]:
    text = str(response or "")
    match = re.search(r"!\[Plot\]\((data:image/png;base64,[^)]+)\)", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return text.strip(), ""
    message = (text[:match.start()] + text[match.end():]).strip()
    return message, match.group(1)


def _normalize_cli_query(query: str) -> str:
    return re.sub(r"^(?:\s*(?:you|query):\s*)+", "", str(query or ""), flags=re.IGNORECASE).strip()

class IAMParisBot:
    def __init__(self, streaming: bool = True):
        self.streaming = streaming
        self.logger = logging.getLogger(__name__)
        self.history: List[Tuple[str, str]] = []
        self.load_env()

    def load_env(self):
        load_dotenv(override=True)
        required = ["OPENAI_API_KEY", "REST_MODELS_URL", "REST_API_FULL"]
        self.env = {k: os.getenv(k) for k in required}
        if missing := [k for k, v in self.env.items() if not v]:
            raise RuntimeError(f"Missing environment variables: {', '.join(missing)}")

    def fetch_json(self, url: str, params=None, payload=None, cache=True, max_retries=3) -> list:
        os.makedirs("cache", exist_ok=True)
        def _strip_internal(d: dict) -> dict:
            return {k: v for k, v in d.items() if not str(k).startswith("_")}
        def _expand_by_workspace(url: str, payload_clean: dict, timeout: int) -> list:
            all_records = []
            seen = set()
            for ws in payload_clean.get("workspace_code", []):
                ws_payload = dict(payload_clean)
                ws_payload["workspace_code"] = [ws]
                resp_ws = requests.post(url, json=ws_payload, timeout=timeout)
                print(f"API call completed: status {resp_ws.status_code} (workspace={ws})")
                if resp_ws.status_code >= 500:
                    continue
                resp_ws.raise_for_status()
                data_ws = resp_ws.json()
                records_ws = data_ws.get("data") if isinstance(data_ws, dict) else data_ws
                for r in records_ws or []:
                    key = (
                        str(r.get("resultId", "")),
                        str(r.get("workspace_code", "")),
                        str(r.get("modelName", "")),
                        str(r.get("scenario", "")),
                        str(r.get("region", "")),
                        str(r.get("variable", "")),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    all_records.append(r)
            return all_records
        # Convert params and payload to strings for hashing if they contain dicts
        params_str = str(sorted(params.items())) if params is not None else ""
        payload_str = str(sorted(payload.items())) if payload is not None else ""
        # Use hashlib for consistent hashing across Python sessions
        import hashlib
        hash_key = hashlib.md5((params_str + payload_str).encode()).hexdigest()[:16]
        cache_file = f"cache/{url.split('/')[-1]}_{hash_key}.json"
        def _load_cache() -> list:
            if cache and os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    return pd.read_json(f).to_dict('records')
            return []

        if cache and payload and payload.get("_force_refresh"):
            # Skip cache lookup when explicitly forced
            pass
        elif cache and os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return pd.read_json(f).to_dict('records')
        # Use POST if payload is provided, otherwise GET
        # Use longer timeout for large data fetches
        timeout = 300 if payload is not None else 60
        print(f"Fetching data from {url}...")
        
        # Retry logic with exponential backoff
        for attempt in range(max_retries):
            try:
                if payload is not None:
                    payload_clean = _strip_internal(payload)
                    # Support paged fetch when limit == -1 for POST endpoints
                    if payload_clean.get("limit") == -1:
                        combined = []
                        seen_ids = set()
                        page_limit = 1000
                        offset = 0
                        while True:
                            paged_payload = dict(payload_clean)
                            paged_payload["limit"] = page_limit
                            paged_payload["offset"] = offset
                            resp = requests.post(url, json=paged_payload, timeout=timeout)
                            print(f"API call completed: status {resp.status_code}")
                            if resp.status_code >= 500:
                                cached = _load_cache()
                                if cached:
                                    print("API returned 5xx; using cached data.")
                                    return cached
                            resp.raise_for_status()
                            data = resp.json()
                            records = data.get("data") if isinstance(data, dict) else data
                            if not records:
                                break
                            # If no id field, stop after first page to avoid duplicates
                            if not isinstance(records, list) or not records or "id" not in records[0]:
                                # If results API is capped and no id field, expand by workspace
                                if (
                                    "results" in url
                                    and isinstance(payload_clean.get("workspace_code"), list)
                                ):
                                    combined = _expand_by_workspace(url, payload_clean, timeout)
                                else:
                                    combined.extend(records if isinstance(records, list) else [])
                                break
                            new_records = [r for r in records if r.get("id") not in seen_ids]
                            for r in new_records:
                                seen_ids.add(r.get("id"))
                            combined.extend(new_records)
                            if len(records) < page_limit or len(new_records) == 0:
                                break
                            offset += page_limit
                        print(f"Records fetched: {len(combined)}")
                        with open(cache_file, 'w') as f:
                            pd.DataFrame(combined).to_json(f)
                        return combined
                    resp = requests.post(url, json=payload_clean, timeout=timeout)
                else:
                    resp = requests.get(url, params=params, timeout=timeout)
                print(f"API call completed: status {resp.status_code}")
                # If server is down, fall back to cache when available
                if resp.status_code >= 500:
                    cached = _load_cache()
                    if cached:
                        print("API returned 5xx; using cached data.")
                        return cached
                resp.raise_for_status()
                data = resp.json()
                records = data.get("data") if isinstance(data, dict) else data
                # If results API appears capped, expand by querying per workspace
                if (
                    isinstance(records, list)
                    and "results" in url
                    and payload is not None
                    and isinstance(payload_clean.get("workspace_code"), list)
                    and len(records) >= 1000
                ):
                    all_records = _expand_by_workspace(url, payload_clean, timeout)
                    print(f"Records fetched: {len(all_records)} (expanded by workspace)")
                    with open(cache_file, 'w') as f:
                        pd.DataFrame(all_records).to_json(f)
                    return all_records

                print(f"Records fetched: {len(records)}")
                with open(cache_file, 'w') as f:
                    pd.DataFrame(records).to_json(f)
                return records
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 5  # Exponential backoff: 5, 10, 20 seconds...
                    print(f"Request failed ({type(e).__name__}), retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    cached = _load_cache()
                    if cached:
                        print("API connection failed; using cached data.")
                        return cached
                    raise RuntimeError(f"Failed to fetch data after {max_retries} attempts: {e}")
        return []

    def create_qa_chain(self, vs: FAISS) -> ConversationalRetrievalChain:
        memory = ConversationBufferMemory(
            chat_memory=ChatMessageHistory(),
            return_messages=True,
            memory_key="chat_history",
            output_key="answer",
            input_key="question"
        )
        system_tpl = """You are an expert climate policy assistant focused on IAM PARIS data and models (https://iamparis.eu/).

Always:
- Provide direct answers without restating the question
- Use Markdown formatting with headers and lists
- Reference IAM PARIS data when available
- Include IAM PARIS links
- Format numbers with units

Context: ```{context}```"""
        user_tpl = "Question: ```{question}```"
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_tpl),
            HumanMessagePromptTemplate.from_template(user_tpl)
        ])
        llm = ChatOpenAI(
            model_name="gpt-4-turbo",
            temperature=0,
            streaming=self.streaming,
            callbacks=[StreamingStdOutCallbackHandler()] if self.streaming else None,
            api_key=self.env["OPENAI_API_KEY"]
        )
        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vs.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
            memory=memory,
            chain_type="stuff",
            combine_docs_chain_kwargs={"prompt": prompt},
            verbose=False
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--query", type=str, help="Single query to process and exit")
    parser.add_argument("--clear-cache", action="store_true", help="Clear all cached data and exit")
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Force refresh API data instead of using cached responses",
    )
    args = parser.parse_args()

    if args.clear_cache:
        clear_cache()
        return

    setup_logging(args.debug)
    logger = logging.getLogger(__name__)

    bot = IAMParisBot(streaming=not args.no_stream)
    try:
        models = bot.fetch_json(bot.env["REST_MODELS_URL"], params={"limit": -1}, cache=True)
        
        # Fetch ALL data from IAMPARIS API (all workspaces)
        # Using workspace_code filter to get all data
        all_workspaces = [
            "afolu", "buildings-transf", "covid-rec", "decarb-potentials", "decipher_1",
            "energy-systems", "eu-headed", "index-decomp", "industrial-transf", "ndcs-impacts",
            "net-zero", "post-glasgow", "power-people", "study-1", "study-2", "study-3",
            "study-4", "study-6", "study-7", "transp-transf", "world-headed"
        ]
        ts_payload = {
            "workspace_code": all_workspaces,
            "limit": -1,
            "_force_refresh": args.refresh_data,
        }
        ts = bot.fetch_json(bot.env["REST_API_FULL"], payload=ts_payload, cache=True)
        ts, ts_source = load_best_cached_results(ts)
        
        print(f"ts fetch: {len(ts)} records ({ts_source})")

        # Create workspace lookup for filtering
        workspace_lookup = {}
        for record in ts:
            ws = record.get('workspace_code', 'unknown')
            if ws not in workspace_lookup:
                workspace_lookup[ws] = []
            workspace_lookup[ws].append(record)
        print(f"Workspaces loaded: {list(workspace_lookup.keys())}")
    except RuntimeError as e:
        logger.error(f"Failed to fetch data: {e}")
        print(f"Error: {e}")
        print("Please check your internet connection and try again.")
        return

    # Check if FAISS cache exists before processing documents
    index_dir = "cache/faiss_index"
    if os.path.exists(os.path.join(index_dir, "index.faiss")):
        print("Loading FAISS index from cache...")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=bot.env["OPENAI_API_KEY"])
        faiss_index = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    else:
        print("Creating new FAISS index...")
        region_docs, variable_docs = load_definitions()
        all_docs = docs_from_records(models) + region_docs + variable_docs
        chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80).split_documents(all_docs)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=bot.env["OPENAI_API_KEY"])
        faiss_index = FAISS.from_documents(chunks, embeddings)
        os.makedirs(index_dir, exist_ok=True)
        faiss_index.save_local(index_dir)

    shared_resources = {
        "models": models,
        "ts": ts,
        "workspace_lookup": workspace_lookup,
        "vector_store": faiss_index,
        "env": bot.env,
        "bot": bot
    }

    manager = MultiAgentManager(shared_resources, streaming=not args.no_stream)

    if args.query:
        # Process single query and exit
        history = []
        query = _normalize_cli_query(args.query)
        response = manager.route_query(query, history)
        message, plot_data = _extract_plot_markdown(response)
        if "No explicit assumptions field is available in the model metadata." in message:
            print("\nNOTICE: No explicit assumptions field is available in the model metadata.\n")
        if message:
            print("Response:", message)
        if plot_data:
            file_path = save_plot_from_base64(plot_data, label=query)
            if file_path:
                print(f"Response: [Plot saved at {file_path}]")
                open_plot_file(file_path)
            else:
                print("Response: [Plot Image]")
        return

    print("\nWelcome to the IAM PARIS Climate Policy Assistant! Type 'exit' to quit.\n")

    history = []
    while True:
        try:
            query = _normalize_cli_query(input("Query: "))
            if query.lower() in ("exit", "quit"):
                break
            if not query:
                continue

            response = manager.route_query(query, history)
            message, plot_data = _extract_plot_markdown(response)
            if "No explicit assumptions field is available in the model metadata." in message:
                print("\nNOTICE: No explicit assumptions field is available in the model metadata.\n")
            if message:
                print("\nBOT:", message, "\n")
            if plot_data:
                file_path = save_plot_from_base64(plot_data, label=query)
                if file_path:
                    print(f"\nBOT: [Plot saved at {file_path}]\n")
                    open_plot_file(file_path)
                    history.append((query, message or file_path))
                else:
                    print("\nBOT: [Plot Image]\n")
                    history.append((query, message or "[Plot Image]"))
            else:
                history.append((query, message))
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print("\nBOT: An error occurred. Please try again.\n")

if __name__ == "__main__":
    main()
