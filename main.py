import os
import sys
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv

# Load project configuration before importing modules whose constants are read
# from the environment at import time (LLM model selection and API settings).
# Deployment-provided environment variables retain precedence over .env.
load_dotenv()

import glob
import requests.exceptions
import time
import hashlib
import re
from llm_config import EXTRACTOR_MODEL, QA_MODEL, ROUTER_MODEL
from pathlib import Path
import argparse
import requests
import subprocess
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime, timezone
import logging
import base64

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llm_factory import get_chat_openai as ChatOpenAI, get_embeddings, is_local_model
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
from model_aliases import is_presentable_model_label
from runtime_context import build_runtime_context
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


def cache_file_timestamp(path: str | Path | None) -> str:
    """Return the UTC modification time for the cache file actually loaded."""
    if not path:
        return ""
    try:
        modified = Path(path).stat().st_mtime
    except OSError:
        return ""
    return datetime.fromtimestamp(modified, tz=timezone.utc).isoformat()




def _definition_source_signature() -> str:
    """Fingerprint definition contents so their parsed cache cannot go stale."""
    digest = hashlib.sha256()
    definition_root = Path("definitions")
    files = sorted(
        path for path in definition_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}
    )
    for path in files:
        digest.update(path.relative_to(definition_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_definitions():
    # Cache parsed definitions only while the YAML source fingerprint matches.
    cache_file = 'cache/yaml_definitions.pkl'
    source_signature = _definition_source_signature()
    if os.path.exists(cache_file):
        logging.getLogger(__name__).info('loading yaml definitions from file cache..')
        try:
            with open(cache_file, 'rb') as f:
                cached = pickle.load(f)
            if (
                isinstance(cached, dict)
                and cached.get("source_signature") == source_signature
                and "result" in cached
            ):
                return cached["result"]
            logging.getLogger(__name__).info(
                'YAML definition sources changed; rebuilding parsed cache.'
            )
        except Exception:
            logging.getLogger(__name__).warning(
                'Failed to load %s; regenerating from YAML.', cache_file, exc_info=True
            )

    #print and parse yaml files
    logging.getLogger(__name__).info('loading and parsing yaml files..')
    region_path = Path('definitions/region').resolve()
    variable_path = Path('definitions/variable').resolve()
    region_yaml = load_all_yaml_files(str(region_path))
    variable_yaml = load_all_yaml_files(str(variable_path))
    result = yaml_to_documents(region_yaml), yaml_to_documents(variable_yaml)

    #save to cache
    os.makedirs('cache',exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump({"source_signature": source_signature, "result": result}, f)
    
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
        if "modelName" in rec and not is_presentable_model_label(rec.get("modelName")):
            # Keep source records in the runtime dataset, but do not put a
            # numeric-only identifier into semantic retrieval where it could
            # be repeated as if it were a meaningful model name.
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
    Treat every supplied current response, including an empty one, as
    authoritative. Only combine result caches when no current response was
    supplied at all (``current_records is None``).

    Historical result files may contain series that were deliberately removed
    upstream, so they must never be merged back into a successful current
    response merely because the union is larger.
    """
    if current_records is not None:
        return list(current_records), "current"

    cache_files = sorted(glob.glob("cache/results*.json"))
    if not cache_files:
        return [], "current"

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

    return merged, "cache-fallback" if merged else "current"


def _faiss_cache_signature(docs: list, embeddings: Any) -> str:
    """Return a stable signature for documents and embedding configuration."""
    digest = hashlib.sha256()
    embedding_config = {
        "class": f"{embeddings.__class__.__module__}.{embeddings.__class__.__qualname__}",
        "model": getattr(embeddings, "model", None),
        "dimensions": getattr(embeddings, "dimensions", None),
    }
    digest.update(json.dumps(embedding_config, sort_keys=True, default=str).encode("utf-8"))
    digest.update(b"\0")
    for doc in docs:
        payload = {
            "page_content": str(getattr(doc, "page_content", "")),
            "metadata": getattr(doc, "metadata", {}) or {},
        }
        digest.update(json.dumps(payload, sort_keys=True, default=str).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()

def build_faiss_index(docs:list, embeddings) ->FAISS:
    # Load only when the indexed content and embedding configuration match.
    index_dir = 'cache/faiss_index'
    index_file = os.path.join(index_dir, 'index.faiss')
    signature_file = Path(index_dir) / "signature.json"
    expected_signature = _faiss_cache_signature(docs, embeddings)

    if os.path.exists(index_file) and signature_file.exists():
        try:
            cached_signature = json.loads(signature_file.read_text()).get("signature")
            if cached_signature == expected_signature:
                logger.info('Loading FAISS index from validated file cache ..')
                return FAISS.load_local(
                    index_dir,
                    embeddings,
                    allow_dangerous_deserialization=True,
                )
            logger.info('FAISS source signature changed; rebuilding index.')
        except (OSError, ValueError, TypeError):
            logger.warning('Could not validate FAISS cache; rebuilding it.', exc_info=True)
    elif os.path.exists(index_file):
        logger.info('Legacy FAISS cache has no source signature; rebuilding index.')

    # Create FAISS index if cache doesn't exist
    logger.info('Creating FAISS index...')
    faiss_index = FAISS.from_documents(docs, embeddings)
    
    # Save to cache using FAISS native save method
    os.makedirs(index_dir, exist_ok=True)
    faiss_index.save_local(index_dir)
    signature_file.write_text(json.dumps({"signature": expected_signature}))
    
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
        logger.error("Error saving plot: %s", e)
        return ""


def open_plot_file(file_path: str) -> None:
    try:
        if file_path and os.path.exists(file_path):
            subprocess.Popen(["open", file_path])
    except Exception as e:
        logger.error("Error opening plot: %s", e)


def _extract_plot_markdown(response: str) -> tuple[str, str]:
    text = str(response or "")
    match = re.search(r"!\[Plot\]\((data:image/png;base64,[^)]+)\)", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return text.strip(), ""
    message = (text[:match.start()] + text[match.end():]).strip()
    return message, match.group(1)


def _normalize_cli_query(query: str) -> str:
    return re.sub(r"^(?:\s*(?:you|query):\s*)+", "", str(query or ""), flags=re.IGNORECASE).strip()


def load_workspace_codes(path: str = "config/workspaces.json") -> list[str]:
    """Load the complete configured workspace catalogue outside application code."""
    data = json.loads(Path(path).read_text())
    workspaces = data.get("workspaces", []) if isinstance(data, dict) else []
    values = list(dict.fromkeys(str(value).strip() for value in workspaces if str(value).strip()))
    if not values:
        raise RuntimeError(f"No workspace codes configured in {path}")
    return values

class IAMParisBot:
    def __init__(self, streaming: bool = True):
        self.streaming = streaming
        self.logger = logging.getLogger(__name__)
        self.history: List[Tuple[str, str]] = []
        self.load_env()

    def load_env(self):
        load_dotenv()
        self.env = {
            k: os.getenv(k) for k in ("OPENAI_API_KEY", "REST_MODELS_URL", "REST_API_FULL")
        }
        required = ["REST_MODELS_URL", "REST_API_FULL"]
        if self._openai_required():
            required.append("OPENAI_API_KEY")
        if missing := [k for k in required if not self.env.get(k)]:
            raise RuntimeError(f"Missing environment variables: {', '.join(missing)}")

    @staticmethod
    def _openai_required() -> bool:
        """OpenAI credentials are only mandatory while a role or embeddings use OpenAI."""
        embeddings_local = bool(os.getenv("IAM_EMBEDDING_MODEL", "").strip())
        roles_local = all(
            is_local_model(model) for model in (ROUTER_MODEL, EXTRACTOR_MODEL, QA_MODEL)
        )
        return not (embeddings_local and roles_local)

    def fetch_json(self, url: str, params=None, payload=None, cache=True, max_retries=3) -> list:
        os.makedirs("cache", exist_ok=True)
        def _strip_internal(d: dict) -> dict:
            return {k: v for k, v in d.items() if not str(k).startswith("_")}
        def _record_key(record: dict):
            stable_id = record.get("resultId") or record.get("id")
            if stable_id not in (None, ""):
                return ("id", str(stable_id))
            return (
                "scope",
                str(record.get("workspace_code", "")),
                str(record.get("study", "")),
                str(record.get("modelName", "")),
                str(record.get("scenario", "")),
                str(record.get("region", "")),
                str(record.get("variable", "")),
                str(record.get("unit", "")),
                json.dumps(record.get("years", {}), sort_keys=True, default=str),
            )
        def _expand_by_workspace(url: str, payload_clean: dict, timeout: int) -> list:
            all_records = []
            seen = set()
            page_limit = 1000
            for ws in payload_clean.get("workspace_code", []):
                page = 0
                while True:
                    ws_payload = dict(payload_clean)
                    ws_payload.update({"workspace_code": [ws], "limit": page_limit, "offset": page})
                    resp_ws = requests.post(url, json=ws_payload, timeout=timeout)
                    self.logger.info(
                        "API call completed: status %s (workspace=%s, page=%s)",
                        resp_ws.status_code, ws, page,
                    )
                    resp_ws.raise_for_status()
                    data_ws = resp_ws.json()
                    records_ws = data_ws.get("data") if isinstance(data_ws, dict) else data_ws
                    if not isinstance(records_ws, list) or not records_ws:
                        break
                    new_count = 0
                    for record in records_ws:
                        key = _record_key(record)
                        if key in seen:
                            continue
                        seen.add(key)
                        all_records.append(record)
                        new_count += 1
                    if len(records_ws) < page_limit or new_count == 0:
                        break
                    page += 1
                self.logger.info("Workspace %s complete after %s page(s)", ws, page + 1)
            return all_records
        # Convert params and payload to strings for hashing if they contain dicts
        # Internal control flags affect how the request is executed, not the
        # identity of the remote data. A forced refresh must overwrite the
        # cache used by the equivalent ordinary startup request.
        params_clean = _strip_internal(params or {})
        payload_for_cache = _strip_internal(payload or {})
        params_str = str(sorted(params_clean.items())) if params is not None else ""
        payload_str = str(sorted(payload_for_cache.items())) if payload is not None else ""
        # Use hashlib for consistent hashing across Python sessions
        import hashlib
        pagination_cache_version = (
            "paged-workspaces-v2"
            if payload and payload.get("limit") == -1 and payload.get("workspace_code")
            else ""
        )
        hash_key = hashlib.md5(
            (url + "\0" + params_str + payload_str + pagination_cache_version).encode()
        ).hexdigest()[:16]
        cache_file = f"cache/{url.split('/')[-1]}_{hash_key}.json"
        self.last_fetch_cache_file = cache_file
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
        self.logger.info("Fetching data from %s ...", url)
        
        # Retry logic with exponential backoff
        for attempt in range(max_retries):
            try:
                if payload is not None:
                    payload_clean = _strip_internal(payload)
                    # Support paged fetch when limit == -1 for POST endpoints
                    if payload_clean.get("limit") == -1:
                        if (
                            "results" in url
                            and isinstance(payload_clean.get("workspace_code"), list)
                        ):
                            combined = _expand_by_workspace(url, payload_clean, timeout)
                            self.logger.info("Records fetched: %s (all workspace pages)", len(combined))
                            with open(cache_file, 'w') as f:
                                pd.DataFrame(combined).to_json(f)
                            return combined
                        combined = []
                        seen_ids = set()
                        page_limit = 1000
                        offset = 0
                        while True:
                            paged_payload = dict(payload_clean)
                            paged_payload["limit"] = page_limit
                            paged_payload["offset"] = offset
                            resp = requests.post(url, json=paged_payload, timeout=timeout)
                            self.logger.info("API call completed: status %s", resp.status_code)
                            if resp.status_code >= 500:
                                cached = _load_cache()
                                if cached:
                                    self.logger.warning("API returned 5xx; using cached data.")
                                    return cached
                            resp.raise_for_status()
                            data = resp.json()
                            records = data.get("data") if isinstance(data, dict) else data
                            if not records:
                                break
                            # If no id field, stop after first page to avoid duplicates
                            if not isinstance(records, list) or not records or not (
                                records[0].get("id") or records[0].get("resultId")
                            ):
                                # If results API is capped and no id field, expand by workspace
                                if (
                                    "results" in url
                                    and isinstance(payload_clean.get("workspace_code"), list)
                                ):
                                    combined = _expand_by_workspace(url, payload_clean, timeout)
                                else:
                                    combined.extend(records if isinstance(records, list) else [])
                                break
                            new_records = [r for r in records if _record_key(r) not in seen_ids]
                            for r in new_records:
                                seen_ids.add(_record_key(r))
                            combined.extend(new_records)
                            if len(records) < page_limit or len(new_records) == 0:
                                break
                            offset += 1
                        self.logger.info("Records fetched: %s", len(combined))
                        with open(cache_file, 'w') as f:
                            pd.DataFrame(combined).to_json(f)
                        return combined
                    resp = requests.post(url, json=payload_clean, timeout=timeout)
                else:
                    resp = requests.get(url, params=params, timeout=timeout)
                self.logger.info("API call completed: status %s", resp.status_code)
                # If server is down, fall back to cache when available
                if resp.status_code >= 500:
                    cached = _load_cache()
                    if cached:
                        self.logger.warning("API returned 5xx; using cached data.")
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
                    self.logger.info("Records fetched: %s (expanded by workspace)", len(all_records))
                    with open(cache_file, 'w') as f:
                        pd.DataFrame(all_records).to_json(f)
                    return all_records

                self.logger.info("Records fetched: %s", len(records))
                with open(cache_file, 'w') as f:
                    pd.DataFrame(records).to_json(f)
                return records
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 5  # Exponential backoff: 5, 10, 20 seconds...
                    self.logger.warning("Request failed (%s), retrying in %ss... (attempt %s/%s)", type(e).__name__, wait_time, attempt + 1, max_retries)
                    time.sleep(wait_time)
                else:
                    cached = _load_cache()
                    if cached:
                        self.logger.warning("API connection failed; using cached data.")
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
            model_name=QA_MODEL,
            temperature=0,
            streaming=self.streaming,
            timeout=30,
            max_retries=1,
            callbacks=[StreamingStdOutCallbackHandler()] if self.streaming else None
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
        
        # Fetch every page for every workspace configured in one data file.
        all_workspaces = load_workspace_codes()
        ts_payload = {
            "workspace_code": all_workspaces,
            "limit": -1,
            "_force_refresh": args.refresh_data,
        }
        ts = bot.fetch_json(bot.env["REST_API_FULL"], payload=ts_payload, cache=True)
        ts, ts_source = load_best_cached_results(ts)
        results_source_path = str(getattr(bot, "last_fetch_cache_file", "") or "")
        results_timestamp = cache_file_timestamp(results_source_path)
        
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

    # Always derive the source signature; build_faiss_index reuses the cache
    # only when the model metadata and YAML-derived chunks still match it.
    region_docs, variable_docs = load_definitions()
    all_docs = docs_from_records(models) + region_docs + variable_docs
    chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80).split_documents(all_docs)
    embeddings = get_embeddings(model="text-embedding-3-small", api_key=bot.env["OPENAI_API_KEY"], timeout=30, max_retries=1)
    faiss_index = build_faiss_index(chunks, embeddings)

    shared_resources = build_runtime_context(
        models=models,
        ts=ts,
        workspace_lookup=workspace_lookup,
        vector_store=faiss_index,
        env=bot.env,
        bot=bot,
        results_source_path=results_source_path,
        results_timestamp=results_timestamp,
    )

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
