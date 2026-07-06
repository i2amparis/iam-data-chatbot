import os
import sys
import pickle
import time
import re
import uuid
import json
import threading
from collections import OrderedDict
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime, timezone
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import requests.exceptions
import logging
logger = logging.getLogger(__name__)
from main import IAMParisBot, docs_from_records, build_faiss_index, load_best_cached_results
from utils.yaml_loader import load_all_yaml_files, yaml_to_documents
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from manager import MultiAgentManager
from runtime_context import build_runtime_context

# Configuration
INITIALIZATION_TIMEOUT = 300  # 5 minutes timeout for cache building
API_REQUEST_TIMEOUT = 120    # 2 minutes timeout for individual API calls

# Global variables for cached data
_cached_resources = None
_initialization_status = "not_started"  # not_started, initializing, ready, error
_initialization_error = None
_initialization_start_time = None
SESSION_TTL_SECONDS = 3600
MAX_SESSIONS = int(os.getenv("IAM_MAX_SESSIONS", "500"))
HISTORY_MAX_TURNS = int(os.getenv("IAM_HISTORY_MAX_TURNS", "20"))
_sessions: "OrderedDict[str, dict]" = OrderedDict()
_sessions_lock = threading.Lock()

# --- Access control ---------------------------------------------------------
# When IAM_API_KEY is set, all data endpoints require a matching X-API-Key
# header. When unset (e.g. local dev) auth is disabled but a warning is logged.
API_KEY = os.getenv("IAM_API_KEY", "").strip()
if not API_KEY:
    logger.warning("IAM_API_KEY is not set: /query, /status and /monitoring are unauthenticated.")

# --- Eval feedback logging ---------------------------------------------------
# Appends low-confidence/no-data queries to a jsonl for later review. Disable
# with IAM_EVAL_FEEDBACK_ENABLED=0; capped so it cannot fill the disk.
EVAL_FEEDBACK_ENABLED = os.getenv("IAM_EVAL_FEEDBACK_ENABLED", "1").strip().lower() not in ("0", "false", "no", "")
EVAL_FEEDBACK_MAX_BYTES = int(os.getenv("IAM_EVAL_FEEDBACK_MAX_BYTES", str(10 * 1024 * 1024)))

# --- Rate limiting (simple in-memory fixed window per client IP) -------------
RATE_LIMIT_PER_MINUTE = int(os.getenv("IAM_RATE_LIMIT_PER_MINUTE", "30"))
_rate_buckets: "OrderedDict[str, list]" = OrderedDict()  # ip -> [window_start, count]
_rate_lock = threading.Lock()

# --- CORS -------------------------------------------------------------------
_origins_env = os.getenv("IAM_ALLOWED_ORIGINS", "").strip()
ALLOWED_ORIGINS = [o.strip() for o in _origins_env.split(",") if o.strip()] or [
    "https://iamparis.eu",
    "https://www.iamparis.eu",
    "http://localhost:3000",
    "http://localhost:8000",
]
# Credentials may not be combined with a wildcard origin.
ALLOW_CREDENTIALS = "*" not in ALLOWED_ORIGINS


def require_api_key(x_api_key: str = Header(default="", alias="X-API-Key")) -> None:
    """Reject requests when an API key is configured and not supplied/matched."""
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


# Behind a proxy/CDN every request shares the proxy's IP, so per-IP limiting
# would throttle all users together. Trust X-Forwarded-For by default (set
# IAM_TRUST_PROXY=0 when the API is exposed directly, since the header is
# client-controlled in that case).
TRUST_PROXY = os.getenv("IAM_TRUST_PROXY", "1").strip().lower() not in ("0", "false", "no")


def _client_ip(request: Request) -> str:
    if TRUST_PROXY:
        forwarded = str(request.headers.get("x-forwarded-for") or "").strip()
        if forwarded:
            # First hop is the original client.
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def enforce_rate_limit(request: Request) -> None:
    """Fixed-window per-IP rate limiter; raises HTTP 429 when exceeded."""
    if RATE_LIMIT_PER_MINUTE <= 0:
        return
    client_ip = _client_ip(request)
    now = time.time()
    with _rate_lock:
        window_start, count = _rate_buckets.get(client_ip, [now, 0])
        if now - window_start >= 60:
            window_start, count = now, 0
        count += 1
        _rate_buckets[client_ip] = [window_start, count]
        # Bound the bucket map so it cannot grow without limit.
        while len(_rate_buckets) > 10000:
            _rate_buckets.popitem(last=False)
        over_limit = count > RATE_LIMIT_PER_MINUTE
    if over_limit:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")
_monitoring_counters = {
    "total_queries": 0,
    "failed_queries": 0,
    "no_data_queries": 0,
    "low_confidence_route_queries": 0,
    "low_confidence_entity_queries": 0,
}
# Persist counters so a restart does not wipe the operational history.
# Set IAM_MONITORING_STATE="" to disable (e.g. read-only filesystems).
_MONITORING_STATE_PATH = os.getenv("IAM_MONITORING_STATE", "cache/monitoring_counters.json")
_monitoring_lock = threading.Lock()


def _load_monitoring_counters() -> None:
    if not _MONITORING_STATE_PATH:
        return
    try:
        data = json.loads(Path(_MONITORING_STATE_PATH).read_text())
    except (OSError, ValueError):
        return
    for key in _monitoring_counters:
        try:
            _monitoring_counters[key] = int(data.get(key, 0))
        except (TypeError, ValueError):
            continue


def _save_monitoring_counters() -> None:
    if not _MONITORING_STATE_PATH:
        return
    try:
        path = Path(_MONITORING_STATE_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_monitoring_counters))
    except OSError:
        pass


_load_monitoring_counters()
MONITORING_THRESHOLDS = {
    "failed_route_rate": float(os.getenv("IAM_MONITOR_FAILED_RATE_THRESHOLD", "0.05")),
    "no_data_rate": float(os.getenv("IAM_MONITOR_NO_DATA_RATE_THRESHOLD", "0.35")),
    "low_confidence_route_rate": float(os.getenv("IAM_MONITOR_LOW_ROUTE_RATE_THRESHOLD", "0.10")),
    "low_confidence_entity_rate": float(os.getenv("IAM_MONITOR_LOW_ENTITY_RATE_THRESHOLD", "0.15")),
}

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
        ts_payload = {
            'workspace_code': all_workspaces,
            'limit': -1,
        }
        ts = bot.fetch_json(bot.env['REST_API_FULL'], payload=ts_payload, cache=True)
        ts, ts_source = load_best_cached_results(ts)
        logger.info(f"Loaded {len(ts)} timeseries records ({ts_source})")
        
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
        embeddings = OpenAIEmbeddings(model='text-embedding-3-small', api_key=bot.env['OPENAI_API_KEY'], timeout=30, max_retries=1)
        faiss_index = build_faiss_index(chunks, embeddings)
        logger.info("FAISS index built successfully")
        
        # Cache shared runtime resources
        _cached_resources = build_runtime_context(
            models=models,
            ts=ts,
            vector_store=faiss_index,
            env=bot.env,
            bot=bot,
        )
        
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
    query: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(default="", max_length=128)
    reset_session: bool = False


class RelevantLink(BaseModel):
    title: str
    url: str
    reason: str = ""
    confidence: float = 0.0
    search_hint: str = ""
    display_label: str = ""
    display_hint: str = ""
    action: str = "open"
    category: str = ""
    verified_direct_url: bool = False
    fallback_instruction: str = ""


class QueryResponse(BaseModel):
    answer: str
    session_id: str = ""
    history: List[Tuple[str, str]] = Field(default_factory=list)
    plot_base64: str = ""
    plot_caption: str = ""
    notices: List[str] = Field(default_factory=list)
    relevant_links: List[RelevantLink] = Field(default_factory=list)
    suggested_next_questions: List[str] = Field(default_factory=list)
    entities: Dict[str, Any] = Field(default_factory=dict)
    data_scope: Dict[str, Any] = Field(default_factory=dict)
    data_provenance: Dict[str, Any] = Field(default_factory=dict)
    route: Dict[str, Any] = Field(default_factory=dict)


def _build_query_trace(
    session_id: str,
    query: str,
    manager: Any,
    answer: str,
) -> Dict[str, Any]:
    entities = dict(getattr(manager, "last_entities", {}) or {})
    route = dict(getattr(manager, "last_route_decision", {}) or {})
    links = list(getattr(manager, "last_links", []) or [])
    text = str(answer or "")
    no_data = "I could not find data" in text or "No data found" in text
    resources = getattr(manager, "shared_resources", {}) or {}
    matched_records = _count_matching_records(resources, entities)
    no_data_reason = ""
    if no_data:
        # Prefer the structured diagnosis over guessing from answer text.
        no_data_reason = _derive_no_data_reason(resources, entities) or _classify_no_data_reason(text)

    return {
        "session_id": session_id,
        "query": query,
        "route": route.get("agent", ""),
        "route_confidence": route.get("confidence", 0.0),
        "route_source": route.get("source", ""),
        "entities": entities,
        "entity_confidence": entities.get("entity_confidence", {}),
        "selected_variable": entities.get("variable", ""),
        "selected_region": entities.get("region", ""),
        "selected_scenario": entities.get("scenario", ""),
        "selected_model": entities.get("model", ""),
        "matched_records": matched_records,
        "no_data_reason": no_data_reason,
        "selected_links": [link.get("title", "") for link in links],
        "link_scores": {
            link.get("title", ""): link.get("confidence", 0.0)
            for link in links
            if link.get("title")
        },
    }


def _latest_cache_timestamp() -> str:
    cache_files = []
    for pattern in ("cache/results*.json", "cache/models*.json", "cache/data_metadata.pkl"):
        cache_files.extend(Path(".").glob(pattern))
    existing = [path for path in cache_files if path.exists()]
    if not existing:
        return ""
    latest_mtime = max(path.stat().st_mtime for path in existing)
    return datetime.fromtimestamp(latest_mtime, tz=timezone.utc).isoformat()


def _answer_scope_from_text(answer: str) -> Dict[str, Any]:
    text = str(answer or "")
    scope: Dict[str, Any] = {}
    header = re.search(r"^###\s+(?P<variable>.+?)(?:\s+in\s+(?P<region>[^\n]+))?$", text, flags=re.MULTILINE)
    if header:
        scope["variable"] = header.group("variable").strip()
        if header.group("region"):
            scope["region"] = header.group("region").strip()

    scope_line = re.search(
        r"Scope:\s*scenario\s+`(?P<scenario>[^`]+)`,\s*model\s+`(?P<model>[^`]+)`,\s*years\s+`(?P<years>[^`]+)`",
        text,
        flags=re.IGNORECASE,
    )
    if scope_line:
        scope["scenario"] = scope_line.group("scenario").strip()
        scope["model"] = scope_line.group("model").strip()
        scope["years"] = scope_line.group("years").strip()

    unit_line = re.search(r"Unit:\s*`(?P<unit>[^`]+)`", text, flags=re.IGNORECASE)
    if unit_line:
        scope["unit"] = unit_line.group("unit").strip()

    return scope


def _build_data_provenance(resources: Dict[str, Any], entities: Dict[str, Any], answer: str, route: Dict[str, Any]) -> Dict[str, Any]:
    agent = str((route or {}).get("agent") or "")
    if agent not in {"data_query", "data_plotting"}:
        return {}

    text_scope = _answer_scope_from_text(answer)
    selected_filters = {
        "variable": entities.get("variable") or text_scope.get("variable", ""),
        "region": entities.get("region") or text_scope.get("region", ""),
        "scenario": entities.get("scenario") or text_scope.get("scenario", ""),
        "model": entities.get("model") or text_scope.get("model", ""),
        "years": text_scope.get("years", ""),
        "unit": text_scope.get("unit", ""),
    }
    selected_filters = {
        key: value
        for key, value in selected_filters.items()
        if str(value or "").strip() and str(value or "").strip().lower() != "multiple"
    }

    matched_record_count = _count_matching_records(resources, selected_filters)
    no_data = "I could not find data" in str(answer or "") or "No data found" in str(answer or "")
    if not selected_filters and matched_record_count is None and not no_data:
        return {}

    provenance = {
        "cache_timestamp": _latest_cache_timestamp(),
        "matched_record_count": matched_record_count,
        "selected_filters": selected_filters,
        "route": {
            "agent": agent,
            "confidence": (route or {}).get("confidence", 0.0),
            "source": (route or {}).get("source", ""),
        },
    }
    if no_data:
        provenance["no_data_reason"] = (
            _derive_no_data_reason(resources, selected_filters) or _classify_no_data_reason(answer)
        )
    provenance.update(_provenance_display_fields(provenance))
    return provenance


def _provenance_display_fields(provenance: Dict[str, Any]) -> Dict[str, Any]:
    selected_filters = provenance.get("selected_filters") or {}
    labels = {
        "variable": "Variable",
        "region": "Region",
        "scenario": "Scenario",
        "model": "Model",
        "years": "Years",
        "unit": "Unit",
    }
    rows: list[dict[str, str]] = []
    for key in ("variable", "region", "scenario", "model", "years", "unit"):
        value = selected_filters.get(key)
        if value:
            rows.append({"label": labels[key], "value": str(value)})
    if provenance.get("matched_record_count") is not None:
        rows.append({"label": "Matched records", "value": str(provenance.get("matched_record_count"))})
    if provenance.get("cache_timestamp"):
        rows.append({"label": "Cache timestamp", "value": str(provenance.get("cache_timestamp"))})
    if provenance.get("no_data_reason"):
        rows.append({"label": "No-data reason", "value": str(provenance.get("no_data_reason"))})
    return {
        "display_title": "Data provenance",
        "display_rows": rows,
    }


def _should_log_eval_candidate(trace: Dict[str, Any]) -> bool:
    route_confidence = float(trace.get("route_confidence") or 0.0)
    entity_confidence = trace.get("entity_confidence") or {}
    low_entity = any(float(value or 0.0) < 0.5 for value in entity_confidence.values())
    return bool(trace.get("no_data_reason") or route_confidence < 0.55 or low_entity)


def _write_eval_feedback_candidate(trace: Dict[str, Any], answer: str, log_path: str | Path | None = None) -> bool:
    if not EVAL_FEEDBACK_ENABLED:
        return False
    if not _should_log_eval_candidate(trace):
        return False
    path = Path(log_path or os.getenv("IAM_EVAL_FEEDBACK_LOG", "docs/eval_feedback_candidates.jsonl"))
    # Size cap: stop appending once the file is large, so a flood of queries
    # cannot exhaust disk space.
    try:
        if path.exists() and path.stat().st_size >= EVAL_FEEDBACK_MAX_BYTES:
            logger.warning("eval feedback log at size cap (%s bytes); skipping write", EVAL_FEEDBACK_MAX_BYTES)
            return False
    except OSError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": trace.get("query", ""),
        "session_id": trace.get("session_id", ""),
        "route": trace.get("route", ""),
        "route_confidence": trace.get("route_confidence", 0.0),
        "entities": trace.get("entities", {}),
        "entity_confidence": trace.get("entity_confidence", {}),
        "matched_records": trace.get("matched_records"),
        "no_data_reason": trace.get("no_data_reason", ""),
        "answer_preview": str(answer or "").replace("\n", " ")[:300],
        "eval_hint": "Add this query to eval_queries.csv or eval_holdout_queries.csv after review.",
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
    return True


def _update_monitoring(trace: Dict[str, Any] | None = None, *, failed: bool = False) -> None:
    with _monitoring_lock:
        _monitoring_counters["total_queries"] += 1
        if failed:
            _monitoring_counters["failed_queries"] += 1
            _save_monitoring_counters()
            return
        trace = trace or {}
        if trace.get("no_data_reason"):
            _monitoring_counters["no_data_queries"] += 1
        if float(trace.get("route_confidence") or 0.0) < 0.55:
            _monitoring_counters["low_confidence_route_queries"] += 1
        entity_confidence = trace.get("entity_confidence") or {}
        if any(float(value or 0.0) < 0.5 for value in entity_confidence.values()):
            _monitoring_counters["low_confidence_entity_queries"] += 1
        _save_monitoring_counters()


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def _feedback_log_summary(path: str | Path | None = None) -> Dict[str, Any]:
    log_path = Path(path or os.getenv("IAM_EVAL_FEEDBACK_LOG", "docs/eval_feedback_candidates.jsonl"))
    if not log_path.exists():
        return {"path": str(log_path), "count": 0, "recent": []}
    rows = []
    with log_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return {"path": str(log_path), "count": len(rows), "recent": rows[-5:]}


def _monitoring_snapshot() -> Dict[str, Any]:
    total = int(_monitoring_counters.get("total_queries", 0))
    failed = int(_monitoring_counters.get("failed_queries", 0))
    no_data = int(_monitoring_counters.get("no_data_queries", 0))
    low_route = int(_monitoring_counters.get("low_confidence_route_queries", 0))
    low_entity = int(_monitoring_counters.get("low_confidence_entity_queries", 0))
    rates = {
        "failed_route_rate": _rate(failed, total),
        "no_data_rate": _rate(no_data, total),
        "low_confidence_route_rate": _rate(low_route, total),
        "low_confidence_entity_rate": _rate(low_entity, total),
    }
    alerts = [
        {
            "metric": metric,
            "value": value,
            "threshold": MONITORING_THRESHOLDS[metric],
            "severity": "warning",
        }
        for metric, value in rates.items()
        if value > MONITORING_THRESHOLDS.get(metric, 1.0)
    ]
    return {
        "counters": dict(_monitoring_counters),
        "rates": rates,
        "thresholds": dict(MONITORING_THRESHOLDS),
        "alerts": alerts,
        "status": "warning" if alerts else "ok",
        "feedback_candidates": _feedback_log_summary(),
    }


def _derive_no_data_reason(resources: Dict[str, Any], entities: Dict[str, Any]) -> str:
    """Structured no-data diagnosis: apply the requested filters one dimension
    at a time and report the first one that eliminates every record. Falls back
    to "" when the data cannot explain the miss (caller then uses the
    text-based classifier)."""
    labels = (
        ("variable", "variable unavailable in current scope"),
        ("region", "region combination unavailable"),
        ("scenario", "scenario combination unavailable"),
        ("model", "model combination unavailable"),
    )
    scope: Dict[str, Any] = {}
    for key, label in labels:
        value = str((entities or {}).get(key) or "").strip()
        if not value:
            continue
        scope[key] = value
        if _count_matching_records(resources, scope) == 0:
            return label
    return ""


def _classify_no_data_reason(answer: str) -> str:
    text = str(answer or "").lower()
    if "scenario combination" in text:
        return "scenario combination unavailable"
    if "region combination" in text:
        return "region combination unavailable"
    if "requested variable is unavailable" in text:
        return "variable unavailable in current scope"
    if "model `" in text or "using model" in text:
        return "model combination unavailable"
    if "under `" in text:
        return "scenario combination unavailable"
    if " in `" in text or "in region" in text:
        return "region combination unavailable"
    return "no matching data slice"


def _prepare_relevant_links(links: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    prepared: List[Dict[str, Any]] = []
    for link in links or []:
        if not isinstance(link, dict):
            continue
        item = dict(link)
        title = str(item.get("title") or "IAM PARIS link")
        search_hint = str(item.get("search_hint") or "").strip()
        url = str(item.get("url") or "")
        if not item.get("display_label"):
            if search_hint and "application_library" in url:
                item["display_label"] = f"Search Application Library for {search_hint}"
            else:
                item["display_label"] = f"Open {title}"
        if not item.get("display_hint"):
            item["display_hint"] = str(item.get("fallback_instruction") or "")
            if not item["display_hint"]:
                item["display_hint"] = (
                    f"Use the site search hint: {search_hint}"
                    if search_hint
                    else str(item.get("reason") or "")
                )
        if not item.get("action"):
            item["action"] = "search" if search_hint and "application_library" in url else "open"
        item["category"] = str(item.get("category") or "")
        item["verified_direct_url"] = bool(item.get("verified_direct_url"))
        item["fallback_instruction"] = str(item.get("fallback_instruction") or "")
        prepared.append(item)
    return prepared


def _suggested_next_questions(query: str, answer: str, manager: Any) -> List[str]:
    route = dict(getattr(manager, "last_route_decision", {}) or {})
    entities = dict(getattr(manager, "last_entities", {}) or {})
    agent = str(route.get("agent") or "")
    text = str(answer or "")
    suggestions: List[str] = []

    def add(item: str) -> None:
        if item and item not in suggestions:
            suggestions.append(item)

    available_scenarios = list(getattr(getattr(manager, "entity_extractor", None), "available_scenarios", []) or [])
    baseline_scenario = next(
        (scen for scen in available_scenarios if "baseline" in str(scen).lower()),
        "",
    )
    current_scenario = str(entities.get("scenario") or "").lower()

    if agent in {"data_query", "data_plotting"}:
        if "I could not find data" in text or "No data found" in text:
            add("Show available scenarios")
            add("Show available regions")
            add("Show available variables")
        else:
            if entities.get("variable") or entities.get("region"):
                add("Plot it")
                if baseline_scenario and "baseline" not in current_scenario:
                    add(f"Compare with {baseline_scenario}")
                add("By 2050")
            else:
                add("Show available variables")
                add("Show available regions")
                add("Help me find data")
    elif agent == "model_explanation":
        model = str(entities.get("model") or "").strip()
        add(f"Show data using {model}" if model else "Show data using this model")
        add("Compare this model with another model")
        add("Show related IAM PARIS model links")
    elif agent == "general_qa":
        add("Show relevant IAM PARIS links")
        add("Help me find data")
        add("Open the related Application Library page")

    # Only offer option selection when a numbered choice is actually pending;
    # otherwise the phrase has no handler and would confuse the router.
    pending_clarification = dict(getattr(manager, "clarification_context", None) or {})
    if pending_clarification.get("suggested_options"):
        add("Use the first option")

    return suggestions[:4]


def _count_matching_records(
    resources: Dict[str, Any],
    entities: Dict[str, Any],
) -> Optional[int]:
    scope = {
        "variable": str(entities.get("variable", "") or "").strip(),
        "region": str(entities.get("region", "") or "").strip(),
        "scenario": str(entities.get("scenario", "") or "").strip(),
        "model": str(entities.get("model", "") or "").strip(),
    }
    active_scope = {key: value for key, value in scope.items() if value}
    if not active_scope:
        return None

    records = resources.get("ts") or []
    if not records:
        return 0

    def _matches(record: Dict[str, Any]) -> bool:
        if scope["variable"] and str(record.get("variable", "")).strip() != scope["variable"]:
            return False
        if scope["region"] and str(record.get("region", "")).strip() != scope["region"]:
            return False
        if scope["scenario"] and str(record.get("scenario", "")).strip() != scope["scenario"]:
            return False
        if scope["model"]:
            model = str(record.get("modelName") or record.get("model") or "").strip()
            if model != scope["model"]:
                return False
        return True

    return sum(1 for record in records if isinstance(record, dict) and _matches(record))


def _extract_notices(answer: str) -> List[str]:
    """
    Extract short UI-friendly notices from the bot response.
    Frontends can display these as a toast/modal without parsing the whole answer.
    """
    text = str(answer or "")
    notices: List[str] = []

    assumptions_msg = "No explicit assumptions field is available in the model metadata."
    if assumptions_msg in text:
        notices.append(assumptions_msg)

    return notices


def _split_answer_payload(answer: str) -> tuple[str, str, str, List[str]]:
    """
    Split mixed text/plot markdown answers into API-friendly fields.
    Returns: cleaned_answer, plot_base64, plot_caption, notices
    """
    text = str(answer or "").strip()
    notices = _extract_notices(text)
    for notice in notices:
        text = re.sub(re.escape(notice), "", text, flags=re.IGNORECASE).strip()

    # Links are returned separately in `relevant_links`; strip the inline list
    # so frontends do not render the same links twice.
    text = re.sub(
        r"\n*Relevant IAM PARIS links:\n(?:- .*(?:\n|$))*",
        "\n",
        text,
    ).strip()

    plot_base64 = ""
    plot_caption = ""
    match = re.search(r"!\[Plot\]\((data:image/png;base64,[^)]+)\)", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        plot_base64 = match.group(1).split("data:image/png;base64,", 1)[-1]
        text = (text[:match.start()] + text[match.end():]).strip()
        # Prefer the plotter's explicit scope line ("Showing ...") as the
        # caption over whatever text happens to come first.
        caption_line = next(
            (line.strip() for line in text.splitlines() if line.strip().startswith("Showing ")),
            "",
        )
        first_line = text.splitlines()[0].strip() if text else ""
        plot_caption = caption_line or first_line or "Generated plot."

    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text, plot_base64, plot_caption, notices


def _cleanup_sessions_locked(now: float) -> None:
    """Drop expired sessions, then evict oldest while above MAX_SESSIONS.

    Caller must hold ``_sessions_lock``.
    """
    expired = [
        session_id
        for session_id, state in _sessions.items()
        if now - state.get("last_access", now) > SESSION_TTL_SECONDS
    ]
    for session_id in expired:
        _sessions.pop(session_id, None)
    # LRU eviction: OrderedDict preserves insertion/refresh order.
    while len(_sessions) > MAX_SESSIONS:
        _sessions.popitem(last=False)


def _get_or_create_session(session_id: str = "", reset_session: bool = False):
    now = time.time()
    with _sessions_lock:
        _cleanup_sessions_locked(now)
        if reset_session and session_id:
            _sessions.pop(session_id, None)

        if not session_id:
            session_id = uuid.uuid4().hex

        state = _sessions.get(session_id)
        if state is not None:
            state["last_access"] = now
            _sessions.move_to_end(session_id)
            return session_id, state

    # Build the (potentially expensive) manager outside the lock so concurrent
    # new-session requests are not serialized on it.
    new_state = {
        "manager": MultiAgentManager(_cached_resources, streaming=False),
        "chat_history": [],
        "last_access": time.time(),
    }
    with _sessions_lock:
        # Another request may have created this session meanwhile; reuse it.
        existing = _sessions.get(session_id)
        if existing is not None:
            existing["last_access"] = time.time()
            _sessions.move_to_end(session_id)
            return session_id, existing
        _sessions[session_id] = new_state
        _cleanup_sessions_locked(time.time())
    return session_id, new_state

# FastAPI Setup
from contextlib import asynccontextmanager


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Initialize resources when the server starts."""
    initialize_resources()
    yield


app = FastAPI(
    title='IAM Paris Data Chatbot API',
    description='Multi-agent conversational AI for IAM Paris climate data',
    version='1.0.0',
    lifespan=_lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=ALLOW_CREDENTIALS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)


@app.post('/query', response_model=QueryResponse)
def query_chatbot(
    req: QueryRequest,
    request: Request,
    _auth: None = Depends(require_api_key),
):
    """
    Process a user query through the multi-agent system.
    Uses cached resources for fast response.
    """
    enforce_rate_limit(request)
    # Check if resources are ready
    if _initialization_status == "initializing":
        raise HTTPException(
            status_code=503, 
            detail="Service is initializing. Please try again in a moment."
        )
    
    if _initialization_status == "error":
        logger.error("Serving 503: initialization error: %s", _initialization_error)
        raise HTTPException(
            status_code=503,
            detail="Service temporarily unavailable."
        )
    
    if _cached_resources is None:
        raise HTTPException(
            status_code=503,
            detail="Service not ready. Resources not loaded."
        )
    
    try:
        session_id, session_state = _get_or_create_session(
            req.session_id,
            reset_session=req.reset_session,
        )
        manager = session_state["manager"]
        chat_history: List[Tuple[str, str]] = session_state["chat_history"]
        
        # Route query
        response = manager.route_query(req.query, chat_history)
        answer_text, plot_base64, plot_caption, notices = _split_answer_payload(response)
        chat_history.append((req.query, answer_text))
        session_state["last_access"] = time.time()
        trace = _build_query_trace(session_id, req.query, manager, answer_text)
        logger.info("query_trace %s", json.dumps(trace, sort_keys=True, default=str))
        _update_monitoring(trace)
        _write_eval_feedback_candidate(trace, answer_text)
        relevant_links = _prepare_relevant_links(getattr(manager, "last_links", []))
        next_questions = _suggested_next_questions(req.query, answer_text, manager)
        entities = getattr(manager, "last_entities", {})
        route = getattr(manager, "last_route_decision", {})
        data_provenance = _build_data_provenance(
            getattr(manager, "shared_resources", {}) or {},
            entities,
            answer_text,
            route,
        )

        return QueryResponse(
            answer=answer_text,
            session_id=session_id,
            # Cap the returned history so long conversations do not grow the
            # payload unbounded; the full history stays in the session state.
            history=chat_history[-HISTORY_MAX_TURNS:],
            plot_base64=plot_base64,
            plot_caption=plot_caption,
            notices=notices,
            relevant_links=relevant_links,
            suggested_next_questions=next_questions,
            entities=entities,
            data_scope=entities,
            data_provenance=data_provenance,
            route=route,
        )
    
    except HTTPException:
        raise
    except Exception:
        _update_monitoring(failed=True)
        logger.exception("Error processing query")
        raise HTTPException(status_code=500, detail="Internal server error.")


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
    
    # Public endpoint: expose only a boolean, never the raw error string
    # (which can contain internal hosts/stack details). Full text is in /status.
    return {
        "status": _initialization_status,
        "resources_loaded": _cached_resources is not None,
        "has_error": _initialization_error is not None,
        "elapsed_seconds": round(elapsed, 1) if elapsed else None,
        "timeout_limit": INITIALIZATION_TIMEOUT
    }


@app.get('/status')
def status_check(_auth: None = Depends(require_api_key)):
    """Detailed status endpoint for monitoring cache readiness."""
    elapsed = None
    if _initialization_start_time:
        elapsed = time.time() - _initialization_start_time
    
    metadata = _cached_resources.get('metadata') if _cached_resources else None
    metadata_summary = metadata.get_summary() if metadata else {}

    # Count cached items
    cache_status = {
        "models_count": len(_cached_resources.get('models', [])) if _cached_resources else 0,
        "timeseries_count": len(_cached_resources.get('ts', [])) if _cached_resources else 0,
        "vector_store_ready": _cached_resources.get('vector_store') is not None if _cached_resources else False,
        "link_catalog_count": len(_cached_resources.get('link_catalog', [])) if _cached_resources else 0,
        "metadata": metadata_summary,
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


@app.get('/monitoring')
def monitoring_check(_auth: None = Depends(require_api_key)):
    """Operational counters for route failures, no-data and low-confidence behavior."""
    return _monitoring_snapshot()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
