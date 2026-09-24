import os
import sys
import time
import re
import uuid
import json
import threading
import hmac
import copy
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
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
from main import (
    IAMParisBot,
    build_faiss_index,
    cache_file_timestamp,
    docs_from_records,
    load_best_cached_results,
    load_definitions as load_cached_definitions,
)
from llm_factory import get_embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from manager import MultiAgentManager
from runtime_context import build_runtime_context
from canonical_aliases import regions_equivalent, scenario_in_family
from model_aliases import (
    UNLABELLED_MODEL_LABEL,
    display_model_label,
    is_presentable_model_label,
    is_unlabelled_model_display,
    resolve_model_family_members,
)
from year_filters import YearFilter, is_finite_numeric_value, is_latest_year_filter
from resolved_scope import consume_resolved_scope, has_numeric_result_table

# Configuration
INITIALIZATION_TIMEOUT = 300  # 5 minutes timeout for cache building
API_REQUEST_TIMEOUT = max(float(os.getenv("IAM_API_REQUEST_TIMEOUT", "120")), 0.1)
QUERY_EXECUTOR_WORKERS = max(int(os.getenv("IAM_QUERY_EXECUTOR_WORKERS", "8")), 1)
_query_executor = ThreadPoolExecutor(
    max_workers=QUERY_EXECUTOR_WORKERS,
    thread_name_prefix="iam-query",
)
_query_slots = threading.BoundedSemaphore(QUERY_EXECUTOR_WORKERS)

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
    logger.warning(
        "IAM_API_KEY is not set: /query and /status are unauthenticated; "
        "/monitoring feedback details are redacted."
    )

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
    if API_KEY and not hmac.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


# Trust forwarded client IPs only when the deployment explicitly opts in and
# its reverse proxy strips user-supplied forwarding headers. Direct clients
# can otherwise rotate X-Forwarded-For to evade the rate limit.
TRUST_PROXY = os.getenv("IAM_TRUST_PROXY", "0").strip().lower() not in ("0", "false", "no", "")


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

_DEFAULT_WORKSPACES = [
    "afolu", "buildings-transf", "covid-rec", "decarb-potentials", "decipher",
    "energy-systems", "eu-headed", "index-decomp", "industrial-transf", "ndcs-impacts",
    "net-zero", "post-glasgow", "power-people", "study-1", "study-2", "study-3",
    "study-4", "study-6", "study-7", "transp-transf", "world-headed",
]


def _load_workspaces() -> List[str]:
    """Workspace codes come from config/workspaces.json so adding a workspace
    is a config edit, not a code change. Falls back to the built-in list."""
    try:
        data = json.loads(Path("config/workspaces.json").read_text())
        workspaces = [str(w).strip() for w in data.get("workspaces", []) if str(w).strip()]
        if workspaces:
            return workspaces
    except (OSError, ValueError):
        pass
    return list(_DEFAULT_WORKSPACES)


def load_definitions():
    """Use the source-validated YAML definition cache shared with the CLI."""
    return load_cached_definitions()


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
        all_workspaces = _load_workspaces()
        logger.info("Fetching %d workspaces", len(all_workspaces))
        ts_payload = {
            'workspace_code': all_workspaces,
            'limit': -1,
        }
        ts = bot.fetch_json(bot.env['REST_API_FULL'], payload=ts_payload, cache=True)
        ts, ts_source = load_best_cached_results(ts)
        results_source_path = str(getattr(bot, "last_fetch_cache_file", "") or "")
        results_timestamp = cache_file_timestamp(results_source_path)
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
        embeddings = get_embeddings(model='text-embedding-3-small', api_key=bot.env['OPENAI_API_KEY'], timeout=30, max_retries=1)
        faiss_index = build_faiss_index(chunks, embeddings)
        logger.info("FAISS index built successfully")
        
        # Cache shared runtime resources
        _cached_resources = build_runtime_context(
            models=models,
            ts=ts,
            vector_store=faiss_index,
            env=bot.env,
            bot=bot,
            results_source_path=results_source_path,
            results_timestamp=results_timestamp,
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
    clarification: Dict[str, Any] = Field(default_factory=dict)
    data_provenance: Dict[str, Any] = Field(default_factory=dict)
    route: Dict[str, Any] = Field(default_factory=dict)


def _sanitize_model_series_label(value: object) -> str:
    """Hide a numeric model ID when it is the leading part of a series label."""
    text = str(value or "").strip()
    return re.sub(
        r"^[+-]?\d+(?:[.,]\d+)*(?=\s+(?:—|-)\s+)",
        UNLABELLED_MODEL_LABEL,
        text,
    )


def _sanitize_response_model_scope(value: Any, key: str = "") -> Any:
    """Sanitize model-labelled response fields without changing source records."""
    normalized_key = str(key or "").casefold()
    # Confidence maps hold numeric scores keyed by dimension name (including
    # "model"), so their values are never model labels.
    if normalized_key.endswith("confidence"):
        return value
    if isinstance(value, dict):
        return {
            child_key: _sanitize_response_model_scope(child_value, child_key)
            for child_key, child_value in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_response_model_scope(item, key) for item in value]
    if normalized_key == "displayed_series":
        return _sanitize_model_series_label(value)
    if "model" in normalized_key and not normalized_key.endswith(("count", "confidence")):
        return display_model_label(value)
    return value


def _manager_response_entities(manager: Any) -> Dict[str, Any]:
    """Return turn-visible scope while preserving legacy manager support."""
    resolver = getattr(manager, "response_entities", None)
    if callable(resolver):
        try:
            return dict(_sanitize_response_model_scope(dict(resolver() or {})) or {})
        except Exception:
            logger.debug("Could not read structured conversation scope", exc_info=True)
    active = dict(getattr(manager, "last_entities", {}) or {})
    if active:
        return dict(_sanitize_response_model_scope(active) or {})
    pending = getattr(manager, "clarification_context", None)
    if pending:
        return dict(_sanitize_response_model_scope(dict(pending.get("entities", {}) or {})) or {})
    return {}


def _manager_clarification_payload(manager: Any) -> Dict[str, Any]:
    resolver = getattr(manager, "pending_clarification_payload", None)
    if callable(resolver):
        try:
            payload = dict(resolver() or {})
            if isinstance(payload.get("options"), list):
                payload["options"] = [
                    option for option in payload["options"]
                    if not (
                        isinstance(option, dict)
                        and str(option.get("kind") or "").casefold() == "model"
                        and not is_presentable_model_label(option.get("value"))
                    )
                ]
            return dict(_sanitize_response_model_scope(payload) or {})
        except Exception:
            logger.debug("Could not read structured clarification", exc_info=True)
    pending = getattr(manager, "clarification_context", None)
    if not pending:
        return {}
    values = list(pending.get("suggested_options") or [])
    kinds = list(pending.get("suggested_option_kinds") or [])
    fallback = str(pending.get("suggested_kind") or "variable")
    options = []
    for index, value in enumerate(values):
        kind = str(kinds[index] if index < len(kinds) else fallback)
        if kind.casefold() == "model" and not is_presentable_model_label(value):
            continue
        options.append({"kind": kind, "value": str(value)})
    return dict(_sanitize_response_model_scope({
        "missing_dimension": fallback,
        "base_scope": dict(pending.get("entities") or {}),
        "options": options,
    }) or {})


_NO_DATA_ANSWER_MARKERS = (
    "i could not find data",
    "i couldn't find data",
    "no data found",
    "no data matched",
    "no time series data",
    "no timeseries data",
    "i can't combine",
    "i cannot combine",
    "incompatible units",
    "i can't plot",
    "i cannot plot",
    "could not identify enough",
    "could not identify variable",
    "could not identify a variable",
    "not found in loaded data",
)


def _runtime_model_scope_override(
    resources: Dict[str, Any],
    entities: Dict[str, Any],
) -> List[str]:
    """Resolve answer model aliases to concrete time-series labels.

    ``result_models`` names the rows a renderer actually used. Model catalogue
    answers expose the resolved family members in the same field. These labels
    are more precise for counting provenance than a display alias such as
    ``PROMETHEUS``, which may not exist verbatim on any row.
    """
    runtime_models = sorted({
        str(record.get("modelName") or record.get("model") or "").strip()
        for record in (resources.get("ts") or [])
        if isinstance(record, dict)
        and str(record.get("modelName") or record.get("model") or "").strip()
    }, key=lambda value: (value.casefold(), value))
    if not runtime_models:
        return []

    raw_values = entities.get("result_models")
    force_runtime_scope = raw_values not in (None, "", [])
    if not force_runtime_scope:
        raw_values = entities.get("models")
        force_runtime_scope = raw_values not in (None, "", [])
    if not force_runtime_scope:
        raw_values = entities.get("model")
        if raw_values in (None, ""):
            return []
        requested = str(raw_values).strip()
        if any(requested.casefold() == name.casefold() for name in runtime_models):
            # Keep the established singular provenance shape when the selected
            # name is already one concrete runtime label.
            return []

    values = (
        list(raw_values)
        if isinstance(raw_values, (list, tuple, set))
        else [raw_values]
    )
    exact_by_key = {name.casefold(): name for name in runtime_models}
    resolved: set[str] = set()
    for value in values:
        requested = str(value or "").strip()
        if not requested:
            continue
        if is_unlabelled_model_display(requested):
            resolved.add(UNLABELLED_MODEL_LABEL)
            continue
        exact = exact_by_key.get(requested.casefold())
        if exact:
            resolved.add(exact)
            continue
        resolved.update(resolve_model_family_members(requested, runtime_models))
    return sorted(resolved, key=lambda value: (value.casefold(), value))


def _is_no_data_answer(answer: str, *, has_plot: bool = False) -> bool:
    """Return whether a data/plot answer failed to produce a usable result.

    Plot answers can include a warning about one missing model while still
    rendering the remaining series.  An attached plot is therefore stronger
    evidence than any failure-like wording in the accompanying notice.
    """
    text = str(answer or "")
    if has_numeric_result_table(text):
        return False
    if has_plot or re.search(r"!\[plot\]\(", text, flags=re.IGNORECASE):
        return False
    lowered = text.casefold()
    return any(marker in lowered for marker in _NO_DATA_ANSWER_MARKERS) or bool(
        re.search(
            r"\bcould(?:n't| not)\s+find\s+.+?\s+as\s+a\s+region\b",
            lowered,
        )
    )


def _build_query_trace(
    session_id: str,
    query: str,
    manager: Any,
    answer: str,
    *,
    has_plot: bool = False,
) -> Dict[str, Any]:
    entities = _manager_response_entities(manager)
    route = dict(getattr(manager, "last_route_decision", {}) or {})
    links = list(getattr(manager, "last_links", []) or [])
    text = str(answer or "")
    unmatched_region = str(entities.get("unmatched_region") or "").strip()
    no_data = bool(unmatched_region) or _is_no_data_answer(text, has_plot=has_plot)
    resources = getattr(manager, "shared_resources", {}) or {}
    runtime_models = _runtime_model_scope_override(resources, entities)
    count_scope = dict(entities)
    if unmatched_region:
        count_scope["region"] = unmatched_region
        count_scope.pop("regions", None)
    matched_records = _count_matching_records(resources, count_scope)
    no_data_reason = ""
    if no_data:
        # Prefer the structured diagnosis over guessing from answer text.
        no_data_reason = _resolve_no_data_reason(resources, count_scope, text)

    return {
        "session_id": session_id,
        "query": query,
        "route": route.get("agent", ""),
        "route_confidence": route.get("confidence", 0.0),
        "route_source": route.get("source", ""),
        "entities": entities,
        "entity_confidence": entities.get("entity_confidence", {}),
        "selected_variable": entities.get("variable", ""),
        "selected_variables": list(entities.get("variables") or []),
        "selected_region": unmatched_region or entities.get("region", ""),
        "selected_regions": list(entities.get("regions") or []),
        "selected_scenario": entities.get("scenario", ""),
        "selected_scenarios": list(entities.get("scenarios") or []),
        "selected_model": "" if runtime_models else entities.get("model", ""),
        "selected_models": runtime_models or list(entities.get("models") or []),
        "comparison_dimension": entities.get("comparison_dimension") or entities.get("comparison", ""),
        "chart_type": entities.get("chart_type", ""),
        "displayed_series": list(entities.get("displayed_series") or []),
        "displayed_series_count": entities.get("displayed_series_count", 0),
        "omitted_series": entities.get("omitted_series", 0),
        "matched_records": matched_records,
        "no_data_reason": no_data_reason,
        "selected_links": [link.get("title", "") for link in links],
        "link_scores": {
            link.get("title", ""): link.get("confidence", 0.0)
            for link in links
            if link.get("title")
        },
    }


def _latest_cache_timestamp(resources: Optional[Dict[str, Any]] = None) -> str:
    """Return the timestamp of the results file used by this runtime."""
    recorded = str((resources or {}).get("results_timestamp", "") or "").strip()
    if recorded:
        return recorded

    source_path = str((resources or {}).get("results_source_path", "") or "").strip()
    if source_path:
        return cache_file_timestamp(source_path)

    # Compatibility fallback for lightweight test resources and older runtime
    # contexts. Metadata/model cache writes must not make results look newer.
    existing = [path for path in Path(".").glob("cache/results*.json") if path.exists()]
    if not existing:
        return ""
    latest_mtime = max(path.stat().st_mtime for path in existing)
    return datetime.fromtimestamp(latest_mtime, tz=timezone.utc).isoformat()


def _answer_scope_from_text(answer: str) -> Dict[str, Any]:
    text = str(answer or "")
    scope: Dict[str, Any] = {}
    header = re.search(r"^###\s+(?P<variable>.+?)(?:\s+in\s+(?P<region>[^\n]+))?$", text, flags=re.MULTILINE)
    if header:
        header_variable = header.group("variable").strip()
        # Discovery headings describe catalogue coverage, not a timeseries
        # variable. Treating ``Data available for GREECE`` as a variable
        # fabricates a zero-match provenance slice.
        if not header_variable.casefold().startswith("data available for "):
            scope["variable"] = header_variable
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


def _build_data_provenance(
    resources: Dict[str, Any],
    entities: Dict[str, Any],
    answer: str,
    route: Dict[str, Any],
    *,
    has_plot: bool = False,
) -> Dict[str, Any]:
    entities = dict(_sanitize_response_model_scope(dict(entities or {})) or {})
    agent = str((route or {}).get("agent") or "")
    if agent not in {"data_query", "data_plotting"}:
        return {}

    # Dataset-catalogue questions report aggregate metadata rather than a
    # selected timeseries slice. A ``latest`` sentinel would otherwise be
    # counted as a literal year filter and displayed as a misleading zero.
    if (
        re.search(r"\blatest available projection year\b", str(answer or ""), re.IGNORECASE)
        and not any(
            entities.get(key)
            for key in (
                "variable", "variables", "region", "regions", "scenario",
                "scenarios", "model", "models",
            )
        )
    ):
        return {}

    text_scope = _answer_scope_from_text(answer)
    runtime_models = _runtime_model_scope_override(resources, entities)
    unmatched_region = str(entities.get("unmatched_region") or "").strip()
    year_text = text_scope.get("years", "")
    if not year_text and (
        entities.get("start_year") is not None
        or entities.get("end_year") is not None
    ):
        year_text = YearFilter(
            entities.get("start_year"),
            entities.get("end_year"),
            explicit=True,
        ).render()
    selected_filters = {
        "workspace_code": entities.get("workspace_code", ""),
        "variable": entities.get("variable") or text_scope.get("variable", ""),
        "variables": list(entities.get("variables") or []),
        "region": entities.get("region") or text_scope.get("region", ""),
        "regions": list(entities.get("regions") or []),
        "scenario": entities.get("scenario") or text_scope.get("scenario", ""),
        "scenarios": list(entities.get("scenarios") or []),
        "model": (
            "" if runtime_models
            else entities.get("model") or text_scope.get("model", "")
        ),
        "models": runtime_models or list(entities.get("models") or []),
        "years": year_text,
        # For rendered data tables the Unit line describes the records that
        # actually survived filtering and formatting. Prefer it over any
        # pre-query extractor guess still present in legacy manager state.
        # Plot captions do not expose a Unit line, so plots continue to use the
        # structured unit recorded by the plotter.
        "unit": (
            entities.get("unit", "")
            if str(entities.get("unit") or "").strip().casefold() == "multiple"
            else text_scope.get("unit") or entities.get("unit", "")
        ),
    }
    if unmatched_region:
        # Preserve the requested invalid geography as the attempted filter.
        # Counting only the resolved variable here would falsely report every
        # CO2 record in the database for a query such as "CO2 for Atlantis".
        selected_filters["region"] = unmatched_region
        selected_filters["regions"] = []
    for singular, plural in (
        ("variable", "variables"),
        ("region", "regions"),
        ("scenario", "scenarios"),
        ("model", "models"),
    ):
        if selected_filters.get(plural):
            selected_filters.pop(singular, None)
    selected_filters = {
        key: value
        for key, value in selected_filters.items()
        if str(value or "").strip() and str(value or "").strip().lower() != "multiple"
    }

    count_scope = dict(selected_filters)
    if entities.get("start_year") is not None:
        count_scope["start_year"] = entities.get("start_year")
    if entities.get("end_year") is not None:
        count_scope["end_year"] = entities.get("end_year")
    # A rendered table may omit empty or incompatible model groups. The
    # manager persists the models that were actually displayed, so provenance
    # should count that visible scope rather than the broader pre-format match.
    matched_record_count = _count_matching_records(resources, count_scope)
    no_data = bool(unmatched_region) or _is_no_data_answer(
        answer,
        has_plot=has_plot,
    )
    if not selected_filters and matched_record_count is None and not no_data:
        return {}

    provenance = {
        "cache_timestamp": _latest_cache_timestamp(resources),
        "matched_record_count": matched_record_count,
        "selected_filters": selected_filters,
        "route": {
            "agent": agent,
            "confidence": (route or {}).get("confidence", 0.0),
            "source": (route or {}).get("source", ""),
        },
    }
    if agent == "data_plotting":
        plot_details = {
            "comparison_dimension": entities.get("comparison_dimension") or entities.get("comparison", ""),
            "chart_type": entities.get("chart_type", ""),
            "displayed_series": list(entities.get("displayed_series") or []),
            "displayed_series_count": entities.get("displayed_series_count"),
            "omitted_series": entities.get("omitted_series"),
        }
        provenance.update({
            key: value for key, value in plot_details.items()
            if value not in (None, "", [], {})
        })
    if no_data:
        provenance["no_data_reason"] = _resolve_no_data_reason(
            resources, selected_filters, answer,
        )
    provenance.update(_provenance_display_fields(provenance))
    return provenance


def _provenance_display_fields(provenance: Dict[str, Any]) -> Dict[str, Any]:
    selected_filters = provenance.get("selected_filters") or {}
    labels = {
        "variable": "Variable",
        "variables": "Variables",
        "region": "Region",
        "regions": "Regions",
        "scenario": "Scenario",
        "scenarios": "Scenarios",
        "model": "Model",
        "models": "Models",
        "years": "Years",
        "unit": "Unit",
    }
    rows: list[dict[str, str]] = []
    for key in (
        "variable", "variables", "region", "regions", "scenario", "scenarios",
        "model", "models", "years", "unit",
    ):
        value = selected_filters.get(key)
        if value:
            rendered = ", ".join(str(item) for item in value) if isinstance(value, list) else str(value)
            rows.append({"label": labels[key], "value": rendered})
    if provenance.get("matched_record_count") is not None:
        rows.append({"label": "Matched records", "value": str(provenance.get("matched_record_count"))})
    if provenance.get("cache_timestamp"):
        rows.append({"label": "Cache timestamp", "value": str(provenance.get("cache_timestamp"))})
    if provenance.get("no_data_reason"):
        rows.append({"label": "No-data reason", "value": str(provenance.get("no_data_reason"))})
    for key, label in (
        ("comparison_dimension", "Comparison"),
        ("chart_type", "Chart type"),
        ("displayed_series_count", "Displayed series"),
        ("omitted_series", "Omitted series"),
    ):
        value = provenance.get(key)
        if value not in (None, ""):
            rows.append({"label": label, "value": str(value)})
    return {
        "display_title": "Data provenance",
        "display_rows": rows,
    }


def _is_low_confidence(value: Any) -> bool:
    """True for a numeric confidence below 0.5; non-numeric values are ignored."""
    try:
        return float(value) < 0.5
    except (TypeError, ValueError):
        return False


def _should_log_eval_candidate(trace: Dict[str, Any]) -> bool:
    route_confidence = float(trace.get("route_confidence") or 0.0)
    entity_confidence = trace.get("entity_confidence") or {}
    low_entity = isinstance(entity_confidence, dict) and any(
        _is_low_confidence(value) for value in entity_confidence.values()
    )
    return bool(trace.get("no_data_reason") or route_confidence < 0.55 or low_entity)


def _write_eval_feedback_candidate(trace: Dict[str, Any], answer: str, log_path: str | Path | None = None) -> bool:
    if not EVAL_FEEDBACK_ENABLED:
        return False
    if not _should_log_eval_candidate(trace):
        return False
    path = Path(log_path or os.getenv("IAM_EVAL_FEEDBACK_LOG", "docs/eval_feedback_candidates.jsonl"))
    try:
        # Size cap: stop appending once the file is large, so a flood of
        # queries cannot exhaust disk space.
        if path.exists() and path.stat().st_size >= EVAL_FEEDBACK_MAX_BYTES:
            logger.warning("eval feedback log at size cap (%s bytes); skipping write", EVAL_FEEDBACK_MAX_BYTES)
            return False
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
    except OSError:
        # Feedback is best-effort telemetry. Never discard an answer because
        # the filesystem is read-only, full, or temporarily unavailable.
        logger.warning("Could not write eval feedback candidate to %s", path, exc_info=True)
        return False


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
        if isinstance(entity_confidence, dict) and any(
            _is_low_confidence(value) for value in entity_confidence.values()
        ):
            _monitoring_counters["low_confidence_entity_queries"] += 1
        _save_monitoring_counters()


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def _feedback_log_summary(
    path: str | Path | None = None,
    *,
    include_sensitive: bool = True,
) -> Dict[str, Any]:
    log_path = Path(path or os.getenv("IAM_EVAL_FEEDBACK_LOG", "docs/eval_feedback_candidates.jsonl"))
    rows = []
    try:
        if not log_path.exists():
            result = {"count": 0, "details_redacted": not include_sensitive}
            if include_sensitive:
                result.update({"path": str(log_path), "recent": []})
            return result
        with log_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        logger.warning("Could not read eval feedback log %s", log_path, exc_info=True)
        result = {
            "count": 0,
            "details_redacted": not include_sensitive,
            "unavailable": True,
        }
        if include_sensitive:
            result.update({"path": str(log_path), "recent": []})
        return result
    result = {"count": len(rows), "details_redacted": not include_sensitive}
    if include_sensitive:
        result.update({"path": str(log_path), "recent": rows[-5:]})
    return result


def _monitoring_snapshot(*, include_sensitive: bool = False) -> Dict[str, Any]:
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
        "feedback_candidates": _feedback_log_summary(include_sensitive=include_sensitive),
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
    if "incompatible units" in text or "can't combine" in text or "cannot combine" in text:
        return "incompatible units"
    if "requested year range" in text or "no time series data" in text:
        return "no values in requested year range"
    if "no timeseries data" in text and "model" in text:
        return "model combination unavailable"
    if "scenario combination" in text:
        return "scenario combination unavailable"
    if "region combination" in text:
        return "region combination unavailable"
    if (
        "requested variable is unavailable" in text
        or "not found in loaded data" in text
        or "could not identify variable" in text
        or "could not identify a variable" in text
    ):
        return "variable unavailable in current scope"
    if "model `" in text or "using model" in text:
        return "model combination unavailable"
    if "under `" in text:
        return "scenario combination unavailable"
    if " in `" in text or "in region" in text:
        return "region combination unavailable"
    return "no matching data slice"


def _resolve_no_data_reason(
    resources: Dict[str, Any],
    entities: Dict[str, Any],
    answer: str,
) -> str:
    """Prefer definitive renderer errors over stale or ambiguous scope clues."""
    classified = _classify_no_data_reason(answer)
    if classified in {
        "incompatible units",
        "no values in requested year range",
        "model combination unavailable",
        "variable unavailable in current scope",
    }:
        return classified
    return _derive_no_data_reason(resources, entities) or classified


def _prepare_relevant_links(links: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    prepared: List[Dict[str, Any]] = []
    for link in links or []:
        if not isinstance(link, dict):
            continue
        item = dict(link)
        title = str(item.get("title") or "IAM PARIS link")
        search_hint = str(item.get("search_hint") or "").strip()
        url = str(item.get("url") or "")
        if (
            str(item.get("category") or "").casefold() == "models"
            and (
                not is_presentable_model_label(search_hint or title)
                or is_unlabelled_model_display(search_hint or title)
            )
        ):
            continue
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


def _suggested_next_questions(
    query: str,
    answer: str,
    manager: Any,
    *,
    has_plot: bool = False,
) -> List[str]:
    route = dict(getattr(manager, "last_route_decision", {}) or {})
    entities = _manager_response_entities(manager)
    agent = str(route.get("agent") or "")
    text = str(answer or "")
    suggestions: List[str] = []

    def add(item: str) -> None:
        if item and item not in suggestions:
            suggestions.append(item)

    available_scenarios = list(getattr(getattr(manager, "entity_extractor", None), "available_scenarios", []) or [])
    # Scenario suggestions must come from the selected study. The extractor's
    # catalogue spans every loaded workspace and can otherwise recommend a
    # scenario that has no records in the user's current study.
    workspace_code = str(entities.get("workspace_code") or "").strip()
    if workspace_code:
        records = list((getattr(manager, "shared_resources", {}) or {}).get("ts") or [])
        available_scenarios = sorted({
            str(record.get("scenario") or "").strip()
            for record in records
            if str(record.get("workspace_code") or "").strip() == workspace_code
            and str(record.get("scenario") or "").strip()
        })
    baseline_scenario = next(
        (scen for scen in available_scenarios if "baseline" in str(scen).lower()),
        "",
    )
    current_scenario = str(entities.get("scenario") or "").lower()
    unmatched_region = str(entities.get("unmatched_region") or "").strip()

    if agent in {"data_query", "data_plotting"}:
        if unmatched_region:
            add("Show available regions")
            add("Help me choose a region")
        elif _is_no_data_answer(text, has_plot=has_plot):
            add("Show available scenarios")
            add("Show available regions")
            add("Show available variables")
        else:
            if entities.get("variable") or entities.get("region"):
                add("Open the data explorer")
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
    runtime_models = _runtime_model_scope_override(resources, entities)
    unmatched_region = str(entities.get("unmatched_region") or "").strip()
    scope = {
        "workspace_code": str(entities.get("workspace_code", "") or "").strip(),
        "variable": str(entities.get("variable", "") or "").strip(),
        "region": unmatched_region or str(entities.get("region", "") or "").strip(),
        "scenario": str(entities.get("scenario", "") or "").strip(),
        "model": (
            "" if runtime_models
            else str(entities.get("model", "") or "").strip()
        ),
    }
    plural_values = {
        dimension: {
            str(value).strip()
            for value in (entities.get(plural) or [])
            if str(value).strip()
        }
        for dimension, plural in (
            ("variable", "variables"),
            ("region", "regions"),
            ("scenario", "scenarios"),
            ("model", "models"),
        )
    }
    if runtime_models:
        plural_values["model"] = set(runtime_models)
    if unmatched_region:
        plural_values["region"] = set()
    years_text = str(entities.get("years", "") or "").strip()
    year_numbers = [int(value) for value in re.findall(r"\b(?:19|20|21)\d{2}\b", years_text)]
    start_bound = entities.get("start_year")
    end_bound = entities.get("end_year")
    latest_requested = is_latest_year_filter(start_bound, end_bound) or (
        start_bound is None and end_bound is None and "latest" in years_text.casefold()
    )
    if start_bound is None and end_bound is None and year_numbers:
        lowered_years = years_text.casefold()
        if re.search(r"\b(?:until|through|up to|before)\b", lowered_years):
            end_bound = max(year_numbers)
        elif re.search(r"\b(?:from|after|since)\b", lowered_years) and len(year_numbers) == 1:
            start_bound = year_numbers[0]
        else:
            start_bound, end_bound = min(year_numbers), max(year_numbers)
    unit = str(entities.get("unit", "") or "").strip()
    active_scope = {key: value for key, value in scope.items() if value}
    for dimension, values in plural_values.items():
        if values:
            active_scope[f"{dimension}s"] = values
    if start_bound is not None or end_bound is not None:
        active_scope["years"] = (start_bound, end_bound)
    if unit:
        active_scope["unit"] = unit
    if not active_scope:
        return None

    records = resources.get("ts") or []
    if not records:
        return 0

    def _normalized_unit(value: object, variable: object = "") -> str:
        normalized = re.sub(r"\s+", " ", str(value or "").strip()).casefold()
        normalized = normalized.replace(" per year", "/yr")
        normalized = re.sub(r"/\s*(?:year|y|a)(?=\b|$)", "/yr", normalized)
        normalized = re.sub(r"\s+", "", normalized)
        variable_key = str(variable or "").strip().casefold()
        if variable_key == "carbon price" or variable_key.startswith("price|carbon"):
            normalized = re.sub(r"/t(?:co2)?$", "/tco2", normalized)
        return normalized

    def _year_values(record: Dict[str, Any]) -> dict:
        values = {
            int(key): value
            for key, value in record.items()
            if str(key).isdigit() and len(str(key)) == 4
        }
        nested = record.get("years")
        if isinstance(nested, dict):
            values.update({
                int(key): value
                for key, value in nested.items()
                if str(key).isdigit() and len(str(key)) == 4
            })
        return values

    latest_by_table = {}

    def _has_value_in_period(record: Dict[str, Any]) -> bool:
        values = _year_values(record)
        if latest_requested:
            year = latest_by_table.get((record.get("variable"), record.get("region")))
            return year is not None and is_finite_numeric_value(values.get(year))
        if start_bound is None and end_bound is None:
            return True
        return any(
            (start_bound is None or year >= int(start_bound))
            and (end_bound is None or year <= int(end_bound))
            and is_finite_numeric_value(value)
            for year, value in values.items()
        )

    def _matches(record: Dict[str, Any]) -> bool:
        if (
            scope["workspace_code"]
            and str(record.get("workspace_code", "")).strip() != scope["workspace_code"]
        ):
            return False
        if scope["variable"] and str(record.get("variable", "")).strip() != scope["variable"]:
            return False
        if plural_values["variable"] and str(record.get("variable", "")).strip() not in plural_values["variable"]:
            return False
        if scope["region"] and not regions_equivalent(record.get("region"), scope["region"]):
            return False
        if plural_values["region"] and not any(
            regions_equivalent(record.get("region"), wanted)
            for wanted in plural_values["region"]
        ):
            return False
        if scope["scenario"]:
            record_scenario = str(record.get("scenario", "")).strip()
            if record_scenario != scope["scenario"] and not scenario_in_family(
                record_scenario, scope["scenario"]
            ):
                return False
        if plural_values["scenario"]:
            record_scenario = str(record.get("scenario", "")).strip()
            if not any(
                record_scenario == wanted or scenario_in_family(record_scenario, wanted)
                for wanted in plural_values["scenario"]
            ):
                return False
        if scope["model"]:
            model = str(record.get("modelName") or record.get("model") or "").strip()
            if is_unlabelled_model_display(scope["model"]):
                if is_presentable_model_label(model):
                    return False
            elif model != scope["model"]:
                return False
        if plural_values["model"]:
            model = str(record.get("modelName") or record.get("model") or "").strip()
            matches_model = any(
                (
                    is_unlabelled_model_display(wanted)
                    and not is_presentable_model_label(model)
                )
                or model == wanted
                for wanted in plural_values["model"]
            )
            if not matches_model:
                return False
        record_variable = str(record.get("variable") or "").strip()
        if unit and _normalized_unit(
            record.get("unit"), record_variable,
        ) != _normalized_unit(unit, record_variable or scope["variable"]):
            return False
        return True

    matching = [record for record in records if isinstance(record, dict) and _matches(record)]
    if latest_requested:
        for record in matching:
            years = _year_values(record)
            finite_years = [
                year for year, value in years.items()
                if is_finite_numeric_value(value)
            ]
            if finite_years:
                key = (record.get("variable"), record.get("region"))
                latest_by_table[key] = max(
                    latest_by_table.get(key, -1),
                    max(finite_years),
                )
    return sum(1 for record in matching if _has_value_in_period(record))


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


def _sanitize_model_presentation_text(value: object) -> str:
    """Defensively sanitize model-labelled Markdown emitted by legacy paths."""
    text = str(value or "")
    numeric = r"[+-]?\d+(?:[.,]\d+)*"
    text = re.sub(
        rf"(\bmodel(?:s)?\s+`)({numeric})(`)",
        rf"\1{UNLABELLED_MODEL_LABEL}\3",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        rf"(\*\*)({numeric})(\s+-\s+[^*]+\*\*)",
        rf"\1{UNLABELLED_MODEL_LABEL}\3",
        text,
    )
    if re.search(r"(?im)^\|\s*Model\s*\|", text):
        text = re.sub(
            rf"(?m)^(\|\s*)({numeric})(\s*\|)",
            rf"\1{UNLABELLED_MODEL_LABEL}\3",
            text,
        )
    return text


def _split_answer_payload(answer: str) -> tuple[str, str, str, List[str]]:
    """
    Split mixed text/plot markdown answers into API-friendly fields.
    Returns: cleaned_answer, plot_base64, plot_caption, notices
    """
    text = _sanitize_model_presentation_text(answer).strip()
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
        # A manager owns mutable conversational scope. Serialize requests for
        # the same session while allowing different sessions to run in parallel.
        "lock": threading.RLock(),
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


class _QueryDeadlineExceeded(Exception):
    """Internal signal used when a query cannot finish within its deadline."""


def _snapshot_manager_state(manager: Any) -> Dict[str, Any]:
    """Capture mutable per-conversation state for transactional routing."""
    snapshot: Dict[str, Any] = {}
    for name in (
        "conversation_state",
        "last_result_models",
        "last_links",
        "last_route_decision",
        "turn_counter",
        "current_turn",
    ):
        if hasattr(manager, name):
            snapshot[name] = copy.deepcopy(getattr(manager, name))
    if "conversation_state" not in snapshot:
        for name in ("last_entities", "clarification_context"):
            if hasattr(manager, name):
                snapshot[name] = copy.deepcopy(getattr(manager, name))
    return snapshot


def _restore_manager_state(manager: Any, snapshot: Dict[str, Any]) -> None:
    for name, value in snapshot.items():
        setattr(manager, name, value)
    # Answer formatters use a worker-local resolved-scope channel. A failed or
    # expired turn must not leak that scope into a later task on the same worker.
    consume_resolved_scope()


def _process_session_query(req: QueryRequest, deadline: float) -> QueryResponse:
    """Run one complete session turn while the worker owns the session lock.

    A timed-out route may not be safely killed in Python. Keeping the lock in
    this worker until routing returns prevents a later request from observing
    partially mutated conversation state.
    """
    session_id, session_state = _get_or_create_session(
        req.session_id,
        reset_session=req.reset_session,
    )
    session_lock = session_state.setdefault("lock", threading.RLock())
    remaining = deadline - time.monotonic()
    if remaining <= 0 or not session_lock.acquire(timeout=remaining):
        raise _QueryDeadlineExceeded("Timed out waiting for the session lock.")

    try:
        manager = session_state["manager"]
        chat_history: List[Tuple[str, str]] = session_state["chat_history"]
        manager_snapshot = _snapshot_manager_state(manager)

        try:
            response = manager.route_query(req.query, list(chat_history))
        except Exception:
            _restore_manager_state(manager, manager_snapshot)
            raise
        if time.monotonic() > deadline:
            _restore_manager_state(manager, manager_snapshot)
            raise _QueryDeadlineExceeded("Query routing exceeded its deadline.")

        answer_text, plot_base64, plot_caption, notices = _split_answer_payload(response)
        history_limit = max(int(HISTORY_MAX_TURNS), 1)
        pending_history = (chat_history + [(req.query, answer_text)])[-history_limit:]
        trace = _build_query_trace(
            session_id,
            req.query,
            manager,
            answer_text,
            has_plot=bool(plot_base64),
        )
        logger.info("query_trace %s", json.dumps(trace, sort_keys=True, default=str))
        _update_monitoring(trace)
        _write_eval_feedback_candidate(trace, answer_text)
        relevant_links = _prepare_relevant_links(getattr(manager, "last_links", []))
        next_questions = _suggested_next_questions(
            req.query,
            answer_text,
            manager,
            has_plot=bool(plot_base64),
        )
        entities = _manager_response_entities(manager)
        clarification = _manager_clarification_payload(manager)
        route = dict(getattr(manager, "last_route_decision", {}) or {})
        data_provenance = _build_data_provenance(
            getattr(manager, "shared_resources", {}) or {},
            entities,
            answer_text,
            route,
            has_plot=bool(plot_base64),
        )

        result = QueryResponse(
            answer=answer_text,
            session_id=session_id,
            history=pending_history,
            plot_base64=plot_base64,
            plot_caption=plot_caption,
            notices=notices,
            relevant_links=relevant_links,
            suggested_next_questions=next_questions,
            entities=entities,
            data_scope=entities,
            clarification=clarification,
            data_provenance=data_provenance,
            route=route,
        )
        if time.monotonic() > deadline:
            raise _QueryDeadlineExceeded("Response preparation exceeded its deadline.")
        chat_history[:] = pending_history
        session_state["last_access"] = time.time()
        return result
    except Exception:
        if "manager_snapshot" in locals():
            _restore_manager_state(manager, manager_snapshot)
        raise
    finally:
        session_lock.release()

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
        deadline = time.monotonic() + API_REQUEST_TIMEOUT
        if not _query_slots.acquire(timeout=API_REQUEST_TIMEOUT):
            _update_monitoring(failed=True)
            raise HTTPException(
                status_code=504,
                detail="Query timed out waiting for processing capacity.",
            )
        try:
            future = _query_executor.submit(_process_session_query, req, deadline)
        except Exception:
            _query_slots.release()
            raise
        future.add_done_callback(lambda _future: _query_slots.release())
        try:
            remaining = max(deadline - time.monotonic(), 0.0)
            return future.result(timeout=remaining)
        except (FutureTimeoutError, _QueryDeadlineExceeded):
            _update_monitoring(failed=True)
            logger.warning(
                "Query exceeded the %.3g second request deadline",
                API_REQUEST_TIMEOUT,
            )
            raise HTTPException(
                status_code=504,
                detail="Query timed out. Please narrow the request and try again.",
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
    """Operational counters; query/session details require configured auth."""
    return _monitoring_snapshot(include_sensitive=bool(API_KEY))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
