# IAM PARIS Data Chatbot

English-only chatbot for IAM PARIS model metadata, compact time-series results, guided follow-up questions, and relevant `iamparis.eu` links.

The app uses cached IAM PARIS API data, YAML region/variable definitions, a shared runtime context, an availability matrix, and an Excel-derived link catalog to answer questions without inventing unavailable data.

## Features

- Model metadata answers with assumptions when available.
- Time-series answers for variables, regions, scenarios, models, and year filters.
- Data Explorer redirects for chart/graph/visualize requests instead of inline plots.
- Multi-turn follow-ups such as `1`, `yes`, `same for China`, and `compare with baseline`.
- No-data recovery with closest valid options.
- Relevant IAM PARIS links from `iamparis_chatbot_links.xlsx`.
- FastAPI responses with structured fields for links, entities, route metadata, and notices.
- Evaluation baseline in `eval_queries.csv`.

## Setup

Create and activate a Python 3.10 environment:

```bash
conda create -n botenv python=3.10
conda activate botenv
pip install -r requirements.txt
```

Or with `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key
REST_MODELS_URL=https://cms.iamparis.eu/items/models
REST_API_FULL=https://api.iamparis.eu/results
IAM_API_KEY=replace_with_a_long_random_secret
IAM_API_REQUEST_TIMEOUT=120
# Set to 1 only behind a trusted proxy that sanitizes X-Forwarded-For.
IAM_TRUST_PROXY=0
```

The app loads this file with `python-dotenv`.

Cached API data is reused until `--refresh-data`. Set
`IAM_CACHE_MAX_AGE_HOURS` (e.g. `168`) to refresh a cache older than that at
startup; if the refresh fails the existing cache is served.

### Local models (Qwen + Ollama) alongside OpenAI

Model selection is configuration-only, so the OpenAI path is never removed. With
nothing set the app behaves exactly as before (OpenAI only). To run everything on
a local Ollama server, set:

```env
LOCAL_LLM_MODEL=qwen3:0.6b-q4_K_M
LOCAL_LLM_REASONING_EFFORT=none
IAM_EMBEDDING_MODEL=nomic-embed-text
LOCAL_LLM_BASE_URL=http://localhost:11434
```

- `LOCAL_LLM_MODEL` makes every role (router, extractor, QA) default to that model.
  Override a single role with `IAM_ROUTER_MODEL`, `IAM_EXTRACTOR_MODEL` or
  `IAM_QA_MODEL`, e.g. keep `IAM_QA_MODEL=gpt-4o` while routing locally.
- Set the matching `IAM_ROUTER_PROVIDER`, `IAM_EXTRACTOR_PROVIDER` or
  `IAM_QA_PROVIDER` to `ollama` or `openai` when overriding a role model whose
  provider cannot be inferred from `IAM_LOCAL_MODELS` / `LOCAL_LLM_MODEL`.
  For example, use `IAM_QA_MODEL=qwen3:4b` with `IAM_QA_PROVIDER=ollama`.
- `IAM_EMBEDDING_MODEL` switches embeddings to Ollama. Because the dimensions
  differ (768 vs 1536), the FAISS index is rebuilt automatically the first time.
  While it is unset, embeddings use OpenAI; override that OpenAI model with
  `IAM_OPENAI_EMBEDDING_MODEL` (default `text-embedding-3-small`).
- `LOCAL_LLM_REASONING_EFFORT=none` disables reasoning tokens (`think: false`) for
  models such as qwen3; without it the `<think>` output breaks JSON parsing.
- `LOCAL_LLM_TIMEOUT` (seconds) overrides the HTTP timeout for local models. CPU
  inference is much slower than the OpenAI endpoints the call sites were tuned
  for, so a long prompt can exceed the default 30s; set e.g. `120`-`240` together
  with a matching `IAM_API_REQUEST_TIMEOUT`.
- `LOCAL_LLM_NUM_CTX` (default `8192`) sets the Ollama context window. Ollama
  silently truncates prompts longer than the server default, and the general
  Q&A prompt (model list, guidance, retrieved chunks, history) can exceed it.
- The project runs on `qwen3:0.6b`. Because a model this small is unreliable
  from its own memory, most answers do not depend on it: data, model profiles,
  glossary terms (`config/glossary.json`) and variable definitions (expanded
  from `definitions/variable`) are answered deterministically, and free-form
  general Q&A uses a short grounded prompt over the knowledge index. Entity
  extraction requests JSON output mode.
- `python scripts/benchmark_local_model.py` measures the configured model on
  the eval questions (route accuracy, LLM calls, latency) and saves its
  general-Q&A answers for review. It writes to a temporary directory only.
- On the server, run Ollama and pull the models:

  ```bash
  ollama pull qwen3:0.6b-q4_K_M
  ollama pull nomic-embed-text
  ```

- Remove those variables (or unset `LOCAL_LLM_MODEL`) to return to OpenAI. If the
  app runs in a container, `localhost` does not reach the host: point
  `LOCAL_LLM_BASE_URL` at the host gateway (e.g. `http://172.17.0.1:11434`) and
  bind Ollama to that gateway only, e.g. `OLLAMA_HOST=172.17.0.1:11434`. Never
  expose port 11434 publicly: Ollama has no authentication, so restrict it with
  the host firewall to the container network.

## Deployment notes (for the server operator)

- **First start after an update rebuilds the knowledge index** (about 700
  chunks embedded with `IAM_EMBEDDING_MODEL`; ~12 s on a laptop, longer on a
  CPU server). It is cached in `cache/faiss_index` and reused while its content
  is unchanged, so keep `cache/` on persistent storage.
- **If the index cannot be built** (embedding server down, startup budget
  exceeded) the API still starts: data, model and definition answers work,
  free-form general Q&A replies that it is unavailable, and `/status` shows
  `knowledge_index_error`. Startup logs the duration of each step.
- **Labels are normalized at load**: whitespace and case-only variants of
  variable, region, model and unit names are merged (scenario names keep their
  published spelling), so the parsed-metadata cache is rebuilt once.
- Optional settings: `LOCAL_LLM_NUM_CTX` (default 8192), `LOCAL_LLM_TIMEOUT`,
  `IAM_CACHE_MAX_AGE_HOURS` (default 0 = never refresh cached API data
  automatically).

## CLI Usage

Interactive mode:

```bash
python main.py
```

Single query:

```bash
python main.py --query "show me carbon dioxide emissions for Europe"
```

Data Explorer redirect query:

```bash
python main.py --query "plot photovoltaic capacity for Greece"
```

Debug logging:

```bash
python main.py --debug
```

Force API data refresh:

```bash
python main.py --refresh-data
```

Clear local cache:

```bash
python main.py --clear-cache
```

## FastAPI Usage

Run the API:

```bash
uvicorn fastapi_app:app --reload
```

Health and status:

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/status
```

Ask a question:

```bash
curl -s -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"show Emissions|CO2 for World under Baseline in 2030"}'
```

Continue a session:

```bash
curl -s -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"session_id":"SESSION_ID_FROM_PREVIOUS_RESPONSE","query":"same for China"}'
```

Reset a session:

```bash
curl -s -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"session_id":"SESSION_ID_FROM_PREVIOUS_RESPONSE","reset_session":true,"query":"fresh question"}'
```

Clients that never send `session_id` (the current iamparis.eu chat sends only
`{"query"}`) still get follow-ups: the backend recognises the visitor by the
address the reverse proxy appends to `X-Forwarded-For` plus the browser's
User-Agent, for 30 minutes of inactivity. An explicit `session_id` always takes
precedence. Requests without a proxy header always start a new conversation.
Set `IAM_IMPLICIT_SESSIONS=0` to disable this, or `IAM_IMPLICIT_SESSION_TTL`
(seconds) to change the window.

Startup (downloading all studies and building the knowledge index) is allowed
`IAM_INITIALIZATION_TIMEOUT` seconds, 900 by default.

## API Response Fields

`POST /query` returns:

- `answer`: markdown answer text.
- `session_id`: conversation session identifier.
- `history`: session turns.
- `plot_base64` and `plot_caption`: legacy compatibility fields; normally empty because plot requests redirect to the Data Explorer.
- `notices`: UI-friendly notices such as missing assumptions metadata.
- `relevant_links`: selected IAM PARIS links with title, URL, reason, confidence, and search hint.
- `suggested_next_questions`: future UI suggestions.
- `entities`: resolved variable, region, scenario, model, year, and confidence metadata.
- `data_scope`: final selected data scope.
- `data_provenance`: cache timestamp, matched record count, selected filters, route metadata, and frontend-friendly `display_rows`.
- `route`: route agent, confidence, source, and reason.

The API also logs a structured `query_trace` with session, route, entity, no-data, and link information.

Frontend rendering hints:

- Render `relevant_links[].action == "open"` as a normal link button.
- Render `relevant_links[].action == "search"` as an Application Library search action using `search_hint`.
- Show `relevant_links[].display_hint` when a direct detail URL is not exposed.
- Render `data_provenance.display_rows` as a compact "Data provenance" details panel.

Monitoring:

```bash
curl -s http://127.0.0.1:8000/monitoring
```

The monitoring endpoint reports total queries, failed query rate, no-data rate, low-confidence route rate, low-confidence entity rate, configured thresholds, and alert status. Recent feedback entries contain query text and session IDs, so they are returned only when `IAM_API_KEY` is configured and supplied in `X-API-Key`; otherwise those details are redacted.

For deployments behind a reverse proxy, enable `IAM_TRUST_PROXY=1` only if the proxy removes client-supplied forwarding headers and writes the trusted `X-Forwarded-For` value itself.

## Link Catalog

The source of truth is:

```text
iamparis_chatbot_links.xlsx
```

The runtime catalog is generated as:

```text
docs/iamparis_link_catalog.json
```

Regenerate it after editing the Excel workbook:

```bash
python link_catalog.py
```

The generator reads `.xlsx` files directly and does not require `openpyxl`.

Expected workbook sheets:

- `00_README`
- `01_Main_Routes`
- `02_Data_Stories`
- `03_Results`
- `04_Models`
- `05_App_Library`
- `06_Routing_Map`
- `07_Summary`

Expected source columns used by `link_catalog.py`:

- `01_Main_Routes`: `Category`, `Subcategory / Page`, `URL`, `Description`, `Chatbot routing hint`, `Status`
- `02_Data_Stories`: `Title`, `URL`, `Description`, `Keywords / intents`, `Status`
- `03_Results`: `Title`, `URL`, `Type`, `Keywords / intents`, `Status`
- `04_Models`: `Model name`, `URL to use`, `URL status`, `Full name / Description`, `Organisation`
- `05_App_Library`: `Title`, `URL to use`, `URL status`, `Type / Subcategory`, `Source / cataloguer`, `Keywords / intents`

Important catalog fields:

- `title`
- `url`
- `category`
- `project`
- `workspace`
- `item_type`
- `keywords`
- `verified_direct_url`
- `search_hint`

Application Library items use a verified detail URL only when available. Otherwise the chatbot links to `https://iamparis.eu/application_library` and tells the user what to search for.

## Runtime Architecture

- `runtime_context.py`: builds shared startup resources.
- `manager.py`: deterministic routing, follow-up state, and LLM fallback.
- `query_normalizer.py`: English synonym normalization.
- `query_extractor.py`: entity extraction and confidence.
- `data_metadata.py`: metadata and availability matrix.
- `data_utils.py`: data answers, discovery, clarifications, and no-data recovery.
- `simple_plotter.py`: plot and comparison handling.
- `link_catalog.py`: Excel to runtime JSON catalog.
- `link_router.py`: relevant IAM PARIS link selection.
- `fastapi_app.py`: API sessions, structured responses, and monitoring logs.

## Example Queries

Model info:

```text
tell me about GCAM model assumptions
```

Data query:

```text
show Emissions|CO2 for World under Baseline in 2030
```

Plot query:

```text
plot solar capacity for EU under Baseline
```

Follow-up:

```text
plot it
same for China
compare with baseline
```

English synonym query:

```text
show me carbon dioxide emissions for Europe
plot photovoltaic capacity for Greece
gross domestic product for World
```

Application Library fallback link:

```text
where can I find Climate Watch
```

## Testing

Use `MPLCONFIGDIR` so matplotlib can write cache files during tests:

```bash
MPLCONFIGDIR=/tmp/mplcache python -m unittest test_clarification_prompts.py test_manager_fallback.py test_fastapi_smoke.py test_query_regressions.py test_data_metadata.py test_year_filters.py
MPLCONFIGDIR=/tmp/mplcache python -m unittest test_query_extractor_confidence.py test_query_normalizer.py test_model_aliases.py test_link_catalog.py test_link_router.py test_runtime_context.py
```

Generate the evaluation report:

```bash
python run_eval.py
```

Run the same evaluation against a local FastAPI server:

```bash
python run_eval.py --live-url http://127.0.0.1:8000/query
```

Run the holdout evaluation:

```bash
python run_eval.py --holdout
python run_eval.py --holdout --live-url http://127.0.0.1:8000/query
```

Re-render the Markdown report from an existing live JSON file without calling the API again:

```bash
python run_eval.py --live-results docs/evaluation_live_results.json
```

This writes:

```text
docs/evaluation_results.md
```

Live mode also writes:

```text
docs/evaluation_live_results.json
```

Review low-confidence/no-data user questions and export eval-ready rows:

```bash
python feedback_review.py
```

This reads `docs/eval_feedback_candidates.jsonl` and writes:

```text
docs/eval_feedback_review.md
docs/eval_feedback_candidates.csv
```

Run the reviewed feedback set through the evaluator:

```bash
python run_eval.py --feedback
python run_eval.py --feedback --live-url http://127.0.0.1:8000/query
```

The evaluation CSV tracks expected route, variable, region, scenario, model, useful clarification, useful link, and no hallucinated data. Live scoring is deterministic and conservative; rows marked `review` still need human inspection.

Run the full local quality gate:

```bash
MPLCONFIGDIR=/tmp/mplcache python quality_gate.py
```

The unit tests use small synthetic records and run without `cache/`, as in GitHub Actions. To add the separate real-data integration check when local IAM PARIS caches are present:

```bash
MPLCONFIGDIR=/tmp/mplcache python quality_gate.py --skip-static-eval --include-real-data
```

The real-data option requires cached model and result files; `cache/` stays excluded from Git.

Run the quality gate against a running local API:

```bash
MPLCONFIGDIR=/tmp/mplcache python quality_gate.py --live-url http://127.0.0.1:8000/query
```

Optionally include IAM PARIS link validation before deployment:

```bash
MPLCONFIGDIR=/tmp/mplcache python quality_gate.py --validate-links
```

Audit saved live responses for frontend rendering safety:

```bash
python frontend_response_audit.py --live-results docs/evaluation_live_results.json
```

Or audit fresh sample responses from a running API:

```bash
python frontend_response_audit.py --live-url http://127.0.0.1:8000/query
```

This writes:

```text
docs/frontend_response_audit.md
```
