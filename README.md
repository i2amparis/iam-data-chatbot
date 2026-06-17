# IAM PARIS Data Chatbot

English-only chatbot for IAM PARIS model metadata, time-series results, plots, guided follow-up questions, and relevant `iamparis.eu` links.

The app uses cached IAM PARIS API data, YAML region/variable definitions, a shared runtime context, an availability matrix, and an Excel-derived link catalog to answer questions without inventing unavailable data.

## Features

- Model metadata answers with assumptions when available.
- Time-series answers for variables, regions, scenarios, models, and year filters.
- Plot generation for chart/graph/visualize requests.
- Multi-turn follow-ups such as `1`, `yes`, `plot it`, `same for China`, and `compare with baseline`.
- No-data recovery with closest valid options.
- Relevant IAM PARIS links from `iamparis_chatbot_links.xlsx`.
- FastAPI responses with structured fields for links, plots, entities, route metadata, and notices.
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

Create an `env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key
REST_MODELS_URL=https://cms.iamparis.eu/items/models
REST_API_FULL=https://api.iamparis.eu/results
```

The app loads this file with `python-dotenv`.

## CLI Usage

Interactive mode:

```bash
python main.py
```

Single query:

```bash
python main.py --query "show me carbon dioxide emissions for Europe"
```

Plot query:

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
  -d '{"session_id":"SESSION_ID_FROM_PREVIOUS_RESPONSE","query":"plot it"}'
```

Reset a session:

```bash
curl -s -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"session_id":"SESSION_ID_FROM_PREVIOUS_RESPONSE","reset_session":true,"query":"fresh question"}'
```

## API Response Fields

`POST /query` returns:

- `answer`: markdown answer text.
- `session_id`: conversation session identifier.
- `history`: session turns.
- `plot_base64`: plot image payload without requiring markdown parsing.
- `plot_caption`: short plot caption.
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

The monitoring endpoint reports total queries, failed query rate, no-data rate, low-confidence route rate, low-confidence entity rate, configured thresholds, alert status, and recent feedback candidates.

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
