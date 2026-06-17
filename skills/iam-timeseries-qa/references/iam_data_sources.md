# IAM PARIS Data Sources and Helpers

## API Endpoints and Env Vars

- `REST_MODELS_URL` (model metadata): `https://cms.iamparis.eu/items/models`
- `REST_API_FULL` (time-series results): `https://api.iamparis.eu/results`

Env vars are loaded in `main.py` and expected in the `env` file.

## Cache and Index Layout

- JSON cache files: `cache/*.json` (hash-based, per URL + params/payload)
- Link catalog source of truth: `iamparis_chatbot_links.xlsx`
- Generated link catalog: `docs/iamparis_link_catalog.json`
- YAML definitions cache:
  - `cache/yaml_definitions.pkl` (definitions/region and definitions/variable)
  - `cache/yaml_dicts.pkl` (variable/region dicts for query resolution)
- FAISS index: `cache/faiss_index/index.faiss`

Use `clear_cache()` in `main.py` to reset all caches.

## Core Entry Points

- Shared startup resources: `build_runtime_context` in `runtime_context.py`
- Model/time-series fetching: `IAMParisBot.fetch_json` in `main.py`
- Data query orchestration: `data_query` in `data_utils.py`
- Plotting: `simple_plot_query` and `plot_multiple_variables` in `simple_plotter.py`
- Link catalog loading/export helpers: `link_catalog.py`
- Link selection: `route_links` in `link_router.py`
- English query normalization: `query_normalizer.py`
- Shared model aliases: `model_aliases.py`
- Year filter helpers: `year_filters.py`
- Variable and region helpers: `utils_query.py`
- YAML loader: `utils/yaml_loader.py`

## Definition Sources

- Regions: `definitions/region/*.yaml`
- Variables: `definitions/variable/*.yaml`

These drive variable/region matching and natural-language resolution.

## Common Query Types

- “What models are available?” -> `get_available_models`
- “Show CO2 emissions for World” -> resolve variable + region, filter ts data
- “Plot solar capacity for EU” -> variable resolution + plot
- “Compare solar and wind capacity for World” -> multi-variable comparison plot
- “Where can I find Climate Watch?” -> Application Library fallback link + search hint
- “Tell me about GCAM assumptions” -> model metadata + Models link
- “AFOLU transformation results” -> relevant Results workspace link

## Response Conventions

- Keep user-facing responses English-only.
- Always include `unit`, `region`, `scenario`, and year range if applicable.
- For data answers, use the standard shape: heading, scope, unit, answer, next step, and relevant IAM PARIS links.
- Validate data combinations before answering; never invent values for unavailable slices.
- For no-data answers, use `Closest valid options:` with numbered choices.
- If multiple scenarios exist and none specified, ask the user to choose.
- If variable match is ambiguous, present top 3 candidate variables.
- Prefer plain-language clarification flows:
  - if the query is meaningful, suggest the top 3 relevant choices
  - if the query is too vague, ask directly for the missing variable/region/scenario
- Support quick replies:
  - `1`, `2`, `3` for numbered options
  - `yes` for option 1
  - `no` to request the next closest alternatives

## FastAPI Session and Structured Fields

`fastapi_app.py` keeps one session state per active `session_id`, including:

- `MultiAgentManager`
- `chat_history`
- last resolved entities/scope
- clarification context
- selected links

`reset_session: true` clears an existing session.

The API response includes structured fields:

- `answer`
- `session_id`
- `history`
- `plot_base64`
- `plot_caption`
- `notices`
- `relevant_links`
- `suggested_next_questions`
- `entities`
- `data_scope`
- `route`

## Link Catalog Rules

The link catalog is generated from `iamparis_chatbot_links.xlsx` and should not be hand-edited in runtime code.

Use direct URLs only when `verified_direct_url` is true. For Application Library entries without a verified detail URL, link to `https://iamparis.eu/application_library` and include the item title as `search_hint`.

Main categories:

- Models
- Results
- Application Library
- Data Stories
- Analysis
- Contact
