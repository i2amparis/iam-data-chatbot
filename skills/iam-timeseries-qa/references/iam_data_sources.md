# IAM PARIS Data Sources and Helpers

## API Endpoints and Env Vars

- `REST_MODELS_URL` (model metadata): `https://cms.iamparis.eu/items/models`
- `REST_API_FULL` (time-series results): `https://api.iamparis.eu/results`

Env vars are loaded in `main.py` and expected in the `env` file.

## Cache and Index Layout

- JSON cache files: `cache/*.json` (hash-based, per URL + params/payload)
- YAML definitions cache:
  - `cache/yaml_definitions.pkl` (definitions/region and definitions/variable)
  - `cache/yaml_dicts.pkl` (variable/region dicts for query resolution)
- FAISS index: `cache/faiss_index/index.faiss`

Use `clear_cache()` in `main.py` to reset all caches.

## Core Entry Points

- Model/time-series fetching: `IAMParisBot.fetch_json` in `main.py`
- Data query orchestration: `data_query` in `data_utils.py`
- Plotting: `simple_plot_query` and `plot_multiple_variables` in `simple_plotter.py`
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

## Response Conventions

- Always include `unit`, `region`, `scenario`, and year range if applicable.
- If multiple scenarios exist and none specified, ask the user to choose.
- If variable match is ambiguous, present top 3 candidate variables.
- Prefer plain-language clarification flows:
  - if the query is meaningful, suggest the top 3 relevant choices
  - if the query is too vague, ask directly for the missing variable/region/scenario
- Support quick replies:
  - `1`, `2`, `3` for numbered options
  - `yes` for option 1
  - `no` to request the next closest alternatives
