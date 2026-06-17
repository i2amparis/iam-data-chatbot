# Baseline And Safety

This file records the starting point for the IAM PARIS chatbot optimization work.

## Baseline Date

- Recorded on: 2026-06-02
- Source: local cache files only, no live API refresh

## Cache Sources

- Model cache: `cache/models_4fd508ff5fa0d20d.json`
- Results cache: `cache/results_merged.json`
- Timeseries merge source: `current`

## Current Counts

- Models: 74
- Timeseries records: 183,743
- Variables: 1,152
- Regions: 180
- Scenarios: 101
- Workspaces: 20

## Workspaces

- `afolu`
- `buildings-transf`
- `covid-rec`
- `decarb-potentials`
- `energy-systems`
- `eu-headed`
- `index-decomp`
- `industrial-transf`
- `ndcs-impacts`
- `net-zero`
- `post-glasgow`
- `power-people`
- `study-1`
- `study-2`
- `study-3`
- `study-4`
- `study-6`
- `study-7`
- `transp-transf`
- `world-headed`

## Baseline Commands

Run the existing regression suite:

```sh
MPLCONFIGDIR=/tmp/mplcache python -m unittest test_query_regressions.py
```

Result on 2026-06-02: `26 tests OK`.

Run the manager fallback tests:

```sh
python -m unittest test_manager_fallback.py
```

Result on 2026-06-02: `8 tests OK`.

Run the FastAPI smoke tests:

```sh
python -m unittest test_fastapi_smoke.py
```

Result on 2026-06-02: `2 tests OK`.

Run the link catalog tests:

```sh
python -m unittest test_link_catalog.py
```

Result on 2026-06-02: `3 tests OK`.

## Link Catalog Decision

- Keep `iamparis_chatbot_links.xlsx` as the source-of-truth input.
- Do not mutate the Excel file during runtime.
- Generate a deterministic runtime catalog in Phase 2.
- Preferred runtime catalog path: `docs/iamparis_link_catalog.json`.
- The generated catalog should be committed once the generator is implemented and stable.
- In production request handling, load the JSON catalog instead of parsing Excel.
- Runtime catalog generated on 2026-06-02: 256 entries.
