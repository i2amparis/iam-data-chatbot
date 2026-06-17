# Query Behavior

This document summarizes the current query behavior implemented in the project. It is meant as a lightweight reference alongside `skills/iam-timeseries-qa/SKILL.md`.

## Purpose

The chatbot supports IAM PARIS model metadata, time-series lookup, discovery questions, clarification follow-ups, and plots. The behavior below reflects the current implementation in:

- `runtime_context.py`
- `manager.py`
- `data_utils.py`
- `simple_plotter.py`
- `link_catalog.py`
- `link_router.py`
- `query_normalizer.py`
- `main.py`
- `fastapi_app.py`

## Query Categories

### Discovery queries

These ask what is available in the dataset.

Examples:

- `What variables can you plot?`
- `List available models`
- `Which scenarios are available?`
- `What countries are included?`

Current behavior:

- Routes to dataset discovery, not plotting.
- Returns short summaries such as available variables, models, scenarios, or regions.
- Avoids overlong lists by default and offers `show all ...` when many items exist.
- Keeps the answer compact and actionable.

### Data lookup queries

These ask for values or trajectories for a variable, region, and optionally scenario/model.

Examples:

- `Show me CO2 emissions for USA`
- `electricity for India`
- `oil demand for EU`

Current behavior:

- Normalizes English synonyms before matching, for example `carbon dioxide` -> CO2, `photovoltaic` -> solar, and `gross domestic product` -> GDP.
- Resolves variables from plain language when possible.
- Uses existing variable and region matching logic.
- Validates requested variable/region/scenario/model combinations before returning values.
- If a required piece is missing, asks only for the next missing item.
- Prefers short clarification prompts with numbered options.
- Uses the standard answer shape: heading, scope, unit, answer, next step, and relevant IAM PARIS links.

### Plot queries

These explicitly ask to plot, graph, chart, or visualize.

Examples:

- `Plot solar capacity for EU under PR_WWH_CP`
- `graph carbon price for China`

Current behavior:

- Routes through `simple_plotter.py`.
- Uses plain-language matching before falling back to raw IAM variable names.
- Produces cleaner captions such as `Showing Solar Capacity in EU for scenario \`PR_WWH_CP\`.`
- Reuses recent session scope for follow-ups such as `plot it`.
- API responses separate `plot_base64` and `plot_caption` from the text answer.

### Model metadata queries

These ask about a specific IAM model.

Examples:

- `Tell me about GCAM`
- `What are the assumptions in the GCAM model?`

Current behavior:

- Returns available model metadata when a model can be matched.
- If assumptions are not explicitly present, it does not invent them.
- Includes a related IAM PARIS Models link.
- Uses the notice text:
  - `No explicit assumptions field is available in the model metadata.`

### Link and study/navigation queries

These ask where to find project pages, result hubs, Application Library tools, Data Stories, or analysis support.

Examples:

- `where can I find Climate Watch`
- `open the Aqueduct raw data application`
- `AFOLU agriculture land forestry transformation results`
- `custom analysis for decarbonisation options`

Current behavior:

- Uses `link_router.py` over the Excel-derived `docs/iamparis_link_catalog.json`.
- Returns top relevant links with title, URL, reason, confidence, and optional search hint.
- Uses direct Application Library URLs only when verified; otherwise links to `https://iamparis.eu/application_library` with a search hint.

### Clarification follow-ups

These are short replies to a previous clarification prompt.

Examples:

- `1`
- `2`
- `yes`
- `GDP|MER`
- `AUS`

Current behavior:

- Continues the active clarification only when the reply clearly answers the current prompt.
- Clarification expires if the user does not answer on the very next turn.
- A fresh full question resets the previous clarification thread.
- FastAPI keeps this state per `session_id` and supports `reset_session`.

## Clarification Rules

Current clarification behavior is intentionally short-lived.

- Full new question: resets previous clarification state.
- Direct short answer: continues clarification.
- Expired clarification: old options are dropped instead of being reused later.

Example:

1. Bot asks: `Choose the scenario: 1. PR_Baseline 2. PR_CurPol_CP`
2. User replies: `2`
3. Bot continues with that scenario.

But:

1. Bot asks: `Choose the scenario: 1. PR_Baseline 2. PR_CurPol_CP`
2. User replies: `What variables can you plot?`
3. Bot treats that as a new discovery question.

## Plain-Language Mapping

The project currently includes English-only normalization and extra support for common plain-language phrases.

Examples:

- `carbon dioxide`
  - maps toward CO2 variables
- `greenhouse gas`
  - maps toward GHG variables
- `solar energy`
  - prefers solar electricity / solar capacity families
- `solar capacity`
  - prefers `Capacity|Electricity|Solar`
- `photovoltaic`
  - maps toward solar/PV capacity or generation families
- `gross domestic product`
  - maps toward GDP variables
- `oil demand`
  - prefers oil-related energy families over unrelated matches
- `chart`, `graph`, `visualize`
  - map toward plot intent

This is implemented as ranking preference, not hardcoded one-off example matching.

## No-Data Recovery

When a requested combination is unavailable, the chatbot should not hallucinate values.

Current behavior:

- Uses an availability matrix before answering or plotting.
- Returns `I could not find data for ...`.
- Shows `Closest valid options:` with numbered choices.
- Lets the user continue with `1`, `2`, or `3`.
- Prioritizes suggestions by current scope where possible: same variable + same region first, same variable next, sector fallback last.

## Plot Output

### CLI behavior

In `main.py`:

- plot responses are parsed out of mixed text + markdown replies
- plots are saved under `plots/`
- the saved file is opened locally once
- raw base64 plot payload is not dumped to the terminal

### API behavior

In `fastapi_app.py`, plot metadata is returned separately:

- `answer`
- `plot_base64`
- `plot_caption`
- `notices`
- `relevant_links`
- `entities`
- `suggested_next_questions`
- `data_scope`
- `route`

This avoids forcing the API/UI layer to parse embedded markdown image strings.

## Structured Logging and Evaluation

The API logs a structured `query_trace` with:

- `session_id`
- `query`
- `route`
- `route_confidence`
- `route_source`
- `entities`
- `entity_confidence`
- selected variable/region/scenario/model
- matched records placeholder
- no-data reason
- selected links and link scores

The baseline evaluation set is in `eval_queries.csv`. Generate the pending-review report with:

- `python run_eval.py`

## Notices

Some responses include a separate notice in addition to the main answer.

Current example:

- `No explicit assumptions field is available in the model metadata.`

Current behavior:

- CLI prints this as a separate notice line.
- API returns it in `notices`.

## Known Routing Intent

The current routing prioritizes detection of:

- clarification follow-ups
- plot requests
- data lookup
- model metadata questions
- discovery/list questions
- study/link suggestions
- general QA fallback

This is important because many user-facing bugs in this project are routing bugs rather than data bugs.

## Tests That Protect This Behavior

These two test files are intentionally kept because they lock in the critical query and follow-up behavior:

- `test_manager_fallback.py` covers clarification lifecycle, follow-up handling (`plot it`), and numeric replies.
- `test_query_regressions.py` covers variable matching (solar capacity vs additions), discovery queries, plot parsing, and notices.
- `test_fastapi_smoke.py` covers session reuse, reset behavior, structured response fields, and query trace shape.
- `test_link_router.py` covers IAM PARIS link routing and Application Library direct/fallback behavior.
- `test_query_normalizer.py` and `test_query_extractor_confidence.py` cover English normalization and extraction confidence.

## Scope of This Document

This file describes the current implementation only. It should be updated when behavior changes in code so that:

- the skill stays accurate
- query handling remains predictable
- tests, docs, and runtime behavior stay aligned
