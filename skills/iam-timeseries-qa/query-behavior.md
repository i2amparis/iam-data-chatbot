# Query Behavior

This document summarizes the current query behavior implemented in the project. It is meant as a lightweight reference alongside `skills/iam-timeseries-qa/SKILL.md`.

## Purpose

The chatbot supports IAM PARIS model metadata, time-series lookup, discovery questions, clarification follow-ups, and plots. The behavior below reflects the current implementation in:

- `manager.py`
- `data_utils.py`
- `simple_plotter.py`
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
- Keeps the answer compact and actionable.

### Data lookup queries

These ask for values or trajectories for a variable, region, and optionally scenario/model.

Examples:

- `Show me CO2 emissions for USA`
- `electricity for India`
- `oil demand for EU`

Current behavior:

- Resolves variables from plain language when possible.
- Uses existing variable and region matching logic.
- If a required piece is missing, asks only for the next missing item.
- Prefers short clarification prompts with numbered options.

### Plot queries

These explicitly ask to plot, graph, chart, or visualize.

Examples:

- `Plot solar capacity for EU under PR_WWH_CP`
- `graph carbon price for China`

Current behavior:

- Routes through `simple_plotter.py`.
- Uses plain-language matching before falling back to raw IAM variable names.
- Produces cleaner captions such as `Showing Solar Capacity in EU for scenario \`PR_WWH_CP\`.`

### Model metadata queries

These ask about a specific IAM model.

Examples:

- `Tell me about GCAM`
- `What are the assumptions in the GCAM model?`

Current behavior:

- Returns available model metadata when a model can be matched.
- If assumptions are not explicitly present, it does not invent them.
- Uses the notice text:
  - `No explicit assumptions field is available in the model metadata.`

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

The project currently includes extra support for a few common plain-language phrases.

Examples:

- `solar energy`
  - prefers solar electricity / solar capacity families
- `solar capacity`
  - prefers `Capacity|Electricity|Solar`
- `oil demand`
  - prefers oil-related energy families over unrelated matches

This is implemented as ranking preference, not hardcoded one-off example matching.

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

This avoids forcing the API/UI layer to parse embedded markdown image strings.

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
- discovery/list questions
- model metadata questions
- plot requests
- general data lookup

This is important because many user-facing bugs in this project are routing bugs rather than data bugs.

## Tests That Protect This Behavior

These two test files are intentionally kept because they lock in the critical query and follow-up behavior:

- `test_manager_fallback.py` covers clarification lifecycle, follow-up handling (`plot it`), and numeric replies.
- `test_query_regressions.py` covers variable matching (solar capacity vs additions), discovery queries, plot parsing, and notices.

## Scope of This Document

This file describes the current implementation only. It should be updated when behavior changes in code so that:

- the skill stays accurate
- query handling remains predictable
- tests, docs, and runtime behavior stay aligned
