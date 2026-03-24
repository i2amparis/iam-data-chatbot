---
name: iam-timeseries-qa
description: Answer IAM PARIS questions about Integrated Assessment Models (IAMs) and IAM PARIS time-series results. Use when a user asks for model metadata, available models/scenarios/variables, region- or scenario-specific time-series values, comparisons across technologies (e.g., solar vs wind), or plotting/visualizing IAM PARIS results sourced via REST_MODELS_URL/REST_API_FULL.
---

# Iam Timeseries Qa

## Overview

Provide reliable answers and plots for IAM PARIS climate/economic model metadata and time-series results. Use existing data fetch, caching, variable/region matching, and plotting utilities in this repo to avoid reinventing query logic.

Companion reference:
- `skills/iam-timeseries-qa/query-behavior.md` summarizes the current implemented behavior in the codebase. Prefer that file when you need to check what the app actually does today.

## Quick Workflow

1. Identify whether the user asks about model metadata, time-series values, or a plot/comparison.
2. Expect plain-language queries, not exact IAM variable strings.
3. Use existing helpers (see references) to resolve variables/regions/scenarios when ambiguous.
4. If the query contains enough semantic signal, offer the top 3 relevant choices and let the user answer with `1`, `2`, `3`, or `yes` for option 1.
5. If the query is too vague, ask only for the next missing piece instead of guessing.
6. Fetch data through existing API helpers with caching enabled.
7. Return concise results with units, region, scenario, and any assumptions called out.
8. Prefer intent- and token-based interpretation over literal example matching. Treat example phrasings in this skill as guidance, not hardcoded trigger strings.

## Routing Order

When a query could fit more than one category, prefer this order:

1. Clarification follow-up
2. Discovery / "what is available" question
3. Model metadata question
4. Plot or visualization request
5. Data lookup question
6. Fallback recovery prompt

Why this order:
- discovery questions like `What variables can you plot?` should not fall into the plotter
- short replies like `1` or `yes` should continue the current clarification
- plot and data lookup should use the same plain-language variable families when possible

## Multi-Query Handling

When a user asks multiple things in one message, split the request into ordered sub-questions and answer in the same order. Examples:
- “Tell me about REMIND and plot solar capacity for EU” → (1) model info, (2) plot solar capacity
- “List models and show CO2 for World” → (1) list models, (2) time-series values

Rules:
1. Do not drop any sub-question.
2. If a sub-question needs clarification, ask it after giving any answers you already have.
3. Keep clarifications short and specific (one question at a time).

## Core Tasks

### 1) Model Metadata Questions

Use when users ask about specific IAMs (e.g., REMIND, GCAM) or list available models.

Do:
- Fetch model metadata from `REST_MODELS_URL` via the project helper.
- Provide model description and assumptions when available.
- If asked “what models are available,” list a short sample and offer to show all.

Avoid:
- Guessing model details not present in metadata.

### 2) Time-Series Value Questions

Use when users ask for values like “CO2 emissions for World” or “capacity for EU.”

Do:
- Resolve meaning from the user’s text first; do not require API-style variable syntax.
- Resolve variable and region using YAML definitions and fuzzy matching helpers.
- Filter results by variable, region, and scenario if provided.
- Return a concise summary (latest year if not specified) with unit.

Ask a clarifying question if:
- Variable is ambiguous or not found.
- Region is missing and multiple regions exist.
- Scenario is required (multiple scenarios present) and user did not specify.

If a requested model is not present in the model metadata, say so clearly and suggest:
- Asking for available models
- Querying time-series by region/variable without model filtering
  
### 2a) Clarifying Questions (Priority Order)

Ask in this order, only if needed:
1. Variable (what indicator?)
2. Region (where?)
3. Scenario (which pathway?)
4. Model (only if the user explicitly wants a model and multiple exist)

### 2b) Plain-Language Follow-Ups

Use this behavior for natural-language requests like:
- `energy in buildings for EU`
- `carbon from transport in Europe`
- `electricity for India`
- `oil demand for EU`

Rules:
1. If the wording contains enough signal, suggest the top 3 most relevant variable choices.
2. Present choices in numbered form and tell the user they can reply with `1`, `2`, `3`, or `yes`.
3. Add a short human-readable hint after each choice when helpful.
4. If the wording is too broad, ask a direct follow-up like:
   - `I found the region \`EU\`. Which variable should I use?`
5. Do not show a generic starter menu when the query has no meaningful signal.

### 3) Plot and Visualization Requests

Use when users say “plot/show/graph/visualize.”

Do:
- Route through the existing plot utilities.
- Return the generated base64 plot (or follow the app’s display mechanism).

If data missing:
- Provide a short message explaining the missing combination and suggest alternatives.

### 4) Comparisons (Multiple Variables)

Use when users ask to compare two technologies (e.g., solar vs wind).

Do:
- Detect multi-variable comparisons.
- Resolve each variable to exact IAM PARIS variable names before plotting.
- Keep the comparison within the same region/scenario unless the user requests otherwise.

### 5) “What’s Available?” Questions

Use existing helpers to list:
- Models
- Scenarios
- Variables
- Workspaces (if requested)
- Regions

Provide a short list with counts and offer to show more.

Example prompts to suggest:
- “list scenarios”
- “list regions”
- “list workspaces”

## Variable/Region/Scenario Resolution

1. Use YAML definitions for variable and region normalization.
2. Prefer exact IAM PARIS variable names when the user already provides pipe-delimited variables.
3. For technology terms (solar, wind, etc.), map to capacity/generation variables using existing matching logic.
4. For fuel terms like **oil**, resolve to the closest IAM variable (e.g., `Secondary Energy|Electricity|Oil`) and ask for clarification if multiple oil-related variables exist.
5. For broad electricity/power requests without a named technology (for example `electricity for India`), prefer a clarification prompt over locking onto a specific technology like wind or solar.
6. If the user’s request is vague (“show me data”, “plot it”) and the variable match is uncertain, ask a follow‑up to confirm the variable instead of guessing.
7. Use confidence thresholds and score gaps: if the best match is low‑confidence or close to the second‑best, ask for confirmation.
8. Reject obviously bad first matches. If the top internal match is semantically far from the user’s wording, fall back to a ranked clarification instead of using it.
9. Prefer sector-aware suggestions for queries like buildings, transport, industry, electricity, and emissions.

If the resolver returns multiple candidates, ask for disambiguation and present the top 3 options ranked by similarity.
Prefer numbered choices for speed:
- Example: `Choose the variable: 1. A 2. B 3. C`.
- Tell users they can reply with just `1`, `2`, or `3` (and `yes` means option 1).

## Missing Info Guidance

When the user’s query is incomplete, guide them with a single, short follow-up that makes it easy to answer.

Principles:
- Ask only for the next missing item in the priority order.
- Offer 2–3 concrete options (top regions/workspaces/scenarios) to reduce user effort.
- Only suggest region/workspace if the user hasn’t already provided one earlier in the conversation.
- Carry forward any previously resolved variable/region/scenario into later sub-queries (e.g., “plot it”).

If the user asks for data without specifying a region or workspace:
- Ask for the missing piece and provide the top 3 available regions/workspaces.
- If a scenario is missing and multiple exist, show the top 3 scenarios.

If the user asks for “data” without a specific variable:
- If the query has meaningful signal, show 2–3 recommended variable candidates that are most similar to the query.
- If the query does not have meaningful signal, ask directly which variable they want.
- Use numbered options so the user can answer with one digit.

If the user gives a region but no variable:
- Confirm the region and ask for the variable before guessing.
- Prefer a short follow-up like “I found `CHN`. Which variable should I use?”

If the user does not know what data exists:
- Switch to discovery mode and summarize what is available.
- Suggest the next best commands: `list variables`, `list regions`, `list scenarios`, `list models`, or a concrete example query.
- Keep it short and actionable so the user can continue without needing domain knowledge.
- Tailor the examples to the user's wording when possible, for example energy-related examples for energy questions and emissions examples for climate-policy questions.

## Clarification State

Clarification prompts should be short-lived and predictable.

Rules:
1. A fresh full question resets any older clarification thread.
2. Short replies continue the current clarification when they clearly answer the prompt.
3. Valid clarification replies include examples like `1`, `2`, `3`, `yes`, `GDP|MER`, `AUS`, or scenario shorthand.
4. Clarification state should only survive for the next direct user reply. If the user does not answer on the next turn, expire it instead of carrying stale state forward.
5. Generic follow-ups like `plot it` or `show it` may reuse recent resolved entities, but they should not revive an expired clarification prompt.

Examples:
- Bot: `Choose the variable: 1. \`GDP|MER\` 2. \`GDP|PPP\``
- User: `2`
- Bot: continue with `GDP|PPP`

- Bot: `Choose the scenario: 1. \`PR_Baseline\` 2. \`PR_CurPol_CP\``
- User: `What variables can you plot?`
- Bot: treat it as a new question, not as a clarification reply.

## Not Available Handling

If the specific combination is missing (variable + region + scenario):
1. Say it is not available.
2. Offer a compact recovery prompt with up to 2–3 closest options for variable, region, or scenario as available.
3. Ask a single follow-up question to proceed (e.g., “Reply with the variable, region, or scenario you want next.”).
4. If the user rejects the first suggestion (`no`), offer the next closest alternatives instead of resetting the whole flow.

If multiple scenarios exist for a variable and the user didn’t specify one:
- Ask the user to choose a scenario and show the 2–3 most relevant scenario options.
- Include short helper labels when possible (for example: `baseline`, `current policy`, `NDC`, `net zero`).

If a model name does not match available model metadata:
- Return a clear message that the model could not be matched.
- Suggest `list models` before continuing with model-filtered requests.

## Data Retrieval and Caching

1. Always use the existing `fetch_json` helper with caching enabled.
2. Respect the cache files and FAISS index; avoid rebuilding unless needed.
3. If the user asks to “refresh” or “reload,” clear cache with the built-in utility.

## Plot Output

When a query generates a plot:

1. Save the rendered PNG under `plots/`.
2. Prefer stable, query-based filenames so repeated test runs are easy to find, for example `plots/plot_solar_capacity_for_eu_<hash>.png`.
3. In the CLI, open the saved plot locally after saving it.
4. Tell the user the file path so they can open it directly from the workspace.
5. If you need to inspect a saved plot, open `plots/plot_*.png` from the workspace.

## Notices and Assumptions

When a model-assumptions query does not have an explicit assumptions field in the metadata:

1. Say so clearly instead of inventing assumptions.
2. Surface the message as a separate notice when the app supports notices.
3. Keep the answer useful by returning any related metadata that is actually available.

Use this exact notice text when appropriate:
- `No explicit assumptions field is available in the model metadata.`

## Example Behaviors

- Vague query:
  - User: `show me data for EU`
  - Bot: `I found the region \`EU (European Union)\`. Which variable should I use?`

- Plain-language query with enough signal:
  - User: `oil demand for EU`
  - Bot: `Based on your wording, I think one of these is the closest match. Choose the variable: 1. ... 2. ... 3. ...`

- Follow-up confirmation:
  - User: `2`
  - Bot: continue with that exact choice instead of re-guessing.

- Rejection:
  - User: `no`
  - Bot: offer the next closest options when available.

## Resources

### references/
- `references/iam_data_sources.md` for API endpoints, cache layout, and code entry points.

## Tools To Use

Use these existing repo tools to answer questions accurately and consistently:

- `IAMParisBot.fetch_json` in `main.py`
  - use for model metadata and time-series API retrieval
  - keep caching enabled unless the user explicitly asks to refresh

- `data_query` in `data_utils.py`
  - use for time-series question answering
  - use for variable/region/scenario clarification
  - use for no-data recovery and ranked follow-up choices

- `simple_plot_query` and plotting helpers in `simple_plotter.py`
  - use for plot, graph, chart, and comparison requests
  - use existing plot saving behavior instead of building a new plotting path

- `MultiAgentManager` in `manager.py`
  - use for multi-intent routing
  - use for follow-up handling across turns
  - use for numbered clarification choices and `yes` / `no` behavior

- `QueryEntityExtractor` in `query_extractor.py`
  - use to extract region, scenario, model, years, and action from plain-language questions

- matching helpers in `utils_query.py`
  - use for fuzzy variable resolution
  - use for region alias matching
  - use for normalizing user phrasing to IAM PARIS concepts

Prefer these runtime artifacts when available:

- `cache/results*.json`
  - use as the main fallback when the API is unavailable
  - useful for local verification and relevance testing

- `cache/faiss_index`
  - use for semantic lookup and model/info support

- `plots/plot_*.png`
  - use to inspect saved plots and report the output path back to the user

Guidance:

- Prefer existing repo helpers over inventing new query logic.
- Prefer cached data over failing outright when API connectivity is unstable.
- Prefer the query/plot manager flow over direct ad hoc filtering unless debugging.

## Validation

Use the existing regression checks after query-flow changes:

- `MPLCONFIGDIR=/tmp/mplcache python -m unittest test_query_regressions.py`

Expected:

- tests pass for vague-query follow-up behavior
- tests pass for ranked top-3 suggestions
- tests pass for scenario clarification prompts
