# Response Fixes TODO

Findings from the 2026-10-07 review: full code read, quality gate, and ~40 real
queries run end-to-end against the cached data (631k records, 21 studies).
Ordered by priority. Each item lists the cause, the steps, and how to verify.

Reproduce any item offline with real data: build the runtime context from the
cached results (as `fastapi_app.initialize_resources` does, with
`vector_store=None`), point `LOCAL_LLM_BASE_URL` at a closed port, and call
`fastapi_app._process_session_query`. Set `IAM_EVAL_FEEDBACK_ENABLED=0` and
`IAM_MONITORING_STATE=""` so nothing is written to `docs/` or `cache/`.

---

## P0: Broken main flow

### 1. Study gate blocks most data questions and loses the question

8 of 12 ordinary data questions ("CO2 emissions for World in 2050", "GDP of the
world", "population of Germany in 2050") return "Please choose a study before I
return numeric results…" with no numbers. Replying with the study title gives
"I need one more detail…" (scope lost); the next "same for China" then returns
**World** data for 2005-2100; replying `1` gives "no active numbered choice".

Cause: `_workspace_choice_response` (`data_utils.py:176`) returns plain text and
records no pending clarification. It is reached from `_data_query`
(`data_utils.py:2744`) and `format_time_series_data` (`data_utils.py:4904`).

- [x] Decide the default policy (recommended: answer from the study with the
      most matching records for the requested variable/region/years, and add a
      line "Also available in: <study A>, <study B> — ask to switch").
- [x] If keeping a choice instead: turn the study prompt into a typed
      clarification (numbered options, `suggested_kind="workspace"`, base
      entities = the extracted variable/region/scenario/years) so `1`, `yes`
      and the study title all resolve it in `_route_single`.
- [x] Make sure the chosen study is persisted as `workspace_code` and the
      original variable/region/years survive the clarification turn.
- [x] Make `_is_no_data_answer` / `_suggested_next_questions`
      (`fastapi_app.py:522`, `:1082`) recognise the study prompt so it does not
      suggest "Compare with Baseline" / "By 2050" for an answer with no data.
- [x] Add regression tests with **two or more workspaces** in the fixture (current
      fixtures have one, which is why the gate never caught this).
- Verify: `CO2 emissions for World in 2050` → numbers; then `same for China` →
  China, 2050, same study; `compare with baseline` → table.

### 2. Own suggestions and follow-ups dead-end on "plot"

Clicking the suggested "Compare with Baseline" returns "Please use the data
explorer to plot these results." The follow-up composer rewrites it as
`plot compare Emissions|CO2 for World versus baseline`, which trips the plot
redirect in `data_query` (`data_utils.py:2410`).

- [x] `manager.py:2622`: build `compare …` / `show …` instead of `plot compare …`.
- [x] `manager.py:1707`: stop telling users to ask `plot compare …`; suggest a
      table query instead.
- [x] Make "compare with <scenario>" without a carried single scenario return a
      table of the current scope vs the named scenario (not a Baseline-only or
      all-scenario table).
- [x] Remove plotting promises from user-facing text: small-talk/hello answer
      ("Plot data, e.g. …"), the scenario list ("You can plot variables for any
      of these scenarios"), and the `ModellingSuggestionsAgent` "try `plot …`"
      hints in `agents.py`.
- [x] grep all user-facing strings for `plot` and check each one against the
      "charts go to the data explorer" policy.
- [x] Test: every string returned by `_suggested_next_questions` must, when sent
      as the next query, not produce a redirect or a "no active choice" answer.

---

## P1: Wrong interpretations

### 3. Region treated as a model ("does Germany have")

`how much solar capacity does Germany have` → "I could not match model
`Germany`". The `does/do X have|report|…` regex at `data_utils.py:3674` assumes
X is a model.

- [x] Skip that branch when the captured word resolves to a region
      (`extract_region_from_query` / `regions_equivalent`) or a variable term.
- [x] Test with Germany, Greece, EU, World.

### 4. "which models report hydrogen" returns the generic model list

Routed as "model availability request"; `_models_covering_topic_answer`
returns `None` for hydrogen, so the generic "75 models available" list is shown.

- [x] Let topic matching fall back to variable-path matching (any variable whose
      path contains the term, e.g. `|Hydrogen`), not only the sector categories.
- [x] When a topic is named but nothing matches, say so instead of listing all
      models.

### 5. Free text mistaken for an unknown region

`what does net zero mean for electricity prices` → "I couldn't find
`electricity prices` as a region".

- [x] In unknown-region detection (`unknown_named_region`, `data_utils.py:780`,
      and `_explicit_unknown_region`, `query_extractor.py:461`) reject
      candidates that match a variable term (price, emissions, capacity, …) or
      a scenario phrase.
- [x] Route conceptual "what does X mean for Y" questions to general QA (or to a
      data answer for the Y variable), not to the region error.
- [x] Found while fixing this: with no region requested, records from several
      regions collapsed into one model/scenario row showing an arbitrary
      region's values. `format_time_series_data` now uses World when present,
      otherwise asks a numbered region question.

### 6. "CO2 for Atlantis" shows the study chooser

The trace records `unmatched_region=Atlantis`, but the study gate answers first.

- [x] Check the unmatched region before the study gate and return the
      "not a region" answer with region suggestions.

### 7. Wrong study link for a named study

`what are the main differences between the scenarios in Where is the EU headed?`
→ link to "Comparison of Fit-for-55…".

- [x] In navigation/link scoring prefer an exact workspace-title match
      (`_matched_workspace`) over token overlap.
- [x] Consider answering this as a study summary (`_workspace_summary`:
      scenarios + explainer link) rather than a bare link.

### 8. Study-suggestion request hijacked by the variable picker

`suggest research ideas on transport` → variable clarification, while
`suggest research ideas` correctly reaches `modelling_suggestions`.

- [x] Run the suggestion check before `_low_confidence_entity_prompt` in
      `_route_single` (`manager.py:6080`), or skip the entity prompt when the
      query has suggestion intent.

### 9. Model comparison adds an unrequested model

`compare GCAM and REMIND` → "GCAM vs REMIND-MFA vs REMIND" with two identical
REMIND descriptions.

- [x] When an exact family name is given, select only that profile; do not add
      variants by prefix.

---

## P1: Answer quality

### 10. Number formatting loses precision

`_format_value` (`data_utils.py:5011`) renders 35,612 as `35.6K Mt CO2/yr` and
0.0034 as `0.00`.

- [x] Use thousands separators and ~3–4 significant digits (`35,612`, `0.0034`).
- [ ] ~~Optionally rescale to a natural unit (Mt → Gt).~~ Not done: values now
      print in full with separators (`35,612 Mt CO2/yr`), so no `K` is mixed
      with the unit; rescaling would change the source unit.
- [x] Update tests that assert the old `K/M` format (none did; values from
      1 to 999 keep two decimals). Units differing only by case ("Million" /
      "million") are now merged.

### 11. No plain-language takeaway; tables are too large

A single-year question returns up to 38 series (12 shown, alphabetical, so only
two models appear).

- [x] Add a deterministic one-to-two sentence headline before the table, computed
      from the data: range (min/max with model and scenario), median, and change
      vs the first year for trajectories. No LLM needed.
- [x] Choose the displayed rows deliberately (e.g. baseline, current policies,
      the most ambitious scenario, one per model) instead of the first N
      alphabetically; keep "Narrow by model or scenario…" for the rest.
- [ ] ~~For a single requested year, show a compact Model × Scenario layout.~~
      Not done: the single-year table already has one value column, and a
      Model × Scenario pivot is mostly empty because each model reports its
      own scenario names.

### 12. Inconsistent model names and counts

Tables mix raw labels (`e3me`, `gcam`) with display names (`E4SMA-EU-TIMES
1.0`); "which models cover transport" lists `ices` next to `ICES-XPS 1.0`; the
counts disagree (75 models vs "of 81").

- [x] Map raw timeseries model labels to catalogue display names in one place
      (`display_model_label`) and use it in every table and list.
- [x] Use one model count everywhere (catalogue vs models with data), labelled
      clearly.
- [x] "which models are available": show families or group by type instead of
      the first 8 alphabetically.

---

## P1: Latency (2.5–8 s per data query before any LLM time)

Profiling one query: `_workspace_entries` regroups all records 3× (2.4 s),
`_runtime_model_scope_override` rebuilds the same model-name set 4× (1.6 s),
`_count_matching_records` scans everything 2× (0.9 s), plus repeated catalogue
set-builds in `manager.py` and `data_utils.py`.

- [x] Precompute once in `RuntimeContext` (`runtime_context.py`): workspace
      entries, runtime model names, scenario/variable/region/model sets, and an
      index `(workspace, variable) → records`.
- [x] Make `_workspace_entries`, `_runtime_model_scope_override`,
      `_count_matching_records`, `_suggested_next_questions` and the setcomps in
      `_data_query` / `_repair_comparison_entities` use those.
- [x] Compute provenance record counts once per request (the trace and
      `_build_data_provenance` currently both count).
- [x] Also fixed: per-catalogue-value regexes (`_match_catalog_value_from_text`,
      exact-variable availability, `rank_catalogue_variable_matches`) now
      pre-check with a substring test; token similarity and stemming are
      memoized; the router LLM client is shared across sessions; caches are
      warmed at startup (`_warm_runtime_caches`, +4 s at boot).
- Result (22 real questions, no LLM): median ~0.25 s, max 0.73 s per
  question, down from 2.5–8 s. Not every question is under 0.3 s yet;
  follow-ups that rebuild scope ("same for China") take ~0.6–0.7 s.

---

## P2: LLM setup

- [ ] **Your decision (not changed):** use a bigger model for user-facing answers: set `IAM_QA_MODEL` to a 4B–8B
      Qwen or `gpt-4o-mini`; keep `qwen3:0.6b` for routing only (`.env`,
      `llm_config.py`). Production settings live in Dokku, so this is left to
      you; the README now recommends a 4B–8B model or `gpt-4o-mini` for the
      QA and extractor roles.
- [x] Extractor (`query_extractor.py:343`): enable JSON mode (`format="json"` for
      Ollama, `response_format={"type": "json_object"}` for OpenAI) instead of
      parsing raw text.
- [x] Extractor prompt: replace "first 40 variables alphabetically"
      (`query_extractor.py:206`) with the most commonly queried variables.
- [x] Router: normalise the LLM reply (strip punctuation/quotes) before checking
      `VALID_AGENT_NAMES` (`manager.py:2198`).
- [x] Set `num_ctx` explicitly for `ChatOllama` in `llm_factory._build_local` and
      check the general-QA prompt fits (model list + ~4k chars skill guidance +
      5 × 800-char chunks + history).
- [x] `GeneralQAAgent` (`agents.py:431`): shorten past answers in `chat_history`
      (e.g. first ~300 chars, tables stripped) so the question-rewrite step
      stays small; consider skipping that extra LLM call on CPU.
- [x] Results cache never expires (`main.py` `fetch_json`): add a max age or a
      scheduled refresh so production does not serve old data indefinitely.
      Done as opt-in `IAM_CACHE_MAX_AGE_HOURS` (default 0 = unchanged), since
      a full refresh is ~600 paged requests; a failed refresh serves the cache.

---

## P2: Tests and tooling

- [ ] **Your action (not changed):** local envs (`botenv`, conda `botenv`, `iam-chatbot`) lack
      `langchain-ollama`: run `pip install -r requirements.txt`. `botenv` is
      Python 3.13 while production and CI use 3.10. Left for you because it
      changes environments outside the repo; recommended:
      `conda activate botenv && pip install -r requirements.txt` (Python 3.10).
- [x] Stop tests writing real files: point `IAM_EVAL_FEEDBACK_LOG`,
      `IAM_MONITORING_STATE` and the metadata cache path at a temp dir in the
      test setup. Done in `quality_gate.py` (scratch feedback log and
      monitoring file), and the default metadata cache is now one file per
      dataset (`cache/data_metadata_<signature>.pkl`, 3 kept), so tests and
      the server no longer overwrite each other's cache. The old
      `cache/data_metadata.pkl` is unused and can be deleted.
- [x] Remove the 43 duplicate test "Atlantis" rows from
      `docs/eval_feedback_candidates.jsonl` and reset
      `cache/monitoring_counters.json` (currently test traffic). Kept the
      most recent Atlantis row; other repeated queries were left as is.
- [x] `quality_gate.py` static eval rewrites tracked `docs/*.md` reports; write
      them to a temp/output dir or only on demand. Now a temp dir by default;
      `python quality_gate.py --write-reports` updates `docs/`.
- [x] Add a real-data "golden questions" check to the gate when the cache is
      present: ~30 questions (including every item above) with the expected
      route, entities, and "has numbers / no redirect" assertions. Done:
      `eval_golden_real_data.json` + `test_golden_real_data.py` (in the gate;
      skipped without a cache, e.g. in CI). Against the original code it
      fails on 15 turns, one per original bug.
- [ ] Found while doing this: `test_query_regressions.py` loads whichever
      `cache/models*.json` / `cache/results*.json` is largest, so its result
      depends on local cache contents (a new models cache file made 4 of its
      tests fail). Pin it to a fixture or a named cache file.

---

## P3: Cleanup

- [x] Remove dead code (partly): `_create_qa_chain` in `DataQueryAgent` and
      `ModelExplanationAgent` (`agents.py`), `llm_wrapper.py`, and the plotting
      paths that can no longer be reached (`simple_plotter.py`,
      `DataPlottingAgent`, the "previous-scope comparison plot" branch in
      `_route_single`) once item 2 confirms plotting stays disabled.
- [ ] **Not done (separate change):** split `_route_single` (`manager.py:4134`, ~2,000 lines) into ordered,
      individually tested route handlers, so one phrasing fix cannot silently
      break another. A ~2,000-line restructure should be its own reviewed
      change; the new golden and regression tests are the safety net for it.
