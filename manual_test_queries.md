# Manual Test Queries — IAM PARIS Chatbot

Use these to manually verify the bot after the recent changes. The bot is
**English-only**. Paste each query into the CLI (`python main.py`) or the API
(`POST /query` with a stable `session_id`).

How to run:
- CLI: `python main.py`
- API: start `uvicorn fastapi_app:app --port 8000`, then POST
  `{"query": "...", "session_id": "test-1"}` to `http://127.0.0.1:8000/query`.

For multi-turn blocks, **reuse the same `session_id`** so follow-ups carry context.

---

## 1. Basic data queries
- [ ] `CO2 emissions for EU` → numeric data, scope (scenario/model/years), Results link
- [ ] `GDP for China` → resolves GDP (no no-data loop), returns values
- [ ] `population projection for World` → resolves Population variable
- [ ] `final energy demand in India` → Final Energy family, NOT silently CO2

## 2. Variable normalization / English synonyms
- [ ] `show me carbon dioxide emissions for Europe` → maps to `Emissions|CO2`
- [ ] `gross domestic product for India` → `GDP|MER`
- [ ] `greenhouse gas pathways by country` → broad GHG, asks/uses sensible scope
- [ ] `solar PV capacity for Greece` → `Capacity|Electricity|Solar`
- [ ] `wind power for Germany` → wind capacity/electricity, not investment/additions

## 3. Regions & scenarios
- [ ] `emissions under current policies for EU` → scenario `Current Policies`
- [ ] `CO2 for Greece` → resolves dataset `GREECE` code
- [ ] `scenarios available for net zero` → filtered list (not all 101)
- [ ] `what scenarios are there for current policies` → category list, filtered

## 4. Model info
- [ ] `what is REMIND` → model description + assumptions, model doc link
- [ ] `information on GCAM` → GCAM profile
- [ ] `tell me about message ix` → resolves `MESSAGEix-GLOBIOM 2.0`
- [ ] `which models are available` → model list (no stale region carry-over)
- [ ] `what is an integrated assessment model` → CONCEPTUAL → general_qa, NOT
      "couldn't match a model"

## 5. Plots
- [ ] `plot CO2 emissions for EU` → returns plot (plot_base64) + caption
- [ ] `visualize final energy demand for India` → Final Energy plot, never silent CO2
- [ ] `plot photovoltaic capacity for Greece` → solar capacity plot
- [ ] `chart GDP for China` → plot via synonym "chart"

## 6. Comparisons (multi-region / multi-variable)
- [ ] `compare CO2 emissions between China and India` → TWO-region plot (not just China)
- [ ] `CO2 emissions for EU vs China` → two regions
- [ ] `compare wind power and solar PV for Greece` → two variables traced

## 7. No-data recovery
- [ ] `electricity capacity for Germany` (not in dataset) → suggests valid options
      that actually have data (no dead-end loop); prefers aggregate (EU) region
- [ ] `CO2 for Atlantis` → graceful no-data, closest valid regions
- [ ] Pick a suggested option (`1`) → returns real data, never re-loops

## 8. Site navigation & links
- [ ] `where can I find Climate Watch` → grounded Application Library link
- [ ] `open the Aqueduct raw data application` → direct/grounded link
- [ ] `link to the model documentation` → link, NOT a variable prompt
- [ ] `AFOLU agriculture land forestry transformation results` → AFOLU result route
- [ ] `policy catalogue climate policies` → Policy Catalogue (Data Stories) link

## 9. Follow-up / multi-turn (same session_id)
Block A — region carry-over:
1. [ ] `CO2 emissions for EU`
2. [ ] `same for China` → keeps variable Emissions|CO2, switches region

Block B — year follow-up:
1. [ ] `GDP for India`
2. [ ] `what about 2050` → same variable/region/scenario, year 2050

Block C — plot follow-up:
1. [ ] `CO2 emissions for World`
2. [ ] `now plot it` → plots previous scope (filler "now" stripped)

Block D — scenario follow-up:
1. [ ] `emissions for EU under current policies`
2. [ ] `compare with baseline` → reuses scope, switches only the scenario

Block E — failed-turn resilience:
1. [ ] `CO2 World`
2. [ ] `now plot it`
3. [ ] `same for India` → still keeps the variable from the last successful turn

## 10. Discovery / lists
- [ ] `which variables are available` → list with "show all" option
- [ ] `which models cover buildings` → buildings-relevant subset (not full 74 list)
- [ ] `list scenarios` → truncated list with show-all

## 11. Edge cases / typos / clarification
- [ ] `emisions for europ` → fuzzy-tolerant or graceful clarification
- [ ] `electricity` (one word) → guided clarification, not empty/over-specific
- [ ] `oil` / `emissions` / `policy` / `NDC` (one word each) → clarification
- [ ] `can I find the buildings results` → must NOT read "can" as region CAN

---

## What to check on every answer
- Correct variable / region / scenario / model resolution (no silent wrong fallback)
- `relevant_links` present and on-topic (model query → that model's page)
- `suggested_next_questions` populated for data/no-data/model/nav answers
- `data_provenance` (cache timestamp, matched record count, filters) on numeric answers
- Plots return `plot_base64` + `plot_caption`
- No hallucinated numbers when data is missing
