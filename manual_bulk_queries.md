# Bulk Manual Smoke Queries — IAM PARIS Chatbot (API)

Paste these into the chatbot (API) in the given order.
For multi-turn blocks, reuse the same `session_id` for all turns in the block.

## Models / general QA
- [ ] `what is an integrated assessment model`
- [ ] `what is REMIND`
- [ ] `information on GCAM`
- [ ] `tell me about message ix`
- [ ] `which models are available`

## Data: variables + regions (happy paths)
- [ ] `CO2 emissions for EU`
- [ ] `GDP for China`
- [ ] `population projection for World`
- [ ] `final energy demand in India`
- [ ] `show me carbon dioxide emissions for Europe`
- [ ] `gross domestic product for India`

## Data from a specific model
Only ~34 of the 74 catalogued models have timeseries data. GCAM / E3ME / MUSE /
ICES / TIAM do; REMIND / MESSAGE / WITCH do NOT (expect a clear "no timeseries
for this model" answer, not a generic no-data).
- [ ] `GDP for China from GCAM`
- [ ] `CO2 emissions for EU from E3ME`
- [ ] `final energy for EU from MUSE`
- [ ] `CO2 emissions for EU from REMIND`   (model has no timeseries — should say so)
- [ ] `what data does GCAM have for the EU`

## Data: normalization / capacity / solar/wind
- [ ] `solar PV capacity for Greece`
- [ ] `wind power for Germany`
- [ ] `electricity capacity for Germany`

## Scenario (current policies) — may need canonical mapping fix
- [ ] `emissions under current policies for EU`
- [ ] `CO2 for Greece under current policies`

## Plots
- [ ] `plot CO2 emissions for EU`
- [ ] `visualize final energy demand for India`
- [ ] `chart GDP for China`
- [ ] `compare CO2 emissions between China and India`
- [ ] `compare wind power and solar PV for Greece`

## Navigation / links
- [ ] `where can I find Climate Watch`
- [ ] `open the Aqueduct raw data application`
- [ ] `link to the model documentation`
- [ ] `policy catalogue climate policies`

## Links — should return a working IAM PARIS link
These queries must produce at least one `relevant_links` entry whose URL resolves
(HTTP 200). Run with `--check-links` to verify the URLs actually work.
- [ ] `link to the REMIND model documentation`
- [ ] `model comparison page`
- [ ] `show me the GCAM model page`
- [ ] `buildings transformation results`
- [ ] `transport transformation results`
- [ ] `industrial transformation results`
- [ ] `AFOLU agriculture land forestry transformation results`
- [ ] `policy catalogue climate policies`
- [ ] `recovery policy database`
- [ ] `where is the application library`
- [ ] `open the Aqueduct raw data application`
- [ ] `where can I find Climate Watch`
- [ ] `NDC ASPECTS results`
- [ ] `IAM COMPACT net zero results`
- [ ] `data stories on decarbonisation`

## Links — find more info about the project
These should point the user to IAM PARIS pages where they can explore more.
- [ ] `where can I learn more about IAM PARIS`
- [ ] `tell me more about the IAM COMPACT project`
- [ ] `what is the NDC ASPECTS project`
- [ ] `where can I read about the project methodology`
- [ ] `give me the link to the scenario explorer`
- [ ] `where do I find the full list of models`
- [ ] `how do I explore the results online`
- [ ] `link to the IAM PARIS homepage`

## No-data recovery
- [ ] `electricity capacity for Germany` (if previously no-data, should suggest options)
- [ ] `CO2 for Atlantis` (should recover gracefully)

## Typos / clarification
- [ ] `emisions for europ`
- [ ] `electricity` (one word)
- [ ] `oil`
- [ ] `emissions`
- [ ] `policy`
- [ ] `NDC`

## Multi-turn blocks (reuse same session_id per block)

### Block A — region carry-over
1. [ ] `CO2 emissions for EU`
2. [ ] `same for China`

### Block B — year follow-up
1. [ ] `GDP for India`
2. [ ] `what about 2050`

### Block C — plot follow-up
1. [ ] `CO2 emissions for World`
2. [ ] `now plot it`

### Block D — scenario follow-up
1. [ ] `emissions for EU under current policies`
2. [ ] `compare with baseline`

### Block E — failed-turn resilience
1. [ ] `CO2 World`
2. [ ] `now plot it`
3. [ ] `same for India`

### Block F — model-scoped data then follow-ups
1. [ ] `GDP for EU from GCAM`
2. [ ] `what about China`
3. [ ] `now plot it`

### Block G — data then ask for more info / link
1. [ ] `final energy demand in India`
2. [ ] `which model is this from`
3. [ ] `where can I read more about this on the project site`

### Block H — scenario carry-over across variables
1. [ ] `CO2 emissions for EU under baseline`
2. [ ] `what about GDP`
3. [ ] `compare with current policies`
