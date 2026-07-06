# Evaluation Results

Status: pending live/manual review

This report is generated from `eval_feedback_candidates.csv`. It records the expected coverage set and can be extended to compare live chatbot outputs.

## Summary

- Total evaluation queries: 37
- Expected `data_plotting` queries: 9
- Expected `data_query` queries: 22
- Expected `general_qa` queries: 4
- Expected `model_explanation` queries: 2

## Tracking Fields

- correct route
- correct variable
- correct region
- correct scenario
- correct model
- useful clarification
- useful link
- no hallucinated data

## Query Set

| ID | Query | Expected route | Actual route | Variable | Region | Scenario | Model | Useful clarification | Useful link | No hallucinated data | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| feedback-001 | show wind data for Greece | data_query |  | Capacity Additions\|Electricity\|Wind | GREECE |  |  | yes | results | yes | pending_manual_review |  |
| feedback-002 | show solar data for Greece | data_query |  |  | GREECE |  |  | yes | results | yes | pending_manual_review |  |
| feedback-003 | show GDP for World under Baseline | data_query |  | GDP\|MER | World | Baseline |  | yes | results | yes | pending_manual_review |  |
| feedback-004 | net zero emissions for Europe | data_query |  | Emissions\|CO2 | EU | NZE_EUPol_Stand |  | yes | iam_compact | yes | pending_manual_review |  |
| feedback-005 | current policy CO2 for Europe | data_query |  | Emissions\|CO2 | EU | Current Policies |  | yes | results | yes | pending_manual_review |  |
| feedback-006 | current policies emissions for EU | data_query |  | Emissions\|CO2 | EU | Current Policies |  | yes | results | yes | pending_manual_review |  |
| feedback-007 | show emissions in Greece | data_query |  | Emissions\|CO2 | GREECE |  |  | yes | results | yes | pending_manual_review |  |
| feedback-008 | display co2 emission trend for China | data_query |  | Emissions\|CO2 | CHN |  |  | yes | results | yes | pending_manual_review |  |
| feedback-009 | show greenhouse gases for World | data_query |  | Emissions\|GHG | World |  |  | yes | results | yes | pending_manual_review |  |
| feedback-010 | current policy scenario emissions for EU | data_query |  | Emissions\|CO2 | EU | Current Policies |  | yes | results | yes | pending_manual_review |  |
| feedback-011 | greenhouse gas pathways by country | data_query |  | Emissions\|GHG | SE |  |  | yes | results | yes | pending_manual_review |  |
| feedback-012 | plot wind capacity for EU under Baseline | data_plotting |  | Capacity\|Electricity\|Wind | EU | Baseline |  | yes | results | yes | pending_manual_review |  |
| feedback-013 | plot solar capacity for EU under Baseline | data_plotting |  | Capacity\|Electricity\|Solar | EU | Baseline |  | yes | results | yes | pending_manual_review |  |
| feedback-014 | plot Emissions\|CO2 for World under Policy | data_plotting |  | Emissions\|CO2 | World | Policy |  | yes | results | yes | pending_manual_review |  |
| feedback-015 | show Emissions\|CO2 for World under Policy | data_query |  | Emissions\|CO2 | World | Policy |  | yes | results | yes | pending_manual_review |  |
| feedback-016 | energy in buildings for EU | data_query |  | Benchmarking\|Buildings\|Energy per capita | EU |  |  | yes | results | yes | pending_manual_review |  |
| feedback-017 | show greenhouse gas emissions for World | data_query |  | Emissions\|GHG | World |  |  | yes | results | yes | pending_manual_review |  |
| feedback-018 | plot photovoltaic capacity for Greece | data_plotting |  | Capacity\|Electricity\|Solar | GREECE |  |  | yes | results | yes | pending_manual_review |  |
| feedback-019 | show current policy emissions for EU | data_query |  | Emissions\|CO2 | EU | Current Policies |  | yes | results | yes | pending_manual_review |  |
| feedback-020 | show solar photovoltaic capacity for Greece | data_query |  | Capacity\|Electricity\|Solar | GREECE |  |  | yes | results | yes | pending_manual_review |  |
| feedback-021 | buildings energy demand for EU | data_query |  | IDA\|Emissions\|CO2\|Energy\|Demand\|Buildings\|CO2 intensity | EU |  |  | yes | results | yes | pending_manual_review |  |
| feedback-022 | CO2 emissions for EU28 under WWH | data_query |  | Emissions\|CO2 | EU28 | WWH |  | yes | results | yes | pending_manual_review |  |
| feedback-023 | CO2 emissions for World using REMIND | data_query |  | Emissions\|CO2 | World |  | REMIND | yes | results | yes | pending_manual_review |  |
| feedback-024 | compare solar and wind capacity | data_plotting |  | Capacity\|Electricity\|Wind |  |  |  | yes | results | yes | pending_manual_review |  |
| feedback-025 | nationally determined contributions emissions | data_query |  | Emissions\|CO2 |  | NDC |  | yes | results | yes | pending_manual_review |  |
| feedback-026 | NDC emissions for World | data_query |  | Emissions\|CO2 | World | NDC |  | yes | ndc_aspects | yes | pending_manual_review |  |
| feedback-027 | plot co2 emission trend for China | data_plotting |  | Emissions\|CO2 | CHN |  |  | yes | results | yes | pending_manual_review |  |
| feedback-028 | NDC impacts for transport and buildings | data_query |  | Energy Service\|Transportation\|Freight |  | NDC |  | yes | ndc_aspects | yes | pending_manual_review |  |
| feedback-029 | compare wind power and solar PV | data_plotting |  | Capacity\|Electricity\|Wind |  |  |  | yes | results | yes | pending_manual_review |  |
| feedback-030 | compare solar and wind electricity capacity | data_plotting |  | Capacity\|Electricity\|Wind |  |  |  | yes | results | yes | pending_manual_review |  |
| feedback-031 | plot CO2 emissions for China | data_plotting |  | Emissions\|CO2 | CHN |  |  | yes | results | yes | pending_manual_review |  |
| feedback-032 | tell me about WITCH model | model_explanation |  |  |  |  | WITCH | no | models | yes | pending_manual_review |  |
| feedback-033 | tell me about REMIND model | model_explanation |  |  |  |  | REMIND | no | models | yes | pending_manual_review |  |
| feedback-034 | NDC buildings results | general_qa |  |  |  |  |  | no | ndc_aspects | yes | pending_manual_review |  |
| feedback-035 | buildings transformation results | general_qa |  |  |  |  |  | no | results | yes | pending_manual_review |  |
| feedback-036 | transportation transformation workspace | general_qa |  |  |  |  |  | no | results | yes | pending_manual_review |  |
| feedback-037 | AFOLU land use results | general_qa |  |  |  |  |  | no | results | yes | pending_manual_review |  |
