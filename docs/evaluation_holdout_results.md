# Evaluation Results

Status: pending live/manual review

This report is generated from `eval_holdout_queries.csv`. It records the expected coverage set and can be extended to compare live chatbot outputs.

## Summary

- Total evaluation queries: 60
- Expected `data_plotting` queries: 7
- Expected `data_query` queries: 29
- Expected `general_qa` queries: 20
- Expected `model_explanation` queries: 4

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
| h001 | show greenhouse gas emissions for World | data_query |  | Emissions\|GHG | World |  |  | yes | results | yes | pending_manual_review |  |
| h002 | plot CO2 emissions for China | data_plotting |  | Emissions\|CO2 | China |  |  | yes | results | yes | pending_manual_review |  |
| h003 | electricity | data_query |  |  |  |  |  | yes | results | yes | pending_manual_review |  |
| h004 | oil | data_query |  |  |  |  |  | yes | results | yes | pending_manual_review |  |
| h005 | emissions | data_query |  |  |  |  |  | yes | results | yes | pending_manual_review |  |
| h006 | policy | data_query |  |  |  |  |  | yes | results | yes | pending_manual_review |  |
| h007 | NDC | data_query |  |  |  |  |  | yes | ndc_aspects | yes | pending_manual_review |  |
| h008 | show current policy emissions for EU | data_query |  | Emissions\|CO2 | EU | Current Policies |  | yes | results | yes | pending_manual_review |  |
| h009 | what scenarios exist for solar capacity | data_query |  | Capacity\|Electricity\|Solar |  |  |  | no | results | yes | pending_manual_review |  |
| h010 | plot wind capacity for India | data_plotting |  | Capacity\|Electricity\|Wind | India |  |  | yes | results | yes | pending_manual_review |  |
| h011 | show methane emissions for World | data_query |  |  | World |  |  | yes | results | yes | pending_manual_review |  |
| h012 | compare solar and wind electricity capacity | data_plotting |  | Capacity\|Electricity\|Solar |  |  |  | yes | results | yes | pending_manual_review |  |
| h013 | what is the WITCH model | model_explanation |  |  |  |  | WITCH | no | models | yes | pending_manual_review |  |
| h014 | explain MESSAGEix assumptions | model_explanation |  |  |  |  | MESSAGEix-GLOBIOM | no | models | yes | pending_manual_review |  |
| h015 | tell me about GCAM-PR 7 | model_explanation |  |  |  |  | GCAM-PR | no | models | yes | pending_manual_review |  |
| h016 | what are REMIND CCS assumptions | model_explanation |  |  |  |  | REMIND | no | models | yes | pending_manual_review |  |
| h017 | which models are available | data_query |  |  |  |  |  | no | models | yes | pending_manual_review |  |
| h018 | list all scenarios | data_query |  |  |  |  |  | no | results | yes | pending_manual_review |  |
| h019 | list all regions | data_query |  |  |  |  |  | no | results | yes | pending_manual_review |  |
| h020 | list variables related to electricity | data_query |  | Electricity |  |  |  | no | results | yes | pending_manual_review |  |
| h021 | show GDP for World | data_query |  | GDP\|MER | World |  |  | no | results | yes | pending_manual_review |  |
| h022 | plot GDP for EU | data_plotting |  | GDP\|MER | EU |  |  | yes | results | yes | pending_manual_review |  |
| h023 | show solar photovoltaic capacity for Greece | data_query |  | Capacity\|Electricity\|Solar | Greece |  |  | yes | results | yes | pending_manual_review |  |
| h024 | plot renewable electricity for Europe | data_plotting |  |  | EU |  |  | yes | results | yes | pending_manual_review |  |
| h025 | show final energy in transport for EU | data_query |  |  | EU |  |  | no | transport | yes | pending_manual_review |  |
| h026 | industry emissions in Europe | data_query |  |  | EU |  |  | yes | results | yes | pending_manual_review |  |
| h027 | buildings energy demand for EU | data_query |  |  | EU |  |  | yes | buildings | yes | pending_manual_review |  |
| h028 | AFOLU land use results | general_qa |  |  |  |  |  | no | afolu | yes | pending_manual_review |  |
| h029 | transportation transformation workspace | general_qa |  |  |  |  |  | no | transport | yes | pending_manual_review |  |
| h030 | buildings transformation results | general_qa |  |  |  |  |  | no | buildings | yes | pending_manual_review |  |
| h031 | where can I find Aqueduct | general_qa |  |  |  |  |  | no | application_library | yes | pending_manual_review |  |
| h032 | open Climate Watch | general_qa |  |  |  |  |  | no | application_library | yes | pending_manual_review |  |
| h033 | CDP Open Data Portal | general_qa |  |  |  |  |  | no | application_library | yes | pending_manual_review |  |
| h034 | policy catalogue explorer | general_qa |  |  |  |  |  | no | data_stories | yes | pending_manual_review |  |
| h035 | recovery policy database | general_qa |  |  |  |  |  | no | data_stories | yes | pending_manual_review |  |
| h036 | technology inventories data story | general_qa |  |  |  |  |  | no | data_stories | yes | pending_manual_review |  |
| h037 | barriers and enablers | general_qa |  |  |  |  |  | no | data_stories | yes | pending_manual_review |  |
| h038 | scenario metadata data story | general_qa |  |  |  |  |  | no | data_stories | yes | pending_manual_review |  |
| h039 | Fit for 55 results | general_qa |  |  | EU |  |  | no | iam_compact | yes | pending_manual_review |  |
| h040 | cost of capital pathways | general_qa |  |  |  |  |  | no | iam_compact | yes | pending_manual_review |  |
| h041 | technology constrained net zero pathways | general_qa |  |  |  |  |  | no | iam_compact | yes | pending_manual_review |  |
| h042 | global impacts of NDCs | general_qa |  |  |  |  |  | no | ndc_aspects | yes | pending_manual_review |  |
| h043 | NDC transport results | general_qa |  |  |  |  |  | no | ndc_aspects | yes | pending_manual_review |  |
| h044 | NDC buildings results | general_qa |  |  |  |  |  | no | ndc_aspects | yes | pending_manual_review |  |
| h045 | show Emissions\|CO2 for World under Baseline in 2050 | data_query |  | Emissions\|CO2 | World | Baseline |  | no | results | yes | pending_manual_review |  |
| h046 | show Emissions\|CO2 for World under Baseline after 2030 | data_query |  | Emissions\|CO2 | World | Baseline |  | no | results | yes | pending_manual_review |  |
| h047 | latest available CO2 emissions for World | data_query |  | Emissions\|CO2 | World |  |  | no | results | yes | pending_manual_review |  |
| h048 | plot Emissions\|CO2 for World under Baseline | data_plotting |  | Emissions\|CO2 | World | Baseline |  | no | results | yes | pending_manual_review |  |
| h049 | CO2 emissions for EU28 under WWH | data_query |  | Emissions\|CO2 | EU28 | WWH |  | yes | results | yes | pending_manual_review |  |
| h050 | CO2 emissions for World using REMIND | data_query |  | Emissions\|CO2 | World |  | REMIND | yes | results | yes | pending_manual_review |  |
| h051 | show data using GCAM | data_query |  |  |  |  | GCAM | yes | results | yes | pending_manual_review |  |
| h052 | compare GCAM and MESSAGE for emissions | data_plotting |  | Emissions\|CO2 |  |  |  | yes | results | yes | pending_manual_review |  |
| h053 | what countries are included | data_query |  |  |  |  |  | no | results | yes | pending_manual_review |  |
| h054 | what variables can you plot | data_query |  |  |  |  |  | no | results | yes | pending_manual_review |  |
| h055 | show all data categories | data_query |  |  |  |  |  | no | results | yes | pending_manual_review |  |
| h056 | custom analysis service | general_qa |  |  |  |  |  | no | analysis | yes | pending_manual_review |  |
| h057 | contact IAM PARIS team | general_qa |  |  |  |  |  | no | contact | yes | pending_manual_review |  |
| h058 | how can I browse online models | general_qa |  |  |  |  |  | no | application_library | yes | pending_manual_review |  |
| h059 | show electricity for India | data_query |  |  | India |  |  | no | results | yes | pending_manual_review |  |
| h060 | show oil demand for EU | data_query |  |  | EU |  |  | no | results | yes | pending_manual_review |  |
