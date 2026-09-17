# Conversation Evaluation Results

Status: live evaluation complete

This report is generated from `eval_conversations.json` and checks multi-turn session behavior.

## Summary

- Total conversations: 10
- Total turns: 28
- Live `pass` conversations: 10/10
- Live `pass` turns: 28/28

## Live Score Summary

- `correct_action`: 3/28
- `correct_chart_type`: 1/28
- `correct_comparison`: 1/28
- `correct_end_year`: 5/28
- `correct_model`: 28/28
- `correct_plot`: 3/28
- `correct_region`: 28/28
- `correct_route`: 28/28
- `correct_scenario`: 28/28
- `correct_start_year`: 5/28
- `correct_variable`: 28/28
- `no_hallucinated_data`: 28/28
- `session_continuity`: 28/28
- `useful_clarification`: 28/28
- `useful_link`: 28/28

## Conversation Set

| Conversation | Turn | Query | Expected route | Actual route | Variable | Region | Scenario | Model | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| conv-001: CO2 Europe Follow-Ups | 1 | show me carbon dioxide emissions for Europe | data_query | data_query | Emissions\|CO2 | EU |  |  | pass | ### Emissions\|CO2 in EU  Scope: scenario `multiple`, model `multiple`, years `2005-2100` Unit: `Mt CO2/yr`  Answer: Summ |
| conv-001: CO2 Europe Follow-Ups | 2 | same for China | data_query | data_query | Emissions\|CO2 | China |  |  | pass | ### Emissions\|CO2 in CHN  Scope: scenario `multiple`, model `multiple`, years `2005-2100` Unit: `Mt CO2/yr`  Answer: Sum |
| conv-001: CO2 Europe Follow-Ups | 3 | what about 2050 | data_query | data_query | Emissions\|CO2 | China |  |  | pass | ### Emissions\|CO2 in CHN  Scope: scenario `multiple`, model `multiple`, years `2050` Unit: `Mt CO2/yr`  Answer: Summary: |
| conv-001: CO2 Europe Follow-Ups | 4 | plot it | data_plotting | data_plotting | Emissions\|CO2 | China |  |  | pass | Showing CO2 Emissions in CHN across available scenarios (2050). Resolved variable: `Emissions\|CO2`. Showing 8 of 49 avai |
| conv-002: Clarify Electricity Variable Then Plot | 1 | electricity for India | data_query | data_query |  | India |  |  | pass | I found several possible variables, but none has enough evidence for automatic selection. Choose the variable: 1. `Secon |
| conv-002: Clarify Electricity Variable Then Plot | 2 | 1 | data_query | data_query | Secondary Energy\|Electricity | India |  |  | pass | ### Secondary Energy\|Electricity in IND  Scope: scenario `multiple`, model `multiple`, years `2005-2100` Unit: `EJ/yr`   |
| conv-002: Clarify Electricity Variable Then Plot | 3 | plot it | data_plotting | data_plotting | Secondary Energy\|Electricity | India |  |  | pass | Showing Electricity Generation in IND across available scenarios. Resolved variable: `Secondary Energy\|Electricity`. Sho |
| conv-003: Baseline Year Follow-Ups | 1 | show Emissions\|CO2 for World under Baseline in 2030 | data_query | data_query | Emissions\|CO2 | World | Baseline |  | pass | ### Emissions\|CO2 in World  Scope: scenario `multiple`, model `multiple`, years `2030` Unit: `Mt CO2/yr`  Answer: Summar |
| conv-003: Baseline Year Follow-Ups | 2 | after 2030 | data_query | data_query | Emissions\|CO2 | World | Baseline |  | pass | ### Emissions\|CO2 in World  Scope: scenario `multiple`, model `multiple`, years `2031-latest available` Unit: `Mt CO2/yr |
| conv-003: Baseline Year Follow-Ups | 3 | compare with current policy | data_plotting | data_plotting | Emissions\|CO2 | World | Current Policies |  | pass | Showing CO2 Emissions in World for selected scenarios `Baseline`, `baseline`, `Baseline\|Harmonized`, `baseline\|Harmonize |
| conv-004: Application Library Grounded Links | 1 | open Aqueduct details | general_qa | general_qa |  |  |  |  | pass | Use these IAM PARIS links for this request: [Aqueduct](https://iamparis.eu/application_library/474). |
| conv-004: Application Library Grounded Links | 2 | where can I find Climate Watch | general_qa | general_qa |  |  |  |  | pass | Use these IAM PARIS links for this request: [Climate Watch](https://iamparis.eu/application_library).  If a direct detai |
| conv-005: IAM COMPACT Workspaces | 1 | Fit-for-55 EU net zero results | general_qa | general_qa |  |  |  |  | pass | Use these IAM PARIS links for this request: [IAM COMPACT](https://iamparis.eu/results/iam-compact). |
| conv-005: IAM COMPACT Workspaces | 2 | cost of capital mitigation pathways | general_qa | general_qa |  |  |  |  | pass | Use these IAM PARIS links for this request: [Cost of capital evolution effects on regional mitigation pathways](https:// |
| conv-005: IAM COMPACT Workspaces | 3 | technology constrained pathways to net zero | general_qa | general_qa |  |  |  |  | pass | Use these IAM PARIS links for this request: [Tech-constrained EU pathways to net zero](https://iamparis.eu/results/iam-c |
| conv-006: NDC ASPECTS Workspaces | 1 | global impacts of NDCs | general_qa | general_qa |  |  |  |  | pass | Use these IAM PARIS links for this request: [Global impacts of NDCs](https://iamparis.eu/results/ndc-aspects/global-impa |
| conv-006: NDC ASPECTS Workspaces | 2 | NDC ASPECTS transport results | general_qa | general_qa |  |  |  |  | pass | Use these IAM PARIS links for this request: [NDC ASPECTS](https://iamparis.eu/results/ndc-aspects). |
| conv-006: NDC ASPECTS Workspaces | 3 | AFOLU land use results | general_qa | general_qa |  |  |  |  | pass | Use these IAM PARIS links for this request: [AFOLU transformation](https://iamparis.eu/results/ndc-aspects/afolu-transfo |
| conv-007: Model Metadata Then Data | 1 | tell me about GCAM model assumptions | model_explanation | model_explanation |  |  |  | GCAM | pass | ### GCAM  Description: GCAM is an integrated assessment model that represents interactions among energy, economy, land,  |
| conv-007: Model Metadata Then Data | 2 | show data using GCAM | data_query | data_query |  |  |  | GCAM | pass | I need one more detail. Which variable should I use? |
| conv-007: Model Metadata Then Data | 3 | show CO2 emissions for World | data_query | data_query | Emissions\|CO2 | World |  |  | pass | ### Emissions\|CO2 in World  Scope: scenario `multiple`, model `multiple`, years `2005-2100` Unit: `Mt CO2/yr`  Answer: S |
| conv-008: No Data Recovery | 1 | show Emissions\|CO2 for World under Policy | data_query | data_query | Emissions\|CO2 | World | Policy |  | pass | I could not find data for `Emissions\|CO2` in `World` under `Policy`.  Reason: the exact scenario combination is not avai |
| conv-008: No Data Recovery | 2 | use the first scenario | data_query | data_query | Emissions\|CO2 | World |  |  | pass | ### Emissions\|CO2 in World  Scope: scenario `multiple`, model `multiple`, years `2005-2100` Unit: `Mt CO2/yr`  Answer: S |
| conv-009: Stale Clarification Reset | 1 | oil demand for EU | data_query | data_query |  | EU |  |  | pass | I found several possible variables, but none has enough evidence for automatic selection. Choose the variable: 1. `Final |
| conv-009: Stale Clarification Reset | 2 | which models are available | data_query | data_query |  |  |  |  | pass | There are 73 models available. Examples: AIM/Enduse India 3.3, ALADIN, ATOM, CHANCE, CLEWs, CLEWs - Ethiopia, CLEWs - Ke |
| conv-009: Stale Clarification Reset | 3 | 1 | data_query | data_query |  |  |  |  | pass | I don't have an active numbered choice right now. Reply with a full variable, region, or scenario so I can continue. |
| conv-010: Contact And Analysis Links | 1 | custom analysis service | general_qa | general_qa |  |  |  |  | pass | Use these IAM PARIS links for this request: [Configurable Analysis](https://iamparis.eu/analysis). |
| conv-010: Contact And Analysis Links | 2 | contact IAM PARIS team | general_qa | general_qa |  |  |  |  | pass | Use these IAM PARIS links for this request: [Contact](https://iamparis.eu/contact). |
