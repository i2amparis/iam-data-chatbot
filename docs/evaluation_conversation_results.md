# Conversation Evaluation Results

Status: pending live/manual review

This report is generated from `eval_conversations.json` and checks multi-turn session behavior.

## Summary

- Total conversations: 10
- Total turns: 28

## Conversation Set

| Conversation | Turn | Query | Expected route | Actual route | Variable | Region | Scenario | Model | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| conv-001: CO2 Europe Follow-Ups | 1 | show me carbon dioxide emissions for Europe | data_query |  | Emissions\|CO2 | EU |  |  | pending_manual_review | pending_manual_review |
| conv-001: CO2 Europe Follow-Ups | 2 | same for China | data_query |  | Emissions\|CO2 | China |  |  | pending_manual_review | pending_manual_review |
| conv-001: CO2 Europe Follow-Ups | 3 | what about 2050 | data_query |  | Emissions\|CO2 | China |  |  | pending_manual_review | pending_manual_review |
| conv-001: CO2 Europe Follow-Ups | 4 | plot it | data_plotting |  | Emissions\|CO2 | China |  |  | pending_manual_review | pending_manual_review |
| conv-002: Clarify Electricity Variable Then Plot | 1 | electricity for India | data_query |  |  | India |  |  | pending_manual_review | pending_manual_review |
| conv-002: Clarify Electricity Variable Then Plot | 2 | 1 | data_query |  | Secondary Energy\|Electricity | India |  |  | pending_manual_review | pending_manual_review |
| conv-002: Clarify Electricity Variable Then Plot | 3 | plot it | data_plotting |  | Secondary Energy\|Electricity | India |  |  | pending_manual_review | pending_manual_review |
| conv-003: Baseline Year Follow-Ups | 1 | show Emissions\|CO2 for World under Baseline in 2030 | data_query |  | Emissions\|CO2 | World | Baseline |  | pending_manual_review | pending_manual_review |
| conv-003: Baseline Year Follow-Ups | 2 | after 2030 | data_query |  | Emissions\|CO2 | World | Baseline |  | pending_manual_review | pending_manual_review |
| conv-003: Baseline Year Follow-Ups | 3 | compare with current policy | data_plotting |  | Emissions\|CO2 | World | Current Policies |  | pending_manual_review | pending_manual_review |
| conv-004: Application Library Grounded Links | 1 | open Aqueduct details | general_qa |  |  |  |  |  | pending_manual_review | pending_manual_review |
| conv-004: Application Library Grounded Links | 2 | where can I find Climate Watch | general_qa |  |  |  |  |  | pending_manual_review | pending_manual_review |
| conv-005: IAM COMPACT Workspaces | 1 | Fit-for-55 EU net zero results | general_qa |  |  | EU |  |  | pending_manual_review | pending_manual_review |
| conv-005: IAM COMPACT Workspaces | 2 | cost of capital mitigation pathways | general_qa |  |  |  |  |  | pending_manual_review | pending_manual_review |
| conv-005: IAM COMPACT Workspaces | 3 | technology constrained pathways to net zero | general_qa |  |  |  |  |  | pending_manual_review | pending_manual_review |
| conv-006: NDC ASPECTS Workspaces | 1 | global impacts of NDCs | general_qa |  |  |  |  |  | pending_manual_review | pending_manual_review |
| conv-006: NDC ASPECTS Workspaces | 2 | NDC ASPECTS transport results | general_qa |  |  |  |  |  | pending_manual_review | pending_manual_review |
| conv-006: NDC ASPECTS Workspaces | 3 | AFOLU land use results | general_qa |  |  |  |  |  | pending_manual_review | pending_manual_review |
| conv-007: Model Metadata Then Data | 1 | tell me about GCAM model assumptions | model_explanation |  |  |  |  | GCAM | pending_manual_review | pending_manual_review |
| conv-007: Model Metadata Then Data | 2 | show data using GCAM | data_query |  |  |  |  | GCAM | pending_manual_review | pending_manual_review |
| conv-007: Model Metadata Then Data | 3 | show CO2 emissions for World | data_query |  | Emissions\|CO2 | World |  |  | pending_manual_review | pending_manual_review |
| conv-008: No Data Recovery | 1 | show Emissions\|CO2 for World under Policy | data_query |  | Emissions\|CO2 | World | Policy |  | pending_manual_review | pending_manual_review |
| conv-008: No Data Recovery | 2 | use the first scenario | data_query |  | Emissions\|CO2 | World |  |  | pending_manual_review | pending_manual_review |
| conv-009: Stale Clarification Reset | 1 | oil demand for EU | data_query |  |  | EU |  |  | pending_manual_review | pending_manual_review |
| conv-009: Stale Clarification Reset | 2 | which models are available | data_query |  |  |  |  |  | pending_manual_review | pending_manual_review |
| conv-009: Stale Clarification Reset | 3 | 1 | data_query |  |  |  |  |  | pending_manual_review | pending_manual_review |
| conv-010: Contact And Analysis Links | 1 | custom analysis service | general_qa |  |  |  |  |  | pending_manual_review | pending_manual_review |
| conv-010: Contact And Analysis Links | 2 | contact IAM PARIS team | general_qa |  |  |  |  |  | pending_manual_review | pending_manual_review |
