# Evaluation Results

Status: pending live/manual review

This report is generated from `eval_queries.csv`. It records the expected coverage set and can be extended to compare live chatbot outputs.

## Summary

- Total evaluation queries: 150
- Expected `data_plotting` queries: 18
- Expected `data_query` queries: 86
- Expected `general_qa` queries: 37
- Expected `model_explanation` queries: 9

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
| 1 | show me carbon dioxide emissions for Europe | data_query |  | Emissions\|CO2 | EU |  |  | yes | results | yes | pending_manual_review |  |
| 2 | plot photovoltaic capacity for Greece | data_plotting |  | Capacity\|Electricity\|Solar | Greece |  |  | yes | results | yes | pending_manual_review |  |
| 3 | which models are available | data_query |  |  |  |  |  | no | models | yes | pending_manual_review |  |
| 4 | compare wind power and solar PV | data_plotting |  | Capacity\|Electricity\|Wind |  |  |  | yes | results | yes | pending_manual_review |  |
| 5 | tell me about GCAM model assumptions | model_explanation |  |  |  |  | GCAM | no | models | yes | pending_manual_review |  |
| 6 | what are the assumptions in the GCAM model | model_explanation |  |  |  |  | GCAM | no | models | yes | pending_manual_review |  |
| 7 | information on gcam | data_query |  |  |  |  | GCAM | no | models | yes | pending_manual_review |  |
| 8 | list variables | data_query |  |  |  |  |  | no | results | yes | pending_manual_review |  |
| 9 | list scenarios | data_query |  |  |  |  |  | no | results | yes | pending_manual_review |  |
| 10 | list regions | data_query |  |  |  |  |  | no | results | yes | pending_manual_review |  |
| 11 | energy in buildings for EU | data_query |  |  | EU |  |  | yes | buildings | yes | pending_manual_review |  |
| 12 | carbon from transport in Europe | data_query |  |  | EU |  |  | yes | transport | yes | pending_manual_review |  |
| 13 | AFOLU agriculture land forestry transformation results | general_qa |  |  |  |  |  | no | afolu | yes | pending_manual_review |  |
| 14 | open the Aqueduct raw data application | general_qa |  |  |  |  |  | no | application_library | yes | pending_manual_review |  |
| 15 | where can I find Climate Watch | general_qa |  |  |  |  |  | no | application_library | yes | pending_manual_review |  |
| 16 | policy catalogue climate policies | general_qa |  |  |  |  |  | no | data_stories | yes | pending_manual_review |  |
| 17 | Fit-for-55 EU net zero results | general_qa |  |  | EU |  |  | no | iam_compact | yes | pending_manual_review |  |
| 18 | NDC impacts for transport and buildings | data_query |  |  |  |  |  | yes | ndc_aspects | yes | pending_manual_review |  |
| 19 | show me data for EU | data_query |  |  | EU |  |  | yes | results | yes | pending_manual_review |  |
| 20 | oil demand for EU | data_query |  |  | EU |  |  | yes | results | yes | pending_manual_review |  |
| 21 | electricity for India | data_query |  |  | India |  |  | yes | results | yes | pending_manual_review |  |
| 22 | show Emissions\|CO2 for World under Policy | data_query |  | Emissions\|CO2 | World | Policy |  | yes | results | yes | pending_manual_review |  |
| 23 | plot Emissions\|CO2 for World under Policy | data_plotting |  | Emissions\|CO2 | World | Policy |  | yes | results | yes | pending_manual_review |  |
| 24 | show Emissions\|CO2 for World under Baseline in 2030 | data_query |  | Emissions\|CO2 | World | Baseline |  | no | results | yes | pending_manual_review |  |
| 25 | show Emissions\|CO2 for World under Baseline by 2050 | data_query |  | Emissions\|CO2 | World | Baseline |  | no | results | yes | pending_manual_review |  |
| 26 | show Emissions\|CO2 for World under Baseline after 2030 | data_query |  | Emissions\|CO2 | World | Baseline |  | no | results | yes | pending_manual_review |  |
| 27 | show the latest available year for Emissions\|CO2 in World | data_query |  | Emissions\|CO2 | World |  |  | no | results | yes | pending_manual_review |  |
| 28 | plot it | data_plotting |  |  |  |  |  | yes | results | yes | pending_manual_review |  |
| 29 | same for China | data_query |  |  | China |  |  | yes | results | yes | pending_manual_review |  |
| 30 | what about 2050 | data_query |  |  |  |  |  | yes | results | yes | pending_manual_review |  |
| 31 | compare with baseline | data_plotting |  |  |  | Baseline |  | yes | results | yes | pending_manual_review |  |
| 32 | show all scenarios | data_query |  |  |  |  |  | yes | results | yes | pending_manual_review |  |
| 33 | use the first scenario | data_query |  |  |  |  |  | yes | results | yes | pending_manual_review |  |
| 34 | yes | data_query |  |  |  |  |  | yes | results | yes | pending_manual_review |  |
| 35 | no | data_query |  |  |  |  |  | yes |  | yes | pending_manual_review |  |
| 36 | 1 | data_query |  |  |  |  |  | yes | results | yes | pending_manual_review |  |
| 37 | 2 | data_query |  |  |  |  |  | yes | results | yes | pending_manual_review |  |
| 38 | what kinds of data are included | data_query |  |  |  |  |  | no | results | yes | pending_manual_review |  |
| 39 | help me find data | data_query |  |  |  |  |  | yes | results | yes | pending_manual_review |  |
| 40 | compare GCAM and MESSAGE for CO2 emissions | data_plotting |  | Emissions\|CO2 |  |  |  | yes | models | yes | pending_manual_review |  |
| 41 | show data using gcampr | data_query |  |  |  |  | GCAM-PR 7.0 | yes | models | yes | pending_manual_review |  |
| 42 | show data with message ix | data_query |  |  |  |  | MESSAGEix-GLOBIOM 2.0 | yes | models | yes | pending_manual_review |  |
| 43 | plot solar capacity for EU under Baseline | data_plotting |  | Capacity\|Electricity\|Solar | EU | Baseline |  | yes | results | yes | pending_manual_review |  |
| 44 | plot wind capacity for EU under Baseline | data_plotting |  | Capacity\|Electricity\|Wind | EU | Baseline |  | yes | results | yes | pending_manual_review |  |
| 45 | gross domestic product for World | data_query |  | GDP\|MER | World |  |  | no | results | yes | pending_manual_review |  |
| 46 | greenhouse gas pathways by country | data_query |  | Emissions\|GHG |  |  |  | yes | results | yes | pending_manual_review |  |
| 47 | current policy scenario emissions for EU | data_query |  | Emissions\|CO2 | EU | Current Policies |  | yes | results | yes | pending_manual_review |  |
| 48 | show all models | data_query |  |  |  |  |  | no | models | yes | pending_manual_review |  |
| 49 | show all variables | data_query |  |  |  |  |  | no | results | yes | pending_manual_review |  |
| 50 | custom analysis for decarbonisation options | general_qa |  |  |  |  |  | no | analysis | yes | pending_manual_review |  |
| 51 | carbon emissions in Europe | data_query |  | Emissions\|CO2 | EU |  |  | yes | results | yes | pending_manual_review |  |
| 52 | CO2 emissions for EU in 2050 | data_query |  | Emissions\|CO2 | EU |  |  | yes | results | yes | pending_manual_review |  |
| 53 | show greenhouse gases for World | data_query |  | Emissions\|GHG | World |  |  | yes | results | yes | pending_manual_review |  |
| 54 | methane emissions for Europe | data_query |  | Emissions\|CH4 | EU |  |  | yes | results | yes | pending_manual_review |  |
| 55 | nitrous oxide emissions for World | data_query |  | Emissions\|N2O | World |  |  | yes | results | yes | pending_manual_review |  |
| 56 | electricity generation for World | data_query |  | Secondary Energy\|Electricity | World |  |  | no | results | yes | pending_manual_review |  |
| 57 | plot electricity generation for World | data_plotting |  | Secondary Energy\|Electricity | World |  |  | yes | results | yes | pending_manual_review |  |
| 58 | wind power capacity in Europe | data_query |  | Capacity\|Electricity\|Wind | EU |  |  | no | results | yes | pending_manual_review |  |
| 59 | solar pv installed capacity in World | data_query |  | Capacity\|Electricity\|Solar | World |  |  | no | results | yes | pending_manual_review |  |
| 60 | plot solar pv installed capacity in World | data_plotting |  | Capacity\|Electricity\|Solar | World |  |  | yes | results | yes | pending_manual_review |  |
| 61 | show GDP for EU | data_query |  | GDP\|MER | EU |  |  | no | results | yes | pending_manual_review |  |
| 62 | gdp values for global economy | data_query |  | GDP\|MER | World |  |  | no | results | yes | pending_manual_review |  |
| 63 | show carbon dioxide emissions globally | data_query |  | Emissions\|CO2 | World |  |  | yes | results | yes | pending_manual_review |  |
| 64 | display co2 emission trend for China | data_query |  | Emissions\|CO2 | China |  |  | yes | results | yes | pending_manual_review |  |
| 65 | plot co2 emission trend for China | data_plotting |  | Emissions\|CO2 | China |  |  | yes | results | yes | pending_manual_review |  |
| 66 | show data for India emissions | data_query |  |  | India |  |  | yes | results | yes | pending_manual_review |  |
| 67 | show emissions in Greece | data_query |  |  | Greece |  |  | yes | results | yes | pending_manual_review |  |
| 68 | show emissions for United States | data_query |  |  |  |  |  | yes | results | yes | pending_manual_review |  |
| 69 | show data for CHN | data_query |  |  | China |  |  | yes | results | yes | pending_manual_review |  |
| 70 | show data for IND | data_query |  |  | India |  |  | yes | results | yes | pending_manual_review |  |
| 71 | current policies emissions for EU | data_query |  | Emissions\|CO2 | EU | Current Policies |  | yes | results | yes | pending_manual_review |  |
| 72 | current policy CO2 for Europe | data_query |  | Emissions\|CO2 | EU | Current Policies |  | yes | results | yes | pending_manual_review |  |
| 73 | baseline emissions for World | data_query |  | Emissions\|CO2 | World | Baseline |  | no | results | yes | pending_manual_review |  |
| 74 | business as usual emissions for World | data_query |  | Emissions\|CO2 | World | Baseline |  | yes | results | yes | pending_manual_review |  |
| 75 | BAU carbon emissions for World | data_query |  | Emissions\|CO2 | World | Baseline |  | yes | results | yes | pending_manual_review |  |
| 76 | NDC emissions for World | data_query |  | Emissions\|CO2 | World | NDC |  | yes | results | yes | pending_manual_review |  |
| 77 | nationally determined contributions emissions | data_query |  | Emissions\|CO2 |  | NDC |  | yes | results | yes | pending_manual_review |  |
| 78 | net zero emissions for Europe | data_query |  | Emissions\|CO2 | EU |  |  | yes | iam_compact | yes | pending_manual_review |  |
| 79 | show net-zero target results | data_query |  |  |  |  |  | yes | iam_compact | yes | pending_manual_review |  |
| 80 | Fit for 55 policy questions | general_qa |  |  |  |  |  | no | iam_compact | yes | pending_manual_review |  |
| 81 | IAM COMPACT renewable energy metrics | general_qa |  |  |  |  |  | no | iam_compact | yes | pending_manual_review |  |
| 82 | post Glasgow climate results | general_qa |  |  |  |  |  | no | iam_compact | yes | pending_manual_review |  |
| 83 | steel relocation EU results | general_qa |  |  |  |  |  | no | iam_compact | yes | pending_manual_review |  |
| 84 | cost of capital mitigation pathways | general_qa |  |  |  |  |  | no | iam_compact | yes | pending_manual_review |  |
| 85 | behavioural change climate policy | general_qa |  |  |  |  |  | no | iam_compact | yes | pending_manual_review |  |
| 86 | technology constrained pathways to net zero | general_qa |  |  |  |  |  | no | iam_compact | yes | pending_manual_review |  |
| 87 | NDC ASPECTS transport results | general_qa |  |  |  |  |  | no | ndc_aspects | yes | pending_manual_review |  |
| 88 | NDC ASPECTS buildings results | general_qa |  |  |  |  |  | no | ndc_aspects | yes | pending_manual_review |  |
| 89 | global impacts of NDCs | general_qa |  |  |  |  |  | no | ndc_aspects | yes | pending_manual_review |  |
| 90 | long term targets scenario results | general_qa |  |  |  |  |  | no | results | yes | pending_manual_review |  |
| 91 | where is the buildings transformation workspace | general_qa |  |  |  |  |  | no | buildings | yes | pending_manual_review |  |
| 92 | open transportation transformation results | general_qa |  |  |  |  |  | no | transport | yes | pending_manual_review |  |
| 93 | industrial transformation workspace | general_qa |  |  |  |  |  | no | results | yes | pending_manual_review |  |
| 94 | AFOLU land use results | general_qa |  |  |  |  |  | no | afolu | yes | pending_manual_review |  |
| 95 | agriculture forestry land results | general_qa |  |  |  |  |  | no | afolu | yes | pending_manual_review |  |
| 96 | open application library | general_qa |  |  |  |  |  | no | application_library | yes | pending_manual_review |  |
| 97 | show me interactive maps in application library | general_qa |  |  |  |  |  | no | application_library | yes | pending_manual_review |  |
| 98 | where is the CDP Open Data Portal | general_qa |  |  |  |  |  | no | application_library | yes | pending_manual_review |  |
| 99 | find EDGAR in application library | general_qa |  |  |  |  |  | no | application_library | yes | pending_manual_review |  |
| 100 | open Climate Watch application | general_qa |  |  |  |  |  | no | application_library | yes | pending_manual_review |  |
| 101 | open Aqueduct details | general_qa |  |  |  |  |  | no | application_library | yes | pending_manual_review |  |
| 102 | policy catalog explorer | general_qa |  |  |  |  |  | no | data_stories | yes | pending_manual_review |  |
| 103 | recovery policy database | general_qa |  |  |  |  |  | no | data_stories | yes | pending_manual_review |  |
| 104 | circularity decarbonisation data story | general_qa |  |  |  |  |  | no | data_stories | yes | pending_manual_review |  |
| 105 | technology inventories data story | general_qa |  |  |  |  |  | no | data_stories | yes | pending_manual_review |  |
| 106 | barriers and enablers database | general_qa |  |  |  |  |  | no | data_stories | yes | pending_manual_review |  |
| 107 | scenario metadata data story | general_qa |  |  |  |  |  | no | data_stories | yes | pending_manual_review |  |
| 108 | custom analysis service | general_qa |  |  |  |  |  | no | analysis | yes | pending_manual_review |  |
| 109 | contact IAM PARIS team | general_qa |  |  |  |  |  | no | contact | yes | pending_manual_review |  |
| 110 | how can I request analysis support | general_qa |  |  |  |  |  | no | analysis | yes | pending_manual_review |  |
| 111 | tell me about MESSAGEix model | model_explanation |  |  |  |  | MESSAGEix-GLOBIOM 2.0 | no | models | yes | pending_manual_review |  |
| 112 | MESSAGE IX assumptions | model_explanation |  |  |  |  | MESSAGEix-GLOBIOM 2.0 | no | models | yes | pending_manual_review |  |
| 113 | tell me about REMIND model | model_explanation |  |  |  |  | REMIND | no | models | yes | pending_manual_review |  |
| 114 | tell me about WITCH model | model_explanation |  |  |  |  | WITCH | no | models | yes | pending_manual_review |  |
| 115 | tell me about PROMETHEUS | model_explanation |  |  |  |  | PROMETHEUS | no | models | yes | pending_manual_review |  |
| 116 | tell me about LEAP | model_explanation |  |  |  |  | LEAP | no | models | yes | pending_manual_review |  |
| 117 | tell me about GCAM PR | model_explanation |  |  |  |  | GCAM-PR 7.0 | no | models | yes | pending_manual_review |  |
| 118 | show data using GCAM | data_query |  |  |  |  | GCAM | yes | models | yes | pending_manual_review |  |
| 119 | show data using MESSAGEix | data_query |  |  |  |  | MESSAGEix-GLOBIOM 2.0 | yes | models | yes | pending_manual_review |  |
| 120 | show data using PROMETHEUS | data_query |  |  |  |  | PROMETHEUS | yes | models | yes | pending_manual_review |  |
| 121 | compare GCAM and WITCH emissions | data_plotting |  | Emissions\|CO2 |  |  |  | yes | models | yes | pending_manual_review |  |
| 122 | compare MESSAGE and REMIND carbon emissions | data_plotting |  | Emissions\|CO2 |  |  |  | yes |  | yes | pending_manual_review |  |
| 123 | compare solar and wind capacity | data_plotting |  | Capacity\|Electricity\|Wind |  |  |  | yes | results | yes | pending_manual_review |  |
| 124 | plot wind power in World | data_plotting |  | Secondary Energy\|Electricity\|Wind | World |  |  | yes | results | yes | pending_manual_review |  |
| 125 | plot methane emissions for World | data_plotting |  | Emissions\|CH4 | World |  |  | yes | results | yes | pending_manual_review |  |
| 126 | show latest CO2 for World | data_query |  | Emissions\|CO2 | World |  |  | no | results | yes | pending_manual_review |  |
| 127 | show latest available GDP for World | data_query |  | GDP\|MER | World |  |  | no | results | yes | pending_manual_review |  |
| 128 | show CO2 for World before 2030 | data_query |  | Emissions\|CO2 | World |  |  | yes | results | yes | pending_manual_review |  |
| 129 | show CO2 for World after 2030 | data_query |  | Emissions\|CO2 | World |  |  | yes | results | yes | pending_manual_review |  |
| 130 | show CO2 for World by 2050 | data_query |  | Emissions\|CO2 | World |  |  | no | results | yes | pending_manual_review |  |
| 131 | show CO2 for World from 2030 to 2050 | data_query |  | Emissions\|CO2 | World |  |  | yes | results | yes | pending_manual_review |  |
| 132 | show data for EU under Baseline | data_query |  |  | EU | Baseline |  | yes | results | yes | pending_manual_review |  |
| 133 | show GDP for World under Baseline | data_query |  | GDP\|MER | World | Baseline |  | yes | results | yes | pending_manual_review |  |
| 134 | show electricity for India | data_query |  |  | India |  |  | no | results | yes | pending_manual_review |  |
| 135 | show solar data for Greece | data_query |  |  | Greece |  |  | yes | results | yes | pending_manual_review |  |
| 136 | show wind data for Greece | data_query |  |  | Greece |  |  | yes | results | yes | pending_manual_review |  |
| 137 | help me choose a variable | data_query |  |  |  |  |  | yes | results | yes | pending_manual_review |  |
| 138 | which scenarios are available for EU | data_query |  |  | EU |  |  | no | results | yes | pending_manual_review |  |
| 139 | which regions are available for CO2 | data_query |  |  |  |  |  | no | results | yes | pending_manual_review |  |
| 140 | which models provide CO2 emissions | data_query |  | Emissions\|CO2 |  |  |  | no | models | yes | pending_manual_review |  |
| 141 | show all data categories | data_query |  |  |  |  |  | no | results | yes | pending_manual_review |  |
| 142 | what variables can I use | data_query |  |  |  |  |  | no | results | yes | pending_manual_review |  |
| 143 | what regions can I use | data_query |  |  |  |  |  | no | results | yes | pending_manual_review |  |
| 144 | what scenarios can I use | data_query |  |  |  |  |  | no | results | yes | pending_manual_review |  |
| 145 | what models can I use | data_query |  |  |  |  |  | no | models | yes | pending_manual_review |  |
| 146 | plot this | data_plotting |  |  |  |  |  | yes | results | yes | pending_manual_review |  |
| 147 | same for India | data_query |  |  | India |  |  | yes | results | yes | pending_manual_review |  |
| 148 | what about 2030 | data_query |  |  |  |  |  | yes | results | yes | pending_manual_review |  |
| 149 | compare to current policy | data_plotting |  |  |  | Current Policies |  | yes | results | yes | pending_manual_review |  |
| 150 | use the second scenario | data_query |  |  |  |  |  | yes | results | yes | pending_manual_review |  |
