# Qwen 200-query audit: full answers

Run: 5 November 2026. Local model: `qwen3:0.6b-q4_K_M` through Ollama.
Data: 631,201 cached IAM PARIS records; local knowledge index enabled.
Each conversation ID uses one session across its turns. The Qwen count is the
number of chat generations for that query; `0` means the answer came from
deterministic routing, metadata, glossary, or cached data.

<a id="q001"></a>

## 001. plot photovoltaic capacity for Greece

Category: charts and comparisons<br>
Route: `general_qa` (charts delegated to data explorer)<br>
Qwen generations: 0<br>
Time: 0.02 s<br>
Conversation: standalone

**Answer**

I do not generate figures in the chat. Please use the [IAM PARIS data explorer](https://iamparis.eu/results) to explore, filter, and plot the data. Ask about a specific study first if you would like its direct explorer link.

<a id="q002"></a>

## 002. compare wind power and solar PV

Category: data and scenarios<br>
Route: `data_query` (chat plotting disabled; return numeric table)<br>
Qwen generations: 1<br>
Time: 1.53 s<br>
Conversation: standalone

**Answer**

Please provide the variable and region for the numeric comparison.

<a id="q003"></a>

## 003. what are the assumptions in the GCAM model

Category: model descriptions<br>
Route: `model_explanation` (named-model metadata request)<br>
Qwen generations: 0<br>
Time: 0.07 s<br>
Conversation: standalone

**Answer**

### GCAM

Description:
GCAM is an integrated assessment model that represents interactions among energy, economy, land, water and climate systems. It is commonly used for scenario analysis of emissions, energy transitions and climate policy.

Developer:
Joint Global Change Research Institute

Model scope:
- energy
- economy
- land
- water
- climate

Useful for:
- emissions and energy pathway analysis
- climate policy scenarios
- land and resource interactions
- regional mitigation comparisons

Assumptions:
No explicit assumptions field is available in the local IAM PARIS model metadata. Use the selected scenario and IAM PARIS result metadata to interpret GCAM assumptions for a specific answer.

Interpretation notes:
- model assumptions vary by scenario and input dataset
- local IAM PARIS metadata may not include detailed assumptions

Related model documentation:
- [IAM PARIS model page](https://iamparis.eu/models/30)

<a id="q004"></a>

## 004. information on gcam

Category: data and scenarios<br>
Route: `data_query` (vague model information request)<br>
Qwen generations: 0<br>
Time: 0.52 s<br>
Conversation: standalone

**Answer**

### GCAM

Description:
GCAM is a dynamic-recursive model with technology-rich representations of the economy, energy sector, land use and water linked to a climate model that can be used to explore climate change mitigation policies including carbon taxes, carbon trading, regulations and accelerated deployment of energy technology. Regional population and labour productivity growth assumptions drive the energy, land-use and water systems employing numerous technology options to produce, transform and provide energy services, as well as to produce agriculture and forest products, and to determine land use and land use cover.

Related model documentation:
- [IAM PARIS model page](https://iamparis.eu/models/30)

Reply with a scenario, model, region, or year to narrow the answer.

<a id="q005"></a>

## 005. list variables

Category: catalogue and coverage<br>
Route: `data_query` (variable availability request)<br>
Qwen generations: 0<br>
Time: 1.21 s<br>
Conversation: standalone

**Answer**

I can work with these variables:
- Agricultural Demand
- Agricultural Demand|Crops
- Agricultural Demand|Crops|Energy
- Agricultural Demand|Crops|Feed
- Agricultural Demand|Crops|Food
- Agricultural Demand|Crops|Other
- Agricultural Demand|Livestock
- Agricultural Demand|Livestock|Food
- Agricultural Demand|Livestock|Other
- Agricultural Production
- Annual Refurbishment Rate|Commercial
- Annual Refurbishment Rate|Residential and 2034 more

Try queries like 'Capacity|Electricity|Solar|Utility for Greece' or 'show [variable name] for Greece'.

Showing 12 of 2046. Say `show all variables` if you need the full list.

<a id="q006"></a>

## 006. list scenarios

Category: data and scenarios<br>
Route: `data_query` (scenario availability request)<br>
Qwen generations: 0<br>
Time: 0.20 s<br>
Conversation: standalone

**Answer**

I found scenarios like 1.5 TECH, 1.5 °C DRI-trade, 1.5 °C Domestic, BAU, BAU_CL1, BAU_CL2, BAU_CL3 and BAU_Familiar and 109 more. Ask for a variable under any of these scenarios to see its values.

Showing 8 of 117. Say `show all scenarios` if you need the full list.

<a id="q007"></a>

## 007. list regions

Category: data and scenarios<br>
Route: `data_query` (region availability request)<br>
Qwen generations: 0<br>
Time: 0.21 s<br>
Conversation: standalone

**Answer**

I found regions like AFR, AGO, ARE, ARG, ASEAN, ASI, AT, ATE, AUS and AZE and 151 more. Ask for a variable in any of these regions to see its values.

Showing 10 of 161. Say `show all regions` if you need the full list.

<a id="q008"></a>

## 008. energy in buildings for EU

Category: data and scenarios<br>
Route: `data_query` (low confidence entity clarification)<br>
Qwen generations: 0<br>
Time: 0.28 s<br>
Conversation: standalone

**Answer**

I found several possible variables, but none has enough evidence for automatic selection. Choose the variable: 1. `Final Energy|Buildings|Residential` (final energy use, buildings) 2. `Final Energy|Buildings|Residential|Gases` (final energy use, buildings, gas) Reply with a number (1-2), or `yes` for option 1.

<a id="q009"></a>

## 009. carbon from transport in Europe

Category: data and scenarios<br>
Route: `data_query` (low confidence entity clarification)<br>
Qwen generations: 0<br>
Time: 0.26 s<br>
Conversation: standalone

**Answer**

I found several possible variables, but none has enough evidence for automatic selection. Choose the variable: 1. `Revenue|Government|Tax|Carbon|Demand|Transport` (demand, transport) 2. `Exports|Air transport` (transport) 3. `Imports|Air transport` (transport) Reply with a number (1-3), or `yes` for option 1.

<a id="q010"></a>

## 010. AFOLU agriculture land forestry transformation results

Category: data and scenarios<br>
Route: `general_qa` (explicit site navigation request)<br>
Qwen generations: 0<br>
Time: 0.08 s<br>
Conversation: standalone

**Answer**

### AFOLU transformation

Available in this study:
- **Topics (28 variables):** Agricultural Demand (9), Land Cover (9), Primary Energy (4), Emissions (3), Price (3)
- **Example variables:** `Agricultural Demand`, `Land Cover`, `Primary Energy|Biomass`, `Emissions|CH4|AFOLU`, `Price|Agriculture|Corn|Index`
- **Scenarios (6):** `Cons_All`, `Cons_NoBiomass`, `Cons_NoFallow`, `Cons_NoForest`, `Reference`, `Uncons`
- **Models (1):** GCAM-PR 7.0
- **Regions (33):** World, Africa_Eastern, Africa_Northern, Africa_Southern, Africa_Western, Argentina and 27 more
- **Years:** 2005–2100

Explore this study:
- [Open the data explorer](https://iamparis.eu/results/ndc-aspects/afolu-transformation/graphs)
- [Read the study explainer](https://iamparis.eu/results/ndc-aspects/afolu-transformation/policy_questions)

<a id="q011"></a>

## 011. open the Aqueduct raw data application

Category: navigation and links<br>
Route: `general_qa` (explicit site navigation request)<br>
Qwen generations: 0<br>
Time: 0.05 s<br>
Conversation: standalone

**Answer**

Use these IAM PARIS links for this request: [Aqueduct](https://iamparis.eu/application_library/474).

<a id="q012"></a>

## 012. show me data for EU

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.33 s<br>
Conversation: standalone

**Answer**

I found the region `EU (European Union)`. Which variable should I use?

<a id="q013"></a>

## 013. plot Emissions|CO2 for World under Policy

Category: charts and comparisons<br>
Route: `general_qa` (charts delegated to data explorer)<br>
Qwen generations: 0<br>
Time: 0.01 s<br>
Conversation: standalone

**Answer**

I do not generate figures in the chat. Please use the [IAM PARIS data explorer](https://iamparis.eu/results) to explore, filter, and plot the data. Ask about a specific study first if you would like its direct explorer link.

<a id="q014"></a>

## 014. show Emissions|CO2 for World under Baseline by 2050

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.34 s<br>
Conversation: standalone

**Answer**

Showing results from the study **Where is the world headed?**. Also available in: Post-Glasgow targets; EU path to net zero; Global impacts of NDCs. Name another study to switch.

### Emissions|CO2 in World

Scope: scenario `multiple`, model `multiple`, years `first available-2050`
Unit: `Mt CO2/yr`

Answer:
In 2050, Emissions|CO2 in World ranges from 29,678 (MUSE, PR_Baseline) to 52,211 Mt CO2/yr (GEMINI-E3, PR_Baseline) across 8 series; median 43,333. Median change since 2005: +47%.

Summary: 8 series across 4 models and 4 scenarios.
Series are ordered by model and scenario.

| Model | Scenario | 2005 | 2010 | 2015 | 2020 | 2025 | 2030 | 2035 | 2040 | 2045 | 2050 | Unit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E3ME | PR_Baseline | — | 31,521 | 34,433 | 34,415 | 35,386 | 36,934 | 38,149 | 38,757 | 39,071 | 40,393 | Mt CO2/yr |
| E3ME | Unharmonised baseline | — | 31,521 | 34,433 | 34,430 | 35,463 | 37,096 | 38,457 | 39,232 | 39,600 | 41,071 | Mt CO2/yr |
| GCAM | Baseline | 31,426 | 36,147 | 39,196 | 38,300 | 40,654 | 42,107 | 43,979 | 45,315 | 45,465 | 45,595 | Mt CO2/yr |
| GCAM | PR_Baseline | 31,426 | 36,147 | 39,196 | 39,429 | 41,308 | 42,663 | 44,466 | 45,961 | 46,635 | 46,979 | Mt CO2/yr |
| GEMINI-E3 | Baseline | — | — | 34,738 | 34,934 | 38,132 | 42,037 | 44,008 | 45,532 | 46,413 | 46,713 | Mt CO2/yr |
| GEMINI-E3 | PR_Baseline | — | — | 34,859 | 37,813 | 41,214 | 44,462 | 46,876 | 48,984 | 50,715 | 52,211 | Mt CO2/yr |
| MUSE | baseline | — | — | 36,386 | 37,813 | 37,955 | 35,891 | 34,436 | 33,696 | 32,387 | 31,448 | Mt CO2/yr |
| MUSE | PR_Baseline | — | — | 36,431 | 38,836 | 40,158 | 41,521 | 38,898 | 36,732 | 33,730 | 29,678 | Mt CO2/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q015"></a>

## 015. show Emissions|CO2 for World under Baseline after 2030

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.34 s<br>
Conversation: standalone

**Answer**

Showing results from the study **Where is the world headed?**. Also available in: Post-Glasgow targets; EU path to net zero; Global impacts of NDCs. Name another study to switch.

### Emissions|CO2 in World

Scope: scenario `multiple`, model `multiple`, years `2031-latest available`
Unit: `Mt CO2/yr`

Answer:
In 2050, Emissions|CO2 in World ranges from 29,678 (MUSE, PR_Baseline) to 52,211 Mt CO2/yr (GEMINI-E3, PR_Baseline) across 8 series; median 43,333. Median change since 2035: +6%.

Summary: 8 series across 4 models and 4 scenarios.
Series are ordered by model and scenario.

| Model | Scenario | 2035 | 2040 | 2045 | 2050 | 2055 | 2060 | 2065 | 2070 | 2075 | 2080 | 2085 | 2090 | 2095 | 2100 | Unit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E3ME | PR_Baseline | 38,149 | 38,757 | 39,071 | 40,393 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| E3ME | Unharmonised baseline | 38,457 | 39,232 | 39,600 | 41,071 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| GCAM | Baseline | 43,979 | 45,315 | 45,465 | 45,595 | 46,708 | 48,206 | 49,791 | 51,521 | 53,322 | 54,770 | 56,170 | 57,563 | 58,673 | 60,068 | Mt CO2/yr |
| GCAM | PR_Baseline | 44,466 | 45,961 | 46,635 | 46,979 | 48,176 | 49,647 | 51,430 | 53,411 | 54,919 | 56,024 | 57,552 | 59,032 | 60,469 | 62,101 | Mt CO2/yr |
| GEMINI-E3 | Baseline | 44,008 | 45,532 | 46,413 | 46,713 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| GEMINI-E3 | PR_Baseline | 46,876 | 48,984 | 50,715 | 52,211 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| MUSE | baseline | 34,436 | 33,696 | 32,387 | 31,448 | 30,979 | 31,285 | 32,420 | 33,030 | 34,518 | 35,378 | 36,141 | 36,237 | 37,010 | 37,729 | Mt CO2/yr |
| MUSE | PR_Baseline | 38,898 | 36,732 | 33,730 | 29,678 | 26,902 | 24,364 | 23,828 | 24,252 | 24,392 | 24,597 | 24,559 | 24,441 | 24,608 | 24,998 | Mt CO2/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q016"></a>

## 016. show the latest available year for Emissions|CO2 in World

Category: catalogue and coverage<br>
Route: `data_query` (catalogue year coverage request)<br>
Qwen generations: 0<br>
Time: 0.29 s<br>
Conversation: standalone

**Answer**

I found the variable `Emissions|CO2` (CO2 emissions) in `World`. I still need the scenario. Choose the scenario: 1. `Baseline` (baseline) 2. `Baseline|Harmonized` (baseline) 3. `CP_EI` Reply with a number (1-3), or `yes` for option 1.

<a id="q017"></a>

## 017. what kinds of data are included

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 1<br>
Time: 2.16 s<br>
Conversation: standalone

**Answer**

### What I can help you with

- **Models:** 75 available. Examples: AIM/Enduse India 3.3, ALADIN, ATOM, CHANCE.
- **Variables:** 2046 available. Examples: Agricultural Demand, Agricultural Demand|Crops, Agricultural Demand|Crops|Energy, Agricultural Demand|Crops|Feed.
- **Scenarios:** 117 available. Examples: 1.5 TECH, 1.5 °C DRI-trade, 1.5 °C Domestic, BAU.
- **Regions:** 161 available. Examples: AFR, AGO, ARE, ARG.
- **Workspaces:** 21 available. Ask `list workspaces`.

Try one of these next:
- `list variables`
- `list regions`
- `list scenarios`
- `Show me CO2 emissions for EU`
- `Compare solar and wind capacity for EU`

Reply with a scenario, model, region, or year to narrow the answer.

<a id="q018"></a>

## 018. help me find data

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 1<br>
Time: 1.78 s<br>
Conversation: standalone

**Answer**

### What I can help you with

- **Models:** 75 available. Examples: AIM/Enduse India 3.3, ALADIN, ATOM, CHANCE.
- **Variables:** 2046 available. Examples: Agricultural Demand, Agricultural Demand|Crops, Agricultural Demand|Crops|Energy, Agricultural Demand|Crops|Feed.
- **Scenarios:** 117 available. Examples: 1.5 TECH, 1.5 °C DRI-trade, 1.5 °C Domestic, BAU.
- **Regions:** 161 available. Examples: AFR, AGO, ARE, ARG.
- **Workspaces:** 21 available. Ask `list workspaces`.

Try one of these next:
- `list variables`
- `list regions`
- `list scenarios`
- `Show me CO2 emissions for EU`
- `Compare solar and wind capacity for EU`

Reply with a scenario, model, region, or year to narrow the answer.

<a id="q019"></a>

## 019. compare GCAM and MESSAGE for CO2 emissions

Category: data and scenarios<br>
Route: `data_query` (chat plotting disabled; return numeric table)<br>
Qwen generations: 0<br>
Time: 0.75 s<br>
Conversation: standalone

**Answer**

I could not find `Emissions|CO2` for the explicitly requested model `MESSAGEix-GLOBIOM`. I did not substitute results from a different model. Models that do have this exact slice include: E3ME, E3ME 6.1, E4SMA-EU-TIMES 1.0, EnergyPLAN, GCAM.

<a id="q020"></a>

## 020. show data using gcampr

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.56 s<br>
Conversation: standalone

**Answer**

I need one more detail. Which variable should I use?

<a id="q021"></a>

## 021. show data with message ix

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.58 s<br>
Conversation: standalone

**Answer**

I need one more detail. Which variable should I use?

<a id="q022"></a>

## 022. plot solar capacity for EU under Baseline

Category: charts and comparisons<br>
Route: `general_qa` (charts delegated to data explorer)<br>
Qwen generations: 0<br>
Time: 0.01 s<br>
Conversation: standalone

**Answer**

I do not generate figures in the chat. Please use the [IAM PARIS data explorer](https://iamparis.eu/results) to explore, filter, and plot the data. Ask about a specific study first if you would like its direct explorer link.

<a id="q023"></a>

## 023. plot wind capacity for EU under Baseline

Category: charts and comparisons<br>
Route: `general_qa` (charts delegated to data explorer)<br>
Qwen generations: 0<br>
Time: 0.00 s<br>
Conversation: standalone

**Answer**

I do not generate figures in the chat. Please use the [IAM PARIS data explorer](https://iamparis.eu/results) to explore, filter, and plot the data. Ask about a specific study first if you would like its direct explorer link.

<a id="q024"></a>

## 024. gross domestic product for World

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.36 s<br>
Conversation: standalone

**Answer**

Showing results from the study **Where is the world headed?**. Also available in: Post-Glasgow targets; EU path to net zero; Global impacts of NDCs; Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU. Name another study to switch.

### GDP|MER in World

Scope: scenario `multiple`, model `multiple`, years `2005-2100`
Unit: `multiple`

Answer:
Summary: 26 series across 4 models and 8 scenarios.
Showing 12 of 26 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 14 remaining series.

| Model | Scenario | 2005 | 2010 | 2015 | 2020 | 2025 | 2030 | 2035 | 2040 | 2045 | 2050 | 2055 | 2060 | 2065 | 2070 | 2075 | 2080 | 2085 | 2090 | 2095 | 2100 | Unit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E3ME | PR_Baseline | — | 65,853 | 75,617 | 87,194 | 99,701 | 113,315 | 127,563 | 142,437 | 158,242 | 175,289 | — | — | — | — | — | — | — | — | — | — | billion US$2010/yr |
| E3ME | PR_CurPol_CP | — | 65,853 | 75,617 | 87,194 | 99,701 | 113,315 | 127,570 | 142,447 | 158,240 | 175,290 | — | — | — | — | — | — | — | — | — | — | billion US$2010/yr |
| E3ME | PR_CurPol_EI | — | 65,853 | 75,617 | 87,276 | 99,786 | 113,415 | 127,709 | 142,508 | 158,284 | 175,296 | — | — | — | — | — | — | — | — | — | — | billion US$2010/yr |
| GCAM | PR_Baseline | 57,891 | 65,619 | 75,256 | 86,062 | 98,447 | 111,867 | 125,882 | 140,380 | 155,661 | 171,683 | 188,396 | 206,259 | 225,685 | 246,272 | 267,947 | 290,862 | 315,135 | 340,678 | 367,520 | 395,501 | billion US$2010/yr OR local currency |
| GCAM | PR_CurPol_CP | 57,891 | 65,619 | 75,256 | 86,062 | 98,447 | 111,867 | 125,882 | 140,380 | 155,661 | 171,683 | 188,396 | 206,259 | 225,685 | 246,272 | 267,947 | 290,862 | 315,135 | 340,678 | 367,520 | 395,501 | billion US$2010/yr OR local currency |
| GCAM | PR_CurPol_CPO | 57,891 | 65,619 | 75,256 | 86,062 | 98,447 | 111,867 | 125,882 | 140,380 | 155,661 | 171,683 | 188,396 | 206,259 | 225,685 | 246,272 | 267,947 | 290,862 | 315,135 | 340,678 | 367,520 | 395,501 | billion US$2010/yr OR local currency |
| GEMINI-E3 | PR_Baseline | — | — | 105,812 | 124,716 | 147,216 | 173,209 | 200,203 | 230,282 | 262,121 | 296,650 | — | — | — | — | — | — | — | — | — | — | billion US$2010/yr OR local currency |
| GEMINI-E3 | PR_CurPol_CP | — | — | 105,812 | 124,690 | 147,077 | 172,969 | 199,923 | 229,974 | 261,783 | 296,277 | — | — | — | — | — | — | — | — | — | — | billion US$2010/yr OR local currency |
| GEMINI-E3 | PR_CurPol_CPO | — | — | 105,812 | 124,685 | 147,113 | 173,008 | 199,990 | 230,068 | 261,901 | 296,418 | — | — | — | — | — | — | — | — | — | — | billion US$2010/yr OR local currency |
| ICES-XPS | PR_Baseline | — | — | 74,357 | 84,932 | 97,365 | 110,895 | 125,105 | 139,889 | 155,533 | 171,964 | — | — | — | — | — | — | — | — | — | — | billion US$2010/yr |
| ICES-XPS | PR_CurPol_CP | — | — | 74,354 | 84,917 | 97,272 | 110,574 | 124,529 | 138,801 | 153,462 | 168,151 | — | — | — | — | — | — | — | — | — | — | billion US$2010/yr |
| ICES-XPS | PR_NDC_EI | — | — | 74,354 | 84,917 | 97,197 | 110,364 | 124,037 | 137,982 | 152,210 | 166,457 | — | — | — | — | — | — | — | — | — | — | billion US$2010/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q025"></a>

## 025. greenhouse gas pathways by country

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.21 s<br>
Conversation: standalone

**Answer**

I need a specific country or region before I can return a grounded pathway. Choose the region: 1. `CHN` (CHN (China)) 2. `IND` (IND (India)) 3. `USA` (USA (United States)) Reply with a number (1-3), or `yes` for option 1.

<a id="q026"></a>

## 026. current policy scenario emissions for EU

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.51 s<br>
Conversation: standalone

**Answer**

Interpretation: I treated bare **emissions** as `Emissions|CO2` (carbon-dioxide emissions).

### Emissions|CO2 in EU

Scope: scenario `multiple`, model `multiple`, years `2005-2100`
Unit: `Mt CO2/yr`

Answer:
In 2050, Emissions|CO2 in EU ranges from 1,024 (GCAM, PR_CurPol_CPO) to 2,633 Mt CO2/yr (E3ME, PR_CurPol_EI) across 11 series; median 1,923. Median change since 2005: -73%.

Summary: 11 series across 4 models and 3 scenarios.
Series are ordered by model and scenario.

| Model | Scenario | 2005 | 2010 | 2015 | 2020 | 2025 | 2030 | 2035 | 2040 | 2045 | 2050 | 2055 | 2060 | 2065 | 2070 | 2075 | 2080 | 2085 | 2090 | 2095 | 2100 | Unit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E3ME | PR_CurPol_CP | — | 4,304 | 3,999 | 3,444 | 3,188 | 3,033 | 2,869 | 2,705 | 2,539 | 2,359 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| E3ME | PR_CurPol_EI | — | 4,304 | 3,999 | 3,445 | 3,188 | 3,032 | 2,919 | 2,827 | 2,737 | 2,633 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| GCAM | PR_CurPol_CP | 4,190 | 3,790 | 3,490 | 3,115 | 2,687 | 2,257 | 2,112 | 1,836 | 1,485 | 1,118 | 962.69 | 912.20 | 870.84 | 863.75 | 839.93 | 771.80 | 656.18 | 503.01 | 281.84 | 10.13 | Mt CO2/yr |
| GCAM | PR_CurPol_CPO | 4,190 | 3,790 | 3,490 | 3,094 | 2,697 | 2,267 | 2,008 | 1,723 | 1,381 | 1,024 | 873.57 | 824.73 | 781.65 | 770.74 | 720.35 | 598.03 | 446.47 | 186.10 | -169.61 | -651.08 | Mt CO2/yr |
| GCAM | PR_CurPol_EI | 4,190 | 3,790 | 3,490 | 3,115 | 2,688 | 2,257 | 1,936 | 1,625 | 1,355 | 1,121 | 921.10 | 724.95 | 558.89 | 407.39 | 280.07 | 166.91 | 64.24 | -27.40 | -106.40 | -175.92 | Mt CO2/yr |
| GEMINI-E3 | PR_CurPol_CP | — | — | 3,411 | 3,197 | 3,006 | 2,733 | 2,630 | 2,545 | 2,441 | 2,342 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| GEMINI-E3 | PR_CurPol_CPO | — | — | 3,411 | 3,197 | 3,006 | 2,731 | 2,626 | 2,541 | 2,437 | 2,337 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| GEMINI-E3 | PR_CurPol_EI | — | — | 3,411 | 3,197 | 3,006 | 2,733 | 2,633 | 2,551 | 2,451 | 2,357 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| MUSE | PR_CurPol_CP | — | — | 3,118 | 3,018 | 2,956 | 2,740 | 2,581 | 2,435 | 2,155 | 1,871 | 1,641 | 1,482 | 1,414 | 1,365 | 1,237 | 1,116 | 975.01 | 910.90 | 844.23 | 817.09 | Mt CO2/yr |
| MUSE | PR_CurPol_CPO | — | — | 3,138 | 3,102 | 3,067 | 2,766 | 2,555 | 2,235 | 1,843 | 1,399 | 1,073 | 844.13 | 773.12 | 755.83 | 731.79 | 715.16 | 673.16 | 661.66 | 660.58 | 660.25 | Mt CO2/yr |
| MUSE | PR_CurPol_EI | — | — | 3,118 | 3,019 | 2,957 | 2,741 | 2,589 | 2,434 | 2,204 | 1,923 | 1,718 | 1,558 | 1,496 | 1,459 | 1,421 | 1,396 | 1,360 | 1,383 | 1,378 | 1,379 | Mt CO2/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q027"></a>

## 027. electricity generation for World

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.52 s<br>
Conversation: standalone

**Answer**

Showing results from the study **Where is the world headed?**. Also available in: Post-Glasgow targets; Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU; Tech-constrained EU pathways to net zero; Global impacts of NDCs; EU path to net zero; Implications of EU steel industry relocation. Name another study to switch.

### Secondary Energy|Electricity in World

Scope: scenario `multiple`, model `multiple`, years `2005-2100`
Unit: `EJ/yr`

Answer:
In 2040, Secondary Energy|Electricity in World ranges from 82.50 (MUSE, PR_Baseline) to 216.23 EJ/yr (ICES-XPS, PR_NDC_EI) across 43 series; median 142.00. Median change since 2005: +139%.

Summary: 43 series across 7 models and 8 scenarios.
Showing 12 of 43 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 31 remaining series.

| Model | Scenario | 2005 | 2010 | 2015 | 2020 | 2025 | 2030 | 2035 | 2040 | 2045 | 2050 | 2055 | 2060 | 2065 | 2070 | 2075 | 2080 | 2085 | 2090 | 2095 | 2100 | Unit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | PR_Baseline | 54.56 | 64.41 | 72.64 | 82.71 | 93.95 | 105.44 | 116.99 | 128.05 | 138.96 | — | — | — | — | — | — | — | — | — | — | — | EJ/yr |
| 42 | PR_CurPol_EI | 54.56 | 64.41 | 72.64 | 82.71 | 93.94 | 106.07 | 118.18 | 130.12 | 141.90 | — | — | — | — | — | — | — | — | — | — | — | EJ/yr |
| E3ME | PR_Baseline | — | 77.28 | 91.59 | 96.94 | 106.81 | 118.66 | 129.97 | 142.00 | 156.90 | 175.34 | — | — | — | — | — | — | — | — | — | — | EJ/yr |
| E3ME | PR_CurPol_CP | — | 77.28 | 91.59 | 96.94 | 106.81 | 118.66 | 129.97 | 141.98 | 156.86 | 175.27 | — | — | — | — | — | — | — | — | — | — | EJ/yr |
| GCAM | PR_Baseline | 64.06 | 75.50 | 85.37 | 102.41 | 120.69 | 142.95 | 155.89 | 168.12 | 183.00 | 198.72 | 214.99 | 231.30 | 247.24 | 263.10 | 277.57 | 291.02 | 303.03 | 314.23 | 323.92 | 332.77 | EJ/yr |
| GCAM | PR_CurPol_CP | 64.06 | 75.50 | 85.37 | 105.69 | 125.92 | 146.40 | 153.02 | 165.29 | 181.43 | 197.76 | 214.51 | 231.38 | 248.29 | 265.68 | 281.12 | 295.56 | 308.25 | 320.19 | 330.87 | 340.98 | EJ/yr |
| GEMINI-E3 | PR_Baseline | — | — | 97.03 | 109.58 | 123.73 | 139.22 | 153.62 | 168.72 | 185.68 | 201.38 | — | — | — | — | — | — | — | — | — | — | EJ/yr |
| GEMINI-E3 | PR_CurPol_CP | — | — | 97.03 | 108.85 | 122.00 | 137.18 | 153.16 | 168.19 | 186.13 | 203.34 | — | — | — | — | — | — | — | — | — | — | EJ/yr |
| ICES-XPS | PR_Baseline | — | — | 84.05 | 86.17 | 106.29 | 133.19 | 163.04 | 200.17 | 246.59 | 309.16 | — | — | — | — | — | — | — | — | — | — | EJ/yr |
| ICES-XPS | PR_NDC_EI | — | — | 84.05 | 86.04 | 107.39 | 136.49 | 171.79 | 216.23 | 268.74 | 330.41 | — | — | — | — | — | — | — | — | — | — | EJ/yr |
| MUSE | PR_Baseline | — | — | 67.53 | 70.22 | 72.05 | 74.56 | 78.47 | 82.50 | 89.60 | 95.17 | 100.12 | 105.55 | 111.07 | 113.88 | 120.49 | 123.09 | 128.11 | 133.85 | 137.65 | 138.76 | EJ/yr |
| TIAM_Grantham | PR_Baseline | 55.14 | 65.72 | — | 85.91 | — | 93.67 | — | 107.72 | — | 127.91 | — | 153.46 | — | 170.72 | — | 196.15 | — | 217.57 | — | 237.88 | EJ/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q028"></a>

## 028. plot electricity generation for World

Category: charts and comparisons<br>
Route: `general_qa` (charts delegated to data explorer)<br>
Qwen generations: 0<br>
Time: 0.01 s<br>
Conversation: standalone

**Answer**

I do not generate figures in the chat. Please use the [IAM PARIS data explorer](https://iamparis.eu/results) to explore, filter, and plot the data. Ask about a specific study first if you would like its direct explorer link.

<a id="q029"></a>

## 029. wind power capacity in Europe

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.37 s<br>
Conversation: standalone

**Answer**

Showing results from the study **Where is the world headed?**. Also available in: Where is the EU headed?. Name another study to switch.

### Capacity|Electricity|Wind in EU

Scope: scenario `multiple`, model `multiple`, years `2005-2100`
Unit: `GW`

Answer:
In 2100, Capacity|Electricity|Wind in EU ranges from 73.32 (TIAM_Grantham, PR_Baseline) to 3,317 GW (TIAM_Grantham, PR_NDC_CPO) across 14 series; median 980.78. Median change since 2005: +3836%.

Summary: 14 series across 2 models and 7 scenarios.
Showing 12 of 14 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 2 remaining series.

| Model | Scenario | 2005 | 2010 | 2015 | 2020 | 2025 | 2030 | 2035 | 2040 | 2045 | 2050 | 2055 | 2060 | 2065 | 2070 | 2075 | 2080 | 2085 | 2090 | 2095 | 2100 | Unit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GCAM | PR_Baseline | 33.27 | 70.49 | 142.25 | 201.02 | 207.61 | 192.81 | 144.63 | 119.29 | 148.69 | 193.11 | 234.53 | 269.70 | 275.61 | 279.09 | 288.14 | 304.67 | 326.61 | 356.56 | 371.25 | 380.50 | GW |
| GCAM | PR_CurPol_CP | 24.92 | 52.78 | 106.52 | 273.87 | 404.51 | 653.48 | 743.70 | 783.52 | 849.88 | 853.32 | 954.06 | 1,028 | 1,084 | 1,160 | 1,232 | 1,308 | 1,358 | 1,397 | 1,420 | 1,420 | GW |
| GCAM | PR_CurPol_CPO | 30.05 | 63.66 | 128.47 | 200.66 | 233.11 | 234.92 | 223.29 | 213.23 | 239.90 | 283.55 | 337.15 | 388.15 | 434.12 | 474.13 | 500.59 | 523.95 | 546.90 | 571.40 | 590.43 | 594.21 | GW |
| GCAM | PR_CurPol_EI | 24.92 | 52.79 | 106.53 | 273.88 | 404.55 | 555.41 | 601.33 | 558.66 | 531.38 | 523.05 | 609.23 | 691.42 | 791.25 | 859.04 | 875.28 | 872.37 | 860.95 | 874.82 | 919.14 | 974.80 | GW |
| GCAM | PR_NDC_CP | 24.92 | 52.78 | 106.52 | 273.87 | 407.13 | 562.32 | 604.69 | 569.96 | 561.73 | 561.28 | 628.42 | 680.15 | 721.76 | 772.24 | 821.25 | 873.41 | 907.78 | 937.40 | 961.91 | 986.75 | GW |
| GCAM | PR_NDC_CPO | 30.07 | 63.70 | 128.54 | 200.66 | 233.46 | 236.28 | 220.44 | 207.16 | 235.28 | 280.73 | 338.08 | 392.21 | 435.58 | 473.69 | 498.89 | 522.20 | 549.26 | 579.53 | 612.89 | 637.93 | GW |
| TIAM_Grantham | PR_Baseline | 38.55 | 93.40 | — | 106.50 | — | 41.56 | — | 10.75 | — | 0.00 | — | 0.00 | — | 5.20 | — | 34.56 | — | 58.28 | — | 73.32 | GW |
| TIAM_Grantham | PR_CurPol_CP | 38.55 | 93.40 | — | 107.11 | — | 221.14 | — | 208.74 | — | 474.25 | — | 736.49 | — | 873.54 | — | 805.55 | — | 960.71 | — | 2,477 | GW |
| TIAM_Grantham | PR_CurPol_CPO | 38.55 | 93.40 | — | 107.11 | — | 462.12 | — | 449.72 | — | 529.84 | — | 718.32 | — | 905.10 | — | 1,064 | — | 2,509 | — | 3,194 | GW |
| TIAM_Grantham | PR_CurPol_EI | 38.55 | 93.40 | — | 107.11 | — | 383.66 | — | 371.25 | — | 498.98 | — | 555.18 | — | 632.13 | — | 736.72 | — | 840.35 | — | 919.33 | GW |
| TIAM_Grantham | PR_NDC_CP | 38.55 | 93.40 | — | 107.11 | — | 391.82 | — | 432.43 | — | 714.18 | — | 873.29 | — | 947.18 | — | 1,129 | — | 2,496 | — | 3,265 | GW |
| TIAM_Grantham | PR_NDC_CPO | 38.55 | 93.40 | — | 107.11 | — | 525.05 | — | 512.65 | — | 692.89 | — | 828.98 | — | 916.97 | — | 1,090 | — | 2,492 | — | 3,317 | GW |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q030"></a>

## 030. solar pv installed capacity in World

Category: data and scenarios<br>
Route: `data_query` (low confidence entity clarification)<br>
Qwen generations: 0<br>
Time: 0.30 s<br>
Conversation: standalone

**Answer**

I could not confidently match all of the requested wording. The loaded variable catalogue does not represent these term(s) in the top match: `installed`. Choose the variable: 1. `Capacity|Electricity|Solar|PV` (power capacity, solar) 2. `Capacity Additions|Electricity|Solar|PV` (capacity, solar) Reply with a number (1-2), or `yes` for option 1.

<a id="q031"></a>

## 031. plot solar pv installed capacity in World

Category: charts and comparisons<br>
Route: `general_qa` (charts delegated to data explorer)<br>
Qwen generations: 0<br>
Time: 0.01 s<br>
Conversation: standalone

**Answer**

I do not generate figures in the chat. Please use the [IAM PARIS data explorer](https://iamparis.eu/results) to explore, filter, and plot the data. Ask about a specific study first if you would like its direct explorer link.

<a id="q032"></a>

## 032. show GDP for EU

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.35 s<br>
Conversation: standalone

**Answer**

Showing results from the study **Where is the world headed?**. Also available in: Where is the EU headed?. Name another study to switch.

### GDP|MER in EU

Scope: scenario `multiple`, model `multiple`, years `2005-2100`
Unit: `multiple`

Answer:
Summary: 26 series across 4 models and 8 scenarios.
Showing 12 of 26 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 14 remaining series.

| Model | Scenario | 2005 | 2010 | 2015 | 2020 | 2025 | 2030 | 2035 | 2040 | 2045 | 2050 | 2055 | 2060 | 2065 | 2070 | 2075 | 2080 | 2085 | 2090 | 2095 | 2100 | Unit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E3ME | PR_Baseline | — | 16,989 | 17,957 | 19,653 | 20,938 | 22,373 | 23,850 | 25,612 | 27,672 | 29,932 | — | — | — | — | — | — | — | — | — | — | billion US$2010/yr |
| E3ME | PR_CurPol_CP | — | 16,989 | 17,957 | 19,653 | 20,938 | 22,373 | 23,848 | 25,609 | 27,668 | 29,929 | — | — | — | — | — | — | — | — | — | — | billion US$2010/yr |
| E3ME | PR_CurPol_EI | — | 16,989 | 17,957 | 19,656 | 20,939 | 22,375 | 23,840 | 25,577 | 27,578 | 29,789 | — | — | — | — | — | — | — | — | — | — | billion US$2010/yr |
| GCAM | PR_Baseline | 16,191 | 16,927 | 17,895 | 19,705 | 21,015 | 22,444 | 23,909 | 25,589 | 27,582 | 29,708 | 31,980 | 34,426 | 37,046 | 39,685 | 42,463 | 45,454 | 48,771 | 52,313 | 56,061 | 59,970 | billion US$2010/yr OR local currency |
| GCAM | PR_CurPol_CP | 16,191 | 16,927 | 17,895 | 19,705 | 21,015 | 22,444 | 23,909 | 25,589 | 27,582 | 29,708 | 31,980 | 34,426 | 37,046 | 39,685 | 42,463 | 45,454 | 48,771 | 52,313 | 56,061 | 59,970 | billion US$2010/yr OR local currency |
| GCAM | PR_CurPol_CPO | 16,191 | 16,927 | 17,895 | 19,705 | 21,015 | 22,444 | 23,909 | 25,589 | 27,582 | 29,708 | 31,980 | 34,426 | 37,046 | 39,685 | 42,463 | 45,454 | 48,771 | 52,313 | 56,061 | 59,970 | billion US$2010/yr OR local currency |
| GEMINI-E3 | PR_Baseline | — | — | 17,962 | 19,766 | 21,129 | 22,463 | 23,920 | 25,436 | 27,013 | 28,689 | — | — | — | — | — | — | — | — | — | — | billion US$2010/yr OR local currency |
| GEMINI-E3 | PR_CurPol_CP | — | — | 17,962 | 19,766 | 21,145 | 22,518 | 24,070 | 25,688 | 27,367 | 29,148 | — | — | — | — | — | — | — | — | — | — | billion US$2010/yr OR local currency |
| GEMINI-E3 | PR_CurPol_CPO | — | — | 17,962 | 19,766 | 21,145 | 22,518 | 24,070 | 25,687 | 27,367 | 29,148 | — | — | — | — | — | — | — | — | — | — | billion US$2010/yr OR local currency |
| ICES-XPS | PR_Baseline | — | — | 18,631 | 20,406 | 21,778 | 23,281 | 24,822 | 26,584 | 28,674 | 30,900 | — | — | — | — | — | — | — | — | — | — | billion US$2010/yr |
| ICES-XPS | PR_CurPol_EI | — | — | 18,630 | 20,396 | 21,639 | 22,892 | 24,064 | 25,210 | 26,300 | 27,114 | — | — | — | — | — | — | — | — | — | — | billion US$2010/yr |
| ICES-XPS | PR_NDC_CPO | — | — | 18,630 | 20,395 | 21,740 | 23,173 | 24,760 | 26,613 | 28,831 | 31,230 | — | — | — | — | — | — | — | — | — | — | billion US$2010/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q033"></a>

## 033. gdp values for global economy

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.49 s<br>
Conversation: standalone

**Answer**

Showing results from the study **Where is the world headed?**. Also available in: Post-Glasgow targets; EU path to net zero; Global impacts of NDCs; Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU. Name another study to switch.

### GDP|MER in World

Scope: scenario `multiple`, model `multiple`, years `2005-2100`
Unit: `multiple`

Answer:
Summary: 26 series across 4 models and 8 scenarios.
Showing 12 of 26 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 14 remaining series.

| Model | Scenario | 2005 | 2010 | 2015 | 2020 | 2025 | 2030 | 2035 | 2040 | 2045 | 2050 | 2055 | 2060 | 2065 | 2070 | 2075 | 2080 | 2085 | 2090 | 2095 | 2100 | Unit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E3ME | PR_Baseline | — | 65,853 | 75,617 | 87,194 | 99,701 | 113,315 | 127,563 | 142,437 | 158,242 | 175,289 | — | — | — | — | — | — | — | — | — | — | billion US$2010/yr |
| E3ME | PR_CurPol_CP | — | 65,853 | 75,617 | 87,194 | 99,701 | 113,315 | 127,570 | 142,447 | 158,240 | 175,290 | — | — | — | — | — | — | — | — | — | — | billion US$2010/yr |
| E3ME | PR_CurPol_EI | — | 65,853 | 75,617 | 87,276 | 99,786 | 113,415 | 127,709 | 142,508 | 158,284 | 175,296 | — | — | — | — | — | — | — | — | — | — | billion US$2010/yr |
| GCAM | PR_Baseline | 57,891 | 65,619 | 75,256 | 86,062 | 98,447 | 111,867 | 125,882 | 140,380 | 155,661 | 171,683 | 188,396 | 206,259 | 225,685 | 246,272 | 267,947 | 290,862 | 315,135 | 340,678 | 367,520 | 395,501 | billion US$2010/yr OR local currency |
| GCAM | PR_CurPol_CP | 57,891 | 65,619 | 75,256 | 86,062 | 98,447 | 111,867 | 125,882 | 140,380 | 155,661 | 171,683 | 188,396 | 206,259 | 225,685 | 246,272 | 267,947 | 290,862 | 315,135 | 340,678 | 367,520 | 395,501 | billion US$2010/yr OR local currency |
| GCAM | PR_CurPol_CPO | 57,891 | 65,619 | 75,256 | 86,062 | 98,447 | 111,867 | 125,882 | 140,380 | 155,661 | 171,683 | 188,396 | 206,259 | 225,685 | 246,272 | 267,947 | 290,862 | 315,135 | 340,678 | 367,520 | 395,501 | billion US$2010/yr OR local currency |
| GEMINI-E3 | PR_Baseline | — | — | 105,812 | 124,716 | 147,216 | 173,209 | 200,203 | 230,282 | 262,121 | 296,650 | — | — | — | — | — | — | — | — | — | — | billion US$2010/yr OR local currency |
| GEMINI-E3 | PR_CurPol_CP | — | — | 105,812 | 124,690 | 147,077 | 172,969 | 199,923 | 229,974 | 261,783 | 296,277 | — | — | — | — | — | — | — | — | — | — | billion US$2010/yr OR local currency |
| GEMINI-E3 | PR_CurPol_CPO | — | — | 105,812 | 124,685 | 147,113 | 173,008 | 199,990 | 230,068 | 261,901 | 296,418 | — | — | — | — | — | — | — | — | — | — | billion US$2010/yr OR local currency |
| ICES-XPS | PR_Baseline | — | — | 74,357 | 84,932 | 97,365 | 110,895 | 125,105 | 139,889 | 155,533 | 171,964 | — | — | — | — | — | — | — | — | — | — | billion US$2010/yr |
| ICES-XPS | PR_CurPol_CP | — | — | 74,354 | 84,917 | 97,272 | 110,574 | 124,529 | 138,801 | 153,462 | 168,151 | — | — | — | — | — | — | — | — | — | — | billion US$2010/yr |
| ICES-XPS | PR_NDC_EI | — | — | 74,354 | 84,917 | 97,197 | 110,364 | 124,037 | 137,982 | 152,210 | 166,457 | — | — | — | — | — | — | — | — | — | — | billion US$2010/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q034"></a>

## 034. show carbon dioxide emissions globally

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.45 s<br>
Conversation: standalone

**Answer**

Showing results from the study **Where is the world headed?**. Also available in: Post-Glasgow targets; Global impacts of NDCs; Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU; EU path to net zero; Tech-constrained EU pathways to net zero. Name another study to switch.

### Emissions|CO2 in World

Scope: scenario `multiple`, model `multiple`, years `2005-2100`
Unit: `Mt CO2/yr`

Answer:
In 2050, Emissions|CO2 in World ranges from 9,173 (GCAM, NDC_LTT) to 52,211 Mt CO2/yr (GEMINI-E3, PR_Baseline) across 38 series; median 33,062. Median change since 2005: +6%.

Summary: 38 series across 4 models and 13 scenarios.
Showing 12 of 38 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 26 remaining series.

| Model | Scenario | 2005 | 2010 | 2015 | 2020 | 2025 | 2030 | 2035 | 2040 | 2045 | 2050 | 2055 | 2060 | 2065 | 2070 | 2075 | 2080 | 2085 | 2090 | 2095 | 2100 | Unit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E3ME | PR_Baseline | — | 31,521 | 34,433 | 34,415 | 35,386 | 36,934 | 38,149 | 38,757 | 39,071 | 40,393 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| E3ME | PR_CurPol_CP | — | 31,521 | 34,433 | 34,416 | 35,385 | 36,935 | 38,006 | 38,479 | 38,772 | 40,054 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| E3ME | PR_CurPol_EI | — | 31,521 | 34,433 | 34,407 | 35,350 | 36,922 | 37,792 | 38,706 | 39,138 | 39,239 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| GCAM | Baseline | 31,426 | 36,147 | 39,196 | 38,300 | 40,654 | 42,107 | 43,979 | 45,315 | 45,465 | 45,595 | 46,708 | 48,206 | 49,791 | 51,521 | 53,322 | 54,770 | 56,170 | 57,563 | 58,673 | 60,068 | Mt CO2/yr |
| GCAM | CP_EI | 31,426 | 36,147 | 39,196 | 37,250 | 37,566 | 37,516 | 35,676 | 33,501 | 31,082 | 28,519 | 26,244 | 24,349 | 22,565 | 20,869 | 19,172 | 17,514 | 15,927 | 14,464 | 13,087 | 11,834 | Mt CO2/yr |
| GCAM | NDC_LTT | 31,426 | 36,147 | 39,196 | 37,250 | 35,865 | 32,324 | 26,749 | 20,969 | 14,887 | 9,173 | 4,316 | 871.19 | 669.19 | 59.16 | -310.14 | -114.52 | -451.30 | -589.76 | -661.35 | -766.86 | Mt CO2/yr |
| GEMINI-E3 | Baseline | — | — | 34,738 | 34,934 | 38,132 | 42,037 | 44,008 | 45,532 | 46,413 | 46,713 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| GEMINI-E3 | CP_EI | — | — | 34,738 | 34,934 | 34,187 | 34,431 | 32,849 | 31,313 | 29,460 | 27,388 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| GEMINI-E3 | PR_Baseline | — | — | 34,859 | 37,813 | 41,214 | 44,462 | 46,876 | 48,984 | 50,715 | 52,211 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| MUSE | baseline | — | — | 36,386 | 37,813 | 37,955 | 35,891 | 34,436 | 33,696 | 32,387 | 31,448 | 30,979 | 31,285 | 32,420 | 33,030 | 34,518 | 35,378 | 36,141 | 36,237 | 37,010 | 37,729 | Mt CO2/yr |
| MUSE | CP_EI | — | — | 36,381 | 36,003 | 35,319 | 33,276 | 30,502 | 27,906 | 26,115 | 23,856 | 22,850 | 21,458 | 20,621 | 19,791 | 18,914 | 18,091 | 18,210 | 17,693 | 18,037 | 18,122 | Mt CO2/yr |
| MUSE | NDC_EI | — | — | 36,381 | 35,946 | 34,574 | 31,089 | 27,917 | 24,700 | 22,635 | 19,999 | 18,772 | 17,549 | 17,234 | 16,328 | 15,290 | 14,893 | 14,669 | 14,811 | 15,232 | 15,044 | Mt CO2/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q035"></a>

## 035. display co2 emission trend for China

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.45 s<br>
Conversation: standalone

**Answer**

Showing results from the study **Where is the world headed?**. Also available in: Post-Glasgow targets; Global impacts of NDCs; Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU; Tech-constrained EU pathways to net zero. Name another study to switch.

### Emissions|CO2 in CHN

Scope: scenario `multiple`, model `multiple`, years `2005-2100`
Unit: `Mt CO2/yr`

Answer:
In 2050, Emissions|CO2 in CHN ranges from 6,159 (MUSE, PR_NDC_CPO) to 16,585 Mt CO2/yr (GEMINI-E3, PR_Baseline) across 28 series; median 7,956. Median change since 2005: +30%.

Summary: 28 series across 4 models and 10 scenarios.
Showing 12 of 28 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 16 remaining series.

| Model | Scenario | 2005 | 2010 | 2015 | 2020 | 2025 | 2030 | 2035 | 2040 | 2045 | 2050 | 2055 | 2060 | 2065 | 2070 | 2075 | 2080 | 2085 | 2090 | 2095 | 2100 | Unit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E3ME | PR_Baseline | — | 7,787 | 9,450 | 10,280 | 10,814 | 11,105 | 10,755 | 9,505 | 8,119 | 7,542 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| E3ME | PR_CurPol_CP | — | 7,787 | 9,450 | 10,280 | 10,814 | 11,105 | 10,679 | 9,336 | 7,972 | 7,420 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| E3ME | PR_CurPol_EI | — | 7,787 | 9,450 | 10,280 | 10,814 | 11,105 | 10,864 | 10,503 | 10,030 | 9,609 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| GCAM | Baseline | 6,734 | 9,611 | 11,226 | 12,529 | 13,314 | 13,627 | 13,705 | 13,727 | 13,025 | 12,277 | 11,947 | 11,734 | 11,498 | 11,264 | 10,996 | 10,591 | 10,158 | 9,781 | 9,360 | 9,015 | Mt CO2/yr |
| GCAM | CP_EI | 6,734 | 9,611 | 11,226 | 12,133 | 12,380 | 12,238 | 10,983 | 9,491 | 7,924 | 6,473 | 5,226 | 4,154 | 3,205 | 2,384 | 1,664 | 1,077 | 594.72 | 204.01 | -104.68 | -345.55 | Mt CO2/yr |
| GCAM | PR_Baseline | 6,734 | 9,611 | 11,226 | 12,605 | 13,076 | 13,122 | 12,990 | 12,815 | 12,141 | 11,321 | 10,825 | 10,359 | 10,052 | 9,812 | 9,288 | 8,620 | 8,212 | 7,843 | 7,549 | 7,317 | Mt CO2/yr |
| GEMINI-E3 | PR_Baseline | — | — | 10,654 | 13,168 | 14,822 | 16,021 | 16,620 | 16,840 | 16,770 | 16,585 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| GEMINI-E3 | PR_CurPol_CP | — | — | 10,654 | 12,013 | 11,479 | 11,072 | 10,866 | 10,788 | 10,415 | 9,939 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| GEMINI-E3 | PR_CurPol_CPO | — | — | 10,654 | 12,070 | 11,935 | 11,655 | 11,520 | 11,502 | 11,153 | 10,692 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| MUSE | PR_Baseline | — | — | 11,234 | 12,637 | 13,262 | 13,488 | 11,591 | 10,284 | 8,467 | 6,552 | 5,214 | 4,313 | 3,843 | 3,402 | 3,148 | 2,929 | 2,748 | 2,609 | 2,472 | 2,356 | Mt CO2/yr |
| MUSE | PR_CurPol_CP | — | — | 11,232 | 12,243 | 12,789 | 12,775 | 11,463 | 9,893 | 8,388 | 6,538 | 5,245 | 4,353 | 3,916 | 3,471 | 3,187 | 2,916 | 2,738 | 2,596 | 2,443 | 2,335 | Mt CO2/yr |
| MUSE | PR_NDC_CPO | — | — | 11,234 | 12,632 | 13,091 | 11,847 | 10,599 | 9,202 | 8,009 | 6,159 | 4,891 | 4,011 | 3,559 | 3,120 | 2,972 | 2,840 | 2,709 | 2,580 | 2,474 | 2,355 | Mt CO2/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q036"></a>

## 036. plot co2 emission trend for China

Category: charts and comparisons<br>
Route: `general_qa` (charts delegated to data explorer)<br>
Qwen generations: 0<br>
Time: 0.01 s<br>
Conversation: standalone

**Answer**

I do not generate figures in the chat. Please use the [IAM PARIS data explorer](https://iamparis.eu/results) to explore, filter, and plot the data. Ask about a specific study first if you would like its direct explorer link.

<a id="q037"></a>

## 037. show data for India emissions

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.38 s<br>
Conversation: standalone

**Answer**

Showing results from the study **Where is the world headed?**. Also available in: Post-Glasgow targets; Global impacts of NDCs; Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU; Tech-constrained EU pathways to net zero. Name another study to switch.

Interpretation: I treated bare **emissions** as `Emissions|CO2` (carbon-dioxide emissions).

### Emissions|CO2 in IND

Scope: scenario `multiple`, model `multiple`, years `2005-2100`
Unit: `Mt CO2/yr`

Answer:
In 2050, Emissions|CO2 in IND ranges from 2,310 (MUSE, PR_NDC_CP) to 7,590 Mt CO2/yr (E3ME, Unharmonised baseline) across 28 series; median 5,874. Median change since 2005: +340%.

Summary: 28 series across 4 models and 10 scenarios.
Showing 12 of 28 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 16 remaining series.

| Model | Scenario | 2005 | 2010 | 2015 | 2020 | 2025 | 2030 | 2035 | 2040 | 2045 | 2050 | 2055 | 2060 | 2065 | 2070 | 2075 | 2080 | 2085 | 2090 | 2095 | 2100 | Unit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E3ME | PR_Baseline | — | 1,664 | 2,265 | 2,736 | 3,238 | 3,918 | 4,719 | 5,658 | 6,553 | 7,582 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| E3ME | PR_CurPol_CP | — | 1,664 | 2,265 | 2,736 | 3,238 | 3,918 | 4,703 | 5,628 | 6,505 | 7,504 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| E3ME | Unharmonised baseline | — | 1,664 | 2,265 | 2,737 | 3,249 | 3,925 | 4,726 | 5,671 | 6,567 | 7,590 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| GCAM | Baseline | 1,379 | 1,857 | 2,316 | 2,685 | 3,424 | 4,017 | 4,642 | 5,049 | 5,316 | 5,530 | 5,749 | 5,972 | 6,150 | 6,270 | 6,419 | 6,437 | 6,423 | 6,349 | 6,245 | 6,197 | Mt CO2/yr |
| GCAM | CP_EI | 1,379 | 1,857 | 2,316 | 2,607 | 3,292 | 3,862 | 4,097 | 4,117 | 3,951 | 3,676 | 3,395 | 3,085 | 2,750 | 2,380 | 1,997 | 1,603 | 1,225 | 870.03 | 525.83 | 202.52 | Mt CO2/yr |
| GCAM | PR_Baseline | 1,379 | 1,857 | 2,316 | 3,175 | 3,872 | 4,507 | 5,197 | 5,909 | 6,496 | 6,868 | 7,287 | 7,692 | 8,030 | 8,309 | 8,499 | 8,529 | 8,567 | 8,580 | 8,557 | 8,538 | Mt CO2/yr |
| GEMINI-E3 | PR_Baseline | — | — | 2,257 | 2,857 | 3,553 | 4,334 | 5,015 | 5,683 | 6,336 | 6,917 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| GEMINI-E3 | PR_CurPol_CP | — | — | 2,257 | 2,862 | 3,488 | 4,281 | 4,916 | 5,544 | 6,149 | 6,675 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| GEMINI-E3 | PR_CurPol_CPO | — | — | 2,257 | 2,870 | 3,575 | 4,291 | 4,936 | 5,572 | 6,177 | 6,704 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| MUSE | PR_Baseline | — | — | 2,393 | 2,865 | 3,125 | 3,458 | 3,271 | 2,959 | 2,753 | 2,365 | 1,709 | 1,381 | 1,395 | 1,419 | 1,437 | 1,443 | 1,427 | 1,386 | 1,353 | 1,319 | Mt CO2/yr |
| MUSE | PR_CurPol_CP | — | — | 2,393 | 2,856 | 3,040 | 3,198 | 3,274 | 2,987 | 2,796 | 2,324 | 1,715 | 1,403 | 1,405 | 1,417 | 1,436 | 1,448 | 1,422 | 1,384 | 1,352 | 1,304 | Mt CO2/yr |
| MUSE | PR_NDC_CP | — | — | 2,393 | 2,856 | 3,040 | 3,199 | 3,264 | 2,983 | 2,767 | 2,310 | 1,715 | 1,413 | 1,438 | 1,474 | 1,495 | 1,507 | 1,497 | 1,460 | 1,437 | 1,390 | Mt CO2/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q038"></a>

## 038. show emissions in Greece

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.36 s<br>
Conversation: standalone

**Answer**

Showing results from the study **EU path to net zero**. Also available in: Where is the world headed?. Name another study to switch.

Interpretation: I treated bare **emissions** as `Emissions|CO2` (carbon-dioxide emissions).

### Emissions|CO2 in GREECE

Scope: scenario `multiple`, model `multiple`, years `2010-2050`
Unit: `Mt CO2/yr`

Answer:
In 2050, Emissions|CO2 in GREECE ranges from -0.667 (NEMESIS 5.1, NZE_Bench_L) to 39.41 Mt CO2/yr (NEMESIS 5.1, WWH21) across 10 series; median 2.70. Median change since 2010: -97%.

Summary: 10 series across 2 models and 5 scenarios.
Series are ordered by model and scenario.

| Model | Scenario | 2010 | 2015 | 2020 | 2025 | 2030 | 2035 | 2040 | 2045 | 2050 | Unit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E4SMA-EU-TIMES 1.0 | NZE_Bench_H | 85.99 | — | 71.87 | 44.73 | 36.59 | — | 20.67 | — | 4.07 | Mt CO2/yr |
| E4SMA-EU-TIMES 1.0 | NZE_Bench_L | 85.99 | — | 71.87 | 41.40 | 34.48 | — | 14.45 | — | 0.192 | Mt CO2/yr |
| E4SMA-EU-TIMES 1.0 | NZE_Bench_M | 85.99 | — | 71.87 | 41.08 | 35.87 | — | 18.43 | — | 1.16 | Mt CO2/yr |
| E4SMA-EU-TIMES 1.0 | NZE_EUPol_Stand | 85.99 | — | 71.87 | 48.85 | 34.70 | — | 20.34 | — | 1.09 | Mt CO2/yr |
| E4SMA-EU-TIMES 1.0 | WWH21 | 85.99 | — | 71.90 | 66.79 | 44.07 | — | 39.42 | — | 37.08 | Mt CO2/yr |
| NEMESIS 5.1 | NZE_Bench_H | 97.32 | 74.94 | 59.35 | 51.64 | 38.12 | 29.46 | 20.70 | 12.69 | 5.59 | Mt CO2/yr |
| NEMESIS 5.1 | NZE_Bench_L | 97.32 | 74.94 | 59.35 | 49.37 | 35.26 | 24.61 | 14.51 | 6.09 | -0.667 | Mt CO2/yr |
| NEMESIS 5.1 | NZE_Bench_M | 97.32 | 74.94 | 59.35 | 50.56 | 36.99 | 27.13 | 18.08 | 9.70 | 2.81 | Mt CO2/yr |
| NEMESIS 5.1 | NZE_EUPol_Stand | 97.32 | 74.94 | 59.29 | 50.45 | 36.19 | 27.60 | 17.89 | 9.96 | 2.59 | Mt CO2/yr |
| NEMESIS 5.1 | WWH21 | 97.32 | 74.94 | 58.89 | 55.11 | 46.60 | 43.51 | 41.27 | 40.41 | 39.41 | Mt CO2/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/eu-path-to-net-zero/graphs).

Reply with a scenario, model, region, or year to narrow the answer.

<a id="q039"></a>

## 039. show emissions for United States

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.41 s<br>
Conversation: standalone

**Answer**

Showing results from the study **Where is the world headed?**. Also available in: Post-Glasgow targets; EU path to net zero; Global impacts of NDCs; Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU. Name another study to switch.

Interpretation: I treated bare **emissions** as `Emissions|CO2` (carbon-dioxide emissions).

### Emissions|CO2 in USA

Scope: scenario `multiple`, model `multiple`, years `2005-2100`
Unit: `Mt CO2/yr`

Answer:
In 2050, Emissions|CO2 in USA ranges from 859.19 (GCAM, PR_NDC_EI) to 5,210 Mt CO2/yr (GCAM, PR_Baseline) across 28 series; median 3,657. Median change since 2005: -37%.

Summary: 28 series across 4 models and 10 scenarios.
Showing 12 of 28 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 16 remaining series.

| Model | Scenario | 2005 | 2010 | 2015 | 2020 | 2025 | 2030 | 2035 | 2040 | 2045 | 2050 | 2055 | 2060 | 2065 | 2070 | 2075 | 2080 | 2085 | 2090 | 2095 | 2100 | Unit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E3ME | PR_Baseline | — | 5,666 | 5,366 | 4,704 | 4,252 | 4,085 | 4,037 | 4,030 | 4,007 | 3,922 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| E3ME | PR_CurPol_CP | — | 5,666 | 5,366 | 4,704 | 4,252 | 4,085 | 4,029 | 4,018 | 3,990 | 3,898 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| E3ME | PR_CurPol_EI | — | 5,666 | 5,366 | 4,704 | 4,252 | 4,085 | 4,012 | 3,970 | 3,907 | 3,761 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| GCAM | Baseline | 5,724 | 5,288 | 4,821 | 4,690 | 4,958 | 4,898 | 4,989 | 5,078 | 4,979 | 5,165 | 5,321 | 5,438 | 5,577 | 5,739 | 5,902 | 6,049 | 6,146 | 6,270 | 6,358 | 6,456 | Mt CO2/yr |
| GCAM | CP_EI | 5,724 | 5,288 | 4,821 | 4,584 | 4,591 | 4,405 | 4,035 | 3,737 | 3,407 | 3,103 | 2,814 | 2,539 | 2,301 | 2,093 | 1,861 | 1,636 | 1,422 | 1,215 | 1,027 | 853.50 | Mt CO2/yr |
| GCAM | PR_Baseline | 5,724 | 5,288 | 4,821 | 4,863 | 4,926 | 4,885 | 4,985 | 5,067 | 5,031 | 5,210 | 5,339 | 5,406 | 5,536 | 5,711 | 5,808 | 5,882 | 5,989 | 6,088 | 6,185 | 6,291 | Mt CO2/yr |
| GCAM | PR_NDC_EI | 5,724 | 5,288 | 4,821 | 4,715 | 3,750 | 3,241 | 2,571 | 1,930 | 1,368 | 859.19 | 449.49 | 128.67 | -113.92 | -311.00 | -458.43 | -581.09 | -682.26 | -760.49 | -819.60 | -862.26 | Mt CO2/yr |
| GEMINI-E3 | PR_Baseline | — | — | 5,247 | 4,957 | 4,823 | 4,714 | 4,525 | 4,355 | 4,110 | 3,830 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| GEMINI-E3 | PR_CurPol_CP | — | — | 5,247 | 4,956 | 4,810 | 4,694 | 4,501 | 4,332 | 4,086 | 3,805 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| GEMINI-E3 | PR_CurPol_CPO | — | — | 5,247 | 4,958 | 4,820 | 4,698 | 4,509 | 4,343 | 4,099 | 3,821 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| MUSE | PR_Baseline | — | — | 5,477 | 5,373 | 5,316 | 5,402 | 4,873 | 4,569 | 4,433 | 3,951 | 3,899 | 3,721 | 3,503 | 3,605 | 3,479 | 3,245 | 3,023 | 3,031 | 3,029 | 3,118 | Mt CO2/yr |
| MUSE | PR_CurPol_CP | — | — | 5,477 | 4,932 | 4,520 | 4,290 | 3,856 | 3,575 | 3,378 | 3,071 | 2,982 | 3,068 | 3,055 | 3,004 | 2,979 | 2,911 | 2,857 | 2,806 | 2,797 | 2,790 | Mt CO2/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q040"></a>

## 040. show data for CHN

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.33 s<br>
Conversation: standalone

**Answer**

I found the region `CHN (China)`. Which variable should I use?

<a id="q041"></a>

## 041. show data for IND

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.32 s<br>
Conversation: standalone

**Answer**

I found the region `IND (India)`. Which variable should I use?

<a id="q042"></a>

## 042. current policies emissions for EU

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.47 s<br>
Conversation: standalone

**Answer**

Interpretation: I treated bare **emissions** as `Emissions|CO2` (carbon-dioxide emissions).

### Emissions|CO2 in EU

Scope: scenario `multiple`, model `multiple`, years `2005-2100`
Unit: `Mt CO2/yr`

Answer:
In 2050, Emissions|CO2 in EU ranges from 1,024 (GCAM, PR_CurPol_CPO) to 2,633 Mt CO2/yr (E3ME, PR_CurPol_EI) across 11 series; median 1,923. Median change since 2005: -73%.

Summary: 11 series across 4 models and 3 scenarios.
Series are ordered by model and scenario.

| Model | Scenario | 2005 | 2010 | 2015 | 2020 | 2025 | 2030 | 2035 | 2040 | 2045 | 2050 | 2055 | 2060 | 2065 | 2070 | 2075 | 2080 | 2085 | 2090 | 2095 | 2100 | Unit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E3ME | PR_CurPol_CP | — | 4,304 | 3,999 | 3,444 | 3,188 | 3,033 | 2,869 | 2,705 | 2,539 | 2,359 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| E3ME | PR_CurPol_EI | — | 4,304 | 3,999 | 3,445 | 3,188 | 3,032 | 2,919 | 2,827 | 2,737 | 2,633 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| GCAM | PR_CurPol_CP | 4,190 | 3,790 | 3,490 | 3,115 | 2,687 | 2,257 | 2,112 | 1,836 | 1,485 | 1,118 | 962.69 | 912.20 | 870.84 | 863.75 | 839.93 | 771.80 | 656.18 | 503.01 | 281.84 | 10.13 | Mt CO2/yr |
| GCAM | PR_CurPol_CPO | 4,190 | 3,790 | 3,490 | 3,094 | 2,697 | 2,267 | 2,008 | 1,723 | 1,381 | 1,024 | 873.57 | 824.73 | 781.65 | 770.74 | 720.35 | 598.03 | 446.47 | 186.10 | -169.61 | -651.08 | Mt CO2/yr |
| GCAM | PR_CurPol_EI | 4,190 | 3,790 | 3,490 | 3,115 | 2,688 | 2,257 | 1,936 | 1,625 | 1,355 | 1,121 | 921.10 | 724.95 | 558.89 | 407.39 | 280.07 | 166.91 | 64.24 | -27.40 | -106.40 | -175.92 | Mt CO2/yr |
| GEMINI-E3 | PR_CurPol_CP | — | — | 3,411 | 3,197 | 3,006 | 2,733 | 2,630 | 2,545 | 2,441 | 2,342 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| GEMINI-E3 | PR_CurPol_CPO | — | — | 3,411 | 3,197 | 3,006 | 2,731 | 2,626 | 2,541 | 2,437 | 2,337 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| GEMINI-E3 | PR_CurPol_EI | — | — | 3,411 | 3,197 | 3,006 | 2,733 | 2,633 | 2,551 | 2,451 | 2,357 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| MUSE | PR_CurPol_CP | — | — | 3,118 | 3,018 | 2,956 | 2,740 | 2,581 | 2,435 | 2,155 | 1,871 | 1,641 | 1,482 | 1,414 | 1,365 | 1,237 | 1,116 | 975.01 | 910.90 | 844.23 | 817.09 | Mt CO2/yr |
| MUSE | PR_CurPol_CPO | — | — | 3,138 | 3,102 | 3,067 | 2,766 | 2,555 | 2,235 | 1,843 | 1,399 | 1,073 | 844.13 | 773.12 | 755.83 | 731.79 | 715.16 | 673.16 | 661.66 | 660.58 | 660.25 | Mt CO2/yr |
| MUSE | PR_CurPol_EI | — | — | 3,118 | 3,019 | 2,957 | 2,741 | 2,589 | 2,434 | 2,204 | 1,923 | 1,718 | 1,558 | 1,496 | 1,459 | 1,421 | 1,396 | 1,360 | 1,383 | 1,378 | 1,379 | Mt CO2/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q043"></a>

## 043. current policy CO2 for Europe

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.39 s<br>
Conversation: standalone

**Answer**

### Emissions|CO2 in EU

Scope: scenario `multiple`, model `multiple`, years `2005-2100`
Unit: `Mt CO2/yr`

Answer:
In 2050, Emissions|CO2 in EU ranges from 1,024 (GCAM, PR_CurPol_CPO) to 2,633 Mt CO2/yr (E3ME, PR_CurPol_EI) across 11 series; median 1,923. Median change since 2005: -73%.

Summary: 11 series across 4 models and 3 scenarios.
Series are ordered by model and scenario.

| Model | Scenario | 2005 | 2010 | 2015 | 2020 | 2025 | 2030 | 2035 | 2040 | 2045 | 2050 | 2055 | 2060 | 2065 | 2070 | 2075 | 2080 | 2085 | 2090 | 2095 | 2100 | Unit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E3ME | PR_CurPol_CP | — | 4,304 | 3,999 | 3,444 | 3,188 | 3,033 | 2,869 | 2,705 | 2,539 | 2,359 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| E3ME | PR_CurPol_EI | — | 4,304 | 3,999 | 3,445 | 3,188 | 3,032 | 2,919 | 2,827 | 2,737 | 2,633 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| GCAM | PR_CurPol_CP | 4,190 | 3,790 | 3,490 | 3,115 | 2,687 | 2,257 | 2,112 | 1,836 | 1,485 | 1,118 | 962.69 | 912.20 | 870.84 | 863.75 | 839.93 | 771.80 | 656.18 | 503.01 | 281.84 | 10.13 | Mt CO2/yr |
| GCAM | PR_CurPol_CPO | 4,190 | 3,790 | 3,490 | 3,094 | 2,697 | 2,267 | 2,008 | 1,723 | 1,381 | 1,024 | 873.57 | 824.73 | 781.65 | 770.74 | 720.35 | 598.03 | 446.47 | 186.10 | -169.61 | -651.08 | Mt CO2/yr |
| GCAM | PR_CurPol_EI | 4,190 | 3,790 | 3,490 | 3,115 | 2,688 | 2,257 | 1,936 | 1,625 | 1,355 | 1,121 | 921.10 | 724.95 | 558.89 | 407.39 | 280.07 | 166.91 | 64.24 | -27.40 | -106.40 | -175.92 | Mt CO2/yr |
| GEMINI-E3 | PR_CurPol_CP | — | — | 3,411 | 3,197 | 3,006 | 2,733 | 2,630 | 2,545 | 2,441 | 2,342 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| GEMINI-E3 | PR_CurPol_CPO | — | — | 3,411 | 3,197 | 3,006 | 2,731 | 2,626 | 2,541 | 2,437 | 2,337 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| GEMINI-E3 | PR_CurPol_EI | — | — | 3,411 | 3,197 | 3,006 | 2,733 | 2,633 | 2,551 | 2,451 | 2,357 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| MUSE | PR_CurPol_CP | — | — | 3,118 | 3,018 | 2,956 | 2,740 | 2,581 | 2,435 | 2,155 | 1,871 | 1,641 | 1,482 | 1,414 | 1,365 | 1,237 | 1,116 | 975.01 | 910.90 | 844.23 | 817.09 | Mt CO2/yr |
| MUSE | PR_CurPol_CPO | — | — | 3,138 | 3,102 | 3,067 | 2,766 | 2,555 | 2,235 | 1,843 | 1,399 | 1,073 | 844.13 | 773.12 | 755.83 | 731.79 | 715.16 | 673.16 | 661.66 | 660.58 | 660.25 | Mt CO2/yr |
| MUSE | PR_CurPol_EI | — | — | 3,118 | 3,019 | 2,957 | 2,741 | 2,589 | 2,434 | 2,204 | 1,923 | 1,718 | 1,558 | 1,496 | 1,459 | 1,421 | 1,396 | 1,360 | 1,383 | 1,378 | 1,379 | Mt CO2/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

Reply with a scenario, model, region, or year to narrow the answer.

<a id="q044"></a>

## 044. baseline emissions for World

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.42 s<br>
Conversation: standalone

**Answer**

Showing results from the study **Where is the world headed?**. Also available in: Post-Glasgow targets; EU path to net zero; Global impacts of NDCs. Name another study to switch.

Interpretation: I treated bare **emissions** as `Emissions|CO2` (carbon-dioxide emissions).

### Emissions|CO2 in World

Scope: scenario `multiple`, model `multiple`, years `2005-2100`
Unit: `Mt CO2/yr`

Answer:
In 2050, Emissions|CO2 in World ranges from 29,678 (MUSE, PR_Baseline) to 52,211 Mt CO2/yr (GEMINI-E3, PR_Baseline) across 8 series; median 43,333. Median change since 2005: +47%.

Summary: 8 series across 4 models and 4 scenarios.
Series are ordered by model and scenario.

| Model | Scenario | 2005 | 2010 | 2015 | 2020 | 2025 | 2030 | 2035 | 2040 | 2045 | 2050 | 2055 | 2060 | 2065 | 2070 | 2075 | 2080 | 2085 | 2090 | 2095 | 2100 | Unit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E3ME | PR_Baseline | — | 31,521 | 34,433 | 34,415 | 35,386 | 36,934 | 38,149 | 38,757 | 39,071 | 40,393 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| E3ME | Unharmonised baseline | — | 31,521 | 34,433 | 34,430 | 35,463 | 37,096 | 38,457 | 39,232 | 39,600 | 41,071 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| GCAM | Baseline | 31,426 | 36,147 | 39,196 | 38,300 | 40,654 | 42,107 | 43,979 | 45,315 | 45,465 | 45,595 | 46,708 | 48,206 | 49,791 | 51,521 | 53,322 | 54,770 | 56,170 | 57,563 | 58,673 | 60,068 | Mt CO2/yr |
| GCAM | PR_Baseline | 31,426 | 36,147 | 39,196 | 39,429 | 41,308 | 42,663 | 44,466 | 45,961 | 46,635 | 46,979 | 48,176 | 49,647 | 51,430 | 53,411 | 54,919 | 56,024 | 57,552 | 59,032 | 60,469 | 62,101 | Mt CO2/yr |
| GEMINI-E3 | Baseline | — | — | 34,738 | 34,934 | 38,132 | 42,037 | 44,008 | 45,532 | 46,413 | 46,713 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| GEMINI-E3 | PR_Baseline | — | — | 34,859 | 37,813 | 41,214 | 44,462 | 46,876 | 48,984 | 50,715 | 52,211 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| MUSE | baseline | — | — | 36,386 | 37,813 | 37,955 | 35,891 | 34,436 | 33,696 | 32,387 | 31,448 | 30,979 | 31,285 | 32,420 | 33,030 | 34,518 | 35,378 | 36,141 | 36,237 | 37,010 | 37,729 | Mt CO2/yr |
| MUSE | PR_Baseline | — | — | 36,431 | 38,836 | 40,158 | 41,521 | 38,898 | 36,732 | 33,730 | 29,678 | 26,902 | 24,364 | 23,828 | 24,252 | 24,392 | 24,597 | 24,559 | 24,441 | 24,608 | 24,998 | Mt CO2/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

Reply with a scenario, model, region, or year to narrow the answer.

<a id="q045"></a>

## 045. business as usual emissions for World

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.42 s<br>
Conversation: standalone

**Answer**

Showing results from the study **Where is the world headed?**. Also available in: Post-Glasgow targets; EU path to net zero; Global impacts of NDCs. Name another study to switch.

Interpretation: I treated bare **emissions** as `Emissions|CO2` (carbon-dioxide emissions).

### Emissions|CO2 in World

Scope: scenario `multiple`, model `multiple`, years `2005-2100`
Unit: `Mt CO2/yr`

Answer:
In 2050, Emissions|CO2 in World ranges from 29,678 (MUSE, PR_Baseline) to 52,211 Mt CO2/yr (GEMINI-E3, PR_Baseline) across 8 series; median 43,333. Median change since 2005: +47%.

Summary: 8 series across 4 models and 4 scenarios.
Series are ordered by model and scenario.

| Model | Scenario | 2005 | 2010 | 2015 | 2020 | 2025 | 2030 | 2035 | 2040 | 2045 | 2050 | 2055 | 2060 | 2065 | 2070 | 2075 | 2080 | 2085 | 2090 | 2095 | 2100 | Unit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E3ME | PR_Baseline | — | 31,521 | 34,433 | 34,415 | 35,386 | 36,934 | 38,149 | 38,757 | 39,071 | 40,393 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| E3ME | Unharmonised baseline | — | 31,521 | 34,433 | 34,430 | 35,463 | 37,096 | 38,457 | 39,232 | 39,600 | 41,071 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| GCAM | Baseline | 31,426 | 36,147 | 39,196 | 38,300 | 40,654 | 42,107 | 43,979 | 45,315 | 45,465 | 45,595 | 46,708 | 48,206 | 49,791 | 51,521 | 53,322 | 54,770 | 56,170 | 57,563 | 58,673 | 60,068 | Mt CO2/yr |
| GCAM | PR_Baseline | 31,426 | 36,147 | 39,196 | 39,429 | 41,308 | 42,663 | 44,466 | 45,961 | 46,635 | 46,979 | 48,176 | 49,647 | 51,430 | 53,411 | 54,919 | 56,024 | 57,552 | 59,032 | 60,469 | 62,101 | Mt CO2/yr |
| GEMINI-E3 | Baseline | — | — | 34,738 | 34,934 | 38,132 | 42,037 | 44,008 | 45,532 | 46,413 | 46,713 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| GEMINI-E3 | PR_Baseline | — | — | 34,859 | 37,813 | 41,214 | 44,462 | 46,876 | 48,984 | 50,715 | 52,211 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| MUSE | baseline | — | — | 36,386 | 37,813 | 37,955 | 35,891 | 34,436 | 33,696 | 32,387 | 31,448 | 30,979 | 31,285 | 32,420 | 33,030 | 34,518 | 35,378 | 36,141 | 36,237 | 37,010 | 37,729 | Mt CO2/yr |
| MUSE | PR_Baseline | — | — | 36,431 | 38,836 | 40,158 | 41,521 | 38,898 | 36,732 | 33,730 | 29,678 | 26,902 | 24,364 | 23,828 | 24,252 | 24,392 | 24,597 | 24,559 | 24,441 | 24,608 | 24,998 | Mt CO2/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

Reply with a scenario, model, region, or year to narrow the answer.

<a id="q046"></a>

## 046. BAU carbon emissions for World

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.40 s<br>
Conversation: standalone

**Answer**

I could not find data for `Emissions|CO2` in `World` under `BAU`.

Reason: `Emissions|CO2` in `World` is not reported under `BAU`.

Closest valid options:

1. scenario `Baseline` (baseline)

2. scenario `Baseline|Harmonized` (baseline)

3. scenario `CP_EI`

Reply with `1`, `2`, or `3`, or type the scenario you want.

<a id="q047"></a>

## 047. NDC emissions for World

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.39 s<br>
Conversation: standalone

**Answer**

Showing results from the study **Where is the world headed?**. Also available in: Post-Glasgow targets; Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU; Global impacts of NDCs; Tech-constrained EU pathways to net zero. Name another study to switch.

Interpretation: I treated bare **emissions** as `Emissions|CO2` (carbon-dioxide emissions).

### Emissions|CO2 in World

Scope: scenario `multiple`, model `multiple`, years `2005-2100`
Unit: `Mt CO2/yr`

Answer:
In 2050, Emissions|CO2 in World ranges from 9,173 (GCAM, NDC_LTT) to 41,731 Mt CO2/yr (GEMINI-E3, PR_NDC_CP) across 16 series; median 26,345. Median change since 2005: -10%.

Summary: 16 series across 4 models and 5 scenarios.
Showing 12 of 16 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 4 remaining series.

| Model | Scenario | 2005 | 2010 | 2015 | 2020 | 2025 | 2030 | 2035 | 2040 | 2045 | 2050 | 2055 | 2060 | 2065 | 2070 | 2075 | 2080 | 2085 | 2090 | 2095 | 2100 | Unit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E3ME | PR_NDC_CP | — | 31,521 | 34,433 | 34,381 | 34,617 | 35,314 | 35,532 | 34,813 | 33,660 | 33,018 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| E3ME | PR_NDC_EI | — | 31,521 | 34,433 | 34,372 | 34,583 | 35,307 | 35,537 | 35,823 | 35,602 | 35,197 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| GCAM | NDC_EI | 31,426 | 36,147 | 39,196 | 37,250 | 35,865 | 32,324 | 28,963 | 25,557 | 21,841 | 18,332 | 15,580 | 13,341 | 11,396 | 9,677 | 8,137 | 6,717 | 5,482 | 4,442 | 3,571 | 2,842 | Mt CO2/yr |
| GCAM | NDC_LTT | 31,426 | 36,147 | 39,196 | 37,250 | 35,865 | 32,324 | 26,749 | 20,969 | 14,887 | 9,173 | 4,316 | 871.19 | 669.19 | 59.16 | -310.14 | -114.52 | -451.30 | -589.76 | -661.35 | -766.86 | Mt CO2/yr |
| GCAM | PR_NDC_CP | 31,426 | 36,147 | 39,196 | 36,891 | 36,234 | 35,221 | 35,376 | 35,480 | 34,601 | 33,106 | 33,030 | 33,339 | 34,200 | 34,773 | 35,054 | 35,030 | 35,032 | 34,802 | 34,441 | 33,950 | Mt CO2/yr |
| GCAM | PR_NDC_CPO | 31,426 | 36,147 | 39,196 | 36,930 | 36,269 | 35,279 | 35,967 | 35,995 | 35,065 | 33,462 | 33,242 | 33,471 | 34,195 | 34,752 | 35,020 | 35,015 | 35,044 | 34,899 | 34,567 | 34,082 | Mt CO2/yr |
| GEMINI-E3 | NDC_EI | — | — | 34,738 | 34,934 | 33,216 | 32,585 | 30,352 | 28,088 | 25,839 | 23,470 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| GEMINI-E3 | NDC_LTT | — | — | 34,738 | 34,934 | 33,264 | 32,585 | 27,597 | 22,434 | 17,208 | 12,090 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| GEMINI-E3 | PR_NDC_CP | — | — | 34,859 | 36,411 | 36,477 | 36,801 | 37,916 | 39,585 | 40,741 | 41,731 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| MUSE | NDC_EI | — | — | 36,381 | 35,946 | 34,574 | 31,089 | 27,917 | 24,700 | 22,635 | 19,999 | 18,772 | 17,549 | 17,234 | 16,328 | 15,290 | 14,893 | 14,669 | 14,811 | 15,232 | 15,044 | Mt CO2/yr |
| MUSE | NDC_LTT | — | — | 36,381 | 35,954 | 34,567 | 30,809 | 25,976 | 21,616 | 17,943 | 13,164 | 10,260 | 7,287 | 6,283 | 5,079 | 3,632 | 2,855 | 3,569 | 4,166 | 4,822 | 5,288 | Mt CO2/yr |
| MUSE | PR_NDC_CP | — | — | 36,322 | 37,758 | 37,803 | 36,999 | 34,280 | 31,704 | 29,062 | 25,043 | 22,018 | 19,751 | 19,638 | 20,092 | 20,510 | 20,956 | 21,256 | 21,873 | 21,961 | 22,243 | Mt CO2/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q048"></a>

## 048. nationally determined contributions emissions

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.60 s<br>
Conversation: standalone

**Answer**

Showing results from the study **Where is the world headed?**. Also available in: Post-Glasgow targets; Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU; Global impacts of NDCs; Tech-constrained EU pathways to net zero; Behavioural changes for climate policy. Name another study to switch.

Interpretation: I treated bare **emissions** as `Emissions|CO2` (carbon-dioxide emissions).

### Emissions|CO2 in World

Scope: scenario `multiple`, model `multiple`, years `2005-2100`
Unit: `Mt CO2/yr`

Answer:
In 2050, Emissions|CO2 in World ranges from 9,173 (GCAM, NDC_LTT) to 41,731 Mt CO2/yr (GEMINI-E3, PR_NDC_CP) across 16 series; median 26,345. Median change since 2005: -10%.

Summary: 16 series across 4 models and 5 scenarios.
Showing 12 of 16 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 4 remaining series.

| Model | Scenario | 2005 | 2010 | 2015 | 2020 | 2025 | 2030 | 2035 | 2040 | 2045 | 2050 | 2055 | 2060 | 2065 | 2070 | 2075 | 2080 | 2085 | 2090 | 2095 | 2100 | Unit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E3ME | PR_NDC_CP | — | 31,521 | 34,433 | 34,381 | 34,617 | 35,314 | 35,532 | 34,813 | 33,660 | 33,018 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| E3ME | PR_NDC_EI | — | 31,521 | 34,433 | 34,372 | 34,583 | 35,307 | 35,537 | 35,823 | 35,602 | 35,197 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| GCAM | NDC_EI | 31,426 | 36,147 | 39,196 | 37,250 | 35,865 | 32,324 | 28,963 | 25,557 | 21,841 | 18,332 | 15,580 | 13,341 | 11,396 | 9,677 | 8,137 | 6,717 | 5,482 | 4,442 | 3,571 | 2,842 | Mt CO2/yr |
| GCAM | NDC_LTT | 31,426 | 36,147 | 39,196 | 37,250 | 35,865 | 32,324 | 26,749 | 20,969 | 14,887 | 9,173 | 4,316 | 871.19 | 669.19 | 59.16 | -310.14 | -114.52 | -451.30 | -589.76 | -661.35 | -766.86 | Mt CO2/yr |
| GCAM | PR_NDC_CP | 31,426 | 36,147 | 39,196 | 36,891 | 36,234 | 35,221 | 35,376 | 35,480 | 34,601 | 33,106 | 33,030 | 33,339 | 34,200 | 34,773 | 35,054 | 35,030 | 35,032 | 34,802 | 34,441 | 33,950 | Mt CO2/yr |
| GCAM | PR_NDC_CPO | 31,426 | 36,147 | 39,196 | 36,930 | 36,269 | 35,279 | 35,967 | 35,995 | 35,065 | 33,462 | 33,242 | 33,471 | 34,195 | 34,752 | 35,020 | 35,015 | 35,044 | 34,899 | 34,567 | 34,082 | Mt CO2/yr |
| GEMINI-E3 | NDC_EI | — | — | 34,738 | 34,934 | 33,216 | 32,585 | 30,352 | 28,088 | 25,839 | 23,470 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| GEMINI-E3 | NDC_LTT | — | — | 34,738 | 34,934 | 33,264 | 32,585 | 27,597 | 22,434 | 17,208 | 12,090 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| GEMINI-E3 | PR_NDC_CP | — | — | 34,859 | 36,411 | 36,477 | 36,801 | 37,916 | 39,585 | 40,741 | 41,731 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| MUSE | NDC_EI | — | — | 36,381 | 35,946 | 34,574 | 31,089 | 27,917 | 24,700 | 22,635 | 19,999 | 18,772 | 17,549 | 17,234 | 16,328 | 15,290 | 14,893 | 14,669 | 14,811 | 15,232 | 15,044 | Mt CO2/yr |
| MUSE | NDC_LTT | — | — | 36,381 | 35,954 | 34,567 | 30,809 | 25,976 | 21,616 | 17,943 | 13,164 | 10,260 | 7,287 | 6,283 | 5,079 | 3,632 | 2,855 | 3,569 | 4,166 | 4,822 | 5,288 | Mt CO2/yr |
| MUSE | PR_NDC_CP | — | — | 36,322 | 37,758 | 37,803 | 36,999 | 34,280 | 31,704 | 29,062 | 25,043 | 22,018 | 19,751 | 19,638 | 20,092 | 20,510 | 20,956 | 21,256 | 21,873 | 21,961 | 22,243 | Mt CO2/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q049"></a>

## 049. net zero emissions for Europe

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.45 s<br>
Conversation: standalone

**Answer**

I could not find data for `Emissions|CO2` in region `EU (European Union)` under the requested scenario family `Net Zero`.

Reason: the exact scenario combination is not available for this data slice.

Closest valid options:

1. region `AT` (AT (Austria))

2. region `BE` (BE (Belgium))

3. region `BG` (BG (Bulgaria))

Reply with `1`, `2`, or `3`, or type the region you want.

<a id="q050"></a>

## 050. show net-zero target results

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 1.37 s<br>
Conversation: standalone

**Answer**

### EU path to net zero

Available in this study:
- **Topics (710 variables):** Final Energy (156), Emissions (104), Secondary Energy (70), Primary Energy (51), Price (37), Capacity (34), Final Energy (excl. feedstocks) (32), Investment (29) and 30 more
- **Example variables:** `Final Energy`, `Emissions|CO`, `Secondary Energy`, `Primary Energy`, `Price|Carbon`
- **Scenarios (7):** `Baseline`, `CP_EI`, `NZE_Bench_H`, `NZE_Bench_L`, `NZE_Bench_M`, `NZE_EUPol_Stand`, `WWH21`
- **Models (9):** ALADIN, E4SMA-EU-TIMES 1.0, FORECAST, GCAM, GCAM-PR 5.3, GEMINI-E3, GEMINI-E3 7.0, ICES-XPS 1.0 and 1 more
- **Regions (38):** World, AT, BE, BE_LU_NL, BG, CH and 32 more
- **Years:** 2005–2100

Explore this study:
- [Open the data explorer](https://iamparis.eu/results/paris-reinforce/eu-path-to-net-zero/graphs)
- [Read the study explainer](https://iamparis.eu/results/paris-reinforce/eu-path-to-net-zero/policy_questions)

<a id="q051"></a>

## 051. Fit for 55 policy questions

Category: data and scenarios<br>
Route: `general_qa` (explicit site navigation request)<br>
Qwen generations: 0<br>
Time: 0.06 s<br>
Conversation: standalone

**Answer**

Use these IAM PARIS links for this request: [Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU](https://iamparis.eu/results/iam-compact/comparison-of-fit-for-55-policy-and-cost-optimal-scenarios-for-the-eu/policy_questions).

<a id="q052"></a>

## 052. IAM COMPACT renewable energy metrics

Category: data and scenarios<br>
Route: `general_qa` (explicit site navigation request)<br>
Qwen generations: 0<br>
Time: 0.05 s<br>
Conversation: standalone

**Answer**

Use these IAM PARIS links for this request: [IAM COMPACT](https://iamparis.eu/results/iam-compact).

<a id="q053"></a>

## 053. post Glasgow climate results

Category: data and scenarios<br>
Route: `general_qa` (explicit site navigation request)<br>
Qwen generations: 0<br>
Time: 0.42 s<br>
Conversation: standalone

**Answer**

### Post-Glasgow targets

Available in this study:
- **Topics (1001 variables):** Final Energy (614), Emissions (104), Secondary Energy (59), Primary Energy (38), Investment (32), Capacity (26), Capacity Additions (26), Price (26) and 18 more
- **Example variables:** `Final Energy`, `Emissions|BC`, `Secondary Energy`, `Primary Energy`, `Investment`
- **Scenarios (10):** `Baseline`, `Baseline|Harmonized`, `CP_EI`, `CP_EI|Harmonized`, `NDC_EI`, `NDC_EI|Harmonized`, `NDC_LTS|Harmonized`, `NDC_LTT`, `baseline`, `baseline|Harmonized`
- **Models (4):** GCAM-PR 5.3, GEMINI-E3 7.0, MUSE, TIAM_Grantham
- **Regions (54):** World, AFR, Africa_Eastern, Africa_Northern, Africa_Southern, Africa_Western and 48 more
- **Years:** 2005–2100

Explore this study:
- [Open the data explorer](https://iamparis.eu/results/paris-reinforce/post-glasgow-targets/graphs)
- [Read the study explainer](https://iamparis.eu/results/paris-reinforce/post-glasgow-targets/policy_questions)

<a id="q054"></a>

## 054. steel relocation EU results

Category: data and scenarios<br>
Route: `general_qa` (explicit site navigation request)<br>
Qwen generations: 0<br>
Time: 0.05 s<br>
Conversation: standalone

**Answer**

Use these IAM PARIS links for this request: [Implications of EU steel industry relocation](https://iamparis.eu/results/iam-compact/implications-of-eu-steel-industry-relocation/policy_questions).

<a id="q055"></a>

## 055. behavioural change climate policy

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 2.28 s<br>
Conversation: standalone

**Answer**

I need one more detail. Which variable should I use?

<a id="q056"></a>

## 056. NDC ASPECTS buildings results

Category: data and scenarios<br>
Route: `general_qa` (explicit site navigation request)<br>
Qwen generations: 0<br>
Time: 0.06 s<br>
Conversation: standalone

**Answer**

Use these IAM PARIS links for this request: [NDC ASPECTS](https://iamparis.eu/results/ndc-aspects).

<a id="q057"></a>

## 057. long term targets scenario results

Category: data and scenarios<br>
Route: `general_qa` (explicit site navigation request)<br>
Qwen generations: 0<br>
Time: 0.05 s<br>
Conversation: standalone

**Answer**

Use these IAM PARIS links for this request: [Results](https://iamparis.eu/results).

<a id="q058"></a>

## 058. tell me about MESSAGEix model

Category: model descriptions<br>
Route: `model_explanation` (model information request)<br>
Qwen generations: 0<br>
Time: 0.31 s<br>
Conversation: standalone

**Answer**

### MESSAGEix-GLOBIOM

Description:
MESSAGEix-GLOBIOM is an integrated assessment framework combining the MESSAGEix energy-system model with GLOBIOM land-use and agriculture components. It is used for energy, land, emissions and climate mitigation pathway analysis.

Model scope:
- energy
- land use
- agriculture
- emissions
- climate

Useful for:
- energy-land-climate mitigation pathways
- technology and resource constraints
- emissions trajectories across sectors
- scenario analysis for climate policy

Interpretation notes:
- results depend on scenario protocol and regional aggregation
- local IAM PARIS metadata may not expose all model internals

Related model documentation:
- [IAM PARIS Models](https://iamparis.eu/models)
- Open the Models page and search for: `MESSAGEix`

<a id="q059"></a>

## 059. MESSAGE IX assumptions

Category: model descriptions<br>
Route: `model_explanation` (named-model metadata request)<br>
Qwen generations: 0<br>
Time: 0.08 s<br>
Conversation: standalone

**Answer**

### MESSAGEix-GLOBIOM

Description:
MESSAGEix-GLOBIOM is an integrated assessment framework combining the MESSAGEix energy-system model with GLOBIOM land-use and agriculture components. It is used for energy, land, emissions and climate mitigation pathway analysis.

Model scope:
- energy
- land use
- agriculture
- emissions
- climate

Useful for:
- energy-land-climate mitigation pathways
- technology and resource constraints
- emissions trajectories across sectors
- scenario analysis for climate policy

Assumptions:
No explicit MESSAGEix-GLOBIOM assumptions field is available in the local IAM PARIS model metadata. Use the scenario name, workspace and result metadata to interpret assumptions for a specific output.

Interpretation notes:
- results depend on scenario protocol and regional aggregation
- local IAM PARIS metadata may not expose all model internals

Related model documentation:
- [IAM PARIS Models](https://iamparis.eu/models)
- Open the Models page and search for: `MESSAGEix`

<a id="q060"></a>

## 060. tell me about REMIND model

Category: model descriptions<br>
Route: `model_explanation` (model information request)<br>
Qwen generations: 0<br>
Time: 0.25 s<br>
Conversation: standalone

**Answer**

There is no exact `REMIND` entry in the loaded IAM PARIS model catalogue. Showing the closest entry, `REMIND-MFA`; check that this is the model you mean.

### REMIND-MFA

Description:
REMIND-MFA includes top-down prospective material flow analysis (MFA) models for the basic materials cement, plastics, and steel. It is designed to provide material demands and flows for the integrated assessment model REMIND.

Related model documentation:
- [IAM PARIS model page](https://iamparis.eu/models/87)

<a id="q061"></a>

## 061. tell me about WITCH model

Category: model descriptions<br>
Route: `model_explanation` (model information request)<br>
Qwen generations: 0<br>
Time: 0.29 s<br>
Conversation: standalone

**Answer**

### WITCH

Description:
WITCH is an integrated assessment model focused on the interaction between the economy, energy system, climate policy and technological change. It is often used for regional mitigation strategies, policy design and technology innovation analysis.

Model scope:
- economy
- energy
- climate policy
- technology innovation

Useful for:
- regional mitigation and burden-sharing analysis
- carbon pricing and policy assessment
- technology innovation and diffusion scenarios
- long-term climate policy pathways

Interpretation notes:
- assumptions are experiment-specific
- IAM PARIS local metadata may not include full methodology notes

Related model documentation:
- [IAM PARIS Models](https://iamparis.eu/models)
- Open the Models page and search for: `WITCH`

<a id="q062"></a>

## 062. tell me about PROMETHEUS

Category: model descriptions<br>
Route: `model_explanation` (model information request)<br>
Qwen generations: 0<br>
Time: 0.25 s<br>
Conversation: standalone

**Answer**

### PROMETHEUS

Description:
The Prometheus model is a comprehensive world energy model with innovative features: it integrates stochastic relations and so all exogenous variables – parameters are stochastic, following explicit probability distributions, including covariance. Therefore, all model results are also stochastic (probability distributions). Among others, stochastic distributions are used for technology change and learning by doing, as well as for fossil fuel resources, both conventional and unconventional. Prometheus models the world energy system divided into 10 regions and produces yearly projections up to 2050; the update to 2070 is ongoing. The model includes a set of long time series (IEA data mostly) on which econometric estimations are carried out. The model produces projections of energy demand by sector (industry, domestic, transport), power generation (representing about 25 technologies), RES, and hydrogen supply and use. Prometheus puts emphasis on oil and gas resources, while coal is assumed to have rather abundant supplies relative to production prospects in the projection time horizon. The model incorporates uncertainty surrounding the amount of oil and gas resources that are yet to be discovered. Data are mostly calibrated on estimation of resources by USGS. The model produces projections of fossil fuel prices, which depend on demand, supply, technology and resources. As all outputs are probabilistic, fossil fuel price projections take the form of probability distributions. The model includes CO2 emissions and can simulate emission reduction pathways.

Related model documentation:
- [IAM PARIS model page](https://iamparis.eu/models/55)

<a id="q063"></a>

## 063. tell me about LEAP

Category: model descriptions<br>
Route: `model_explanation` (model information request)<br>
Qwen generations: 0<br>
Time: 0.24 s<br>
Conversation: standalone

**Answer**

### LEAP

Description:
The Long-range Energy Alternatives Planning system (LEAP) is a scenario-based energy-environment modeling tool. Its scenarios are based on comprehensive accounting of how energy is consumed, converted and produced in a given region or economy under a range of alternative assumptions on population, economic development, technology, price and so on.

Related model documentation:
- [IAM PARIS model page](https://iamparis.eu/models/38)

<a id="q064"></a>

## 064. tell me about GCAM PR

Category: model descriptions<br>
Route: `model_explanation` (model information request)<br>
Qwen generations: 0<br>
Time: 0.23 s<br>
Conversation: standalone

**Answer**

### GCAM

Description:
GCAM is a dynamic-recursive model with technology-rich representations of the economy, energy sector, land use and water linked to a climate model that can be used to explore climate change mitigation policies including carbon taxes, carbon trading, regulations and accelerated deployment of energy technology. Regional population and labour productivity growth assumptions drive the energy, land-use and water systems employing numerous technology options to produce, transform and provide energy services, as well as to produce agriculture and forest products, and to determine land use and land use cover.

Related model documentation:
- [IAM PARIS model page](https://iamparis.eu/models/30)

<a id="q065"></a>

## 065. show data using MESSAGEix

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.58 s<br>
Conversation: standalone

**Answer**

I need one more detail. Which variable should I use?

<a id="q066"></a>

## 066. show data using PROMETHEUS

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.45 s<br>
Conversation: standalone

**Answer**

I found the region `USA (United States)`. Which variable should I use?

<a id="q067"></a>

## 067. compare GCAM and WITCH emissions

Category: data and scenarios<br>
Route: `data_query` (chat plotting disabled; return numeric table)<br>
Qwen generations: 0<br>
Time: 0.56 s<br>
Conversation: standalone

**Answer**

Showing results from the study **Where is the world headed?**. Also available in: EU path to net zero. Name another study to switch.

Interpretation: I treated bare **emissions** as `Emissions|CO2` (carbon-dioxide emissions).

### Emissions|CO2 in World

Scope: scenario `multiple`, model `GCAM`, years `2005-2100`
Unit: `Mt CO2/yr`

Answer:
In 2100, Emissions|CO2 in World ranges from -766.86 (GCAM, NDC_LTT) to 62,101 Mt CO2/yr (GCAM, PR_Baseline) across 11 series; median 33,950. Median change since 2005: +8%.

Summary: 11 series across 1 model and 11 scenarios.
Series are ordered by model and scenario.

| Model | Scenario | 2005 | 2010 | 2015 | 2020 | 2025 | 2030 | 2035 | 2040 | 2045 | 2050 | 2055 | 2060 | 2065 | 2070 | 2075 | 2080 | 2085 | 2090 | 2095 | 2100 | Unit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GCAM | Baseline | 31,426 | 36,147 | 39,196 | 38,300 | 40,654 | 42,107 | 43,979 | 45,315 | 45,465 | 45,595 | 46,708 | 48,206 | 49,791 | 51,521 | 53,322 | 54,770 | 56,170 | 57,563 | 58,673 | 60,068 | Mt CO2/yr |
| GCAM | CP_EI | 31,426 | 36,147 | 39,196 | 37,250 | 37,566 | 37,516 | 35,676 | 33,501 | 31,082 | 28,519 | 26,244 | 24,349 | 22,565 | 20,869 | 19,172 | 17,514 | 15,927 | 14,464 | 13,087 | 11,834 | Mt CO2/yr |
| GCAM | NDC_EI | 31,426 | 36,147 | 39,196 | 37,250 | 35,865 | 32,324 | 28,963 | 25,557 | 21,841 | 18,332 | 15,580 | 13,341 | 11,396 | 9,677 | 8,137 | 6,717 | 5,482 | 4,442 | 3,571 | 2,842 | Mt CO2/yr |
| GCAM | NDC_LTT | 31,426 | 36,147 | 39,196 | 37,250 | 35,865 | 32,324 | 26,749 | 20,969 | 14,887 | 9,173 | 4,316 | 871.19 | 669.19 | 59.16 | -310.14 | -114.52 | -451.30 | -589.76 | -661.35 | -766.86 | Mt CO2/yr |
| GCAM | PR_Baseline | 31,426 | 36,147 | 39,196 | 39,429 | 41,308 | 42,663 | 44,466 | 45,961 | 46,635 | 46,979 | 48,176 | 49,647 | 51,430 | 53,411 | 54,919 | 56,024 | 57,552 | 59,032 | 60,469 | 62,101 | Mt CO2/yr |
| GCAM | PR_CurPol_CP | 31,426 | 36,147 | 39,196 | 36,891 | 37,709 | 38,074 | 38,798 | 39,485 | 39,081 | 38,385 | 38,926 | 39,800 | 41,260 | 42,548 | 43,526 | 44,277 | 45,175 | 45,925 | 46,625 | 47,266 | Mt CO2/yr |
| GCAM | PR_CurPol_CPO | 31,426 | 36,147 | 39,196 | 36,935 | 37,721 | 38,288 | 39,098 | 39,709 | 39,235 | 38,523 | 39,108 | 39,865 | 41,255 | 42,501 | 43,408 | 44,090 | 44,971 | 45,679 | 46,314 | 46,840 | Mt CO2/yr |
| GCAM | PR_CurPol_EI | 31,426 | 36,147 | 39,196 | 36,902 | 37,668 | 38,059 | 38,421 | 37,896 | 37,091 | 36,082 | 35,311 | 34,135 | 33,100 | 32,024 | 30,910 | 29,685 | 28,298 | 26,863 | 25,376 | 23,919 | Mt CO2/yr |
| GCAM | PR_NDC_CP | 31,426 | 36,147 | 39,196 | 36,891 | 36,234 | 35,221 | 35,376 | 35,480 | 34,601 | 33,106 | 33,030 | 33,339 | 34,200 | 34,773 | 35,054 | 35,030 | 35,032 | 34,802 | 34,441 | 33,950 | Mt CO2/yr |
| GCAM | PR_NDC_CPO | 31,426 | 36,147 | 39,196 | 36,930 | 36,269 | 35,279 | 35,967 | 35,995 | 35,065 | 33,462 | 33,242 | 33,471 | 34,195 | 34,752 | 35,020 | 35,015 | 35,044 | 34,899 | 34,567 | 34,082 | Mt CO2/yr |
| GCAM | PR_NDC_EI | 31,426 | 36,147 | 39,196 | 36,891 | 36,232 | 35,176 | 34,076 | 32,310 | 30,369 | 28,158 | 26,286 | 24,304 | 22,552 | 20,863 | 19,321 | 17,751 | 16,146 | 14,613 | 13,142 | 11,757 | Mt CO2/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

Reply with a scenario, model, region, or year to narrow the answer.

<a id="q068"></a>

## 068. compare MESSAGE and REMIND carbon emissions

Category: data and scenarios<br>
Route: `data_query` (chat plotting disabled; return numeric table)<br>
Qwen generations: 0<br>
Time: 0.88 s<br>
Conversation: standalone

**Answer**

I could not find `Emissions|CO2` for the explicitly requested model `MESSAGEix-GLOBIOM`. I did not substitute results from a different model. Models that do have this exact slice include: E3ME, E3ME 6.1, E4SMA-EU-TIMES 1.0, EnergyPLAN, GCAM.

<a id="q069"></a>

## 069. compare solar and wind capacity

Category: data and scenarios<br>
Route: `data_query` (chat plotting disabled; return numeric table)<br>
Qwen generations: 0<br>
Time: 0.19 s<br>
Conversation: standalone

**Answer**

Please provide the variable and region for the numeric comparison.

<a id="q070"></a>

## 070. policy catalogue climate policies

Category: data and scenarios<br>
Route: `general_qa` (explicit site navigation request)<br>
Qwen generations: 0<br>
Time: 0.07 s<br>
Conversation: standalone

**Answer**

Use these IAM PARIS links for this request: [Policy Catalogue Interactive Explorer](https://iamparis.eu/datastories/policyCatalog).

<a id="q071"></a>

## 071. NDC impacts for transport and buildings

Category: data and scenarios<br>
Route: `data_query` (low confidence entity clarification)<br>
Qwen generations: 0<br>
Time: 3.12 s<br>
Conversation: standalone

**Answer**

I could not confidently match all of the requested wording. The loaded variable catalogue does not represent these term(s) in the top match: `impacts`, `transport`. Choose the variable: 1. `Emissions|Kyoto Gases|Buildings` (emissions, buildings, gas) 2. `Final Energy|Buildings|Residential` (final energy use, buildings) 3. `Benchmarking|Buildings|CO2 intensity` (buildings) Reply with a number (1-3), or `yes` for option 1.

<a id="q072"></a>

## 072. compare with baseline

Category: data and scenarios<br>
Route: `data_query` (extracted data entities)<br>
Qwen generations: 0<br>
Time: 2.53 s<br>
Conversation: standalone

**Answer**

I can answer questions about IAM PARIS results, models and climate-policy terms, but I need a little more to go on. Try for example:
- `CO2 emissions for World in 2050`
- `which scenario has the lowest emissions in 2050?`
- `which models cover transport?`
- `what does net zero mean?`

<a id="q073"></a>

## 073. show all scenarios

Category: data and scenarios<br>
Route: `data_query` (scenario availability request)<br>
Qwen generations: 0<br>
Time: 0.19 s<br>
Conversation: standalone

**Answer**

I found scenarios like 1.5 TECH, 1.5 °C DRI-trade, 1.5 °C Domestic, BAU, BAU_CL1, BAU_CL2, BAU_CL3, BAU_Familiar, Baseline, Baseline_HP, Baseline_High_Adapt vs Baseline, Baseline_High_Adapt_Biod vs Baseline_High_Adapt, Baseline_Low_Adapt vs Baseline, Baseline|Harmonized, CBAM, COVID Recovery - Average portfolio, COVID Recovery - Max Emissions Cuts, COVID Recovery - Max Jobs 2025, COVID Recovery - Max Jobs 2030, CP_EI, CP_EI_baseWACC, CP_EI|Harmonized, Centralized_Unified, Cluster 1, Cluster 2, Cluster 3, Cluster 4, Cons_All, Cons_NoBiomass, Cons_NoFallow, Cons_NoForest, Current Ambition, Current_ambition, DECARB_HELEC, DECARB_HP_HE, DECARB_HP_HELEC, DECARB_HP_HELEC_ETS, Decarb_HE, Decentralized_Familiar, Decentralized_Unified, Fossil_NDC, Gas_Fragmented, INDEPENDENCE, Increased_ambition, Lignite_Fragmented, NDC_EI, NDC_EI_baseWACC, NDC_EI|Harmonized, NDC_LTS vs Baseline, NDC_LTS_High_Adapt vs NDC_LTS, NDC_LTS_High_Adapt_Biod vs NDC_LTS_High_Adapt, NDC_LTS_Low_Adapt vs NDC_LTS, NDC_LTS|Harmonized, NDC_LTT, NDC_LTT_LED, NDC_LTT_LIFE, NDC_LTT_LimBio, NDC_LTT_LimCCS, NDC_LTT_NoRus, NDC_LTT_TRANS, NDC_LTT_limBIO, NDC_LTT_limCCS, NDC_LTT_limRES, NDC_LTT_limres, NDC_LTT_norusimp, NZE, NZE_Bench_H, NZE_Bench_L, NZE_Bench_M, NZE_EUPol_Stand, NonFossil_NDC, PR_Baseline, PR_CurPol_CP, PR_CurPol_CPO, PR_CurPol_EI, PR_NDC_CP, PR_NDC_CPO, PR_NDC_EI, PR_WWH_CP, PR_WWH_NDC, Reference, Smart Energy Europe, Uncons, Unharmonised baseline, WWH, WWH21, With LEV Substitution, baseline, baseline|Harmonized, cluster_1_FiT, cluster_1_NEM, cluster_1_NEMBESS, cluster_2_FiT, cluster_2_NEM, cluster_2_NEMBESS, cluster_3_FiT, cluster_3_NEM, cluster_3_NEMBESS, cluster_4_FiT, cluster_4_NEM, cluster_4_NEMBESS, decarb_helec_ets, familiar_world, familiar_world_FiT, familiar_world_NEM, familiar_world_NEMBESS, fragmented_world, fragmented_world_FiT, fragmented_world_NEM, fragmented_world_NEMBESS, total_FiT, total_NEM, total_NEMBESS, unified_world, unified_world_FiT, unified_world_NEM and unified_world_NEMBESS. Ask for a variable under any of these scenarios to see its values.

Reply with a scenario, model, region, or year to narrow the answer.

<a id="q074"></a>

## 074. yes

Category: data and scenarios<br>
Route: `general_qa` (yes/no without active question)<br>
Qwen generations: 0<br>
Time: 0.11 s<br>
Conversation: standalone

**Answer**

I don't have an open question for that answer right now. Ask about data (for example `CO2 emissions for World in 2050`), a model (`what is GCAM?`) or a term (`what does net zero mean?`).

<a id="q075"></a>

## 075. 2

Category: data and scenarios<br>
Route: `data_query` (number without active clarification)<br>
Qwen generations: 0<br>
Time: 0.04 s<br>
Conversation: standalone

**Answer**

I don't have an active numbered choice right now. Reply with a full variable, region, or scenario so I can continue.

<a id="q076"></a>

## 076. show all models

Category: data and scenarios<br>
Route: `data_query` (model availability request)<br>
Qwen generations: 0<br>
Time: 0.10 s<br>
Conversation: standalone

**Answer**

I found these models in the IAM PARIS dataset: AIM/Enduse India 3.3, ALADIN, ATOM, CHANCE, CLEWs, CLEWs - Ethiopia, CLEWs - Kenya, CLEWs - Sri Lanka, CLEWs-EU, CLIMACRED, CONTO, China-MORE, DICE, DIVA, DREEM, DYNERIO, E3ME 6.1, E3ME-FTT, E4SMA-EU-TIMES 1.0, EGMM, EPMM, ETCU-RICE, EU MFA, EXPANSE, EnergyPLAN, Euro-Calliope, FORECAST, GCAM, GCAM-Europe, GCAM-PR 5.3, GCAM-PR 7.0, GEM-E3, GEMINI-E3, GEMINI-E3 7.0, GEMINI-E3 EU, GLOBIO, HEB, HU-TIMES, ICES-XPS 1.0, ICES-XPS 1.1, IMACLIM-China, LEAP, MANAGE, MAPLE, MARIO, MARKAL-India, MEDEAS, MENA-EDS model, MUSE, MicroGridsPy, NATEM, NEMESIS 5.0, NEMESIS 5.1, NEMESIS-World, OMNIA, OPEN-PROM, OSeMOSYS, OSeMOSYS OSeMBE, OSeMOSYS-Ukraine, OnSSET, POLES, PROMETHEUS, PSM-EU, PowerPlan, REMIND-MFA, SISGEMA, TIAM_Grantham, TIAM_Grantham 3.2, TIMES-CAC, TIMES-GEO, WILIAM, WISEE-EDM, WISEE-ESM, WISEE-Global Steel and WTMBT. Which one would you like to know more about?

<a id="q077"></a>

## 077. show all variables

Category: data and scenarios<br>
Route: `data_query` (variable availability request)<br>
Qwen generations: 0<br>
Time: 0.89 s<br>
Conversation: standalone

**Answer**

I can work with these variables:
- Agricultural Demand
- Agricultural Demand|Crops
- Agricultural Demand|Crops|Energy
- Agricultural Demand|Crops|Feed
- Agricultural Demand|Crops|Food
- Agricultural Demand|Crops|Other
- Agricultural Demand|Livestock
- Agricultural Demand|Livestock|Food
- Agricultural Demand|Livestock|Other
- Agricultural Production
- Annual Refurbishment Rate|Commercial
- Annual Refurbishment Rate|Residential
- Annual Replacement Rate|Commercial
- Annual Replacement Rate|Residential
- Average Annual Investment|Energy Supply|Electricity|Battery|Rooftop
- Average Annual Investment|Energy Supply|Electricity|Battery|Utility
- Average Annual Investment|Energy Supply|Electricity|Biomass
- Average Annual Investment|Energy Supply|Electricity|CCS
- Average Annual Investment|Energy Supply|Electricity|Coal
- Average Annual Investment|Energy Supply|Electricity|FuelCell
- Average Annual Investment|Energy Supply|Electricity|Gas
- Average Annual Investment|Energy Supply|Electricity|Geothermal
- Average Annual Investment|Energy Supply|Electricity|Hydro
- Average Annual Investment|Energy Supply|Electricity|Oil
- Average Annual Investment|Energy Supply|Electricity|PumpedHydro
- Average Annual Investment|Energy Supply|Electricity|Solar|Commercial
- Average Annual Investment|Energy Supply|Electricity|Solar|Rooftop
- Average Annual Investment|Energy Supply|Electricity|Solar|Utility
- Average Annual Investment|Energy Supply|Electricity|WindOffshore
- Average Annual Investment|Energy Supply|Electricity|WindOnshore
- Average Annual Investment|Energy Supply|Hydrogen|Electrolyser
- Benchmarking|Buildings|CO2 intensity
- Benchmarking|Buildings|District heat rate
- Benchmarking|Buildings|Electrification rate
- Benchmarking|Buildings|Energy per capita
- Benchmarking|Industry|CO2 intensity (gross)
- Benchmarking|Industry|CO2 intensity (net)
- Benchmarking|Industry|Electrification rate
- Benchmarking|Industry|Energy intensity
- Benchmarking|Industry|GDP per capita
- Benchmarking|Transport|Biofuel admixture quota
- Benchmarking|Transport|CO2 intensity
- Benchmarking|Transport|Electrification rate
- Benchmarking|Transport|Energy per capita
- Biodiversity|Conservation Costs
- Biodiversity|Potentially Disappeared Fraction of species
- Biofuels Investment Share
- Biomass Investment Share
- CO2 emissions cuts|Absolute
- CO2 emissions cuts|Relative
- CSP Investment Share
- Capacity Additions|Electricity
- Capacity Additions|Electricity|Biomass
- Capacity Additions|Electricity|Biomass|w/ CCS
- Capacity Additions|Electricity|Biomass|w/o CCS
- Capacity Additions|Electricity|Coal
- Capacity Additions|Electricity|Coal|w/ CCS
- Capacity Additions|Electricity|Coal|w/o CCS
- Capacity Additions|Electricity|Gas
- Capacity Additions|Electricity|Gas|CCS
- Capacity Additions|Electricity|Gas|w/ CCS
- Capacity Additions|Electricity|Gas|w/o CCS
- Capacity Additions|Electricity|Geothermal
- Capacity Additions|Electricity|Hydro
- Capacity Additions|Electricity|Nuclear
- Capacity Additions|Electricity|Ocean
- Capacity Additions|Electricity|Oil
- Capacity Additions|Electricity|Oil|w/ CCS
- Capacity Additions|Electricity|Oil|w/o CCS
- Capacity Additions|Electricity|Solar
- Capacity Additions|Electricity|Solar|CSP
- Capacity Additions|Electricity|Solar|PV
- Capacity Additions|Electricity|Wind
- Capacity Additions|Electricity|Wind|Offshore
- Capacity Additions|Electricity|Wind|Onshore
- Capacity Additions|Electricty|Biomass|w/ CCS
- Capacity Additions|Electricty|Coal|CCS
- Capacity additions|Electricity|Solar|PV|Average
- Capacity additions|Electricity|Solar|PV|Max
- Capacity additions|Electricity|Solar|PV|Min
- Capacity|Electricity
- Capacity|Electricity|Battery|Rooftop
- Capacity|Electricity|Battery|Utility
- Capacity|Electricity|Biomass
- Capacity|Electricity|Biomass|w/ CCS
- Capacity|Electricity|Biomass|w/o CCS
- Capacity|Electricity|Coal
- Capacity|Electricity|Coal|w/ CCS
- Capacity|Electricity|Coal|w/o CCS
- Capacity|Electricity|FuelCell
- Capacity|Electricity|Gas
- Capacity|Electricity|Gas|CCS
- Capacity|Electricity|Gas|w/ CCS
- Capacity|Electricity|Gas|w/o CCS
- Capacity|Electricity|Geothermal
- Capacity|Electricity|Hydro
- Capacity|Electricity|Nuclear
- Capacity|Electricity|Ocean
- Capacity|Electricity|Oil
- Capacity|Electricity|Oil|w/ CCS
- Capacity|Electricity|Oil|w/o CCS
- Capacity|Electricity|Other
- Capacity|Electricity|PumpedHydro
- Capacity|Electricity|Solar
- Capacity|Electricity|Solar|CSP
- Capacity|Electricity|Solar|Commercial
- Capacity|Electricity|Solar|PV
- Capacity|Electricity|Solar|Rooftop
- Capacity|Electricity|Solar|Utility
- Capacity|Electricity|Wind
- Capacity|Electricity|Wind|Offshore
- Capacity|Electricity|Wind|Onshore
- Capacity|Electricty|Biomass|w/ CCS
- Capacity|Electricty|Coal|CCS
- Capacity|Hydrogen|Biomass
- Capacity|Hydrogen|Biomass|w/ CCS
- Capacity|Hydrogen|Biomass|w/o CCS
- Capacity|Hydrogen|Coal
- Capacity|Hydrogen|Coal|w/ CCS
- Capacity|Hydrogen|Coal|w/o CCS
- Capacity|Hydrogen|Electricity
- Capacity|Hydrogen|Electrolyser
- Capacity|Hydrogen|Gas
- Capacity|Hydrogen|Gas|w/ CCS
- Capacity|Hydrogen|Gas|w/o CCS
- Capital Cost|Electricity|Biomass|w/ CCS
- Capital Cost|Electricity|Biomass|w/o CCS
- Capital Cost|Electricity|Coal|w/ CCS
- Capital Cost|Electricity|Coal|w/o CCS
- Capital Cost|Electricity|Gas|w/ CCS
- Capital Cost|Electricity|Gas|w/o CCS
- Capital Cost|Electricity|Geothermal
- Capital Cost|Electricity|Hydro
- Capital Cost|Electricity|Nuclear
- Capital Cost|Electricity|Solar|CSP
- Capital Cost|Electricity|Solar|PV
- Capital Cost|Electricity|Wind|Offshore
- Capital Cost|Electricity|Wind|Onshore
- Capital Formation
- Capital Investment|Energy Supply
- Capital Stock
- Carbon Capture
- Carbon Capture|Biomass
- Carbon Capture|Fossil
- Carbon Intensity|Production|Steel
- Carbon Sequestration|CCS
- Carbon Sequestration|CCS|Biomass
- Carbon Sequestration|CCS|Biomass|Energy
- Carbon Sequestration|CCS|Biomass|Energy|Demand|Industry
- Carbon Sequestration|CCS|Biomass|Energy|Demand|Industry|Cement
- Carbon Sequestration|CCS|Biomass|Energy|Demand|Industry|Steel
- Carbon Sequestration|CCS|Biomass|Energy|Supply
- Carbon Sequestration|CCS|Biomass|Energy|Supply|Electricity
- Carbon Sequestration|CCS|Biomass|Energy|Supply|Gases
- Carbon Sequestration|CCS|Biomass|Energy|Supply|Hydrogen
- Carbon Sequestration|CCS|Biomass|Energy|Supply|Liquids
- Carbon Sequestration|CCS|Biomass|Energy|Supply|Other
- Carbon Sequestration|CCS|Biomass|Industrial Processes
- Carbon Sequestration|CCS|Energy|Demand|Industry
- Carbon Sequestration|CCS|Energy|Supply
- Carbon Sequestration|CCS|Energy|Supply|Electricity
- Carbon Sequestration|CCS|Energy|Supply|Gases
- Carbon Sequestration|CCS|Energy|Supply|Hydrogen
- Carbon Sequestration|CCS|Energy|Supply|Liquids
- Carbon Sequestration|CCS|Energy|Supply|Other
- Carbon Sequestration|CCS|Fossil
- Carbon Sequestration|CCS|Fossil|Energy
- Carbon Sequestration|CCS|Fossil|Energy|Demand|Industry
- Carbon Sequestration|CCS|Fossil|Energy|Demand|Industry|Cement
- Carbon Sequestration|CCS|Fossil|Energy|Demand|Industry|Steel
- Carbon Sequestration|CCS|Fossil|Energy|Supply
- Carbon Sequestration|CCS|Fossil|Energy|Supply|Electricity
- Carbon Sequestration|CCS|Fossil|Energy|Supply|Gases
- Carbon Sequestration|CCS|Fossil|Energy|Supply|Hydrogen
- Carbon Sequestration|CCS|Fossil|Energy|Supply|Liquids
- Carbon Sequestration|CCS|Fossil|Energy|Supply|Other
- Carbon Sequestration|CCS|Fossil|Industrial Processes
- Carbon Sequestration|CCS|Fossil|Supply|Liquids
- Carbon Sequestration|CCS|Industrial Processes
- Carbon Sequestration|CCS|Industrial Processes|Cement
- Carbon Sequestration|CCS|Industrial Processes|Demand
- Carbon Sequestration|Direct Air Capture
- Carbon Sequestration|Feedstocks
- Carbon Utilization|CCS|Industry
- Carbon footprint|CO2|Energy|Supply|Electricity
- Concentration|CO2
- Consumption
- Consumption|Steel
- Cost of Capital
- Demand|Transportation|Road|Passenger|2W
- Demand|Transportation|Road|Passenger|3W
- Demand|Transportation|Road|Passenger|Battery Electric Car
- Demand|Transportation|Road|Passenger|Battery electric 3W
- Demand|Transportation|Road|Passenger|CNG/LPG Car
- Demand|Transportation|Road|Passenger|Car
- Demand|Transportation|Road|Passenger|Diesel Car
- Demand|Transportation|Road|Passenger|EPAC
- Demand|Transportation|Road|Passenger|EPAC+
- Demand|Transportation|Road|Passenger|Gasoline Car
- Demand|Transportation|Road|Passenger|Hybrid Electric Car
- Demand|Transportation|Road|Passenger|Minibus
- Demand|Transportation|Road|Passenger|Plug-in Hybrid Electric Car
- Demand|Transportation|Road|Passenger|e-Microcar 120kmph
- Demand|Transportation|Road|Passenger|e-Microcar 45kmph
- Demand|Transportation|Road|Passenger|e-Microcar 90 kmph
- Demand|Transportation|Road|Passenger|e-Moped
- Demand|Transportation|Road|Passenger|e-Motorcycle
- Demand|Transportation|Road|Passenger|e-Scooter
- Demand|Transportation|Road|Passenger|e-Speed Bike
- Discount rate|Industry
- Economy|Commercial|Number of buildings
- Economy|Commercial|Value added
- Economy|Residential & Commercial|Number of buildings
- Economy|Residential|Number of households
- Economy|Residential|Population
- Economy|Residential|Private Consumption
- Efficiency|Electricity|Biomass|w/ CCS
- Efficiency|Electricity|Biomass|w/o CCS
- Efficiency|Electricity|Coal|w/ CCS
- Efficiency|Electricity|Coal|w/o CCS
- Efficiency|Electricity|Gas|w/ CCS
- Efficiency|Electricity|Gas|w/o CCS
- Emission Factor|CO2|Tailpipe|2W
- Emission Factor|CO2|Tailpipe|3W
- Emission Factor|CO2|Tailpipe|Battery Electric Car
- Emission Factor|CO2|Tailpipe|Battery electric 3W
- Emission Factor|CO2|Tailpipe|CNG/LPG Car
- Emission Factor|CO2|Tailpipe|Car
- Emission Factor|CO2|Tailpipe|Diesel Car
- Emission Factor|CO2|Tailpipe|EPAC
- Emission Factor|CO2|Tailpipe|EPAC+
- Emission Factor|CO2|Tailpipe|Gasoline Car
- Emission Factor|CO2|Tailpipe|Hybrid Electric Car
- Emission Factor|CO2|Tailpipe|Minibus
- Emission Factor|CO2|Tailpipe|Plug-in Hybrid Electric Car
- Emission Factor|CO2|Tailpipe|e-Microcar 120kmph
- Emission Factor|CO2|Tailpipe|e-Microcar 45kmph
- Emission Factor|CO2|Tailpipe|e-Microcar 90 kmph
- Emission Factor|CO2|Tailpipe|e-Moped
- Emission Factor|CO2|Tailpipe|e-Motorcycle
- Emission Factor|CO2|Tailpipe|e-Scooter
- Emission Factor|CO2|Tailpipe|e-Speed Bike
- Emissions Penalty|Energy Supply
- Emissions avoided|CO2|Residential and Commercial|Residential
- Emissions per capita|CO2|Energy
- Emissions/GDP|CO2|Energy
- Emissions|BC
- Emissions|BC|AFOLU
- Emissions|BC|Energy
- Emissions|BC|Energy|Demand|Industry
- Emissions|BC|Energy|Demand|Residential and Commercial
- Emissions|BC|Energy|Demand|Transportation
- Emissions|BC|Energy|Supply
- Emissions|C2F6
- Emissions|CF4
- Emissions|CH4
- Emissions|CH4|AFOFI
- Emissions|CH4|AFOLU
- Emissions|CH4|AFOLU|Agriculture
- Emissions|CH4|Energy
- Emissions|CH4|Energy|Demand|Industry
- Emissions|CH4|Energy|Demand|Residential and Commercial
- Emissions|CH4|Energy|Demand|Transportation
- Emissions|CH4|Energy|Supply
- Emissions|CH4|Industrial Processes
- Emissions|CH4|LULUCF
- Emissions|CH4|Other
- Emissions|CH4|Waste
- Emissions|CO
- Emissions|CO2
- Emissions|CO2 (w/o bunkers)
- Emissions|CO2|AFOLU
- Emissions|CO2|AFOLU|Agriculture
- Emissions|CO2|Demand|Transportation|Passenger|2W
- Emissions|CO2|Demand|Transportation|Passenger|3W
- Emissions|CO2|Demand|Transportation|Passenger|Battery Electric Car
- Emissions|CO2|Demand|Transportation|Passenger|Battery electric 3W
- Emissions|CO2|Demand|Transportation|Passenger|CNG/LPG Car
- Emissions|CO2|Demand|Transportation|Passenger|Car
- Emissions|CO2|Demand|Transportation|Passenger|Diesel Car
- Emissions|CO2|Demand|Transportation|Passenger|EPAC
- Emissions|CO2|Demand|Transportation|Passenger|EPAC+
- Emissions|CO2|Demand|Transportation|Passenger|Gasoline Car
- Emissions|CO2|Demand|Transportation|Passenger|Hybrid Electric Car
- Emissions|CO2|Demand|Transportation|Passenger|Minibus
- Emissions|CO2|Demand|Transportation|Passenger|Plug-in Hybrid Electric Car
- Emissions|CO2|Demand|Transportation|Passenger|e-Microcar 120kmph
- Emissions|CO2|Demand|Transportation|Passenger|e-Microcar 45kmph
- Emissions|CO2|Demand|Transportation|Passenger|e-Microcar 90 kmph
- Emissions|CO2|Demand|Transportation|Passenger|e-Moped
- Emissions|CO2|Demand|Transportation|Passenger|e-Motorcycle
- Emissions|CO2|Demand|Transportation|Passenger|e-Scooter
- Emissions|CO2|Demand|Transportation|Passenger|e-Speed Bike
- Emissions|CO2|EUETS
- Emissions|CO2|Electricity
- Emissions|CO2|Energy
- Emissions|CO2|Energy and Industrial Processes
- Emissions|CO2|Energy and Industrial Processes|EUETS
- Emissions|CO2|Energy and Industrial Processes|Non EUETS
- Emissions|CO2|Energy|Demand
- Emissions|CO2|Energy|Demand|AFOFI
- Emissions|CO2|Energy|Demand|Bunkers
- Emissions|CO2|Energy|Demand|Bunkers|International Aviation
- Emissions|CO2|Energy|Demand|Bunkers|International Shipping
- Emissions|CO2|Energy|Demand|Commercial
- Emissions|CO2|Energy|Demand|Industry
- Emissions|CO2|Energy|Demand|Industry|Cement
- Emissions|CO2|Energy|Demand|Industry|Chemicals
- Emissions|CO2|Energy|Demand|Industry|Chemicals|Ammonia
- Emissions|CO2|Energy|Demand|Industry|Chemicals|High value chemicals
- Emissions|CO2|Energy|Demand|Industry|Chemicals|Methanol
- Emissions|CO2|Energy|Demand|Industry|Chemicals|Other
- Emissions|CO2|Energy|Demand|Industry|EUETS
- Emissions|CO2|Energy|Demand|Industry|Non EUETS
- Emissions|CO2|Energy|Demand|Industry|Non-ferrous metals
- Emissions|CO2|Energy|Demand|Industry|Other
- Emissions|CO2|Energy|Demand|Industry|Pulp and Paper
- Emissions|CO2|Energy|Demand|Industry|Steel
- Emissions|CO2|Energy|Demand|Other Sector
- Emissions|CO2|Energy|Demand|Residential
- Emissions|CO2|Energy|Demand|Residential and Commercial
- Emissions|CO2|Energy|Demand|Residential and Commercial|Commercial
- Emissions|CO2|Energy|Demand|Residential and Commercial|Commercial|Heating
- Emissions|CO2|Energy|Demand|Residential and Commercial|Commercial|Heating|Cooking
- Emissions|CO2|Energy|Demand|Residential and Commercial|Commercial|Heating|Space
- Emissions|CO2|Energy|Demand|Residential and Commercial|Commercial|Heating|Water
- Emissions|CO2|Energy|Demand|Residential and Commercial|Residential
- Emissions|CO2|Energy|Demand|Residential and Commercial|Residential|Heating
- Emissions|CO2|Energy|Demand|Residential and Commercial|Residential|Heating|Cooking
- Emissions|CO2|Energy|Demand|Residential and Commercial|Residential|Heating|Space
- Emissions|CO2|Energy|Demand|Residential and Commercial|Residential|Heating|Water
- Emissions|CO2|Energy|Demand|Transport
- Emissions|CO2|Energy|Demand|Transportation
- Emissions|CO2|Energy|Demand|Transportation (w/ bunkers)
- Emissions|CO2|Energy|Demand|Transportation|Aviation
- Emissions|CO2|Energy|Demand|Transportation|Aviation and Maritime|Domestic
- Emissions|CO2|Energy|Demand|Transportation|Aviation and Maritime|Extra-EU
- Emissions|CO2|Energy|Demand|Transportation|Aviation and Maritime|International
- Emissions|CO2|Energy|Demand|Transportation|Aviation and Maritime|Intra-EU
- Emissions|CO2|Energy|Demand|Transportation|Aviation|Freight
- Emissions|CO2|Energy|Demand|Transportation|Aviation|International
- Emissions|CO2|Energy|Demand|Transportation|Aviation|Passenger
- Emissions|CO2|Energy|Demand|Transportation|EUETS
- Emissions|CO2|Energy|Demand|Transportation|Freight
- Emissions|CO2|Energy|Demand|Transportation|Gases
- Emissions|CO2|Energy|Demand|Transportation|Liquids
- Emissions|CO2|Energy|Demand|Transportation|Maritime
- Emissions|CO2|Energy|Demand|Transportation|Maritime|Freight
- Emissions|CO2|Energy|Demand|Transportation|Maritime|International
- Emissions|CO2|Energy|Demand|Transportation|Passenger
- Emissions|CO2|Energy|Demand|Transportation|Rail
- Emissions|CO2|Energy|Demand|Transportation|Rail|Freight
- Emissions|CO2|Energy|Demand|Transportation|Rail|Passenger
- Emissions|CO2|Energy|Demand|Transportation|Road
- Emissions|CO2|Energy|Demand|Transportation|Road|Freight
- Emissions|CO2|Energy|Demand|Transportation|Road|Passenger
- Emissions|CO2|Energy|Demand|Transportation|Road|Passenger|Bus
- Emissions|CO2|Energy|Demand|Transportation|Road|Passenger|LDV
- Emissions|CO2|Energy|Demand|Transportation|non EUETS
- Emissions|CO2|Energy|Supply
- Emissions|CO2|Energy|Supply|EUETS
- Emissions|CO2|Energy|Supply|Electricity
- Emissions|CO2|Energy|Supply|Gases
- Emissions|CO2|Energy|Supply|Heat
- Emissions|CO2|Energy|Supply|Hydrogen
- Emissions|CO2|Energy|Supply|Liquids
- Emissions|CO2|Energy|Supply|Non EUETS
- Emissions|CO2|Energy|Supply|Other Sector
- Emissions|CO2|Energy|Supply|Solids
- Emissions|CO2|Gases
- Emissions|CO2|Heat
- Emissions|CO2|Industrial Processes
- Emissions|CO2|Industrial Processes|Cement
- Emissions|CO2|Industrial Processes|Chemicals
- Emissions|CO2|Industrial Processes|Chemicals|Ammonia
- Emissions|CO2|Industrial Processes|Chemicals|High value chemicals
- Emissions|CO2|Industrial Processes|Chemicals|Methanol
- Emissions|CO2|Industrial Processes|Chemicals|Other
- Emissions|CO2|Industrial Processes|Non-ferrous metals
- Emissions|CO2|Industrial Processes|Other
- Emissions|CO2|Industrial Processes|Pulp and Paper
- Emissions|CO2|Industrial Processes|Steel
- Emissions|CO2|Industry and Industrial Processes
- Emissions|CO2|Industry|Steel
- Emissions|CO2|LULUCF
- Emissions|CO2|Liquids
- Emissions|CO2|Non EUETS
- Emissions|CO2|Other
- Emissions|CO2|Other|DACCS
- Emissions|CO2|Solids
- Emissions|CO2|Waste
- Emissions|CO|AFOLU
- Emissions|CO|Energy
- Emissions|CO|Energy|Demand|Industry
- Emissions|CO|Energy|Demand|Residential and Commercial
- Emissions|CO|Energy|Demand|Transportation
- Emissions|CO|Energy|Supply
- Emissions|F-Gases
- Emissions|GHG|EUETS
- Emissions|GHG|Non EUETS
- Emissions|HFC
- Emissions|HFC|HFC125
- Emissions|HFC|HFC134a
- Emissions|HFC|HFC143a
- Emissions|HFC|HFC227ea
- Emissions|HFC|HFC23
- Emissions|HFC|HFC245fa
- Emissions|HFC|HFC32
- Emissions|HFC|HFC43-10
- Emissions|HFC|Industrial Processes
- Emissions|Kyoto Gases
- Emissions|Kyoto Gases (AR5-GWP100)
- Emissions|Kyoto Gases (AR6-GWP100)
- Emissions|Kyoto Gases|AFOLU
- Emissions|Kyoto Gases|Buildings
- Emissions|Kyoto Gases|Cement
- Emissions|Kyoto Gases|Chemicals
- Emissions|Kyoto Gases|Electricity
- Emissions|Kyoto Gases|Industry
- Emissions|Kyoto Gases|Non-ferrous Metals
- Emissions|Kyoto Gases|Other
- Emissions|Kyoto Gases|Other Energy Supply
- Emissions|Kyoto Gases|Other Industry
- Emissions|Kyoto Gases|Steel
- Emissions|Kyoto Gases|Supply
- Emissions|Kyoto Gases|Transportation
- Emissions|N2O
- Emissions|N2O|AFOFI
- Emissions|N2O|AFOLU
- Emissions|N2O|AFOLU|Agriculture
- Emissions|N2O|Energy
- Emissions|N2O|Industrial Processes
- Emissions|N2O|LULUCF
- Emissions|N2O|Other
- Emissions|N2O|Waste
- Emissions|NH3
- Emissions|NH3|AFOLU
- Emissions|NH3|Energy
- Emissions|NH3|Energy|Demand|Industry
- Emissions|NH3|Energy|Demand|Residential and Commercial
- Emissions|NH3|Energy|Demand|Transportation
- Emissions|NH3|Energy|Supply
- Emissions|NOx
- Emissions|NOx|AFOLU
- Emissions|NOx|Energy
- Emissions|NOx|Energy|Demand|Industry
- Emissions|NOx|Energy|Demand|Residential and Commercial
- Emissions|NOx|Energy|Demand|Transportation
- Emissions|NOx|Energy|Supply
- Emissions|OC
- Emissions|OC|AFOLU
- Emissions|OC|Energy
- Emissions|OC|Energy|Demand|Industry
- Emissions|OC|Energy|Demand|Residential and Commercial
- Emissions|OC|Energy|Demand|Transportation
- Emissions|OC|Energy|Supply
- Emissions|PFC
- Emissions|PFC|C2F6
- Emissions|PFC|C6F14
- Emissions|PFC|CF4
- Emissions|SF6
- Emissions|Sulfur
- Emissions|Sulfur|AFOLU
- Emissions|Sulfur|Energy
- Emissions|Sulfur|Energy|Demand|Industry
- Emissions|Sulfur|Energy|Demand|Residential and Commercial
- Emissions|Sulfur|Energy|Demand|Transportation
- Emissions|Sulfur|Energy|Supply
- Emissions|VOC
- Emissions|VOC|AFOLU
- Emissions|VOC|Energy
- Emissions|VOC|Energy|Demand|Industry
- Emissions|VOC|Energy|Demand|Residential and Commercial
- Emissions|VOC|Energy|Demand|Transportation
- Emissions|VOC|Energy|Supply
- Employment
- Employment|Agricultural and Unskilled
- Employment|Agriculture
- Employment|Clerks
- Employment|Industry
- Employment|Industry|Energy
- Employment|Industry|Energy Intensive
- Employment|Industry|Manufacturing
- Employment|Officials and Managers
- Employment|Service
- Employment|Service and Shop Workers
- Employment|Skilled
- Employment|Technicians
- Employment|Unskilled
- Employment|Unskilled|Rate
- Energy Jobs 2025|Absolute
- Energy Jobs 2025|Relative
- Energy Jobs 2030|Absolute
- Energy Jobs 2030|Relative
- Energy Service|Commercial|Floor Space
- Energy Service|Residential and Commercial|Floor Space
- Energy Service|Residential and Commercial|Residential|Floor Space
- Energy Service|Residential|Floor Space
- Energy Service|Transportation|Freight
- Energy Service|Transportation|Freight|Aviation
- Energy Service|Transportation|Freight|International Shipping
- Energy Service|Transportation|Freight|Navigation
- Energy Service|Transportation|Freight|Railways
- Energy Service|Transportation|Freight|Road
- Energy Service|Transportation|Passenger
- Energy Service|Transportation|Passenger|Aviation
- Energy Service|Transportation|Passenger|Bicycling and Walking
- Energy Service|Transportation|Passenger|Railways
- Energy Service|Transportation|Passenger|Road
- Energy Service|Transportation|Passenger|Road|Bus
- Energy Service|Transportation|Passenger|Road|LDV
- Expenditure|Energy|Fossil
- Expenditure|Household|Total
- Expenditure|Household|Total|Clothing and footwear
- Expenditure|Household|Total|Communication
- Expenditure|Household|Total|Education
- Expenditure|Household|Total|Food beverages and tobacco
- Expenditure|Household|Total|Fuels and power
- Expenditure|Household|Total|Heating and cooking appliances
- Expenditure|Household|Total|Household equipment and operation
- Expenditure|Household|Total|Household equipment and operation (excl. heating and cooking appliances)
- Expenditure|Household|Total|Housing and water charges
- Expenditure|Household|Total|Medical care and health
- Expenditure|Household|Total|Miscellaneous goods and services
- Expenditure|Household|Total|Operation of vehicles
- Expenditure|Household|Total|Purchase of vehicles
- Expenditure|Household|Total|Recreational services
- Expenditure|Household|Total|Transport services
- Expenditure|RnD
- Expenditure|government
- Expenditure|household
- Expenditure|household|Energy
- Expenditure|household|Food
- Expenditure|household|Industry
- Expenditure|household|Services
- Expenditure|household|Services|Telecommunications
- Expenditure|household|Services|health
- Export
- Exports
- Exports|Agriculture
- Exports|Air transport
- Exports|Basic metals
- Exports|Basic pharmaceutical products
- Exports|Batteries
- Exports|Biofuels
- Exports|Biomass
- Exports|Biomass Solid
- Exports|CCS coal
- Exports|CSS Bio
- Exports|CSS Gas
- Exports|Chemical Products
- Exports|Clean Gas
- Exports|Coal fired
- Exports|Computer, electronic and optical products
- Exports|Construction
- Exports|Consumer Goods Industries
- Exports|EV Transport Equipment
- Exports|Electrical Equipment
- Exports|Electricity
- Exports|Fabricated Metal products
- Exports|Gas Fired
- Exports|Gas supply
- Exports|Geothermal
- Exports|Hydro electric
- Exports|Hydrogen
- Exports|Land transport
- Exports|Land transport and Warehousing services
- Exports|Manufactured fuels
- Exports|Market Services
- Exports|Mechanical Engineering
- Exports|Mining
- Exports|Non Market Services
- Exports|Non-metallic minerals
- Exports|Nuclear
- Exports|Oil Fired
- Exports|Other Equipment Goods
- Exports|PV
- Exports|Paper products, publishing
- Exports|R&D
- Exports|Rubber and plastic products
- Exports|Steel
- Exports|Transport equipment
- Exports|Transport equipment (excluding EV)
- Exports|Warehousing
- Exports|Water transport
- Exports|Wind
- Export|Agriculture
- Export|Energy
- Export|Industry
- Export|Industry|Energy
- Export|Industry|Energy Intensive
- Export|Industry|Manufacturing
- Export|Other
- Final Energy
- Final Energy (excl. feedstocks)|Industry
- Final Energy (excl. feedstocks)|Industry|Chemicals|Electricity
- Final Energy (excl. feedstocks)|Industry|Chemicals|Gases
- Final Energy (excl. feedstocks)|Industry|Chemicals|Gases|Bioenergy
- Final Energy (excl. feedstocks)|Industry|Chemicals|Gases|Fossil
- Final Energy (excl. feedstocks)|Industry|Chemicals|Heat
- Final Energy (excl. feedstocks)|Industry|Chemicals|Liquids
- Final Energy (excl. feedstocks)|Industry|Chemicals|Liquids|Bioenergy
- Final Energy (excl. feedstocks)|Industry|Chemicals|Liquids|Fossil
- Final Energy (excl. feedstocks)|Industry|Chemicals|Solids
- Final Energy (excl. feedstocks)|Industry|Chemicals|Solids|Bioenergy
- Final Energy (excl. feedstocks)|Industry|Chemicals|Solids|Fossil
- Final Energy (excl. feedstocks)|Industry|Electricity
- Final Energy (excl. feedstocks)|Industry|Gases
- Final Energy (excl. feedstocks)|Industry|Gases|Bioenergy
- Final Energy (excl. feedstocks)|Industry|Gases|Fossil
- Final Energy (excl. feedstocks)|Industry|Gases|Hydrogen synfuel
- Final Energy (excl. feedstocks)|Industry|Heat
- Final Energy (excl. feedstocks)|Industry|Hydrogen
- Final Energy (excl. feedstocks)|Industry|Liquids
- Final Energy (excl. feedstocks)|Industry|Liquids|Bioenergy
- Final Energy (excl. feedstocks)|Industry|Liquids|Fossil
- Final Energy (excl. feedstocks)|Industry|Liquids|Hydrogen synfuel
- Final Energy (excl. feedstocks)|Industry|Liquids|Oil
- Final Energy (excl. feedstocks)|Industry|Other
- Final Energy (excl. feedstocks)|Industry|Other|Bioenergy
- Final Energy (excl. feedstocks)|Industry|Other|Fossil
- Final Energy (excl. feedstocks)|Industry|Solids
- Final Energy (excl. feedstocks)|Industry|Solids|Bioenergy
- Final Energy (excl. feedstocks)|Industry|Solids|Biomass
- Final Energy (excl. feedstocks)|Industry|Solids|Coal
- Final Energy (excl. feedstocks)|Industry|Solids|Fossil
- Final Energy (w/o bunkers)
- Final Energy|AFOFI
- Final Energy|AFOFI|Electricity
- Final Energy|AFOFI|Gases
- Final Energy|AFOFI|Gases|Bioenergy
- Final Energy|AFOFI|Gases|Fossil
- Final Energy|AFOFI|Hydrogen
- Final Energy|AFOFI|Liquids
- Final Energy|AFOFI|Liquids|Bioenergy
- Final Energy|AFOFI|Liquids|Fossil
- Final Energy|AFOFI|Modern
- Final Energy|AFOFI|Modern|Electricity
- Final Energy|AFOFI|Modern|Gases
- Final Energy|AFOFI|Modern|Gases|Bioenergy
- Final Energy|AFOFI|Modern|Gases|Fossil
- Final Energy|AFOFI|Modern|Hydrogen
- Final Energy|AFOFI|Modern|Liquids
- Final Energy|AFOFI|Modern|Liquids|Bioenergy
- Final Energy|AFOFI|Modern|Liquids|Fossil
- Final Energy|AFOFI|Modern|Other Fuel
- Final Energy|AFOFI|Modern|Solids
- Final Energy|AFOFI|Modern|Solids|Biomass
- Final Energy|AFOFI|Modern|Solids|Coal
- Final Energy|AFOFI|Other Fuel
- Final Energy|AFOFI|Solids
- Final Energy|AFOFI|Solids|Biomass
- Final Energy|AFOFI|Solids|Coal
- Final Energy|AFOFI|Traditional
- Final Energy|AFOFI|Traditional|Electricity
- Final Energy|AFOFI|Traditional|Gases
- Final Energy|AFOFI|Traditional|Gases|Bioenergy
- Final Energy|AFOFI|Traditional|Gases|Fossil
- Final Energy|AFOFI|Traditional|Hydrogen
- Final Energy|AFOFI|Traditional|Liquids
- Final Energy|AFOFI|Traditional|Liquids|Bioenergy
- Final Energy|AFOFI|Traditional|Liquids|Fossil
- Final Energy|AFOFI|Traditional|Other Fuel
- Final Energy|AFOFI|Traditional|Solids
- Final Energy|AFOFI|Traditional|Solids|Biomass
- Final Energy|AFOFI|Traditional|Solids|Coal
- Final Energy|AFOFI|Transitional
- Final Energy|AFOFI|Transitional|Electricity
- Final Energy|AFOFI|Transitional|Gases
- Final Energy|AFOFI|Transitional|Gases|Bioenergy
- Final Energy|AFOFI|Transitional|Gases|Fossil
- Final Energy|AFOFI|Transitional|Hydrogen
- Final Energy|AFOFI|Transitional|Liquids
- Final Energy|AFOFI|Transitional|Liquids|Bioenergy
- Final Energy|AFOFI|Transitional|Liquids|Fossil
- Final Energy|AFOFI|Transitional|Other Fuel
- Final Energy|AFOFI|Transitional|Solids
- Final Energy|AFOFI|Transitional|Solids|Biomass
- Final Energy|AFOFI|Transitional|Solids|Coal
- Final Energy|Buildings|Residential
- Final Energy|Buildings|Residential|Electricity
- Final Energy|Buildings|Residential|Gases
- Final Energy|Buildings|Residential|Liquids
- Final Energy|Buildings|Residential|Solids|Coal
- Final Energy|Commercial
- Final Energy|Commerciall|Appliances
- Final Energy|Commercial|Cooking
- Final Energy|Commercial|Electricity
- Final Energy|Commercial|Gases
- Final Energy|Commercial|Heat
- Final Energy|Commercial|Hydrogen
- Final Energy|Commercial|Lighting
- Final Energy|Commercial|Liquids
- Final Energy|Commercial|Other
- Final Energy|Commercial|Solids
- Final Energy|Commercial|Solids|Biomass
- Final Energy|Commercial|Solids|Fossil
- Final Energy|Commercial|Space Cooling
- Final Energy|Commercial|Space Heating
- Final Energy|Commercial|Water Heating
- Final Energy|Electricity
- Final Energy|Gases
- Final Energy|Gases|Bioenergy
- Final Energy|Gases|Electricity
- Final Energy|Gases|Fossil
- Final Energy|Geothermal
- Final Energy|Heat
- Final Energy|Hydrogen
- Final Energy|Industry
- Final Energy|Industry|Cement
- Final Energy|Industry|Cement|Biomass
- Final Energy|Industry|Cement|Coal
- Final Energy|Industry|Cement|Electricity
- Final Energy|Industry|Cement|Gases
- Final Energy|Industry|Cement|Hydrogen
- Final Energy|Industry|Cement|Natural gas
- Final Energy|Industry|Cement|Other fossils
- Final Energy|Industry|Cement|Pet coke
- Final Energy|Industry|Cement|Waste based fuels
- Final Energy|Industry|Chemicals
- Final Energy|Industry|Chemicals|Ammonia
- Final Energy|Industry|Chemicals|Ammonia|Electricity
- Final Energy|Industry|Chemicals|Ammonia|Gases
- Final Energy|Industry|Chemicals|Ammonia|Gases|Bioenergy
- Final Energy|Industry|Chemicals|Ammonia|Gases|Fossil
- Final Energy|Industry|Chemicals|Ammonia|Hydrogen
- Final Energy|Industry|Chemicals|Ammonia|Liquids
- Final Energy|Industry|Chemicals|Ammonia|Liquids|Bioenergy
- Final Energy|Industry|Chemicals|Ammonia|Liquids|Fossil
- Final Energy|Industry|Chemicals|Ammonia|Other Fuel
- Final Energy|Industry|Chemicals|Ammonia|Solids
- Final Energy|Industry|Chemicals|Ammonia|Solids|Biomass
- Final Energy|Industry|Chemicals|Ammonia|Solids|Coal
- Final Energy|Industry|Chemicals|Ammonia|Solids|Fossil
- Final Energy|Industry|Chemicals|Electricity
- Final Energy|Industry|Chemicals|Gases
- Final Energy|Industry|Chemicals|Gases|Bioenergy
- Final Energy|Industry|Chemicals|Gases|Fossil
- Final Energy|Industry|Chemicals|Heat
- Final Energy|Industry|Chemicals|High value chemicals
- Final Energy|Industry|Chemicals|High value chemicals|Electricity
- Final Energy|Industry|Chemicals|High value chemicals|Gases
- Final Energy|Industry|Chemicals|High value chemicals|Gases|Bioenergy
- Final Energy|Industry|Chemicals|High value chemicals|Gases|Fossil
- Final Energy|Industry|Chemicals|High value chemicals|Hydrogen
- Final Energy|Industry|Chemicals|High value chemicals|Liquids
- Final Energy|Industry|Chemicals|High value chemicals|Liquids|Bioenergy
- Final Energy|Industry|Chemicals|High value chemicals|Liquids|Fossil
- Final Energy|Industry|Chemicals|High value chemicals|Other Fuel
- Final Energy|Industry|Chemicals|High value chemicals|Solids
- Final Energy|Industry|Chemicals|High value chemicals|Solids|Biomass
- Final Energy|Industry|Chemicals|High value chemicals|Solids|Coal
- Final Energy|Industry|Chemicals|Hydrogen
- Final Energy|Industry|Chemicals|Liquids
- Final Energy|Industry|Chemicals|Liquids|Bioenergy
- Final Energy|Industry|Chemicals|Liquids|Fossil
- Final Energy|Industry|Chemicals|Methanol
- Final Energy|Industry|Chemicals|Methanol|Electricity
- Final Energy|Industry|Chemicals|Methanol|Gases
- Final Energy|Industry|Chemicals|Methanol|Gases|Bioenergy
- Final Energy|Industry|Chemicals|Methanol|Gases|Fossil
- Final Energy|Industry|Chemicals|Methanol|Hydrogen
- Final Energy|Industry|Chemicals|Methanol|Liquids
- Final Energy|Industry|Chemicals|Methanol|Liquids|Bioenergy
- Final Energy|Industry|Chemicals|Methanol|Liquids|Fossil
- Final Energy|Industry|Chemicals|Methanol|Other Fuel
- Final Energy|Industry|Chemicals|Methanol|Solids
- Final Energy|Industry|Chemicals|Methanol|Solids|Biomass
- Final Energy|Industry|Chemicals|Methanol|Solids|Coal
- Final Energy|Industry|Chemicals|Other
- Final Energy|Industry|Chemicals|Other Fuel
- Final Energy|Industry|Chemicals|Other|Electricity
- Final Energy|Industry|Chemicals|Other|Gases
- Final Energy|Industry|Chemicals|Other|Gases|Bioenergy
- Final Energy|Industry|Chemicals|Other|Gases|Fossil
- Final Energy|Industry|Chemicals|Other|Hydrogen
- Final Energy|Industry|Chemicals|Other|Liquids
- Final Energy|Industry|Chemicals|Other|Liquids|Bioenergy
- Final Energy|Industry|Chemicals|Other|Liquids|Fossil
- Final Energy|Industry|Chemicals|Other|Other Fuel
- Final Energy|Industry|Chemicals|Other|Solids
- Final Energy|Industry|Chemicals|Other|Solids|Biomass
- Final Energy|Industry|Chemicals|Other|Solids|Coal
- Final Energy|Industry|Chemicals|Solids
- Final Energy|Industry|Chemicals|Solids|Bioenergy
- Final Energy|Industry|Chemicals|Solids|Biomass
- Final Energy|Industry|Chemicals|Solids|Coal
- Final Energy|Industry|Chemicals|Solids|Fossil
- Final Energy|Industry|Electricity
- Final Energy|Industry|Gases
- Final Energy|Industry|Gases|
- Final Energy|Industry|Gases|Bioenergy
- Final Energy|Industry|Gases|Electricity
- Final Energy|Industry|Gases|Fossil
- Final Energy|Industry|Gases|Fossils
- Final Energy|Industry|Gases|Hydrogen synfuel
- Final Energy|Industry|Heat
- Final Energy|Industry|Hydrogen
- Final Energy|Industry|Liquids
- Final Energy|Industry|Liquids|
- Final Energy|Industry|Liquids|Bioenergy
- Final Energy|Industry|Liquids|Biomass
- Final Energy|Industry|Liquids|Fossil
- Final Energy|Industry|Liquids|Fossils
- Final Energy|Industry|Liquids|Hydrogen synfuel
- Final Energy|Industry|Liquids|Oil
- Final Energy|Industry|Non-ferrous metals
- Final Energy|Industry|Non-ferrous metals|Electricity
- Final Energy|Industry|Non-ferrous metals|Gases
- Final Energy|Industry|Non-ferrous metals|Gases|Bioenergy
- Final Energy|Industry|Non-ferrous metals|Gases|Fossil
- Final Energy|Industry|Non-ferrous metals|Hydrogen
- Final Energy|Industry|Non-ferrous metals|Liquids
- Final Energy|Industry|Non-ferrous metals|Liquids|Bioenergy
- Final Energy|Industry|Non-ferrous metals|Liquids|Fossil
- Final Energy|Industry|Non-ferrous metals|Other Fuel
- Final Energy|Industry|Non-ferrous metals|Solids
- Final Energy|Industry|Non-ferrous metals|Solids|Bioenergy
- Final Energy|Industry|Non-ferrous metals|Solids|Biomass
- Final Energy|Industry|Non-ferrous metals|Solids|Coal
- Final Energy|Industry|Non-ferrous metals|Solids|Fossil
- Final Energy|Industry|Non-metallic minerals
- Final Energy|Industry|Non-metallic minerals|Electricity
- Final Energy|Industry|Non-metallic minerals|Gases
- Final Energy|Industry|Non-metallic minerals|Gases|Bioenergy
- Final Energy|Industry|Non-metallic minerals|Gases|Fossil
- Final Energy|Industry|Non-metallic minerals|Hydrogen
- Final Energy|Industry|Non-metallic minerals|Liquids
- Final Energy|Industry|Non-metallic minerals|Liquids|Bioenergy
- Final Energy|Industry|Non-metallic minerals|Liquids|Fossil
- Final Energy|Industry|Non-metallic minerals|Other Fuel
- Final Energy|Industry|Non-metallic minerals|Solids
- Final Energy|Industry|Non-metallic minerals|Solids|Biomass
- Final Energy|Industry|Non-metallic minerals|Solids|Coal
- Final Energy|Industry|Other
- Final Energy|Industry|Other Fuel
- Final Energy|Industry|Other|Bioenergy
- Final Energy|Industry|Other|Electricity
- Final Energy|Industry|Other|Fossil
- Final Energy|Industry|Other|Gases
- Final Energy|Industry|Other|Gases|Bioenergy
- Final Energy|Industry|Other|Gases|Fossil
- Final Energy|Industry|Other|Hydrogen
- Final Energy|Industry|Other|Liquids
- Final Energy|Industry|Other|Liquids|Bioenergy
- Final Energy|Industry|Other|Liquids|Fossil
- Final Energy|Industry|Other|Other Fuel
- Final Energy|Industry|Other|Solids
- Final Energy|Industry|Other|Solids|Biomass
- Final Energy|Industry|Other|Solids|Coal
- Final Energy|Industry|Pulp and paper
- Final Energy|Industry|Pulp and paper|Electricity
- Final Energy|Industry|Pulp and paper|Gases
- Final Energy|Industry|Pulp and paper|Gases|Bioenergy
- Final Energy|Industry|Pulp and paper|Gases|Fossil
- Final Energy|Industry|Pulp and paper|Hydrogen
- Final Energy|Industry|Pulp and paper|Liquids
- Final Energy|Industry|Pulp and paper|Liquids|Bioenergy
- Final Energy|Industry|Pulp and paper|Liquids|Fossil
- Final Energy|Industry|Pulp and paper|Other Fuel
- Final Energy|Industry|Pulp and paper|Solids
- Final Energy|Industry|Pulp and paper|Solids|Biomass
- Final Energy|Industry|Pulp and paper|Solids|Coal
- Final Energy|Industry|Solids
- Final Energy|Industry|Solids|Bioenergy
- Final Energy|Industry|Solids|Biomass
- Final Energy|Industry|Solids|Coal
- Final Energy|Industry|Solids|Fossil
- Final Energy|Industry|Solids|Fossils
- Final Energy|Industry|Steel
- Final Energy|Industry|Steel|
- Final Energy|Industry|Steel|Biomass
- Final Energy|Industry|Steel|Coal
- Final Energy|Industry|Steel|Electricity
- Final Energy|Industry|Steel|Gases
- Final Energy|Industry|Steel|Gases|Bioenergy
- Final Energy|Industry|Steel|Gases|Bioenergy|Biomethane
- Final Energy|Industry|Steel|Gases|Fossil
- Final Energy|Industry|Steel|Gases|Fossil|Natural Gas
- Final Energy|Industry|Steel|Gases|Fossil|Off-gases
- Final Energy|Industry|Steel|Gases|Hydrogen synfuel
- Final Energy|Industry|Steel|Hydrogen
- Final Energy|Industry|Steel|Liquids
- Final Energy|Industry|Steel|Liquids|Bioenergy
- Final Energy|Industry|Steel|Liquids|Fossil
- Final Energy|Industry|Steel|Liquids|Hydrogen synfuel
- Final Energy|Industry|Steel|Natural gas
- Final Energy|Industry|Steel|Other Fuel
- Final Energy|Industry|Steel|Solids
- Final Energy|Industry|Steel|Solids|Bioenergy
- Final Energy|Industry|Steel|Solids|Biomass
- Final Energy|Industry|Steel|Solids|Coal
- Final Energy|Industry|Steel|Solids|Fossil
- Final Energy|Industry|Steel|Solids|Fossil|Coal
- Final Energy|Liquids
- Final Energy|Liquids|Bioenergy
- Final Energy|Liquids|Biomass
- Final Energy|Liquids|Fossil
- Final Energy|Non-Energy Use
- Final Energy|Non-Energy Use|Biomass
- Final Energy|Non-Energy Use|Coal
- Final Energy|Non-Energy Use|Gas
- Final Energy|Non-Energy Use|Gases
- Final Energy|Non-Energy Use|Liquids
- Final Energy|Non-Energy Use|Oil
- Final Energy|Non-Energy Use|Solids
- Final Energy|Other
- Final Energy|Other Fuel
- Final Energy|Other Sector
- Final Energy|Other Sector|Electricity
- Final Energy|Other Sector|Gases
- Final Energy|Other Sector|Gases|Bioenergy
- Final Energy|Other Sector|Gases|Fossil
- Final Energy|Other Sector|Heat
- Final Energy|Other Sector|Hydrogen
- Final Energy|Other Sector|Liquids
- Final Energy|Other Sector|Liquids|Bioenergy
- Final Energy|Other Sector|Liquids|Fossil
- Final Energy|Other Sector|Liquids|Oil
- Final Energy|Other Sector|Other
- Final Energy|Other Sector|Other Fuel
- Final Energy|Other Sector|Solids
- Final Energy|Other Sector|Solids|Biomass
- Final Energy|Other Sector|Solids|Coal
- Final Energy|Other|Electricity
- Final Energy|Other|Gases
- Final Energy|Other|Heat
- Final Energy|Other|Liquids
- Final Energy|Other|Solids
- Final Energy|Other|Solids|Biomass
- Final Energy|Other|Solids|Coal
- Final Energy|Residential
- Final Energy|Residential & Commercial|Appliances
- Final Energy|Residential & Commercial|Cooking
- Final Energy|Residential & Commercial|Lighting
- Final Energy|Residential & Commercial|Solids|Fossil
- Final Energy|Residential & Commercial|Space Cooling
- Final Energy|Residential & Commercial|Space Heating
- Final Energy|Residential & Commercial|Water Heating
- Final Energy|Residential and Commercial
- Final Energy|Residential and Commercial|Commercial
- Final Energy|Residential and Commercial|Commercial|Appliances
- Final Energy|Residential and Commercial|Commercial|Cooling
- Final Energy|Residential and Commercial|Commercial|Electricity
- Final Energy|Residential and Commercial|Commercial|Gases
- Final Energy|Residential and Commercial|Commercial|Gases|Bioenergy
- Final Energy|Residential and Commercial|Commercial|Gases|Fossil
- Final Energy|Residential and Commercial|Commercial|Gases|Fossils
- Final Energy|Residential and Commercial|Commercial|Heat
- Final Energy|Residential and Commercial|Commercial|Heating
- Final Energy|Residential and Commercial|Commercial|Heating|Cooking
- Final Energy|Residential and Commercial|Commercial|Heating|Cooking|Electricity
- Final Energy|Residential and Commercial|Commercial|Heating|Cooking|Gases
- Final Energy|Residential and Commercial|Commercial|Heating|Cooking|Gases|Bioenergy
- Final Energy|Residential and Commercial|Commercial|Heating|Cooking|Gases|Fossil
- Final Energy|Residential and Commercial|Commercial|Heating|Cooking|Hydrogen
- Final Energy|Residential and Commercial|Commercial|Heating|Cooking|Liquids
- Final Energy|Residential and Commercial|Commercial|Heating|Cooking|Liquids|Bioenergy
- Final Energy|Residential and Commercial|Commercial|Heating|Cooking|Liquids|Fossil
- Final Energy|Residential and Commercial|Commercial|Heating|Cooking|Other Fuel
- Final Energy|Residential and Commercial|Commercial|Heating|Cooking|Solids
- Final Energy|Residential and Commercial|Commercial|Heating|Cooking|Solids|Biomass
- Final Energy|Residential and Commercial|Commercial|Heating|Cooking|Solids|Coal
- Final Energy|Residential and Commercial|Commercial|Heating|Space
- Final Energy|Residential and Commercial|Commercial|Heating|Space|Electricity
- Final Energy|Residential and Commercial|Commercial|Heating|Space|Gases
- Final Energy|Residential and Commercial|Commercial|Heating|Space|Gases|Bioenergy
- Final Energy|Residential and Commercial|Commercial|Heating|Space|Gases|Fossil
- Final Energy|Residential and Commercial|Commercial|Heating|Space|Hydrogen
- Final Energy|Residential and Commercial|Commercial|Heating|Space|Liquids
- Final Energy|Residential and Commercial|Commercial|Heating|Space|Liquids|Bioenergy
- Final Energy|Residential and Commercial|Commercial|Heating|Space|Liquids|Fossil
- Final Energy|Residential and Commercial|Commercial|Heating|Space|Other Fuel
- Final Energy|Residential and Commercial|Commercial|Heating|Space|Solids
- Final Energy|Residential and Commercial|Commercial|Heating|Space|Solids|Biomass
- Final Energy|Residential and Commercial|Commercial|Heating|Space|Solids|Coal
- Final Energy|Residential and Commercial|Commercial|Heating|Water
- Final Energy|Residential and Commercial|Commercial|Hydrogen
- Final Energy|Residential and Commercial|Commercial|Lighting
- Final Energy|Residential and Commercial|Commercial|Liquids
- Final Energy|Residential and Commercial|Commercial|Liquids|Bioenergy
- Final Energy|Residential and Commercial|Commercial|Liquids|Fossil
- Final Energy|Residential and Commercial|Commercial|Liquids|Fossils
- Final Energy|Residential and Commercial|Commercial|Ofice equipment
- Final Energy|Residential and Commercial|Commercial|Other
- Final Energy|Residential and Commercial|Commercial|Other Fuel
- Final Energy|Residential and Commercial|Commercial|Other|Electricity
- Final Energy|Residential and Commercial|Commercial|Other|Gases
- Final Energy|Residential and Commercial|Commercial|Other|Gases|Bioenergy
- Final Energy|Residential and Commercial|Commercial|Other|Gases|Fossil
- Final Energy|Residential and Commercial|Commercial|Other|Hydrogen
- Final Energy|Residential and Commercial|Commercial|Other|Liquids
- Final Energy|Residential and Commercial|Commercial|Other|Liquids|Bioenergy
- Final Energy|Residential and Commercial|Commercial|Other|Liquids|Fossil
- Final Energy|Residential and Commercial|Commercial|Other|Other Fuel
- Final Energy|Residential and Commercial|Commercial|Other|Solids
- Final Energy|Residential and Commercial|Commercial|Other|Solids|Biomass
- Final Energy|Residential and Commercial|Commercial|Other|Solids|Coal
- Final Energy|Residential and Commercial|Commercial|Solids
- Final Energy|Residential and Commercial|Commercial|Solids|Biomass
- Final Energy|Residential and Commercial|Commercial|Solids|Coal
- Final Energy|Residential and Commercial|Commercial|Solids|Fossils
- Final Energy|Residential and Commercial|Commercial|Water
- Final Energy|Residential and Commercial|Commercial|Water|Electricity
- Final Energy|Residential and Commercial|Commercial|Water|Gases
- Final Energy|Residential and Commercial|Commercial|Water|Gases|Bioenergy
- Final Energy|Residential and Commercial|Commercial|Water|Gases|Fossil
- Final Energy|Residential and Commercial|Commercial|Water|Hydrogen
- Final Energy|Residential and Commercial|Commercial|Water|Liquids
- Final Energy|Residential and Commercial|Commercial|Water|Liquids|Bioenergy
- Final Energy|Residential and Commercial|Commercial|Water|Liquids|Fossil
- Final Energy|Residential and Commercial|Commercial|Water|Other Fuel
- Final Energy|Residential and Commercial|Commercial|Water|Solids
- Final Energy|Residential and Commercial|Commercial|Water|Solids|Biomass
- Final Energy|Residential and Commercial|Commercial|Water|Solids|Coal
- Final Energy|Residential and Commercial|Cooling
- Final Energy|Residential and Commercial|Electricity
- Final Energy|Residential and Commercial|Gases
- Final Energy|Residential and Commercial|Gases|Bioenergy
- Final Energy|Residential and Commercial|Gases|Electricity
- Final Energy|Residential and Commercial|Gases|Fossil
- Final Energy|Residential and Commercial|Gases|Fossils
- Final Energy|Residential and Commercial|Heat
- Final Energy|Residential and Commercial|Heating|Cooking
- Final Energy|Residential and Commercial|Heating|Cooking|Electricity
- Final Energy|Residential and Commercial|Heating|Cooking|Gases
- Final Energy|Residential and Commercial|Heating|Cooking|Gases|Bioenergy
- Final Energy|Residential and Commercial|Heating|Cooking|Gases|Fossil
- Final Energy|Residential and Commercial|Heating|Cooking|Hydrogen
- Final Energy|Residential and Commercial|Heating|Cooking|Liquids
- Final Energy|Residential and Commercial|Heating|Cooking|Liquids|Bioenergy
- Final Energy|Residential and Commercial|Heating|Cooking|Liquids|Fossil
- Final Energy|Residential and Commercial|Heating|Cooking|Other Fuel
- Final Energy|Residential and Commercial|Heating|Cooking|Solids
- Final Energy|Residential and Commercial|Heating|Cooking|Solids|Biomass
- Final Energy|Residential and Commercial|Heating|Cooking|Solids|Coal
- Final Energy|Residential and Commercial|Heating|Space
- Final Energy|Residential and Commercial|Heating|Space|Electricity
- Final Energy|Residential and Commercial|Heating|Space|Gases
- Final Energy|Residential and Commercial|Heating|Space|Gases|Bioenergy
- Final Energy|Residential and Commercial|Heating|Space|Gases|Fossil
- Final Energy|Residential and Commercial|Heating|Space|Hydrogen
- Final Energy|Residential and Commercial|Heating|Space|Liquids
- Final Energy|Residential and Commercial|Heating|Space|Liquids|Bioenergy
- Final Energy|Residential and Commercial|Heating|Space|Liquids|Fossil
- Final Energy|Residential and Commercial|Heating|Space|Other Fuel
- Final Energy|Residential and Commercial|Heating|Space|Solids
- Final Energy|Residential and Commercial|Heating|Space|Solids|Biomass
- Final Energy|Residential and Commercial|Heating|Space|Solids|Coal
- Final Energy|Residential and Commercial|Hydrogen
- Final Energy|Residential and Commercial|Liquids
- Final Energy|Residential and Commercial|Liquids|Bioenergy
- Final Energy|Residential and Commercial|Liquids|Biomass
- Final Energy|Residential and Commercial|Liquids|Fossil
- Final Energy|Residential and Commercial|Liquids|Fossils
- Final Energy|Residential and Commercial|Other
- Final Energy|Residential and Commercial|Other Fuel
- Final Energy|Residential and Commercial|Other|Electricity
- Final Energy|Residential and Commercial|Other|Gases
- Final Energy|Residential and Commercial|Other|Gases|Bioenergy
- Final Energy|Residential and Commercial|Other|Gases|Fossil
- Final Energy|Residential and Commercial|Other|Hydrogen
- Final Energy|Residential and Commercial|Other|Liquids
- Final Energy|Residential and Commercial|Other|Liquids|Bioenergy
- Final Energy|Residential and Commercial|Other|Liquids|Fossil
- Final Energy|Residential and Commercial|Other|Other Fuel
- Final Energy|Residential and Commercial|Other|Solids
- Final Energy|Residential and Commercial|Other|Solids|Biomass
- Final Energy|Residential and Commercial|Other|Solids|Coal
- Final Energy|Residential and Commercial|Residential
- Final Energy|Residential and Commercial|Residential|Cooling
- Final Energy|Residential and Commercial|Residential|Electricity
- Final Energy|Residential and Commercial|Residential|Gases
- Final Energy|Residential and Commercial|Residential|Gases|Bioenergy
- Final Energy|Residential and Commercial|Residential|Gases|Fossil
- Final Energy|Residential and Commercial|Residential|Gases|Fossils
- Final Energy|Residential and Commercial|Residential|Heat
- Final Energy|Residential and Commercial|Residential|Heating
- Final Energy|Residential and Commercial|Residential|Heating|Cooking
- Final Energy|Residential and Commercial|Residential|Heating|Cooking|Electricity
- Final Energy|Residential and Commercial|Residential|Heating|Cooking|Gases
- Final Energy|Residential and Commercial|Residential|Heating|Cooking|Gases|Bioenergy
- Final Energy|Residential and Commercial|Residential|Heating|Cooking|Gases|Fossil
- Final Energy|Residential and Commercial|Residential|Heating|Cooking|Hydrogen
- Final Energy|Residential and Commercial|Residential|Heating|Cooking|Liquids
- Final Energy|Residential and Commercial|Residential|Heating|Cooking|Liquids|Bioenergy
- Final Energy|Residential and Commercial|Residential|Heating|Cooking|Liquids|Fossil
- Final Energy|Residential and Commercial|Residential|Heating|Cooking|Other Fuel
- Final Energy|Residential and Commercial|Residential|Heating|Cooking|Solids
- Final Energy|Residential and Commercial|Residential|Heating|Cooking|Solids|Biomass
- Final Energy|Residential and Commercial|Residential|Heating|Cooking|Solids|Coal
- Final Energy|Residential and Commercial|Residential|Heating|Space
- Final Energy|Residential and Commercial|Residential|Heating|Space|Electricity
- Final Energy|Residential and Commercial|Residential|Heating|Space|Gases
- Final Energy|Residential and Commercial|Residential|Heating|Space|Gases|Bioenergy
- Final Energy|Residential and Commercial|Residential|Heating|Space|Gases|Fossil
- Final Energy|Residential and Commercial|Residential|Heating|Space|Hydrogen
- Final Energy|Residential and Commercial|Residential|Heating|Space|Liquids
- Final Energy|Residential and Commercial|Residential|Heating|Space|Liquids|Bioenergy
- Final Energy|Residential and Commercial|Residential|Heating|Space|Liquids|Fossil
- Final Energy|Residential and Commercial|Residential|Heating|Space|Other Fuel
- Final Energy|Residential and Commercial|Residential|Heating|Space|Solids
- Final Energy|Residential and Commercial|Residential|Heating|Space|Solids|Biomass
- Final Energy|Residential and Commercial|Residential|Heating|Space|Solids|Coal
- Final Energy|Residential and Commercial|Residential|Heating|Water
- Final Energy|Residential and Commercial|Residential|Hydrogen
- Final Energy|Residential and Commercial|Residential|Liquids
- Final Energy|Residential and Commercial|Residential|Liquids|Bioenergy
- Final Energy|Residential and Commercial|Residential|Liquids|Fossil
- Final Energy|Residential and Commercial|Residential|Liquids|Fossils
- Final Energy|Residential and Commercial|Residential|Other
- Final Energy|Residential and Commercial|Residential|Other Fuel
- Final Energy|Residential and Commercial|Residential|Other|Electricity
- Final Energy|Residential and Commercial|Residential|Other|Gases
- Final Energy|Residential and Commercial|Residential|Other|Gases|Bioenergy
- Final Energy|Residential and Commercial|Residential|Other|Gases|Fossil
- Final Energy|Residential and Commercial|Residential|Other|Hydrogen
- Final Energy|Residential and Commercial|Residential|Other|Liquids
- Final Energy|Residential and Commercial|Residential|Other|Liquids|Bioenergy
- Final Energy|Residential and Commercial|Residential|Other|Liquids|Fossil
- Final Energy|Residential and Commercial|Residential|Other|Other Fuel
- Final Energy|Residential and Commercial|Residential|Other|Solids
- Final Energy|Residential and Commercial|Residential|Other|Solids|Biomass
- Final Energy|Residential and Commercial|Residential|Other|Solids|Coal
- Final Energy|Residential and Commercial|Residential|Solids
- Final Energy|Residential and Commercial|Residential|Solids|Biomass
- Final Energy|Residential and Commercial|Residential|Solids|Coal
- Final Energy|Residential and Commercial|Residential|Solids|Fossils
- Final Energy|Residential and Commercial|Residential|Water
- Final Energy|Residential and Commercial|Residential|Water|Electricity
- Final Energy|Residential and Commercial|Residential|Water|Gases
- Final Energy|Residential and Commercial|Residential|Water|Gases|Bioenergy
- Final Energy|Residential and Commercial|Residential|Water|Gases|Fossil
- Final Energy|Residential and Commercial|Residential|Water|Hydrogen
- Final Energy|Residential and Commercial|Residential|Water|Liquids
- Final Energy|Residential and Commercial|Residential|Water|Liquids|Bioenergy
- Final Energy|Residential and Commercial|Residential|Water|Liquids|Fossil
- Final Energy|Residential and Commercial|Residential|Water|Other Fuel
- Final Energy|Residential and Commercial|Residential|Water|Solids
- Final Energy|Residential and Commercial|Residential|Water|Solids|Biomass
- Final Energy|Residential and Commercial|Residential|Water|Solids|Coal
- Final Energy|Residential and Commercial|Solids
- Final Energy|Residential and Commercial|Solids|Biomass
- Final Energy|Residential and Commercial|Solids|Biomass|Traditional
- Final Energy|Residential and Commercial|Solids|Coal
- Final Energy|Residential and Commercial|Solids|Fossils
- Final Energy|Residential and Commercial|Water
- Final Energy|Residential and Commercial|Water|Electricity
- Final Energy|Residential and Commercial|Water|Gases
- Final Energy|Residential and Commercial|Water|Gases|Bioenergy
- Final Energy|Residential and Commercial|Water|Gases|Fossil
- Final Energy|Residential and Commercial|Water|Hydrogen
- Final Energy|Residential and Commercial|Water|Liquids
- Final Energy|Residential and Commercial|Water|Liquids|Bioenergy
- Final Energy|Residential and Commercial|Water|Liquids|Fossil
- Final Energy|Residential and Commercial|Water|Other Fuel
- Final Energy|Residential and Commercial|Water|Solids
- Final Energy|Residential and Commercial|Water|Solids|Biomass
- Final Energy|Residential and Commercial|Water|Solids|Coal
- Final Energy|Residential|Black Appliances
- Final Energy|Residential|Cooking
- Final Energy|Residential|Electricity
- Final Energy|Residential|Gases
- Final Energy|Residential|Heat
- Final Energy|Residential|Hydrogen
- Final Energy|Residential|Lighting
- Final Energy|Residential|Liquids
- Final Energy|Residential|Other
- Final Energy|Residential|Solids
- Final Energy|Residential|Solids|Biomass
- Final Energy|Residential|Solids|Fossil
- Final Energy|Residential|Space Cooling
- Final Energy|Residential|Space Heating
- Final Energy|Residential|Water Heating
- Final Energy|Residential|White Appliances
- Final Energy|Solar
- Final Energy|Solids
- Final Energy|Solids|Biomass
- Final Energy|Solids|Biomass|Traditional
- Final Energy|Solids|Coal
- Final Energy|Solids|Fossil
- Final Energy|Transportation
- Final Energy|Transportation (w/ bunkers)
- Final Energy|Transportation|Aviation
- Final Energy|Transportation|Aviation|Electricity
- Final Energy|Transportation|Aviation|Freight
- Final Energy|Transportation|Aviation|Freight|Electricity
- Final Energy|Transportation|Aviation|Freight|Gases
- Final Energy|Transportation|Aviation|Freight|Gases|Bioenergy
- Final Energy|Transportation|Aviation|Freight|Gases|Fossil
- Final Energy|Transportation|Aviation|Freight|Hydrogen
- Final Energy|Transportation|Aviation|Freight|Liquids
- Final Energy|Transportation|Aviation|Freight|Liquids|Bioenergy
- Final Energy|Transportation|Aviation|Freight|Liquids|Fossil
- Final Energy|Transportation|Aviation|Freight|Other Fuel
- Final Energy|Transportation|Aviation|Freight|Solids
- Final Energy|Transportation|Aviation|Freight|Solids|Biomass
- Final Energy|Transportation|Aviation|Freight|Solids|Coal
- Final Energy|Transportation|Aviation|Gases
- Final Energy|Transportation|Aviation|Gases|Bioenergy
- Final Energy|Transportation|Aviation|Gases|Fossil
- Final Energy|Transportation|Aviation|Hydrogen
- Final Energy|Transportation|Aviation|Liquids
- Final Energy|Transportation|Aviation|Liquids|Bioenergy
- Final Energy|Transportation|Aviation|Liquids|Fossil
- Final Energy|Transportation|Aviation|Other Fuel
- Final Energy|Transportation|Aviation|Passenger
- Final Energy|Transportation|Aviation|Passenger|Electricity
- Final Energy|Transportation|Aviation|Passenger|Gases
- Final Energy|Transportation|Aviation|Passenger|Gases|Bioenergy
- Final Energy|Transportation|Aviation|Passenger|Gases|Fossil
- Final Energy|Transportation|Aviation|Passenger|Hydrogen
- Final Energy|Transportation|Aviation|Passenger|Liquids
- Final Energy|Transportation|Aviation|Passenger|Liquids|Bioenergy
- Final Energy|Transportation|Aviation|Passenger|Liquids|Fossil
- Final Energy|Transportation|Aviation|Passenger|Other Fuel
- Final Energy|Transportation|Aviation|Passenger|Solids
- Final Energy|Transportation|Aviation|Passenger|Solids|Biomass
- Final Energy|Transportation|Aviation|Passenger|Solids|Coal
- Final Energy|Transportation|Aviation|Solids
- Final Energy|Transportation|Aviation|Solids|Biomass
- Final Energy|Transportation|Aviation|Solids|Coal
- Final Energy|Transportation|Electricity
- Final Energy|Transportation|Electricity|Rail
- Final Energy|Transportation|Electricity|Road|Freight
- Final Energy|Transportation|Electricity|Road|Passenger
- Final Energy|Transportation|Electricity|Road|Passenger|Cars
- Final Energy|Transportation|Freight
- Final Energy|Transportation|Freight|Electricity
- Final Energy|Transportation|Freight|Gases
- Final Energy|Transportation|Freight|Hydrogen
- Final Energy|Transportation|Freight|Liquid
- Final Energy|Transportation|Freight|Liquids
- Final Energy|Transportation|Freight|Liquids|Bioenergy
- Final Energy|Transportation|Freight|Liquids|Fossil synfuel
- Final Energy|Transportation|Freight|Liquids|Oil
- Final Energy|Transportation|Freight|Other
- Final Energy|Transportation|Freight|Solid
- Final Energy|Transportation|Gases
- Final Energy|Transportation|Gases|Bioenergy
- Final Energy|Transportation|Gases|Electricity
- Final Energy|Transportation|Gases|Fossil
- Final Energy|Transportation|Hydrogen
- Final Energy|Transportation|Hydrogen|Electricity
- Final Energy|Transportation|LDV|BEV
- Final Energy|Transportation|LDV|Electricity
- Final Energy|Transportation|LDV|Gases
- Final Energy|Transportation|LDV|Gases|Electricity
- Final Energy|Transportation|LDV|Gases|Fossil
- Final Energy|Transportation|LDV|Hybrid
- Final Energy|Transportation|LDV|Hybrid|Electricity
- Final Energy|Transportation|LDV|Hybrid|Gases
- Final Energy|Transportation|LDV|Hybrid|Liquids
- Final Energy|Transportation|LDV|Hydrogen
- Final Energy|Transportation|LDV|Hydrogen|Electricity
- Final Energy|Transportation|LDV|ICE
- Final Energy|Transportation|LDV|ICE|Gases
- Final Energy|Transportation|LDV|ICE|Liquids
- Final Energy|Transportation|LDV|Liquids
- Final Energy|Transportation|LDV|Liquids|Biomass
- Final Energy|Transportation|LDV|Liquids|Fossil
- Final Energy|Transportation|Liquids
- Final Energy|Transportation|Liquids|Bioenergy
- Final Energy|Transportation|Liquids|Biomass
- Final Energy|Transportation|Liquids|Fossil
- Final Energy|Transportation|Liquids|Fossil synfuel
- Final Energy|Transportation|Liquids|Oil
- Final Energy|Transportation|Maritime
- Final Energy|Transportation|Maritime|Electricity
- Final Energy|Transportation|Maritime|Freight
- Final Energy|Transportation|Maritime|Freight|Electricity
- Final Energy|Transportation|Maritime|Freight|Gases
- Final Energy|Transportation|Maritime|Freight|Gases|Bioenergy
- Final Energy|Transportation|Maritime|Freight|Gases|Fossil
- Final Energy|Transportation|Maritime|Freight|Hydrogen
- Final Energy|Transportation|Maritime|Freight|Liquids
- Final Energy|Transportation|Maritime|Freight|Liquids|Bioenergy
- Final Energy|Transportation|Maritime|Freight|Liquids|Fossil
- Final Energy|Transportation|Maritime|Freight|Other Fuel
- Final Energy|Transportation|Maritime|Freight|Solids
- Final Energy|Transportation|Maritime|Freight|Solids|Biomass
- Final Energy|Transportation|Maritime|Freight|Solids|Coal
- Final Energy|Transportation|Maritime|Gases
- Final Energy|Transportation|Maritime|Gases|Bioenergy
- Final Energy|Transportation|Maritime|Gases|Fossil
- Final Energy|Transportation|Maritime|Hydrogen
- Final Energy|Transportation|Maritime|Liquids
- Final Energy|Transportation|Maritime|Liquids|Bioenergy
- Final Energy|Transportation|Maritime|Liquids|Fossil
- Final Energy|Transportation|Maritime|Other Fuel
- Final Energy|Transportation|Maritime|Solids
- Final Energy|Transportation|Maritime|Solids|Biomass
- Final Energy|Transportation|Maritime|Solids|Coal
- Final Energy|Transportation|Other
- Final Energy|Transportation|Other Fuel
- Final Energy|Transportation|Passenger
- Final Energy|Transportation|Passenger|Electricity
- Final Energy|Transportation|Passenger|Gases
- Final Energy|Transportation|Passenger|Hydrogen
- Final Energy|Transportation|Passenger|Liquid
- Final Energy|Transportation|Passenger|Liquids
- Final Energy|Transportation|Passenger|Liquids|Bioenergy
- Final Energy|Transportation|Passenger|Liquids|Fossil synfuel
- Final Energy|Transportation|Passenger|Liquids|Oil
- Final Energy|Transportation|Passenger|Solid
- Final Energy|Transportation|Rail
- Final Energy|Transportation|Rail|Electricity
- Final Energy|Transportation|Rail|Freight
- Final Energy|Transportation|Rail|Freight|Electricity
- Final Energy|Transportation|Rail|Freight|Gases
- Final Energy|Transportation|Rail|Freight|Gases|Bioenergy
- Final Energy|Transportation|Rail|Freight|Gases|Fossil
- Final Energy|Transportation|Rail|Freight|Hydrogen
- Final Energy|Transportation|Rail|Freight|Liquids
- Final Energy|Transportation|Rail|Freight|Liquids|Bioenergy
- Final Energy|Transportation|Rail|Freight|Liquids|Fossil
- Final Energy|Transportation|Rail|Freight|Other Fuel
- Final Energy|Transportation|Rail|Freight|Solids
- Final Energy|Transportation|Rail|Freight|Solids|Biomass
- Final Energy|Transportation|Rail|Freight|Solids|Coal
- Final Energy|Transportation|Rail|Gases
- Final Energy|Transportation|Rail|Gases|Bioenergy
- Final Energy|Transportation|Rail|Gases|Fossil
- Final Energy|Transportation|Rail|Hydrogen
- Final Energy|Transportation|Rail|Liquids
- Final Energy|Transportation|Rail|Liquids|Bioenergy
- Final Energy|Transportation|Rail|Liquids|Fossil
- Final Energy|Transportation|Rail|Other Fuel
- Final Energy|Transportation|Rail|Passenger
- Final Energy|Transportation|Rail|Passenger|Electricity
- Final Energy|Transportation|Rail|Passenger|Gases
- Final Energy|Transportation|Rail|Passenger|Gases|Bioenergy
- Final Energy|Transportation|Rail|Passenger|Gases|Fossil
- Final Energy|Transportation|Rail|Passenger|Hydrogen
- Final Energy|Transportation|Rail|Passenger|Liquids
- Final Energy|Transportation|Rail|Passenger|Liquids|Bioenergy
- Final Energy|Transportation|Rail|Passenger|Liquids|Fossil
- Final Energy|Transportation|Rail|Passenger|Other Fuel
- Final Energy|Transportation|Rail|Passenger|Solids
- Final Energy|Transportation|Rail|Passenger|Solids|Biomass
- Final Energy|Transportation|Rail|Passenger|Solids|Coal
- Final Energy|Transportation|Rail|Solids
- Final Energy|Transportation|Rail|Solids|Biomass
- Final Energy|Transportation|Rail|Solids|Coal
- Final Energy|Transportation|Road
- Final Energy|Transportation|Road|Electricity
- Final Energy|Transportation|Road|Freight
- Final Energy|Transportation|Road|Freight|Electric
- Final Energy|Transportation|Road|Freight|Electricity
- Final Energy|Transportation|Road|Freight|FC
- Final Energy|Transportation|Road|Freight|Gases
- Final Energy|Transportation|Road|Freight|Gases|Bioenergy
- Final Energy|Transportation|Road|Freight|Gases|Fossil
- Final Energy|Transportation|Road|Freight|Hydrogen
- Final Energy|Transportation|Road|Freight|Liquids
- Final Energy|Transportation|Road|Freight|Liquids|Bioenergy
- Final Energy|Transportation|Road|Freight|Liquids|Fossil
- Final Energy|Transportation|Road|Freight|Other Fuel
- Final Energy|Transportation|Road|Freight|Solids
- Final Energy|Transportation|Road|Freight|Solids|Biomass
- Final Energy|Transportation|Road|Freight|Solids|Coal
- Final Energy|Transportation|Road|Gases
- Final Energy|Transportation|Road|Gases|Bioenergy
- Final Energy|Transportation|Road|Gases|Fossil
- Final Energy|Transportation|Road|Hydrogen
- Final Energy|Transportation|Road|Liquids
- Final Energy|Transportation|Road|Liquids|Bioenergy
- Final Energy|Transportation|Road|Liquids|Fossil
- Final Energy|Transportation|Road|Other Fuel
- Final Energy|Transportation|Road|Passenger
- Final Energy|Transportation|Road|Passenger|2W&3W
- Final Energy|Transportation|Road|Passenger|4W
- Final Energy|Transportation|Road|Passenger|Bus
- Final Energy|Transportation|Road|Passenger|Bus|Electricity
- Final Energy|Transportation|Road|Passenger|Bus|Gases
- Final Energy|Transportation|Road|Passenger|Bus|Gases|Bioenergy
- Final Energy|Transportation|Road|Passenger|Bus|Gases|Fossil
- Final Energy|Transportation|Road|Passenger|Bus|Hydrogen
- Final Energy|Transportation|Road|Passenger|Bus|Liquids
- Final Energy|Transportation|Road|Passenger|Bus|Liquids|Bioenergy
- Final Energy|Transportation|Road|Passenger|Bus|Liquids|Fossil
- Final Energy|Transportation|Road|Passenger|Bus|Other Fuel
- Final Energy|Transportation|Road|Passenger|Bus|Solids
- Final Energy|Transportation|Road|Passenger|Bus|Solids|Biomass
- Final Energy|Transportation|Road|Passenger|Bus|Solids|Coal
- Final Energy|Transportation|Road|Passenger|LDV
- Final Energy|Transportation|Road|Passenger|LDV|Electricity
- Final Energy|Transportation|Road|Passenger|LDV|Gases
- Final Energy|Transportation|Road|Passenger|LDV|Gases|Bioenergy
- Final Energy|Transportation|Road|Passenger|LDV|Gases|Fossil
- Final Energy|Transportation|Road|Passenger|LDV|Hydrogen
- Final Energy|Transportation|Road|Passenger|LDV|Liquids
- Final Energy|Transportation|Road|Passenger|LDV|Liquids|Bioenergy
- Final Energy|Transportation|Road|Passenger|LDV|Liquids|Fossil
- Final Energy|Transportation|Road|Passenger|LDV|Other Fuel
- Final Energy|Transportation|Road|Passenger|LDV|Solids
- Final Energy|Transportation|Road|Passenger|LDV|Solids|Biomass
- Final Energy|Transportation|Road|Passenger|LDV|Solids|Coal
- Final Energy|Transportation|Road|Solids
- Final Energy|Transportation|Road|Solids|Biomass
- Final Energy|Transportation|Road|Solids|Coal
- Final Energy|Transportation|Solids
- Final Energy|Transportation|Solids|Biomass
- Final Energy|Transportation|Solids|Coal
- Final Energy|Water
- Final Energy|Water|Desalination
- Final Energy|Water|Desalination|Electricity
- Final Energy|Water|Desalination|Gases
- Final Energy|Water|Desalination|Liquids
- Final Energy|Water|Industry
- Final Energy|Water|Industry|Electricity
- Final Energy|Water|Irrigation
- Final Energy|Water|Irrigation|Electricity
- Final Energy|Water|Municipal
- Final Energy|Water|Municipal|Electricity
- Fixed Operating Cost|Energy Supply
- Flood Risk|Adaptation Investments
- Flood Risk|Damages
- Forcing
- GDP
- GDP|Inventories
- GDP|MER
- GDP|PPP
- Geothermal Investment Share
- Gross Emissions|CH4
- Gross Emissions|CO2 energy
- Gross Emissions|CO2 process
- Gross Emissions|CO2|Energy|Demand|Industry
- Gross Emissions|CO2|Industrial Processes
- Gross Emissions|GHG
- Gross Emissions|HFC
- Gross Emissions|N2O
- Gross Emissions|PFC
- Gross Emissions|SF6
- Housing Electricity Demand
- Hydro Investment Share
- IDA|Emissions|CO2|Energy|Demand|Buildings|CO2 intensity
- IDA|Emissions|CO2|Energy|Demand|Buildings|Electrification
- IDA|Emissions|CO2|Energy|Demand|Buildings|Emission end level 2050
- IDA|Emissions|CO2|Energy|Demand|Buildings|Energy efficiency
- IDA|Emissions|CO2|Energy|Demand|Buildings|Heat
- IDA|Emissions|CO2|Energy|Demand|Buildings|Population
- IDA|Emissions|CO2|Energy|Demand|Buildings|PtG/PtL
- IDA|Emissions|CO2|Energy|Demand|Buildings|Renewables
- IDA|Emissions|CO2|Energy|Demand|Industry|CCS
- IDA|Emissions|CO2|Energy|Demand|Industry|CO2 intensity
- IDA|Emissions|CO2|Energy|Demand|Industry|Electrification
- IDA|Emissions|CO2|Energy|Demand|Industry|Emission end level 2050
- IDA|Emissions|CO2|Energy|Demand|Industry|Energy efficiency
- IDA|Emissions|CO2|Energy|Demand|Industry|GDP
- IDA|Emissions|CO2|Energy|Demand|Industry|GDP rel. to WWH
- IDA|Emissions|CO2|Energy|Demand|Industry|Heat
- IDA|Emissions|CO2|Energy|Demand|Industry|PtG/PtL
- IDA|Emissions|CO2|Energy|Demand|Industry|Renewables
- IDA|Emissions|CO2|Energy|Demand|Transport|CO2 intensity
- IDA|Emissions|CO2|Energy|Demand|Transport|Electrification
- IDA|Emissions|CO2|Energy|Demand|Transport|Emission end level 2050
- IDA|Emissions|CO2|Energy|Demand|Transport|Energy efficiency
- IDA|Emissions|CO2|Energy|Demand|Transport|Population
- IDA|Emissions|CO2|Energy|Demand|Transport|PtG/PtL
- IDA|Emissions|CO2|Energy|Demand|Transport|Renewables
- IDA|Emissions|CO2|Energy|Supply|Electricity|CCS
- IDA|Emissions|CO2|Energy|Supply|Electricity|CO2 intensity
- IDA|Emissions|CO2|Energy|Supply|Electricity|Conversion efficiency
- IDA|Emissions|CO2|Energy|Supply|Electricity|Emission end level 2050
- IDA|Emissions|CO2|Energy|Supply|Electricity|Nuclear
- IDA|Emissions|CO2|Energy|Supply|Electricity|Power generation
- IDA|Emissions|CO2|Energy|Supply|Electricity|Power generation rel. to WWH
- IDA|Emissions|CO2|Energy|Supply|Electricity|Renewables
- Import
- Imports
- Imports|Agriculture
- Imports|Air transport
- Imports|Basic metals
- Imports|Basic pharmaceutical products
- Imports|Batteries
- Imports|Biofuels
- Imports|Biomass
- Imports|Biomass Solid
- Imports|CCS coal
- Imports|CSS Bio
- Imports|CSS Gas
- Imports|Chemical Products
- Imports|Clean Gas
- Imports|Coal fired
- Imports|Computer, electronic and optical products
- Imports|Construction
- Imports|Consumer Goods Industries
- Imports|EV Transport Equipment
- Imports|Electrical Equipment
- Imports|Electricity
- Imports|Fabricated Metal products
- Imports|Gas Fired
- Imports|Gas supply
- Imports|Geothermal
- Imports|Hydro electric
- Imports|Hydrogen
- Imports|Land transport
- Imports|Land transport and Warehousing services
- Imports|Manufactured fuels
- Imports|Market Services
- Imports|Mechanical Engineering
- Imports|Mining
- Imports|Non Market Services
- Imports|Non-metallic minerals
- Imports|Nuclear
- Imports|Oil Fired
- Imports|Other Equipment Goods
- Imports|PV
- Imports|Paper products, publishing
- Imports|R&D
- Imports|Rubber and plastic products
- Imports|Steel
- Imports|Steel|High-Carbon
- Imports|Steel|Low-Carbon
- Imports|Steel|Med-Carbon
- Imports|Transport equipment
- Imports|Transport equipment (excluding EV)
- Imports|Warehousing
- Imports|Water transport
- Imports|Wind
- Import|Agriculture
- Import|Energy
- Import|Industry
- Import|Industry|Energy
- Import|Industry|Energy Intensive
- Import|Industry|Manufacturing
- Import|Other
- Interest Rate
- Interest Rate|Real
- Investment
- Investments|Agriculture
- Investments|Air transport
- Investments|Basic metals
- Investments|Basic pharmaceutical products
- Investments|Batteries
- Investments|Biofuels
- Investments|Biomass
- Investments|Biomass Solid
- Investments|CCS coal
- Investments|CSS Bio
- Investments|CSS Gas
- Investments|Chemical Products
- Investments|Clean Gas
- Investments|Coal fired
- Investments|Computer electronic and optical products
- Investments|Computer, electronic and optical products
- Investments|Construction
- Investments|Consumer Goods Industries
- Investments|EV Transport Equipment
- Investments|Electrical Equipment
- Investments|Electricity
- Investments|Fabricated Metal products
- Investments|Gas Fired
- Investments|Gas supply
- Investments|Geothermal
- Investments|Hydro electric
- Investments|Hydrogen
- Investments|Land transport
- Investments|Land transport and Warehousing services
- Investments|Manufactured fuels
- Investments|Market Services
- Investments|Mechanical Engineering
- Investments|Mining
- Investments|Mitigation Investments
- Investments|Non Market Services
- Investments|Non-metallic minerals
- Investments|Nuclear
- Investments|Oil Fired
- Investments|Other Equipment Goods
- Investments|PV
- Investments|Paper products publishing
- Investments|Paper products, publishing
- Investments|R&D
- Investments|Rubber and plastic products
- Investments|Transport equipment
- Investments|Transport equipment (excluding EV)
- Investments|Warehousing
- Investments|Water transport
- Investments|Wind
- Investment|Energy Supply
- Investment|Energy Supply|CO2 Transport and Storage
- Investment|Energy Supply|Electricity
- Investment|Energy Supply|Electricity|Battery|Rooftop
- Investment|Energy Supply|Electricity|Battery|Rootop
- Investment|Energy Supply|Electricity|Battery|Utility
- Investment|Energy Supply|Electricity|Biomass
- Investment|Energy Supply|Electricity|Biomass|w/ CCS
- Investment|Energy Supply|Electricity|Biomass|w/o CCS
- Investment|Energy Supply|Electricity|Coal
- Investment|Energy Supply|Electricity|Coal|w/ CCS
- Investment|Energy Supply|Electricity|Coal|w/o CCS
- Investment|Energy Supply|Electricity|Electricity Storage
- Investment|Energy Supply|Electricity|Fossil
- Investment|Energy Supply|Electricity|FuelCell
- Investment|Energy Supply|Electricity|Gas
- Investment|Energy Supply|Electricity|Gas|w/ CCS
- Investment|Energy Supply|Electricity|Gas|w/o CCS
- Investment|Energy Supply|Electricity|Geothermal
- Investment|Energy Supply|Electricity|Hydro
- Investment|Energy Supply|Electricity|Non-Biomass Renewables
- Investment|Energy Supply|Electricity|Non-fossil
- Investment|Energy Supply|Electricity|Nuclear
- Investment|Energy Supply|Electricity|Ocean
- Investment|Energy Supply|Electricity|Oil
- Investment|Energy Supply|Electricity|Oil|w/ CCS
- Investment|Energy Supply|Electricity|Oil|w/o CCS
- Investment|Energy Supply|Electricity|PumpedHydro
- Investment|Energy Supply|Electricity|Solar
- Investment|Energy Supply|Electricity|Solar|Commercial
- Investment|Energy Supply|Electricity|Solar|Rooftop
- Investment|Energy Supply|Electricity|Solar|Utility
- Investment|Energy Supply|Electricity|Transmission and Distribution
- Investment|Energy Supply|Electricity|Wind
- Investment|Energy Supply|Electricity|WindOffshore
- Investment|Energy Supply|Electricity|WindOnshore
- Investment|Energy Supply|Extraction|Coal
- Investment|Energy Supply|Extraction|Fossil
- Investment|Energy Supply|Extraction|Gas
- Investment|Energy Supply|Extraction|Oil
- Investment|Energy Supply|Hydrogen
- Investment|Energy Supply|Hydrogen|Electrolyser
- Investment|Energy Supply|Hydrogen|Fossil
- Investment|Energy Supply|Hydrogen|Other
- Investment|Energy Supply|Hydrogen|Renewable
- Investment|Energy Supply|Liquids
- Investment|Energy Supply|Liquids|Biomass
- Investment|Energy Supply|Liquids|Coal and Gas
- Investment|Energy Supply|Liquids|Oil
- Investment|Private
- Investment|Public
- Investment|RnD|Energy Supply
- Labour Force
- Land Cover
- Land Cover|Built-up Area
- Land Cover|Cropland
- Land Cover|Cropland|Energy Crops
- Land Cover|Forest
- Land Cover|Forest|Managed
- Land Cover|Forest|Natural Forest
- Land Cover|Other Land
- Land Cover|Pasture
- Lifetime|Electricity|Biomass|w/ CCS
- Lifetime|Electricity|Biomass|w/o CCS
- Lifetime|Electricity|Coal|w/ CCS
- Lifetime|Electricity|Coal|w/o CCS
- Lifetime|Electricity|Gas|w/ CCS
- Lifetime|Electricity|Gas|w/o CCS
- Lifetime|Electricity|Geothermal
- Lifetime|Electricity|Hydro
- Lifetime|Electricity|Nuclear
- Lifetime|Electricity|Solar|CSP
- Lifetime|Electricity|Solar|PV
- Lifetime|Electricity|Wind|Offshore
- Lifetime|Electricity|Wind|Onshore
- Mobility Electricity Demand
- Net Present Value|100% Net Billing|Solar|Rooftop
- Net Present Value|70% Net billing & 30% Net metering|Solar|Rooftop
- Net Present Value|Cumulative|Solar|PV|Residential and Commercial|Residential
- Nuclear Investment Share
- OM Cost|Fixed|Electricity|Biomass|w/ CCS
- OM Cost|Fixed|Electricity|Biomass|w/o CCS
- OM Cost|Fixed|Electricity|Coal|w/ CCS
- OM Cost|Fixed|Electricity|Coal|w/o CCS
- OM Cost|Fixed|Electricity|Gas|w/ CCS
- OM Cost|Fixed|Electricity|Gas|w/o CCS
- OM Cost|Fixed|Electricity|Geothermal
- OM Cost|Fixed|Electricity|Hydro
- OM Cost|Fixed|Electricity|Nuclear
- OM Cost|Fixed|Electricity|Solar|CSP
- OM Cost|Fixed|Electricity|Solar|PV
- OM Cost|Fixed|Electricity|Wind|Offshore
- OM Cost|Fixed|Electricity|Wind|Onshore
- Offshore wind Investment Share
- Onshore wind Investment Share
- PV Investment Share
- Policy Cost|Consumption Loss
- Policy Cost|Equivalent Variation
- Policy Cost|Expenditures|Government
- Policy Cost|Expenditures|Households
- Policy Cost|Exports
- Policy Cost|GDP Loss
- Policy Cost|GDP Loss|w/o transfers
- Policy Cost|Imports
- Policy Cost|Investment
- Policy Cost|Other
- Population
- Population|Female
- Population|Male
- Population|Tertiary education
- Population|Working Age
- Power Generation Technologies|Biomass
- Power Generation Technologies|CCS bio
- Power Generation Technologies|CCS coal
- Power Generation Technologies|CCS gas
- Power Generation Technologies|Coal fired
- Power Generation Technologies|Gas fired
- Power Generation Technologies|Geothermal
- Power Generation Technologies|Hydro Electric
- Power Generation Technologies|Nuclear
- Power Generation Technologies|Oil fired
- Power Generation Technologies|PV
- Power Generation Technologies|Wind
- Price|Agriculture|Corn|Index
- Price|Agriculture|Rice|Index
- Price|Agriculture|Soybean|Index
- Price|Agriculture|Wheat|Index
- Price|Carbon
- Price|Carbon|Demand|Industry
- Price|Carbon|Demand|Residential and Commercial
- Price|Carbon|Demand|Transportation
- Price|Carbon|EU ETS
- Price|Carbon|EUETS
- Price|Carbon|Non EU ETS
- Price|Carbon|Non EUETS
- Price|Carbon|Supply
- Price|Carbon|TOTCO2
- Price|Coal
- Price|Electricity|Steel
- Price|Final Energy|Commercial|Biomass
- Price|Final Energy|Commercial|Electricity
- Price|Final Energy|Commercial|Fossil
- Price|Final Energy|Commercial|Gases
- Price|Final Energy|Commercial|Heat
- Price|Final Energy|Commercial|Hydrogen
- Price|Final Energy|Commercial|Liquids
- Price|Final Energy|Commercial|Solids
- Price|Final Energy|Residential & Commercial|Biomass
- Price|Final Energy|Residential & Commercial|Electricity
- Price|Final Energy|Residential & Commercial|Fossil
- Price|Final Energy|Residential & Commercial|Gases
- Price|Final Energy|Residential & Commercial|Heat
- Price|Final Energy|Residential & Commercial|Hydrogen
- Price|Final Energy|Residential & Commercial|Liquids
- Price|Final Energy|Residential & Commercial|Solids
- Price|Final Energy|Residential and Commercial|Residential|Electricity
- Price|Final Energy|Residential and Commercial|Residential|Gases|Natural Gas
- Price|Final Energy|Residential and Commercial|Residential|Liquids
- Price|Final Energy|Residential and Commercial|Residential|Liquids|Biomass
- Price|Final Energy|Residential and Commercial|Residential|Liquids|Oil
- Price|Final Energy|Residential and Commercial|Residential|Solids|Biomass
- Price|Final Energy|Residential and Commercial|Residential|Solids|Coal
- Price|Final Energy|Residential|Biomass
- Price|Final Energy|Residential|Electricity
- Price|Final Energy|Residential|Fossil
- Price|Final Energy|Residential|Gases
- Price|Final Energy|Residential|Heat
- Price|Final Energy|Residential|Hydrogen
- Price|Final Energy|Residential|Liquids
- Price|Final Energy|Residential|Solids
- Price|Final Energy|Transportation|Electricity
- Price|Final Energy|Transportation|Gases
- Price|Final Energy|Transportation|Gases|Bioenergy
- Price|Final Energy|Transportation|Gases|Fossil
- Price|Final Energy|Transportation|Hydrogen
- Price|Final Energy|Transportation|Liquids
- Price|Final Energy|Transportation|Liquids|Bioenergy
- Price|Final Energy|Transportation|Liquids|Fossil
- Price|H2|Steel
- Price|Industry|Steel
- Price|Natural Gas
- Price|Primary Energy|Biomass
- Price|Primary Energy|Coal
- Price|Primary Energy|Gas
- Price|Primary Energy|Oil
- Price|Secondary Energy|Electricity
- Price|Secondary Energy|Gases|Natural Gas
- Price|Secondary Energy|Hydrogen
- Price|Secondary Energy|Liquids
- Price|Secondary Energy|Liquids|Biomass
- Price|Secondary Energy|Liquids|Oil
- Price|Secondary Energy|Solids|Biomass
- Price|Secondary Energy|Solids|Coal
- Price|Steel|Consumption
- Price|Steel|Imports|High-Carbon
- Price|Steel|Imports|Low-Carbon
- Price|Steel|Imports|Med-Carbon
- Price|Steel|Production
- Price|Steel|Subsidy
- Primary Energy
- Primary Energy|Biomass
- Primary Energy|Biomass|1st Generation
- Primary Energy|Biomass|Electricity
- Primary Energy|Biomass|Electricity|w/ CCS
- Primary Energy|Biomass|Electricity|w/o CCS
- Primary Energy|Biomass|Gases
- Primary Energy|Biomass|Hydrogen
- Primary Energy|Biomass|Liquids
- Primary Energy|Biomass|Modern
- Primary Energy|Biomass|Modern|w/ CCS
- Primary Energy|Biomass|Modern|w/o CCS
- Primary Energy|Biomass|Solids
- Primary Energy|Biomass|Traditional
- Primary Energy|Biomass|w/ CCS
- Primary Energy|Biomass|w/o CCS
- Primary Energy|Coal
- Primary Energy|Coal|Electricity
- Primary Energy|Coal|Electricity|w/ CCS
- Primary Energy|Coal|Electricity|w/o CCS
- Primary Energy|Coal|Hydrogen
- Primary Energy|Coal|w/ CCS
- Primary Energy|Coal|w/o CCS
- Primary Energy|Fossil
- Primary Energy|Fossil|w/ CCS
- Primary Energy|Fossil|w/o CCS
- Primary Energy|Gas
- Primary Energy|Gases
- Primary Energy|Gas|Electricity
- Primary Energy|Gas|Electricity|w/ CCS
- Primary Energy|Gas|Electricity|w/o CCS
- Primary Energy|Gas|Hydrogen
- Primary Energy|Gas|w/ CCS
- Primary Energy|Gas|w/o CCS
- Primary Energy|Geothermal
- Primary Energy|Hydro
- Primary Energy|Non-Biomass Renewables
- Primary Energy|Non-Biomass Renewables|Geothermal
- Primary Energy|Non-Biomass Renewables|Hydro
- Primary Energy|Non-Biomass Renewables|Ocean
- Primary Energy|Non-Biomass Renewables|Solar
- Primary Energy|Non-Biomass Renewables|Wind
- Primary Energy|Nuclear
- Primary Energy|Ocean
- Primary Energy|Oil
- Primary Energy|Oil|Electricity
- Primary Energy|Oil|Electricity|w/ CCS
- Primary Energy|Oil|Electricity|w/o CCS
- Primary Energy|Oil|Hydrogen
- Primary Energy|Oil|w/ CCS
- Primary Energy|Oil|w/o CCS
- Primary Energy|Other
- Primary Energy|Secondary Energy Trade
- Primary Energy|Solar
- Primary Energy|Wind
- Production|Agriculture
- Production|Air transport
- Production|Basic metals
- Production|Basic pharmaceutical products
- Production|Batteries
- Production|Biofuels
- Production|Biomass
- Production|Biomass Solid
- Production|CCS coal
- Production|CSS Bio
- Production|CSS Gas
- Production|Cement
- Production|Chemical Products
- Production|Clean Gas
- Production|Clinker
- Production|Coal fired
- Production|Computer, electronic and optical products
- Production|Construction
- Production|Consumer Goods Industries
- Production|EV Transport Equipment
- Production|Electrical Equipment
- Production|Electricity
- Production|Fabricated Metal products
- Production|Gas Fired
- Production|Gas supply
- Production|Geothermal
- Production|Hydro electric
- Production|Hydrogen
- Production|Land transport
- Production|Land transport and Warehousing services
- Production|Manufactured fuels
- Production|Market Services
- Production|Mechanical Engineering
- Production|Mining
- Production|Non Market Services
- Production|Non-ferrous metals
- Production|Non-metallic minerals
- Production|Nuclear
- Production|Oil Fired
- Production|Other Equipment Goods
- Production|PV
- Production|Paper products, publishing
- Production|R&D
- Production|Rubber and plastic products
- Production|Steel
- Production|Steel|AEL_EAF
- Production|Steel|BF/BOF
- Production|Steel|BF/BOF_CCS
- Production|Steel|BF_BOF
- Production|Steel|BF_BOF_BECCSmax
- Production|Steel|BF_BOF_BECCSmin
- Production|Steel|BF_BOF_CCS_73%
- Production|Steel|BF_BOF_CCS_86%
- Production|Steel|DRI/EAF_NG_CCS
- Production|Steel|DRI_EAF_BECCS
- Production|Steel|DRI_EAF_H2
- Production|Steel|DRI_EAF_NG
- Production|Steel|DRI_EAF_NG_CCS
- Production|Steel|DRI_EAF_coal
- Production|Steel|DRI_EAF_coal_CCS
- Production|Steel|DRI_H2
- Production|Steel|DRI_NG
- Production|Steel|DRI_NG_CCS
- Production|Steel|DRI_SAF_BOF_BECCS
- Production|Steel|DRI_SAF_BOF_H2
- Production|Steel|DRI_SAF_BOF_NG
- Production|Steel|DRI_bio
- Production|Steel|EAF_scrap_bio_NG_finish
- Production|Steel|EAF_scrap_bio_elec_finish
- Production|Steel|EAF_scrap_fossil_NG_finish
- Production|Steel|EAF_scrap_fossil_elec_finish
- Production|Steel|Electrolysis
- Production|Steel|MOE
- Production|Steel|SR_BOF
- Production|Steel|SR_BOF_CCS
- Production|Steel|Secondary steel
- Production|Transport equipment
- Production|Transport equipment (excluding EV)
- Production|Warehousing
- Production|Water transport
- Production|Wind
- Recycled material consumption|Steel scrap
- Renewable energy development
- Revenue|Government|Tax|Carbon
- Revenue|Government|Tax|Carbon|Demand|Buildings
- Revenue|Government|Tax|Carbon|Demand|Industry
- Revenue|Government|Tax|Carbon|Demand|Transport
- Revenue|Government|Tax|Carbon|Supply
- Savings|Private
- Savings|Public
- Savings|Total
- Secondary Energy
- Secondary Energy|Electricity
- Secondary Energy|Electricity|Biomass
- Secondary Energy|Electricity|Biomass|w/ CCS
- Secondary Energy|Electricity|Biomass|w/o CCS
- Secondary Energy|Electricity|Coal
- Secondary Energy|Electricity|Coal|w/ CCS
- Secondary Energy|Electricity|Coal|w/o CCS
- Secondary Energy|Electricity|Fossil
- Secondary Energy|Electricity|Fossil|Coal
- Secondary Energy|Electricity|Fossil|Gas
- Secondary Energy|Electricity|Fossil|Oil
- Secondary Energy|Electricity|Fossil|w/ CCS
- Secondary Energy|Electricity|Fossil|w/o CCS
- Secondary Energy|Electricity|FuelCell
- Secondary Energy|Electricity|Gas
- Secondary Energy|Electricity|Gas|w/ CCS
- Secondary Energy|Electricity|Gas|w/o CCS
- Secondary Energy|Electricity|Geothermal
- Secondary Energy|Electricity|Hydro
- Secondary Energy|Electricity|Imports
- Secondary Energy|Electricity|Non-Biomass Renewables
- Secondary Energy|Electricity|Nuclear
- Secondary Energy|Electricity|Ocean
- Secondary Energy|Electricity|Oil
- Secondary Energy|Electricity|Oil w/ CCS
- Secondary Energy|Electricity|Oil w/o CCS
- Secondary Energy|Electricity|Oil|w/ CCS
- Secondary Energy|Electricity|Oil|w/o CCS
- Secondary Energy|Electricity|Other
- Secondary Energy|Electricity|Solar
- Secondary Energy|Electricity|Solar|CSP
- Secondary Energy|Electricity|Solar|Commercial
- Secondary Energy|Electricity|Solar|PV
- Secondary Energy|Electricity|Solar|Rooftop
- Secondary Energy|Electricity|Solar|Utility
- Secondary Energy|Electricity|Transmission Losses
- Secondary Energy|Electricity|Wind
- Secondary Energy|Electricity|Wind|Offshore
- Secondary Energy|Electricity|Wind|Onshore
- Secondary Energy|Gases
- Secondary Energy|Gases|Biomass
- Secondary Energy|Gases|Coal
- Secondary Energy|Gases|Fossil
- Secondary Energy|Gases|Natural Gas
- Secondary Energy|Heat
- Secondary Energy|Heat|Geothermal
- Secondary Energy|Hydrogen
- Secondary Energy|Hydrogen|Biomass
- Secondary Energy|Hydrogen|Biomass|w/ CCS
- Secondary Energy|Hydrogen|Biomass|w/o CCS
- Secondary Energy|Hydrogen|Coal
- Secondary Energy|Hydrogen|Coal|w/ CCS
- Secondary Energy|Hydrogen|Coal|w/o CCS
- Secondary Energy|Hydrogen|Electricity
- Secondary Energy|Hydrogen|Electrolyser
- Secondary Energy|Hydrogen|Fossil
- Secondary Energy|Hydrogen|Fossil|w/ CCS
- Secondary Energy|Hydrogen|Fossil|w/o CCS
- Secondary Energy|Hydrogen|Gas
- Secondary Energy|Hydrogen|Gas|w/ CCS
- Secondary Energy|Hydrogen|Gas|w/o CCS
- Secondary Energy|Hydrogen|Nuclear
- Secondary Energy|Hydrogen|Oil
- Secondary Energy|Hydrogen|Solar
- Secondary Energy|Liquids
- Secondary Energy|Liquids|Biomass
- Secondary Energy|Liquids|Biomass|w/ CCS
- Secondary Energy|Liquids|Biomass|w/o CCS
- Secondary Energy|Liquids|Coal
- Secondary Energy|Liquids|Coal|w/ CCS
- Secondary Energy|Liquids|Coal|w/o CCS
- Secondary Energy|Liquids|Fossil
- Secondary Energy|Liquids|Fossil|w/ CCS
- Secondary Energy|Liquids|Fossil|w/o CCS
- Secondary Energy|Liquids|Gas
- Secondary Energy|Liquids|Gas|w/o CCS
- Secondary Energy|Liquids|Oil
- Secondary Energy|Liquids|Other
- Secondary Energy|Non-Electricity
- Secondary Energy|Other Carrier
- Secondary Energy|Solids
- Secondary Energy|Solids|Biomass
- Secondary Energy|Solids|Coal
- Secondary Energy|Solids|Fossil
- Specific Energy Consumption|Production|Steel
- Temperature Increase
- Temperature|Global Mean
- Total Energy System Cost|Commercial|Investments
- Total Energy System Cost|Commercial|Investments|Energy Equipment
- Total Energy System Cost|Commercial|Investments|New Constructions
- Total Energy System Cost|Commercial|Investments|Renovation of old constructions
- Total Energy System Cost|Residential & Commercial|Investments
- Total Energy System Cost|Residential & Commercial|Investments|Energy Equipment
- Total Energy System Cost|Residential & Commercial|Investments|New Constructions
- Total Energy System Cost|Residential & Commercial|Investments|Renovation of old constructions
- Total Energy System Cost|Residential|Investments
- Total Energy System Cost|Residential|Investments|Energy Equipment
- Total Energy System Cost|Residential|Investments|New Constructions
- Total Energy System Cost|Residential|Investments|Renovation of old constructions
- Trade
- Trade|Energy|Volume
- Trade|Primary Energy|Biomass|Volume
- Trade|Primary Energy|Coal|Volume
- Trade|Primary Energy|Gas|Volume
- Trade|Primary Energy|Oil|Volume
- Trade|Secondary Energy|Electricity|Volume
- Trade|Secondary Energy|Hydrogen|Volume
- Trade|Secondary Energy|Liquids|Biomass|Volume
- Trade|Secondary Energy|Liquids|Oil|Volume
- Unemployment
- Unemployment Rate
- Unemployment|Rate
- Value Added|Agriculture
- Value Added|Industry
- Value Added|Industry|Energy
- Value Added|Industry|Energy Intensive
- Value Added|Industry|Manufacturing
- Value Added|Residential and Commercial|Commercial
- Variable Operating Cost|Energy Supply
- Wages and salaries

Try queries like 'Capacity|Electricity|Solar|Utility for Greece' or 'show [variable name] for Greece'.

<a id="q078"></a>

## 078. custom analysis for decarbonisation options

Category: data and scenarios<br>
Route: `general_qa` (explicit site navigation request)<br>
Qwen generations: 0<br>
Time: 0.06 s<br>
Conversation: standalone

**Answer**

Use these IAM PARIS links for this request: [Configurable Analysis](https://iamparis.eu/analysis).

<a id="q079"></a>

## 079. carbon emissions in Europe

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.42 s<br>
Conversation: standalone

**Answer**

Showing results from the study **Where is the world headed?**. Also available in: Where is the EU headed?; Post-Glasgow targets. Name another study to switch.

### Emissions|CO2 in EU

Scope: scenario `multiple`, model `multiple`, years `2005-2100`
Unit: `Mt CO2/yr`

Answer:
In 2050, Emissions|CO2 in EU ranges from 1,024 (GCAM, PR_CurPol_CPO) to 3,429 Mt CO2/yr (GCAM, PR_Baseline) across 26 series; median 2,002. Median change since 2005: -71%.

Summary: 26 series across 4 models and 8 scenarios.
Showing 12 of 26 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 14 remaining series.

| Model | Scenario | 2005 | 2010 | 2015 | 2020 | 2025 | 2030 | 2035 | 2040 | 2045 | 2050 | 2055 | 2060 | 2065 | 2070 | 2075 | 2080 | 2085 | 2090 | 2095 | 2100 | Unit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E3ME | PR_Baseline | — | 4,304 | 3,999 | 3,444 | 3,188 | 3,033 | 2,864 | 2,692 | 2,516 | 2,323 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| E3ME | PR_CurPol_CP | — | 4,304 | 3,999 | 3,444 | 3,188 | 3,033 | 2,869 | 2,705 | 2,539 | 2,359 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| E3ME | PR_CurPol_EI | — | 4,304 | 3,999 | 3,445 | 3,188 | 3,032 | 2,919 | 2,827 | 2,737 | 2,633 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| GCAM | PR_Baseline | 4,190 | 3,790 | 3,490 | 3,543 | 3,524 | 3,454 | 3,524 | 3,549 | 3,509 | 3,429 | 3,476 | 3,589 | 3,712 | 3,829 | 3,926 | 4,029 | 4,168 | 4,311 | 4,472 | 4,672 | Mt CO2/yr |
| GCAM | PR_CurPol_CP | 4,190 | 3,790 | 3,490 | 3,115 | 2,687 | 2,257 | 2,112 | 1,836 | 1,485 | 1,118 | 962.69 | 912.20 | 870.84 | 863.75 | 839.93 | 771.80 | 656.18 | 503.01 | 281.84 | 10.13 | Mt CO2/yr |
| GCAM | PR_CurPol_CPO | 4,190 | 3,790 | 3,490 | 3,094 | 2,697 | 2,267 | 2,008 | 1,723 | 1,381 | 1,024 | 873.57 | 824.73 | 781.65 | 770.74 | 720.35 | 598.03 | 446.47 | 186.10 | -169.61 | -651.08 | Mt CO2/yr |
| GCAM | PR_CurPol_EI | 4,190 | 3,790 | 3,490 | 3,115 | 2,688 | 2,257 | 1,936 | 1,625 | 1,355 | 1,121 | 921.10 | 724.95 | 558.89 | 407.39 | 280.07 | 166.91 | 64.24 | -27.40 | -106.40 | -175.92 | Mt CO2/yr |
| GEMINI-E3 | PR_Baseline | — | — | 3,411 | 3,201 | 3,230 | 3,235 | 3,150 | 3,048 | 2,941 | 2,829 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| GEMINI-E3 | PR_CurPol_CP | — | — | 3,411 | 3,197 | 3,006 | 2,733 | 2,630 | 2,545 | 2,441 | 2,342 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| GEMINI-E3 | PR_CurPol_CPO | — | — | 3,411 | 3,197 | 3,006 | 2,731 | 2,626 | 2,541 | 2,437 | 2,337 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| MUSE | PR_Baseline | — | — | 3,141 | 3,106 | 3,076 | 2,994 | 2,842 | 2,701 | 2,384 | 2,074 | 1,838 | 1,664 | 1,588 | 1,543 | 1,400 | 1,279 | 1,101 | 1,034 | 977.08 | 947.41 | Mt CO2/yr |
| MUSE | PR_CurPol_CP | — | — | 3,118 | 3,018 | 2,956 | 2,740 | 2,581 | 2,435 | 2,155 | 1,871 | 1,641 | 1,482 | 1,414 | 1,365 | 1,237 | 1,116 | 975.01 | 910.90 | 844.23 | 817.09 | Mt CO2/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q080"></a>

## 080. CO2 emissions for EU in 2050

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.29 s<br>
Conversation: standalone

**Answer**

Showing results from the study **Where is the world headed?**. Also available in: Where is the EU headed?; Post-Glasgow targets. Name another study to switch.

### Emissions|CO2 in EU

Scope: scenario `multiple`, model `multiple`, years `2050`
Unit: `Mt CO2/yr`

Answer:
In 2050, Emissions|CO2 in EU ranges from 1,024 (GCAM, PR_CurPol_CPO) to 3,429 Mt CO2/yr (GCAM, PR_Baseline) across 26 series; median 2,002.

Summary: 26 series across 4 models and 8 scenarios.
Showing 12 of 26 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 14 remaining series.

| Model | Scenario | 2050 | Unit |
|---|---|---|---|
| E3ME | PR_Baseline | 2,323 | Mt CO2/yr |
| E3ME | PR_CurPol_CP | 2,359 | Mt CO2/yr |
| E3ME | PR_CurPol_EI | 2,633 | Mt CO2/yr |
| GCAM | PR_Baseline | 3,429 | Mt CO2/yr |
| GCAM | PR_CurPol_CP | 1,118 | Mt CO2/yr |
| GCAM | PR_CurPol_CPO | 1,024 | Mt CO2/yr |
| GCAM | PR_CurPol_EI | 1,121 | Mt CO2/yr |
| GEMINI-E3 | PR_Baseline | 2,829 | Mt CO2/yr |
| GEMINI-E3 | PR_CurPol_CP | 2,342 | Mt CO2/yr |
| GEMINI-E3 | PR_CurPol_CPO | 2,337 | Mt CO2/yr |
| MUSE | PR_Baseline | 2,074 | Mt CO2/yr |
| MUSE | PR_CurPol_CP | 1,871 | Mt CO2/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q081"></a>

## 081. show greenhouse gases for World

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.41 s<br>
Conversation: standalone

**Answer**

Showing results from the study **Where is the world headed?**. Also available in: Global impacts of NDCs; Post-Glasgow targets; EU path to net zero. Name another study to switch.

### Emissions|Kyoto Gases in World

Scope: scenario `multiple`, model `GCAM`, years `2005-2100`
Unit: `Mt CO2-equiv/yr`

Answer:
In 2100, Emissions|Kyoto Gases in World ranges from 25,009 (GCAM, PR_NDC_EI) to 86,184 Mt CO2-equiv/yr (GCAM, PR_Baseline) across 7 series; median 49,965. Median change since 2005: +13%.

Summary: 7 series across 1 model and 7 scenarios.
Series are ordered by model and scenario.

| Model | Scenario | 2005 | 2010 | 2015 | 2020 | 2025 | 2030 | 2035 | 2040 | 2045 | 2050 | 2055 | 2060 | 2065 | 2070 | 2075 | 2080 | 2085 | 2090 | 2095 | 2100 | Unit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GCAM | PR_Baseline | 44,360 | 49,723 | 53,484 | 55,605 | 58,586 | 61,179 | 63,583 | 65,646 | 67,044 | 67,972 | 69,616 | 71,539 | 73,747 | 76,093 | 77,927 | 79,295 | 81,092 | 82,764 | 84,372 | 86,184 | Mt CO2-equiv/yr |
| GCAM | PR_CurPol_CP | 44,359 | 49,722 | 53,483 | 51,244 | 53,010 | 54,478 | 55,491 | 56,592 | 56,775 | 56,543 | 57,398 | 58,601 | 60,361 | 61,860 | 63,001 | 63,861 | 64,812 | 65,485 | 66,075 | 66,585 | Mt CO2-equiv/yr |
| GCAM | PR_CurPol_CPO | 44,360 | 49,723 | 53,484 | 51,140 | 53,069 | 54,762 | 55,848 | 56,867 | 56,972 | 56,735 | 57,633 | 58,708 | 60,394 | 61,839 | 62,902 | 63,693 | 64,624 | 65,257 | 65,788 | 66,189 | Mt CO2-equiv/yr |
| GCAM | PR_CurPol_EI | 44,359 | 49,722 | 53,483 | 51,254 | 52,968 | 54,461 | 54,824 | 54,341 | 53,861 | 53,131 | 52,202 | 50,861 | 49,658 | 48,364 | 47,037 | 45,611 | 44,008 | 42,350 | 40,633 | 38,859 | Mt CO2-equiv/yr |
| GCAM | PR_NDC_CP | 44,359 | 49,722 | 53,483 | 51,239 | 50,413 | 49,750 | 50,231 | 50,602 | 50,180 | 49,029 | 49,084 | 49,508 | 50,430 | 50,977 | 51,181 | 51,017 | 50,812 | 50,311 | 49,709 | 48,998 | Mt CO2-equiv/yr |
| GCAM | PR_NDC_CPO | 44,360 | 49,723 | 53,484 | 51,137 | 50,668 | 50,084 | 51,022 | 51,307 | 50,821 | 49,551 | 49,472 | 49,842 | 50,661 | 51,220 | 51,456 | 51,411 | 51,351 | 51,076 | 50,591 | 49,965 | Mt CO2-equiv/yr |
| GCAM | PR_NDC_EI | 44,359 | 49,722 | 53,483 | 51,251 | 50,410 | 49,745 | 48,534 | 46,692 | 44,989 | 43,010 | 40,977 | 38,887 | 37,020 | 35,178 | 33,485 | 31,767 | 29,984 | 28,238 | 26,563 | 25,009 | Mt CO2-equiv/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

Reply with a scenario, model, region, or year to narrow the answer.

<a id="q082"></a>

## 082. methane emissions for Europe

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.40 s<br>
Conversation: standalone

**Answer**

Showing results from the study **Where is the world headed?**. Also available in: Where is the EU headed?. Name another study to switch.

### Emissions|CH4 in EU

Scope: scenario `multiple`, model `multiple`, years `2005-2100`
Unit: `Mt CH4/yr`

Answer:
In 2050, Emissions|CH4 in EU ranges from 10.56 (GEMINI-E3, PR_NDC_EI) to 20.97 Mt CH4/yr (GCAM, PR_Baseline) across 13 series; median 16.52. Median change since 2005: -25%.

Summary: 13 series across 2 models and 7 scenarios.
Showing 12 of 13 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 1 remaining series.

| Model | Scenario | 2005 | 2010 | 2015 | 2020 | 2025 | 2030 | 2035 | 2040 | 2045 | 2050 | 2055 | 2060 | 2065 | 2070 | 2075 | 2080 | 2085 | 2090 | 2095 | 2100 | Unit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GCAM | PR_Baseline | 22.15 | 20.08 | 18.87 | 21.65 | 22.16 | 21.66 | 21.32 | 21.11 | 21.24 | 20.97 | 20.80 | 20.81 | 20.81 | 20.74 | 20.73 | 20.89 | 21.22 | 21.33 | 21.49 | 21.66 | Mt CH4/yr |
| GCAM | PR_CurPol_CP | 22.15 | 20.08 | 18.87 | 18.76 | 18.67 | 17.74 | 17.16 | 16.77 | 16.81 | 16.52 | 16.27 | 16.18 | 16.11 | 15.98 | 15.84 | 15.91 | 16.03 | 16.11 | 16.11 | 16.12 | Mt CH4/yr |
| GCAM | PR_CurPol_CPO | 22.15 | 20.08 | 18.87 | 18.65 | 18.86 | 18.18 | 17.61 | 17.19 | 17.19 | 16.86 | 16.57 | 16.48 | 16.39 | 16.23 | 16.06 | 16.11 | 16.19 | 16.24 | 16.20 | 16.18 | Mt CH4/yr |
| GCAM | PR_CurPol_EI | 22.15 | 20.08 | 18.87 | 18.75 | 18.68 | 17.74 | 17.20 | 16.86 | 16.93 | 16.63 | 16.29 | 16.13 | 15.98 | 15.80 | 15.55 | 15.52 | 15.52 | 15.57 | 15.54 | 15.54 | Mt CH4/yr |
| GCAM | PR_NDC_CP | 22.15 | 20.08 | 18.87 | 18.76 | 19.36 | 19.15 | 16.21 | 15.43 | 15.42 | 16.13 | 16.35 | 16.40 | 15.99 | 15.50 | 15.04 | 14.81 | 14.66 | 14.58 | 14.26 | 14.15 | Mt CH4/yr |
| GCAM | PR_NDC_CPO | 22.15 | 20.08 | 18.87 | 18.65 | 18.77 | 18.05 | 17.50 | 17.08 | 17.06 | 16.72 | 16.42 | 16.32 | 16.22 | 16.10 | 15.90 | 15.90 | 15.95 | 16.07 | 16.03 | 16.01 | Mt CH4/yr |
| GEMINI-E3 | PR_Baseline | — | — | 18.82 | 18.46 | 18.29 | 18.22 | 18.33 | 18.43 | 18.55 | 18.68 | — | — | — | — | — | — | — | — | — | — | Mt CH4/yr |
| GEMINI-E3 | PR_CurPol_CP | — | — | 18.82 | 18.53 | 18.52 | 15.55 | 15.83 | 15.97 | 16.07 | 16.16 | — | — | — | — | — | — | — | — | — | — | Mt CH4/yr |
| GEMINI-E3 | PR_CurPol_CPO | — | — | 18.82 | 18.53 | 18.52 | 15.59 | 15.75 | 15.91 | 16.05 | 16.14 | — | — | — | — | — | — | — | — | — | — | Mt CH4/yr |
| GEMINI-E3 | PR_CurPol_EI | — | — | 18.82 | 18.53 | 18.52 | 15.55 | 15.83 | 15.97 | 16.06 | 16.15 | — | — | — | — | — | — | — | — | — | — | Mt CH4/yr |
| GEMINI-E3 | PR_NDC_CP | — | — | 18.82 | 18.53 | 18.51 | 15.69 | 15.56 | 15.78 | 15.92 | 16.02 | — | — | — | — | — | — | — | — | — | — | Mt CH4/yr |
| GEMINI-E3 | PR_NDC_EI | — | — | 18.82 | 18.53 | 18.51 | 15.69 | 14.87 | 13.58 | 12.27 | 10.56 | — | — | — | — | — | — | — | — | — | — | Mt CH4/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q083"></a>

## 083. how much CO2 will the world emit in 2050?

Category: data and scenarios<br>
Route: `data_query` (extracted data entities)<br>
Qwen generations: 0<br>
Time: 0.46 s<br>
Conversation: standalone

**Answer**

Showing results from the study **Where is the world headed?**. Also available in: Post-Glasgow targets; Global impacts of NDCs; Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU; EU path to net zero; Tech-constrained EU pathways to net zero. Name another study to switch.

### Emissions|CO2 in World

Scope: scenario `multiple`, model `multiple`, years `2050`
Unit: `Mt CO2/yr`

Answer:
In 2050, Emissions|CO2 in World ranges from 9,173 (GCAM, NDC_LTT) to 52,211 Mt CO2/yr (GEMINI-E3, PR_Baseline) across 38 series; median 33,062.

Summary: 38 series across 4 models and 13 scenarios.
Showing 12 of 38 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 26 remaining series.

| Model | Scenario | 2050 | Unit |
|---|---|---|---|
| E3ME | PR_Baseline | 40,393 | Mt CO2/yr |
| E3ME | PR_CurPol_CP | 40,054 | Mt CO2/yr |
| E3ME | PR_CurPol_EI | 39,239 | Mt CO2/yr |
| GCAM | Baseline | 45,595 | Mt CO2/yr |
| GCAM | CP_EI | 28,519 | Mt CO2/yr |
| GCAM | NDC_LTT | 9,173 | Mt CO2/yr |
| GEMINI-E3 | Baseline | 46,713 | Mt CO2/yr |
| GEMINI-E3 | CP_EI | 27,388 | Mt CO2/yr |
| GEMINI-E3 | PR_Baseline | 52,211 | Mt CO2/yr |
| MUSE | baseline | 31,448 | Mt CO2/yr |
| MUSE | CP_EI | 23,856 | Mt CO2/yr |
| MUSE | NDC_EI | 19,999 | Mt CO2/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q084"></a>

## 084. global emissions 2050

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.34 s<br>
Conversation: standalone

**Answer**

Showing results from the study **Where is the world headed?**. Also available in: Post-Glasgow targets; Global impacts of NDCs; Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU; EU path to net zero; Tech-constrained EU pathways to net zero. Name another study to switch.

Interpretation: I treated bare **emissions** as `Emissions|CO2` (carbon-dioxide emissions).

### Emissions|CO2 in World

Scope: scenario `multiple`, model `multiple`, years `2050`
Unit: `Mt CO2/yr`

Answer:
In 2050, Emissions|CO2 in World ranges from 9,173 (GCAM, NDC_LTT) to 52,211 Mt CO2/yr (GEMINI-E3, PR_Baseline) across 38 series; median 33,062.

Summary: 38 series across 4 models and 13 scenarios.
Showing 12 of 38 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 26 remaining series.

| Model | Scenario | 2050 | Unit |
|---|---|---|---|
| E3ME | PR_Baseline | 40,393 | Mt CO2/yr |
| E3ME | PR_CurPol_CP | 40,054 | Mt CO2/yr |
| E3ME | PR_CurPol_EI | 39,239 | Mt CO2/yr |
| GCAM | Baseline | 45,595 | Mt CO2/yr |
| GCAM | CP_EI | 28,519 | Mt CO2/yr |
| GCAM | NDC_LTT | 9,173 | Mt CO2/yr |
| GEMINI-E3 | Baseline | 46,713 | Mt CO2/yr |
| GEMINI-E3 | CP_EI | 27,388 | Mt CO2/yr |
| GEMINI-E3 | PR_Baseline | 52,211 | Mt CO2/yr |
| MUSE | baseline | 31,448 | Mt CO2/yr |
| MUSE | CP_EI | 23,856 | Mt CO2/yr |
| MUSE | NDC_EI | 19,999 | Mt CO2/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q085"></a>

## 085. co2 emisions world 2040

Category: data and scenarios<br>
Route: `data_query` (extracted data entities)<br>
Qwen generations: 0<br>
Time: 0.40 s<br>
Conversation: standalone

**Answer**

Showing results from the study **Where is the world headed?**. Also available in: Post-Glasgow targets; Global impacts of NDCs; Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU; EU path to net zero; Tech-constrained EU pathways to net zero. Name another study to switch.

### Emissions|CO2 in World

Scope: scenario `multiple`, model `multiple`, years `2040`
Unit: `Mt CO2/yr`

Answer:
In 2040, Emissions|CO2 in World ranges from 20,969 (GCAM, NDC_LTT) to 48,984 Mt CO2/yr (GEMINI-E3, PR_Baseline) across 38 series; median 35,646.

Summary: 38 series across 4 models and 13 scenarios.
Showing 12 of 38 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 26 remaining series.

| Model | Scenario | 2040 | Unit |
|---|---|---|---|
| E3ME | PR_Baseline | 38,757 | Mt CO2/yr |
| E3ME | PR_CurPol_CP | 38,479 | Mt CO2/yr |
| E3ME | PR_CurPol_EI | 38,706 | Mt CO2/yr |
| GCAM | Baseline | 45,315 | Mt CO2/yr |
| GCAM | CP_EI | 33,501 | Mt CO2/yr |
| GCAM | NDC_LTT | 20,969 | Mt CO2/yr |
| GEMINI-E3 | Baseline | 45,532 | Mt CO2/yr |
| GEMINI-E3 | CP_EI | 31,313 | Mt CO2/yr |
| GEMINI-E3 | PR_Baseline | 48,984 | Mt CO2/yr |
| MUSE | baseline | 33,696 | Mt CO2/yr |
| MUSE | CP_EI | 27,906 | Mt CO2/yr |
| MUSE | NDC_EI | 24,700 | Mt CO2/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q086"></a>

## 086. What's the projected solar capacity in Greece by 2040?

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.29 s<br>
Conversation: standalone

**Answer**

### Capacity|Electricity|Solar in GREECE

Scope: scenario `multiple`, model `multiple`, years `first available-2040`
Unit: `GW`

Answer:
In 2040, Capacity|Electricity|Solar in GREECE ranges from 3.72 (NEMESIS 5.1, WWH21) to 15.27 GW (E4SMA-EU-TIMES 1.0, WWH21) across 10 series; median 11.39. Median change since 2010: +6616%.

Summary: 10 series across 2 models and 5 scenarios.
Series are ordered by model and scenario.

| Model | Scenario | 2010 | 2015 | 2020 | 2025 | 2030 | 2035 | 2040 | Unit |
|---|---|---|---|---|---|---|---|---|---|
| E4SMA-EU-TIMES 1.0 | NZE_Bench_H | 0.202 | — | 2.60 | 7.22 | 12.77 | — | 12.77 | GW |
| E4SMA-EU-TIMES 1.0 | NZE_Bench_L | 0.202 | — | 2.60 | 13.63 | 13.63 | — | 13.63 | GW |
| E4SMA-EU-TIMES 1.0 | NZE_Bench_M | 0.202 | — | 2.60 | 13.58 | 13.58 | — | 13.58 | GW |
| E4SMA-EU-TIMES 1.0 | NZE_EUPol_Stand | 0.202 | — | 2.60 | 2.60 | 12.97 | — | 12.97 | GW |
| E4SMA-EU-TIMES 1.0 | WWH21 | 0.202 | — | 2.60 | 2.60 | 15.27 | — | 15.27 | GW |
| NEMESIS 5.1 | NZE_Bench_H | — | 2.23 | 2.11 | 3.63 | 5.35 | 6.41 | 7.63 | GW |
| NEMESIS 5.1 | NZE_Bench_L | — | 2.23 | 2.11 | 4.19 | 5.97 | 7.60 | 10.01 | GW |
| NEMESIS 5.1 | NZE_Bench_M | — | 2.23 | 2.11 | 3.92 | 5.62 | 6.94 | 8.47 | GW |
| NEMESIS 5.1 | NZE_EUPol_Stand | — | 2.23 | 2.09 | 3.68 | 5.75 | 6.86 | 8.73 | GW |
| NEMESIS 5.1 | WWH21 | — | 2.23 | 2.24 | 3.00 | 3.66 | 3.70 | 3.72 | GW |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/eu-path-to-net-zero/graphs).

<a id="q087"></a>

## 087. Will Germany's electricity demand grow?

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.52 s<br>
Conversation: standalone

**Answer**

Showing results from the study **EU path to net zero**. Also available in: Where is the world headed?. Name another study to switch.

### Final Energy|Electricity in DEU

Scope: scenario `multiple`, model `multiple`, years `2010-2050`
Unit: `EJ/yr`

Answer:
**From 2010 to 2050, Final Energy|Electricity in DEU rises in 13 and falls in 0 of 13 series** (median change +61%).

Summary: 13 series across 3 models and 5 scenarios.
Showing 12 of 13 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 1 remaining series.

| Model | Scenario | 2010 | 2015 | 2020 | 2025 | 2030 | 2035 | 2040 | 2045 | 2050 | Unit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E4SMA-EU-TIMES 1.0 | NZE_Bench_H | 1.96 | — | 1.90 | 1.86 | 1.82 | — | 1.96 | — | 2.78 | EJ/yr |
| E4SMA-EU-TIMES 1.0 | NZE_Bench_L | 1.96 | — | 1.90 | 1.86 | 1.82 | — | 2.16 | — | 3.17 | EJ/yr |
| E4SMA-EU-TIMES 1.0 | NZE_Bench_M | 1.96 | — | 1.90 | 1.86 | 1.82 | — | 2.03 | — | 2.77 | EJ/yr |
| E4SMA-EU-TIMES 1.0 | NZE_EUPol_Stand | 1.96 | — | 1.90 | 1.89 | 1.87 | — | 2.21 | — | 2.84 | EJ/yr |
| E4SMA-EU-TIMES 1.0 | WWH21 | 1.96 | — | 1.92 | 2.04 | 2.12 | — | 2.27 | — | 2.78 | EJ/yr |
| ICES-XPS 1.0 | NZE_Bench_H | — | 2.46 | 2.72 | 3.11 | 3.48 | 3.97 | 4.46 | 5.11 | 5.91 | EJ/yr |
| ICES-XPS 1.0 | NZE_EUPol_Stand | — | 2.46 | 2.73 | 3.12 | 3.53 | 4.03 | 4.52 | 5.15 | 5.92 | EJ/yr |
| ICES-XPS 1.0 | WWH21 | — | 2.46 | 2.75 | 3.18 | 3.51 | 3.95 | 4.36 | 4.88 | 5.48 | EJ/yr |
| NEMESIS 5.1 | NZE_Bench_H | 1.92 | 1.85 | 1.92 | 1.91 | 1.90 | 1.95 | 2.14 | 2.49 | 2.99 | EJ/yr |
| NEMESIS 5.1 | NZE_Bench_L | 1.92 | 1.85 | 1.92 | 1.90 | 1.91 | 2.05 | 2.42 | 3.06 | 4.18 | EJ/yr |
| NEMESIS 5.1 | NZE_Bench_M | 1.92 | 1.85 | 1.92 | 1.90 | 1.90 | 1.99 | 2.24 | 2.70 | 3.37 | EJ/yr |
| NEMESIS 5.1 | WWH21 | 1.92 | 1.85 | 1.90 | 1.96 | 2.07 | 2.05 | 2.09 | 2.14 | 2.18 | EJ/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/eu-path-to-net-zero/graphs).

<a id="q088"></a>

## 088. show me renewables share EU

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.38 s<br>
Conversation: standalone

**Answer**

Showing results from the study **Where is the world headed?**. Also available in: Where is the EU headed?. Name another study to switch.

### Primary Energy|Non-Biomass Renewables in EU

Scope: scenario `multiple`, model `multiple`, years `2005-2100`
Unit: `EJ/yr`

Answer:
In 2040, Primary Energy|Non-Biomass Renewables in EU ranges from 4.62 (TIAM_Grantham, PR_Baseline) to 12.55 EJ/yr (TIAM_Grantham, PR_NDC_CPO) across 22 series; median 7.48. Median change since 2005: +407%.

Summary: 22 series across 4 models and 8 scenarios.
Showing 12 of 22 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 10 remaining series.

| Model | Scenario | 2005 | 2010 | 2015 | 2020 | 2025 | 2030 | 2035 | 2040 | 2045 | 2050 | 2060 | 2070 | 2080 | 2090 | 2100 | Unit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | PR_Baseline | 1.72 | 2.34 | 3.23 | 3.50 | 3.77 | 4.15 | 4.61 | 5.09 | 5.62 | — | — | — | — | — | — | EJ/yr |
| 42 | PR_CurPol_EI | 1.72 | 2.34 | 3.23 | 3.62 | 4.37 | 5.41 | 6.42 | 7.43 | 8.50 | — | — | — | — | — | — | EJ/yr |
| 42 | PR_NDC_EI | 1.72 | 2.34 | 3.23 | 3.62 | 4.51 | 5.71 | 6.94 | 8.48 | 10.10 | — | — | — | — | — | — | EJ/yr |
| E3ME | PR_Baseline | — | 2.01 | 2.81 | 4.02 | 4.97 | 5.85 | 6.85 | 8.05 | 9.18 | 10.44 | — | — | — | — | — | EJ/yr |
| E3ME | PR_CurPol_CP | — | 2.01 | 2.81 | 4.02 | 4.97 | 5.85 | 6.82 | 7.97 | 9.04 | 10.25 | — | — | — | — | — | EJ/yr |
| E3ME | PR_CurPol_EI | — | 2.01 | 2.81 | 4.02 | 4.97 | 5.85 | 6.67 | 7.53 | 8.22 | 9.08 | — | — | — | — | — | EJ/yr |
| GEMINI-E3 | PR_Baseline | — | — | 2.72 | 3.24 | 3.50 | 3.89 | 4.43 | 5.05 | 5.61 | 6.38 | — | — | — | — | — | EJ/yr |
| GEMINI-E3 | PR_CurPol_CP | — | — | 2.72 | 3.24 | 3.76 | 4.35 | 5.09 | 5.82 | 6.51 | 7.44 | — | — | — | — | — | EJ/yr |
| GEMINI-E3 | PR_CurPol_CPO | — | — | 2.72 | 3.24 | 3.76 | 4.35 | 5.09 | 5.81 | 6.51 | 7.44 | — | — | — | — | — | EJ/yr |
| TIAM_Grantham | PR_Baseline | 2.18 | 2.71 | — | 3.10 | — | 2.77 | — | 4.62 | — | 7.20 | 7.51 | 7.66 | 8.45 | 9.34 | 9.99 | EJ/yr |
| TIAM_Grantham | PR_CurPol_CP | 2.18 | 2.71 | — | 5.22 | — | 7.59 | — | 10.40 | — | 17.74 | 23.40 | 26.77 | 27.82 | 34.52 | 48.60 | EJ/yr |
| TIAM_Grantham | PR_NDC_CPO | 2.18 | 2.71 | — | 5.22 | — | 9.77 | — | 12.55 | — | 20.87 | 26.01 | 28.68 | 35.15 | 50.23 | 65.58 | EJ/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q089"></a>

## 089. how does GCAM compare to REMIND on emissions in 2050

Category: conceptual explanations<br>
Route: `data_query` (chat plotting disabled; return numeric table)<br>
Qwen generations: 0<br>
Time: 0.40 s<br>
Conversation: standalone

**Answer**

Showing results from the study **Where is the world headed?**. Also available in: EU path to net zero. Name another study to switch.

Interpretation: I treated bare **emissions** as `Emissions|CO2` (carbon-dioxide emissions).

### Emissions|CO2 in World

Scope: scenario `multiple`, model `GCAM`, years `2050`
Unit: `Mt CO2/yr`

Answer:
In 2050, Emissions|CO2 in World ranges from 9,173 (GCAM, NDC_LTT) to 46,979 Mt CO2/yr (GCAM, PR_Baseline) across 11 series; median 33,462.

Summary: 11 series across 1 model and 11 scenarios.
Series are ordered by model and scenario.

| Model | Scenario | 2050 | Unit |
|---|---|---|---|
| GCAM | Baseline | 45,595 | Mt CO2/yr |
| GCAM | CP_EI | 28,519 | Mt CO2/yr |
| GCAM | NDC_EI | 18,332 | Mt CO2/yr |
| GCAM | NDC_LTT | 9,173 | Mt CO2/yr |
| GCAM | PR_Baseline | 46,979 | Mt CO2/yr |
| GCAM | PR_CurPol_CP | 38,385 | Mt CO2/yr |
| GCAM | PR_CurPol_CPO | 38,523 | Mt CO2/yr |
| GCAM | PR_CurPol_EI | 36,082 | Mt CO2/yr |
| GCAM | PR_NDC_CP | 33,106 | Mt CO2/yr |
| GCAM | PR_NDC_CPO | 33,462 | Mt CO2/yr |
| GCAM | PR_NDC_EI | 28,158 | Mt CO2/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q090"></a>

## 090. which scenario has the lowest emissions in 2050 for the world?

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.32 s<br>
Conversation: standalone

**Answer**

Showing results from the study **Where is the world headed?**. Also available in: Post-Glasgow targets; Global impacts of NDCs; Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU; EU path to net zero; Tech-constrained EU pathways to net zero. Name another study to switch.

Interpretation: I treated bare **emissions** as `Emissions|CO2` (carbon-dioxide emissions).

### Emissions|CO2 in World

Scope: scenario `multiple`, model `multiple`, years `2050`
Unit: `Mt CO2/yr`

Answer:
**Lowest in 2050:** 9,173 Mt CO2/yr (GCAM, NDC_LTT), out of 38 series.

Summary: 38 series across 4 models and 13 scenarios.
Showing 12 of 38 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 26 remaining series.

| Model | Scenario | 2050 | Unit |
|---|---|---|---|
| E3ME | PR_Baseline | 40,393 | Mt CO2/yr |
| E3ME | PR_CurPol_CP | 40,054 | Mt CO2/yr |
| E3ME | PR_CurPol_EI | 39,239 | Mt CO2/yr |
| GCAM | Baseline | 45,595 | Mt CO2/yr |
| GCAM | CP_EI | 28,519 | Mt CO2/yr |
| GCAM | NDC_LTT | 9,173 | Mt CO2/yr |
| GEMINI-E3 | Baseline | 46,713 | Mt CO2/yr |
| GEMINI-E3 | CP_EI | 27,388 | Mt CO2/yr |
| GEMINI-E3 | PR_Baseline | 52,211 | Mt CO2/yr |
| MUSE | baseline | 31,448 | Mt CO2/yr |
| MUSE | CP_EI | 23,856 | Mt CO2/yr |
| MUSE | NDC_EI | 19,999 | Mt CO2/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q091"></a>

## 091. what is the highest carbon price in 2050?

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.41 s<br>
Conversation: standalone

**Answer**

Showing results from the study **Where is the world headed?**. Also available in: Decipher modelling studies; EU path to net zero; Post-Glasgow targets; Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU; Global impacts of NDCs; Tech-constrained EU pathways to net zero; Behavioural changes for climate policy; Where is the EU headed?; Enhancing renewable energy system metrics. Name another study to switch.

### Price|Carbon in World

Scope: scenario `multiple`, model `multiple`, years `2050`
Unit: `US$2010/t CO2`

Answer:
**Highest in 2050:** 127.64 US$2010/t CO2 (ICES-XPS, PR_NDC_EI), out of 38 series.

Summary: 38 series across 6 models and 8 scenarios.
Showing 12 of 38 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 26 remaining series.

| Model | Scenario | 2050 | Unit |
|---|---|---|---|
| E3ME | PR_Baseline | 2.95 | US$2010/t CO2 |
| E3ME | PR_CurPol_CP | 4.04 | US$2010/t CO2 |
| GCAM | PR_CurPol_CP | 12.43 | US$2010/t CO2 |
| GCAM | PR_CurPol_CPO | 13.17 | US$2010/t CO2 |
| GEMINI-E3 | PR_Baseline | 0.282 | US$2010/t CO2 |
| GEMINI-E3 | PR_CurPol_CP | 5.62 | US$2010/t CO2 |
| ICES-XPS | PR_Baseline | 0.00 | US$2010/t CO2 |
| ICES-XPS | PR_CurPol_CP | 3.04 | US$2010/t CO2 |
| ICES-XPS | PR_NDC_EI | 127.64 | US$2010/t CO2 |
| MUSE | PR_Baseline | 41.61 | US$2010/t CO2 |
| MUSE | PR_CurPol_CP | 41.96 | US$2010/t CO2 |
| TIAM_Grantham | PR_CurPol_CP | 22.22 | US$2010/t CO2 |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q092"></a>

## 092. by how much do emissions fall between 2020 and 2050 in the EU?

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.33 s<br>
Conversation: standalone

**Answer**

Showing results from the study **Where is the world headed?**. Also available in: Where is the EU headed?; Post-Glasgow targets. Name another study to switch.

Interpretation: I treated bare **emissions** as `Emissions|CO2` (carbon-dioxide emissions).

### Emissions|CO2 in EU

Scope: scenario `multiple`, model `multiple`, years `2020-2050`
Unit: `Mt CO2/yr`

Answer:
**Change from 2020 to 2050:** median -36% across 26 series (range -67% to -3%).

Summary: 26 series across 4 models and 8 scenarios.
Showing 12 of 26 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 14 remaining series.

| Model | Scenario | 2020 | 2025 | 2030 | 2035 | 2040 | 2045 | 2050 | Unit |
|---|---|---|---|---|---|---|---|---|---|
| E3ME | PR_Baseline | 3,444 | 3,188 | 3,033 | 2,864 | 2,692 | 2,516 | 2,323 | Mt CO2/yr |
| E3ME | PR_CurPol_CP | 3,444 | 3,188 | 3,033 | 2,869 | 2,705 | 2,539 | 2,359 | Mt CO2/yr |
| E3ME | PR_CurPol_EI | 3,445 | 3,188 | 3,032 | 2,919 | 2,827 | 2,737 | 2,633 | Mt CO2/yr |
| GCAM | PR_Baseline | 3,543 | 3,524 | 3,454 | 3,524 | 3,549 | 3,509 | 3,429 | Mt CO2/yr |
| GCAM | PR_CurPol_CP | 3,115 | 2,687 | 2,257 | 2,112 | 1,836 | 1,485 | 1,118 | Mt CO2/yr |
| GCAM | PR_CurPol_CPO | 3,094 | 2,697 | 2,267 | 2,008 | 1,723 | 1,381 | 1,024 | Mt CO2/yr |
| GCAM | PR_CurPol_EI | 3,115 | 2,688 | 2,257 | 1,936 | 1,625 | 1,355 | 1,121 | Mt CO2/yr |
| GEMINI-E3 | PR_Baseline | 3,201 | 3,230 | 3,235 | 3,150 | 3,048 | 2,941 | 2,829 | Mt CO2/yr |
| GEMINI-E3 | PR_CurPol_CP | 3,197 | 3,006 | 2,733 | 2,630 | 2,545 | 2,441 | 2,342 | Mt CO2/yr |
| GEMINI-E3 | PR_CurPol_CPO | 3,197 | 3,006 | 2,731 | 2,626 | 2,541 | 2,437 | 2,337 | Mt CO2/yr |
| MUSE | PR_Baseline | 3,106 | 3,076 | 2,994 | 2,842 | 2,701 | 2,384 | 2,074 | Mt CO2/yr |
| MUSE | PR_CurPol_CP | 3,018 | 2,956 | 2,740 | 2,581 | 2,435 | 2,155 | 1,871 | Mt CO2/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q093"></a>

## 093. is coal phased out in India by 2050?

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.37 s<br>
Conversation: standalone

**Answer**

Showing results from the study **Where is the world headed?**. Also available in: Post-Glasgow targets; Global impacts of NDCs; Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU; Tech-constrained EU pathways to net zero. Name another study to switch.

### Primary Energy|Coal in IND

Scope: scenario `multiple`, model `multiple`, years `first available-2050`
Unit: `EJ/yr`

Answer:
**No series phases Primary Energy|Coal in IND out by 2050** (below 5% of its starting level). The lowest is 102% of its 2015 level (ICES-XPS, PR_NDC_EI).

Summary: 43 series across 7 models and 8 scenarios.
Showing 12 of 43 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 31 remaining series.

| Model | Scenario | 2005 | 2010 | 2015 | 2020 | 2025 | 2030 | 2035 | 2040 | 2045 | 2050 | Unit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | PR_Baseline | 7.71 | 11.68 | 15.86 | 17.66 | 21.95 | 26.20 | 30.25 | 34.06 | 37.45 | — | EJ/yr |
| 42 | PR_NDC_EI | 7.71 | 11.68 | 15.86 | 17.38 | 18.42 | 18.12 | 18.09 | 17.27 | 16.38 | — | EJ/yr |
| E3ME | PR_Baseline | — | 11.22 | 14.91 | 15.52 | 19.56 | 24.55 | 30.22 | 36.17 | 43.42 | 52.46 | EJ/yr |
| E3ME | PR_CurPol_CP | — | 11.22 | 14.91 | 15.52 | 19.56 | 24.55 | 30.06 | 35.88 | 42.96 | 51.73 | EJ/yr |
| GCAM | PR_Baseline | 7.63 | 11.53 | 15.39 | 21.71 | 27.07 | 31.61 | 36.43 | 41.74 | 46.22 | 48.80 | EJ/yr |
| GCAM | PR_CurPol_CP | 7.63 | 11.53 | 15.39 | 18.41 | 25.15 | 29.10 | 32.96 | 37.53 | 41.22 | 43.93 | EJ/yr |
| GEMINI-E3 | PR_Baseline | — | — | 16.75 | 21.44 | 27.38 | 34.17 | 40.00 | 45.74 | 51.77 | 57.08 | EJ/yr |
| GEMINI-E3 | PR_CurPol_CP | — | — | 16.75 | 21.49 | 26.70 | 33.62 | 38.98 | 44.30 | 49.81 | 54.53 | EJ/yr |
| ICES-XPS | PR_Baseline | — | — | 17.58 | 20.57 | 22.59 | 23.60 | 25.86 | 26.63 | 26.63 | 26.30 | EJ/yr |
| ICES-XPS | PR_CurPol_CP | — | — | 17.58 | 20.60 | 22.76 | 24.01 | 26.32 | 27.09 | 27.05 | 26.69 | EJ/yr |
| MUSE | PR_Baseline | — | — | 16.52 | 19.21 | 21.16 | 25.07 | 24.03 | 23.70 | 23.48 | 22.33 | EJ/yr |
| TIAM_Grantham | PR_Baseline | 7.93 | 8.42 | — | 14.25 | — | 20.54 | — | 25.89 | — | 32.73 | EJ/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q094"></a>

## 094. tell me something interesting about the data

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 1<br>
Time: 1.35 s<br>
Conversation: standalone

**Answer**

I can answer questions about IAM PARIS results, models and climate-policy terms, but I need a little more to go on. Try for example:
- `CO2 emissions for World in 2050`
- `which scenario has the lowest emissions in 2050?`
- `which models cover transport?`
- `what does net zero mean?`

<a id="q095"></a>

## 095. which studies cover transport?

Category: data and scenarios<br>
Route: `data_query` (study/results catalogue request)<br>
Qwen generations: 0<br>
Time: 0.07 s<br>
Conversation: standalone

**Answer**

I found these workspaces (studies): AFOLU transformation; Behavioural changes for climate policy; Buildings Transformation; Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU; Cost of capital evolution effects on regional mitigation pathways; COVID recovery packages; Decarbonization potentials of energy citizens; Decipher modelling studies; Enhancing renewable energy system metrics; EU path to net zero; Global impacts of NDCs; Implications of EU steel industry relocation and 9 more. Ask about a study to see its variables, scenarios, and direct IAM PARIS links.

Showing 12 of 21. Say `show all workspaces` if you need the full list.

<a id="q096"></a>

## 096. what scenarios exist for the EU?

Category: data and scenarios<br>
Route: `data_query` (scenario availability request)<br>
Qwen generations: 0<br>
Time: 0.42 s<br>
Conversation: standalone

**Answer**

I found scenarios for EU (European Union) like CBAM, COVID Recovery - Average portfolio, COVID Recovery - Max Emissions Cuts, COVID Recovery - Max Jobs 2025, COVID Recovery - Max Jobs 2030, CP_EI, INDEPENDENCE and NDC_EI and 12 more. Ask for a variable under any of these scenarios to see its values.

Showing 8 of 20. Say `show all scenarios` if you need the full list.

Reply with a scenario, model, region, or year to narrow the answer.

<a id="q097"></a>

## 097. I want hydrogen data for Europe

Category: data and scenarios<br>
Route: `data_query` (low confidence entity clarification)<br>
Qwen generations: 0<br>
Time: 0.21 s<br>
Conversation: standalone

**Answer**

I found several possible variables, but none has enough evidence for automatic selection. Choose the variable: 1. `Exports|Hydrogen` (hydrogen) 2. `Imports|Hydrogen` (hydrogen) 3. `Production|Hydrogen` (hydrogen) Reply with a number (1-3), or `yes` for option 1.

<a id="q098"></a>

## 098. Greece

Category: data and scenarios<br>
Route: `data_query` (extracted data entities)<br>
Qwen generations: 0<br>
Time: 0.31 s<br>
Conversation: standalone

**Answer**

### Data available for GREECE (Greece)

- **Variables:** 676 (`Average Annual Investment|Energy Supply|Electricity|Battery|Rooftop`, `Average Annual Investment|Energy Supply|Electricity|Battery|Utility`, `Average Annual Investment|Energy Supply|Electricity|Biomass`, `Average Annual Investment|Energy Supply|Electricity|CCS`, `Average Annual Investment|Energy Supply|Electricity|Coal`, `Average Annual Investment|Energy Supply|Electricity|FuelCell`, `Average Annual Investment|Energy Supply|Electricity|Gas`, `Average Annual Investment|Energy Supply|Electricity|Geothermal` and 668 more)
- **Scenarios:** 47
- **Models:** 9
- **Years:** 2005–2100

Ask for one variable, for example `CO2 emissions`, to retrieve its values.

Reply with a scenario, model, region, or year to narrow the answer.

<a id="q099"></a>

## 099. methane

Category: data and scenarios<br>
Route: `data_query` (extracted data entities)<br>
Qwen generations: 0<br>
Time: 0.39 s<br>
Conversation: standalone

**Answer**

Showing results from the study **Where is the world headed?**. Also available in: Post-Glasgow targets; Global impacts of NDCs; Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU; EU path to net zero; Behavioural changes for climate policy; Tech-constrained EU pathways to net zero; Where is the EU headed?. Name another study to switch.

### Emissions|CH4 in World

Scope: scenario `multiple`, model `multiple`, years `2005-2100`
Unit: `Mt CH4/yr`

Answer:
In 2050, Emissions|CH4 in World ranges from 353.64 (GEMINI-E3, PR_NDC_EI) to 465.12 Mt CH4/yr (GEMINI-E3, PR_Baseline) across 13 series; median 412.03. Median change since 2005: +14%.

Summary: 13 series across 2 models and 7 scenarios.
Showing 12 of 13 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 1 remaining series.

| Model | Scenario | 2005 | 2010 | 2015 | 2020 | 2025 | 2030 | 2035 | 2040 | 2045 | 2050 | 2055 | 2060 | 2065 | 2070 | 2075 | 2080 | 2085 | 2090 | 2095 | 2100 | Unit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GCAM | PR_Baseline | 345.08 | 367.57 | 380.58 | 409.93 | 426.25 | 434.59 | 440.83 | 446.37 | 456.36 | 461.74 | 464.43 | 467.42 | 469.26 | 469.24 | 468.33 | 466.04 | 464.71 | 461.86 | 459.73 | 458.11 | Mt CH4/yr |
| GCAM | PR_CurPol_CP | 345.08 | 367.57 | 380.58 | 373.28 | 386.98 | 395.62 | 395.44 | 399.09 | 407.70 | 412.03 | 412.29 | 413.05 | 412.59 | 409.56 | 406.03 | 402.27 | 398.71 | 393.34 | 388.40 | 384.11 | Mt CH4/yr |
| GCAM | PR_CurPol_CPO | 345.08 | 367.57 | 380.58 | 370.17 | 389.35 | 398.65 | 397.74 | 400.91 | 409.06 | 413.38 | 413.35 | 413.66 | 413.01 | 409.59 | 405.86 | 402.14 | 398.44 | 393.11 | 388.29 | 384.13 | Mt CH4/yr |
| GCAM | PR_CurPol_EI | 345.08 | 367.57 | 380.58 | 373.24 | 386.93 | 395.57 | 390.88 | 386.20 | 391.11 | 393.89 | 388.44 | 384.01 | 378.38 | 370.63 | 362.96 | 356.29 | 349.31 | 341.90 | 334.45 | 324.79 | Mt CH4/yr |
| GCAM | PR_NDC_CP | 345.08 | 367.57 | 380.58 | 373.25 | 361.69 | 360.85 | 365.09 | 367.51 | 375.59 | 380.05 | 379.01 | 377.72 | 375.52 | 371.18 | 365.30 | 358.90 | 351.72 | 343.45 | 336.28 | 329.44 | Mt CH4/yr |
| GCAM | PR_NDC_CPO | 345.08 | 367.57 | 380.58 | 370.22 | 370.14 | 370.28 | 372.85 | 374.59 | 382.02 | 385.56 | 384.39 | 383.61 | 382.06 | 378.20 | 372.95 | 368.86 | 364.34 | 359.48 | 353.67 | 348.28 | Mt CH4/yr |
| GEMINI-E3 | PR_Baseline | — | — | 373.71 | 383.52 | 396.98 | 408.65 | 421.97 | 436.21 | 450.51 | 465.12 | — | — | — | — | — | — | — | — | — | — | Mt CH4/yr |
| GEMINI-E3 | PR_CurPol_CP | — | — | 373.71 | 381.93 | 392.86 | 399.76 | 410.28 | 423.17 | 435.48 | 447.37 | — | — | — | — | — | — | — | — | — | — | Mt CH4/yr |
| GEMINI-E3 | PR_CurPol_CPO | — | — | 373.71 | 378.79 | 380.33 | 380.15 | 389.79 | 402.15 | 414.54 | 427.06 | — | — | — | — | — | — | — | — | — | — | Mt CH4/yr |
| GEMINI-E3 | PR_CurPol_EI | — | — | 373.71 | 381.93 | 392.86 | 399.76 | 403.52 | 402.04 | 401.17 | 406.35 | — | — | — | — | — | — | — | — | — | — | Mt CH4/yr |
| GEMINI-E3 | PR_NDC_CP | — | — | 373.71 | 379.98 | 380.03 | 377.92 | 384.16 | 395.54 | 406.12 | 416.97 | — | — | — | — | — | — | — | — | — | — | Mt CH4/yr |
| GEMINI-E3 | PR_NDC_EI | — | — | 373.71 | 379.98 | 380.03 | 377.92 | 373.54 | 365.62 | 357.62 | 353.64 | — | — | — | — | — | — | — | — | — | — | Mt CH4/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q100"></a>

## 100. thanks!

Category: data and scenarios<br>
Route: `general_qa` (greeting/small talk)<br>
Qwen generations: 0<br>
Time: 0.01 s<br>
Conversation: standalone

**Answer**

You're welcome! Ask me anything else about IAM PARIS data whenever you like.

<a id="q101"></a>

## 101. what can you do?

Category: data and scenarios<br>
Route: `general_qa` (greeting/small talk)<br>
Qwen generations: 0<br>
Time: 0.00 s<br>
Conversation: standalone

**Answer**

I answer questions about IAM PARIS climate data (https://iamparis.eu/). You can ask me to:
- Show data, e.g. `show CO2 emissions for Europe under Baseline`
- Compare results, e.g. `compare solar capacity for Greece and Germany`
- List what is available, e.g. `list models`, `show all scenarios`
- Explain a model, e.g. `what is GCAM?`
- Find IAM PARIS pages, e.g. `where can I find the policy catalogue?`

<a id="q102"></a>

## 102. explain the NDC scenario

Category: conceptual explanations<br>
Route: `general_qa` (glossary definition)<br>
Qwen generations: 0<br>
Time: 0.01 s<br>
Conversation: standalone

**Answer**

### Nationally Determined Contribution (NDC)

A Nationally Determined Contribution (NDC) is a country's climate pledge under the Paris Agreement, setting out its emission-reduction targets and actions. In IAM studies, an NDC scenario models the emission pathway implied by these pledges, usually extended beyond the pledge year with assumptions about later effort.

Source: UNFCCC; IPCC AR6 WGIII

<a id="q103"></a>

## 103. what's the difference between NDC and current policies?

Category: conceptual explanations<br>
Route: `general_qa` (glossary definition)<br>
Qwen generations: 0<br>
Time: 0.00 s<br>
Conversation: standalone

**Answer**

### Nationally Determined Contribution (NDC)

A Nationally Determined Contribution (NDC) is a country's climate pledge under the Paris Agreement, setting out its emission-reduction targets and actions. In IAM studies, an NDC scenario models the emission pathway implied by these pledges, usually extended beyond the pledge year with assumptions about later effort.

Source: UNFCCC; IPCC AR6 WGIII

### Current policies scenario

A current policies scenario assumes that only climate policies already implemented are in place and that no further policies are added. It differs from an NDC scenario, which also includes pledged but not yet implemented targets, and from a baseline, which assumes no (or no new) climate policy.

Source: IPCC AR6 WGIII scenario categories

To see how these differ in the data, ask for a variable under each scenario, for example `CO2 emissions for World in 2050`, then `compare with` another scenario.

<a id="q104"></a>

## 104. CO2 for the US in 2030

Category: data and scenarios<br>
Route: `data_query` (extracted data entities)<br>
Qwen generations: 0<br>
Time: 0.41 s<br>
Conversation: standalone

**Answer**

Showing results from the study **Where is the world headed?**. Also available in: Post-Glasgow targets; EU path to net zero; Global impacts of NDCs; Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU. Name another study to switch.

### Emissions|CO2 in USA

Scope: scenario `multiple`, model `multiple`, years `2030`
Unit: `Mt CO2/yr`

Answer:
In 2030, Emissions|CO2 in USA ranges from 3,241 (GCAM, PR_NDC_CP) to 5,402 Mt CO2/yr (MUSE, PR_Baseline) across 28 series; median 4,204.

Summary: 28 series across 4 models and 10 scenarios.
Showing 12 of 28 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 16 remaining series.

| Model | Scenario | 2030 | Unit |
|---|---|---|---|
| E3ME | PR_Baseline | 4,085 | Mt CO2/yr |
| E3ME | PR_CurPol_CP | 4,085 | Mt CO2/yr |
| E3ME | PR_CurPol_EI | 4,085 | Mt CO2/yr |
| GCAM | Baseline | 4,898 | Mt CO2/yr |
| GCAM | CP_EI | 4,405 | Mt CO2/yr |
| GCAM | PR_NDC_CP | 3,241 | Mt CO2/yr |
| GEMINI-E3 | PR_Baseline | 4,714 | Mt CO2/yr |
| GEMINI-E3 | PR_CurPol_CP | 4,694 | Mt CO2/yr |
| GEMINI-E3 | PR_CurPol_CPO | 4,698 | Mt CO2/yr |
| MUSE | PR_Baseline | 5,402 | Mt CO2/yr |
| MUSE | PR_CurPol_CP | 4,290 | Mt CO2/yr |
| MUSE | PR_CurPol_CPO | 4,947 | Mt CO2/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q105"></a>

## 105. emissions in the UK

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.37 s<br>
Conversation: standalone

**Answer**

Showing results from the study **EU path to net zero**. Also available in: Where is the world headed?. Name another study to switch.

Interpretation: I treated bare **emissions** as `Emissions|CO2` (carbon-dioxide emissions).

### Emissions|CO2 in GBR

Scope: scenario `multiple`, model `multiple`, years `2010-2050`
Unit: `Mt CO2/yr`

Answer:
In 2050, Emissions|CO2 in GBR ranges from -27.80 (NEMESIS 5.1, NZE_Bench_L) to 328.07 Mt CO2/yr (NEMESIS 5.1, NZE_EUPol_Stand) across 10 series; median 89.58. Median change since 2010: -83%.

Summary: 10 series across 2 models and 5 scenarios.
Series are ordered by model and scenario.

| Model | Scenario | 2010 | 2015 | 2020 | 2025 | 2030 | 2035 | 2040 | 2045 | 2050 | Unit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E4SMA-EU-TIMES 1.0 | NZE_Bench_H | 519.85 | — | 379.89 | 377.86 | 304.35 | — | 207.15 | — | 94.04 | Mt CO2/yr |
| E4SMA-EU-TIMES 1.0 | NZE_Bench_L | 519.85 | — | 379.89 | 367.26 | 281.33 | — | 168.49 | — | 46.50 | Mt CO2/yr |
| E4SMA-EU-TIMES 1.0 | NZE_Bench_M | 519.85 | — | 379.89 | 371.23 | 294.92 | — | 189.95 | — | 85.12 | Mt CO2/yr |
| E4SMA-EU-TIMES 1.0 | NZE_EUPol_Stand | 519.85 | — | 379.89 | 293.47 | 283.79 | — | 232.05 | — | 110.94 | Mt CO2/yr |
| E4SMA-EU-TIMES 1.0 | WWH21 | 519.85 | — | 380.11 | 378.41 | 338.95 | — | 297.20 | — | 285.48 | Mt CO2/yr |
| NEMESIS 5.1 | NZE_Bench_H | 509.77 | 420.38 | 341.49 | 287.62 | 203.74 | 109.77 | 62.94 | 17.50 | -27.43 | Mt CO2/yr |
| NEMESIS 5.1 | NZE_Bench_L | 509.77 | 420.38 | 341.49 | 287.62 | 203.73 | 109.81 | 62.96 | 17.39 | -27.80 | Mt CO2/yr |
| NEMESIS 5.1 | NZE_Bench_M | 509.77 | 420.38 | 341.49 | 287.62 | 203.73 | 109.79 | 62.94 | 17.45 | -27.55 | Mt CO2/yr |
| NEMESIS 5.1 | NZE_EUPol_Stand | 509.77 | 420.38 | 351.73 | 375.94 | 343.42 | 340.47 | 335.27 | 332.31 | 328.07 | Mt CO2/yr |
| NEMESIS 5.1 | WWH21 | 509.77 | 420.38 | 353.00 | 357.61 | 317.85 | 312.49 | 305.79 | 301.68 | 296.17 | Mt CO2/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/eu-path-to-net-zero/graphs).

Reply with a scenario, model, region, or year to narrow the answer.

<a id="q106"></a>

## 106. energy mix of China in 2050

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.27 s<br>
Conversation: standalone

**Answer**

Showing results from the study **Where is the world headed?**. Also available in: Post-Glasgow targets; Global impacts of NDCs; Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU; Tech-constrained EU pathways to net zero. Name another study to switch.

### Primary Energy in CHN

Scope: scenario `multiple`, model `multiple`, years `2050`
Unit: `EJ/yr`

Answer:
In 2050, Primary Energy in CHN ranges from 108.04 (GEMINI-E3, PR_NDC_EI) to 203.49 EJ/yr (GEMINI-E3, PR_Baseline) across 40 series; median 147.67.

Summary: 40 series across 6 models and 8 scenarios.
Showing 12 of 40 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 28 remaining series.

| Model | Scenario | 2050 | Unit |
|---|---|---|---|
| E3ME | PR_Baseline | 155.98 | EJ/yr |
| E3ME | PR_CurPol_CP | 155.38 | EJ/yr |
| GCAM | PR_Baseline | 161.96 | EJ/yr |
| GCAM | PR_CurPol_CP | 153.46 | EJ/yr |
| GEMINI-E3 | PR_Baseline | 203.49 | EJ/yr |
| GEMINI-E3 | PR_CurPol_CP | 143.10 | EJ/yr |
| GEMINI-E3 | PR_NDC_EI | 108.04 | EJ/yr |
| ICES-XPS | PR_Baseline | 193.02 | EJ/yr |
| ICES-XPS | PR_CurPol_CP | 190.52 | EJ/yr |
| MUSE | PR_Baseline | 125.19 | EJ/yr |
| MUSE | PR_CurPol_CP | 135.00 | EJ/yr |
| TIAM_Grantham | PR_Baseline | 139.05 | EJ/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q107"></a>

## 107. how many models are there?

Category: data and scenarios<br>
Route: `data_query` (model availability request)<br>
Qwen generations: 0<br>
Time: 0.11 s<br>
Conversation: standalone

**Answer**

There are **76 distinct models** in the loaded database.

<a id="q108"></a>

## 108. who developed PROMETHEUS?

Category: model descriptions<br>
Route: `model_explanation` (named-model metadata request)<br>
Qwen generations: 0<br>
Time: 0.08 s<br>
Conversation: standalone

**Answer**

### PROMETHEUS

`PROMETHEUS` is developed by E3Modelling S.A. https://e3modelling.com/.

Related model documentation:
- [IAM PARIS model page](https://iamparis.eu/models/55)

<a id="q109"></a>

## 109. which models are open source?

Category: model descriptions<br>
Route: `model_explanation` (model attribute not in catalogue)<br>
Qwen generations: 0<br>
Time: 0.02 s<br>
Conversation: standalone

**Answer**

The IAM PARIS model catalogue does not record licensing or code availability, so I can't tell whether the models are open source or downloadable. Each model's IAM PARIS page links to its documentation and references, which usually state how the model can be accessed.

<a id="q110"></a>

## 110. give me final energy for transport in EU in 2030 under net zero

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.46 s<br>
Conversation: standalone

**Answer**

I could not find data for `Final Energy|Transportation` in region `EU (European Union)` under the requested scenario family `Net Zero`.

Reason: the exact scenario combination is not available for this data slice.

Closest valid options:

1. region `AT` (AT (Austria))

2. region `BE` (BE (Belgium))

3. region `BE_LU_NL`

Reply with `1`, `2`, or `3`, or type the region you want.

<a id="q111"></a>

## 111. average CO2 across models for World 2050

Category: data and scenarios<br>
Route: `data_query` (extracted data entities)<br>
Qwen generations: 0<br>
Time: 0.40 s<br>
Conversation: standalone

**Answer**

Showing results from the study **Where is the world headed?**. Also available in: Post-Glasgow targets; Global impacts of NDCs; Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU; EU path to net zero; Tech-constrained EU pathways to net zero. Name another study to switch.

### Emissions|CO2 in World

Scope: scenario `multiple`, model `multiple`, years `2050`
Unit: `Mt CO2/yr`

Answer:
**Average across 38 series in 2050:** 32,464 Mt CO2/yr (range 9,173 to 52,211).

Summary: 38 series across 4 models and 13 scenarios.
Showing 12 of 38 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 26 remaining series.

| Model | Scenario | 2050 | Unit |
|---|---|---|---|
| E3ME | PR_Baseline | 40,393 | Mt CO2/yr |
| E3ME | PR_CurPol_CP | 40,054 | Mt CO2/yr |
| E3ME | PR_CurPol_EI | 39,239 | Mt CO2/yr |
| GCAM | Baseline | 45,595 | Mt CO2/yr |
| GCAM | CP_EI | 28,519 | Mt CO2/yr |
| GCAM | NDC_LTT | 9,173 | Mt CO2/yr |
| GEMINI-E3 | Baseline | 46,713 | Mt CO2/yr |
| GEMINI-E3 | CP_EI | 27,388 | Mt CO2/yr |
| GEMINI-E3 | PR_Baseline | 52,211 | Mt CO2/yr |
| MUSE | baseline | 31,448 | Mt CO2/yr |
| MUSE | CP_EI | 23,856 | Mt CO2/yr |
| MUSE | NDC_EI | 19,999 | Mt CO2/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q112"></a>

## 112. what are the main uncertainties in IAM projections?

Category: conceptual explanations<br>
Route: `general_qa` (conceptual climate explanation)<br>
Qwen generations: 1<br>
Time: 0.65 s<br>
Conversation: standalone

**Answer**

The main uncertainties in IAM projections include assumptions about socioeconomic development, technology costs, policy, and model structure. These uncertainties lead to variability in the outcomes, as shown by model intercomparisons.

<a id="q113"></a>

## 113. What does net zero mean for electricity prices?

Category: conceptual explanations<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.53 s<br>
Conversation: standalone

**Answer**

`Price|Secondary Energy|Electricity` is reported for 32 regions. Choose the region: 1. `AT` (AT (Austria)) 2. `BE` (BE (Belgium)) 3. `BG` (BG (Bulgaria)) 4. `CY` (CY (Cyprus)) 5. `CZ` (CZ (Czechia)) Reply with a number (1-5), or `yes` for option 1.

<a id="q114"></a>

## 114. Why can carbon prices differ between models?

Category: conceptual explanations<br>
Route: `general_qa` (conceptual climate explanation)<br>
Qwen generations: 1<br>
Time: 0.54 s<br>
Conversation: standalone

**Answer**

Carbon prices differ between models because they consider different assumptions about technology costs, fuel switching options, economic structures, and baseline emissions. Models with more cheap low-carbon options require lower carbon prices, while those with fewer options need higher prices.

<a id="q115"></a>

## 115. How does carbon pricing affect emissions?

Category: conceptual explanations<br>
Route: `general_qa` (conceptual climate explanation)<br>
Qwen generations: 1<br>
Time: 0.61 s<br>
Conversation: standalone

**Answer**

Carbon pricing increases the cost of each fuel in proportion to its CO2 emissions, which encourages investment in low-carbon energy sources, energy efficiency, and CCS. This shifts investment towards sectors that emit less, reducing fossil fuel demand. The economic impact depends on the model and how carbon revenues are used.

<a id="q116"></a>

## 116. What role does hydrogen play in net zero pathways?

Category: conceptual explanations<br>
Route: `general_qa` (conceptual climate explanation)<br>
Qwen generations: 1<br>
Time: 0.57 s<br>
Conversation: standalone

**Answer**

Hydrogen plays a role in net zero pathways by acting as a reducing agent in steelmaking and a feedstock for ammonia and refining, replacing fossil fuels in industries where direct electrification is challenging. Its effectiveness depends on hydrogen production costs and the cost of low-carbon electricity.

<a id="q117"></a>

## 117. Why might renewable capacity rise while emissions stay high?

Category: conceptual explanations<br>
Route: `general_qa` (conceptual climate explanation)<br>
Qwen generations: 1<br>
Time: 0.90 s<br>
Conversation: standalone

**Answer**

Renewable capacity rising while emissions remain high could be due to factors like increased demand for energy from population and economic growth, fossil fuel competitiveness, and efficiency gains from renewables. Additionally, the integration of variable renewables into the grid and storage systems may add costs, which could offset the benefits of renewables without reversing the trend.

<a id="q118"></a>

## 118. How should I interpret differences between two IAM studies?

Category: conceptual explanations<br>
Route: `general_qa` (conceptual climate explanation)<br>
Qwen generations: 1<br>
Time: 0.63 s<br>
Conversation: standalone

**Answer**

To interpret differences between two IAM studies, compare them based on the same variable, region, unit, and year. Ensure scenarios are defined consistently, and check for harmonisation in model versions and scenarios. Differences often reflect scenario design rather than model behavior.

<a id="q119"></a>

## 119. Why do baseline scenarios sometimes show falling emissions?

Category: conceptual explanations<br>
Route: `general_qa` (conceptual climate explanation)<br>
Qwen generations: 1<br>
Time: 0.59 s<br>
Conversation: standalone

**Answer**

Baseline scenarios sometimes show falling emissions because they assume no new climate policies beyond what is already in place, leading to increased demand for energy and fossil fuels. Population and economic growth also contribute to rising emissions, and efficiency gains and cheaper renewables slow growth, but without additional policy, emissions may not fall.

<a id="q120"></a>

## 120. What does a negative emissions value mean?

Category: conceptual explanations<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.53 s<br>
Conversation: standalone

**Answer**

Showing results from the study **Where is the world headed?**. Also available in: EU path to net zero; Post-Glasgow targets; Global impacts of NDCs; Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU; Tech-constrained EU pathways to net zero; Behavioural changes for climate policy; Where is the EU headed?; Enhancing renewable energy system metrics. Name another study to switch.

Interpretation: I treated bare **emissions** as `Emissions|CO2` (carbon-dioxide emissions).

### Emissions|CO2 in World

Scope: scenario `multiple`, model `multiple`, years `2005-2100`
Unit: `Mt CO2/yr`

Answer:
**Average across 22 series in 2100:** 25,865 Mt CO2/yr (range -766.86 to 62,101).

Summary: 38 series across 4 models and 13 scenarios.
Showing 12 of 38 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 26 remaining series.

| Model | Scenario | 2005 | 2010 | 2015 | 2020 | 2025 | 2030 | 2035 | 2040 | 2045 | 2050 | 2055 | 2060 | 2065 | 2070 | 2075 | 2080 | 2085 | 2090 | 2095 | 2100 | Unit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E3ME | PR_Baseline | — | 31,521 | 34,433 | 34,415 | 35,386 | 36,934 | 38,149 | 38,757 | 39,071 | 40,393 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| E3ME | PR_CurPol_CP | — | 31,521 | 34,433 | 34,416 | 35,385 | 36,935 | 38,006 | 38,479 | 38,772 | 40,054 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| E3ME | PR_CurPol_EI | — | 31,521 | 34,433 | 34,407 | 35,350 | 36,922 | 37,792 | 38,706 | 39,138 | 39,239 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| GCAM | Baseline | 31,426 | 36,147 | 39,196 | 38,300 | 40,654 | 42,107 | 43,979 | 45,315 | 45,465 | 45,595 | 46,708 | 48,206 | 49,791 | 51,521 | 53,322 | 54,770 | 56,170 | 57,563 | 58,673 | 60,068 | Mt CO2/yr |
| GCAM | CP_EI | 31,426 | 36,147 | 39,196 | 37,250 | 37,566 | 37,516 | 35,676 | 33,501 | 31,082 | 28,519 | 26,244 | 24,349 | 22,565 | 20,869 | 19,172 | 17,514 | 15,927 | 14,464 | 13,087 | 11,834 | Mt CO2/yr |
| GCAM | NDC_LTT | 31,426 | 36,147 | 39,196 | 37,250 | 35,865 | 32,324 | 26,749 | 20,969 | 14,887 | 9,173 | 4,316 | 871.19 | 669.19 | 59.16 | -310.14 | -114.52 | -451.30 | -589.76 | -661.35 | -766.86 | Mt CO2/yr |
| GEMINI-E3 | Baseline | — | — | 34,738 | 34,934 | 38,132 | 42,037 | 44,008 | 45,532 | 46,413 | 46,713 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| GEMINI-E3 | CP_EI | — | — | 34,738 | 34,934 | 34,187 | 34,431 | 32,849 | 31,313 | 29,460 | 27,388 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| GEMINI-E3 | PR_Baseline | — | — | 34,859 | 37,813 | 41,214 | 44,462 | 46,876 | 48,984 | 50,715 | 52,211 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| MUSE | baseline | — | — | 36,386 | 37,813 | 37,955 | 35,891 | 34,436 | 33,696 | 32,387 | 31,448 | 30,979 | 31,285 | 32,420 | 33,030 | 34,518 | 35,378 | 36,141 | 36,237 | 37,010 | 37,729 | Mt CO2/yr |
| MUSE | CP_EI | — | — | 36,381 | 36,003 | 35,319 | 33,276 | 30,502 | 27,906 | 26,115 | 23,856 | 22,850 | 21,458 | 20,621 | 19,791 | 18,914 | 18,091 | 18,210 | 17,693 | 18,037 | 18,122 | Mt CO2/yr |
| MUSE | NDC_EI | — | — | 36,381 | 35,946 | 34,574 | 31,089 | 27,917 | 24,700 | 22,635 | 19,999 | 18,772 | 17,549 | 17,234 | 16,328 | 15,290 | 14,893 | 14,669 | 14,811 | 15,232 | 15,044 | Mt CO2/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q121"></a>

## 121. How does carbon capture affect the carbon budget?

Category: conceptual explanations<br>
Route: `general_qa` (conceptual climate explanation)<br>
Qwen generations: 1<br>
Time: 0.61 s<br>
Conversation: standalone

**Answer**

Carbon capture affects the carbon budget by reducing emissions from fossil fuels and power plants, while removing CO2 from the atmosphere. This helps balance the remaining emissions that remain, contributing to the carbon budget's limits.

<a id="q122"></a>

## 122. What uncertainties matter for projected electricity demand?

Category: conceptual explanations<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.52 s<br>
Conversation: standalone

**Answer**

I couldn't find `projected electricity demand` as a region in the IAM PARIS data, so I can't return results for it. Try one of the available regions, e.g. `EU (European Union)`, `AGO (Angola)`, `ARE (United Arab Emirates)`, `ARG (Argentina)`, `AUS (Australia)`, or ask `list regions` to see all options.

<a id="q123"></a>

## 123. Explain why net zero is not the same as zero emissions.

Category: conceptual explanations<br>
Route: `general_qa` (conceptual climate explanation)<br>
Qwen generations: 1<br>
Time: 0.85 s<br>
Conversation: standalone

**Answer**

Net zero is not the same as zero emissions because it aims to balance anthropogenic CO₂ emissions with removals (like afforestation or direct air capture) over a specified period, not eliminate them entirely. Zero emissions would mean no anthropogenic CO₂ is emitted, which contradicts the concept of net zero. The key difference lies in the balance between emissions and removals, making net zero a more realistic and long-term goal.

<a id="q124"></a>

## 124. What are the limitations of long term climate projections?

Category: data and scenarios<br>
Route: `general_qa` (conceptual climate explanation)<br>
Qwen generations: 1<br>
Time: 0.56 s<br>
Conversation: standalone

**Answer**

The limitations of long-term climate projections include the short-lived nature of methane, which has a shorter atmospheric lifetime compared to CO2, and the fact that methane reductions primarily slow warming within decades, while CO2 reductions mainly limit long-term warming. These factors can affect the accuracy of projections over time.

<a id="q125"></a>

## 125. Why might two scenarios with the same target have different costs?

Category: conceptual explanations<br>
Route: `general_qa` (conceptual climate explanation)<br>
Qwen generations: 1<br>
Time: 0.64 s<br>
Conversation: standalone

**Answer**

Two scenarios with the same target might have different costs because they assume different technology costs, fuel switching methods, economic structures, or baseline emissions. A model with cheaper low-carbon options requires a lower carbon price, while a model with higher technology costs needs a higher price. (Source: IAM PARIS; IPCC AR6 WGIII Ch. 3)

<a id="q126"></a>

## 126. How does land use influence mitigation pathways?

Category: conceptual explanations<br>
Route: `general_qa` (conceptual climate explanation)<br>
Qwen generations: 1<br>
Time: 0.74 s<br>
Conversation: standalone

**Answer**

Land use influences mitigation pathways by affecting carbon emissions through deforestation, reforestation, and soil management. For example, clearing forests releases stored carbon as CO₂, while afforestation and soil management take CO₂ from the atmosphere. IAMs report these flows under AFOLU, such as `Emissions|CO2|AFOLU`.

<a id="q127"></a>

## 127. What role do removals play if emissions reach net zero?

Category: conceptual explanations<br>
Route: `general_qa` (conceptual climate explanation)<br>
Qwen generations: 1<br>
Time: 0.44 s<br>
Conversation: standalone

**Answer**

Removals play a role in balancing residual emissions from sectors like aviation, shipping, and agriculture, ensuring net-zero goals can be achieved. They also help offset cumulative emissions if pathways overshoot the temperature or carbon budget.

<a id="q128"></a>

## 128. Describe the GCAM model and its main sectors.

Category: data and scenarios<br>
Route: `model_explanation` (named-model metadata request)<br>
Qwen generations: 0<br>
Time: 0.06 s<br>
Conversation: standalone

**Answer**

### GCAM

Description:
GCAM is an integrated assessment model that represents interactions among energy, economy, land, water and climate systems. It is commonly used for scenario analysis of emissions, energy transitions and climate policy.

Developer:
Joint Global Change Research Institute

Model scope:
- energy
- economy
- land
- water
- climate

Useful for:
- emissions and energy pathway analysis
- climate policy scenarios
- land and resource interactions
- regional mitigation comparisons

Interpretation notes:
- model assumptions vary by scenario and input dataset
- local IAM PARIS metadata may not include detailed assumptions

Related model documentation:
- [IAM PARIS model page](https://iamparis.eu/models/30)

<a id="q129"></a>

## 129. How does REMIND represent energy transitions?

Category: conceptual explanations<br>
Route: `model_explanation` (model information request)<br>
Qwen generations: 0<br>
Time: 0.48 s<br>
Conversation: standalone

**Answer**

There is no exact `REMIND` entry in the loaded IAM PARIS model catalogue. Showing the closest entry, `REMIND-MFA`; check that this is the model you mean.

### REMIND-MFA

Description:
REMIND-MFA includes top-down prospective material flow analysis (MFA) models for the basic materials cement, plastics, and steel. It is designed to provide material demands and flows for the integrated assessment model REMIND.

Model scope:
- energy
- economy
- climate
- land-use interactions

Useful for:
- global and regional mitigation pathways
- carbon pricing and policy scenarios
- energy-system transformation
- technology deployment including renewables, CCS and negative emissions

Interpretation notes:
- assumptions vary by scenario, model version and experiment design
- local IAM PARIS metadata may not expose every technology assumption

Related model documentation:
- [IAM PARIS model page](https://iamparis.eu/models/87)

<a id="q130"></a>

## 130. What is the difference between WITCH and GEM-E3?

Category: conceptual explanations<br>
Route: `model_explanation` (multi-model metadata comparison)<br>
Qwen generations: 0<br>
Time: 0.09 s<br>
Conversation: standalone

**Answer**

### GEM-E3 vs WITCH

**GEM-E3**
- Developer: E3Modelling - RICARDO

**WITCH**
- Systems/sectors: economy, energy, climate policy, technology innovation
- Common uses: regional mitigation and burden-sharing analysis; carbon pricing and policy assessment; technology innovation and diffusion scenarios; long-term climate policy pathways

Catalogue wording match for `between`, `difference`: **WITCH**.
Sector coverage is not described in the loaded metadata for **GEM-E3**, so coverage cannot be compared.

[Browse IAM PARIS model documentation](https://iamparis.eu/models)

<a id="q131"></a>

## 131. Who develops IMAGE and what does it cover?

Category: conceptual explanations<br>
Route: `model_explanation` (named-model metadata request)<br>
Qwen generations: 0<br>
Time: 0.08 s<br>
Conversation: standalone

**Answer**

### IMAGE

`IMAGE` is developed by IMAGE team under the authority of PBL Netherlands Environmental Assessment Agency.

Related model documentation:
- [Official IMAGE model documentation (PBL)](https://www.pbl.nl/en/image/home)
- [IAM PARIS Models](https://iamparis.eu/models)
- Open the Models page and search for: `IMAGE`

<a id="q132"></a>

## 132. Does PROMETHEUS model transport?

Category: data and scenarios<br>
Route: `model_explanation` (named-model metadata request)<br>
Qwen generations: 0<br>
Time: 0.06 s<br>
Conversation: standalone

**Answer**

### PROMETHEUS

Yes. The loaded IAM PARIS profile explicitly indicates that `PROMETHEUS` models `transport`.
Supporting profile evidence: the description or use-case metadata explicitly mentions `transport`.

Related model documentation:
- [IAM PARIS model page](https://iamparis.eu/models/55)

<a id="q133"></a>

## 133. Which models report hydrogen variables?

Category: catalogue and coverage<br>
Route: `data_query` (topic-wide model availability)<br>
Qwen generations: 0<br>
Time: 0.20 s<br>
Conversation: standalone

**Answer**

### Models reporting hydrogen variables

16 model(s) report at least one of 113 variable(s) mentioning hydrogen, e.g. `Average Annual Investment|Energy Supply|Hydrogen|Electrolyser`, `Capacity|Hydrogen|Biomass`, `Capacity|Hydrogen|Biomass|w/ CCS`, `Capacity|Hydrogen|Biomass|w/o CCS`, `Capacity|Hydrogen|Coal` and 108 more:

ALADIN, E3ME, E4SMA-EU-TIMES 1.0, EnergyPLAN, FORECAST, GCAM 7.0, GCAM-PR 5.3, GEM-E3, MUSE, OSeMOSYS-GR, PRIMES-Buildings, PROMETHEUS V1, tiam, TIAM_Grantham 3.2, WISEE EDM-I Steel, WISEE-EDM

Ask for one of these variables (e.g. `show Average Annual Investment|Energy Supply|Hydrogen|Electrolyser for World`) to see values.

<a id="q134"></a>

## 134. Which models have data for Greece?

Category: catalogue and coverage<br>
Route: `data_query` (topic-wide model availability)<br>
Qwen generations: 0<br>
Time: 0.06 s<br>
Conversation: standalone

**Answer**

No loaded IAM PARIS variable mentions `greece`, so I cannot list models that report it. Ask `list variables` to see the available variables.

<a id="q135"></a>

## 135. What data do you have for Africa?

Category: catalogue and coverage<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.25 s<br>
Conversation: standalone

**Answer**

### Data available for AFR (Africa)

- **Variables:** 296 (`Capacity Additions|Electricity`, `Capacity Additions|Electricity|Biomass`, `Capacity Additions|Electricity|Biomass|w/o CCS`, `Capacity Additions|Electricity|Coal`, `Capacity Additions|Electricity|Coal|w/o CCS`, `Capacity Additions|Electricity|Gas`, `Capacity Additions|Electricity|Gas|w/o CCS`, `Capacity Additions|Electricity|Geothermal` and 288 more)
- **Scenarios:** 10
- **Models:** 2
- **Years:** 2005–2100

Ask for one variable, for example `CO2 emissions`, to retrieve its values.

Reply with a scenario, model, region, or year to narrow the answer.

<a id="q136"></a>

## 136. Where can I download the underlying dataset?

Category: data and scenarios<br>
Route: `data_query` (unclear deterministic route)<br>
Qwen generations: 2<br>
Time: 1.34 s<br>
Conversation: standalone

**Answer**

I can answer questions about IAM PARIS results, models and climate-policy terms, but I need a little more to go on. Try for example:
- `CO2 emissions for World in 2050`
- `which scenario has the lowest emissions in 2050?`
- `which models cover transport?`
- `what does net zero mean?`

<a id="q137"></a>

## 137. How can I cite the results shown here?

Category: data and scenarios<br>
Route: `general_qa` (explicit site navigation request)<br>
Qwen generations: 0<br>
Time: 0.05 s<br>
Conversation: standalone

**Answer**

Use these IAM PARIS links for this request: [Results](https://iamparis.eu/results).

<a id="q138"></a>

## 138. show me carbon dioxide emissions for Europe

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.42 s<br>
Conversation: `conv-001`

**Answer**

Showing results from the study **Where is the world headed?**. Also available in: Where is the EU headed?; Post-Glasgow targets. Name another study to switch.

### Emissions|CO2 in EU

Scope: scenario `multiple`, model `multiple`, years `2005-2100`
Unit: `Mt CO2/yr`

Answer:
In 2050, Emissions|CO2 in EU ranges from 1,024 (GCAM, PR_CurPol_CPO) to 3,429 Mt CO2/yr (GCAM, PR_Baseline) across 26 series; median 2,002. Median change since 2005: -71%.

Summary: 26 series across 4 models and 8 scenarios.
Showing 12 of 26 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 14 remaining series.

| Model | Scenario | 2005 | 2010 | 2015 | 2020 | 2025 | 2030 | 2035 | 2040 | 2045 | 2050 | 2055 | 2060 | 2065 | 2070 | 2075 | 2080 | 2085 | 2090 | 2095 | 2100 | Unit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E3ME | PR_Baseline | — | 4,304 | 3,999 | 3,444 | 3,188 | 3,033 | 2,864 | 2,692 | 2,516 | 2,323 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| E3ME | PR_CurPol_CP | — | 4,304 | 3,999 | 3,444 | 3,188 | 3,033 | 2,869 | 2,705 | 2,539 | 2,359 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| E3ME | PR_CurPol_EI | — | 4,304 | 3,999 | 3,445 | 3,188 | 3,032 | 2,919 | 2,827 | 2,737 | 2,633 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| GCAM | PR_Baseline | 4,190 | 3,790 | 3,490 | 3,543 | 3,524 | 3,454 | 3,524 | 3,549 | 3,509 | 3,429 | 3,476 | 3,589 | 3,712 | 3,829 | 3,926 | 4,029 | 4,168 | 4,311 | 4,472 | 4,672 | Mt CO2/yr |
| GCAM | PR_CurPol_CP | 4,190 | 3,790 | 3,490 | 3,115 | 2,687 | 2,257 | 2,112 | 1,836 | 1,485 | 1,118 | 962.69 | 912.20 | 870.84 | 863.75 | 839.93 | 771.80 | 656.18 | 503.01 | 281.84 | 10.13 | Mt CO2/yr |
| GCAM | PR_CurPol_CPO | 4,190 | 3,790 | 3,490 | 3,094 | 2,697 | 2,267 | 2,008 | 1,723 | 1,381 | 1,024 | 873.57 | 824.73 | 781.65 | 770.74 | 720.35 | 598.03 | 446.47 | 186.10 | -169.61 | -651.08 | Mt CO2/yr |
| GCAM | PR_CurPol_EI | 4,190 | 3,790 | 3,490 | 3,115 | 2,688 | 2,257 | 1,936 | 1,625 | 1,355 | 1,121 | 921.10 | 724.95 | 558.89 | 407.39 | 280.07 | 166.91 | 64.24 | -27.40 | -106.40 | -175.92 | Mt CO2/yr |
| GEMINI-E3 | PR_Baseline | — | — | 3,411 | 3,201 | 3,230 | 3,235 | 3,150 | 3,048 | 2,941 | 2,829 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| GEMINI-E3 | PR_CurPol_CP | — | — | 3,411 | 3,197 | 3,006 | 2,733 | 2,630 | 2,545 | 2,441 | 2,342 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| GEMINI-E3 | PR_CurPol_CPO | — | — | 3,411 | 3,197 | 3,006 | 2,731 | 2,626 | 2,541 | 2,437 | 2,337 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| MUSE | PR_Baseline | — | — | 3,141 | 3,106 | 3,076 | 2,994 | 2,842 | 2,701 | 2,384 | 2,074 | 1,838 | 1,664 | 1,588 | 1,543 | 1,400 | 1,279 | 1,101 | 1,034 | 977.08 | 947.41 | Mt CO2/yr |
| MUSE | PR_CurPol_CP | — | — | 3,118 | 3,018 | 2,956 | 2,740 | 2,581 | 2,435 | 2,155 | 1,871 | 1,641 | 1,482 | 1,414 | 1,365 | 1,237 | 1,116 | 975.01 | 910.90 | 844.23 | 817.09 | Mt CO2/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q139"></a>

## 139. same for China

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.56 s<br>
Conversation: `conv-001`

**Answer**

### Emissions|CO2 in CHN

Scope: scenario `multiple`, model `multiple`, years `2005-2100`
Unit: `Mt CO2/yr`

Answer:
In 2050, Emissions|CO2 in CHN ranges from 6,159 (MUSE, PR_NDC_CPO) to 16,585 Mt CO2/yr (GEMINI-E3, PR_Baseline) across 28 series; median 7,956. Median change since 2005: +30%.

Summary: 28 series across 4 models and 10 scenarios.
Showing 12 of 28 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 16 remaining series.

| Model | Scenario | 2005 | 2010 | 2015 | 2020 | 2025 | 2030 | 2035 | 2040 | 2045 | 2050 | 2055 | 2060 | 2065 | 2070 | 2075 | 2080 | 2085 | 2090 | 2095 | 2100 | Unit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E3ME | PR_Baseline | — | 7,787 | 9,450 | 10,280 | 10,814 | 11,105 | 10,755 | 9,505 | 8,119 | 7,542 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| E3ME | PR_CurPol_CP | — | 7,787 | 9,450 | 10,280 | 10,814 | 11,105 | 10,679 | 9,336 | 7,972 | 7,420 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| E3ME | PR_CurPol_EI | — | 7,787 | 9,450 | 10,280 | 10,814 | 11,105 | 10,864 | 10,503 | 10,030 | 9,609 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| GCAM | Baseline | 6,734 | 9,611 | 11,226 | 12,529 | 13,314 | 13,627 | 13,705 | 13,727 | 13,025 | 12,277 | 11,947 | 11,734 | 11,498 | 11,264 | 10,996 | 10,591 | 10,158 | 9,781 | 9,360 | 9,015 | Mt CO2/yr |
| GCAM | CP_EI | 6,734 | 9,611 | 11,226 | 12,133 | 12,380 | 12,238 | 10,983 | 9,491 | 7,924 | 6,473 | 5,226 | 4,154 | 3,205 | 2,384 | 1,664 | 1,077 | 594.72 | 204.01 | -104.68 | -345.55 | Mt CO2/yr |
| GCAM | PR_Baseline | 6,734 | 9,611 | 11,226 | 12,605 | 13,076 | 13,122 | 12,990 | 12,815 | 12,141 | 11,321 | 10,825 | 10,359 | 10,052 | 9,812 | 9,288 | 8,620 | 8,212 | 7,843 | 7,549 | 7,317 | Mt CO2/yr |
| GEMINI-E3 | PR_Baseline | — | — | 10,654 | 13,168 | 14,822 | 16,021 | 16,620 | 16,840 | 16,770 | 16,585 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| GEMINI-E3 | PR_CurPol_CP | — | — | 10,654 | 12,013 | 11,479 | 11,072 | 10,866 | 10,788 | 10,415 | 9,939 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| GEMINI-E3 | PR_CurPol_CPO | — | — | 10,654 | 12,070 | 11,935 | 11,655 | 11,520 | 11,502 | 11,153 | 10,692 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| MUSE | PR_Baseline | — | — | 11,234 | 12,637 | 13,262 | 13,488 | 11,591 | 10,284 | 8,467 | 6,552 | 5,214 | 4,313 | 3,843 | 3,402 | 3,148 | 2,929 | 2,748 | 2,609 | 2,472 | 2,356 | Mt CO2/yr |
| MUSE | PR_CurPol_CP | — | — | 11,232 | 12,243 | 12,789 | 12,775 | 11,463 | 9,893 | 8,388 | 6,538 | 5,245 | 4,353 | 3,916 | 3,471 | 3,187 | 2,916 | 2,738 | 2,596 | 2,443 | 2,335 | Mt CO2/yr |
| MUSE | PR_NDC_CPO | — | — | 11,234 | 12,632 | 13,091 | 11,847 | 10,599 | 9,202 | 8,009 | 6,159 | 4,891 | 4,011 | 3,559 | 3,120 | 2,972 | 2,840 | 2,709 | 2,580 | 2,474 | 2,355 | Mt CO2/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q140"></a>

## 140. what about 2050

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.37 s<br>
Conversation: `conv-001`

**Answer**

### Emissions|CO2 in CHN

Scope: scenario `multiple`, model `multiple`, years `2050`
Unit: `Mt CO2/yr`

Answer:
In 2050, Emissions|CO2 in CHN ranges from 6,159 (MUSE, PR_NDC_CPO) to 16,585 Mt CO2/yr (GEMINI-E3, PR_Baseline) across 28 series; median 7,956.

Summary: 28 series across 4 models and 10 scenarios.
Showing 12 of 28 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 16 remaining series.

| Model | Scenario | 2050 | Unit |
|---|---|---|---|
| E3ME | PR_Baseline | 7,542 | Mt CO2/yr |
| E3ME | PR_CurPol_CP | 7,420 | Mt CO2/yr |
| E3ME | PR_CurPol_EI | 9,609 | Mt CO2/yr |
| GCAM | Baseline | 12,277 | Mt CO2/yr |
| GCAM | CP_EI | 6,473 | Mt CO2/yr |
| GCAM | PR_Baseline | 11,321 | Mt CO2/yr |
| GEMINI-E3 | PR_Baseline | 16,585 | Mt CO2/yr |
| GEMINI-E3 | PR_CurPol_CP | 9,939 | Mt CO2/yr |
| GEMINI-E3 | PR_CurPol_CPO | 10,692 | Mt CO2/yr |
| MUSE | PR_Baseline | 6,552 | Mt CO2/yr |
| MUSE | PR_CurPol_CP | 6,538 | Mt CO2/yr |
| MUSE | PR_NDC_CPO | 6,159 | Mt CO2/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q141"></a>

## 141. plot it

Category: charts and comparisons<br>
Route: `general_qa` (charts delegated to data explorer)<br>
Qwen generations: 0<br>
Time: 0.01 s<br>
Conversation: `conv-001`

**Answer**

I do not generate figures in the chat. Please use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs) to explore, filter, and plot the data. Ask about a specific study first if you would like its direct explorer link.

<a id="q142"></a>

## 142. electricity for India

Category: data and scenarios<br>
Route: `data_query` (low confidence entity clarification)<br>
Qwen generations: 0<br>
Time: 0.39 s<br>
Conversation: `conv-002`

**Answer**

I found several possible variables, but none has enough evidence for automatic selection. Choose the variable: 1. `Secondary Energy|Electricity` (electricity output) 2. `Final Energy|Electricity` (final energy use, electricity) 3. `Capacity|Electricity` (power capacity, electricity) Reply with a number (1-3), or `yes` for option 1.

<a id="q143"></a>

## 143. 1

Category: data and scenarios<br>
Route: `data_query` (clarification selection; resolved)<br>
Qwen generations: 0<br>
Time: 0.29 s<br>
Conversation: `conv-002`

**Answer**

Showing results from the study **Where is the world headed?**. Also available in: Post-Glasgow targets; Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU; Tech-constrained EU pathways to net zero; Global impacts of NDCs; Implications of EU steel industry relocation. Name another study to switch.

### Secondary Energy|Electricity in IND

Scope: scenario `multiple`, model `multiple`, years `2005-2100`
Unit: `EJ/yr`

Answer:
In 2040, Secondary Energy|Electricity in IND ranges from 0.188 (MUSE, PR_CurPol_EI) to 23.23 EJ/yr (GCAM, PR_Baseline) across 43 series; median 15.00. Median change since 2005: +585%.

Summary: 43 series across 7 models and 8 scenarios.
Showing 12 of 43 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 31 remaining series.

| Model | Scenario | 2005 | 2010 | 2015 | 2020 | 2025 | 2030 | 2035 | 2040 | 2045 | 2050 | 2055 | 2060 | 2065 | 2070 | 2075 | 2080 | 2085 | 2090 | 2095 | 2100 | Unit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | PR_Baseline | 1.76 | 2.59 | 3.70 | 4.90 | 6.43 | 8.16 | 10.05 | 12.03 | 14.10 | — | — | — | — | — | — | — | — | — | — | — | EJ/yr |
| 42 | PR_CurPol_EI | 1.76 | 2.59 | 3.70 | 4.90 | 6.43 | 8.16 | 10.05 | 12.03 | 14.10 | — | — | — | — | — | — | — | — | — | — | — | EJ/yr |
| E3ME | PR_Baseline | — | 3.53 | 4.94 | 6.68 | 8.34 | 10.35 | 12.57 | 15.00 | 18.08 | 22.04 | — | — | — | — | — | — | — | — | — | — | EJ/yr |
| E3ME | PR_CurPol_CP | — | 3.53 | 4.94 | 6.68 | 8.34 | 10.35 | 12.55 | 14.96 | 18.02 | 21.94 | — | — | — | — | — | — | — | — | — | — | EJ/yr |
| GCAM | PR_Baseline | 2.58 | 3.53 | 4.95 | 8.47 | 12.21 | 16.06 | 19.34 | 23.23 | 27.33 | 31.26 | 34.85 | 38.09 | 40.98 | 43.52 | 45.48 | 46.97 | 47.92 | 48.46 | 48.58 | 48.56 | EJ/yr |
| GCAM | PR_CurPol_CP | 2.58 | 3.53 | 4.95 | 7.67 | 11.76 | 15.11 | 17.98 | 21.55 | 25.31 | 28.92 | 32.27 | 35.27 | 37.97 | 40.33 | 42.17 | 43.54 | 44.45 | 44.95 | 45.08 | 45.09 | EJ/yr |
| GEMINI-E3 | PR_Baseline | — | — | 5.15 | 6.77 | 8.76 | 11.14 | 13.47 | 16.01 | 18.87 | 21.78 | — | — | — | — | — | — | — | — | — | — | EJ/yr |
| GEMINI-E3 | PR_CurPol_CP | — | — | 5.15 | 6.82 | 9.34 | 11.86 | 14.69 | 17.82 | 21.46 | 25.25 | — | — | — | — | — | — | — | — | — | — | EJ/yr |
| ICES-XPS | PR_Baseline | — | — | 3.73 | 4.80 | 7.22 | 10.27 | 13.66 | 17.70 | 22.44 | 27.88 | — | — | — | — | — | — | — | — | — | — | EJ/yr |
| ICES-XPS | PR_CurPol_CP | — | — | 3.73 | 4.80 | 7.19 | 10.19 | 13.51 | 17.35 | 21.64 | 26.23 | — | — | — | — | — | — | — | — | — | — | EJ/yr |
| MUSE | PR_CurPol_EI | — | — | 0.129 | 0.128 | 0.126 | 0.118 | 0.2 | 0.188 | 0.179 | 0.176 | 0.175 | 0.17 | 0.183 | 0.188 | 0.194 | 0.198 | 0.203 | 0.209 | 0.214 | 0.213 | EJ/yr |
| TIAM_Grantham | PR_Baseline | 2.35 | 3.09 | — | 5.44 | — | 6.78 | — | 9.54 | — | 13.21 | — | 16.05 | — | 18.26 | — | 22.40 | — | 24.02 | — | 27.13 | EJ/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q144"></a>

## 144. plot it

Category: charts and comparisons<br>
Route: `general_qa` (charts delegated to data explorer)<br>
Qwen generations: 0<br>
Time: 0.01 s<br>
Conversation: `conv-002`

**Answer**

I do not generate figures in the chat. Please use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs) to explore, filter, and plot the data. Ask about a specific study first if you would like its direct explorer link.

<a id="q145"></a>

## 145. show Emissions|CO2 for World under Baseline in 2030

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.28 s<br>
Conversation: `conv-003`

**Answer**

Showing results from the study **Where is the world headed?**. Also available in: Post-Glasgow targets; EU path to net zero; Global impacts of NDCs. Name another study to switch.

### Emissions|CO2 in World

Scope: scenario `multiple`, model `multiple`, years `2030`
Unit: `Mt CO2/yr`

Answer:
In 2030, Emissions|CO2 in World ranges from 35,891 (MUSE, baseline) to 44,462 Mt CO2/yr (GEMINI-E3, PR_Baseline) across 8 series; median 41,779.

Summary: 8 series across 4 models and 4 scenarios.
Series are ordered by model and scenario.

| Model | Scenario | 2030 | Unit |
|---|---|---|---|
| E3ME | PR_Baseline | 36,934 | Mt CO2/yr |
| E3ME | Unharmonised baseline | 37,096 | Mt CO2/yr |
| GCAM | Baseline | 42,107 | Mt CO2/yr |
| GCAM | PR_Baseline | 42,663 | Mt CO2/yr |
| GEMINI-E3 | Baseline | 42,037 | Mt CO2/yr |
| GEMINI-E3 | PR_Baseline | 44,462 | Mt CO2/yr |
| MUSE | baseline | 35,891 | Mt CO2/yr |
| MUSE | PR_Baseline | 41,521 | Mt CO2/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q146"></a>

## 146. after 2030

Category: data and scenarios<br>
Route: `data_query` (year change for previous question)<br>
Qwen generations: 0<br>
Time: 0.11 s<br>
Conversation: `conv-003`

**Answer**

### Emissions|CO2 in World

Scope: scenario `multiple`, model `multiple`, years `2031-latest available`
Unit: `Mt CO2/yr`

Answer:
In 2050, Emissions|CO2 in World ranges from 29,678 (MUSE, PR_Baseline) to 52,211 Mt CO2/yr (GEMINI-E3, PR_Baseline) across 8 series; median 43,333. Median change since 2035: +6%.

Summary: 8 series across 4 models and 4 scenarios.
Series are ordered by model and scenario.

| Model | Scenario | 2035 | 2040 | 2045 | 2050 | 2055 | 2060 | 2065 | 2070 | 2075 | 2080 | 2085 | 2090 | 2095 | 2100 | Unit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E3ME | PR_Baseline | 38,149 | 38,757 | 39,071 | 40,393 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| E3ME | Unharmonised baseline | 38,457 | 39,232 | 39,600 | 41,071 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| GCAM | Baseline | 43,979 | 45,315 | 45,465 | 45,595 | 46,708 | 48,206 | 49,791 | 51,521 | 53,322 | 54,770 | 56,170 | 57,563 | 58,673 | 60,068 | Mt CO2/yr |
| GCAM | PR_Baseline | 44,466 | 45,961 | 46,635 | 46,979 | 48,176 | 49,647 | 51,430 | 53,411 | 54,919 | 56,024 | 57,552 | 59,032 | 60,469 | 62,101 | Mt CO2/yr |
| GEMINI-E3 | Baseline | 44,008 | 45,532 | 46,413 | 46,713 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| GEMINI-E3 | PR_Baseline | 46,876 | 48,984 | 50,715 | 52,211 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| MUSE | baseline | 34,436 | 33,696 | 32,387 | 31,448 | 30,979 | 31,285 | 32,420 | 33,030 | 34,518 | 35,378 | 36,141 | 36,237 | 37,010 | 37,729 | Mt CO2/yr |
| MUSE | PR_Baseline | 38,898 | 36,732 | 33,730 | 29,678 | 26,902 | 24,364 | 23,828 | 24,252 | 24,392 | 24,597 | 24,559 | 24,441 | 24,608 | 24,998 | Mt CO2/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q147"></a>

## 147. compare with current policy

Category: data and scenarios<br>
Route: `data_query` (chat plotting disabled; return numeric table)<br>
Qwen generations: 0<br>
Time: 0.24 s<br>
Conversation: `conv-003`

**Answer**

I couldn't find `current policy` as a region in the loaded IAM PARIS data. Please choose a loaded region using `list regions`; I cannot substitute a different geographic aggregate for the requested comparison.

<a id="q148"></a>

## 148. open Aqueduct details

Category: navigation and links<br>
Route: `general_qa` (explicit site navigation request)<br>
Qwen generations: 0<br>
Time: 0.05 s<br>
Conversation: `conv-004`

**Answer**

Use these IAM PARIS links for this request: [Aqueduct](https://iamparis.eu/application_library/474).

<a id="q149"></a>

## 149. where can I find Climate Watch

Category: navigation and links<br>
Route: `general_qa` (explicit site navigation request)<br>
Qwen generations: 0<br>
Time: 0.04 s<br>
Conversation: `conv-004`

**Answer**

Use these IAM PARIS links for this request: [Climate Watch](https://iamparis.eu/application_library).

If a direct detail page is not available:
- Open the Application Library and search for: Climate Watch

<a id="q150"></a>

## 150. Fit-for-55 EU net zero results

Category: data and scenarios<br>
Route: `general_qa` (explicit site navigation request)<br>
Qwen generations: 0<br>
Time: 0.45 s<br>
Conversation: `conv-005`

**Answer**

### EU path to net zero

Available in this study:
- **Topics (710 variables):** Final Energy (156), Emissions (104), Secondary Energy (70), Primary Energy (51), Price (37), Capacity (34), Final Energy (excl. feedstocks) (32), Investment (29) and 30 more
- **Example variables:** `Final Energy`, `Emissions|CO`, `Secondary Energy`, `Primary Energy`, `Price|Carbon`
- **Scenarios (7):** `Baseline`, `CP_EI`, `NZE_Bench_H`, `NZE_Bench_L`, `NZE_Bench_M`, `NZE_EUPol_Stand`, `WWH21`
- **Models (9):** ALADIN, E4SMA-EU-TIMES 1.0, FORECAST, GCAM, GCAM-PR 5.3, GEMINI-E3, GEMINI-E3 7.0, ICES-XPS 1.0 and 1 more
- **Regions (38):** World, AT, BE, BE_LU_NL, BG, CH and 32 more
- **Years:** 2005–2100

Explore this study:
- [Open the data explorer](https://iamparis.eu/results/paris-reinforce/eu-path-to-net-zero/graphs)
- [Read the study explainer](https://iamparis.eu/results/paris-reinforce/eu-path-to-net-zero/policy_questions)

<a id="q151"></a>

## 151. cost of capital mitigation pathways

Category: data and scenarios<br>
Route: `general_qa` (explicit site navigation request)<br>
Qwen generations: 0<br>
Time: 0.05 s<br>
Conversation: `conv-005`

**Answer**

Use these IAM PARIS links for this request: [Cost of capital evolution effects on regional mitigation pathways](https://iamparis.eu/results/iam-compact/cost-of-capital-evolution-effects-on-regional-mitigation-pathways/policy_questions).

<a id="q152"></a>

## 152. technology constrained pathways to net zero

Category: data and scenarios<br>
Route: `general_qa` (explicit site navigation request)<br>
Qwen generations: 0<br>
Time: 0.06 s<br>
Conversation: `conv-005`

**Answer**

Use these IAM PARIS links for this request: [Tech-constrained EU pathways to net zero](https://iamparis.eu/results/iam-compact/tech-constrained-eu-pathways-to-net-zero/policy_questions).

<a id="q153"></a>

## 153. global impacts of NDCs

Category: data and scenarios<br>
Route: `general_qa` (study summary)<br>
Qwen generations: 0<br>
Time: 0.15 s<br>
Conversation: `conv-006`

**Answer**

### Global impacts of NDCs

Available in this study:
- **Topics (355 variables):** Final Energy (77), Emissions (55), Secondary Energy (45), Primary Energy (39), Price (26), Capacity (22), Capacity Additions (20), Investment (20) and 14 more
- **Example variables:** `Final Energy`, `Emissions|CO`, `Secondary Energy|Gases`, `Primary Energy`, `Price|Carbon`
- **Scenarios (4):** `Baseline`, `CP_EI`, `NDC_EI`, `NDC_LTT`
- **Models (2):** GCAM-PR 5.3, PROMETHEUS V1
- **Regions (40):** World, Africa_Eastern, Africa_Northern, Africa_Southern, Africa_Western, Argentina and 34 more
- **Years:** 2005–2100

Explore this study:
- [Open the data explorer](https://iamparis.eu/results/ndc-aspects/global-impacts-of-ndcs/graphs)
- [Read the study explainer](https://iamparis.eu/results/ndc-aspects/global-impacts-of-ndcs/policy_questions)

<a id="q154"></a>

## 154. NDC ASPECTS transport results

Category: data and scenarios<br>
Route: `general_qa` (explicit site navigation request)<br>
Qwen generations: 0<br>
Time: 0.08 s<br>
Conversation: `conv-006`

**Answer**

Use these IAM PARIS links for this request: [NDC ASPECTS](https://iamparis.eu/results/ndc-aspects).

<a id="q155"></a>

## 155. AFOLU land use results

Category: data and scenarios<br>
Route: `general_qa` (explicit site navigation request)<br>
Qwen generations: 0<br>
Time: 0.04 s<br>
Conversation: `conv-006`

**Answer**

### AFOLU transformation

Available in this study:
- **Topics (28 variables):** Agricultural Demand (9), Land Cover (9), Primary Energy (4), Emissions (3), Price (3)
- **Example variables:** `Agricultural Demand`, `Land Cover`, `Primary Energy|Biomass`, `Emissions|CH4|AFOLU`, `Price|Agriculture|Corn|Index`
- **Scenarios (6):** `Cons_All`, `Cons_NoBiomass`, `Cons_NoFallow`, `Cons_NoForest`, `Reference`, `Uncons`
- **Models (1):** GCAM-PR 7.0
- **Regions (33):** World, Africa_Eastern, Africa_Northern, Africa_Southern, Africa_Western, Argentina and 27 more
- **Years:** 2005–2100

Explore this study:
- [Open the data explorer](https://iamparis.eu/results/ndc-aspects/afolu-transformation/graphs)
- [Read the study explainer](https://iamparis.eu/results/ndc-aspects/afolu-transformation/policy_questions)

<a id="q156"></a>

## 156. tell me about GCAM model assumptions

Category: model descriptions<br>
Route: `model_explanation` (named-model metadata request)<br>
Qwen generations: 0<br>
Time: 0.06 s<br>
Conversation: `conv-007`

**Answer**

### GCAM

Description:
GCAM is an integrated assessment model that represents interactions among energy, economy, land, water and climate systems. It is commonly used for scenario analysis of emissions, energy transitions and climate policy.

Developer:
Joint Global Change Research Institute

Model scope:
- energy
- economy
- land
- water
- climate

Useful for:
- emissions and energy pathway analysis
- climate policy scenarios
- land and resource interactions
- regional mitigation comparisons

Assumptions:
No explicit assumptions field is available in the local IAM PARIS model metadata. Use the selected scenario and IAM PARIS result metadata to interpret GCAM assumptions for a specific answer.

Interpretation notes:
- model assumptions vary by scenario and input dataset
- local IAM PARIS metadata may not include detailed assumptions

Related model documentation:
- [IAM PARIS model page](https://iamparis.eu/models/30)

<a id="q157"></a>

## 157. show data using GCAM

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.77 s<br>
Conversation: `conv-007`

**Answer**

I need one more detail. Which variable should I use?

<a id="q158"></a>

## 158. show CO2 emissions for World

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.55 s<br>
Conversation: `conv-007`

**Answer**

Showing results from the study **Where is the world headed?**. Also available in: Post-Glasgow targets; Global impacts of NDCs; Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU; EU path to net zero; Tech-constrained EU pathways to net zero. Name another study to switch.

### Emissions|CO2 in World

Scope: scenario `multiple`, model `multiple`, years `2005-2100`
Unit: `Mt CO2/yr`

Answer:
In 2050, Emissions|CO2 in World ranges from 9,173 (GCAM, NDC_LTT) to 52,211 Mt CO2/yr (GEMINI-E3, PR_Baseline) across 38 series; median 33,062. Median change since 2005: +6%.

Summary: 38 series across 4 models and 13 scenarios.
Showing 12 of 38 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 26 remaining series.

| Model | Scenario | 2005 | 2010 | 2015 | 2020 | 2025 | 2030 | 2035 | 2040 | 2045 | 2050 | 2055 | 2060 | 2065 | 2070 | 2075 | 2080 | 2085 | 2090 | 2095 | 2100 | Unit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E3ME | PR_Baseline | — | 31,521 | 34,433 | 34,415 | 35,386 | 36,934 | 38,149 | 38,757 | 39,071 | 40,393 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| E3ME | PR_CurPol_CP | — | 31,521 | 34,433 | 34,416 | 35,385 | 36,935 | 38,006 | 38,479 | 38,772 | 40,054 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| E3ME | PR_CurPol_EI | — | 31,521 | 34,433 | 34,407 | 35,350 | 36,922 | 37,792 | 38,706 | 39,138 | 39,239 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| GCAM | Baseline | 31,426 | 36,147 | 39,196 | 38,300 | 40,654 | 42,107 | 43,979 | 45,315 | 45,465 | 45,595 | 46,708 | 48,206 | 49,791 | 51,521 | 53,322 | 54,770 | 56,170 | 57,563 | 58,673 | 60,068 | Mt CO2/yr |
| GCAM | CP_EI | 31,426 | 36,147 | 39,196 | 37,250 | 37,566 | 37,516 | 35,676 | 33,501 | 31,082 | 28,519 | 26,244 | 24,349 | 22,565 | 20,869 | 19,172 | 17,514 | 15,927 | 14,464 | 13,087 | 11,834 | Mt CO2/yr |
| GCAM | NDC_LTT | 31,426 | 36,147 | 39,196 | 37,250 | 35,865 | 32,324 | 26,749 | 20,969 | 14,887 | 9,173 | 4,316 | 871.19 | 669.19 | 59.16 | -310.14 | -114.52 | -451.30 | -589.76 | -661.35 | -766.86 | Mt CO2/yr |
| GEMINI-E3 | Baseline | — | — | 34,738 | 34,934 | 38,132 | 42,037 | 44,008 | 45,532 | 46,413 | 46,713 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| GEMINI-E3 | CP_EI | — | — | 34,738 | 34,934 | 34,187 | 34,431 | 32,849 | 31,313 | 29,460 | 27,388 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| GEMINI-E3 | PR_Baseline | — | — | 34,859 | 37,813 | 41,214 | 44,462 | 46,876 | 48,984 | 50,715 | 52,211 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| MUSE | baseline | — | — | 36,386 | 37,813 | 37,955 | 35,891 | 34,436 | 33,696 | 32,387 | 31,448 | 30,979 | 31,285 | 32,420 | 33,030 | 34,518 | 35,378 | 36,141 | 36,237 | 37,010 | 37,729 | Mt CO2/yr |
| MUSE | CP_EI | — | — | 36,381 | 36,003 | 35,319 | 33,276 | 30,502 | 27,906 | 26,115 | 23,856 | 22,850 | 21,458 | 20,621 | 19,791 | 18,914 | 18,091 | 18,210 | 17,693 | 18,037 | 18,122 | Mt CO2/yr |
| MUSE | NDC_EI | — | — | 36,381 | 35,946 | 34,574 | 31,089 | 27,917 | 24,700 | 22,635 | 19,999 | 18,772 | 17,549 | 17,234 | 16,328 | 15,290 | 14,893 | 14,669 | 14,811 | 15,232 | 15,044 | Mt CO2/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q159"></a>

## 159. show Emissions|CO2 for World under Policy

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.47 s<br>
Conversation: `conv-008`

**Answer**

I could not find data for `Emissions|CO2` in `World` under `Policy`.

Reason: `Emissions|CO2` in `World` is not reported under `Policy`.

Closest valid options:

1. scenario `Baseline` (baseline)

2. scenario `Baseline|Harmonized` (baseline)

3. scenario `CP_EI`

Reply with `1`, `2`, or `3`, or type the scenario you want.

<a id="q160"></a>

## 160. use the first scenario

Category: data and scenarios<br>
Route: `data_query` (clarification selection; resolved)<br>
Qwen generations: 0<br>
Time: 0.51 s<br>
Conversation: `conv-008`

**Answer**

Showing results from the study **Where is the world headed?**. Also available in: Post-Glasgow targets; EU path to net zero; Global impacts of NDCs. Name another study to switch.

### Emissions|CO2 in World

Scope: scenario `multiple`, model `multiple`, years `2005-2100`
Unit: `Mt CO2/yr`

Answer:
In 2050, Emissions|CO2 in World ranges from 29,678 (MUSE, PR_Baseline) to 52,211 Mt CO2/yr (GEMINI-E3, PR_Baseline) across 8 series; median 43,333. Median change since 2005: +47%.

Summary: 8 series across 4 models and 4 scenarios.
Series are ordered by model and scenario.

| Model | Scenario | 2005 | 2010 | 2015 | 2020 | 2025 | 2030 | 2035 | 2040 | 2045 | 2050 | 2055 | 2060 | 2065 | 2070 | 2075 | 2080 | 2085 | 2090 | 2095 | 2100 | Unit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E3ME | PR_Baseline | — | 31,521 | 34,433 | 34,415 | 35,386 | 36,934 | 38,149 | 38,757 | 39,071 | 40,393 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| E3ME | Unharmonised baseline | — | 31,521 | 34,433 | 34,430 | 35,463 | 37,096 | 38,457 | 39,232 | 39,600 | 41,071 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| GCAM | Baseline | 31,426 | 36,147 | 39,196 | 38,300 | 40,654 | 42,107 | 43,979 | 45,315 | 45,465 | 45,595 | 46,708 | 48,206 | 49,791 | 51,521 | 53,322 | 54,770 | 56,170 | 57,563 | 58,673 | 60,068 | Mt CO2/yr |
| GCAM | PR_Baseline | 31,426 | 36,147 | 39,196 | 39,429 | 41,308 | 42,663 | 44,466 | 45,961 | 46,635 | 46,979 | 48,176 | 49,647 | 51,430 | 53,411 | 54,919 | 56,024 | 57,552 | 59,032 | 60,469 | 62,101 | Mt CO2/yr |
| GEMINI-E3 | Baseline | — | — | 34,738 | 34,934 | 38,132 | 42,037 | 44,008 | 45,532 | 46,413 | 46,713 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| GEMINI-E3 | PR_Baseline | — | — | 34,859 | 37,813 | 41,214 | 44,462 | 46,876 | 48,984 | 50,715 | 52,211 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| MUSE | baseline | — | — | 36,386 | 37,813 | 37,955 | 35,891 | 34,436 | 33,696 | 32,387 | 31,448 | 30,979 | 31,285 | 32,420 | 33,030 | 34,518 | 35,378 | 36,141 | 36,237 | 37,010 | 37,729 | Mt CO2/yr |
| MUSE | PR_Baseline | — | — | 36,431 | 38,836 | 40,158 | 41,521 | 38,898 | 36,732 | 33,730 | 29,678 | 26,902 | 24,364 | 23,828 | 24,252 | 24,392 | 24,597 | 24,559 | 24,441 | 24,608 | 24,998 | Mt CO2/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q161"></a>

## 161. oil demand for EU

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.76 s<br>
Conversation: `conv-009`

**Answer**

Interpretation: the available liquid-fuel demand indicator is broader than oil and may include other liquid fuels. It is not an oil-only estimate.

Showing results from the study **Where is the world headed?**. Also available in: Where is the EU headed?; Post-Glasgow targets. Name another study to switch.

### Final Energy|Liquids in EU

Scope: scenario `multiple`, model `multiple`, years `2005-2100`
Unit: `EJ/yr`

Answer:
In 2040, Final Energy|Liquids in EU ranges from 11.18 (TIAM_Grantham, PR_NDC_CP) to 27.82 EJ/yr (ICES-XPS, PR_Baseline) across 43 series; median 16.55. Median change since 2005: -40%.

Summary: 43 series across 7 models and 8 scenarios.
Showing 12 of 43 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 31 remaining series.

| Model | Scenario | 2005 | 2010 | 2015 | 2020 | 2025 | 2030 | 2035 | 2040 | 2045 | 2050 | 2055 | 2060 | 2065 | 2070 | 2075 | 2080 | 2085 | 2090 | 2095 | 2100 | Unit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | PR_Baseline | 27.62 | 25.08 | 23.61 | 22.84 | 22.50 | 21.97 | 21.30 | 20.48 | 19.55 | — | — | — | — | — | — | — | — | — | — | — | EJ/yr |
| 42 | PR_CurPol_EI | 27.62 | 25.08 | 23.61 | 22.61 | 21.45 | 19.53 | 18.05 | 16.51 | 14.91 | — | — | — | — | — | — | — | — | — | — | — | EJ/yr |
| E3ME | PR_Baseline | — | 24.45 | 24.49 | 24.33 | 23.48 | 22.50 | 21.82 | 20.82 | 19.45 | 17.36 | — | — | — | — | — | — | — | — | — | — | EJ/yr |
| E3ME | PR_CurPol_CP | — | 24.45 | 24.49 | 24.33 | 23.48 | 22.50 | 21.82 | 20.82 | 19.46 | 17.37 | — | — | — | — | — | — | — | — | — | — | EJ/yr |
| GCAM | PR_Baseline | 23.14 | 21.67 | 20.37 | 20.12 | 19.26 | 17.24 | 17.27 | 16.83 | 16.69 | 16.87 | 17.30 | 17.86 | 18.59 | 19.28 | 20.02 | 20.84 | 21.79 | 22.85 | 23.96 | 25.07 | EJ/yr |
| GCAM | PR_CurPol_CP | 23.14 | 21.67 | 20.37 | 19.29 | 18.29 | 16.16 | 15.28 | 13.96 | 13.59 | 13.65 | 13.78 | 14.01 | 14.20 | 14.32 | 14.50 | 14.75 | 15.04 | 15.36 | 15.65 | 15.91 | EJ/yr |
| GEMINI-E3 | PR_Baseline | — | — | 22.52 | 21.99 | 21.66 | 21.23 | 21.07 | 20.84 | 19.80 | 18.97 | — | — | — | — | — | — | — | — | — | — | EJ/yr |
| GEMINI-E3 | PR_CurPol_CP | — | — | 22.52 | 21.97 | 21.46 | 20.57 | 20.47 | 20.31 | 19.35 | 18.57 | — | — | — | — | — | — | — | — | — | — | EJ/yr |
| ICES-XPS | PR_Baseline | — | — | 33.52 | 39.34 | 33.59 | 29.95 | 28.94 | 27.82 | 27.31 | 26.50 | — | — | — | — | — | — | — | — | — | — | EJ/yr |
| ICES-XPS | PR_CurPol_CP | — | — | 33.52 | 38.34 | 31.19 | 25.45 | 24.68 | 23.74 | 23.21 | 22.36 | — | — | — | — | — | — | — | — | — | — | EJ/yr |
| MUSE | PR_Baseline | — | — | 19.70 | 19.38 | 18.36 | 17.26 | 16.17 | 14.45 | 11.17 | 8.36 | 7.21 | 6.28 | 5.82 | 5.57 | 5.51 | 5.36 | 5.27 | 5.20 | 5.15 | 4.84 | EJ/yr |
| TIAM_Grantham | PR_NDC_CP | 23.49 | 22.26 | — | 20.79 | — | 13.77 | — | 11.18 | — | 8.11 | — | 4.70 | — | 3.01 | — | 2.12 | — | 1.77 | — | 1.46 | EJ/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q162"></a>

## 162. which models are available

Category: catalogue and coverage<br>
Route: `data_query` (model availability request)<br>
Qwen generations: 0<br>
Time: 0.12 s<br>
Conversation: `conv-009`

**Answer**

There are 75 models available in the IAM PARIS model catalogue; 33 of them have loaded results. Models with the most results: MUSE, E3ME, GCAM-PR 5.3, NEMESIS 5.1, E4SMA-EU-TIMES 1.0, GCAM 7.0, GCAM, GEM-E3. Ask `tell me about [model name]` for details, or filter by topic, e.g. `which models cover transport`.

<a id="q163"></a>

## 163. 1

Category: data and scenarios<br>
Route: `data_query` (number without active clarification)<br>
Qwen generations: 0<br>
Time: 0.06 s<br>
Conversation: `conv-009`

**Answer**

I don't have an active numbered choice right now. Reply with a full variable, region, or scenario so I can continue.

<a id="q164"></a>

## 164. custom analysis service

Category: data and scenarios<br>
Route: `general_qa` (explicit site navigation request)<br>
Qwen generations: 0<br>
Time: 0.07 s<br>
Conversation: `conv-010`

**Answer**

Use these IAM PARIS links for this request: [Configurable Analysis](https://iamparis.eu/analysis).

<a id="q165"></a>

## 165. contact IAM PARIS team

Category: navigation and links<br>
Route: `general_qa` (explicit site navigation request)<br>
Qwen generations: 0<br>
Time: 0.05 s<br>
Conversation: `conv-010`

**Answer**

Use these IAM PARIS links for this request: [Contact](https://iamparis.eu/contact).

<a id="q166"></a>

## 166. Tell me about GCAM.

Category: model descriptions<br>
Route: `model_explanation` (model information request)<br>
Qwen generations: 0<br>
Time: 0.26 s<br>
Conversation: `conv-011`

**Answer**

### GCAM

Description:
GCAM is a dynamic-recursive model with technology-rich representations of the economy, energy sector, land use and water linked to a climate model that can be used to explore climate change mitigation policies including carbon taxes, carbon trading, regulations and accelerated deployment of energy technology. Regional population and labour productivity growth assumptions drive the energy, land-use and water systems employing numerous technology options to produce, transform and provide energy services, as well as to produce agriculture and forest products, and to determine land use and land use cover.

Related model documentation:
- [IAM PARIS model page](https://iamparis.eu/models/30)

<a id="q167"></a>

## 167. Who develops this model?

Category: model descriptions<br>
Route: `model_explanation` (singular-model qualitative follow-up)<br>
Qwen generations: 0<br>
Time: 0.06 s<br>
Conversation: `conv-011`

**Answer**

### GCAM

`GCAM` is developed by Joint Global Change Research Institute.

Related model documentation:
- [IAM PARIS model page](https://iamparis.eu/models/30)

<a id="q168"></a>

## 168. What limitations should I keep in mind?

Category: model descriptions<br>
Route: `model_explanation` (singular-model qualitative follow-up)<br>
Qwen generations: 0<br>
Time: 0.06 s<br>
Conversation: `conv-011`

**Answer**

### GCAM

Interpretation notes:
- model assumptions vary by scenario and input dataset
- local IAM PARIS metadata may not include detailed assumptions

Related model documentation:
- [IAM PARIS model page](https://iamparis.eu/models/30)

<a id="q169"></a>

## 169. Which regions does it cover?

Category: catalogue and coverage<br>
Route: `data_query` (model-scoped catalogue follow-up)<br>
Qwen generations: 0<br>
Time: 1.12 s<br>
Conversation: `conv-011`

**Answer**

Model `GCAM` has these regions: ARG, Africa_Eastern, Africa_Northern, Africa_Southern, Africa_Western, Australia_NZ, BRA, CAN, CHN, COL, Central America and Caribbean, Central Asia, EU, EU-12, EU-15 and 26 more. Runtime labels included: GCAM 7.0, GCAM 7_0, gcam.

Showing 15 of 41. Say `show all regions for GCAM` if you need the full list.

<a id="q170"></a>

## 170. Compare GCAM and WITCH as models.

Category: model descriptions<br>
Route: `model_explanation` (multi-model metadata comparison)<br>
Qwen generations: 0<br>
Time: 0.07 s<br>
Conversation: `conv-012`

**Answer**

### GCAM vs WITCH

**GCAM**
- Developer: Joint Global Change Research Institute
- Systems/sectors: energy, economy, land, water, climate
- Common uses: emissions and energy pathway analysis; climate policy scenarios; land and resource interactions; regional mitigation comparisons

**WITCH**
- Systems/sectors: economy, energy, climate policy, technology innovation
- Common uses: regional mitigation and burden-sharing analysis; carbon pricing and policy assessment; technology innovation and diffusion scenarios; long-term climate policy pathways

Shared coverage: economy, energy.
Distinctive in GCAM: climate, land, water.
Distinctive in WITCH: climate policy, technology innovation.

[Browse IAM PARIS model documentation](https://iamparis.eu/models)

<a id="q171"></a>

## 171. How do these two differ in methodology?

Category: model descriptions<br>
Route: `model_explanation` (referenced-model qualitative comparison)<br>
Qwen generations: 0<br>
Time: 0.13 s<br>
Conversation: `conv-012`

**Answer**

### GCAM vs WITCH

**GCAM**
- Methodology/model type: Partial equilibrium

**WITCH**
- Methodology/model type: WITCH is an integrated assessment model focused on the interaction between the economy, energy system, climate policy and technological change. It is often used for regional mitigation strategies, policy design and technology innovation analysis.

Comparison focus: method.

[Browse IAM PARIS model documentation](https://iamparis.eu/models)

<a id="q172"></a>

## 172. Give me documentation links for both models

Category: model descriptions<br>
Route: `model_explanation` (referenced model links)<br>
Qwen generations: 0<br>
Time: 0.04 s<br>
Conversation: `conv-012`

**Answer**

IAM PARIS model links for: GCAM, WITCH.

<a id="q173"></a>

## 173. show Emissions|CO2 for India under NDC_EI from 2030 to 2050

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.29 s<br>
Conversation: `conv-013`

**Answer**

Showing results from the study **Post-Glasgow targets**. Also available in: Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU; Global impacts of NDCs; Tech-constrained EU pathways to net zero. Name another study to switch.

### Emissions|CO2 in IND

Scope: scenario `NDC_EI`, model `multiple`, years `2030-2050`
Unit: `Mt CO2/yr`

Answer:
In 2050, Emissions|CO2 in IND ranges from 1,310 (MUSE, NDC_EI) to 3,197 Mt CO2/yr (GEMINI-E3 7.0, NDC_EI) across 3 series; median 1,740. Median change since 2030: -45%.

**GCAM-PR 5.3 - NDC_EI**
| Year | Value | Unit |
|------|-------|------|
| 2030 | 3,135 | Mt CO2/yr |
| 2035 | 2,994 | Mt CO2/yr |
| 2040 | 2,656 | Mt CO2/yr |
| 2045 | 2,211 | Mt CO2/yr |
| 2050 | 1,740 | Mt CO2/yr |

**GEMINI-E3 7.0 - NDC_EI**
| Year | Value | Unit |
|------|-------|------|
| 2030 | 3,316 | Mt CO2/yr |
| 2035 | 3,431 | Mt CO2/yr |
| 2040 | 3,518 | Mt CO2/yr |
| 2045 | 3,361 | Mt CO2/yr |
| 2050 | 3,197 | Mt CO2/yr |

**MUSE - NDC_EI**
| Year | Value | Unit |
|------|-------|------|
| 2030 | 2,664 | Mt CO2/yr |
| 2035 | 2,436 | Mt CO2/yr |
| 2040 | 2,020 | Mt CO2/yr |
| 2045 | 1,692 | Mt CO2/yr |
| 2050 | 1,310 | Mt CO2/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/post-glasgow-targets/graphs).

<a id="q174"></a>

## 174. now for CHN

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.53 s<br>
Conversation: `conv-013`

**Answer**

### Emissions|CO2 in CHN

Scope: scenario `NDC_EI`, model `multiple`, years `2030-2050`
Unit: `Mt CO2/yr`

Answer:
In 2050, Emissions|CO2 in CHN ranges from 4,858 (MUSE, NDC_EI) to 5,882 Mt CO2/yr (GCAM-PR 5.3, NDC_EI) across 2 series; median 5,370. Median change since 2030: -52%.

**GCAM-PR 5.3 - NDC_EI**
| Year | Value | Unit |
|------|-------|------|
| 2030 | 11,823 | Mt CO2/yr |
| 2035 | 10,506 | Mt CO2/yr |
| 2040 | 8,941 | Mt CO2/yr |
| 2045 | 7,324 | Mt CO2/yr |
| 2050 | 5,882 | Mt CO2/yr |

**MUSE - NDC_EI**
| Year | Value | Unit |
|------|-------|------|
| 2030 | 10,474 | Mt CO2/yr |
| 2035 | 9,154 | Mt CO2/yr |
| 2040 | 7,759 | Mt CO2/yr |
| 2045 | 6,627 | Mt CO2/yr |
| 2050 | 4,858 | Mt CO2/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/post-glasgow-targets/graphs).

Reply with a scenario, model, region, or year to narrow the answer.

<a id="q175"></a>

## 175. until 2040

Category: data and scenarios<br>
Route: `data_query` (year change for previous question)<br>
Qwen generations: 0<br>
Time: 0.22 s<br>
Conversation: `conv-013`

**Answer**

### Emissions|CO2 in CHN

Scope: scenario `NDC_EI`, model `multiple`, years `first available-2040`
Unit: `Mt CO2/yr`

Answer:
In 2040, Emissions|CO2 in CHN ranges from 7,759 (MUSE, NDC_EI) to 8,941 Mt CO2/yr (GCAM-PR 5.3, NDC_EI) across 2 series; median 8,350.

**GCAM-PR 5.3 - NDC_EI**
| Year | Value | Unit |
|------|-------|------|
| 2005 | 6,734 | Mt CO2/yr |
| 2010 | 9,611 | Mt CO2/yr |
| 2015 | 11,226 | Mt CO2/yr |
| 2020 | 12,133 | Mt CO2/yr |
| 2025 | 12,371 | Mt CO2/yr |
| 2030 | 11,823 | Mt CO2/yr |
| 2035 | 10,506 | Mt CO2/yr |
| 2040 | 8,941 | Mt CO2/yr |

**MUSE - NDC_EI**
| Year | Value | Unit |
|------|-------|------|
| 2015 | 11,771 | Mt CO2/yr |
| 2020 | 11,564 | Mt CO2/yr |
| 2025 | 11,356 | Mt CO2/yr |
| 2030 | 10,474 | Mt CO2/yr |
| 2035 | 9,154 | Mt CO2/yr |
| 2040 | 7,759 | Mt CO2/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/post-glasgow-targets/graphs).

<a id="q176"></a>

## 176. plot it

Category: charts and comparisons<br>
Route: `general_qa` (charts delegated to data explorer)<br>
Qwen generations: 0<br>
Time: 0.01 s<br>
Conversation: `conv-013`

**Answer**

I do not generate figures in the chat. Please use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/post-glasgow-targets/graphs) to explore, filter, and plot the data. Ask about a specific study first if you would like its direct explorer link.

<a id="q177"></a>

## 177. plot Primary Energy|Coal for CHN under PR_CurPol_CP from 2030 to 2050

Category: charts and comparisons<br>
Route: `general_qa` (charts delegated to data explorer)<br>
Qwen generations: 0<br>
Time: 0.00 s<br>
Conversation: `conv-014`

**Answer**

I do not generate figures in the chat. Please use the [IAM PARIS data explorer](https://iamparis.eu/results) to explore, filter, and plot the data. Ask about a specific study first if you would like its direct explorer link.

<a id="q178"></a>

## 178. change only the variable to Primary Energy|Gas

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.50 s<br>
Conversation: `conv-014`

**Answer**

Showing results from the study **Where is the world headed?**. Also available in: EU path to net zero; Post-Glasgow targets; Global impacts of NDCs; Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU; Tech-constrained EU pathways to net zero; Behavioural changes for climate policy; Where is the EU headed?; People-centered national energy systems; Decarbonization potentials of energy citizens; Enhancing renewable energy system metrics. Name another study to switch.

### Primary Energy|Gas in World

Scope: scenario `multiple`, model `multiple`, years `2005-2100`
Unit: `EJ/yr`

Answer:
In 2045, Primary Energy|Gas in World ranges from 113.57 (ICES-XPS, PR_NDC_EI) to 227.49 EJ/yr (GEMINI-E3, PR_Baseline) across 36 series; median 177.95. Median change since 2005: +93%.

Summary: 36 series across 6 models and 8 scenarios.
Showing 12 of 36 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 24 remaining series.

| Model | Scenario | 2005 | 2010 | 2015 | 2020 | 2025 | 2030 | 2035 | 2040 | 2045 | 2050 | 2055 | 2060 | 2065 | 2070 | 2075 | 2080 | 2085 | 2090 | 2095 | 2100 | Unit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | PR_Baseline | 98.90 | 114.57 | 123.61 | 140.31 | 153.17 | 167.16 | 181.69 | 195.42 | 209.92 | — | — | — | — | — | — | — | — | — | — | — | EJ/yr |
| 42 | PR_CurPol_EI | 98.90 | 114.57 | 123.61 | 140.23 | 155.12 | 169.12 | 183.33 | 196.44 | 209.95 | — | — | — | — | — | — | — | — | — | — | — | EJ/yr |
| E3ME | PR_Baseline | — | 115.01 | 123.34 | 137.31 | 147.94 | 157.82 | 165.08 | 171.55 | 178.23 | 186.91 | — | — | — | — | — | — | — | — | — | — | EJ/yr |
| E3ME | PR_CurPol_CP | — | 115.01 | 123.34 | 137.31 | 147.94 | 157.82 | 164.85 | 171.19 | 177.67 | 186.20 | — | — | — | — | — | — | — | — | — | — | EJ/yr |
| GCAM | PR_Baseline | 102.47 | 119.24 | 127.94 | 140.51 | 151.01 | 165.35 | 177.53 | 191.10 | 204.66 | 211.41 | 223.52 | 238.56 | 253.38 | 270.25 | 288.75 | 306.31 | 322.67 | 332.02 | 338.27 | 344.65 | EJ/yr |
| GCAM | PR_CurPol_CP | 102.47 | 119.23 | 127.94 | 138.03 | 149.30 | 164.89 | 176.80 | 188.94 | 199.98 | 206.25 | 217.92 | 231.59 | 245.47 | 259.61 | 274.59 | 288.97 | 306.02 | 316.05 | 322.35 | 328.88 | EJ/yr |
| GEMINI-E3 | PR_Baseline | — | — | 132.25 | 146.42 | 164.91 | 183.95 | 197.47 | 210.72 | 227.49 | 240.91 | — | — | — | — | — | — | — | — | — | — | EJ/yr |
| GEMINI-E3 | PR_CurPol_CP | — | — | 132.25 | 146.19 | 163.52 | 181.30 | 193.41 | 205.61 | 220.92 | 232.87 | — | — | — | — | — | — | — | — | — | — | EJ/yr |
| ICES-XPS | PR_Baseline | — | — | 118.37 | 114.01 | 127.31 | 137.54 | 150.67 | 159.00 | 165.59 | 167.40 | — | — | — | — | — | — | — | — | — | — | EJ/yr |
| ICES-XPS | PR_NDC_EI | — | — | 118.37 | 113.01 | 119.33 | 122.32 | 120.48 | 117.58 | 113.57 | 107.32 | — | — | — | — | — | — | — | — | — | — | EJ/yr |
| MUSE | PR_Baseline | — | — | 130.88 | 139.11 | 140.52 | 138.66 | 135.13 | 134.12 | 130.84 | 131.25 | 126.07 | 121.84 | 122.91 | 122.97 | 121.35 | 117.90 | 115.21 | 112.07 | 111.76 | 112.99 | EJ/yr |
| MUSE | PR_CurPol_CP | — | — | 131.24 | 141.96 | 142.44 | 143.44 | 136.71 | 135.11 | 137.28 | 136.31 | 136.09 | 130.62 | 131.30 | 130.75 | 132.46 | 132.05 | 132.24 | 134.47 | 136.76 | 139.43 | EJ/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q179"></a>

## 179. plot it as a bar chart

Category: charts and comparisons<br>
Route: `general_qa` (charts delegated to data explorer)<br>
Qwen generations: 0<br>
Time: 0.01 s<br>
Conversation: `conv-014`

**Answer**

I do not generate figures in the chat. Please use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs) to explore, filter, and plot the data. Ask about a specific study first if you would like its direct explorer link.

<a id="q180"></a>

## 180. show Emissions|CO2 for EU under PR_CurPol_CP in 2050

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.30 s<br>
Conversation: `conv-015`

**Answer**

### Emissions|CO2 in EU

Scope: scenario `PR_CurPol_CP`, model `multiple`, years `2050`
Unit: `Mt CO2/yr`

Answer:
In 2050, Emissions|CO2 in EU ranges from 1,118 (GCAM, PR_CurPol_CP) to 2,359 Mt CO2/yr (E3ME, PR_CurPol_CP) across 4 series; median 2,107.

**E3ME - PR_CurPol_CP**
| Year | Value | Unit |
|------|-------|------|
| 2050 | 2,359 | Mt CO2/yr |

**GCAM - PR_CurPol_CP**
| Year | Value | Unit |
|------|-------|------|
| 2050 | 1,118 | Mt CO2/yr |

**GEMINI-E3 - PR_CurPol_CP**
| Year | Value | Unit |
|------|-------|------|
| 2050 | 2,342 | Mt CO2/yr |

**MUSE - PR_CurPol_CP**
| Year | Value | Unit |
|------|-------|------|
| 2050 | 1,871 | Mt CO2/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q181"></a>

## 181. same for China

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.57 s<br>
Conversation: `conv-015`

**Answer**

### Emissions|CO2 in CHN

Scope: scenario `PR_CurPol_CP`, model `multiple`, years `2050`
Unit: `Mt CO2/yr`

Answer:
In 2050, Emissions|CO2 in CHN ranges from 6,538 (MUSE, PR_CurPol_CP) to 9,939 Mt CO2/yr (GEMINI-E3, PR_CurPol_CP) across 4 series; median 8,075.

**E3ME - PR_CurPol_CP**
| Year | Value | Unit |
|------|-------|------|
| 2050 | 7,420 | Mt CO2/yr |

**GCAM - PR_CurPol_CP**
| Year | Value | Unit |
|------|-------|------|
| 2050 | 8,729 | Mt CO2/yr |

**GEMINI-E3 - PR_CurPol_CP**
| Year | Value | Unit |
|------|-------|------|
| 2050 | 9,939 | Mt CO2/yr |

**MUSE - PR_CurPol_CP**
| Year | Value | Unit |
|------|-------|------|
| 2050 | 6,538 | Mt CO2/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

Reply with a scenario, model, region, or year to narrow the answer.

<a id="q182"></a>

## 182. plot the comparison between the two regions

Category: charts and comparisons<br>
Route: `general_qa` (charts delegated to data explorer)<br>
Qwen generations: 0<br>
Time: 0.01 s<br>
Conversation: `conv-015`

**Answer**

I do not generate figures in the chat. Please use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs) to explore, filter, and plot the data. Ask about a specific study first if you would like its direct explorer link.

<a id="q183"></a>

## 183. Open the IAM PARIS Models page.

Category: navigation and links<br>
Route: `general_qa` (explicit site navigation request)<br>
Qwen generations: 0<br>
Time: 0.05 s<br>
Conversation: `conv-016`

**Answer**

Use these IAM PARIS links for this request: [Models](https://iamparis.eu/models).

<a id="q184"></a>

## 184. Now open the model comparison page.

Category: navigation and links<br>
Route: `general_qa` (explicit site navigation request)<br>
Qwen generations: 0<br>
Time: 0.04 s<br>
Conversation: `conv-016`

**Answer**

Use these IAM PARIS links for this request: [Model Comparison](https://iamparis.eu/models/comparison).

<a id="q185"></a>

## 185. Take me to My Workspace.

Category: data and scenarios<br>
Route: `general_qa` (explicit site navigation request)<br>
Qwen generations: 0<br>
Time: 0.03 s<br>
Conversation: `conv-016`

**Answer**

Use these IAM PARIS links for this request: [My Workspace](https://iamparis.eu/scientific_module).

<a id="q186"></a>

## 186. Where can I find Aqueduct?

Category: navigation and links<br>
Route: `general_qa` (explicit site navigation request)<br>
Qwen generations: 0<br>
Time: 0.04 s<br>
Conversation: `conv-017`

**Answer**

Use these IAM PARIS links for this request: [Aqueduct](https://iamparis.eu/application_library/474).

<a id="q187"></a>

## 187. What about Climate Watch?

Category: data and scenarios<br>
Route: `general_qa` (general climate/policy question)<br>
Qwen generations: 3<br>
Time: 1.87 s<br>
Conversation: `conv-017`

**Answer**

The Climate Watch is a tool used to assess the impact of climate change on various sectors, such as energy and finance, based on integrated assessment models. It helps identify potential risks and opportunities related to climate change.

<a id="q188"></a>

## 188. Open the Application Library.

Category: navigation and links<br>
Route: `general_qa` (explicit site navigation request)<br>
Qwen generations: 0<br>
Time: 0.06 s<br>
Conversation: `conv-017`

**Answer**

Use these IAM PARIS links for this request: [Application Library](https://iamparis.eu/application_library).

<a id="q189"></a>

## 189. electricity for India in 2050

Category: data and scenarios<br>
Route: `data_query` (low confidence entity clarification)<br>
Qwen generations: 0<br>
Time: 0.55 s<br>
Conversation: `conv-018`

**Answer**

I found several possible variables, but none has enough evidence for automatic selection. Choose the variable: 1. `Secondary Energy|Electricity` (electricity output) 2. `Final Energy|Electricity` (final energy use, electricity) 3. `Capacity|Electricity` (power capacity, electricity) Reply with a number (1-3), or `yes` for option 1.

<a id="q190"></a>

## 190. no

Category: data and scenarios<br>
Route: `data_query` (clarification rejection; additional clarification required)<br>
Qwen generations: 0<br>
Time: 0.32 s<br>
Conversation: `conv-018`

**Answer**

Okay, here are the next closest options. Choose the variable: 1. `Final Energy|Electricity` (final energy use, electricity) 2. `Capacity|Electricity` (power capacity, electricity) Reply with a number (1-2), or `yes` for option 1.

<a id="q191"></a>

## 191. 1

Category: data and scenarios<br>
Route: `data_query` (clarification selection; resolved)<br>
Qwen generations: 0<br>
Time: 0.29 s<br>
Conversation: `conv-018`

**Answer**

Showing results from the study **Where is the world headed?**. Also available in: Post-Glasgow targets; Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU; Tech-constrained EU pathways to net zero; Global impacts of NDCs; Implications of EU steel industry relocation. Name another study to switch.

### Final Energy|Electricity in IND

Scope: scenario `multiple`, model `multiple`, years `2050`
Unit: `EJ/yr`

Answer:
In 2050, Final Energy|Electricity in IND ranges from 7.66 (MUSE, PR_Baseline) to 26.13 EJ/yr (GCAM, PR_Baseline) across 40 series; median 19.21.

Summary: 40 series across 6 models and 8 scenarios.
Showing 12 of 40 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 28 remaining series.

| Model | Scenario | 2050 | Unit |
|---|---|---|---|
| E3ME | PR_Baseline | 20.59 | EJ/yr |
| E3ME | PR_CurPol_CP | 20.49 | EJ/yr |
| GCAM | PR_Baseline | 26.13 | EJ/yr |
| GCAM | PR_CurPol_CP | 24.21 | EJ/yr |
| GEMINI-E3 | PR_Baseline | 18.66 | EJ/yr |
| GEMINI-E3 | PR_CurPol_CP | 19.46 | EJ/yr |
| ICES-XPS | PR_Baseline | 24.93 | EJ/yr |
| ICES-XPS | PR_CurPol_CP | 24.42 | EJ/yr |
| MUSE | PR_Baseline | 7.66 | EJ/yr |
| MUSE | PR_CurPol_CP | 8.10 | EJ/yr |
| TIAM_Grantham | PR_Baseline | 12.65 | EJ/yr |
| TIAM_Grantham | PR_CurPol_CP | 11.46 | EJ/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q192"></a>

## 192. plot it

Category: charts and comparisons<br>
Route: `general_qa` (charts delegated to data explorer)<br>
Qwen generations: 0<br>
Time: 0.01 s<br>
Conversation: `conv-018`

**Answer**

I do not generate figures in the chat. Please use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs) to explore, filter, and plot the data. Ask about a specific study first if you would like its direct explorer link.

<a id="q193"></a>

## 193. emisions for europ

Category: data and scenarios<br>
Route: `data_query` (extracted data entities)<br>
Qwen generations: 0<br>
Time: 0.46 s<br>
Conversation: `conv-019`

**Answer**

Showing results from the study **Where is the world headed?**. Also available in: Where is the EU headed?; Post-Glasgow targets. Name another study to switch.

Interpretation: I treated bare **emissions** as `Emissions|CO2` (carbon-dioxide emissions).

### Emissions|CO2 in EU

Scope: scenario `multiple`, model `multiple`, years `2005-2100`
Unit: `Mt CO2/yr`

Answer:
In 2050, Emissions|CO2 in EU ranges from 1,024 (GCAM, PR_CurPol_CPO) to 3,429 Mt CO2/yr (GCAM, PR_Baseline) across 26 series; median 2,002. Median change since 2005: -71%.

Summary: 26 series across 4 models and 8 scenarios.
Showing 12 of 26 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 14 remaining series.

| Model | Scenario | 2005 | 2010 | 2015 | 2020 | 2025 | 2030 | 2035 | 2040 | 2045 | 2050 | 2055 | 2060 | 2065 | 2070 | 2075 | 2080 | 2085 | 2090 | 2095 | 2100 | Unit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E3ME | PR_Baseline | — | 4,304 | 3,999 | 3,444 | 3,188 | 3,033 | 2,864 | 2,692 | 2,516 | 2,323 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| E3ME | PR_CurPol_CP | — | 4,304 | 3,999 | 3,444 | 3,188 | 3,033 | 2,869 | 2,705 | 2,539 | 2,359 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| E3ME | PR_CurPol_EI | — | 4,304 | 3,999 | 3,445 | 3,188 | 3,032 | 2,919 | 2,827 | 2,737 | 2,633 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| GCAM | PR_Baseline | 4,190 | 3,790 | 3,490 | 3,543 | 3,524 | 3,454 | 3,524 | 3,549 | 3,509 | 3,429 | 3,476 | 3,589 | 3,712 | 3,829 | 3,926 | 4,029 | 4,168 | 4,311 | 4,472 | 4,672 | Mt CO2/yr |
| GCAM | PR_CurPol_CP | 4,190 | 3,790 | 3,490 | 3,115 | 2,687 | 2,257 | 2,112 | 1,836 | 1,485 | 1,118 | 962.69 | 912.20 | 870.84 | 863.75 | 839.93 | 771.80 | 656.18 | 503.01 | 281.84 | 10.13 | Mt CO2/yr |
| GCAM | PR_CurPol_CPO | 4,190 | 3,790 | 3,490 | 3,094 | 2,697 | 2,267 | 2,008 | 1,723 | 1,381 | 1,024 | 873.57 | 824.73 | 781.65 | 770.74 | 720.35 | 598.03 | 446.47 | 186.10 | -169.61 | -651.08 | Mt CO2/yr |
| GCAM | PR_CurPol_EI | 4,190 | 3,790 | 3,490 | 3,115 | 2,688 | 2,257 | 1,936 | 1,625 | 1,355 | 1,121 | 921.10 | 724.95 | 558.89 | 407.39 | 280.07 | 166.91 | 64.24 | -27.40 | -106.40 | -175.92 | Mt CO2/yr |
| GEMINI-E3 | PR_Baseline | — | — | 3,411 | 3,201 | 3,230 | 3,235 | 3,150 | 3,048 | 2,941 | 2,829 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| GEMINI-E3 | PR_CurPol_CP | — | — | 3,411 | 3,197 | 3,006 | 2,733 | 2,630 | 2,545 | 2,441 | 2,342 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| GEMINI-E3 | PR_CurPol_CPO | — | — | 3,411 | 3,197 | 3,006 | 2,731 | 2,626 | 2,541 | 2,437 | 2,337 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| MUSE | PR_Baseline | — | — | 3,141 | 3,106 | 3,076 | 2,994 | 2,842 | 2,701 | 2,384 | 2,074 | 1,838 | 1,664 | 1,588 | 1,543 | 1,400 | 1,279 | 1,101 | 1,034 | 977.08 | 947.41 | Mt CO2/yr |
| MUSE | PR_CurPol_CP | — | — | 3,118 | 3,018 | 2,956 | 2,740 | 2,581 | 2,435 | 2,155 | 1,871 | 1,641 | 1,482 | 1,414 | 1,365 | 1,237 | 1,116 | 975.01 | 910.90 | 844.23 | 817.09 | Mt CO2/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q194"></a>

## 194. same for World in 2050

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.48 s<br>
Conversation: `conv-019`

**Answer**

### Emissions|CO2 in World

Scope: scenario `multiple`, model `multiple`, years `2050`
Unit: `Mt CO2/yr`

Answer:
In 2050, Emissions|CO2 in World ranges from 9,173 (GCAM, NDC_LTT) to 52,211 Mt CO2/yr (GEMINI-E3, PR_Baseline) across 38 series; median 33,062.

Summary: 38 series across 4 models and 13 scenarios.
Showing 12 of 38 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 26 remaining series.

| Model | Scenario | 2050 | Unit |
|---|---|---|---|
| E3ME | PR_Baseline | 40,393 | Mt CO2/yr |
| E3ME | PR_CurPol_CP | 40,054 | Mt CO2/yr |
| E3ME | PR_CurPol_EI | 39,239 | Mt CO2/yr |
| GCAM | Baseline | 45,595 | Mt CO2/yr |
| GCAM | CP_EI | 28,519 | Mt CO2/yr |
| GCAM | NDC_LTT | 9,173 | Mt CO2/yr |
| GEMINI-E3 | Baseline | 46,713 | Mt CO2/yr |
| GEMINI-E3 | CP_EI | 27,388 | Mt CO2/yr |
| GEMINI-E3 | PR_Baseline | 52,211 | Mt CO2/yr |
| MUSE | baseline | 31,448 | Mt CO2/yr |
| MUSE | CP_EI | 23,856 | Mt CO2/yr |
| MUSE | NDC_EI | 19,999 | Mt CO2/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q195"></a>

## 195. plot it

Category: charts and comparisons<br>
Route: `general_qa` (charts delegated to data explorer)<br>
Qwen generations: 0<br>
Time: 0.00 s<br>
Conversation: `conv-019`

**Answer**

I do not generate figures in the chat. Please use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs) to explore, filter, and plot the data. Ask about a specific study first if you would like its direct explorer link.

<a id="q196"></a>

## 196. Which models contributed to it?

Category: catalogue and coverage<br>
Route: `data_query` (scoped availability request)<br>
Qwen generations: 0<br>
Time: 0.13 s<br>
Conversation: `conv-019`

**Answer**

### Availability for Emissions|CO2

**Models (0):** none

Recorded unit: `Mt CO2/yr`.

<a id="q197"></a>

## 197. show Emissions|CO2 for World under Baseline in 2050

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.33 s<br>
Conversation: `conv-020`

**Answer**

Showing results from the study **Where is the world headed?**. Also available in: Post-Glasgow targets; EU path to net zero; Global impacts of NDCs. Name another study to switch.

### Emissions|CO2 in World

Scope: scenario `multiple`, model `multiple`, years `2050`
Unit: `Mt CO2/yr`

Answer:
In 2050, Emissions|CO2 in World ranges from 29,678 (MUSE, PR_Baseline) to 52,211 Mt CO2/yr (GEMINI-E3, PR_Baseline) across 8 series; median 43,333.

Summary: 8 series across 4 models and 4 scenarios.
Series are ordered by model and scenario.

| Model | Scenario | 2050 | Unit |
|---|---|---|---|
| E3ME | PR_Baseline | 40,393 | Mt CO2/yr |
| E3ME | Unharmonised baseline | 41,071 | Mt CO2/yr |
| GCAM | Baseline | 45,595 | Mt CO2/yr |
| GCAM | PR_Baseline | 46,979 | Mt CO2/yr |
| GEMINI-E3 | Baseline | 46,713 | Mt CO2/yr |
| GEMINI-E3 | PR_Baseline | 52,211 | Mt CO2/yr |
| MUSE | baseline | 31,448 | Mt CO2/yr |
| MUSE | PR_Baseline | 29,678 | Mt CO2/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q198"></a>

## 198. under Policy instead

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.61 s<br>
Conversation: `conv-020`

**Answer**

No data found for **Emissions|CO2** in region `World` under scenario `Policy`. It is reported for `ARG`, `AUS`, `AUT`, `Africa_Eastern`, `Africa_Northern` and 90 more. Choose the variable: 1. `Emissions|CO2|Heat` (CO2 emissions) 2. `Emissions|CO2|AFOLU` (CO2 emissions, land use) 3. `Emissions|CO2|Gases` (CO2 emissions, gas) Reply with a number (1-3), or `yes` for option 1.

<a id="q199"></a>

## 199. plot the comparison between the two scenarios

Category: charts and comparisons<br>
Route: `general_qa` (charts delegated to data explorer)<br>
Qwen generations: 0<br>
Time: 0.02 s<br>
Conversation: `conv-020`

**Answer**

I do not generate figures in the chat. Please use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs) to explore, filter, and plot the data. Ask about a specific study first if you would like its direct explorer link.

<a id="q200"></a>

## 200. under PR_CurPol_CP instead

Category: data and scenarios<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.48 s<br>
Conversation: `conv-020`

**Answer**

### Emissions|CO2 in World

Scope: scenario `PR_CurPol_CP`, model `multiple`, years `2050`
Unit: `Mt CO2/yr`

Answer:
In 2050, Emissions|CO2 in World ranges from 28,295 (MUSE, PR_CurPol_CP) to 44,164 Mt CO2/yr (GEMINI-E3, PR_CurPol_CP) across 4 series; median 39,220.

**E3ME - PR_CurPol_CP**
| Year | Value | Unit |
|------|-------|------|
| 2050 | 40,054 | Mt CO2/yr |

**GCAM - PR_CurPol_CP**
| Year | Value | Unit |
|------|-------|------|
| 2050 | 38,385 | Mt CO2/yr |

**GEMINI-E3 - PR_CurPol_CP**
| Year | Value | Unit |
|------|-------|------|
| 2050 | 44,164 | Mt CO2/yr |

**MUSE - PR_CurPol_CP**
| Year | Value | Unit |
|------|-------|------|
| 2050 | 28,295 | Mt CO2/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

Reply with a scenario, model, region, or year to narrow the answer.
