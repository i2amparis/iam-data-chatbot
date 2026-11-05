# Fresh Qwen audit: all 200 questions, routes, and full answers

Run: 5 November 2026. Configured local model: `qwen3:0.6b-q4_K_M` via Ollama.
Dataset: 631,201 cached IAM PARIS records and the local knowledge index.
Each named conversation kept one session across its six turns. A Qwen generation count of
`0` means the response used deterministic routing, metadata, glossary, or cached data.

<a id="q001"></a>

## 001. Give the 2045 CO2 emissions estimate for Brazil under current policies.

Category: numeric data and comparisons<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.46 s<br>
Conversation: standalone

**Full answer**

### Emissions|CO2 in BRA

Scope: scenario `multiple`, model `multiple`, years `2045`
Unit: `Mt CO2/yr`

Answer:
In 2045, Emissions|CO2 in BRA ranges from 52.73 (GCAM, PR_CurPol_EI) to 744.48 Mt CO2/yr (E3ME, PR_CurPol_CP) across 11 series; median 465.80.

Summary: 11 series across 4 models and 3 scenarios.
Series are ordered by model and scenario.

| Model | Scenario | 2045 | Unit |
|---|---|---|---|
| E3ME | PR_CurPol_CP | 744.48 | Mt CO2/yr |
| E3ME | PR_CurPol_EI | 682.14 | Mt CO2/yr |
| GCAM | PR_CurPol_CP | 343.38 | Mt CO2/yr |
| GCAM | PR_CurPol_CPO | 408.44 | Mt CO2/yr |
| GCAM | PR_CurPol_EI | 52.73 | Mt CO2/yr |
| GEMINI-E3 | PR_CurPol_CP | 492.61 | Mt CO2/yr |
| GEMINI-E3 | PR_CurPol_CPO | 507.52 | Mt CO2/yr |
| GEMINI-E3 | PR_CurPol_EI | 465.80 | Mt CO2/yr |
| MUSE | PR_CurPol_CP | 402.78 | Mt CO2/yr |
| MUSE | PR_CurPol_CPO | 419.51 | Mt CO2/yr |
| MUSE | PR_CurPol_EI | 503.16 | Mt CO2/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q002"></a>

## 002. How does methane output in India change from 2020 through 2050?

Category: numeric data and comparisons<br>
Route: `data_query` (extracted data entities)<br>
Qwen generations: 0<br>
Time: 0.53 s<br>
Conversation: standalone

**Full answer**

Showing results from the study **Where is the world headed?**. Also available in: Post-Glasgow targets; Global impacts of NDCs; Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU. Name another study to switch.

### Emissions|CH4 in IND

Scope: scenario `multiple`, model `multiple`, years `2020-2050`
Unit: `Mt CH4/yr`

Answer:
**Change from 2020 to 2050:** median +27% across 13 series (range +20% to +37%).

Summary: 13 series across 2 models and 7 scenarios.
Showing 12 of 13 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 1 remaining series.

| Model | Scenario | 2020 | 2025 | 2030 | 2035 | 2040 | 2045 | 2050 | Unit |
|---|---|---|---|---|---|---|---|---|---|
| GCAM | PR_Baseline | 36.67 | 39.00 | 40.69 | 42.61 | 44.32 | 46.75 | 48.37 | Mt CH4/yr |
| GCAM | PR_CurPol_CP | 36.66 | 38.69 | 40.35 | 41.97 | 43.55 | 45.76 | 47.54 | Mt CH4/yr |
| GCAM | PR_CurPol_CPO | 34.87 | 38.78 | 40.50 | 42.10 | 43.71 | 46.00 | 47.63 | Mt CH4/yr |
| GCAM | PR_CurPol_EI | 36.66 | 38.70 | 40.36 | 42.26 | 42.71 | 44.63 | 46.42 | Mt CH4/yr |
| GCAM | PR_NDC_CP | 36.66 | 39.41 | 41.25 | 43.17 | 44.69 | 46.85 | 48.67 | Mt CH4/yr |
| GCAM | PR_NDC_EI | 36.66 | 39.42 | 41.38 | 43.88 | 44.86 | 46.94 | 48.80 | Mt CH4/yr |
| GEMINI-E3 | PR_Baseline | 35.49 | 37.00 | 38.43 | 40.08 | 41.84 | 43.56 | 44.91 | Mt CH4/yr |
| GEMINI-E3 | PR_CurPol_CP | 35.49 | 36.96 | 38.41 | 39.99 | 41.74 | 43.43 | 44.77 | Mt CH4/yr |
| GEMINI-E3 | PR_CurPol_CPO | 35.49 | 37.00 | 38.17 | 39.67 | 41.34 | 42.94 | 44.19 | Mt CH4/yr |
| GEMINI-E3 | PR_CurPol_EI | 35.49 | 36.96 | 38.41 | 39.69 | 41.06 | 41.85 | 42.53 | Mt CH4/yr |
| GEMINI-E3 | PR_NDC_CP | 35.51 | 36.99 | 38.47 | 39.78 | 41.52 | 43.17 | 44.49 | Mt CH4/yr |
| GEMINI-E3 | PR_NDC_EI | 35.51 | 36.99 | 38.47 | 39.63 | 41.00 | 41.80 | 42.50 | Mt CH4/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q003"></a>

## 003. Report EU solar electricity capacity in 2040 under a net-zero pathway.

Category: numeric data and comparisons<br>
Route: `data_query` (scoped availability request)<br>
Qwen generations: 0<br>
Time: 0.33 s<br>
Conversation: standalone

**Full answer**

### Availability for Secondary Energy|Electricity|Solar

**Scenarios (13):** CP_EI, NDC_EI, NDC_LTT, PR_Baseline, PR_CurPol_CP, PR_CurPol_CPO, PR_CurPol_EI, PR_NDC_CP, PR_NDC_CPO, PR_NDC_EI, PR_WWH_CP, Unharmonised baseline, baseline

Recorded unit: `multiple`.

<a id="q004"></a>

## 004. What is the projected wind generation for Germany in 2035?

Category: numeric data and comparisons<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.36 s<br>
Conversation: standalone

**Full answer**

Showing results from the study **Where is the world headed?**. Also available in: EU path to net zero. Name another study to switch.

### Secondary Energy|Electricity|Wind in DEU

Scope: scenario `multiple`, model `multiple`, years `2035`
Unit: `EJ/yr`

Answer:
In 2035, Secondary Energy|Electricity|Wind in DEU ranges from 0.695 (ICES-XPS, PR_CurPol_EI) to 0.973 EJ/yr (E3ME, PR_CurPol_EI) across 13 series; median 0.874.

Summary: 13 series across 2 models and 8 scenarios.
Showing 12 of 13 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 1 remaining series.

| Model | Scenario | 2035 | Unit |
|---|---|---|---|
| E3ME | PR_Baseline | 0.97 | EJ/yr |
| E3ME | PR_CurPol_CP | 0.967 | EJ/yr |
| E3ME | PR_CurPol_EI | 0.973 | EJ/yr |
| E3ME | PR_NDC_CP | 0.941 | EJ/yr |
| E3ME | PR_NDC_EI | 0.949 | EJ/yr |
| E3ME | Unharmonised baseline | 0.969 | EJ/yr |
| ICES-XPS | PR_Baseline | 0.818 | EJ/yr |
| ICES-XPS | PR_CurPol_CP | 0.725 | EJ/yr |
| ICES-XPS | PR_CurPol_CPO | 0.867 | EJ/yr |
| ICES-XPS | PR_CurPol_EI | 0.695 | EJ/yr |
| ICES-XPS | PR_NDC_CP | 0.743 | EJ/yr |
| ICES-XPS | PR_NDC_CPO | 0.874 | EJ/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q005"></a>

## 005. Show primary energy from coal for China between 2030 and 2060.

Category: numeric data and comparisons<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.37 s<br>
Conversation: standalone

**Full answer**

Showing results from the study **Where is the world headed?**. Also available in: Post-Glasgow targets; Global impacts of NDCs; Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU; Tech-constrained EU pathways to net zero. Name another study to switch.

### Primary Energy|Coal in CHN

Scope: scenario `multiple`, model `multiple`, years `2030-2060`
Unit: `EJ/yr`

Answer:
In 2040, Primary Energy|Coal in CHN ranges from 52.14 (GEMINI-E3, PR_NDC_EI) to 129.19 EJ/yr (GEMINI-E3, PR_Baseline) across 43 series; median 75.65. Median change since 2030: -7%.

Summary: 43 series across 7 models and 8 scenarios.
Showing 12 of 43 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 31 remaining series.

| Model | Scenario | 2030 | 2035 | 2040 | 2045 | 2050 | 2055 | 2060 | Unit |
|---|---|---|---|---|---|---|---|---|---|
| 42 | PR_Baseline | 93.82 | 94.99 | 93.53 | 90.92 | — | — | — | EJ/yr |
| 42 | PR_CurPol_EI | 82.69 | 81.10 | 78.13 | 73.74 | — | — | — | EJ/yr |
| E3ME | PR_Baseline | 88.29 | 85.79 | 75.35 | 64.46 | 60.39 | — | — | EJ/yr |
| E3ME | PR_CurPol_CP | 88.29 | 85.28 | 73.98 | 63.27 | 59.41 | — | — | EJ/yr |
| GCAM | PR_Baseline | 99.72 | 97.87 | 96.49 | 90.85 | 84.20 | 80.67 | 76.99 | EJ/yr |
| GCAM | PR_CurPol_CP | 79.48 | 77.02 | 75.77 | 68.79 | 61.71 | 58.02 | 54.97 | EJ/yr |
| GEMINI-E3 | PR_Baseline | 123.01 | 127.61 | 129.19 | 129.38 | 128.71 | — | — | EJ/yr |
| GEMINI-E3 | PR_CurPol_CP | 73.66 | 71.84 | 70.68 | 67.97 | 64.57 | — | — | EJ/yr |
| GEMINI-E3 | PR_NDC_EI | 73.86 | 62.32 | 52.14 | 41.62 | 30.67 | — | — | EJ/yr |
| ICES-XPS | PR_Baseline | 112.94 | 118.98 | 120.76 | 121.37 | 123.17 | — | — | EJ/yr |
| MUSE | PR_Baseline | 94.57 | 80.84 | 75.13 | 63.75 | 45.06 | 34.01 | 28.60 | EJ/yr |
| TIAM_Grantham | PR_Baseline | 82.88 | — | 89.30 | — | 81.43 | — | 90.56 | EJ/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q006"></a>

## 006. Compare 2050 gas use in Japan and South Korea.

Category: numeric data and comparisons<br>
Route: `data_query` (chat plotting disabled; return numeric table)<br>
Qwen generations: 0<br>
Time: 0.26 s<br>
Conversation: standalone

**Full answer**

Showing results from the study **Where is the world headed?**. Also available in: Post-Glasgow targets; Global impacts of NDCs; Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU. Name another study to switch.

### Primary Energy|Gas in JPN

Scope: scenario `multiple`, model `multiple`, years `2050`
Unit: `EJ/yr`

Answer:
In 2050, Primary Energy|Gas in JPN ranges from 0.801 (ICES-XPS, PR_NDC_EI) to 6.44 EJ/yr (E3ME, PR_CurPol_EI) across 27 series; median 3.92.

Summary: 27 series across 4 models and 8 scenarios.
Showing 12 of 27 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 15 remaining series.

| Model | Scenario | 2050 | Unit |
|---|---|---|---|
| E3ME | PR_Baseline | 5.85 | EJ/yr |
| E3ME | PR_CurPol_CP | 5.76 | EJ/yr |
| E3ME | PR_CurPol_EI | 6.44 | EJ/yr |
| GCAM | PR_Baseline | 4.11 | EJ/yr |
| GCAM | PR_CurPol_CP | 4.14 | EJ/yr |
| GCAM | PR_CurPol_CPO | 4.03 | EJ/yr |
| ICES-XPS | PR_Baseline | 1.72 | EJ/yr |
| ICES-XPS | PR_CurPol_CP | 1.73 | EJ/yr |
| ICES-XPS | PR_NDC_EI | 0.801 | EJ/yr |
| MUSE | PR_Baseline | 4.00 | EJ/yr |
| MUSE | PR_CurPol_CP | 3.99 | EJ/yr |
| MUSE | PR_CurPol_CPO | 4.41 | EJ/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

### Primary Energy|Gas in KOR

Scope: scenario `multiple`, model `multiple`, years `2050`
Unit: `EJ/yr`

Answer:
In 2050, Primary Energy|Gas in KOR ranges from 0.101 (ICES-XPS, PR_NDC_EI) to 3.05 EJ/yr (E3ME, Unharmonised baseline) across 27 series; median 1.78.

Summary: 27 series across 4 models and 8 scenarios.
Showing 12 of 27 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 15 remaining series.

| Model | Scenario | 2050 | Unit |
|---|---|---|---|
| E3ME | PR_Baseline | 3.01 | EJ/yr |
| E3ME | PR_CurPol_CP | 3.01 | EJ/yr |
| E3ME | Unharmonised baseline | 3.05 | EJ/yr |
| GCAM | PR_Baseline | 2.20 | EJ/yr |
| GCAM | PR_CurPol_CP | 2.09 | EJ/yr |
| GCAM | PR_CurPol_CPO | 2.23 | EJ/yr |
| ICES-XPS | PR_Baseline | 1.12 | EJ/yr |
| ICES-XPS | PR_CurPol_CP | 1.09 | EJ/yr |
| ICES-XPS | PR_NDC_EI | 0.101 | EJ/yr |
| MUSE | PR_Baseline | 1.27 | EJ/yr |
| MUSE | PR_CurPol_CP | 1.40 | EJ/yr |
| MUSE | PR_CurPol_CPO | 1.27 | EJ/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q007"></a>

## 007. How much final energy does transport use in the EU in 2040?

Category: numeric data and comparisons<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.33 s<br>
Conversation: standalone

**Full answer**

Showing results from the study **Where is the world headed?**. Also available in: Where is the EU headed?; Post-Glasgow targets. Name another study to switch.

### Final Energy|Transportation in EU

Scope: scenario `multiple`, model `multiple`, years `2040`
Unit: `EJ/yr`

Answer:
In 2040, Final Energy|Transportation in EU ranges from 7.90 (TIAM_Grantham, PR_NDC_CP) to 18.68 EJ/yr (ICES-XPS, PR_Baseline) across 43 series; median 13.93.

Summary: 43 series across 7 models and 8 scenarios.
Showing 12 of 43 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 31 remaining series.

| Model | Scenario | 2040 | Unit |
|---|---|---|---|
| 42 | PR_Baseline | 16.25 | EJ/yr |
| 42 | PR_CurPol_EI | 14.30 | EJ/yr |
| E3ME | PR_Baseline | 13.92 | EJ/yr |
| E3ME | PR_CurPol_CP | 13.92 | EJ/yr |
| GCAM | PR_Baseline | 15.95 | EJ/yr |
| GCAM | PR_CurPol_CP | 14.11 | EJ/yr |
| GEMINI-E3 | PR_Baseline | 16.60 | EJ/yr |
| GEMINI-E3 | PR_CurPol_CP | 16.02 | EJ/yr |
| ICES-XPS | PR_Baseline | 18.68 | EJ/yr |
| ICES-XPS | PR_CurPol_CP | 16.14 | EJ/yr |
| MUSE | PR_Baseline | 12.16 | EJ/yr |
| TIAM_Grantham | PR_NDC_CP | 7.90 | EJ/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q008"></a>

## 008. Show population for Nigeria at the latest reported year.

Category: numeric data and comparisons<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.64 s<br>
Conversation: standalone

**Full answer**

Showing results from the study **Where is the world headed?**. Also available in: Decipher modelling studies. Name another study to switch.

### Population in NGA

Scope: scenario `multiple`, model `ICES-XPS`, years `latest available`
Unit: `million`

Answer:
In 2050, Population in NGA ranges from 385.27 (ICES-XPS, PR_NDC_EI) to 385.27 million (ICES-XPS, PR_CurPol_EI) across 7 series; median 385.27.

Summary: 7 series across 1 model and 7 scenarios.
Series are ordered by model and scenario.

| Model | Scenario | 2050 | Unit |
|---|---|---|---|
| ICES-XPS | PR_Baseline | 385.27 | million |
| ICES-XPS | PR_CurPol_CP | 385.27 | million |
| ICES-XPS | PR_CurPol_CPO | 385.27 | million |
| ICES-XPS | PR_CurPol_EI | 385.27 | million |
| ICES-XPS | PR_NDC_CP | 385.27 | million |
| ICES-XPS | PR_NDC_CPO | 385.27 | million |
| ICES-XPS | PR_NDC_EI | 385.27 | million |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q009"></a>

## 009. Find the highest carbon price for World in 2050 across loaded scenarios.

Category: numeric data and comparisons<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.37 s<br>
Conversation: standalone

**Full answer**

Showing results from the study **Where is the world headed?**. Also available in: Decipher modelling studies; Post-Glasgow targets; Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU; Global impacts of NDCs; Tech-constrained EU pathways to net zero; EU path to net zero. Name another study to switch.

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

<a id="q010"></a>

## 010. What is the median GDP projection for the EU in 2060?

Category: numeric data and comparisons<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.29 s<br>
Conversation: standalone

**Full answer**

Showing results from the study **Where is the world headed?**. Also available in: Where is the EU headed?. Name another study to switch.

### GDP|MER in EU

Scope: scenario `multiple`, model `GCAM`, years `2060`
Unit: `billion US$2010/yr OR local currency`

Answer:
In 2060, GDP|MER in EU ranges from 34,426 (GCAM, PR_Baseline) to 34,426 billion US$2010/yr OR local currency (GCAM, PR_NDC_EI) across 7 series; median 34,426.

Summary: 7 series across 1 model and 7 scenarios.
Series are ordered by model and scenario.

| Model | Scenario | 2060 | Unit |
|---|---|---|---|
| GCAM | PR_Baseline | 34,426 | billion US$2010/yr OR local currency |
| GCAM | PR_CurPol_CP | 34,426 | billion US$2010/yr OR local currency |
| GCAM | PR_CurPol_CPO | 34,426 | billion US$2010/yr OR local currency |
| GCAM | PR_CurPol_EI | 34,426 | billion US$2010/yr OR local currency |
| GCAM | PR_NDC_CP | 34,426 | billion US$2010/yr OR local currency |
| GCAM | PR_NDC_CPO | 34,426 | billion US$2010/yr OR local currency |
| GCAM | PR_NDC_EI | 34,426 | billion US$2010/yr OR local currency |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q011"></a>

## 011. How far do global CO2 emissions fall between 2030 and 2070?

Category: numeric data and comparisons<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.34 s<br>
Conversation: standalone

**Full answer**

Showing results from the study **Where is the world headed?**. Also available in: Post-Glasgow targets; Global impacts of NDCs; Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU; EU path to net zero; Tech-constrained EU pathways to net zero. Name another study to switch.

### Emissions|CO2 in World

Scope: scenario `multiple`, model `multiple`, years `2030-2070`
Unit: `Mt CO2/yr`

Answer:
**From 2030 to 2070, Emissions|CO2 in World rises in 4 and falls in 18 of 22 series** (median change -41%).

Summary: 38 series across 4 models and 13 scenarios.
Showing 12 of 38 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 26 remaining series.

| Model | Scenario | 2030 | 2035 | 2040 | 2045 | 2050 | 2055 | 2060 | 2065 | 2070 | Unit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E3ME | PR_Baseline | 36,934 | 38,149 | 38,757 | 39,071 | 40,393 | — | — | — | — | Mt CO2/yr |
| E3ME | PR_CurPol_CP | 36,935 | 38,006 | 38,479 | 38,772 | 40,054 | — | — | — | — | Mt CO2/yr |
| E3ME | PR_CurPol_EI | 36,922 | 37,792 | 38,706 | 39,138 | 39,239 | — | — | — | — | Mt CO2/yr |
| GCAM | Baseline | 42,107 | 43,979 | 45,315 | 45,465 | 45,595 | 46,708 | 48,206 | 49,791 | 51,521 | Mt CO2/yr |
| GCAM | CP_EI | 37,516 | 35,676 | 33,501 | 31,082 | 28,519 | 26,244 | 24,349 | 22,565 | 20,869 | Mt CO2/yr |
| GCAM | NDC_LTT | 32,324 | 26,749 | 20,969 | 14,887 | 9,173 | 4,316 | 871.19 | 669.19 | 59.16 | Mt CO2/yr |
| GEMINI-E3 | Baseline | 42,037 | 44,008 | 45,532 | 46,413 | 46,713 | — | — | — | — | Mt CO2/yr |
| GEMINI-E3 | CP_EI | 34,431 | 32,849 | 31,313 | 29,460 | 27,388 | — | — | — | — | Mt CO2/yr |
| GEMINI-E3 | PR_Baseline | 44,462 | 46,876 | 48,984 | 50,715 | 52,211 | — | — | — | — | Mt CO2/yr |
| MUSE | baseline | 35,891 | 34,436 | 33,696 | 32,387 | 31,448 | 30,979 | 31,285 | 32,420 | 33,030 | Mt CO2/yr |
| MUSE | CP_EI | 33,276 | 30,502 | 27,906 | 26,115 | 23,856 | 22,850 | 21,458 | 20,621 | 19,791 | Mt CO2/yr |
| MUSE | NDC_EI | 31,089 | 27,917 | 24,700 | 22,635 | 19,999 | 18,772 | 17,549 | 17,234 | 16,328 | Mt CO2/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q012"></a>

## 012. Compare electricity generation from solar and nuclear in Europe in 2050.

Category: numeric data and comparisons<br>
Route: `data_query` (chat plotting disabled; return numeric table)<br>
Qwen generations: 0<br>
Time: 0.44 s<br>
Conversation: standalone

**Full answer**

Showing results from the study **Where is the world headed?**. Also available in: Where is the EU headed?; Post-Glasgow targets. Name another study to switch.

### Secondary Energy|Electricity|Solar in EU

Scope: scenario `multiple`, model `multiple`, years `2050`
Unit: `EJ/yr`

Answer:
In 2050, Secondary Energy|Electricity|Solar in EU ranges from 0.118 (ICES-XPS, PR_Baseline) to 22.82 EJ/yr (ICES-XPS, PR_NDC_EI) across 40 series; median 2.95.

Summary: 40 series across 6 models and 8 scenarios.
Showing 12 of 40 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 28 remaining series.

| Model | Scenario | 2050 | Unit |
|---|---|---|---|
| E3ME | PR_Baseline | 3.09 | EJ/yr |
| E3ME | PR_CurPol_CP | 2.98 | EJ/yr |
| GCAM | PR_Baseline | 3.58 | EJ/yr |
| GCAM | PR_CurPol_CP | 4.88 | EJ/yr |
| GEMINI-E3 | PR_Baseline | 1.69 | EJ/yr |
| GEMINI-E3 | PR_CurPol_CP | 2.07 | EJ/yr |
| ICES-XPS | PR_Baseline | 0.118 | EJ/yr |
| ICES-XPS | PR_CurPol_CP | 19.46 | EJ/yr |
| ICES-XPS | PR_NDC_EI | 22.82 | EJ/yr |
| MUSE | PR_Baseline | 1.96 | EJ/yr |
| MUSE | PR_CurPol_CP | 2.92 | EJ/yr |
| TIAM_Grantham | PR_Baseline | 1.99 | EJ/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q013"></a>

## 013. Give hydropower capacity for Greece in 2030.

Category: numeric data and comparisons<br>
Route: `data_query` (low confidence entity clarification)<br>
Qwen generations: 0<br>
Time: 0.22 s<br>
Conversation: standalone

**Full answer**

I could not confidently match all of the requested wording. The loaded variable catalogue does not represent these term(s) in the top match: `hydropower`. Choose the variable: 1. `Capacity|Electricity` (power capacity, electricity) 2. `Capacity|Hydrogen|Gas` (capacity, gas) 3. `Capacity|Hydrogen|Coal` (capacity, hydrogen) Reply with a number (1-3), or `yes` for option 1.

<a id="q014"></a>

## 014. What are the 2050 industrial CO2 emissions for the United States?

Category: numeric data and comparisons<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.37 s<br>
Conversation: standalone

**Full answer**

Showing results from the study **Where is the world headed?**. Also available in: Post-Glasgow targets; EU path to net zero; Global impacts of NDCs; Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU. Name another study to switch.

### Emissions|CO2|Industrial Processes in USA

Scope: scenario `multiple`, model `multiple`, years `2050`
Unit: `Mt CO2/yr`

Answer:
In 2050, Emissions|CO2|Industrial Processes in USA ranges from -35.76 (GCAM, PR_NDC_EI) to 137.56 Mt CO2/yr (GEMINI-E3, PR_CurPol_EI) across 21 series; median 80.67.

Summary: 21 series across 3 models and 8 scenarios.
Showing 12 of 21 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 9 remaining series.

| Model | Scenario | 2050 | Unit |
|---|---|---|---|
| GCAM | Baseline | 114.31 | Mt CO2/yr |
| GCAM | PR_Baseline | 115.82 | Mt CO2/yr |
| GCAM | PR_CurPol_CP | 79.76 | Mt CO2/yr |
| GCAM | PR_NDC_EI | -35.76 | Mt CO2/yr |
| GEMINI-E3 | PR_Baseline | 136.34 | Mt CO2/yr |
| GEMINI-E3 | PR_CurPol_CP | 136.78 | Mt CO2/yr |
| GEMINI-E3 | PR_CurPol_CPO | 135.60 | Mt CO2/yr |
| GEMINI-E3 | PR_CurPol_EI | 137.56 | Mt CO2/yr |
| MUSE | PR_Baseline | 130.32 | Mt CO2/yr |
| MUSE | PR_CurPol_CP | 130.32 | Mt CO2/yr |
| MUSE | PR_CurPol_CPO | 130.32 | Mt CO2/yr |
| MUSE | PR_CurPol_EI | 69.31 | Mt CO2/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q015"></a>

## 015. Show final electricity demand in France from 2025 to 2045.

Category: numeric data and comparisons<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.47 s<br>
Conversation: standalone

**Full answer**

Showing results from the study **EU path to net zero**. Also available in: Where is the world headed?. Name another study to switch.

### Final Energy|Electricity in FRA

Scope: scenario `multiple`, model `multiple`, years `2025-2045`
Unit: `EJ/yr`

Answer:
In 2040, Final Energy|Electricity in FRA ranges from 1.95 (E4SMA-EU-TIMES 1.0, NZE_Bench_H) to 2.91 EJ/yr (ICES-XPS 1.0, NZE_EUPol_Stand) across 13 series; median 2.19. Median change since 2025: +30%.

Summary: 13 series across 3 models and 5 scenarios.
Showing 12 of 13 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 1 remaining series.

| Model | Scenario | 2025 | 2030 | 2035 | 2040 | 2045 | Unit |
|---|---|---|---|---|---|---|---|
| E4SMA-EU-TIMES 1.0 | NZE_Bench_H | 1.51 | 1.54 | — | 1.95 | — | EJ/yr |
| E4SMA-EU-TIMES 1.0 | NZE_Bench_L | 1.54 | 1.57 | — | 2.16 | — | EJ/yr |
| E4SMA-EU-TIMES 1.0 | NZE_Bench_M | 1.51 | 1.55 | — | 2.11 | — | EJ/yr |
| E4SMA-EU-TIMES 1.0 | NZE_EUPol_Stand | 1.64 | 1.68 | — | 2.14 | — | EJ/yr |
| E4SMA-EU-TIMES 1.0 | WWH21 | 1.68 | 1.78 | — | 2.05 | — | EJ/yr |
| ICES-XPS 1.0 | NZE_Bench_H | 2.01 | 2.42 | 2.62 | 2.84 | 3.19 | EJ/yr |
| ICES-XPS 1.0 | NZE_EUPol_Stand | 2.04 | 2.52 | 2.70 | 2.91 | 3.23 | EJ/yr |
| ICES-XPS 1.0 | WWH21 | 1.99 | 2.34 | 2.40 | 2.45 | 2.61 | EJ/yr |
| NEMESIS 5.1 | NZE_Bench_H | 1.75 | 1.89 | 2.00 | 2.19 | 2.53 | EJ/yr |
| NEMESIS 5.1 | NZE_Bench_L | 1.76 | 1.93 | 2.11 | 2.41 | 3.08 | EJ/yr |
| NEMESIS 5.1 | NZE_Bench_M | 1.75 | 1.90 | 2.05 | 2.27 | 2.72 | EJ/yr |
| NEMESIS 5.1 | NZE_EUPol_Stand | 1.72 | 1.87 | 1.99 | 2.23 | 2.62 | EJ/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/eu-path-to-net-zero/graphs).

<a id="q016"></a>

## 016. Report carbon capture deployment for World by 2050.

Category: numeric data and comparisons<br>
Route: `data_query` (low confidence entity clarification)<br>
Qwen generations: 0<br>
Time: 0.70 s<br>
Conversation: standalone

**Full answer**

I could not confidently match all of the requested wording. The loaded variable catalogue does not represent these term(s) in the top match: `deployment`. Choose the variable: 1. `Carbon Capture` 2. `Carbon Capture|Fossil` 3. `Carbon Capture|Biomass` Reply with a number (1-3), or `yes` for option 1.

<a id="q017"></a>

## 017. How much bioenergy is used globally in 2100?

Category: numeric data and comparisons<br>
Route: `data_query` (low confidence entity clarification)<br>
Qwen generations: 0<br>
Time: 0.45 s<br>
Conversation: standalone

**Full answer**

I could not confidently match all of the requested wording. The loaded variable catalogue does not represent these term(s) in the top match: `used`. Choose the variable: 1. `Final Energy|Gases|Bioenergy` (final energy use, gas) 2. `Final Energy|Liquids|Bioenergy` (final energy use) 3. `Final Energy|AFOFI|Gases|Bioenergy` (final energy use, gas) Reply with a number (1-3), or `yes` for option 1.

<a id="q018"></a>

## 018. Compare baseline and NDC CO2 results for India in 2040.

Category: numeric data and comparisons<br>
Route: `data_query` (chat plotting disabled; return numeric table)<br>
Qwen generations: 0<br>
Time: 0.41 s<br>
Conversation: standalone

**Full answer**

Showing results from the study **Where is the world headed?**. Also available in: Post-Glasgow targets; Global impacts of NDCs. Name another study to switch.

### Emissions|CO2 in IND

Scope: scenario `multiple`, model `multiple`, years `2040`
Unit: `Mt CO2/yr`

Answer:
In 2040, Emissions|CO2 in IND ranges from 2,959 (MUSE, PR_Baseline) to 5,909 Mt CO2/yr (GCAM, PR_Baseline) across 6 series; median 5,664.

**E3ME - PR_Baseline**
| Year | Value | Unit |
|------|-------|------|
| 2040 | 5,658 | Mt CO2/yr |

**E3ME - Unharmonised baseline**
| Year | Value | Unit |
|------|-------|------|
| 2040 | 5,671 | Mt CO2/yr |

**GCAM - Baseline**
| Year | Value | Unit |
|------|-------|------|
| 2040 | 5,049 | Mt CO2/yr |

**GCAM - PR_Baseline**
| Year | Value | Unit |
|------|-------|------|
| 2040 | 5,909 | Mt CO2/yr |

**GEMINI-E3 - PR_Baseline**
| Year | Value | Unit |
|------|-------|------|
| 2040 | 5,683 | Mt CO2/yr |

**MUSE - PR_Baseline**
| Year | Value | Unit |
|------|-------|------|
| 2040 | 2,959 | Mt CO2/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q019"></a>

## 019. What is the latest methane value for Brazil?

Category: numeric data and comparisons<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.35 s<br>
Conversation: standalone

**Full answer**

Showing results from the study **Where is the world headed?**. Also available in: Post-Glasgow targets; Global impacts of NDCs; Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU. Name another study to switch.

### Emissions|CH4 in BRA

Scope: scenario `multiple`, model `GCAM`, years `latest available`
Unit: `Mt CH4/yr`

Answer:
In 2100, Emissions|CH4 in BRA ranges from 14.64 (GCAM, PR_NDC_EI) to 21.04 Mt CH4/yr (GCAM, PR_Baseline) across 7 series; median 17.62.

Summary: 7 series across 1 model and 7 scenarios.
Series are ordered by model and scenario.

| Model | Scenario | 2100 | Unit |
|---|---|---|---|
| GCAM | PR_Baseline | 21.04 | Mt CH4/yr |
| GCAM | PR_CurPol_CP | 17.62 | Mt CH4/yr |
| GCAM | PR_CurPol_CPO | 17.73 | Mt CH4/yr |
| GCAM | PR_CurPol_EI | 16.52 | Mt CH4/yr |
| GCAM | PR_NDC_CP | 17.19 | Mt CH4/yr |
| GCAM | PR_NDC_CPO | 17.66 | Mt CH4/yr |
| GCAM | PR_NDC_EI | 14.64 | Mt CH4/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q020"></a>

## 020. Show power-sector emissions for Europe under current policies.

Category: numeric data and comparisons<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.53 s<br>
Conversation: standalone

**Full answer**

### Emissions|CO2|Energy|Supply|Other Sector in EU

Scope: scenario `multiple`, model `multiple`, years `2005-2100`
Unit: `Mt CO2/yr`

Answer:
In 2040, Emissions|CO2|Energy|Supply|Other Sector in EU ranges from -15.08 (GCAM, PR_CurPol_EI) to 167.86 Mt CO2/yr (42, PR_CurPol_EI) across 13 series; median 18.82. Median change since 2005: -101%.

Summary: 13 series across 5 models and 3 scenarios.
Showing 12 of 13 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 1 remaining series.

| Model | Scenario | 2005 | 2010 | 2015 | 2020 | 2025 | 2030 | 2035 | 2040 | 2045 | 2050 | 2055 | 2060 | 2065 | 2070 | 2075 | 2080 | 2085 | 2090 | 2095 | 2100 | Unit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | PR_CurPol_EI | 302.27 | 280.00 | 266.73 | 243.01 | 205.75 | 191.14 | 178.36 | 167.86 | 158.99 | — | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| GCAM | PR_CurPol_CP | 0.00 | 0.00 | 0.00 | 7.12 | 16.83 | 28.70 | 6.40 | -5.18 | -12.88 | -2.84 | 3.92 | 10.71 | 6.59 | -3.32 | -12.64 | -18.14 | -23.30 | -29.20 | -48.89 | -73.25 | Mt CO2/yr |
| GCAM | PR_CurPol_CPO | 0.00 | 0.00 | 0.00 | 3.16 | 3.74 | 0.19 | -2.57 | -5.67 | -6.20 | -4.99 | -2.30 | -1.98 | -3.63 | -8.31 | -14.76 | -24.31 | -31.71 | -56.25 | -93.24 | -137.28 | Mt CO2/yr |
| GCAM | PR_CurPol_EI | 0.00 | 0.00 | 0.00 | 7.12 | 16.83 | 28.70 | -9.44 | -15.08 | -12.12 | 9.49 | 5.32 | -3.76 | -23.80 | -45.90 | -48.06 | -38.08 | -32.02 | -31.53 | -42.95 | -56.89 | Mt CO2/yr |
| GEMINI-E3 | PR_CurPol_CP | — | — | 19.16 | 18.35 | 19.04 | 19.40 | 20.53 | 21.27 | 22.18 | 23.25 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| GEMINI-E3 | PR_CurPol_CPO | — | — | 19.16 | 18.35 | 19.03 | 19.39 | 20.52 | 21.26 | 22.18 | 23.24 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| GEMINI-E3 | PR_CurPol_EI | — | — | 19.16 | 18.35 | 19.04 | 19.40 | 20.54 | 21.28 | 22.20 | 23.28 | — | — | — | — | — | — | — | — | — | — | Mt CO2/yr |
| MUSE | PR_CurPol_CP | — | — | 37.62 | 36.69 | 32.49 | 28.24 | 23.53 | 18.82 | 14.11 | 9.41 | 4.70 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | Mt CO2/yr |
| MUSE | PR_CurPol_CPO | — | — | 37.62 | 36.69 | 32.48 | 28.24 | 23.53 | 18.82 | 14.11 | 9.40 | 4.70 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | Mt CO2/yr |
| MUSE | PR_CurPol_EI | — | — | 37.62 | 36.69 | 32.49 | 28.24 | 23.53 | 18.82 | 14.11 | 9.41 | 4.70 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | Mt CO2/yr |
| TIAM_Grantham | PR_CurPol_CP | 158.01 | 47.59 | — | 14.24 | — | 14.04 | — | -1.46 | — | -18.85 | — | -86.52 | — | -354.38 | — | -1,071 | — | -2,383 | — | -6,911 | Mt CO2/yr |
| TIAM_Grantham | PR_CurPol_CPO | 158.01 | 47.59 | — | 14.24 | — | 11.70 | — | -1.48 | — | -20.31 | — | -80.29 | — | -333.10 | — | -1,399 | — | -5,769 | — | -9,934 | Mt CO2/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q021"></a>

## 021. How much investment in solar electricity is reported for China in 2050?

Category: numeric data and comparisons<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.47 s<br>
Conversation: standalone

**Full answer**

Showing results from the study **Where is the world headed?**. Also available in: Post-Glasgow targets; Global impacts of NDCs; Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU; Tech-constrained EU pathways to net zero. Name another study to switch.

### Secondary Energy|Electricity|Solar in CHN

Scope: scenario `multiple`, model `multiple`, years `2050`
Unit: `EJ/yr`

Answer:
In 2050, Secondary Energy|Electricity|Solar in CHN ranges from 0.0123 (ICES-XPS, PR_Baseline) to 25.96 EJ/yr (E3ME, Unharmonised baseline) across 40 series; median 3.37.

Summary: 40 series across 6 models and 8 scenarios.
Showing 12 of 40 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 28 remaining series.

| Model | Scenario | 2050 | Unit |
|---|---|---|---|
| E3ME | PR_Baseline | 25.55 | EJ/yr |
| E3ME | Unharmonised baseline | 25.96 | EJ/yr |
| GCAM | PR_Baseline | 9.34 | EJ/yr |
| GCAM | PR_CurPol_CP | 10.18 | EJ/yr |
| GEMINI-E3 | PR_Baseline | 6.85 | EJ/yr |
| GEMINI-E3 | PR_CurPol_CP | 9.27 | EJ/yr |
| ICES-XPS | PR_Baseline | 0.0123 | EJ/yr |
| ICES-XPS | PR_CurPol_CP | 0.0331 | EJ/yr |
| MUSE | PR_Baseline | 3.13 | EJ/yr |
| MUSE | PR_CurPol_CP | 3.14 | EJ/yr |
| TIAM_Grantham | PR_Baseline | 1.52 | EJ/yr |
| TIAM_Grantham | PR_CurPol_CP | 1.56 | EJ/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q022"></a>

## 022. Give the 2030 coal electricity generation figure for Africa.

Category: numeric data and comparisons<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.41 s<br>
Conversation: standalone

**Full answer**

Showing results from the study **Post-Glasgow targets**. Also available in: Tech-constrained EU pathways to net zero; Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU; Implications of EU steel industry relocation. Name another study to switch.

### Secondary Energy|Electricity|Coal in AFR

Scope: scenario `multiple`, model `multiple`, years `2030`
Unit: `EJ/yr`

Answer:
In 2030, Secondary Energy|Electricity|Coal in AFR ranges from 0.443 (GEMINI-E3 7.0, CP_EI) to 0.952 EJ/yr (TIAM_Grantham, NDC_LTT) across 8 series; median 0.733.

Summary: 8 series across 2 models and 4 scenarios.
Series are ordered by model and scenario.

| Model | Scenario | 2030 | Unit |
|---|---|---|---|
| GEMINI-E3 7.0 | Baseline | 0.515 | EJ/yr |
| GEMINI-E3 7.0 | CP_EI | 0.443 | EJ/yr |
| GEMINI-E3 7.0 | NDC_EI | 0.443 | EJ/yr |
| GEMINI-E3 7.0 | NDC_LTT | 0.443 | EJ/yr |
| TIAM_Grantham | Baseline | 0.952 | EJ/yr |
| TIAM_Grantham | CP_EI | 0.952 | EJ/yr |
| TIAM_Grantham | NDC_EI | 0.952 | EJ/yr |
| TIAM_Grantham | NDC_LTT | 0.952 | EJ/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/post-glasgow-targets/graphs).

<a id="q023"></a>

## 023. Which scenario has the smallest EU carbon price in 2050?

Category: numeric data and comparisons<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.43 s<br>
Conversation: standalone

**Full answer**

Showing results from the study **Where is the world headed?**. Also available in: Where is the EU headed?; Post-Glasgow targets. Name another study to switch.

### Price|Carbon in EU

Scope: scenario `multiple`, model `multiple`, years `2050`
Unit: `US$2010/t CO2`

Answer:
**Lowest in 2050:** 0.00 US$2010/t CO2 (ICES-XPS, PR_Baseline), out of 38 series.

Summary: 38 series across 6 models and 8 scenarios.
Showing 12 of 38 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 26 remaining series.

| Model | Scenario | 2050 | Unit |
|---|---|---|---|
| E3ME | PR_Baseline | 52.65 | US$2010/t CO2 |
| E3ME | PR_CurPol_CP | 42.79 | US$2010/t CO2 |
| GCAM | PR_CurPol_CP | 70.59 | US$2010/t CO2 |
| GCAM | PR_CurPol_CPO | 90.27 | US$2010/t CO2 |
| GEMINI-E3 | PR_Baseline | 4.97 | US$2010/t CO2 |
| GEMINI-E3 | PR_CurPol_CP | 56.10 | US$2010/t CO2 |
| ICES-XPS | PR_Baseline | 0.00 | US$2010/t CO2 |
| ICES-XPS | PR_CurPol_CP | 39.06 | US$2010/t CO2 |
| ICES-XPS | PR_NDC_EI | 432.96 | US$2010/t CO2 |
| MUSE | PR_Baseline | 66.00 | US$2010/t CO2 |
| MUSE | PR_CurPol_CP | 66.00 | US$2010/t CO2 |
| TIAM_Grantham | PR_CurPol_CP | 305.47 | US$2010/t CO2 |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q024"></a>

## 024. What is the average 2040 global electricity output across models?

Category: numeric data and comparisons<br>
Route: `data_query` (topic-wide model availability)<br>
Qwen generations: 0<br>
Time: 1.01 s<br>
Conversation: standalone

**Full answer**

### Models covering Energy Supply

24 model(s) of the 33 with loaded results report at least one energy supply variable in IAM PARIS:

ALADIN, ATOM, E3ME 6.1, E4SMA-EU-TIMES 1.0, EnergyPLAN, FORECAST, GCAM 7.0, GCAM-PR 5.3, GCAM-PR 7.0, GEM-E3, GEMINI-E3 7.0, ices, ICES-XPS 1.0, ICES-XPS 1.1, MUSE, NEMESIS 5.0, NEMESIS 5.1, OSeMOSYS-GR, PRIMES-Buildings, PROMETHEUS V1 … and 4 more

Ask for a specific model (e.g. `tell me about ALADIN`) or a data query (e.g. `energy supply emissions for Europe`) to go deeper.

<a id="q025"></a>

## 025. Report residential final energy for Italy in 2050.

Category: numeric data and comparisons<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.46 s<br>
Conversation: standalone

**Full answer**

I could not find data for `Final Energy|Residential` in `ITA`.

Reason: `Final Energy|Residential` is not reported for `ITA` in any scenario of the loaded data.

Closest valid options:

1. region `EU27noUK`

Reply with `1`, or type the region you want.

<a id="q026"></a>

## 026. Compare 2035 wind capacity in Spain and Germany.

Category: numeric data and comparisons<br>
Route: `data_query` (chat plotting disabled; return numeric table)<br>
Qwen generations: 0<br>
Time: 0.14 s<br>
Conversation: standalone

**Full answer**

Showing results from the study **EU path to net zero**.

### Capacity|Electricity|Wind in ESP

Scope: scenario `multiple`, model `NEMESIS 5.1`, years `2035`
Unit: `GW`

Answer:
In 2035, Capacity|Electricity|Wind in ESP ranges from 37.18 (NEMESIS 5.1, WWH21) to 56.59 GW (NEMESIS 5.1, NZE_Bench_L) across 5 series; median 53.12.

**NEMESIS 5.1 - NZE_Bench_H**
| Year | Value | Unit |
|------|-------|------|
| 2035 | 50.53 | GW |

**NEMESIS 5.1 - NZE_Bench_L**
| Year | Value | Unit |
|------|-------|------|
| 2035 | 56.59 | GW |

**NEMESIS 5.1 - NZE_Bench_M**
| Year | Value | Unit |
|------|-------|------|
| 2035 | 53.29 | GW |

**NEMESIS 5.1 - NZE_EUPol_Stand**
| Year | Value | Unit |
|------|-------|------|
| 2035 | 53.12 | GW |

**NEMESIS 5.1 - WWH21**
| Year | Value | Unit |
|------|-------|------|
| 2035 | 37.18 | GW |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/eu-path-to-net-zero/graphs).

### Capacity|Electricity|Wind in DEU

Scope: scenario `multiple`, model `NEMESIS 5.1`, years `2035`
Unit: `GW`

Answer:
In 2035, Capacity|Electricity|Wind in DEU ranges from 71.75 (NEMESIS 5.1, WWH21) to 106.65 GW (NEMESIS 5.1, NZE_Bench_L) across 5 series; median 98.30.

**NEMESIS 5.1 - NZE_Bench_H**
| Year | Value | Unit |
|------|-------|------|
| 2035 | 93.97 | GW |

**NEMESIS 5.1 - NZE_Bench_L**
| Year | Value | Unit |
|------|-------|------|
| 2035 | 106.65 | GW |

**NEMESIS 5.1 - NZE_Bench_M**
| Year | Value | Unit |
|------|-------|------|
| 2035 | 99.84 | GW |

**NEMESIS 5.1 - NZE_EUPol_Stand**
| Year | Value | Unit |
|------|-------|------|
| 2035 | 98.30 | GW |

**NEMESIS 5.1 - WWH21**
| Year | Value | Unit |
|------|-------|------|
| 2035 | 71.75 | GW |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/eu-path-to-net-zero/graphs).

<a id="q027"></a>

## 027. How does world natural-gas primary energy evolve after 2040?

Category: numeric data and comparisons<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.38 s<br>
Conversation: standalone

**Full answer**

Showing results from the study **Where is the world headed?**. Also available in: Post-Glasgow targets; Global impacts of NDCs; Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU; EU path to net zero; Tech-constrained EU pathways to net zero. Name another study to switch.

### Primary Energy|Gas in World

Scope: scenario `multiple`, model `multiple`, years `2041-latest available`
Unit: `EJ/yr`

Answer:
In 2045, Primary Energy|Gas in World ranges from 113.57 (ICES-XPS, PR_NDC_EI) to 227.49 EJ/yr (GEMINI-E3, PR_Baseline) across 36 series; median 177.95.

Summary: 36 series across 6 models and 8 scenarios.
Showing 12 of 36 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 24 remaining series.

| Model | Scenario | 2045 | 2050 | 2055 | 2060 | 2065 | 2070 | 2075 | 2080 | 2085 | 2090 | 2095 | 2100 | Unit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | PR_Baseline | 209.92 | — | — | — | — | — | — | — | — | — | — | — | EJ/yr |
| 42 | PR_CurPol_EI | 209.95 | — | — | — | — | — | — | — | — | — | — | — | EJ/yr |
| E3ME | PR_Baseline | 178.23 | 186.91 | — | — | — | — | — | — | — | — | — | — | EJ/yr |
| E3ME | PR_CurPol_CP | 177.67 | 186.20 | — | — | — | — | — | — | — | — | — | — | EJ/yr |
| GCAM | PR_Baseline | 204.66 | 211.41 | 223.52 | 238.56 | 253.38 | 270.25 | 288.75 | 306.31 | 322.67 | 332.02 | 338.27 | 344.65 | EJ/yr |
| GCAM | PR_CurPol_CP | 199.98 | 206.25 | 217.92 | 231.59 | 245.47 | 259.61 | 274.59 | 288.97 | 306.02 | 316.05 | 322.35 | 328.88 | EJ/yr |
| GEMINI-E3 | PR_Baseline | 227.49 | 240.91 | — | — | — | — | — | — | — | — | — | — | EJ/yr |
| GEMINI-E3 | PR_CurPol_CP | 220.92 | 232.87 | — | — | — | — | — | — | — | — | — | — | EJ/yr |
| ICES-XPS | PR_Baseline | 165.59 | 167.40 | — | — | — | — | — | — | — | — | — | — | EJ/yr |
| ICES-XPS | PR_NDC_EI | 113.57 | 107.32 | — | — | — | — | — | — | — | — | — | — | EJ/yr |
| MUSE | PR_Baseline | 130.84 | 131.25 | 126.07 | 121.84 | 122.91 | 122.97 | 121.35 | 117.90 | 115.21 | 112.07 | 111.76 | 112.99 | EJ/yr |
| MUSE | PR_CurPol_CP | 137.28 | 136.31 | 136.09 | 130.62 | 131.30 | 130.75 | 132.46 | 132.05 | 132.24 | 134.47 | 136.76 | 139.43 | EJ/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q028"></a>

## 028. Give the 2050 CO2 value from GCAM for China under PR_NDC_EI.

Category: numeric data and comparisons<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.36 s<br>
Conversation: standalone

**Full answer**

### Emissions|CO2 in CHN

Scope: scenario `PR_NDC_EI`, model `GCAM`, years `2050`
Unit: `Mt CO2/yr`

Answer:
In 2050, Emissions|CO2 in CHN is 7,989 Mt CO2/yr (GCAM, PR_NDC_EI).

**GCAM - PR_NDC_EI**
| Year | Value | Unit |
|------|-------|------|
| 2050 | 7,989 | Mt CO2/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q029"></a>

## 029. Show electricity production from biomass in India through 2060.

Category: numeric data and comparisons<br>
Route: `data_query` (low confidence entity clarification)<br>
Qwen generations: 0<br>
Time: 0.52 s<br>
Conversation: standalone

**Full answer**

I could not confidently match all of the requested wording. The loaded variable catalogue does not represent these term(s) in the top match: `through`. Choose the variable: 1. `Secondary Energy|Electricity` (electricity output) 2. `Final Energy|Electricity` (final energy use, electricity) 3. `Capacity|Electricity` (power capacity, electricity) Reply with a number (1-3), or `yes` for option 1.

<a id="q030"></a>

## 030. How much does EU GDP change between 2020 and 2040?

Category: numeric data and comparisons<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.35 s<br>
Conversation: standalone

**Full answer**

Showing results from the study **Where is the world headed?**. Also available in: Where is the EU headed?. Name another study to switch.

### GDP|MER in EU

Scope: scenario `multiple`, model `multiple`, years `2020-2040`
Unit: `multiple`

Answer:
Summary: 26 series across 4 models and 8 scenarios.
Showing 12 of 26 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 14 remaining series.

| Model | Scenario | 2020 | 2025 | 2030 | 2035 | 2040 | Unit |
|---|---|---|---|---|---|---|---|
| E3ME | PR_Baseline | 19,653 | 20,938 | 22,373 | 23,850 | 25,612 | billion US$2010/yr |
| E3ME | PR_CurPol_CP | 19,653 | 20,938 | 22,373 | 23,848 | 25,609 | billion US$2010/yr |
| E3ME | PR_CurPol_EI | 19,656 | 20,939 | 22,375 | 23,840 | 25,577 | billion US$2010/yr |
| GCAM | PR_Baseline | 19,705 | 21,015 | 22,444 | 23,909 | 25,589 | billion US$2010/yr OR local currency |
| GCAM | PR_CurPol_CP | 19,705 | 21,015 | 22,444 | 23,909 | 25,589 | billion US$2010/yr OR local currency |
| GCAM | PR_CurPol_CPO | 19,705 | 21,015 | 22,444 | 23,909 | 25,589 | billion US$2010/yr OR local currency |
| GEMINI-E3 | PR_Baseline | 19,766 | 21,129 | 22,463 | 23,920 | 25,436 | billion US$2010/yr OR local currency |
| GEMINI-E3 | PR_CurPol_CP | 19,766 | 21,145 | 22,518 | 24,070 | 25,688 | billion US$2010/yr OR local currency |
| GEMINI-E3 | PR_CurPol_CPO | 19,766 | 21,145 | 22,518 | 24,070 | 25,687 | billion US$2010/yr OR local currency |
| ICES-XPS | PR_Baseline | 20,406 | 21,778 | 23,281 | 24,822 | 26,584 | billion US$2010/yr |
| ICES-XPS | PR_CurPol_EI | 20,396 | 21,639 | 22,892 | 24,064 | 25,210 | billion US$2010/yr |
| ICES-XPS | PR_NDC_CPO | 20,395 | 21,740 | 23,173 | 24,760 | 26,613 | billion US$2010/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q031"></a>

## 031. Report global hydrogen capacity in 2050.

Category: numeric data and comparisons<br>
Route: `data_query` (low confidence entity clarification)<br>
Qwen generations: 0<br>
Time: 0.56 s<br>
Conversation: standalone

**Full answer**

I found several possible variables, but none has enough evidence for automatic selection. Choose the variable: 1. `Capacity|Hydrogen|Gas` (capacity, gas) 2. `Capacity|Hydrogen|Coal` (capacity, hydrogen) 3. `Capacity|Hydrogen|Biomass` (capacity, hydrogen) Reply with a number (1-3), or `yes` for option 1.

<a id="q032"></a>

## 032. Compare two models on 2030 EU solar capacity.

Category: numeric data and comparisons<br>
Route: `data_query` (chat plotting disabled; return numeric table)<br>
Qwen generations: 0<br>
Time: 0.31 s<br>
Conversation: standalone

**Full answer**

Showing results from the study **Where is the world headed?**. Also available in: Where is the EU headed?. Name another study to switch.

### Capacity|Electricity|Solar in EU

Scope: scenario `multiple`, model `multiple`, years `2030`
Unit: `GW`

Answer:
In 2030, Capacity|Electricity|Solar in EU ranges from 50.07 (TIAM_Grantham, PR_CurPol_EI) to 1,043 GW (GCAM, PR_CurPol_CP) across 14 series; median 361.79.

Summary: 14 series across 2 models and 7 scenarios.
Showing 12 of 14 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 2 remaining series.

| Model | Scenario | 2030 | Unit |
|---|---|---|---|
| GCAM | PR_Baseline | 421.67 | GW |
| GCAM | PR_CurPol_CP | 1,043 | GW |
| GCAM | PR_CurPol_CPO | 461.24 | GW |
| GCAM | PR_CurPol_EI | 868.25 | GW |
| GCAM | PR_NDC_CP | 871.73 | GW |
| GCAM | PR_NDC_CPO | 463.27 | GW |
| TIAM_Grantham | PR_Baseline | 50.07 | GW |
| TIAM_Grantham | PR_CurPol_CP | 246.65 | GW |
| TIAM_Grantham | PR_CurPol_CPO | 278.75 | GW |
| TIAM_Grantham | PR_CurPol_EI | 50.07 | GW |
| TIAM_Grantham | PR_NDC_CP | 50.07 | GW |
| TIAM_Grantham | PR_NDC_CPO | 301.91 | GW |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q033"></a>

## 033. Show 2050 CO2 values in the Post-Glasgow targets study for World.

Category: numeric data and comparisons<br>
Route: `data_query` (study/results catalogue request)<br>
Qwen generations: 0<br>
Time: 0.10 s<br>
Conversation: standalone

**Full answer**

I found these workspaces (studies): Post-Glasgow targets. Ask about a study to see its variables, scenarios, and direct IAM PARIS links.

<a id="q034"></a>

## 034. What is the earliest available global population value?

Category: numeric data and comparisons<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.47 s<br>
Conversation: standalone

**Full answer**

Yes. The loaded IAM PARIS results reports `Population` for region `World` (runtime model label(s): MUSE, gcam, gemini_e3, ices, tiam; 7 scenario(s); years 2005–2100).

<a id="q035"></a>

## 035. Give 2050 methane emissions for Canada under a baseline scenario.

Category: numeric data and comparisons<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.30 s<br>
Conversation: standalone

**Full answer**

Showing results from the study **Global impacts of NDCs**. Also available in: Post-Glasgow targets; Where is the world headed?. Name another study to switch.

### Emissions|CH4 in CAN

Scope: scenario `Baseline`, model `GCAM-PR 5.3`, years `2050`
Unit: `Mt CH4/yr`

Answer:
In 2050, Emissions|CH4 in CAN is 5.91 Mt CH4/yr (GCAM-PR 5.3, Baseline).

**GCAM-PR 5.3 - Baseline**
| Year | Value | Unit |
|------|-------|------|
| 2050 | 5.91 | Mt CH4/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/ndc-aspects/global-impacts-of-ndcs/graphs).

<a id="q036"></a>

## 036. Explain the main structure of GCAM 7.0.

Category: model descriptions and methodology<br>
Route: `model_explanation` (named-model metadata request)<br>
Qwen generations: 0<br>
Time: 0.08 s<br>
Conversation: standalone

**Full answer**

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

<a id="q037"></a>

## 037. What distinguishes WITCH from a simple energy accounting model?

Category: model descriptions and methodology<br>
Route: `data_query` (data request)<br>
Qwen generations: 2<br>
Time: 6.37 s<br>
Conversation: standalone

**Full answer**

Interpreted `WITCH` as `with`.

I found matching model names in the results data, but no metadata description is available. Examples: EnergyPLAN. If you want, ask for values from one of these models.

Reply with a scenario, model, region, or year to narrow the answer.

<a id="q038"></a>

## 038. Who maintains the GEM-E3 model?

Category: model descriptions and methodology<br>
Route: `model_explanation` (model information request)<br>
Qwen generations: 0<br>
Time: 0.23 s<br>
Conversation: standalone

**Full answer**

### GEM-E3

Description:
The GEM-E3 model is a multi-regional, multi-sectoral, recursive dynamic computable general equilibrium (CGE) model which provides details on the macro-economy and its interaction with the environment and the energy system. It is an empirical, large scale model, written entirely in structural form. GEM-E3 allows for a consistent comparative analysis of policy scenarios since it ensures that in all scenarios, the economic system remains in general equilibrium. In addition, it incorporates micro-economic mechanisms and institutional features within a consistent macro-economic framework and avoids the representation of behaviour in reduced form. Particularly valuable are the insights the model provides regarding the distributional aspects of long-term structural adjustments. The GEM-E3 model is extensively used as a tool of policy analysis and impact assessment.

Related model documentation:
- [IAM PARIS model page](https://iamparis.eu/models/75)

<a id="q039"></a>

## 039. Describe the geographic scope of E3ME.

Category: model descriptions and methodology<br>
Route: `model_explanation` (explicit model-family match)<br>
Qwen generations: 0<br>
Time: 0.07 s<br>
Conversation: standalone

**Full answer**

I matched `E3ME` to the only loaded catalogue entry in that family, `E3ME 6.1`.

### E3ME 6.1

Description:
The Post Keynesian energy-environment-economy (E3) macroeconometric model (E3ME). Economic growth in E3ME is demand-driven and supply-constrained, with no assumption of the economy being in full-employment equilibrium. Empirically-validated dynamics (the time path of an economy), is a key feature of E3ME.

Developer:
Cambridge Econometrics

Related model documentation:
- [IAM PARIS model page](https://iamparis.eu/models/18)

<a id="q040"></a>

## 040. How does IMAGE connect land use and climate?

Category: model descriptions and methodology<br>
Route: `model_explanation` (model information request)<br>
Qwen generations: 0<br>
Time: 0.47 s<br>
Conversation: standalone

**Full answer**

### IMAGE

Description:
IMAGE (Integrated Model to Assess the Global Environment) is a global integrated assessment modelling framework representing interactions among society, the biosphere and the climate system.

Developer:
IMAGE team under the authority of PBL Netherlands Environmental Assessment Agency

Model scope:
- energy
- land
- climate
- biodiversity
- human development

Useful for:
- long-term global environmental and sustainable-development pathways
- climate, biodiversity and land-use policy assessment

Related model documentation:
- [Official IMAGE model documentation (PBL)](https://www.pbl.nl/en/image/home)
- [IAM PARIS Models](https://iamparis.eu/models)
- Open the Models page and search for: `IMAGE`

<a id="q041"></a>

## 041. Summarize PROMETHEUS model assumptions.

Category: model descriptions and methodology<br>
Route: `model_explanation` (named-model metadata request)<br>
Qwen generations: 0<br>
Time: 0.06 s<br>
Conversation: standalone

**Full answer**

### PROMETHEUS

Description:
The Prometheus model is a comprehensive world energy model with innovative features: it integrates stochastic relations and so all exogenous variables – parameters are stochastic, following explicit probability distributions, including covariance. Therefore, all model results are also stochastic (probability distributions). Among others, stochastic distributions are used for technology change and learning by doing, as well as for fossil fuel resources, both conventional and unconventional. Prometheus models the world energy system divided into 10 regions and produces yearly projections up to 2050; the update to 2070 is ongoing. The model includes a set of long time series (IEA data mostly) on which econometric estimations are carried out. The model produces projections of energy demand by sector (industry, domestic, transport), power generation (representing about 25 technologies), RES, and hydrogen supply and use. Prometheus puts emphasis on oil and gas resources, while coal is assumed to have rather abundant supplies relative to production prospects in the projection time horizon. The model incorporates uncertainty surrounding the amount of oil and gas resources that are yet to be discovered. Data are mostly calibrated on estimation of resources by USGS. The model produces projections of fossil fuel prices, which depend on demand, supply, technology and resources. As all outputs are probabilistic, fossil fuel price projections take the form of probability distributions. The model includes CO2 emissions and can simulate emission reduction pathways.

Developer:
E3Modelling S.A. https://e3modelling.com/

Assumptions:
Key scenario assumptions for PROMETHEUS include socioeconomics (population, labour participation, and GDP); energy technology characteristics (e.g. costs, performance); energy and other resources & potentials, such as fossil fuels, wind, solar, uranium; and policies, including emissions constraints, renewable portfolio standards, etc.\nKey scenario results (outputs) from PROMETHEUS include an analysis of the energy system (energy demands, flows, technology deployments, and prices throughout); prices of energy products; and emissions for the major greenhouse gases \nOutcomes of PROMETHEUS depend strongly on the assumptions made for socioeconomic, techno-economic, and policy parameters. \nParameters can be revised and updated in the framework of the project, following the feedback of local and national experts (stakeholder engagement), the comparative assessment with other modelling experiences, and the discussion with the partners (modellers).

Related model documentation:
- [IAM PARIS model page](https://iamparis.eu/models/55)

<a id="q042"></a>

## 042. What does MUSE simulate?

Category: model descriptions and methodology<br>
Route: `model_explanation` (model information request)<br>
Qwen generations: 0<br>
Time: 0.25 s<br>
Conversation: standalone

**Full answer**

### MUSE

Description:
MUSE is a partial equilibrium global integrated assessment model. It uses an agent-based modelling approach to model each sector of the energy system.

Related model documentation:
- [IAM PARIS model page](https://iamparis.eu/models/45)

<a id="q043"></a>

## 043. Explain the purpose of ALADIN.

Category: model descriptions and methodology<br>
Route: `model_explanation` (named-model metadata request)<br>
Qwen generations: 0<br>
Time: 0.07 s<br>
Conversation: standalone

**Full answer**

### ALADIN

The loaded profile describes its purpose as follows:
The ALADIN model (ALternative Automobiles Diffusion and INfrastructure) is an agent-based simulation of alternative fuel vehicles purchase decision. It uses driving data from several thousand individual vehicles.

Related model documentation:
- [IAM PARIS model page](https://iamparis.eu/models/6)

<a id="q044"></a>

## 044. How is ATOM used in energy analysis?

Category: model descriptions and methodology<br>
Route: `data_query` (low confidence entity clarification)<br>
Qwen generations: 0<br>
Time: 0.24 s<br>
Conversation: standalone

**Full answer**

I could not confidently match all of the requested wording. The loaded variable catalogue does not represent these term(s) in the top match: `analysis`, `used`. Choose the variable: 1. `Final Energy` (final energy use) 2. `Export|Energy` 3. `Import|Energy` Reply with a number (1-3), or `yes` for option 1.

<a id="q045"></a>

## 045. What sectors are represented by FORECAST?

Category: model descriptions and methodology<br>
Route: `model_explanation` (named-model metadata request)<br>
Qwen generations: 0<br>
Time: 0.06 s<br>
Conversation: standalone

**Full answer**

### FORECAST

Description:
The FORECAST model is designed as a tool that can be used to support strategic decisions. Its main objective is to support the scenario design and analysis for the long-term development of energy demand and greenhouse gas emissions for the industry, residential and tertiary sectors on country level.

Developer:
Fraunhofer ISI

Related model documentation:
- [IAM PARIS model page](https://iamparis.eu/models/27)

<a id="q046"></a>

## 046. Describe MESSAGEix-GLOBIOM in plain language.

Category: model descriptions and methodology<br>
Route: `model_explanation` (named-model metadata request)<br>
Qwen generations: 0<br>
Time: 0.07 s<br>
Conversation: standalone

**Full answer**

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

<a id="q047"></a>

## 047. How does REMIND-MFA differ from REMIND?

Category: model descriptions and methodology<br>
Route: `model_explanation` (multi-model metadata comparison)<br>
Qwen generations: 0<br>
Time: 0.09 s<br>
Conversation: standalone

**Full answer**

### REMIND-MFA vs REMIND

**REMIND-MFA**
- Developer: PIK

**REMIND**
- Developer: PIK
- Systems/sectors: energy, economy, climate, land-use interactions
- Common uses: global and regional mitigation pathways; carbon pricing and policy scenarios; energy-system transformation; technology deployment including renewables, CCS and negative emissions

Sector coverage is not described in the loaded metadata for **REMIND-MFA**, so coverage cannot be compared.

[Browse IAM PARIS model documentation](https://iamparis.eu/models)

<a id="q048"></a>

## 048. Who develops the LEAP model?

Category: model descriptions and methodology<br>
Route: `model_explanation` (named-model metadata request)<br>
Qwen generations: 0<br>
Time: 0.06 s<br>
Conversation: standalone

**Full answer**

### LEAP

`LEAP` is developed by Stockholm Environment Institute.

Related model documentation:
- [IAM PARIS model page](https://iamparis.eu/models/38)

<a id="q049"></a>

## 049. Is TIAM a bottom-up energy model?

Category: model descriptions and methodology<br>
Route: `model_explanation` (named-model metadata request)<br>
Qwen generations: 0<br>
Time: 0.08 s<br>
Conversation: standalone

**Full answer**

### TIAM_Grantham

The loaded catalogue classifies `TIAM_Grantham` as `Partial Equilibrium`, but it does not explicitly establish whether that is a bottom-up model.

Related model documentation:
- [IAM PARIS model page](https://iamparis.eu/models/59)

<a id="q050"></a>

## 050. What is the role of OSeMOSYS-GR?

Category: model descriptions and methodology<br>
Route: `model_explanation` (model information request)<br>
Qwen generations: 0<br>
Time: 0.27 s<br>
Conversation: standalone

**Full answer**

### OSeMOSYS

Description:
The Open Source energy MOdelling SYStem – OSeMOSYS is a long-term energy and resource planning tool that stands for an open, transparent and accessible approach of energy planning. It is a bottom-up linear optimization modelling system. In the last few years OSeMOSYS has already been used to build regional models like the The Electricity Model Base for Africa – TEMBA, and The open source South American Model Base – SAMBA. In the Horizon 2020 EU REEEM project the Open Source energy Modelling Base for the European Union – OSeMBE was developed. OSeMOSYS has a low learning threshold and is available in three modelling languages (Python, GAMS and GNU MathProg), fitting the needs of different communities of users. That allowed its spread in numerous countries worldwide, for academic research, capacity development and support to national planning processes.

Related model documentation:
- [IAM PARIS model page](https://iamparis.eu/models/52)

<a id="q051"></a>

## 051. Describe PRIMES-Buildings and its intended uses.

Category: model descriptions and methodology<br>
Route: `data_query` (unclear deterministic route)<br>
Qwen generations: 2<br>
Time: 1.59 s<br>
Conversation: standalone

**Full answer**

I can answer questions about IAM PARIS results, models and climate-policy terms, but I need a little more to go on. Try for example:
- `CO2 emissions for World in 2050`
- `which scenario has the lowest emissions in 2050?`
- `which models cover transport?`
- `what does net zero mean?`

<a id="q052"></a>

## 052. What sort of model is EnergyPLAN?

Category: model descriptions and methodology<br>
Route: `model_explanation` (named-model metadata request)<br>
Qwen generations: 0<br>
Time: 0.06 s<br>
Conversation: standalone

**Full answer**

### EnergyPLAN

Methodology/model type: Simulation model

Related model documentation:
- [IAM PARIS model page](https://iamparis.eu/models/22)

<a id="q053"></a>

## 053. Compare GCAM and WITCH on methodology.

Category: model descriptions and methodology<br>
Route: `model_explanation` (multi-model metadata comparison)<br>
Qwen generations: 0<br>
Time: 0.09 s<br>
Conversation: standalone

**Full answer**

### GCAM vs WITCH

**GCAM**
- Methodology/model type: Partial equilibrium

**WITCH**
- Methodology/model type: WITCH is an integrated assessment model focused on the interaction between the economy, energy system, climate policy and technological change. It is often used for regional mitigation strategies, policy design and technology innovation analysis.

Comparison focus: method.

[Browse IAM PARIS model documentation](https://iamparis.eu/models)

<a id="q054"></a>

## 054. Contrast IMAGE and GEM-E3 in their sector coverage.

Category: model descriptions and methodology<br>
Route: `data_query` (chat plotting disabled; return numeric table)<br>
Qwen generations: 0<br>
Time: 0.88 s<br>
Conversation: standalone

**Full answer**

I couldn't find `their sector coverage` as a region in the IAM PARIS data, so I can't return results for it. Try one of the available regions, e.g. `AT (Austria)`, `BE (Belgium)`, `BG (Bulgaria)`, `CH (Switzerland)`, `CY (Cyprus)`, or ask `list regions` to see all options.

<a id="q055"></a>

## 055. What are the limitations of the MUSE model description?

Category: model descriptions and methodology<br>
Route: `model_explanation` (named-model metadata request)<br>
Qwen generations: 0<br>
Time: 0.08 s<br>
Conversation: standalone

**Full answer**

### MUSE

Interpretation notes:
- The loaded IAM PARIS profile does not list specific limitations for `MUSE`. Results depend on the scenario assumptions and input data, so check the model documentation and the study's experiment metadata before interpreting a result.

Related model documentation:
- [IAM PARIS model page](https://iamparis.eu/models/45)

<a id="q056"></a>

## 056. Does E3ME use general equilibrium?

Category: model descriptions and methodology<br>
Route: `model_explanation` (named-model metadata request)<br>
Qwen generations: 0<br>
Time: 0.09 s<br>
Conversation: standalone

**Full answer**

### E3ME-FTT

The loaded catalogue classifies `E3ME-FTT` as `Macro-econometric E3 model`, but it does not explicitly establish whether that is a general equilibrium model.

Related model documentation:
- [IAM PARIS model page](https://iamparis.eu/models/19)

<a id="q057"></a>

## 057. How does PROMETHEUS treat technology choices?

Category: model descriptions and methodology<br>
Route: `model_explanation` (named-model metadata request)<br>
Qwen generations: 0<br>
Time: 0.07 s<br>
Conversation: standalone

**Full answer**

### PROMETHEUS

Technology representation:
PROMETHEUS is a technology-rich energy system model that represents most major fossil fuel and low-carbon technologies  that are envisaged to be available for at least the first half of the 21st century, and also including disruptive technological options, including zero and negative-emission technologies (incuding Biomass with CCS and Direct Air Capture) . By simulating the substitution of low or zero-carbon for high-carbon technologies in response to their relative costs, as well as emissions constraints and/or carbon prices, the PROMETHEUS energy system model simulates mitigation through a large set of different measures and technological configurations.

Related model documentation:
- [IAM PARIS model page](https://iamparis.eu/models/55)

<a id="q058"></a>

## 058. What documentation is available for GCAM-PR 5.3?

Category: model descriptions and methodology<br>
Route: `data_query` (extracted data entities)<br>
Qwen generations: 0<br>
Time: 0.70 s<br>
Conversation: standalone

**Full answer**

No. I could not find `Food Availability [per capita]` for model `GCAM-PR 7.0` in the loaded IAM PARIS timeseries data.

<a id="q059"></a>

## 059. Which institution is responsible for the WITCH model?

Category: model descriptions and methodology<br>
Route: `model_explanation` (named-model metadata request)<br>
Qwen generations: 0<br>
Time: 0.08 s<br>
Conversation: standalone

**Full answer**

### WITCH

The developer or maintaining organization for `WITCH` is not described in the loaded IAM PARIS profile.

Related model documentation:
- [IAM PARIS Models](https://iamparis.eu/models)
- Open the Models page and search for: `WITCH`

<a id="q060"></a>

## 060. Compare REMIND and MESSAGEix-GLOBIOM as energy-system tools.

Category: model descriptions and methodology<br>
Route: `model_explanation` (named-model metadata request)<br>
Qwen generations: 0<br>
Time: 0.32 s<br>
Conversation: standalone

**Full answer**

Interpreted `GLOBIOM` as `globio`.

### REMIND

Description:
REMIND is a global integrated assessment model that links macroeconomic development, the energy system, climate policy and technology choices. It is commonly used to assess mitigation pathways, carbon pricing, energy transitions and technology deployment across regions and sectors.

Developer:
PIK

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

<a id="q061"></a>

## 061. Which models have loaded results for Brazil?

Category: coverage and catalogue<br>
Route: `data_query` (topic-wide model availability)<br>
Qwen generations: 0<br>
Time: 0.07 s<br>
Conversation: standalone

**Full answer**

No loaded IAM PARIS variable mentions `loaded results for brazil`, so I cannot list models that report it. Ask `list variables` to see the available variables.

<a id="q062"></a>

## 062. What variables does WITCH report for the EU?

Category: coverage and catalogue<br>
Route: `data_query` (scope-filtered variable availability)<br>
Qwen generations: 0<br>
Time: 1.75 s<br>
Conversation: standalone

**Full answer**

### Variables available for the requested scope

Scope: region `EU`.

**Variables (1229):** Agricultural Demand, Agricultural Demand|Crops|Energy, Agricultural Demand|Crops|Feed, Agricultural Demand|Crops|Food, Agricultural Demand|Crops|Other, Agricultural Demand|Livestock, Agricultural Demand|Livestock|Food, Agricultural Demand|Livestock|Other, Biofuels Investment Share, Biomass Investment Share, CO2 emissions cuts|Absolute, CO2 emissions cuts|Relative, CSP Investment Share, Capacity Additions|Electricity|Biomass, Capacity Additions|Electricity|Biomass|w/ CCS, Capacity Additions|Electricity|Biomass|w/o CCS, Capacity Additions|Electricity|Coal, Capacity Additions|Electricity|Coal|w/ CCS, Capacity Additions|Electricity|Coal|w/o CCS, Capacity Additions|Electricity|Gas … and 1209 more

<a id="q063"></a>

## 063. List the regions covered by MUSE data.

Category: coverage and catalogue<br>
Route: `data_query` (region availability request)<br>
Qwen generations: 0<br>
Time: 0.19 s<br>
Conversation: standalone

**Full answer**

I found regions like AFR, AGO, ARE, ARG, ASEAN, ASI, AT, ATE, AUS and AZE and 151 more. Ask for a variable in any of these regions to see its values.

Showing 10 of 161. Say `show all regions` if you need the full list.

<a id="q064"></a>

## 064. Which years have GDP results for World?

Category: coverage and catalogue<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.39 s<br>
Conversation: standalone

**Full answer**

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

<a id="q065"></a>

## 065. How many scenarios contain solar-capacity data for Germany?

Category: coverage and catalogue<br>
Route: `data_query` (scenario availability request)<br>
Qwen generations: 0<br>
Time: 0.39 s<br>
Conversation: standalone

**Full answer**

There are **15 distinct scenarios** for region `DEU (Germany)`.

<a id="q066"></a>

## 066. Which studies include methane emissions for India?

Category: coverage and catalogue<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.41 s<br>
Conversation: standalone

**Full answer**

Showing results from the study **Where is the world headed?**. Also available in: Post-Glasgow targets; Global impacts of NDCs; Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU. Name another study to switch.

### Emissions|CH4 in IND

Scope: scenario `multiple`, model `multiple`, years `2005-2100`
Unit: `Mt CH4/yr`

Answer:
In 2050, Emissions|CH4 in IND ranges from 42.50 (GEMINI-E3, PR_NDC_EI) to 48.80 Mt CH4/yr (GCAM, PR_NDC_EI) across 13 series; median 46.42. Median change since 2005: +64%.

Summary: 13 series across 2 models and 7 scenarios.
Showing 12 of 13 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 1 remaining series.

| Model | Scenario | 2005 | 2010 | 2015 | 2020 | 2025 | 2030 | 2035 | 2040 | 2045 | 2050 | 2055 | 2060 | 2065 | 2070 | 2075 | 2080 | 2085 | 2090 | 2095 | 2100 | Unit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GCAM | PR_Baseline | 29.03 | 31.98 | 33.95 | 36.67 | 39.00 | 40.69 | 42.61 | 44.32 | 46.75 | 48.37 | 48.38 | 48.35 | 48.11 | 47.73 | 47.68 | 47.44 | 47.08 | 46.33 | 45.44 | 44.41 | Mt CH4/yr |
| GCAM | PR_CurPol_CP | 29.03 | 31.98 | 33.95 | 36.66 | 38.69 | 40.35 | 41.97 | 43.55 | 45.76 | 47.54 | 47.43 | 47.27 | 46.90 | 46.33 | 46.02 | 45.61 | 45.12 | 44.22 | 43.24 | 42.19 | Mt CH4/yr |
| GCAM | PR_CurPol_CPO | 29.03 | 31.98 | 33.95 | 34.87 | 38.78 | 40.50 | 42.10 | 43.71 | 46.00 | 47.63 | 47.53 | 47.37 | 46.99 | 46.42 | 46.10 | 45.71 | 45.20 | 44.28 | 43.30 | 42.28 | Mt CH4/yr |
| GCAM | PR_CurPol_EI | 29.03 | 31.98 | 33.95 | 36.66 | 38.70 | 40.36 | 42.26 | 42.71 | 44.63 | 46.42 | 46.34 | 46.03 | 45.43 | 44.37 | 43.50 | 42.97 | 42.25 | 41.20 | 39.99 | 38.74 | Mt CH4/yr |
| GCAM | PR_NDC_CP | 29.03 | 31.98 | 33.95 | 36.66 | 39.41 | 41.25 | 43.17 | 44.69 | 46.85 | 48.67 | 48.56 | 48.43 | 48.19 | 47.89 | 48.00 | 48.02 | 47.89 | 47.25 | 46.73 | 46.15 | Mt CH4/yr |
| GCAM | PR_NDC_EI | 29.03 | 31.98 | 33.95 | 36.66 | 39.42 | 41.38 | 43.88 | 44.86 | 46.94 | 48.80 | 49.15 | 49.39 | 49.49 | 49.48 | 49.90 | 50.50 | 50.91 | 50.59 | 50.41 | 50.18 | Mt CH4/yr |
| GEMINI-E3 | PR_Baseline | — | — | 33.88 | 35.49 | 37.00 | 38.43 | 40.08 | 41.84 | 43.56 | 44.91 | — | — | — | — | — | — | — | — | — | — | Mt CH4/yr |
| GEMINI-E3 | PR_CurPol_CP | — | — | 33.88 | 35.49 | 36.96 | 38.41 | 39.99 | 41.74 | 43.43 | 44.77 | — | — | — | — | — | — | — | — | — | — | Mt CH4/yr |
| GEMINI-E3 | PR_CurPol_CPO | — | — | 33.88 | 35.49 | 37.00 | 38.17 | 39.67 | 41.34 | 42.94 | 44.19 | — | — | — | — | — | — | — | — | — | — | Mt CH4/yr |
| GEMINI-E3 | PR_CurPol_EI | — | — | 33.88 | 35.49 | 36.96 | 38.41 | 39.69 | 41.06 | 41.85 | 42.53 | — | — | — | — | — | — | — | — | — | — | Mt CH4/yr |
| GEMINI-E3 | PR_NDC_CP | — | — | 33.88 | 35.51 | 36.99 | 38.47 | 39.78 | 41.52 | 43.17 | 44.49 | — | — | — | — | — | — | — | — | — | — | Mt CH4/yr |
| GEMINI-E3 | PR_NDC_EI | — | — | 33.88 | 35.51 | 36.99 | 38.47 | 39.63 | 41.00 | 41.80 | 42.50 | — | — | — | — | — | — | — | — | — | — | Mt CH4/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q067"></a>

## 067. What models report power-sector CO2 for Europe?

Category: coverage and catalogue<br>
Route: `data_query` (topic-wide model availability)<br>
Qwen generations: 0<br>
Time: 1.10 s<br>
Conversation: standalone

**Full answer**

### Models covering Emissions or Energy Supply

**Emissions (28):** ALADIN, ATOM, DLR_LEV, DREEM, E3ME 6.1, E4SMA-EU-TIMES 1.0, EnergyPLAN, eu_times, FORECAST, GCAM 7.0, GCAM 7_0, GCAM-PR 5.3, GCAM-PR 7.0, GEM-E3, GEMINI-E3 7.0, ices, ICES-XPS 1.0, ICES-XPS 1.1, MUSE, NEMESIS 5.0 … and 8 more

**Energy Supply (24):** ALADIN, ATOM, E3ME 6.1, E4SMA-EU-TIMES 1.0, EnergyPLAN, FORECAST, GCAM 7.0, GCAM-PR 5.3, GCAM-PR 7.0, GEM-E3, GEMINI-E3 7.0, ices, ICES-XPS 1.0, ICES-XPS 1.1, MUSE, NEMESIS 5.0, NEMESIS 5.1, OSeMOSYS-GR, PRIMES-Buildings, PROMETHEUS V1 … and 4 more

Combined coverage: 28 distinct model(s) of the 33 with loaded results.

<a id="q068"></a>

## 068. Which variables are available for Nigeria?

Category: coverage and catalogue<br>
Route: `data_query` (scope-filtered variable availability)<br>
Qwen generations: 0<br>
Time: 0.58 s<br>
Conversation: standalone

**Full answer**

### Variables available for the requested scope

Scope: region `Nigeria`.

**Variables (156):** Biodiversity|Conservation Costs, Biodiversity|Potentially Disappeared Fraction of species, Employment, Employment|Agricultural and Unskilled, Employment|Clerks, Employment|Officials and Managers, Employment|Service and Shop Workers, Employment|Technicians, Expenditure|Household|Total, Expenditure|Household|Total|Clothing and footwear, Expenditure|Household|Total|Communication, Expenditure|Household|Total|Food beverages and tobacco, Expenditure|Household|Total|Fuels and power, Expenditure|Household|Total|Household equipment and operation, Expenditure|Household|Total|Housing and water charges, Expenditure|Household|Total|Medical care and health, Expenditure|Household|Total|Miscellaneous goods and services, Expenditure|Household|Total|Operation of vehicles, Expenditure|Household|Total|Purchase of vehicles, Expenditure|Household|Total|Recreational services … and 136 more

<a id="q069"></a>

## 069. What scenarios are loaded for the Post-Glasgow targets study?

Category: coverage and catalogue<br>
Route: `general_qa` (explicit site navigation request)<br>
Qwen generations: 0<br>
Time: 0.53 s<br>
Conversation: standalone

**Full answer**

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

<a id="q070"></a>

## 070. Which regions contain carbon-price projections?

Category: coverage and catalogue<br>
Route: `data_query` (scoped availability request)<br>
Qwen generations: 0<br>
Time: 0.18 s<br>
Conversation: standalone

**Full answer**

### Availability for Price|Carbon

**Regions (190):** AFR, ARG, ASEAN, ASI, AT, ATE, AUS, AUT, Africa_Eastern, Africa_Northern, Africa_Southern, Africa_Western, Argentina, Australia, Australia_NZ, BE, BEL, BE_LU_NL, BG, BGD … and 170 more

Recorded unit: `multiple`.

<a id="q071"></a>

## 071. What is the year span for GCAM-PR 5.3 results?

Category: coverage and catalogue<br>
Route: `model_explanation` (model information request)<br>
Qwen generations: 0<br>
Time: 0.29 s<br>
Conversation: standalone

**Full answer**

### GCAM-PR 5.3

Description:
GCAM (v2022) is a dynamic-recursive model with technology-rich representations of the economy, energy sector, land use and water linked to a climate model that can be used to explore climate change mitigation policies including carbon taxes, carbon trading, regulations and accelerated deployment of energy technology. Regional population and labour productivity growth assumptions drive the energy, land-use and water systems employing numerous technology options to produce, transform and provide energy services, as well as to produce agriculture and forest products, and to determine land use and land use cover.

Related model documentation:
- [IAM PARIS model page](https://iamparis.eu/models/29)

<a id="q072"></a>

## 072. Which models include nuclear electricity output?

Category: coverage and catalogue<br>
Route: `data_query` (topic-wide model availability)<br>
Qwen generations: 0<br>
Time: 0.97 s<br>
Conversation: standalone

**Full answer**

### Models covering Energy Supply

24 model(s) of the 33 with loaded results report at least one energy supply variable in IAM PARIS:

ALADIN, ATOM, E3ME 6.1, E4SMA-EU-TIMES 1.0, EnergyPLAN, FORECAST, GCAM 7.0, GCAM-PR 5.3, GCAM-PR 7.0, GEM-E3, GEMINI-E3 7.0, ices, ICES-XPS 1.0, ICES-XPS 1.1, MUSE, NEMESIS 5.0, NEMESIS 5.1, OSeMOSYS-GR, PRIMES-Buildings, PROMETHEUS V1 … and 4 more

Ask for a specific model (e.g. `tell me about ALADIN`) or a data query (e.g. `energy supply emissions for Europe`) to go deeper.

<a id="q073"></a>

## 073. How many variables are in the EU path to net zero study?

Category: coverage and catalogue<br>
Route: `data_query` (variable availability request)<br>
Qwen generations: 0<br>
Time: 0.19 s<br>
Conversation: standalone

**Full answer**

There are **710 distinct variables** in the loaded database.

<a id="q074"></a>

## 074. Does PROMETHEUS have population results?

Category: coverage and catalogue<br>
Route: `data_query` (extracted data entities)<br>
Qwen generations: 0<br>
Time: 0.40 s<br>
Conversation: standalone

**Full answer**

No. I could not find `Population` in region `USA (United States)` for model `PROMETHEUS` in the loaded IAM PARIS timeseries data. Models that do have this slice include: E3ME, GCAM 7.0, GCAM-PR 5.3, GEM-E3, GEMINI-E3 7.0.

<a id="q075"></a>

## 075. Which regions have hydrogen-capacity data?

Category: coverage and catalogue<br>
Route: `data_query` (data request)<br>
Qwen generations: 2<br>
Time: 3.68 s<br>
Conversation: standalone

**Full answer**

### What I can help you with

- **Models:** 75 available. Examples: AIM/Enduse India 3.3, ALADIN, ATOM, CHANCE.
- **Variables:** 2046 available. Examples: Agricultural Demand|Crops|Energy, Average Annual Investment|Energy Supply|Electricity|Battery|Rooftop, Average Annual Investment|Energy Supply|Electricity|Battery|Utility, Average Annual Investment|Energy Supply|Electricity|Biomass.
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

<a id="q076"></a>

## 076. What are the available model names for wind generation?

Category: coverage and catalogue<br>
Route: `data_query` (topic-wide model availability)<br>
Qwen generations: 0<br>
Time: 1.01 s<br>
Conversation: standalone

**Full answer**

### Models covering Renewables or Energy Supply

**Renewables (25):** ALADIN, ATOM, DREEM, E3ME 6.1, E4SMA-EU-TIMES 1.0, EnergyPLAN, FORECAST, GCAM 7.0, GCAM-PR 5.3, GCAM-PR 7.0, GEM-E3, GEMINI-E3 7.0, ices, ICES-XPS 1.0, ICES-XPS 1.1, MUSE, NEMESIS 5.0, NEMESIS 5.1, OSeMOSYS-GR, PRIMES-Buildings … and 5 more

**Energy Supply (24):** ALADIN, ATOM, E3ME 6.1, E4SMA-EU-TIMES 1.0, EnergyPLAN, FORECAST, GCAM 7.0, GCAM-PR 5.3, GCAM-PR 7.0, GEM-E3, GEMINI-E3 7.0, ices, ICES-XPS 1.0, ICES-XPS 1.1, MUSE, NEMESIS 5.0, NEMESIS 5.1, OSeMOSYS-GR, PRIMES-Buildings, PROMETHEUS V1 … and 4 more

Combined coverage: 25 distinct model(s) of the 33 with loaded results.

<a id="q077"></a>

## 077. Which scenarios report emissions for Canada?

Category: coverage and catalogue<br>
Route: `data_query` (scoped availability request)<br>
Qwen generations: 0<br>
Time: 0.43 s<br>
Conversation: standalone

**Full answer**

### Availability for Emissions|CO2

**Scenarios (4):** Baseline, CP_EI, NDC_EI, NDC_LTT

Recorded unit: `Mt CO2/yr`.

<a id="q078"></a>

## 078. What data categories exist for Brazil?

Category: coverage and catalogue<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 1.18 s<br>
Conversation: standalone

**Full answer**

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

<a id="q079"></a>

## 079. Are there any projections beyond 2080?

Category: coverage and catalogue<br>
Route: `data_query` (chat plotting disabled; return numeric table)<br>
Qwen generations: 1<br>
Time: 6.88 s<br>
Conversation: standalone

**Full answer**

I can answer questions about IAM PARIS results, models and climate-policy terms, but I need a little more to go on. Try for example:
- `CO2 emissions for World in 2050`
- `which scenario has the lowest emissions in 2050?`
- `which models cover transport?`
- `what does net zero mean?`

<a id="q080"></a>

## 080. Which models provide coal primary-energy data for China?

Category: coverage and catalogue<br>
Route: `data_query` (topic-wide model availability)<br>
Qwen generations: 0<br>
Time: 1.06 s<br>
Conversation: standalone

**Full answer**

### Models covering Energy Demand

22 model(s) of the 33 with loaded results report at least one energy demand variable in IAM PARIS:

ALADIN, E3ME 6.1, E4SMA-EU-TIMES 1.0, EnergyPLAN, FORECAST, GCAM 7.0, GCAM-PR 5.3, GCAM-PR 7.0, GEMINI-E3 7.0, ices, ICES-XPS 1.0, ICES-XPS 1.1, MUSE, NEMESIS 5.0, NEMESIS 5.1, OSeMOSYS-GR, PRIMES-Buildings, PROMETHEUS V1, tiam, TIAM_Grantham 3.2 … and 2 more

Ask for a specific model (e.g. `tell me about ALADIN`) or a data query (e.g. `energy demand emissions for Europe`) to go deeper.

<a id="q081"></a>

## 081. List the studies that include Africa.

Category: coverage and catalogue<br>
Route: `data_query` (study/results catalogue request)<br>
Qwen generations: 0<br>
Time: 0.06 s<br>
Conversation: standalone

**Full answer**

I found these workspaces (studies): AFOLU transformation; Behavioural changes for climate policy; Buildings Transformation; Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU; Cost of capital evolution effects on regional mitigation pathways; COVID recovery packages; Decarbonization potentials of energy citizens; Decipher modelling studies; Enhancing renewable energy system metrics; EU path to net zero; Global impacts of NDCs; Implications of EU steel industry relocation and 9 more. Ask about a study to see its variables, scenarios, and direct IAM PARIS links.

Showing 12 of 21. Say `show all workspaces` if you need the full list.

<a id="q082"></a>

## 082. What model versions are present in the results cache?

Category: coverage and catalogue<br>
Route: `data_query` (low confidence entity clarification)<br>
Qwen generations: 2<br>
Time: 2.53 s<br>
Conversation: standalone

**Full answer**

I couldn't find a model called `model` in the IAM PARIS catalogue. Ask `list models` to see the available models.

<a id="q083"></a>

## 083. Which electricity-price variables are recorded?

Category: coverage and catalogue<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.93 s<br>
Conversation: standalone

**Full answer**

Showing results from the study **EU path to net zero**. Also available in: Where is the world headed?; Global impacts of NDCs; Post-Glasgow targets; Where is the EU headed?. Name another study to switch.

### Price|Secondary Energy|Electricity in World

Scope: scenario `multiple`, model `GCAM`, years `2005-2100`
Unit: `US$2010/GJ or local currency/GJ`

Answer:
In 2100, Price|Secondary Energy|Electricity in World ranges from 22.37 (GCAM, Baseline) to 25.12 US$2010/GJ or local currency/GJ (GCAM, CP_EI) across 2 series; median 23.75. Median change since 2005: +10%.

**GCAM - Baseline**
| Year | Value | Unit |
|------|-------|------|
| 2005 | 21.64 | US$2010/GJ or local currency/GJ |
| 2010 | 25.52 | US$2010/GJ or local currency/GJ |
| 2015 | 24.53 | US$2010/GJ or local currency/GJ |
| 2020 | 22.69 | US$2010/GJ or local currency/GJ |
| 2025 | 22.29 | US$2010/GJ or local currency/GJ |
| 2030 | 21.54 | US$2010/GJ or local currency/GJ |
| 2035 | 20.96 | US$2010/GJ or local currency/GJ |
| 2040 | 20.84 | US$2010/GJ or local currency/GJ |
| 2045 | 21.31 | US$2010/GJ or local currency/GJ |
| 2050 | 21.97 | US$2010/GJ or local currency/GJ |
| 2055 | 22.13 | US$2010/GJ or local currency/GJ |
| 2060 | 22.01 | US$2010/GJ or local currency/GJ |
| 2065 | 21.90 | US$2010/GJ or local currency/GJ |
| 2070 | 21.90 | US$2010/GJ or local currency/GJ |
| 2075 | 22.03 | US$2010/GJ or local currency/GJ |
| 2080 | 22.31 | US$2010/GJ or local currency/GJ |
| 2085 | 22.25 | US$2010/GJ or local currency/GJ |
| 2090 | 22.38 | US$2010/GJ or local currency/GJ |
| 2095 | 22.20 | US$2010/GJ or local currency/GJ |
| 2100 | 22.37 | US$2010/GJ or local currency/GJ |

**GCAM - CP_EI**
| Year | Value | Unit |
|------|-------|------|
| 2005 | 21.64 | US$2010/GJ or local currency/GJ |
| 2010 | 25.52 | US$2010/GJ or local currency/GJ |
| 2015 | 24.53 | US$2010/GJ or local currency/GJ |
| 2020 | 22.87 | US$2010/GJ or local currency/GJ |
| 2025 | 22.35 | US$2010/GJ or local currency/GJ |
| 2030 | 21.61 | US$2010/GJ or local currency/GJ |
| 2035 | 22.44 | US$2010/GJ or local currency/GJ |
| 2040 | 23.33 | US$2010/GJ or local currency/GJ |
| 2045 | 24.63 | US$2010/GJ or local currency/GJ |
| 2050 | 25.13 | US$2010/GJ or local currency/GJ |
| 2055 | 25.23 | US$2010/GJ or local currency/GJ |
| 2060 | 25.19 | US$2010/GJ or local currency/GJ |
| 2065 | 25.28 | US$2010/GJ or local currency/GJ |
| 2070 | 25.27 | US$2010/GJ or local currency/GJ |
| 2075 | 25.66 | US$2010/GJ or local currency/GJ |
| 2080 | 25.76 | US$2010/GJ or local currency/GJ |
| 2085 | 25.57 | US$2010/GJ or local currency/GJ |
| 2090 | 25.40 | US$2010/GJ or local currency/GJ |
| 2095 | 25.21 | US$2010/GJ or local currency/GJ |
| 2100 | 25.12 | US$2010/GJ or local currency/GJ |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/eu-path-to-net-zero/graphs).

Reply with a scenario, model, region, or year to narrow the answer.

<a id="q084"></a>

## 084. How many models have results for France?

Category: coverage and catalogue<br>
Route: `data_query` (sector-filtered model availability request)<br>
Qwen generations: 0<br>
Time: 0.09 s<br>
Conversation: standalone

**Full answer**

No loaded IAM PARIS variable mentions `results for france`, so I cannot list models that report it. Ask `list variables` to see the available variables.

<a id="q085"></a>

## 085. Do the loaded results include land-cover variables?

Category: coverage and catalogue<br>
Route: `data_query` (data request)<br>
Qwen generations: 2<br>
Time: 2.95 s<br>
Conversation: standalone

**Full answer**

Yes. The loaded IAM PARIS results reports `Land Cover` (runtime model label(s): GCAM 7.0, GCAM-PR 7.0; 16 scenario(s); years 2005–2100).

<a id="q086"></a>

## 086. Why might a model show falling coal use but rising gas use?

Category: conceptual explanations for Qwen<br>
Route: `data_query` (model availability request)<br>
Qwen generations: 0<br>
Time: 0.13 s<br>
Conversation: standalone

**Full answer**

There are 75 models available in the IAM PARIS model catalogue; 33 of them have loaded results. Models with the most results: MUSE, E3ME, GCAM-PR 5.3, NEMESIS 5.1, E4SMA-EU-TIMES 1.0, GCAM 7.0, GCAM, GEM-E3. Ask `tell me about [model name]` for details, or filter by topic, e.g. `which models cover transport`.

<a id="q087"></a>

## 087. How can two net-zero scenarios reach the same target with different technology mixes?

Category: conceptual explanations for Qwen<br>
Route: `general_qa` (conceptual climate explanation)<br>
Qwen generations: 1<br>
Time: 2.11 s<br>
Conversation: standalone

**Full answer**

Two net-zero scenarios can reach the same target with different technology mixes by adjusting the assumptions about the generation mix, carbon prices, and technology costs. For example, if one scenario assumes a higher proportion of renewable energy sources (like wind or solar) and another scenario uses a mix of fossil fuels, the resulting emissions would differ, even if the overall energy mix is the same. This variation in technology mix directly impacts the final energy output, which can be adjusted to meet the same target.

<a id="q088"></a>

## 088. What drives uncertainty in future hydrogen demand?

Category: conceptual explanations for Qwen<br>
Route: `general_qa` (conceptual climate explanation)<br>
Qwen generations: 1<br>
Time: 0.31 s<br>
Conversation: standalone

**Full answer**

The uncertainty in future hydrogen demand is driven by hydrogen production costs and the cost of low-carbon electricity.

<a id="q089"></a>

## 089. Why can carbon prices rise while emissions decline?

Category: conceptual explanations for Qwen<br>
Route: `general_qa` (conceptual climate explanation)<br>
Qwen generations: 1<br>
Time: 1.34 s<br>
Conversation: standalone

**Full answer**

Carbon prices can rise while emissions decline because of the **integrated assessment model (IAM)** framework, which considers how carbon pricing affects energy transitions. In IAMs, a higher carbon price signals a shift towards low-carbon energy sources, such as electrification and energy efficiency, and reduces fossil fuel demand. This leads to lower emissions, even if the price rises. The economic effect depends on the model's assumptions about how carbon revenues are used, which can influence the overall impact on emissions.

<a id="q090"></a>

## 090. How should I interpret a negative land-use CO2 value?

Category: conceptual explanations for Qwen<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.68 s<br>
Conversation: standalone

**Full answer**

Showing results from the study **Where is the world headed?**. Also available in: EU path to net zero; Post-Glasgow targets; Global impacts of NDCs; Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU; Tech-constrained EU pathways to net zero; Behavioural changes for climate policy; Where is the EU headed?; Enhancing renewable energy system metrics. Name another study to switch.

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

<a id="q091"></a>

## 091. What makes electricity-price projections sensitive to storage costs?

Category: conceptual explanations for Qwen<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.50 s<br>
Conversation: standalone

**Full answer**

Showing results from the study **EU path to net zero**. Also available in: Where is the world headed?; Global impacts of NDCs; Post-Glasgow targets; Where is the EU headed?. Name another study to switch.

### Price|Secondary Energy|Electricity in World

Scope: scenario `multiple`, model `GCAM`, years `2005-2100`
Unit: `US$2010/GJ or local currency/GJ`

Answer:
In 2100, Price|Secondary Energy|Electricity in World ranges from 22.37 (GCAM, Baseline) to 25.12 US$2010/GJ or local currency/GJ (GCAM, CP_EI) across 2 series; median 23.75. Median change since 2005: +10%.

**GCAM - Baseline**
| Year | Value | Unit |
|------|-------|------|
| 2005 | 21.64 | US$2010/GJ or local currency/GJ |
| 2010 | 25.52 | US$2010/GJ or local currency/GJ |
| 2015 | 24.53 | US$2010/GJ or local currency/GJ |
| 2020 | 22.69 | US$2010/GJ or local currency/GJ |
| 2025 | 22.29 | US$2010/GJ or local currency/GJ |
| 2030 | 21.54 | US$2010/GJ or local currency/GJ |
| 2035 | 20.96 | US$2010/GJ or local currency/GJ |
| 2040 | 20.84 | US$2010/GJ or local currency/GJ |
| 2045 | 21.31 | US$2010/GJ or local currency/GJ |
| 2050 | 21.97 | US$2010/GJ or local currency/GJ |
| 2055 | 22.13 | US$2010/GJ or local currency/GJ |
| 2060 | 22.01 | US$2010/GJ or local currency/GJ |
| 2065 | 21.90 | US$2010/GJ or local currency/GJ |
| 2070 | 21.90 | US$2010/GJ or local currency/GJ |
| 2075 | 22.03 | US$2010/GJ or local currency/GJ |
| 2080 | 22.31 | US$2010/GJ or local currency/GJ |
| 2085 | 22.25 | US$2010/GJ or local currency/GJ |
| 2090 | 22.38 | US$2010/GJ or local currency/GJ |
| 2095 | 22.20 | US$2010/GJ or local currency/GJ |
| 2100 | 22.37 | US$2010/GJ or local currency/GJ |

**GCAM - CP_EI**
| Year | Value | Unit |
|------|-------|------|
| 2005 | 21.64 | US$2010/GJ or local currency/GJ |
| 2010 | 25.52 | US$2010/GJ or local currency/GJ |
| 2015 | 24.53 | US$2010/GJ or local currency/GJ |
| 2020 | 22.87 | US$2010/GJ or local currency/GJ |
| 2025 | 22.35 | US$2010/GJ or local currency/GJ |
| 2030 | 21.61 | US$2010/GJ or local currency/GJ |
| 2035 | 22.44 | US$2010/GJ or local currency/GJ |
| 2040 | 23.33 | US$2010/GJ or local currency/GJ |
| 2045 | 24.63 | US$2010/GJ or local currency/GJ |
| 2050 | 25.13 | US$2010/GJ or local currency/GJ |
| 2055 | 25.23 | US$2010/GJ or local currency/GJ |
| 2060 | 25.19 | US$2010/GJ or local currency/GJ |
| 2065 | 25.28 | US$2010/GJ or local currency/GJ |
| 2070 | 25.27 | US$2010/GJ or local currency/GJ |
| 2075 | 25.66 | US$2010/GJ or local currency/GJ |
| 2080 | 25.76 | US$2010/GJ or local currency/GJ |
| 2085 | 25.57 | US$2010/GJ or local currency/GJ |
| 2090 | 25.40 | US$2010/GJ or local currency/GJ |
| 2095 | 25.21 | US$2010/GJ or local currency/GJ |
| 2100 | 25.12 | US$2010/GJ or local currency/GJ |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/eu-path-to-net-zero/graphs).

Reply with a scenario, model, region, or year to narrow the answer.

<a id="q092"></a>

## 092. Why do IAMs report several regional aggregates for the same country?

Category: conceptual explanations for Qwen<br>
Route: `general_qa` (conceptual climate explanation)<br>
Qwen generations: 1<br>
Time: 0.51 s<br>
Conversation: standalone

**Full answer**

IAMs report regional aggregates for the same country because they capture differences in energy transitions across regions, such as income levels, energy resources, and policy frameworks. These variations lead to distinct regional pathways, which are reflected in the aggregated data.

<a id="q093"></a>

## 093. How does a carbon budget differ from an annual emissions target?

Category: conceptual explanations for Qwen<br>
Route: `general_qa` (conceptual climate explanation)<br>
Qwen generations: 1<br>
Time: 1.20 s<br>
Conversation: standalone

**Full answer**

A carbon budget is the maximum cumulative net CO₂ emissions allowed while limiting warming to a given level (e.g., 1.5°C or 2°C), and it accounts for a reference date. An annual emissions target is a specific estimate of emissions for a given year, based on assumptions like climate policy, population, and economic growth. The key difference is that the carbon budget sets a long-term upper limit, while the annual target is a specific year's estimate.

<a id="q094"></a>

## 094. What assumptions make direct air capture attractive in a scenario?

Category: conceptual explanations for Qwen<br>
Route: `data_query` (extracted data entities)<br>
Qwen generations: 2<br>
Time: 8.81 s<br>
Conversation: standalone

**Full answer**

I couldn't find `a scenario` as a region in the IAM PARIS data, so I can't return results for it.

Try one of the available regions, e.g. `AT (Austria)`, `BE (Belgium)`, `BG (Bulgaria)`, `CH (Switzerland)`, `CY (Cyprus)`, or ask `list regions` to see all options.

<a id="q095"></a>

## 095. Why might economic growth and final energy use move in opposite directions?

Category: conceptual explanations for Qwen<br>
Route: `general_qa` (conceptual climate explanation)<br>
Qwen generations: 1<br>
Time: 1.06 s<br>
Conversation: standalone

**Full answer**

Economic growth and final energy use may move in opposite directions due to the interplay between population, income per person, and energy intensity. As economic growth increases, both energy demand and emissions rise, while lower energy intensity and higher carbon intensity could offset this trend. Conversely, if economic growth slows, energy use might decrease, and emissions could also decline.

<a id="q096"></a>

## 096. How do integrated assessment models represent policy uncertainty?

Category: conceptual explanations for Qwen<br>
Route: `general_qa` (conceptual climate explanation)<br>
Qwen generations: 1<br>
Time: 0.82 s<br>
Conversation: standalone

**Full answer**

Integrated assessment models represent policy uncertainty by incorporating assumptions about socioeconomic development, technology, and policy behavior, which can lead to varied outcomes. This uncertainty is reflected in model intercomparisons, where different models on the same scenario show varying results, indicating potential structural differences.

<a id="q097"></a>

## 097. What is the difference between installed capacity and electricity generation?

Category: conceptual explanations for Qwen<br>
Route: `data_query` (chat plotting disabled; return numeric table)<br>
Qwen generations: 0<br>
Time: 0.75 s<br>
Conversation: standalone

**Full answer**

Showing results from the study **Where is the world headed?**. Also available in: EU path to net zero; Post-Glasgow targets; Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU; Global impacts of NDCs; Tech-constrained EU pathways to net zero; Implications of EU steel industry relocation; Behavioural changes for climate policy; Where is the EU headed?; People-centered national energy systems; Decarbonization potentials of energy citizens; Enhancing renewable energy system metrics. Name another study to switch.

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

<a id="q098"></a>

## 098. Why can historical values differ across model submissions?

Category: conceptual explanations for Qwen<br>
Route: `general_qa` (conceptual climate explanation)<br>
Qwen generations: 1<br>
Time: 0.63 s<br>
Conversation: standalone

**Full answer**

Historical values can differ across model submissions due to variations in assumptions about future policy, population, economic growth, technology costs, and resource availability, as well as differences in model structures and baseline emissions.

<a id="q099"></a>

## 099. How should I compare carbon prices reported in different currency-year units?

Category: conceptual explanations for Qwen<br>
Route: `general_qa` (conceptual climate explanation)<br>
Qwen generations: 1<br>
Time: 1.08 s<br>
Conversation: standalone

**Full answer**

To compare carbon prices reported in different currency-year units, ensure consistency in the time frames and units used across models. For example, if one model uses a specific time period (e.g., 2020–2025) and another uses a different timeframe, align the units or specify the time frame explicitly. If the context does not provide specific details, you can state that the comparison should be based on the same time frame and currency.

<a id="q100"></a>

## 100. What does a large spread between model results tell us?

Category: conceptual explanations for Qwen<br>
Route: `data_query` (chat plotting disabled; return numeric table)<br>
Qwen generations: 3<br>
Time: 2.42 s<br>
Conversation: standalone

**Full answer**

I couldn't match that to a known model. Try `list models` to see available options.

<a id="q101"></a>

## 101. Why is a baseline pathway not necessarily a forecast?

Category: conceptual explanations for Qwen<br>
Route: `data_query` (extracted data entities)<br>
Qwen generations: 0<br>
Time: 0.52 s<br>
Conversation: standalone

**Full answer**

I can answer questions about IAM PARIS results, models and climate-policy terms, but I need a little more to go on. Try for example:
- `CO2 emissions for World in 2050`
- `which scenario has the lowest emissions in 2050?`
- `which models cover transport?`
- `what does net zero mean?`

<a id="q102"></a>

## 102. How could faster battery learning change a power-sector pathway?

Category: conceptual explanations for Qwen<br>
Route: `general_qa` (conceptual climate explanation)<br>
Qwen generations: 1<br>
Time: 0.56 s<br>
Conversation: standalone

**Full answer**

Faster battery learning could enable the power sector to adopt cleaner technologies more efficiently, thereby accelerating decarbonization efforts. This would allow for quicker implementation of low-carbon power sources and electrification strategies, which are critical for decarbonizing the entire energy system.

<a id="q103"></a>

## 103. What causes transport emissions to persist in deep-mitigation scenarios?

Category: conceptual explanations for Qwen<br>
Route: `general_qa` (conceptual climate explanation)<br>
Qwen generations: 1<br>
Time: 1.06 s<br>
Conversation: standalone

**Full answer**

Transport emissions persist in deep-mitigation scenarios due to the difficulty in decarbonizing these sectors. The main reasons include the reliance on fossil fuels with limited alternatives, slow infrastructure changes, and the relatively short lifespan of current transportation systems (cars lasting 15–18 years, aircraft and ships lasting longer). These factors make it challenging to significantly reduce transport emissions under such scenarios.

<a id="q104"></a>

## 104. Why do some models use more bioenergy than others?

Category: conceptual explanations for Qwen<br>
Route: `general_qa` (conceptual climate explanation)<br>
Qwen generations: 1<br>
Time: 0.57 s<br>
Conversation: standalone

**Full answer**

Some models use more bioenergy than others due to differences in how they account for land-use changes and sustainability constraints. These variations arise from varying assumptions about the supply of biomass, competition with food production, and the extent of CO2 removal through CCS.

<a id="q105"></a>

## 105. How do scenario assumptions affect projected GDP?

Category: conceptual explanations for Qwen<br>
Route: `general_qa` (conceptual climate explanation)<br>
Qwen generations: 1<br>
Time: 1.28 s<br>
Conversation: standalone

**Full answer**

Scenario assumptions affect projected GDP by influencing the baseline assumptions used in the IAM PARIS model. For example, a scenario with current policies assumes existing climate policies, while an NDC scenario assumes countries meet their pledges. Scenarios can also vary in population and economic growth assumptions, and different models may yield different GDP outcomes due to their structure and assumptions. The exact impact depends on the specific assumptions and models used in the scenario.

<a id="q106"></a>

## 106. What is the practical meaning of net-negative CO2 after 2050?

Category: conceptual explanations for Qwen<br>
Route: `data_query` (extracted data entities)<br>
Qwen generations: 0<br>
Time: 0.90 s<br>
Conversation: standalone

**Full answer**

Showing results from the study **Where is the world headed?**. Also available in: Post-Glasgow targets; Global impacts of NDCs; Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU; Behavioural changes for climate policy; EU path to net zero; Tech-constrained EU pathways to net zero; Where is the EU headed?; Enhancing renewable energy system metrics. Name another study to switch.

### Emissions|CO2 in World

Scope: scenario `multiple`, model `multiple`, years `2051-latest available`
Unit: `Mt CO2/yr`

Answer:
In 2100, Emissions|CO2 in World ranges from -766.86 (GCAM, NDC_LTT) to 62,101 Mt CO2/yr (GCAM, PR_Baseline) across 22 series; median 23,148. Median change since 2055: -7%.

Summary: 22 series across 2 models and 12 scenarios.
Showing 12 of 22 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 10 remaining series.

| Model | Scenario | 2055 | 2060 | 2065 | 2070 | 2075 | 2080 | 2085 | 2090 | 2095 | 2100 | Unit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GCAM | Baseline | 46,708 | 48,206 | 49,791 | 51,521 | 53,322 | 54,770 | 56,170 | 57,563 | 58,673 | 60,068 | Mt CO2/yr |
| GCAM | CP_EI | 26,244 | 24,349 | 22,565 | 20,869 | 19,172 | 17,514 | 15,927 | 14,464 | 13,087 | 11,834 | Mt CO2/yr |
| GCAM | NDC_EI | 15,580 | 13,341 | 11,396 | 9,677 | 8,137 | 6,717 | 5,482 | 4,442 | 3,571 | 2,842 | Mt CO2/yr |
| GCAM | NDC_LTT | 4,316 | 871.19 | 669.19 | 59.16 | -310.14 | -114.52 | -451.30 | -589.76 | -661.35 | -766.86 | Mt CO2/yr |
| GCAM | PR_Baseline | 48,176 | 49,647 | 51,430 | 53,411 | 54,919 | 56,024 | 57,552 | 59,032 | 60,469 | 62,101 | Mt CO2/yr |
| GCAM | PR_CurPol_CP | 38,926 | 39,800 | 41,260 | 42,548 | 43,526 | 44,277 | 45,175 | 45,925 | 46,625 | 47,266 | Mt CO2/yr |
| GCAM | PR_CurPol_CPO | 39,108 | 39,865 | 41,255 | 42,501 | 43,408 | 44,090 | 44,971 | 45,679 | 46,314 | 46,840 | Mt CO2/yr |
| MUSE | baseline | 30,979 | 31,285 | 32,420 | 33,030 | 34,518 | 35,378 | 36,141 | 36,237 | 37,010 | 37,729 | Mt CO2/yr |
| MUSE | CP_EI | 22,850 | 21,458 | 20,621 | 19,791 | 18,914 | 18,091 | 18,210 | 17,693 | 18,037 | 18,122 | Mt CO2/yr |
| MUSE | NDC_EI | 18,772 | 17,549 | 17,234 | 16,328 | 15,290 | 14,893 | 14,669 | 14,811 | 15,232 | 15,044 | Mt CO2/yr |
| MUSE | NDC_LTT | 10,260 | 7,287 | 6,283 | 5,079 | 3,632 | 2,855 | 3,569 | 4,166 | 4,822 | 5,288 | Mt CO2/yr |
| MUSE | PR_Baseline | 26,902 | 24,364 | 23,828 | 24,252 | 24,392 | 24,597 | 24,559 | 24,441 | 24,608 | 24,998 | Mt CO2/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q107"></a>

## 107. Why might a country reach net zero later than the global average?

Category: conceptual explanations for Qwen<br>
Route: `general_qa` (conceptual climate explanation)<br>
Qwen generations: 1<br>
Time: 0.75 s<br>
Conversation: standalone

**Full answer**

A country might reach net zero later than the global average due to differences in energy transitions, such as varying income levels, energy resources, and policy frameworks. These factors influence how countries prioritize and implement energy transitions, leading to different timelines for achieving net-zero goals.

<a id="q108"></a>

## 108. How can demand-side changes lower mitigation costs?

Category: conceptual explanations for Qwen<br>
Route: `general_qa` (conceptual climate explanation)<br>
Qwen generations: 1<br>
Time: 0.60 s<br>
Conversation: standalone

**Full answer**

Demand-side changes, such as shifting to public transport, cycling, or walking, can lower mitigation costs by reducing reliance on energy services and carbon removal technologies. These changes help achieve climate targets with less carbon-intensive solutions.

<a id="q109"></a>

## 109. What should I check before calling one model more optimistic than another?

Category: conceptual explanations for Qwen<br>
Route: `data_query` (chat plotting disabled; return numeric table)<br>
Qwen generations: 3<br>
Time: 3.09 s<br>
Conversation: standalone

**Full answer**

I couldn't match that to a known model. Try `list models` to see available options.

<a id="q110"></a>

## 110. Why can renewable capacity grow even when electricity demand is flat?

Category: conceptual explanations for Qwen<br>
Route: `general_qa` (conceptual climate explanation)<br>
Qwen generations: 1<br>
Time: 1.33 s<br>
Conversation: standalone

**Full answer**

Renewable capacity can grow even when electricity demand is flat because of factors like energy efficiency improvements, electrification of transport and heating, and industrial production, which reduce the overall demand for electricity. In IAM models, these factors are considered as part of the demand projection, allowing for growth in renewable capacity without a sharp increase in electricity prices.

<a id="q111"></a>

## 111. Take me to the page listing IAM PARIS models.

Category: website and documentation links<br>
Route: `general_qa` (explicit site navigation request)<br>
Qwen generations: 0<br>
Time: 0.09 s<br>
Conversation: standalone

**Full answer**

Use these IAM PARIS links for this request: [Models](https://iamparis.eu/models).

<a id="q112"></a>

## 112. Where is the EU path to net zero explorer?

Category: website and documentation links<br>
Route: `general_qa` (explicit site navigation request)<br>
Qwen generations: 0<br>
Time: 0.74 s<br>
Conversation: standalone

**Full answer**

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

<a id="q113"></a>

## 113. Find the Aqueduct entry in the application library.

Category: website and documentation links<br>
Route: `general_qa` (explicit site navigation request)<br>
Qwen generations: 0<br>
Time: 0.07 s<br>
Conversation: standalone

**Full answer**

Use these IAM PARIS links for this request: [Aqueduct](https://iamparis.eu/application_library/474).

<a id="q114"></a>

## 114. Open the project contact form.

Category: website and documentation links<br>
Route: `general_qa` (explicit site navigation request)<br>
Qwen generations: 0<br>
Time: 0.05 s<br>
Conversation: standalone

**Full answer**

Use these IAM PARIS links for this request: [Contact](https://iamparis.eu/contact).

<a id="q115"></a>

## 115. Where are the data stories listed?

Category: website and documentation links<br>
Route: `general_qa` (explicit site navigation request)<br>
Qwen generations: 0<br>
Time: 0.14 s<br>
Conversation: standalone

**Full answer**

Use these IAM PARIS links for this request: [Data Stories](https://iamparis.eu/datastories).

<a id="q116"></a>

## 116. Give me the workspace index link.

Category: website and documentation links<br>
Route: `general_qa` (explicit site navigation request)<br>
Qwen generations: 0<br>
Time: 0.04 s<br>
Conversation: standalone

**Full answer**

I could not match this request to a reliable IAM PARIS link.

<a id="q117"></a>

## 117. How can I reach the model comparison tool?

Category: website and documentation links<br>
Route: `general_qa` (explicit site navigation request)<br>
Qwen generations: 0<br>
Time: 0.04 s<br>
Conversation: standalone

**Full answer**

Use these IAM PARIS links for this request: [Model Comparison](https://iamparis.eu/models/comparison).

<a id="q118"></a>

## 118. Find documentation for the IMAGE framework.

Category: website and documentation links<br>
Route: `general_qa` (explicit site navigation request)<br>
Qwen generations: 0<br>
Time: 0.04 s<br>
Conversation: standalone

**Full answer**

Use these IAM PARIS links for this request: [Models](https://iamparis.eu/models).

<a id="q119"></a>

## 119. Where is the policy catalogue on IAM PARIS?

Category: website and documentation links<br>
Route: `general_qa` (explicit site navigation request)<br>
Qwen generations: 0<br>
Time: 0.03 s<br>
Conversation: standalone

**Full answer**

Use these IAM PARIS links for this request: [Policy Catalogue Interactive Explorer](https://iamparis.eu/datastories/policyCatalog).

<a id="q120"></a>

## 120. Show the transport transformation study page.

Category: website and documentation links<br>
Route: `general_qa` (explicit site navigation request)<br>
Qwen generations: 0<br>
Time: 0.08 s<br>
Conversation: standalone

**Full answer**

Use these IAM PARIS links for this request: [Transportation Transformation](https://iamparis.eu/results/ndc-aspects/transportation-transformation/policy_questions).

<a id="q121"></a>

## 121. Where can I read about the Fit-for-55 comparison study?

Category: website and documentation links<br>
Route: `general_qa` (explicit site navigation request)<br>
Qwen generations: 0<br>
Time: 0.06 s<br>
Conversation: standalone

**Full answer**

Use these IAM PARIS links for this request: [Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU](https://iamparis.eu/results/iam-compact/comparison-of-fit-for-55-policy-and-cost-optimal-scenarios-for-the-eu/policy_questions).

<a id="q122"></a>

## 122. Give me a link to the configurable analysis page.

Category: website and documentation links<br>
Route: `general_qa` (explicit site navigation request)<br>
Qwen generations: 0<br>
Time: 0.03 s<br>
Conversation: standalone

**Full answer**

Use these IAM PARIS links for this request: [Configurable Analysis](https://iamparis.eu/analysis).

<a id="q123"></a>

## 123. Find the Climate Watch application listing.

Category: website and documentation links<br>
Route: `general_qa` (explicit site navigation request)<br>
Qwen generations: 0<br>
Time: 0.04 s<br>
Conversation: standalone

**Full answer**

Use these IAM PARIS links for this request: [Climate Watch](https://iamparis.eu/application_library).

If a direct detail page is not available:
- Open the Application Library and search for: Climate Watch

<a id="q124"></a>

## 124. Where can I access NDC ASPECTS results?

Category: website and documentation links<br>
Route: `general_qa` (explicit site navigation request)<br>
Qwen generations: 0<br>
Time: 0.04 s<br>
Conversation: standalone

**Full answer**

Use these IAM PARIS links for this request: [NDC ASPECTS](https://iamparis.eu/results/ndc-aspects).

<a id="q125"></a>

## 125. Show me the page for the GCAM model.

Category: website and documentation links<br>
Route: `general_qa` (explicit site navigation request)<br>
Qwen generations: 0<br>
Time: 0.04 s<br>
Conversation: standalone

**Full answer**

Use these IAM PARIS links for this request: [Models](https://iamparis.eu/models).

If a direct detail page is not available:
- Open the Models directory and search for: GCAM

<a id="q126"></a>

## 126. Show me energy numbers for Europe.

Category: ambiguity and recovery<br>
Route: `data_query` (low confidence entity clarification)<br>
Qwen generations: 0<br>
Time: 0.44 s<br>
Conversation: standalone

**Full answer**

I could not confidently match all of the requested wording. The loaded variable catalogue does not represent these term(s) in the top match: `energy`. Choose the variable: 1. `Economy|Commercial|Number of buildings` (buildings) 2. `Economy|Residential|Number of households` (buildings) 3. `'Economy|Residential & Commercial|Number of buildings` (buildings) Reply with a number (1-3), or `yes` for option 1.

<a id="q127"></a>

## 127. I need electricity data but I do not know which variable.

Category: ambiguity and recovery<br>
Route: `data_query` (variable availability request)<br>
Qwen generations: 0<br>
Time: 1.30 s<br>
Conversation: standalone

**Full answer**

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

<a id="q128"></a>

## 128. Give me emissions for an unspecified region.

Category: ambiguity and recovery<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.54 s<br>
Conversation: standalone

**Full answer**

I couldn't find `an unspecified region` as a region in the IAM PARIS data, so I can't return results for it. Try one of the available regions, e.g. `EU (European Union)`, `AGO (Angola)`, `ARE (United Arab Emirates)`, `ARG (Argentina)`, `AUS (Australia)`, or ask `list regions` to see all options.

<a id="q129"></a>

## 129. Show CO2 for El Dorado in 2050.

Category: ambiguity and recovery<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.15 s<br>
Conversation: standalone

**Full answer**

I couldn't find `El Dorado` as a region in the loaded IAM PARIS data. Please choose a loaded region using `list regions`; I cannot substitute a different geographic aggregate for the requested comparison.

<a id="q130"></a>

## 130. Is MODEL-X available in the catalogue?

Category: ambiguity and recovery<br>
Route: `general_qa` (explicit site navigation request)<br>
Qwen generations: 0<br>
Time: 0.06 s<br>
Conversation: standalone

**Full answer**

Use these IAM PARIS links for this request: [Models](https://iamparis.eu/models).

<a id="q131"></a>

## 131. Plot the data I have not selected yet.

Category: ambiguity and recovery<br>
Route: `general_qa` (charts delegated to data explorer)<br>
Qwen generations: 0<br>
Time: 0.01 s<br>
Conversation: standalone

**Full answer**

I do not generate figures in the chat. Please use the [IAM PARIS data explorer](https://iamparis.eu/results) to explore, filter, and plot the data. Ask about a specific study first if you would like its direct explorer link.

<a id="q132"></a>

## 132. Compare solar with the other technology.

Category: ambiguity and recovery<br>
Route: `data_query` (chat plotting disabled; return numeric table)<br>
Qwen generations: 1<br>
Time: 2.07 s<br>
Conversation: standalone

**Full answer**

I found variable `Final Energy|Solar`, but I still need the region. Choose the region: 1. `AFR (Africa)` 2. `AGO (Angola)` 3. `ARE (United Arab Emirates)` Reply with a number (1-3), or `yes` for option 1.

<a id="q133"></a>

## 133. What does “Policy” refer to in these results?

Category: ambiguity and recovery<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 5.91 s<br>
Conversation: standalone

**Full answer**

I couldn't find `these results` as a region in the IAM PARIS data, so I can't return results for it. Try one of the available regions, e.g. `AT (Austria)`, `BE (Belgium)`, `BG (Bulgaria)`, `CH (Switzerland)`, `CY (Cyprus)`, or ask `list regions` to see all options.

<a id="q134"></a>

## 134. I meant solar generation, not solar capacity.

Category: ambiguity and recovery<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.22 s<br>
Conversation: standalone

**Full answer**

Please provide the variable and region for the numeric comparison.

<a id="q135"></a>

## 135. Can you show a value for the year 2300?

Category: ambiguity and recovery<br>
Route: `data_query` (data request)<br>
Qwen generations: 1<br>
Time: 7.48 s<br>
Conversation: standalone

**Full answer**

I couldn't find `the year` as a region in the IAM PARIS data, so I can't return results for it. Try one of the available regions, e.g. `AT (Austria)`, `BE (Belgium)`, `BG (Bulgaria)`, `CH (Switzerland)`, `CY (Cyprus)`, or ask `list regions` to see all options.

<a id="q136"></a>

## 136. Please use an unknown scenario called Blue Pathway.

Category: ambiguity and recovery<br>
Route: `data_query` (low confidence entity clarification)<br>
Qwen generations: 1<br>
Time: 3.18 s<br>
Conversation: standalone

**Full answer**

I matched `Blue Pathway` as the scenario, but confidence is low. Which scenario should I use?

<a id="q137"></a>

## 137. How many models cover an imaginary sector called moon mining?

Category: ambiguity and recovery<br>
Route: `data_query` (topic-wide model availability)<br>
Qwen generations: 0<br>
Time: 0.08 s<br>
Conversation: standalone

**Full answer**

No loaded IAM PARIS variable mentions `an imaginary sector called moon mining`, so I cannot list models that report it. Ask `list variables` to see the available variables.

<a id="q138"></a>

## 138. Show me results for a country that is not in the dataset.

Category: ambiguity and recovery<br>
Route: `data_query` (region availability request)<br>
Qwen generations: 0<br>
Time: 0.26 s<br>
Conversation: standalone

**Full answer**

I found regions like AFR, AGO, ARE, ARG, ASEAN, ASI, AT, ATE, AUS and AZE and 151 more. Ask for a variable in any of these regions to see its values.

Showing 10 of 161. Say `show all regions` if you need the full list.

<a id="q139"></a>

## 139. Can you explain a model called QUASAR-IAM?

Category: ambiguity and recovery<br>
Route: `model_explanation` (model information request)<br>
Qwen generations: 1<br>
Time: 1.58 s<br>
Conversation: standalone

**Full answer**

I found matching model names in the results data, but no metadata description is available. Examples: TIAM_Grantham, TIAM_Grantham 3.2, tiam. If you want, ask for values from one of these models.

<a id="q140"></a>

## 140. Give me everything at once.

Category: ambiguity and recovery<br>
Route: `data_query` (data request)<br>
Qwen generations: 1<br>
Time: 1.42 s<br>
Conversation: standalone

**Full answer**

I need one more detail. Which variable should I use?

<a id="q141"></a>

## 141. Start with CO2 emissions for the EU in 2040.

Category: multi-turn follow-up<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.36 s<br>
Conversation: `co2_region_model`

**Full answer**

Showing results from the study **Where is the world headed?**. Also available in: Where is the EU headed?; Post-Glasgow targets. Name another study to switch.

### Emissions|CO2 in EU

Scope: scenario `multiple`, model `multiple`, years `2040`
Unit: `Mt CO2/yr`

Answer:
In 2040, Emissions|CO2 in EU ranges from 1,625 (GCAM, PR_CurPol_EI) to 3,549 Mt CO2/yr (GCAM, PR_Baseline) across 26 series; median 2,453.

Summary: 26 series across 4 models and 8 scenarios.
Showing 12 of 26 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 14 remaining series.

| Model | Scenario | 2040 | Unit |
|---|---|---|---|
| E3ME | PR_Baseline | 2,692 | Mt CO2/yr |
| E3ME | PR_CurPol_CP | 2,705 | Mt CO2/yr |
| E3ME | PR_CurPol_EI | 2,827 | Mt CO2/yr |
| GCAM | PR_Baseline | 3,549 | Mt CO2/yr |
| GCAM | PR_CurPol_CP | 1,836 | Mt CO2/yr |
| GCAM | PR_CurPol_CPO | 1,723 | Mt CO2/yr |
| GCAM | PR_CurPol_EI | 1,625 | Mt CO2/yr |
| GEMINI-E3 | PR_Baseline | 3,048 | Mt CO2/yr |
| GEMINI-E3 | PR_CurPol_CP | 2,545 | Mt CO2/yr |
| GEMINI-E3 | PR_CurPol_CPO | 2,541 | Mt CO2/yr |
| MUSE | PR_Baseline | 2,701 | Mt CO2/yr |
| MUSE | PR_CurPol_CP | 2,435 | Mt CO2/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q142"></a>

## 142. Switch the geography to Brazil while keeping the year.

Category: multi-turn follow-up<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.57 s<br>
Conversation: `co2_region_model`

**Full answer**

### Emissions|CO2 in BRA

Scope: scenario `multiple`, model `multiple`, years `2040`
Unit: `Mt CO2/yr`

Answer:
In 2040, Emissions|CO2 in BRA ranges from 90.46 (GCAM, PR_CurPol_EI) to 830.23 Mt CO2/yr (GCAM, PR_Baseline) across 28 series; median 472.60.

Summary: 28 series across 4 models and 10 scenarios.
Showing 12 of 28 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 16 remaining series.

| Model | Scenario | 2040 | Unit |
|---|---|---|---|
| E3ME | PR_Baseline | 688.37 | Mt CO2/yr |
| E3ME | PR_CurPol_CP | 685.18 | Mt CO2/yr |
| E3ME | PR_CurPol_EI | 660.92 | Mt CO2/yr |
| GCAM | Baseline | 750.47 | Mt CO2/yr |
| GCAM | CP_EI | 187.60 | Mt CO2/yr |
| GCAM | PR_Baseline | 830.23 | Mt CO2/yr |
| GCAM | PR_CurPol_EI | 90.46 | Mt CO2/yr |
| GEMINI-E3 | PR_Baseline | 527.51 | Mt CO2/yr |
| GEMINI-E3 | PR_CurPol_CP | 500.43 | Mt CO2/yr |
| GEMINI-E3 | PR_CurPol_CPO | 513.99 | Mt CO2/yr |
| MUSE | PR_Baseline | 522.71 | Mt CO2/yr |
| MUSE | PR_CurPol_CP | 463.32 | Mt CO2/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q143"></a>

## 143. Now limit that result to GCAM.

Category: multi-turn follow-up<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.53 s<br>
Conversation: `co2_region_model`

**Full answer**

### Emissions|CO2 in BRA

Scope: scenario `multiple`, model `GCAM`, years `2040`
Unit: `Mt CO2/yr`

Answer:
In 2040, Emissions|CO2 in BRA ranges from 90.46 (GCAM, PR_CurPol_EI) to 830.23 Mt CO2/yr (GCAM, PR_Baseline) across 9 series; median 295.89.

Summary: 9 series across 1 model and 9 scenarios.
Series are ordered by model and scenario.

| Model | Scenario | 2040 | Unit |
|---|---|---|---|
| GCAM | Baseline | 750.47 | Mt CO2/yr |
| GCAM | CP_EI | 187.60 | Mt CO2/yr |
| GCAM | PR_Baseline | 830.23 | Mt CO2/yr |
| GCAM | PR_CurPol_CP | 295.89 | Mt CO2/yr |
| GCAM | PR_CurPol_CPO | 347.63 | Mt CO2/yr |
| GCAM | PR_CurPol_EI | 90.46 | Mt CO2/yr |
| GCAM | PR_NDC_CP | 293.55 | Mt CO2/yr |
| GCAM | PR_NDC_CPO | 346.29 | Mt CO2/yr |
| GCAM | PR_NDC_EI | 174.46 | Mt CO2/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

Reply with a scenario, model, region, or year to narrow the answer.

<a id="q144"></a>

## 144. Use 2050 instead of 2040.

Category: multi-turn follow-up<br>
Route: `data_query` (explicit result scope edit)<br>
Qwen generations: 0<br>
Time: 0.33 s<br>
Conversation: `co2_region_model`

**Full answer**

### Emissions|CO2 in BRA

Scope: scenario `multiple`, model `GCAM`, years `2050`
Unit: `Mt CO2/yr`

Answer:
In 2050, Emissions|CO2 in BRA ranges from 58.66 (GCAM, PR_CurPol_EI) to 853.70 Mt CO2/yr (GCAM, PR_Baseline) across 9 series; median 373.26.

Summary: 9 series across 1 model and 9 scenarios.
Series are ordered by model and scenario.

| Model | Scenario | 2050 | Unit |
|---|---|---|---|
| GCAM | Baseline | 744.89 | Mt CO2/yr |
| GCAM | CP_EI | 136.84 | Mt CO2/yr |
| GCAM | PR_Baseline | 853.70 | Mt CO2/yr |
| GCAM | PR_CurPol_CP | 335.51 | Mt CO2/yr |
| GCAM | PR_CurPol_CPO | 431.33 | Mt CO2/yr |
| GCAM | PR_CurPol_EI | 58.66 | Mt CO2/yr |
| GCAM | PR_NDC_CP | 373.26 | Mt CO2/yr |
| GCAM | PR_NDC_CPO | 461.45 | Mt CO2/yr |
| GCAM | PR_NDC_EI | 155.97 | Mt CO2/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q145"></a>

## 145. Which scenarios contributed to this table?

Category: multi-turn follow-up<br>
Route: `data_query` (scoped availability request)<br>
Qwen generations: 0<br>
Time: 0.16 s<br>
Conversation: `co2_region_model`

**Full answer**

### Availability for Emissions|CO2

**Scenarios (7):** PR_Baseline, PR_CurPol_CP, PR_CurPol_CPO, PR_CurPol_EI, PR_NDC_CP, PR_NDC_CPO, PR_NDC_EI

Recorded unit: `Mt CO2/yr`.

<a id="q146"></a>

## 146. Open the explorer for that study.

Category: multi-turn follow-up<br>
Route: `general_qa` (explicit site navigation request)<br>
Qwen generations: 0<br>
Time: 0.05 s<br>
Conversation: `co2_region_model`

**Full answer**

I could not match this request to a reliable IAM PARIS link.

<a id="q147"></a>

## 147. Show solar electricity capacity for Germany from 2030 to 2050.

Category: multi-turn follow-up<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.36 s<br>
Conversation: `solar_variable_change`

**Full answer**

Showing results from the study **EU path to net zero**. Also available in: Where is the world headed?. Name another study to switch.

### Secondary Energy|Electricity|Solar in DEU

Scope: scenario `multiple`, model `multiple`, years `2030-2050`
Unit: `EJ/yr`

Answer:
In 2050, Secondary Energy|Electricity|Solar in DEU ranges from 0.0196 (ICES-XPS 1.0, WWH21) to 2.09 EJ/yr (E4SMA-EU-TIMES 1.0, NZE_Bench_L) across 13 series; median 1.32. Median change since 2030: +144%.

Summary: 13 series across 3 models and 5 scenarios.
Showing 12 of 13 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 1 remaining series.

| Model | Scenario | 2030 | 2035 | 2040 | 2045 | 2050 | Unit |
|---|---|---|---|---|---|---|---|
| E4SMA-EU-TIMES 1.0 | NZE_Bench_H | 0.164 | — | 0.23 | — | 0.683 | EJ/yr |
| E4SMA-EU-TIMES 1.0 | NZE_Bench_L | 0.171 | — | 0.346 | — | 2.09 | EJ/yr |
| E4SMA-EU-TIMES 1.0 | NZE_Bench_M | 0.164 | — | 0.26 | — | 1.34 | EJ/yr |
| E4SMA-EU-TIMES 1.0 | NZE_EUPol_Stand | 0.164 | — | 0.274 | — | 1.39 | EJ/yr |
| E4SMA-EU-TIMES 1.0 | WWH21 | 0.164 | — | 0.163 | — | 0.261 | EJ/yr |
| ICES-XPS 1.0 | NZE_Bench_H | 0.019 | 0.0199 | 0.0214 | 0.0249 | 0.0321 | EJ/yr |
| ICES-XPS 1.0 | NZE_EUPol_Stand | 0.0199 | 0.0205 | 0.0216 | 0.0243 | 0.0295 | EJ/yr |
| ICES-XPS 1.0 | WWH21 | 0.0174 | 0.0171 | 0.017 | 0.0178 | 0.0196 | EJ/yr |
| NEMESIS 5.1 | NZE_Bench_H | 0.572 | 0.724 | 0.864 | 1.08 | 1.32 | EJ/yr |
| NEMESIS 5.1 | NZE_Bench_L | 0.649 | 0.856 | 1.06 | 1.39 | 1.80 | EJ/yr |
| NEMESIS 5.1 | NZE_Bench_M | 0.608 | 0.786 | 0.942 | 1.20 | 1.49 | EJ/yr |
| NEMESIS 5.1 | NZE_EUPol_Stand | 0.629 | 0.784 | 0.959 | 1.21 | 1.53 | EJ/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/eu-path-to-net-zero/graphs).

<a id="q148"></a>

## 148. Change just the variable to wind electricity capacity.

Category: multi-turn follow-up<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.42 s<br>
Conversation: `solar_variable_change`

**Full answer**

Using `DEU` from your previous question; name another region to change it.

### Secondary Energy|Electricity|Wind in DEU

Scope: scenario `multiple`, model `multiple`, years `2030-2050`
Unit: `EJ/yr`

Answer:
In 2050, Secondary Energy|Electricity|Wind in DEU ranges from 0.512 (E4SMA-EU-TIMES 1.0, WWH21) to 5.96 EJ/yr (ICES-XPS 1.0, NZE_Bench_H) across 13 series; median 1.42. Median change since 2030: +119%.

Summary: 13 series across 3 models and 5 scenarios.
Showing 12 of 13 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 1 remaining series.

| Model | Scenario | 2030 | 2035 | 2040 | 2045 | 2050 | Unit |
|---|---|---|---|---|---|---|---|
| E4SMA-EU-TIMES 1.0 | NZE_Bench_H | 0.466 | — | 1.10 | — | 1.42 | EJ/yr |
| E4SMA-EU-TIMES 1.0 | NZE_Bench_L | 0.513 | — | 1.10 | — | 1.41 | EJ/yr |
| E4SMA-EU-TIMES 1.0 | NZE_Bench_M | 0.513 | — | 1.10 | — | 1.41 | EJ/yr |
| E4SMA-EU-TIMES 1.0 | NZE_EUPol_Stand | 0.486 | — | 1.08 | — | 1.42 | EJ/yr |
| E4SMA-EU-TIMES 1.0 | WWH21 | 0.393 | — | 0.422 | — | 0.512 | EJ/yr |
| ICES-XPS 1.0 | NZE_Bench_H | 2.43 | 3.14 | 3.88 | 4.80 | 5.96 | EJ/yr |
| ICES-XPS 1.0 | NZE_EUPol_Stand | 2.52 | 3.21 | 3.92 | 4.79 | 5.88 | EJ/yr |
| ICES-XPS 1.0 | WWH21 | 2.22 | 2.72 | 3.20 | 3.78 | 4.56 | EJ/yr |
| NEMESIS 5.1 | NZE_Bench_H | 0.757 | 0.884 | 1.00 | 1.17 | 1.36 | EJ/yr |
| NEMESIS 5.1 | NZE_Bench_L | 0.823 | 1.01 | 1.18 | 1.44 | 1.81 | EJ/yr |
| NEMESIS 5.1 | NZE_Bench_M | 0.785 | 0.941 | 1.08 | 1.27 | 1.52 | EJ/yr |
| NEMESIS 5.1 | NZE_EUPol_Stand | 0.798 | 0.927 | 1.08 | 1.25 | 1.53 | EJ/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/eu-path-to-net-zero/graphs).

<a id="q149"></a>

## 149. Keep Germany but stop at 2040.

Category: multi-turn follow-up<br>
Route: `data_query` (explicit result scope edit)<br>
Qwen generations: 0<br>
Time: 0.44 s<br>
Conversation: `solar_variable_change`

**Full answer**

### Secondary Energy|Electricity|Wind in DEU

Scope: scenario `multiple`, model `multiple`, years `2040`
Unit: `EJ/yr`

Answer:
In 2040, Secondary Energy|Electricity|Wind in DEU ranges from 0.422 (E4SMA-EU-TIMES 1.0, WWH21) to 3.92 EJ/yr (ICES-XPS 1.0, NZE_EUPol_Stand) across 13 series; median 1.10.

Summary: 13 series across 3 models and 5 scenarios.
Showing 12 of 13 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 1 remaining series.

| Model | Scenario | 2040 | Unit |
|---|---|---|---|
| E4SMA-EU-TIMES 1.0 | NZE_Bench_H | 1.10 | EJ/yr |
| E4SMA-EU-TIMES 1.0 | NZE_Bench_L | 1.10 | EJ/yr |
| E4SMA-EU-TIMES 1.0 | NZE_Bench_M | 1.10 | EJ/yr |
| E4SMA-EU-TIMES 1.0 | NZE_EUPol_Stand | 1.08 | EJ/yr |
| E4SMA-EU-TIMES 1.0 | WWH21 | 0.422 | EJ/yr |
| ICES-XPS 1.0 | NZE_Bench_H | 3.88 | EJ/yr |
| ICES-XPS 1.0 | NZE_EUPol_Stand | 3.92 | EJ/yr |
| ICES-XPS 1.0 | WWH21 | 3.20 | EJ/yr |
| NEMESIS 5.1 | NZE_Bench_H | 1.00 | EJ/yr |
| NEMESIS 5.1 | NZE_Bench_L | 1.18 | EJ/yr |
| NEMESIS 5.1 | NZE_Bench_M | 1.08 | EJ/yr |
| NEMESIS 5.1 | NZE_EUPol_Stand | 1.08 | EJ/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/eu-path-to-net-zero/graphs).

<a id="q150"></a>

## 150. Which models produced those values?

Category: multi-turn follow-up<br>
Route: `data_query` (result provenance follow-up)<br>
Qwen generations: 0<br>
Time: 0.06 s<br>
Conversation: `solar_variable_change`

**Full answer**

That result includes data from these models: `E4SMA-EU-TIMES 1.0`, `ICES-XPS 1.0`, `NEMESIS 5.1`.

<a id="q151"></a>

## 151. Explain why model values could differ.

Category: multi-turn follow-up<br>
Route: `data_query` (chat plotting disabled; return numeric table)<br>
Qwen generations: 0<br>
Time: 0.50 s<br>
Conversation: `solar_variable_change`

**Full answer**

### Secondary Energy|Electricity|Wind in DEU

Scope: scenario `multiple`, model `multiple`, years `2040`
Unit: `EJ/yr`

Answer:
In 2040, Secondary Energy|Electricity|Wind in DEU ranges from 0.422 (E4SMA-EU-TIMES 1.0, WWH21) to 3.92 EJ/yr (ICES-XPS 1.0, NZE_EUPol_Stand) across 13 series; median 1.10.

Summary: 13 series across 3 models and 5 scenarios.
Showing 12 of 13 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 1 remaining series.

| Model | Scenario | 2040 | Unit |
|---|---|---|---|
| E4SMA-EU-TIMES 1.0 | NZE_Bench_H | 1.10 | EJ/yr |
| E4SMA-EU-TIMES 1.0 | NZE_Bench_L | 1.10 | EJ/yr |
| E4SMA-EU-TIMES 1.0 | NZE_Bench_M | 1.10 | EJ/yr |
| E4SMA-EU-TIMES 1.0 | NZE_EUPol_Stand | 1.08 | EJ/yr |
| E4SMA-EU-TIMES 1.0 | WWH21 | 0.422 | EJ/yr |
| ICES-XPS 1.0 | NZE_Bench_H | 3.88 | EJ/yr |
| ICES-XPS 1.0 | NZE_EUPol_Stand | 3.92 | EJ/yr |
| ICES-XPS 1.0 | WWH21 | 3.20 | EJ/yr |
| NEMESIS 5.1 | NZE_Bench_H | 1.00 | EJ/yr |
| NEMESIS 5.1 | NZE_Bench_L | 1.18 | EJ/yr |
| NEMESIS 5.1 | NZE_Bench_M | 1.08 | EJ/yr |
| NEMESIS 5.1 | NZE_EUPol_Stand | 1.08 | EJ/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/eu-path-to-net-zero/graphs).

<a id="q152"></a>

## 152. Return to the solar series for 2040.

Category: multi-turn follow-up<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.52 s<br>
Conversation: `solar_variable_change`

**Full answer**

### Secondary Energy|Electricity|Wind in DEU

Scope: scenario `multiple`, model `multiple`, years `2040`
Unit: `EJ/yr`

Answer:
In 2040, Secondary Energy|Electricity|Wind in DEU ranges from 0.422 (E4SMA-EU-TIMES 1.0, WWH21) to 3.92 EJ/yr (ICES-XPS 1.0, NZE_EUPol_Stand) across 13 series; median 1.10.

Summary: 13 series across 3 models and 5 scenarios.
Showing 12 of 13 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 1 remaining series.

| Model | Scenario | 2040 | Unit |
|---|---|---|---|
| E4SMA-EU-TIMES 1.0 | NZE_Bench_H | 1.10 | EJ/yr |
| E4SMA-EU-TIMES 1.0 | NZE_Bench_L | 1.10 | EJ/yr |
| E4SMA-EU-TIMES 1.0 | NZE_Bench_M | 1.10 | EJ/yr |
| E4SMA-EU-TIMES 1.0 | NZE_EUPol_Stand | 1.08 | EJ/yr |
| E4SMA-EU-TIMES 1.0 | WWH21 | 0.422 | EJ/yr |
| ICES-XPS 1.0 | NZE_Bench_H | 3.88 | EJ/yr |
| ICES-XPS 1.0 | NZE_EUPol_Stand | 3.92 | EJ/yr |
| ICES-XPS 1.0 | WWH21 | 3.20 | EJ/yr |
| NEMESIS 5.1 | NZE_Bench_H | 1.00 | EJ/yr |
| NEMESIS 5.1 | NZE_Bench_L | 1.18 | EJ/yr |
| NEMESIS 5.1 | NZE_Bench_M | 1.08 | EJ/yr |
| NEMESIS 5.1 | NZE_EUPol_Stand | 1.08 | EJ/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/eu-path-to-net-zero/graphs).

<a id="q153"></a>

## 153. Describe the PROMETHEUS model and its sectors.

Category: multi-turn follow-up<br>
Route: `model_explanation` (named-model metadata request)<br>
Qwen generations: 0<br>
Time: 0.07 s<br>
Conversation: `model_pronouns`

**Full answer**

### PROMETHEUS

Description:
The Prometheus model is a comprehensive world energy model with innovative features: it integrates stochastic relations and so all exogenous variables – parameters are stochastic, following explicit probability distributions, including covariance. Therefore, all model results are also stochastic (probability distributions). Among others, stochastic distributions are used for technology change and learning by doing, as well as for fossil fuel resources, both conventional and unconventional. Prometheus models the world energy system divided into 10 regions and produces yearly projections up to 2050; the update to 2070 is ongoing. The model includes a set of long time series (IEA data mostly) on which econometric estimations are carried out. The model produces projections of energy demand by sector (industry, domestic, transport), power generation (representing about 25 technologies), RES, and hydrogen supply and use. Prometheus puts emphasis on oil and gas resources, while coal is assumed to have rather abundant supplies relative to production prospects in the projection time horizon. The model incorporates uncertainty surrounding the amount of oil and gas resources that are yet to be discovered. Data are mostly calibrated on estimation of resources by USGS. The model produces projections of fossil fuel prices, which depend on demand, supply, technology and resources. As all outputs are probabilistic, fossil fuel price projections take the form of probability distributions. The model includes CO2 emissions and can simulate emission reduction pathways.

Developer:
E3Modelling S.A. https://e3modelling.com/

Related model documentation:
- [IAM PARIS model page](https://iamparis.eu/models/55)

<a id="q154"></a>

## 154. Who built that model?

Category: multi-turn follow-up<br>
Route: `model_explanation` (singular-model qualitative follow-up)<br>
Qwen generations: 0<br>
Time: 0.07 s<br>
Conversation: `model_pronouns`

**Full answer**

### PROMETHEUS

`PROMETHEUS` is developed by E3Modelling S.A. https://e3modelling.com/.

Related model documentation:
- [IAM PARIS model page](https://iamparis.eu/models/55)

<a id="q155"></a>

## 155. What limitations does it have?

Category: multi-turn follow-up<br>
Route: `model_explanation` (singular-model qualitative follow-up)<br>
Qwen generations: 0<br>
Time: 0.06 s<br>
Conversation: `model_pronouns`

**Full answer**

### PROMETHEUS

Interpretation notes:
- The loaded IAM PARIS profile does not list specific limitations for `PROMETHEUS`. Results depend on the scenario assumptions and input data, so check the model documentation and the study's experiment metadata before interpreting a result.

Related model documentation:
- [IAM PARIS model page](https://iamparis.eu/models/55)

<a id="q156"></a>

## 156. Which regions does it report?

Category: multi-turn follow-up<br>
Route: `data_query` (model-scoped catalogue follow-up)<br>
Qwen generations: 0<br>
Time: 0.55 s<br>
Conversation: `model_pronouns`

**Full answer**

Model `PROMETHEUS` has these regions: CHN, CIS, EMRG, EPROD, EU28, IND, NOAM, RESTW, WPAC, World.

<a id="q157"></a>

## 157. Does that model have EU carbon-price data?

Category: multi-turn follow-up<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.95 s<br>
Conversation: `model_pronouns`

**Full answer**

### What I can help you with

- **Models:** 75 available. Examples: AIM/Enduse India 3.3, ALADIN, ATOM, CHANCE.
- **Variables:** 516 available. Examples: Benchmarking|Buildings|CO2 intensity, Benchmarking|Industry|CO2 intensity (gross), Benchmarking|Industry|CO2 intensity (net), Benchmarking|Transport|CO2 intensity.
- **Scenarios:** 15 available. Examples: Baseline, CP_EI, NDC_EI, NDC_LTT.
- **Regions:** 109 available. Examples: AGO, ARE, ARG, AUS.
- **Workspaces:** 1 available. Ask `list workspaces`.

Try one of these next:
- `list variables`
- `list regions`
- `list scenarios`
- `Show me CO2 emissions for EU`
- `Compare solar and wind capacity for EU`

Reply with a scenario, model, region, or year to narrow the answer.

<a id="q158"></a>

## 158. Show its documentation link.

Category: multi-turn follow-up<br>
Route: `general_qa` (explicit site navigation request)<br>
Qwen generations: 0<br>
Time: 0.03 s<br>
Conversation: `model_pronouns`

**Full answer**

Use these IAM PARIS links for this request: [Models](https://iamparis.eu/models).

<a id="q159"></a>

## 159. I want energy data for India in 2050.

Category: multi-turn follow-up<br>
Route: `data_query` (low confidence entity clarification)<br>
Qwen generations: 0<br>
Time: 0.80 s<br>
Conversation: `clarify_energy`

**Full answer**

I found several possible variables, but none has enough evidence for automatic selection. Choose the variable: 1. `Final Energy` (final energy use) 2. `Export|Energy` 3. `Import|Energy` Reply with a number (1-3), or `yes` for option 1.

<a id="q160"></a>

## 160. Choose option 2 if it is available.

Category: multi-turn follow-up<br>
Route: `data_query` (chat plotting disabled; return numeric table)<br>
Qwen generations: 2<br>
Time: 2.01 s<br>
Conversation: `clarify_energy`

**Full answer**

No. I could not find `Carbon Removal|Land Use|{Land Removal Option}` for that model scope in the loaded IAM PARIS timeseries data.

<a id="q161"></a>

## 161. Actually use electricity output rather than final energy.

Category: multi-turn follow-up<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.67 s<br>
Conversation: `clarify_energy`

**Full answer**

Showing results from the study **Where is the world headed?**. Also available in: EU path to net zero; Post-Glasgow targets; Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU; Global impacts of NDCs; Tech-constrained EU pathways to net zero; Implications of EU steel industry relocation; Behavioural changes for climate policy; Where is the EU headed?; Enhancing renewable energy system metrics. Name another study to switch.

### Final Energy|Electricity in World

Scope: scenario `multiple`, model `multiple`, years `2005-2100`
Unit: `EJ/yr`

Answer:
In 2040, Final Energy|Electricity in World ranges from 79.09 (MUSE, PR_CurPol_EI) to 204.55 EJ/yr (ICES-XPS, PR_NDC_EI) across 43 series; median 130.12. Median change since 2005: +139%.

Summary: 43 series across 7 models and 8 scenarios.
Showing 12 of 43 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 31 remaining series.

| Model | Scenario | 2005 | 2010 | 2015 | 2020 | 2025 | 2030 | 2035 | 2040 | 2045 | 2050 | 2055 | 2060 | 2065 | 2070 | 2075 | 2080 | 2085 | 2090 | 2095 | 2100 | Unit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | PR_Baseline | 54.55 | 64.43 | 72.73 | 82.70 | 93.94 | 105.44 | 116.99 | 128.04 | 138.96 | — | — | — | — | — | — | — | — | — | — | — | EJ/yr |
| 42 | PR_CurPol_EI | 54.55 | 64.43 | 72.73 | 82.70 | 93.93 | 106.07 | 118.17 | 130.12 | 141.89 | — | — | — | — | — | — | — | — | — | — | — | EJ/yr |
| E3ME | PR_Baseline | — | 64.18 | 72.26 | 83.20 | 92.95 | 104.58 | 115.71 | 127.53 | 142.14 | 160.27 | — | — | — | — | — | — | — | — | — | — | EJ/yr |
| E3ME | PR_CurPol_CP | — | 64.18 | 72.26 | 83.20 | 92.95 | 104.58 | 115.71 | 127.52 | 142.11 | 160.21 | — | — | — | — | — | — | — | — | — | — | EJ/yr |
| GCAM | PR_Baseline | 55.73 | 66.07 | 74.77 | 89.30 | 105.22 | 124.88 | 136.55 | 147.45 | 160.78 | 174.89 | 189.63 | 204.57 | 219.03 | 233.23 | 246.25 | 258.31 | 269.11 | 279.21 | 287.95 | 295.91 | EJ/yr |
| GCAM | PR_CurPol_CP | 55.73 | 66.07 | 74.77 | 92.02 | 109.78 | 127.91 | 134.00 | 144.96 | 159.39 | 174.19 | 189.45 | 204.84 | 220.07 | 235.60 | 249.42 | 262.34 | 273.69 | 284.38 | 293.93 | 302.94 | EJ/yr |
| GEMINI-E3 | PR_Baseline | — | — | 93.16 | 106.60 | 121.69 | 137.82 | 152.57 | 167.71 | 185.14 | 201.33 | — | — | — | — | — | — | — | — | — | — | EJ/yr |
| GEMINI-E3 | PR_CurPol_CP | — | — | 93.16 | 106.15 | 120.16 | 135.14 | 149.70 | 164.34 | 181.53 | 197.53 | — | — | — | — | — | — | — | — | — | — | EJ/yr |
| ICES-XPS | PR_Baseline | — | — | 77.83 | 79.19 | 98.18 | 123.37 | 151.18 | 185.81 | 229.29 | 288.52 | — | — | — | — | — | — | — | — | — | — | EJ/yr |
| ICES-XPS | PR_NDC_EI | — | — | 77.83 | 79.13 | 99.70 | 127.46 | 161.33 | 204.55 | 255.47 | 315.10 | — | — | — | — | — | — | — | — | — | — | EJ/yr |
| MUSE | PR_CurPol_EI | — | — | 61.24 | 64.18 | 67.72 | 72.16 | 76.12 | 79.09 | 84.71 | 89.54 | 94.07 | 98.63 | 104.05 | 108.84 | 113.85 | 118.05 | 121.97 | 126.53 | 130.94 | 133.48 | EJ/yr |
| TIAM_Grantham | PR_Baseline | 53.15 | 64.99 | — | 85.03 | — | 94.00 | — | 109.22 | — | 130.47 | — | 156.92 | — | 176.35 | — | 196.77 | — | 214.10 | — | 230.64 | EJ/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q162"></a>

## 162. Keep India and show the year 2040.

Category: multi-turn follow-up<br>
Route: `data_query` (explicit result scope edit)<br>
Qwen generations: 0<br>
Time: 0.44 s<br>
Conversation: `clarify_energy`

**Full answer**

### Final Energy|Electricity in IND

Scope: scenario `multiple`, model `multiple`, years `2040`
Unit: `EJ/yr`

Answer:
In 2040, Final Energy|Electricity in IND ranges from 5.99 (MUSE, PR_Baseline) to 19.04 EJ/yr (GCAM, PR_Baseline) across 43 series; median 13.58.

Summary: 43 series across 7 models and 8 scenarios.
Showing 12 of 43 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 31 remaining series.

| Model | Scenario | 2040 | Unit |
|---|---|---|---|
| 42 | PR_Baseline | 12.03 | EJ/yr |
| 42 | PR_CurPol_EI | 12.03 | EJ/yr |
| E3ME | PR_Baseline | 13.58 | EJ/yr |
| E3ME | PR_CurPol_CP | 13.53 | EJ/yr |
| GCAM | PR_Baseline | 19.04 | EJ/yr |
| GCAM | PR_CurPol_CP | 17.68 | EJ/yr |
| GEMINI-E3 | PR_Baseline | 13.72 | EJ/yr |
| GEMINI-E3 | PR_CurPol_CP | 14.16 | EJ/yr |
| ICES-XPS | PR_Baseline | 15.94 | EJ/yr |
| ICES-XPS | PR_CurPol_CP | 15.83 | EJ/yr |
| MUSE | PR_Baseline | 5.99 | EJ/yr |
| TIAM_Grantham | PR_Baseline | 8.93 | EJ/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q163"></a>

## 163. Which study supplied these results?

Category: multi-turn follow-up<br>
Route: `data_query` (study/results catalogue request)<br>
Qwen generations: 0<br>
Time: 0.07 s<br>
Conversation: `clarify_energy`

**Full answer**

I found these workspaces (studies): AFOLU transformation; Behavioural changes for climate policy; Buildings Transformation; Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU; Cost of capital evolution effects on regional mitigation pathways; COVID recovery packages; Decarbonization potentials of energy citizens; Decipher modelling studies; Enhancing renewable energy system metrics; EU path to net zero; Global impacts of NDCs; Implications of EU steel industry relocation and 9 more. Ask about a study to see its variables, scenarios, and direct IAM PARIS links.

Showing 12 of 21. Say `show all workspaces` if you need the full list.

<a id="q164"></a>

## 164. Can I compare the same metric with China?

Category: multi-turn follow-up<br>
Route: `data_query` (scope-filtered variable availability)<br>
Qwen generations: 0<br>
Time: 0.77 s<br>
Conversation: `clarify_energy`

**Full answer**

### Variables available for the requested scope

Scope: region `China`, scenario `PR_Baseline`, `PR_CurPol_CP`, `PR_CurPol_CPO`, `PR_CurPol_EI`, `PR_NDC_CP`, `PR_NDC_CPO`, `PR_NDC_EI`, `Unharmonised baseline`.

**Variables (0):** none

<a id="q165"></a>

## 165. Give World CO2 emissions under Baseline in 2040.

Category: multi-turn follow-up<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.34 s<br>
Conversation: `scenario_comparison`

**Full answer**

Showing results from the study **Where is the world headed?**. Also available in: Post-Glasgow targets; EU path to net zero; Global impacts of NDCs. Name another study to switch.

### Emissions|CO2 in World

Scope: scenario `multiple`, model `multiple`, years `2040`
Unit: `Mt CO2/yr`

Answer:
In 2040, Emissions|CO2 in World ranges from 33,696 (MUSE, baseline) to 48,984 Mt CO2/yr (GEMINI-E3, PR_Baseline) across 8 series; median 42,274.

Summary: 8 series across 4 models and 4 scenarios.
Series are ordered by model and scenario.

| Model | Scenario | 2040 | Unit |
|---|---|---|---|
| E3ME | PR_Baseline | 38,757 | Mt CO2/yr |
| E3ME | Unharmonised baseline | 39,232 | Mt CO2/yr |
| GCAM | Baseline | 45,315 | Mt CO2/yr |
| GCAM | PR_Baseline | 45,961 | Mt CO2/yr |
| GEMINI-E3 | Baseline | 45,532 | Mt CO2/yr |
| GEMINI-E3 | PR_Baseline | 48,984 | Mt CO2/yr |
| MUSE | baseline | 33,696 | Mt CO2/yr |
| MUSE | PR_Baseline | 36,732 | Mt CO2/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q166"></a>

## 166. Compare that with current policies for the same year.

Category: multi-turn follow-up<br>
Route: `data_query` (scenario-comparison table follow-up)<br>
Qwen generations: 0<br>
Time: 0.20 s<br>
Conversation: `scenario_comparison`

**Full answer**

### Emissions|CO2 in World

Scope: scenario `multiple`, model `multiple`, years `2040`
Unit: `Mt CO2/yr`

Answer:
In 2040, Emissions|CO2 in World ranges from 33,696 (MUSE, baseline) to 48,984 Mt CO2/yr (GEMINI-E3, PR_Baseline) across 19 series; median 39,232.

Summary: 19 series across 4 models and 7 scenarios.
Showing 12 of 19 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 7 remaining series.

| Model | Scenario | 2040 | Unit |
|---|---|---|---|
| E3ME | PR_Baseline | 38,757 | Mt CO2/yr |
| E3ME | PR_CurPol_CP | 38,479 | Mt CO2/yr |
| E3ME | PR_CurPol_EI | 38,706 | Mt CO2/yr |
| GCAM | Baseline | 45,315 | Mt CO2/yr |
| GCAM | PR_Baseline | 45,961 | Mt CO2/yr |
| GCAM | PR_CurPol_CP | 39,485 | Mt CO2/yr |
| GEMINI-E3 | Baseline | 45,532 | Mt CO2/yr |
| GEMINI-E3 | PR_Baseline | 48,984 | Mt CO2/yr |
| GEMINI-E3 | PR_CurPol_CP | 41,810 | Mt CO2/yr |
| MUSE | baseline | 33,696 | Mt CO2/yr |
| MUSE | PR_Baseline | 36,732 | Mt CO2/yr |
| MUSE | PR_CurPol_CP | 34,835 | Mt CO2/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q167"></a>

## 167. Which of the two scenario families is lower?

Category: multi-turn follow-up<br>
Route: `data_query` (scenario availability request)<br>
Qwen generations: 0<br>
Time: 0.18 s<br>
Conversation: `scenario_comparison`

**Full answer**

I found scenarios like 1.5 TECH, 1.5 °C DRI-trade, 1.5 °C Domestic, BAU, BAU_CL1, BAU_CL2, BAU_CL3 and BAU_Familiar and 109 more. Ask for a variable under any of these scenarios to see its values.

Showing 8 of 117. Say `show all scenarios` if you need the full list.

Reply with a scenario, model, region, or year to narrow the answer.

<a id="q168"></a>

## 168. Change the region to India for this comparison.

Category: multi-turn follow-up<br>
Route: `data_query` (extracted data entities)<br>
Qwen generations: 0<br>
Time: 0.54 s<br>
Conversation: `scenario_comparison`

**Full answer**

I can answer questions about IAM PARIS results, models and climate-policy terms, but I need a little more to go on. Try for example:
- `CO2 emissions for World in 2050`
- `which scenario has the lowest emissions in 2050?`
- `which models cover transport?`
- `what does net zero mean?`

<a id="q169"></a>

## 169. Show only 2050 now.

Category: multi-turn follow-up<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 7.06 s<br>
Conversation: `scenario_comparison`

**Full answer**

I need one more detail. Which variable should I use?

<a id="q170"></a>

## 170. What assumptions could explain the gap?

Category: multi-turn follow-up<br>
Route: `data_query` (unclear deterministic route)<br>
Qwen generations: 2<br>
Time: 1.67 s<br>
Conversation: `scenario_comparison`

**Full answer**

I can answer questions about IAM PARIS results, models and climate-policy terms, but I need a little more to go on. Try for example:
- `CO2 emissions for World in 2050`
- `which scenario has the lowest emissions in 2050?`
- `which models cover transport?`
- `what does net zero mean?`

<a id="q171"></a>

## 171. Show 2050 CO2 emissions for Atlantis.

Category: multi-turn follow-up<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.20 s<br>
Conversation: `failed_scope_recovery`

**Full answer**

I couldn't find `Atlantis` as a region in the loaded IAM PARIS data. Please choose a loaded region using `list regions`; I cannot substitute a different geographic aggregate for the requested comparison.

<a id="q172"></a>

## 172. Use Greece as the region instead.

Category: multi-turn follow-up<br>
Route: `data_query` (extracted data entities)<br>
Qwen generations: 0<br>
Time: 0.39 s<br>
Conversation: `failed_scope_recovery`

**Full answer**

I can answer questions about IAM PARIS results, models and climate-policy terms, but I need a little more to go on. Try for example:
- `CO2 emissions for World in 2050`
- `which scenario has the lowest emissions in 2050?`
- `which models cover transport?`
- `what does net zero mean?`

<a id="q173"></a>

## 173. Keep the same metric but use a baseline pathway.

Category: multi-turn follow-up<br>
Route: `data_query` (extracted data entities)<br>
Qwen generations: 0<br>
Time: 2.13 s<br>
Conversation: `failed_scope_recovery`

**Full answer**

I can answer questions about IAM PARIS results, models and climate-policy terms, but I need a little more to go on. Try for example:
- `CO2 emissions for World in 2050`
- `which scenario has the lowest emissions in 2050?`
- `which models cover transport?`
- `what does net zero mean?`

<a id="q174"></a>

## 174. What model names are in that result?

Category: multi-turn follow-up<br>
Route: `data_query` (unclear deterministic route)<br>
Qwen generations: 3<br>
Time: 2.70 s<br>
Conversation: `failed_scope_recovery`

**Full answer**

I couldn't match that to a known model. Try `list models` to see available options.

<a id="q175"></a>

## 175. Chart this in the data explorer.

Category: multi-turn follow-up<br>
Route: `general_qa` (charts delegated to data explorer)<br>
Qwen generations: 0<br>
Time: 0.02 s<br>
Conversation: `failed_scope_recovery`

**Full answer**

I do not generate figures in the chat. Please use the [IAM PARIS data explorer](https://iamparis.eu/results) to explore, filter, and plot the data. Ask about a specific study first if you would like its direct explorer link.

<a id="q176"></a>

## 176. Give the numeric table for 2045 too.

Category: multi-turn follow-up<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 6.16 s<br>
Conversation: `failed_scope_recovery`

**Full answer**

I need one more detail. Which variable should I use?

<a id="q177"></a>

## 177. Find the IAM PARIS Application Library.

Category: multi-turn follow-up<br>
Route: `general_qa` (explicit site navigation request)<br>
Qwen generations: 0<br>
Time: 0.05 s<br>
Conversation: `navigation_context`

**Full answer**

Use these IAM PARIS links for this request: [Application Library](https://iamparis.eu/application_library).

<a id="q178"></a>

## 178. What about Aqueduct there?

Category: multi-turn follow-up<br>
Route: `general_qa` (site navigation follow-up)<br>
Qwen generations: 0<br>
Time: 0.04 s<br>
Conversation: `navigation_context`

**Full answer**

Use these IAM PARIS links for this request: [Aqueduct](https://iamparis.eu/application_library/474).

<a id="q179"></a>

## 179. Where can I find its detail page?

Category: multi-turn follow-up<br>
Route: `general_qa` (explicit site navigation request)<br>
Qwen generations: 0<br>
Time: 0.03 s<br>
Conversation: `navigation_context`

**Full answer**

I could not match this request to a reliable IAM PARIS link.

<a id="q180"></a>

## 180. Now look for Climate Watch in that library.

Category: multi-turn follow-up<br>
Route: `general_qa` (general climate/policy question)<br>
Qwen generations: 3<br>
Time: 1.79 s<br>
Conversation: `navigation_context`

**Full answer**

The Climate Watch detail page can be found on the IAM PARIS website, where users can access detailed information about climate-related data and scenarios.

<a id="q181"></a>

## 181. Return to the main Results page.

Category: multi-turn follow-up<br>
Route: `general_qa` (explicit site navigation request)<br>
Qwen generations: 0<br>
Time: 0.07 s<br>
Conversation: `navigation_context`

**Full answer**

Use these IAM PARIS links for this request: [Results](https://iamparis.eu/results).

<a id="q182"></a>

## 182. How do I download a study dataset from there?

Category: multi-turn follow-up<br>
Route: `general_qa` (data download navigation)<br>
Qwen generations: 0<br>
Time: 0.03 s<br>
Conversation: `navigation_context`

**Full answer**

Use these IAM PARIS links for this request: [Results](https://iamparis.eu/results).

I do not have a verified direct file-download URL. Open the Results page and select the study whose data you need.

<a id="q183"></a>

## 183. Plot coal primary energy for China from 2030 to 2050.

Category: multi-turn follow-up<br>
Route: `general_qa` (charts delegated to data explorer)<br>
Qwen generations: 0<br>
Time: 0.00 s<br>
Conversation: `plot_context`

**Full answer**

I do not generate figures in the chat. Please use the [IAM PARIS data explorer](https://iamparis.eu/results) to explore, filter, and plot the data. Ask about a specific study first if you would like its direct explorer link.

<a id="q184"></a>

## 184. Change only the variable to gas primary energy.

Category: multi-turn follow-up<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.42 s<br>
Conversation: `plot_context`

**Full answer**

Showing results from the study **Where is the world headed?**. Also available in: Post-Glasgow targets; Global impacts of NDCs; Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU; Tech-constrained EU pathways to net zero. Name another study to switch.

### Primary Energy|Gas in China

Scope: scenario `multiple`, model `multiple`, years `2030-2050`
Unit: `EJ/yr`

Answer:
In 2045, Primary Energy|Gas in China ranges from 5.10 (ICES-XPS, PR_NDC_CP) to 37.74 EJ/yr (42, PR_CurPol_EI) across 36 series; median 15.44. Median change since 2030: +11%.

Summary: 36 series across 6 models and 8 scenarios.
Showing 12 of 36 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 24 remaining series.

| Model | Scenario | 2030 | 2035 | 2040 | 2045 | 2050 | Unit |
|---|---|---|---|---|---|---|---|
| 42 | PR_Baseline | 20.17 | 24.29 | 27.92 | 31.41 | — | EJ/yr |
| 42 | PR_CurPol_EI | 24.02 | 29.01 | 33.50 | 37.74 | — | EJ/yr |
| E3ME | PR_Baseline | 13.22 | 13.64 | 13.77 | 13.89 | 14.87 | EJ/yr |
| E3ME | PR_CurPol_CP | 13.22 | 13.56 | 13.65 | 13.78 | 14.79 | EJ/yr |
| GCAM | PR_Baseline | 10.74 | 11.91 | 12.80 | 13.80 | 14.43 | EJ/yr |
| GCAM | PR_CurPol_CP | 23.33 | 23.63 | 23.81 | 23.51 | 23.40 | EJ/yr |
| GEMINI-E3 | PR_Baseline | 13.93 | 14.66 | 15.13 | 15.42 | 15.56 | EJ/yr |
| GEMINI-E3 | PR_CurPol_CP | 15.22 | 15.47 | 15.54 | 15.45 | 15.07 | EJ/yr |
| ICES-XPS | PR_Baseline | 4.68 | 5.26 | 5.61 | 6.18 | 6.64 | EJ/yr |
| ICES-XPS | PR_NDC_CP | 4.60 | 4.71 | 4.82 | 5.10 | 5.29 | EJ/yr |
| MUSE | PR_Baseline | 13.03 | 13.41 | 15.29 | 16.05 | 19.58 | EJ/yr |
| MUSE | PR_CurPol_CP | 13.12 | 13.59 | 15.77 | 18.01 | 22.15 | EJ/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q185"></a>

## 185. Add the PR_CurPol_CP scenario.

Category: multi-turn follow-up<br>
Route: `data_query` (extracted data entities)<br>
Qwen generations: 0<br>
Time: 0.42 s<br>
Conversation: `plot_context`

**Full answer**

I can answer questions about IAM PARIS results, models and climate-policy terms, but I need a little more to go on. Try for example:
- `CO2 emissions for World in 2050`
- `which scenario has the lowest emissions in 2050?`
- `which models cover transport?`
- `what does net zero mean?`

<a id="q186"></a>

## 186. Narrow the interval to 2035 through 2045.

Category: multi-turn follow-up<br>
Route: `data_query` (unclear deterministic route)<br>
Qwen generations: 1<br>
Time: 0.45 s<br>
Conversation: `plot_context`

**Full answer**

I can answer questions about IAM PARIS results, models and climate-policy terms, but I need a little more to go on. Try for example:
- `CO2 emissions for World in 2050`
- `which scenario has the lowest emissions in 2050?`
- `which models cover transport?`
- `what does net zero mean?`

<a id="q187"></a>

## 187. Show a numeric answer for that scope.

Category: multi-turn follow-up<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.55 s<br>
Conversation: `plot_context`

**Full answer**

### Primary Energy|Gas in China

Scope: scenario `multiple`, model `multiple`, years `2030-2050`
Unit: `EJ/yr`

Answer:
In 2045, Primary Energy|Gas in China ranges from 5.10 (ICES-XPS, PR_NDC_CP) to 37.74 EJ/yr (42, PR_CurPol_EI) across 36 series; median 15.44. Median change since 2030: +11%.

Summary: 36 series across 6 models and 8 scenarios.
Showing 12 of 36 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 24 remaining series.

| Model | Scenario | 2030 | 2035 | 2040 | 2045 | 2050 | Unit |
|---|---|---|---|---|---|---|---|
| 42 | PR_Baseline | 20.17 | 24.29 | 27.92 | 31.41 | — | EJ/yr |
| 42 | PR_CurPol_EI | 24.02 | 29.01 | 33.50 | 37.74 | — | EJ/yr |
| E3ME | PR_Baseline | 13.22 | 13.64 | 13.77 | 13.89 | 14.87 | EJ/yr |
| E3ME | PR_CurPol_CP | 13.22 | 13.56 | 13.65 | 13.78 | 14.79 | EJ/yr |
| GCAM | PR_Baseline | 10.74 | 11.91 | 12.80 | 13.80 | 14.43 | EJ/yr |
| GCAM | PR_CurPol_CP | 23.33 | 23.63 | 23.81 | 23.51 | 23.40 | EJ/yr |
| GEMINI-E3 | PR_Baseline | 13.93 | 14.66 | 15.13 | 15.42 | 15.56 | EJ/yr |
| GEMINI-E3 | PR_CurPol_CP | 15.22 | 15.47 | 15.54 | 15.45 | 15.07 | EJ/yr |
| ICES-XPS | PR_Baseline | 4.68 | 5.26 | 5.61 | 6.18 | 6.64 | EJ/yr |
| ICES-XPS | PR_NDC_CP | 4.60 | 4.71 | 4.82 | 5.10 | 5.29 | EJ/yr |
| MUSE | PR_Baseline | 13.03 | 13.41 | 15.29 | 16.05 | 19.58 | EJ/yr |
| MUSE | PR_CurPol_CP | 13.12 | 13.59 | 15.77 | 18.01 | 22.15 | EJ/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q188"></a>

## 188. Which models are represented in it?

Category: multi-turn follow-up<br>
Route: `data_query` (scoped availability request)<br>
Qwen generations: 0<br>
Time: 0.15 s<br>
Conversation: `plot_context`

**Full answer**

### Availability for Primary Energy|Gas

**Models (0):** none

Recorded unit: `multiple`.

<a id="q189"></a>

## 189. Compare GCAM with IMAGE as model frameworks.

Category: multi-turn follow-up<br>
Route: `model_explanation` (multi-model metadata comparison)<br>
Qwen generations: 0<br>
Time: 0.08 s<br>
Conversation: `model_compare_followup`

**Full answer**

### GCAM vs IMAGE

**GCAM**
- Developer: Joint Global Change Research Institute
- Systems/sectors: energy, economy, land, water, climate
- Common uses: emissions and energy pathway analysis; climate policy scenarios; land and resource interactions; regional mitigation comparisons

**IMAGE**
- Developer: IMAGE team under the authority of PBL Netherlands Environmental Assessment Agency
- Systems/sectors: energy, land, climate, biodiversity, human development
- Common uses: long-term global environmental and sustainable-development pathways; climate, biodiversity and land-use policy assessment

Shared coverage: climate, energy, land.
Distinctive in GCAM: economy, water.
Distinctive in IMAGE: biodiversity, human development.

[Browse IAM PARIS model documentation](https://iamparis.eu/models)

<a id="q190"></a>

## 190. Which one includes land-use interactions?

Category: multi-turn follow-up<br>
Route: `model_explanation` (referenced-model qualitative comparison)<br>
Qwen generations: 0<br>
Time: 0.12 s<br>
Conversation: `model_compare_followup`

**Full answer**

### GCAM vs IMAGE

**GCAM**
- Developer: Joint Global Change Research Institute
- Systems/sectors: energy, economy, land, water, climate
- Common uses: emissions and energy pathway analysis; climate policy scenarios; land and resource interactions; regional mitigation comparisons

**IMAGE**
- Developer: IMAGE team under the authority of PBL Netherlands Environmental Assessment Agency
- Systems/sectors: energy, land, climate, biodiversity, human development
- Common uses: long-term global environmental and sustainable-development pathways; climate, biodiversity and land-use policy assessment

Catalogue wording match for `includes`, `land`, `use`: **IMAGE**.
Shared coverage: climate, energy, land.
Distinctive in GCAM: economy, water.
Distinctive in IMAGE: biodiversity, human development.

[Browse IAM PARIS model documentation](https://iamparis.eu/models)

<a id="q191"></a>

## 191. How do their methodological approaches differ?

Category: multi-turn follow-up<br>
Route: `model_explanation` (referenced-model qualitative comparison)<br>
Qwen generations: 0<br>
Time: 0.13 s<br>
Conversation: `model_compare_followup`

**Full answer**

### GCAM vs IMAGE

**GCAM**
- Methodology/model type: Partial equilibrium

**IMAGE**
- Methodology/model type: IMAGE (Integrated Model to Assess the Global Environment) is a global integrated assessment modelling framework representing interactions among society, the biosphere and the climate system.

The loaded catalogue metadata does not explicitly match `methodological` for either model.
Comparison focus: method.

[Browse IAM PARIS model documentation](https://iamparis.eu/models)

<a id="q192"></a>

## 192. Give documentation for the first model.

Category: multi-turn follow-up<br>
Route: `general_qa` (explicit site navigation request)<br>
Qwen generations: 0<br>
Time: 0.04 s<br>
Conversation: `model_compare_followup`

**Full answer**

Use these IAM PARIS links for this request: [Models](https://iamparis.eu/models).

<a id="q193"></a>

## 193. Which one has loaded World CO2 data?

Category: multi-turn follow-up<br>
Route: `data_query` (referenced-model availability)<br>
Qwen generations: 0<br>
Time: 0.30 s<br>
Conversation: `model_compare_followup`

**Full answer**

### Availability among the referenced models

Scope: variable `Emissions|CO2`, region `World`.

- `GCAM`: available via `gcam`.
- `IMAGE`: no matching rows in the loaded time-series data.

<a id="q194"></a>

## 194. Now compare their 2050 emissions if both are available.

Category: multi-turn follow-up<br>
Route: `data_query` (referenced-model availability)<br>
Qwen generations: 0<br>
Time: 0.30 s<br>
Conversation: `model_compare_followup`

**Full answer**

### Availability among the referenced models

Scope: variable `Emissions|CO2`, year `2050`.

- `GCAM`: available via `gcam`.
- `IMAGE`: no matching rows in the loaded time-series data.

<a id="q195"></a>

## 195. Show China methane emissions from 2020 through 2060.

Category: multi-turn follow-up<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.31 s<br>
Conversation: `year_refinement`

**Full answer**

Showing results from the study **Where is the world headed?**. Also available in: Global impacts of NDCs; Post-Glasgow targets; Comparison of Fit-for-55 Policy and Cost Optimal Scenarios for the EU. Name another study to switch.

### Emissions|CH4 in CHN

Scope: scenario `multiple`, model `multiple`, years `2020-2060`
Unit: `Mt CH4/yr`

Answer:
In 2050, Emissions|CH4 in CHN ranges from 36.34 (GCAM, PR_CurPol_EI) to 85.82 Mt CH4/yr (GEMINI-E3, PR_Baseline) across 13 series; median 50.59. Median change since 2020: -28%.

Summary: 13 series across 2 models and 7 scenarios.
Showing 12 of 13 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 1 remaining series.

| Model | Scenario | 2020 | 2025 | 2030 | 2035 | 2040 | 2045 | 2050 | 2055 | 2060 | Unit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GCAM | PR_Baseline | 71.06 | 68.76 | 64.72 | 61.26 | 57.78 | 53.70 | 50.59 | 48.48 | 46.79 | Mt CH4/yr |
| GCAM | PR_CurPol_CP | 51.17 | 49.38 | 46.74 | 44.13 | 41.50 | 38.83 | 36.65 | 34.77 | 33.12 | Mt CH4/yr |
| GCAM | PR_CurPol_CPO | 51.99 | 50.56 | 48.03 | 45.37 | 42.50 | 39.64 | 37.37 | 35.33 | 33.53 | Mt CH4/yr |
| GCAM | PR_CurPol_EI | 51.17 | 49.38 | 46.74 | 44.12 | 41.24 | 38.54 | 36.34 | 33.32 | 31.04 | Mt CH4/yr |
| GCAM | PR_NDC_CP | 51.18 | 50.33 | 48.06 | 46.18 | 43.57 | 41.09 | 38.95 | 36.95 | 35.04 | Mt CH4/yr |
| GCAM | PR_NDC_CPO | 51.99 | 50.12 | 47.34 | 44.67 | 41.81 | 38.97 | 36.60 | 34.49 | 32.64 | Mt CH4/yr |
| GEMINI-E3 | PR_Baseline | 75.54 | 78.23 | 80.33 | 82.03 | 83.59 | 84.87 | 85.82 | — | — | Mt CH4/yr |
| GEMINI-E3 | PR_CurPol_CP | 73.95 | 74.05 | 74.44 | 74.98 | 75.65 | 75.51 | 74.52 | — | — | Mt CH4/yr |
| GEMINI-E3 | PR_CurPol_CPO | 70.87 | 61.93 | 57.66 | 57.54 | 58.05 | 58.39 | 58.37 | — | — | Mt CH4/yr |
| GEMINI-E3 | PR_CurPol_EI | 73.95 | 74.05 | 74.44 | 68.96 | 59.50 | 53.06 | 51.82 | — | — | Mt CH4/yr |
| GEMINI-E3 | PR_NDC_CP | 73.97 | 74.07 | 74.53 | 74.82 | 75.49 | 75.31 | 74.35 | — | — | Mt CH4/yr |
| GEMINI-E3 | PR_NDC_EI | 73.97 | 74.07 | 74.53 | 68.75 | 59.49 | 52.87 | 51.72 | — | — | Mt CH4/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q196"></a>

## 196. Only retain values through 2045.

Category: multi-turn follow-up<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.39 s<br>
Conversation: `year_refinement`

**Full answer**

### Emissions|CH4 in CHN

Scope: scenario `multiple`, model `multiple`, years `first available-2045`
Unit: `Mt CH4/yr`

Answer:
In 2045, Emissions|CH4 in CHN ranges from 38.54 (GCAM, PR_CurPol_EI) to 84.87 Mt CH4/yr (GEMINI-E3, PR_Baseline) across 13 series; median 52.87. Median change since 2005: -30%.

Summary: 13 series across 2 models and 7 scenarios.
Showing 12 of 13 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 1 remaining series.

| Model | Scenario | 2005 | 2010 | 2015 | 2020 | 2025 | 2030 | 2035 | 2040 | 2045 | Unit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GCAM | PR_Baseline | 56.32 | 68.73 | 73.23 | 71.06 | 68.76 | 64.72 | 61.26 | 57.78 | 53.70 | Mt CH4/yr |
| GCAM | PR_CurPol_CP | 56.32 | 68.73 | 73.23 | 51.17 | 49.38 | 46.74 | 44.13 | 41.50 | 38.83 | Mt CH4/yr |
| GCAM | PR_CurPol_CPO | 56.32 | 68.73 | 73.23 | 51.99 | 50.56 | 48.03 | 45.37 | 42.50 | 39.64 | Mt CH4/yr |
| GCAM | PR_CurPol_EI | 56.32 | 68.73 | 73.23 | 51.17 | 49.38 | 46.74 | 44.12 | 41.24 | 38.54 | Mt CH4/yr |
| GCAM | PR_NDC_CP | 56.32 | 68.73 | 73.23 | 51.18 | 50.33 | 48.06 | 46.18 | 43.57 | 41.09 | Mt CH4/yr |
| GCAM | PR_NDC_CPO | 56.32 | 68.73 | 73.23 | 51.99 | 50.12 | 47.34 | 44.67 | 41.81 | 38.97 | Mt CH4/yr |
| GEMINI-E3 | PR_Baseline | — | — | 73.09 | 75.54 | 78.23 | 80.33 | 82.03 | 83.59 | 84.87 | Mt CH4/yr |
| GEMINI-E3 | PR_CurPol_CP | — | — | 73.09 | 73.95 | 74.05 | 74.44 | 74.98 | 75.65 | 75.51 | Mt CH4/yr |
| GEMINI-E3 | PR_CurPol_CPO | — | — | 73.09 | 70.87 | 61.93 | 57.66 | 57.54 | 58.05 | 58.39 | Mt CH4/yr |
| GEMINI-E3 | PR_CurPol_EI | — | — | 73.09 | 73.95 | 74.05 | 74.44 | 68.96 | 59.50 | 53.06 | Mt CH4/yr |
| GEMINI-E3 | PR_NDC_CP | — | — | 73.09 | 73.97 | 74.07 | 74.53 | 74.82 | 75.49 | 75.31 | Mt CH4/yr |
| GEMINI-E3 | PR_NDC_EI | — | — | 73.09 | 73.97 | 74.07 | 74.53 | 68.75 | 59.49 | 52.87 | Mt CH4/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q197"></a>

## 197. What is the change from 2020 to 2045?

Category: multi-turn follow-up<br>
Route: `data_query` (data request)<br>
Qwen generations: 0<br>
Time: 0.48 s<br>
Conversation: `year_refinement`

**Full answer**

### Emissions|CH4 in CHN

Scope: scenario `multiple`, model `multiple`, years `2020-2045`
Unit: `Mt CH4/yr`

Answer:
**Change from 2020 to 2045:** median -24% across 13 series (range -29% to +12%).

Summary: 13 series across 2 models and 7 scenarios.
Showing 12 of 13 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 1 remaining series.

| Model | Scenario | 2020 | 2025 | 2030 | 2035 | 2040 | 2045 | Unit |
|---|---|---|---|---|---|---|---|---|
| GCAM | PR_Baseline | 71.06 | 68.76 | 64.72 | 61.26 | 57.78 | 53.70 | Mt CH4/yr |
| GCAM | PR_CurPol_CP | 51.17 | 49.38 | 46.74 | 44.13 | 41.50 | 38.83 | Mt CH4/yr |
| GCAM | PR_CurPol_CPO | 51.99 | 50.56 | 48.03 | 45.37 | 42.50 | 39.64 | Mt CH4/yr |
| GCAM | PR_CurPol_EI | 51.17 | 49.38 | 46.74 | 44.12 | 41.24 | 38.54 | Mt CH4/yr |
| GCAM | PR_NDC_CP | 51.18 | 50.33 | 48.06 | 46.18 | 43.57 | 41.09 | Mt CH4/yr |
| GCAM | PR_NDC_CPO | 51.99 | 50.12 | 47.34 | 44.67 | 41.81 | 38.97 | Mt CH4/yr |
| GEMINI-E3 | PR_Baseline | 75.54 | 78.23 | 80.33 | 82.03 | 83.59 | 84.87 | Mt CH4/yr |
| GEMINI-E3 | PR_CurPol_CP | 73.95 | 74.05 | 74.44 | 74.98 | 75.65 | 75.51 | Mt CH4/yr |
| GEMINI-E3 | PR_CurPol_CPO | 70.87 | 61.93 | 57.66 | 57.54 | 58.05 | 58.39 | Mt CH4/yr |
| GEMINI-E3 | PR_CurPol_EI | 73.95 | 74.05 | 74.44 | 68.96 | 59.50 | 53.06 | Mt CH4/yr |
| GEMINI-E3 | PR_NDC_CP | 73.97 | 74.07 | 74.53 | 74.82 | 75.49 | 75.31 | Mt CH4/yr |
| GEMINI-E3 | PR_NDC_EI | 73.97 | 74.07 | 74.53 | 68.75 | 59.49 | 52.87 | Mt CH4/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q198"></a>

## 198. Use Brazil for the same metric and dates.

Category: multi-turn follow-up<br>
Route: `data_query` (explicit result scope edit)<br>
Qwen generations: 0<br>
Time: 0.53 s<br>
Conversation: `year_refinement`

**Full answer**

### Emissions|CH4 in BRA

Scope: scenario `multiple`, model `multiple`, years `2020-2045`
Unit: `Mt CH4/yr`

Answer:
In 2045, Emissions|CH4 in BRA ranges from 12.67 (GEMINI-E3, PR_NDC_EI) to 23.88 Mt CH4/yr (GCAM, PR_Baseline) across 13 series; median 20.03. Median change since 2020: +1%.

Summary: 13 series across 2 models and 7 scenarios.
Showing 12 of 13 series (every model and the highest and lowest values included), ordered by model and scenario. Narrow by model or scenario to view the 1 remaining series.

| Model | Scenario | 2020 | 2025 | 2030 | 2035 | 2040 | 2045 | Unit |
|---|---|---|---|---|---|---|---|---|
| GCAM | PR_Baseline | 21.31 | 22.00 | 22.63 | 23.10 | 23.48 | 23.88 | Mt CH4/yr |
| GCAM | PR_CurPol_CP | 20.84 | 20.63 | 21.28 | 21.70 | 21.94 | 22.00 | Mt CH4/yr |
| GCAM | PR_CurPol_CPO | 20.97 | 20.32 | 20.71 | 21.10 | 21.39 | 21.73 | Mt CH4/yr |
| GCAM | PR_CurPol_EI | 20.84 | 20.63 | 21.28 | 20.80 | 20.50 | 20.51 | Mt CH4/yr |
| GCAM | PR_NDC_CP | 20.84 | 20.97 | 21.82 | 22.52 | 22.63 | 22.59 | Mt CH4/yr |
| GCAM | PR_NDC_CPO | 20.98 | 20.30 | 20.66 | 21.03 | 21.32 | 21.65 | Mt CH4/yr |
| GEMINI-E3 | PR_Baseline | 19.26 | 19.51 | 19.71 | 19.73 | 19.69 | 19.51 | Mt CH4/yr |
| GEMINI-E3 | PR_CurPol_CP | 19.26 | 19.49 | 19.68 | 19.67 | 19.58 | 19.36 | Mt CH4/yr |
| GEMINI-E3 | PR_CurPol_CPO | 19.26 | 19.50 | 19.31 | 19.31 | 19.21 | 18.96 | Mt CH4/yr |
| GEMINI-E3 | PR_CurPol_EI | 19.26 | 19.49 | 19.68 | 19.33 | 18.66 | 17.99 | Mt CH4/yr |
| GEMINI-E3 | PR_NDC_CP | 19.22 | 18.08 | 16.79 | 16.59 | 16.25 | 15.91 | Mt CH4/yr |
| GEMINI-E3 | PR_NDC_EI | 19.22 | 18.08 | 16.79 | 15.75 | 14.38 | 12.67 | Mt CH4/yr |

For interactive charts and additional filters, use the [IAM PARIS data explorer](https://iamparis.eu/results/paris-reinforce/where-is-the-world-headed/graphs).

<a id="q199"></a>

## 199. What is the latest available year for that slice?

Category: multi-turn follow-up<br>
Route: `data_query` (catalogue year coverage request)<br>
Qwen generations: 0<br>
Time: 0.08 s<br>
Conversation: `year_refinement`

**Full answer**

The latest available projection year is **2100**. The loaded records span 2005–2100 (45 distinct years).

<a id="q200"></a>

## 200. Explain why a projection might end before 2100.

Category: multi-turn follow-up<br>
Route: `data_query` (low confidence entity clarification)<br>
Qwen generations: 0<br>
Time: 0.25 s<br>
Conversation: `year_refinement`

**Full answer**

I could not confidently match all of the requested wording. The loaded variable catalogue does not represent these term(s) in the top match: `explain`, `might`, `projection`. Choose the variable: 1. `IDA|Emissions|CO2|Energy|Demand|Industry|Emission end level 2050` (CO2 emissions, industry) 2. `IDA|Emissions|CO2|Energy|Demand|Buildings|Emission end level 2050` (CO2 emissions, buildings) 3. `IDA|Emissions|CO2|Energy|Demand|Transport|Emission end level 2050` (CO2 emissions, transport) Reply with a number (1-3), or `yes` for option 1.
