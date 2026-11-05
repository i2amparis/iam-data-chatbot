# Fresh 200-query local Qwen audit

Run on 5 November 2026 through the app session-query path, with the configured local
`qwen3:0.6b-q4_K_M` model, 631,201 cached IAM PARIS records, and the local knowledge index.
The [question set](qwen_unique_200_questions.json) has 140 standalone prompts and ten six-turn
conversations. Its normalized prompt strings are unique and do not repeat the earlier 200-question audit.
Read [all routes and full answers](qwen_unique_200_answers.md) or the
[raw JSONL results](qwen_unique_200_audit.jsonl) with entities, links, timings, and Qwen calls.

## Run summary

- Completed: **200/200**; execution errors: **0**.
- Routes: **127** data query, **28** model explanation, **45** general QA.
- Qwen generated a response or routing output for **35** prompts, across **51** chat generations. The other **165** turns used deterministic code or cached material.
- Median turn time: **0.39 s**; total measured turn time: **153.8 s**.

| Question category | Count |
|---|---:|
| numeric data and comparisons | 35 |
| model descriptions and methodology | 25 |
| coverage and catalogue | 25 |
| conceptual explanations for Qwen | 25 |
| website and documentation links | 15 |
| ambiguity and recovery | 15 |
| multi-turn follow-up | 60 |

## Wrong or incomplete answer examples

These are manually checked examples from this run. The list is not an exhaustive accuracy score.
Some are routing or follow-up-state failures; others are answer-quality failures.

| Query | Observed problem |
|---:|---|
| [024](qwen_unique_200_answers.md#q024) | Asked for average global electricity output across models; returned a broad list of energy-supply models. |
| [033](qwen_unique_200_answers.md#q033) | Asked for CO2 values in a named study; returned a study listing without values. |
| [034](qwen_unique_200_answers.md#q034) | Asked for the earliest global population value; confirmed availability but did not give the value. |
| [037](qwen_unique_200_answers.md#q037) | Misread the model name WITCH as “with” and gave unrelated model information. |
| [044](qwen_unique_200_answers.md#q044) | ATOM methodology question was treated as a numeric data query. |
| [054](qwen_unique_200_answers.md#q054) | IMAGE versus GEM-E3 sector comparison was treated as an invalid region. |
| [058](qwen_unique_200_answers.md#q058) | GCAM-PR documentation question returned an unrelated food-availability lookup. |
| [061](qwen_unique_200_answers.md#q061) | Asked which models have Brazil results; searched for a variable called “loaded results for brazil”. |
| [074](qwen_unique_200_answers.md#q074) | PROMETHEUS population coverage question silently used USA although no region was requested. |
| [084](qwen_unique_200_answers.md#q084) | Asked how many models have France results; searched for a variable called “results for france”. |
| [086](qwen_unique_200_answers.md#q086) | Coal-versus-gas explanation returned a catalogue summary. |
| [090](qwen_unique_200_answers.md#q090) | Meaning of a negative land-use CO2 value became an unrelated numeric CO2 table. |
| [097](qwen_unique_200_answers.md#q097) | Capacity-versus-generation definition became a numeric lookup. |
| [099](qwen_unique_200_answers.md#q099) | Currency-year carbon-price comparison omitted the required currency and price-year conversion. |
| [100](qwen_unique_200_answers.md#q100) | Model-spread interpretation was treated as an unknown model name. |
| [101](qwen_unique_200_answers.md#q101) | Baseline-versus-forecast explanation returned generic help. |
| [110](qwen_unique_200_answers.md#q110) | Qwen claimed flat electricity demand follows from electrification; this explanation is internally inconsistent. |
| [130](qwen_unique_200_answers.md#q130) | Unknown MODEL-X availability question returned the model directory link without answering availability. |
| [133](qwen_unique_200_answers.md#q133) | Meaning of “Policy” was treated as an invalid region. |
| [146](qwen_unique_200_answers.md#q146) | Explorer follow-up lost the named study context and returned no link. |
| [149](qwen_unique_200_answers.md#q149) | “Stop at 2040” collapsed the 2030–2050 interval to a single year. |
| [151](qwen_unique_200_answers.md#q151) | Asked why model values differ; repeated the numeric data table. |
| [152](qwen_unique_200_answers.md#q152) | Asked to return to solar; kept the wind variable. |
| [160](qwen_unique_200_answers.md#q160) | Option 2 after an energy-variable clarification resolved to unrelated carbon removal. |
| [166](qwen_unique_200_answers.md#q166) | Baseline versus current-policy comparison returned multiple scenarios without separating the two families. |
| [168](qwen_unique_200_answers.md#q168) | Region switch to India lost the comparison context. |
| [180](qwen_unique_200_answers.md#q180) | Climate Watch detail-page follow-up produced an unsourced description without a URL. |
| [185](qwen_unique_200_answers.md#q185) | Adding a scenario lost the prior China/gas scope. |
| [187](qwen_unique_200_answers.md#q187) | After interval refinement, numeric answer still showed the old 2030–2050 range. |
| [188](qwen_unique_200_answers.md#q188) | After a numeric gas answer, model-list follow-up reported zero models. |
| [196](qwen_unique_200_answers.md#q196) | “Only retain values through 2045” lost the original 2020 start year. |
| [200](qwen_unique_200_answers.md#q200) | General explanation of why projections end early was mistaken for a variable lookup. |

## Patterns and next fixes

1. Route conceptual questions before attempting data or model entity extraction. Common trigger words
   such as “model”, “energy”, “capacity”, and “emissions” currently pull explanations into lookups.
2. Handle coverage questions as scoped catalogue queries: models by region, variable, study, year,
   and scenario. Preserve explicit constraints and avoid silently injecting a default region.
3. Keep conversation state through clarification selections and study, scenario, variable, and
   interval edits. Apply edits to the last valid scope and check that the resulting answer reflects them.
4. Resolve model names against the catalogue before fuzzy common-word matching; preserve model
   families and versions in comparisons and documentation requests.
5. Add an answer check for unsupported external-site details and for conceptual claims that are
   self-contradictory or omit essential unit conversion steps.

The audit used local cached data and Ollama. It did not verify external URLs or recalculate every
numeric value against the source dataset, so the findings are examples rather than a full correctness rate.
