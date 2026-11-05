# Second 1,000-query local Qwen audit

Run on 5 November 2026 through the app session-query path using `qwen3:0.6b-q4_K_M`, 631,201 cached IAM PARIS records, and the local knowledge index. The [question set](qwen_second_1000_questions.json) contains 700 standalone prompts and 50 six-turn conversations. Normalized prompts are unique and have no overlap with the previous 1,400 audited prompts.

Read [all 1,000 routes and full answers](qwen_second_1000_answers.md), or inspect the [raw JSONL results](qwen_second_1000_audit.jsonl) for entities, links, timing, and Qwen-call metadata.

## Run totals

- Completed: **1,000/1,000**; execution errors: **0**.
- Routes: **613** data query, **150** model explanation, **237** general Q&A.
- Qwen was invoked on **175** prompts, across **207** chat generations. The remaining turns used deterministic or cached answer logic.
- Median turn time: **0.41 s**; total measured turn time: **588.1 s**.

## Routes by category

| Category | Queries | Data query | Model explanation | General Q&A | Qwen invoked |
|---|---:|---:|---:|---:|---:|
| numeric values and comparisons | 160 | 160 | 0 | 0 | 0 |
| model profiles and methodology | 120 | 11 | 86 | 23 | 8 |
| data coverage and provenance | 100 | 98 | 0 | 2 | 3 |
| conceptual explanation for Qwen | 240 | 91 | 33 | 116 | 148 |
| navigation and links | 40 | 4 | 0 | 36 | 0 |
| ambiguous and invalid requests | 40 | 40 | 0 | 0 | 0 |
| six-turn follow-up | 300 | 209 | 31 | 60 | 16 |

## Checked wrong or incomplete responses

These are concrete examples checked against the prompt and answer. The list is not exhaustive, and zero execution errors does not mean all answers are correct.

| Query | Finding |
|---:|---|
| [0001](qwen_second_1000_answers.md#q0001) | The question names Argentina, but the bot parsed “what is the” as a region. |
| [0160](qwen_second_1000_answers.md#q0160) | Morocco was rejected as an unknown region; this is an availability/alias issue to inspect, rather than proof the source has Morocco data. |
| [0280](qwen_second_1000_answers.md#q0280) | Asked about DLR_LEV scenario inputs and outputs; answer listed scenario names only. |
| [0330](qwen_second_1000_answers.md#q0330) | Asked which study has widest solar coverage for Japan; answer listed all studies without ranking them. |
| [0400](qwen_second_1000_answers.md#q0400) | A conceptual bioenergy assumption question misread “change” as the CHANCE model. |
| [0430](qwen_second_1000_answers.md#q0430) | A renewable-curtailment explanation made the same “change” to CHANCE substitution. |
| [0520](qwen_second_1000_answers.md#q0520) | An industrial-electrification explanation returned a study link instead. |
| [0610](qwen_second_1000_answers.md#q0610) | An explanation about uncertainty bands returned generic help. |
| [0670](qwen_second_1000_answers.md#q0670) | An invented Fantasia model was ignored while a numeric data response was returned. |
| [0702](qwen_second_1000_answers.md#q0702) | A follow-up switched to Kenya, but the answer stayed scoped to Argentina. |
| [0703](qwen_second_1000_answers.md#q0703) | The following year edit still answered for Argentina despite explicitly saying Kenya. |
| [0704](qwen_second_1000_answers.md#q0704) | A GCAM filter did not replace the previously selected GCAM-PR 5.3 model. |
| [0710](qwen_second_1000_answers.md#q0710) | MESSAGEix-GLOBIOM coverage request rejected Vietnam instead of preserving the model and explaining availability. |
| [0711](qwen_second_1000_answers.md#q0711) | The next turn parsed “model MESSAGEix-GLOBIOM” as a region. |
| [0712](qwen_second_1000_answers.md#q0712) | The model documentation request linked to a policy catalogue rather than the model page. |
| [0716](qwen_second_1000_answers.md#q0716) | A year edit lost the Baseline/NDC comparison and returned generic help. |
| [0717](qwen_second_1000_answers.md#q0717) | A follow-up reported zero models after a carbon-price comparison with model values. |
| [0722](qwen_second_1000_answers.md#q0722) | After a chart redirect, an explicit request for numeric values was redirected to charting again. |
| [0724](qwen_second_1000_answers.md#q0724) | The requested study link could not be resolved from the conversation context. |

## Main patterns

- Conceptual questions containing “change” repeatedly matched the CHANCE model. Other explanations were sent to data lookup or website navigation.
- Some coverage and provenance requests returned broad catalogues or study lists rather than the requested scoped answer.
- Follow-ups sometimes lost the region, selected model, year range, or scenario comparison.
- Invented model names were not always rejected before the bot produced data for the remaining scope.

## Suggested fixes

1. Resolve explanatory intent before fuzzy model-name matching; require a word-boundary or catalogue-backed match for CHANCE.
2. Validate each requested region, model, variable, scenario, and year against the output scope before returning a data answer.
3. Preserve and edit conversation state through follow-ups and chart redirects.
4. Make coverage and provenance answers apply the exact requested filters and ranking.

The run used local cached data and Ollama. It did not verify external links or independently recompute all numeric values, so these findings are examples rather than an overall accuracy rate.
