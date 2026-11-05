# Local Qwen 1,000-query audit

Run on 5 November 2026 through the application session-query path with `qwen3:0.6b-q4_K_M`, 631,201 cached IAM PARIS records, and the local knowledge index. The [question fixture](qwen_1000_questions.json) contains 700 standalone prompts and 50 six-turn conversations. Normalized prompts are unique and have no overlap with either earlier 200-question audit.

Read [all 1,000 routes and full answers](qwen_1000_answers.md) or the [raw results](qwen_1000_audit.jsonl), which include answer, route reason, entities, clarification, provenance, links, timing, and per-turn Qwen-call metadata.

## Run totals

- Completed: **1000/1,000**; execution errors: **0**.
- Routes: **670** data query, **124** model explanation, **206** general Q&A.
- Qwen was invoked on **69** prompts, for **77** chat generations. The other turns used deterministic routing, metadata, glossary, or cached-data logic.
- Median turn time: **0.35 s**. Total measured turn time: **439.16 s**.

## Routes by category

| Category | Queries | Data query | Model explanation | General Q&A | Qwen invoked |
|---|---:|---:|---:|---:|---:|
| numeric data and comparison | 220 | 219 | 1 | 0 | 0 |
| model descriptions and methods | 100 | 2 | 98 | 0 | 2 |
| catalogue and coverage | 110 | 110 | 0 | 0 | 0 |
| conceptual explanation | 100 | 27 | 3 | 70 | 60 |
| navigation and documentation | 59 | 3 | 1 | 55 | 3 |
| ambiguity and recovery | 60 | 60 | 0 | 0 | 0 |
| charts and visualization | 51 | 0 | 0 | 51 | 0 |
| multi-turn follow-up | 300 | 249 | 21 | 30 | 4 |

## Checked wrong or incomplete answers

The examples below were checked against the prompt and full answer. They are not an exhaustive accuracy count; completing a request without an exception does not establish correctness.

| Query | What happened |
|---:|---|
| [0220](qwen_1000_answers.md#q0220) | A bioenergy trend question was routed to model explanation after “change” was matched to the CHANCE model. |
| [0297](qwen_1000_answers.md#q0297) | A PRIMES-Buildings description request returned generic help. |
| [0300](qwen_1000_answers.md#q0300) | Another PRIMES-Buildings profile request truncated the name to “PRIMES-” and said it could not resolve it. |
| [0321](qwen_1000_answers.md#q0321) | Asked for models reporting World CO2, but the answer listed models reporting any emissions variable. |
| [0325](qwen_1000_answers.md#q0325) | Asked for model versions reporting electricity demand in Italy; answer listed broad energy-supply models without the requested scope. |
| [0380](qwen_1000_answers.md#q0380) | Asked how many models have South Africa nuclear-generation data; answer listed general energy-supply models. |
| [0440](qwen_1000_answers.md#q0440) | A capacity-versus-generation interpretation question returned only a Results link. |
| [0450](qwen_1000_answers.md#q0450) | A conceptual example about baseline versus forecast became a data-query clarification. |
| [0470](qwen_1000_answers.md#q0470) | A conceptual example about assumptions versus parameters became an unknown-model error. |
| [0500](qwen_1000_answers.md#q0500) | A climate-neutrality versus net-zero explanation returned a study listing. |
| [0510](qwen_1000_answers.md#q0510) | A currency-year comparison explanation treated “current dollars and constant” as a region. |
| [0600](qwen_1000_answers.md#q0600) | The invented Silver Horizon scenario was accepted into a data response instead of clearly rejected or clarified. |
| [0706](qwen_1000_answers.md#q0706) | Asked why scenario values differ; answer repeated the numeric table. |
| [0710](qwen_1000_answers.md#q0710) | Model pronoun “it” was treated as a literal model name and lost MESSAGEix-GLOBIOM context. |
| [0714](qwen_1000_answers.md#q0714) | Follow-up asking which pathway is lower lost the comparison and listed general scenarios. |
| [0715](qwen_1000_answers.md#q0715) | Region change lost the previous pathway comparison. |
| [0721](qwen_1000_answers.md#q0721) | “Through 2045” lost the original 2030 start year and used first available. |

## Repeated patterns

- Conceptual explanation prompts can be pulled into data or model lookup by terms such as “model”, “capacity”, “price”, or “results”.
- Coverage answers sometimes broaden the requested variable or region and provide a topic-wide model list.
- Some model-family names and hyphenated names are misread or truncated.
- Follow-ups often lose scenario-comparison state, pronoun references, or the beginning of a year interval.
- Chart requests consistently redirect to the data explorer; this is the current product behavior.

## Suggested priorities

1. Resolve intent before entity extraction for explanatory and interpretation prompts.
2. Enforce requested variable, region, scenario, year, and model constraints on coverage answers.
3. Preserve the last valid scope through pronoun references, comparison edits, chart redirects, and range refinements.
4. Use catalogue-first matching for model names and validate unsupported scenario names before a numeric answer.

The audit used local cached data and Ollama. It did not verify external URLs or independently recalculate every numeric value, so this report gives concrete examples rather than an overall correctness rate.
