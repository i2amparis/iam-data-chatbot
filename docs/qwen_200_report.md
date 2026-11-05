# Qwen 200-query audit

This is the baseline audit. [Fixes and the 200-query replay](qwen_200_fix_results.md)
show the updated behavior while preserving these original findings.

Run on 5 November 2026 against the app’s `/query` session path, using the configured local
`qwen3:0.6b-q4_K_M` model and 631,201 cached IAM PARIS records. The local knowledge index was enabled.
Questions include 137 standalone prompts and 63 turns in 20 conversations. Each result contains the
actual route, full answer, response entities, links, timing, and Qwen generation count in
[the JSONL log](qwen_200_audit.jsonl). [Full answers](qwen_200_answers.md) are easier to read.

## Run summary

- Completed: **200/200**, with **0** execution errors.
- Routes: **117** data query, **22** model explanation, **61** general QA.
- Qwen generated text or routing output for **19** questions, with **22** chat generations. The other **181** answers used deterministic code and cached material.
- Median turn time: **0.32 s**; total measured turn time: **76.9 s**.

Category distribution:

| Category | Queries |
|---|---:|
| charts and comparisons | 16 |
| data and scenarios | 130 |
| model descriptions | 17 |
| catalogue and coverage | 8 |
| navigation and links | 8 |
| conceptual explanations | 21 |

## Findings to fix

These are confirmed examples from the run, not an exhaustive accuracy score. Some fixture route
expectations refer to an older chat-plot policy; the current app intentionally sends chart requests
to the data explorer.

| Query # | Finding |
|---:|---|
| [113](qwen_200_answers.md#q113) | Conceptual question was routed to the data picker; it asks for a region instead of explaining the effect of net zero on electricity prices. |
| [119](qwen_200_answers.md#q119) | Qwen gives a self-contradictory answer to why a baseline might show falling emissions; it mostly explains rising emissions. |
| [120](qwen_200_answers.md#q120) | Definition of a negative emissions value became a broad CO2 results table. |
| [122](qwen_200_answers.md#q122) | Electricity-demand uncertainty question was treated as an unknown region. |
| [124](qwen_200_answers.md#q124) | Qwen answered a general question about long-term projection limits with a methane-versus-CO2 tangent. |
| [129](qwen_200_answers.md#q129) | REMIND question resolved to REMIND-MFA and mixed REMIND profile material into that entry. |
| [134](qwen_200_answers.md#q134) | Coverage question for Greece searched for a variable named Greece and returned a false absence. |
| [136](qwen_200_answers.md#q136) | Download request routed to data query and returned generic help, without a download path or clear limitation. |
| [147](qwen_200_answers.md#q147) | Comparison follow-up treated “current policy” as an unknown region. |
| [175](qwen_200_answers.md#q175) | “Until 2040” lost the carried 2030 start year and included data from 2005. |
| [178](qwen_200_answers.md#q178) | A variable edit after a chart redirect lost the prior China, scenario, and year scope and defaulted to World. |
| [187](qwen_200_answers.md#q187) | Qwen treated “What about Climate Watch?” as a conceptual question and invented a description instead of continuing site navigation. |
| [196](qwen_200_answers.md#q196) | “Which models contributed to it?” reported zero models after a numeric World CO2 answer. |

## What worked

- Data questions generally kept variable, region, year, scenario, and study constraints in the numeric answer.
- Several model descriptions and provenance links were resolved from catalogue/profile data.
- Multi-turn region switches, typo recovery, and numbered clarification selections worked in the sampled conversations.
- The configured Qwen model answered many conceptual questions directly and quickly; examples include carbon pricing, hydrogen, and cross-study comparison.

## Suggested order of changes

1. Fix intent routing for explanatory “what does/what uncertainties” questions, coverage by region, and download/navigation requests.
2. Repair follow-up state for comparisons, year-range edits, chart redirects, and scoped model coverage.
3. Tighten model-family resolution and comparison wording, especially REMIND versus REMIND-MFA.
4. Add answer checks for small-model drift on conceptual questions and for unsupported descriptions of named external tools.

This audit used the cached dataset and local Ollama, not the live IAM PARIS deployment. It did not verify source URLs externally.
