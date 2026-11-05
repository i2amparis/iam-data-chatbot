# Qwen audit fixes: verification

The original [200-query audit](qwen_200_report.md) remains the baseline. The
[after-fixes JSONL](qwen_200_after_fixes.jsonl) contains the same 200 questions,
routes, answers, links and Qwen call counts after the changes. Query number equals
the JSONL line number.

## Results

- Completed **200/200** queries with **0** execution errors using local
  `qwen3:0.6b-q4_K_M` and the same cached IAM PARIS dataset.
- Route counts changed from 117 data / 61 general QA / 22 model explanation to
  113 data / 65 general QA / 22 model explanation. Four explanation or
  navigation questions that had been sent to data routing are now classified
  correctly.
- The focused regression suite passed **329 tests**.

| Query | Original failure | Result after fix |
|---:|---|---|
| 113 | Net zero and electricity prices became a region picker. | Qwen explains that net zero does not imply a fixed electricity price and names the relevant scenario assumptions. |
| 119 | Qwen answered why baseline emissions rise when asked why they fall. | A matching explanation grounds Qwen's answer in existing policy effects, efficiency, technology and demand. |
| 120 | Negative-emissions meaning became a broad CO2 table. | The bot explains the meaning of a negative net-emissions value and cautions about sign conventions. |
| 122 | Electricity-demand uncertainty was treated as an unknown region. | Qwen explains the demand assumptions and their uncertainty. |
| 124 | Long-term projection limits became a methane tangent. | Qwen explains conditional scenarios and model/assumption uncertainty. |
| 129 | REMIND was answered as REMIND-MFA. | The curated REMIND profile is used. |
| 134 | Greece was searched as a variable. | Models with loaded results for Greece are listed. |
| 136 | Download question returned generic help. | The bot links to Results and states that no direct file URL is verified. |
| 147 | “Current policy” comparison was mistaken for a region. | Baseline and current-policy series are compared in a numeric table. |
| 175 | “Until 2040” dropped the carried 2030 lower bound. | The answer covers 2030–2040. |
| 178 | A variable edit after a chart redirect lost China, scenario and years. | The answer keeps China, `PR_CurPol_CP`, and 2030–2050. |
| 187 | Climate Watch navigation became an unsupported Qwen description. | The bot returns the catalogued Application Library link and search hint. |
| 196 | A provenance follow-up claimed zero contributing models. | The answer names E3ME, GCAM, GEMINI-E3 and MUSE from the prior result. |

## Remaining limits

The link catalogue has no verified direct file-download URL for the underlying
dataset, so the answer directs users to Results instead of inventing one. The
replay uses local cached data and does not validate external links or fact-check
every answer beyond the reported failures.
