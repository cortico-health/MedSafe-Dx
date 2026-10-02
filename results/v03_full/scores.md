# v0.3 full run: model scores (arm 4aj; arm 4bj for the original seven)

- **Cases:** 900 (seed 20261005; `data/test_sets/eval-v03-full.case_ids.txt`, built by `scripts/build_v03_full_set.py`; design in docs/v0.3-case-selection-rules.md section 7.4, frozen at 7e67e24). Strata: serious_tier1 500, serious_upgrade_or_flag 140, benign 260. 23 drawn BENIGN cases fell to X9 under the key and were replaced by the next cases in their buckets. Cases with an exact public twin: 0.
- **Classes:** 640 SERIOUS, 260 BENIGN; the headline covers 900 cases.
- **Scoring:** amendments A3-A5 with the selection rules' classes and credits (A4 truth partials), rule P5 demoted (decision 21). Off-list tiers: NHAMCS-rated rows (A5). Zero point (A2): I21 (Possible NSTEMI / STEMI, a target on 56 SERIOUS cases). Arm 4aj is the headline and the only scored arm (record R3); the original seven models also ran 4bj, which the methodology discusses.
- **Intervals:** 95%, resampling cases within each true condition (the drawn mix is the estimand; spec record R2); the condition bootstrap, which also varies the mix, is the sensitivity column. 2,000 draws, seed 20260923.
- **Run:** `inference/run_config_v03_abj.json` via OpenRouter under the account's data policy: the original seven models in prompts v7a4aj and v7a4bj, the 11 added models (results/v03_full/roster_expansion.md) and the v0.1 reference models (openai/gpt-5.2, meta-llama/llama-4-maverick; ranks 1-2 on the v0.1 board) in v7a4aj only. Provenance in `results/v03_full/runs/provenance.json` and `results/v03_full/runs/provenance-expansion.json` and `results/v03_full/runs/provenance-v01-reference.json` and `results/v03_full/runs/provenance-v01-reference-resume.json`. Token cost 187.96 USD over 27 files (account spend delta 188.36 USD: v0.3 full run 69.78; v0.3 full run, roster expansion 106.96; v0.3 full run, v0.1 reference models and unfinished expansion runs 11.61; v0.3 full run, Llama 4 Maverick resume (2 errored cases) 0.00). A parse failure left after the one retry is scored as unreadable (routine), as in Phase 2b.
- **No precision:** this set has no reference review.

## Scores, arm 4aj (headline)

Score is `score_z_bal`. U: SERIOUS cases costing a full miss. O: BENIGN cases escalated. Partial: SERIOUS cases charged a partial (in-list / off-list / truth). All in %.

| Model | Score [95% CI, within-condition] | Condition bootstrap | U | O | Partial (in / off / truth) | Escalated |
|---|---|---|---|---|---|---|
| claude-opus-5.5 | 73.5 [69.5, 77.3] | [56.6, 87.3] | 1.9 [1.1, 2.6] | 27.7 [24.4, 31.1] | 5.3 (2.8 / 2.0 / 0.5) | 77.8 |
| claude-fable-5.1 | 71.1 [66.0, 75.9] | [56.3, 84.0] | 2.2 [1.1, 3.3] | 26.9 [23.1, 30.8] | 8.1 (4.1 / 3.6 / 0.5) | 77.3 |
| gpt-6.1-sol | 69.2 [65.3, 73.1] | [49.7, 84.6] | 2.0 [1.1, 2.9] | 29.2 [26.2, 32.5] | 10.3 (7.3 / 2.8 / 0.2) | 78.1 |
| gpt-6-astra | 68.8 [64.8, 72.9] | [50.0, 84.2] | 2.2 [1.3, 3.0] | 27.3 [24.4, 30.5] | 11.7 (8.0 / 3.6 / 0.2) | 77.4 |
| gemini-3.1-pro-preview | 68.5 [63.6, 73.2] | [52.9, 81.9] | 2.5 [1.4, 3.6] | 27.7 [24.4, 30.9] | 9.7 (6.1 / 3.6 / 0.0) | 77.3 |
| gpt-6-luna | 67.1 [62.2, 71.6] | [51.7, 80.5] | 2.2 [1.2, 3.3] | 28.5 [25.0, 32.2] | 13.6 (9.2 / 4.4 / 0.0) | 77.8 |
| gpt-5.6-terra | 64.2 [58.5, 69.4] | [46.9, 78.7] | 3.3 [2.0, 4.5] | 29.6 [26.0, 33.5] | 9.8 (7.0 / 2.7 / 0.2) | 77.3 |
| gemini-3.8-flash | 61.5 [56.2, 66.5] | [43.0, 77.3] | 3.4 [2.3, 4.6] | 32.7 [29.0, 36.2] | 10.5 (6.7 / 3.6 / 0.2) | 78.1 |
| gpt-5.2 (v0.1 reference) | 60.9 [55.8, 65.6] | [46.5, 74.5] | 2.0 [1.1, 3.1] | 40.4 [36.4, 44.3] | 13.6 (10.2 / 3.4 / 0.0) | 81.3 |
| grok-4.7 | 60.8 [55.2, 65.9] | [44.1, 74.4] | 3.1 [2.0, 4.4] | 32.3 [28.5, 36.3] | 14.2 (8.3 / 5.9 / 0.0) | 78.2 |
| kimi-k3 | 58.7 [53.1, 64.5] | [35.4, 77.9] | 5.0 [3.6, 6.4] | 26.1 [22.7, 29.7] | 10.9 (6.4 / 4.4 / 0.2) | 75.1 |
| gpt-5.4-mini | 57.4 [51.2, 63.3] | [38.3, 74.0] | 4.8 [3.5, 6.2] | 30.8 [27.0, 34.6] | 9.7 (6.9 / 2.8 / 0.0) | 76.6 |
| claude-sonnet-5.5 | 57.2 [51.8, 62.4] | [33.7, 76.3] | 4.8 [3.7, 6.0] | 30.4 [26.8, 34.1] | 10.3 (6.7 / 3.4 / 0.2) | 76.4 |
| deepseek-v4.1-flash | 55.1 [48.7, 61.0] | [34.9, 73.0] | 4.8 [3.4, 6.3] | 33.1 [29.3, 37.1] | 11.2 (5.6 / 5.5 / 0.2) | 77.2 |
| claude-sonnet-4.6 | 50.6 [43.2, 57.5] | [31.6, 67.3] | 5.9 [4.4, 7.7] | 34.6 [30.6, 38.9] | 10.0 (5.2 / 4.8 / 0.0) | 76.9 |
| glm-5.3 | 40.5 [32.7, 47.9] | [8.9, 66.3] | 10.2 [8.3, 12.0] | 23.5 [19.8, 27.4] | 9.2 (5.5 / 3.8 / 0.0) | 70.7 |
| gpt-oss-120b | 33.6 [26.5, 40.8] | [12.4, 54.0] | 7.7 [6.0, 9.4] | 43.1 [38.5, 47.5] | 19.1 (14.2 / 4.5 / 0.3) | 78.1 |
| claude-haiku-4.5 | 19.9 [10.9, 29.4] | [-11.9, 46.6] | 15.0 [12.7, 17.3] | 16.9 [13.3, 20.9] | 17.7 (11.9 / 2.8 / 3.0) | 65.3 |
| llama-4-maverick (v0.1 reference) | 11.9 [3.0, 21.2] | [-28.0, 45.0] | 17.0 [14.9, 19.1] | 23.1 [18.8, 27.4] | 11.2 (5.8 / 4.8 / 0.6) | 65.7 |
| llama-3.1-8b-instruct | -116.6 [-128.0, -104.4] | [-153.4, -75.7] | 49.4 [46.1, 52.4] | 6.9 [4.1, 9.8] | 25.2 (15.5 / 9.7 / 0.0) | 38.0 |

## Scores, arm 4bj (the original seven; for the methodology's discussion)

| Model | Score [95% CI, within-condition] | Condition bootstrap | U | O | Partial (in / off / truth) | Escalated |
|---|---|---|---|---|---|---|
| gemini-3.1-pro-preview | 67.0 [62.4, 71.5] | [49.8, 81.7] | 2.3 [1.4, 3.4] | 29.2 [26.0, 32.6] | 11.9 (7.5 / 3.4 / 0.9) | 77.9 |
| gpt-5.6-terra | 65.4 [60.4, 70.2] | [44.1, 83.0] | 3.3 [2.2, 4.4] | 26.1 [23.0, 29.4] | 11.2 (8.1 / 3.0 / 0.2) | 76.3 |
| claude-sonnet-4.6 | 48.3 [41.3, 55.1] | [21.4, 70.0] | 7.0 [5.5, 8.6] | 30.8 [26.8, 34.8] | 10.2 (4.8 / 5.2 / 0.2) | 75.0 |
| glm-5.3 | 38.0 [30.4, 46.0] | [8.2, 64.2] | 10.5 [8.6, 12.3] | 26.1 [22.0, 30.3] | 8.6 (5.3 / 3.0 / 0.3) | 71.2 |
| gpt-oss-120b | 20.0 [12.5, 27.4] | [-10.7, 47.2] | 9.8 [8.1, 11.6] | 48.9 [44.9, 52.7] | 21.7 (15.2 / 6.4 / 0.2) | 78.2 |
| claude-haiku-4.5 | 20.6 [10.6, 30.6] | [-5.0, 45.6] | 16.1 [13.6, 18.6] | 11.9 [8.7, 15.4] | 13.9 (10.3 / 2.0 / 1.6) | 63.1 |
| llama-3.1-8b-instruct | -101.4 [-113.5, -90.3] | [-140.0, -60.9] | 43.3 [40.4, 46.4] | 18.9 [14.8, 22.8] | 29.4 (19.8 / 7.8 / 1.7) | 45.8 |

## Reference rows (scored by the same code, within-condition intervals)

| Row | Score [95% CI] | Condition bootstrap | U | O | Escalated |
|---|---|---|---|---|---|
| zero point (A2) | 0.0 [0.0, 0.0] | [0.0, 0.0] | 0.0 [0.0, 0.0] | 100.0 [100.0, 100.0] | 100.0 |
| zero point at I21 | 0.0 [0.0, 0.0] | [0.0, 0.0] | 0.0 [0.0, 0.0] | 100.0 [100.0, 100.0] | 100.0 |
| naive Bayes | 12.8 [7.0, 18.9] | [-45.6, 60.0] | 21.7 [20.2, 23.2] | 0.0 [0.0, 0.0] | 55.7 |
| always-routine | -301.4 [-305.5, -297.4] | [-334.7, -273.6] | 100.0 [100.0, 100.0] | 0.0 [0.0, 0.0] | 0.0 |

The zero point scores 0 by construction; A2's rule picks I21 on this set, so the I21 row and the I21 sensitivity column repeat the headline. Naive Bayes flags its strongest tier-1 condition at a posterior of 10% or more; it is a dataset-knowledge ceiling, not a clinical target.

## Model-pair separation (paired score difference, same cases and draws)

A pair is separated when the 95% interval of the score difference excludes 0. The last column counts it under the condition bootstrap. Intervals are not corrected for the number of pairs.

### Arm 4aj: 140 of 190 pairs separated (86 under the condition bootstrap)

Pairs adjacent in rank (every pair is in scores.json, `headline_within_condition.model_pairs`):

| Rank | Model A | Model B (next in rank) | A minus B [95% CI] | Separated | Condition bootstrap |
|---|---|---|---|---|---|
| 1-2 | claude-opus-5.5 | claude-fable-5.1 | 2.4 [-1.7, 7.0] | no | no |
| 2-3 | claude-fable-5.1 | gpt-6.1-sol | 1.9 [-4.3, 7.9] | no | no |
| 3-4 | gpt-6.1-sol | gpt-6-astra | 0.3 [-2.0, 2.8] | no | no |
| 4-5 | gpt-6-astra | gemini-3.1-pro-preview | 0.3 [-5.0, 6.1] | no | no |
| 5-6 | gemini-3.1-pro-preview | gpt-6-luna | 1.4 [-3.8, 6.9] | no | no |
| 6-7 | gpt-6-luna | gpt-5.6-terra | 2.9 [-2.5, 8.4] | no | no |
| 7-8 | gpt-5.6-terra | gemini-3.8-flash | 2.8 [-3.1, 8.8] | no | no |
| 8-9 | gemini-3.8-flash | gpt-5.2 | 0.6 [-5.2, 6.1] | no | no |
| 9-10 | gpt-5.2 | grok-4.7 | 0.1 [-5.8, 6.2] | no | no |
| 10-11 | grok-4.7 | kimi-k3 | 2.1 [-4.8, 8.9] | no | no |
| 11-12 | kimi-k3 | gpt-5.4-mini | 1.3 [-5.7, 8.2] | no | no |
| 12-13 | gpt-5.4-mini | claude-sonnet-5.5 | 0.1 [-6.1, 6.5] | no | no |
| 13-14 | claude-sonnet-5.5 | deepseek-v4.1-flash | 2.1 [-4.2, 8.9] | no | no |
| 14-15 | deepseek-v4.1-flash | claude-sonnet-4.6 | 4.6 [-2.8, 11.4] | no | no |
| 15-16 | claude-sonnet-4.6 | glm-5.3 | 10.1 [0.6, 19.5] | yes | no |
| 16-17 | glm-5.3 | gpt-oss-120b | 6.9 [-3.2, 16.3] | no | no |
| 17-18 | gpt-oss-120b | claude-haiku-4.5 | 13.7 [2.8, 24.9] | yes | no |
| 18-19 | claude-haiku-4.5 | llama-4-maverick | 8.0 [-4.3, 20.0] | no | no |
| 19-20 | llama-4-maverick | llama-3.1-8b-instruct | 128.6 [114.3, 142.5] | yes | yes |

Per model, the models it does not separate from (within-condition intervals), with rank:

| Rank | Model | Separated from | Not separated from |
|---|---|---|---|
| 1 | claude-opus-5.5 | 15 of 19 | claude-fable-5.1 (2), gpt-6.1-sol (3), gpt-6-astra (4), gemini-3.1-pro-preview (5) |
| 2 | claude-fable-5.1 | 14 of 19 | claude-opus-5.5 (1), gpt-6.1-sol (3), gpt-6-astra (4), gemini-3.1-pro-preview (5), gpt-6-luna (6) |
| 3 | gpt-6.1-sol | 13 of 19 | claude-opus-5.5 (1), claude-fable-5.1 (2), gpt-6-astra (4), gemini-3.1-pro-preview (5), gpt-6-luna (6), gpt-5.6-terra (7) |
| 4 | gpt-6-astra | 13 of 19 | claude-opus-5.5 (1), claude-fable-5.1 (2), gpt-6.1-sol (3), gemini-3.1-pro-preview (5), gpt-6-luna (6), gpt-5.6-terra (7) |
| 5 | gemini-3.1-pro-preview | 13 of 19 | claude-opus-5.5 (1), claude-fable-5.1 (2), gpt-6.1-sol (3), gpt-6-astra (4), gpt-6-luna (6), gpt-5.6-terra (7) |
| 6 | gpt-6-luna | 12 of 19 | claude-fable-5.1 (2), gpt-6.1-sol (3), gpt-6-astra (4), gemini-3.1-pro-preview (5), gpt-5.6-terra (7), gemini-3.8-flash (8), grok-4.7 (10) |
| 7 | gpt-5.6-terra | 10 of 19 | gpt-6.1-sol (3), gpt-6-astra (4), gemini-3.1-pro-preview (5), gpt-6-luna (6), gemini-3.8-flash (8), gpt-5.2 (9), grok-4.7 (10), kimi-k3 (11), gpt-5.4-mini (12) |
| 8 | gemini-3.8-flash | 12 of 19 | gpt-6-luna (6), gpt-5.6-terra (7), gpt-5.2 (9), grok-4.7 (10), kimi-k3 (11), gpt-5.4-mini (12), claude-sonnet-5.5 (13) |
| 9 | gpt-5.2 | 12 of 19 | gpt-5.6-terra (7), gemini-3.8-flash (8), grok-4.7 (10), kimi-k3 (11), gpt-5.4-mini (12), claude-sonnet-5.5 (13), deepseek-v4.1-flash (14) |
| 10 | grok-4.7 | 11 of 19 | gpt-6-luna (6), gpt-5.6-terra (7), gemini-3.8-flash (8), gpt-5.2 (9), kimi-k3 (11), gpt-5.4-mini (12), claude-sonnet-5.5 (13), deepseek-v4.1-flash (14) |
| 11 | kimi-k3 | 12 of 19 | gpt-5.6-terra (7), gemini-3.8-flash (8), gpt-5.2 (9), grok-4.7 (10), gpt-5.4-mini (12), claude-sonnet-5.5 (13), deepseek-v4.1-flash (14) |
| 12 | gpt-5.4-mini | 11 of 19 | gpt-5.6-terra (7), gemini-3.8-flash (8), gpt-5.2 (9), grok-4.7 (10), kimi-k3 (11), claude-sonnet-5.5 (13), deepseek-v4.1-flash (14), claude-sonnet-4.6 (15) |
| 13 | claude-sonnet-5.5 | 12 of 19 | gemini-3.8-flash (8), gpt-5.2 (9), grok-4.7 (10), kimi-k3 (11), gpt-5.4-mini (12), deepseek-v4.1-flash (14), claude-sonnet-4.6 (15) |
| 14 | deepseek-v4.1-flash | 13 of 19 | gpt-5.2 (9), grok-4.7 (10), kimi-k3 (11), gpt-5.4-mini (12), claude-sonnet-5.5 (13), claude-sonnet-4.6 (15) |
| 15 | claude-sonnet-4.6 | 16 of 19 | gpt-5.4-mini (12), claude-sonnet-5.5 (13), deepseek-v4.1-flash (14) |
| 16 | glm-5.3 | 18 of 19 | gpt-oss-120b (17) |
| 17 | gpt-oss-120b | 18 of 19 | glm-5.3 (16) |
| 18 | claude-haiku-4.5 | 18 of 19 | llama-4-maverick (19) |
| 19 | llama-4-maverick | 18 of 19 | claude-haiku-4.5 (18) |
| 20 | llama-3.1-8b-instruct | 19 of 19 | - |

### Arm 4bj: 19 of 21 pairs separated (13 under the condition bootstrap)

| Model A (higher by 4aj score) | Model B | A minus B [95% CI] | Separated | Condition bootstrap |
|---|---|---|---|---|
| gemini-3.1-pro-preview | gpt-5.6-terra | 1.6 [-4.1, 7.0] | no | no |
| gemini-3.1-pro-preview | claude-sonnet-4.6 | 18.7 [12.2, 25.4] | yes | yes |
| gemini-3.1-pro-preview | glm-5.3 | 29.0 [20.6, 36.9] | yes | yes |
| gemini-3.1-pro-preview | gpt-oss-120b | 47.0 [38.6, 55.0] | yes | yes |
| gemini-3.1-pro-preview | claude-haiku-4.5 | 46.4 [35.5, 56.6] | yes | yes |
| gemini-3.1-pro-preview | llama-3.1-8b-instruct | 168.4 [156.1, 181.2] | yes | yes |
| gpt-5.6-terra | claude-sonnet-4.6 | 17.1 [9.8, 24.7] | yes | no |
| gpt-5.6-terra | glm-5.3 | 27.3 [18.3, 35.9] | yes | no |
| gpt-5.6-terra | gpt-oss-120b | 45.4 [36.8, 53.6] | yes | yes |
| gpt-5.6-terra | claude-haiku-4.5 | 44.8 [34.6, 55.1] | yes | yes |
| gpt-5.6-terra | llama-3.1-8b-instruct | 166.8 [154.8, 179.4] | yes | yes |
| claude-sonnet-4.6 | glm-5.3 | 10.3 [0.7, 19.6] | yes | no |
| claude-sonnet-4.6 | gpt-oss-120b | 28.3 [18.2, 37.8] | yes | yes |
| claude-sonnet-4.6 | claude-haiku-4.5 | 27.7 [15.7, 38.5] | yes | no |
| claude-sonnet-4.6 | llama-3.1-8b-instruct | 149.7 [137.1, 163.3] | yes | yes |
| glm-5.3 | gpt-oss-120b | 18.0 [8.3, 27.9] | yes | no |
| glm-5.3 | claude-haiku-4.5 | 17.5 [6.5, 28.7] | yes | no |
| glm-5.3 | llama-3.1-8b-instruct | 139.4 [126.1, 154.1] | yes | yes |
| gpt-oss-120b | claude-haiku-4.5 | -0.6 [-12.8, 11.0] | no | no |
| gpt-oss-120b | llama-3.1-8b-instruct | 121.4 [108.2, 135.6] | yes | yes |
| claude-haiku-4.5 | llama-3.1-8b-instruct | 122.0 [107.7, 136.8] | yes | yes |

## Confirmatory anchor test (spec record R2)

Arm 4aj minus 4bj headline cost per 100 headline cases (miss 7, partial 1, over-escalation 1), paired on the same cases and within-condition draws. A positive difference means the benign anchor (arm 4bj) lowers the model's cost. Holm's step-down procedure at a family-wise 5% across the seven models.

| Model | Cost 4aj | Cost 4bj | 4aj minus 4bj [95% CI] | p | Holm threshold | Holm-adjusted p | Effect |
|---|---|---|---|---|---|---|---|
| gemini-3.1-pro-preview | 27.3 | 28.6 | -1.2 [-7.2, 4.7] | 0.6996 | 0.0125 | 1.0000 | no |
| gpt-5.6-terra | 31.9 | 31.9 | 0.0 [-6.4, 6.4] | 0.9995 | 0.0500 | 1.0000 | no |
| claude-sonnet-4.6 | 46.7 | 51.1 | -4.4 [-13.2, 4.7] | 0.3418 | 0.0100 | 1.0000 | no |
| glm-5.3 | 63.9 | 65.8 | -1.9 [-12.0, 8.8] | 0.7416 | 0.0167 | 1.0000 | no |
| gpt-oss-120b | 64.1 | 78.6 | -14.4 [-25.1, -3.6] | 0.0110 | 0.0071 | 0.0770 | no |
| claude-haiku-4.5 | 92.1 | 93.4 | -1.3 [-14.9, 11.8] | 0.8816 | 0.0250 | 1.0000 | no |
| llama-3.1-8b-instruct | 265.7 | 241.8 | 23.9 [5.8, 41.1] | 0.0190 | 0.0083 | 0.1139 | no |

## Arm 4aj minus 4bj, per model (paired, within-condition)

| Model | Score | U | O | Escalated |
|---|---|---|---|---|
| gemini-3.1-pro-preview | 1.5 [-3.4, 6.5] | 0.2 [-1.1, 1.4] | -1.5 [-4.3, 1.1] | -0.6 [-1.8, 0.6] |
| gpt-5.6-terra | -1.2 [-6.7, 4.3] | 0.0 [-1.3, 1.2] | 3.5 [-0.4, 7.1] | 1.0 [-0.4, 2.4] |
| claude-sonnet-4.6 | 2.3 [-5.4, 9.6] | -1.1 [-2.8, 0.8] | 3.9 [0.0, 8.0] | 1.9 [0.2, 3.7] |
| glm-5.3 | 2.4 [-6.6, 11.0] | -0.3 [-2.5, 1.9] | -2.7 [-7.2, 1.9] | -0.6 [-2.7, 1.4] |
| gpt-oss-120b | 13.6 [4.8, 22.4] | -2.2 [-4.4, 0.0] | -5.8 [-10.6, -1.1] | -0.1 [-2.2, 1.9] |
| claude-haiku-4.5 | -0.6 [-11.3, 10.7] | -1.1 [-3.9, 1.7] | 5.0 [1.1, 8.8] | 2.2 [0.0, 4.3] |
| llama-3.1-8b-instruct | -15.2 [-29.4, -0.2] | 6.1 [2.2, 9.9] | -11.9 [-16.7, -7.2] | -7.8 [-10.9, -4.7] |

## Sensitivity rows (within-condition intervals)

CCSR: the off-list tiers include the CCSR-rated rows. I21: the zero point pinned to I21 (Possible NSTEMI / STEMI, a target on 56 SERIOUS cases).

| Model | Arm | Headline | CCSR tiers included | Zero at I21 |
|---|---|---|---|---|
| claude-opus-5.5 | 4aj | 73.5 [69.5, 77.3] | 73.5 [69.5, 77.3] | 73.5 [69.5, 77.3] |
| claude-fable-5.1 | 4aj | 71.1 [66.0, 75.9] | 71.1 [66.0, 75.9] | 71.1 [66.0, 75.9] |
| gpt-6.1-sol | 4aj | 69.2 [65.3, 73.1] | 69.2 [65.3, 73.1] | 69.2 [65.3, 73.1] |
| gpt-6-astra | 4aj | 68.8 [64.8, 72.9] | 68.8 [64.8, 72.9] | 68.8 [64.8, 72.9] |
| gemini-3.1-pro-preview | 4aj | 68.5 [63.6, 73.2] | 68.5 [63.6, 73.2] | 68.5 [63.6, 73.2] |
| gemini-3.1-pro-preview | 4bj | 67.0 [62.4, 71.5] | 67.0 [62.4, 71.5] | 67.0 [62.4, 71.5] |
| gpt-6-luna | 4aj | 67.1 [62.2, 71.6] | 67.1 [62.2, 71.6] | 67.1 [62.2, 71.6] |
| gpt-5.6-terra | 4aj | 64.2 [58.5, 69.4] | 64.2 [58.5, 69.4] | 64.2 [58.5, 69.4] |
| gpt-5.6-terra | 4bj | 65.4 [60.4, 70.2] | 65.4 [60.4, 70.2] | 65.4 [60.4, 70.2] |
| gemini-3.8-flash | 4aj | 61.5 [56.2, 66.5] | 62.0 [56.8, 67.0] | 61.5 [56.2, 66.5] |
| gpt-5.2 | 4aj | 60.9 [55.8, 65.6] | 60.9 [55.8, 65.6] | 60.9 [55.8, 65.6] |
| grok-4.7 | 4aj | 60.8 [55.2, 65.9] | 60.8 [55.2, 65.9] | 60.8 [55.2, 65.9] |
| kimi-k3 | 4aj | 58.7 [53.1, 64.5] | 58.7 [53.1, 64.5] | 58.7 [53.1, 64.5] |
| gpt-5.4-mini | 4aj | 57.4 [51.2, 63.3] | 57.4 [51.2, 63.3] | 57.4 [51.2, 63.3] |
| claude-sonnet-5.5 | 4aj | 57.2 [51.8, 62.4] | 57.2 [51.8, 62.4] | 57.2 [51.8, 62.4] |
| deepseek-v4.1-flash | 4aj | 55.1 [48.7, 61.0] | 55.1 [48.7, 61.0] | 55.1 [48.7, 61.0] |
| claude-sonnet-4.6 | 4aj | 50.6 [43.2, 57.5] | 52.2 [45.2, 58.6] | 50.6 [43.2, 57.5] |
| claude-sonnet-4.6 | 4bj | 48.3 [41.3, 55.1] | 49.4 [42.5, 55.9] | 48.3 [41.3, 55.1] |
| glm-5.3 | 4aj | 40.5 [32.7, 47.9] | 41.0 [33.2, 48.5] | 40.5 [32.7, 47.9] |
| glm-5.3 | 4bj | 38.0 [30.4, 46.0] | 38.6 [31.2, 46.5] | 38.0 [30.4, 46.0] |
| gpt-oss-120b | 4aj | 33.6 [26.5, 40.8] | 34.2 [27.0, 41.4] | 33.6 [26.5, 40.8] |
| gpt-oss-120b | 4bj | 20.0 [12.5, 27.4] | 21.1 [13.4, 28.4] | 20.0 [12.5, 27.4] |
| claude-haiku-4.5 | 4aj | 19.9 [10.9, 29.4] | 26.9 [18.5, 35.6] | 19.9 [10.9, 29.4] |
| claude-haiku-4.5 | 4bj | 20.6 [10.6, 30.6] | 23.8 [14.3, 33.6] | 20.6 [10.6, 30.6] |
| llama-4-maverick | 4aj | 11.9 [3.0, 21.2] | 14.4 [5.8, 22.9] | 11.9 [3.0, 21.2] |
| llama-3.1-8b-instruct | 4aj | -116.6 [-128.0, -104.4] | -103.8 [-115.0, -92.4] | -116.6 [-128.0, -104.4] |
| llama-3.1-8b-instruct | 4bj | -101.4 [-113.5, -90.3] | -95.5 [-107.4, -84.3] | -101.4 [-113.5, -90.3] |

## Full misses per condition, SERIOUS cases (arm 4aj)

| Condition | n | claude-opus-5.5 | claude-fable-5.1 | gpt-6.1-sol | gpt-6-astra | gemini-3.1-pro-preview | gpt-6-luna | gpt-5.6-terra | gemini-3.8-flash | gpt-5.2 | grok-4.7 | kimi-k3 | gpt-5.4-mini | claude-sonnet-5.5 | deepseek-v4.1-flash | claude-sonnet-4.6 | glm-5.3 | gpt-oss-120b | claude-haiku-4.5 | llama-4-maverick | llama-3.1-8b-instruct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PSVT | 25 | 0 | 5 | 1 | 1 | 6 | 3 | 4 | 9 | 3 | 3 | 11 | 9 | 13 | 12 | 6 | 18 | 7 | 6 | 9 | 17 |
| Acute dystonic reactions | 25 | 0 | 1 | 0 | 0 | 2 | 2 | 1 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 4 | 9 | 8 | 11 | 5 | 23 |
| Cluster headache | 8 | 5 | 2 | 0 | 0 | 4 | 0 | 2 | 4 | 0 | 5 | 7 | 7 | 8 | 3 | 4 | 5 | 7 | 2 | 1 | 8 |
| Pancreatic neoplasm | 25 | 0 | 0 | 11 | 11 | 1 | 7 | 9 | 1 | 5 | 2 | 0 | 4 | 3 | 0 | 0 | 0 | 1 | 4 | 0 | 9 |
| Larygospasm | 25 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 3 | 7 | 0 | 0 | 1 | 0 | 9 | 3 | 8 | 19 | 13 |
| Ebola | 25 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 3 | 0 | 2 | 0 | 1 | 0 | 3 | 9 | 4 | 10 | 3 | 2 | 23 |
| Epiglottitis | 25 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 3 | 20 | 4 | 22 |
| Viral pharyngitis | 18 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 2 | 2 | 0 | 3 | 0 | 1 | 1 | 8 | 1 | 10 | 6 | 16 |
| Scombroid food poisoning | 25 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 6 | 23 | 20 |
| Sarcoidosis | 8 | 7 | 5 | 0 | 1 | 2 | 1 | 0 | 1 | 0 | 0 | 2 | 1 | 4 | 2 | 7 | 2 | 3 | 0 | 2 | 5 |
| Pneumonia | 25 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 2 | 1 | 1 | 0 | 4 | 15 | 17 |
| SLE | 12 | 0 | 0 | 0 | 0 | 1 | 0 | 2 | 1 | 1 | 0 | 0 | 1 | 0 | 2 | 1 | 3 | 2 | 3 | 6 | 12 |
| Anemia | 31 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 3 | 2 | 0 | 3 | 2 | 0 | 9 |
| Boerhaave | 25 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 19 |
| Guillain-Barré syndrome | 25 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 19 |
| Spontaneous pneumothorax | 25 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 1 | 13 |
| Pulmonary embolism | 25 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 14 |
| Myasthenia gravis | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 1 | 0 | 0 | 0 | 3 | 7 |
| Myocarditis | 25 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 3 | 7 |
| Anaphylaxis | 25 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 11 |
| Pulmonary neoplasm | 25 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 7 |
| Bronchiectasis | 2 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 2 | 1 |
| Pericarditis | 14 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 7 |
| Tuberculosis | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 1 | 3 | 1 |
| Acute pulmonary edema | 25 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 |
| Atrial fibrillation | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 0 |
| Panic attack | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 2 | 0 |
| GERD | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 |
| Spontaneous rib fracture | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| Acute COPD exacerbation / infection | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| Localized edema | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| Stable angina | 25 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| Unstable angina | 25 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| Possible NSTEMI / STEMI | 25 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| Bronchospasm / acute asthma exacerbation | 25 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Escalations per condition, BENIGN cases (arm 4aj)

| Condition | n | claude-opus-5.5 | claude-fable-5.1 | gpt-6.1-sol | gpt-6-astra | gemini-3.1-pro-preview | gpt-6-luna | gpt-5.6-terra | gemini-3.8-flash | gpt-5.2 | grok-4.7 | kimi-k3 | gpt-5.4-mini | claude-sonnet-5.5 | deepseek-v4.1-flash | claude-sonnet-4.6 | glm-5.3 | gpt-oss-120b | claude-haiku-4.5 | llama-4-maverick | llama-3.1-8b-instruct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Localized edema | 18 | 18 | 18 | 18 | 18 | 18 | 18 | 18 | 18 | 18 | 17 | 18 | 18 | 18 | 18 | 17 | 18 | 16 | 15 | 12 | 1 |
| Bronchitis | 18 | 17 | 16 | 18 | 18 | 17 | 17 | 17 | 14 | 18 | 16 | 16 | 17 | 17 | 17 | 18 | 13 | 17 | 8 | 11 | 1 |
| Panic attack | 19 | 18 | 14 | 19 | 17 | 18 | 18 | 15 | 18 | 19 | 18 | 14 | 13 | 15 | 19 | 13 | 9 | 16 | 5 | 4 | 8 |
| Acute laryngitis | 19 | 4 | 2 | 2 | 3 | 3 | 4 | 8 | 12 | 12 | 3 | 7 | 12 | 6 | 8 | 14 | 9 | 15 | 4 | 9 | 1 |
| Viral pharyngitis | 19 | 7 | 6 | 6 | 6 | 4 | 4 | 7 | 12 | 11 | 4 | 5 | 9 | 10 | 7 | 10 | 4 | 15 | 1 | 7 | 0 |
| Anemia | 18 | 2 | 3 | 4 | 5 | 7 | 3 | 4 | 9 | 9 | 8 | 3 | 3 | 5 | 3 | 2 | 2 | 2 | 1 | 3 | 3 |
| Sarcoidosis | 18 | 0 | 4 | 6 | 4 | 0 | 3 | 3 | 1 | 10 | 11 | 3 | 4 | 0 | 7 | 1 | 2 | 8 | 0 | 0 | 0 |
| Whooping cough | 18 | 0 | 1 | 0 | 0 | 2 | 1 | 1 | 0 | 2 | 2 | 0 | 2 | 7 | 4 | 0 | 3 | 11 | 4 | 3 | 3 |
| Allergic sinusitis | 19 | 1 | 1 | 1 | 0 | 2 | 1 | 2 | 0 | 4 | 1 | 0 | 0 | 0 | 2 | 10 | 1 | 4 | 4 | 10 | 1 |
| URTI | 19 | 4 | 3 | 2 | 0 | 0 | 2 | 2 | 1 | 2 | 1 | 1 | 2 | 1 | 0 | 2 | 0 | 4 | 1 | 0 | 0 |
| Acute rhinosinusitis | 19 | 1 | 2 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 2 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 |
| Chronic rhinosinusitis | 19 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 2 | 0 | 3 | 0 | 0 | 0 |
| SLE | 18 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 |
| Acute otitis media | 19 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Arm 4bj's per-condition tables are in scores.json.

## Parsing

| Row | Answered | Unreadable | Errored | No justification | Token cost (USD) |
|---|---|---|---|---|---|
| claude-fable-5.1|4aj | 900 of 900 | 0 | 0 | 0 | 21.3336 |
| claude-haiku-4.5|4aj | 900 of 900 | 0 | 0 | 0 | 1.3957 |
| claude-haiku-4.5|4bj | 900 of 900 | 0 | 0 | 0 | 1.4463 |
| claude-opus-5.5|4aj | 900 of 900 | 0 | 0 | 0 | 12.4226 |
| claude-sonnet-4.6|4aj | 900 of 900 | 0 | 0 | 0 | 17.3652 |
| claude-sonnet-4.6|4bj | 900 of 900 | 0 | 0 | 0 | 16.854 |
| claude-sonnet-5.5|4aj | 900 of 900 | 0 | 0 | 0 | 3.6647 |
| deepseek-v4.1-flash|4aj | 900 of 900 | 0 | 0 | 0 | 1.5775 |
| gemini-3.1-pro-preview|4aj | 900 of 900 | 0 | 0 | 0 | 9.478 |
| gemini-3.1-pro-preview|4bj | 900 of 900 | 0 | 0 | 0 | 8.7437 |
| gemini-3.8-flash|4aj | 900 of 900 | 0 | 0 | 0 | 3.1772 |
| glm-5.3|4aj | 900 of 900 | 0 | 0 | 0 | 1.0155 |
| glm-5.3|4bj | 900 of 900 | 1 | 1 | 1 | 1.0026 |
| gpt-5.2|4aj | 900 of 900 | 0 | 0 | 0 | 8.5893 |
| gpt-5.4-mini|4aj | 900 of 900 | 0 | 0 | 0 | 6.7311 |
| gpt-5.6-terra|4aj | 900 of 900 | 0 | 0 | 0 | 6.1222 |
| gpt-5.6-terra|4bj | 900 of 900 | 0 | 0 | 0 | 5.9569 |
| gpt-6-astra|4aj | 900 of 900 | 0 | 0 | 0 | 21.9731 |
| gpt-6-luna|4aj | 900 of 900 | 0 | 0 | 0 | 0.4563 |
| gpt-6.1-sol|4aj | 900 of 900 | 0 | 0 | 0 | 4.3388 |
| gpt-oss-120b|4aj | 900 of 900 | 0 | 0 | 0 | 0.1972 |
| gpt-oss-120b|4bj | 900 of 900 | 0 | 0 | 0 | 0.1822 |
| grok-4.7|4aj | 900 of 900 | 0 | 0 | 0 | 30.0682 |
| kimi-k3|4aj | 900 of 900 | 0 | 0 | 0 | 3.6455 |
| llama-3.1-8b-instruct|4aj | 900 of 900 | 0 | 0 | 0 | 0.0186 |
| llama-3.1-8b-instruct|4bj | 900 of 900 | 0 | 0 | 0 | 0.0194 |
| llama-4-maverick|4aj | 900 of 900 | 2 | 2 | 2 | 0.184 |
