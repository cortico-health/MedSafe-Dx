# Phase 2 model scores (arms 4aj and 4bj, post hoc, rule X11)

Blinded to the reference: these are benchmark scores only. Precision against the adjudicated reference comes from `scripts/analysis/v03_phase2_precision.py` once `results/phase2/reference_adjudicated.jsonl` exists.

- **Cases:** the 250 Phase 2 cases (seed 20261003; `data/test_sets/eval-v03-phase2.case_ids.txt`, built by `scripts/build_v03_phase2_set.py --phase 2`). Strata: serious_tier1 50, serious_upgrade_or_flag 70, serious_dxa_only 20, benign 60, excluded 50. 18 of the 20 drawn DXA-only cases had no DXA-only target left once the key was built (red herrings), so the next cases in the stratum's shuffled pool replaced them (section 7); the models also answered the 18 replaced cases, which are not scored here.
- **Classes** (selection rules, key targets): 120 SERIOUS, 58 BENIGN, 72 EXCLUDED; the headline covers 178 cases.
- **Scoring:** amendments A3-A5 with the selection rules' classes and credits (A4 truth partials). Off-list tiers: NHAMCS-rated rows only (A5). Zero reference (A2): I471 (PSVT, a target on 15 SERIOUS cases).
- **Run:** `inference/run_config_v03_abj.json` (as the 4aj and 4bj runs on the 150), prompt v7a4aj and v7a4bj with the justification line, via OpenRouter; provenance in `results/phase2/runs/provenance.json`. Token cost 21.46 USD over 14 files and every case the models answered (account spend delta 21.27 USD). A json_parse_failure left after the one retry is scored as unreadable (routine), as on the 150 (spec section 9); the runner lists those files as incomplete in the provenance, and we did not re-run them.

## Scores

Score is `score_z_bal` with its 95% interval. U: SERIOUS cases costing a full miss. O: BENIGN cases escalated. Partial: SERIOUS cases charged a partial (in-list / off-list / truth). All in %.

| Model | Arm | Score [95% CI] | Score, sample mix | U | O | Partial (in / off / truth) | Escalated |
|---|---|---|---|---|---|---|---|
| gemini-3.1-pro-preview | 4aj | 81.0 [68.2, 90.8] | 78.4 | 1.7 [0.0, 3.9] | 12.1 [3.0, 28.6] | 11.7 (6.7 / 4.2 / 0.8) | 66.0 |
| gemini-3.1-pro-preview | 4bj | 81.9 [71.0, 90.4] | 79.6 | 0.8 [0.0, 2.3] | 12.1 [2.2, 30.4] | 15.8 (8.3 / 4.2 / 3.3) | 66.4 |
| gpt-5.6-terra | 4aj | 81.0 [67.5, 90.2] | 79.6 | 1.7 [0.0, 3.8] | 15.5 [5.0, 34.3] | 8.3 (6.7 / 0.8 / 0.8) | 69.6 |
| gpt-5.6-terra | 4bj | 76.5 [62.6, 87.9] | 74.7 | 2.5 [0.0, 4.8] | 19.0 [8.5, 34.1] | 7.5 (5.0 / 1.7 / 0.8) | 68.8 |
| claude-sonnet-4.6 | 4aj | 46.3 [21.0, 72.2] | 38.9 | 7.5 [2.6, 12.1] | 34.5 [10.8, 65.7] | 13.3 (5.0 / 8.3 / 0.0) | 70.0 |
| claude-sonnet-4.6 | 4bj | 65.2 [42.7, 83.4] | 61.7 | 3.3 [0.0, 7.5] | 25.9 [5.3, 55.8] | 15.8 (5.8 / 9.2 / 0.8) | 68.8 |
| glm-5.3 | 4aj | 46.2 [8.5, 74.9] | 30.9 | 10.8 [4.0, 19.4] | 13.8 [3.3, 29.6] | 10.8 (5.8 / 4.2 / 0.8) | 59.2 |
| glm-5.3 | 4bj | 42.3 [-0.3, 72.5] | 22.8 | 13.3 [5.5, 23.1] | 6.9 [0.0, 21.9] | 7.5 (3.3 / 3.3 / 0.8) | 54.8 |
| gpt-oss-120b | 4aj | 40.0 [7.9, 65.0] | 30.2 | 7.5 [1.8, 14.3] | 34.5 [12.5, 65.5] | 25.0 (15.0 / 9.2 / 0.8) | 72.8 |
| gpt-oss-120b | 4bj | 18.9 [-14.3, 52.6] | 3.7 | 12.5 [4.0, 20.9] | 41.4 [14.9, 74.4] | 22.5 (12.5 / 9.2 / 0.8) | 72.0 |
| claude-haiku-4.5 | 4aj | 21.3 [-13.8, 51.1] | -5.6 | 17.5 [10.2, 26.9] | 8.6 [0.0, 24.0] | 15.8 (5.0 / 2.5 / 8.3) | 50.8 |
| claude-haiku-4.5 | 4bj | 11.4 [-38.7, 59.2] | -17.9 | 20.0 [8.2, 30.9] | 12.1 [2.3, 29.4] | 13.3 (6.7 / 2.5 / 4.2) | 51.2 |
| llama-3.1-8b-instruct | 4aj | -101.9 [-148.2, -56.8] | -176.5 | 49.2 [36.7, 62.6] | 6.9 [0.0, 18.0] | 25.8 (12.5 / 11.7 / 1.7) | 31.6 |
| llama-3.1-8b-instruct | 4bj | -89.5 [-131.4, -51.8] | -158.0 | 44.2 [32.2, 58.0] | 10.3 [2.9, 20.0] | 34.2 (20.8 / 9.2 / 4.2) | 36.8 |

## Arm 4aj minus 4bj, per model (paired)

| Model | Score | U | O | Escalated |
|---|---|---|---|---|
| gemini-3.1-pro-preview | -0.9 [-8.2, 4.9] | 0.8 [0.0, 2.9] | 0.0 [-5.3, 5.0] | -0.4 [-3.1, 2.9] |
| gpt-5.6-terra | 4.5 [-9.1, 17.2] | -0.8 [-3.7, 2.1] | -3.5 [-9.8, 7.7] | 0.8 [-1.9, 3.4] |
| claude-sonnet-4.6 | -18.9 [-31.5, -4.2] | 4.2 [0.9, 7.2] | 8.6 [1.9, 17.3] | 1.2 [-1.8, 3.8] |
| glm-5.3 | 3.9 [-16.6, 26.7] | -2.5 [-8.5, 3.3] | 6.9 [-2.9, 17.1] | 4.4 [0.4, 8.7] |
| gpt-oss-120b | 21.1 [-8.0, 46.8] | -5.0 [-12.3, 2.2] | -6.9 [-14.6, 6.7] | 0.8 [-3.6, 5.7] |
| claude-haiku-4.5 | 9.9 [-41.0, 53.9] | -2.5 [-14.5, 12.1] | -3.5 [-10.4, 0.0] | -0.4 [-7.8, 6.8] |
| llama-3.1-8b-instruct | -12.4 [-45.0, 21.4] | 5.0 [-3.9, 13.3] | -3.5 [-12.8, 2.9] | -5.2 [-10.6, 0.3] |

## Model-pair separation (paired score difference, same cases and draws)

A pair is separated when the 95% interval of the score difference excludes 0.

### Arm 4aj: 14 of 21 pairs separated

| Model A (higher by 4aj score) | Model B | A minus B, score [95% CI] | Separated |
|---|---|---|---|
| gemini-3.1-pro-preview | gpt-5.6-terra | 0.1 [-8.4, 10.4] | no |
| gemini-3.1-pro-preview | claude-sonnet-4.6 | 34.8 [14.4, 54.8] | yes |
| gemini-3.1-pro-preview | glm-5.3 | 34.9 [7.0, 71.8] | yes |
| gemini-3.1-pro-preview | gpt-oss-120b | 41.0 [18.8, 70.3] | yes |
| gemini-3.1-pro-preview | claude-haiku-4.5 | 59.8 [29.2, 95.5] | yes |
| gemini-3.1-pro-preview | llama-3.1-8b-instruct | 182.9 [139.5, 227.2] | yes |
| gpt-5.6-terra | claude-sonnet-4.6 | 34.7 [11.9, 56.4] | yes |
| gpt-5.6-terra | glm-5.3 | 34.8 [6.9, 71.1] | yes |
| gpt-5.6-terra | gpt-oss-120b | 41.0 [18.6, 70.1] | yes |
| gpt-5.6-terra | claude-haiku-4.5 | 59.7 [30.1, 95.2] | yes |
| gpt-5.6-terra | llama-3.1-8b-instruct | 182.9 [137.5, 229.0] | yes |
| claude-sonnet-4.6 | glm-5.3 | 0.1 [-24.0, 28.8] | no |
| claude-sonnet-4.6 | gpt-oss-120b | 6.2 [-12.1, 30.6] | no |
| claude-sonnet-4.6 | claude-haiku-4.5 | 25.0 [-8.5, 64.7] | no |
| claude-sonnet-4.6 | llama-3.1-8b-instruct | 148.2 [107.9, 192.1] | yes |
| glm-5.3 | gpt-oss-120b | 6.2 [-19.1, 31.9] | no |
| glm-5.3 | claude-haiku-4.5 | 24.9 [-9.8, 58.5] | no |
| glm-5.3 | llama-3.1-8b-instruct | 148.1 [106.5, 192.3] | yes |
| gpt-oss-120b | claude-haiku-4.5 | 18.7 [-13.8, 53.8] | no |
| gpt-oss-120b | llama-3.1-8b-instruct | 141.9 [101.4, 182.6] | yes |
| claude-haiku-4.5 | llama-3.1-8b-instruct | 123.2 [79.1, 170.6] | yes |

### Arm 4bj: 15 of 21 pairs separated

| Model A (higher by 4aj score) | Model B | A minus B, score [95% CI] | Separated |
|---|---|---|---|
| gemini-3.1-pro-preview | gpt-5.6-terra | 5.5 [-3.5, 15.8] | no |
| gemini-3.1-pro-preview | claude-sonnet-4.6 | 16.8 [2.1, 37.0] | yes |
| gemini-3.1-pro-preview | glm-5.3 | 39.6 [10.8, 79.5] | yes |
| gemini-3.1-pro-preview | gpt-oss-120b | 63.0 [32.2, 95.2] | yes |
| gemini-3.1-pro-preview | claude-haiku-4.5 | 70.5 [24.1, 119.5] | yes |
| gemini-3.1-pro-preview | llama-3.1-8b-instruct | 171.4 [132.5, 215.8] | yes |
| gpt-5.6-terra | claude-sonnet-4.6 | 11.3 [-0.7, 28.1] | no |
| gpt-5.6-terra | glm-5.3 | 34.2 [8.6, 69.5] | yes |
| gpt-5.6-terra | gpt-oss-120b | 57.5 [29.4, 87.2] | yes |
| gpt-5.6-terra | claude-haiku-4.5 | 65.1 [20.2, 111.5] | yes |
| gpt-5.6-terra | llama-3.1-8b-instruct | 165.9 [128.0, 211.9] | yes |
| claude-sonnet-4.6 | glm-5.3 | 22.9 [-5.2, 54.6] | no |
| claude-sonnet-4.6 | gpt-oss-120b | 46.3 [17.4, 74.3] | yes |
| claude-sonnet-4.6 | claude-haiku-4.5 | 53.8 [6.8, 102.0] | yes |
| claude-sonnet-4.6 | llama-3.1-8b-instruct | 154.6 [114.7, 199.6] | yes |
| glm-5.3 | gpt-oss-120b | 23.4 [-10.9, 52.7] | no |
| glm-5.3 | claude-haiku-4.5 | 30.9 [-27.4, 86.3] | no |
| glm-5.3 | llama-3.1-8b-instruct | 131.8 [81.7, 181.3] | yes |
| gpt-oss-120b | claude-haiku-4.5 | 7.5 [-32.9, 51.9] | no |
| gpt-oss-120b | llama-3.1-8b-instruct | 108.4 [63.6, 161.8] | yes |
| claude-haiku-4.5 | llama-3.1-8b-instruct | 100.9 [34.0, 168.3] | yes |

## Sensitivity rows

The zero reference's flag passes on 16 SERIOUS cases at I471 and on 35 at I21 (a target or a credited layer-a danger); the point scores move accordingly.

CCSR: the off-list tiers include the CCSR-rated rows (zero reference I471). I21: the zero reference pinned to I21 (Possible NSTEMI / STEMI, a target on 8 SERIOUS cases).

| Model | Arm | Headline | CCSR tiers included | Zero at I21 |
|---|---|---|---|---|
| gemini-3.1-pro-preview | 4aj | 81.0 [68.2, 90.8] | 81.0 [68.2, 90.8] | 79.3 [65.0, 89.8] |
| gemini-3.1-pro-preview | 4bj | 81.9 [71.0, 90.4] | 81.9 [71.0, 90.4] | 80.2 [68.7, 89.4] |
| gpt-5.6-terra | 4aj | 81.0 [67.5, 90.2] | 81.0 [67.5, 90.2] | 79.2 [64.8, 89.2] |
| gpt-5.6-terra | 4bj | 76.5 [62.6, 87.9] | 76.5 [62.6, 87.9] | 74.3 [60.7, 86.4] |
| claude-sonnet-4.6 | 4aj | 46.3 [21.0, 72.2] | 48.9 [24.0, 74.1] | 41.3 [17.5, 68.0] |
| claude-sonnet-4.6 | 4bj | 65.2 [42.7, 83.4] | 65.2 [42.7, 83.4] | 61.9 [39.1, 80.9] |
| glm-5.3 | 4aj | 46.2 [8.5, 74.9] | 46.2 [8.5, 74.9] | 41.2 [4.4, 70.7] |
| glm-5.3 | 4bj | 42.3 [-0.3, 72.5] | 42.3 [-0.3, 72.5] | 36.9 [-3.6, 68.2] |
| gpt-oss-120b | 4aj | 40.0 [7.9, 65.0] | 40.0 [7.9, 65.0] | 34.5 [4.8, 60.0] |
| gpt-oss-120b | 4bj | 18.9 [-14.3, 52.6] | 18.9 [-14.3, 52.6] | 11.4 [-21.8, 45.7] |
| claude-haiku-4.5 | 4aj | 21.3 [-13.8, 51.1] | 26.6 [-5.8, 55.9] | 14.0 [-24.8, 45.6] |
| claude-haiku-4.5 | 4bj | 11.4 [-38.7, 59.2] | 11.4 [-38.7, 59.2] | 3.2 [-45.6, 53.3] |
| llama-3.1-8b-instruct | 4aj | -101.9 [-148.2, -56.8] | -88.5 [-131.9, -44.1] | -120.6 [-170.1, -76.7] |
| llama-3.1-8b-instruct | 4bj | -89.5 [-131.4, -51.8] | -89.5 [-131.4, -51.8] | -107.0 [-156.4, -68.1] |

## Parsing

| Row | Answered | Unreadable | Errored | No justification | Token cost (USD) |
|---|---|---|---|---|---|
| claude-haiku-4.5|4aj | 250 of 250 | 0 | 0 | 0 | 0.3877 |
| claude-haiku-4.5|4bj | 250 of 250 | 0 | 0 | 0 | 0.3994 |
| claude-sonnet-4.6|4aj | 250 of 250 | 0 | 0 | 0 | 5.045 |
| claude-sonnet-4.6|4bj | 250 of 250 | 0 | 0 | 0 | 4.9655 |
| gemini-3.1-pro-preview|4aj | 250 of 250 | 0 | 0 | 0 | 2.6949 |
| gemini-3.1-pro-preview|4bj | 250 of 250 | 1 | 1 | 1 | 2.4927 |
| glm-5.3|4aj | 250 of 250 | 6 | 6 | 6 | 0.2497 |
| glm-5.3|4bj | 250 of 250 | 10 | 10 | 11 | 0.2594 |
| gpt-5.6-terra|4aj | 250 of 250 | 0 | 0 | 0 | 1.7526 |
| gpt-5.6-terra|4bj | 250 of 250 | 0 | 0 | 0 | 1.7274 |
| gpt-oss-120b|4aj | 250 of 250 | 1 | 1 | 1 | 0.054 |
| gpt-oss-120b|4bj | 250 of 250 | 2 | 2 | 2 | 0.0514 |
| llama-3.1-8b-instruct|4aj | 250 of 250 | 0 | 0 | 0 | 0.0053 |
| llama-3.1-8b-instruct|4bj | 250 of 250 | 0 | 0 | 0 | 0.0055 |
