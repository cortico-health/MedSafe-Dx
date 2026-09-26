# Phase 2 model scores (arms 4aj and 4bj)

Blinded to the reference: these are benchmark scores only. Precision against the adjudicated reference comes from `scripts/analysis/v03_phase2_precision.py` once `results/phase2/reference_adjudicated.jsonl` exists.

- **Cases:** the 250 Phase 2 cases (seed 20261003; `data/test_sets/eval-v03-phase2.case_ids.txt`, built by `scripts/build_v03_phase2_set.py`). Strata: serious_tier1 50, serious_upgrade_or_flag 70, serious_dxa_only 20, benign 60, excluded 50. 18 of the 20 drawn DXA-only cases had no DXA-only target left once the key was built (red herrings), so the next cases in the stratum's shuffled pool replaced them (section 7); the models also answered the 18 replaced cases, which are not scored here.
- **Classes** (selection rules, key targets): 140 SERIOUS, 58 BENIGN, 52 EXCLUDED; the headline covers 198 cases.
- **Scoring:** amendments A3-A5 with the selection rules' classes and credits (A4 truth partials). Off-list tiers: NHAMCS-rated rows only (A5). Zero reference (A2): I471 (PSVT, a target on 34 SERIOUS cases).
- **Run:** `inference/run_config_v03_abj.json` (as the 4aj and 4bj runs on the 150), prompt v7a4aj and v7a4bj with the justification line, via OpenRouter; provenance in `results/phase2/runs/provenance.json`. Token cost 21.46 USD over 14 files and every case the models answered (account spend delta 21.27 USD). A json_parse_failure left after the one retry is scored as unreadable (routine), as on the 150 (spec section 9); the runner lists those files as incomplete in the provenance, and we did not re-run them.

## Scores

Score is `score_z_bal` with its 95% interval. U: SERIOUS cases costing a full miss. O: BENIGN cases escalated. Partial: SERIOUS cases charged a partial (in-list / off-list / truth). All in %.

| Model | Arm | Score [95% CI] | Score, sample mix | U | O | Partial (in / off / truth) | Escalated |
|---|---|---|---|---|---|---|---|
| gpt-5.6-terra | 4aj | 72.0 [46.0, 89.3] | 65.6 | 2.1 [0.0, 4.0] | 15.5 [5.0, 34.3] | 18.6 (15.0 / 2.9 / 0.7) | 69.6 |
| gpt-5.6-terra | 4bj | 65.5 [36.0, 86.9] | 57.7 | 3.6 [0.8, 6.0] | 19.0 [8.5, 34.1] | 16.4 (13.6 / 2.1 / 0.7) | 68.8 |
| gemini-3.1-pro-preview | 4aj | 62.1 [19.8, 88.9] | 49.1 | 5.0 [0.0, 10.6] | 12.1 [3.0, 28.6] | 19.3 (15.0 / 3.6 / 0.7) | 66.0 |
| gemini-3.1-pro-preview | 4bj | 67.4 [33.3, 88.8] | 57.1 | 2.9 [0.0, 6.2] | 12.1 [2.2, 30.4] | 25.0 (18.6 / 3.6 / 2.9) | 66.4 |
| gpt-oss-120b | 4aj | 33.0 [-1.5, 62.0] | 16.6 | 7.1 [2.3, 13.3] | 34.5 [12.5, 65.5] | 32.9 (22.9 / 9.3 / 0.7) | 72.8 |
| gpt-oss-120b | 4bj | 14.7 [-17.1, 47.7] | -7.4 | 10.7 [3.2, 19.4] | 41.4 [14.9, 74.4] | 32.9 (24.3 / 7.9 / 0.7) | 72.0 |
| claude-sonnet-4.6 | 4aj | 32.1 [-5.9, 65.8] | 15.3 | 9.3 [3.6, 13.7] | 34.5 [10.8, 65.7] | 19.3 (11.4 / 7.9 / 0.0) | 70.0 |
| claude-sonnet-4.6 | 4bj | 47.3 [2.6, 80.4] | 33.7 | 6.4 [0.9, 12.3] | 25.9 [5.3, 55.8] | 21.4 (12.1 / 8.6 / 0.7) | 68.8 |
| glm-5.3 | 4aj | 15.4 [-60.0, 69.0] | -20.2 | 17.1 [5.3, 30.1] | 13.8 [3.3, 29.6] | 14.3 (9.3 / 4.3 / 0.7) | 59.2 |
| glm-5.3 | 4bj | 9.5 [-77.2, 70.3] | -32.5 | 20.0 [6.2, 34.7] | 6.9 [0.0, 21.9] | 11.4 (7.9 / 2.9 / 0.7) | 54.8 |
| claude-haiku-4.5 | 4aj | -13.1 [-91.5, 45.9] | -65.6 | 25.0 [11.6, 39.9] | 8.6 [0.0, 24.0] | 14.3 (5.0 / 2.1 / 7.1) | 50.8 |
| claude-haiku-4.5 | 4bj | -23.6 [-103.4, 47.9] | -79.8 | 27.1 [11.4, 40.6] | 12.1 [2.3, 29.4] | 14.3 (8.6 / 2.1 / 3.6) | 51.2 |
| llama-3.1-8b-instruct | 4aj | -125.2 [-183.4, -70.1] | -235.0 | 51.4 [39.4, 63.2] | 6.9 [0.0, 18.0] | 27.1 (12.1 / 13.6 / 1.4) | 31.6 |
| llama-3.1-8b-instruct | 4bj | -118.2 [-187.3, -62.2] | -222.7 | 48.6 [34.9, 61.4] | 10.3 [2.9, 20.0] | 31.4 (19.3 / 8.6 / 3.6) | 36.8 |

## Arm 4aj minus 4bj, per model (paired)

| Model | Score | U | O | Escalated |
|---|---|---|---|---|
| gpt-5.6-terra | 6.5 [-7.4, 18.4] | -1.4 [-3.9, 1.7] | -3.5 [-9.8, 7.7] | 0.8 [-1.9, 3.4] |
| gemini-3.1-pro-preview | -5.3 [-17.0, 3.6] | 2.1 [0.0, 4.9] | 0.0 [-5.3, 5.0] | -0.4 [-3.1, 2.9] |
| gpt-oss-120b | 18.2 [-9.1, 44.4] | -3.6 [-11.1, 2.6] | -6.9 [-14.6, 6.7] | 0.8 [-3.6, 5.7] |
| claude-sonnet-4.6 | -15.1 [-29.7, 0.6] | 2.9 [-0.7, 6.6] | 8.6 [1.9, 17.3] | 1.2 [-1.8, 3.8] |
| glm-5.3 | 5.9 [-14.7, 27.2] | -2.9 [-7.9, 2.9] | 6.9 [-2.9, 17.1] | 4.4 [0.4, 8.7] |
| claude-haiku-4.5 | 10.5 [-35.7, 52.9] | -2.1 [-13.5, 10.5] | -3.5 [-10.4, 0.0] | -0.4 [-7.8, 6.8] |
| llama-3.1-8b-instruct | -7.0 [-38.9, 26.2] | 2.9 [-4.9, 11.2] | -3.5 [-12.8, 2.9] | -5.2 [-10.6, 0.3] |

## Model-pair separation (paired score difference, same cases and draws)

A pair is separated when the 95% interval of the score difference excludes 0.

### Arm 4aj: 13 of 21 pairs separated

| Model A (higher by 4aj score) | Model B | A minus B, score [95% CI] | Separated |
|---|---|---|---|
| gpt-5.6-terra | gemini-3.1-pro-preview | 9.9 [-6.2, 30.5] | no |
| gpt-5.6-terra | gpt-oss-120b | 39.0 [17.9, 66.7] | yes |
| gpt-5.6-terra | claude-sonnet-4.6 | 39.8 [17.6, 61.1] | yes |
| gpt-5.6-terra | glm-5.3 | 56.6 [13.4, 111.6] | yes |
| gpt-5.6-terra | claude-haiku-4.5 | 85.0 [37.0, 143.2] | yes |
| gpt-5.6-terra | llama-3.1-8b-instruct | 197.1 [148.2, 243.4] | yes |
| gemini-3.1-pro-preview | gpt-oss-120b | 29.1 [-1.8, 64.4] | no |
| gemini-3.1-pro-preview | claude-sonnet-4.6 | 29.9 [9.5, 51.7] | yes |
| gemini-3.1-pro-preview | glm-5.3 | 46.7 [11.9, 87.6] | yes |
| gemini-3.1-pro-preview | claude-haiku-4.5 | 75.2 [35.6, 117.8] | yes |
| gemini-3.1-pro-preview | llama-3.1-8b-instruct | 187.2 [145.6, 227.7] | yes |
| gpt-oss-120b | claude-sonnet-4.6 | 0.8 [-24.7, 21.6] | no |
| gpt-oss-120b | glm-5.3 | 17.6 [-25.1, 70.2] | no |
| gpt-oss-120b | claude-haiku-4.5 | 46.0 [-8.0, 110.0] | no |
| gpt-oss-120b | llama-3.1-8b-instruct | 158.1 [112.0, 204.5] | yes |
| claude-sonnet-4.6 | glm-5.3 | 16.8 [-18.3, 61.5] | no |
| claude-sonnet-4.6 | claude-haiku-4.5 | 45.2 [-0.5, 99.8] | no |
| claude-sonnet-4.6 | llama-3.1-8b-instruct | 157.3 [115.4, 201.7] | yes |
| glm-5.3 | claude-haiku-4.5 | 28.5 [-3.9, 59.0] | no |
| glm-5.3 | llama-3.1-8b-instruct | 140.6 [100.0, 186.5] | yes |
| claude-haiku-4.5 | llama-3.1-8b-instruct | 112.1 [65.0, 164.0] | yes |

### Arm 4bj: 15 of 21 pairs separated

| Model A (higher by 4aj score) | Model B | A minus B, score [95% CI] | Separated |
|---|---|---|---|
| gpt-5.6-terra | gemini-3.1-pro-preview | -1.9 [-13.3, 8.1] | no |
| gpt-5.6-terra | gpt-oss-120b | 50.8 [21.1, 80.7] | yes |
| gpt-5.6-terra | claude-sonnet-4.6 | 18.2 [1.6, 38.5] | yes |
| gpt-5.6-terra | glm-5.3 | 56.0 [11.0, 117.4] | yes |
| gpt-5.6-terra | claude-haiku-4.5 | 89.1 [30.2, 148.8] | yes |
| gpt-5.6-terra | llama-3.1-8b-instruct | 183.7 [136.4, 234.8] | yes |
| gemini-3.1-pro-preview | gpt-oss-120b | 52.7 [15.5, 87.1] | yes |
| gemini-3.1-pro-preview | claude-sonnet-4.6 | 20.1 [4.0, 40.2] | yes |
| gemini-3.1-pro-preview | glm-5.3 | 57.9 [13.3, 115.5] | yes |
| gemini-3.1-pro-preview | claude-haiku-4.5 | 91.0 [34.4, 147.7] | yes |
| gemini-3.1-pro-preview | llama-3.1-8b-instruct | 185.6 [141.3, 231.9] | yes |
| gpt-oss-120b | claude-sonnet-4.6 | -32.5 [-68.8, 7.4] | no |
| gpt-oss-120b | glm-5.3 | 5.2 [-50.5, 80.8] | no |
| gpt-oss-120b | claude-haiku-4.5 | 38.4 [-23.5, 111.5] | no |
| gpt-oss-120b | llama-3.1-8b-instruct | 132.9 [73.6, 203.0] | yes |
| claude-sonnet-4.6 | glm-5.3 | 37.7 [-2.6, 86.6] | no |
| claude-sonnet-4.6 | claude-haiku-4.5 | 70.9 [17.1, 122.2] | yes |
| claude-sonnet-4.6 | llama-3.1-8b-instruct | 165.4 [122.0, 209.7] | yes |
| glm-5.3 | claude-haiku-4.5 | 33.2 [-20.1, 85.7] | no |
| glm-5.3 | llama-3.1-8b-instruct | 127.7 [80.9, 178.5] | yes |
| claude-haiku-4.5 | llama-3.1-8b-instruct | 94.5 [33.2, 160.7] | yes |

## Sensitivity rows

The zero reference's flag passes on 35 SERIOUS cases at I471 and on 35 at I21 (a target or a credited layer-a danger); equal pass counts give equal point scores, and the intervals differ because the passes fall on different conditions.

CCSR: the off-list tiers include the CCSR-rated rows (zero reference I471). I21: the zero reference pinned to I21 (Possible NSTEMI / STEMI, a target on 8 SERIOUS cases).

| Model | Arm | Headline | CCSR tiers included | Zero at I21 |
|---|---|---|---|---|
| gpt-5.6-terra | 4aj | 72.0 [46.0, 89.3] | 72.0 [46.0, 89.3] | 72.0 [52.4, 88.1] |
| gpt-5.6-terra | 4bj | 65.5 [36.0, 86.9] | 65.5 [36.0, 86.9] | 65.5 [45.2, 85.5] |
| gemini-3.1-pro-preview | 4aj | 62.1 [19.8, 88.9] | 62.1 [19.8, 88.9] | 62.1 [29.1, 87.7] |
| gemini-3.1-pro-preview | 4bj | 67.4 [33.3, 88.8] | 67.4 [33.3, 88.8] | 67.4 [42.7, 87.5] |
| gpt-oss-120b | 4aj | 33.0 [-1.5, 62.0] | 33.0 [-1.5, 62.0] | 33.0 [6.7, 57.3] |
| gpt-oss-120b | 4bj | 14.7 [-17.1, 47.7] | 14.7 [-17.1, 47.7] | 14.7 [-15.5, 43.2] |
| claude-sonnet-4.6 | 4aj | 32.1 [-5.9, 65.8] | 34.6 [-3.4, 68.3] | 32.1 [6.8, 61.1] |
| claude-sonnet-4.6 | 4bj | 47.3 [2.6, 80.4] | 47.3 [2.6, 80.4] | 47.3 [14.3, 78.0] |
| glm-5.3 | 4aj | 15.4 [-60.0, 69.0] | 15.4 [-60.0, 69.0] | 15.4 [-36.8, 64.7] |
| glm-5.3 | 4bj | 9.5 [-77.2, 70.3] | 9.5 [-77.2, 70.3] | 9.5 [-51.8, 65.9] |
| claude-haiku-4.5 | 4aj | -13.1 [-91.5, 45.9] | -8.2 [-86.5, 50.0] | -13.1 [-67.5, 39.9] |
| claude-haiku-4.5 | 4bj | -23.6 [-103.4, 47.9] | -23.6 [-103.4, 47.9] | -23.6 [-79.3, 42.2] |
| llama-3.1-8b-instruct | 4aj | -125.2 [-183.4, -70.1] | -112.9 [-172.8, -57.0] | -125.2 [-165.6, -85.1] |
| llama-3.1-8b-instruct | 4bj | -118.2 [-187.3, -62.2] | -118.2 [-187.3, -62.2] | -118.2 [-160.7, -77.1] |

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
