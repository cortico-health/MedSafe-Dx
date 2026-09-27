# Phase 2b model scores (arms 4aj and 4bj)

Blinded to the reference: these are benchmark scores only. Precision against the adjudicated reference comes from `scripts/analysis/v03_phase2_precision.py` once `results/phase2b/reference_adjudicated.jsonl` exists.

- **Cases:** the 250 Phase 2b cases (seed 20261004; `data/test_sets/eval-v03-phase2b.case_ids.txt`, built by `scripts/build_v03_phase2_set.py --phase 2b`). Strata: serious_tier1 60, serious_upgrade_or_flag 80, benign 60, excluded 50.
- **Classes** (selection rules, key targets): 140 SERIOUS, 64 BENIGN, 46 EXCLUDED; the headline covers 204 cases.
- **Scoring:** amendments A3-A5 with the selection rules' classes and credits (A4 truth partials). Off-list tiers: NHAMCS-rated rows only (A5). Zero reference (A2): I471 (PSVT, a target on 11 SERIOUS cases).
- **Run:** `inference/run_config_v03_abj.json` (as the 4aj and 4bj runs on the 150), prompt v7a4aj and v7a4bj with the justification line, via OpenRouter; provenance in `results/phase2b/runs/provenance.json`. Token cost 19.55 USD over 14 files and every case the models answered (account spend delta 19.57 USD). A json_parse_failure left after the one retry is scored as unreadable (routine), as on the 150 (spec section 9); the runner lists those files as incomplete in the provenance, and we did not re-run them.

## Scores

Score is `score_z_bal` with its 95% interval. U: SERIOUS cases costing a full miss. O: BENIGN cases escalated. Partial: SERIOUS cases charged a partial (in-list / off-list / truth). All in %.

| Model | Arm | Score [95% CI] | Score, sample mix | U | O | Partial (in / off / truth) | Escalated |
|---|---|---|---|---|---|---|---|
| gpt-5.6-terra | 4aj | 79.1 [59.1, 92.2] | 76.4 | 2.1 [0.0, 7.2] | 14.1 [4.5, 29.8] | 10.7 (7.1 / 0.7 / 2.9) | 70.0 |
| gpt-5.6-terra | 4bj | 68.7 [43.6, 86.6] | 64.9 | 4.3 [0.0, 10.4] | 21.9 [9.9, 43.2] | 7.9 (5.7 / 0.7 / 1.4) | 70.4 |
| gemini-3.1-pro-preview | 4aj | 77.6 [62.4, 89.1] | 74.9 | 2.9 [0.0, 6.5] | 15.6 [6.7, 30.6] | 7.1 (5.7 / 0.7 / 0.7) | 69.2 |
| gemini-3.1-pro-preview | 4bj | 85.6 [76.0, 92.7] | 84.8 | 0.7 [0.0, 2.4] | 12.5 [3.8, 26.8] | 10.0 (5.0 / 1.4 / 3.6) | 70.0 |
| gpt-oss-120b | 4aj | 57.9 [36.2, 75.1] | 55.5 | 3.6 [1.0, 6.6] | 35.9 [16.1, 68.4] | 19.3 (10.7 / 5.7 / 2.9) | 75.6 |
| gpt-oss-120b | 4bj | 56.7 [38.4, 73.5] | 54.5 | 3.6 [0.8, 6.5] | 37.5 [19.6, 67.9] | 20.0 (12.1 / 7.1 / 0.7) | 75.6 |
| claude-sonnet-4.6 | 4aj | 57.8 [37.3, 76.5] | 52.9 | 5.7 [1.6, 10.5] | 29.7 [14.3, 50.8] | 10.7 (4.3 / 5.0 / 1.4) | 70.8 |
| claude-sonnet-4.6 | 4bj | 69.8 [52.4, 83.3] | 69.6 | 2.1 [0.0, 4.4] | 29.7 [14.5, 55.2] | 12.9 (3.6 / 7.1 / 2.1) | 73.2 |
| glm-5.3 | 4aj | 55.2 [27.2, 77.7] | 42.9 | 9.3 [3.9, 16.1] | 14.1 [4.2, 30.8] | 6.4 (3.6 / 0.7 / 2.1) | 63.2 |
| glm-5.3 | 4bj | 69.4 [48.1, 86.5] | 62.8 | 5.0 [0.9, 10.0] | 14.1 [3.6, 34.1] | 9.3 (3.6 / 4.3 / 1.4) | 67.2 |
| claude-haiku-4.5 | 4aj | 13.9 [-15.2, 46.6] | -17.3 | 20.0 [12.2, 26.7] | 7.8 [0.0, 22.1] | 16.4 (5.7 / 2.9 / 7.9) | 54.4 |
| claude-haiku-4.5 | 4bj | 16.9 [-20.2, 61.1] | -13.6 | 20.0 [8.5, 29.2] | 6.2 [1.2, 14.7] | 12.1 (7.9 / 2.1 / 2.1) | 53.2 |
| llama-3.1-8b-instruct | 4aj | -101.0 [-140.5, -65.0] | -179.1 | 50.7 [39.3, 62.9] | 4.7 [0.0, 13.2] | 23.6 (9.3 / 14.3 / 0.0) | 33.2 |
| llama-3.1-8b-instruct | 4bj | -52.2 [-95.4, -18.7] | -107.8 | 35.0 [25.0, 48.4] | 12.5 [1.6, 31.9] | 32.9 (13.6 / 12.1 / 7.1) | 48.4 |

## Arm 4aj minus 4bj, per model (paired)

| Model | Score | U | O | Escalated |
|---|---|---|---|---|
| gpt-5.6-terra | 10.5 [-0.0, 25.0] | -2.1 [-5.7, 0.0] | -7.8 [-19.4, -1.4] | -0.4 [-2.4, 1.7] |
| gemini-3.1-pro-preview | -8.0 [-19.5, 0.4] | 2.1 [0.0, 5.7] | 3.1 [0.0, 9.8] | -0.8 [-2.4, 0.5] |
| gpt-oss-120b | 1.2 [-10.4, 10.7] | 0.0 [-2.4, 3.3] | -1.6 [-11.8, 8.6] | 0.0 [-3.5, 3.0] |
| claude-sonnet-4.6 | -12.0 [-22.8, -2.2] | 3.6 [0.9, 6.7] | 0.0 [-13.0, 7.8] | -2.4 [-6.1, 0.5] |
| glm-5.3 | -14.2 [-26.8, -3.3] | 4.3 [1.6, 7.8] | 0.0 [-15.0, 11.5] | -4.0 [-8.1, -0.7] |
| claude-haiku-4.5 | -3.1 [-35.6, 23.8] | 0.0 [-7.7, 9.8] | 1.6 [-4.8, 10.3] | 1.2 [-4.9, 7.2] |
| llama-3.1-8b-instruct | -48.7 [-78.5, -12.2] | 15.7 [5.5, 24.4] | -7.8 [-26.3, 4.3] | -15.2 [-22.1, -8.0] |

## Model-pair separation (paired score difference, same cases and draws)

A pair is separated when the 95% interval of the score difference excludes 0.

### Arm 4aj: 13 of 21 pairs separated

| Model A (higher by 4aj score) | Model B | A minus B, score [95% CI] | Separated |
|---|---|---|---|
| gpt-5.6-terra | gemini-3.1-pro-preview | 1.6 [-20.6, 20.4] | no |
| gpt-5.6-terra | gpt-oss-120b | 21.2 [-4.2, 43.9] | no |
| gpt-5.6-terra | claude-sonnet-4.6 | 21.3 [-4.8, 44.3] | no |
| gpt-5.6-terra | glm-5.3 | 24.0 [-5.2, 55.2] | no |
| gpt-5.6-terra | claude-haiku-4.5 | 65.3 [31.6, 96.4] | yes |
| gpt-5.6-terra | llama-3.1-8b-instruct | 180.1 [141.1, 221.9] | yes |
| gemini-3.1-pro-preview | gpt-oss-120b | 19.6 [4.1, 39.3] | yes |
| gemini-3.1-pro-preview | claude-sonnet-4.6 | 19.7 [8.9, 31.2] | yes |
| gemini-3.1-pro-preview | glm-5.3 | 22.4 [-0.0, 50.8] | no |
| gemini-3.1-pro-preview | claude-haiku-4.5 | 63.7 [26.1, 95.0] | yes |
| gemini-3.1-pro-preview | llama-3.1-8b-instruct | 178.5 [143.0, 217.7] | yes |
| gpt-oss-120b | claude-sonnet-4.6 | 0.1 [-18.6, 17.5] | no |
| gpt-oss-120b | glm-5.3 | 2.8 [-16.3, 23.9] | no |
| gpt-oss-120b | claude-haiku-4.5 | 44.0 [13.2, 69.2] | yes |
| gpt-oss-120b | llama-3.1-8b-instruct | 158.9 [122.6, 196.1] | yes |
| claude-sonnet-4.6 | glm-5.3 | 2.7 [-16.9, 27.6] | no |
| claude-sonnet-4.6 | claude-haiku-4.5 | 44.0 [7.9, 73.3] | yes |
| claude-sonnet-4.6 | llama-3.1-8b-instruct | 158.8 [122.3, 201.2] | yes |
| glm-5.3 | claude-haiku-4.5 | 41.3 [8.2, 70.5] | yes |
| glm-5.3 | llama-3.1-8b-instruct | 156.1 [121.7, 193.6] | yes |
| claude-haiku-4.5 | llama-3.1-8b-instruct | 114.8 [73.2, 163.3] | yes |

### Arm 4bj: 14 of 21 pairs separated

| Model A (higher by 4aj score) | Model B | A minus B, score [95% CI] | Separated |
|---|---|---|---|
| gpt-5.6-terra | gemini-3.1-pro-preview | -16.9 [-36.0, -3.1] | yes |
| gpt-5.6-terra | gpt-oss-120b | 11.9 [-11.9, 32.1] | no |
| gpt-5.6-terra | claude-sonnet-4.6 | -1.1 [-25.2, 17.3] | no |
| gpt-5.6-terra | glm-5.3 | -0.7 [-21.9, 17.4] | no |
| gpt-5.6-terra | claude-haiku-4.5 | 51.7 [2.2, 94.8] | yes |
| gpt-5.6-terra | llama-3.1-8b-instruct | 120.9 [77.5, 171.8] | yes |
| gemini-3.1-pro-preview | gpt-oss-120b | 28.8 [13.1, 45.4] | yes |
| gemini-3.1-pro-preview | claude-sonnet-4.6 | 15.8 [5.5, 28.6] | yes |
| gemini-3.1-pro-preview | glm-5.3 | 16.2 [2.9, 32.3] | yes |
| gemini-3.1-pro-preview | claude-haiku-4.5 | 68.6 [23.1, 108.0] | yes |
| gemini-3.1-pro-preview | llama-3.1-8b-instruct | 137.8 [100.3, 182.3] | yes |
| gpt-oss-120b | claude-sonnet-4.6 | -13.1 [-27.0, 1.5] | no |
| gpt-oss-120b | glm-5.3 | -12.7 [-30.2, 8.4] | no |
| gpt-oss-120b | claude-haiku-4.5 | 39.8 [-2.7, 75.5] | no |
| gpt-oss-120b | llama-3.1-8b-instruct | 109.0 [71.4, 156.8] | yes |
| claude-sonnet-4.6 | glm-5.3 | 0.4 [-16.1, 18.8] | no |
| claude-sonnet-4.6 | claude-haiku-4.5 | 52.9 [6.2, 89.9] | yes |
| claude-sonnet-4.6 | llama-3.1-8b-instruct | 122.1 [85.1, 169.4] | yes |
| glm-5.3 | claude-haiku-4.5 | 52.5 [1.0, 96.2] | yes |
| glm-5.3 | llama-3.1-8b-instruct | 121.7 [82.7, 166.1] | yes |
| claude-haiku-4.5 | llama-3.1-8b-instruct | 69.2 [20.0, 136.0] | yes |

## Sensitivity rows

The zero reference's flag passes on 13 SERIOUS cases at I471 and on 42 at I21 (a target or a credited layer-a danger); the point scores move accordingly.

CCSR: the off-list tiers include the CCSR-rated rows (zero reference I471). I21: the zero reference pinned to I21 (Possible NSTEMI / STEMI, a target on 10 SERIOUS cases).

| Model | Arm | Headline | CCSR tiers included | Zero at I21 |
|---|---|---|---|---|
| gpt-5.6-terra | 4aj | 79.1 [59.1, 92.2] | 79.1 [59.1, 92.2] | 76.6 [53.6, 91.3] |
| gpt-5.6-terra | 4bj | 68.7 [43.6, 86.6] | 68.7 [43.6, 86.6] | 64.9 [37.0, 85.0] |
| gemini-3.1-pro-preview | 4aj | 77.6 [62.4, 89.1] | 77.6 [62.4, 89.1] | 74.8 [57.3, 87.7] |
| gemini-3.1-pro-preview | 4bj | 85.6 [76.0, 92.7] | 85.6 [76.0, 92.7] | 83.8 [72.3, 91.9] |
| gpt-oss-120b | 4aj | 57.9 [36.2, 75.1] | 57.9 [36.2, 75.1] | 52.8 [27.5, 71.6] |
| gpt-oss-120b | 4bj | 56.7 [38.4, 73.5] | 56.7 [38.4, 73.5] | 51.5 [31.1, 68.8] |
| claude-sonnet-4.6 | 4aj | 57.8 [37.3, 76.5] | 60.1 [41.5, 77.2] | 52.7 [29.6, 72.7] |
| claude-sonnet-4.6 | 4bj | 69.8 [52.4, 83.3] | 69.8 [52.4, 83.3] | 66.2 [46.2, 81.1] |
| glm-5.3 | 4aj | 55.2 [27.2, 77.7] | 55.2 [27.2, 77.7] | 49.7 [18.5, 74.7] |
| glm-5.3 | 4bj | 69.4 [48.1, 86.5] | 69.4 [48.1, 86.5] | 65.7 [41.6, 84.7] |
| claude-haiku-4.5 | 4aj | 13.9 [-15.2, 46.6] | 18.4 [-10.6, 51.6] | 3.4 [-25.9, 37.3] |
| claude-haiku-4.5 | 4bj | 16.9 [-20.2, 61.1] | 16.9 [-20.2, 61.1] | 6.8 [-29.6, 53.6] |
| llama-3.1-8b-instruct | 4aj | -101.0 [-140.5, -65.0] | -92.0 [-128.8, -56.0] | -125.5 [-172.6, -88.1] |
| llama-3.1-8b-instruct | 4bj | -52.2 [-95.4, -18.7] | -50.0 [-94.9, -15.9] | -70.8 [-116.1, -36.2] |

## Parsing

| Row | Answered | Unreadable | Errored | No justification | Token cost (USD) |
|---|---|---|---|---|---|
| claude-haiku-4.5|4aj | 250 of 250 | 0 | 0 | 0 | 0.3872 |
| claude-haiku-4.5|4bj | 250 of 250 | 0 | 0 | 0 | 0.3984 |
| claude-sonnet-4.6|4aj | 250 of 250 | 0 | 0 | 0 | 4.9837 |
| claude-sonnet-4.6|4bj | 250 of 250 | 1 | 1 | 1 | 4.7118 |
| gemini-3.1-pro-preview|4aj | 250 of 250 | 0 | 0 | 0 | 2.5865 |
| gemini-3.1-pro-preview|4bj | 250 of 250 | 0 | 0 | 0 | 2.4344 |
| glm-5.3|4aj | 250 of 250 | 2 | 2 | 2 | 0.2257 |
| glm-5.3|4bj | 250 of 250 | 1 | 1 | 1 | 0.2396 |
| gpt-5.6-terra|4aj | 250 of 250 | 0 | 0 | 0 | 1.7524 |
| gpt-5.6-terra|4bj | 250 of 250 | 0 | 0 | 0 | 1.7089 |
| gpt-oss-120b|4aj | 250 of 250 | 0 | 0 | 0 | 0.0527 |
| gpt-oss-120b|4bj | 250 of 250 | 0 | 0 | 0 | 0.0535 |
| llama-3.1-8b-instruct|4aj | 250 of 250 | 0 | 0 | 0 | 0.0053 |
| llama-3.1-8b-instruct|4bj | 250 of 250 | 0 | 0 | 0 | 0.0053 |
