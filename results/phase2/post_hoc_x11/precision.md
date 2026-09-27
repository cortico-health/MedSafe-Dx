# Model-level criteria (phase2, post hoc, rule X11)

Reference: `results/phase2/reference_adjudicated.jsonl`. Runs: `results/phase2/runs`. Off-list tiers: NHAMCS only (A5 headline). Cases scored: 250 (classes {'SERIOUS': 120, 'EXCLUDED': 72, 'BENIGN': 58}; reference {'ESCALATE': 166, 'ROUTINE': 77, 'UNCERTAIN': 7}). Cases without a reference row: 18.

| Criterion | Value [95% CI] | Threshold | Result |
|---|---|---|---|
| 6. SAFETY (cost-7 precision) | 88.7 [84.0, 92.2] (205 of 231) | >= 75% | pass |
| 7. Point-weighted precision | 77.8 [75.9, 79.6] | >= 65% | pass |
| 8. Reason specificity: partials over SERIOUS escalations | 18.8 [16.9, 20.9] (273 of 1449: in-list 143, off-list 96, truth 34) | reported | - |
| 9. FN rate on kept cases | 5.7 [3.8, 8.6] (21 FN, 347 TP) | reported | - |

## By model and arm

| Row | TP / FP / FN | SAFETY | Point-weighted | FN rate, kept | Partials / SERIOUS escalations |
|---|---|---|---|---|---|
| arm 4aj | 169 / 146 / 163 | 87.0 [79.6, 91.9] (100 of 115) | 76.5 [73.8, 79.0] | 6.6 [3.8, 11.2] | 133 of 725 |
| arm 4bj | 178 / 148 / 155 | 90.5 [83.8, 94.6] (105 of 116) | 79.1 [76.5, 81.4] | 4.8 [2.6, 8.9] | 140 of 724 |
| claude-haiku-4.5 | 50 / 42 / 55 | 88.9 [76.5, 95.2] (40 of 45) | 80.1 [75.7, 83.9] | 5.7 [1.9, 15.4] | 35 of 195 |
| claude-sonnet-4.6 | 37 / 44 / 31 | 53.8 [29.1, 76.8] (7 of 13) | 49.7 [42.0, 57.4] | 9.8 [3.9, 22.5] | 35 of 227 |
| gemini-3.1-pro-preview | 12 / 36 / 38 | 100.0 [43.8, 100.0] (3 of 3) | 45.5 [34.0, 57.4] | 7.7 [1.4, 33.3] | 33 of 237 |
| glm-5.3 | 31 / 31 / 31 | 75.9 [57.9, 87.8] (22 of 29) | 69.1 [62.9, 74.6] | 8.8 [3.0, 23.0] | 22 of 211 |
| gpt-5.6-terra | 17 / 25 / 38 | 80.0 [37.6, 96.4] (4 of 5) | 56.9 [45.4, 67.7] | 0.0 [0.0, 18.4] | 19 of 235 |
| gpt-oss-120b | 63 / 60 / 56 | 87.5 [69.0, 95.7] (21 of 24) | 70.8 [65.1, 75.9] | 6.0 [2.3, 14.4] | 57 of 216 |
| llama-3.1-8b-instruct | 137 / 56 / 69 | 96.4 [91.2, 98.6] (108 of 112) | 90.8 [88.6, 92.5] | 4.2 [1.9, 8.9] | 72 of 128 |

## 10. Anchor check (4aj minus 4bj, cost per 100 headline cases on reference-agreed penalties)

| Model | 4aj | 4bj | Difference [95% CI] | Excludes 0 |
|---|---|---|---|---|
| claude-haiku-4.5 | 73.03 | 89.89 | -16.85 [-75.15, 42.08] | no |
| claude-sonnet-4.6 | 29.21 | 15.17 | 14.04 [0.0, 29.59] | no |
| gemini-3.1-pro-preview | 10.67 | 6.18 | 4.49 [-0.63, 14.0] | no |
| glm-5.3 | 42.7 | 48.88 | -6.18 [-34.19, 18.98] | no |
| gpt-5.6-terra | 10.67 | 12.36 | -1.69 [-12.42, 11.54] | no |
| gpt-oss-120b | 38.2 | 67.98 | -29.78 [-66.28, 3.09] | no |
| llama-3.1-8b-instruct | 227.53 | 213.48 | 14.04 [-26.67, 52.32] | no |

Zero reference (A2): I471 (PSVT, a target on 15 SERIOUS cases).
