# Model-level criteria (pooled)

Reference: `results/phase2/reference_adjudicated.jsonl, results/phase2b/reference_adjudicated.jsonl`. Runs: `results/phase2/runs, results/phase2b/runs`. Off-list tiers: NHAMCS only (A5 headline). Cases scored: 500 (classes {'SERIOUS': 260, 'EXCLUDED': 118, 'BENIGN': 122}; reference {'ESCALATE': 344, 'ROUTINE': 148, 'UNCERTAIN': 8}). Cases without a reference row: 18.

| Criterion | Value [95% CI] | Threshold | Result |
|---|---|---|---|
| 6. SAFETY (cost-7 precision) | 86.6 [83.1, 89.4] (399 of 461) | >= 75% | pass |
| 7. Point-weighted precision | 76.2 [74.9, 77.5] | >= 65% | pass |
| 8. Reason specificity: partials over SERIOUS escalations | 17.4 [16.1, 18.7] (552 of 3178: in-list 280, off-list 187, truth 85) | reported | - |
| 9. FN rate on kept cases | 6.2 [4.7, 8.1] (47 FN, 716 TP) | reported | - |

## By model and arm

| Row | TP / FP / FN | SAFETY | Point-weighted | FN rate, kept | Partials / SERIOUS escalations |
|---|---|---|---|---|---|
| arm 4aj | 368 / 289 / 256 | 86.6 [81.8, 90.3] (214 of 247) | 77.2 [75.4, 79.0] | 6.6 [4.5, 9.5] | 265 of 1573 |
| arm 4bj | 348 / 308 / 244 | 86.4 [81.2, 90.4] (185 of 214) | 75.2 [73.2, 77.0] | 5.7 [3.8, 8.5] | 287 of 1605 |
| claude-haiku-4.5 | 112 / 85 / 81 | 91.1 [83.9, 95.2] (92 of 101) | 82.7 [79.9, 85.1] | 8.9 [5.1, 15.3] | 75 of 419 |
| claude-sonnet-4.6 | 79 / 84 / 46 | 58.3 [38.8, 75.5] (14 of 24) | 53.1 [47.5, 58.6] | 4.8 [1.9, 11.7] | 68 of 496 |
| gemini-3.1-pro-preview | 29 / 66 / 60 | 62.5 [30.6, 86.3] (5 of 8) | 41.3 [33.5, 49.5] | 12.1 [4.8, 27.3] | 57 of 512 |
| glm-5.3 | 57 / 64 / 52 | 65.3 [51.3, 77.1] (32 of 49) | 60.0 [55.2, 64.6] | 8.1 [3.5, 17.5] | 44 of 471 |
| gpt-5.6-terra | 36 / 64 / 60 | 35.7 [16.3, 61.2] (5 of 14) | 35.9 [29.3, 43.0] | 10.0 [4.0, 23.1] | 45 of 506 |
| gpt-oss-120b | 116 / 119 / 75 | 85.3 [69.9, 93.6] (29 of 34) | 66.1 [61.5, 70.3] | 4.9 [2.3, 10.3] | 112 of 486 |
| llama-3.1-8b-instruct | 287 / 115 / 126 | 96.1 [92.8, 97.9] (222 of 231) | 90.5 [89.1, 91.8] | 4.3 [2.5, 7.3] | 151 of 288 |

## 10. Anchor check (4aj minus 4bj, cost per 100 headline cases on reference-agreed penalties)

| Model | 4aj | 4bj | Difference [95% CI] | Excludes 0 |
|---|---|---|---|---|
| claude-haiku-4.5 | 82.98 | 90.84 | -7.85 [-52.33, 38.46] | no |
| claude-sonnet-4.6 | 29.58 | 13.09 | 16.49 [5.72, 27.85] | yes |
| gemini-3.1-pro-preview | 10.73 | 4.71 | 6.02 [-0.22, 19.7] | no |
| glm-5.3 | 36.65 | 28.53 | 8.12 [-10.44, 24.6] | no |
| gpt-5.6-terra | 6.81 | 10.47 | -3.66 [-13.3, 4.89] | no |
| gpt-oss-120b | 32.46 | 43.46 | -10.99 [-29.33, 7.46] | no |
| llama-3.1-8b-instruct | 233.25 | 190.58 | 42.67 [6.47, 78.08] | yes |

Zero reference (A2): I471 (PSVT, a target on 26 SERIOUS cases).
