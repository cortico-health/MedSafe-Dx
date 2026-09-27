# Model-level criteria (phase2b)

Reference: `results/phase2b/reference_adjudicated.jsonl`. Runs: `results/phase2b/runs`. Off-list tiers: NHAMCS only (A5 headline). Cases scored: 250 (classes {'SERIOUS': 140, 'EXCLUDED': 46, 'BENIGN': 64}; reference {'ESCALATE': 178, 'ROUTINE': 71, 'UNCERTAIN': 1}). Cases without a reference row: 0.

| Criterion | Value [95% CI] | Threshold | Result |
|---|---|---|---|
| 6. SAFETY (cost-7 precision) | 84.3 [79.1, 88.5] (194 of 230) | >= 75% | pass |
| 7. Point-weighted precision | 74.7 [72.8, 76.5] | >= 65% | pass |
| 8. Reason specificity: partials over SERIOUS escalations | 16.1 [14.5, 17.9] (279 of 1729: in-list 137, off-list 91, truth 51) | reported | - |
| 9. FN rate on kept cases | 6.6 [4.5, 9.5] (26 FN, 369 TP) | reported | - |

## By model and arm

| Row | TP / FP / FN | SAFETY | Point-weighted | FN rate, kept | Partials / SERIOUS escalations |
|---|---|---|---|---|---|
| arm 4aj | 199 / 143 / 93 | 86.4 [79.5, 91.2] (114 of 132) | 77.9 [75.4, 80.2] | 6.6 [4.0, 10.7] | 132 of 848 |
| arm 4bj | 170 / 160 / 89 | 81.6 [72.8, 88.1] (80 of 98) | 70.8 [67.8, 73.7] | 6.6 [3.8, 11.2] | 147 of 881 |
| claude-haiku-4.5 | 62 / 43 / 26 | 92.9 [83.0, 97.2] (52 of 56) | 84.8 [81.2, 87.9] | 11.4 [5.9, 21.0] | 40 of 224 |
| claude-sonnet-4.6 | 42 / 40 / 15 | 63.6 [35.4, 84.8] (7 of 11) | 56.8 [48.7, 64.5] | 0.0 [0.0, 8.4] | 33 of 269 |
| gemini-3.1-pro-preview | 17 / 30 / 22 | 40.0 [11.8, 76.9] (2 of 5) | 37.7 [27.7, 48.8] | 15.0 [5.2, 36.0] | 24 of 275 |
| glm-5.3 | 26 / 33 / 21 | 50.0 [29.9, 70.1] (10 of 20) | 48.0 [40.8, 55.3] | 7.1 [2.0, 22.6] | 22 of 260 |
| gpt-5.6-terra | 19 / 39 / 22 | 11.1 [2.0, 43.5] (1 of 9) | 22.3 [15.6, 30.9] | 17.4 [7.0, 37.1] | 26 of 271 |
| gpt-oss-120b | 53 / 59 / 19 | 80.0 [49.0, 94.3] (8 of 10) | 58.7 [51.3, 65.8] | 3.6 [1.0, 12.3] | 55 of 270 |
| llama-3.1-8b-instruct | 150 / 59 / 57 | 95.8 [90.5, 98.2] (114 of 119) | 90.4 [88.3, 92.1] | 4.5 [2.2, 8.9] | 79 of 160 |

## 10. Anchor check (4aj minus 4bj, cost per 100 headline cases on reference-agreed penalties)

| Model | 4aj | 4bj | Difference [95% CI] | Excludes 0 |
|---|---|---|---|---|
| claude-haiku-4.5 | 91.67 | 91.67 | 0.0 [-40.34, 45.05] | no |
| claude-sonnet-4.6 | 29.9 | 11.27 | 18.63 [6.5, 33.34] | yes |
| gemini-3.1-pro-preview | 10.78 | 3.43 | 7.35 [0.0, 23.85] | no |
| glm-5.3 | 31.37 | 10.78 | 20.59 [2.7, 45.46] | yes |
| gpt-5.6-terra | 3.43 | 8.82 | -5.39 [-14.68, 0.0] | no |
| gpt-oss-120b | 27.45 | 22.06 | 5.39 [-2.73, 16.91] | no |
| llama-3.1-8b-instruct | 238.24 | 170.59 | 67.65 [26.06, 111.62] | yes |

Zero reference (A2): I471 (PSVT, a target on 11 SERIOUS cases).
