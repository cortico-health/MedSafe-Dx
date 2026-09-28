# v0.3 full run: model scores (arms 4aj and 4bj)

- **Cases:** 900 (seed 20261005; `data/test_sets/eval-v03-full.case_ids.txt`, built by `scripts/build_v03_full_set.py`; design in docs/v0.3-case-selection-rules.md section 7.4, frozen at 7e67e24). Strata: serious_tier1 500, serious_upgrade_or_flag 140, benign 260. 23 drawn BENIGN cases fell to X9 under the key and were replaced by the next cases in their buckets. Cases with an exact public twin: 0.
- **Classes:** 640 SERIOUS, 260 BENIGN; the headline covers 900 cases.
- **Scoring:** amendments A3-A5 with the selection rules' classes and credits (A4 truth partials), rule P5 demoted (decision 21). Off-list tiers: NHAMCS-rated rows (A5). Zero point (A2): I21 (Possible NSTEMI / STEMI, a target on 56 SERIOUS cases). Headline arm 4aj (decision 18); 4bj secondary.
- **Intervals:** 95%, resampling cases within each true condition (the drawn mix is the estimand; spec record R2); the condition bootstrap, which also varies the mix, is the sensitivity column. 2,000 draws, seed 20260923.
- **Run:** `inference/run_config_v03_abj.json`, prompts v7a4aj and v7a4bj, via OpenRouter; provenance in `results/v03_full/runs/provenance.json`. Token cost 69.80 USD over 14 files (account spend delta 69.78 USD). A parse failure left after the one retry is scored as unreadable (routine), as in Phase 2b.
- **No precision:** this set has no reference review.

## Scores, arm 4aj (headline)

Score is `score_z_bal`. U: SERIOUS cases costing a full miss. O: BENIGN cases escalated. Partial: SERIOUS cases charged a partial (in-list / off-list / truth). All in %.

| Model | Score [95% CI, within-condition] | Condition bootstrap | U | O | Partial (in / off / truth) | Escalated |
|---|---|---|---|---|---|---|
| gemini-3.1-pro-preview | 68.5 [63.6, 73.2] | [52.9, 81.9] | 2.5 [1.4, 3.6] | 27.7 [24.4, 30.9] | 9.7 (6.1 / 3.6 / 0.0) | 77.3 |
| gpt-5.6-terra | 64.2 [58.5, 69.4] | [46.9, 78.7] | 3.3 [2.0, 4.5] | 29.6 [26.0, 33.5] | 9.8 (7.0 / 2.7 / 0.2) | 77.3 |
| claude-sonnet-4.6 | 50.6 [43.2, 57.5] | [31.6, 67.3] | 5.9 [4.4, 7.7] | 34.6 [30.6, 38.9] | 10.0 (5.2 / 4.8 / 0.0) | 76.9 |
| glm-5.3 | 40.5 [32.7, 47.9] | [8.9, 66.3] | 10.2 [8.3, 12.0] | 23.5 [19.8, 27.4] | 9.2 (5.5 / 3.8 / 0.0) | 70.7 |
| gpt-oss-120b | 33.6 [26.5, 40.8] | [12.4, 54.0] | 7.7 [6.0, 9.4] | 43.1 [38.5, 47.5] | 19.1 (14.2 / 4.5 / 0.3) | 78.1 |
| claude-haiku-4.5 | 19.9 [10.9, 29.4] | [-11.9, 46.6] | 15.0 [12.7, 17.3] | 16.9 [13.3, 20.9] | 17.7 (11.9 / 2.8 / 3.0) | 65.3 |
| llama-3.1-8b-instruct | -116.6 [-128.0, -104.4] | [-153.4, -75.7] | 49.4 [46.1, 52.4] | 6.9 [4.1, 9.8] | 25.2 (15.5 / 9.7 / 0.0) | 38.0 |

## Scores, arm 4bj (secondary)

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

A pair is separated when the 95% interval of the score difference excludes 0. The last column counts it under the condition bootstrap. Intervals are not corrected for the 21 pairs.

### Arm 4aj: 19 of 21 pairs separated (13 under the condition bootstrap)

| Model A (higher by 4aj score) | Model B | A minus B [95% CI] | Separated | Condition bootstrap |
|---|---|---|---|---|
| gemini-3.1-pro-preview | gpt-5.6-terra | 4.3 [-1.2, 9.9] | no | no |
| gemini-3.1-pro-preview | claude-sonnet-4.6 | 17.9 [11.0, 25.2] | yes | yes |
| gemini-3.1-pro-preview | glm-5.3 | 28.0 [20.5, 35.7] | yes | yes |
| gemini-3.1-pro-preview | gpt-oss-120b | 34.9 [27.0, 42.6] | yes | yes |
| gemini-3.1-pro-preview | claude-haiku-4.5 | 48.6 [38.8, 58.8] | yes | yes |
| gemini-3.1-pro-preview | llama-3.1-8b-instruct | 185.1 [172.1, 197.5] | yes | yes |
| gpt-5.6-terra | claude-sonnet-4.6 | 13.6 [5.8, 21.5] | yes | no |
| gpt-5.6-terra | glm-5.3 | 23.7 [15.7, 32.2] | yes | no |
| gpt-5.6-terra | gpt-oss-120b | 30.6 [22.0, 39.4] | yes | yes |
| gpt-5.6-terra | claude-haiku-4.5 | 44.2 [34.0, 54.4] | yes | yes |
| gpt-5.6-terra | llama-3.1-8b-instruct | 180.8 [167.3, 192.8] | yes | yes |
| claude-sonnet-4.6 | glm-5.3 | 10.1 [0.6, 19.5] | yes | no |
| claude-sonnet-4.6 | gpt-oss-120b | 16.9 [7.0, 26.3] | yes | yes |
| claude-sonnet-4.6 | claude-haiku-4.5 | 30.6 [19.6, 41.7] | yes | no |
| claude-sonnet-4.6 | llama-3.1-8b-instruct | 167.2 [153.9, 180.5] | yes | yes |
| glm-5.3 | gpt-oss-120b | 6.9 [-3.2, 16.3] | no | no |
| glm-5.3 | claude-haiku-4.5 | 20.5 [9.3, 31.7] | yes | no |
| glm-5.3 | llama-3.1-8b-instruct | 157.1 [143.6, 170.6] | yes | yes |
| gpt-oss-120b | claude-haiku-4.5 | 13.7 [2.8, 24.9] | yes | no |
| gpt-oss-120b | llama-3.1-8b-instruct | 150.2 [136.7, 163.8] | yes | yes |
| claude-haiku-4.5 | llama-3.1-8b-instruct | 136.6 [121.6, 150.7] | yes | yes |

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
| gemini-3.1-pro-preview | 4aj | 68.5 [63.6, 73.2] | 68.5 [63.6, 73.2] | 68.5 [63.6, 73.2] |
| gemini-3.1-pro-preview | 4bj | 67.0 [62.4, 71.5] | 67.0 [62.4, 71.5] | 67.0 [62.4, 71.5] |
| gpt-5.6-terra | 4aj | 64.2 [58.5, 69.4] | 64.2 [58.5, 69.4] | 64.2 [58.5, 69.4] |
| gpt-5.6-terra | 4bj | 65.4 [60.4, 70.2] | 65.4 [60.4, 70.2] | 65.4 [60.4, 70.2] |
| claude-sonnet-4.6 | 4aj | 50.6 [43.2, 57.5] | 52.2 [45.2, 58.6] | 50.6 [43.2, 57.5] |
| claude-sonnet-4.6 | 4bj | 48.3 [41.3, 55.1] | 49.4 [42.5, 55.9] | 48.3 [41.3, 55.1] |
| glm-5.3 | 4aj | 40.5 [32.7, 47.9] | 41.0 [33.2, 48.5] | 40.5 [32.7, 47.9] |
| glm-5.3 | 4bj | 38.0 [30.4, 46.0] | 38.6 [31.2, 46.5] | 38.0 [30.4, 46.0] |
| gpt-oss-120b | 4aj | 33.6 [26.5, 40.8] | 34.2 [27.0, 41.4] | 33.6 [26.5, 40.8] |
| gpt-oss-120b | 4bj | 20.0 [12.5, 27.4] | 21.1 [13.4, 28.4] | 20.0 [12.5, 27.4] |
| claude-haiku-4.5 | 4aj | 19.9 [10.9, 29.4] | 26.9 [18.5, 35.6] | 19.9 [10.9, 29.4] |
| claude-haiku-4.5 | 4bj | 20.6 [10.6, 30.6] | 23.8 [14.3, 33.6] | 20.6 [10.6, 30.6] |
| llama-3.1-8b-instruct | 4aj | -116.6 [-128.0, -104.4] | -103.8 [-115.0, -92.4] | -116.6 [-128.0, -104.4] |
| llama-3.1-8b-instruct | 4bj | -101.4 [-113.5, -90.3] | -95.5 [-107.4, -84.3] | -101.4 [-113.5, -90.3] |

## Full misses per condition, SERIOUS cases (arm 4aj)

| Condition | n | gemini-3.1-pro-preview | gpt-5.6-terra | claude-sonnet-4.6 | glm-5.3 | gpt-oss-120b | claude-haiku-4.5 | llama-3.1-8b-instruct |
|---|---|---|---|---|---|---|---|---|
| PSVT | 25 | 6 | 4 | 6 | 18 | 7 | 6 | 17 |
| Acute dystonic reactions | 25 | 2 | 1 | 4 | 9 | 8 | 11 | 23 |
| Ebola | 25 | 0 | 1 | 9 | 4 | 10 | 3 | 23 |
| Epiglottitis | 25 | 0 | 0 | 0 | 4 | 3 | 20 | 22 |
| Viral pharyngitis | 18 | 0 | 2 | 1 | 8 | 1 | 10 | 16 |
| Larygospasm | 25 | 0 | 0 | 0 | 9 | 3 | 8 | 13 |
| Cluster headache | 8 | 4 | 2 | 4 | 5 | 7 | 2 | 8 |
| Scombroid food poisoning | 25 | 0 | 0 | 0 | 0 | 1 | 6 | 20 |
| Pancreatic neoplasm | 25 | 1 | 9 | 0 | 0 | 1 | 4 | 9 |
| SLE | 12 | 1 | 2 | 1 | 3 | 2 | 3 | 12 |
| Pneumonia | 25 | 0 | 0 | 1 | 1 | 0 | 4 | 17 |
| Guillain-Barré syndrome | 25 | 0 | 0 | 1 | 0 | 0 | 0 | 19 |
| Boerhaave | 25 | 0 | 0 | 0 | 0 | 0 | 0 | 19 |
| Sarcoidosis | 8 | 2 | 0 | 7 | 2 | 3 | 0 | 5 |
| Spontaneous pneumothorax | 25 | 0 | 0 | 0 | 0 | 0 | 5 | 13 |
| Anemia | 31 | 0 | 0 | 2 | 0 | 3 | 2 | 9 |
| Pulmonary embolism | 25 | 0 | 0 | 0 | 0 | 0 | 0 | 14 |
| Anaphylaxis | 25 | 0 | 0 | 0 | 0 | 0 | 0 | 11 |
| Myocarditis | 25 | 0 | 0 | 0 | 0 | 0 | 3 | 7 |
| Myasthenia gravis | 8 | 0 | 0 | 1 | 0 | 0 | 0 | 7 |
| Pulmonary neoplasm | 25 | 0 | 0 | 1 | 0 | 0 | 0 | 7 |
| Pericarditis | 14 | 0 | 0 | 0 | 0 | 0 | 0 | 7 |
| Acute pulmonary edema | 25 | 0 | 0 | 0 | 0 | 0 | 0 | 4 |
| Atrial fibrillation | 8 | 0 | 0 | 0 | 0 | 0 | 4 | 0 |
| Tuberculosis | 8 | 0 | 0 | 0 | 2 | 0 | 1 | 1 |
| Spontaneous rib fracture | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| Acute COPD exacerbation / infection | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| Bronchiectasis | 2 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| GERD | 5 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| Panic attack | 5 | 0 | 0 | 0 | 0 | 0 | 2 | 0 |
| Stable angina | 25 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| Unstable angina | 25 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| Localized edema | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| Possible NSTEMI / STEMI | 25 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| Bronchospasm / acute asthma exacerbation | 25 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Escalations per condition, BENIGN cases (arm 4aj)

| Condition | n | gemini-3.1-pro-preview | gpt-5.6-terra | claude-sonnet-4.6 | glm-5.3 | gpt-oss-120b | claude-haiku-4.5 | llama-3.1-8b-instruct |
|---|---|---|---|---|---|---|---|---|
| Localized edema | 18 | 18 | 18 | 17 | 18 | 16 | 15 | 1 |
| Bronchitis | 18 | 17 | 17 | 18 | 13 | 17 | 8 | 1 |
| Panic attack | 19 | 18 | 15 | 13 | 9 | 16 | 5 | 8 |
| Acute laryngitis | 19 | 3 | 8 | 14 | 9 | 15 | 4 | 1 |
| Viral pharyngitis | 19 | 4 | 7 | 10 | 4 | 15 | 1 | 0 |
| Allergic sinusitis | 19 | 2 | 2 | 10 | 1 | 4 | 4 | 1 |
| Whooping cough | 18 | 2 | 1 | 0 | 3 | 11 | 4 | 3 |
| Anemia | 18 | 7 | 4 | 2 | 2 | 2 | 1 | 3 |
| Sarcoidosis | 18 | 0 | 3 | 1 | 2 | 8 | 0 | 0 |
| URTI | 19 | 0 | 2 | 2 | 0 | 4 | 1 | 0 |
| Chronic rhinosinusitis | 19 | 0 | 0 | 2 | 0 | 3 | 0 | 0 |
| Acute rhinosinusitis | 19 | 1 | 0 | 1 | 0 | 0 | 1 | 0 |
| SLE | 18 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| Acute otitis media | 19 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Arm 4bj's per-condition tables are in scores.json.

## Parsing

| Row | Answered | Unreadable | Errored | No justification | Token cost (USD) |
|---|---|---|---|---|---|
| claude-haiku-4.5|4aj | 900 of 900 | 0 | 0 | 0 | 1.3957 |
| claude-haiku-4.5|4bj | 900 of 900 | 0 | 0 | 0 | 1.4463 |
| claude-sonnet-4.6|4aj | 900 of 900 | 0 | 0 | 0 | 17.3652 |
| claude-sonnet-4.6|4bj | 900 of 900 | 0 | 0 | 0 | 16.854 |
| gemini-3.1-pro-preview|4aj | 900 of 900 | 0 | 0 | 0 | 9.478 |
| gemini-3.1-pro-preview|4bj | 900 of 900 | 0 | 0 | 0 | 8.7437 |
| glm-5.3|4aj | 900 of 900 | 0 | 0 | 0 | 1.0155 |
| glm-5.3|4bj | 900 of 900 | 1 | 1 | 1 | 1.0026 |
| gpt-5.6-terra|4aj | 900 of 900 | 0 | 0 | 0 | 6.1222 |
| gpt-5.6-terra|4bj | 900 of 900 | 0 | 0 | 0 | 5.9569 |
| gpt-oss-120b|4aj | 900 of 900 | 0 | 0 | 0 | 0.1972 |
| gpt-oss-120b|4bj | 900 of 900 | 0 | 0 | 0 | 0.1822 |
| llama-3.1-8b-instruct|4aj | 900 of 900 | 0 | 0 | 0 | 0.0186 |
| llama-3.1-8b-instruct|4bj | 900 of 900 | 0 | 0 | 0 | 0.0194 |
