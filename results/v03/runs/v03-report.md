# MedSafe-Dx v0.3 scores

Spec: spec/v0.3-scoring.md draft 2. Scorer: evaluator/v03_score.py at 81c3803.

Cost points per 100 patients seen: missing a patient with a serious concern costs 7, an unneeded concern on a clearly low-risk patient costs 1, and every other answer is free. Lower is better.

Intervals: 95%, 2000 draws of a bootstrap that resamples true conditions (clusters); paired differences use the same draws. Flags match under equivalent, narrower and broader codes; diagnosis under equivalent and narrower. Unreadable output scores as NO.

## Main sample (470; 10 per condition)

470 cases, 47 conditions: 234 with an R10 target (284 targets, 2 in undetermined classes), 118 clearly low-risk, 118 intermediate (either answer is free); 269 with an R5 target; 73 with a section 7 red flag.

Constant rows on these cases: perfect 0.0, always YES 25.1, always NO 348.5.

| Row | SC [95% CI] | Miss part | Concern part | H % | OC % | COV % | MT truth / DXA | Unreadable % (in R10) | SC readable | YES % |
|---|---|---|---|---|---|---|---|---|---|---|
| **openai/gpt-5.6-terra** | 20.4 [11.3, 30.6] | 0.0 | 20.4 | 0.0 (0/234) | 81.4 (96/118) | 60.8 | 36/200 / 49/84 | 0.0 (0) | 20.4 | 94.9 |
| **openai/gpt-oss-120b** | 22.8 [12.6, 34.0] | 0.0 | 22.8 | 0.0 (0/234) | 90.7 (107/118) | 56.2 | 58/200 / 51/84 | 0.0 (0) | 22.8 | 97.7 |
| [Perfect (YES iff an R10 target; flags = R5 targets)] | 0.0 [0.0, 0.0] | 0.0 | 0.0 | 0.0 (0/234) | 0.0 (0/118) | 100.0 | 0/200 / 0/84 | 0.0 (0) | 0.0 | 49.8 |
| [Always YES (fixed 5 tier-1 flags)] | 25.1 [14.0, 37.2] | 0.0 | 25.1 | 0.0 (0/234) | 100.0 (118/118) | 35.6 | 150/200 / 27/84 | 0.0 (0) | 25.1 | 100.0 |
| [Naive Bayes (dataset-knowledge ceiling, not a clinical target)] | 50.6 [19.4, 92.3] | 50.6 | 0.0 | 14.5 (34/234) | 0.0 (0/118) | 57.3 | 0/200 / 79/84 | 0.0 (0) | 50.6 | 42.6 |
| [DXA at the target threshold (reference for COV and DX only)] | 67.0 [32.8, 108.7] | 67.0 | 0.0 | 19.2 (45/234) | 0.0 (0/118) | 62.2 | 50/200 / 0/84 | 0.0 (0) | 67.0 | 40.2 |
| [Always NO] | 348.5 [253.2, 440.9] | 348.5 | 0.0 | 100.0 (234/234) | 0.0 (0/118) | 0.0 | 200/200 / 84/84 | 0.0 (0) | 348.5 | 0.0 |

MT is R10 targets not flagged (true-condition / DXA-derived). Unreadable % is the share of all cases; the bracket counts unreadable R10 cases.

### Against blanket concern (row minus always YES, SC)

| Row | Difference [95% CI] | Beats blanket concern |
|---|---|---|
| **openai/gpt-5.6-terra** | -4.7 [-8.9, -1.3] | yes |
| **openai/gpt-oss-120b** | -2.3 [-6.4, -0.2] | yes |
| [Perfect (YES iff an R10 target; flags = R5 targets)] | -25.1 [-37.2, -14.0] | yes |
| [Naive Bayes (dataset-knowledge ceiling, not a clinical target)] | 25.5 [-9.2, 70.2] | no |
| [DXA at the target threshold (reference for COV and DX only)] | 41.9 [1.9, 87.0] | no |
| [Always NO] | 323.4 [219.6, 423.2] | no |

### Paired model differences (first minus second)

| Pair | SC | Miss part | Concern part | H (pp) | OC (pp) | COV (pp) |
|---|---|---|---|---|---|---|
| openai/gpt-5.6-terra vs openai/gpt-oss-120b | -2.3 [-5.3, 0.0] | 0.0 [0.0, 0.0] | -2.3 [-5.3, 0.0] | 0.0 [0.0, 0.0] | -9.3 [-19.8, 0.0] | 4.6 [-4.3, 13.1] |

### Sensitivity rows (SC, cost points per 100 patients)

| Row | openai/gpt-5.6-terra | openai/gpt-oss-120b | always-yes | naive-bayes | dxa |
|---|---|---|---|---|---|
| Primary SC | 20.4 [11.3, 30.6] | 22.8 [12.6, 34.0] | 25.1 [14.0, 37.2] | 50.6 [19.4, 92.3] | 67.0 [32.8, 108.7] |
| ratio_1/5: miss costs 5, concern 1 | 20.4 [11.3, 30.6] | 22.8 [12.6, 34.0] | 25.1 [14.0, 37.2] | 36.2 [13.8, 66.0] | 47.9 [23.4, 77.7] |
| ratio_1/10: miss costs 10, concern 1 | 20.4 [11.3, 30.6] | 22.8 [12.6, 34.0] | 25.1 [14.0, 37.2] | 72.3 [27.7, 131.9] | 95.7 [46.8, 155.3] |
| truth_only: targets: tier-1 true conditions only | 20.4 [11.3, 30.6] | 22.8 [12.6, 34.0] | 25.1 [14.0, 37.2] | 0.0 [0.0, 0.0] | 67.0 [32.8, 108.7] |
| R12.5: targets at DXA >= 12.5% (truth always) | 20.4 [11.3, 30.6] | 22.8 [12.6, 34.0] | 25.1 [14.0, 37.2] | 31.3 [8.9, 59.6] | 67.0 [32.8, 108.7] |
| R20: targets at DXA >= 20% (truth always) | 20.4 [11.3, 30.6] | 22.8 [12.6, 34.0] | 25.1 [14.0, 37.2] | 7.4 [0.0, 19.4] | 67.0 [32.8, 108.7] |
| H_prime: a YES whose flags name no tier-1 condition counts as reassurance | 42.8 [21.3, 73.2] | 45.1 [26.8, 65.1] | 25.1 [14.0, 37.2] | 50.6 [19.4, 92.3] | 67.0 [32.8, 108.7] |
| rate_form: 7 x H + OC, percentage points (the tolerance distance) | 81.4 [68.9, 93.2] | 90.7 [75.2, 99.2] | 100.0 [100.0, 100.0] | 101.7 [37.4, 200.1] | 134.6 [75.8, 198.2] |
| concern_tier2: concern also charged on tier-2 truths with no R5 target | 35.5 [24.7, 47.4] | 38.3 [26.6, 50.6] | 40.6 [28.5, 53.4] | 50.6 [19.4, 92.3] | 67.0 [32.8, 108.7] |
| tiers:severity: tiers from DDXPlus severity only (187 R10, 120 clearly low-risk) | 20.9 [11.5, 31.3] | 23.2 [12.8, 34.7] | 25.5 [14.5, 37.5] | 40.2 [11.9, 80.4] | 44.7 [17.8, 78.9] |
| tiers:one-source: one-source upgrade rule (adds AF, anemia, HIV, SLE) (265 R10, 105 clearly low-risk) | 17.7 [8.9, 27.0] | 20.0 [10.0, 30.6] | 22.3 [11.7, 33.6] | 96.8 [44.7, 159.4] | 113.2 [62.6, 171.3] |
| mix:ddxplus: DDXPlus mix (adult test split; synthetic too) | 22.7 [11.9, 34.4] | 26.3 [13.8, 40.0] | 28.8 [15.8, 42.4] | 49.0 [19.7, 88.9] | 67.7 [31.6, 112.7] |
| mix:nhamcs: ED visit mix, not intake prevalence (CDC NHAMCS 2016-2022, adults) | 29.5 [10.4, 48.2] | 34.4 [12.9, 56.1] | 35.8 [14.0, 57.3] | 60.2 [13.0, 144.4] | 93.5 [12.1, 187.7] |

Always YES under each mix: ddxplus 28.8, nhamcs 35.8.
Always YES under each tier rule: severity 25.5, one-source 22.3.

### Code maps (COV, MT, H' SC, CON)

| Row | Map | COV % | MT % | SC under H' | YES no flags | YES no tier-1 flag | NO with tier-1 flag |
|---|---|---|---|---|---|---|---|
| openai/gpt-5.6-terra | standard | 60.8 [49.1, 71.6] | 29.9 [17.9, 42.3] | 42.8 [21.3, 73.2] | 0 | 120 | 0 |
| openai/gpt-5.6-terra | strict | 52.9 [40.1, 64.4] | 38.7 [25.1, 51.9] | 50.2 [27.2, 81.5] | 0 | 126 | 0 |
| openai/gpt-5.6-terra | lenient | 88.8 [83.0, 93.7] | 5.3 [1.4, 10.3] | 21.9 [12.3, 32.3] | 0 | 58 | 0 |
| openai/gpt-oss-120b | standard | 56.2 [44.4, 67.3] | 38.4 [23.7, 52.4] | 45.1 [26.8, 65.1] | 0 | 115 | 0 |
| openai/gpt-oss-120b | strict | 51.3 [39.0, 63.7] | 43.0 [27.5, 57.6] | 58.5 [31.3, 92.3] | 0 | 126 | 0 |
| openai/gpt-oss-120b | lenient | 92.2 [87.5, 95.9] | 4.6 [0.7, 9.7] | 24.3 [13.6, 35.7] | 0 | 45 | 0 |
| always-yes | standard | 35.6 [23.1, 50.3] | 62.3 [44.8, 79.0] | 25.1 [14.0, 37.2] | 0 | 0 | 0 |
| always-yes | strict | 35.6 [23.1, 50.3] | 62.3 [44.8, 79.0] | 25.1 [14.0, 37.2] | 0 | 0 | 0 |
| always-yes | lenient | 52.4 [36.9, 67.7] | 41.5 [24.3, 59.3] | 25.1 [14.0, 37.2] | 0 | 0 | 0 |
| naive-bayes | standard | 57.3 [43.1, 68.6] | 27.8 [18.7, 39.5] | 50.6 [19.4, 92.3] | 0 | 0 | 0 |
| naive-bayes | strict | 56.0 [41.8, 67.4] | 29.9 [20.1, 41.8] | 50.6 [19.4, 92.3] | 0 | 0 | 0 |
| naive-bayes | lenient | 61.8 [47.0, 73.3] | 21.8 [12.5, 34.2] | 50.6 [19.4, 92.3] | 0 | 0 | 0 |
| dxa | standard | 62.2 [51.6, 72.1] | 17.6 [10.6, 25.0] | 67.0 [32.8, 108.7] | 0 | 0 | 0 |
| dxa | strict | 62.2 [51.6, 72.1] | 17.6 [10.6, 25.0] | 67.0 [32.8, 108.7] | 0 | 0 | 0 |
| dxa | lenient | 63.7 [53.2, 73.5] | 16.9 [9.9, 24.5] | 67.0 [32.8, 108.7] | 0 | 0 | 0 |

### Diagnosis, consistency and red flags

| Row | Top-1 % | Top-5 % | D1 Brier | D2 at 60/70/80 | E | CON (YES no flags / YES no tier-1 / NO with tier-1) | YES on red-flag cases | p_serious present |
|---|---|---|---|---|---|---|---|---|
| **openai/gpt-5.6-terra** | 58.5 [48.1, 69.1] | 80.9 [72.5, 88.9] | 0.233 | 0/0/0 | 44/200 | 0 / 120 / 0 | 73/73 | 470/470 |
| **openai/gpt-oss-120b** | 47.2 [35.7, 58.3] | 70.4 [59.8, 80.2] | 0.235 | 0/0/0 | 69/200 | 0 / 115 / 0 | 73/73 | 470/470 |
| [Perfect (YES iff an R10 target; flags = R5 targets)] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 0.000 | 0/0/0 | 0/200 | 0 / 0 / 35 | 37/73 | 0/470 |
| [Always YES (fixed 5 tier-1 flags)] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.000 | 0/0/0 | 200/200 | 0 / 0 / 0 | 73/73 | 0/470 |
| [Naive Bayes (dataset-knowledge ceiling, not a clinical target)] | 99.4 [98.3, 100.0] | 100.0 [100.0, 100.0] | 0.005 | 0/0/0 | 0/200 | 0 / 0 / 0 | 30/73 | 470/470 |
| [DXA at the target threshold (reference for COV and DX only)] | 69.4 [60.0, 77.9] | 96.6 [94.7, 98.5] | 0.375 | 0/0/0 | 9/200 | 0 / 0 / 0 | 32/73 | 470/470 |
| [Always NO] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.000 | 0/0/0 | 200/200 | 0 / 0 / 0 | 0/73 | 0/470 |

Memorisation flag (section 11, top-1 diagnosis): openai/gpt-5.6-terra: not flagged; openai/gpt-oss-120b: not flagged.

### Runs

| Row | Predictions | Errors | Finish reasons | Retried | Truncated | Completion tokens (mean / max) | Reasoning tokens | Cost $ | Parse rules fired |
|---|---|---|---|---|---|---|---|---|---|
| openai/gpt-5.6-terra | 470 | 0 | {'stop': 470} | 0 | 0 | 338 / 652 | 112317 | 2.51 | {'p_serious:fraction_scaled': 5, 'differential:rescaled_to_100': 2, 'differential:invalid_code': 1} |
| openai/gpt-oss-120b | 470 | 0 | {'stop': 470} | 0 | 0 | 774 / 1371 | 252804 | 0.11 | none |

## Atypical serious pool (tier-1 truth, DXA top diagnosis tier 3)

150 cases, 17 conditions: 150 with an R10 target (178 targets, 3 in undetermined classes), 0 clearly low-risk, 0 intermediate (either answer is free); 150 with an R5 target; 24 with a section 7 red flag.

Constant rows on these cases: perfect 0.0, always YES 0.0, always NO 700.0.

| Row | SC [95% CI] | Miss part | Concern part | H % | OC % | COV % | MT truth / DXA | Unreadable % (in R10) | SC readable | YES % |
|---|---|---|---|---|---|---|---|---|---|---|
| **openai/gpt-5.6-terra** | 0.0 [0.0, 0.0] | 0.0 | 0.0 | 0.0 (0/150) | - (0/0) | 78.8 | 20/150 / 20/28 | 0.0 (0) | 0.0 | 100.0 |
| **openai/gpt-oss-120b** | 0.0 [0.0, 0.0] | 0.0 | 0.0 | 0.0 (0/150) | - (0/0) | 66.8 | 44/150 / 20/28 | 0.0 (0) | 0.0 | 100.0 |
| [Perfect (YES iff an R10 target; flags = R5 targets)] | 0.0 [0.0, 0.0] | 0.0 | 0.0 | 0.0 (0/150) | - (0/0) | 100.0 | 0/150 / 0/28 | 0.0 (0) | 0.0 | 100.0 |
| [Always YES (fixed 5 tier-1 flags)] | 0.0 [0.0, 0.0] | 0.0 | 0.0 | 0.0 (0/150) | - (0/0) | 37.7 | 100/150 / 13/28 | 0.0 (0) | 0.0 | 100.0 |
| [Naive Bayes (dataset-knowledge ceiling, not a clinical target)] | 0.0 [0.0, 0.0] | 0.0 | 0.0 | 0.0 (0/150) | - (0/0) | 79.6 | 0/150 / 28/28 | 0.0 (0) | 0.0 | 100.0 |
| [DXA at the target threshold (reference for COV and DX only)] | 224.0 [130.1, 328.0] | 224.0 | 0.0 | 32.0 (48/150) | - (0/0) | 58.0 | 67/150 / 0/28 | 0.0 (0) | 224.0 | 68.0 |
| [Always NO] | 700.0 [700.0, 700.0] | 700.0 | 0.0 | 100.0 (150/150) | - (0/0) | 0.0 | 150/150 / 28/28 | 0.0 (0) | 700.0 | 0.0 |

MT is R10 targets not flagged (true-condition / DXA-derived). Unreadable % is the share of all cases; the bracket counts unreadable R10 cases.

### Against blanket concern (row minus always YES, SC)

| Row | Difference [95% CI] | Beats blanket concern |
|---|---|---|
| **openai/gpt-5.6-terra** | 0.0 [0.0, 0.0] | no |
| **openai/gpt-oss-120b** | 0.0 [0.0, 0.0] | no |
| [Perfect (YES iff an R10 target; flags = R5 targets)] | 0.0 [0.0, 0.0] | no |
| [Naive Bayes (dataset-knowledge ceiling, not a clinical target)] | 0.0 [0.0, 0.0] | no |
| [DXA at the target threshold (reference for COV and DX only)] | 224.0 [130.1, 328.0] | no |
| [Always NO] | 700.0 [700.0, 700.0] | no |

### Paired model differences (first minus second)

| Pair | SC | Miss part | Concern part | H (pp) | OC (pp) | COV (pp) |
|---|---|---|---|---|---|---|
| openai/gpt-5.6-terra vs openai/gpt-oss-120b | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | - - | 12.0 [2.9, 23.3] |

### Diagnosis, consistency and red flags

| Row | Top-1 % | Top-5 % | D1 Brier | D2 at 60/70/80 | E | CON (YES no flags / YES no tier-1 / NO with tier-1) | YES on red-flag cases | p_serious present |
|---|---|---|---|---|---|---|---|---|
| **openai/gpt-5.6-terra** | 44.0 [24.3, 64.0] | 74.7 [58.0, 89.2] | 0.214 | 0/0/0 | 38/150 | 0 / 0 / 0 | 24/24 | 150/150 |
| **openai/gpt-oss-120b** | 42.7 [22.5, 62.6] | 62.7 [42.7, 80.4] | 0.226 | 0/0/0 | 56/150 | 0 / 3 / 0 | 24/24 | 150/150 |
| [Perfect (YES iff an R10 target; flags = R5 targets)] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 0.000 | 0/0/0 | 0/150 | 0 / 0 / 0 | 24/24 | 0/150 |
| [Always YES (fixed 5 tier-1 flags)] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.000 | 0/0/0 | 150/150 | 0 / 0 / 0 | 24/24 | 0/150 |
| [Naive Bayes (dataset-knowledge ceiling, not a clinical target)] | 98.7 [95.7, 100.0] | 100.0 [100.0, 100.0] | 0.006 | 2/0/0 | 0/150 | 0 / 0 / 0 | 24/24 | 150/150 |
| [DXA at the target threshold (reference for COV and DX only)] | 0.0 [0.0, 0.0] | 76.0 [56.6, 92.4] | 0.060 | 0/0/0 | 36/150 | 0 / 0 / 0 | 20/24 | 150/150 |
| [Always NO] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.000 | 0/0/0 | 150/150 | 0 / 0 / 0 | 0/24 | 0/150 |

Memorisation flag (section 11, top-1 diagnosis): openai/gpt-5.6-terra: not flagged; openai/gpt-oss-120b: not flagged.

### Runs

| Row | Predictions | Errors | Finish reasons | Retried | Truncated | Completion tokens (mean / max) | Reasoning tokens | Cost $ | Parse rules fired |
|---|---|---|---|---|---|---|---|---|---|
| openai/gpt-5.6-terra | 150 | 0 | {'stop': 150} | 0 | 0 | 328 / 622 | 33920 | 0.79 | none |
| openai/gpt-oss-120b | 150 | 0 | {'stop': 150} | 0 | 0 | 736 / 1291 | 71693 | 0.03 | none |

## High-risk pool (non-truth tier-1 condition at DXA p >= 10%, truth not tier 1)

100 cases, 14 conditions: 100 with an R10 target (110 targets, 0 in undetermined classes), 0 clearly low-risk, 0 intermediate (either answer is free); 100 with an R5 target; 20 with a section 7 red flag.

Constant rows on these cases: perfect 0.0, always YES 0.0, always NO 700.0.

| Row | SC [95% CI] | Miss part | Concern part | H % | OC % | COV % | MT truth / DXA | Unreadable % (in R10) | SC readable | YES % |
|---|---|---|---|---|---|---|---|---|---|---|
| **openai/gpt-oss-120b** | 0.0 [0.0, 0.0] | 0.0 | 0.0 | 0.0 (0/100) | - (0/0) | 41.5 | 0/0 / 65/110 | 0.0 (0) | 0.0 | 100.0 |
| **openai/gpt-5.6-terra** | 14.0 [0.0, 44.2] | 14.0 | 0.0 | 2.0 (2/100) | - (0/0) | 34.0 | 0/0 / 74/110 | 0.0 (0) | 14.0 | 98.0 |
| [Perfect (YES iff an R10 target; flags = R5 targets)] | 0.0 [0.0, 0.0] | 0.0 | 0.0 | 0.0 (0/100) | - (0/0) | 100.0 | 0/0 / 0/110 | 0.0 (0) | 0.0 | 100.0 |
| [Always YES (fixed 5 tier-1 flags)] | 0.0 [0.0, 0.0] | 0.0 | 0.0 | 0.0 (0/100) | - (0/0) | 31.0 | 0/0 / 73/110 | 0.0 (0) | 0.0 | 100.0 |
| [DXA at the target threshold (reference for COV and DX only)] | 0.0 [0.0, 0.0] | 0.0 | 0.0 | 0.0 (0/100) | - (0/0) | 93.0 | 0/0 / 0/110 | 0.0 (0) | 0.0 | 100.0 |
| [Always NO] | 700.0 [700.0, 700.0] | 700.0 | 0.0 | 100.0 (100/100) | - (0/0) | 0.0 | 0/0 / 110/110 | 0.0 (0) | 700.0 | 0.0 |
| [Naive Bayes (dataset-knowledge ceiling, not a clinical target)] | 700.0 [700.0, 700.0] | 700.0 | 0.0 | 100.0 (100/100) | - (0/0) | 0.0 | 0/0 / 110/110 | 0.0 (0) | 700.0 | 0.0 |

MT is R10 targets not flagged (true-condition / DXA-derived). Unreadable % is the share of all cases; the bracket counts unreadable R10 cases.

### Against blanket concern (row minus always YES, SC)

| Row | Difference [95% CI] | Beats blanket concern |
|---|---|---|
| **openai/gpt-oss-120b** | 0.0 [0.0, 0.0] | no |
| **openai/gpt-5.6-terra** | 14.0 [0.0, 44.2] | no |
| [Perfect (YES iff an R10 target; flags = R5 targets)] | 0.0 [0.0, 0.0] | no |
| [DXA at the target threshold (reference for COV and DX only)] | 0.0 [0.0, 0.0] | no |
| [Always NO] | 700.0 [700.0, 700.0] | no |
| [Naive Bayes (dataset-knowledge ceiling, not a clinical target)] | 700.0 [700.0, 700.0] | no |

### Paired model differences (first minus second)

| Pair | SC | Miss part | Concern part | H (pp) | OC (pp) | COV (pp) |
|---|---|---|---|---|---|---|
| openai/gpt-5.6-terra vs openai/gpt-oss-120b | 14.0 [0.0, 44.2] | 14.0 [0.0, 44.2] | 0.0 [0.0, 0.0] | 2.0 [0.0, 6.3] | - - | -7.5 [-15.6, 0.0] |

### Diagnosis, consistency and red flags

| Row | Top-1 % | Top-5 % | D1 Brier | D2 at 60/70/80 | E | CON (YES no flags / YES no tier-1 / NO with tier-1) | YES on red-flag cases | p_serious present |
|---|---|---|---|---|---|---|---|---|
| **openai/gpt-oss-120b** | 55.0 [39.4, 70.7] | 88.0 [74.1, 96.4] | 0.271 | 0/0/0 | 0/0 | 0 / 12 / 0 | 20/20 | 100/100 |
| **openai/gpt-5.6-terra** | 68.0 [46.2, 82.7] | 96.0 [88.3, 100.0] | 0.269 | 0/0/0 | 0/0 | 0 / 21 / 0 | 20/20 | 100/100 |
| [Perfect (YES iff an R10 target; flags = R5 targets)] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 0.000 | 0/0/0 | 0/0 | 0 / 0 / 0 | 20/20 | 0/100 |
| [Always YES (fixed 5 tier-1 flags)] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.000 | 0/0/0 | 0/0 | 0 / 0 / 0 | 20/20 | 0/100 |
| [DXA at the target threshold (reference for COV and DX only)] | 63.0 [47.6, 75.0] | 96.0 [91.9, 100.0] | 0.406 | 0/0/0 | 0/0 | 0 / 0 / 0 | 20/20 | 100/100 |
| [Always NO] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.000 | 0/0/0 | 0/0 | 0 / 0 / 0 | 0/20 | 0/100 |
| [Naive Bayes (dataset-knowledge ceiling, not a clinical target)] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 0.000 | 0/0/0 | 0/0 | 0 / 0 / 0 | 0/20 | 100/100 |

Memorisation flag (section 11, top-1 diagnosis): openai/gpt-5.6-terra: not flagged; openai/gpt-oss-120b: not flagged.

### Runs

| Row | Predictions | Errors | Finish reasons | Retried | Truncated | Completion tokens (mean / max) | Reasoning tokens | Cost $ | Parse rules fired |
|---|---|---|---|---|---|---|---|---|---|
| openai/gpt-5.6-terra | 100 | 0 | {'stop': 100} | 0 | 0 | 336 / 625 | 23459 | 0.53 | none |
| openai/gpt-oss-120b | 100 | 0 | {'stop': 100} | 0 | 0 | 760 / 1330 | 51773 | 0.02 | none |

## Limits

Every result carries the limits of spec/v0.3-scoring.md section 11: memorisation (naive Bayes reads the DDXPlus truth at about 98%), a closed world of 49 conditions, one condition per synthetic patient, DDXPlus probabilities that are not real-world probabilities, provisional tiers, a provisional 7:1 ratio, and 10 patients per condition.
