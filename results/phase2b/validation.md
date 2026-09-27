# Phase 2b validation of the case-selection rules

Freeze commit 54dbc3f (docs/v0.3-case-selection-rules.md sections 7.2 and 9); draw seed 20261004; 250 fresh cases (tier-1 60, upgraded or flagged 80, BENIGN 60, EXCLUDED 50). Label level: `label_validation.md` in this directory (`scripts/analysis/v03_phase2_validation.py --phase 2b`). Model level: `precision.md` and `model_scores.md` (`v03_phase2_precision.py --set phase2b`, `v03_phase2_scores.py --phase 2b`). No rule was changed after unblinding. The Phase 2 column is the post-hoc rerun under rule X11 (`results/phase2/post_hoc_x11/`), and the pooled column is SECONDARY: Phase 2 and Phase 2b as one 500-case set (`pooled/`).

## Pass/fail table

| # | Criterion | Phase 2b (pre-registered) | Phase 2 (post hoc, X11) | Pooled (secondary) |
|---|---|---|---|---|
| 1 | Class agreement on decided kept cases (>= 90%, Wilson lower bound >= 85%) | PASS: 196 of 203, 96.6% [93.1%, 98.3%] | PASS: 169 of 176, 96.0% [92.0%, 98.1%] | PASS: 365 of 379, 96.3% [93.9%, 97.8%] |
| 2 | Per-stratum agreement (SERIOUS >= 92% ESCALATE, BENIGN >= 85% ROUTINE) | PASS: serious_tier1 58 of 59 (98.3%); serious_upgrade_or_flag 78 of 80 (97.5%); benign 53 of 55 (96.4%) | PASS: serious_tier1 46 of 50 (92.0%); serious_upgrade_or_flag 70 of 70 (100.0%); benign 53 of 56 (94.6%) | PASS: serious_tier1 104 of 109 (95.4%); serious_upgrade_or_flag 148 of 150 (98.7%); benign 106 of 111 (95.5%) |
| 3 | Per-rule agreement (>= 80% on every rule with >= 5 cases) | FAIL: P5 flagged | PASS: no rule flagged (12 rules judged) | PASS: no rule flagged (14 rules judged) |
| 4 | Reviewer kappa >= 0.6 (three-way) | PASS: kappa 0.700 (raw agreement 87.2%) on 250 | FAIL: kappa 0.552 (raw agreement 75.6%) on 250 | PASS: kappa 0.620 (raw agreement 81.4%) on 500 |
| 4 | Reported (decision 14): kappa on cases both decided; UNCERTAIN rate per reviewer | kappa 0.884 on the 224 cases both decided (raw agreement 96.0%); UNCERTAIN Fable 14 of 250 (5.6%), Astra 15 of 250 (6.0%) | kappa 0.918 on the 187 cases both decided (raw agreement 96.8%); UNCERTAIN Fable 46 of 250 (18.4%), Astra 25 of 250 (10.0%) | kappa 0.901 on the 411 cases both decided (raw agreement 96.4%); UNCERTAIN Fable 60 of 500 (12.0%), Astra 40 of 500 (8.0%) |
| 5 | EXCLUDED stratum (reported) | 46 cases: ESCALATE 82.6%, ROUTINE 17.4%, UNCERTAIN 0.0%, split 15.2%; candidate PATCH: K30 (7 of 7), X8 (1 of 1) | 72 cases: ESCALATE 65.3%, ROUTINE 27.8%, UNCERTAIN 6.9%, split 36.1%; candidate PATCH: K30 (4 of 5) | 118 cases: ESCALATE 72.0%, ROUTINE 23.7%, UNCERTAIN 4.2%, split 28.0%; candidate PATCH: K30 (11 of 12) |
| 6 | SAFETY: precision of the cost-7 charges >= 75% | PASS: 84.3 [79.1, 88.5] (194 of 230) | PASS: 88.7 [84.0, 92.2] (205 of 231) | PASS: 86.6 [83.1, 89.4] (399 of 461) |
| 7 | Point-weighted precision >= 65% | PASS: 74.7 [72.8, 76.5] | PASS: 77.8 [75.9, 79.6] | PASS: 76.2 [74.9, 77.5] |
| 8 | Reason specificity (reported) | 16.1 [14.5, 17.9] (279 of 1729: in-list 137, off-list 91, truth 51) | 18.8 [16.9, 20.9] (273 of 1449: in-list 143, off-list 96, truth 34) | 17.4 [16.1, 18.7] (552 of 3178: in-list 280, off-list 187, truth 85) |
| 9 | FN rate on kept cases (reported) | 6.6 [4.5, 9.5] (26 FN, 369 TP) | 5.7 [3.8, 8.6] (21 FN, 347 TP) | 6.2 [4.7, 8.1] (47 FN, 716 TP) |
| 10 | Anchor check (reported as "no consistent effect", decision 15) | interval excludes 0 for claude-sonnet-4.6 (18.63 [6.5, 33.34]), glm-5.3 (20.59 [2.7, 45.46]), llama-3.1-8b-instruct (67.65 [26.06, 111.62]) | no interval excludes 0 | interval excludes 0 for claude-sonnet-4.6 (16.49 [5.72, 27.85]), llama-3.1-8b-instruct (42.67 [6.47, 78.08]) |

Phase 2b: 5 of 6 judged criteria pass; failed: 3 (causes below).

**Criterion 3 cause.** P5: 7 of 9 decided cases agree (77.8%). The disagreeing cases and the reference's rationale are listed in `label_validation.md`. Section 7 demotes a flagged rule to EXCLUDE; that is a rule change after unblinding, so this document reports it and the demotion is a decision for the next freeze, not an edit here.

## Scores (score_z_bal, 95% interval; the headline covers 204 cases: 140 SERIOUS, 64 BENIGN; zero reference I471)

| Model | 4aj | 4bj | 4aj minus 4bj (paired) |
|---|---|---|---|
| gpt-5.6-terra | 79.1 [59.1, 92.2] | 68.7 [43.6, 86.6] | 10.5 [-0.0, 25.0] |
| gemini-3.1-pro-preview | 77.6 [62.4, 89.1] | 85.6 [76.0, 92.7] | -8.0 [-19.5, 0.4] |
| gpt-oss-120b | 57.9 [36.2, 75.1] | 56.7 [38.4, 73.5] | 1.2 [-10.4, 10.7] |
| claude-sonnet-4.6 | 57.8 [37.3, 76.5] | 69.8 [52.4, 83.3] | -12.0 [-22.8, -2.2] |
| glm-5.3 | 55.2 [27.2, 77.7] | 69.4 [48.1, 86.5] | -14.2 [-26.8, -3.3] |
| claude-haiku-4.5 | 13.9 [-15.2, 46.6] | 16.9 [-20.2, 61.1] | -3.1 [-35.6, 23.8] |
| llama-3.1-8b-instruct | -101.0 [-140.5, -65.0] | -52.2 [-95.4, -18.7] | -48.7 [-78.5, -12.2] |

## Anchor check (criterion 10): 4aj minus 4bj cost per 100 headline cases on reference-agreed penalties

| Model | Phase 2b | Phase 2 (post hoc) | Pooled |
|---|---|---|---|
| gpt-5.6-terra | -5.39 [-14.68, 0.0] | -1.69 [-12.42, 11.54] | -3.66 [-13.3, 4.89] |
| gemini-3.1-pro-preview | 7.35 [0.0, 23.85] | 4.49 [-0.63, 14.0] | 6.02 [-0.22, 19.7] |
| gpt-oss-120b | 5.39 [-2.73, 16.91] | -29.78 [-66.28, 3.09] | -10.99 [-29.33, 7.46] |
| claude-sonnet-4.6 | 18.63 [6.5, 33.34] (excludes 0) | 14.04 [0.0, 29.59] | 16.49 [5.72, 27.85] (excludes 0) |
| glm-5.3 | 20.59 [2.7, 45.46] (excludes 0) | -6.18 [-34.19, 18.98] | 8.12 [-10.44, 24.6] |
| claude-haiku-4.5 | 0.0 [-40.34, 45.05] | -16.85 [-75.15, 42.08] | -7.85 [-52.33, 38.46] |
| llama-3.1-8b-instruct | 67.65 [26.06, 111.62] (excludes 0) | 14.04 [-26.67, 52.32] | 42.67 [6.47, 78.08] (excludes 0) |

Read as "no consistent effect" (decision 15): an effect is claimed only when a model's interval excludes 0 in the same direction on two independent samples.

## Criterion 3 by rule (Phase 2b)

| Rule | cases | decided | agree | rate | result |
|---|---|---|---|---|---|
| P1 | 15 | 15 | 15 | 100.0% | PASS |
| P10 | 2 | 2 | 2 | 100.0% | reported (< 5) |
| P12 | 8 | 8 | 8 | 100.0% | PASS |
| P2 | 16 | 16 | 16 | 100.0% | PASS |
| P3 | 13 | 13 | 12 | 92.3% | PASS |
| P4 | 4 | 4 | 4 | 100.0% | reported (< 5) |
| P5 | 9 | 9 | 7 | 77.8% | FAIL |
| P6 | 2 | 2 | 2 | 100.0% | reported (< 5) |
| P7 | 8 | 8 | 8 | 100.0% | PASS |
| P9 | 8 | 8 | 8 | 100.0% | PASS |
| K22 | 4 | 4 | 4 | 100.0% | reported (< 5) |
| K23 | 2 | 2 | 2 | 100.0% | reported (< 5) |
| K24 | 8 | 8 | 8 | 100.0% | PASS |
| K33 | 8 | 8 | 8 | 100.0% | PASS |
| K42 | 8 | 8 | 8 | 100.0% | PASS |

## Spend

Account spend delta for the Phase 2b run: 19.57 USD (token cost 19.55 USD over the 14 files); the 3-case smoke: 0.00 USD. Phase 2 was 21.27 USD. Model outputs stay git-ignored in `runs/`.
