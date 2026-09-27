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
| 6 | SAFETY: precision of the cost-7 charges >= 75% | PENDING: model run incomplete (OpenRouter key cap); the watcher resumes it | PASS: 88.7 [84.0, 92.2] (205 of 231) | PENDING: model run incomplete (OpenRouter key cap); the watcher resumes it |
| 7 | Point-weighted precision >= 65% | PENDING: model run incomplete (OpenRouter key cap); the watcher resumes it | PASS: 77.8 [75.9, 79.6] | PENDING: model run incomplete (OpenRouter key cap); the watcher resumes it |
| 8 | Reason specificity (reported) | PENDING: model run incomplete (OpenRouter key cap); the watcher resumes it | 18.8 [16.9, 20.9] (273 of 1449: in-list 143, off-list 96, truth 34) | PENDING: model run incomplete (OpenRouter key cap); the watcher resumes it |
| 9 | FN rate on kept cases (reported) | PENDING: model run incomplete (OpenRouter key cap); the watcher resumes it | 5.7 [3.8, 8.6] (21 FN, 347 TP) | PENDING: model run incomplete (OpenRouter key cap); the watcher resumes it |
| 10 | Anchor check (reported as "no consistent effect", decision 15) | PENDING: model run incomplete (OpenRouter key cap); the watcher resumes it | no interval excludes 0 | PENDING: model run incomplete (OpenRouter key cap); the watcher resumes it |

Phase 2b: 3 of 4 judged criteria pass; failed: 3 (causes below).

**Criterion 3 cause.** P5: 7 of 9 decided cases agree (77.8%). The disagreeing cases and the reference's rationale are listed in `label_validation.md`. Section 7 demotes a flagged rule to EXCLUDE; that is a rule change after unblinding, so this document reports it and the demotion is a decision for the next freeze, not an edit here.

**Model level pending.** The OpenRouter key reached its hard cap (299 USD) during the run: 1,289 of 3,500 requests succeeded (7.97 USD) and the rest returned http_403. The run resumes on the same command once the cap is raised (the runner re-runs errored entries); criteria 6-10, the scores and the pooled model rows are then filled by `v03_phase2_scores.py --phase 2b`, `v03_phase2_precision.py --set phase2b`, `--set pooled` and this script.

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