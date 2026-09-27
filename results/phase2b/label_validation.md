# Phase 2b label validation of the frozen case-selection rules

Freeze commit 54dbc3f; draw seed 20261004 (docs/v0.3-case-selection-rules.md section 7). Script: `scripts/analysis/v03_phase2_validation.py`; cases from `scripts/analysis/v03_phase2_cases.py`. Reference: two blind Fable reviews with verified citations (A on cases 0-124, B on 125-249), a blind Astra review from knowledge (all 250), and a blind adjudication of every disagreement and every UNCERTAIN (35 cases); the 215 agreed cases adopt the agreed decision, and 10 of them were spot-checked (seed 20261004). No reviewer saw the truth, the stratum, the rules or any model output.

## Summary

1. **Criterion 1 passes: the rules' class agrees with the blind reference on 196 of 203 decided kept cases (96.6%, Wilson lower bound 93.1%).**
2. **Criterion 2 passes in every stratum: serious_tier1 98.3%; serious_upgrade_or_flag 97.5%; benign 96.4%.** The lowest stratum is benign at 96.4%. SERIOUS cases the reference keeps routine: 2 Anemia (serious_upgrade_or_flag), 1 PSVT (serious_tier1). BENIGN cases it escalates: Sarcoidosis at index 166, Sarcoidosis at index 176, Sarcoidosis at index 220, Panic attack at index 248.
3. **Criterion 3 fails: P5 flagged for demotion.**
4. **Criterion 4 passes: three-way kappa 0.700 (raw agreement 87.2%).** On the 224 cases both reviewers decided, kappa is 0.884 (raw agreement 96.0%); Fable said UNCERTAIN on 5.6% of cases and Astra on 6.0%. The adjudication decided 34 of the 35 queued cases; 1 stay UNCERTAIN.
5. **The EXCLUDED stratum is 82.6% ESCALATE, 17.4% ROUTINE, 0.0% UNCERTAIN, and the reviewers split on 15.2%.** Exclusion rules whose cases are ESCALATE at confidence 4 or more in at least 80% of draws (candidate PATCH for a later phase): K30 (7 of 7), X8 (1 of 1).

## Reviewer agreement (criterion 4: kappa >= 0.6 on the three-way decision)

| Fable \ Astra | ESCALATE | ROUTINE | UNCERTAIN |
|---|---|---|---|
| ESCALATE | 170 | 0 | 3 |
| ROUTINE | 9 | 45 | 9 |
| UNCERTAIN | 10 | 1 | 3 |

Cohen's kappa 0.700, raw agreement 87.2% (218 of 250). **PASS.**

Reported measures (decision 14):

- kappa on the 224 cases both reviewers decided (ESCALATE/ROUTINE only): 0.884, raw agreement 96.0%;
- UNCERTAIN rate: Fable 14 of 250 (5.6%), Astra 15 of 250 (6.0%).

## The reference by stratum

| Stratum | n | ESCALATE | ROUTINE | UNCERTAIN | class under the key |
|---|---|---|---|---|---|
| serious_tier1 | 60 | 58 | 1 | 1 | serious 60 |
| serious_upgrade_or_flag | 80 | 78 | 2 | 0 | serious 80 |
| benign | 60 | 3 | 57 | 0 | benign 55, excluded 5 |
| excluded | 50 | 39 | 11 | 0 | benign 9, excluded 41 |

## Criterion 1: class agreement on decided kept cases (>= 90%, Wilson lower bound >= 85%)

196 of 203 decided kept cases agree: 96.6% [93.1, 98.3]; 1 kept cases are UNCERTAIN. **PASS.**

| Class | ESCALATE | ROUTINE | UNCERTAIN | total |
|---|---|---|---|---|
| SERIOUS | 136 | 3 | 1 | 140 |
| BENIGN | 4 | 60 | 0 | 64 |
| EXCLUDED | 38 | 8 | 0 | 46 |

## Criterion 2: BENIGN >= 85% ROUTINE; each SERIOUS stratum >= 92% ESCALATE (among decided cases)

| Stratum | decided | agree | rate [Wilson 95%] | target | result |
|---|---|---|---|---|---|
| serious_tier1 | 59 | 58 | 98.3% [91.0, 99.7] | >= 92% | PASS |
| serious_upgrade_or_flag | 80 | 78 | 97.5% [91.3, 99.3] | >= 92% | PASS |
| benign | 55 | 53 | 96.4% [87.7, 99.0] | >= 85% | PASS |

5 drawn kept case(s) fall to EXCLUDED under the key (ddxplus_38189 by X9, ddxplus_62677 by X9, ddxplus_18166 by X9, ddxplus_4215 by X9, ddxplus_46931 by X9); they are counted in the EXCLUDED stratum below, not here.

## Criterion 3: each rule with >= 5 fresh cases it decided, >= 80% agreement

| Rule | cases | decided | agree | rate | ESCALATE / ROUTINE / UNCERTAIN | result |
|---|---|---|---|---|---|---|
| P1 | 15 | 15 | 15 | 100.0% | 15 / 0 / 0 | PASS |
| P10 | 2 | 2 | 2 | 100.0% | 2 / 0 / 0 | reported only (< 5 cases) |
| P12 | 8 | 8 | 8 | 100.0% | 8 / 0 / 0 | PASS |
| P2 | 16 | 16 | 16 | 100.0% | 16 / 0 / 0 | PASS |
| P3 | 13 | 13 | 12 | 92.3% | 12 / 1 / 0 | PASS |
| P4 | 4 | 4 | 4 | 100.0% | 4 / 0 / 0 | reported only (< 5 cases) |
| P5 | 9 | 9 | 7 | 77.8% | 7 / 2 / 0 | FAIL, flagged for demotion |
| P6 | 2 | 2 | 2 | 100.0% | 2 / 0 / 0 | reported only (< 5 cases) |
| P7 | 8 | 8 | 8 | 100.0% | 8 / 0 / 0 | PASS |
| P9 | 8 | 8 | 8 | 100.0% | 8 / 0 / 0 | PASS |
| K22 | 4 | 4 | 4 | 100.0% | 4 / 0 / 0 | reported only (< 5 cases) |
| K23 | 2 | 2 | 2 | 100.0% | 2 / 0 / 0 | reported only (< 5 cases) |
| K24 | 8 | 8 | 8 | 100.0% | 8 / 0 / 0 | PASS |
| K33 | 8 | 8 | 8 | 100.0% | 8 / 0 / 0 | PASS |
| K42 | 8 | 8 | 8 | 100.0% | 8 / 0 / 0 | PASS |

Rules flagged for demotion to EXCLUDE: P5.

## Criterion 5 (reported only): the EXCLUDED stratum

46 cases: ESCALATE 38 (82.6%), ROUTINE 8 (17.4%), UNCERTAIN 0 (0.0%); the two reviewers split on 7 (15.2%).

| Exclusion rule | cases | ESCALATE | ROUTINE | UNCERTAIN | split | ESCALATE at confidence >= 4 | candidate PATCH (>= 80%) |
|---|---|---|---|---|---|---|---|
| K27 | 2 | 2 | 0 | 0 | 0 | 1 (50.0%) | no |
| K30 | 7 | 7 | 0 | 0 | 0 | 7 (100.0%) | yes |
| K32 | 2 | 1 | 1 | 0 | 1 | 0 (0.0%) | no |
| X10 | 28 | 26 | 2 | 0 | 2 | 19 (67.9%) | no |
| X11 | 1 | 0 | 1 | 0 | 0 | 0 (0.0%) | no |
| X6 | 6 | 5 | 1 | 0 | 0 | 1 (16.7%) | no |
| X7 | 3 | 3 | 0 | 0 | 0 | 0 (0.0%) | no |
| X8 | 1 | 1 | 0 | 0 | 0 | 1 (100.0%) | yes |
| X9 | 8 | 4 | 4 | 0 | 5 | 2 (25.0%) | no |

## Kept cases by condition

| Condition | tier | class | n | ESCALATE | ROUTINE | UNCERTAIN | agree |
|---|---|---|---|---|---|---|---|
| Acute laryngitis | 3 | benign | 4 | 0 | 4 | 0 | 4 |
| Acute otitis media | 3 | benign | 10 | 0 | 10 | 0 | 10 |
| Acute rhinosinusitis | 3 | benign | 7 | 0 | 7 | 0 | 7 |
| Allergic sinusitis | 3 | benign | 6 | 0 | 6 | 0 | 6 |
| Bronchitis | 3 | benign | 2 | 0 | 2 | 0 | 2 |
| Chronic rhinosinusitis | 3 | benign | 6 | 0 | 6 | 0 | 6 |
| Panic attack | 3 | benign | 3 | 1 | 2 | 0 | 2 |
| Sarcoidosis | 3 | benign | 3 | 3 | 0 | 0 | 0 |
| URTI | 3 | benign | 17 | 0 | 17 | 0 | 17 |
| Viral pharyngitis | 3 | benign | 5 | 0 | 5 | 0 | 5 |
| Whooping cough | 3 | benign | 1 | 0 | 1 | 0 | 1 |
| Acute dystonic reactions | 1 | serious | 3 | 3 | 0 | 0 | 3 |
| Acute pulmonary edema | 1 | serious | 5 | 5 | 0 | 0 | 5 |
| Boerhaave | 1 | serious | 1 | 1 | 0 | 0 | 1 |
| Bronchospasm / acute asthma exacerbation | 1 | serious | 2 | 2 | 0 | 0 | 2 |
| Epiglottitis | 1 | serious | 4 | 4 | 0 | 0 | 4 |
| Guillain-Barré syndrome | 1 | serious | 4 | 3 | 0 | 1 | 3 |
| Larygospasm | 1 | serious | 3 | 3 | 0 | 0 | 3 |
| Myocarditis | 1 | serious | 1 | 1 | 0 | 0 | 1 |
| PSVT | 1 | serious | 1 | 0 | 1 | 0 | 0 |
| Pancreatic neoplasm | 1 | serious | 6 | 6 | 0 | 0 | 6 |
| Pneumonia | 1 | serious | 3 | 3 | 0 | 0 | 3 |
| Possible NSTEMI / STEMI | 1 | serious | 3 | 3 | 0 | 0 | 3 |
| Pulmonary embolism | 1 | serious | 4 | 4 | 0 | 0 | 4 |
| Pulmonary neoplasm | 1 | serious | 2 | 2 | 0 | 0 | 2 |
| Scombroid food poisoning | 1 | serious | 5 | 5 | 0 | 0 | 5 |
| Spontaneous pneumothorax | 1 | serious | 3 | 3 | 0 | 0 | 3 |
| Stable angina | 1 | serious | 6 | 6 | 0 | 0 | 6 |
| Unstable angina | 1 | serious | 4 | 4 | 0 | 0 | 4 |
| Acute COPD exacerbation / infection | 2 | serious | 2 | 2 | 0 | 0 | 2 |
| Atrial fibrillation | 2 | serious | 8 | 8 | 0 | 0 | 8 |
| Bronchiectasis | 2 | serious | 4 | 4 | 0 | 0 | 4 |
| GERD | 2 | serious | 4 | 4 | 0 | 0 | 4 |
| Myasthenia gravis | 2 | serious | 8 | 8 | 0 | 0 | 8 |
| Spontaneous rib fracture | 2 | serious | 1 | 1 | 0 | 0 | 1 |
| Tuberculosis | 2 | serious | 4 | 4 | 0 | 0 | 4 |
| Anemia | 3 | serious | 23 | 21 | 2 | 0 | 21 |
| Panic attack | 3 | serious | 3 | 3 | 0 | 0 | 3 |
| Pericarditis | 3 | serious | 8 | 8 | 0 | 0 | 8 |
| SLE | 3 | serious | 2 | 2 | 0 | 0 | 2 |
| Sarcoidosis | 3 | serious | 4 | 4 | 0 | 0 | 4 |
| Viral pharyngitis | 3 | serious | 9 | 9 | 0 | 0 | 9 |

## Kept cases where the reference disagrees with the class (7)

| Idx | Case | Truth (tier) | Stratum | Rules | Class | Reference (conf.) | Source | Rationale |
|---|---|---|---|---|---|---|---|---|
| 17 | ddxplus_65159 | PSVT (1) | serious_tier1 | - | SERIOUS | ROUTINE (4) | adjudicated | Palpitations with near-fainting, no syncope and a stimulant history are closed by an office 12-lead ECG and vitals, the initial evaluation the ESC 2018 syncope guideline names. The anaemia label is a benign mislabel that omits no dangerous treatment. |
| 77 | ddxplus_64600 | Anemia (3) | serious_upgrade_or_flag | P5 | SERIOUS | ROUTINE (3) | adjudicated | The anaemia label fits heavy periods, pallor, poor diet and low weight, and no upper-GI bleeding is reported, so a blood count, vitals and a urine pregnancy test at the visit close it. BSG 2021 reserves transfusion for severe symptomatic or circulatory compromise, which the haemoglobin and vitals decide. |
| 78 | ddxplus_5302 | Anemia (3) | serious_upgrade_or_flag | P3|P5 | SERIOUS | ROUTINE (3) | adjudicated | Fresh red rectal blood under 50 with heavy periods is lower-GI and menstrual loss; NG12 1.3.1 needs abdominal pain or weight loss to trigger FIT under 50, and the BSG 2019 lower-GI guideline decides place of care on vitals and haemoglobin, which the visit supplies. Melaena would score on Blatchford and escalate; fresh blood does not. |
| 166 | ddxplus_78338 | Sarcoidosis (3) | benign | - | BENIGN | ESCALATE (4) | adjudicated | Unexplained lymphadenopathy at 82 meets NICE NG12 1.10.6 and 1.10.8 for a suspected cancer pathway referral, and first-time vaginal discharge at 55 or over meets NG12 1.5.13 for an urgent endometrial ultrasound. A sarcoidosis label explains neither without tissue and imaging. |
| 176 | ddxplus_111673 | Sarcoidosis (3) | benign | - | BENIGN | ESCALATE (3) | adjudicated | Unexplained lymphadenopathy at 60 meets NICE NG12 1.10.6 and 1.10.8 for a lymphoma pathway referral, and a red eye under a sarcoidosis label needs a slit-lamp examination for uveitis, which the ATS 2020 guideline asks for even without eye symptoms. Sarcoidosis is a diagnosis of exclusion that itself mandates a chest film and tissue. |
| 220 | ddxplus_102395 | Sarcoidosis (3) | excluded | - | BENIGN | ESCALATE (4) | spot-checked | Sarcoidosis is not a working diagnosis to hold without imaging, and this patient has red eyes and significant breathlessness. |
| 248 | ddxplus_4193 | Panic attack (3) | excluded | - | BENIGN | ESCALATE (4) | agreed | Chest pain with sweating, choking and palpitations in a 47-year-old woman is how ACS presents in women, and panic attack is the diagnosis left after an ECG and troponin, not before. |

## Kept cases left UNCERTAIN (1)

| Idx | Case | Truth (tier) | Stratum | Rules | Class | Rationale |
|---|---|---|---|---|---|---|
| 41 | ddxplus_26708 | Guillain-Barré syndrome (1) | serious_tier1 | - | SERIOUS | NICE NG127 1.7.2 and 1.10.2 refer symmetrical limb weakness with numbness immediately only when it is rapidly progressive, and the intake carries the weakness as a bare yes with no onset or progression. The panic label rests on no anxiety history, so the evidence is balanced and reflexes and power at the visit decide. |

## Verdict

| Criterion | Result |
|---|---|
| 1. Class agreement on decided kept cases | PASS: 96.6% [93.1, 98.3] |
| 2. Per-stratum agreement | PASS: serious_tier1 98.3%; serious_upgrade_or_flag 97.5%; benign 96.4% |
| 3. Per-rule agreement (>= 5 cases) | FAIL: P5 flagged |
| 4. Reviewer kappa (three-way) | PASS: 0.700 |
| 4 (reported). Kappa on cases both reviewers decided; UNCERTAIN rate | 0.884 on 224; Fable 5.6%, Astra 6.0% |
| 5. EXCLUDED stratum (reported) | ESCALATE 82.6%, ROUTINE 17.4%, UNCERTAIN 0.0%, split 15.2% |

Section 7 says failing criterion 1 or 2 means the slice is not certified; a rule flagged under criterion 3 is demoted to EXCLUDE. This document reports; it changes no rule.
