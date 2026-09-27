# Phase 2 label validation of the frozen case-selection rules (post hoc, rule X11)

Freeze commit 45a7599; draw seed 20261003 (docs/v0.3-case-selection-rules.md section 7). Script: `scripts/analysis/v03_phase2_validation.py`; cases from `scripts/analysis/v03_phase2_cases.py`. Reference: two blind Fable reviews with verified citations (A on cases 0-124, B on 125-249), a blind Astra review from knowledge (all 250), and a blind adjudication of every disagreement and every UNCERTAIN (69 cases); the 181 agreed cases adopt the agreed decision, and 10 of them were spot-checked (seed 20261003). No reviewer saw the truth, the stratum, the rules or any model output. POST HOC: the classes and deciding rules are recomputed under the rules file as it stands (rule X11 added after this phase was unblinded); the pre-registered result is the committed label_validation.md.

## Summary

1. **Criterion 1 passes: the rules' class agrees with the blind reference on 169 of 176 decided kept cases (96.0%, Wilson lower bound 92.0%).**
2. **Criterion 2 passes in every stratum: serious_tier1 92.0%; serious_upgrade_or_flag 100.0%; benign 94.6%.** The lowest stratum is serious_tier1 at 92.0%. SERIOUS cases the reference keeps routine: 4 PSVT (serious_tier1). BENIGN cases it escalates: Localized edema at index 157, Localized edema at index 173, Sarcoidosis at index 191.
3. **Criterion 3 passes: no rule with 5 or more fresh cases falls below 80% agreement.**
4. **Criterion 4 fails: three-way kappa 0.552 (raw agreement 75.6%).** On the 187 cases both reviewers decided, kappa is 0.918 (raw agreement 96.8%); Fable said UNCERTAIN on 18.4% of cases and Astra on 10.0%. The adjudication decided 62 of the 69 queued cases; 7 stay UNCERTAIN.
5. **The EXCLUDED stratum is 65.3% ESCALATE, 27.8% ROUTINE, 6.9% UNCERTAIN, and the reviewers split on 36.1%.** Exclusion rules whose cases are ESCALATE at confidence 4 or more in at least 80% of draws (candidate PATCH for a later phase): K30 (4 of 5).

## Reviewer agreement (criterion 4: kappa >= 0.6 on the three-way decision)

| Fable \ Astra | ESCALATE | ROUTINE | UNCERTAIN |
|---|---|---|---|
| ESCALATE | 134 | 0 | 1 |
| ROUTINE | 6 | 47 | 16 |
| UNCERTAIN | 38 | 0 | 8 |

Cohen's kappa 0.552, raw agreement 75.6% (189 of 250). **FAIL.**

Reported measures (decision 14):

- kappa on the 187 cases both reviewers decided (ESCALATE/ROUTINE only): 0.918, raw agreement 96.8%;
- UNCERTAIN rate: Fable 46 of 250 (18.4%), Astra 25 of 250 (10.0%).

## The reference by stratum

| Stratum | n | ESCALATE | ROUTINE | UNCERTAIN | class under the key |
|---|---|---|---|---|---|
| serious_tier1 | 50 | 46 | 4 | 0 | serious 50 |
| serious_upgrade_or_flag | 70 | 70 | 0 | 0 | serious 70 |
| serious_dxa_only | 20 | 12 | 8 | 0 | excluded 20 |
| benign | 60 | 3 | 55 | 2 | benign 58, excluded 2 |
| excluded | 50 | 35 | 10 | 5 | excluded 50 |

## Criterion 1: class agreement on decided kept cases (>= 90%, Wilson lower bound >= 85%)

169 of 176 decided kept cases agree: 96.0% [92.0, 98.1]; 2 kept cases are UNCERTAIN. **PASS.**

| Class | ESCALATE | ROUTINE | UNCERTAIN | total |
|---|---|---|---|---|
| SERIOUS | 116 | 4 | 0 | 120 |
| BENIGN | 3 | 53 | 2 | 58 |
| EXCLUDED | 47 | 20 | 5 | 72 |

## Criterion 2: BENIGN >= 85% ROUTINE; each SERIOUS stratum >= 92% ESCALATE (among decided cases)

| Stratum | decided | agree | rate [Wilson 95%] | target | result |
|---|---|---|---|---|---|
| serious_tier1 | 50 | 46 | 92.0% [81.2, 96.8] | >= 92% | PASS |
| serious_upgrade_or_flag | 70 | 70 | 100.0% [94.8, 100.0] | >= 92% | PASS |
| benign | 56 | 53 | 94.6% [85.4, 98.2] | >= 85% | PASS |
| serious_dxa_only | 0 | 0 | n/a | | not judged: no kept case (rule X11 excludes the stratum) |

22 drawn kept case(s) fall to EXCLUDED under the key (ddxplus_53176 by X11, ddxplus_27844 by X11, ddxplus_103631 by X11, ddxplus_115592 by X11, ddxplus_49506 by X11, ddxplus_30054 by X11, ddxplus_12248 by X11, ddxplus_35900 by X11, ddxplus_18411 by X11, ddxplus_87310 by X11, ddxplus_5213 by X11, ddxplus_77420 by X11, ddxplus_117822 by X11, ddxplus_18367 by X11, ddxplus_93100 by X11, ddxplus_45695 by X11, ddxplus_81398 by X11, ddxplus_4519 by X11, ddxplus_77892 by X11, ddxplus_60530 by X11, ddxplus_23603 by X9, ddxplus_68301 by X9); they are counted in the EXCLUDED stratum below, not here.

## Criterion 3: each rule with >= 5 fresh cases it decided, >= 80% agreement

| Rule | cases | decided | agree | rate | ESCALATE / ROUTINE / UNCERTAIN | result |
|---|---|---|---|---|---|---|
| P1 | 10 | 10 | 10 | 100.0% | 10 / 0 / 0 | PASS |
| P10 | 6 | 6 | 6 | 100.0% | 6 / 0 / 0 | PASS |
| P12 | 8 | 8 | 8 | 100.0% | 8 / 0 / 0 | PASS |
| P2 | 16 | 16 | 16 | 100.0% | 16 / 0 / 0 | PASS |
| P3 | 9 | 9 | 9 | 100.0% | 9 / 0 / 0 | PASS |
| P4 | 4 | 4 | 4 | 100.0% | 4 / 0 / 0 | reported only (< 5 cases) |
| P5 | 8 | 8 | 8 | 100.0% | 8 / 0 / 0 | PASS |
| P6 | 2 | 2 | 2 | 100.0% | 2 / 0 / 0 | reported only (< 5 cases) |
| P7 | 8 | 8 | 8 | 100.0% | 8 / 0 / 0 | PASS |
| P9 | 8 | 8 | 8 | 100.0% | 8 / 0 / 0 | PASS |
| K22 | 1 | 1 | 1 | 100.0% | 1 / 0 / 0 | reported only (< 5 cases) |
| K23 | 6 | 6 | 6 | 100.0% | 6 / 0 / 0 | PASS |
| K24 | 8 | 8 | 8 | 100.0% | 8 / 0 / 0 | PASS |
| K33 | 8 | 8 | 8 | 100.0% | 8 / 0 / 0 | PASS |
| K42 | 5 | 5 | 5 | 100.0% | 5 / 0 / 0 | PASS |

No rule with 5 or more fresh cases falls below 80% agreement.

## Criterion 5 (reported only): the EXCLUDED stratum

72 cases: ESCALATE 47 (65.3%), ROUTINE 20 (27.8%), UNCERTAIN 5 (6.9%); the two reviewers split on 26 (36.1%).

| Exclusion rule | cases | ESCALATE | ROUTINE | UNCERTAIN | split | ESCALATE at confidence >= 4 | candidate PATCH (>= 80%) |
|---|---|---|---|---|---|---|---|
| K27 | 3 | 3 | 0 | 0 | 1 | 1 (33.3%) | no |
| K30 | 5 | 5 | 0 | 0 | 0 | 4 (80.0%) | yes |
| K32 | 8 | 4 | 4 | 0 | 5 | 2 (25.0%) | no |
| X10 | 31 | 21 | 7 | 3 | 11 | 10 (32.3%) | no |
| X11 | 20 | 12 | 8 | 0 | 8 | 4 (20.0%) | no |
| X6 | 9 | 4 | 3 | 2 | 2 | 2 (22.2%) | no |
| X7 | 5 | 5 | 0 | 0 | 3 | 0 (0.0%) | no |
| X8 | 2 | 2 | 0 | 0 | 0 | 1 (50.0%) | no |
| X9 | 7 | 5 | 2 | 0 | 2 | 3 (42.9%) | no |

## Kept cases by condition

| Condition | tier | class | n | ESCALATE | ROUTINE | UNCERTAIN | agree |
|---|---|---|---|---|---|---|---|
| Acute laryngitis | 3 | benign | 8 | 0 | 8 | 0 | 8 |
| Acute otitis media | 3 | benign | 15 | 0 | 15 | 0 | 15 |
| Acute rhinosinusitis | 3 | benign | 4 | 0 | 4 | 0 | 4 |
| Allergic sinusitis | 3 | benign | 5 | 0 | 5 | 0 | 5 |
| Chronic rhinosinusitis | 3 | benign | 7 | 0 | 7 | 0 | 7 |
| Localized edema | 3 | benign | 2 | 2 | 0 | 0 | 0 |
| Panic attack | 3 | benign | 1 | 0 | 1 | 0 | 1 |
| SLE | 3 | benign | 1 | 0 | 1 | 0 | 1 |
| Sarcoidosis | 3 | benign | 1 | 1 | 0 | 0 | 0 |
| URTI | 3 | benign | 6 | 0 | 5 | 1 | 5 |
| Viral pharyngitis | 3 | benign | 7 | 0 | 6 | 1 | 6 |
| Whooping cough | 3 | benign | 1 | 0 | 1 | 0 | 1 |
| Acute dystonic reactions | 1 | serious | 2 | 2 | 0 | 0 | 2 |
| Acute pulmonary edema | 1 | serious | 2 | 2 | 0 | 0 | 2 |
| Anaphylaxis | 1 | serious | 4 | 4 | 0 | 0 | 4 |
| Boerhaave | 1 | serious | 4 | 4 | 0 | 0 | 4 |
| Bronchospasm / acute asthma exacerbation | 1 | serious | 2 | 2 | 0 | 0 | 2 |
| Epiglottitis | 1 | serious | 3 | 3 | 0 | 0 | 3 |
| Guillain-Barré syndrome | 1 | serious | 3 | 3 | 0 | 0 | 3 |
| Larygospasm | 1 | serious | 4 | 4 | 0 | 0 | 4 |
| Myocarditis | 1 | serious | 1 | 1 | 0 | 0 | 1 |
| PSVT | 1 | serious | 6 | 2 | 4 | 0 | 2 |
| Pancreatic neoplasm | 1 | serious | 2 | 2 | 0 | 0 | 2 |
| Pneumonia | 1 | serious | 3 | 3 | 0 | 0 | 3 |
| Possible NSTEMI / STEMI | 1 | serious | 4 | 4 | 0 | 0 | 4 |
| Scombroid food poisoning | 1 | serious | 2 | 2 | 0 | 0 | 2 |
| Spontaneous pneumothorax | 1 | serious | 3 | 3 | 0 | 0 | 3 |
| Stable angina | 1 | serious | 2 | 2 | 0 | 0 | 2 |
| Unstable angina | 1 | serious | 3 | 3 | 0 | 0 | 3 |
| Acute COPD exacerbation / infection | 2 | serious | 6 | 6 | 0 | 0 | 6 |
| Atrial fibrillation | 2 | serious | 8 | 8 | 0 | 0 | 8 |
| Bronchiectasis | 2 | serious | 2 | 2 | 0 | 0 | 2 |
| GERD | 2 | serious | 4 | 4 | 0 | 0 | 4 |
| Myasthenia gravis | 2 | serious | 8 | 8 | 0 | 0 | 8 |
| Spontaneous rib fracture | 2 | serious | 4 | 4 | 0 | 0 | 4 |
| Tuberculosis | 2 | serious | 1 | 1 | 0 | 0 | 1 |
| Anemia | 3 | serious | 17 | 17 | 0 | 0 | 17 |
| Panic attack | 3 | serious | 1 | 1 | 0 | 0 | 1 |
| Pericarditis | 3 | serious | 5 | 5 | 0 | 0 | 5 |
| SLE | 3 | serious | 2 | 2 | 0 | 0 | 2 |
| Sarcoidosis | 3 | serious | 4 | 4 | 0 | 0 | 4 |
| Viral pharyngitis | 3 | serious | 8 | 8 | 0 | 0 | 8 |

## Kept cases where the reference disagrees with the class (7)

| Idx | Case | Truth (tier) | Stratum | Rules | Class | Reference (conf.) | Source | Rationale |
|---|---|---|---|---|---|---|---|---|
| 13 | ddxplus_43608 | PSVT (1) | serious_tier1 | - | SERIOUS | ROUTINE (3) | adjudicated | Palpitations with light-headedness on caffeine, energy drinks and decongestants in a 19-year-old: the ESC 2018 syncope initial evaluation (history, orthostatic blood pressure, 12-lead ECG) is an office check, and no chest pain or high-risk marker mandates troponin or imaging. |
| 23 | ddxplus_7249 | PSVT (1) | serious_tier1 | - | SERIOUS | ROUTINE (3) | adjudicated | Headache, palpitations and light-headedness on regular stimulants plus decongestants is a sympathomimetic pattern closed by blood pressure and an office ECG; onset speed 3/10 is gradual so the Ottawa SAH rule is not triggered, and with no chest pain NICE CG95 1.2.1.3 does not apply. |
| 24 | ddxplus_35803 | PSVT (1) | serious_tier1 | - | SERIOUS | ROUTINE (3) | adjudicated | Same sympathomimetic pattern as index 23 at age 50: blood pressure and an office ECG (higher yield for atrial fibrillation at this age) are the checks; gradual onset (3/10) and no chest pain mean no rule mandates imaging or troponin. |
| 31 | ddxplus_61816 | PSVT (1) | serious_tier1 | - | SERIOUS | ROUTINE (3) | adjudicated | Palpitations, light-headedness and breathlessness in a 31-year-old on four sympathomimetic exposures: the ESC 2018 initial syncope evaluation is an office ECG with vitals, PERC completes with pulse and saturation at age under 50, and no chest pain triggers NICE CG95. |
| 157 | ddxplus_28541 | Localized edema (3) | benign | - | BENIGN | ESCALATE (4) | agreed | A 23-year-old man with a documented prior DVT now has rapid-onset painful swelling that includes the left thigh, which scores at least 2 on the two-level Wells score (previous DVT 1, whole-leg or thigh swelling 1, localised tenderness likely 1) and makes DVT 'likely'. |
| 173 | ddxplus_82870 | Localized edema (3) | benign | - | BENIGN | ESCALATE (4) | adjudicated | Painful (7/10) swelling reaching both thighs in a man with a previously documented DVT scores at least 2 on the NICE NG158 table 1 Wells score (prior DVT, entire leg swollen, tenderness), which mandates a proximal leg-vein ultrasound within 4 hours; bilateral oedema with weight gain also needs heart-failure work-up that the localised-oedema label omits. |
| 191 | ddxplus_40108 | Sarcoidosis (3) | benign | - | BENIGN | ESCALATE (3) | agreed | The joint and shoulder pain fit lupus, but significant shortness of breath in SLE is never explained by the arthritis and the serious causes are common in this disease: pericardial effusion, pleural effusion, pulmonary embolism (antiphospholipid antibodies), pneumonitis and infection. |

## Kept cases left UNCERTAIN (2)

| Idx | Case | Truth (tier) | Stratum | Rules | Class | Rationale |
|---|---|---|---|---|---|---|
| 143 | ddxplus_3445 | Viral pharyngitis (3) | benign | - | BENIGN | Unilateral 9/10 tonsillar pain with submandibular and laryngeal-level pain in a 56-year-old without coryza is the quinsy or supraglottitis pattern, but the deciding features (trismus, drooling, dysphagia, pain out of proportion to the pharynx) reach the intake only as a bare pain score; NICE NG84 1.1.13 referral turns on the examination. |
| 197 | ddxplus_73470 | URTI (3) | benign | - | BENIGN | NICE NG79 1.1.9 lists severe frontal headache verbatim as a hospital-referral sign of intracranial complication, but the intake carries it only as a bare 8/10 score inside a slow-onset (0/10) coryzal febrile illness with no frontal swelling, meningism or neurology; the score alone cannot separate sinus pain from a complication. |

## Verdict

| Criterion | Result |
|---|---|
| 1. Class agreement on decided kept cases | PASS: 96.0% [92.0, 98.1] |
| 2. Per-stratum agreement | PASS: serious_tier1 92.0%; serious_upgrade_or_flag 100.0%; benign 94.6% |
| 3. Per-rule agreement (>= 5 cases) | PASS: no rule flagged |
| 4. Reviewer kappa (three-way) | FAIL: 0.552 |
| 4 (reported). Kappa on cases both reviewers decided; UNCERTAIN rate | 0.918 on 187; Fable 18.4%, Astra 10.0% |
| 5. EXCLUDED stratum (reported) | ESCALATE 65.3%, ROUTINE 27.8%, UNCERTAIN 6.9%, split 36.1% |

Section 7 says failing criterion 1 or 2 means the slice is not certified; a rule flagged under criterion 3 is demoted to EXCLUDE. This document reports; it changes no rule.
