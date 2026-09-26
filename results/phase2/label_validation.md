# Phase 2 label validation of the frozen case-selection rules

Freeze commit 45a7599; draw seed 20261003 (docs/v0.3-case-selection-rules.md section 7). Script: `scripts/analysis/v03_phase2_validation.py`; cases from `scripts/analysis/v03_phase2_cases.py`. Reference: two blind Fable reviews with verified citations (A on cases 0-124, B on 125-249), a blind Astra review from knowledge (all 250), and a blind adjudication of every disagreement and every UNCERTAIN (69 cases); the 181 agreed cases adopt the agreed decision, and 10 of them were spot-checked (seed 20261003). No reviewer saw the truth, the stratum, the rules or any model output.

## Summary

1. **Criterion 1 passes: the rules' class agrees with the blind reference on 181 of 196 decided kept cases (92.3%, Wilson lower bound 87.8%).** The layer-a and UPGRADE rules hold on fresh cases: every rule with 5 or more cases agrees on 100% of them (criterion 3 passes, no rule flagged), and the BENIGN stratum is 94.6% ROUTINE.
2. **Criterion 2 fails on one stratum: SERIOUS by a DXA-only target is 60.0% ESCALATE against the 92% target.** After the red-herring rule the stratum is Panic attack 19, Bronchitis 1 with targets PSVT 19, Bronchospasm / acute asthma exacerbation 1; the reference keeps 8 routine, because a panic attack in a young adult with chronic anxiety and no cardiac risk factor is closed by an office ECG and vitals (HEART age 0; NICE CG113). Section 7 says a failing stratum's rules go back to EXCLUDE and Phase 2 repeats on seed 20261004; this document changes nothing.
3. **The tier-1 stratum sits exactly on its 92% target (92.0%).** The reference keeps routine 4 PSVT cases: palpitations with light-headedness on caffeine, energy drinks, stimulants or decongestants, without chest pain or syncope, which the audit also kept routine (case 36). The condition table lists PSVT as INCLUDE; this is the one tier-1 condition the reference does not treat as an escalation by default.
4. **Criterion 4 fails: Cohen's kappa is 0.552 (raw agreement 75.6%), below 0.6.** The disagreement is almost entirely one reviewer's UNCERTAIN against the other's decision (Fable 46 UNCERTAIN, Astra 25); only 6 of 250 cases are outright ESCALATE/ROUTINE splits. The adjudication decided 62 of the 69 queued cases; 7 stay UNCERTAIN.
5. **The EXCLUDED stratum is 67.3% ESCALATE, 23.1% ROUTINE, 9.6% UNCERTAIN, and the reviewers split on 34.6%.** Exclusion rules whose cases are ESCALATE at confidence 4 or more in at least 80% of draws (candidate PATCH for a later phase): K30 (4 of 5). Among BENIGN cases the reference escalates 3 (listed under the disagreements below): Localized edema at index 157, Localized edema at index 173, Sarcoidosis at index 191.

## Reviewer agreement (criterion 4: kappa >= 0.6)

| Fable \ Astra | ESCALATE | ROUTINE | UNCERTAIN |
|---|---|---|---|
| ESCALATE | 134 | 0 | 1 |
| ROUTINE | 6 | 47 | 16 |
| UNCERTAIN | 38 | 0 | 8 |

Cohen's kappa 0.552, raw agreement 75.6% (189 of 250). **FAIL.**

## The reference by stratum

| Stratum | n | ESCALATE | ROUTINE | UNCERTAIN | class under the key |
|---|---|---|---|---|---|
| serious_tier1 | 50 | 46 | 4 | 0 | serious 50 |
| serious_upgrade_or_flag | 70 | 70 | 0 | 0 | serious 70 |
| serious_dxa_only | 20 | 12 | 8 | 0 | serious 20 |
| benign | 60 | 3 | 55 | 2 | benign 58, excluded 2 |
| excluded | 50 | 35 | 10 | 5 | excluded 50 |

## Criterion 1: class agreement on decided kept cases (>= 90%, Wilson lower bound >= 85%)

181 of 196 decided kept cases agree: 92.3% [87.8, 95.3]; 2 kept cases are UNCERTAIN. **PASS.**

| Class | ESCALATE | ROUTINE | UNCERTAIN | total |
|---|---|---|---|---|
| SERIOUS | 128 | 12 | 0 | 140 |
| BENIGN | 3 | 53 | 2 | 58 |
| EXCLUDED | 35 | 12 | 5 | 52 |

## Criterion 2: BENIGN >= 85% ROUTINE; each SERIOUS stratum >= 92% ESCALATE (among decided cases)

| Stratum | decided | agree | rate [Wilson 95%] | target | result |
|---|---|---|---|---|---|
| serious_tier1 | 50 | 46 | 92.0% [81.2, 96.8] | >= 92% | PASS |
| serious_upgrade_or_flag | 70 | 70 | 100.0% [94.8, 100.0] | >= 92% | PASS |
| serious_dxa_only | 20 | 12 | 60.0% [38.7, 78.1] | >= 92% | FAIL |
| benign | 56 | 53 | 94.6% [85.4, 98.2] | >= 85% | PASS |

Two BENIGN candidates (ddxplus_23603, ddxplus_68301) carry an R5-only target under the key and fall to EXCLUDED under X9; they are counted in the EXCLUDED stratum below, not here.

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

52 cases: ESCALATE 35 (67.3%), ROUTINE 12 (23.1%), UNCERTAIN 5 (9.6%); the two reviewers split on 18 (34.6%).

| Exclusion rule | cases | ESCALATE | ROUTINE | UNCERTAIN | split | ESCALATE at confidence >= 4 | candidate PATCH (>= 80%) |
|---|---|---|---|---|---|---|---|
| K27 | 3 | 3 | 0 | 0 | 1 | 1 (33.3%) | no |
| K30 | 5 | 5 | 0 | 0 | 0 | 4 (80.0%) | yes |
| K32 | 8 | 4 | 4 | 0 | 5 | 2 (25.0%) | no |
| X10 | 31 | 21 | 7 | 3 | 11 | 10 (32.3%) | no |
| X6 | 9 | 4 | 3 | 2 | 2 | 2 (22.2%) | no |
| X7 | 5 | 5 | 0 | 0 | 3 | 0 (0.0%) | no |
| X8 | 2 | 2 | 0 | 0 | 0 | 1 (50.0%) | no |
| X9 | 6 | 4 | 2 | 0 | 2 | 3 (50.0%) | no |

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
| Bronchitis | 3 | serious | 1 | 0 | 1 | 0 | 0 |
| Panic attack | 3 | serious | 20 | 13 | 7 | 0 | 13 |
| Pericarditis | 3 | serious | 5 | 5 | 0 | 0 | 5 |
| SLE | 3 | serious | 2 | 2 | 0 | 0 | 2 |
| Sarcoidosis | 3 | serious | 4 | 4 | 0 | 0 | 4 |
| Viral pharyngitis | 3 | serious | 8 | 8 | 0 | 0 | 8 |

## Kept cases where the reference disagrees with the class (15)

| Idx | Case | Truth (tier) | Stratum | Rules | Class | Reference (conf.) | Source | Rationale |
|---|---|---|---|---|---|---|---|---|
| 13 | ddxplus_43608 | PSVT (1) | serious_tier1 | - | SERIOUS | ROUTINE (3) | adjudicated | Palpitations with light-headedness on caffeine, energy drinks and decongestants in a 19-year-old: the ESC 2018 syncope initial evaluation (history, orthostatic blood pressure, 12-lead ECG) is an office check, and no chest pain or high-risk marker mandates troponin or imaging. |
| 23 | ddxplus_7249 | PSVT (1) | serious_tier1 | - | SERIOUS | ROUTINE (3) | adjudicated | Headache, palpitations and light-headedness on regular stimulants plus decongestants is a sympathomimetic pattern closed by blood pressure and an office ECG; onset speed 3/10 is gradual so the Ottawa SAH rule is not triggered, and with no chest pain NICE CG95 1.2.1.3 does not apply. |
| 24 | ddxplus_35803 | PSVT (1) | serious_tier1 | - | SERIOUS | ROUTINE (3) | adjudicated | Same sympathomimetic pattern as index 23 at age 50: blood pressure and an office ECG (higher yield for atrial fibrillation at this age) are the checks; gradual onset (3/10) and no chest pain mean no rule mandates imaging or troponin. |
| 31 | ddxplus_61816 | PSVT (1) | serious_tier1 | - | SERIOUS | ROUTINE (3) | adjudicated | Palpitations, light-headedness and breathlessness in a 31-year-old on four sympathomimetic exposures: the ESC 2018 initial syncope evaluation is an office ECG with vitals, PERC completes with pulse and saturation at age under 50, and no chest pain triggers NICE CG95. |
| 125 | ddxplus_30054 | Panic attack (3) | serious_dxa_only | - | SERIOUS | ROUTINE (4) | adjudicated | Fear of dying, choking, palpitations, derealisation and peri-oral tingling in a 27-year-old with chronic anxiety and no risk factors: HEART age scores 0, NICE CG113 1.3.6 asks only for the minimum investigations, and an office ECG with vitals closes the visit. |
| 126 | ddxplus_12248 | Panic attack (3) | serious_dxa_only | - | SERIOUS | ROUTINE (4) | adjudicated | Mild non-radiating pain with palpitations and light-headedness in a 28-year-old with chronic anxiety: young age, female sex and atypical pain predict panic (Huffman 2003), the ESC 2018 initial evaluation is an office ECG with vitals, and no feature mandates troponin or imaging. |
| 128 | ddxplus_18411 | Panic attack (3) | serious_dxa_only | - | SERIOUS | ROUTINE (3) | adjudicated | Sudden diffuse abdominal pain with the hyperventilation signature in a 27-year-old with chronic anxiety: abdominal examination, vitals and an office pregnancy test (NICE NG126 1.4.5-1.4.7 routes only a positive test with tenderness to an early pregnancy service) close the visit; bilateral, diffuse pain does not fit torsion. |
| 129 | ddxplus_87310 | Panic attack (3) | serious_dxa_only | - | SERIOUS | ROUTINE (3) | adjudicated | Panic cluster (fear of dying, tingling, derealisation) in a 39-year-old with chronic anxiety and no cardiac risk factors: HEART age scores 0, and NICE CG113 1.3.6 minimum investigations (office ECG, vitals, abdominal examination) are the closing checks; near-syncope is why confidence is 3. |
| 135 | ddxplus_45695 | Panic attack (3) | serious_dxa_only | - | SERIOUS | ROUTINE (3) | agreed | A 30-year-old woman with chronic anxiety, depression and fibromyalgia has the hyperventilation signature with derealisation and no syncope, dyspnoea or exertional features, so panic is the leading diagnosis and her HEART age score is 0. |
| 137 | ddxplus_4519 | Panic attack (3) | serious_dxa_only | - | SERIOUS | ROUTINE (3) | adjudicated | Textbook panic attack in a 32-year-old with chronic anxiety and depression: HEART age 0, NICE CG113 1.3.6 minimum investigations (office ECG, glucose, abdominal examination) close it; excess alcohol adds a withdrawal question but no mandated test. |
| 138 | ddxplus_77892 | Bronchitis (3) | serious_dxa_only | - | SERIOUS | ROUTINE (4) | agreed | A 31-year-old smoker with COPD (unusually young, so asthma or an alternative diagnosis is worth revisiting) has increased purulent sputum, cough and wheeze without reported dyspnoea, fever or haemoptysis, which is a mild infective exacerbation. |
| 139 | ddxplus_60530 | Panic attack (3) | serious_dxa_only | - | SERIOUS | ROUTINE (3) | adjudicated | Abrupt panic attack with sweating and tingling in an 18-year-old with chronic anxiety: pain is cramping and spreads to the abdomen rather than pleuritic and unilateral, so pneumothorax is excluded by chest examination and saturation (ordinary checks), and HEART age scores 0. |
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
| 1. Class agreement on decided kept cases | PASS: 92.3% [87.8, 95.3] |
| 2. Per-stratum agreement | FAIL: serious_tier1 92.0%; serious_upgrade_or_flag 100.0%; serious_dxa_only 60.0%; benign 94.6% |
| 3. Per-rule agreement (>= 5 cases) | PASS: no rule flagged |
| 4. Reviewer kappa | FAIL: 0.552 |
| 5. EXCLUDED stratum (reported) | ESCALATE 67.3%, ROUTINE 23.1%, UNCERTAIN 9.6%, split 34.6% |

Section 7 says failing criterion 1 or 2 means the slice is not certified and Phase 2 repeats on seed 20261004; a rule flagged under criterion 3 is demoted to EXCLUDE. This document reports; it changes no rule.
