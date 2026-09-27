# Phase 2 and Phase 2b pooled label validation (secondary)

SECONDARY, POOLED: the Phase 2 reference (results/phase2/post_hoc_x11/reference_adjudicated.jsonl, classes recomputed post hoc under rule X11) and the Phase 2b reference (results/phase2b/reference_adjudicated.jsonl) as one set of 500 cases. The pre-registered result for each phase is its own label_validation.md; this file pools them for precision, not for the verdict. Index prefix P2: or P2b: names the phase.

## Summary

1. **Criterion 1 passes: the rules' class agrees with the blind reference on 365 of 379 decided kept cases (96.3%, Wilson lower bound 93.9%).**
2. **Criterion 2 passes in every stratum: serious_tier1 95.4%; serious_upgrade_or_flag 98.7%; benign 95.5%.** The lowest stratum is serious_tier1 at 95.4%. SERIOUS cases the reference keeps routine: 5 PSVT (serious_tier1), 2 Anemia (serious_upgrade_or_flag). BENIGN cases it escalates: Localized edema at index 157, Localized edema at index 173, Sarcoidosis at index 191, Sarcoidosis at index 166, Sarcoidosis at index 176, Sarcoidosis at index 220, Panic attack at index 248.
3. **Criterion 3 passes: no rule with 5 or more fresh cases falls below 80% agreement.**
4. **Criterion 4 passes: three-way kappa 0.620 (raw agreement 81.4%).** On the 411 cases both reviewers decided, kappa is 0.901 (raw agreement 96.4%); Fable said UNCERTAIN on 12.0% of cases and Astra on 8.0%. The adjudication decided 96 of the 104 queued cases; 8 stay UNCERTAIN.
5. **The EXCLUDED stratum is 72.0% ESCALATE, 23.7% ROUTINE, 4.2% UNCERTAIN, and the reviewers split on 28.0%.** Exclusion rules whose cases are ESCALATE at confidence 4 or more in at least 80% of draws (candidate PATCH for a later phase): K30 (11 of 12).

## Reviewer agreement (criterion 4: kappa >= 0.6 on the three-way decision)

| Fable \ Astra | ESCALATE | ROUTINE | UNCERTAIN |
|---|---|---|---|
| ESCALATE | 304 | 0 | 4 |
| ROUTINE | 15 | 92 | 25 |
| UNCERTAIN | 48 | 1 | 11 |

Cohen's kappa 0.620, raw agreement 81.4% (407 of 500). **PASS.**

Reported measures (decision 14):

- kappa on the 411 cases both reviewers decided (ESCALATE/ROUTINE only): 0.901, raw agreement 96.4%;
- UNCERTAIN rate: Fable 60 of 500 (12.0%), Astra 40 of 500 (8.0%).

## The reference by stratum

| Stratum | n | ESCALATE | ROUTINE | UNCERTAIN | class under the key |
|---|---|---|---|---|---|
| serious_tier1 | 110 | 104 | 5 | 1 | serious 110 |
| serious_upgrade_or_flag | 150 | 148 | 2 | 0 | serious 150 |
| serious_dxa_only | 20 | 12 | 8 | 0 | excluded 20 |
| benign | 120 | 6 | 112 | 2 | benign 113, excluded 7 |
| excluded | 100 | 74 | 21 | 5 | benign 9, excluded 91 |

## Criterion 1: class agreement on decided kept cases (>= 90%, Wilson lower bound >= 85%)

365 of 379 decided kept cases agree: 96.3% [93.9, 97.8]; 3 kept cases are UNCERTAIN. **PASS.**

| Class | ESCALATE | ROUTINE | UNCERTAIN | total |
|---|---|---|---|---|
| SERIOUS | 252 | 7 | 1 | 260 |
| BENIGN | 7 | 113 | 2 | 122 |
| EXCLUDED | 85 | 28 | 5 | 118 |

## Criterion 2: BENIGN >= 85% ROUTINE; each SERIOUS stratum >= 92% ESCALATE (among decided cases)

| Stratum | decided | agree | rate [Wilson 95%] | target | result |
|---|---|---|---|---|---|
| serious_tier1 | 109 | 104 | 95.4% [89.7, 98.0] | >= 92% | PASS |
| serious_upgrade_or_flag | 150 | 148 | 98.7% [95.3, 99.6] | >= 92% | PASS |
| benign | 111 | 106 | 95.5% [89.9, 98.1] | >= 85% | PASS |
| serious_dxa_only | 0 | 0 | n/a | | not judged: no kept case (rule X11 excludes the stratum) |

27 drawn kept case(s) fall to EXCLUDED under the key (ddxplus_53176 by X11, ddxplus_27844 by X11, ddxplus_103631 by X11, ddxplus_115592 by X11, ddxplus_49506 by X11, ddxplus_30054 by X11, ddxplus_12248 by X11, ddxplus_35900 by X11, ddxplus_18411 by X11, ddxplus_87310 by X11, ddxplus_5213 by X11, ddxplus_77420 by X11, ddxplus_117822 by X11, ddxplus_18367 by X11, ddxplus_93100 by X11, ddxplus_45695 by X11, ddxplus_81398 by X11, ddxplus_4519 by X11, ddxplus_77892 by X11, ddxplus_60530 by X11, ddxplus_23603 by X9, ddxplus_68301 by X9, ddxplus_38189 by X9, ddxplus_62677 by X9, ddxplus_18166 by X9, ddxplus_4215 by X9, ddxplus_46931 by X9); they are counted in the EXCLUDED stratum below, not here.

## Criterion 3: each rule with >= 5 fresh cases it decided, >= 80% agreement

| Rule | cases | decided | agree | rate | ESCALATE / ROUTINE / UNCERTAIN | result |
|---|---|---|---|---|---|---|
| P1 | 25 | 25 | 25 | 100.0% | 25 / 0 / 0 | PASS |
| P10 | 8 | 8 | 8 | 100.0% | 8 / 0 / 0 | PASS |
| P12 | 16 | 16 | 16 | 100.0% | 16 / 0 / 0 | PASS |
| P2 | 32 | 32 | 32 | 100.0% | 32 / 0 / 0 | PASS |
| P3 | 22 | 22 | 21 | 95.5% | 21 / 1 / 0 | PASS |
| P4 | 8 | 8 | 8 | 100.0% | 8 / 0 / 0 | PASS |
| P5 | 17 | 17 | 15 | 88.2% | 15 / 2 / 0 | PASS |
| P6 | 4 | 4 | 4 | 100.0% | 4 / 0 / 0 | reported only (< 5 cases) |
| P7 | 16 | 16 | 16 | 100.0% | 16 / 0 / 0 | PASS |
| P9 | 16 | 16 | 16 | 100.0% | 16 / 0 / 0 | PASS |
| K22 | 5 | 5 | 5 | 100.0% | 5 / 0 / 0 | PASS |
| K23 | 8 | 8 | 8 | 100.0% | 8 / 0 / 0 | PASS |
| K24 | 16 | 16 | 16 | 100.0% | 16 / 0 / 0 | PASS |
| K33 | 16 | 16 | 16 | 100.0% | 16 / 0 / 0 | PASS |
| K42 | 13 | 13 | 13 | 100.0% | 13 / 0 / 0 | PASS |

No rule with 5 or more fresh cases falls below 80% agreement.

## Criterion 5 (reported only): the EXCLUDED stratum

118 cases: ESCALATE 85 (72.0%), ROUTINE 28 (23.7%), UNCERTAIN 5 (4.2%); the two reviewers split on 33 (28.0%).

| Exclusion rule | cases | ESCALATE | ROUTINE | UNCERTAIN | split | ESCALATE at confidence >= 4 | candidate PATCH (>= 80%) |
|---|---|---|---|---|---|---|---|
| K27 | 5 | 5 | 0 | 0 | 1 | 2 (40.0%) | no |
| K30 | 12 | 12 | 0 | 0 | 0 | 11 (91.7%) | yes |
| K32 | 10 | 5 | 5 | 0 | 6 | 2 (20.0%) | no |
| X10 | 59 | 47 | 9 | 3 | 13 | 29 (49.2%) | no |
| X11 | 21 | 12 | 9 | 0 | 8 | 4 (19.0%) | no |
| X6 | 15 | 9 | 4 | 2 | 2 | 3 (20.0%) | no |
| X7 | 8 | 8 | 0 | 0 | 3 | 0 (0.0%) | no |
| X8 | 3 | 3 | 0 | 0 | 0 | 2 (66.7%) | no |
| X9 | 15 | 9 | 6 | 0 | 7 | 5 (33.3%) | no |

## Kept cases by condition

| Condition | tier | class | n | ESCALATE | ROUTINE | UNCERTAIN | agree |
|---|---|---|---|---|---|---|---|
| Acute laryngitis | 3 | benign | 12 | 0 | 12 | 0 | 12 |
| Acute otitis media | 3 | benign | 25 | 0 | 25 | 0 | 25 |
| Acute rhinosinusitis | 3 | benign | 11 | 0 | 11 | 0 | 11 |
| Allergic sinusitis | 3 | benign | 11 | 0 | 11 | 0 | 11 |
| Bronchitis | 3 | benign | 2 | 0 | 2 | 0 | 2 |
| Chronic rhinosinusitis | 3 | benign | 13 | 0 | 13 | 0 | 13 |
| Localized edema | 3 | benign | 2 | 2 | 0 | 0 | 0 |
| Panic attack | 3 | benign | 4 | 1 | 3 | 0 | 3 |
| SLE | 3 | benign | 1 | 0 | 1 | 0 | 1 |
| Sarcoidosis | 3 | benign | 4 | 4 | 0 | 0 | 0 |
| URTI | 3 | benign | 23 | 0 | 22 | 1 | 22 |
| Viral pharyngitis | 3 | benign | 12 | 0 | 11 | 1 | 11 |
| Whooping cough | 3 | benign | 2 | 0 | 2 | 0 | 2 |
| Acute dystonic reactions | 1 | serious | 5 | 5 | 0 | 0 | 5 |
| Acute pulmonary edema | 1 | serious | 7 | 7 | 0 | 0 | 7 |
| Anaphylaxis | 1 | serious | 4 | 4 | 0 | 0 | 4 |
| Boerhaave | 1 | serious | 5 | 5 | 0 | 0 | 5 |
| Bronchospasm / acute asthma exacerbation | 1 | serious | 4 | 4 | 0 | 0 | 4 |
| Epiglottitis | 1 | serious | 7 | 7 | 0 | 0 | 7 |
| Guillain-Barré syndrome | 1 | serious | 7 | 6 | 0 | 1 | 6 |
| Larygospasm | 1 | serious | 7 | 7 | 0 | 0 | 7 |
| Myocarditis | 1 | serious | 2 | 2 | 0 | 0 | 2 |
| PSVT | 1 | serious | 7 | 2 | 5 | 0 | 2 |
| Pancreatic neoplasm | 1 | serious | 8 | 8 | 0 | 0 | 8 |
| Pneumonia | 1 | serious | 6 | 6 | 0 | 0 | 6 |
| Possible NSTEMI / STEMI | 1 | serious | 7 | 7 | 0 | 0 | 7 |
| Pulmonary embolism | 1 | serious | 4 | 4 | 0 | 0 | 4 |
| Pulmonary neoplasm | 1 | serious | 2 | 2 | 0 | 0 | 2 |
| Scombroid food poisoning | 1 | serious | 7 | 7 | 0 | 0 | 7 |
| Spontaneous pneumothorax | 1 | serious | 6 | 6 | 0 | 0 | 6 |
| Stable angina | 1 | serious | 8 | 8 | 0 | 0 | 8 |
| Unstable angina | 1 | serious | 7 | 7 | 0 | 0 | 7 |
| Acute COPD exacerbation / infection | 2 | serious | 8 | 8 | 0 | 0 | 8 |
| Atrial fibrillation | 2 | serious | 16 | 16 | 0 | 0 | 16 |
| Bronchiectasis | 2 | serious | 6 | 6 | 0 | 0 | 6 |
| GERD | 2 | serious | 8 | 8 | 0 | 0 | 8 |
| Myasthenia gravis | 2 | serious | 16 | 16 | 0 | 0 | 16 |
| Spontaneous rib fracture | 2 | serious | 5 | 5 | 0 | 0 | 5 |
| Tuberculosis | 2 | serious | 5 | 5 | 0 | 0 | 5 |
| Anemia | 3 | serious | 40 | 38 | 2 | 0 | 38 |
| Panic attack | 3 | serious | 4 | 4 | 0 | 0 | 4 |
| Pericarditis | 3 | serious | 13 | 13 | 0 | 0 | 13 |
| SLE | 3 | serious | 4 | 4 | 0 | 0 | 4 |
| Sarcoidosis | 3 | serious | 8 | 8 | 0 | 0 | 8 |
| Viral pharyngitis | 3 | serious | 17 | 17 | 0 | 0 | 17 |

## Kept cases where the reference disagrees with the class (14)

| Idx | Case | Truth (tier) | Stratum | Rules | Class | Reference (conf.) | Source | Rationale |
|---|---|---|---|---|---|---|---|---|
| P2:13 | ddxplus_43608 | PSVT (1) | serious_tier1 | - | SERIOUS | ROUTINE (3) | adjudicated | Palpitations with light-headedness on caffeine, energy drinks and decongestants in a 19-year-old: the ESC 2018 syncope initial evaluation (history, orthostatic blood pressure, 12-lead ECG) is an office check, and no chest pain or high-risk marker mandates troponin or imaging. |
| P2:23 | ddxplus_7249 | PSVT (1) | serious_tier1 | - | SERIOUS | ROUTINE (3) | adjudicated | Headache, palpitations and light-headedness on regular stimulants plus decongestants is a sympathomimetic pattern closed by blood pressure and an office ECG; onset speed 3/10 is gradual so the Ottawa SAH rule is not triggered, and with no chest pain NICE CG95 1.2.1.3 does not apply. |
| P2:24 | ddxplus_35803 | PSVT (1) | serious_tier1 | - | SERIOUS | ROUTINE (3) | adjudicated | Same sympathomimetic pattern as index 23 at age 50: blood pressure and an office ECG (higher yield for atrial fibrillation at this age) are the checks; gradual onset (3/10) and no chest pain mean no rule mandates imaging or troponin. |
| P2:31 | ddxplus_61816 | PSVT (1) | serious_tier1 | - | SERIOUS | ROUTINE (3) | adjudicated | Palpitations, light-headedness and breathlessness in a 31-year-old on four sympathomimetic exposures: the ESC 2018 initial syncope evaluation is an office ECG with vitals, PERC completes with pulse and saturation at age under 50, and no chest pain triggers NICE CG95. |
| P2:157 | ddxplus_28541 | Localized edema (3) | benign | - | BENIGN | ESCALATE (4) | agreed | A 23-year-old man with a documented prior DVT now has rapid-onset painful swelling that includes the left thigh, which scores at least 2 on the two-level Wells score (previous DVT 1, whole-leg or thigh swelling 1, localised tenderness likely 1) and makes DVT 'likely'. |
| P2:173 | ddxplus_82870 | Localized edema (3) | benign | - | BENIGN | ESCALATE (4) | adjudicated | Painful (7/10) swelling reaching both thighs in a man with a previously documented DVT scores at least 2 on the NICE NG158 table 1 Wells score (prior DVT, entire leg swollen, tenderness), which mandates a proximal leg-vein ultrasound within 4 hours; bilateral oedema with weight gain also needs heart-failure work-up that the localised-oedema label omits. |
| P2:191 | ddxplus_40108 | Sarcoidosis (3) | benign | - | BENIGN | ESCALATE (3) | agreed | The joint and shoulder pain fit lupus, but significant shortness of breath in SLE is never explained by the arthritis and the serious causes are common in this disease: pericardial effusion, pleural effusion, pulmonary embolism (antiphospholipid antibodies), pneumonitis and infection. |
| P2b:17 | ddxplus_65159 | PSVT (1) | serious_tier1 | - | SERIOUS | ROUTINE (4) | adjudicated | Palpitations with near-fainting, no syncope and a stimulant history are closed by an office 12-lead ECG and vitals, the initial evaluation the ESC 2018 syncope guideline names. The anaemia label is a benign mislabel that omits no dangerous treatment. |
| P2b:77 | ddxplus_64600 | Anemia (3) | serious_upgrade_or_flag | P5 | SERIOUS | ROUTINE (3) | adjudicated | The anaemia label fits heavy periods, pallor, poor diet and low weight, and no upper-GI bleeding is reported, so a blood count, vitals and a urine pregnancy test at the visit close it. BSG 2021 reserves transfusion for severe symptomatic or circulatory compromise, which the haemoglobin and vitals decide. |
| P2b:78 | ddxplus_5302 | Anemia (3) | serious_upgrade_or_flag | P3|P5 | SERIOUS | ROUTINE (3) | adjudicated | Fresh red rectal blood under 50 with heavy periods is lower-GI and menstrual loss; NG12 1.3.1 needs abdominal pain or weight loss to trigger FIT under 50, and the BSG 2019 lower-GI guideline decides place of care on vitals and haemoglobin, which the visit supplies. Melaena would score on Blatchford and escalate; fresh blood does not. |
| P2b:166 | ddxplus_78338 | Sarcoidosis (3) | benign | - | BENIGN | ESCALATE (4) | adjudicated | Unexplained lymphadenopathy at 82 meets NICE NG12 1.10.6 and 1.10.8 for a suspected cancer pathway referral, and first-time vaginal discharge at 55 or over meets NG12 1.5.13 for an urgent endometrial ultrasound. A sarcoidosis label explains neither without tissue and imaging. |
| P2b:176 | ddxplus_111673 | Sarcoidosis (3) | benign | - | BENIGN | ESCALATE (3) | adjudicated | Unexplained lymphadenopathy at 60 meets NICE NG12 1.10.6 and 1.10.8 for a lymphoma pathway referral, and a red eye under a sarcoidosis label needs a slit-lamp examination for uveitis, which the ATS 2020 guideline asks for even without eye symptoms. Sarcoidosis is a diagnosis of exclusion that itself mandates a chest film and tissue. |
| P2b:220 | ddxplus_102395 | Sarcoidosis (3) | excluded | - | BENIGN | ESCALATE (4) | spot-checked | Sarcoidosis is not a working diagnosis to hold without imaging, and this patient has red eyes and significant breathlessness. |
| P2b:248 | ddxplus_4193 | Panic attack (3) | excluded | - | BENIGN | ESCALATE (4) | agreed | Chest pain with sweating, choking and palpitations in a 47-year-old woman is how ACS presents in women, and panic attack is the diagnosis left after an ECG and troponin, not before. |

## Kept cases left UNCERTAIN (3)

| Idx | Case | Truth (tier) | Stratum | Rules | Class | Rationale |
|---|---|---|---|---|---|---|
| P2:143 | ddxplus_3445 | Viral pharyngitis (3) | benign | - | BENIGN | Unilateral 9/10 tonsillar pain with submandibular and laryngeal-level pain in a 56-year-old without coryza is the quinsy or supraglottitis pattern, but the deciding features (trismus, drooling, dysphagia, pain out of proportion to the pharynx) reach the intake only as a bare pain score; NICE NG84 1.1.13 referral turns on the examination. |
| P2:197 | ddxplus_73470 | URTI (3) | benign | - | BENIGN | NICE NG79 1.1.9 lists severe frontal headache verbatim as a hospital-referral sign of intracranial complication, but the intake carries it only as a bare 8/10 score inside a slow-onset (0/10) coryzal febrile illness with no frontal swelling, meningism or neurology; the score alone cannot separate sinus pain from a complication. |
| P2b:41 | ddxplus_26708 | Guillain-Barré syndrome (1) | serious_tier1 | - | SERIOUS | NICE NG127 1.7.2 and 1.10.2 refer symmetrical limb weakness with numbness immediately only when it is rapidly progressive, and the intake carries the weakness as a bare yes with no onset or progression. The panic label rests on no anxiety history, so the evidence is balanced and reflexes and power at the visit decide. |

## Verdict

| Criterion | Result |
|---|---|
| 1. Class agreement on decided kept cases | PASS: 96.3% [93.9, 97.8] |
| 2. Per-stratum agreement | PASS: serious_tier1 95.4%; serious_upgrade_or_flag 98.7%; benign 95.5% |
| 3. Per-rule agreement (>= 5 cases) | PASS: no rule flagged |
| 4. Reviewer kappa (three-way) | PASS: 0.620 |
| 4 (reported). Kappa on cases both reviewers decided; UNCERTAIN rate | 0.901 on 411; Fable 12.0%, Astra 8.0% |
| 5. EXCLUDED stratum (reported) | ESCALATE 72.0%, ROUTINE 23.7%, UNCERTAIN 4.2%, split 28.0% |

Section 7 says failing criterion 1 or 2 means the slice is not certified; a rule flagged under criterion 3 is demoted to EXCLUDE. This document reports; it changes no rule.
