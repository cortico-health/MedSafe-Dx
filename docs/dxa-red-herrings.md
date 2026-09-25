# Red herrings in DXA's differentials

Date: 2026-09-24. Scope: the 109,938 adults in the DDXPlus test split and the 470-case `eval-v02-adult` sample, plus the 250-case v0 set for the model-flag check. Code: `scripts/analysis/dxa_red_herrings.py`; outputs in `results/analysis/dxa_red_herrings/`. No inference spend.

A red herring is a (patient, serious condition c) pair where DXA puts p >= 10% on c, but c is essentially never the true condition of DDXPlus adults who look the same. "Never true in DDXPlus's world" is the whole claim; it is not a clinical judgement, and section 6 lists the cases where the two part ways. Serious means DDXPlus severity <= 2 (17 conditions), as in `evaluator/answer_key_v02.py`.

## Summary

1. **57% of DXA's serious probability mass at >= 10% is red herring, and one rule explains almost all of it: the patient has none of the condition's five hallmark symptoms.** Over 116,711 (adult, serious condition) pairs at p >= 10%, the recommended detector M1' labels 57.5% of the mass red herring (33% of all serious mass, including mass under 10%); on the 470 sample it labels 50.6% (351 of 547 pairs; 186 of the 262 cases with a serious condition at >= 10% carry at least one). 75,668 pairs, 55% of the mass, have no hallmark symptom of the condition, and the condition is true in 50 of them (0.07%). With one hallmark it is true 14% of the time, with two 48%, with three or more 88-99% (section 3).
2. **DXA is calibrated on average within each probability band, so the band-only check never fires; red herrings exist only conditionally on the evidence.** MI at DXA 10-20% is true 8.1% of the time over 16,790 adults, at 35-50% 19.5%. The same 10-20% band is 0.02% true (3 of 12,684) when no MI hallmark is present and 100% (478 of 478) with three. Check A (band only) labels 0.6% of the mass, so the "M1' AND A" rule labels 0.6% too. The recommendation is M1' alone, with A reported as calibration context and B (DXA top-1 grouping, 82% agreement) as a second check (sections 2 and 3).
3. **The known knowledge-base quirks are red herrings; two suspected ones are not.** MI under a laryngitis, pharyngitis or bronchitis top diagnosis: 7,092 pairs, 0 true, 97% labelled. Anaphylaxis under inguinal hernia: 1,711 pairs, 0 true, 100%; under pancreatic neoplasm 1,031, 0 true, 99%. Pulmonary oedema under atrial fibrillation 457, 0 true, 100%. Dystonic reactions under sarcoidosis or SLE 600, 0 true, 100%. PSVT under atrial fibrillation is genuine (1,751 pairs, 183 true, 3% labelled), and MI under GERD is borderline (674 pairs, 8 true, 34% labelled). The largest single profile is DXA ranking MI first at a mean 31% with no MI hallmark: 3,540 pairs, 0.4% true, the truths being pericarditis, viral pharyngitis, unstable angina and sarcoidosis (section 4).
4. **On the v0.3 uses: the 20% excuse rule keeps 5 of its 32 excused patients, the atypical-serious subset is 4 cases and untouched, and 36-72% of the models' DXA-supported wrong severe flags are red herrings.** On the 470 sample the single-condition excuse at >= 20% drops from 32 non-serious patients to 5 (Guillain-Barre under myasthenia gravis 2, PSVT under atrial fibrillation 2, myocarditis under atrial fibrillation 1); at >= 10% it drops from 128 to 27. The 19 MI-at-20-50% cases that `docs/risk-proxy-validation.md` called defensible escalations are all red herrings (MI cell rates 0.000-0.002 over cells of 1,000-12,000). Only 4 of the 160 serious cases have a benign DXA top diagnosis, no detector labels their true condition a red herring, and 3 of the 4 carry another serious red herring. Across the 19 v0.1 models on the 250-case set, 4,362 severe top-5 flags split into 1,260 on the truth, 2,598 on conditions DXA itself puts under 10% on (outside the detector), 295 red herrings and 209 DXA-plausible; the commonest red-herring flag is anaphylaxis on a scombroid patient (85 flags), a genuine clinical differential (section 5).
5. **The real-world referee cannot choose between plausibility estimators, because it only sees presentation-level means.** Against the eight published pre-test probabilities, leave-one-pattern-out, the lowest raw error belongs to the tempered naive Bayes at T = 0.02 (mean |log ratio| 1.05, 95% interval 0.68-1.46), which is the prior in disguise: it gives every adult P(any serious) of about 0.23 and labels 18% of true serious mass a red herring. The band-only check A (1.20) and DXA itself (1.23) come next; M1' scores 2.39 and the full naive Bayes 6.07, and the shape-only ranking is the same. Any estimator calibrated to DDXPlus returns the generator's own per-presentation truth rate, which sits further from the real world than DXA's number (pleuritic pain: PE 1.1% published, 11.7% DXA, 29.1% DDXPlus truth), so the referee ranks estimators by how little evidence they use. The choice in this doc rests on the reference-class argument and DDXPlus-internal error rates, not on the referee (section 2).

## 1. Method

**Data.** `data/ddxplus_v0/release_test_patients`, adults (age >= 18): 109,938 rows, each with one true condition (PATHOLOGY) and DXA's differential, normalised over the 49 conditions. The reference population is these adults minus the 470 sample cases and the 250 v0 cases (109,219 rows), so no sample case sees its own truth. Reference rows are scored leave-one-out. Case ids map to rows as `ddxplus_<row>`.

**Pairs.** One row per (adult, serious condition c) with DXA p_c >= 5%: 334,068 pairs, of which 116,711 have p_c >= 10% (23,493 of them true) and 24,307 have p_c >= 20%. The main threshold X is 10%, because it covers both thresholds the v0.3 design uses (the pilot's 10% excuse set, the risk-proxy doc's 20% excuse rule) and is the level below which DXA's mass is spread over a dozen conditions and enters no rule. Sensitivity at 5% and 20% is in section 3.

**Estimators.** Each gives P(c | patient). The cell-based ones read the empirical rate of c among reference adults in the same cell. The table states what each one assumes.

| Name | Cell, or model | Assumes | Role |
|---|---|---|---|
| DXA | DXA's own p_c | DXA is right | the thing under test |
| A | (c, p_c band) | nothing beyond DXA's number: "when DXA says 10-20% for c, how often is c true?" | calibration check |
| M1' | (c, p_c band, count of c's hallmark symptoms present, 0-5) | c's plausibility depends on DXA's number and on evidence about c itself; nothing about other diseases | recommended |
| M1'p | (c, p_c band, which hallmarks are present) | as M1', finer; 41% of cells under 30 patients | sensitivity |
| M1'ant | as M1', hallmarks chosen from all evidence including antecedents | as M1', but DDXPlus samples risk-factor antecedents only for the true condition, so it leans to the oracle | sensitivity |
| B | (c, p_c band, DXA's top-1 condition) | DXA's grouping of diseases: patients whose top diagnosis is the same are alike | check; assumes DXA's own similarity model |
| M2 | (c, DXA's top-3 set) | DXA's grouping, finer | sensitivity |
| KNN | 200 nearest reference adults by evidence Jaccard | all evidence, no model; sample cases only | sensitivity |
| NB T | naive Bayes over all evidence codes, likelihoods raised to T; T = 1 is the v0.2 reference reader | all evidence, conditional independence; T = 1 is the dataset oracle | sensitivity, and the referee's fitted candidate |

Bands are 0-5, 5-10, 10-20, 20-35, 35-50 and 50-100%. Cells under 30 reference patients are suppressed: the pair is "undetermined", never a red herring. M1' has 1,197 cells, 866 with 30 or more, median 132 patients; 0.8% of pairs at >= 10% fall in a suppressed cell. Full cell tables: `cells.csv`, sizes in `cell_sizes.csv`.

**Hallmarks.** For each condition, the five symptom tokens (an evidence code with its value, antecedents excluded) with the largest likelihood ratio P(token | c) / P(token | not c) among tokens present in at least 20% of c's reference patients. For MI they are pain characterised as "scary" or "sickening" and pain radiating to the thyroid cartilage, under the jaw or to the right shoulder; for PE, pain radiating to the side of the chest and swelling behind the ankle; for anaphylaxis, contact with a possible allergen and swelling of the nose, cheeks or forehead; for pulmonary oedema, bouts of choking or breathlessness and swelling over the tibia. Full list: `hallmarks.csv`. Antecedents are excluded because DDXPlus samples them only for the true condition (HIV, high cholesterol and family history for MI; DVT history, immobility and recent surgery for PE), which makes "no risk factor" a dataset artefact rather than a presentation; the M1'ant row shows the difference is small anyway.

**Rule.** (i, c) is a red herring when DXA p >= X, the cell has >= 30 reference patients, and the estimator's rate < max(1%, p / 10). The floor stops a 10% claim being called a red herring by a 0.9% cell; the ratio keeps the same tenfold shortfall at higher p. At p = 10% a cell of 30 with 3 expected truths shows 0 with probability 4%, and one of 100 with probability 0.003%, so the labelled cells are not sampling noise. Section 3 shows the label is insensitive to the floor, the ratio, the minimum cell size and to replacing the rate by its Wilson 95% upper bound.

**Referee.** The published pre-test probabilities and NHAMCS cells from `docs/risk-proxy-validation.md`, reused through `scripts/analysis/risk_proxy_validation.py`: eight primary patterns with a nonzero published rate (chest pain, pleuritic pain, dyspnoea, haemoptysis, rash, fever with cough, palpitations, wheeze) and twelve secondary rows. For each estimator and pattern we take the mean estimated probability of the published conditions over the DDXPlus adults with the pattern and compare it with the published rate as a log ratio. T is fitted on seven patterns and scored on the eighth, raw and shape-only (a free log offset absorbs the constant prior inflation). Intervals come from 2,000 bootstrap draws over the eight patterns. The NHAMCS comparison takes each estimator's mean P(any on-list serious condition) per presentation x age cell against the ED on-list serious-diagnosis rate, over the 36 banded cells.

## 2. Estimator selection against real-world rates

`selection_summary.csv`, `selection_patterns.csv`, `selection_T_folds.csv`.

| Estimator | Raw error, 8 patterns (95% CI) | Shape error (95% CI) | Median ratio | NHAMCS Spearman, 36 cells | NHAMCS median ratio |
|---|---|---|---|---|---|
| DXA | 1.23 (0.73-1.72) | 0.90 (0.57-1.27) | 3.3x | 0.52 | 12.0x |
| A | 1.20 (0.84-1.61) | 0.75 (0.32-1.27) | 2.7x | 0.44 | 6.6x |
| M1' | 2.39 (1.40-3.72) | 2.06 (0.71-4.02) | 4.6x | 0.33 | 6.7x |
| M1'ant | 2.46 (1.41-3.92) | 2.20 (0.78-4.27) | 4.6x | 0.33 | 6.7x |
| B | 2.04 (1.32-2.86) | 1.59 (0.56-2.99) | 4.4x | 0.37 | 5.7x |
| NB 0.1 | 2.29 (1.40-3.43) | 1.89 (0.63-3.67) | 4.9x | 0.34 | 5.6x |
| NB 0.25 | 3.16 (1.42-5.98) | 3.57 (1.56-6.82) | 4.8x | 0.27 | 5.8x |
| NB 1 (oracle) | 6.07 (1.41-14.7) | 9.40 (4.90-17.6) | 4.6x | 0.26 | 5.8x |
| NB fitted, T = 0.02 in every fold | 1.05 (0.68-1.46) | 2.53 (0.51-6.04) | 1.8x | 0.48 | 10.1x |

Per pattern, the estimators calibrated to DDXPlus return the generator's own truth rate:

| Pattern | Published | DXA | A | M1' | NB 1 | DDXPlus truth |
|---|---|---|---|---|---|---|
| Chest pain, ACS | 3.6% | 19.5% | 11.3% | 19.2% | 19.4% | 19.3% |
| Pleuritic pain, PE | 1.1% | 11.7% | 10.8% | 29.0% | 29.1% | 29.1% |
| Dyspnoea, PE | 1.2% | 5.3% | 4.4% | 4.7% | 4.7% | 4.7% |
| Haemoptysis, PE | 2.6% | 6.5% | 5.9% | 16.5% | 16.6% | 16.6% |
| Rash, anaphylaxis | 1.0% | 7.5% | 6.1% | 16.1% | 16.4% | 16.4% |
| Fever and cough, pneumonia | 5.0% | 7.1% | 8.7% | 12.0% | 11.9% | 11.9% |
| Palpitations, PSVT | 8.8% | 14.4% | 19.2% | 22.5% | 22.6% | 22.6% |
| Wheeze, PE | 5.9% | 3.6% | 2.4% | 0.0% | 0.0% | 0.0% |

What the referee can and cannot say:

- **It ranks estimators by how little evidence they use.** The order on every metric is prior, A, DXA, B, NB 0.1, M1', NB 0.25, NB 1: the sharper the estimator, the worse. That is not a finding about within-pattern plausibility. An estimator that is calibrated to DDXPlus reproduces the generator's truth rate per pattern (M1' and NB 1 agree with the truth column to within 0.3 points in seven of eight rows), and the generator's serious priors are 1.5-26x the real world, further out than DXA's numbers, which under-weight the truth in pleuritic pain, haemoptysis and rash. The referee measures the distance between DDXPlus's condition mix and the world. It is blind to reshuffling within a pattern, which is all a red-herring detector does.
- **The fitted temperature is degenerate.** T = 0.02 wins every fold because flattening toward the prior lowers the constant inflation; the result gives every adult P(any serious) between 0.24 (median) and 0.31 (99th percentile of non-serious adults) and 0.40 for serious adults, and it labels 1,334 true pairs (18% of true serious mass) red herrings. The shape-only fit picks T = 0.02 in seven folds and 0.25 in the wheeze fold, where the generator never produces PE and any naive Bayes returns 0 (`selection_T_folds.csv`). Full naive Bayes is degenerate the other way: P(any serious) is 0.0000 up to the 95th percentile of non-serious adults (`nb_degeneracy.csv`).
- **Intervals overlap.** With eight patterns the 95% intervals of DXA, A, the prior and B overlap, and M1' overlaps B and NB 0.1. No estimator is separated from its neighbours.

**Verdict.** The referee cannot select a plausibility estimator. Read strictly it would pick the prior (raw) or A (shape), neither of which can detect a red herring: A labels 0.6% of the mass and the prior labels true conditions. The selection in this doc therefore rests on the reference-class argument in section 1 (M1' conditions only on DXA's number and evidence about c itself; B and M2 inherit DXA's grouping; naive Bayes and KNN are the oracle end) and on the DDXPlus-internal error rates in section 3. A referee that could do the job would need within-presentation real-world rates, for example the pre-test probability of MI in chest pain without radiation or exertional character; none is in the collected material.

## 3. Detector comparison inside DDXPlus

`detector_comparison.csv`, X = 10%, population (sample-470 values in the file).

| Detector | Mass labelled red herring | Undetermined | Wrong mass labelled | True mass labelled | True pairs labelled (of 23,493) | Agreement with M1' |
|---|---|---|---|---|---|---|
| A (band only) | 0.6% | 0.3% | 0.9% | 0.0% | 2 | 34% |
| M1' | 57.5% | 1.6% | 82.9% | 0.2% | 57 | 100% |
| M1'p (hallmark pattern) | 59.4% | 10.2% | 85.7% | 0.2% | 67 | 97% |
| M1'ant (antecedents allowed) | 60.9% | 1.9% | 87.9% | 0.2% | 67 | 94% |
| B (top-1 grouping) | 49.3% | 2.4% | 71.2% | 0.1% | 32 | 82% |
| M2 (top-3 set) | 51.5% | 5.6% | 74.2% | 0.3% | 75 | 81% |
| KNN, sample only | 59.6% | 0 | 98.1% | 0.0% | 0 | 88% |
| NB 0.1 | 58.1% | 0 | 84.0% | 0.0% | 0 | 88% |
| NB 0.25 | 66.7% | 0 | 96.4% | 0.0% | 0 | 89% |
| NB 1 (oracle) | 68.6% | 0 | 99.0% | 0.0% | 8 | 87% |
| NB fitted (prior) | 26.4% | 0 | 30.0% | 18.3% | 1,334 | 50% |
| M1' AND A | 0.6% | 1.6% | 0.9% | 0.0% | 2 | 34% |

Reading the table:

- **Finer matching approaches the oracle; coarser matching approaches the clinician's blank page.** The oracle (NB 1) labels 99% of wrong mass, KNN 98%, NB 0.25 96%; M1' 83%; B 71%; A 1%. What separates the middle from the oracle is what the label means: NB 1 says "given every finding, this is not the patient's condition", which is true of every wrong condition in a deterministic generator; M1' says "among 12,684 adults where DXA said 10-20% MI and none of MI's five hallmark symptoms was present, 3 had MI", a statement about a large, describable reference class.
- **The hallmark count carries the whole signal.** Pairs at >= 10% by count of the condition's hallmark symptoms present: 0 hallmarks, 75,668 pairs, 50 true (0.07%), 99.7% labelled; 1 hallmark, 12,599 pairs, 14.0% true, 18% labelled; 2 hallmarks, 10,551 pairs, 48.2% true, 0.4% labelled; 3, 4 and 5 hallmarks, 17,893 pairs, 87.7-99.4% true, none labelled. 75,422 of the 75,674 labelled pairs have no hallmark. The practical reading of M1' is "DXA >= 10% and none of the condition's five hallmark symptoms".
- **DXA is calibrated by band, so A sees nothing.** Rates by band for MI: 5-10% 3.0%, 10-20% 8.1%, 20-35% 8.6%, 35-50% 19.5%, 50-100% 28.3%; anaphylaxis 10-20% 14.7%, 20-35% 15.0%; PE 10-20% 15.7%, 20-35% 30.7%. Every band sits within a factor of about 2 of DXA's claim, which `docs/risk-proxy-validation.md` found at the presentation level. The red herring is invisible at this level, and any rule that requires A to agree finds nothing.
- **Errors on the truth are rare and explainable.** M1' labels 57 true pairs (0.24%), 14 MI, 13 unstable angina, 11 PE and 9 stable angina, all in cells whose rate is under 1% (an MI with none of its five hallmarks). The Wilson upper-bound rule below cuts this to 36 for a 2-point loss of coverage.

**Sensitivity** (`sensitivity.csv`; M1', population mass share at X = 10% unless stated, and the non-serious patients still excused at 20% on the 470 sample, from 32):

| Variation | Mass labelled | True pairs labelled | Excused at 20% after |
|---|---|---|---|
| Rule max(1%, p/10), cells >= 30 (chosen) | 57.5% | 57 | 5 |
| Rule 1% absolute | 56.0% | 39 | 5 |
| Rule p/10, no floor | 57.5% | 57 | 5 |
| Rule max(0.5%, p/20) | 56.2% | 44 | 5 |
| Rule max(2%, p/5) | 59.5% | 125 | 5 |
| Wilson 95% upper bound < max(1%, p/10) | 55.2% | 36 | 7 |
| Cells >= 100 instead of 30 | 56.8% | 52 | 7 |
| X = 5% | 70.5% of mass at >= 5% | 148 | 5 |
| X = 20% | 40.6% of mass at >= 20% | 27 | 5 |

The label moves by at most 2 points across rules and cell sizes; X moves it because low-probability mass is mostly hallmark-free.

## 4. Which conditions and profiles

Per serious condition, population, X = 10%, M1' (`per_condition.csv`):

| Condition | Severity | True adults | Pairs >= 10% | True rate at >= 10% | Mean DXA p | Mass labelled, >= 10% | Mass labelled, all p | Dominant DXA top-1 in the labelled mass |
|---|---|---|---|---|---|---|---|---|
| Acute pulmonary oedema | 1 | 2,598 | 6,230 | 11.0% | 12.2% | 86.8% | 26.4% | anaemia 28%, myasthenia gravis 14%, dystonic reactions 12% |
| Anaphylaxis | 1 | 2,928 | 11,913 | 14.7% | 15.4% | 85.3% | 45.7% | inguinal hernia 23%, localized oedema 18%, scombroid 16% |
| Pulmonary embolism | 2 | 2,864 | 10,704 | 19.1% | 16.1% | 78.8% | 41.5% | localized oedema 19%, MI 14%, dystonic reactions 11% |
| MI | 1 | 2,911 | 23,999 | 9.3% | 19.5% | 73.8% | 52.7% | MI 31%, viral pharyngitis 15%, bronchitis 11% |
| Ebola | 1 | 71 | 734 | 9.3% | 14.5% | 73.0% | 35.1% | anaemia 85% |
| Unstable angina | 2 | 2,880 | 16,755 | 10.4% | 15.0% | 72.0% | 45.0% | bronchitis 19%, viral pharyngitis 17%, laryngitis 11% |
| Myocarditis | 2 | 1,155 | 5,796 | 12.6% | 14.6% | 68.5% | 21.5% | anaemia 29%, sarcoidosis 21%, dystonic reactions 14% |
| Scombroid | 2 | 1,893 | 6,855 | 27.6% | 17.7% | 55.5% | 24.0% | inguinal hernia 38%, pancreatic neoplasm 16%, HIV 12% |
| Stable angina | 2 | 2,386 | 4,925 | 29.3% | 15.3% | 42.9% | 13.6% | unstable angina 38%, MI 29% |
| Boerhaave | 2 | 1,850 | 4,101 | 33.9% | 21.9% | 37.6% | 22.0% | MI 23%, bronchitis 18%, pancreatic neoplasm 17% |
| Dystonic reactions | 2 | 2,566 | 5,967 | 42.9% | 29.2% | 33.5% | 16.7% | sarcoidosis 25%, myasthenia gravis 17%, MI 12% |
| Guillain-Barre | 2 | 1,986 | 6,762 | 29.3% | 22.5% | 31.4% | 13.4% | anaemia 28%, dystonic reactions 15%, panic attack 14% |
| Epiglottitis | 2 | 1,545 | 2,584 | 58.2% | 17.1% | 24.8% | 17.8% | viral pharyngitis 73%, MI 19% |
| Laryngospasm | 1 | 629 | 1,410 | 44.6% | 53.7% | 15.2% | 12.7% | epiglottitis 63%, anaphylaxis 27% |
| Pneumothorax | 2 | 1,124 | 1,749 | 53.1% | 12.9% | 4.9% | 1.2% | MI 100% |
| PSVT | 2 | 2,003 | 6,227 | 30.6% | 18.6% | 2.9% | 1.6% | PSVT 62%, atrial fibrillation 35% |
| Croup | 2 | 0 | 0 | - | - | - | - | no adults |

Two groups. The five tier-1 and cardiac conditions (pulmonary oedema, anaphylaxis, PE, MI, unstable angina) and myocarditis carry 69-87% red-herring mass: DXA spreads them across respiratory, dermatological and neuromuscular presentations. PSVT, pneumothorax and laryngospasm carry under 16%: DXA raises them only where their findings are present.

The largest (DXA top-1, condition) profiles among the labelled pairs (`dominant_profiles.csv`; mass in patient-equivalents, mean p, true rate, mean hallmarks):

| DXA top-1 | Red herring | Pairs | Mass | Mean p | True rate | Hallmarks |
|---|---|---|---|---|---|---|
| MI | MI | 3,540 | 1,083 | 30.6% | 0.4% | 0.03 |
| Viral pharyngitis | MI | 2,765 | 522 | 18.9% | 0.0% | 0.00 |
| Bronchitis | MI | 2,593 | 374 | 14.4% | 0.0% | 0.00 |
| Inguinal hernia | Anaphylaxis | 1,711 | 356 | 20.8% | 0.0% | 0.00 |
| Bronchitis | Unstable angina | 2,539 | 337 | 13.3% | 0.0% | 0.00 |
| Acute laryngitis | MI | 1,521 | 334 | 22.0% | 0.0% | 0.00 |
| Viral pharyngitis | Unstable angina | 2,345 | 311 | 13.3% | 0.0% | 0.00 |
| Localized oedema | Anaphylaxis | 2,249 | 286 | 12.7% | 0.0% | 0.00 |
| Inguinal hernia | Scombroid | 1,764 | 256 | 14.5% | 0.0% | 0.00 |
| Localized oedema | PE | 814 | 252 | 31.0% | 0.0% | 0.00 |
| Scombroid | Anaphylaxis | 1,517 | 251 | 16.6% | 0.0% | 0.00 |

The first row is the one B cannot see: DXA ranks MI first, at a mean 31%, in 6,968 adults; 2,225 have MI, and the 3,540 with no MI hallmark symptom have pericarditis (772), viral pharyngitis (616), unstable angina (585) or sarcoidosis (561) and MI 0.4% of the time. B labels 58% of MI's mass against M1''s 74% because it pools these with the true MIs.

Known quirks (`quirks.csv`, pairs at >= 10%):

| Quirk | Pairs | True | True rate | Mean p | Labelled by M1' | by A | by B |
|---|---|---|---|---|---|---|---|
| MI under laryngitis, pharyngitis or bronchitis | 7,092 | 0 | 0.0% | 17.8% | 97% | 0% | 100% |
| MI under sarcoidosis or SLE | 505 | 3 | 0.6% | 24.2% | 94% | 0% | 78% |
| MI under pericarditis | 861 | 0 | 0.0% | 13.7% | 100% | 0% | 99% |
| MI under pulmonary neoplasm | 416 | 0 | 0.0% | 14.2% | 100% | 0% | 99% |
| MI under GERD | 674 | 8 | 1.2% | 16.9% | 34% | 0% | 77% |
| Unstable angina under GERD | 1,224 | 4 | 0.3% | 14.3% | 100% | 0% | 100% |
| Anaphylaxis under inguinal hernia | 1,711 | 0 | 0.0% | 20.8% | 100% | 0% | 100% |
| Anaphylaxis under pancreatic neoplasm | 1,031 | 0 | 0.0% | 15.5% | 99% | 0% | 99% |
| PSVT under atrial fibrillation | 1,751 | 183 | 10.5% | 17.4% | 3% | 0% | 0% |
| Pulmonary oedema under atrial fibrillation | 457 | 0 | 0.0% | 11.3% | 100% | 0% | 100% |
| Dystonic reactions under sarcoidosis or SLE | 600 | 0 | 0.0% | 24.6% | 100% | 0% | 95% |
| Scombroid under URTI, pharyngitis or sinusitis | 98 | 0 | 0.0% | 11.7% | 100% | 0% | 92% |
| PE under bronchitis or pneumonia | 226 | 0 | 0.0% | 14.0% | 100% | 0% | 97% |
| Guillain-Barre or myocarditis under COPD or asthma | 0 | - | - | - | - | - | - |

The pilot's quirk list (`docs/dangerous-if-missed-pilot.md`, section 1) holds for MI in the pharyngitis family, anaphylaxis in hernia and pancreatic neoplasm, and pulmonary oedema in atrial fibrillation. PSVT in atrial fibrillation is not a quirk: DDXPlus produces PSVT there one time in ten, and M1' keeps it. MI in GERD is the one borderline family: GERD patients share one MI pain feature on average, and M1' labels a third. The Guillain-Barre and myocarditis mass in dyspnoea that the risk-proxy doc described sits under other top diagnoses (anaemia, dystonic reactions, sarcoidosis), where it is labelled.

## 5. Impact on the v0.3 uses

**Excuse rule** (`excuse_rule.csv`, `excuse_cases_470.csv`). A non-serious patient in the 470 sample is excused when some serious condition has DXA p >= t and that pair is not a red herring.

| Threshold t | Excused before | After M1' | After B | After A, or M1' AND A | After NB 1 |
|---|---|---|---|---|---|
| 20% (risk-proxy doc's rule) | 32 | 5 | 19 | 32 | 0 |
| 10% (pilot's K5 excuse set) | 128 | 27 | 41 | 128 | 0 |

The five patients still excused at 20% are the ones whose serious condition has its findings: Guillain-Barre at 21-35% in two myasthenia gravis patients (cell rates 0.42 and 0.58), PSVT at 33% in two atrial fibrillation patients (0.70), and myocarditis at 20% in an atrial fibrillation patient (0.07). The 27 dropped are MI at 21-55% in sarcoidosis, SLE, laryngitis, pharyngitis, pericarditis and pancreatic neoplasm (cell rates 0.000-0.002, cells of 1,000-12,700), myocarditis and dystonic reactions in sarcoidosis and myasthenia gravis, anaphylaxis and PE in localized oedema, and scombroid in inguinal hernia. The risk-proxy doc left it to clinicians whether MI at 20-50% in a sarcoidosis or SLE patient is a real differential; inside DDXPlus it is not: 505 such pairs, 3 true.

**Atypical serious subset** (`atypical_serious_470.csv`). Of the 160 serious cases in the sample, 4 have a benign DXA top diagnosis: PSVT under anaemia (truth at 15.1%, rank 2, M1' rate 0.40), Boerhaave under panic attack twice (truth at 5.2%, rank 6, below X) and Ebola under pneumonia (10.5%, rank 2, cell of 13, undetermined). No detector labels any of these truths a red herring; 3 of the 4 carry another serious condition that is one (pulmonary oedema at 11% in the PSVT case, dystonic reactions at 16% and myocarditis at 18% in both Boerhaave cases). The subset is too small to change under any detector, and the detector's job there is the opposite one: to show that the atypical case's own serious mass is genuine, which it does for the two above X.

**Model flags** (`model_flags.csv`, `model_flag_pairs.csv`). Each v0.1 model's top-5 differential on the 250-case v0 set, mapped to DDXPlus conditions as in the pilot; a severe flag is a listed severity <= 2 condition.

| Model | Severe flags | On the truth | DXA < 10% (outside the detector) | Red herring | DXA-plausible, not true | Red herring share of DXA-supported wrong flags |
|---|---|---|---|---|---|---|
| GPT-OSS 120B | 237 | 55 | 150 | 23 | 9 | 72% |
| GPT-5.6 Luna | 254 | 60 | 160 | 23 | 11 | 68% |
| Kimi K3 | 253 | 77 | 148 | 18 | 10 | 64% |
| GPT-5 Mini | 252 | 61 | 161 | 19 | 11 | 63% |
| GPT-5.4 Mini | 159 | 49 | 94 | 10 | 6 | 63% |
| DeepSeek R1 | 172 | 52 | 99 | 13 | 8 | 62% |
| GPT-5.2 | 306 | 69 | 201 | 22 | 14 | 61% |
| Gemini 3.1 Pro | 277 | 79 | 160 | 23 | 15 | 61% |
| Opus 5 | 251 | 87 | 138 | 15 | 11 | 58% |
| GPT-6 Astra | 249 | 71 | 155 | 13 | 10 | 57% |
| GPT-5.6 Terra | 256 | 75 | 156 | 14 | 11 | 56% |
| Grok 4.6 | 283 | 79 | 170 | 19 | 15 | 56% |
| GPT-5.6 Sol | 274 | 70 | 175 | 16 | 13 | 55% |
| GPT-5 Chat | 172 | 54 | 98 | 11 | 9 | 55% |
| Fable 5 | 252 | 79 | 140 | 18 | 15 | 55% |
| GLM 5.3 | 240 | 66 | 150 | 13 | 11 | 54% |
| Gemini 3 Pro | 210 | 67 | 117 | 13 | 13 | 50% |
| Haiku 4.5 | 104 | 46 | 43 | 7 | 8 | 47% |
| Sonnet 4.6 | 161 | 64 | 83 | 5 | 9 | 36% |

Across the 19 models: 4,362 severe flags, 1,260 on the truth, 2,598 on a condition DXA itself puts under 10% on, 295 red herrings, 209 DXA-plausible. The detector covers only the DXA-supported wrong flags (504 of the 3,102 wrong ones), of which 36-72% are red herrings, median 57%. The commonest red-herring flags are anaphylaxis on a scombroid patient (85 across models), PE on pulmonary neoplasm (36), MI on PE (21), MI on unstable angina (20) and PE on pneumothorax (18). Every one of these is a differential a clinician would list, and the two largest are the textbook mimic pairs. This is the clearest demonstration that "never true in DDXPlus" and "clinically implausible" are different claims: models do not echo DXA's laryngitis-MI quirk (11 flags of MI on panic attack is the nearest), they flag the mimic, and DDXPlus's generator never produces the mimic under that truth.

## 6. Recommendation and limits

Recommended detector and rule for the spec:

1. **Detector M1'.** For a serious condition c with DXA p_c >= 10%, the reference class is the DDXPlus adults with p_c in the same band and the same count (0-5) of c's hallmark symptoms, the five symptom tokens with the largest likelihood ratio among those present in at least 20% of c's patients, learned on the test split with the sample cases held out. The pair is a red herring when the class has at least 30 patients and its rate of c is under max(1%, p_c / 10). The practical reading: DXA >= 10% and no hallmark symptom of the condition.
2. **Report A beside it as calibration context, not as a veto.** A says whether DXA's number is right on average for c at that level (it is, within about 2x); M1' says whether it is right for this presentation. Requiring both finds nothing, because the red herring is a conditional error.
3. **Use the label only to strip DXA-derived sets.** Remove red-herring pairs from the excuse set (rule (a)) and from any plausible-set key or "severe flags not excused" count. Never let the label create a miss, an over-flag or a penalty, because a red herring in DDXPlus can be a genuine clinical differential (anaphylaxis on scombroid, MI on unstable angina, PE on pneumothorax).
4. **Publish the per-case list** (`sample470_cases.csv`: case_id, condition, dxa_p, empirical_rate, cell_n, red_herring, plus every check's rate) with the sample, so the excuse set is reproducible from a CSV rather than from the detector.

Limits:

1. **Closed world.** The hallmarks and rates are learned from DDXPlus's evidence pool, which gives most conditions findings no other condition produces (contact with an allergen appears only in anaphylaxis; pain radiating to the side of the chest only in PE). The 0.07% rate for hallmark-free pairs is a property of the generator. The label means "DDXPlus never produces this", which section 5 shows differs from "a clinician would not list this" on the models' actual flags.
2. **The referee is blind to the question.** Section 2 shows published pattern-level rates rank estimators by how little evidence they use; the choice of M1' is an argument about reference classes and internal error rates, not a measured real-world calibration. A within-presentation referee (real pre-test rates conditioned on pain features, for example) would be needed to test it, and none is in the collected material.
3. **Hallmark choice is ours.** Five symptom tokens, prevalence >= 20%, likelihood-ratio order. M1'p (presence pattern) and M1'ant (antecedents allowed) move the mass share by 2-3 points and agree with M1' on 94-97% of pairs, and the count 0 cell carries 99.7% of the labels, so the number and the ordering matter little; the exclusion of antecedents is a judgement that risk factors are not a presentation.
4. **Reference and sample share the test split.** Cells are learned on the test split minus the 720 sample and v0 cases; the validate split (132k rows) would give an independent check of the cell rates, which we did not run.
5. **Small conditions.** Ebola has 71 adults and croup none; 331 of the 1,197 M1' cells are under 30 patients and leave 977 pairs (0.8%) undetermined: Ebola at 10-35% (89 pairs, 15% of its pairs), pulmonary oedema at 20-35% (74), and the 35-50% band of PE, unstable angina and dystonic reactions (36-41 each).
6. **Ties in DXA's differential** make the top-1 and top-3 arbitrary for B and M2; M1' and A do not use them.
7. **The model-flag check** uses v0.1 top-5 differentials on the 250-case set as a proxy for a flag list, and the ICD-10 map's strictness decides which flags count as a DDXPlus condition, both as in the pilot.

## Files

- `scripts/analysis/dxa_red_herrings.py`: the analysis. It caches parsed adults and the pair table under `results/analysis/dxa_red_herrings/cache/` (160 MB, safe to delete).
- `results/analysis/dxa_red_herrings/sample470_cases.csv`: per (case, serious condition) row for the 470 sample: case_id, condition, dxa_p, truth, band, hallmarks present, empirical_rate and cell_n (M1'), red_herring, undetermined, and the rate and label of every check.
- `selection_summary.csv`, `selection_patterns.csv`, `selection_T_folds.csv`, `nb_degeneracy.csv`: section 2.
- `detector_comparison.csv`, `cells.csv`, `cell_sizes.csv`, `hallmarks.csv`, `sensitivity.csv`: section 3.
- `per_condition.csv`, `dominant_profiles.csv`, `quirks.csv`: section 4.
- `excuse_rule.csv`, `excuse_cases_470.csv`, `atypical_serious_470.csv`, `model_flags.csv`, `model_flag_pairs.csv`: section 5.
- `summary.json`, `summary.md`: every table rendered.
