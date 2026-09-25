# "Dangerous if missed" labels for the 49 DDXPlus conditions

Date: 2026-09-24. Scope: the 49 conditions in `data/ddxplus_v0/release_conditions.json` and the 470-case adult sample (`data/test_sets/eval-v02-adult.json`, 47 conditions with 10 cases each; croup and bronchiolitis have no adult cases). Companion data: `spec/dangerous_if_missed_tiers.csv`. No scoring code, spec or existing file changed. Every figure below was read this session from a fetched abstract, full text or data file, except where marked "via docs/ddxplus-severity-validation.md" (verified there earlier and reused) or "reported by the coordinator" (not re-verified).

Web search was unavailable (session budget spent), so sources were reached by direct URL: PubMed E-utilities for abstracts, PubMed Central for full texts, archive.org's full-text index for Murtagh, WikEM's API, and the NHAMCS outputs already in `results/analysis/nhamcs_urgency/`.

## Summary

1. **The only third-party source that ranks our conditions by harm when missed is Newman-Toker et al., and it names 4 of the 49 (MI, pulmonary embolism, pneumonia, lung cancer), 5 with the "other cancers" row (pancreatic neoplasm).** Its 2023 Table 1 gives each a diagnostic-error rate and a serious-harm rate per case (section 2). That is 5 of 49 conditions and 50 of 470 cases. NHAMCS covers 29 conditions and 270 cases, but its construct is ED disposition, and it puts anaphylaxis, PSVT and croup in the lowest tier (section 4).
2. **We recommend option (c): DDXPlus severity stays the primary label for all 49, with Newman-Toker as validation and a sensitivity row.** External coverage is too thin for a fallback design to change the benchmark's meaning, and the fallback would relabel only 3 conditions (pneumonia, lung cancer and pancreatic neoplasm move from tier 2 to tier 1; PE is already tier 1 under the severity <= 2 line). The sensitivity row moves 30 of 470 cases (section 5).
3. **Against a 3-tier crosswalk of severity (1-2, 3, 4-5), the Newman-Toker-first label agrees at quadratic weighted kappa 0.61 and the NHAMCS-only label at 0.43; against the NTS translation, 0.61 and 0.53.** The disagreements the user asked about: pneumonia, lung cancer and pancreatic cancer move up (Newman-Toker); AF moves up only under NHAMCS (ICU 10.2%); COPD, HIV, TB, stable angina and scombroid do not move under any external source, because none covers them at condition level (section 6).
4. **The literature does not support a numeric weight for "right diagnosis" against "dangerous condition flagged". Report them separately.** The closest evidence: about 40-60% of diagnostic errors in dangerous diseases end in death or permanent disability (Newman-Toker 2023: harm 4.4% per case over error 11.1% overall; 0.48-0.63 for the four named rows), while a wrong admission diagnosis in an admitted patient raises in-hospital mortality from 3.8% to 8.6% (Hautz 2019). These are different constructs, so the order-of-magnitude reading "flag-failure harm is roughly 10 times diagnosis-error harm" is a bound, not a weight (section 7).
5. **Time sensitivity and treatment benefit are not in the label.** Published rule-out thresholds exist for two tier-1 conditions only: PE (PERC test threshold 1.8%, Kline 2004) and ACS (HEART 0-3, MACE 1.7%, Backus 2013). They are recorded as supporting documentation in section 8 and in the CSV, not used to set tiers.

Also, as reported by the coordinator: the design weights true-condition misses by tier and does not build a required-flag set from DXA differential probabilities, which proved unreliable (0 of 7,092 adults with laryngitis-led DXA differentials and DXA MI >= 10% truly had MI). That figure is the coordinator's, not re-verified here.

## 1. Candidate sources for "dangerous if missed"

| Source | What it lists | How it defines "dangerous if missed" | Availability and licence | Coverage of our 49 | Verdict |
|---|---|---|---|---|---|
| **Newman-Toker et al. 2019, Diagnosis 6(3):227-240, PMID 31535832** ("Big Three" malpractice claims) | CRICO Comparative Benchmarking System, 2006-2015, 28.7% of US claims; 55,377 closed claims, 11,592 diagnostic-error cases, 7,379 with high-severity harm (53.0% death). Vascular events, infections and cancers = 74.1% of high-severity cases (22.8%, 13.5%, 37.8%); top 5 of each = 15 diseases = 47.1%; most frequent: stroke, sepsis, lung cancer | High-severity harm = NAIC severity 6-9 (serious permanent disability or death) | De Gruyter; free-to-read PDF (OpenAlex "bronze", `licenseType=free`); publisher copyright; the site blocked our fetch, so figures come from the PubMed abstract | 4 named diseases match (MI, VTE, pneumonia, lung cancer) | Use, through the 2023 paper's table |
| **Newman-Toker et al. 2020/2021, Diagnosis 8(1):67-84, PMID 32412440** (rates for the 15 diseases) | 28 studies, 91,755 patients; error rates 2.2% (MI) to 62.1% (spinal abscess), median 13.6%; serious harm per incident case 1.2% (MI) to 35.6% (spinal abscess), median 5.5%, aggregate 5.2% | Serious harm = morbidity or mortality, from a disease-agnostic harm-per-error rate times claims-based severity weights | Subscription (OpenAlex "closed"); abstract only | Same 4 | Cite for method; per-disease numbers taken from the 2023 paper, which reuses them |
| **Newman-Toker et al. 2023/2024, BMJ Qual Saf 33(2):109-120, PMID 37460118, PMC10792094** (US burden) | Table 1: incidence, diagnostic-error rate and serious misdiagnosis-related harm rate for 15 diseases plus "other" rows; 795,000 serious harms per year (range 598,000-1,023,000); 15 diseases = 50.7% of serious harms, top 5 (stroke, sepsis, pneumonia, VTE, lung cancer) = 38.7% | Serious harm = permanent morbidity or mortality (NAIC 6-9, Box in the paper) | Author manuscript on PMC; "No commercial re-use" (BMJ); full text read | 5 rows match (MI, VTE, pneumonia, lung cancer; "Other cancers" for pancreatic) | **Primary external source** |
| **AHRQ EPC report 22(23)-EHC043, 2022, PMID 36574484** (ED diagnostic errors, Newman-Toker et al.) | Top 15 ED conditions by serious harm (68% of serious harms): stroke, MI, aortic aneurysm/dissection, spinal cord compression, VTE, meningitis/encephalitis, sepsis, lung cancer, TBI/ICH, arterial thromboembolism, spinal/intracranial abscess, cardiac arrhythmia, pneumonia, GI perforation/rupture, intestinal obstruction; error rates 1.5% (MI) to 56% (spinal abscess) | Serious harm = permanent disability or death | Free (AHRQ, NCBI Bookshelf NBK588118); the Bookshelf and AHRQ pages blocked our fetch, so figures come from the PubMed abstract; public-domain statement not verified | Adds category matches: "cardiac arrhythmia" (AF, PSVT), "GI perforation and rupture" (Boerhaave) | Supporting only: categories, not conditions, and the coordinator limited the label to three datasets |
| **Murtagh's General Practice, "diagnostic strategy model"** (McGraw-Hill; 6th ed. 2015 ISBN 9781743760031; 1999 ed.; Companion Handbook 1996, 1999, 2007, 2011) | Per presenting complaint: probability diagnosis, "serious disorders not to be missed", pitfalls, masquerades. Verified fragments from archive.org's full-text index: cough table "Serious disorders not to be missed: Cardiovascular: left ventricular failure; Neoplasia: lung cancer" (6th ed.); chest pain "Cardiovascular: myocardial infarction, dissecting aneurysm, pulmonary [embolism]" (Companion Handbook 1996/1999) and "acute coronary syndrome, aortic dissection, pulmonary embolism" (2011); general rule "the three important serious disorders not to be missed with any painful [condition]" | Expert opinion, one author; no harm data | Copyright John Murtagh and McGraw-Hill; no open licence; the full lists are behind the publisher's paywall and archive.org's lending restriction, so they could not be read in full | Unknown in full; for cough, only left ventricular failure and lung cancer were verified as list members; the rest of the list was not read | Cite as the GP framing; cannot be a label without the book |
| **WikEM** (OpenEM Foundation; CC BY-SA 4.0, template "WikEM Copyright") | Per chief complaint. Cough (rev. 389165): "emergent causes (PE, pneumothorax, foreign body, anaphylaxis, acute heart failure)". Sore throat (389356): "dangerous causes (peritonsillar abscess, retropharyngeal abscess, epiglottitis, Ludwig's angina)". Acute chest pain (522303): "big 5" ACS, PE, aortic dissection, tension pneumothorax, Boerhaave; DDX template splits Critical / Emergent / Nonemergent. Acute dyspnea DDX (386276): Emergent includes anaphylaxis, asthma, epiglottitis, pneumonia, PE, pulmonary edema, MI, pericarditis, myocarditis, anemia, Guillain-Barre, myasthenia; Non-emergent includes COPD exacerbation, panic attack, rib fracture, spontaneous pneumothorax, URI | ED consensus wiki; no harm data | Free, CC BY-SA 4.0; wiki edited by contributors with an editorial board | About 20 conditions appear on an emergent, critical or dangerous list | Not a label: lists conflict across pages (pneumonia emergent for dyspnea, nonemergent for chest pain; COPD exacerbation nonemergent), and a wiki is not a dataset |
| **CRICO / Candello benchmarking reports** | Claims by diagnosis | Claims severity | Reports are "exclusive content" behind a Candello login; the public CBS analysis is Newman-Toker 2019 | - | Use through Newman-Toker 2019 |
| **The Doctors Company diagnostic-error studies** | Claims by diagnosis | Claims | Site fetch returned no article index this session; not verified | - | Not used |
| **BMJ "Easily missed?" series** (from 2009, PMID 19228766) | One article per condition (e.g. subarachnoid haemorrhage, infective endocarditis, giant cell arteritis) | Editorial choice | Paywalled; no consolidated list found on PubMed | Few | Not used |
| **Primary-care and inpatient error series**: Schiff 2009 (583 errors: PE 4.5%, drug reactions 4.5%, lung cancer 3.9%, colorectal cancer 3.3%, ACS 3.1%; 28% major harm), Singh 2013 (190 primary-care errors: pneumonia 6.7%, decompensated CHF 5.7%, acute renal failure 5.3%, cancer 5.3%; 86.8% with moderate-to-severe potential harm), Gunderson 2020 (0.7% harmful diagnostic errors per admission; malignancy 11%, PE 9.6% of described errors) | Frequency of misses | Frequency, not danger | Abstracts free; Singh full text on PMC | Supports PE, pneumonia, lung cancer, ACS | Corroboration only |
| **CDC NHAMCS ED 2016-2022** (`results/analysis/nhamcs_urgency/derived_levels.csv`) | Per condition: visits, admission, critical-care admission, ED death, arrival triage | ED disposition, not harm from missing | Public domain; already in `data/external/` | 29 conditions with >= 30 primary-diagnosis visits | Validation only (section 4) |
| **DDXPlus severity** | 1-5, undocumented (docs/ddxplus-severity-validation.md) | Unknown | Ships with the dataset | 49 | Primary label (section 5) |
| **NTS translation** (`spec/acuity_reference_levels.csv`, superseded) | U1-U5 by our translation | Urgency, our judgement | Ours | 49 | Comparison only |

Not reachable this session: NICE red-flag pages (geo-restricted), Rosen's and ACEP policies (paywalled, not list-form). NICE NG12 figures for the two cancers are reused from docs/ddxplus-severity-validation.md.

## 2. The Newman-Toker rows we use

From Newman-Toker 2023, Table 1 ("Annual US incidence of dangerous diseases, diagnostic errors, & serious misdiagnosis-related harms"), read on PMC10792094:

| Table 1 row | Our condition | Diagnostic error rate | Serious harm rate per case | Serious harms per year (thousands) |
|---|---|---|---|---|
| Myocardial Infarction | Possible NSTEMI / STEMI | 1.5% (CI 1.0-2.2) | 0.8% (0.5-1.2) | 10 |
| Venous Thromboembolism | Pulmonary embolism | 20.4% (17.0-23.9) | 10.9% (8.9-13.1) | 35 |
| Pneumonia | Pneumonia | 9.5% (2.3-14.3) | 4.6% (1.1-7.0) | 68 |
| Lung Cancer | Pulmonary neoplasm | 22.5% (PR 11.3-37.8) | 14.2% (7.1-24.1) | 32 |
| Other Cancers | Pancreatic neoplasm | 11.1% (PPR 10.1-20.9) | 7.4% (6.7-14.2) | 47 |
| TOTAL BIG 3 (Top 5 + Other) | - | 11.1% | 4.4% | 603 |

The method statement, from the same text: "Disease-specific misdiagnosis-related harm rates were derived by multiplying high-quality data on disease-agnostic (non-disease specific) harms per diagnostic error (from well-respected clinical studies) by disease-specific harm-severity weights (from malpractice claims)".

Two rows we do not use, and why: "Other Infections" (harm 3.3%) and "Other Vascular Events" (1.4%) count hospital discharges, so they describe admitted infections and vascular events, not URTI, pharyngitis or sinusitis. Unstable angina is not a named row (the claims category is acute MI), so it falls back to DDXPlus severity 2. The AHRQ ED list's "cardiac arrhythmia" and "GI perforation and rupture" are category matches for AF, PSVT and Boerhaave; we record them but do not use them, per the three-dataset limit.

## 3. Candidate rules

Each rule is written once, in full.

- **A. Newman-Toker first, DDXPlus fallback.** Tier 1 if the condition is a row of Newman-Toker 2023 Table 1 (named disease or "Other Cancers"), else DDXPlus severity 1; tier 2 if DDXPlus severity 2; tier 3 if DDXPlus severity 3-5. Judgement calls: the crosswalk, and counting the "Other Cancers" row.
- **B. NHAMCS only.** Among conditions with >= 30 primary-diagnosis visits: tier 1 if critical-care admission >= 10%; tier 2 if admission >= 30%; tier 3 otherwise; the rest uncovered. Judgement calls: two thresholds and the visit floor.
- **C. Three sources.** Tier 1 if a Newman-Toker row, or NHAMCS critical-care admission >= 10%, or DDXPlus severity 1; tier 2 if NHAMCS admission >= 30% or DDXPlus severity 2; tier 3 otherwise. Judgement calls: the same two thresholds plus the crosswalk.
- **Primary crosswalk (option (c)).** Tier 1 = DDXPlus severity 1-2 (the frozen "serious" line of spec/v0.2-scoring.md), tier 2 = severity 3, tier 3 = severity 4-5. No new judgement call. The "2+ levels" penalty can be taken on the 5-level scale directly, as measure D2 already does.

| Rule | Conditions covered | Cases covered | Tier 1 / 2 / 3 counts | Judgement calls | Traceable to an external row |
|---|---|---|---|---|---|
| A | 49 | 470 | 9 / 11 / 29 | 2 | 5 conditions |
| B | 29 | 270 | 4 / 7 / 18 | 3 | 29, but the construct is ED disposition |
| C | 49 | 470 | 10 / 14 / 25 | 4 | 5 + 3 from NHAMCS (AF, anemia, HIV, COPD) |
| Primary crosswalk | 49 | 470 | 17 / 17 / 15 | 0 | none |

## 4. Coverage options and the recommendation

| Option | What it does | Conditions with an external label | Cases | Assessment |
|---|---|---|---|---|
| (a) fallback | Rule A: Newman-Toker where it has a row, DDXPlus crosswalk elsewhere, source marked per condition | 5 of 49 | 50 of 470 (10.6%) | Changes 3 conditions against the primary crosswalk (pneumonia, lung cancer, pancreatic neoplasm: tier 2 -> 1). Adds a source column for 3 rows' worth of change |
| (b) covered-only | Score only conditions with an external label | 5 (Newman-Toker) or 29 (NHAMCS) | 50 or 270 | With Newman-Toker, 50 cases and 5 conditions cannot carry a headline. With NHAMCS, the label misranks anaphylaxis (tier 3: admit 9.2%, ICU 8.0%), PSVT (tier 3) and croup (tier 3), and drops Ebola, Boerhaave, GBS, epiglottitis, myocarditis, TB and 14 others |
| (c) DDXPlus primary, external as validation and sensitivity | Severity for all 49; Newman-Toker and NHAMCS reported as agreement statistics; one sensitivity row re-scored with rule A | 0 as label; 5 as sensitivity | 470 | Keeps the frozen label; every case has a label from one source; the external evidence is visible where it disagrees |

**Recommendation: (c).** External coverage is 5 of 49 conditions by the only harm-if-missed source, and the alternative with wider coverage (NHAMCS) measures the wrong construct. A fallback design would relabel 30 cases and add a second label source for a 6% gain. Under (c), the board reports the primary crosswalk and one sensitivity row under rule A, plus the agreement statistics of section 6.

## 5. The 49 conditions

Tier (primary) is the severity crosswalk. External tier (A) is rule A. NHAMCS columns are primary-diagnosis visits 2016-2022, weighted admission and critical-care admission rates from `results/analysis/nhamcs_urgency/derived_levels.csv`; "uncovered" means fewer than 30 visits. NTS is the superseded translation.

| Condition | ICD-10 | DDX sev | Tier (primary) | External tier (A) | Source row that set the external tier | NHAMCS n / admit / ICU | NTS |
|---|---|---|---|---|---|---|---|
| Acute pulmonary edema | J81.0 | 1 | 1 | 1 | DDXPlus severity 1 (fallback) | 36 / 73% / 24% | U1 |
| Anaphylaxis | T78.0 | 1 | 1 | 1 | DDXPlus severity 1 (fallback) | 76 / 9% / 8% | U1 |
| Ebola | A98.4 | 1 | 1 | 1 | DDXPlus severity 1 (fallback) | uncovered | U1 |
| Larygospasm | J38.5 | 1 | 1 | 1 | DDXPlus severity 1 (fallback) | uncovered | U2 |
| Possible NSTEMI / STEMI | I21 | 1 | 1 | 1 | Newman-Toker 2023 Table 1 'Myocardial Infarction' (error 1.5%, serious harm 0.8%) | 258 / 89% / 16% | U1 |
| Acute dystonic reactions | G24.02 | 2 | 1 | 2 | DDXPlus severity 2 (fallback) | uncovered | U2 |
| Boerhaave | K22.3 | 2 | 1 | 2 | DDXPlus severity 2 (fallback) | uncovered | U1 |
| Croup | J05.0 | 2 | 1 | 2 | DDXPlus severity 2 (fallback) | 341 / 3% / 1% | U2 |
| Epiglottitis | J05.1 | 2 | 1 | 2 | DDXPlus severity 2 (fallback) | uncovered | U1 |
| Guillain-Barré syndrome | G61.0 | 2 | 1 | 2 | DDXPlus severity 2 (fallback) | uncovered | U2 |
| Myocarditis | I51.4 | 2 | 1 | 2 | DDXPlus severity 2 (fallback) | uncovered | U2 |
| PSVT | I47.1 | 2 | 1 | 2 | DDXPlus severity 2 (fallback) | 88 / 19% / 5% | U2 |
| Pulmonary embolism | I26 | 2 | 1 | 1 | Newman-Toker 2023 Table 1 'Venous Thromboembolism' (error 20.4%, serious harm 10.9%) | 117 / 80% / 14% | U1 |
| Scombroid food poisoning | T61.1 | 2 | 1 | 2 | DDXPlus severity 2 (fallback) | uncovered | U3 |
| Spontaneous pneumothorax | J93 | 2 | 1 | 2 | DDXPlus severity 2 (fallback) | 30 / 50% / 1% | U2 |
| Stable angina | I20.9 | 2 | 1 | 2 | DDXPlus severity 2 (fallback) | 86 / 57% / 1% | U4 |
| Unstable angina | I20.0 | 2 | 1 | 2 | DDXPlus severity 2 (fallback) | 84 / 71% / 6% | U1 |
| Acute COPD exacerbation / infection | J44.1 | 3 | 2 | 3 | DDXPlus severity 3 (fallback) | 579 / 38% / 4% | U3 |
| Atrial fibrillation | I48.91 | 3 | 2 | 3 | DDXPlus severity 3 (fallback) | 479 / 52% / 10% | U4 |
| Bronchiectasis | J47 | 3 | 2 | 3 | DDXPlus severity 3 (fallback) | uncovered | U5 |
| Bronchiolitis | J21 | 3 | 2 | 3 | DDXPlus severity 3 (fallback) | 325 / 19% / 2% | U3 |
| Bronchospasm / acute asthma exacerbation | J45 | 3 | 2 | 3 | DDXPlus severity 3 (fallback) | 1371 / 7% / 1% | U2 |
| Chagas | B57 | 3 | 2 | 3 | DDXPlus severity 3 (fallback) | uncovered | U5 |
| Cluster headache | G44.009 | 3 | 2 | 3 | DDXPlus severity 3 (fallback) | uncovered | U3 |
| GERD | K21 | 3 | 2 | 3 | DDXPlus severity 3 (fallback) | 286 / 2% / 0% | U5 |
| HIV (initial infection) | B20 | 3 | 2 | 3 | DDXPlus severity 3 (fallback) | 37 / 45% / 0% | U5 |
| Influenza | J11.1 | 3 | 2 | 3 | DDXPlus severity 3 (fallback) | 851 / 4% / 0% | U5 |
| Inguinal hernia | K40 | 3 | 2 | 3 | DDXPlus severity 3 (fallback) | 105 / 16% / 0% | U5 |
| Myasthenia gravis | G70.0 | 3 | 2 | 3 | DDXPlus severity 3 (fallback) | uncovered | U5 |
| Pancreatic neoplasm | C25 | 3 | 2 | 1 | Newman-Toker 2023 Table 1 'Other Cancers (row)' (error 11.1%, serious harm 7.4%) | uncovered | U5 |
| Pneumonia | J17/J18 | 3 | 2 | 1 | Newman-Toker 2023 Table 1 'Pneumonia' (error 9.5%, serious harm 4.6%) | 1186 / 44% / 6% | U3 |
| Pulmonary neoplasm | C34 | 3 | 2 | 1 | Newman-Toker 2023 Table 1 'Lung Cancer' (error 22.5%, serious harm 14.2%) | uncovered | U5 |
| Spontaneous rib fracture | S22.9 | 3 | 2 | 3 | DDXPlus severity 3 (fallback) | 225 / 18% / 3% | U5 |
| Tuberculosis | A15 | 3 | 2 | 3 | DDXPlus severity 3 (fallback) | uncovered | U5 |
| Acute laryngitis | J04.0 | 4 | 3 | 3 | DDXPlus severity 4 (fallback) | uncovered | U5 |
| Acute otitis media | H66.90 | 4 | 3 | 3 | DDXPlus severity 4 (fallback) | 1091 / 0% / 0% | U5 |
| Acute rhinosinusitis | J01 | 4 | 3 | 3 | DDXPlus severity 4 (fallback) | 264 / 0% / 0% | U5 |
| Allergic sinusitis | J30 | 4 | 3 | 3 | DDXPlus severity 4 (fallback) | 116 / 1% / 0% | U5 |
| Anemia | D64.9 | 4 | 3 | 3 | DDXPlus severity 4 (fallback) | 612 / 40% / 4% | U5 |
| Bronchitis | J40 | 4 | 3 | 3 | DDXPlus severity 4 (fallback) | 1045 / 2% / 0% | U5 |
| Localized edema | R60.0 | 4 | 3 | 3 | DDXPlus severity 4 (fallback) | 155 / 8% / 0% | U3 |
| Pericarditis | I30 | 4 | 3 | 3 | DDXPlus severity 4 (fallback) | uncovered | U3 |
| SLE | M32 | 4 | 3 | 3 | DDXPlus severity 4 (fallback) | uncovered | U5 |
| Sarcoidosis | D86 | 4 | 3 | 3 | DDXPlus severity 4 (fallback) | uncovered | U5 |
| Viral pharyngitis | J02.9 | 4 | 3 | 3 | DDXPlus severity 4 (fallback) | 1131 / 0% / 0% | U5 |
| Whooping cough | A37 | 4 | 3 | 3 | DDXPlus severity 4 (fallback) | uncovered | U5 |
| Chronic rhinosinusitis | J32 | 5 | 3 | 3 | DDXPlus severity 5 (fallback) | 182 / 2% / 0% | U5 |
| Panic attack | F41 | 5 | 3 | 3 | DDXPlus severity 5 (fallback) | 901 / 3% / 0% | U5 |
| URTI | J06.9 | 5 | 3 | 3 | DDXPlus severity 5 (fallback) | 2527 / 1% / 0% | U5 |

Conditions only DDXPlus covers (no Newman-Toker row and under 30 NHAMCS visits): Ebola, laryngospasm, acute dystonic reactions, Boerhaave, epiglottitis, Guillain-Barre syndrome, myocarditis, scombroid, bronchiectasis, Chagas, cluster headache, myasthenia gravis, tuberculosis, acute laryngitis, pericarditis, SLE, sarcoidosis, whooping cough (18 conditions, 180 cases). Pulmonary and pancreatic neoplasm have a Newman-Toker row but fewer than 30 NHAMCS visits.

## 6. Agreement with severity and NTS

Quadratic weighted kappa on three tiers. The NTS crosswalk is U1-U2 -> 1, U3 -> 2, U4-U5 -> 3.

| Label | Against primary crosswalk (49) | Against NTS crosswalk (49) | Exact matches with primary |
|---|---|---|---|
| A (Newman-Toker first) | 0.61 | 0.61 | 21 of 49 |
| B (NHAMCS only; 29 conditions) | 0.43 | - | 15 of 29 |
| C (three sources) | 0.60 | 0.53 | 22 of 49 |
| Primary crosswalk itself | - | 0.73 | - |

Where rule A disagrees with the primary crosswalk, all 28 moves but 3 are the crosswalk's own doing: severity 2 conditions become tier 2 (11 conditions: dystonic reactions, Boerhaave, croup, epiglottitis, GBS, myocarditis, PSVT, scombroid, pneumothorax, stable angina, unstable angina) and severity 3 conditions become tier 3 (14). The three external moves are pneumonia, lung cancer and pancreatic neoplasm, up to tier 1. PE stays tier 1 under both.

The conditions the user named:

| Condition | DDX sev | Newman-Toker | NHAMCS (n, admit, ICU) | Move under A | Move under B or C |
|---|---|---|---|---|---|
| Pneumonia | 3 | row: harm 4.6% | 1186, 44%, 5.8% | up to 1 | C: 1; B: 2 |
| Pulmonary neoplasm | 3 | row: harm 14.2% | 28, uncovered | up to 1 | C: 1 |
| Pancreatic neoplasm | 3 | "Other cancers" 7.4% | 6, uncovered | up to 1 | C: 1 |
| Atrial fibrillation | 3 | none | 479, 52%, 10.2% | none | B and C: 1 (ICU >= 10%) |
| Acute COPD exacerbation | 3 | none | 579, 38%, 4.4% | none | B and C: 2 |
| HIV (initial) | 3 | none | 37, 45%, 0% | none | B and C: 2 |
| Tuberculosis | 3 | none | 0 | none | uncovered |
| GERD, inguinal hernia, influenza | 3 | none | 286/105/851; admit 2-16% | none | B: 3 |
| Cluster headache, Chagas | 3 | none | 17 / 0 | none | uncovered |
| Stable angina | 2 | none (MI row only) | 86, 57%, 1.1% | none (tier 2 by crosswalk) | B and C: 2 |
| Scombroid | 2 | none | 1 | none | uncovered |

So the external sources cannot demote stable angina or scombroid, which docs/ddxplus-severity-validation.md rated over-severe on evidence; only our own reference level does that. They also cannot promote COPD exacerbation or AF on harm-if-missed grounds; NHAMCS promotes AF on critical-care admission, a disposition measure.

## 7. Weight of a correct diagnosis against flagging danger

What the literature measures:

| Study | Setting | What it says about harm | Bears on |
|---|---|---|---|
| Newman-Toker 2023 (PMC10792094) | US, all settings, 15 dangerous diseases | Serious harm 4.4% per case against error 11.1% per case: about 4 in 10 diagnostic errors in dangerous disease end in death or permanent disability (0.53 for MI, 0.53 for VTE, 0.48 for pneumonia, 0.63 for lung cancer, from Table 1 rows) | Harm when a dangerous condition is not surfaced |
| Pope 2000 NEJM (PMID 10770981) | 10 US EDs, 10,689 patients | 2.1% of MI and 2.3% of unstable angina mistakenly discharged; risk-adjusted mortality ratio 1.9 (95% CI 0.7-5.2) for MI, 1.7 (0.2-17.0) for UA | Harm of a missed flag for one tier-1 condition; wide intervals |
| Hautz 2019 Scand J Trauma Resusc Emerg Med (PMID 31068188) | 755 admitted ED patients, Switzerland | Discharge diagnosis differed substantially from admission diagnosis in 12.3%; mortality 8.6% vs 3.8% (OR 2.40, 95% CI 1.05-5.5); longer stay | Harm of a wrong diagnosis in a patient who was escalated |
| Zwaan 2010 Arch Intern Med (PMID 20585065) | 7,926 Dutch records | Diagnostic adverse events in 0.4% of admissions, 83.3% preventable, more severe (higher mortality) than other adverse events | Diagnosis errors are the high-severity class |
| Schiff 2009 (PMID 19901140) | 583 physician-reported errors | 28% major, 41% moderate, 31% minor harm | Distribution of harm by error, not by type |
| Singh 2013 (PMC3690001) | 190 primary-care errors | 86.8% rated moderate-to-severe potential harm; the missed diagnoses are common conditions (pneumonia 6.7%, CHF 5.7%) | Primary-care misses of "tier 2" conditions still carry harm |
| Cheraghi-Sohi 2021 (PMC8606447, CC BY) | 2,057 English GP consultations | Missed diagnostic opportunities in 4.3%; 37% of them with moderate-to-severe avoidable harm | Same |
| Auerbach 2024 JAMA Intern Med (PMID 38190122) | 2,428 inpatients who died or went to ICU | Diagnostic error in 23.0%; contributed to harm in 17.8%; to death in 6.6% of those who died | Inpatient, after escalation |

What clinical AI evaluations do:

- Semigran 2015 (BMJ, PMID 26157077) reports diagnosis (correct first 34%, top 20 58%) and triage (57% appropriate) as separate outcomes, with no composite.
- Gilbert 2020 (BMJ Open, PMID 33328258) reports top-3 condition accuracy and "safe urgency advice" separately; GPs 82.1% and 97.0%.
- HealthBench (arXiv 2505.08775) scores physician-written rubric criteria with physician-assigned weights across themes including emergencies; it publishes no diagnosis-against-safety ratio.

Assessment. No study measures the harm of "dangerous possibility flagged, specific diagnosis wrong" against "dangerous possibility not flagged" in one population. The two closest numbers are different constructs: Newman-Toker's 0.40-0.63 serious harms per error is death or permanent disability per missed dangerous diagnosis across settings; Hautz's 4.8 percentage-point mortality excess is in-hospital death per wrong admission diagnosis among admitted patients. Reading them together says the harm of a wrong diagnosis after escalation is about an order of magnitude below the harm of a missed dangerous diagnosis, with wide uncertainty (Hautz OR CI 1.05-5.5; Newman-Toker plausible ranges span a factor of 2-3).

Recommendation: report the two measures separately as primary outputs, as the symptom-checker literature does. If the design needs one number, pre-register the diagnosis credit at one tenth of the tier-1 miss penalty, label it a convention bounded by the two studies above, and report sensitivity at 1/5 and 1/20. Do not present it as an evidence-based weight.

## 8. Supporting documentation: thresholds and time sensitivity (not used for tiers)

The threshold model. Pauker and Kassirer 1975 (NEJM 293:229-234, PMID 1143303) derive a therapeutic threshold: treat if the probability of disease exceeds it, withhold if below. Pauker and Kassirer 1980 (NEJM 302:1109-1117, PMID 7366635) add a testing threshold and a test-treatment threshold, set "from data on the reliability and potential risks of the diagnostic test and the benefits and risks of a specific treatment". The abstracts define the thresholds; the algebraic form p* = C / (B + C) is the standard statement of the 1975 result and was not checked against full text this session.

Published thresholds for our tier-1 conditions:

| Condition | Threshold | Source (verified abstract) |
|---|---|---|
| Pulmonary embolism | Test threshold for D-dimer estimated at 1.8% "using the method of Pauker and Kassirer"; PERC derived on 3,148 ED patients; PE prevalence 1.4% (0.5-3.0%) in rule-negative low-risk patients | Kline 2004 J Thromb Haemost, PMID 15304025 |
| Pulmonary embolism | Gestalt < 15% plus PERC-negative: VTE or death within 45 days 16 of 1,666 (1.0%, 0.6-1.6%), designed to sit below 2.0% | Kline 2008 J Thromb Haemost, PMID 18318689 |
| MI and unstable angina (ACS) | HEART 0-3: MACE 1.7% (n = 2,440; 36.4% of patients); 4-6: 16.6%; 7-10: 50.1% | Backus 2013 Int J Cardiol, PMID 23465250 |
| ACS | HEART 0-3: endpoint risk 2.5%, "supports an immediate discharge" | Six 2008 Neth Heart J, PMID 18665203 |
| ACS | Than 2013 Int J Cardiol 166:752-754 (PMID 23084108), the survey of an acceptable MACE miss rate, has no abstract on PubMed; not verified |
| Pneumonia | CURB-65 0-1 for outpatient care; no probability threshold | Lim 2003, via docs/ddxplus-severity-validation.md |
| Anaphylaxis, pneumothorax, epiglottitis, asthma, COPD | No probability threshold published; management follows clinical criteria or severity bands | via docs/ddxplus-severity-validation.md |

Time-dependent benefit, for the record: STEMI door-to-balloon <= 90 min (ACC/AHA, via the severity doc); untreated PE 10 of 19 recurrent or fatal against 1 of 54 treated (Barritt and Jordan 1960, via the severity doc); septic shock survival falls 7.6% per hour of antimicrobial delay over the first 6 hours (Kumar 2006, PMID 16625125); NICE NG12 2-week pathways for lung and pancreatic cancer (via the severity doc). These would change tiers if used: they put PE and MI above pneumonia, and the cancers well below. That is exactly the dimension the recommended label leaves out (section 10).

## 9. Judgement: is a third-party "dangerous if missed" tier a better label?

| | For | Against |
|---|---|---|
| Newman-Toker tier | Peer-reviewed, per-disease harm rates, the construct the design wants (harm when missed), one table, one rule | 5 of 49 conditions; US claims and hospital data; malpractice severity weights; cannot demote the over-rated severity-2 conditions |
| NHAMCS tier | 29 conditions; public domain; outcome rates on the same record | Measures ED disposition, not harm from missing; ranks anaphylaxis, PSVT and croup lowest; two thresholds of ours |
| DDXPlus severity | All 49; frozen and pre-registered; QWK 0.77 against evidence (severity doc) | Undocumented; 6 lenient and 2 strict conditions at the serious line |
| NTS translation | 49; anchored to a published scale | Our translation; urgency, not harm; superseded by user decision |
| Murtagh or WikEM tier | GP and ED framing per complaint | Unverifiable in full (Murtagh) or a wiki with page-to-page conflicts (WikEM); neither is a dataset |

Recommendation: keep DDXPlus severity as the primary label; add rule A as the named sensitivity row; report Newman-Toker and NHAMCS agreement beside the headline. A third-party "dangerous if missed" tier is the right construct but does not exist at the coverage this benchmark needs.

## 10. Limits

1. The recommended proxy is harm-if-missed only. It ignores time-sensitive treatment benefit and the cost of a work-up, the two terms of the threshold model that decide whether flagging a low-probability condition is worth it (section 8).
2. Newman-Toker's harm rates mix a disease-agnostic harm-per-error rate with malpractice severity weights, and their incidence is hospital discharges; the rates describe US care in 2012-2014.
3. NHAMCS rates are arrival triage and disposition in US EDs, with no variance estimates, and 20 conditions have too few visits.
4. Murtagh's lists were read as index snippets, not pages; WikEM pages change (revision ids recorded above).
5. The De Gruyter, AHRQ and NCBI Bookshelf pages blocked automated fetching; those figures rest on PubMed abstracts, which for the AHRQ report and both Diagnosis papers carry the numbers used here.
6. The diagnosis-weight bound in section 7 combines two constructs from two populations; it supports an order of magnitude, not a coefficient.
7. The coordinator's DXA reliability figure (0 of 7,092) was not re-verified.
