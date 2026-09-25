# Which DDXPlus-derived proxy tracks real-world serious risk?

Date: 2026-09-24. Scope: adults in the full DDXPlus test split (109,938 patients aged 18+) and the 470-case `eval-v02-adult` sample. Code: `scripts/analysis/risk_proxy_validation.py`, outputs in `results/analysis/risk_proxy/`. No inference spend. Every published figure below was read on the cited page in this session; where we could not verify a number we say so.

The two proxies for "flag this patient as possibly having a serious condition":

- **(1) Truth**: the true condition has DDXPlus severity <= 2 (`serious` in `evaluator/answer_key_v02.py`).
- **(2) DXA mass**: DXA's differential puts >= t on severity <= 2 conditions. The answer key sums the mass (`p_serious_risk`, threshold 12.5%). The task also words it as "some condition >= t", so we report both readings: **sum** and **max** (the largest single serious probability).

## Summary

1. **DXA's probabilities are calibrated to DDXPlus's synthetic priors, not to real-world pre-test probabilities.** On every presentation DXA's mean probability for the serious target sits within a factor of about 2 of the DDXPlus true rate (chest pain: DXA 19.5% ACS vs DDXPlus 19.3%; dyspnoea: PE 5.3% vs 4.7%), and both are inflated against primary care by 4-16x (ACS in chest pain 5.4x against 3.6%; PE in pleuritic pain 10.6x against 1.1%; anaphylaxis in rash 7.5x against 1%). Against emergency department rates the inflation is 1.5-4x (ACS 1.5x against 13%; PE in dyspnoea 4.4x against 1.2%). DXA is realistic only where the DDXPlus prior happens to match the world: lung cancer in haemoptysis (1.2x against 4.9%), pneumonia in febrile cough (1.4x against 5%), PE in wheeze (0.6x against 5.9%) (section 2).
2. **Proxy (2) as summed mass over-flags every real-world cell by 5-6x.** Across 36 NHAMCS presentation x age cells (2016-2022, 42,000 sampled adult ED visits), the ED rate of a severity <= 2-type diagnosis is 1-76% (median 12%); summed DXA mass >= 12.5% flags 23-100% (median ratio 5.7x, 32 of 36 cells over 2x). It flags 59% of cough visits and 23% of sore-throat visits, whose ED serious-diagnosis rates are 9% and 4%. The mass comes from conditions DXA spreads across everything: Guillain-Barre, myocarditis and dystonic reactions are DXA's top three serious contributors in dyspnoea, and scombroid poisoning is in the top two for sore throat, rash and wheeze (section 3).
3. **The max reading at 20% is the only threshold that is calibrated to ED serious-diagnosis rates, and it misses 61% of serious truths.** "Some severity <= 2 condition >= 20%" gives a median flag/serious-diagnosis ratio of 0.76 with 16 of 36 cells within 2x, and the best cell-level rank correlation with the serious-diagnosis rate (Spearman 0.46 vs 0.37 for the truth and 0.29 for summed mass at 12.5%). But on the eval sample it leaves 97 of 160 serious patients below threshold, because DXA puts under 20% on the true condition in most serious cases. No threshold makes proxy (2) both sensitive and calibrated (sections 3 and 4).
4. **Neither proxy carries the strongest real-world risk signal, which is age.** In NHAMCS the admission rate rises 2.4-14.7x and the serious-diagnosis rate 2.3-14.9x from the 18-39 to the 65+ band within every presentation; the DDXPlus proxies stay flat (0.3-1.0x for the truth, 0.9-1.0x for DXA), because DDXPlus samples conditions with age-independent priors. This caps any DDXPlus-derived proxy's cell-level rank correlation at about 0.5, and it is the same for both proxies (section 3).
5. **Verdict: keep proxy (1) as the anchor for misses, and let DXA excuse escalations only at the max reading >= 20%, or keep the summed 12.5% while stating that it excuses 63% of non-serious patients.** Proxy (1) is the better anchor because its errors are legible and one-directional per condition (it under-flags cough and fever cells, where real-world danger is pneumonia and sepsis, which are severity 3 or off-list), while proxy (2) adds DXA's spread onto implausible serious conditions on top of the same synthetic prior. At max >= 20% the excuse set on the eval sample shrinks from 196 to 32 non-serious patients, and 19 of the 32 are MI at 20-50% in sarcoidosis, SLE and laryngitis, which a clinician would still call a defensible escalation. This agrees with the sibling pilot's K5 key (truth is the target, DXA only excuses). What no data here can settle is the level at which a DXA probability on a wrong-but-serious condition is a real clinical signal rather than a generator artefact; that needs clinician review of the 32 excused cases (section 4).

## 1. Method

**Part 1, published pre-test probabilities.** We define 11 presentation patterns in DDXPlus evidence codes (table in section 2), select the adults with the pattern in the full test split, and compute DXA's mean probability for the matching serious condition(s), the true-condition rate the generator produced, and the flag rate of each proxy. Published real-world rates come from primary care where one exists, else from the ED. Three subagents fetched abstracts and full text from PubMed, PMC, Europe PMC and the CDC; WebSearch was unavailable, so coverage leans on known studies. We re-fetched the ten figures that carry the most weight (Haasenritter 2015, Bhuiya 2010, Jones 2009, van Vugt 2013, Kelly 2016, Gaeta 2007, Zwietering 1998, Abdulmalak 2015, Hendriksen 2015, Berger 2003).

**Part 2, NHAMCS.** We use the CDC NHAMCS ED public-use files 2016-2022 already in `data/external/nhamcs/` (URLs and SHA-256 in `SHA256SUMS.txt`; see `docs/third-party-urgency-source.md` for the files and their limits). A visit belongs to a presentation cell when any of its first three reason-for-visit codes (RFV1-3) is in the cell's code set, and to an age band 18-39, 40-64 or 65+. Real-world outcomes per cell, visit-weighted by `PATWT`: admission (`ADMITHOS`, `OBSHOS` or `TRANOTH`), critical-care admission (`ADMIT = 1`), death in the ED (`DIEDED` or `DOA`), triage immediacy 1-2, and a **serious diagnosis**: any of the five discharge diagnoses matches an ICD-10 code of a severity <= 2 DDXPlus condition (`spec/ddxplus_icd10_map.csv`, relations equivalent and narrower) or one of 21 off-list categories from `spec/ddxplus_offlist_categories.csv` that need care within hours (aortic dissection, GI haemorrhage, sepsis, stroke, intracranial haemorrhage, heart failure, respiratory failure, DVT, cardiac arrest and other arrhythmia, CNS infection, endocarditis, tamponade, pancreatitis, appendicitis and obstruction, deep neck abscess, empyema, airway foreign body, aspiration, shock, tetanus and botulism, viral haemorrhagic fever; the list is `OFFLIST_SERIOUS` in the script). DDXPlus patients map to the same cells by any-listed evidence code and age. Per cell we compare each proxy's flag rate with the real-world rates: Spearman rank correlation across the 36 age-banded cells and across the 12 presentations, the 65+/18-39 ratio, and the flag rate divided by the serious-diagnosis rate.

| Cell | NHAMCS RFV codes | DDXPlus evidence |
|---|---|---|
| chest_pain | 1050.0-1050.3 chest pain, discomfort, burning | E_55 at V_29, V_101, V_55, V_56 (lower, upper, side of chest) or E_14 (chest pain at rest) |
| dyspnoea | 1415.0 shortness of breath; 1420.0 laboured breathing | E_66 or E_64 |
| cough | 1440.0 | E_201 |
| sore_throat | 1455.0-1455.6 | E_97 |
| haemoptysis | 1470.1 coughing up blood | E_45 |
| fever | 1010.0 | E_91 |
| palpitations | 1260.0-1260.3 | E_155 |
| wheeze | 1425.0 | E_214 or E_112 |
| rash | 1860.0 skin rash | E_130 with a colour other than NA |
| heartburn | 1535.0 heartburn and indigestion | E_173 |
| haematemesis | 1580.2 vomiting blood | E_210 |
| swelling | 1035.1 oedema | E_151 |

## 2. Published pre-test probabilities vs DXA vs the DDXPlus generator

Primary comparison per pattern. DXA and DDXPlus rates are over the adults with the pattern in the full test split; the published rate is for the conditions in the second column. Factor = rate / published. Full list of every rate checked, including secondary settings: `results/analysis/risk_proxy/published_rates.csv`.

| Pattern (DDXPlus codes) | N | Conditions compared | DXA mean | DDXPlus true | Published | Setting, source | DXA / pub | DDX / pub | Reading |
|---|---|---|---|---|---|---|---|---|---|
| Chest pain (E_55 chest sites or E_14) | 29,775 | MI, unstable angina | 19.5% | 19.3% | 3.6% (pooled range 1.5-3.6%) | Primary care; Haasenritter 2015 Croat Med J PMID 26526879 ("1.5 to 3.6% (acute coronary syndrome/myocardial infarction)"); 3.6% in Marburg N=1212, PMID 19883149 | 5.4x (13x at 1.5%) | 5.4x | Inflated vs primary care |
| same | | same | 19.5% | 19.3% | 13.0% | US ED 2007-08; Bhuiya 2010 NCHS Data Brief 43 ("from 23.6% in 1999-2000 to 13.0% in 2007-2008") | 1.5x | 1.5x | Near ED rate |
| same | | MI, unstable and stable angina | 24.7% | 27.2% | 14.7% | Primary care; stable IHD 11.1% + ACS 3.6%, PMID 19883149 | 1.7x | 1.8x | Mildly inflated |
| Heartburn (E_173) | 2,077 | MI, unstable angina | 16.2% | 0.0% | none found | No study of burning-quality pain vs ACS. NHAMCS heartburn RFV cell: 6.3% on-list serious diagnosis, 18% any (section 3) | ~2.6x vs ED cell | 0 | DXA inflated vs ED; the generator never produces ACS here |
| Pleuritic pain (E_220) | 7,215 | PE | 11.7% | 29.1% | 1.1% (1/92) | UK ED; Hall 1991 PMID 1854394 ("Only one of the patients had a diagnosis of pulmonary embolus"). 0.8% of all ED chest pain in Le Gal 2020 PMID 32097173 | 10.6x | 26x | Both inflated; generator most |
| Dyspnoea (E_66 or E_64) | 43,522 | PE | 5.3% | 4.7% | 1.2% (12/1007) | ANZ ED ambulance arrivals, median age 74; Kelly 2016 PMID 27658711 table 2 ("12, 1.2 % (0.7-2.1 %)") | 4.4x | 3.9x | Inflated |
| same | | PE or lung neoplasm | 7.9% | 8.3% | 0.5% | Primary care; Viniol 2015 PMID 26498502 citing Okkes ("0.5% (0.3-0.8)") | 16x | 17x | Inflated |
| same | | Pulmonary oedema | 5.6% | 4.7% | 20.3% (cardiac failure) | Kelly 2016 table 2 | 0.3x | 0.2x | Deflated vs an elderly ED cohort |
| Haemoptysis (E_45) | 11,162 | PE | 6.5% | 16.6% | 2.6% | French hospital admissions for haemoptysis; Abdulmalak 2015 PMID 26022949 ("tuberculosis (2.7%), pulmonary embolism (2.6%)") | 2.5x | 6.4x | DXA mildly inflated |
| same | | Lung cancer | 5.9% | 14.7% | 4.9% men, 2.8% women (90-day) | UK primary care, 4,812 first episodes; Jones 2009 BMJ PMID 19679615 table 2 ("4.9 (4.1 to 5.7)", "2.8 (2.1 to 3.7)") | 1.2-2.1x | 3.0-5.2x | DXA realistic; generator inflated |
| same | | TB | 8.5% | 11.1% | 2.7% | Abdulmalak 2015 | 3.2x | 4.1x | Inflated |
| Sore throat, no chest pain or dyspnoea (E_97) | 8,373 | Epiglottitis | 0.2% | 0.0% | incidence 3.1 per 100,000 adults per year; no consultation denominator found | Berger 2003 PMID 14608569 ("increased from 0.88 ... to 3.1 (from 1996-2000)"). NHAMCS sore-throat cell: 0.6% on-list serious diagnosis | - | - | DXA's epiglottitis mass is small; but summed serious mass >= 5% still flags 31% of these patients, from scombroid 2.3%, anaphylaxis 0.6%, MI 0.4% |
| Rash with colour (E_130) | 17,894 | Anaphylaxis | 7.5% | 16.4% | 1% | US ED allergy-related visits 1993-2004, 12.4M; Gaeta 2007 PMID 17458433 ("Anaphylaxis coding was rare (1%)") | 7.5x | 16x | Both inflated |
| Fever with cough (E_91, E_201) | 12,810 | Pneumonia (severity 3) | 7.1% | 11.9% | 5% (140/2820) | European primary care acute cough; van Vugt 2013 BMJ PMID 23633005 ("140 (5%) had pneumonia"); fever subgroup not reported | 1.4x | 2.4x | DXA realistic |
| Palpitations (E_155) | 8,773 | PSVT | 14.4% | 22.6% | 8.8% (any clinically relevant arrhythmia) | Dutch general practice N=762; Zwietering 1998 PMID 9792350 ("8.8% were clinically relevant") | 1.6x (upper bound) | 2.6x | DXA near realistic |
| Haematemesis (E_210) | 2,159 | Boerhaave | 23.7% | 60.7% | about 0 (oesophageal perforation of any cause 3.1 per million per year) | Aburumman 2025 PMID 40854992 | >100x | >100x | Both wildly inflated; yet real haematemesis is serious for a reason DDXPlus lacks (upper GI bleed: 10% in-hospital mortality, Hearnshaw 2011 Gut PMID 21490373) |
| Wheeze (E_214 or E_112) | 8,019 | PE | 3.6% | 0.0% | 5.9% (44/740) | Hospitalised COPD exacerbations; Couturaud 2021 JAMA PMID 33399840 ("pulmonary embolism was detected in 5.9% of patients") | 0.6x | 0 | DXA realistic; generator never produces it |

Not found or not verified: any age-split ACS rate in chest pain; a burning-quality chest pain vs ACS study; pneumothorax and pericarditis shares among pleuritic pain; a sore-throat consultation rate to turn epiglottitis incidence into a per-consultation rate; MI or myocarditis as a share of palpitations; the anaphylaxis share of acute urticaria alone; admission rates of adult wheeze visits (the NHAMCS cells in section 3 fill the last gap).

What the table shows:

- **DXA tracks the generator, not the world.** In 10 of 11 patterns DXA's target probability is within 0.4-2.5x of the DDXPlus true rate. The exceptions are heartburn and wheeze, where the generator never produces the serious target (0%) and DXA gives 16% and 3.6%. DXA is the rule-based system whose priors and likelihoods DDXPlus sampled from, so its probabilities describe the microcosm by construction.
- **The microcosm's serious priors are 4-16x primary care and 1.5-4x the ED.** DDXPlus clamps disease priors to 10-100% and samples one condition per patient from 49; a 3.6% event cannot exist at that rate. So the inflation belongs to DDXPlus, and proxy (1) inherits it exactly as proxy (2) does. Within the benchmark this is a fixed base rate, not a per-patient error.
- **Where DXA is realistic, it is by coincidence of the prior**: lung cancer in haemoptysis, pneumonia in febrile cough, PE in wheeze, and within 2x for PSVT in palpitations.
- **DXA's serious mass often comes from conditions no clinician would rank.** The top four serious contributors by mean mass are, for dyspnoea: Guillain-Barre 6.4%, myocarditis 6.0%, dystonic reactions 5.9%, pulmonary oedema 5.6%; for rash: scombroid 8.3%, anaphylaxis 7.5%, MI 3.6%, unstable angina 2.6%; for haemoptysis: MI 9.4%, unstable angina 6.8%, PE 6.5%. This is why summed mass exceeds 5% in 78% of all adults.

## 3. NHAMCS cells: real outcomes vs proxy flag rates

Per-cell table: `results/analysis/risk_proxy/nhamcs_cells.csv` (and `summary.md`). Selected cells, adults, visit-weighted:

| Cell | Age | NHAMCS n | Admit | ICU | Died | Serious dx, on-list | Serious dx, any | DDX n | (1) truth | (2) sum >= 12.5% | (2) max >= 20% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chest_pain | 18-39 | 2,556 | 5.8% | 0.9% | 0.1% | 1.8% | 5.0% | 12,162 | 57.0% | 98.9% | 17.4% |
| chest_pain | 65+ | 1,896 | 39.7% | 6.3% | 0.4% | 16.0% | 30.9% | 5,214 | 58.8% | 98.5% | 17.9% |
| dyspnoea | 18-39 | 1,699 | 11.7% | 2.0% | 0.2% | 2.5% | 8.2% | 17,330 | 49.2% | 99.9% | 7.7% |
| dyspnoea | 65+ | 2,697 | 52.9% | 10.5% | 0.4% | 10.2% | 42.8% | 8,084 | 44.0% | 99.9% | 6.9% |
| cough | 18-39 | 1,940 | 2.5% | 0.2% | 0.0% | 0.4% | 1.6% | 13,082 | 0.2% | 57.5% | 0.5% |
| cough | 65+ | 1,132 | 34.8% | 6.5% | 0.1% | 5.7% | 24.3% | 6,347 | 0.1% | 60.0% | 0.7% |
| sore_throat | 18-39 | 1,532 | 1.5% | 0.0% | 0.0% | 0.4% | 2.8% | 4,068 | 0.5% | 23.8% | 0.3% |
| haemoptysis | 40-64 | 67 | 14.4% | 1.7% | 0.4% | 3.9% | 10.9% | 4,883 | 15.7% | 91.2% | 14.7% |
| fever | 65+ | 648 | 50.8% | 6.6% | 0.0% | 3.6% | 25.3% | 3,796 | 4.2% | 48.5% | 0.3% |
| palpitations | 65+ | 496 | 36.6% | 7.2% | 0.9% | 11.5% | 52.3% | 1,548 | 40.6% | 98.8% | 26.4% |
| rash | 18-39 | 565 | 3.5% | 0.8% | 0.0% | 1.1% | 2.3% | 7,385 | 27.2% | 78.3% | 13.6% |
| heartburn | 65+ | 50 | 35.9% | 8.2% | 0.0% | 13.3% | 39.5% | 355 | 0.0% | 83.1% | 18.7% |
| haematemesis | 65+ | 34 | 61.9% | 24.4% | 2.3% | 8.1% | 75.5% | 365 | 58.9% | 78.1% | 34.4% |

Rank correlation (Spearman) of proxy flag rate with real-world rate, across the 36 presentation x age cells and across the 12 presentations (all adults):

| Proxy | Admit, cells | Serious dx, cells | Triage 1-2, cells | Admit, presentations | Serious dx, presentations | Triage 1-2, presentations |
|---|---|---|---|---|---|---|
| (1) truth | 0.29 | 0.37 | 0.40 | 0.62 | 0.55 | 0.64 |
| DXA mean serious mass | 0.27 | 0.37 | 0.58 | 0.55 | 0.64 | 0.92 |
| (2) sum >= 5% | 0.18 | 0.19 | 0.42 | 0.41 | 0.44 | 0.68 |
| (2) sum >= 12.5% | 0.23 | 0.29 | 0.56 | 0.48 | 0.60 | 0.88 |
| (2) sum >= 20% | 0.25 | 0.30 | 0.56 | 0.52 | 0.56 | 0.87 |
| (2) sum >= 50% | 0.28 | 0.37 | 0.54 | 0.63 | 0.64 | 0.86 |
| (2) max >= 10% | 0.17 | 0.38 | 0.25 | 0.48 | 0.80 | 0.60 |
| (2) max >= 20% | 0.23 | 0.46 | 0.32 | 0.39 | 0.71 | 0.61 |
| (2) max >= 50% | 0.20 | 0.36 | 0.33 | 0.34 | 0.45 | 0.57 |

Calibration against the ED serious-diagnosis rate, per age-banded cell (flag rate divided by real rate):

| Proxy | Median ratio | Cells within 2x | Over 2x | Under 0.5x | Mean abs difference |
|---|---|---|---|---|---|
| (1) truth | 1.45 | 11 | 14 | 11 | 21 points |
| DXA mean serious mass | 2.34 | 14 | 21 | 1 | 23 points |
| (2) sum >= 5% | 6.41 | 3 | 33 | 0 | 65 points |
| (2) sum >= 12.5% | 5.69 | 4 | 32 | 0 | 59 points |
| (2) sum >= 20% | 5.35 | 5 | 31 | 0 | 54 points |
| (2) sum >= 50% | 1.78 | 9 | 17 | 10 | 27 points |
| (2) max >= 10% | 2.73 | 12 | 24 | 0 | 31 points |
| (2) max >= 20% | 0.76 | 16 | 8 | 12 | 12 points |
| (2) max >= 50% | 0.04 | 3 | 0 | 33 | 17 points |

Age gradient, 65+ cell divided by 18-39 cell:

| Cell | NHAMCS admit | NHAMCS serious dx | (1) truth | (2) sum >= 12.5% | DXA mean |
|---|---|---|---|---|---|
| chest_pain | 6.9x | 6.2x | 1.0x | 1.0x | 1.0x |
| dyspnoea | 4.5x | 5.2x | 0.9x | 1.0x | 1.0x |
| cough | 13.9x | 14.9x | 0.4x | 1.0x | 1.0x |
| fever | 6.2x | 5.7x | 0.5x | 0.9x | 0.9x |
| wheeze | 8.7x | 9.5x | 0.7x | 0.9x | 0.9x |
| heartburn | 14.7x | 12.8x | - (0% both) | 1.0x | 1.0x |
| all 12 cells | 2.4-14.7x | 2.3-14.9x | 0.3-1.0x | 0.9-1.0x | 0.9-1.0x |

What the cells show:

- **Both proxies rank presentations moderately well and age cells poorly.** Across presentations the truth reaches 0.62 with admission and DXA's mean mass 0.92 with triage 1-2; across age cells every proxy falls to 0.2-0.5, because age drives most of the real-world variation and neither proxy sees it. This is a property of DDXPlus, not of either proxy.
- **The truth under-flags where real danger is off the severity <= 2 list.** Cough 65+ (35% admitted, 24% serious diagnosis, 0.1% flagged), fever 65+ (51% admitted, 4% flagged) and heartburn 65+ (36% admitted, 0% flagged) are the cells where proxy (1) sits under half the real rate. Their real-world serious diagnoses are pneumonia with sepsis, heart failure, COPD and MI presenting as indigestion; DDXPlus rates pneumonia and COPD at severity 3 and has no sepsis or heart failure. `docs/ddxplus-severity-validation.md` already flags pneumonia, COPD and asthma exacerbation as the six same-day conditions the cutoff misses.
- **Summed mass over-flags everywhere.** At any threshold from 5% to 20% it flags 23-100% of every cell, against real serious-diagnosis rates of 1-76%; 31-33 of 36 cells sit over 2x. It is not a risk measure but a near-constant, and its rank correlations come from the few low cells (sore throat, cough, fever).
- **The max reading at 20% is the calibrated DXA reading.** It is the only proxy with a median ratio near 1 and the fewest cells beyond 2x in either direction, and the best cell-level rank correlation with the serious-diagnosis rate (0.46). It is still blind to age and it flags only 20% of the eval sample.

## 4. Verdict

**Which proxy tracks real risk?** Neither tracks it well at the cell level (all Spearman under 0.5), for the shared reason that DDXPlus has no age or comorbidity gradient. At the presentation level proxy (1) and DXA's mean mass are similar (0.55-0.64 against the serious-diagnosis rate). Proxy (1) is the safer anchor because its inflation is the generator's fixed base rate (5x primary care, 1.5x the ED, on chest pain) and its misses are a known short list of conditions, while proxy (2) adds DXA's habit of spreading 5-10% onto Guillain-Barre, myocarditis, scombroid and MI in patients with no cardiac or neurological findings.

**Does any DXA threshold make (2) realistic?** As summed mass, no: 50% is the first threshold that brings the median cell within 2x of the ED rate, and it already drops 16 of 160 serious patients on the eval sample and 10 of 36 cells under half. As a single-condition maximum, 20% is calibrated to the ED serious-diagnosis rate (median ratio 0.76, 16 of 36 cells within 2x), but it leaves 97 of 160 serious truths below threshold, so it cannot define misses.

**Hybrid.** (1) for misses; DXA only to excuse escalations of non-serious patients. Real-world rates support a strict excuse threshold: on the eval sample the current summed 12.5% excuses 196 of 310 non-serious patients (63%), including every laryngitis, bronchitis, panic attack and rib fracture case, and their real-world cells (sore throat, cough) carry 4-9% ED serious-diagnosis rates. The max reading at 20% excuses 32 (10%): 19 with MI at 20-50% (sarcoidosis 9, SLE 4, laryngitis 4, others), plus myocarditis 3, anaphylaxis 3, PSVT 2, Guillain-Barre 2, dystonic reactions 2, scombroid 1. Recommendation:

| # | Element | Recommended | Why |
|---|---|---|---|
| 1 | Miss definition | Proxy (1): true condition severity <= 2, unchanged | The only per-patient signal the generator guarantees; DXA under-weights the truth in most serious cases |
| 2 | Excuse rule | A single severity <= 2 condition at DXA >= 20% (max reading), replacing summed mass >= 12.5% | Only reading calibrated to ED serious-diagnosis rates; shrinks excuses from 63% to 10% of non-serious patients and removes the sore-throat, cough and fever excuses that real-world rates do not support |
| 3 | If the summed 12.5% is kept | State on the board that it excuses 63% of non-serious patients and flags 5-6x the ED serious-diagnosis rate in every cell | So readers do not mistake B for a real-world over-escalation measure |
| 4 | Sensitivity rows | Report the excuse set at max >= 10% (128 excused) and >= 50% (1 excused) | The 10-50% band is where the two readings disagree |
| 5 | Age | Add an age modifier to the "reference urgency" sensitivity label, not to the primary label | NHAMCS shows 2-15x age gradients that DDXPlus cannot express; the primary label should stay comparable with published DDXPlus work |

This aligns with `docs/dangerous-if-missed-pilot.md` (K5: the target is the true condition; DXA conditions only form the excuse set, at p >= 10% per condition). The pilot's per-condition 10% and this analysis's per-condition 20% differ by one step; the NHAMCS calibration favours 20% (median ratio 0.76 vs 2.73 at 10%), and the eval-sample excuse set is 32 vs 128. Either is defensible; 10% keeps more of DXA's view, 20% keeps only what ED discharge diagnoses support.

**What stays unanswerable without clinicians.** (a) Whether MI at 20-50% in a DDXPlus sarcoidosis or SLE patient is a genuine chest-pain-driven differential or an artefact of the shared evidence pool: the 32 excused cases need a read. (b) Whether the generator's 5x-inflated serious base rate should be corrected by reweighting the eval sample toward real-world priors, which changes every headline. (c) Which severity-3 and off-list conditions (pneumonia, COPD exacerbation, sepsis-type presentations) belong in the miss set for the cough and fever cells where proxy (1) under-flags the ED by 10-100x.

## 5. Limits

1. **ED population.** NHAMCS visits are people who chose the ED; admission and serious-diagnosis rates for cough or sore throat are far above a GP queue's. The ratios in section 3 therefore flatter proxy (2): against primary care its over-flagging is larger.
2. **RFV is coarser than DDXPlus evidence.** A DDXPlus "chest pain" patient has a site, a character, an intensity and a radiation; an NHAMCS 1050.x visit has one code. Both sides use any-listed matching, so a visit or patient can sit in several cells.
3. **The serious-diagnosis outcome mixes ICD granularity.** NHAMCS codes carry 4 characters; 3-character map codes match whole categories (I21 all MI, J93 all pneumothorax), and stable vs unstable angina under I25.11x cannot be separated. Off-list serious categories are our choice (21 of 106 categories); moving heart failure or COVID-19 across the line changes the fever and cough cells most.
4. **Published rates are single studies, not pooled, for most patterns**, and several are from ambulance or hospital cohorts (Kelly 2016 median age 74; Abdulmalak 2015 admitted patients), which overstate serious shares relative to primary care. Hall 1991 has 92 patients. The factors are order-of-magnitude readings, not estimates.
5. **No age split in the published rates**, so section 2 compares all adults; NHAMCS supplies the age dimension instead.
6. **Triage immediacy is missing for 27-36% of NHAMCS visits per year**; the triage 1-2 column uses triaged visits only.
7. **The literature check ran without WebSearch**, so it leaned on studies named in advance and fetched through PubMed, PMC and Europe PMC; a broader search could add primary-care rates for pleuritic pain, palpitations and haematemesis.

## Files

- `scripts/analysis/risk_proxy_validation.py`: the analysis; caches parsed inputs under `results/analysis/risk_proxy/cache/`.
- `results/analysis/risk_proxy/patterns.csv`, `published_rates.csv`: section 2.
- `results/analysis/risk_proxy/nhamcs_cells.csv`, `cell_correlations.csv`, `calibration.csv`, `age_gradient.csv`: section 3.
- `results/analysis/risk_proxy/summary.md`: all tables rendered.
