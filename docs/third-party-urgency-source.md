Superseded: NTS is not used in v0.2 (user decision 2026-09-23); kept as research record.

# A third-party source for the urgency answer key

Date: 2026-09-23. Scope: the 49 DDXPlus conditions in `spec/acuity_reference_levels.csv`. Code: `scripts/analysis/nhamcs_urgency.py`, outputs in `results/analysis/nhamcs_urgency/`. Every figure below was read on a fetched page or computed by that script in this session; anything we could not verify says so.

## Summary

1. **We recommend the CDC National Hospital Ambulatory Medical Care Survey (NHAMCS) emergency department public-use files, 2016-2022, as the third-party source.** They are the only candidate that is free, public-domain, downloadable without registration, coded in ICD-10-CM, and carry a nurse-assigned 5-level triage immediacy (`IMMEDR`), patient age, and disposition (admission, critical-care unit, death) on the same record. NYU's ED Algorithm is free but leaves 22 of our 49 conditions unclassified and rates avoidability, not urgency. MIMIC-IV-ED and HCUP NEDS are gated (credentialing, purchase). Published Canadian, Australian and English aggregates do not cross diagnosis with triage level (section 1).
2. **One rule derives the key: the condition's level is the visit-weighted mean `IMMEDR` of ED visits with the condition as primary diagnosis, rounded to the nearest integer; conditions with fewer than 30 sampled visits get no level.** Level 1-3 is urgent, which the rounding makes equal to "mean immediacy under 3.5". The same rule gives an age-band level wherever the band has 30 visits (section 3).
3. **The rule covers 29 of 49 conditions (33 if we accept any-listed diagnosis as a fallback).** Sixteen conditions have 7 or fewer US ED visits in seven survey years (Ebola, Chagas, Boerhaave, Guillain-Barre, laryngospasm, tuberculosis, epiglottitis, myocarditis, dystonic reaction, whooping cough, scombroid, sarcoidosis, myasthenia, bronchiectasis, pericarditis, pancreatic neoplasm); a larger public ED survey would not change that (section 4).
4. **Agreement with the existing keys is moderate: quadratic weighted kappa 0.57 against DDXPlus severity (14 of 29 exact, 27 within one) and 0.41 against our NTS translation (5 of 29 exact, 19 within one).** On the urgent line NHAMCS keeps 18 of DDXPlus's 19 urgent conditions and all 13 of the NTS translation's, and marks 4 (DDXPlus) and 9 (NTS) more as urgent: GERD, panic attack, bronchitis, localized edema, anemia, rib fracture, inguinal hernia, HIV and stable angina (section 5).
5. **The main limit is the construct: arrival triage in an ED compresses everything to levels 2-4.** Anaphylaxis lands at 3 (mean 2.58) and pulmonary embolism at 3 (2.61) because the triage nurse scores presentation, not the diagnosis made later, while GERD and panic attack land at 3 because ED patients with those final diagnoses presented with chest pain. Triage immediacy is missing for 27-36% of visits per year. NHAMCS's 4-character diagnosis codes cannot separate stable from unstable angina under I25.11x (section 6).

## 1. Candidates

| Source | What it maps | Age | Licence and access | Validation | Verdict |
|---|---|---|---|---|---|
| **NHAMCS ED public-use microdata** (CDC/NCHS) | Each sampled ED visit: up to 5 ICD-10-CM diagnoses (4 characters, implied decimal), `IMMEDR` triage immediacy 1 immediate, 2 emergent, 3 urgent, 4 semi-urgent, 5 nonurgent, age (1-93, top-coded), sex, disposition flags (`ADMITHOS`, `OBSHOS`, `TRANOTH`, `DIEDED`, `DOA`), `ADMIT` unit (1 = critical care), visit weight `PATWT`, masked strata. ICD-10-CM from survey year 2016 ("the codes used to define injury visits ... were changed to reflect the adoption of ... ICD-10-CM", 2016 documentation). 2015 file is ICD-9-CM | Yes, per visit | US federal statistical data; NCHS data-use restrictions: "Use the data in this dataset for statistical reporting and analysis only", no re-identification or linkage. No registration. Fixed-width ASCII plus Stata (2016-2021) and SAS (2022) files at `ftp.cdc.gov` | The immediacy item is the nurse's ESI-style rating at arrival. Since 2012 NCHS does not impute it; the 2022 documentation warns users to "be careful when combining data across years for trending" | **Recommended.** The only free source with diagnosis, triage level, age and outcome on one record |
| **NYU ED Algorithm, ICD-10 version** (Billings; file "Updated 5.5.25") | Per ICD-10-CM code (75,243 rows): shares Non_Emergent, Emergent/PC treatable, ED care needed/preventable, ED care needed/not preventable, Alcohol, Drug, Injury, Psych, Unclassified. Built from "a sample of almost 6,000 full ED records" and "mapped to the discharge diagnosis" | Age was abstracted when the sample cases were classified, but the output is per code with no age term | Free download from the NYU Wagner page (the server refuses requests without a browser Referer header). Johnston et al. 2017: "the algorithm is free for anyone to download and use". No licence text on the page beyond the site-wide "Copyright and Fair Use" link | Ballard et al. 2010 (Med Care), 2.26 million commercial and 261,091 Medicare members: visits classed emergent had OR 3.37 (95% CI 3.31-3.44) for hospitalisation within 1 day and OR 2.81 (2.62-3.00) for death within 30 days versus non-emergent. Johnston 2017 documents 11.2% to 15.5% unclassifiable visits (2006 to 2012) and patches the code list | Not usable as the key. NYU itself states "the algorithm is not intended as a triage tool". 22 of 49 conditions are 100% Unclassified or Injury (including pulmonary embolism, pneumothorax, all neoplasms, anaphylaxis); see `results/analysis/nhamcs_urgency/nyu_eda_levels.csv` |
| Minnesota revision of the NYU algorithm | Reported in our memory as an ICD-10 update | - | **Not verified.** No page matching "algorithm" or "preventable" under health.state.mn.us was found in the Wayback index, and web search was unavailable in this session | - | Do not cite |
| **MIMIC-IV-ED v2.2** (PhysioNet) | Six tables; `triage.acuity` where "1 indicates the highest severity and 5 indicates the lowest severity"; `diagnosis` with up to 9 ICD-9 or ICD-10 codes per stay, `seq_num` ordering | Yes | "PhysioNet Credentialed Health Data License 1.5.0"; requires credentialing, the Credentialed Health Data Use Agreement 1.5.0 and "CITI Data or Specimens Only Research" training | Single centre (BIDMC) | Restricted; note only. Not used |
| **HCUP NEDS** (AHRQ) | All-payer ED visit records, ICD-10-CM from 2016 (2015 transition year) | Yes | "Purchase HCUP data from the HCUP Central Distributor"; DUA training and signed agreement required; price set per data organisation | - | Paid and gated. The documentation page we fetched does not mention a triage acuity variable. Not used |
| **CIHI NACRS** supplementary tables (2003-04 to 2021-22) | Table 3: visits and median length of stay by triage level (CTAS) x age group x sex x year. Table 4: by main problem (named condition groups such as "Acute myocardial infarction", "Asthma") x age group | Yes, in bands | Excel download from cihi.ca, no registration. Licence text not verified in this session | - | Triage level and diagnosis are separate tables; no cross-tabulation. Not usable |
| AIHW emergency department care (Australia, ATS) | Reported to publish presentations by principal diagnosis and by triage category | - | **Not verified**: aihw.gov.au returned HTTP 403 to every fetch, including via the Wayback Machine | - | Unknown |
| NHS Digital Hospital Accident and Emergency Activity (England, ECDS) | Reported to publish attendances by diagnosis and by acuity | - | **Not verified**: digital.nhs.uk returned HTTP 403 | - | Unknown |

Data files, URLs and checksums (SHA-256, first 16 characters; full values in `data/external/nhamcs/SHA256SUMS.txt`):

| File | URL | SHA-256 |
|---|---|---|
| ED2016-stata.zip | https://ftp.cdc.gov/pub/Health_Statistics/NCHS/dataset_documentation/nhamcs/stata/ED2016-stata.zip | 537832e04aaad594 |
| ed2017-stata.zip | .../stata/ed2017-stata.zip | 1c2ef8fb29e24d6e |
| ED2018-stata.zip | .../stata/ED2018-stata.zip | 78cbde244c3ec6e1 |
| ED2019-stata.zip | .../stata/ED2019-stata.zip | 1c62f4f240390b9f |
| ed2020-stata.zip | .../stata/ed2020-stata.zip | 1c9de2ddd3e3ada0 |
| ed2021-stata.zip | .../stata/ed2021-stata.zip | d1e33d8189077b6a |
| ed2022_sas.zip | https://ftp.cdc.gov/pub/Health_Statistics/NCHS/dataset_documentation/nhamcs/sas/ed2022_sas.zip | f4479e0b72c084b9 |
| ed2016.zip ... ed2022.zip (fixed-width ASCII, not used by the script) | https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHAMCS/ | in SHA256SUMS.txt |
| doc22-ed-508.pdf, doc16_ed.pdf (codebooks) | https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/NHAMCS/ | - |
| NYU ED Algorithm - ICD10 Codes - Updated 5.5.25.xlsx | https://wagner.nyu.edu/files/faculty/NYU%20ED%20Algorithm%20-%20ICD10%20Codes%20-%20Updated%205.5.25.xlsx | 4aaf4c12a16e4b82 |

`data/external/` is not yet listed in `.gitignore`; add it before committing anything else, because the raw files are 300 MB.

## 2. What NHAMCS measures

- **Construct: arrival triage, not diagnosis urgency.** `IMMEDR` is the "Immediacy with which patient should be seen (based on PRF item Triage Level)", recorded by the triage nurse before any diagnosis. Across 2016-2022 the weighted split among triaged visits is 1.6% immediate, 14.5% emergent, 49.1% urgent, 30.5% semi-urgent, 4.3% nonurgent. Conditioning on the final diagnosis therefore asks "how urgent did patients who turned out to have X look when they walked in", which is close to what an intake tool must judge, but it is blind to diagnoses that are time-critical yet look benign at the door.
- **Population: US emergency departments**, not primary care. People who bring GERD or a panic attack to an ED are the subset whose presentation looked like chest pain; their immediacy overstates the condition's urgency in a GP intake queue.
- **Outcomes on the same record.** We also report the weighted admission rate (`ADMITHOS`, `OBSHOS` or `TRANOTH`), critical-care admission (`ADMIT = 1`) and death in the ED (`DIEDED` or `DOA`). Admission measures "needs a hospital bed", which is a different construct again: anaphylaxis is admitted 9% of the time because it is treated and discharged.
- **Sample: 123,040 visit records, 84,148 with a valid immediacy (1-5).** The rest are unknown (18-25% by year), blank, "no triage" or from ESAs that do not triage. NCHS stopped imputing this item in 2012. The 2022 ESA response rate was 40.2%.

## 3. Derivation rule

```
level(condition) = round( sum_v PATWT_v * IMMEDR_v / sum_v PATWT_v )
    over ED visits v, 2016-2022, with DIAG1 in the condition's code set and IMMEDR in 1..5
    undefined when the condition has fewer than 30 sampled visits (unweighted, any IMMEDR)
urgent(condition) = level <= 3
```

Code sets come from `spec/ddxplus_icd10_map.csv`, relations `equivalent` and `narrower` only, as the scoring spec already uses for diagnosis matching. NHAMCS codes carry 4 characters, so a 3-character map code matches on the category and a longer code matches on its first 4 characters. Age bands (0-17, 18-64, 65+) apply the same rule to the band's visits. We chose the rounded mean over the modal level (which is 3 for 22 of 29 conditions) and over threshold rules on P(immediate or emergent) (which need cut points nobody published). Rounding at 3.5 makes the 5-level rule and the binary rule one rule.

## 4. Derived levels for the 49 conditions

Columns: N primary = unweighted visits with the condition as first-listed diagnosis; N any = as any of five diagnoses; Mean IMMEDR, P(1-2), Admit and ICU are weighted, primary scope, all ages; NHAMCS level applies the rule; Fallback uses any-listed diagnosis when N primary is under 30; 18-64 and 65+ apply the rule to the band (level shown only when the band has 30 visits); DDXPlus and NTS are the existing keys. "-" means no level.

| Condition | N primary | N any | Mean IMMEDR | P(1-2) | Admit | ICU | NHAMCS level | Fallback (any dx) | 18-64 (N) | 65+ (N) | DDXPlus | NTS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Acute COPD exacerbation / infection | 579 | 859 | 2.71 | 0.39 | 0.38 | 0.04 | 3 | 3 | 3 (303) | 3 (275) | 3 | 3 |
| Acute dystonic reactions | 1 | 6 | 3.00 | 0.00 | 1.00 | 0.00 | - | - | - (1) | - (0) | 2 | 2 |
| Acute laryngitis | 29 | 44 | 3.56 | 0.02 | 0.03 | 0.00 | - | 4 | - (20) | - (5) | 4 | 5 |
| Acute otitis media | 1091 | 1615 | 3.88 | 0.02 | 0.00 | 0.00 | 4 | 4 | 4 (198) | - (16) | 4 | 5 |
| Acute pulmonary edema | 36 | 119 | 2.44 | 0.51 | 0.73 | 0.24 | 2 | 2 | - (12) | - (22) | 1 | 1 |
| Acute rhinosinusitis | 264 | 403 | 3.64 | 0.05 | 0.00 | 0.00 | 4 | 4 | 4 (201) | - (29) | 4 | 5 |
| Allergic sinusitis | 116 | 318 | 4.05 | 0.02 | 0.01 | 0.00 | 4 | 4 | 4 (45) | - (11) | 4 | 5 |
| Anaphylaxis | 76 | 87 | 2.58 | 0.49 | 0.09 | 0.08 | 3 | 3 | 3 (46) | - (2) | 1 | 1 |
| Anemia | 612 | 1761 | 2.81 | 0.26 | 0.40 | 0.04 | 3 | 3 | 3 (375) | 3 (179) | 4 | 5 |
| Atrial fibrillation | 479 | 1217 | 2.44 | 0.63 | 0.52 | 0.10 | 2 | 2 | 2 (165) | 2 (314) | 3 | 4 |
| Boerhaave | 0 | 0 |  |  |  |  | - | - | - (0) | - (0) | 2 | 1 |
| Bronchiectasis | 3 | 7 | 3.00 | 0.00 | 0.36 | 0.00 | - | - | - (1) | - (2) | 3 | 5 |
| Bronchiolitis | 325 | 424 | 3.33 | 0.11 | 0.19 | 0.02 | 3 | 3 | - (5) | - (1) | 3 | 3 |
| Bronchitis | 1045 | 1513 | 3.32 | 0.09 | 0.02 | 0.00 | 3 | 3 | 3 (723) | 3 (167) | 4 | 5 |
| Bronchospasm / acute asthma exacerbation | 1371 | 3345 | 3.12 | 0.16 | 0.07 | 0.01 | 3 | 3 | 3 (742) | 3 (59) | 3 | 2 |
| Chagas | 0 | 0 |  |  |  |  | - | - | - (0) | - (0) | 3 | 5 |
| Chronic rhinosinusitis | 182 | 366 | 3.60 | 0.02 | 0.01 | 0.00 | 4 | 4 | 4 (136) | - (22) | 5 | 5 |
| Cluster headache | 17 | 22 | 3.07 | 0.08 | 0.00 | 0.00 | - | - | - (14) | - (3) | 3 | 3 |
| Croup | 341 | 407 | 3.34 | 0.15 | 0.03 | 0.01 | 3 | 3 | - (0) | - (0) | 2 | 2 |
| Ebola | 0 | 0 |  |  |  |  | - | - | - (0) | - (0) | 1 | 1 |
| Epiglottitis | 1 | 1 |  |  | 0.00 | 0.00 | - | - | - (1) | - (0) | 2 | 1 |
| GERD | 286 | 1447 | 3.06 | 0.13 | 0.02 | 0.00 | 3 | 3 | 3 (186) | 3 (41) | 3 | 5 |
| Guillain-Barré syndrome | 0 | 0 |  |  |  |  | - | - | - (0) | - (0) | 2 | 2 |
| HIV (initial infection) | 37 | 147 | 3.21 | 0.19 | 0.45 | 0.00 | 3 | 3 | 3 (34) | - (2) | 3 | 5 |
| Influenza | 851 | 1129 | 3.50 | 0.06 | 0.04 | 0.00 | 4 | 4 | 3 (355) | 3 (70) | 3 | 5 |
| Inguinal hernia | 105 | 152 | 3.13 | 0.11 | 0.16 | 0.00 | 3 | 3 | 3 (64) | - (27) | 3 | 5 |
| Larygospasm | 0 | 0 |  |  |  |  | - | - | - (0) | - (0) | 1 | 2 |
| Localized edema | 155 | 323 | 3.04 | 0.21 | 0.08 | 0.00 | 3 | 3 | 3 (92) | 3 (61) | 4 | 3 |
| Myasthenia gravis | 3 | 5 | 2.40 | 0.30 | 0.93 | 0.00 | - | - | - (2) | - (1) | 3 | 5 |
| Myocarditis | 1 | 2 | 2.00 | 1.00 | 1.00 | 1.00 | - | - | - (1) | - (0) | 2 | 2 |
| PSVT | 88 | 144 | 2.22 | 0.72 | 0.19 | 0.05 | 2 | 2 | 2 (54) | - (27) | 2 | 2 |
| Pancreatic neoplasm | 6 | 34 | 3.00 | 0.00 | 0.62 | 0.00 | - | 2 | - (1) | - (5) | 3 | 5 |
| Panic attack | 901 | 2781 | 3.08 | 0.21 | 0.03 | 0.00 | 3 | 3 | 3 (736) | 3 (90) | 5 | 5 |
| Pericarditis | 7 | 15 | 2.68 | 0.32 | 0.37 | 0.14 | - | - | - (4) | - (3) | 4 | 3 |
| Pneumonia | 1186 | 2255 | 2.93 | 0.24 | 0.44 | 0.06 | 3 | 3 | 3 (513) | 3 (456) | 3 | 3 |
| Possible NSTEMI / STEMI | 258 | 505 | 2.10 | 0.76 | 0.89 | 0.16 | 2 | 2 | 2 (121) | 2 (137) | 1 | 1 |
| Pulmonary embolism | 117 | 216 | 2.61 | 0.42 | 0.81 | 0.14 | 3 | 3 | 3 (61) | 3 (56) | 2 | 1 |
| Pulmonary neoplasm | 28 | 115 | 2.63 | 0.52 | 0.56 | 0.02 | - | 2 | - (18) | - (10) | 3 | 5 |
| SLE | 17 | 96 | 3.08 | 0.19 | 0.18 | 0.03 | - | 3 | - (17) | - (0) | 4 | 5 |
| Sarcoidosis | 1 | 2 |  |  | 0.00 | 0.00 | - | - | - (1) | - (0) | 4 | 5 |
| Scombroid food poisoning | 1 | 1 |  |  | 0.00 | 0.00 | - | - | - (1) | - (0) | 2 | 3 |
| Spontaneous pneumothorax | 30 | 42 | 2.60 | 0.55 | 0.50 | 0.01 | 3 | 3 | - (23) | - (5) | 2 | 2 |
| Spontaneous rib fracture | 225 | 353 | 3.16 | 0.17 | 0.18 | 0.03 | 3 | 3 | 3 (117) | 3 (103) | 3 | 5 |
| Stable angina | 86 | 779 | 2.68 | 0.43 | 0.57 | 0.01 | 3 | 3 | 3 (46) | 3 (40) | 2 | 4 |
| Tuberculosis | 0 | 0 |  |  |  |  | - | - | - (0) | - (0) | 3 | 5 |
| URTI | 2527 | 3819 | 3.66 | 0.05 | 0.01 | 0.00 | 4 | 4 | 4 (945) | 3 (84) | 5 | 5 |
| Unstable angina | 84 | 763 | 2.48 | 0.54 | 0.71 | 0.06 | 2 | 2 | 2 (41) | 3 (43) | 2 | 1 |
| Viral pharyngitis | 1131 | 1710 | 3.73 | 0.03 | 0.01 | 0.00 | 4 | 4 | 4 (634) | 4 (34) | 4 | 5 |
| Whooping cough | 1 | 2 | 4.00 | 0.00 | 0.00 | 0.00 | - | - | - (0) | - (0) | 4 | 5 |

Level counts under the rule: 5 at level 2, 17 at level 3, 7 at level 4, none at 1 or 5, 20 uncovered. The fallback adds acute laryngitis (4), pancreatic neoplasm (2), pulmonary neoplasm (2) and SLE (3).

Age bands: 17 conditions have 30 visits in both adult bands. Two differ between bands: URTI (18-64 level 4, 65+ level 3) and unstable angina (18-64 level 2, 65+ level 3). Influenza is level 4 over all ages but 3 in both adult bands, because children pull the all-ages mean down.

## 5. Agreement with the existing keys

Quadratic weighted kappa on 1-5; urgent means level 1-3 in every key (spec/v0.2-scoring.md).

| NHAMCS level against | N | QWK | Exact | Within one | Same side of urgent line | Reference's urgent conditions kept | Reference's non-urgent marked urgent |
|---|---|---|---|---|---|---|---|
| DDXPlus severity | 29 | 0.57 | 14 | 27 | 24 | 18 of 19 | 4 |
| NTS translation (`scale_level`) | 29 | 0.41 | 5 | 19 | 20 | 13 of 13 | 9 |
| DDXPlus severity, fallback scope | 33 | 0.55 | 15 | 31 | 27 | 20 of 21 | 5 |
| NTS translation, fallback scope | 33 | 0.31 | 5 | 20 | 21 | 13 of 13 | 12 |
| (check) DDXPlus severity against NTS translation, all 49 | 49 | 0.66 | 17 | 37 | 34 | 21 of 23 | 13 |

The last row reproduces the 0.66 and 37-of-49 reported in `docs/triage-scale-anchor.md`, which checks the kappa code. Where NHAMCS disagrees, it is almost always more urgent than the NTS translation (9 conditions up, 0 down at the urgent line) and never less urgent than DDXPlus except anaphylaxis (DDXPlus 1, NHAMCS 3). The conditions NHAMCS marks urgent that the NTS translation does not: anemia, bronchitis, GERD, HIV, inguinal hernia, panic attack, rib fracture, stable angina (all level 3) and atrial fibrillation (level 2).

## 6. Limits

1. **Compression.** Every covered condition lands at 2-4. The rule cannot produce level 1 (no condition's ED population averages under 1.5) or level 5 (none averages above 4.5), so it cannot express "call an ambulance" or "self-care". Possible NSTEMI/STEMI is level 2 with 76% of visits immediate or emergent, and anaphylaxis, pulmonary edema and pulmonary embolism sit at 2.4-2.6.
2. **Coverage.** 20 of 49 conditions have fewer than 30 primary-diagnosis visits, 16 of them fewer than 8. These are rare in US EDs by nature, so a larger public ED survey would not fix it. Any key built on NHAMCS needs a documented fallback for them, or they leave the scored sample.
3. **Arrival triage blind spots.** The rating is made before diagnosis, on the ED's own population. Anaphylaxis (3) and pulmonary embolism (3) are under-rated relative to their danger; GERD (3), panic attack (3) and bronchitis (3) are over-rated relative to a primary-care queue.
4. **Missing immediacy.** 27-36% of visits per year have no valid `IMMEDR`, higher in 2020-2022 (unknown 24%). We drop them. NCHS notes the collection changed over the years and does not impute since 2012.
5. **Four-character codes.** The public-use file truncates ICD-10-CM to 4 characters. I25.110 (unstable) and I25.118/119 (stable) both become "I251", so the stable- and unstable-angina visit sets overlap by construction. Any-listed matching is likewise broader than the map intends for 5- and 6-character codes.
6. **No variance.** We use the visit weight only, not the masked strata and PSU markers, so there are no confidence intervals. Small-N levels (30-40 visits: acute pulmonary edema, spontaneous pneumothorax, HIV) can move by one level with a different year range.
7. **Map alias.** The map spells "Larygospasm" as the levels file does; the script matches on that spelling.
8. **NYU EDA.** Beyond its 22 unclassified conditions, its probabilities come from a 1990s New York sample mapped forward through code crosswalks, and its categories mix urgency with avoidability. We list its shares per condition for reference only.
