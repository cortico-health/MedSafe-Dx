# Systematic sources for upgrading "dangerous if missed" tiers (v0.3)

Date: 2026-09-24. Scope: the 49 DDXPlus conditions and the 470-case adult sample (`data/test_sets/eval-v02-adult.json`). Companion data: `spec/dangerous_if_missed_tiers_v03.csv`. Base rule and the Newman-Toker upgrade: `docs/dangerous-if-missed-labels.md` and `spec/dangerous_if_missed_tiers.csv`. No scoring code, spec or existing file changed.

Every table row and count below was read this session from the source's full text (PubMed Central or Europe PMC XML) or its PubMed abstract, as marked. Web search was unavailable (session budget spent), so we reached sources by direct URL: PubMed E-utilities for abstracts, Europe PMC and NCBI efetch for full texts. Publisher pages (JAMA, BMJ, Oxford, NCBI Bookshelf, The Doctors Company) blocked automated fetching.

## Summary

1. **Three independent, per-condition, harm-enriched lists exist and are citable: Singh 2013 (US primary care, EHR triggers), Hussain 2019 (England and Wales ED incident reports) and Miyagami 2023 (Japan ED lawsuits).** Each publishes a table of missed diagnoses (section 1). The other candidates either give no per-condition table (Cheraghi-Sohi, Avery, Calder, Sklar, Watari), are abstract-only behind a paywall (Schiff, Gunderson, Kostopoulou, Gandhi, Kachalia), belong to the Newman-Toker family (AHRQ 2022), or were unreachable (CRICO, The Doctors Company, NHS Resolution, MPS).
2. **The pre-registered rule: tier 1 if a Newman-Toker 2023 Table 1 row, or if named by at least 2 of the 3 independent sources; otherwise the severity crosswalk.** One source is too weak because the lists are frequency tables with n=1 rows and benign entries (otitis, hyperlipidemia); three sources would add nothing beyond Newman-Toker (section 2).
3. **The rule upgrades one condition beyond Newman-Toker: bronchospasm / acute asthma exacerbation (severity 3), named by Singh 2013 ("Asthma exacerbation", n=1) and Miyagami 2023 ("Bronchial asthma", 2 of 46 non-trauma claims).** Pneumonia, pulmonary neoplasm and pancreatic neoplasm keep their Newman-Toker upgrade; pneumonia and MI are also named by all 3 independent sources, PE by 2 (section 3).
4. **Tier 1 holds 21 of 49 conditions and 200 of 470 adult cases: 160 by severity alone, +30 from Newman-Toker, +10 from the new rule.** Tier 2 falls to 13 conditions (120 cases); tier 3 stays at 15 (150 cases) (section 4).
5. **Four conditions are candidates, named by one independent source only: atrial fibrillation, anemia, HIV (initial infection) and SLE, all from Singh 2013.** They stay at their base tier. The rule cannot reach conditions that rarely present in US, UK or Japanese practice (Chagas, tuberculosis, Ebola), so absence there is not evidence of safety (sections 5 and 6).

## 1. Sources checked

"Systematic" here means the source drew its cases from a defined population by a stated method (trigger query, incident database, court database, systematic literature search), not by editorial choice.

| # | Source | Design and population | Per-condition list | Harm enrichment | Availability and licence | Verdict |
|---|---|---|---|---|---|---|
| S1 | **Singh et al. 2013, JAMA Intern Med 173(6):418-425, PMC3690001** | Two US health systems (a VA facility and an integrated private system), primary-care visits Oct 2006-Sep 2007. EHR triggers (unplanned hospitalisation within 14 days; return primary-care, ED or urgent-care visit within 14 days) selected records; trained physicians reviewed them. 190 records with diagnostic error | Table 2 "Frequencies of Most Commonly Missed Diagnoses in 190 Unique Patient Records", by site (129 at site A, 61 at site B); 68 unique diagnoses | Trigger-selected sample; potential harm rated 4-8 on an 8-point scale in 86.8% of cases, mode 5 "considerable harm" | Author manuscript on PMC; NCBI efetch note: "available for text mining ... fair use"; publisher copyright (AMA). Full text read via efetch | **Independent source 1** |
| S2 | **Hussain et al. 2019, BMC Emerg Med 19(Suppl 5):77, PMC6894198** | All patient-safety incident reports describing diagnostic error in EDs in England and Wales, National Reporting and Learning System, 2013-2015; 5,412 reports screened, 2,288 analysed | Table 1 "Frequency of commonly reported diagnoses": fracture 1,007 (44%), other/not specified 679 (30%), MI 161 (7%), intracranial bleed 140 (6%), stroke 97 (4%), acute abdomen 77 (3%), PE 34 (2%), ectopic pregnancy 31 (1%), appendicitis 17, ischaemic limb 15, DVT 11, meningitis 11, pneumonia 8 (each <1%) | Reporter-selected incidents; harm graded per report (Table 2: 4-15% death by incident type) | CC BY 4.0. Full text read via Europe PMC | **Independent source 2** |
| S3 | **Miyagami et al. 2023, West J Emerg Med 24(2):340, PMC10047720** | Japan's largest legal database, medical lawsuits 1961-2017 involving the ED: 108 claims, 74 with diagnostic error, 46 non-trauma | Table 3 "Non-trauma related final diagnosis (n=46)": vascular 18 (AMI 5, SAH 5, aortic dissection 4), infection 16 (epiglottitis 4, meningitis 4, peritonitis 3), tumour 0, others 12 (bronchial asthma 2, acute pancreatitis 2, intestinal obstruction 2). Table 4: initial diagnosis "upper respiratory tract infection" (10) ended as epiglottitis 4, meningitis 2, appendicitis, pneumonia, stroke, heat illness 1 each | Litigation-selected; death in 82.4% of diagnostic-error claims | CC BY 4.0. Full text read via Europe PMC | **Independent source 3** |
| S4 | Newman-Toker et al. 2022, AHRQ report 22(23)-EHC043 (ED systematic review), PMID 36574484 | 279 studies; top 15 ED conditions by serious misdiagnosis-related harm: stroke, MI, aortic aneurysm/dissection, spinal cord compression, VTE, meningitis/encephalitis, sepsis, lung cancer, TBI/ICH, arterial thromboembolism, spinal/intracranial abscess, cardiac arrhythmia, pneumonia, GI perforation/rupture, intestinal obstruction | Yes, 15 conditions | Serious harm = permanent disability or death; distribution from malpractice claims and incident reports | Abstract via PubMed; NCBI Bookshelf blocked (CAPTCHA) | Same group and claims data as Newman-Toker 2019/2023: counted as the Newman-Toker family, not as an independent source. Corroboration only |
| S5 | Schiff et al. 2009, Arch Intern Med 169(20):1881-1887, PMID 19901140 | Survey at 20 grand rounds and 2 institutions; 583 physician-recalled errors; 28% major harm | Abstract names the most common: PE 26 (4.5%), drug reactions or overdose 26, lung cancer 23 (3.9%), colorectal cancer 19, ACS 18 (3.1%), breast cancer 18, stroke 15 | Recall-weighted, harm rated by the reporter | Paywalled; abstract only | Convenience sample, not systematic. Corroboration only |
| S6 | Gunderson et al. 2020, BMJ Qual Saf 29(12):1008-1018, PMID 32269070 | Systematic review, 22 studies, 80,026 hospitalised adults; 136 harmful errors described in detail | Abstract names malignancy 15 (11%) and PE 13 (9.6%); "Fourteen diagnoses account for more than half" but the table is paywalled | Harmful errors only | "No commercial re-use" (BMJ); abstract only | Table not readable. Corroboration for PE only |
| S7 | Kostopoulou, Delaney, Munro 2008, Fam Pract 25(6):400-413, PMID 18842618 | Systematic review, 21 papers on diagnostic error or delay in primary care | Abstract: malignancies, MI, meningitis, dementia, iron deficiency anaemia, asthma, tremor in the elderly, HIV | None; the authors say some conditions reflect "a specialist research interest rather than an increased rate of misdiagnosis" | Paywalled; abstract only | Lists research interest, not harm. Not used |
| S8 | Gandhi et al. 2006, Ann Intern Med 145(7):488-496, PMID 17015866 | 307 ambulatory closed claims, 4 insurers; 181 harmful errors, 59% cancer | Abstract names breast (44) and colorectal (13) cancer only | Claims | Paywalled; abstract only | No usable list for our conditions |
| S9 | Kachalia et al. 2007, Ann Emerg Med 49(2):196-205, PMID 16997424 | 122 ED closed claims, 4 insurers; 79 harmful missed diagnoses | Abstract names no diagnoses | Claims; 39% death | Paywalled; abstract only | No usable list |
| S10 | Watari et al. 2020, PLoS One, PMC7398551 | 1,802 Japanese claims, 709 with diagnostic error | Table 2 gives initial-diagnosis categories only (malignant neoplasm, respiratory tract infection, ischaemic heart disease ...), no final diagnoses | Claims | CC BY 4.0; full text read | Category level. Not used |
| S11 | Cheraghi-Sohi et al. 2021, BMJ Qual Saf, PMC8606447 | 2,057 English GP consultations, 89 missed diagnostic opportunities | ICD chapters only | 37% moderate-to-severe avoidable harm | CC BY; full text read | No per-condition list |
| S12 | Avery et al. 2021, BMJ Qual Saf, PMC8606464 | 12 English practices, 74 avoidable significant harms, 45 diagnostic | Incident types only (wrong, delayed non-cancer, delayed cancer) | Significant harm only | CC BY-NC; full text read | No per-condition list |
| S13 | Calder et al. 2015, BMJ Qual Saf, PMC4316869 (ED return visits) | 13,495 ED patients, 923 return visits, 53 adverse events, 15 diagnostic | Discharge diagnostic categories only (Table 1); no missed-diagnosis table | Adverse events | CC BY-NC; full text read | No per-condition list |
| S14 | Sklar 2007 (PMID 17210204) and Aaronson 2020 (PMID 31699427), ED discharge deaths and ICU returns | Themes and process framework | None | Death or ICU | Abstracts | No per-condition list |
| S15 | CRICO/Candello, The Doctors Company, NHS Resolution, Medical Protection Society | Claims analyses | Unknown | Claims | Candello login; The Doctors Company search and article URLs returned no study; NHS Resolution and MPS not indexed in Europe PMC | Not reachable this session |

Why S1-S3 and not the rest: they are the only sources with (a) a published per-diagnosis table, (b) a defined sampling method, (c) a harm-enriched population and (d) readable full text. They also differ in country (US, England and Wales, Japan), setting (primary care, ED incident reports, ED litigation) and detection (EHR triggers with chart review, staff reports, court records), so agreement between two of them is not one dataset's noise.

## 2. The pre-registered rule

**Rule (v0.3).** We set `final_tier = 1` when either holds, and `final_tier = base_tier` otherwise:

1. the condition is a row of Newman-Toker 2023 BMJ Qual Saf Table 1 (a named disease, or "Other Cancers" for pancreatic neoplasm, as agreed in `docs/dangerous-if-missed-labels.md`); or
2. at least 2 of the 3 independent sources (S1 Singh 2013 Table 2; S2 Hussain 2019 Table 1; S3 Miyagami 2023 Tables 3-4) name the condition.

`base_tier` is the severity crosswalk (DDXPlus severity 1-2 = tier 1, 3 = tier 2, 4-5 = tier 3). Absence from a source never lowers a tier.

**"Names" means:** the condition, or the title of its ICD-10 code, appears in the text of a row of the source's per-diagnosis table, at any count. A row that gives only a broader class ("Fracture", "Cancer (primary)", "Cardiac dysrhythmia", "Otitis", "Viral syndrome", "Psychiatric disorder", "Respiratory tract infection") is a category match: we record it but do not count it, because it does not say which member of the class was dangerous. A lumped row that spells out the condition ("Angina/myocardial infarction/acute coronary syndrome") names each condition it spells out. Initial (wrong) diagnoses in S3 Table 4 do not name a missed condition; the final diagnoses in that table do.

**Why the threshold is 2.**

- Not 1: each independent table is a frequency list of everything missed, so it carries n=1 rows from a single chart (Singh: HIV 1, complicated lupus 1, atrial fibrillation 1) and benign entries (otitis 3, hyperlipidemia 1, carpal tunnel 1). One mention cannot separate "dangerous if missed" from "seen in that clinic that year". A 1-source rule would upgrade 4 more conditions (40 cases) on Singh's table alone (section 5).
- Not 3: only MI and pneumonia are named by all three, and both are already tier 1. A 3-source rule would leave the Newman-Toker rule unchanged and make the exercise empty. Miyagami's non-trauma set is 46 claims, so absence there is weak evidence.
- 2: the smallest concordance across two unrelated detection systems in two countries.

**Sensitivity rows to report beside the headline:** (a) Newman-Toker only (the v0.2 rule); (b) the v0.3 rule with a count floor of 2 per row (Singh's "Asthma exacerbation" n=1 then fails, and asthma stays tier 2); (c) a 1-source rule (adds AF, anemia, HIV, SLE).

## 3. Per-condition source rows for every severity 3-5 condition

Only severity 3-5 conditions can move. Severity 1-2 conditions are tier 1 already; for the record, the independent sources name MI (all three), PE (S1, S2), epiglottitis (S3, 4 of 46), and stable and unstable angina (S1's lumped row), and give category matches for acute pulmonary edema (S1 "Decompensated congestive heart failure", 8 + 4), PSVT (S1 "Cardiac dysrhythmia", 3) and Boerhaave (S4 "GI perforation and rupture").

| Condition | Sev | Base | Newman-Toker 2023 Table 1 | S1 Singh 2013 Table 2 | S2 Hussain 2019 Table 1 | S3 Miyagami 2023 Tables 3-4 | Independent count | Final | Status |
|---|---|---|---|---|---|---|---|---|---|
| Pneumonia | 3 | 2 | row "Pneumonia" (error 9.5%, harm 4.6%) | "Pneumonia" site A 9, site B 5 | "Pneumonia" 8 (<1%) | Table 4 final diagnosis "pneumonia" 1 | 3 | 1 | Upgraded (Newman-Toker; also 3 of 3) |
| Pulmonary neoplasm | 3 | 2 | row "Lung Cancer" (error 22.5%, harm 14.2%) | category "Cancer (primary)" 8 + 3 | - | - (tumour 0) | 0 | 1 | Upgraded (Newman-Toker). Corroboration: Schiff 2009 lung cancer 3.9% |
| Pancreatic neoplasm | 3 | 2 | row "Other Cancers" (error 11.1%, harm 7.4%) | category "Cancer (primary)" | - | - | 0 | 1 | Upgraded (Newman-Toker) |
| Bronchospasm / acute asthma exacerbation | 3 | 2 | - | "Asthma exacerbation" site B 1 | - | "Bronchial asthma" 2 of 46 | 2 | 1 | **Upgraded (v0.3 rule)**. Corroboration: Kostopoulou 2008 lists asthma |
| Atrial fibrillation | 3 | 2 | - | "Atrial fibrillation (new onset)" site B 1; category "Cardiac dysrhythmia" site A 3 | - | - | 1 | 2 | Candidate, not upgraded. Category: S4 "cardiac arrhythmia" (family) |
| HIV (initial infection) | 3 | 2 | - | "HIV" site A 1 | - | - | 1 | 2 | Candidate, not upgraded. Kostopoulou 2008 lists HIV |
| Acute COPD exacerbation / infection | 3 | 2 | - | - | - | - | 0 | 2 | Not named |
| Bronchiectasis | 3 | 2 | - | - | - | - | 0 | 2 | Not named |
| Bronchiolitis | 3 | 2 | - | - | - | - | 0 | 2 | Not named (no adult cases) |
| Chagas | 3 | 2 | - | - | - | - | 0 | 2 | Not named |
| Cluster headache | 3 | 2 | - | - ("Migraine", "Basilar migraine" are other conditions) | - | - | 0 | 2 | Not named |
| GERD | 3 | 2 | - | - | - | - | 0 | 2 | Not named |
| Influenza | 3 | 2 | - | category "Viral syndrome" site B 1 | - | - | 0 | 2 | Not named |
| Inguinal hernia | 3 | 2 | - | - ("Hernia" appears only as a chief complaint, Table 3) | - | - | 0 | 2 | Not named |
| Myasthenia gravis | 3 | 2 | - | - | - | - | 0 | 2 | Not named |
| Spontaneous rib fracture | 3 | 2 | - | category "Fracture" 1 + 1 | category "Fracture" 1,007 (hip 29%, cervical spine 14%) | - | 0 | 2 | Category only, not upgraded |
| Tuberculosis | 3 | 2 | - | - | - | - | 0 | 2 | Not named |
| Anemia | 4 | 3 | - | "Symptomatic anemia" site A 7, site B 2 | - | - | 1 | 3 | Candidate, not upgraded. Kostopoulou 2008 lists iron deficiency anaemia |
| SLE | 4 | 3 | - | "Complicated lupus" site B 1 | - | - | 1 | 3 | Candidate, not upgraded |
| Acute otitis media | 4 | 3 | - | category "Otitis" 1 + 2 | - | - | 0 | 3 | Category only |
| Acute laryngitis | 4 | 3 | - | - | - | - | 0 | 3 | Not named |
| Acute rhinosinusitis | 4 | 3 | - | - | - | - | 0 | 3 | Not named |
| Allergic sinusitis | 4 | 3 | - | - | - | - | 0 | 3 | Not named |
| Bronchitis | 4 | 3 | - | - | - | - | 0 | 3 | Not named |
| Localized edema | 4 | 3 | - | - ("Leg edema/swelling" is a chief complaint) | - | - | 0 | 3 | Not named |
| Pericarditis | 4 | 3 | - | - | - | - | 0 | 3 | Not named |
| Sarcoidosis | 4 | 3 | - | - | - | - | 0 | 3 | Not named |
| Viral pharyngitis | 4 | 3 | - | - | - | - | 0 | 3 | Not named |
| Whooping cough | 4 | 3 | - | - | - | - | 0 | 3 | Not named |
| Chronic rhinosinusitis | 5 | 3 | - | - | - | - | 0 | 3 | Not named |
| Panic attack | 5 | 3 | - | category "Psychiatric disorder" 1 + 1 | - | - | 0 | 3 | Category only |
| URTI | 5 | 3 | - | category "Viral syndrome" | - | initial diagnosis only (10 of 46 non-trauma claims began as URTI) | 0 | 3 | Not named as a missed condition |

The S3 Table 4 finding belongs in the benchmark's framing rather than the label: in 10 of 46 non-trauma ED lawsuits the initial diagnosis was URTI and the final diagnosis was epiglottitis (4), meningitis (2), appendicitis, pneumonia, stroke or heat illness. That is the "unsafe reassurance" pattern the scorer measures.

## 4. Final tiers and case counts

Adult sample: 47 conditions with 10 cases each (croup and bronchiolitis have no adult cases).

| Label | Tier-1 conditions (of 49) | Tier-1 cases (of 470) | Tier 2 conditions / cases | Tier 3 conditions / cases |
|---|---|---|---|---|
| Severity crosswalk alone | 17 | 160 | 17 / 160 | 15 / 150 |
| + Newman-Toker (v0.2 rule) | 20 | 190 | 14 / 130 | 15 / 150 |
| + 2-of-3 independent sources (v0.3 rule) | 21 | 200 | 13 / 120 | 15 / 150 |
| Sensitivity: count floor 2 per row | 20 | 190 | 14 / 130 | 15 / 150 |
| Sensitivity: 1 independent source | 25 | 240 | 11 / 100 | 13 / 130 |

Tier 1 under v0.3: acute pulmonary edema, anaphylaxis, Ebola, laryngospasm, MI, acute dystonic reactions, Boerhaave, croup, epiglottitis, Guillain-Barre syndrome, myocarditis, PSVT, PE, scombroid, spontaneous pneumothorax, stable angina, unstable angina (severity 1-2); pneumonia, pulmonary neoplasm, pancreatic neoplasm (Newman-Toker); bronchospasm / acute asthma exacerbation (2 of 3 independent sources).

The change from severity alone is +40 tier-1 cases (8.5% of the sample): 30 from Newman-Toker, 10 from the new rule.

## 5. Candidates not upgraded

| Condition | Named by | Why it stays | What would move it |
|---|---|---|---|
| Atrial fibrillation (sev 3, tier 2) | S1 "Atrial fibrillation (new onset)" n=1; category rows S1 "Cardiac dysrhythmia" 3 and S4 "cardiac arrhythmia" | One independent source; the arrhythmia rows are categories | A second independent source naming AF; or a decision to count S4 category rows |
| Anemia (sev 4, tier 3) | S1 "Symptomatic anemia" 7 + 2 (the sixth most frequent miss at site A) | One source | A second source; Kostopoulou 2008 names iron deficiency anaemia but is excluded (research interest, not harm) |
| HIV (initial infection) (sev 3, tier 2) | S1 "HIV" n=1 | One source, one chart | A second source |
| SLE (sev 4, tier 3) | S1 "Complicated lupus" n=1 | One source, one chart | A second source |

## 6. Possible counterexamples and limits

Severity 3-5 conditions a clinician could call dangerous if missed that no source, or one source, names:

1. Atrial fibrillation: stroke risk; NHAMCS critical-care admission 10.2% (`docs/dangerous-if-missed-labels.md`); one source (S1) plus two category rows.
2. Acute COPD exacerbation: NHAMCS admission 38%; no source names it, although Singh's Table 2 has no COPD row at all in a VA population where it must have been common, which suggests the trigger method under-detects known chronic disease decompensation.
3. Tuberculosis and Chagas: dangerous to the patient and to others, but rare in the three source populations, so they cannot appear.
4. Myasthenia gravis: myasthenic crisis; not named.
5. Pericarditis (sev 4): tamponade; not named.
6. Inguinal hernia: strangulation; not named (S1 lists "Hernia" only as a chief complaint).
7. Anemia and HIV: one source each (section 5).
8. Whooping cough (sev 4): transmission to infants; not named.
9. Spontaneous rib fracture and acute otitis media: category matches only ("Fracture", "Otitis").

Limits of the rule:

1. The three independent sources are frequency lists of harmful misses, not harm-per-miss rates. A condition is named when it is both dangerous and common in that population, so the rule can only upgrade conditions common in US, UK or Japanese practice. Absence never lowers a tier, but absence is also not evidence of safety.
2. S1 counts are per site (129 and 61 records) and include n=1 rows; the asthma upgrade rests on one Singh chart plus two Japanese lawsuits. The count-floor sensitivity row (section 4) shows the label without it.
3. S3 covers 1961-2017 and 46 non-trauma claims; litigation selects deaths (82%) and settled cases.
4. S2 is voluntary staff reporting; 30% of reports gave no diagnosis, and fractures dominate because imaging misreads are easy to report.
5. The category-match rule is ours. Counting S4's "cardiac arrhythmia" or S1's "Cardiac dysrhythmia" would upgrade AF and PSVT; counting "Fracture" would upgrade rib fracture. We chose not to count categories because they do not say which member was dangerous.
6. Schiff 2009, Gunderson 2020, Kostopoulou 2008, Gandhi 2006 and Kachalia 2007 were read as abstracts only; their full tables might name more of our conditions. The AHRQ 2022 report was read as its PubMed abstract, which carries the 15-condition list.
7. CRICO, The Doctors Company, NHS Resolution and MPS claims breakdowns were not reachable; if one publishes a per-diagnosis table it could replace S3 as the claims source.
8. The rule does not weight time sensitivity or treatment benefit, as before (`docs/dangerous-if-missed-labels.md`, section 8).
