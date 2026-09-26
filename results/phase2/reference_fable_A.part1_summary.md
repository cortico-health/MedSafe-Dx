# Reference review, Part A, cases 0-62: summary

Reviewer: Fable 5.1, blind. Input: `results/phase2/cases_blind.json` entries 0-62 only. Output: `results/phase2/reference_fable_A.part1.jsonl` (63 rows, one per case, in file order). No other file under `results/`, `docs/`, `spec/` or `data/` was opened, and the repo was not searched for case ids.

## Decisions and confidence

| Decision  | Cases | Confidence 3 | Confidence 4 | Confidence 5 |
|-----------|------:|-------------:|-------------:|-------------:|
| ESCALATE  |    49 |            3 |           13 |           33 |
| UNCERTAIN |    14 |           14 |            0 |            0 |
| ROUTINE   |     0 |            0 |            0 |            0 |
| Total     |    63 |           17 |           13 |           33 |

I gave no ROUTINE decisions because every intake in this slice contains at least one finding that the working diagnosis does not explain. The UNCERTAIN cases are those where the deciding data (airway examination, ECG, blood pressure, chest radiograph) is missing from the intake and the history alone cannot say which way it falls.

Decisions by working diagnosis:

| Working diagnosis  | ESCALATE | UNCERTAIN |
|--------------------|---------:|----------:|
| Anemia             |       14 |         4 |
| Panic attack       |       10 |         1 |
| Bronchitis         |       10 |         1 |
| Pericarditis       |        8 |         0 |
| Acute laryngitis   |        2 |         4 |
| Viral pharyngitis  |        1 |         4 |
| Localized edema    |        2 |         0 |
| SLE                |        2 |         0 |

## Citation verification

The 63 rows use 29 distinct sources; 21 rows also carry a "clinical reasoning" entry where a specific point had no rule or guideline behind it (confidence was set at 3 or 4 in those rows unless a verified source covered the main judgement).

How I verified them, on 2026-09-26:

1. Every DOI (22 sources) was resolved on the Crossref REST API and the returned title, journal and year matched the citation as written: HEART (Six 2008), AHA/ACC chest pain 2021, ESC ACS 2023, ADD-RS (Rogers 2011), ACC/AHA aortic disease 2022, Wells 2000, PERC (Kline 2004), BTS pleural disease 2023, ESC pericardial 2015, ESC syncope 2018, ACC/AHA/HRS syncope 2017, WAO anaphylaxis 2020, Feng scombroid 2016, van Harten dystonia 1999, Guardiani supraglottitis 2010, CURB-65 (Lim 2003), Glasgow-Blatchford (Blatchford 2000), Brinster oesophageal perforation 2004, Leonhard GBS 2019, ESC heart failure 2021, BSG lower GI bleeding 2019, UK malaria guidelines 2016, ESC hypertension 2024.
2. Every guideline URL (NICE CG141, NG12, NG128, CG191, NG84, CG134; SIGN 158; AAFP hemoptysis 2015) was fetched and the page title checked.
3. The specific point used was confirmed from the text where a fetch was possible: Guardiani abstract (stridor, dyspnea, rapid onset predict airway intervention; Europe PMC); Wells abstract (surgery within 4 weeks 1.5, haemoptysis 1.0; Europe PMC); ADD-RS abstract (95.7% sensitivity; Europe PMC); WAO 2020 Table 2 criteria (PMC7607509 full text); SIGN 158 section 9.1 near-fatal asthma risk factors (PDF text); NICE NG12 1.2.4, 1.2.5 and 1.8.1 (page text); NICE NG84 1.1.13 (page text); ESC syncope Table 6 high-risk features and ESC pericarditis admission predictors (publisher page); van Harten summary points (OpenAlex abstract: 7-day window, cocaine as risk factor, IM biperiden).
4. Not confirmed from text, cited on title and DOI only, with the point taken from standard knowledge of the document: BTS pleural 2023 (secondary spontaneous pneumothorax definition), CG191 (CRB65 in primary care), CG141 (Blatchford at first assessment), ESC HF 2021 (orthopnoea and PND as typical symptoms), BSG LGIB 2019 (admission of anticoagulated patients), Lalloo 2016 (test unwell returned travellers), ESC hypertension 2024 (hypertensive emergency definition), Brinster 2004 (delay beyond 24 h raises mortality). The van Harten point about laryngeal dystonia threatening the airway could not be confirmed from the retrievable text and is marked as clinical reasoning in the rows that use it.
5. Two sources I intended to use were dropped because they could not be fetched (NICE CKS is UK-only; the GINA report sits behind a bot wall) and replaced with NG84 and SIGN 158.

WebSearch was unavailable for most of the session (session budget exhausted before this task started), so verification ran through Crossref, Europe PMC, OpenAlex, Semantic Scholar and direct page fetches.

## Recurring patterns in the intakes

1. The working diagnosis is usually orthogonal to the dominant finding. "Anemia" is attached to haematemesis, acute dystonia, anaphylaxis and chest pain; "bronchitis" to obstructive jaundice and to haematemesis with melaena; "panic attack" to crescendo angina in patients with prior MI. In 49 of 63 cases the benign label fails to account for the single most dangerous symptom.
2. Cases come in near-duplicate families that differ only in age or one modifier: isolated stridor (indices 1, 18, 25, 47); supraglottitis (0, 27, 42); ACS with a full risk profile (2, 7, 12, 16, 20, 33, 34, 37, 41); sudden pleuritic pain in COPD smokers with pneumothorax history (6, 26, 29); alcohol plus vomiting plus haematemesis with tearing pain (17, 35, 40, 49); food-allergen anaphylaxis (9, 15, 21, 32) and scombroid (39, 43); antipsychotic dystonia (5, 10); post-viral bilateral weakness (11, 19, 46); palpitations with near-syncope on stimulants (3, 8, 13, 23, 24, 31); pharyngitis with "coughing up blood" in immunosuppressed patients (51-57); melaena on a DOAC (58-62); dialysis heart-failure decompensation (45, 48).
3. Risk profiles are implausibly dense for the stated age: 21-year-olds with prior MI, diabetes, hypertension and hypercholesterolaemia (12, 16, 34); a 25-year-old on dialysis with heart failure and prior MI (48); a 25-year-old with COPD (6). I judged these on the stated history, because the question is what a clinician must do with the intake as given, but noted the age effect in the confidence.
4. Intakes give no vitals, examination or duration, and several symptom phrasings are ambiguous ("noticed a high pitched sound when breathing in" may be wheeze; "been coughing up blood" in a sore-throat case may be pharyngeal). Nearly all UNCERTAIN decisions come from these two gaps.
5. Ages of 100 and 104 appear (0, 47) with otherwise ordinary histories, and a 61-year-old "thinks she may be pregnant" (59). These look like generator artefacts; I recorded them but did not let them drive the decision.
