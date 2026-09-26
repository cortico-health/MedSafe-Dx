# Reference review, Fable B part 1 (cases 125-187)

Blind clinical reference assessment of 63 synthetic intakes, entries 125-187 of `results/phase2/cases_blind.json`. Output: `results/phase2/reference_fable_B.part1.jsonl`. The reviewer read only `cases_blind.json` and did not open any other file under `results/`, `docs/`, `spec/` or `data/`.

## Decisions

| Decision  | Count |
|-----------|------:|
| ROUTINE   | 51 |
| ESCALATE  | 8 |
| UNCERTAIN | 4 |

| Confidence | Count |
|-----------:|------:|
| 3 | 16 |
| 4 | 34 |
| 5 | 13 |

Decisions by working diagnosis:

| Working diagnosis | ROUTINE | ESCALATE | UNCERTAIN |
|---|---:|---:|---:|
| Panic attack (15) | 7 | 7 | 1 |
| Bronchitis (13) | 13 | 0 | 0 |
| Acute laryngitis (8) | 8 | 0 | 0 |
| Viral pharyngitis (6) | 4 | 0 | 2 |
| Acute otitis media (6) | 6 | 0 | 0 |
| Acute rhinosinusitis (5) | 5 | 0 | 0 |
| URTI (3) | 3 | 0 | 0 |
| Chronic rhinosinusitis (2) | 2 | 0 | 0 |
| Allergic sinusitis (2) | 2 | 0 | 0 |
| Localized edema (2) | 0 | 1 | 1 |
| SLE (1) | 1 | 0 | 0 |

The ESCALATE calls are seven panic-attack intakes (indices 127, 130, 131, 132, 133, 134, 136) and one leg-swelling intake (157). The UNCERTAIN calls are 129 (39-year-old with near-syncope), 143 (56-year-old with 9/10 unilateral throat pain), 173 (bilateral oedema with weight gain and prior DVT) and 179 (immunosuppressed 21-year-old with 8/10 sore throat).

## The threshold I applied to the panic-attack intakes

The panic intakes share one template: diffuse chest and abdominal pain, fear of dying, choking, palpitations, peri-oral tingling, derealisation, and a history of anxiety or depression. Nothing in the template discriminates between them except age, sex and comorbidity, and none of them carries vitals, ECG or duration. I used the HEART age bands and the ACC/AHA 2021 rule that panic is a diagnosis made after exclusion:

1. Age under 40, no cardiac risk factor, no syncope: ROUTINE, with an ECG as the minimum investigation (NICE CG113 1.3.6).
2. Age 45 or over with chest pain, or any age with near-syncope plus alcohol excess or diaphoresis: ESCALATE, because ECG and troponin are required before the working diagnosis can stand.
3. Age 35-44 with near-syncope and mostly abdominal pain: UNCERTAIN.

Reviewers who prefer the strict ACC/AHA reading (ECG for all acute chest pain, so every panic intake escalates) will disagree with my ROUTINE calls at 125, 126, 128, 135, 137, 139 and 182; I kept those routine because a HEART score of 0-3 supports discharge and the ECG is a routine-visit test, not an escalation.

## Citation verification

24 distinct sources are cited across the 63 cases; every case has at least one. Verification, all done on 2026-09-26:

1. 15 journal sources were resolved through the Crossref API (title, journal and year matched the citation as written): Six 2008 (HEART), Kline 2004 (PERC), Huffman 2003, Fanaroff 2015, Wells 2003, Rosenfeld 2015 (AAO-HNS sinusitis), Stachler 2018 (AAO-HNS hoarseness), Aringer 2019 (EULAR/ACR SLE), Cornia 2010 (pertussis), Guldfred 2008 (adult epiglottitis), Scadding 2017 (BSACI rhinitis), Lim 2003 (CRB-65), Andersohn 2007 (drug-induced agranulocytosis), Smetana 2002 (temporal arteritis), Gulati 2021 (ACC/AHA chest pain). Three more were resolved but not used (Backus 2013, Freund 2018, Fleet 1996).
2. For 9 of those I also read the abstract through Europe PMC and confirmed the specific figure I cite (HEART 0-3 = 2.5% risk; PERC's eight items and 1.4% prevalence; Huffman's five predictors and 25% prevalence; Fanaroff's 10% ACS base rate and LR for bilateral arm radiation; Cornia's LRs of 1.8 and 1.9; Guldfred's 1.9/100,000 incidence and 29% airway rate; Andersohn's list of top drugs; Smetana's warning about visual loss).
3. 9 NICE guidelines were fetched from nice.org.uk with curl and the quoted recommendation was read in the Recommendations chapter: CG113 (1.3.6), NG158 (table 1, 1.1.3), NG106 (1.2.2-1.2.4), NG79 (1.1.9), NG84 (1.1.3, 1.1.13), NG91 (hospital-referral recommendation in section 1.1), NG12 (1.8.1), NG115 (1.3.1, table 7), NG120 (1.1.1, 1.1.4).
4. Points I state from professional knowledge of a verified source, without reading the full text this session: the AAO-HNS sinusitis 10-day and "double worsening" rule, the AAO-HNS hoarseness 4-week laryngoscopy statement, the BSACI clinical-diagnosis criteria, the EULAR/ACR ANA 1:80 entry criterion, and Smetana's individual likelihood ratios for jaw claudication and diplopia. These are well-known headline recommendations, but a checker who wants full certainty should confirm them against the full text.
5. Not obtained: the GOLD 2025 report (site reachable, PDF link 404) and the BNF/NICE CKS pages (403). I cited NICE NG115 for COPD exacerbations and Andersohn 2007 for agranulocytosis in their place.

Web search was unavailable (session budget exhausted before this task started), so all verification went through Crossref, Europe PMC and direct fetches.

## Patterns in the intakes

1. Template-driven presentations. Each working diagnosis maps to a near-identical symptom set. The panic template (15 cases) varies only in age, sex, pain locations and the anxiety-history items; the sinusitis template (10 cases across acute, chronic, bronchitis and pharyngitis labels) is facial pain plus anosmia, purulent or clear discharge, polyps, deviated septum and asthma; the laryngitis template is a recent cold plus hoarseness; the otitis template is ear pain "despite current oral antibiotics". Discrimination between cases therefore rests almost entirely on age and comorbidity rather than on the symptoms.
2. No vitals, examination or duration. Every guideline rule I cite (HEART, PERC, CRB-65, NG115 table 7, NG79 and NG84 referral signs, Wells) needs at least one observation the intake lacks, so most decisions are stated as "the working diagnosis is adequate provided X is normal at the visit".
3. Working-diagnosis mismatches between benign labels. Several intakes carry a bronchitis or pharyngitis label but describe rhinosinusitis (150, 159, 186), allergic rhinitis (166, 167, 168), otitis media (149, 152, 178) or pertussis (154). None of these mismatches changes the escalation decision, but they would matter for a benchmark that scores the working diagnosis itself.
4. Pain locations are scattered. "Side of the chest", "iliac fossa", "hypochondrium" and "pubis" are combined freely in the panic intakes; the anatomical spread reads as generated rather than clinical and I did not weight individual locations heavily.
5. Immunosuppression appears as a bare flag on five sore-throat intakes (148, 155, 171, 179, 183) with no drug or cause. I treated it as requiring a same-day full blood count in every case, and it pushed 179 to UNCERTAIN because of the 8/10 pain.
6. Age extremes with chronic-disease labels: COPD at age 28 (169) and 31 (138), which reads as a label artefact; I noted it in the reasoning without changing the decision.

## Process note

The session scratchpad is shared with a sibling reviewer, whose extract of entries 63-124 overwrote my first scratch file. I saw those blind intakes (they are the same input file, no labels) before re-extracting my own slice to a uniquely named file. No answer-key material was involved.
