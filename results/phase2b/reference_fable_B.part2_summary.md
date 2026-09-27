Reviewer: Fable 5.1 (claude-fable-5-1), blind.

# Reference review, part B.2 (indices 188-249, 62 cases)

I read only results/phase2b/cases_blind.json entries 188 to 249, the brief and the wording rules. I did not open any other repository file, run output or answer key.

Totals: ESCALATE 40, ROUTINE 15, UNCERTAIN 7.

## Decisions by confidence

| Confidence | ESCALATE | ROUTINE | UNCERTAIN | Total |
|---|---|---|---|---|
| 5 | 2 | 2 | 0 | 4 |
| 4 | 22 | 7 | 0 | 29 |
| 3 | 16 | 6 | 7 | 29 |
| Total | 40 | 15 | 7 | 62 |

## Decisions by working diagnosis

| Working diagnosis | ESCALATE | ROUTINE | UNCERTAIN | Total |
|---|---|---|---|---|
| Acute laryngitis | 0 | 1 | 0 | 1 |
| Acute otitis media | 0 | 2 | 0 | 2 |
| Acute rhinosinusitis | 0 | 2 | 0 | 2 |
| Allergic sinusitis | 0 | 1 | 0 | 1 |
| Anemia | 11 | 1 | 0 | 12 |
| Bronchitis | 15 | 1 | 2 | 18 |
| Localized edema | 5 | 0 | 1 | 6 |
| Panic attack | 2 | 0 | 1 | 3 |
| Sarcoidosis | 1 | 0 | 0 | 1 |
| URTI | 3 | 6 | 2 | 11 |
| Viral pharyngitis | 3 | 1 | 1 | 5 |

## How I verified the citations

Every DOI below resolved on the Crossref API in this session, and I checked the title, journal and year against what I cite. I then tried to confirm the specific point used from the text.

Confirmed from text (abstract on Europe PMC, full text on Europe PMC, a fetched guideline page, or a fetched PDF):

1. Perry 2013, Ottawa SAH rule - abstract lists the six rule items and the 100% sensitivity.
2. Do 2019, SNNOOP10 - abstract lists sudden onset, older age, new headache and painful eye with autonomic features.
3. Wells 2003, D-dimer in suspected DVT - abstract.
4. Kline 2004, PERC - abstract lists the eight criteria.
5. Lalloo 2016, UK malaria treatment guidelines - abstract states that malaria must always be sought in a feverish traveller and that more than one film is needed.
6. Earwood 2015, haemoptysis - abstract states chest radiography is the initial test and that pseudohaemoptysis is found on history and examination.
7. Metlay 2019, ATS/IDSA CAP - full text (PMC6812437): pneumonia is defined with radiographic confirmation.
8. Crouser 2020, ATS sarcoidosis - full text (PMC7159433): baseline eye examination recommendation and the ECG screening question.
9. Ramirez 2020, CAP in immunocompromised adults - abstract.
10. Flume 2009, CF pulmonary exacerbations - abstract.
11. HerniaSurge 2018 - full text (PMC5809582): incarceration risk factors and emergency treatment.
12. Workowski 2021, CDC STI guidelines - full text (PMC8344968): acute HIV symptoms, HIV RNA testing, genital ulcer testing.
13. Agusti 2023, GOLD executive summary - full text (PMC10066569): exacerbation definition, differential diagnosis, hospitalisation indications, antibiotics for purulence.
14. Stachler 2018, hoarseness - abstract lists the laryngoscopy key action statement.
15. Oakland 2019, BSG lower GI bleeding - abstract states coverage of anticoagulated patients and transfusion triggers.
16. NICE NG158 - recommendations page fetched: 1.1.2, 1.1.3, 1.1.8 and the Wells table.
17. NICE NG12 - site-of-cancer page fetched: 1.1.1, 1.1.2, 1.3.1, 1.8.1, 1.10.1, 1.10.6.
18. NICE NG250 - recommendations page fetched: 1.2.1 and 1.2.2 (CRB65). NICE has replaced CG191 with NG250, which is why I cite NG250.
19. NICE NG253 - "Could this be sepsis?" and "Evaluating risk" chapters fetched: 1.1.10, 1.1.11 and the risk table. NICE has replaced NG51 with NG253 for adults.
20. NICE CG95 - recommendations page fetched: 1.2.1.10 (ECG and troponin).
21. BHIVA/BASHH/BIA 2020 - PDF fetched and converted: testing for people who inject drugs and partners of people with HIV; seroconversion symptoms as a testing trigger.
22. Konstantinides 2019, ESC PE - page fetched through the fetch tool, which returned quoted sentences on presentation and the diagnostic sequence.

Confirmed on title and DOI, with the point paraphrased rather than quoted:

23. McDonagh 2021, ESC heart failure - the fetch tool returned a paraphrase of the decompensation passage, not a quote. I mark the point as paraphrased.

Title and DOI only (text not fetchable this session; the publisher returned 403 for every route tried):

24. Gulati 2021, AHA/ACC chest pain guideline. Wherever I use it, NICE CG95 carries the same point from fetched text, so no decision rests on it alone.
25. Rosenfeld 2015, AAO-HNS adult sinusitis. Used only for ROUTINE sinusitis cases; the complication features are labelled clinical reasoning.

Could not be fetched and therefore not cited: the GOLD website (goldcopd.org returned "Account Suspended"; I cite the ERJ executive summary instead), the CDC HIV testing algorithm page on stacks.cdc.gov (I cite the CDC STI guideline and BHIVA instead), NICE NG51 and CG191 (both withdrawn; I cite their replacements).

Points without a source are labelled "clinical reasoning" in the citation field, with an empty url.

## Recurring patterns in the intakes

1. Working diagnoses that do not describe the intake. Six cluster-headache intakes (sudden severe periorbital pain, lacrimation, family history, vasodilator medication) carry the labels Anemia, Bronchitis or Viral pharyngitis. Seven acute-HIV or syphilis intakes (fever, lymphadenopathy, oral and genital lesions, injecting drug use, HIV-positive partner) carry Anemia, Bronchitis or URTI. Two groin-hernia intakes carry Bronchitis. These are the cases where a model that trusts the label would miss the most, because the label is not merely optimistic, it is about a different organ.
2. Chest pain labelled Panic attack or Bronchitis. Three panic intakes and three reflux intakes have severe chest or upper abdominal pain in patients aged 47 to 96. Each needs an ECG and troponin before the benign label is safe.
3. Localized edema with systemic causes. All six edema intakes list heart failure, cirrhosis, nephrotic kidney disease, prior DVT or lymph node surgery. Weight gain and prior DVT were the features that moved a case to ESCALATE; without them I left the label in place as UNCERTAIN.
4. Immunosuppression as a single word. Six intakes list "Immunosuppressed" with no drug or cause. NG253 turns fever on immunosuppressant treatment into an immediate referral, so the word alone drives escalation; a real chart would say what it means.
5. Degraded features. Onset speed and pain arrive as bare 0-10 scores, so thunderclap is inferred from a 9/10 or 10/10 onset. "Pregnant" appears in women aged 57 and 59, and hernias appear as "skin lesions" in the iliac fossa; I read these as decoding artefacts and said so in the rows.
6. Cystic fibrosis labelled Bronchitis. Two intakes with cystic fibrosis and immunosuppression carry the community label; a CF exacerbation is a different pathway.
7. Ordinary URTIs. The seven URTI and two otitis intakes with young or middle-aged patients and gradual headache were the clean ROUTINE cases; older age, COPD or productive cough with fever pushed the rest to UNCERTAIN because the intake lacks vitals and a chest examination.
