# Reference assessment, reviewer Fable B, part 2 (entries 188-249)

Blind Part A review of the last 62 entries of `results/phase2/cases_blind.json`, written 2026-09-26. I read only that file, never searched the repo for case ids, and did not open any other file under `results/`, `docs/`, `spec/` or `data/`. Output rows are in `results/phase2/reference_fable_B.part2.jsonl`, one per case in file order.

## Decisions and confidence

| Decision | n | conf 3 | conf 4 | conf 5 |
|---|---|---|---|---|
| ESCALATE | 27 | 15 | 11 | 1 |
| ROUTINE | 18 | 11 | 7 | 0 |
| UNCERTAIN | 17 | 17 | 0 | 0 |
| Total | 62 | 43 | 18 | 1 |

The single confidence-5 call is index 245 (fever within four weeks of West Africa in a person who injects drugs). Confidence is mostly 3 because the intakes carry no vitals, examination or duration, so many calls say what the clinician must check at the visit rather than what the intake proves.

## Citation verification

Every case carries at least one citation; no case rests on "clinical reasoning" alone (it appears 7 times, always beside a guideline). 26 distinct sources were used, each verified on 2026-09-26 as follows.

| How verified | Sources |
|---|---|
| Crossref API resolved the DOI to the stated title, journal and year | Perry 2013 (Ottawa SAH), Wells 2003, Kline 2004 (PERC), Roberts 2023 (BTS pleural), Lim 2003 (CRB-65), Adler 2015 (ESC pericarditis), Blatchford 2000, Oakland 2019 (BSG LGIB), Lalloo 2016 (UK malaria), Delgado 2023 (ESC endocarditis), HerniaSurge 2018, Flume 2009 (CF exacerbations), Do 2019 (SNNOOP10), McDonagh 2021 (ESC HF), Snook 2021 (BSG IDA), Mackie 2020 (BSR GCA), Baddour 2015 (AHA endocarditis) |
| Europe PMC abstract or full text read for the specific point | Lalloo 2016 (points 5-6 quoted), Kline 2004 (eight criteria and 1.4% prevalence), Do 2019 (red-flag list), HerniaSurge 2018 (full text: incarceration risk factors, watchful waiting), Snook 2021 (full text: urgent GI investigation), Oakland 2019 (abstract: DOAC management), Flume 2009 (abstract) |
| WebFetch of the OUP article page returned the quoted text | Adler 2015 (major and minor predictors, admission rule), Mackie 2020 (GCA is a medical emergency, feature list), Delgado 2023 (PWID portal of entry and section 12.6.2) |
| NICE recommendations chapter fetched with curl and the recommendation text extracted | NG253 (1.1.10-1.1.11), NG240 (1.1.4-1.1.6, rash table), NG91 (1.1.1, 1.1.9), NG79 (1.1.9), NG84 (1.1.12-1.1.13), NG120 (1.1.8-1.1.10, 1.1.15), NG158 (two-level Wells table, 1.1.3), NG250 (1.2.1-1.2.2 CRB65), NG115 (Table 7) |
| PDF downloaded and text searched | BHIVA/BASHH/BIA 2020 (indicator-condition table: mononucleosis-like illness, unexplained lymphadenopathy, fever, weight loss, grade 1C) |

Three points are weaker and are marked in the `point` field: BTS pleural 2023, ESC HF 2021 and AHA endocarditis 2015 were verified by title and year only, and the point is the guideline's well-known content rather than retrieved text. Two other checks changed what I cited: NICE NG51 (sepsis) has been replaced by NG253 and NG138 (pneumonia) by NG250, so the current guidelines are cited; NICE CG151 (neutropenic sepsis) was verified but NG253 1.1.10 covers the same point for non-cancer immunosuppression, so CG151 was not needed. GOLD 2025 Figure 4.3 is an image in the PDF and could not be quoted, so COPD disposition cites NG115 Table 7 instead. WebFetch was blocked (403) by nice.org.uk and bmj.com, which is why curl and Europe PMC were used. The web-search budget was exhausted before this task began, so no searches were run; every source was reached by direct URL, DOI or the Europe PMC API.

## Recurring patterns in the intakes

1. The working diagnosis often does not name the illness the intake describes. "Bronchitis" is attached to eight groin-hernia intakes (205, 207-209, 221, 232, 242, 243), to three thunderclap or cluster-type headache intakes (202, 241) and "Anemia" to five more headache intakes (204, 212, 229, 236) and to five HIV-exposure or injecting-drug-use febrile syndromes (203, 233, 245, 246, 249). I judged the intake, not the label, and said so in the reasoning.
2. Immunosuppression with fever appears in nine cases (195 without fever; 214, 218, 220, 222, 225, 235 with fever; 210 with dyspnoea). I applied NG253 1.1.10 consistently: fever plus systemic features (bedbound, rigors, neck pain, lesions) escalates; fever with localised URTI symptoms only is UNCERTAIN; no fever is ROUTINE.
3. Nine "localized edema" intakes (200, 201, 206, 215, 219, 224, 230, 234, 238) carry the same heavy list of systemic causes (heart failure, cirrhosis, nephrotic syndrome, calcium channel blocker, steroids, NSAIDs, lymph node surgery). I scored them on NG158 Wells logic: prior DVT escalates (200, 201, 238), unilateral calf pain or very high pain is UNCERTAIN (206, 224, 230), and symmetric low-pain swelling is ROUTINE (215, 219, 234).
4. Groin-hernia intakes hinge on the item "Been able to pass stools or gas since your symptoms increased". Per the brief, an unlisted item was asked and is absent, so its absence means the patient has not passed stool or gas; combined with "bloated or distended" that reads as possible obstruction (207, 208, 242, 243). I flagged this inversion in each red_flags entry because it is an interpretive step a reader may want to audit.
5. Six headache intakes share the same template (violent periorbital pain, lacrimation, family history of cluster headache, alcohol, vasodilators). I used the Ottawa SAH Rule: instantaneous onset or age 40 or over escalates (202, 204, 212); under 40 with fast but not instantaneous onset is UNCERTAIN (229, 236, 241).
6. Fever after travel (213 Central America, 245 West Africa) and fever in a person who injects drugs (233, 245, 246, 249) were the clearest escalations; malaria and endocarditis are the dangers that no benign working diagnosis can cover.
7. No intake in this slice contains vital signs, examination findings or symptom duration, so PERC, CRB65, Glasgow-Blatchford and the meningitis red-flag combination can only be partially applied. Where a rule needs those data I said what the clinician must measure at the visit and left the decision as UNCERTAIN unless the history alone already crossed a threshold.
