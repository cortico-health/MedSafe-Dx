Adjudicator: Fable 5.1, blind.

Model ID: claude-fable-5-1. I read only `results/phase2b/adjudication_queue.json`, `results/phase2b/spot_check_queue.json`, the brief and the wording rules. I used the web only to verify guideline text (NICE pages by curl, Crossref, Europe PMC).

## Result

| Decision | Count |
|---|---|
| ESCALATE | 8 |
| ROUTINE | 26 |
| UNCERTAIN | 1 |

Spot check: all ten agreed decisions hold, so `adjudications.json` carries the 35 queued cases and nothing more.

## The standard as applied

The brief's standard leaves one question open in most of the queue: what to do when the danger is real but the visit's own examination can find or exclude it. I fixed one test for it, because both reviewers reached for "needs assessment at the visit" as a reason to escalate or to abstain, and the standard treats the visit's ordinary checks as given.

The test: a feature escalates when a normal visit examination would still leave the dangerous condition open, so the guideline's next step is a check beyond the visit (a film, a troponin, a D-dimer, an ultrasound, laryngoscopy, a referral pathway). A feature stays routine when a normal examination closes the danger, because the standard says "depends on the vitals" is not a reason to escalate, and the same holds for "depends on the examination".

The fixed answers per recurring question:

1. Adult inspiratory stridor (cases 1, 2, 37): ESCALATE. The pharyngeal examination does not see the larynx, so stridor mandates laryngeal visualisation. Sideris 2020 and Guardiani 2010 rank stridor among the strongest predictors of airway intervention in adult supraglottitis. Stridor reaches the intake as a categorical yes, not a degraded score.
2. Hoarseness or unilateral throat pain with fever and no airway symptom (151, 168, 177): ROUTINE. Quinsy, deep neck infection and supraglottitis show on examination (trismus, uvular deviation, drooling, stridor, distress), and NICE NG84 1.1.13 refers on those signs. A slow onset score and no dysphagia or breathlessness leave nothing for laryngoscopy to answer.
3. Panic cluster with chest pain (100, 140, 143, 219, 225): the office ECG, vitals and PERC close it under 50 with no cardiac risk factor (ROUTINE); at 50 or over a normal ECG does not exclude NSTEMI, so the AHA/ACC 2021 chest pain guideline's troponin applies (ESCALATE). I set the age line at 50 because PERC uses it and because the panic label carries less weight as pre-test probability of ACS rises. Case 100 (57-year-old man) escalates; the four under 40 do not.
4. Palpitations with near-fainting and no syncope (17, 225): ROUTINE. The ESC 2018 syncope guideline's initial evaluation is history, examination and a 12-lead ECG, all visit checks.
5. Right iliac fossa pain as a bare score (143): ROUTINE. Appendicitis scoring (Alvarado, AIR) needs the examination, fever and bloods the visit supplies; no rule mandates imaging on a pain score with no nausea, vomiting or anorexia.
6. Symmetrical limb weakness with numbness and no time course (41): UNCERTAIN. NICE NG127 1.7.2 and 1.10.2 refer immediately only when the weakness is rapidly progressive, and the intake carries it as a bare yes. The panic label rests on no anxiety history. This is the standard's degraded-feature case.
7. Anaemia with a stated bleeding source (77, 78; spot check 75): melaena or haematemesis escalates, because the Blatchford score counts presentation features and NICE CG141 allows early discharge only at a score of 0 (case 75). Heavy periods and fresh red rectal blood under 50 stay routine: NG12 1.3.1 needs abdominal pain or weight loss with rectal bleeding under 50 before it triggers FIT, the BSG 2019 lower-GI guideline decides place of care on vitals and haemoglobin, and the BSG 2021 iron-deficiency guideline reserves transfusion for severe symptomatic or circulatory compromise that the haemoglobin and vitals show. "Think you are pregnant" is closed by the urine test the standard names as ordinary.
8. COPD or bronchitis with sputum change and wheeze (119, 169, 183, 197, 200): reported significant breathlessness in severe COPD with prior flare-ups escalates, because NICE NG115 table 7 puts severe breathlessness and significant comorbidity on the hospital side (case 119). Without reported breathlessness the visit treats the exacerbation under NG114 and NG115 keeps mild breathlessness at home (ROUTINE).
9. Fever with cough in an older smoker (145, 181, 195, 202, 245): ROUTINE. NICE NG250 makes pneumonia a clinical diagnosis at the visit on focal signs, respiratory rate and saturation, and a CRB65 of 0 or 1 (age 65 or over alone) stays in primary care with safety netting. NG12 1.1.2's urgent chest film needs an unexplained symptom, and an acute infective cough is explained.
10. Headache inside a febrile respiratory illness (174, 181, 208, 245): ROUTINE. NICE NG240 1.1.4 strongly suspects meningitis on fever and headache with neck stiffness or altered consciousness, and the intake lists none. Temporal pain in a multi-site headache list without jaw claudication, visual symptoms or scalp tenderness does not make giant cell arteritis a suspicion. A pain score is a bare number.
11. Severe otalgia with fever, with or without antibiotic failure (175, 182, 187): ROUTINE. Mastoiditis is an examination diagnosis, and NICE NG91 1.1.7 reassesses and changes the antibiotic. The bronchitis mislabel omits an antibiotic switch the ear examination supplies, not a treatment whose delay is dangerous.
12. Sarcoidosis as a working diagnosis with lymphadenopathy (166, 176; spot check 220): ESCALATE. Unexplained lymphadenopathy in an adult meets NG12 1.10.6 and 1.10.8 for a suspected cancer pathway referral, and a sarcoidosis label does not explain the nodes until tissue and imaging do. A red eye adds a slit-lamp examination (ATS 2020). Vaginal discharge at 55 or over presenting for the first time meets NG12 1.5.13 for an urgent ultrasound (case 166).
13. Reported coughing up blood (240; spot check 15): ESCALATE. The chest film is the initial test for haemoptysis (Earwood 2015) and haemoptysis fails PERC, so PE probability must be assessed; a normal pharynx does not prove pseudohaemoptysis.
14. Groin lumps and bilateral leg swelling (238, 239): ROUTINE. Hernia reducibility and limb asymmetry are examination findings. NICE NG158 1.1.2 applies Wells only when DVT is suspected, and an alternative diagnosis at least as likely scores -2; heart failure, cirrhosis and nephrotic kidney disease are three such alternatives.

## Decisions by reviewer pair

The pair is written Fable / Astra.

| Pair | Cases | ESCALATE | ROUTINE | UNCERTAIN |
|---|---|---|---|---|
| ROUTINE / ESCALATE | 17, 77, 140, 143, 151, 174, 175, 177, 182 | 0 | 9 | 0 |
| UNCERTAIN / ESCALATE | 1, 2, 37, 78, 100, 119, 197, 200, 238, 245 | 5 | 5 | 0 |
| ESCALATE / UNCERTAIN | 166, 176, 240 | 3 | 0 | 0 |
| ROUTINE / UNCERTAIN | 145, 168, 169, 181, 183, 187, 195, 208, 225 | 0 | 9 | 0 |
| UNCERTAIN / ROUTINE | 202 | 0 | 1 | 0 |
| UNCERTAIN / UNCERTAIN | 41, 219, 239 | 0 | 2 | 1 |
| Total | 35 | 8 | 26 | 1 |

In every outright split I sided with Fable's ROUTINE, because each of Astra's escalations rested on a check the visit performs (an ECG, an ear or throat examination, an abdominal examination, a blood count). Where Fable abstained and Astra escalated, I escalated the stridor cases, the severe-COPD breathlessness and the 57-year-old's chest pain, and kept the rest routine.

## Spot check

| Index | Case | Agreed | Holds | Why |
|---|---|---|---|---|
| 15 | ddxplus_119324 | ESCALATE | Yes | Haemoptysis at 62 in a smoker with breathlessness: NG12 1.1.1 pathway and a chest film. |
| 49 | ddxplus_47644 | ESCALATE | Yes | Haematemesis after retching with instantaneous pain: Blatchford scoring and endoscopy under CG141. |
| 53 | ddxplus_19177 | ESCALATE | Yes | Crescendo angina with rest pain and prior MI: ECG and troponin under CG95. |
| 75 | ddxplus_45315 | ESCALATE | Yes | Melaena with presyncope scores on Blatchford, so early discharge is not allowed. |
| 84 | ddxplus_36039 | ESCALATE | Yes | Fatigable ptosis and diplopia with significant breathlessness: impending myasthenic crisis needs admission. |
| 108 | ddxplus_111233 | ESCALATE | Yes | Significant breathlessness with prior pericarditis and lupus features: ECG and echocardiogram under the ESC 2015 pericardial guideline. |
| 144 | ddxplus_101550 | ROUTINE | Yes | Acute rhinosinusitis at 19; NG79 refers only on orbital or intracranial signs. |
| 220 | ddxplus_102395 | ESCALATE | Yes | Sarcoidosis label with red eyes and significant breathlessness: slit lamp, chest film and ECG. |
| 223 | ddxplus_85272 | ROUTINE | Yes | Viral URTI at 21 with sick contacts; vitals and auscultation close it. |
| 246 | ddxplus_26554 | ESCALATE | Yes | Weight loss, mucosal ulcers and injecting drug use: HIV and syphilis testing that the anaemia label omits. |

The ten agreed decisions match the standard as applied above, so no spot-check object was appended.

## Sources I added and how I verified them

| Source | Used for | Verification |
|---|---|---|
| Sideris A et al. A systematic review and meta-analysis of predictors of airway intervention in adult epiglottitis. Laryngoscope 2020;130:465-473. doi:10.1002/lary.28076 | Stridor rule (1, 2, 37) | Europe PMC record PMID 31173373; the abstract names stridor, an epiglottic abscess and diabetes as the most reliable features associated with airway intervention across 30 studies and 10,148 patients. |
| NICE NG127 Suspected neurological conditions: recognition and referral, recs 1.7.2 and 1.10.2 | Weakness rule (41) | Fetched the recommendations page by curl; 1.7.2 refers immediately adults with rapidly (within 4 weeks) progressive symmetrical limb weakness, 1.10.2 adults with rapidly progressive (hours to days) symmetrical numbness and weakness. |
| NICE NG84 Sore throat (acute), rec 1.1.13 | Throat rule (151, 168, 177) | Fetched by curl; refer to hospital for severe systemic infection or severe suppurative complications such as quinsy, parapharyngeal or retropharyngeal abscess. |
| NICE NG115 COPD in over 16s, table 7 | COPD rule (119, 169, 183, 197, 200) | Fetched by curl; the table lists breathlessness (mild at home, severe in hospital), significant comorbidity, rapid onset and SaO2 under 90 percent among the hospital factors. |
| NICE NG250 Pneumonia, recs 1.2.1 to 1.2.4 | Fever-and-cough rule (145, 181, 195, 197, 200, 202, 245) | Fetched by curl; pneumonia is a clinical diagnosis in primary care, CRB65 0 stays in primary care, CRB65 1 has primary-care-led care with safety netting among its options, and referral follows signs of a more serious illness. |
| NICE NG12 Suspected cancer, recs 1.1.2, 1.3.1, 1.5.13, 1.8.1, 1.10.6, 1.10.8 | Cancer pathway rules (78, 166, 176, 197) | Fetched by curl; the text matches the points Fable cited, and 1.3.1 requires abdominal pain or weight loss with rectal bleeding under 50. |
| NICE NG158 Venous thromboembolic diseases, rec 1.1.2 and table 1 | Leg swelling (239) | Fetched by curl; Wells applies if DVT is suspected, and an alternative diagnosis at least as likely scores -2. |
| Oakland K et al. Diagnosis and management of acute lower gastrointestinal bleeding: guidelines from the British Society of Gastroenterology. Gut 2019. doi:10.1136/gutjnl-2018-317807 | Rectal bleeding (78) | Crossref confirms title, journal and year; the Europe PMC abstract (PMID 30792244) confirms it covers risk assessment and transfusion triggers. I did not fetch the full text, so the Oakland outpatient threshold is from the guideline's known content, not verified text. |
| Snook J et al. British Society of Gastroenterology guidelines for the management of iron deficiency anaemia in adults. Gut 2021. doi:10.1136/gutjnl-2021-325210 | Symptomatic anaemia (77, 78) | Europe PMC full text (PMC8515119); the text reserves transfusion for severe symptomatic and/or circulatory compromise and describes ambulatory pathways for more severe anaemia. |
| Crossref checks on DOIs the reviewers cited | Guardiani 2010, Kline 2004 (PERC), ESC 2018 syncope, AHA/ACC 2021 chest pain, Sanders 2016 myasthenia, Crouser 2020 ATS sarcoidosis | Each DOI resolves to the cited title, journal and year. |

NICE CKS refuses requests from outside the UK, so I could not verify the CKS admission text Astra recalled for otitis media and anaemia. I did not rely on it.

## Why the reviewers said UNCERTAIN

Fable abstained 14 times, Astra 15 times, and they overlapped on three cases. The reasons fall into four groups.

1. The deciding feature is a visit measurement the intake lacks. Fable used this for 78, 119, 197, 200, 202, 239 and 245 (haemoglobin, pulse, saturation, chest findings), and Astra for 145, 169, 183, 195 and 239 (respiratory rate, oxygenation, physiological severity). The standard treats those checks as given, so I decided each case on whether a normal check would still leave the danger open.
2. A categorical feature arrived without severity or time course. Fable used this for the three stridor cases and for the weakness in 41. I decided stridor on the feature itself, because the guideline attaches laryngoscopy to stridor rather than to its severity, and kept 41 uncertain, because NG127 attaches referral to a tempo the intake cannot give.
3. A secondary cause was raised without its specific features. Astra abstained on giant cell arteritis (181, 208), uveitis (176), disseminated gonococcal infection (166), deep neck infection (168), mastoiditis (187) and the source of the blood (240). I decided each on whether a rule attaches a beyond-visit check to the features present.
4. The evidence is balanced between a benign cluster and a dangerous mimic. Both reviewers used this for 219 (panic cluster with sudden chest pain at 31), Fable for 100 (the same cluster at 57) and 238 (hernias under a bronchitis label), and Astra for 225 and 41. I fixed the age line at 50 for troponin so that 219 and 100 get like decisions for a like cluster, and decided 238 on the examination.

Both reviewers wrote "needs assessment at this visit" as if it settled the direction. Under the standard it settles nothing, because every case gets the visit; the question is what the visit cannot answer.
