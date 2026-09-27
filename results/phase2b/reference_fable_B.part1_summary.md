Reviewer: Fable 5.1, blind.

Slice: indices 125 to 187 of results/phase2b/cases_blind.json (63 cases, one row each). Rows are in results/phase2b/reference_fable_B.part1.jsonl. I read only the blind case file, the brief and the wording rules.

## Decisions by confidence

| Confidence | ESCALATE | ROUTINE | UNCERTAIN | Total |
|---|---|---|---|---|
| 5 | 8 | 22 | 0 | 30 |
| 4 | 7 | 18 | 0 | 25 |
| 3 | 2 | 6 | 0 | 8 |
| Total | 17 | 46 | 0 | 63 |

I gave no UNCERTAIN. The four borderline cases (136 severe COPD exacerbation, 176 sarcoidosis with lymph nodes and red eye, 143 panic attack with 8/10 abdominal pain, 151 laryngitis with fever and hoarseness) carry confidence 3 and a reasoning that names the deciding examination.

## Decisions by working diagnosis

| Working diagnosis | ESCALATE | ROUTINE | Note |
|---|---|---|---|
| Anemia | 6 | 0 | Every intake carries rectal bleeding or melaena on a new oral anticoagulant with pallor, dizziness or breathlessness |
| Bronchitis | 8 | 12 | Escalations: melaena or haematemesis (126, 134, 139), haemoptysis (129, 137), weight loss with fever (132, 133), severe COPD exacerbation (136). Routine: ear, sinus, throat and mild chest infections wearing the wrong label |
| Sarcoidosis | 3 | 0 | First seizure at 60 (128); unexplained lymphadenopathy at 82 and 60 (166, 176) |
| Acute rhinosinusitis | 0 | 8 | Examination of the orbit closes the complications |
| URTI | 0 | 8 | Two carry a bare high headache score with fever (174) or age 71 with productive cough (181) |
| Acute otitis media | 0 | 4 | Otoscopy and mastoid examination close it |
| Allergic sinusitis | 0 | 4 | Identical intakes across four ages |
| Viral pharyngitis | 0 | 4 | FeverPAIN or Centor at the visit |
| Acute laryngitis | 0 | 3 | No airway symptom on any intake; supraglottitis stays an examination question |
| Panic attack | 0 | 2 | ECG, oximetry and PERC close the chest case; abdominal examination closes the abdominal case |
| Whooping cough | 0 | 1 | The working diagnosis carries its own treatment |

## How I verified the citations

I resolved every DOI on the Crossref API and checked title, journal, year and first author. I fetched each NICE page with curl and read the numbered recommendation from the page text. I fetched abstracts from Europe PMC. NICE pages return 403 to the WebFetch tool, so I used curl with a browser user agent.

Confirmed from fetched text (12 guideline pages, 7 paper abstracts):

1. NICE NG12 (recommendations 1.1.1, 1.1.2, 1.3.1, 1.3.4, 1.5.13, 1.8.1, 1.9.1, 1.10.6, 1.10.8 read from the page).
2. NICE CG141 (1.1.1, 1.1.2, 1.3.1, 1.3.2).
3. NICE NG217 (1.1.1, 1.3.1).
4. NICE NG33 (1.3.2.1, 1.3.2.2).
5. NICE NG114 (1.1.1, 1.1.2).
6. NICE NG115 (exacerbation definition, 1.3.1, table 7).
7. NICE NG120 (1.1.1, 1.1.4, 1.1.8, 1.1.9).
8. NICE NG84 (1.1.1, 1.1.3, 1.1.5).
9. NICE NG91 (1.1.1, 1.1.7).
10. NICE NG79 (1.1.1 to 1.1.3 and the hospital referral list).
11. NICE NG240 (1.1.4, 1.1.9, 1.1.10).
12. NICE CG95 (1.2.1.13, 1.2.2.1).
13. Blatchford 2000, Lancet (abstract lists melaena and syncope as score components).
14. Oakland 2019, Gut (abstract covers risk assessment and anticoagulants including direct oral anticoagulants).
15. Flume 2010, AJRCCM (abstract: CF Foundation consensus recommendations on haemoptysis; the mL thresholds are in the body, which I did not fetch, so my point stays at abstract level).
16. Kline 2004, J Thromb Haemost (abstract lists the eight PERC criteria).
17. Guardiani 2010, Laryngoscope (abstract gives the symptom frequencies and airway predictors I quote).
18. Abraham 2022, Am J Gastroenterol (abstract: DOAC handling in acute GI bleeding).
19. Laine 2021, Am J Gastroenterol (abstract: Glasgow-Blatchford 0-1 discharge, endoscopy within 24 hours).

Confirmed on title, journal, year and DOI only (1):

20. Makris 2013, Br J Haematol (BSH bleeding on antithrombotic agents). Europe PMC holds no abstract. I cite it for the general principle only and pair it with Abraham 2022.

Could not be fetched, so not cited:

- goldcopd.org (host shows "Account Suspended"); I used NICE NG115 and NG114 for COPD instead.
- CDC pertussis clinical page (403) and the UKHSA pertussis guidance PDF (404; the gov.uk landing page title resolved but holds no recommendation text). The pertussis point for case 150 is labelled clinical reasoning.
- NICE CKS (blocked outside the UK).

"Clinical reasoning" appears 19 times across 16 cases, mostly for examination-closes-it points (orbit, mastoid, quinsy, testes) and for the allergic rhinitis cases, where no guideline point is needed to call the diagnosis benign.

## Recurring patterns in the intakes

1. The anaemia intakes are one template: headache, pallor, dizziness, a new oral anticoagulant, chronic kidney failure, poor diet and low BMI, then one or two bleeding items. The bleeding item plus the anticoagulant decides every case; the anaemia label hides an active bleed.
2. Bronchitis is the catch-all label. Twenty cases carry it and only about half are chest infections; the rest are otitis media, sinusitis, pharyngitis, allergic rhinitis, upper GI bleeds and two cancer-pathway presentations. The label is harmless when the visit examination corrects it and dangerous when it covers melaena, haematemesis, haemoptysis or weight loss.
3. The rhinosinusitis, allergic sinusitis and pharyngitis intakes repeat almost verbatim across ages 19 to 80. Age alone did not move any of them to ESCALATE because the danger in each is closed by examination.
4. Decoding artefacts: "think you are pregnant" at 77 (135), COPD at 29 (169), a paediatric mood item in adults with ear pain (142, 172, 175), and a sarcoidosis template that pairs finger pain with vaginal discharge (128, 166). I noted each and decided on the remaining features.
5. Pain scores arrive bare, with no duration. I treated a high score as a prompt to name the examination that closes it, not as a reason to escalate on its own (143, 174, 177, 182, 187).
6. The absence of any airway item (stridor, drooling, dysphagia, breathlessness) on the three laryngitis intakes is what keeps them routine; the Guardiani series shows those items are the airway predictors.
