Reviewer: Fable 5.1, blind.

# Reference review, slice A part 1 (cases 0-62)

I read only results/phase2b/cases_blind.json entries 0 to 62, the brief and the wording rules. I did not open any other repository file. This file records the decisions, how I checked the sources, and the patterns in the intakes, so the study team can weigh the reference before they compare it with model output.

Totals: ESCALATE 58, ROUTINE 1, UNCERTAIN 4 (63 cases).

## Decisions by confidence

| Confidence | ESCALATE | ROUTINE | UNCERTAIN | Total |
|---|---|---|---|---|
| 3 | 3 | 1 | 4 | 8 |
| 4 | 20 | 0 | 0 | 20 |
| 5 | 35 | 0 | 0 | 35 |

## Decisions by working diagnosis

| Working diagnosis | ESCALATE | ROUTINE | UNCERTAIN | Total |
|---|---|---|---|---|
| Acute laryngitis | 3 | 0 | 3 | 6 |
| Anemia | 16 | 1 | 0 | 17 |
| Bronchitis | 13 | 0 | 0 | 13 |
| Localized edema | 1 | 0 | 0 | 1 |
| Panic attack | 15 | 0 | 1 | 16 |
| Pericarditis | 7 | 0 | 0 | 7 |
| SLE | 1 | 0 | 0 | 1 |
| Viral pharyngitis | 2 | 0 | 0 | 2 |

## How I verified the citations

I used 25 distinct sources. I resolved every DOI on the Crossref API and checked title, journal, year, volume and pages. I fetched every guideline page with curl and checked the page title. Where a fetch returned text, I searched it for the recommendation I quote. Where the publisher blocked the full text (academic.oup.com and thorax.bmj.com return 403, the escardio.org landing pages hold only summaries), I cite the source on title and DOI only and say so inside the evidence point, and the case confidence takes that into account. Points I could not ground in a source at all are labelled "clinical reasoning" in the citation field.

Confirmed from fetched text or abstract (13):

1. NICE NG12 Suspected cancer: recognition and referral (2015, updated) - https://www.nice.org.uk/guidance/ng12/chapter/Recommendations-organised-by-site-of-cancer
2. NICE NG158 Venous thromboembolic diseases: diagnosis, management and thrombophilia testing (2020) - https://www.nice.org.uk/guidance/ng158/chapter/Recommendations
3. NICE CG95 Recent-onset chest pain of suspected cardiac origin: assessment and diagnosis (2010, updated 2016) - https://www.nice.org.uk/guidance/cg95/chapter/Recommendations
4. NICE CG141 Acute upper gastrointestinal bleeding in over 16s: management (2012, updated) - https://www.nice.org.uk/guidance/cg141/chapter/Recommendations
5. NICE NG250 Pneumonia: diagnosis and management (2025) - https://www.nice.org.uk/guidance/ng250/chapter/Recommendations
6. BTS/SIGN 158 British guideline on the management of asthma (2019 revision) - https://www.sign.ac.uk/media/1773/sign158-updated.pdf
7. Cardona V et al. World Allergy Organization Anaphylaxis Guidance 2020. World Allergy Organ J 2020;13:100472 - https://doi.org/10.1016/j.waojou.2020.100472
8. Wells PS et al. Derivation of a simple clinical model to categorize patients probability of pulmonary embolism. Thromb Haemost 2000;83:416-420 - https://doi.org/10.1055/s-0037-1613830
9. Sideris A et al. A systematic review and meta-analysis of predictors of airway intervention in adult epiglottitis. Laryngoscope 2020;130:465-473 - https://doi.org/10.1002/lary.28076
10. Guardiani E et al. Supraglottitis in the era following widespread immunization against Haemophilus influenzae type B. Laryngoscope 2010;120:2183-2188 - https://doi.org/10.1002/lary.21083
11. Feng C, Teuber S, Gershwin ME. Histamine (Scombroid) Fish Poisoning: a Comprehensive Review. Clin Rev Allergy Immunol 2016;50:64-69 - https://doi.org/10.1007/s12016-015-8467-x
12. Leonhard SE et al. Diagnosis and management of Guillain-Barre syndrome in ten steps. Nat Rev Neurol 2019;15:671-683 - https://doi.org/10.1038/s41582-019-0250-9
13. Flume PA et al. Cystic fibrosis pulmonary guidelines: pulmonary complications: hemoptysis and pneumothorax. Am J Respir Crit Care Med 2010;182:298-306 - https://doi.org/10.1164/rccm.201002-0157OC

Notes on the text checks: NG12 recommendations 1.1.1, 1.1.2, 1.2.4, 1.2.5 and 1.8.1 were read from the site-of-cancer page. NG158 1.1.2, table 1 and 1.1.3 were read from the recommendations page. CG95 1.2.1.3, 1.2.1.7 and 1.2.2.1 were read from the recommendations page. CG141 1.1.1 was read from the recommendations page. NG250 1.2.1 and 1.4.1 were read from the recommendations page; NICE has withdrawn CG191 in favour of NG250. SIGN 158 was read from the PDF, section 9.1 (risk factors for asthma death, including admission in the last year) and the acute-attack steroid recommendation. The WAO 2020 anaphylaxis criteria (table 2) were read from the Europe PMC full text (PMC7607509). The Wells 2000 abstract (seven items and cut-offs), the Sideris 2020 abstract (stridor, abscess and diabetes predict airway intervention), the Guardiani 2010 abstract (stridor, respiratory distress, rapid onset and dyspnoea predicted airway intervention), the Feng 2016 abstract (scombroid mimics allergy), the Leonhard 2019 abstract (potentially fatal, ten-step management) and the Flume 2010 abstract (CF Foundation consensus on hemoptysis) were read on Europe PMC.

Confirmed on title, journal, year and DOI only (12):

1. Newsome PN et al. Guidelines on the management of abnormal liver blood tests. Gut 2018;67:6-19 - https://doi.org/10.1136/gutjnl-2017-314924
2. Byrne RA et al. 2023 ESC Guidelines for the management of acute coronary syndromes. Eur Heart J 2023;44:3720-3826 - https://doi.org/10.1093/eurheartj/ehad191
3. Gulati M et al. 2021 AHA/ACC/ASE/CHEST/SAEM/SCCT/SCMR Guideline for the Evaluation and Diagnosis of Chest Pain. Circulation 2021;144:e368-e454 - https://doi.org/10.1161/CIR.0000000000001029
4. Konstantinides SV et al. 2019 ESC Guidelines for the diagnosis and management of acute pulmonary embolism. Eur Heart J 2020;41:543-603 - https://doi.org/10.1093/eurheartj/ehz405
5. Wells PS et al. Value of assessment of pretest probability of deep-vein thrombosis in clinical management. Lancet 1997;350:1795-1798 - https://doi.org/10.1016/S0140-6736(97)08140-3
6. Adler Y et al. 2015 ESC Guidelines for the diagnosis and management of pericardial diseases. Eur Heart J 2015;36:2921-2964 - https://doi.org/10.1093/eurheartj/ehv318
7. McDonagh TA et al. 2021 ESC Guidelines for the diagnosis and treatment of acute and chronic heart failure. Eur Heart J 2021;42:3599-3726 - https://doi.org/10.1093/eurheartj/ehab368
8. Roberts ME et al. British Thoracic Society Guideline for pleural disease. Thorax 2023;78:s1-s42 - https://doi.org/10.1136/thorax-2022-219784
9. van Harten PN, Hoek HW, Kahn RS. Acute dystonia induced by drug treatment. BMJ 1999;319:623-626 - https://doi.org/10.1136/bmj.319.7210.623
10. Brignole M et al. 2018 ESC Guidelines for the diagnosis and management of syncope. Eur Heart J 2018;39:1883-1948 - https://doi.org/10.1093/eurheartj/ehy037
11. Mazzolai L et al. 2024 ESC Guidelines for the management of peripheral arterial and aortic diseases. Eur Heart J 2024;45:3538-3700 - https://doi.org/10.1093/eurheartj/ehae179
12. Rogers AM et al. Sensitivity of the Aortic Dissection Detection Risk Score. Circulation 2011;123:2213-2218 - https://doi.org/10.1161/CIRCULATIONAHA.110.988568

For these the specific recommendation I rely on is stated from clinical knowledge of the document and is flagged as unconfirmed inside the evidence point. The Newsome 2018 abstract was fetched and confirms the scope (management of abnormal liver blood tests in primary and secondary care) but not the ultrasound recommendation itself, so I list it here. The van Harten 1999 BMJ review has no abstract on Europe PMC. The ESC 2015 pericardial guideline has a PMC record (PMC7539677) without full text.

Could not be fetched: the NICE CKS sore throat topic (UK-only access). I did not cite it; the airway red flags for sore throat rest on the two Laryngoscope studies instead.

## Recurring patterns in the intakes

1. Chest pain with a cardiac risk profile dominates the slice: 22 intakes carry exertional or sudden chest pain radiating to the arms, jaw or throat with prior MI, diabetes or smoking. The working diagnosis is panic attack, anaemia, pericarditis or SLE. Every one of these needs an ECG and troponin, so every one is ESCALATE.
2. Dialysis with heart failure appears five times (cases 0, 11, 20, 24, 28), always with orthopnoea, nocturnal choking and leg swelling, and with pericarditis or panic attack as the working diagnosis. The intake mixes three dangerous diagnoses (ACS, decompensated heart failure, uraemic effusion) into one pattern.
3. Obstructive jaundice with chronic pancreatitis and a family history of pancreatic cancer appears six times (cases 12, 14, 22, 29, 33, 48) under bronchitis or anaemia. The yellow "skin lesion" on the epigastrium is jaundice reaching the intake as a decoding artefact. Five of the six patients are under 40, so the NG12 pathway does not apply by age, but the imaging requirement does.
4. The scombroid pattern (flushing after dark-fleshed fish, itchy rash, wheeze, palpitations) appears five times (cases 6, 8, 35, 40, 42), each labelled anaemia. Each meets the WAO anaphylaxis criteria on the intake alone.
5. Acute dystonia (oculogyric crisis, trismus, torticollis, tongue protrusion) appears three times (cases 10, 13, 27), each labelled anaemia; two include laryngeal symptoms.
6. Polyneuropathy after a viral illness (bilateral weakness, distal and perioral tingling, facial palsy) appears four times (cases 39, 41, 47, 51), each labelled panic attack. Perioral tingling is the one feature that genuinely overlaps with hyperventilation, and case 41, which has no facial palsy or viral trigger, is the only one I left UNCERTAIN.
7. Haemoptysis appears nine times; six patients are 40 or over and meet NG12 1.1.1 outright. The two immunosuppressed patients with sore throats (cases 60, 62) are the hardest, because the pharynx may be the bleeding site.
8. Isolated inspiratory stridor after a cold (cases 1, 2, 37) is the one recurring intake where the working diagnosis of laryngitis is plausible; the single symptom arrives without duration, voice change or fever, so I left all three UNCERTAIN.
9. Pain descriptors arrive as long location lists with 0-10 scores for intensity, localisability and speed of onset. The speed score was informative (9-10 read as abrupt onset and pushed dissection into the differential); the intensity score was not, since 3/10 exertional pain with jaw radiation carries the same risk as 9/10.
10. Every intake states no vitals, examination or duration. Ordinary office checks therefore closed only one case (17), and the borderline cases are all ones where the deciding feature is a single degraded symptom.

## Source use counts

| Citation | Cases citing it |
|---|---|
| clinical reasoning | 27 |
| NICE CG95 Recent-onset chest pain of suspected cardiac origin: assessment and diagnosis (2010, updated 2016) | 19 |
| Byrne RA et al. 2023 ESC Guidelines for the management of acute coronary syndromes. Eur Heart J 2023;44:3720-3826 | 18 |
| NICE NG12 Suspected cancer: recognition and referral (2015, updated) | 14 |
| Guardiani E et al. Supraglottitis in the era following widespread immunization against Haemophilus influenzae type B. Laryngoscope 2010;120:2183-2188 | 7 |
| Adler Y et al. 2015 ESC Guidelines for the diagnosis and management of pericardial diseases. Eur Heart J 2015;36:2921-2964 | 6 |
| Newsome PN et al. Guidelines on the management of abnormal liver blood tests. Gut 2018;67:6-19 | 6 |
| McDonagh TA et al. 2021 ESC Guidelines for the diagnosis and treatment of acute and chronic heart failure. Eur Heart J 2021;42:3599-3726 | 5 |
| Cardona V et al. World Allergy Organization Anaphylaxis Guidance 2020. World Allergy Organ J 2020;13:100472 | 5 |
| Feng C, Teuber S, Gershwin ME. Histamine (Scombroid) Fish Poisoning: a Comprehensive Review. Clin Rev Allergy Immunol 2016;50:64-69 | 5 |
| NICE NG158 Venous thromboembolic diseases: diagnosis, management and thrombophilia testing (2020) | 5 |
| Rogers AM et al. Sensitivity of the Aortic Dissection Detection Risk Score. Circulation 2011;123:2213-2218 | 5 |
| Sideris A et al. A systematic review and meta-analysis of predictors of airway intervention in adult epiglottitis. Laryngoscope 2020;130:465-473 | 4 |
| Wells PS et al. Derivation of a simple clinical model to categorize patients probability of pulmonary embolism. Thromb Haemost 2000;83:416-420 | 4 |
| Leonhard SE et al. Diagnosis and management of Guillain-Barre syndrome in ten steps. Nat Rev Neurol 2019;15:671-683 | 4 |
| Roberts ME et al. British Thoracic Society Guideline for pleural disease. Thorax 2023;78:s1-s42 | 3 |
| NICE NG250 Pneumonia: diagnosis and management (2025) | 3 |
| van Harten PN, Hoek HW, Kahn RS. Acute dystonia induced by drug treatment. BMJ 1999;319:623-626 | 3 |
| Konstantinides SV et al. 2019 ESC Guidelines for the diagnosis and management of acute pulmonary embolism. Eur Heart J 2020;41:543-603 | 3 |
| BTS/SIGN 158 British guideline on the management of asthma (2019 revision) | 2 |
| Brignole M et al. 2018 ESC Guidelines for the diagnosis and management of syncope. Eur Heart J 2018;39:1883-1948 | 2 |
| NICE CG141 Acute upper gastrointestinal bleeding in over 16s: management (2012, updated) | 1 |
| Wells PS et al. Value of assessment of pretest probability of deep-vein thrombosis in clinical management. Lancet 1997;350:1795-1798 | 1 |
| Flume PA et al. Cystic fibrosis pulmonary guidelines: pulmonary complications: hemoptysis and pneumothorax. Am J Respir Crit Care Med 2010;182:298-306 | 1 |
