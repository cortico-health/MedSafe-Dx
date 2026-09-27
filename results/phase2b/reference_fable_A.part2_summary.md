Reviewer: Fable 5.1, blind.

Slice: indices 63 to 124 of results/phase2b/cases_blind.json (62 cases). I read only that file, the brief and the wording rules. Each case has one row in reference_fable_A.part2.jsonl.

## Decisions by confidence

| Confidence | ESCALATE | ROUTINE | UNCERTAIN | Total |
|---|---|---|---|---|
| 5 | 17 | 0 | 0 | 17 |
| 4 | 30 | 0 | 0 | 30 |
| 3 | 11 | 1 | 3 | 15 |
| 2 | 0 | 0 | 0 | 0 |
| 1 | 0 | 0 | 0 | 0 |
| Total | 58 | 1 | 3 | 62 |

## Decisions by working diagnosis

| Working diagnosis | ESCALATE | ROUTINE | UNCERTAIN | Total |
|---|---|---|---|---|
| Anemia (D64.9) | 28 | 1 | 1 | 30 |
| Pericarditis (I30) | 7 | 0 | 0 | 7 |
| Viral pharyngitis (J02.9) | 6 | 0 | 0 | 6 |
| Bronchitis (J40) | 5 | 0 | 1 | 6 |
| Panic attack (F41) | 5 | 0 | 1 | 6 |
| Sarcoidosis (D86) | 3 | 0 | 0 | 3 |
| Acute laryngitis (J04.0) | 2 | 0 | 0 | 2 |
| SLE (M32) | 2 | 0 | 0 | 2 |

The slice escalates heavily because the intakes carry hard features: melaena or haematemesis on an anticoagulant, haemoptysis in smokers over 40, bulbar and respiratory weakness, chest pain radiating to the back, irregular palpitations with structural heart disease, and first seizures. The three UNCERTAIN cases are a 57-year-old man with a textbook panic cluster and mild chest pain (100), a 33-year-old woman with rectal bleeding and heavy periods but no anticoagulant (78), and a severe-COPD exacerbation whose home-or-hospital decision rests on saturations the intake lacks (119). The one ROUTINE case is menstrual blood-loss anaemia with no other bleeding source and no anticoagulant (77).

## How I verified the citations

I resolved every DOI on the Crossref API and checked title, journal, year, volume and pages. I fetched guideline pages with curl or the page fetcher and checked the title and the quoted recommendation. I used Europe PMC for abstracts and one full text. Seventeen distinct sources appear in the file, plus "clinical reasoning" entries.

| Source | Verified | How |
|---|---|---|
| NICE NG12 recs 1.1.1, 1.3.1 | From text | Recommendations page fetched; haemoptysis and rectal-bleeding rules read verbatim |
| NICE CG141 recs 1.1.1, 1.1.2, 1.3.2 | From text | Recommendations page fetched; Blatchford and 24-hour endoscopy rules read verbatim |
| Blatchford 2000, Lancet | From text | Crossref plus Europe PMC abstract naming melaena and syncope as score components |
| Oakland 2019, Gut (BSG lower GI bleeding) | Title, DOI and abstract | Crossref plus Europe PMC abstract, which states the guideline covers risk assessment and DOAC management; the full text is subscription-only and the BSG page returned 404 |
| Sanders 2016, Neurology (MG consensus) | From text | Crossref plus Europe PMC full text; impending-crisis definition and admission statement read verbatim |
| Raviele 2011, Europace (EHRA palpitations) | From text | Crossref plus fetched article page; structural-heart-disease and exertional-palpitation statements read verbatim |
| Van Gelder 2024, Eur Heart J (ESC AF) | From text | Crossref plus fetched article page; ECG documentation, TTE and anticoagulation statements read |
| Adler 2015, Eur Heart J (ESC pericardial diseases) | From text | Crossref plus fetched article page; diagnostic criteria, work-up tests and major and minor predictors read |
| Rogers 2011, Circulation (ADD-RS) | From text | Crossref plus Europe PMC abstract with the 95.7% sensitivity figure |
| Gulati 2021, Circulation (AHA/ACC chest pain) | Title, DOI and abstract | Crossref plus Europe PMC abstract; the AHA page returned 403, so the ECG-and-troponin point is paired with a clinical reasoning entry |
| NICE NG158 rec 1.1.17, table 2 | From text | Recommendations page fetched; Wells items for haemoptysis and malignancy read verbatim |
| NICE NG217 recs 1.1.1, 1.2.2 | From text | Chapter 1 fetched; two-week first-seizure referral and ECG rule read verbatim |
| NICE NG115 table 7 | From text | Recommendations page fetched; home-versus-hospital factors read verbatim |
| WHO TB screening module 2 (2021) | Title and page summary | Publication page fetched; it names people with HIV as a priority group and symptom screening as a tool; the PDF returned 403, so the W4SS wording is not quoted |
| Earwood 2015, Am Fam Physician (hemoptysis) | From text | Europe PMC abstract and the AAFP article page; chest radiography and CT statements read |
| Flume 2010, AJRCCM (CF hemoptysis) | Title, DOI and abstract | Crossref plus Europe PMC abstract; the specific recommendations are not quoted, so the point is kept general |
| clinical reasoning | Not a source | Used where no fetched text carried the point (immunosuppressed haemoptysis, DOAC rectal bleeding, first-seizure imaging, lupus dyspnoea, older panic-attack chest pain, CF and cancer chest pain); confidence lowered to 3 where it carried the decision |

Sources I could not fetch at all: none that appear in the file. I dropped GOLD (site suspended) and the ACG and ESC ACS guidelines because I did not need them.

## Recurring patterns in the intakes

1. The anaemia template repeats one symptom block (headache with tugging or cramping character, dizziness, fatigue, pallor) and varies the bleeding items. Melaena appears in 10 of the 30 anaemia intakes and a new oral anticoagulant in 13, so the decision often turns on two words the working diagnosis ignores.
2. Decoding artefacts are common: "think you are pregnant" appears at ages 53, 57 and 93 and alongside a NOAC, "breastfed a child for more than 9 months" appears at 19, and "coughing up blood" is attached to 7 of the 8 pharyngitis and laryngitis intakes. I decided on the face value of the intake and noted the artefact in the reasoning.
3. Eight intakes (82 to 89) are the same myasthenia block (ptosis, diplopia, jaw weakness, dysphagia, dyspnoea, fatigable weakness) filed under anaemia, laryngitis or SLE. Every one is a neuromuscular emergency on the intake alone.
4. Eight intakes (90 to 97) pair "heart beating very irregularly" and dyspnoea with a stacked cardiac history (valve disease, heart defect, prior infarction, hyperthyroidism) and file it as anaemia or panic. The history block, not the symptom block, drives the escalation.
5. The pericarditis intakes all include radiation to the thoracic spine and posterior chest wall, which is unusual for pericarditis and forces dissection into the differential; onset speed and intensity are the only discriminators and reach the intake as bare scores.
6. "Experiencing shortness of breath in a significant way" appears in 41 of 62 intakes, across every diagnosis, so it carries little weight on its own; I used it only in combination with a bleeding, neuromuscular or cardiac feature.
7. Pain scores and onset speed arrive as bare numbers with no duration or time course, so the UNCERTAIN cases are the ones where those numbers would have decided the call.
