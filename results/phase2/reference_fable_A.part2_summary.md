# Reference review, Part 2 (cases 63-124): summary

Blind clinical reference assessment of entries 63-124 of `results/phase2/cases_blind.json` (62 cases). I read only that file and the intakes for my slice. Output: `results/phase2/reference_fable_A.part2.jsonl`, one record per case in file order.

## Decisions

| Decision | Count |
|---|---|
| ESCALATE | 51 |
| UNCERTAIN | 11 |
| ROUTINE | 0 |

| Confidence | Count |
|---|---|
| 5 | 19 |
| 4 | 25 |
| 3 | 18 |

By working diagnosis:

| Working diagnosis | ESCALATE | UNCERTAIN |
|---|---|---|
| Anemia (D64.9) | 27 | 3 |
| Bronchitis (J40) | 8 | 2 |
| Panic attack (F41) | 6 | 2 |
| Pericarditis (I30) | 3 | 2 |
| Sarcoidosis (D86) | 4 | 0 |
| SLE (M32) | 0 | 2 |
| Acute laryngitis (J04.0) | 2 | 0 |
| Viral pharyngitis (J02.9) | 1 | 0 |

The 11 UNCERTAIN calls (all confidence 3) are cases where the working diagnosis is plausible or close to right but the safety of routine care depends on a same-visit test the intake cannot contain (ECG, SpO2, urinalysis, haemoglobin): 70, 96, 97, 100, 102, 103, 105, 107, 113, 122, 123. I gave no ROUTINE call because every intake in this slice carries at least one finding that a dangerous condition explains better than the working diagnosis does.

## Citations

30 distinct sources, 1-3 per case. Verification, all done without LLM inference spend:

| How verified | Sources |
|---|---|
| DOI resolved on Crossref and the abstract or open article read on Europe PMC / PMC / CDC, so the quoted point is confirmed | Blatchford 2000; Stanley 2017; Oakland 2017; Quinn 2004 (SFSR); Wendell 2011; Sanders 2016 (MG consensus, definitions read in PMC full text); Rao 2021 (CDC botulism); Thavendiranathan 2009; Rogers 2011 (ADD-RS); Kline 2008 (PERC); Adler 2015 (ESC pericarditis, via OUP page); Six 2008 (HEART); Fleet 1996; Krumholz 2007; Couturaud 2021 (PEP); Hamada 2018 (W4SS); Flume 2010 (thresholds confirmed in the open-access review PMC8727888) |
| Guideline page fetched with curl and the recommendation text read | NICE CG141 (1.1.1-1.1.2), NICE NG196 (1.1.1-1.1.2), NICE CG95 (1.2.1.5-1.2.2.1), NICE NG217 (1.1.1, from the PDF), NICE NG84 (1.1.13), GOLD 2025 (pocket guide and full report text: exacerbation definition, PE/pneumonia/heart-failure differential, 5.9% PE figure; the hospitalisation-indications figure is an image, so I cite only its title) |
| DOI resolved on Crossref and abstract read; the publisher blocked the body, so the specific point rests on my knowledge of the document | BSG LGIB 2019; ESC PE 2019; HRS cardiac sarcoidosis 2014; EULAR/ACR SLE criteria 2019; EULAR SLE 2023; ESVS AAA 2024; ACC/AHA AF 2023 |

Tools: Crossref REST API (26 DOIs, all resolved), Europe PMC REST API (abstracts and two full texts), curl for NICE and GOLD, WebFetch for CDC, PMC and OUP pages. The WebSearch budget for the session was already exhausted, and the AHA, JAMA, Gut, hiv.gov and NICE CKS sites returned 403, so I swapped the AHA 2021 chest-pain guideline for NICE CG95 (fetched and read), the ESC syncope guideline for the San Francisco Syncope Rule (abstract read), and the WHO TB screening module for the Hamada 2018 meta-analysis of the same rule (abstract read).

## Recurring patterns in the intakes

1. The working diagnosis is often the complication, not the cause. 30 cases carry "Anemia" while the intake reports melaena, haematochezia or haematemesis (16 cases), a bulbar neuromuscular syndrome (8), an irregular tachycardia with structural heart disease (8), or a COPD exacerbation (4). An anaemia label leaves the bleed, the airway or the rhythm unaddressed, which is why most of these are ESCALATE.
2. Anticoagulant plus bleeding plus CKD is a repeated template (65-71, 110, 117, 118). Several of these also flag "think you are pregnant" in men or in an 83-year-old, which looks like a generator artefact; I noted it and did not let it drive the decision.
3. The chest-pain intakes separate cleanly on pain descriptors. "Tearing", "abrupt onset 9-10/10" and "radiates to thoracic spine" mark the dissection/PE/MI group (88-90, 92-95); "worse lying down, better sitting up" plus "previous pericarditis" marks the pericarditis group (96, 97), where I stayed UNCERTAIN because ECG and echo decide safety.
4. The panic-attack intakes share perioral paraesthesia, derealisation and fear of dying, and differ mainly in age, pain intensity and accompanying features (sweating, presyncope, nausea). I escalated those at 50 or older or with diaphoresis/presyncope (91, 120, 121, 124) and marked the rest UNCERTAIN, on the CG95 rule that a 12-lead ECG must precede the label.
5. No intake has vitals, examination or duration, so for COPD, pericarditis and SLE the escalation decision hinges on measurements the clinician would take in the room. I marked those UNCERTAIN rather than guess.
6. Off-list dangers I flagged that a fixed list might miss: botulism (A05.1) and brainstem stroke in the myasthenia-pattern cases; cardiac sarcoidosis conduction disease (D86.85) as a mimic of a first seizure; ectopic pregnancy (O00.9) in the possibly-pregnant bleeders; ruptured AAA (I71.3) and mesenteric ischaemia (K55.0) in the 71-year-old "panic attack".
