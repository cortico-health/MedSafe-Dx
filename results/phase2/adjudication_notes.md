# Phase 2 adjudication notes

Adjudicator: Claude (Fable 5.1), 2026-09-26. Blinded: I read only `adjudication_queue.json` and `spot_check_queue.json`. I did not open any other file under results/, docs/, spec/ or data/, and did not search the repo for case ids. Output: `adjudications.json` (69 objects, queue order; no spot-check case was overturned, so no extra objects).

## The standard as applied

I applied the three-way standard as written. Because most disagreements came from the same few questions, I fixed one answer for each and applied it to every case it touched, so that like cases got like decisions.

1. Office checks do not escalate. Vitals, examination, an office ECG, a urine pregnancy test and a peak flow are the ordinary checks of a visit. A case whose danger is settled by those checks is ROUTINE even when the reviewer wrote "depends on the vitals". A case escalates when a guideline or validated rule mandates a test beyond them (troponin, chest film, D-dimer, ultrasound, laryngoscopy, a scan) or a referral pathway (same-day or two-week), or when the working diagnosis omits treatment whose delay is dangerous.
2. A benign mislabel is not an escalation. Pertussis labelled bronchitis, otitis media labelled bronchitis and a reducible hernia labelled bronchitis are ROUTINE, because the axis is danger, not label accuracy. The label matters when it omits dangerous treatment (an asthma exacerbation in a patient admitted last year, a COPD exacerbation labelled anaemia).
3. Chest pain and age. For chest pain with panic features I used the HEART age band as the line: under 45 the age score is 0 and an office ECG closes the visit (ROUTINE); at 45 and over the band scores and troponin is needed to reach low risk, and NICE CG95 1.2.1.3 suspects ACS on chest pain with nausea, breathlessness or presyncope and refers as an emergency while pain is present (ESCALATE).
4. Stridor. Self-reported inspiratory stridor is an AAO-HNS 2018 expedited-laryngoscopy indicator and is not a laryngitis symptom, so the four stridor cases escalate: airway visualisation is on the standard's own list.
5. Haemoptysis. A chest film is the first test for any reported haemoptysis (AFP 2015), and NICE NG12 1.1.1 makes unexplained haemoptysis at 40 and over a suspected-cancer pathway referral. All five pharyngitis-with-blood cases escalate, at confidence 4 when 40 or over and 3 when younger.
6. Fever while immunosuppressed. NICE NG253 1.1.10-1.1.11: suspect neutropenic sepsis in people who become unwell on immunosuppressant treatment and refer immediately for secondary-care assessment. Fever plus immunosuppression escalates; immunosuppression without fever is the NG84 high-risk tier (immediate antibiotic, blood count) and stays ROUTINE.
7. COPD with significant dyspnoea. NICE NG115 table 7 puts severe breathlessness in the treat-in-hospital column, and GOLD 2025 asks for pneumonia, heart failure and PE to be considered. All six COPD exacerbation cases escalate, at confidence 4 when COPD is stated as severe or the label (anaemia) omits airway treatment, and 3 when the label is bronchitis and severity is unstated.
8. Leg swelling. NICE NG158 table 1 Wells points that the intake itself supplies (previously documented DVT, painful swelling of one whole leg) escalate, because the pathway ends in a D-dimer or ultrasound. Symmetrical mild swelling with systemic causes and no tenderness is ROUTINE. A single calf pain site among scattered sites is a degraded feature (UNCERTAIN).
9. Hernia. Distension with no passage of stool or gas (asked, absent) beside a groin lump escalates for obstruction; a lump that hurts on straining with preserved stool and gas is ROUTINE whatever the age, with HerniaSurge risk factors setting the referral pace.
10. UNCERTAIN was reserved for the standard's own examples: a deciding feature that reaches the intake as a bare pain score (idx 143, 197), an onset-speed score that cannot say whether a headache peaked within an hour (idx 229, 236, 241), or a Wells tenderness point that cannot be assigned from scattered pain sites (idx 224, 230).

## Decisions by reviewer pair

| Fable | Astra | Cases | My ESCALATE | My ROUTINE | My UNCERTAIN |
|---|---|---|---|---|---|
| UNCERTAIN | ESCALATE | 38 | 26 | 9 | 3 |
| ROUTINE | UNCERTAIN | 16 | 0 | 16 | 0 |
| UNCERTAIN | UNCERTAIN | 8 | 5 | 0 | 3 |
| ROUTINE | ESCALATE | 6 | 0 | 5 | 1 |
| ESCALATE | UNCERTAIN | 1 | 1 | 0 | 0 |
| Total | | 69 | 32 | 30 | 7 |

Confidence: 30 cases at 4, 39 at 3. No case reached 5, because every queued case was a disagreement and the deciding feature was a single intake item.

Where I sided against Astra's ESCALATE (14 cases): the young panic cluster (idx 126, 128, 129, 137, 182) and the stimulant cluster (idx 13, 23, 24, 31), where the office ECG and vitals are the closing checks; 215 (symmetrical mild oedema); 232 (reducible hernias); and the three headaches and one sinusitis that stayed UNCERTAIN (idx 197, 229, 236, 241). Where I sided against Fable's UNCERTAIN toward ESCALATE (26 plus 5 double-uncertain cases): stridor, haemoptysis, COPD, pericarditis with dyspnoea, SLE with dyspnoea, fever while immunosuppressed, obstruction features with a hernia, prior-DVT swelling, symptomatic bleeding on a DOAC, and the pancreatitis case (NG104 1.3.10 and 1.3.22-1.3.23). Where I sided against Fable's ROUTINE: none, except that 197 moved to UNCERTAIN because NG79 1.1.9 lists severe frontal headache verbatim.

## Spot check

All 10 agreed decisions stand (idx 2, 9, 36, 45, 88, 108, 162, 170, 204, 209). Two of them anchor my rules: 108 (85-year-old with severe COPD, both ESCALATE) matches rule 7, and 162 (SLE without dyspnoea, both ROUTINE) matches the dyspnoea line I drew for idx 103 and 105. 204 (headache with onset 9/10 at age 49, both ESCALATE) is consistent with 229, 236 and 241 staying UNCERTAIN: at 49 the Ottawa age criterion is met on its own, whereas under 40 the decision turns entirely on the onset score.

## Sources I added and how I verified them

I fetched each NICE recommendations page with curl and read the recommendation text (the NICE site refuses the built-in fetcher). Journal sources were confirmed through Crossref or the Europe PMC REST API (title, journal, year, volume, pages, PMID). Fable's citations that I relied on unchanged (ESC 2015 pericardial, ESC 2018 syncope, PEP study, BTS pleural 2023, Guardiani 2010, Stachler 2018, HerniaSurge 2018) I re-confirmed bibliographically the same way.

| Source | Used for | Verified |
|---|---|---|
| NICE NG12 1.1.1 (lung cancer: unexplained haemoptysis at 40 and over), 1.8.1 (laryngeal cancer at 45 and over) | idx 18, 47, 51, 52, 53 | page text |
| NICE NG253 1.1.10-1.1.11 (suspect neutropenic sepsis on immunosuppressant treatment; refer immediately) | idx 179, 183, 220, 235 | page text |
| NICE NG79 1.1.9 (sinusitis hospital referral; "severe frontal headache" verbatim) | idx 197 | page text |
| NICE NG104 1.3.10, 1.3.22-1.3.23 (symptomatic pseudocyst; hereditary pancreatitis cancer risk) | idx 14 | page text |
| NICE NG115 1.3.1 and table 7 (where to treat a COPD exacerbation) | idx 100, 102, 107, 113, 237, 248 | page text |
| NICE CG95 1.2.1.3, 1.2.1.5, 1.2.1.7, 1.2.1.12 (ACS suspicion, emergency referral, ECG) | idx 122, 123, 211, 237, 248 | page text |
| NICE NG158 table 1 and 1.1.3-1.1.8 (two-level Wells, D-dimer within 4 hours) | idx 173, 206, 215, 224, 230 | page text |
| NICE CG150 1.1.2 and the first-bout cluster neuroimaging recommendation | idx 229, 236, 241 | page text |
| NICE NG84 1.1.12-1.1.13 (high-risk tier; hospital referral) | idx 51, 143, 179, 183 | page text |
| NICE NG126 1.4.5-1.4.7 (pregnancy test; refer a positive test with tenderness immediately) | idx 70, 128, 182 | page text |
| NICE NG240 1.1.4-1.1.7 and table 2 (meningitis red flags) | idx 153, 231, 239 | page text |
| NICE NG91 1.1.14 (otitis media hospital referral) | idx 152, 165, 174, 175, 178, 187, 196 | page text |
| Stachler 2018 AAO-HNS dysphonia guideline KAS 3 (laryngoscopy; stridor as an expedited indicator) | idx 1, 18, 25, 47, 172 | Europe PMC abstract, PMID 29494321 |
| Earwood 2015 AFP hemoptysis (chest film first; CT at 40 and over with smoking) | idx 51, 52, 53, 56, 217 | article page |
| GOLD 2025 pocket guide (consider pneumonia, heart failure and PE) | COPD cases | PDF text; the hospitalisation-indications figure is an image and was not verified |

Two things I did not verify and did not lean on: the WSES 2017 complicated-hernia guideline's imaging wording (the abstract does not carry it; the hernia escalations rest on the obstruction features and HerniaSurge risk factors), and SIGN 158 section 9.1 (Fable verified it; the SIGN site did not expose a PDF link to my fetcher).

## Why the reviewers said UNCERTAIN

Fable's 47 UNCERTAIN calls had four recurring shapes; Astra's 25 had the same shapes in a different mix.

1. "Escalation hinges on the vitals" (Fable, about 20 cases: COPD, stimulant, panic, pericarditis, SLE). The intake never carries vitals, so this reason applies to every case; I treated vitals as ordinary checks and decided from the intake feature itself.
2. "The label is wrong but the true condition is usually outpatient-treatable" (Fable: COPD labelled anaemia, hernia labelled bronchitis). I decided by whether the omitted treatment is dangerous to delay.
3. "The examination decides" (both reviewers: ear pain on antibiotics, sore throat severity, hernia reducibility, leg asymmetry, meningism). Where the intake feature that would trigger a rule is an examination finding, this is ROUTINE; where the intake feature is present but degraded, UNCERTAIN.
4. "The intake cannot give the time course or the degree" (both: headache onset score, pain score, degree of immunosuppression, bleeding source and amount). This is the standard's own UNCERTAIN definition and is where my seven UNCERTAIN calls sit, except that a stated guideline rule (haemoptysis, fever while immunosuppressed) overrides the missing degree.

Astra's ESCALATE calls that I overturned all had one shape: naming an office check (ECG, blood pressure, pregnancy test, oximetry) as the reason for escalation.
