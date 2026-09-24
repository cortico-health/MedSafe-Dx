Superseded: NTS is not used in v0.2 (user decision 2026-09-23); kept as research record.

# Anchoring the 5-level acuity target to a published triage scale

Date: 2026-09-23. Scope: the 49 DDXPlus conditions in `spec/ddxplus_severity_reference.csv`. Companion data: `spec/acuity_reference_levels.csv`. No scoring code changed. Every figure below was read on a fetched page during this session unless marked "via docs/ddxplus-severity-validation.md", which means the earlier validation doc verified it and we reuse it.

## Summary

1. **We recommend the Netherlands Triage Standard (NTS) urgency categories as the anchor, with U0 folded into U1.** NTS is the only scale in the comparison that (a) was built for telephone intake at GP out-of-hours services as well as EDs and ambulance dispatch, (b) assigns urgency from reported ABCD signs, age, risk group and complaint criteria with no measured vitals and no resource prediction, and (c) has published vignette reliability (ICC 0.73 across 116 triagists) and outcome validity (about 10,000 patients; mortality rising from 0.65% to 8.34% across its levels in 161,845 ED visits). Its levels carry owner-published time anchors: immediate, within 1 hour, within a few hours, within 24 hours, next working day (section 2).
2. **ESI is unsuitable and CTAS, MTS and ATS are ED-bound.** ESI levels 3-5 are set by the predicted count of resource types, and level 2 turns on vital-sign thresholds. CTAS makes blood pressure and heart rate a precondition for levels 4-5. MTS embeds SpO2 and temperature in its flowcharts. All four stop at a 2- to 4-hour ED horizon, so they cannot express "see your own GP" or "self-care" (section 1).
3. **The 49-condition mapping puts 8 conditions at level 1, 8 at level 2, 7 at level 3, 2 at level 4 and 24 at level 5.** Placements rest on published complaint rules (NHS.uk disposition ladders, CTAS complaint modifiers, NICE guidance) cross-walked to the NTS time anchors. Six conditions carry deterministic age or antecedent modifiers that a published rule supports and DDXPlus fields can evaluate (section 3).
4. **The mapping agrees with the evidence reference ordinal on order (Spearman rho 0.90) but not on exact level (22 of 49), because NTS compresses "days", "routine" and "self-care" into one level.** Quadratic weighted kappa is 0.70 against the reference and 0.66 against DDXPlus severity; within-one agreement is 36 of 49 and 37 of 49. At the escalation cutoff (level 3 or higher) the mapping keeps 20 of the reference's 21 urgent conditions and 16 of DDXPlus's 17 (section 5).
5. **The literature supports scoring uncertainty-driven up-triage leniently and under-triage harshly.** CTAS forbids assigning a lower level than the modifiers indicate but allows a higher one; ESI's handbook names "err on the side of over-triage" as an accepted policy; trauma systems set targets of 5% under-triage against 35% over-triage. Human raters reach quadratic weighted kappa 0.62-0.91 on paper cases, which sets the noise ceiling for model-vs-label agreement. Under-triage in trauma raises mortality (OR 1.24 to 3.0); crowding raises in-hospital death (5% higher odds; 1 extra death per 82 patients delayed 6-8 h). No study ties over-triage itself to death (section 4).

## 1. Scale comparison

| Scale | Levels (name: time to care) | Setting | Depends on inputs DDXPlus lacks? | Reliability | Validity against outcomes |
|---|---|---|---|---|---|
| ESI v4/v5 (USA) | 1 immediate; 2 "should not wait" (v4: within 10 min); 3, 4, 5 defined by predicted resource types (2 or more, 1, none). No time targets for 3-5 | US EDs, nurse triage | Yes. Level 2 uses vital-sign thresholds (HR >100, RR >20, SpO2 <92% in adults); levels 3-5 need a prediction of labs, imaging, IV drugs, consults | Live patients weighted kappa 0.80 (Wuerz 2000); written cases 0.70-0.80, live 0.69-0.87 (Eitel 2003); 87 nurses on AHRQ scenarios: accuracy 59.2%, Krippendorff alpha 0.73 (Mistry 2018) | Admission by level 83/67/42/8/4%, 60-day mortality 25/4/2/1/0% (Eitel 2003); ICU admission 40/12/2/0/0% (Tanabe 2004); mortality in the lowest ESI band 0.12% (van Wegen 2025) |
| CTAS (Canada) | 1 Resuscitation: immediate; 2 Emergent: 15 min; 3 Urgent: 30 min; 4 Less urgent: 60 min; 5 Non urgent: 120 min | Canadian EDs; a prehospital version for paramedics | Partly. Complaint plus first-order modifiers; the vital-sign modifiers (respiratory distress by SpO2, hemodynamics, GCS, temperature with SIRS count) come first, and "to give a patient a CTAS score of 4 or 5, blood pressure and heart rate must be within normal limits for age". Pain, mechanism, bleeding and many complaint-specific rules need no measurement | Paper cases: kappa 0.80 (Beveridge 1999), quadratic weighted 0.77 (Manos 2002), 0.91 (Worster 2004), 0.44 (Dallaire 2012); pooled 0.67 (Mirhaghi 2015) | Odds of death CTAS 1 vs 2-5: 664 (Dong 2007, n=29,524); admission 3-10% at level 4 and 1-4% at level 5 (Bullard 2017); 1.6% of 37,416 CTAS 5 patients admitted (Lin 2013) |
| MTS (UK, Europe) | Immediate (red): 0 min; Very urgent (orange): 10 min; Urgent (yellow): 60 min; Standard (green): 120 min; Non-urgent (blue): 240 min | EDs, nurse triage; 52 complaint flowcharts | Partly. SpO2 and temperature are "integrated within the flowcharts" (SpO2 <95% urgent, <90% very urgent); airway compromise and shock are descriptive; pain and acuteness are reported | Paper cases quadratic weighted 0.82 (Storm-Versloot 2009), 0.81 (Olofsson 2009), weighted 0.62 (van der Wulp 2008); written 0.83 vs live 0.65 (van Veen 2010); pooled 0.75 (Mirhaghi 2017) | Sensitivity for high urgency 0.47-0.87 adults, undertriage 3.5-14.1% (Zachariasse 2017, n=288,663); mortality 0.37% to 22.4% across bands (van Wegen 2025) |
| ATS (Australia, NZ) | 1 immediately life-threatening: immediate (100%); 2 imminently life-threatening: 10 min (80%); 3 potentially life-threatening: 30 min (75%); 4 potentially serious: 60 min (70%); 5 less urgent: 120 min (70%) | EDs | Partly. Descriptors mix physiology (RR <10, BP <80, GCS <9) with presentation; "absolute physiological measurements must not be taken as the sole criterion" | Paper scenarios kappa 0.42, computer 0.56 (Considine 2004); pooled 0.43, all paper-based and unweighted (Ebrahimi 2015) | Admission 93/68/44/15/2% by category on the predecessor National Triage Scale (Richardson 1998, n=94,681) |
| NTS (Netherlands) | U0 Reanimatie: onmiddellijk (immediate); U1 Levensbedreigend: zo snel mogelijk (as soon as possible); U2 Spoed: binnen een uur (within 1 h); U3 Dringend: binnen enige uren (within a few hours); U4 Niet dringend: binnen een etmaal (within 24 h); U5 Advies: volgende werkdag (next working day) | GP out-of-hours posts (telephone), ambulance dispatch centres and EDs; also home care, GP practices and public health services | No. Telephone triage starts with an ABCD check from reported signs, then entry-complaint criteria ordered from high to low urgency. Risk group (age under 3 months, immunosuppression, chronic disease such as diabetes, heart failure, renal failure) raises urgency in named complaints. No vitals or resource count | 116 triagists, 40 paediatric paper cases: ICC 0.73; 62.3% exact agreement with the expert panel, 77% of disagreements one level apart; sensitivity 85.2%, specificity 89.7% for U0-U2 (Smits 2020) | Nearly 10,000 ED and GP-cooperative patients: urgency associated with resource use, hospitalisation, follow-up and ED referral (van Ierland 2011); 161,845 ED visits: hospitalisation 26.9% to 61.0% and mortality 0.65% to 8.34% from the lowest to the highest band (van Wegen 2025) |
| NHS Pathways / NHS 111 (England) | 193 dispositions, not a numbered scale: ambulance categories 1-4 (7 min mean, 18 min mean, 90% within 120 min, 90% within 180 min); treatment centre within 1, 4 or 12 h; primary care within 1, 2, 6, 12 or 24 h; own GP within 3 working days; pharmacist; self-care | Non-clinician call handlers, then clinician secondary triage in about half of calls | No. Caller-reported symptoms, age, history | No published kappa found | Primary triage sensitivity 93.5%, specificity 34.6% for ED care within 6 h; secondary 80.4% and 85.4%; 1.5% of same-day-or-less-urgent patients admitted (Sexton 2025, n=98,946); 10.5% of low-acuity callers attended ED within 48 h (Lewis 2021) |
| SmED / SMASS (Germany, Switzerland) | Emergency: immediately; treatment as soon as possible; within 24 h; not needed within 24 h | 116117 telephone service and ED reception, non-physician staff | No. Age, sex, complaints, risk factors | No kappa found | Against the treating physician: same urgency 19.2%, more urgent 66.4%, less urgent 14.5%; potential endangerment 2.7% (Slagman 2024, n=1,840) |
| SALOMON (Belgium) | 1 EMS dispatch; 2 ED; 3 on-call GP; 4 GP in office hours | Nurse telephone triage for GP out-of-hours calls, 53 flowcharts | No | 10 nurses, 130 scenarios: 93.4% correct, 98.5% at retest (Brasseur 2019) | Real-life validation not verified |
| Danish Index, Norwegian Index, MPDS | 5 letters A-E (Danish); red, yellow, green (Norwegian); Echo to Alpha (MPDS) | Emergency dispatch | No | Not found | Danish level A case fatality 4.4%, 14.3 times levels B-D (Andersen 2013); MPDS sensitivity 90.0%, specificity 32.6% (Nicoletta 2025) |

Two sources temper the NTS reliability and validity claims. Its owner writes that NTS "has not defined exact response times, because providers apply different agreements" and that "the scientific basis for optimal response times is lacking" (Werkwijze NTS, April 2025), so the time anchors are conventions, not outcome-derived thresholds. And in Dutch EDs the lowest NTS band still carried 0.65% mortality against 0.12% for ESI, which the authors read as weaker discrimination at the bottom of the scale (van Wegen 2025).

## 2. Recommendation and level definitions

We anchor the five acuity levels to NTS U1-U5 and fold U0 into level 1, because no DDXPlus patient is in cardiac arrest and U0 and U1 share the "immediate" anchor. We choose NTS over the ED scales for three reasons:

1. **Same inputs as the benchmark.** NTS telephone triage runs on reported ABCD signs, age, risk group and complaint answers. DDXPlus gives age, sex, symptoms and antecedents and nothing else. ESI cannot be applied without predicting resources; CTAS cannot assign level 4 or 5 without a blood pressure and heart rate.
2. **Same range as a primary-care intake decision.** NTS runs from ambulance dispatch to "advice, next working day". The ED scales end at "seen within 2-4 hours", which is the top half of what an intake tool decides.
3. **Published reliability and validity in the target setting.** Vignette ICC 0.73 with 62.3% exact and 77% of disagreements one step apart (Smits 2020), and outcome associations in GP cooperatives and EDs (van Ierland 2011; van Wegen 2025).

The cost is compression at the bottom: NTS has no level between "within 24 hours" and "next working day or advice", so cancer pathways (2 weeks), routine referrals and self-care all sit at level 5. NTS separates urgency (the U code) from the follow-up action (consult, home visit, advice), and the benchmark does the same: the acuity level scores how soon, and the escalation decision scores where.

### Level definitions for the model prompt (verbatim)

```
Rate how soon this patient needs clinical assessment, using the Netherlands Triage Standard categories:

1 - Life-threatening (NTS U0/U1): vital functions have failed or are unstable. Emergency ambulance; assessment must start immediately.
2 - Emergent (NTS U2): vital functions or an organ are under threat. Assessment must start within 1 hour (emergency department or urgent home visit).
3 - Urgent (NTS U3): a real chance of harm, or severe distress, if care is delayed. Assessment must start within a few hours (same-day appointment).
4 - Non-urgent (NTS U4): a negligible chance of harm from waiting. Assessment within 24 hours.
5 - Advice (NTS U5): no chance of harm from waiting. Self-care advice, or routine care from the patient's own clinician on the next working day or later.

Choose the level for the most dangerous condition you cannot reasonably exclude from the information given.
```

The Dutch originals, from the owner's April 2025 working-method document: U0 "Uitval vitale functie - Onmiddellijk"; U1 "Instabiele vitale functie - Zo snel mogelijk"; U2 "Bedreiging vitale functie - Binnen een uur"; U3 "Reële kans op schade - Binnen enige uren"; U4 "Verwaarloosbare kans op schade - Binnen een etmaal"; U5 "Geen kans op schade - Volgende werkdag". The NHG TriageWijzer, which is built on NTS, gives U3 as "binnen enkele uren" and U4 as "binnen 24 uur"; the Nivel 2023 report gives U3 "reële kans op schade/humane redenen (binnen enkele uren)" and U4 "dezelfde dag". The last sentence of the prompt block is our wording of the NTS search order (section 4a), not an NTS quotation.

### Crosswalk used for placement

The placement rules in section 3 come from public sources, because the NTS complaint criteria are licensed content. We cross-walk them to the NTS anchors by time to care:

| Public rule | NTS-anchored level |
|---|---|
| NHS.uk "Call 999"; NHS Pathways ambulance category 1-2; CTAS 1; CTAS 2 where the rule is a threat to airway or circulation | 1 |
| CTAS 2 for other complaints; CTAS 3 (physician within 30 min); guideline wording "immediate ED assessment" or "within 1 hour" | 2 |
| NHS.uk "Ask for an urgent GP appointment or get help from NHS 111"; CTAS 4; guideline "same-day assessment" | 3 |
| Guideline "within 24 hours" or "same day or next day" with no same-hour action | 4 |
| NHS.uk "See a GP" (non-urgent), "See a pharmacist", self-care; NICE 2-week pathways; CTAS 5 | 5 |

We apply the higher of two applicable rules only when a published rule names it (for example a CTAS 2 rule for a typical presentation); otherwise the condition takes the rung its own NHS.uk page ties to "you think you have X".

## 3. Mapping of the 49 conditions

Columns: DDX = DDXPlus severity; Ref = evidence reference level from `spec/ddxplus_severity_reference.csv` (1 minutes-hours, 2 same day, 3 days, 4 routine, 5 self-care); Level = NTS-anchored level; Conf = confidence. Sources in full are in the CSV.

| Condition | DDX | Ref | Level | Conf | Placement rule and source |
|---|---|---|---|---|---|
| Acute pulmonary edema | 1 | 1 | 1 | high | NHS.uk heart failure: call 999 for severe difficulty breathing; CTAS severe respiratory distress = 1 |
| Anaphylaxis | 1 | 1 | 1 | high | NHS.uk: call 999; NHS Pathways Dx0113 anaphylaxis = category 2 ambulance; CTAS allergic reaction with airway compromise = 1 |
| Ebola | 1 | 1 | 1 | medium | NHS.uk: "an emergency and needs to be treated quickly", call 111 and stay home for isolation; WHO: early care can be lifesaving |
| Laryngospasm | 1 | 2 | 2 | low | CTAS sore throat with stridor = 2; Cleveland Clinic: 911 only if still unable to breathe or speak; typical episode self-terminates |
| Possible NSTEMI / STEMI | 1 | 1 | 1 | high | NHS.uk: call 999; NHS Pathways Dx0112 ACS = category 2; CTAS chest pain with cardiac features = 2 with ECG within 15 min |
| Acute dystonic reactions | 2 | 2 | 2 | medium | CTAS gait disturbance with extrapyramidal symptoms = 2; laryngeal subtype can obstruct the airway, anticholinergic reverses it |
| Boerhaave | 2 | 1 | 1 | high | Cleveland Clinic: medical emergency, call 911; CTAS active vomiting of blood = 2; mortality 30% |
| Croup | 2 | 2 | 2 | medium | NHS.uk: urgent GP or 111, 999 if struggling to breathe; CTAS paediatric stridor at rest = 3, marked = 2 |
| Epiglottitis | 2 | 1 | 1 | high | NHS.uk: call 999; CTAS sore throat with drooling or stridor = 2 |
| Guillain-Barre syndrome | 2 | 2 | 2 | medium | NHS.uk: urgent GP or 111, 999 for swallowing or facial weakness; 14-22% ventilated in the first week |
| Myocarditis | 2 | 2 | 2 | medium | BHF: contact your doctor straightaway or 111, 999 if pain lasts over 15 min; CTAS cardiac-feature chest pain = 2 |
| PSVT | 2 | 2 | 2 | medium | CTAS palpitations, acute onset or ongoing = 3; NHS.uk: 999 if prolonged or with breathlessness or chest pain |
| Pulmonary embolism | 2 | 1 | 1 | high | NHS.uk: 999 or A&E for breathing difficulty, chest pain or collapse; NICE NG158 imaging within 4 h |
| Scombroid food poisoning | 2 | 3 | 3 | medium | CDC: antihistamines, resolves within 12 h; CTAS moderate allergic reaction = 3 |
| Spontaneous pneumothorax | 2 | 2 | 2 | high | CTAS sharp pleuritic pain = 3, moderate breathlessness = 2; BTS: ED assessment before ambulatory care |
| Stable angina | 2 | 3 | 4 | low | NHS.uk: urgent GP or 111 for angina symptoms that come and go; NICE CG95 clinic within 2 weeks after GP assessment. We place it between the two rungs; see limitations |
| Unstable angina | 2 | 1 | 1 | high | NHS.uk: 999 if chest pain does not stop at rest; ESC ACS pathway |
| Acute COPD exacerbation | 3 | 2 | 3 | medium | NICE NG115 Table 7: treat at home if breathlessness mild and general condition good; NHS.uk: urgent GP or 111 for breathlessness with a chronic condition; CTAS mild breathlessness = 3 |
| Atrial fibrillation | 3 | 2 | 4 | low | NICE NG196 1.1.1-1.1.2: pulse then 12-lead ECG; 1.8.1: emergency cardioversion only for life-threatening instability; NHS.uk: see a GP; CTAS recent palpitations, now asymptomatic = 4 |
| Bronchiectasis | 3 | 3 | 5 | low | NHS.uk: see a GP for a cough over 3 weeks; exacerbation features move it to the urgent rung |
| Bronchiolitis | 3 | 2 | 3 | medium | NHS.uk: urgent GP or 111; NICE NG9 1.2.1 immediate 999 referral needs examination findings; modifier below |
| Acute asthma exacerbation | 3 | 2 | 2 | high | NHS.uk: 999 if not better after maximum reliever; CTAS asthmatic breathlessness mild = 3, moderate = 2; modifier below |
| Chagas | 3 | 3 | 5 | low | CDC: treat all acute cases; no time-critical disposition |
| Cluster headache | 3 | 3 | 3 | medium | NTS U3 includes "humane redenen" (severe pain); NTS: severe pain is U2 in most complaints; NHS.uk: see a GP; CTAS recurring headache = 5 |
| GERD | 3 | 4 | 5 | high | NHS.uk: pharmacist or self-care first |
| HIV (initial infection) | 3 | 3 | 5 | medium | NHS.uk: sexual health clinic or GP; CDC: start treatment as soon as possible after diagnosis |
| Influenza | 3 | 3 | 5 | high | NHS.uk: self-care; modifier below |
| Inguinal hernia | 3 | 4 | 5 | high | NHS.uk: see a GP; 111 only if painful with vomiting or fever |
| Myasthenia gravis | 3 | 3 | 5 | medium | NHS.uk: see a GP; hospital only if breathing or swallowing suddenly worsens |
| Pancreatic neoplasm | 3 | 3 | 5 | high | NICE NG12 1.2.4-1.2.5: 2-week pathway |
| Pneumonia | 3 | 2 | 3 | high | NHS.uk: urgent GP or 111 for cough with chest pain or breathlessness; modifier below |
| Pulmonary neoplasm | 3 | 3 | 5 | high | NICE NG12 1.1.1-1.1.2: 2-week pathway, chest X-ray within 2 weeks |
| Spontaneous rib fracture | 3 | 4 | 5 | medium | NHS.uk: self-care, heals in 2-6 weeks |
| Tuberculosis | 3 | 3 | 5 | medium | NHS.uk: see a GP; NICE NG33 urgent referral to a TB service |
| Acute laryngitis | 4 | 4 | 5 | high | NHS.uk: self-care or pharmacist |
| Acute otitis media | 4 | 4 | 5 | high | NHS.uk: pharmacist first; modifier below |
| Acute rhinosinusitis | 4 | 4 | 5 | high | NHS.uk: self-care then pharmacist; modifier below |
| Allergic sinusitis | 4 | 5 | 5 | high | NHS.uk: often treated without a GP; CTAS hay fever = 5 |
| Anemia | 4 | 3 | 5 | medium | NHS.uk: non-urgent, see a GP |
| Bronchitis | 4 | 5 | 5 | high | NHS.uk: self-care; CTAS URTI, well, no fever = 5; modifier below |
| Localized edema | 4 | 3 | 3 | medium | NHS.uk: urgent GP or 111 for one swollen leg without a known cause; CTAS swelling above the ankles = 4 |
| Pericarditis | 4 | 2 | 3 | medium | NHS.uk: urgent GP or 111 for sharp pleuritic chest pain; ESC same-day risk stratification; CTAS acute non-cardiac chest pain = 3 would give level 2 |
| SLE | 4 | 3 | 5 | medium | NHS.uk: see a GP |
| Sarcoidosis | 4 | 3 | 5 | low | NHS.uk gives no urgent disposition |
| Viral pharyngitis | 4 | 4 | 5 | high | NHS.uk: self-care or pharmacist; CTAS sore throat, not severe, under 39 C = 5; modifier below |
| Whooping cough | 4 | 3 | 5 | medium | NHS.uk: urgent only for babies under 6 months or a rapidly worsening cough; modifier below |
| Chronic rhinosinusitis | 5 | 4 | 5 | high | NHS.uk: GP may refer to ENT after 3 months |
| Panic attack | 5 | 4 | 5 | medium | NHS.uk: self-care during an attack, see a GP for panic disorder |
| URTI | 5 | 5 | 5 | high | NHS.uk: treat at home, passes within 1-2 weeks; CTAS URTI = 5 |

### Patient-level modifiers

Each rule uses only fields DDXPlus provides (age in whole years, sex, antecedent codes) and a published rule that names the same trigger. The CSV `modifiers` column holds the same rules.

| Condition | Rule | Published basis |
|---|---|---|
| Pneumonia | age >= 65, or chronic lung disease (E_123, E_31), heart failure (E_106) or immunosuppression (E_227, E_2, E_34): level 2 | NHS.uk: may need hospital treatment if over 65 or with cardiovascular or chronic lung disease; CRB-65 gives one point for age >= 65 (Lim 2003, via the validation doc); NTS risk-group rule raises urgency for chronic disease and immunosuppression |
| Acute asthma exacerbation | E_101 hospitalised for asthma in the past year, or E_46 two or more attacks in a year: level 1 | BTS/SIGN near-fatal asthma risk factors (via the validation doc) |
| Bronchiolitis | age < 1 and (E_160 prematurity or E_139 congenital heart defect): level 2 | NICE NG9 1.3.3 risk factors for severe bronchiolitis: under 3 months, prematurity, congenital heart disease. DDXPlus age is in years, so the 3-month cut is not testable |
| Influenza | age >= 65, pregnancy (E_167), chronic disease (E_123, E_31, E_124, E_106, E_69, E_113, E_126) or immunosuppression (E_227, E_2, E_34): level 3 | NHS.uk flu: urgent GP or 111 for these groups; CDC: antivirals as soon as possible for higher-risk groups |
| Bronchitis | same trigger set as influenza: level 3 | NHS.uk bronchitis: urgent GP or 111 if over 65, pregnant, chronic disease or immunosuppressed |
| Acute otitis media | immunosuppression (E_227), diabetes (E_69), heart failure (E_106), COPD (E_123, E_31), CKD (E_113): level 3 | NHS.uk ear infections: urgent GP or 111 for chronic disease or weakened immunity; NTS risk-group rule |
| Acute rhinosinusitis, viral pharyngitis | immunosuppression (E_227): level 3 | NHS.uk sinusitis and sore throat: urgent GP or 111 if immunosuppressed |
| Whooping cough | age < 1: level 3 | NHS.uk: urgent GP or 111 if the baby is under 6 months; whole-year ages make under 1 the nearest testable cut |

The validation doc's caution still applies: DDXPlus samples antecedents from condition priors, so a young pneumothorax patient can carry a COPD flag, and the 250-case eval set has no patient under 10, so the infant rules never fire there.

### Disagreements with the reference ordinal and with DDXPlus

Against the reference ordinal, no condition moves to a more urgent level and 27 move to a less urgent one. All 27 are compression, not reversal. 21 are reference levels 3 and 4 collapsing into level 5 (12 from level 3, 9 from level 4). Four are same-day (reference 2) conditions placed at level 3, "within a few hours": COPD exacerbation, bronchiolitis, pneumonia and pericarditis. Atrial fibrillation goes from reference 2 to level 4 on the NICE NG196 and NHS.uk wording, and stable angina from reference 3 to level 4 because the NHS.uk angina page ties new angina symptoms to the urgent rung. The three reference level-5 conditions (allergic sinusitis, bronchitis, URTI) match exactly.

Against DDXPlus severity, 7 conditions are more urgent on the new scale and 25 less. The 7 are the set the validation doc flagged, minus laryngospasm: Boerhaave, epiglottitis, pulmonary embolism and unstable angina (DDXPlus 2, level 1); acute asthma (3 to 2); pericarditis and localized edema (4 to 3). The two-step moves are all downward: 11 DDXPlus-3 conditions sit at level 5 (bronchiectasis, Chagas, GERD, HIV, influenza, inguinal hernia, myasthenia gravis, pancreatic neoplasm, pulmonary neoplasm, rib fracture, tuberculosis) and stable angina goes from DDXPlus 2 to level 4.

At the escalation cutoff (levels 1-3 count as urgent on the new scale, matching "within a few hours"): the mapping keeps 20 of the reference's 21 urgent conditions (atrial fibrillation is the exception, at level 4) and adds scombroid, cluster headache and localized edema. Against DDXPlus severity <= 2, it keeps 16 of 17 (stable angina drops to level 4) and adds COPD exacerbation, bronchiolitis, pneumonia, cluster headache, localized edema and pericarditis.

## 4. Methodology questions

### 4a. Is "up-triage when a dangerous condition cannot be excluded" an endorsed principle?

Yes, in four places, with two counterweights.

1. **CTAS 2014 revisions (Bullard et al., CJEM):** "Nurses are trained to assign a higher score if their clinical judgment suggests that the patient may be sicker than the score indicated applying the most relevant CTAS modifiers. The CTAS NWG does not approve assigning a lower score than CTAS indicates." The rule is one-way: up is allowed, down is not.
2. **ESI v4 Implementation Handbook (AHRQ 2005):** "the ED management team might stipulate that, when in doubt about a patient's triage rating, nurses err on the side of over-triage. While this approach might result in some patients being mis-triaged as more acute than they actually are, it is preferable to risking an adverse event because the patient was triaged to a less urgent category." The v4 vital-sign rule reads "Consider uptriage to ESI 2 if any vital sign criterion is exceeded."
3. **Manchester Triage System (Emergency Triage, 3rd ed.):** "Discriminators that indicate higher levels of priority are sought first, and to a large degree patients who are allocated to the standard / 4 / green clinical priority are selected by default." The method is "reductive discriminator seeking": the practitioner rules out the top of the chart before settling lower.
4. **NTS working method (April 2025):** "De NTS is opgebouwd van hoog- naar laagurgente vragen. Per ingangsklacht moeten eerst de hoog urgente criteria beoordeeld worden" (NTS runs from high- to low-urgency questions; the high-urgency criteria of each complaint must be assessed first), and "Bij twijfel over ABCD ... moeten de ABCD-criteria allemaal worden beoordeeld" (when in doubt about ABCD, all ABCD criteria must be assessed). The document also warns that a wrongly completed ABCD "adviseert NTS mogelijk een te lage urgentie" (may make NTS advise too low an urgency).
5. **Trauma systems (Newgard 2022, national field triage guideline):** "Trauma systems have prioritized the goal of minimizing undertriage and accepting a higher level of overtriage to avoid increased mortality, with targets set at <=5% and <=35%, respectively."

Counterweights: the same ESI handbook says "Initial overcategorization (overtriage) uses scarce resources, limiting availability of an open ED bed for another patient who may require immediate care", and the telephone-triage review by Huibers 2011 says "over-triage could also compromise patient safety, if it results in an overload with insufficient resources and delay in treatment." The observed behaviour of algorithmic telephone triage is risk-averse: NHS 111 primary triage shows "high sensitivity but low specificity" (Sexton 2025), and symptom checkers are "generally risk averse" (Semigran 2015).

Implication for scoring: the endorsed principle is to rate for the most dangerous condition that the information cannot exclude, not to rate everything up. A model that rates a URTI at level 3 is not following the principle; a model that rates chest pain at level 1 when its differential holds a level-1 condition is. The asymmetric ordinal cost already charges over-rating less than under-rating. To reward the principle rather than blanket caution, score over-rating against the most urgent condition in the DDXPlus differential with meaningful probability, not only against the true condition, so a rating that matches a plausible dangerous differential costs less than one with no such support. The per-level probabilities the v0.2 design asks for make that test direct.

### 4b. What inter-rater reliability do humans reach on vignettes?

| Study | Scale | Format | Raters x cases | Agreement |
|---|---|---|---|---|
| Beveridge 1999 | CTAS | written case summaries | 20 x 50 | kappa 0.80 (nurses 0.84, physicians 0.83) |
| Manos 2002 | CTAS | written scenarios | 20 x 42 | quadratic weighted 0.77 |
| Worster 2004 | CTAS vs ESI | 200 cases | 10 nurses | quadratic weighted 0.91 vs 0.89 |
| Dallaire 2012 | CTAS | written scenarios | 5 x 100 | quadratic weighted 0.44 (pairwise 0.30-0.61) |
| Eitel 2003 | ESI v2 | written and live | >200 nurses x 40 written; 386 live | weighted 0.70-0.80 written, 0.69-0.87 live |
| Mistry 2018 | ESI | AHRQ scenarios | 87 nurses, 3 countries | accuracy 59.2%, alpha 0.73 |
| Storm-Versloot 2009 | MTS vs ESI | 50 scenarios | 18 nurses | quadratic weighted 0.82 (MTS), unweighted 0.76 vs 0.46 |
| van der Wulp 2008 | MTS | vignettes | - | weighted 0.62; test-retest ICC 0.75 |
| van Veen 2010 | MTS | written vs live | - | weighted 0.83 written, 0.65 live |
| Considine 2004 | ATS | paper vs computer scenarios | 167 x 2,349 adult | kappa 0.42 paper, 0.56 computer; 61% expected, 18% under, 21% over |
| Ebrahimi 2015 (pooled) | ATS | all paper, unweighted | 6 studies | 0.43 (0.34-0.51); 60.8% agreement, 20.7% over, 18.5% under |
| Mirhaghi 2015 (pooled) | CTAS | 14 studies | - | 0.67 (0.60-0.74) |
| Mirhaghi 2017 (pooled) | MTS | 7 studies | - | 0.75 (0.68-0.81) |
| Smits 2020 | NTS | paper cases, telephone and physical | 116 x 40 | ICC 0.73; 62.3% exact vs expert panel; 17.4% under, 20.2% over; 77% of disagreements one level |
| Giesen/Derkx 2016 | Dutch GP telephone urgency | written cases | 973 respondents | 63.6% adequate, 19.3% over, 17.1% under |
| Giesen 2007 | Dutch GP cooperatives | mystery patients | 352 calls | 69% correct, 19% under; sensitivity 0.76, specificity 0.95 |

Hinson 2019 (systematic review, 42 reliability evaluations) found "only a minority (11 of 42) reporting kappa above 0.8". The two format comparisons disagree: Eitel found written and live agreement similar; van Veen found written higher (0.83 vs 0.65).

Noise ceiling: against a single labeller, expect quadratic weighted kappa of about 0.75-0.85 and exact agreement of about 60-65% from trained humans on paper cases; the NTS figures (ICC 0.73, 62.3% exact, 77% of misses one step) are the closest match to our setting. A model whose quadratic weighted kappa against the label exceeds 0.85 is at the ceiling of what the label can resolve, and differences between models above that line are not interpretable.

### 4c. Evidence linking under-triage and over-triage or crowding to harm

Under-triage:

| Study | Setting, n | Effect |
|---|---|---|
| Haas 2010, J Am Coll Surg | major trauma, 11,398 | mortality OR 1.24 (1.10-1.40) for undertriage to a non-trauma centre after survivor-bias correction |
| Rogers 2013, Eur J Trauma Emerg Surg | level I trauma centre, 18,324; 6.3% undertriaged | mortality OR 3.0 (2.4-3.8) |
| MacKenzie 2006, NEJM | 18 trauma vs 51 non-trauma centres | in-hospital mortality 7.6% vs 9.5%, RR 0.80 (0.66-0.98); the mechanism behind trauma undertriage harm |
| Sax 2023, JAMA Netw Open | 21 EDs, 5,315,176 encounters, ESI | mistriage 32.2%, undertriage 3.3%; 60.9% of patients who received level-1 interventions were undertriaged; no mortality association reported |
| Sax 2026, Ann Emerg Med | same cohort | undertriaged high-acuity patients waited 8 minutes longer |
| Hinson 2018, Int J Emerg Med | one ED, 96,071 | undertriaged patients had "significantly higher" admission and critical-outcome rates; no per-stratum mortality |
| Seymour 2017, NEJM | sepsis, 49,331 | in-hospital mortality OR 1.04 per hour of delay to antibiotics (the delay mechanism) |
| Huibers 2011 (review) | out-of-hours telephone triage | safe in 97% of all contacts but 89% of high-urgency contacts and 46% of simulated high-risk patients; adverse events included deaths in 6 studies |
| Sexton 2025 | NHS 111, 98,946 | 1.5% of patients given same-day-or-less-urgent dispositions were admitted; 63.4% of those discharged within a day |

Over-triage and crowding:

| Study | Setting, n | Effect |
|---|---|---|
| Sun 2013, Ann Emerg Med | 187 hospitals, 995,379 visits | admission on high-crowding days: 5% greater odds of inpatient death (2-8%), about 300 excess deaths |
| Guttmann 2011, BMJ | Ontario, 13.9 million discharged patients | shifts with mean ED stay >= 6 h vs < 1 h: 7-day death aOR 1.79 (high acuity), 1.71 (low acuity) |
| Jones 2022, Emerg Med J | England, 7.5 million admissions | mortality rises linearly from 5 h to 12 h in the ED; one extra death per 82 patients delayed 6-8 h |
| Roussel 2023, JAMA Intern Med | France, 1,598 patients >= 75 | overnight ED stay: in-hospital mortality aRR 1.39 (1.07-1.81) |
| Bernstein 2009 (review) | 41 studies | crowding linked to higher mortality, slower pneumonia and pain treatment; no RCTs |
| Sax 2026 | 5.3 million | overtriaged low-acuity patients stayed 42 minutes longer |
| Newgard 2013, Health Aff | 7 regions, 301,214 injured | 34.3% of low-risk patients went to major trauma centres; $5,590 more per episode |
| Turner 2013, BMJ Open | NHS 111 pilots, 277,163 calls | ambulance incidents +2.9% (1.0-4.8); no fall in ED attendances |
| Semigran 2015, BMJ | 23 symptom checkers | self-care vignettes triaged correctly 33%; advice "generally risk averse" |

What this supports: under-triage harms through delay, with direct mortality odds ratios in trauma and a delay mechanism in sepsis; crowding and long ED stays raise mortality with consistent effects across four countries. What it does not support: no verified study shows that over-triage itself causes deaths. The over-triage harm chain runs through cost, longer stays, extra ambulance demand and crowding, and the crowding studies do not attribute crowding to triage decisions. The asymmetry in the harm evidence matches the asymmetry in the benchmark's cost function and in the U = 5%, O = 35% tolerances in `spec/triage_tolerances.md`.

## 5. Agreement statistics

Computed over the 49 conditions with the script in the scratchpad (`agree.py`; it reproduces the validation doc's DDXPlus-vs-reference figures exactly).

| Statistic | Mapping vs DDXPlus severity | Mapping vs evidence reference | DDXPlus vs reference (for comparison) |
|---|---|---|---|
| Exact agreement | 17 of 49 (34.7%) | 22 of 49 (44.9%) | 24 of 49 (49.0%) |
| Within one level | 37 of 49 (75.5%) | 36 of 49 (73.5%) | 48 of 49 (98.0%) |
| Quadratic weighted kappa | 0.657 | 0.698 | 0.766 |
| Linear weighted kappa | 0.466 | 0.535 | 0.567 |
| Unweighted kappa | 0.235 | 0.349 | 0.325 |
| Spearman rho | 0.797 | 0.898 | 0.781 |
| Mapping more urgent | 7 | 0 | - |
| Mapping less urgent | 25 | 27 | - |

Confusion, mapping (rows) against the evidence reference (columns 1-5): row 1 [8, 0, 0, 0, 0]; row 2 [0, 8, 0, 0, 0]; row 3 [0, 4, 3, 0, 0]; row 4 [0, 1, 1, 0, 0]; row 5 [0, 0, 12, 9, 3]. The mapping is a monotone coarsening of the reference: every reference level-1 condition is level 1, every level-2 condition is level 2 or 3 except atrial fibrillation, and levels 3-5 collapse into level 5. The lower kappas against the reference therefore measure the compression, not disagreement on order (rho 0.90).

Confusion against DDXPlus (columns 1-5): row 1 [4, 4, 0, 0, 0]; row 2 [1, 6, 1, 0, 0]; row 3 [0, 1, 4, 2, 0]; row 4 [0, 1, 1, 0, 0]; row 5 [0, 0, 11, 10, 3].

## 6. Limitations

1. **The NTS complaint criteria are licensed, so no placement cites an NTS criterion directly.** We placed conditions with public NHS.uk, CTAS and NICE rules and cross-walked them to NTS time anchors by the table in section 2. A reader with NTS access should check the 15 conditions at levels 2-4 against the actual entry-complaint criteria.
2. **Levels 3 and 4 are hard to separate by condition.** NHS.uk's single "urgent GP or 111" rung spans NHS Pathways dispositions from 1 to 24 hours, and the NTS owner does not define exact response times. Only two conditions sit at level 4 (stable angina, atrial fibrillation), both low confidence. Moving either to level 3 or 5 changes no kappa by more than 0.02.
3. **Level 5 holds 24 conditions.** That is the price of a primary-care scale: "not today" is the last decision it makes. The benchmark's escalation label and the DDXPlus severity field keep the finer ordering among those 24 if a future analysis needs it.
4. **NICE CG95 (chest pain), NG158 (PE), NG33 (TB), the 2015 ESC pericarditis guideline, BTS/SIGN asthma and GOLD were not re-fetched this session** (nice.org.uk returned 403 for guidance pages and the session's web-search budget was exhausted). Where they support a placement, the table says "via docs/ddxplus-severity-validation.md". NICE NG9, NG12, NG115 and NG196 were read as PDF text this session.
5. **Several reliability figures come from abstracts** that do not state whether the kappa was weighted (Beveridge 1999, Tanabe 2004, Considine 2004). The Smits 2020 NTS study is in Dutch and reports an ICC, not kappa, and used paediatric cases only.
6. **The van Wegen 2025 outcome figures group NTS levels into four bands** whose composition the paper does not state in the text we read, so we cite them as a range from lowest to highest band.
7. **The mapping is one author's application of the crosswalk.** The validation doc's 20 medium- and 6 low-confidence placements carry over; the additional judgment here is the level-3-versus-4 split and the CTAS-2-to-level-1-or-2 rule.

## Sources

```
https://www.de-nts.nl/app/uploads/2025/05/Werkwijze-NTS-april_25.pdf
https://www.de-nts.nl/home/wat-is-nts/
https://triagewijzer.nhg.org/verantwoording
https://www.nivel.nl/sites/default/files/bestanden/1004494.pdf
https://doi.org/10.1093/fampra/cmq097
https://pubmed.ncbi.nlm.nih.gov/32940982/
https://pmc.ncbi.nlm.nih.gov/articles/PMC12044865/
https://pmc.ncbi.nlm.nih.gov/articles/PMC13064533/
https://media.emscimprovement.center/documents/Emergency_Severity_Index_Handbook.pdf
https://sgnor.ch/fileadmin/user_upload/Dokumente/Downloads/Esi_Handbook.pdf
https://doi.org/10.1111/j.1553-2712.2000.tb01066.x
https://doi.org/10.1111/j.1553-2712.2003.tb00577.x
https://doi.org/10.1197/j.aem.2003.06.013
https://doi.org/10.1016/j.annemergmed.2017.09.036
https://doi.org/10.1016/j.annemergmed.2018.09.022
https://ctas-phctas.ca/wp-content/uploads/2018/05/ctased16_98.pdf
https://ctas-phctas.ca/wp-content/uploads/2018/05/revisions_to_the_canadian_emergency_department_triage_and_acuity_scale_ctas_guidelines_2016.pdf
http://ctas-phctas.ca/wp-content/uploads/2018/05/participant_manual_v2.5b_november_2013_0.pdf
http://ctas-phctas.ca/wp-content/uploads/2018/05/ctas_guidelines_-_2014.pdf
https://files.ontario.ca/moh_3/moh-manuals-prehospital-ctas-paramedic-guide-v2-0-en-2016-12-31.pdf
https://staffnet.southernhealth.ca/wp-content/uploads/CTAS-Interactive-Quick-Look-Booklet.pdf
https://doi.org/10.1016/s0196-0644(99)70223-4
https://doi.org/10.1017/s1481803500006023
https://doi.org/10.1017/s1481803500009192
https://doi.org/10.1016/j.jemermed.2011.05.085
https://pmc.ncbi.nlm.nih.gov/articles/PMC4525387/
https://pmc.ncbi.nlm.nih.gov/articles/PMC5289484/
https://pmc.ncbi.nlm.nih.gov/articles/PMC2548283/
https://pmc.ncbi.nlm.nih.gov/articles/PMC7872278/
https://pmc.ncbi.nlm.nih.gov/articles/PMC3893080/
https://doi.org/10.1136/emj.2008.059378
https://doi.org/10.1111/jebm.12231
https://content.e-bookshelf.de/media/reading/L-4033541-3f8b45baf4.pdf
https://policy.acem.org.au/index.php/policies-menu/p06-policy-on-the-australasian-triage-scale
https://acem.org.au/getmedia/51dc74f7-9ff0-42ce-872a-0437f3db640a/G24_04_Guidelines_on_Implementation_of_ATS_Jul-16.aspx
https://doi.org/10.1016/j.annemergmed.2004.04.007
https://pmc.ncbi.nlm.nih.gov/articles/PMC4458479/
https://doi.org/10.1111/j.1553-2712.1998.tb02599.x
https://pmc.ncbi.nlm.nih.gov/articles/PMC6549628/
https://pmc.ncbi.nlm.nih.gov/articles/PMC3150303/
https://www.england.nhs.uk/statistics/wp-content/uploads/sites/2/2022/11/IUC-ADC-REVISED-Dx-code-mapping-Oct-2022-v2-1.xlsx
https://www.londonambulance.nhs.uk/about-us/our-performance/ambulance-response-categories/
https://pmc.ncbi.nlm.nih.gov/articles/PMC11792721/
https://pmc.ncbi.nlm.nih.gov/articles/PMC8109810/
https://pmc.ncbi.nlm.nih.gov/articles/PMC12005381/
https://pmc.ncbi.nlm.nih.gov/articles/PMC11240063/
https://pmc.ncbi.nlm.nih.gov/articles/PMC6567030/
https://pmc.ncbi.nlm.nih.gov/articles/PMC3844302/
https://pmc.ncbi.nlm.nih.gov/articles/PMC12096499/
https://pmc.ncbi.nlm.nih.gov/articles/PMC9323557/
https://pmc.ncbi.nlm.nih.gov/articles/PMC3308461/
https://pmc.ncbi.nlm.nih.gov/articles/PMC4496786/
https://pmc.ncbi.nlm.nih.gov/articles/PMC4911030/
https://doi.org/10.1136/qshc.2006.018846
https://pmc.ncbi.nlm.nih.gov/articles/PMC10024207/
https://doi.org/10.1016/j.annemergmed.2025.11.018
https://pmc.ncbi.nlm.nih.gov/articles/PMC5768578/
https://doi.org/10.1016/j.jamcollsurg.2010.08.014
https://doi.org/10.1007/s00068-013-0289-z
https://doi.org/10.1056/NEJMsa052049
https://doi.org/10.1056/NEJMoa1703058
https://doi.org/10.1016/j.annemergmed.2012.10.026
https://doi.org/10.1136/bmj.d2983
https://doi.org/10.1136/emermed-2021-211572
https://doi.org/10.1001/jamainternmed.2023.5961
https://doi.org/10.1111/j.1553-2712.2008.00295.x
https://pmc.ncbi.nlm.nih.gov/articles/PMC4044817/
https://doi.org/10.1136/bmjopen-2013-003451
https://www.nice.org.uk/guidance/ng196
https://www.nice.org.uk/guidance/ng9
https://www.nice.org.uk/guidance/ng115
https://docs.bvsalud.org/biblioref/2022/02/1355289/suspected-cancer-recognition-and-referral-pdf-1837268071621.pdf
https://www.nhs.uk/conditions/
https://www.cdc.gov/flu/hcp/antivirals/summary-clinicians.html
https://www.cdc.gov/yellow-book/hcp/environmental-hazards-risks/food-poisoning-from-marine-toxins.html
https://www.who.int/news-room/fact-sheets/detail/ebola-disease
```
