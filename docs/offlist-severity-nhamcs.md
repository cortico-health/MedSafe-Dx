# Off-list severity tiers from NHAMCS

Date: 2026-09-26. Scope: ICD-10-CM codes that models emit but that name none of the 49 DDXPlus conditions under the standard map policy (`evaluator/condition_match.py`). Code: `scripts/analysis/offlist_severity_nhamcs.py`. Data outputs under `results/analysis/nhamcs_offlist/`; the tier table the scorer reads is `spec/offlist_tiers_nhamcs.csv`. Every figure below comes from that script run on the NHAMCS files in `data/external/nhamcs/` and the run files present at the time of writing; the coverage figures move as new runs land.

## Summary

1. **One rule on ED disposition reproduces our v0.3 tiers well on the conditions NHAMCS covers: tier 1 if critical-care admission >= 5% or admission >= 50%; tier 3 if admission < 5%; else tier 2.** On the 26 DDXPlus conditions with 30+ adult primary-diagnosis visits it agrees with `spec/dangerous_if_missed_tiers_v03b.csv` at quadratic weighted kappa 0.83, 20 of 26 exact, all 26 within one tier. Leave-one-out kappa is 0.72 and 24 of 26 folds pick the same thresholds (section 2).
2. **The rule's six misses are the known construct gap: ED disposition rates treat-and-release danger low and chronic admission high.** Asthma exacerbation (admission 5.7%) and PSVT (20%) fall to tier 2 against our tier 1; anemia (40%) and localized edema (8%) rise to tier 2 against our tier 3; AF (ICU 10%) rises to tier 1; GERD (2.2%) falls to tier 3 (section 2).
3. **`spec/offlist_tiers_nhamcs.csv` has 1,425 prefixes: 266 tier 1 (145 by Newman-Toker or AHRQ override, 121 by the rule), 237 tier 2, 178 tier 3, 744 unscored.** Off-list mentions in the A/B and v6 runs land 31% tier 1, 20% tier 2, 19% tier 3, 31% unscored; the unscored share is mostly codes with under 30 ED visits (21% of mentions) and symptom codes (8%). Arm-4 flags in the v6 runs are 61% tier 1 (section 4).
4. **Clinically surprising assignments exist and are reported, not corrected.** Venomous bites T63 and adverse effects T78 are tier 3 (treated and released); shock R57 is unscored as a symptom code despite 45% critical care; ARDS J80 is unscored at 29 visits; sleep disorders G47, cellulitis L03, gastritis K29 and adjustment disorder F43 are tier 2; 31 tier-1 rows rest on fewer than five critical-care visits (2.5% of off-list mentions) (section 5).
5. **The tiers measure where US ED patients with a coded diagnosis went, not the harm of missing it.** Coding, visit weights, the 4-character public codes and small counts all shape the result (section 6). The Newman-Toker and AHRQ overrides carry the "dangerous if missed" construct; the NHAMCS rule fills in behind them.

## 1. Data and outcomes by code

We use the NHAMCS ED public-use files 2016-2022 (`docs/third-party-urgency-source.md` section 1), restricted to patients aged 18 and over: 96,539 of 123,040 visit records. For every ICD-10-CM code (4 characters, as the public file carries them) and every 3-character group, as primary diagnosis (`DIAG1`) and as any of the five listed diagnoses, the script writes to `results/analysis/nhamcs_offlist/outcomes_by_code.csv`:

| Column | Meaning |
|---|---|
| n, n_admit, n_icu, n_death | unweighted visits, and how many of them were admitted, went to a critical-care unit, or died |
| weighted_n | sum of `PATWT` |
| admission | weighted share with `ADMITHOS`, `OBSHOS` or `TRANOTH` = 1 |
| icu | weighted share with `ADMIT` = 1 (admitted to a critical care unit) |
| death | weighted share with `DIEDED` or `DOA` = 1 |
| n_triaged, wmean_immedr, p_immedr_1..5 | arrival triage immediacy among visits with `IMMEDR` 1-5 |

We drop rows under 30 unweighted visits, as NCHS advises for published estimates. Published rows: 477 codes and 368 groups as primary diagnosis, 871 codes and 566 groups as any diagnosis; 6,235 rows suppressed.

## 2. Calibration on the overlap

The reference is `final_tier` in `spec/dangerous_if_missed_tiers_v03b.csv`. Condition code sets come from `spec/ddxplus_icd10_map.csv` (equivalent and narrower rows), matched on the first four characters as `scripts/analysis/nhamcs_urgency.py` does. Twenty-six conditions have 30+ adult primary-diagnosis visits; the other 23 (Ebola, Boerhaave, croup, bronchiolitis, pneumothorax at 28, pulmonary neoplasm at 28, and 17 more) do not.

The rule family has three rates and up to four thresholds: tier 1 if ICU >= a, or ED death >= b, or admission >= c; tier 3 if admission < d; else tier 2. We search round values only (a in 5-25%, b in none/0.5/1/2%, c in none/50-90%, d in 5-40%) and rank by kappa, then exact matches, then fewer clauses. The death clause never helps: ED deaths are rare for every overlap condition except MI (4.9%), which the other clauses already catch. The chosen rule:

```
tier 1 if ICU >= 5% or admission >= 50%
tier 3 if admission < 5%
tier 2 otherwise
```

| Condition | n | Admission | ICU | Death | Our tier | Rule tier |
|---|---|---|---|---|---|---|
| Possible NSTEMI / STEMI | 258 | 0.891 | 0.157 | 0.0488 | 1 | 1 |
| Pulmonary embolism | 117 | 0.805 | 0.137 | 0.0000 | 1 | 1 |
| Acute pulmonary edema | 34 | 0.724 | 0.250 | 0.0000 | 1 | 1 |
| Unstable angina | 84 | 0.709 | 0.060 | 0.0024 | 1 | 1 |
| Stable angina | 86 | 0.569 | 0.011 | 0.0020 | 1 | 1 |
| Pneumonia | 969 | 0.493 | 0.061 | 0.0002 | 1 | 1 |
| Anaphylaxis | 48 | 0.118 | 0.106 | 0.0000 | 1 | 1 |
| PSVT | 81 | 0.201 | 0.049 | 0.0000 | 1 | **2** |
| Bronchospasm / acute asthma exacerbation | 801 | 0.057 | 0.006 | 0.0000 | 1 | **2** |
| Atrial fibrillation | 479 | 0.522 | 0.102 | 0.0000 | 2 | **1** |
| Acute COPD exacerbation / infection | 578 | 0.377 | 0.044 | 0.0000 | 2 | 2 |
| HIV (initial infection) | 36 | 0.417 | 0.000 | 0.0000 | 2 | 2 |
| Spontaneous rib fracture | 220 | 0.187 | 0.029 | 0.0000 | 2 | 2 |
| Inguinal hernia | 91 | 0.144 | 0.000 | 0.0000 | 2 | 2 |
| Influenza | 425 | 0.071 | 0.004 | 0.0005 | 2 | 2 |
| GERD | 227 | 0.022 | 0.002 | 0.0000 | 2 | **3** |
| Anemia | 554 | 0.400 | 0.048 | 0.0003 | 3 | **2** |
| Localized edema | 153 | 0.080 | 0.000 | 0.0000 | 3 | **2** |
| Panic attack | 826 | 0.032 | 0.004 | 0.0000 | 3 | 3 |
| Allergic sinusitis | 56 | 0.020 | 0.000 | 0.0000 | 3 | 3 |
| Bronchitis | 890 | 0.018 | 0.000 | 0.0000 | 3 | 3 |
| URTI | 1029 | 0.017 | 0.004 | 0.0000 | 3 | 3 |
| Viral pharyngitis | 668 | 0.008 | 0.000 | 0.0000 | 3 | 3 |
| Acute otitis media | 214 | 0.006 | 0.000 | 0.0000 | 3 | 3 |
| Acute rhinosinusitis | 230 | 0.004 | 0.000 | 0.0000 | 3 | 3 |
| Chronic rhinosinusitis | 158 | 0.002 | 0.000 | 0.0000 | 3 | 3 |

Confusion table (rows: our tier; columns: rule tier):

| | Rule 1 | Rule 2 | Rule 3 |
|---|---|---|---|
| Ours 1 | 7 | 2 | 0 |
| Ours 2 | 1 | 5 | 1 |
| Ours 3 | 0 | 2 | 8 |

Agreement: quadratic weighted kappa 0.833, 20 of 26 exact, 26 of 26 within one tier. Against `base_tier` (the DDXPlus severity crosswalk before upgrades) kappa is 0.822; against `final_tier_formal` 0.811.

Stability:

| Check | Result |
|---|---|
| Leave-one-out (refit on 25, predict the held-out condition) | kappa 0.72, 19 of 26 exact |
| Thresholds chosen across the 26 folds | 24 folds: the chosen rule; 1 fold: routine line at 10%; 1 fold: drops the admission >= 50% clause |
| Routine line d at 5 / 10 / 15 / 20 / 25 / 30 / 40% (a, c fixed) | kappa 0.833 / 0.767 / 0.748 / 0.729 / 0.662 / 0.662 / 0.677 |
| ICU line a at 5 / 10 / 15 / 20 / 25% (c, d fixed) | kappa 0.833 / 0.799 / 0.763 / 0.763 / 0.763 |
| Same kappa with a death clause added (any b) | 0.833; the clause changes no overlap condition, so we leave it out |

The routine line at 5% is the sharpest choice and the least comfortable: it puts every code with more than one admission in twenty into tier 2. Raising it to 10% costs 0.07 kappa on the overlap (GERD would still be tier 3; influenza would drop to tier 3) and moves the emitted-code shares in section 4 from tier 2 toward tier 3.

## 3. Rule for off-list codes and the spec table

The script applies three steps in order to every 3-character group that NHAMCS, the runs or the override list mention, and to every finer prefix that has a basis of its own:

1. **Override, tier 1:** the code falls under a Newman-Toker 2023 Table 1 group (`spec/offlist_escalation_groups.csv`: stroke, VTE, arterial thromboembolism, aortic aneurysm and dissection, MI, sepsis, pneumonia, meningitis and encephalitis, spinal abscess, endocarditis, cancers) or under one of the five AHRQ 2022 ED top-15 groups that Table 1 lacks. The override is upgrade-only: NHAMCS rates are reported beside it but cannot lower it.
2. **Unscored:** symptom codes R00-R99, factor codes Z00-Z99, external-cause codes V00-Y99, and any prefix with under 30 adult visits both as primary and as any listed diagnosis.
3. **NHAMCS rule:** the section 2 rule on the prefix's primary-diagnosis rates; when the primary count is under 30 but the any-listed count is not, the any-listed rates, marked "any diagnosis" in `source` (208 of the 536 rule rows).

AHRQ 2022 prefixes are our reading of the report's condition names; the ten other AHRQ conditions are Table 1 rows already in the spec CSV:

| AHRQ ED top-15 condition | Prefixes |
|---|---|
| Spinal cord compression | G95.2 |
| Traumatic brain injury and intracranial haemorrhage | S06 (non-traumatic I60-I62 sit in the stroke group) |
| Cardiac arrhythmia | I44, I45, I46, I47, I49 (I48 is on the DDXPlus list) |
| GI perforation and rupture | K63.1; K25-K28 with perforation (.1 .2 .5 .6); K35.2, K35.3; K57.0, K57.2, K57.4, K57.8; K65 |
| Intestinal obstruction | K56, K31.5, and hernia codes with obstruction (K41.0, K41.3, K42.0, K43.0, K43.3, K43.6, K44.0, K45.0, K46.0) |

`spec/offlist_tiers_nhamcs.csv` columns: `icd10_prefix`, `description` (CMS ICD-10-CM FY2026 order file; "not an ICD-10-CM FY2026 code" for WHO-only codes models emit, such as I64), `tier` (1, 2, 3 or `unscored`), `rule_path` (`override`, `nhamcs`, `unscored`), `admission`, `icu`, `death`, `n` (the NHAMCS rates and unweighted count behind the row, blank rates when suppressed), `source`. Rows are 3-character groups plus finer prefixes only where the finer prefix has its own override or 30+ visits and its tier differs from the group's (T78 tier 3 but T78.2 anaphylactic shock and T78.3 angioedema tier 1; R65 unscored but R65.2 severe sepsis tier 1). A consumer takes the longest matching prefix; a subcode with no row of its own inherits its group. The scorer resolves the DDXPlus map first: 107 group rows whose family holds a DDXPlus code say so at the start of `source` ("DDXPlus map first (...)"), because the on-list tier wins there.

Row counts: 1,425 prefixes; 266 tier 1 (145 override, 121 rule), 237 tier 2, 178 tier 3, 744 unscored.

## 4. Coverage of emitted off-list codes

Sources: differentials and flags in `results/v03/ab/runs/*v7a*.json` (arms 1-4) and `results/v03/runs/*.json` (v6: 470-case, atypical and high-risk pools). A mention is one code in one answer. On-list mentions (8,183 A/B differential, 983 A/B flag, 4,569 v6 differential, 3,505 v6 flag) are outside this table; 25 emitted strings are not single ICD-10-CM codes (ranges such as C00-C97, ICD-9 E-codes) and get no tier.

| Source | Off-list mentions | Unique codes | Tier 1 | Tier 2 | Tier 3 | Unscored |
|---|---|---|---|---|---|---|
| A/B differentials | 5,656 | 1,234 | 20.0% | 20.9% | 24.1% | 35.0% |
| A/B flags (arms 4a, 4b) | 391 | 210 | 31.5% | 14.3% | 18.2% | 36.1% |
| v6 differentials | 2,597 | 540 | 34.3% | 21.3% | 18.2% | 26.2% |
| v6 flags | 1,874 | 399 | 60.9% | 14.0% | 2.7% | 22.4% |
| All | 10,518 | 1,461 | 31.2% | 19.5% | 18.6% | 30.6% |

Unscored mentions by reason: under 30 ED visits 2,260 (21.5% of all off-list mentions), symptom codes 852 (8.1%), Z codes 108 (1.0%).

The prefixes the brief named: I71 (545 mentions) and I63 (267) tier 1 by override; K85 pancreatitis (266) tier 1 by rule (admission 62%); U07 COVID-19 (247) tier 2 (admission 18.5%, ICU 4.6%); G47 sleep disorders (91) tier 2; M54 back pain (317), M79 soft tissue (224), M94 cartilage (149), J02 pharyngitis (143) and L50 urticaria (172) tier 3; R07 chest pain (185) and R06 breathing (125) unscored as symptoms.

Top 30 emitted off-list codes:

| Code | Mentions | Tier | Path | Row used | n | Admission | ICU | Description |
|---|---|---|---|---|---|---|---|---|
| I71.0 | 358 | 1 | override | I71 | 31 | 0.676 | 0.168 | Aortic aneurysm and dissection |
| U07.1 | 237 | 2 | nhamcs | U07 | 974 | 0.185 | 0.046 | COVID-19 |
| I63.9 | 219 | 1 | override | I63 | 298 | 0.824 | 0.145 | Cerebral infarction |
| K85.9 | 152 | 1 | nhamcs | K85 | 232 | 0.621 | 0.050 | Acute pancreatitis |
| M31.6 | 152 | unscored | unscored | M31 | 1 | | | Other giant cell arteritis |
| M94.0 | 148 | 3 | nhamcs | M94 | 82 | 0.000 | 0.000 | Chondrocostal junction syndrome (Tietze) |
| J02.0 | 143 | 3 | nhamcs | J02 | 890 | 0.004 | 0.000 | Streptococcal pharyngitis |
| R09.1 | 114 | unscored | unscored | R09 | 276 | 0.339 | 0.079 | Pleurisy |
| M79.1 | 113 | 3 | nhamcs | M79 | 1604 | 0.038 | 0.002 | Myalgia |
| A05.1 | 108 | unscored | unscored | A05 | 20 | | | Botulism food poisoning |
| G00.9 | 101 | 1 | override | G00 | 0 | | | Bacterial meningitis, unspecified |
| A41.9 | 93 | 1 | override | A41 | 408 | 0.879 | 0.224 | Sepsis, unspecified organism |
| I71.01 | 92 | 1 | override | I71 | 31 | 0.676 | 0.168 | Dissection of ascending aorta |
| L50.9 | 87 | 3 | nhamcs | L50 | 153 | 0.000 | 0.000 | Urticaria, unspecified |
| K86.1 | 82 | 1 | nhamcs | K86.1 | 59 | 0.159 | 0.054 | Other chronic pancreatitis |
| G50.0 | 80 | unscored | unscored | G50 | 0 | | | Trigeminal neuralgia |
| K92.2 | 76 | 1 | nhamcs | K92 | 386 | 0.638 | 0.096 | Gastrointestinal hemorrhage, unspecified |
| B54 | 72 | unscored | unscored | B54 | 0 | | | Unspecified malaria |
| R51.9 | 71 | unscored | unscored | R51 | 1420 | 0.031 | 0.004 | Headache, unspecified |
| M54.6 | 69 | 3 | nhamcs | M54 | 2976 | 0.028 | 0.001 | Pain in thoracic spine |
| R07.89 | 69 | unscored | unscored | R07 | 4480 | 0.167 | 0.018 | Other chest pain |
| G35 | 68 | 2 | nhamcs | G35 | 87 | 0.215 | 0.037 | Multiple sclerosis |
| I71.00 | 68 | 1 | override | I71 | 31 | 0.676 | 0.168 | Dissection of unspecified site of aorta |
| L50.0 | 67 | 3 | nhamcs | L50 | 153 | 0.000 | 0.000 | Allergic urticaria |
| K27.9 | 64 | 2 | nhamcs | K27 | 56 | 0.182 | 0.022 | Peptic ulcer, unspecified |
| G03.9 | 59 | 1 | override | G03 | 5 | | | Meningitis, unspecified |
| K27.4 | 58 | 2 | nhamcs | K27 | 56 | 0.182 | 0.022 | Chronic peptic ulcer with hemorrhage |
| K85.2 | 57 | 1 | nhamcs | K85 | 232 | 0.621 | 0.050 | Alcohol induced acute pancreatitis |
| I60.9 | 56 | 1 | override | I60 | 22 | | | Subarachnoid hemorrhage, unspecified |
| M54.1 | 54 | 3 | nhamcs | M54 | 2976 | 0.028 | 0.001 | Radiculopathy |

Descriptions in this table are the subcode's; the CSV row used carries the group's description where the subcode has no row.

## 5. Clinically surprising assignments

The script checks two watch lists of prefixes (`WATCH_DANGEROUS`, `WATCH_ROUTINE`) and writes the mismatches to `coverage.json`. We report them so a reader can judge the rule; we do not hand-edit tiers, because a hand edit would turn a third-party rule back into our judgement.

Dangerous families that come out tier 3 or unscored:

| Prefix | Tier | Why | Off-list mentions |
|---|---|---|---|
| T63 venomous animals and plants | 3 | admission 4.1%, ICU 3.1% over 103 visits: treated and released | 0 |
| T78 adverse effects NEC (includes T78.0 anaphylaxis due to food) | 3 | admission 4.5% over 426 visits. T78.0 itself is on the DDXPlus list (anaphylaxis, tier 1), so the map catches it first; T78.2 anaphylactic shock (ICU 7.1%, 32 visits) and T78.3 angioedema (ICU 9.6%) have their own tier-1 rows | 0 |
| R57 shock | unscored | symptom code, though admission 79%, ICU 45%, death 3.5% over 49 visits | 1 |
| J80 ARDS | unscored | 29 visits, one under the floor | 13 |
| L51.1 Stevens-Johnson | unscored | 0 adult ED visits in the sample | 5 |
| I31.4 cardiac tamponade, G00.9 meningitis, I60.9 subarachnoid haemorrhage | 1 by override (I31.4 unscored, 0 visits) | rare in a 96k-visit sample; only the override list reaches them | 21 / 101 / 56 |

Routine families that come out tier 1 or 2:

| Prefix | Tier | Why | Off-list mentions |
|---|---|---|---|
| G47 sleep disorders | 2 | admission 9.8%, ICU 4.1% over 55 visits (2 critical-care visits) | 91 |
| L03 cellulitis | 2 | admission 22% over 1,331 visits | 75 |
| K29 gastritis | 2 | admission 6.5% over 274 visits: just over the 5% line | 64 |
| M62 muscle disorders (rhabdomyolysis sits here) | 2 | admission 7.5% | 43 |
| F43 stress and adjustment disorders | 2 | admission 6.0% | 0 |
| K86.1 chronic pancreatitis | 1 | ICU 5.4% over 59 visits: 3 critical-care visits | 82 |

The last row is the general case: 57 rule rows are tier 1 by the ICU clause alone (admission under 50%), and 31 of them rest on fewer than five critical-care visits, among them chronic pancreatitis K86.1, Clostridioides difficile A04.7, chronic hepatitis C B18.2 and anaphylactic shock T78.2. They carry 261 off-list mentions (2.5%). Dementia F03 (ICU 6.3% of 81 visits), Alzheimer's G30 (20.9% of 36) and emphysema J43 (8.8% of 88) are tier 1 by the same clause on five to eight visits. A stricter rule (ICU >= 10%) would drop most of them at a cost of 0.03 kappa on the overlap; we keep the calibrated rule and flag the rows here and through `n_icu` in `outcomes_by_code.csv`.

Other notable rows: U07.1 COVID-19 is tier 2 (admission 18.5%) and is the second most emitted off-list code; M31.6 giant cell arteritis, A05.1 botulism, B54 malaria and G50.0 trigeminal neuralgia are unscored because US EDs almost never record them as a visit diagnosis, although models emit each 70-150 times.

## 6. Limits

1. **Construct.** ED disposition says where a patient with a coded diagnosis went, not what happens when the diagnosis is missed. Treat-and-release emergencies (anaphylaxis, asthma, envenomation, PSVT) rate low; chronic conditions with frail patients (dementia, heart failure, anemia) rate high. The overrides carry the missed-diagnosis construct for 15 disease groups; every other tier is disposition.
2. **Coding.** NHAMCS codes the ED's own discharge diagnosis, as abstracted from the record, and often a symptom code (R07 chest pain has 4,480 adult primary visits, more than any disease group). A model that emits the disease code gets the disease's tier; the visits where the ED wrote the symptom instead are not in the disease's denominator, which raises the disease's rates. Rare and specialist diagnoses (GCA, botulism, malaria, neuralgia) are unscored because EDs do not close visits with them.
3. **Weights and variance.** Rates use the visit weight `PATWT` only, without the masked strata and PSUs, so there are no confidence intervals. A 5% ICU threshold on 30-60 visits is one to three sampled visits.
4. **Four-character codes.** The public file truncates codes to four characters, so I71.00 and I71.01 share one row and any 5- or 6-character emitted code inherits its 4-character parent.
5. **Small N.** The 30-visit floor leaves 744 of 1,425 prefixes unscored and puts 21.5% of off-list mentions outside the tiers. Any-listed rates (208 rule rows) mix the code as a secondary diagnosis of an admitted patient with the code as the reason for the visit, and lean high.
6. **Calibration sample.** Twenty-six conditions, 10 of them tier 1 and none rare, set the thresholds. The routine line at 5% is the least stable choice (kappa 0.767 at 10%) and drives the tier-2/tier-3 split for the bulk of emitted codes.
7. **Adults only.** Age 18+ matches the benchmark's adult sample; the tiers do not apply to paediatric codes such as croup or bronchiolitis.
