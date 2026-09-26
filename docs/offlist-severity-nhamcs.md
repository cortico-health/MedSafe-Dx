# Off-list severity tiers from NHAMCS

Date: 2026-09-26. Scope: ICD-10-CM codes that models emit but that name none of the 49 DDXPlus conditions under the standard map policy (`evaluator/condition_match.py`). Code: `scripts/analysis/offlist_severity_nhamcs.py`, then `scripts/analysis/offlist_tier_fill.py` (the fill, change note three). Data outputs under `results/analysis/nhamcs_offlist/`; the tier table the scorer reads is `spec/offlist_tiers_nhamcs.csv`. Sections 1-6 describe the table before the fill (`spec/offlist_tiers_nhamcs_pre_fill.csv`); every figure in them comes from the build script run on the NHAMCS files in `data/external/nhamcs/` and the run files present at the time of writing (including arms 4aj and 4bj); the coverage figures move as new runs land.

## Summary

**Change, 2026-09-26 (third): we fill unscored rows from more public data, with the same rule.** 901 of 1,369 prefixes were unscored, most because they had under 30 adult primary-diagnosis visits, and 42% of off-list flags in the A/B runs landed on them. An unscored flag cannot earn a partial credit, so a model that flags giant cell arteritis scores as if it had flagged nothing. `scripts/analysis/offlist_tier_fill.py` now reads the table the build script writes (kept as `spec/offlist_tiers_nhamcs_pre_fill.csv`), keeps every scored row, and tries two sources, in order, on each unscored row that is not an R, Z or V-Y code. The new column `tier_source` names the source that scored the row (`override`, `nhamcs`, `nhamcs_pooled`, `ccsr` or `unscored`). The rule, the 30-visit floor, primary-diagnosis rates, the overrides and the group-tier rule are unchanged.

1. **`nhamcs_pooled`: NHAMCS ED 2011-2022.** CDC's 2011-2015 ED files add 101,765 adult visits to the 96,539 from 2016-2022 (CDC has not released 2023). They code diagnoses in ICD-9-CM, so we map each primary diagnosis with the CMS 2018 General Equivalence Mapping (GEM) to the longest ICD-10-CM prefix every GEM target shares (4 or 3 characters), and drop the visit when the targets fall in different 3-character groups (88,253 of the 2011-2015 visits map). Rates on common groups agree across the two eras, 2011-2015 vs 2016-2022 (pneumonia J18 admission 54% vs 47%, AF I48 53% vs 52%, back pain M54 2.3% vs 2.8%). A 3-character group follows the group-tier rule on the pooled rates.
2. **`ccsr`: the AHRQ CCSR category.** When the prefix still has under 30 pooled visits, we rate its CCSR category (v2026.1, default outpatient category, since an ED visit is an outpatient encounter) on the pooled primary-diagnosis visits whose code falls in that category, with the same rule and a floor of 30 visits per category. A prefix takes the category most of its ICD-10-CM codes fall in. AHRQ puts codes that cannot be a first-listed diagnosis in placeholder categories (XXX000, XXX111); for those we take the code's first clinical category, because a placeholder mixes unrelated diseases. A finer prefix inside a filled group gets its own row when its category gives a different tier (B44 aspergillosis tier 3 under fungal infections, B44.0 and B44.1 pulmonary aspergillosis tier 1 under lower respiratory disease).
3. **HCUP NEDS is not used.** HCUPnet is an interactive query builder behind a use agreement, with no public API, so a script cannot pull its admission rates.

| | Before the fill | After |
|---|---|---|
| Prefixes (rows) | 1,369 | 1,498 (129 new finer rows) |
| Tier 1 / 2 / 3 / unscored | 199 / 127 / 142 / 901 | 301 / 437 / 377 / 383 |
| Rows scored by pooled NHAMCS (tier 1 / 2 / 3) | - | 99 (18 / 36 / 45) |
| Rows scored by CCSR category (tier 1 / 2 / 3) | - | 548 (84 / 274 / 190) |
| Weak-evidence tier-1 rows | 9 | 20 |
| A/B flags tier 1 / 2 / 3 / unscored | 30.5% / 8.0% / 19.1% / 42.4% | 35.7% / 22.1% / 31.5% / 10.6% |
| All off-list mentions tier 1 / 2 / 3 / unscored | 24.3% / 12.0% / 20.4% / 43.3% | 28.4% / 23.9% / 33.0% / 14.7% |
| Unscored mentions under 30 visits / symptom / Z | 4,761 / 1,198 / 146 | 732 / 1,198 / 146 |

Most filled rows land in tier 2 or 3, because a CCSR category's rates are mostly those of its common members. On the audited kept cases (`results/analysis/case_selection/audited_150_verdicts.csv`, bucket not EXCLUDE, flag off-list unscored), the fill moves few full misses to a partial:

| Audited full misses | Cases | Now tier 1 (partial, cost 1) | Now tier 2 or 3 (cost 7) | Still unscored |
|---|---|---|---|---|
| Disputed (kind FP) | 32 | 7: B44.1 x3, G12.9 x2, G21.01, G21.1 | 22: M31.6 x11 (tier 2); B54 x7, A98.3 x2, J38.3, F40.1 (tier 3) | 3: H40.2 x2, E84.0 |
| Reference-agreed (kind TP) | 83 | 8: J84.0 x3, J84.1 x2, G08, G21.01, I70.3 | 44 | 31: G73.0 x5, I36.9 x3, 7 R codes, 4 F45 codes and 12 more |

Example codes:

| Code | Row used | Tier | Source | n | Admission / ICU |
|---|---|---|---|---|---|
| M31.6 giant cell arteritis | M31 | 2 | CCSR MUS024 connective tissue disorders | 46 | 0.179 / 0.023 |
| B54 malaria | B54 | 3 | CCSR INF009 parasitic infections | 167 | 0.002 / 0.000 |
| B44 aspergillosis | B44 (B44.0, B44.1: tier 1) | 3 | CCSR INF004 fungal infections (RSP016 for B44.0-B44.1) | 212 | 0.007 / 0.000 |
| H40.2 angle-closure glaucoma | H40 | unscored | CCSR EYE003 glaucoma, 15 visits | 15 | - |
| A98.3 Marburg | A98 | 3 | CCSR INF008 viral infection | 1,337 | 0.016 / 0.001 |
| G12.9 spinal muscular atrophy | G12 | 1 | CCSR NVS006 other nervous system disorders | 137 | 0.573 / 0.123 |
| G21.0 neuroleptic malignant syndrome | G21 | 1 | CCSR NVS006 | 137 | 0.573 / 0.123 |
| E84.0 cystic fibrosis | E84 | unscored | CCSR END012, 2 visits | 2 | - |
| G73.0 myasthenic syndromes | G73 | unscored | CCSR NVS018, 9 visits | 0 | - |
| M47.0 spondylosis | M47 | 3 | pooled NHAMCS | 33 | 0.036 / 0.000 |
| A35 tetanus | A35 | 2 | CCSR INF003 bacterial infections | 102 | 0.141 / 0.043 |
| I36.9 tricuspid valve disorder | I36 | unscored | CCSR CIR003, 21 visits | 0 | - |
| J84 interstitial lung disease | J84 | 1 | CCSR RSP016 lower respiratory disease | 215 | 0.314 / 0.060 |

The CCSR fallback inherits the construct gap of section 6 and adds one of its own: a rare, dangerous disease takes the rates of the common diseases it is grouped with. Malaria B54 and Marburg A98.3 come out tier 3 because parasitic and viral infection visits are mostly routine; tetanus A35 is tier 2. We report these rather than override them, as with section 5, because a hand edit would turn the third-party rule back into our judgement. Outputs: `results/analysis/nhamcs_offlist/fill_rates.csv` (pooled rates by code, group and category) and `fill_report.json` (the figures above). Tests: `evaluator/tests/test_offlist_tier_fill.py`. The build script now writes `spec/offlist_tiers_nhamcs_pre_fill.csv`; run the fill after it.

**Change, 2026-09-26 (second): the group-tier rule.** A 3-character group the NHAMCS rule scores now takes the tier of the sub-codes that hold the majority of its primary-diagnosis visits, and sub-codes with a different tier keep rows of their own. We made this change because pooled group rates let a small, severe sub-code set the tier of a group whose visits are mostly routine: type 2 diabetes E11 was tier 1 (ICU 8.7% over 483 visits) because E11.1 ketoacidosis (54 visits, ICU 45%) sits inside it, while E11, E11.6 and E11.9 (396 visits) are tier 2 on their own. A sub-code has a tier of its own when it has an override or 30+ primary visits; a group with no majority tier among those keeps its pooled tier. Override and unscored groups are unchanged. The previous table is kept as `spec/offlist_tiers_nhamcs_pooled.csv`; the new column `pooled_tier` gives the pooled tier on every rule-scored group row. The rule changes 16 of 317 rule-scored groups:

| Group | Pooled tier | New tier | Pooled admission / ICU | Majority sub-codes: visits | Sub-code rows (tier) | Off-list mentions changed |
|---|---|---|---|---|---|---|
| A08 Viral and other specified intestinal infections | 2 | 3 | 0.057 / 0.023 | A08.4: 104 of 109 | - | 8 |
| D64 Other anemias | 1 | 2 | 0.500 / 0.048 | D64.9: 278 of 281 | - | 0 |
| E11 Type 2 diabetes mellitus | 1 | 2 | 0.237 / 0.087 | E11, E11.6, E11.9: 396 of 483 | E11.1 (1) | 21 |
| F15 Other stimulant related disorders | 2 | 3 | 0.085 / 0.000 | F15.1: 90 of 141 | - | 53 |
| G47 Sleep disorders | 2 | 3 | 0.098 / 0.041 | G47.0: 40 of 55 | - | 78 |
| J09 Influenza due to certain identified influenza viruses | 3 | 2 | 0.046 / 0.000 | J09.X: 53 of 71 | - | 0 |
| J10 Influenza due to other identified influenza virus | 2 | 3 | 0.062 / 0.015 | J10.1: 108 of 121 | - | 0 |
| K29 Gastritis and duodenitis | 2 | 3 | 0.065 / 0.001 | K29.0, K29.7: 201 of 274 | - | 41 |
| L02 Cutaneous abscess, furuncle and carbuncle | 2 | 3 | 0.069 / 0.002 | L02, L02.0, L02.3, L02.9: 404 of 758 | L02.2, L02.4, L02.5 (2) | 0 |
| O21 Excessive vomiting in pregnancy | 2 | 3 | 0.058 / 0.000 | O21.9: 111 of 210 | O21.0 (2) | 0 |
| O36 Maternal care for other fetal problems | 2 | 3 | 0.181 / 0.006 | O36.8: 39 of 60 | - | 0 |
| S22 Fracture of rib(s), sternum and thoracic spine | 2 | 1 (weak evidence) | 0.239 / 0.041 | S22.0, S22.4: 140 of 271 | S22.3 (2) | 0 |
| S43 Dislocation and sprain of shoulder girdle | 3 | 2 | 0.047 / 0.000 | S43.0: 114 of 214 | S43.1, S43.4 (3) | 0 |
| S52 Fracture of forearm | 3 | 2 | 0.049 / 0.000 | S52.5: 153 of 258 | S52.1 (3) | 0 |
| S70 Superficial injury of hip and thigh | 2 | 3 | 0.053 / 0.000 | S70.0: 98 of 143 | - | 0 |
| S81 Open wound of knee and lower leg | 3 | 2 | 0.048 / 0.000 | S81.8: 213 of 253 | S81.0 (3) | 0 |

Every sub-code row with its own tier keeps that tier; only sub-codes without a row of their own (under 30 visits) follow the new group tier. D64, J09, J10 and S22 hold DDXPlus codes, which the map resolves first. S22 is tier 1 because S22.0 and S22.4, both weak-evidence rows, hold 140 of 271 visits, so the group row carries `weak_evidence` in their place (9 weak rows, 65 mentions, as before). Off-list mentions that change tier: 180 from tier 2 to 3 (G47, F15, K29, A08) and 21 from tier 1 to 2 (E11). Rows: 1,369 prefixes; 199 tier 1 (151 override, 48 rule), 127 tier 2, 142 tier 3, 901 unscored. Tests: `evaluator/tests/test_offlist_group_tier.py`.

**Change, 2026-09-26: the rule reads primary-diagnosis rates only.** The first version (commit 9323b23, kept as `spec/offlist_tiers_nhamcs_anylisted.csv`) fell back to any-listed rates when a prefix had under 30 primary visits (208 of 536 rule rows). Any-listed rates count the code as a secondary diagnosis of an admitted patient, so they rate chronic comorbidities as dangerous: unspecified viral hepatitis B19 and interstitial lung disease J84 were tier 1 on 124 and 54 any-listed visits. Now a prefix under 30 primary visits is unscored, and a new column `weak_evidence` marks tier-1 rows that rest on the ICU clause with fewer than 5 critical-care visits (10 rows; new column `n_icu` gives the count). The outcome, on the same emitted codes:

| | Any-listed fallback (old) | Primary only (new) |
|---|---|---|
| Prefixes | 1,425 | 1,380 |
| Tier 1 (override / rule) | 266 (145 / 121) | 201 (151 / 50) |
| Tier 2 / tier 3 / unscored | 237 / 178 / 744 | 133 / 145 / 901 |
| Off-list mentions tier 1 / 2 / 3 / unscored | 28.8% / 20.3% / 19.7% / 31.2% | 24.5% / 13.1% / 19.1% / 43.3% |
| A/B flags tier 1 / unscored | 33.8% / 32.5% | 30.7% / 42.4% |

Mentions that change tier: tier 2 to unscored 1,117; tier 1 to unscored 441; tier 1 to 2 176; tier 3 to unscored 145; tier 2 to 3 67; tier 2 to 1 6. B19 and J84 are now unscored (3 and 9 primary visits). **Type 2 diabetes E11 stays tier 1** (the group-tier rule above later moved it to tier 2): as a primary diagnosis it has 483 visits, 23.7% admission and 8.7% ICU (37 critical-care visits), because ED visits coded E11 as the reason include ketoacidosis and hyperosmolar states; E11.6 and E11.9 are tier 2. The rule is unchanged, so we report E11 rather than override it. Of the 6 new override rows, C17 and C20 are groups the 4aj and 4bj runs emitted for the first time; K25.1, K25.2, K25.5 and K25.6 (gastric ulcer with perforation, an AHRQ group) need rows of their own now that their group K25 is unscored.

1. **One rule on ED disposition reproduces our v0.3 tiers well on the conditions NHAMCS covers: tier 1 if critical-care admission >= 5% or admission >= 50%; tier 3 if admission < 5%; else tier 2.** On the 26 DDXPlus conditions with 30+ adult primary-diagnosis visits it agrees with `spec/dangerous_if_missed_tiers_v03b.csv` at quadratic weighted kappa 0.83, 20 of 26 exact, all 26 within one tier. Leave-one-out kappa is 0.72 and 24 of 26 folds pick the same thresholds (section 2).
2. **The rule's six misses are the known construct gap: ED disposition rates treat-and-release danger low and chronic admission high.** Asthma exacerbation (admission 5.7%) and PSVT (20%) fall to tier 2 against our tier 1; anemia (40%) and localized edema (8%) rise to tier 2 against our tier 3; AF (ICU 10%) rises to tier 1; GERD (2.2%) falls to tier 3 (section 2).
3. **`spec/offlist_tiers_nhamcs.csv` has 1,369 prefixes: 199 tier 1 (151 by Newman-Toker or AHRQ override, 48 by the rule), 127 tier 2, 142 tier 3, 901 unscored.** Off-list mentions in the A/B and v6 runs land 24% tier 1, 12% tier 2, 20% tier 3, 43% unscored; the unscored share is mostly codes with under 30 primary ED visits (34% of mentions) and symptom codes (8%). Arm-4 flags in the v6 runs are 54% tier 1 (section 4).
4. **Clinically surprising assignments exist and are reported, not corrected.** Venomous bites T63 and adverse effects T78 are tier 3 (treated and released); shock R57 is unscored as a symptom code despite 45% critical care; ARDS J80 is unscored at 29 visits; cellulitis L03 and adjustment disorder F43 are tier 2; 9 tier-1 rows rest on fewer than five critical-care visits and carry `weak_evidence` (65 off-list mentions) (section 5).
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
3. **NHAMCS rule:** the section 2 rule on the prefix's primary-diagnosis rates. A 3-character group then takes the tier of the sub-codes that hold the majority of its primary visits (the group-tier rule; a sub-code counts when it has an override or 30+ primary visits, and with no majority tier the pooled tier stands). A prefix under 30 primary visits is unscored, whatever its any-listed count, because a secondary diagnosis of an admitted patient says little about the code as the reason for the visit. A tier-1 row that the ICU clause alone puts there (admission under 50%) on fewer than 5 critical-care visits carries `weak_evidence = true`; the scorer can read those rows as no reason (`TierFileRule(exclude_weak=True)`).

AHRQ 2022 prefixes are our reading of the report's condition names; the ten other AHRQ conditions are Table 1 rows already in the spec CSV:

| AHRQ ED top-15 condition | Prefixes |
|---|---|
| Spinal cord compression | G95.2 |
| Traumatic brain injury and intracranial haemorrhage | S06 (non-traumatic I60-I62 sit in the stroke group) |
| Cardiac arrhythmia | I44, I45, I46, I47, I49 (I48 is on the DDXPlus list) |
| GI perforation and rupture | K63.1; K25-K28 with perforation (.1 .2 .5 .6); K35.2, K35.3; K57.0, K57.2, K57.4, K57.8; K65 |
| Intestinal obstruction | K56, K31.5, and hernia codes with obstruction (K41.0, K41.3, K42.0, K43.0, K43.3, K43.6, K44.0, K45.0, K46.0) |

`spec/offlist_tiers_nhamcs.csv` columns: `icd10_prefix`, `description` (CMS ICD-10-CM FY2026 order file; "not an ICD-10-CM FY2026 code" for WHO-only codes models emit, such as I64), `tier` (1, 2, 3 or `unscored`), `rule_path` (`override`, `nhamcs`, `unscored`), `pooled_tier` (on rule-scored group rows, the tier the group's pooled rates give), `admission`, `icu`, `death`, `n` (the NHAMCS rates and unweighted primary-diagnosis count behind the row, blank rates when suppressed), `n_icu` (critical-care visits among them), `weak_evidence` (`true` or `false`), `source`. Rows are 3-character groups plus finer prefixes only where the finer prefix has its own override or 30+ visits and its tier differs from the group's (E11 tier 2 but E11.1 ketoacidosis tier 1; T78 tier 3 but T78.2 anaphylactic shock and T78.3 angioedema tier 1; R65 unscored but R65.2 severe sepsis tier 1). A consumer takes the longest matching prefix; a subcode with no row of its own inherits its group. The scorer resolves the DDXPlus map first: 107 group rows whose family holds a DDXPlus code say so at the start of `source` ("DDXPlus map first (...)"), because the on-list tier wins there.

Row counts: 1,369 prefixes; 199 tier 1 (151 override, 48 rule; 9 weak evidence), 127 tier 2, 142 tier 3, 901 unscored.

## 4. Coverage of emitted off-list codes

Sources: differentials and flags in `results/v03/ab/runs/*v7a*.json` (arms 1-4) and `results/v03/runs/*.json` (v6: 470-case, atypical and high-risk pools). A mention is one code in one answer. On-list mentions (12,695 A/B differential, 2,004 A/B flag, 4,569 v6 differential, 3,505 v6 flag) are outside this table; 25 emitted strings are not single ICD-10-CM codes (ranges such as C00-C97, ICD-9 E-codes) and get no tier.

| Source | Off-list mentions | Unique codes | Tier 1 | Tier 2 | Tier 3 | Unscored |
|---|---|---|---|---|---|---|
| A/B differentials | 8,798 | 1,498 | 15.9% | 12.5% | 25.1% | 46.5% |
| A/B flags (arms 4a, 4b, 4aj, 4bj) | 837 | 318 | 30.5% | 8.0% | 19.1% | 42.4% |
| v6 differentials | 2,597 | 540 | 29.7% | 13.3% | 18.3% | 38.7% |
| v6 flags | 1,874 | 399 | 53.7% | 9.6% | 1.9% | 34.8% |
| All | 14,106 | 1,692 | 24.3% | 12.0% | 20.4% | 43.3% |

Unscored mentions by reason: under 30 primary ED visits 4,761 (33.8% of all off-list mentions), symptom codes 1,198 (8.5%), Z codes 146 (1.0%).

The prefixes the brief named: I71 and I63 tier 1 by override; K85 pancreatitis tier 1 by rule (admission 62%); U07 COVID-19 tier 2 (admission 18.5%, ICU 4.6%); G47 sleep disorders, M54 back pain, M79 soft tissue, M94 cartilage, J02 pharyngitis and L50 urticaria tier 3; R07 chest pain and R06 breathing unscored as symptoms. Mention counts per code are in the top-30 table and `coverage.json`.

Top 30 emitted off-list codes:

| Code | Mentions | Tier | Path | Row used | n | Admission | ICU | Description |
|---|---|---|---|---|---|---|---|---|
| I71.0 | 420 | 1 | override | I71 | 31 | 0.676 | 0.168 | Aortic aneurysm and dissection |
| U07.1 | 318 | 2 | nhamcs | U07 | 974 | 0.185 | 0.046 | Emergency use of U07 (U07.1 COVID-19; U07.0 vaping-related disorder) |
| I63.9 | 274 | 1 | override | I63 | 298 | 0.824 | 0.145 | Cerebral infarction |
| J02.0 | 225 | 3 | nhamcs | J02 | 890 | 0.004 | 0.000 | Acute pharyngitis |
| M31.6 | 218 | unscored | unscored | M31 | 1 |  |  | Other necrotizing vasculopathies |
| M94.0 | 214 | 3 | nhamcs | M94 | 82 | 0.000 | 0.000 | Other disorders of cartilage |
| R09.1 | 191 | unscored | unscored | R09 | 276 | 0.339 | 0.079 | Other symptoms and signs involving the circulatory and respiratory system |
| K85.9 | 189 | 1 | nhamcs | K85 | 232 | 0.621 | 0.050 | Acute pancreatitis |
| M79.1 | 137 | 3 | nhamcs | M79 | 1604 | 0.038 | 0.002 | Other and unspecified soft tissue disorders, not elsewhere classified |
| A05.1 | 129 | unscored | unscored | A05 | 11 |  |  | Other bacterial foodborne intoxications, not elsewhere classified |
| G50.0 | 121 | unscored | unscored | G50 | 0 |  |  | Disorders of trigeminal nerve |
| G00.9 | 113 | 1 | override | G00 | 0 |  |  | Bacterial meningitis, not elsewhere classified |
| B54 | 110 | unscored | unscored | B54 | 0 |  |  | Unspecified malaria |
| A41.9 | 104 | 1 | override | A41 | 408 | 0.879 | 0.224 | Other sepsis |
| G35 | 102 | unscored | unscored | G35 | 23 |  |  | Multiple sclerosis |
| I71.01 | 100 | 1 | override | I71 | 31 | 0.676 | 0.168 | Aortic aneurysm and dissection |
| L50.9 | 99 | 3 | nhamcs | L50 | 153 | 0.000 | 0.000 | Urticaria |
| K86.1 | 99 | 2 | nhamcs | K86 | 47 | 0.201 | 0.017 | Other diseases of pancreas |
| R51.9 | 97 | unscored | unscored | R51 | 1420 | 0.031 | 0.004 | Headache |
| K92.2 | 97 | 1 | nhamcs | K92 | 386 | 0.638 | 0.096 | Other diseases of digestive system |
| M54.6 | 95 | 3 | nhamcs | M54 | 2976 | 0.028 | 0.001 | Dorsalgia |
| I71.00 | 94 | 1 | override | I71 | 31 | 0.676 | 0.168 | Aortic aneurysm and dissection |
| G03.9 | 91 | 1 | override | G03 | 3 |  |  | Meningitis due to other and unspecified causes |
| L50.0 | 85 | 3 | nhamcs | L50 | 153 | 0.000 | 0.000 | Urticaria |
| M54.5 | 83 | 3 | nhamcs | M54 | 2976 | 0.028 | 0.001 | Dorsalgia |
| K27.9 | 83 | unscored | unscored | K27 | 17 |  |  | Peptic ulcer, site unspecified |
| R07.9 | 77 | unscored | unscored | R07 | 4480 | 0.167 | 0.018 | Pain in throat and chest |
| K27.4 | 76 | unscored | unscored | K27 | 17 |  |  | Peptic ulcer, site unspecified |
| R07.89 | 75 | unscored | unscored | R07 | 4480 | 0.167 | 0.018 | Pain in throat and chest |
| M54.1 | 75 | 3 | nhamcs | M54 | 2976 | 0.028 | 0.001 | Dorsalgia |

Descriptions in this table are the row used; a subcode with no row of its own carries its group's description.

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
| L03 cellulitis | 2 | admission 22% over 1,331 visits | 75 |
| M62 muscle disorders (rhabdomyolysis sits here) | 2 | admission 7.5% | 43 |
| F43 stress and adjustment disorders | 2 | admission 6.0% | 0 |
| K86.1 chronic pancreatitis | 1 | ICU 5.4% over 59 visits: 3 critical-care visits | 82 |

The last row was the general case under the any-listed fallback (31 weak tier-1 rows, 2.5% of mentions); with primary rates only, K86.1 falls to tier 2 with its group. Now 15 rule rows are tier 1 by the ICU clause alone (admission under 50%), and 9 of them rest on fewer than five critical-care visits: D72, F03, J44.0, K22, N28, S22, T78.2, T78.3, T88. They carry `weak_evidence = true` and 65 off-list mentions. A stricter rule (ICU >= 10%) would drop most of them at a cost of 0.03 kappa on the overlap; we keep the calibrated rule, mark the rows, and score a sensitivity row without them.

Other notable rows: U07.1 COVID-19 is tier 2 (admission 18.5%) and is the second most emitted off-list code; M31.6 giant cell arteritis, A05.1 botulism, B54 malaria and G50.0 trigeminal neuralgia are unscored because US EDs almost never record them as a visit diagnosis, although models emit each 70-150 times.

## 6. Limits

1. **Construct.** ED disposition says where a patient with a coded diagnosis went, not what happens when the diagnosis is missed. Treat-and-release emergencies (anaphylaxis, asthma, envenomation, PSVT) rate low; chronic conditions with frail patients (dementia, heart failure, anemia) rate high. The overrides carry the missed-diagnosis construct for 15 disease groups; every other tier is disposition.
2. **Coding.** NHAMCS codes the ED's own discharge diagnosis, as abstracted from the record, and often a symptom code (R07 chest pain has 4,480 adult primary visits, more than any disease group). A model that emits the disease code gets the disease's tier; the visits where the ED wrote the symptom instead are not in the disease's denominator, which raises the disease's rates. Rare and specialist diagnoses (GCA, botulism, malaria, neuralgia) are unscored because EDs do not close visits with them.
3. **Weights and variance.** Rates use the visit weight `PATWT` only, without the masked strata and PSUs, so there are no confidence intervals. A 5% ICU threshold on 30-60 visits is one to three sampled visits.
4. **Four-character codes.** The public file truncates codes to four characters, so I71.00 and I71.01 share one row and any 5- or 6-character emitted code inherits its 4-character parent.
5. **Small N.** The 30-visit floor on primary diagnoses leaves 901 of 1,380 prefixes unscored and puts 33.8% of off-list mentions outside the tiers. We no longer fall back to any-listed rates, which mixed the code as a secondary diagnosis of an admitted patient with the code as the reason for the visit and leaned high; the price is more unscored rare codes. The fill (change note three) cuts this to 383 of 1,498 prefixes and 5.2% of mentions, by pooling NHAMCS 2011-2022 and falling back to the CCSR category.
6. **Calibration sample.** Twenty-six conditions, 10 of them tier 1 and none rare, set the thresholds. The routine line at 5% is the least stable choice (kappa 0.767 at 10%) and drives the tier-2/tier-3 split for the bulk of emitted codes.
7. **Adults only.** Age 18+ matches the benchmark's adult sample; the tiers do not apply to paediatric codes such as croup or bronchiolitis.
