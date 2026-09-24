# Condition-level ICD-10 map for the 49 DDXPlus conditions

Status: draft for review, 2026-09-23. Nothing in the evaluator uses this map yet.

## Summary

1. The current prefix matcher recognises 44.3% of the 22,115 codes the 18 available models emitted (9,787 codes). The condition map recognises 52.1% as equivalent or narrower (11,531), and 55.0% once broader codes are included. The other 45% are off-list conditions: symptom codes (7.9% of all codes), GI haemorrhage (2.7%), urticaria and angioedema (1.8%), aortic dissection (1.8%), biliary disease (1.7%).
2. Mean "true condition in top 5" across the 18 models rises from 73.6% to 83.9%; top 3 rises from 66.3% to 76.3%. For severe true conditions (DDXPlus severity 1-2) top 5 rises from 72.5% to 79.6%. Every model gains; the spread between models narrows slightly (top 5: 56.0-80.8% before, 66.0-92.4% after).
3. Seven conditions account for most of the gain, all because DDXPlus picked a code models do not use: Anemia 26% to 100% (models code the anemia type, D62 and D50), Acute otitis media 25% to 100% (H66.93 by ear), Bronchitis 3% to 94% (J20.9 acute), HIV 44% to 97% (B23.0 and B20 variants), Myocarditis 7% to 59% (I40.9 acute), Anaphylaxis 72% to 100% (T78.2), Spontaneous rib fracture 0% to 41% (S22.3x).
4. Two conditions lose ground on purpose. Panic attack falls from 93.4% to 85.2% because generalised anxiety (F41.1) and anxiety NOS (F41.9) stop counting; atrial fibrillation falls from 90.0% to 89.4% because atrial flutter stops counting. Both are visible as "broader" in the report.
5. Acute pulmonary edema (severity 1) barely moves, 35% to 39%, because the brief classes I50.9 (heart failure, unspecified) as related. Models answer I50.9 on 103 of 540 code slots for these cases. Counting it lifts top 5 to 99%. This is the one mapping that ICD-10 text does not settle; it needs a policy decision (see "Open decisions").

## Method

We map by condition, not by code. For each of the 49 conditions we list every WHO ICD-10 (2019) and ICD-10-CM (FY2026) code that denotes the condition, with a relation:

| Relation | Meaning | Counted as a match |
|---|---|---|
| equivalent | the same condition, including the DDXPlus code and its unspecified-form siblings | yes |
| narrower | a subtype, site, stage or cause-specified form of the condition | yes |
| broader | a parent category or an unspecified code that contains the condition | reported, never counted |
| related | a neighbouring condition, symptom, complication or cause that is not the condition | never counted |

Sources. We verified every code in two places: the CMS FY2026 ICD-10-CM tabular list and alphabetic index (https://www.cms.gov/files/zip/2026-code-tables-tabular-and-index.zip, files `icd10cm_tabular_2026.xml` and `icd10cm_index_2026.xml`) and the WHO ICD-10 2019 browser (https://icd.who.int/browse10/2019/en, JSON concept endpoint). The build script fails if a code is found in neither system. The `system` column of the map records where the code is valid, and the `note` column carries both titles so a reviewer can audit a row without opening either source.

Decision rules, applied in this order:

1. The DDXPlus code is always equivalent, even where it is non-standard (for example `S22.9` for spontaneous rib fracture, `J30` for "allergic sinusitis", `F41` for panic attack). Those codes are what the benchmark currently treats as gold.
2. An inclusion term or index entry settles equivalence. Examples: CM T78.2 carries the inclusion term "Anaphylaxis"; CM I50.1 carries "Pulmonary edema with heart failure"; CM I20.89 carries "Stable angina" and the index sends "Angina, stable" there; CM J93.83 carries "Spontaneous pneumothorax NOS"; CM F41.0 carries "Panic attack"; the CM index sends "Pharyngitis, viral" to J02.8 and "Boerhaave's syndrome" to K22.3.
3. Children of an equivalent code are narrower unless they name a different condition. The map lists those exceptions explicitly, so the matcher can read them (J30.0 vasomotor rhinitis under J30, F41.1 generalised anxiety under F41, I48.3 flutter under I48).
4. When WHO and CM give one code different scopes, we read it in the system where it is a valid leaf code, because a model that emits it can only mean that reading. G44.0 is "Cluster headache syndrome" in WHO (leaf) but a header for all trigeminal autonomic cephalgias in CM, so it is equivalent. Likewise J81 (WHO leaf "Pulmonary oedema"), I26.9, H66.9, J45.9, I20.8 and G24.0.
5. Ownership. A code is counted for at most one condition. If it would be equivalent or narrower for two, the condition whose DDXPlus code is its parent owns it, and it becomes related for the other. The build script refuses a map with two owners. Broader and related rows may repeat across conditions.

Matching rule for the scorer. Normalise the model's code (upper case, drop the dot). For each condition, take the longest map code that is a prefix of the model's code and use its relation. A 7th-character code such as `T78.2XXA` therefore resolves to `T78.2`; `S22.31XA` resolves to `S22.3`. A code with no prefix in a condition's rows does not map to that condition.

## Ambiguous codes and tie rules

| Code(s) | Conditions in play | Rule | Why |
|---|---|---|---|
| I20.0 vs I20.9 vs I20.89 | Unstable angina, Stable angina | I20.0 is Unstable angina. I20.9 and I20.89 are Stable angina (equivalent). I20 and I20.9 are broader for Unstable angina. I25.110 (CAD with unstable angina) is Unstable angina; I25.119 and I25.118 are Stable angina | CM inclusion terms: I20.89 "Stable angina", I20.9 "Angina NOS". DDXPlus itself chose I20.9 for stable angina |
| I24.9 acute coronary syndrome | Unstable angina, NSTEMI/STEMI | broader for both, counted for neither | Index: "Syndrome, coronary, acute NEC I24.9" covers both |
| J01 vs J32 vs J30 | Acute rhinosinusitis, Chronic rhinosinusitis, Allergic sinusitis | Each category owns its own children. J32.9 is chronic only; J01.9 is acute only; J30.x is allergic rhinitis. Each is related to the other two | CM J01 excludes1 "sinusitis NOS (J32.9)", so "sinusitis" without a qualifier is chronic, and models that emit J32.9 on an acute case do not get credit |
| J20 vs J40 vs J44.1 | Bronchitis, COPD exacerbation | J40 owns J20 (acute), J41 and J42 (chronic) as narrower. J44.x belongs to the COPD condition only | CM J40 inclusion "Bronchitis NOS"; DDXPlus Bronchitis is the unqualified code, so the acute form is a subtype |
| J06.9 vs J02.9 vs J04.0 vs J00 vs J03 | URTI, Viral pharyngitis, Acute laryngitis | Each named condition owns its code. URTI owns J00 (common cold), J03 (tonsillitis), J06.0 and J04.1 as narrower because no other DDXPlus condition names them. J06.9 is broader for pharyngitis and laryngitis | The J00-J06 block is "acute upper respiratory infections"; DDXPlus splits it four ways |
| J05.0 vs J04.0 vs J04.2 vs J38.5 | Croup, Acute laryngitis, Laryngospasm | Croup owns J05.0 only. Laryngitis owns J04.0 and J04.2 (laryngotracheitis). Laryngospasm owns J38.5. Each is related to the others | Index: "Croup J05.0", "Croup, spasmodic J38.5" |
| J05.1 vs J04.3 | Epiglottitis | J04.3 supraglottitis is related, not equivalent | CM indexes "Supraglottitis J04.30" and "Epiglottitis J05.10" as separate entries, even though adult clinicians use the words interchangeably |
| I51.4 vs I40 vs I30 | Myocarditis, Pericarditis | I40.x (acute myocarditis) is narrower for Myocarditis. I30.x belongs to Pericarditis. Neither counts for the other | CM I51.4 excludes1 acute myocarditis (I40.-): same condition, different course. Myopericarditis has no code of its own |
| I50.1 vs J81.0 vs I50.9 | Acute pulmonary edema | I50.1 is equivalent; J81 and J81.0 are equivalent; I50 is broader; I50.9 and I50.2-I50.8 are related | WHO and CM put "pulmonary oedema with heart failure" under I50.1 and exclude it from J81. I50.9 is heart failure NOS, the code ICD-10 tells a coder not to use for pulmonary oedema. See "Open decisions" |
| T78.0 vs T78.2 vs T78.3 | Anaphylaxis, Scombroid | T78.2 is equivalent for Anaphylaxis (the brief's example listed it as related; the CM inclusion term "Anaphylaxis" under T78.2 overrides that). T78.3 angioedema is related. Scombroid gets only T61.1 and its broader parents | CM T78.2 inclusion terms: "Anaphylaxis", "Anaphylactic reaction", "Allergic shock". The condition is "Anaphylaxis" with no cause; DDXPlus's food code is a representative, not a restriction |
| I47.1 vs R00.0 vs I49.9 vs I48.91 | PSVT, Atrial fibrillation | PSVT owns I47.1, I47.10, I47.19. Tachycardia NOS (R00.0), paroxysmal tachycardia NOS (I47.9) and arrhythmia NOS (I49.9) are broader for both PSVT and AF. Flutter (I48.3, I48.4, I48.92) is related to AF | CM index sends PAT, AVNRT, AVRT and junctional tachycardia to I47.19. I47.11 inappropriate sinus tachycardia is related, not PSVT |
| F41 vs F41.0 vs F41.9 | Panic attack | F41 (DDXPlus code) and F41.0 are equivalent; F41.9 is broader; F41.1, F41.3, F41.8 are related | CM F41.0 inclusion "Panic attack". DDXPlus chose the category. See "Open decisions" |
| A15 vs A16 vs A17-A19 | Tuberculosis | A15 and WHO A16 are equivalent (WHO splits respiratory TB by confirmation status); A19 miliary is narrower; A17 and A18 (non-respiratory sites) are related | DDXPlus's code and symptom list are respiratory |
| C34 vs C78.0 vs D38.1 vs D14.3 | Pulmonary neoplasm | C34.x and C7A.090 (bronchial carcinoid) count. Secondary (C78.0), in situ (D02.2) and benign (D14.3) are related. Uncertain behaviour (D38.1, D49.1) is broader | DDXPlus's code is the primary malignancy; the condition name says "neoplasm" but the code decides |
| S22.9 vs S22.3 vs M84.4 | Spontaneous rib fracture | S22.3 and S22.4 (rib fractures) are narrower; M84.4 and M84.48 pathological fracture are equivalent; M84.3 stress fracture is related | Index: "Fracture, spontaneous (cause unknown) - see Fracture, pathological". No model emitted M84.4x in the 250-case runs, so this row has no effect today |
| B20 vs B23.0 vs Z21 | HIV (initial infection) | B20-B24 are equivalent (HIV disease, any manifestation axis); Z21 asymptomatic status and R75 are related | CM B20 excludes1 Z21 |

## Coverage before and after

Population: every code in the top-5 differential of every prediction in the 18 leaderboard runs whose prediction files exist in this checkout (22,115 codes, 1,184 distinct). Five leaderboard rows have no prediction file here and are excluded: claude-opus-4.7, claude-sonnet-4.6, llama-4-maverick, o3-pro, grok-4.20.

| Matcher | Codes recognised | Share |
|---|---|---|
| prefix (current `icd10_prefix_match`, either direction against the 49 DDXPlus codes) | 9,787 | 44.3% |
| category (what `MetricsAccumulator.top3_hits` counts: prefix or same 3-character category) | 10,770 | 48.7% |
| map, equivalent + narrower | 11,531 | 52.1% |
| map, plus broader | 12,154 | 55.0% |

The category rule already credits T78.2 for anaphylaxis and I48.92 for atrial fibrillation, which is why it sits between the other two. The map is stricter than the category rule on those sibling codes and looser on cross-category synonyms (I40 vs I51.4, J20 vs J40, D62 vs D64.9).

## Per-model change

"True condition" is the case's DDXPlus pathology (one condition per case), not the three-condition gold list the evaluator uses. Severe means DDXPlus severity 1 or 2 (about 77 of 250 cases).

| Model | top 3 prefix | top 3 map | top 5 prefix | top 5 map | top 5 map+broader | severe top 5 prefix | severe top 5 map |
|---|---|---|---|---|---|---|---|
| anthropic-claude-fable-5 | 72.8 | 84.4 | 80.8 | 92.4 | 92.8 | 81.8 | 89.6 |
| anthropic-claude-haiku-4.5 | 50.8 | 61.2 | 56.0 | 66.0 | 74.0 | 55.8 | 57.1 |
| anthropic-claude-opus-5 | 69.2 | 81.6 | 78.4 | 91.2 | 91.6 | 84.4 | 93.5 |
| deepseek-deepseek-r1 | 62.6 | 70.7 | 69.9 | 78.3 | 82.7 | 55.8 | 67.5 |
| google-gemini-3-pro-preview | 68.1 | 76.8 | 77.3 | 85.4 | 86.5 | 83.1 | 89.2 |
| google-gemini-3.1-pro-preview | 70.5 | 79.3 | 78.1 | 88.7 | 89.1 | 81.8 | 89.6 |
| moonshotai-kimi-k3 | 72.8 | 83.2 | 78.8 | 88.8 | 90.8 | 80.5 | 87.0 |
| openai-gpt-5-chat | 64.0 | 69.6 | 70.4 | 75.2 | 78.8 | 59.7 | 64.9 |
| openai-gpt-5-mini | 66.1 | 77.1 | 72.7 | 84.9 | 85.3 | 70.7 | 77.3 |
| openai-gpt-5.2 | 59.2 | 71.6 | 70.8 | 84.8 | 85.2 | 74.0 | 85.7 |
| openai-gpt-5.4-mini | 68.4 | 74.4 | 72.0 | 78.4 | 82.0 | 55.8 | 62.3 |
| openai-gpt-5.6-luna | 60.0 | 69.2 | 72.8 | 81.2 | 85.2 | 68.8 | 76.6 |
| openai-gpt-5.6-sol | 65.6 | 76.0 | 72.4 | 83.2 | 85.6 | 77.9 | 85.7 |
| openai-gpt-5.6-terra | 71.4 | 80.7 | 79.8 | 88.7 | 89.9 | 81.8 | 87.0 |
| openai-gpt-6-astra | 70.4 | 83.2 | 76.8 | 90.0 | 92.4 | 77.9 | 87.0 |
| openai-gpt-oss-120b | 58.6 | 71.5 | 63.0 | 76.7 | 77.9 | 61.8 | 63.2 |
| x-ai-grok-4.6 | 72.0 | 83.2 | 80.0 | 91.2 | 92.0 | 81.8 | 89.6 |
| z-ai-glm-5.3 | 70.8 | 80.4 | 74.0 | 84.8 | 85.6 | 71.4 | 79.2 |
| mean | 66.3 | 76.3 | 73.6 | 83.9 | 86.0 | 72.5 | 79.6 |

## Per-condition change

Conditions whose top-5 rate changes. Rates pool all model-cases for that true condition (18 models x cases with that pathology). The 31 other conditions do not move.

| Condition | DDXPlus code | Severity | Model-cases | top 5 prefix | top 5 map | top 5 map+broader | top 3 prefix | top 3 map |
|---|---|---|---|---|---|---|---|---|
| Anemia | D64.9 | 4 | 247 | 26.3 | 100.0 | 100.0 | 20.2 | 96.0 |
| Myocarditis | I51.4 | 2 | 106 | 6.6 | 58.5 | 58.5 | 5.7 | 46.2 |
| HIV (initial infection) | B20 | 3 | 101 | 43.6 | 97.0 | 97.0 | 41.6 | 97.0 |
| Acute otitis media | H66.90 | 4 | 51 | 25.5 | 100.0 | 100.0 | 25.5 | 100.0 |
| Anaphylaxis | T78.0 | 1 | 124 | 71.8 | 100.0 | 100.0 | 71.0 | 100.0 |
| Bronchitis | j40 | 4 | 36 | 2.8 | 94.4 | 94.4 | 0.0 | 88.9 |
| Cluster headache | g44.009 | 3 | 156 | 83.3 | 96.8 | 98.7 | 81.4 | 94.2 |
| URTI | j06.9 | 5 | 232 | 84.9 | 89.7 | 90.1 | 75.9 | 79.3 |
| Tuberculosis | a15 | 3 | 72 | 72.2 | 86.1 | 86.1 | 61.1 | 75.0 |
| Influenza | j11.1 | 3 | 86 | 62.8 | 73.3 | 89.5 | 46.5 | 54.6 |
| Spontaneous rib fracture | S22.9 | 3 | 17 | 0.0 | 41.2 | 41.2 | 0.0 | 0.0 |
| Acute COPD exacerbation / infection | j44.1 | 3 | 180 | 94.4 | 96.7 | 100.0 | 94.4 | 96.7 |
| Acute pulmonary edema | J81.0 | 1 | 108 | 35.2 | 38.9 | 38.9 | 9.3 | 13.9 |
| Stable angina | I20.9 | 2 | 36 | 61.1 | 69.4 | 72.2 | 50.0 | 55.6 |
| Viral pharyngitis | J02.9 | 4 | 214 | 76.2 | 77.1 | 83.2 | 62.2 | 62.2 |
| Acute laryngitis | J04.0 | 4 | 53 | 90.6 | 92.5 | 92.5 | 79.2 | 83.0 |
| Atrial fibrillation | I48.91 | 3 | 180 | 90.0 | 89.4 | 100.0 | 89.4 | 88.9 |
| Panic attack | f41 | 5 | 122 | 93.4 | 85.2 | 86.9 | 59.0 | 55.7 |

Where the anemia gain comes from: the first counted code in each anemia top 5 is D62 (acute posthaemorrhagic) 105 times, D50 (iron deficiency) 70, D64.x 47, D63 (anemia in chronic disease) 25. Models diagnose the anemia and its type; DDXPlus's D64.9 says only "anemia".

## Sensitivity of the contested rows

Top 5 for the affected condition, pooled over the 18 models, if one row were flipped.

| Condition | Change tested | top 5 now | top 5 if changed |
|---|---|---|---|
| Acute pulmonary edema | count I50.9 heart failure NOS | 38.9 | 99.1 |
| Acute pulmonary edema | stop counting I50.1 | 38.9 | 35.2 |
| Anaphylaxis | stop counting T78.2 | 100.0 | 71.8 |
| Panic attack | count the whole F41 category (current behaviour) | 85.2 | 93.4 |
| Myocarditis | stop counting I40 acute myocarditis | 58.5 | 7.5 |
| Myocarditis | count I30 pericarditis (myopericarditis) | 58.5 | 88.7 |
| Atrial fibrillation | count atrial flutter | 89.4 | 93.9 |
| URTI | count J02.9, J04.0, J01 as URTI subtypes | 89.7 | 100.0 |
| URTI | stop counting J03 tonsillitis | 89.7 | 89.7 |
| PSVT | count R00.0, I47.9, I49.9 (broader) | 9.1 | 45.5 |
| Stable angina | count I25.10 CAD without angina | 69.4 | 77.8 |

## Top unmapped codes

Codes with no equivalent, narrower or broader mapping to any of the 49 conditions: 9,961 codes (45.0%), 931 distinct. The 25 most frequent, with the reporting category from `spec/ddxplus_offlist_categories.csv`:

| Code | Count | Category |
|---|---|---|
| I50.9 | 337 | Heart failure |
| U07.1 | 242 | COVID-19 |
| R04.2 | 224 | Symptoms and signs (R chapter) |
| R07.9 | 223 | Symptoms and signs (R chapter) |
| I71.00 | 216 | Aortic aneurysm and dissection |
| R06.02 | 193 | Symptoms and signs (R chapter) |
| K92.2 | 159 | GI haemorrhage |
| L50.0 | 147 | Urticaria and angioedema |
| K83.1 | 137 | Biliary disease |
| I63.9 | 126 | Ischaemic stroke and TIA |
| R51.9 | 126 | Migraine and other headache |
| K92.1 | 123 | GI haemorrhage |
| J36 | 115 | Peritonsillar and deep neck abscess |
| K86.1 | 114 | Pancreatitis |
| K85.9 | 113 | Pancreatitis |
| I60.9 | 105 | Intracranial haemorrhage |
| M31.6 | 98 | Systemic vasculitis |
| I71.0 | 98 | Aortic aneurysm and dissection |
| G43.909 | 93 | Migraine and other headache |
| E05.90 | 89 | Thyroid disease |
| R07.89 | 89 | Symptoms and signs (R chapter) |
| A05.1 | 88 | Tetanus and botulism |
| K92.0 | 83 | GI haemorrhage |
| T78.3 | 83 | Urticaria and angioedema |
| I33.0 | 79 | Endocarditis |

Off-list categories by volume (share of all 22,115 codes): symptoms and signs 7.9%, GI haemorrhage 2.7%, urticaria and angioedema 1.8%, aortic aneurysm and dissection 1.8%, biliary disease 1.7%, migraine and other headache 1.7%, heart failure 1.6%, musculoskeletal pain 1.4%, pancreatitis 1.3%, venous thromboembolism 1.2%, COVID-19 1.1%, then a long tail (CNS infection 0.5%, intracranial haemorrhage 0.5%, sepsis under 0.5%). These categories are for reporting only. They have no DDXPlus severity, and we assign none: a model that lists GI haemorrhage for an anemia case is neither credited nor penalised on the diagnosis axis by this map.

## Open decisions

These are policy choices, not coding facts. Each has a default.

1. I50.9 (heart failure, unspecified) for Acute pulmonary edema. Default: keep it related, as the brief asked. ICD-10 sends pulmonary oedema with heart failure to I50.1, not I50.9, so a model that says I50.9 has named the cause, not the condition. Cost of the default: the severity-1 condition stays at 39% top 5 when 99% of top-5 lists contain a heart-failure code. If the benchmark's question is "did the model recognise cardiogenic pulmonary oedema", count I50.9 as narrower-by-cause.
2. F41 category for Panic attack. Default: count F41.0 and F40.01 only; treat F41.9 as broader and F41.1/F41.3/F41.8 as related. This drops the condition from 93.4% to 85.2%. Counting the whole category keeps DDXPlus's own code choice but credits generalised anxiety disorder for an acute attack.
3. T78.2 for Anaphylaxis. Default: equivalent, on the CM inclusion term. The brief's example put it under related; keeping that reading would drop the severity-1 condition from 100% to 72%.
4. Broader codes for PSVT. Default: never counted. R00.0 tachycardia NOS, I47.9 and I49.9 would lift PSVT from 9% to 45%, but they also fit atrial fibrillation, and DDXPlus keeps the two apart.

## Limitations

- The map covers codes models emitted in these runs plus the codes ICD-10 lists for each condition. A future model can emit a valid code the map lacks; the coverage script's `unmapped_codes.csv` is the place to look for it. Adding a row needs the same two-source check the build script runs.
- Longest-prefix matching cannot see a code that is more general than any row. A model that emits a bare chapter letter or a 3-character category the map does not list gets no mapping. Every DDXPlus category is listed, so this affects only off-list codes.
- Five leaderboard models have no prediction file in this checkout, so their numbers are missing from the per-model table. The map itself does not depend on them.
- "True condition" here is the single DDXPlus pathology. The evaluator scores against a three-condition gold list, so the evaluator's top-3 numbers will differ from this report even after it adopts the map.
- The relation labels are ours. The two ICD sources settle whether a code exists and what it is called; they do not say whether "acute myocarditis" is a subtype of "myocarditis NOS" for benchmark purposes. Rule 3 in the method section states how we decided, and the sensitivity table shows what each contested row is worth.
- WHO ICD-10 2019 and ICD-10-CM FY2026 disagree on some codes (J30.4 vs J30.9, A16, B23.0). The `system` column records which applies. A model trained on ICD-10-CA or ICD-10-AM may emit codes valid in neither; those fall to the unmapped list.

## Files

- `spec/ddxplus_icd10_map.csv`: 1,464 rows (98 equivalent, 236 narrower, 88 broader, 1,042 related) over 49 conditions.
- `spec/ddxplus_offlist_categories.csv`: code prefix to reporting category, longest prefix wins.
- `scripts/analysis/icd10_map_coverage.py`: produces `results/analysis/icd10_map/` (coverage_summary.json, per_model_topk.csv, per_condition_topk.csv, code_tally.csv, unmapped_codes.csv, offlist_categories_tally.csv). It reads the evaluator's matcher but changes nothing in it.
