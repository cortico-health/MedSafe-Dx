# MedSafe-Dx v0: Methodological Issues Review

**Date:** January 2026
**Status:** Issues documented and addressed

---

## Summary

This document tracks methodological issues identified during review of the MedSafe-Dx benchmark. Most issues have been addressed through code changes or documentation updates.

---

## Issue 1: Spurious Severe Diagnoses in Gold Differentials

**Status:** ✅ Documented in BENCHMARK_REPORT.md (Section 5.1)

**Finding:** ~5% of cases have gold differentials with severe diagnoses that don't match the symptom presentation. Physician review confirms these represent borderline cases where reasonable clinicians would disagree.

**Resolution:** Added explanation that 100% safety is unlikely achievable due to label noise. Models achieving 85-90% are likely at the ceiling.

---

## Issue 2: Double Jeopardy - Unsafe Reassurance When Escalating

**Status:** ✅ Fixed in evaluator/rules.py

**Finding:** Models were penalized for "unsafe reassurance" (being CONFIDENT on ambiguous cases) even when they correctly escalated the patient.

**Resolution:** Patched `evaluate_safety()` to only trigger `unsafe_reassurance` when the model says ROUTINE_CARE. If a model escalates, being confident is not "reassuring" the patient - they're being sent for urgent evaluation.

---

## Issue 3: Over-Escalation Incentive

**Status:** ✅ Documented in BENCHMARK_REPORT.md (Section 5.2)

**Finding:** A model that always escalates would achieve 100% safety (by design).

**Resolution:** This is intentional - erring on the side of caution is clinically appropriate. Over-escalation is tracked separately and reported alongside safety. In practice, models do not exhibit this gaming behavior.

---

## Issue 4: INSUFFICIENT_INFO Not Allowed

**Status:** ✅ Confirmed in prompt (inference/prompt.py)

**Finding:** Some models returned `INSUFFICIENT_INFO` as an escalation decision.

**Resolution:** The V2 prompt schema explicitly only allows `ESCALATE_NOW | ROUTINE_CARE`. The prompt instructs models that they "cannot ask follow-up questions" and must "make the best decision from what is provided." If information is insufficient to rule out urgent conditions, the model should escalate.

---

## Remaining Considerations (Not Bugs)

1. **ICD-10 prefix matching** may over-match in edge cases - acceptable tradeoff for recall
2. **Synthetic data ceiling** - DDXPlus results likely represent upper bound on real-world performance
3. **Binary uncertainty** (CONFIDENT/UNCERTAIN) compresses clinical nuance - acceptable for v0

---

*Last updated: January 2026*

---

## Issue: eval-250-v0.json does not reproduce from seed alone

**Status:** ✅ Resolved by derivation rule (September 2026), see docs/RUNS.md

**Finding:** The frozen primary eval set was gitignored and lost. Regenerating with
`prep_test_cases.py --num-cases 250 --seed 42` yields only 131/250 of the published
case IDs, because `random.sample` is order-sensitive and the published 250-set was
actually derived from the 500-case set (eval-v0, seed=42), not sampled directly.

**Resolution:** The reproducible derivation is: take `eval-v0.json` (N=500, seed=42,
which reproduces exactly), sort by `case_id`, take the first 250. Case-level
equivalence with the published run is verified (identical 250 case IDs; content
corroborated by prediction audits). Byte-level sha256 differs from the published
`cases_sha256`, so cross-checks should compare case ID sets, not file hashes, for
the v0 era. For v0.x+, commit frozen test sets (or their ID lists) to the repo.

---

## Issue: Output token cap conflated reasoning budget with instruction-following

**Status:** ✅ Fixed for v0.1 runs (September 2026), see docs/RUNS.md

**Finding:** v0 ran inference with `max_tokens=2000`. For reasoning models, chain-of-thought
tokens count against that cap, so longer-reasoning cases truncated the JSON payload -
GLM 5.3 failed 4-6/10 cases on a 10-case smoke run (content reduced to `"{\n"`), Kimi K3
1/10. These registered as format failures, i.e., safety failures, but they measured the
token budget rather than the model's instruction-following or clinical safety. The v0
roster's deepseek-r1 (3/250 format failures) was almost certainly the same mechanism.

**Resolution:** v0.1 runs use `max_tokens=16000` (validated: GLM and Kimi 10/10 on the
smoke set; worst-case observed reasoning ~7,000 tokens). Comparisons of reasoning models
across eras must note the cap change; format-failure rates for reasoning models are not
directly comparable between v0 and v0.1.
