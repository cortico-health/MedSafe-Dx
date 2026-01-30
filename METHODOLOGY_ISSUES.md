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
