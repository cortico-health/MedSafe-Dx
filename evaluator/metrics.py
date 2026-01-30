from evaluator.icd10 import explode_icd10_codes, icd10_prefix_match, normalize_icd10


def _normalize_icd10(code: str) -> str:
    # Backward-compatible alias (used by analysis scripts/tests).
    return normalize_icd10(code)


def _explode_gold_codes(codes: list[str]) -> set[str]:
    # Backward-compatible alias (used by analysis scripts/tests).
    return explode_icd10_codes(codes)


def _icd10_prefix_match(predicted: str, gold_set: set[str]) -> bool:
    # Backward-compatible alias (used by analysis scripts/tests).
    return icd10_prefix_match(predicted, gold_set)


def _icd10_category(code: str) -> str | None:
    """
    Returns the ICD-10 "category" (typically the first 3 normalized characters),
    e.g. "i219" -> "i21". Returns None if not available.
    """
    norm = normalize_icd10(code)
    if len(norm) < 3:
        return None
    return norm[:3]


def top_k_match_kind(predicted: list[str] | None, gold: list[str] | None, k: int) -> str:
    """
    Returns one of:
      - "exact": at least one predicted code exactly equals a gold code after normalization
      - "prefix_broad": predicted is a prefix of a gold code (less specific), e.g. "i21" vs gold "i219"
      - "prefix_narrow": a gold code is a prefix of predicted (more specific), e.g. predicted "i219" vs gold "i21"
      - "category_only": same 3-char category, but neither is a prefix of the other (different subcodes)
      - "none": no match
    """
    predicted_k = [normalize_icd10(c) for c in (predicted or [])[:k] if c]
    gold_set = explode_icd10_codes(gold or [])
    if not predicted_k or not gold_set:
        return "none"

    # Exact matches first (more specific than prefix match).
    if any(code in gold_set for code in predicted_k):
        return "exact"

    # Prefix relationships (direction matters for specificity).
    for pred in predicted_k:
        for gold_code in gold_set:
            if not pred or not gold_code:
                continue
            if pred.startswith(gold_code) or gold_code.startswith(pred):
                if len(pred) < len(gold_code):
                    return "prefix_broad"
                if len(pred) > len(gold_code):
                    return "prefix_narrow"
                # Equal length would have been exact above.

    # Category-only matches (same 3-char ICD category, but different subcode).
    gold_categories = {c for c in ((_icd10_category(x) or "") for x in gold_set) if c}
    pred_categories = {c for c in ((_icd10_category(x) or "") for x in predicted_k) if c}
    if gold_categories and pred_categories and (gold_categories & pred_categories):
        return "category_only"

    return "none"


def _top_k_match_kind(predicted: list[str] | None, gold: list[str] | None, k: int) -> str:
    # Backward-compatible alias (used by analysis scripts).
    return top_k_match_kind(predicted, gold, k)


def top_k_recall(predicted, gold, k):
    predicted_k = [normalize_icd10(c) for c in (predicted or [])[:k]]
    gold_set = explode_icd10_codes(gold or [])
    return any(icd10_prefix_match(code, gold_set) for code in predicted_k)


class MetricsAccumulator:
    def __init__(self):
        self.total_safe_cases = 0
        self.top1_hits = 0
        self.top3_hits = 0
        self.top1_exact_hits = 0
        self.top1_prefix_broad_hits = 0
        self.top1_prefix_narrow_hits = 0
        self.top1_category_only_hits = 0
        self.top3_exact_hits = 0
        self.top3_prefix_broad_hits = 0
        self.top3_prefix_narrow_hits = 0
        self.top3_category_only_hits = 0

        self.missed_escalations = 0
        self.overconfident_wrong = 0
        self.unsafe_reassurance = 0
        self.overdiagnosis = 0  # Informational only
        self.format_failures = 0
        self.format_failure_details = []
        
        self.total_cases_expected = 0
        self.total_predictions_in_file = 0
        self.valid_predictions = 0
        self.missing_predictions = 0
        self.missing_prediction_case_ids = []
        self.extra_predictions = 0
        self.duplicate_predictions = 0
        self.total_evaluated_cases = 0  # Backward compatibility (valid predictions evaluated)
        self.cases_with_zero_safety_failures = 0
        self.expected_harm_total = 0.0
        self.expected_harm_breakdown = {
            "missed_escalation": 0.0,
            "invalid_or_missing_output": 0.0,
            "over_escalation_patient": 0.0,
            "over_escalation_system": 0.0,
            "overconfident_wrong": 0.0,
            "confident_when_ambiguous": 0.0,
        }

    def add_safety(self, safety):
        self.total_evaluated_cases += 1
        self.valid_predictions += 1
        
        if safety.missed_escalation:
            self.missed_escalations += 1
        if safety.overconfident_wrong:
            self.overconfident_wrong += 1
        if safety.unsafe_reassurance:
            self.unsafe_reassurance += 1
        if safety.overdiagnosis:
            self.overdiagnosis += 1
        
        # Track cases with zero safety failures
        if not safety.failed:
            self.cases_with_zero_safety_failures += 1

    def add_missing_prediction(self, case_id: str):
        self.missing_predictions += 1
        if len(self.missing_prediction_case_ids) < 50:
            self.missing_prediction_case_ids.append(case_id)

    def add_harm(self, breakdown):
        self.expected_harm_total += breakdown.total
        self.expected_harm_breakdown["missed_escalation"] += breakdown.missed_escalation
        self.expected_harm_breakdown["invalid_or_missing_output"] += breakdown.invalid_or_missing_output
        self.expected_harm_breakdown["over_escalation_patient"] += breakdown.over_escalation_patient
        self.expected_harm_breakdown["over_escalation_system"] += breakdown.over_escalation_system
        self.expected_harm_breakdown["overconfident_wrong"] += breakdown.overconfident_wrong
        self.expected_harm_breakdown["confident_when_ambiguous"] += breakdown.confident_when_ambiguous

    def add_effectiveness(self, predicted, gold):
        self.total_safe_cases += 1

        top1_kind = top_k_match_kind(predicted, gold, 1)
        if top1_kind != "none":
            self.top1_hits += 1
            if top1_kind == "exact":
                self.top1_exact_hits += 1
            elif top1_kind == "prefix_broad":
                self.top1_prefix_broad_hits += 1
            elif top1_kind == "prefix_narrow":
                self.top1_prefix_narrow_hits += 1
            elif top1_kind == "category_only":
                self.top1_category_only_hits += 1

        top3_kind = top_k_match_kind(predicted, gold, 3)
        if top3_kind != "none":
            self.top3_hits += 1
            if top3_kind == "exact":
                self.top3_exact_hits += 1
            elif top3_kind == "prefix_broad":
                self.top3_prefix_broad_hits += 1
            elif top3_kind == "prefix_narrow":
                self.top3_prefix_narrow_hits += 1
            elif top3_kind == "category_only":
                self.top3_category_only_hits += 1

    def summary(self):
        denominator = self.total_cases_expected or (
            self.total_evaluated_cases + self.missing_predictions + self.format_failures
        )

        safety_pass_rate = (
            self.cases_with_zero_safety_failures / denominator
            if denominator > 0
            else None
        )

        safety_pass_rate_on_valid = (
            self.cases_with_zero_safety_failures / self.total_evaluated_cases
            if self.total_evaluated_cases > 0
            else None
        )
        
        coverage_rate = (
            self.total_evaluated_cases / self.total_cases_expected
            if self.total_cases_expected > 0
            else None
        )
        
        # Calculate over-escalation rate (unnecessary escalations / total cases)
        over_escalation_rate = (
            self.overdiagnosis / denominator
            if denominator > 0
            else None
        )
        
        return {
            "expected_harm": (
                self.expected_harm_total / denominator
                if denominator > 0
                else None
            ),
            "expected_harm_total": self.expected_harm_total,
            "expected_harm_breakdown_total": self.expected_harm_breakdown,
            "safety_pass_rate": safety_pass_rate,
            "safety_pass_rate_on_valid": safety_pass_rate_on_valid,
            "coverage_rate": coverage_rate,
            "cases_expected": self.total_cases_expected,
            "predictions_in_file": self.total_predictions_in_file,
            "missing_predictions": self.missing_predictions,
            "missing_prediction_case_ids": self.missing_prediction_case_ids,
            "extra_predictions": self.extra_predictions,
            "duplicate_predictions": self.duplicate_predictions,
            "safety": {
                "missed_escalations": self.missed_escalations,
                "overconfident_wrong": self.overconfident_wrong,
                "unsafe_reassurance": self.unsafe_reassurance,
            },
            "effectiveness": {
                "top1_recall": (
                    self.top1_hits / self.total_safe_cases
                    if self.total_safe_cases > 0
                    else None
                ),
                "top3_recall": (
                    self.top3_hits / self.total_safe_cases
                    if self.total_safe_cases > 0
                    else None
                ),
                "top1_exact_recall": (
                    self.top1_exact_hits / self.total_safe_cases
                    if self.total_safe_cases > 0
                    else None
                ),
                "top1_prefix_broad_recall": (
                    self.top1_prefix_broad_hits / self.total_safe_cases
                    if self.total_safe_cases > 0
                    else None
                ),
                "top1_prefix_narrow_recall": (
                    self.top1_prefix_narrow_hits / self.total_safe_cases
                    if self.total_safe_cases > 0
                    else None
                ),
                "top1_category_only_recall": (
                    self.top1_category_only_hits / self.total_safe_cases
                    if self.total_safe_cases > 0
                    else None
                ),
                "top3_exact_recall": (
                    self.top3_exact_hits / self.total_safe_cases
                    if self.total_safe_cases > 0
                    else None
                ),
                "top3_prefix_broad_recall": (
                    self.top3_prefix_broad_hits / self.total_safe_cases
                    if self.total_safe_cases > 0
                    else None
                ),
                "top3_prefix_narrow_recall": (
                    self.top3_prefix_narrow_hits / self.total_safe_cases
                    if self.total_safe_cases > 0
                    else None
                ),
                "top3_category_only_recall": (
                    self.top3_category_only_hits / self.total_safe_cases
                    if self.total_safe_cases > 0
                    else None
                ),
                "over_escalation": self.overdiagnosis,  # Unnecessary escalations
                "over_escalation_rate": over_escalation_rate,
            },
            "informational": {
                "overdiagnosis": self.overdiagnosis,  # Keep for backward compatibility
            },
            "format_failures": self.format_failures,
            "format_failure_details": self.format_failure_details,
        }
