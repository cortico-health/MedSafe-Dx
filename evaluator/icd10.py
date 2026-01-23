from __future__ import annotations


def normalize_icd10(code: str) -> str:
    return (code or "").strip().lower().replace(".", "").replace(" ", "")


def explode_icd10_codes(codes: list[str]) -> set[str]:
    exploded: set[str] = set()
    for code in codes or []:
        if not code:
            continue
        for part in str(code).split(","):
            normalized = normalize_icd10(part)
            if normalized:
                exploded.add(normalized)
    return exploded


def icd10_prefix_match(predicted: str, gold_set: set[str]) -> bool:
    """
    Check if predicted code matches any gold code using prefix matching.
    Matches if either code is a prefix of the other (after normalization).
    E.g., "I21" matches "I21.9" and vice versa.
    """
    if not predicted:
        return False
    for gold in gold_set:
        if predicted.startswith(gold) or gold.startswith(predicted):
            return True
    return False

