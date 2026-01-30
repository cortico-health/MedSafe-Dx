#!/usr/bin/env python3
"""
Generate publication-oriented tables for MedSafe-Dx.

Design goals:
- Use the same evaluator logic as the leaderboard (pydantic schema + evaluator.rules)
- Provide stratified rates and uncertainty summaries
- Keep output deterministic and reproducible (fixed seed for bootstrap)

Outputs:
- results/analysis/publish_tables.md
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
LEADERBOARD_DIR = PROJECT_ROOT / "leaderboard"

DEFAULT_CASES_PATH = DATA_DIR / "test_sets" / "test-v0.json"
DEFAULT_CONDITIONS_PATH = DATA_DIR / "ddxplus_v0" / "release_conditions.json"
DEFAULT_OUT_MD = RESULTS_DIR / "analysis" / "publish_tables.md"


def _load_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def _load_cases(cases_path: Path) -> list[dict]:
    data = _load_json(cases_path)
    if isinstance(data, dict) and "cases" in data:
        return data["cases"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unexpected cases format in {cases_path}")


def _build_icd10_to_severity(conditions_path: Path) -> dict[str, int]:
    raw = _load_json(conditions_path)
    mapping: dict[str, int] = {}
    if not isinstance(raw, dict):
        return mapping
    for _cond_name, details in raw.items():
        if not isinstance(details, dict):
            continue
        icd10 = details.get("icd10-id")
        severity = details.get("severity")
        if not icd10 or severity is None:
            continue
        try:
            sev_int = int(severity)
        except Exception:
            continue
        key = str(icd10).lower().replace(".", "").replace(" ", "")
        mapping[key] = sev_int
    return mapping


def _severity_category(severity: Optional[int]) -> str:
    if severity is None:
        return "unknown"
    if severity <= 2:
        return "critical"
    if severity == 3:
        return "moderate"
    return "mild"


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    phat = k / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2.0 * n)) / denom
    half = (z * math.sqrt((phat * (1.0 - phat) + (z * z) / (4.0 * n)) / n)) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return (lo, hi)


def _bootstrap_ci(
    values: list[float],
    rng: random.Random,
    iters: int = 2000,
    alpha: float = 0.05,
) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    n = len(values)
    stats: list[float] = []
    for _ in range(iters):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        stats.append(sum(sample) / n)
    stats.sort()
    lo_idx = int((alpha / 2.0) * iters)
    hi_idx = int((1.0 - alpha / 2.0) * iters) - 1
    lo_idx = max(0, min(iters - 1, lo_idx))
    hi_idx = max(0, min(iters - 1, hi_idx))
    return (stats[lo_idx], stats[hi_idx])


def _format_pct(x: Optional[float], digits: int = 1) -> str:
    if x is None:
        return "—"
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "—"
    return f"{100.0 * x:.{digits}f}%"


def _format_ci(lo: float, hi: float, digits: int = 1) -> str:
    return f"{_format_pct(lo, digits=digits)}–{_format_pct(hi, digits=digits)}"


@dataclass(frozen=True)
class CaseOutcome:
    case_id: str
    escalation_required: bool
    uncertainty_acceptable: bool
    severity_category: str
    valid: bool
    safety_pass: bool
    missed_escalation: bool
    overconfident_wrong: bool
    unsafe_reassurance: bool
    over_escalation: bool
    top1_match: bool
    top3_match: bool
    top1_match_kind: str
    top3_match_kind: str
    information_sufficiency: Optional[str]
    followup_kind: Optional[str]
    followup_recommendation: Optional[str]
    decode_total_codes: Optional[int]
    decode_unknown_evidence_count: Optional[int]
    decode_unknown_value_count: Optional[int]


def _iter_eval_files() -> list[Path]:
    return sorted(LEADERBOARD_DIR.glob("*-eval.json"))


def _load_predictions(predictions_path: Path) -> list[dict]:
    data = _load_json(predictions_path)
    if isinstance(data, dict) and "predictions" in data:
        preds = data["predictions"]
    else:
        preds = data
    if not isinstance(preds, list):
        raise ValueError(f"Unexpected predictions format in {predictions_path}")
    return [p for p in preds if isinstance(p, dict)]


def _generate_outcomes_for_model(
    eval_json: dict,
    gold_by_id: dict[str, Any],
    icd10_to_severity: dict[str, int],
) -> list[CaseOutcome]:
    import sys

    sys.path.insert(0, str(PROJECT_ROOT))
    from evaluator.schemas import ModelPrediction
    from evaluator.rules import evaluate_safety
    from evaluator.metrics import top_k_recall, top_k_match_kind

    predictions_path = Path(eval_json["predictions_path"])
    if not predictions_path.is_absolute():
        predictions_path = PROJECT_ROOT / predictions_path

    raw_preds = _load_predictions(predictions_path)
    by_case_id: dict[str, dict] = {}
    for p in raw_preds:
        cid = p.get("case_id")
        if not cid or cid in by_case_id:
            continue
        by_case_id[str(cid)] = p

    outcomes: list[CaseOutcome] = []
    for cid, gold in gold_by_id.items():
        gold_top3 = getattr(gold, "gold_top3", []) or []
        primary = gold_top3[0] if gold_top3 else None
        primary_norm = str(primary).lower().replace(".", "").replace(" ", "") if primary else ""
        severity = icd10_to_severity.get(primary_norm)
        sev_cat = _severity_category(severity)

        raw = by_case_id.get(cid)
        if not raw:
            outcomes.append(
                CaseOutcome(
                    case_id=cid,
                    escalation_required=bool(getattr(gold, "escalation_required", False)),
                    uncertainty_acceptable=bool(getattr(gold, "uncertainty_acceptable", False)),
                    severity_category=sev_cat,
                    valid=False,
                    safety_pass=False,
                    missed_escalation=False,
                    overconfident_wrong=False,
                    unsafe_reassurance=False,
                    over_escalation=False,
                    top1_match=False,
                    top3_match=False,
                    top1_match_kind="none",
                    top3_match_kind="none",
                    information_sufficiency=None,
                    followup_kind=None,
                    followup_recommendation=None,
                    decode_total_codes=None,
                    decode_unknown_evidence_count=None,
                    decode_unknown_value_count=None,
                )
            )
            continue

        try:
            pred = ModelPrediction(**raw)
        except Exception:
            outcomes.append(
                CaseOutcome(
                    case_id=cid,
                    escalation_required=bool(getattr(gold, "escalation_required", False)),
                    uncertainty_acceptable=bool(getattr(gold, "uncertainty_acceptable", False)),
                    severity_category=sev_cat,
                    valid=False,
                    safety_pass=False,
                    missed_escalation=False,
                    overconfident_wrong=False,
                    unsafe_reassurance=False,
                    over_escalation=False,
                    top1_match=False,
                    top3_match=False,
                    top1_match_kind="none",
                    top3_match_kind="none",
                    information_sufficiency=None,
                    followup_kind=None,
                    followup_recommendation=None,
                    decode_total_codes=None,
                    decode_unknown_evidence_count=None,
                    decode_unknown_value_count=None,
                )
            )
            continue

        safety = evaluate_safety(pred, gold)
        predicted_codes = [d.code for d in (pred.differential_diagnoses or [])]
        gold_top3_for_case = getattr(gold, "gold_top3", []) or []
        top1_match = bool(top_k_recall(predicted_codes, gold_top3_for_case, 1))
        top3_match = bool(top_k_recall(predicted_codes, gold_top3_for_case, 3))
        top1_kind = top_k_match_kind(predicted_codes, gold_top3_for_case, 1)
        top3_kind = top_k_match_kind(predicted_codes, gold_top3_for_case, 3)

        decode_audit = raw.get("input_decode_audit") if isinstance(raw, dict) else None
        symptoms_audit = decode_audit.get("symptoms") if isinstance(decode_audit, dict) else None
        decode_total = symptoms_audit.get("total_codes") if isinstance(symptoms_audit, dict) else None
        decode_unk_e = symptoms_audit.get("unknown_evidence_count") if isinstance(symptoms_audit, dict) else None
        decode_unk_v = symptoms_audit.get("unknown_value_count") if isinstance(symptoms_audit, dict) else None

        outcomes.append(
            CaseOutcome(
                case_id=cid,
                escalation_required=bool(getattr(gold, "escalation_required", False)),
                uncertainty_acceptable=bool(getattr(gold, "uncertainty_acceptable", False)),
                severity_category=sev_cat,
                valid=True,
                safety_pass=not safety.failed,
                missed_escalation=bool(safety.missed_escalation),
                overconfident_wrong=bool(safety.overconfident_wrong),
                unsafe_reassurance=bool(safety.unsafe_reassurance),
                over_escalation=bool(safety.overdiagnosis),
                top1_match=top1_match,
                top3_match=top3_match,
                top1_match_kind=top1_kind,
                top3_match_kind=top3_kind,
                information_sufficiency=getattr(pred, "information_sufficiency", None),
                followup_kind=getattr(pred, "followup_kind", None),
                followup_recommendation=getattr(pred, "followup_recommendation", None),
                decode_total_codes=int(decode_total) if isinstance(decode_total, int) else None,
                decode_unknown_evidence_count=int(decode_unk_e) if isinstance(decode_unk_e, int) else None,
                decode_unknown_value_count=int(decode_unk_v) if isinstance(decode_unk_v, int) else None,
            )
        )

    return outcomes


def _group_by(items: Iterable[CaseOutcome], key_fn) -> dict[str, list[CaseOutcome]]:
    out: dict[str, list[CaseOutcome]] = {}
    for it in items:
        k = key_fn(it)
        out.setdefault(k, []).append(it)
    return out


def _rate(k: int, n: int) -> Optional[float]:
    return (k / n) if n > 0 else None


def _summarize_group(cases: list[CaseOutcome]) -> dict[str, Any]:
    n = len(cases)
    valid = sum(1 for c in cases if c.valid)
    safety_pass = sum(1 for c in cases if c.safety_pass)
    missed = sum(1 for c in cases if c.missed_escalation)
    overconf = sum(1 for c in cases if c.overconfident_wrong)
    unsafe = sum(1 for c in cases if c.unsafe_reassurance)
    overesc = sum(1 for c in cases if c.over_escalation)
    unusable = sum(1 for c in cases if not c.valid)
    top1 = sum(1 for c in cases if c.valid and c.top1_match)
    top3 = sum(1 for c in cases if c.valid and c.top3_match)
    top1_exact = sum(1 for c in cases if c.valid and c.top1_match_kind == "exact")
    top1_prefix_broad = sum(1 for c in cases if c.valid and c.top1_match_kind == "prefix_broad")
    top1_prefix_narrow = sum(1 for c in cases if c.valid and c.top1_match_kind == "prefix_narrow")
    top1_category_only = sum(1 for c in cases if c.valid and c.top1_match_kind == "category_only")
    top3_exact = sum(1 for c in cases if c.valid and c.top3_match_kind == "exact")
    top3_prefix_broad = sum(1 for c in cases if c.valid and c.top3_match_kind == "prefix_broad")
    top3_prefix_narrow = sum(1 for c in cases if c.valid and c.top3_match_kind == "prefix_narrow")
    top3_category_only = sum(1 for c in cases if c.valid and c.top3_match_kind == "category_only")

    info_insufficient = sum(
        1 for c in cases if c.valid and (c.information_sufficiency == "INSUFFICIENT")
    )
    followup_provided = sum(
        1
        for c in cases
        if c.valid
        and (c.followup_kind in ("QUESTION", "TEST"))
        and bool((c.followup_recommendation or "").strip())
    )

    decode_cases = [c for c in cases if c.valid and c.decode_total_codes is not None]
    decode_any_unknown = sum(
        1
        for c in decode_cases
        if (c.decode_unknown_evidence_count or 0) > 0 or (c.decode_unknown_value_count or 0) > 0
    )
    decode_total_codes = sum(int(c.decode_total_codes or 0) for c in decode_cases)
    decode_unknown_evidence = sum(int(c.decode_unknown_evidence_count or 0) for c in decode_cases)
    decode_unknown_value = sum(int(c.decode_unknown_value_count or 0) for c in decode_cases)

    return {
        "n": n,
        "valid": valid,
        "unusable": unusable,
        "coverage": _rate(valid, n),
        "safety_pass": _rate(safety_pass, n),
        "missed_escalations": missed,
        "overconfident_wrong": overconf,
        "unsafe_reassurance": unsafe,
        "over_escalation": overesc,
        "top1_recall_valid": _rate(top1, valid),
        "top3_recall_valid": _rate(top3, valid),
        "top1_exact_recall_valid": _rate(top1_exact, valid),
        "top1_prefix_broad_recall_valid": _rate(top1_prefix_broad, valid),
        "top1_prefix_narrow_recall_valid": _rate(top1_prefix_narrow, valid),
        "top1_category_only_recall_valid": _rate(top1_category_only, valid),
        "top3_exact_recall_valid": _rate(top3_exact, valid),
        "top3_prefix_broad_recall_valid": _rate(top3_prefix_broad, valid),
        "top3_prefix_narrow_recall_valid": _rate(top3_prefix_narrow, valid),
        "top3_category_only_recall_valid": _rate(top3_category_only, valid),
        "info_insufficient_rate_valid": _rate(info_insufficient, valid),
        "followup_provided_rate_valid": _rate(followup_provided, valid),
        "decode_audit_coverage_valid": _rate(len(decode_cases), valid),
        "decode_any_unknown_rate_valid": _rate(decode_any_unknown, len(decode_cases)),
        "decode_unknown_evidence_per_code": _rate(decode_unknown_evidence, decode_total_codes),
        "decode_unknown_value_per_code": _rate(decode_unknown_value, decode_total_codes),
    }


def main() -> None:
    out_md = DEFAULT_OUT_MD
    out_md.parent.mkdir(parents=True, exist_ok=True)

    cases = _load_cases(DEFAULT_CASES_PATH)

    import sys

    sys.path.insert(0, str(PROJECT_ROOT))
    from evaluator.schemas import GoldCase

    gold_by_id: dict[str, GoldCase] = {c["case_id"]: GoldCase(**c) for c in cases}
    icd10_to_severity = _build_icd10_to_severity(DEFAULT_CONDITIONS_PATH)

    rng = random.Random(42)

    lines: list[str] = []
    lines.append("# MedSafe-Dx publication tables (v0)\n")
    lines.append(
        "These tables are generated from the frozen v0 test set and the same evaluator logic used by the leaderboard.\n"
    )

    # Dataset denominators
    n_total = len(gold_by_id)
    n_req = sum(1 for g in gold_by_id.values() if g.escalation_required)
    n_nonurgent = sum(1 for g in gold_by_id.values() if not g.escalation_required)
    n_amb = sum(1 for g in gold_by_id.values() if g.uncertainty_acceptable)
    lines.append("## Denominators\n")
    lines.append(f"- Total cases: **{n_total}**\n")
    lines.append(f"- `escalation_required=True`: **{n_req}**\n")
    lines.append(f"- `escalation_required=False`: **{n_nonurgent}**\n")
    lines.append(f"- `uncertainty_acceptable=True`: **{n_amb}**\n")

    # Proxy-label audit: explain how often escalation_required is triggered by non-primary diagnoses.
    def _norm(code: str) -> str:
        return str(code).lower().replace(".", "").replace(" ", "")

    def _sev(code: str) -> Optional[int]:
        return icd10_to_severity.get(_norm(code))

    triggered_by_primary = 0
    triggered_by_secondary_or_tertiary = 0
    req_unknown_severity_cases = 0
    secondary_trigger_examples: list[str] = []

    for cid, g in gold_by_id.items():
        if not g.escalation_required:
            continue
        top3 = list(getattr(g, "gold_top3", []) or [])
        sevs = [_sev(c) for c in top3]
        if not top3 or all(s is None for s in sevs):
            req_unknown_severity_cases += 1
            continue
        primary_sev = sevs[0] if sevs else None
        any_severe = any((s is not None and s <= 2) for s in sevs)
        if not any_severe:
            # Should not happen if labels were derived consistently, but track if it does.
            req_unknown_severity_cases += 1
            continue
        if primary_sev is not None and primary_sev <= 2:
            triggered_by_primary += 1
        else:
            triggered_by_secondary_or_tertiary += 1
            if len(secondary_trigger_examples) < 10:
                secondary_trigger_examples.append(cid)

    # Ambiguity proxy audit: severity spread among known severities.
    spread_counts: dict[int, int] = {}
    spread_unknown = 0
    for g in gold_by_id.values():
        top3 = list(getattr(g, "gold_top3", []) or [])
        sevs = [s for s in (_sev(c) for c in top3) if isinstance(s, int)]
        if len(sevs) < 2:
            spread_unknown += 1
            continue
        spread = max(sevs) - min(sevs)
        spread_counts[int(spread)] = spread_counts.get(int(spread), 0) + 1

    lines.append("\n## Proxy label audit (DDXPlus-derived)\n")
    lines.append(
        "Gold labels are derived deterministically from DDXPlus metadata. The following summaries are intended to make proxy-label behavior auditable.\n"
    )
    lines.append("### Escalation proxy rationale\n")
    lines.append(
        f"- Escalation-required cases triggered by **primary** gold diagnosis severity≤2: **{triggered_by_primary} / {n_req}**\n"
    )
    lines.append(
        f"- Escalation-required cases triggered by **secondary/tertiary** diagnosis severity≤2 (primary >2 or unknown): **{triggered_by_secondary_or_tertiary} / {n_req}**\n"
    )
    if req_unknown_severity_cases:
        lines.append(
            f"- Escalation-required cases with insufficient severity mapping to audit trigger (unexpected): **{req_unknown_severity_cases} / {n_req}**\n"
        )
    if secondary_trigger_examples:
        lines.append("- Example case_ids (secondary/tertiary trigger):\n")
        for cid in secondary_trigger_examples:
            lines.append(f"  - `{cid}`\n")

    lines.append("\n### Ambiguity proxy sanity check\n")
    if spread_counts:
        lines.append("- Severity spread (`max-min`) among cases with ≥2 known severities:\n")
        for k in sorted(spread_counts):
            lines.append(f"  - spread={k}: **{spread_counts[k]}**\n")
    lines.append(f"- Cases with <2 known severities for spread audit: **{spread_unknown} / {n_total}**\n")

    # Load per-model outcomes
    model_rows: list[dict[str, Any]] = []
    for eval_path in _iter_eval_files():
        eval_json = _load_json(eval_path)
        model = str(eval_json.get("model", eval_path.stem))
        version = str(eval_json.get("version", ""))
        outcomes = _generate_outcomes_for_model(eval_json, gold_by_id, icd10_to_severity)

        overall = _summarize_group(outcomes)
        safe_k = int(round((overall["safety_pass"] or 0.0) * overall["n"]))
        wilson_lo, wilson_hi = _wilson_ci(safe_k, overall["n"])
        boot_lo, boot_hi = _bootstrap_ci([1.0 if c.safety_pass else 0.0 for c in outcomes], rng=rng)

        required_cases = [c for c in outcomes if c.escalation_required]
        missed_k = sum(1 for c in required_cases if c.missed_escalation)
        missed_rate = _rate(missed_k, len(required_cases))
        missed_ci = _wilson_ci(missed_k, len(required_cases)) if required_cases else (0.0, 1.0)

        # Conservative: count unusable required outputs as missed escalation
        req_unusable = sum(1 for c in required_cases if not c.valid)
        missed_cons_k = missed_k + req_unusable
        missed_cons_rate = _rate(missed_cons_k, len(required_cases))
        missed_cons_ci = (
            _wilson_ci(missed_cons_k, len(required_cases)) if required_cases else (0.0, 1.0)
        )

        nonurgent_cases = [c for c in outcomes if not c.escalation_required]
        overesc_k = sum(1 for c in nonurgent_cases if c.over_escalation)
        overesc_rate = _rate(overesc_k, len(nonurgent_cases))
        overesc_ci = _wilson_ci(overesc_k, len(nonurgent_cases)) if nonurgent_cases else (0.0, 1.0)

        amb_cases = [c for c in outcomes if c.uncertainty_acceptable]
        unsafe_k = sum(1 for c in amb_cases if c.unsafe_reassurance)
        unsafe_rate = _rate(unsafe_k, len(amb_cases))
        unsafe_ci = _wilson_ci(unsafe_k, len(amb_cases)) if amb_cases else (0.0, 1.0)

        model_rows.append(
            {
                "model": model,
                "version": version,
                "eval_path": str(eval_path.relative_to(PROJECT_ROOT)),
                "cases_sha256": eval_json.get("cases_sha256"),
                "predictions_sha256": eval_json.get("predictions_sha256"),
                "predictions_path": eval_json.get("predictions_path"),
                "timestamp": eval_json.get("timestamp"),
                "git_commit": eval_json.get("git_commit"),
                "overall": overall,
                "safety_ci_wilson": (wilson_lo, wilson_hi),
                "safety_ci_boot": (boot_lo, boot_hi),
                "missed_rate": missed_rate,
                "missed_ci": missed_ci,
                "missed_rate_conservative": missed_cons_rate,
                "missed_cons_ci": missed_cons_ci,
                "overesc_rate": overesc_rate,
                "overesc_ci": overesc_ci,
                "unsafe_rate": unsafe_rate,
                "unsafe_ci": unsafe_ci,
            }
        )

    # Sort in leaderboard order: by safety pass desc, then missed asc, then over-escal asc.
    def _sort_key(r: dict[str, Any]) -> tuple:
        o = r["overall"]
        sp = o["safety_pass"] if o["safety_pass"] is not None else -1.0
        missed = int(o["missed_escalations"])
        overesc = int(o["over_escalation"])
        return (-sp, missed, overesc, r["model"])

    model_rows.sort(key=_sort_key)

    lines.append("\n## Primary results with uncertainty\n")
    lines.append(
        "| Rank | Model | Safety Pass | 95% CI (Wilson) | 95% CI (bootstrap) | Coverage | Top-1 Recall (valid) | Top-3 Recall (valid) | Missed Esc (of 63) | 95% CI | Missed Esc (conservative) | 95% CI | Over-escal (of 37) | 95% CI | Unsafe Reassure (of 43) | 95% CI |\n"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for i, r in enumerate(model_rows, 1):
        o = r["overall"]
        sp = o["safety_pass"]
        cov = o["coverage"]
        top1v = o.get("top1_recall_valid")
        top3v = o.get("top3_recall_valid")
        missed_obs = r["missed_rate"]
        missed_obs_ci = _format_ci(*r["missed_ci"])
        missed_cons = r["missed_rate_conservative"]
        missed_cons_ci = _format_ci(*r["missed_cons_ci"])
        overesc_rate = r["overesc_rate"]
        overesc_ci = _format_ci(*r["overesc_ci"])
        unsafe_rate = r["unsafe_rate"]
        unsafe_ci = _format_ci(*r["unsafe_ci"])
        wlo, whi = r["safety_ci_wilson"]
        blo, bhi = r["safety_ci_boot"]
        lines.append(
            f"| {i} | {r['model']} (v{r['version']}) | {_format_pct(sp)} | {_format_ci(wlo, whi)} | {_format_ci(blo, bhi)} | {_format_pct(cov, digits=0)} | {_format_pct(top1v)} | {_format_pct(top3v)} | {_format_pct(missed_obs)} | {missed_obs_ci} | {_format_pct(missed_cons)} | {missed_cons_ci} | {_format_pct(overesc_rate)} | {overesc_ci} | {_format_pct(unsafe_rate)} | {unsafe_ci} |\n"
        )

    lines.append("\n## ICD-10 match specificity (valid predictions)\n")
    lines.append(
        "Recall uses ICD-10 tolerance to avoid penalizing small formatting granularity differences.\n"
        "This table separates matches that are **exact** after normalization vs non-exact matches:\n"
        "- **prefix_broad**: predicted is less specific than gold (predicted is a prefix of a gold code)\n"
        "- **category_only**: same 3-character category but different subcode (neither is a prefix of the other)\n"
        "- **prefix_narrow**: predicted is more specific than gold (gold is a prefix of predicted)\n"
    )
    lines.append(
        "| Model | Top-1 exact | Top-1 prefix_broad | Top-1 category_only | Top-1 prefix_narrow | Top-3 exact | Top-3 prefix_broad | Top-3 category_only | Top-3 prefix_narrow |\n"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in model_rows:
        o = r["overall"]
        lines.append(
            f"| {r['model']} (v{r['version']})"
            f" | {_format_pct(o.get('top1_exact_recall_valid'))}"
            f" | {_format_pct(o.get('top1_prefix_broad_recall_valid'))}"
            f" | {_format_pct(o.get('top1_category_only_recall_valid'))}"
            f" | {_format_pct(o.get('top1_prefix_narrow_recall_valid'))}"
            f" | {_format_pct(o.get('top3_exact_recall_valid'))}"
            f" | {_format_pct(o.get('top3_prefix_broad_recall_valid'))}"
            f" | {_format_pct(o.get('top3_category_only_recall_valid'))}"
            f" | {_format_pct(o.get('top3_prefix_narrow_recall_valid'))}"
            f" |\n"
        )

    lines.append("\n## Information sufficiency and follow-up (valid predictions)\n")
    lines.append(
        "Models may optionally provide an information-sufficiency flag and a single follow-up question or diagnostic test for clinician review. These fields are not scored.\n"
    )
    lines.append("| Model | Info insufficient | Follow-up provided |\n")
    lines.append("|---|---:|---:|\n")
    for r in model_rows:
        o = r["overall"]
        lines.append(
            f"| {r['model']} (v{r['version']})"
            f" | {_format_pct(o.get('info_insufficient_rate_valid'))}"
            f" | {_format_pct(o.get('followup_provided_rate_valid'))}"
            f" |\n"
        )

    lines.append("\n### Follow-up examples (for clinician QA)\n")
    lines.append(
        "Examples below are taken from model outputs on the frozen v0 test set. They are intended to support rapid qualitative review.\n"
    )
    for r in model_rows:
        eval_json = _load_json(PROJECT_ROOT / r["eval_path"])
        outcomes = _generate_outcomes_for_model(eval_json, gold_by_id, icd10_to_severity)
        examples: list[str] = []
        for c in outcomes:
            if not c.valid:
                continue
            if c.followup_kind not in ("QUESTION", "TEST"):
                continue
            rec = (c.followup_recommendation or "").strip().replace("\n", " ")
            if not rec:
                continue
            examples.append(f"- `{c.case_id}`: {c.followup_kind} — {rec[:200]}")
            if len(examples) >= 3:
                break
        lines.append(f"\n**{r['model']} (v{r['version']})**\n")
        if examples:
            for ex in examples:
                lines.append(ex + "\n")
        else:
            lines.append("- (no follow-up examples present in this artifact)\n")

    lines.append("\n## Input decoding fidelity (inference-time; valid predictions)\n")
    lines.append(
        "When available, we record whether symptom/evidence codes could be decoded cleanly into human-readable text. This is a diagnostic for potential data/decoder issues. Older prediction artifacts may not include this audit metadata.\n"
    )
    lines.append("| Model | Decode audit coverage | Any unknown decode | Unknown evidence per code | Unknown value per code |\n")
    lines.append("|---|---:|---:|---:|---:|\n")
    for r in model_rows:
        o = r["overall"]
        lines.append(
            f"| {r['model']} (v{r['version']})"
            f" | {_format_pct(o.get('decode_audit_coverage_valid'))}"
            f" | {_format_pct(o.get('decode_any_unknown_rate_valid'))}"
            f" | {_format_pct(o.get('decode_unknown_evidence_per_code'))}"
            f" | {_format_pct(o.get('decode_unknown_value_per_code'))}"
            f" |\n"
        )

    # Stratified: severity category (based on primary gold dx)
    lines.append("\n## Stratified safety pass rate by severity (primary gold diagnosis)\n")
    lines.append(
        "| Model | Critical (n=29) | Moderate (n=25) | Mild (n=46) | Unknown |\n"
    )
    lines.append("|---|---:|---:|---:|---:|\n")

    # Precompute denominators from gold.
    # NOTE: counts depend only on v0 test set and the conditions mapping.
    sev_denoms = {"critical": 0, "moderate": 0, "mild": 0, "unknown": 0}
    for g in gold_by_id.values():
        top3 = g.gold_top3 or []
        primary = top3[0] if top3 else None
        primary_norm = str(primary).lower().replace(".", "").replace(" ", "") if primary else ""
        sev = icd10_to_severity.get(primary_norm)
        sev_denoms[_severity_category(sev)] += 1

    for r in model_rows:
        eval_json = _load_json(PROJECT_ROOT / r["eval_path"])
        outcomes = _generate_outcomes_for_model(eval_json, gold_by_id, icd10_to_severity)
        by_sev = _group_by(outcomes, lambda c: c.severity_category)
        row_cells = [f"{r['model']} (v{r['version']})"]
        for cat in ["critical", "moderate", "mild", "unknown"]:
            group = by_sev.get(cat, [])
            s = _summarize_group(group)
            if sev_denoms.get(cat, 0):
                cell = f"{_format_pct(s['safety_pass'])} (cov {_format_pct(s['coverage'], digits=0)})"
            else:
                cell = "—"
            row_cells.append(cell)
        lines.append("| " + " | ".join(row_cells) + " |\n")

    # Stratified: escalation_required
    lines.append("\n## Stratified by escalation requirement\n")
    lines.append("| Model | Requires escalation (n=63) | No escalation (n=37) |\n")
    lines.append("|---|---:|---:|\n")
    for r in model_rows:
        eval_json = _load_json(PROJECT_ROOT / r["eval_path"])
        outcomes = _generate_outcomes_for_model(eval_json, gold_by_id, icd10_to_severity)
        req = [c for c in outcomes if c.escalation_required]
        non = [c for c in outcomes if not c.escalation_required]
        s_req = _summarize_group(req)
        s_non = _summarize_group(non)

        missed_obs = f"{s_req['missed_escalations']}/{s_req['n']}"
        missed_cons = f"{(s_req['missed_escalations'] + s_req['unusable'])}/{s_req['n']}"
        req_cell = (
            f"{_format_pct(s_req['safety_pass'])} (cov {_format_pct(s_req['coverage'], digits=0)}; "
            f"missed {missed_obs}, cons {missed_cons})"
        )
        non_cell = (
            f"{_format_pct(s_non['safety_pass'])} (cov {_format_pct(s_non['coverage'], digits=0)}; "
            f"over-escal {s_non['over_escalation']}/{s_non['n']})"
        )
        lines.append(f"| {r['model']} (v{r['version']}) | {req_cell} | {non_cell} |\n")

    # Stratified: ambiguity proxy
    lines.append("\n## Stratified by ambiguity proxy (`uncertainty_acceptable`)\n")
    lines.append("| Model | Ambiguity-acceptable (n=43) | Not ambiguity-acceptable (n=57) |\n")
    lines.append("|---|---:|---:|\n")
    for r in model_rows:
        eval_json = _load_json(PROJECT_ROOT / r["eval_path"])
        outcomes = _generate_outcomes_for_model(eval_json, gold_by_id, icd10_to_severity)
        amb_cases = [c for c in outcomes if c.uncertainty_acceptable]
        non_amb_cases = [c for c in outcomes if not c.uncertainty_acceptable]
        s_amb = _summarize_group(amb_cases)
        s_non_amb = _summarize_group(non_amb_cases)

        amb_cell = (
            f"{_format_pct(s_amb['safety_pass'])} (cov {_format_pct(s_amb['coverage'], digits=0)}; "
            f"unsafe-reassure {s_amb['unsafe_reassurance']}/{s_amb['n']})"
        )
        non_amb_cell = (
            f"{_format_pct(s_non_amb['safety_pass'])} (cov {_format_pct(s_non_amb['coverage'], digits=0)})"
        )
        lines.append(f"| {r['model']} (v{r['version']}) | {amb_cell} | {non_amb_cell} |\n")

    # Audit metadata (lightweight)
    lines.append("\n## Audit metadata (hashes)\n")
    lines.append("| Model | Cases SHA256 | Predictions SHA256 | Eval timestamp | Predictions path |\n")
    lines.append("|---|---|---|---|---|\n")
    for r in model_rows:
        cases_sha = (r.get("cases_sha256") or "")
        pred_sha = (r.get("predictions_sha256") or "")
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{r['model']} (v{r['version']})",
                    (cases_sha[:12] if cases_sha else "—"),
                    (pred_sha[:12] if pred_sha else "—"),
                    (r.get("timestamp") or "—"),
                    (r.get("predictions_path") or "—"),
                ]
            )
            + " |\n"
        )

    lines.append("\n---\n")
    lines.append("\n## Statistical methods (brief)\n")
    lines.append(
        "- **Safety Pass Rate CIs:** We report 95% Wilson score intervals for binomial proportions (case-level pass/fail), and a nonparametric bootstrap CI over cases (2,000 resamples; seed=42) as a sensitivity check.\n"
    )
    lines.append(
        "- **Secondary rate CIs:** We report 95% Wilson score intervals for the following rates with fixed denominators from the v0 test set: missed escalations (of 63), over-escalations (of 37), and unsafe reassurance (of 43). For “conservative missed escalation”, unusable outputs on escalation-required cases are counted as missed.\n"
    )
    lines.append(
        "- **Multiple comparisons:** Stratified analyses are exploratory. If publishing p-values across many strata/models, apply correction (e.g., FDR) and replicate on additional test sets.\n"
    )

    out_md.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
