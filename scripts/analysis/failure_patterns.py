#!/usr/bin/env python3
"""
Failure Pattern Analysis for MedSafe-Dx.

Analyzes systematic failure patterns across models:
- Missed escalations: Which gold conditions? Common symptom patterns?
- Overconfident wrong: What did models predict vs actual?
- Unsafe reassurance: Case characteristics?
- Universal failures: Cases that ALL models fail

Outputs:
- results/analysis/failure_patterns.json
- results/analysis/failure_case_studies.md
"""

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .config import PATHS, SAFETY_FAILURES
from .utils import (
    load_test_cases,
    load_predictions,
    list_model_results,
    evaluate_case,
    get_icd10_description,
    format_diagnosis_code,
    decode_symptoms,
    get_condition_severity,
)


def analyze_missed_escalations(
    failures: List[Dict],
    cases_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Analyze patterns in missed escalation failures."""
    if not failures:
        return {"count": 0}

    # Which gold conditions were missed?
    gold_conditions = Counter()
    for f in failures:
        case = cases_df[cases_df["case_id"] == f["case_id"]].iloc[0]
        for code in case["gold_top3"]:
            gold_conditions[code] += 1

    # What did models predict instead?
    predicted_escalations = Counter()
    for f in failures:
        esc = f.get("pred_escalation", "")
        predicted_escalations[esc] += 1

    # Severity distribution
    severity_dist = Counter()
    for f in failures:
        case = cases_df[cases_df["case_id"] == f["case_id"]].iloc[0]
        primary_code = case["gold_top3"][0] if case["gold_top3"] else None
        if primary_code:
            sev = get_condition_severity(primary_code)
            severity_dist[sev or "unknown"] += 1

    # Format top conditions with descriptions
    top_conditions = []
    for code, count in gold_conditions.most_common(10):
        desc = get_icd10_description(code)
        top_conditions.append({
            "code": code,
            "description": desc,
            "count": count,
            "pct": count / len(failures) * 100,
        })

    return {
        "count": len(failures),
        "top_gold_conditions": top_conditions,
        "predicted_escalation_distribution": dict(predicted_escalations),
        "severity_distribution": dict(severity_dist),
    }


def analyze_overconfident_wrong(
    failures: List[Dict],
    cases_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Analyze patterns in overconfident wrong predictions."""
    if not failures:
        return {"count": 0}

    # What did models predict (wrong)?
    predicted_codes = Counter()
    for f in failures:
        for dx in f.get("pred_dx", [])[:3]:
            code = dx.get("code", dx) if isinstance(dx, dict) else dx
            predicted_codes[str(code).upper()] += 1

    # What were the actual gold diagnoses?
    gold_codes = Counter()
    for f in failures:
        case = cases_df[cases_df["case_id"] == f["case_id"]].iloc[0]
        for code in case["gold_top3"]:
            gold_codes[code.upper()] += 1

    # Near-miss analysis: same ICD-10 chapter but wrong specific code
    near_misses = 0
    for f in failures:
        case = cases_df[cases_df["case_id"] == f["case_id"]].iloc[0]
        gold_chapters = set(str(c)[0].upper() for c in case["gold_top3"])
        pred_chapters = set()
        for dx in f.get("pred_dx", [])[:3]:
            code = dx.get("code", dx) if isinstance(dx, dict) else dx
            if code:
                pred_chapters.add(str(code)[0].upper())
        if gold_chapters & pred_chapters:
            near_misses += 1

    # Format top predicted codes
    top_predicted = []
    for code, count in predicted_codes.most_common(10):
        desc = get_icd10_description(code)
        top_predicted.append({
            "code": code,
            "description": desc,
            "count": count,
        })

    # Format top gold codes
    top_gold = []
    for code, count in gold_codes.most_common(10):
        desc = get_icd10_description(code)
        top_gold.append({
            "code": code,
            "description": desc,
            "count": count,
        })

    return {
        "count": len(failures),
        "top_predicted_codes": top_predicted,
        "top_gold_codes": top_gold,
        "near_miss_count": near_misses,
        "near_miss_rate": near_misses / len(failures) * 100 if failures else 0,
    }


def analyze_unsafe_reassurance(
    failures: List[Dict],
    cases_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Analyze patterns in unsafe reassurance failures."""
    if not failures:
        return {"count": 0}

    # Gold conditions that got unsafe reassurance
    gold_conditions = Counter()
    for f in failures:
        case = cases_df[cases_df["case_id"] == f["case_id"]].iloc[0]
        for code in case["gold_top3"]:
            gold_conditions[code] += 1

    # Symptom count distribution
    symptom_counts = []
    for f in failures:
        case = cases_df[cases_df["case_id"] == f["case_id"]].iloc[0]
        symptom_counts.append(len(case["presenting_symptoms"]))

    avg_symptoms = sum(symptom_counts) / len(symptom_counts) if symptom_counts else 0

    # Format top conditions
    top_conditions = []
    for code, count in gold_conditions.most_common(10):
        desc = get_icd10_description(code)
        top_conditions.append({
            "code": code,
            "description": desc,
            "count": count,
        })

    return {
        "count": len(failures),
        "top_gold_conditions": top_conditions,
        "avg_symptom_count": avg_symptoms,
        "symptom_count_range": (min(symptom_counts), max(symptom_counts)) if symptom_counts else (0, 0),
    }


def find_universal_failures(
    all_model_failures: Dict[str, Dict[str, List[str]]],
    n_models: int,
) -> Dict[str, List[str]]:
    """Find cases that fail across ALL models."""
    # For each failure type, find cases that appear in all models
    universal = {}

    for failure_type in SAFETY_FAILURES:
        case_counts = Counter()
        for model_id, failures_by_type in all_model_failures.items():
            for case_id in failures_by_type.get(failure_type, []):
                case_counts[case_id] += 1

        # Cases that failed in all models
        universal_cases = [
            case_id for case_id, count in case_counts.items()
            if count == n_models
        ]
        universal[failure_type] = universal_cases

    # Also find cases with ANY safety failure across all models
    any_failure_counts = Counter()
    for model_id, failures_by_type in all_model_failures.items():
        all_cases = set()
        for case_list in failures_by_type.values():
            all_cases.update(case_list)
        for case_id in all_cases:
            any_failure_counts[case_id] += 1

    universal["any_safety_failure"] = [
        case_id for case_id, count in any_failure_counts.items()
        if count == n_models
    ]

    return universal


def find_consensus_successes(
    all_model_results: Dict[str, List[Dict]],
    n_models: int,
) -> List[str]:
    """Find cases that ALL models pass."""
    pass_counts = Counter()

    for model_id, cases in all_model_results.items():
        for case in cases:
            if case["safety_pass"]:
                pass_counts[case["case_id"]] += 1

    return [
        case_id for case_id, count in pass_counts.items()
        if count == n_models
    ]


def generate_case_study(
    case_id: str,
    cases_df: pd.DataFrame,
    all_model_results: Dict[str, List[Dict]],
) -> Dict[str, Any]:
    """Generate a detailed case study for a specific case."""
    case = cases_df[cases_df["case_id"] == case_id].iloc[0]

    # Decode symptoms
    symptoms = case["presenting_symptoms"]
    active, antecedents = decode_symptoms(symptoms)

    # Format gold diagnoses
    gold_dx = []
    for code in case["gold_top3"]:
        gold_dx.append(format_diagnosis_code(code))

    # Collect model results for this case
    model_results = []
    for model_id, cases in all_model_results.items():
        for c in cases:
            if c["case_id"] == case_id:
                model_results.append({
                    "model": model_id,
                    "safety_pass": c["safety_pass"],
                    "failures": c.get("safety_failures", []),
                    "top1_match": c["top1_match"],
                    "top3_match": c["top3_match"],
                    "pred_escalation": c.get("pred_escalation", ""),
                    "pred_uncertainty": c.get("pred_uncertainty", ""),
                    "pred_dx": c.get("pred_dx", []),
                })
                break

    return {
        "case_id": case_id,
        "demographics": f"{case['age']}yo {case['sex']}",
        "presenting_symptoms": active,
        "antecedents": antecedents,
        "gold_diagnoses": gold_dx,
        "escalation_required": case["escalation_required"],
        "uncertainty_acceptable": case["uncertainty_acceptable"],
        "model_results": model_results,
    }


def generate_markdown_report(
    analysis: Dict[str, Any],
    case_studies: List[Dict],
) -> str:
    """Generate markdown report for failure patterns."""
    lines = [
        "# MedSafe-Dx Failure Pattern Analysis",
        "",
        "This report identifies systematic failure patterns across models.",
        "",
    ]

    # Summary stats
    lines.extend([
        "## Summary Statistics",
        "",
        f"- Total models analyzed: {analysis['n_models']}",
        f"- Total cases per model: {analysis['n_cases']}",
        "",
    ])

    # Missed Escalations
    me = analysis["missed_escalations"]
    lines.extend([
        "## Missed Escalations",
        "",
        f"Total occurrences across all models: {me['count']}",
        "",
        "### Most Commonly Missed Conditions",
        "",
        "| ICD-10 | Condition | Count | % of Missed |",
        "|--------|-----------|-------|-------------|",
    ])
    for cond in me.get("top_gold_conditions", [])[:10]:
        desc = cond["description"] or "Unknown"
        lines.append(f"| {cond['code']} | {desc} | {cond['count']} | {cond['pct']:.1f}% |")

    lines.extend([
        "",
        "### Model Escalation Decisions (when missed)",
        "",
    ])
    for esc, count in me.get("predicted_escalation_distribution", {}).items():
        lines.append(f"- **{esc or 'Unknown'}**: {count} times")

    lines.append("")

    # Overconfident Wrong
    ow = analysis["overconfident_wrong"]
    lines.extend([
        "## Overconfident Wrong Predictions",
        "",
        f"Total occurrences across all models: {ow['count']}",
        f"Near-miss rate (correct chapter, wrong code): {ow.get('near_miss_rate', 0):.1f}%",
        "",
        "### Most Commonly Over-predicted Codes",
        "",
        "| ICD-10 | Predicted Condition | Count |",
        "|--------|---------------------|-------|",
    ])
    for cond in ow.get("top_predicted_codes", [])[:10]:
        desc = cond["description"] or "Unknown"
        lines.append(f"| {cond['code']} | {desc} | {cond['count']} |")

    lines.extend([
        "",
        "### Actual Gold Conditions (when overconfident)",
        "",
        "| ICD-10 | Actual Condition | Count |",
        "|--------|------------------|-------|",
    ])
    for cond in ow.get("top_gold_codes", [])[:10]:
        desc = cond["description"] or "Unknown"
        lines.append(f"| {cond['code']} | {desc} | {cond['count']} |")

    lines.append("")

    # Unsafe Reassurance
    ur = analysis["unsafe_reassurance"]
    lines.extend([
        "## Unsafe Reassurance",
        "",
        f"Total occurrences across all models: {ur['count']}",
        f"Average symptom count: {ur.get('avg_symptom_count', 0):.1f}",
        "",
        "### Conditions Most Often Given Unsafe Reassurance",
        "",
        "| ICD-10 | Condition | Count |",
        "|--------|-----------|-------|",
    ])
    for cond in ur.get("top_gold_conditions", [])[:10]:
        desc = cond["description"] or "Unknown"
        lines.append(f"| {cond['code']} | {desc} | {cond['count']} |")

    lines.append("")

    # Universal Failures
    uf = analysis.get("universal_failures", {})
    lines.extend([
        "## Universal Failures (All Models Fail)",
        "",
        "Cases where every model exhibits the same failure type:",
        "",
        f"- **Any safety failure**: {len(uf.get('any_safety_failure', []))} cases",
        f"- **Missed escalation**: {len(uf.get('missed_escalation', []))} cases",
        f"- **Overconfident wrong**: {len(uf.get('overconfident_wrong', []))} cases",
        f"- **Unsafe reassurance**: {len(uf.get('unsafe_reassurance', []))} cases",
        "",
    ])

    # Consensus Successes
    cs = analysis.get("consensus_successes", [])
    lines.extend([
        f"## Consensus Successes: {len(cs)} cases",
        "",
        "Cases where all models pass safety checks.",
        "",
    ])

    # Case Studies
    if case_studies:
        lines.extend([
            "---",
            "",
            "## Case Studies",
            "",
            "Detailed analysis of notable failure cases.",
            "",
        ])

        for i, study in enumerate(case_studies[:10], 1):  # Limit to 10
            lines.extend([
                f"### Case Study {i}: {study['case_id']}",
                "",
                f"**Demographics:** {study['demographics']}",
                "",
                "**Presenting Symptoms:**",
            ])
            for symptom in study["presenting_symptoms"][:10]:
                lines.append(f"- {symptom}")
            if len(study["presenting_symptoms"]) > 10:
                lines.append(f"- ... and {len(study['presenting_symptoms']) - 10} more")

            if study["antecedents"]:
                lines.extend(["", "**Antecedents:**"])
                for ant in study["antecedents"][:5]:
                    lines.append(f"- {ant}")

            lines.extend([
                "",
                "**Gold Diagnoses:**",
            ])
            for dx in study["gold_diagnoses"]:
                lines.append(f"- {dx}")

            lines.extend([
                "",
                f"**Escalation Required:** {'Yes' if study['escalation_required'] else 'No'}",
                f"**Uncertainty Acceptable:** {'Yes' if study['uncertainty_acceptable'] else 'No'}",
                "",
                "**Model Results:**",
                "",
                "| Model | Safety | Failures | Top-3 Match | Escalation |",
                "|-------|--------|----------|-------------|------------|",
            ])

            for mr in study["model_results"]:
                status = "PASS" if mr["safety_pass"] else "FAIL"
                failures = ", ".join(mr["failures"]) if mr["failures"] else "-"
                t3 = "Yes" if mr["top3_match"] else "No"
                esc = mr["pred_escalation"] or "-"
                lines.append(f"| {mr['model'][:30]} | {status} | {failures} | {t3} | {esc} |")

            lines.append("")

    return "\n".join(lines)


def main():
    """Run failure pattern analysis."""
    print("Loading test cases...")
    cases_df = load_test_cases()
    print(f"  Loaded {len(cases_df)} test cases")

    # Load all model results
    print("\nLoading model results...")
    model_files = list_model_results()
    print(f"  Found {len(model_files)} models")

    # Collect all failures and results
    all_failures = {ft: [] for ft in SAFETY_FAILURES}
    all_model_failures = {}  # model_id -> {failure_type -> [case_ids]}
    all_model_results = {}  # model_id -> [case dicts]

    for mf in model_files:
        model_id = mf["model_id"]
        print(f"  Processing {model_id}...")

        try:
            predictions_df = load_predictions(mf["predictions_path"])
            merged = cases_df.merge(predictions_df, on="case_id", how="inner")

            model_failures = {ft: [] for ft in SAFETY_FAILURES}
            model_results = []

            for _, row in merged.iterrows():
                prediction = {
                    "differential_diagnoses": row.get("differential_diagnoses", []),
                    "escalation_decision": row.get("escalation_decision", ""),
                    "uncertainty": row.get("uncertainty", ""),
                }
                gold = {
                    "gold_top3": row.get("gold_top3", []),
                    "escalation_required": row.get("escalation_required", False),
                    "uncertainty_acceptable": row.get("uncertainty_acceptable", False),
                }
                eval_result = evaluate_case(prediction, gold)

                case_info = {
                    "case_id": row["case_id"],
                    "model_id": model_id,
                    "pred_dx": row.get("differential_diagnoses", []),
                    "pred_escalation": row.get("escalation_decision", ""),
                    "pred_uncertainty": row.get("uncertainty", ""),
                    **eval_result,
                }

                model_results.append(case_info)

                # Categorize failures
                if eval_result["missed_escalation"]:
                    all_failures["missed_escalation"].append(case_info)
                    model_failures["missed_escalation"].append(row["case_id"])

                if eval_result["overconfident_wrong"]:
                    all_failures["overconfident_wrong"].append(case_info)
                    model_failures["overconfident_wrong"].append(row["case_id"])

                if eval_result["unsafe_reassurance"]:
                    all_failures["unsafe_reassurance"].append(case_info)
                    model_failures["unsafe_reassurance"].append(row["case_id"])

            all_model_failures[model_id] = model_failures
            all_model_results[model_id] = model_results

        except Exception as e:
            print(f"    Error: {e}")

    if not all_model_results:
        print("No results to analyze!")
        return

    n_models = len(all_model_results)

    # Analyze each failure type
    print("\nAnalyzing failure patterns...")

    analysis = {
        "n_models": n_models,
        "n_cases": len(cases_df),
        "missed_escalations": analyze_missed_escalations(
            all_failures["missed_escalation"], cases_df
        ),
        "overconfident_wrong": analyze_overconfident_wrong(
            all_failures["overconfident_wrong"], cases_df
        ),
        "unsafe_reassurance": analyze_unsafe_reassurance(
            all_failures["unsafe_reassurance"], cases_df
        ),
    }

    # Find universal failures
    print("Finding universal failures...")
    analysis["universal_failures"] = find_universal_failures(
        all_model_failures, n_models
    )

    # Find consensus successes
    analysis["consensus_successes"] = find_consensus_successes(
        all_model_results, n_models
    )

    # Generate case studies for universal failures
    print("Generating case studies...")
    case_studies = []

    # Prioritize cases with universal failures
    study_cases = set()
    for failure_type, case_ids in analysis["universal_failures"].items():
        study_cases.update(case_ids[:3])  # Top 3 per failure type

    for case_id in list(study_cases)[:10]:  # Limit total studies
        study = generate_case_study(case_id, cases_df, all_model_results)
        case_studies.append(study)

    # Write JSON output
    output_path = PATHS["analysis_output_dir"] / "failure_patterns.json"
    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"\nWrote {output_path}")

    # Generate and write markdown report
    markdown = generate_markdown_report(analysis, case_studies)
    md_path = PATHS["analysis_output_dir"] / "failure_case_studies.md"
    with open(md_path, "w") as f:
        f.write(markdown)
    print(f"Wrote {md_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("FAILURE PATTERN SUMMARY")
    print("=" * 60)

    me = analysis["missed_escalations"]
    print(f"\nMissed Escalations: {me['count']} total")
    if me.get("top_gold_conditions"):
        print(f"  Most missed: {me['top_gold_conditions'][0]['description']} ({me['top_gold_conditions'][0]['count']}x)")

    ow = analysis["overconfident_wrong"]
    print(f"\nOverconfident Wrong: {ow['count']} total")
    print(f"  Near-miss rate: {ow.get('near_miss_rate', 0):.1f}%")

    ur = analysis["unsafe_reassurance"]
    print(f"\nUnsafe Reassurance: {ur['count']} total")

    uf = analysis["universal_failures"]
    print(f"\nUniversal Failures (all {n_models} models fail):")
    print(f"  Any safety failure: {len(uf.get('any_safety_failure', []))} cases")

    print(f"\nConsensus Successes: {len(analysis['consensus_successes'])} cases")


if __name__ == "__main__":
    main()
