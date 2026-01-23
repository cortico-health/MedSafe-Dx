#!/usr/bin/env python3
"""
Case Breakdown Analysis for MedSafe-Dx.

Analyzes which case types are hardest by stratifying results across:
- escalation_required (true/false)
- uncertainty_acceptable (true/false)
- Symptom count (low/medium/high terciles)
- Gold diagnosis severity (from release_conditions.json)

Outputs:
- results/analysis/case_breakdown.json
- results/analysis/case_breakdown_tables.md
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np
from scipy import stats

from .config import PATHS, SAFETY_FAILURES
from .utils import (
    load_test_cases,
    load_predictions,
    list_model_results,
    evaluate_case,
    get_condition_severity,
)


def get_primary_severity(gold_top3: List[str]) -> Optional[int]:
    """Get severity of the primary (first) gold diagnosis."""
    if not gold_top3:
        return None
    primary = gold_top3[0]
    return get_condition_severity(primary)


def categorize_symptom_count(count: int, terciles: tuple) -> str:
    """Categorize symptom count into low/medium/high based on terciles."""
    if count <= terciles[0]:
        return "low"
    elif count <= terciles[1]:
        return "medium"
    else:
        return "high"


def categorize_severity(severity: Optional[int]) -> str:
    """Categorize severity into critical/moderate/mild groups."""
    if severity is None:
        return "unknown"
    elif severity <= 2:
        return "critical"  # Severity 1-2: life-threatening
    elif severity <= 3:
        return "moderate"  # Severity 3: serious but not immediately life-threatening
    else:
        return "mild"  # Severity 4-5: less severe


def compute_stratum_metrics(cases_in_stratum: List[Dict]) -> Dict[str, Any]:
    """Compute all metrics for a stratum of cases."""
    if not cases_in_stratum:
        return {
            "n_cases": 0,
            "safety_pass_rate": None,
            "missed_escalation_rate": None,
            "overconfident_wrong_rate": None,
            "unsafe_reassurance_rate": None,
            "top1_recall": None,
            "top3_recall": None,
        }

    n = len(cases_in_stratum)
    safety_passes = sum(1 for c in cases_in_stratum if c["safety_pass"])
    missed_esc = sum(1 for c in cases_in_stratum if c["missed_escalation"])
    overconf = sum(1 for c in cases_in_stratum if c["overconfident_wrong"])
    unsafe = sum(1 for c in cases_in_stratum if c["unsafe_reassurance"])
    top1 = sum(1 for c in cases_in_stratum if c["top1_match"])
    top3 = sum(1 for c in cases_in_stratum if c["top3_match"])

    return {
        "n_cases": n,
        "safety_pass_rate": safety_passes / n,
        "missed_escalation_rate": missed_esc / n,
        "overconfident_wrong_rate": overconf / n,
        "unsafe_reassurance_rate": unsafe / n,
        "top1_recall": top1 / n,
        "top3_recall": top3 / n,
    }


def chi_square_test(group_a: List[bool], group_b: List[bool]) -> Dict[str, float]:
    """Perform chi-square test for difference in proportions."""
    if len(group_a) < 5 or len(group_b) < 5:
        return {"chi2": None, "p_value": None, "significant": None}

    a_success = sum(group_a)
    a_fail = len(group_a) - a_success
    b_success = sum(group_b)
    b_fail = len(group_b) - b_success

    contingency = [[a_success, a_fail], [b_success, b_fail]]

    try:
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
        return {
            "chi2": chi2,
            "p_value": p_value,
            "significant": p_value < 0.05,
        }
    except Exception:
        return {"chi2": None, "p_value": None, "significant": None}


def analyze_model_results(
    model_id: str,
    predictions_df: pd.DataFrame,
    cases_df: pd.DataFrame,
    symptom_terciles: tuple,
) -> Dict[str, Any]:
    """Analyze results for a single model across all stratifications."""

    # Merge predictions with cases
    merged = cases_df.merge(predictions_df, on="case_id", how="inner")

    # Evaluate each case
    evaluated_cases = []
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

        # Add case info
        symptoms = row.get("presenting_symptoms", [])
        gold_top3 = row.get("gold_top3", [])

        evaluated_cases.append({
            "case_id": row["case_id"],
            "escalation_required": row.get("escalation_required", False),
            "uncertainty_acceptable": row.get("uncertainty_acceptable", False),
            "symptom_count": len(symptoms),
            "symptom_category": categorize_symptom_count(len(symptoms), symptom_terciles),
            "primary_severity": get_primary_severity(gold_top3),
            "severity_category": categorize_severity(get_primary_severity(gold_top3)),
            **eval_result,
        })

    # Overall metrics
    overall = compute_stratum_metrics(evaluated_cases)

    # Stratify by escalation_required
    by_escalation = {
        "requires_escalation": compute_stratum_metrics(
            [c for c in evaluated_cases if c["escalation_required"]]
        ),
        "no_escalation": compute_stratum_metrics(
            [c for c in evaluated_cases if not c["escalation_required"]]
        ),
    }

    # Stratify by uncertainty_acceptable
    by_uncertainty = {
        "uncertainty_acceptable": compute_stratum_metrics(
            [c for c in evaluated_cases if c["uncertainty_acceptable"]]
        ),
        "uncertainty_not_acceptable": compute_stratum_metrics(
            [c for c in evaluated_cases if not c["uncertainty_acceptable"]]
        ),
    }

    # Stratify by symptom count
    by_symptom_count = {
        "low": compute_stratum_metrics(
            [c for c in evaluated_cases if c["symptom_category"] == "low"]
        ),
        "medium": compute_stratum_metrics(
            [c for c in evaluated_cases if c["symptom_category"] == "medium"]
        ),
        "high": compute_stratum_metrics(
            [c for c in evaluated_cases if c["symptom_category"] == "high"]
        ),
    }

    # Stratify by severity
    by_severity = {
        "critical": compute_stratum_metrics(
            [c for c in evaluated_cases if c["severity_category"] == "critical"]
        ),
        "moderate": compute_stratum_metrics(
            [c for c in evaluated_cases if c["severity_category"] == "moderate"]
        ),
        "mild": compute_stratum_metrics(
            [c for c in evaluated_cases if c["severity_category"] == "mild"]
        ),
        "unknown": compute_stratum_metrics(
            [c for c in evaluated_cases if c["severity_category"] == "unknown"]
        ),
    }

    # Statistical tests for key comparisons
    statistical_tests = {}

    # Test: escalation_required vs not
    esc_cases = [c for c in evaluated_cases if c["escalation_required"]]
    no_esc_cases = [c for c in evaluated_cases if not c["escalation_required"]]
    if esc_cases and no_esc_cases:
        statistical_tests["escalation_safety_pass"] = chi_square_test(
            [c["safety_pass"] for c in esc_cases],
            [c["safety_pass"] for c in no_esc_cases],
        )

    # Test: critical vs mild severity
    critical_cases = [c for c in evaluated_cases if c["severity_category"] == "critical"]
    mild_cases = [c for c in evaluated_cases if c["severity_category"] == "mild"]
    if critical_cases and mild_cases:
        statistical_tests["severity_safety_pass"] = chi_square_test(
            [c["safety_pass"] for c in critical_cases],
            [c["safety_pass"] for c in mild_cases],
        )

    return {
        "model_id": model_id,
        "overall": overall,
        "by_escalation": by_escalation,
        "by_uncertainty": by_uncertainty,
        "by_symptom_count": by_symptom_count,
        "by_severity": by_severity,
        "statistical_tests": statistical_tests,
        "evaluated_cases": evaluated_cases,
    }


def compute_aggregate_breakdown(all_results: Dict[str, Dict]) -> Dict[str, Any]:
    """Compute aggregate breakdown across all models."""

    # Collect all evaluated cases across models
    all_cases_by_stratum = defaultdict(list)

    for model_id, result in all_results.items():
        for case in result["evaluated_cases"]:
            # Add to appropriate strata
            esc_key = "requires_escalation" if case["escalation_required"] else "no_escalation"
            all_cases_by_stratum[f"escalation:{esc_key}"].append(case)

            unc_key = "uncertainty_acceptable" if case["uncertainty_acceptable"] else "uncertainty_not_acceptable"
            all_cases_by_stratum[f"uncertainty:{unc_key}"].append(case)

            all_cases_by_stratum[f"symptoms:{case['symptom_category']}"].append(case)
            all_cases_by_stratum[f"severity:{case['severity_category']}"].append(case)
            all_cases_by_stratum["all"].append(case)

    # Compute metrics for each stratum
    aggregate = {}
    for stratum_name, cases in all_cases_by_stratum.items():
        aggregate[stratum_name] = compute_stratum_metrics(cases)

    return aggregate


def format_pct(value: Optional[float], decimals: int = 1) -> str:
    """Format a float as a percentage string."""
    if value is None:
        return "N/A"
    return f"{value * 100:.{decimals}f}%"


def generate_markdown_tables(
    all_results: Dict[str, Dict],
    aggregate: Dict[str, Any],
) -> str:
    """Generate markdown tables for the breakdown analysis."""
    lines = [
        "# MedSafe-Dx Case Breakdown Analysis",
        "",
        "This report analyzes model performance across different case types.",
        "",
    ]

    # Table 1: Overall by Model
    lines.extend([
        "## Overall Performance by Model",
        "",
        "| Model | N | Safety Pass | Missed Esc | Overconf Wrong | Unsafe Reassure | Top-1 | Top-3 |",
        "|-------|---|-------------|------------|----------------|-----------------|-------|-------|",
    ])

    for model_id, result in sorted(all_results.items(), key=lambda x: x[0]):
        o = result["overall"]
        lines.append(
            f"| {model_id} | {o['n_cases']} | {format_pct(o['safety_pass_rate'])} | "
            f"{format_pct(o['missed_escalation_rate'])} | {format_pct(o['overconfident_wrong_rate'])} | "
            f"{format_pct(o['unsafe_reassurance_rate'])} | {format_pct(o['top1_recall'])} | "
            f"{format_pct(o['top3_recall'])} |"
        )

    lines.append("")

    # Table 2: By Escalation Required (Aggregate)
    lines.extend([
        "## Performance by Escalation Requirement (All Models)",
        "",
        "| Stratum | N | Safety Pass | Missed Esc | Top-1 | Top-3 |",
        "|---------|---|-------------|------------|-------|-------|",
    ])

    for key in ["escalation:requires_escalation", "escalation:no_escalation"]:
        if key in aggregate:
            a = aggregate[key]
            label = key.split(":")[1].replace("_", " ").title()
            lines.append(
                f"| {label} | {a['n_cases']} | {format_pct(a['safety_pass_rate'])} | "
                f"{format_pct(a['missed_escalation_rate'])} | {format_pct(a['top1_recall'])} | "
                f"{format_pct(a['top3_recall'])} |"
            )

    lines.append("")

    # Table 3: By Severity (Aggregate)
    lines.extend([
        "## Performance by Case Severity (All Models)",
        "",
        "Severity categories:",
        "- **Critical**: Severity 1-2 (life-threatening conditions)",
        "- **Moderate**: Severity 3 (serious conditions)",
        "- **Mild**: Severity 4-5 (less severe conditions)",
        "",
        "| Severity | N | Safety Pass | Missed Esc | Overconf Wrong | Top-1 | Top-3 |",
        "|----------|---|-------------|------------|----------------|-------|-------|",
    ])

    for sev in ["critical", "moderate", "mild"]:
        key = f"severity:{sev}"
        if key in aggregate:
            a = aggregate[key]
            lines.append(
                f"| {sev.title()} | {a['n_cases']} | {format_pct(a['safety_pass_rate'])} | "
                f"{format_pct(a['missed_escalation_rate'])} | {format_pct(a['overconfident_wrong_rate'])} | "
                f"{format_pct(a['top1_recall'])} | {format_pct(a['top3_recall'])} |"
            )

    lines.append("")

    # Table 4: By Symptom Count (Aggregate)
    lines.extend([
        "## Performance by Symptom Count (All Models)",
        "",
        "| Symptom Count | N | Safety Pass | Overconf Wrong | Top-1 | Top-3 |",
        "|---------------|---|-------------|----------------|-------|-------|",
    ])

    for sym in ["low", "medium", "high"]:
        key = f"symptoms:{sym}"
        if key in aggregate:
            a = aggregate[key]
            lines.append(
                f"| {sym.title()} | {a['n_cases']} | {format_pct(a['safety_pass_rate'])} | "
                f"{format_pct(a['overconfident_wrong_rate'])} | {format_pct(a['top1_recall'])} | "
                f"{format_pct(a['top3_recall'])} |"
            )

    lines.append("")

    # Table 5: By Model - Escalation Cases Only
    lines.extend([
        "## Performance on Escalation-Required Cases (by Model)",
        "",
        "| Model | N | Safety Pass | Missed Esc | Top-1 | Top-3 |",
        "|-------|---|-------------|------------|-------|-------|",
    ])

    for model_id, result in sorted(all_results.items(), key=lambda x: x[0]):
        e = result["by_escalation"]["requires_escalation"]
        if e["n_cases"] > 0:
            lines.append(
                f"| {model_id} | {e['n_cases']} | {format_pct(e['safety_pass_rate'])} | "
                f"{format_pct(e['missed_escalation_rate'])} | {format_pct(e['top1_recall'])} | "
                f"{format_pct(e['top3_recall'])} |"
            )

    lines.append("")

    # Table 6: By Model - Critical Severity Only
    lines.extend([
        "## Performance on Critical Severity Cases (by Model)",
        "",
        "| Model | N | Safety Pass | Missed Esc | Top-1 | Top-3 |",
        "|-------|---|-------------|------------|-------|-------|",
    ])

    for model_id, result in sorted(all_results.items(), key=lambda x: x[0]):
        c = result["by_severity"]["critical"]
        if c["n_cases"] > 0:
            lines.append(
                f"| {model_id} | {c['n_cases']} | {format_pct(c['safety_pass_rate'])} | "
                f"{format_pct(c['missed_escalation_rate'])} | {format_pct(c['top1_recall'])} | "
                f"{format_pct(c['top3_recall'])} |"
            )

    lines.append("")

    # Statistical significance notes
    lines.extend([
        "## Statistical Tests",
        "",
        "Chi-square tests for significant differences in safety pass rate:",
        "",
    ])

    for model_id, result in sorted(all_results.items(), key=lambda x: x[0]):
        tests = result.get("statistical_tests", {})

        esc_test = tests.get("escalation_safety_pass", {})
        if esc_test.get("p_value") is not None:
            sig = "**significant**" if esc_test["significant"] else "not significant"
            lines.append(
                f"- **{model_id}** - Escalation vs No-Escalation: "
                f"χ²={esc_test['chi2']:.2f}, p={esc_test['p_value']:.4f} ({sig})"
            )

        sev_test = tests.get("severity_safety_pass", {})
        if sev_test.get("p_value") is not None:
            sig = "**significant**" if sev_test["significant"] else "not significant"
            lines.append(
                f"- **{model_id}** - Critical vs Mild Severity: "
                f"χ²={sev_test['chi2']:.2f}, p={sev_test['p_value']:.4f} ({sig})"
            )

    lines.append("")

    return "\n".join(lines)


def main():
    """Run case breakdown analysis."""
    print("Loading test cases...")
    cases_df = load_test_cases()
    print(f"  Loaded {len(cases_df)} test cases")

    # Calculate symptom count terciles
    symptom_counts = cases_df["presenting_symptoms"].apply(len)
    terciles = (
        symptom_counts.quantile(0.33),
        symptom_counts.quantile(0.67),
    )
    print(f"  Symptom count terciles: {terciles}")

    # Load all model results
    print("\nLoading model results...")
    model_files = list_model_results()
    print(f"  Found {len(model_files)} models")

    all_results = {}
    for mf in model_files:
        model_id = mf["model_id"]
        print(f"  Analyzing {model_id}...")

        try:
            predictions_df = load_predictions(mf["predictions_path"])
            result = analyze_model_results(model_id, predictions_df, cases_df, terciles)
            all_results[model_id] = result
        except Exception as e:
            print(f"    Error: {e}")

    if not all_results:
        print("No results to analyze!")
        return

    # Compute aggregate breakdown
    print("\nComputing aggregate breakdown...")
    aggregate = compute_aggregate_breakdown(all_results)

    # Prepare output (remove evaluated_cases to reduce file size)
    output_results = {}
    for model_id, result in all_results.items():
        output_results[model_id] = {k: v for k, v in result.items() if k != "evaluated_cases"}

    output = {
        "symptom_terciles": terciles,
        "n_models": len(all_results),
        "n_cases_per_model": len(cases_df),
        "results_by_model": output_results,
        "aggregate": aggregate,
    }

    # Write JSON output
    output_path = PATHS["analysis_output_dir"] / "case_breakdown.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nWrote {output_path}")

    # Generate markdown tables
    markdown = generate_markdown_tables(all_results, aggregate)
    md_path = PATHS["analysis_output_dir"] / "case_breakdown_tables.md"
    with open(md_path, "w") as f:
        f.write(markdown)
    print(f"Wrote {md_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_agg = aggregate.get("all", {})
    print(f"\nAggregate across all models:")
    print(f"  Total evaluations: {all_agg.get('n_cases', 0)}")
    print(f"  Safety pass rate: {format_pct(all_agg.get('safety_pass_rate'))}")
    print(f"  Top-1 recall: {format_pct(all_agg.get('top1_recall'))}")
    print(f"  Top-3 recall: {format_pct(all_agg.get('top3_recall'))}")

    print("\nBy escalation requirement:")
    for key in ["escalation:requires_escalation", "escalation:no_escalation"]:
        if key in aggregate:
            a = aggregate[key]
            label = key.split(":")[1]
            print(f"  {label}: safety={format_pct(a['safety_pass_rate'])}, n={a['n_cases']}")

    print("\nBy severity:")
    for sev in ["critical", "moderate", "mild"]:
        key = f"severity:{sev}"
        if key in aggregate:
            a = aggregate[key]
            print(f"  {sev}: safety={format_pct(a['safety_pass_rate'])}, n={a['n_cases']}")


if __name__ == "__main__":
    main()
