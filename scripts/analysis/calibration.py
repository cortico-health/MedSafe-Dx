#!/usr/bin/env python3
"""
Calibration Analysis for MedSafe-Dx.

Assesses model uncertainty calibration:
- Group by uncertainty (CONFIDENT vs UNCERTAIN)
- Calculate accuracy in each group
- Expected Calibration Error (ECE)
- Generate calibration comparison charts

Outputs:
- results/analysis/calibration_results.json
- results/analysis/calibration_curves.png
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping visualizations")

from .config import PATHS, MODELS
from .utils import (
    load_test_cases,
    load_predictions,
    list_model_results,
    evaluate_case,
)


def normalize_uncertainty(uncertainty: str) -> str:
    """Normalize uncertainty value to standard form."""
    if not uncertainty:
        return "UNKNOWN"
    u = str(uncertainty).upper().strip()
    if "CONFIDENT" in u and "UNCERTAIN" not in u:
        return "CONFIDENT"
    elif "UNCERTAIN" in u:
        return "UNCERTAIN"
    else:
        return u


def analyze_model_calibration(
    predictions_df: pd.DataFrame,
    cases_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Analyze calibration for a single model."""

    merged = cases_df.merge(predictions_df, on="case_id", how="inner")

    # Categorize by uncertainty
    confident_cases = []
    uncertain_cases = []
    unknown_cases = []

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

        uncertainty = normalize_uncertainty(row.get("uncertainty", ""))

        case_result = {
            "case_id": row["case_id"],
            "uncertainty": uncertainty,
            "top1_match": eval_result["top1_match"],
            "top3_match": eval_result["top3_match"],
            "safety_pass": eval_result["safety_pass"],
        }

        if uncertainty == "CONFIDENT":
            confident_cases.append(case_result)
        elif uncertainty == "UNCERTAIN":
            uncertain_cases.append(case_result)
        else:
            unknown_cases.append(case_result)

    # Calculate metrics for each group
    def calc_group_metrics(cases: List[Dict]) -> Dict:
        if not cases:
            return {
                "n": 0,
                "top1_accuracy": None,
                "top3_accuracy": None,
                "safety_pass_rate": None,
            }
        n = len(cases)
        return {
            "n": n,
            "top1_accuracy": sum(1 for c in cases if c["top1_match"]) / n,
            "top3_accuracy": sum(1 for c in cases if c["top3_match"]) / n,
            "safety_pass_rate": sum(1 for c in cases if c["safety_pass"]) / n,
        }

    confident_metrics = calc_group_metrics(confident_cases)
    uncertain_metrics = calc_group_metrics(uncertain_cases)
    unknown_metrics = calc_group_metrics(unknown_cases)

    # Calculate Expected Calibration Error (ECE)
    # For binary calibration: |confident_accuracy - expected_confident_accuracy|
    # Ideally, confident predictions should have higher accuracy than uncertain ones
    ece = None
    calibration_gap = None

    if confident_metrics["n"] > 0 and uncertain_metrics["n"] > 0:
        # The "gap" between confident and uncertain accuracy
        # Positive means well-calibrated (confident > uncertain in accuracy)
        # Negative means poorly calibrated (uncertain cases are more accurate)
        calibration_gap = (
            confident_metrics["top3_accuracy"] - uncertain_metrics["top3_accuracy"]
        )

        # ECE: weighted average of miscalibration
        # For simplicity with binary confidence, we measure how far from ideal
        # Ideal: confident = 1.0, uncertain = 0.0 (extreme)
        # More realistic ideal: confident should be 20%+ higher than uncertain
        total = confident_metrics["n"] + uncertain_metrics["n"]
        w_conf = confident_metrics["n"] / total
        w_unc = uncertain_metrics["n"] / total

        # Expected accuracy for confident should be high (e.g., 0.8)
        # Expected accuracy for uncertain should be lower (e.g., 0.4)
        # This is a simplified ECE for binary confidence
        ideal_gap = 0.3  # Confident should be 30% more accurate than uncertain
        ece = abs(calibration_gap - ideal_gap)

    # Calculate confidence ratio (what % of predictions are confident)
    total_known = confident_metrics["n"] + uncertain_metrics["n"]
    confidence_ratio = confident_metrics["n"] / total_known if total_known > 0 else None

    return {
        "confident": confident_metrics,
        "uncertain": uncertain_metrics,
        "unknown": unknown_metrics,
        "calibration_gap": calibration_gap,
        "ece": ece,
        "confidence_ratio": confidence_ratio,
        "total_cases": len(merged),
    }


def create_calibration_chart(
    all_results: Dict[str, Dict],
    output_path: Path,
) -> None:
    """Create calibration comparison chart."""
    if not HAS_MATPLOTLIB:
        return

    # Prepare data
    models = []
    confident_acc = []
    uncertain_acc = []
    confidence_ratios = []

    for model_id, result in sorted(all_results.items()):
        if result["confident"]["n"] > 0 and result["uncertain"]["n"] > 0:
            models.append(model_id.replace("-", "\n", 1))  # Wrap long names
            confident_acc.append(result["confident"]["top3_accuracy"] * 100)
            uncertain_acc.append(result["uncertain"]["top3_accuracy"] * 100)
            confidence_ratios.append(result["confidence_ratio"] * 100)

    if not models:
        print("Not enough data for calibration chart")
        return

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Confident vs Uncertain accuracy
    x = np.arange(len(models))
    width = 0.35

    bars1 = ax1.bar(x - width/2, confident_acc, width, label='Confident', color='#2ecc71', alpha=0.8)
    bars2 = ax1.bar(x + width/2, uncertain_acc, width, label='Uncertain', color='#e74c3c', alpha=0.8)

    ax1.set_ylabel('Top-3 Accuracy (%)')
    ax1.set_title('Accuracy by Confidence Level')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=8)
    ax1.legend()
    ax1.set_ylim(0, 100)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.0f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=7)

    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.0f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=7)

    # Plot 2: Calibration gap vs confidence ratio
    calibration_gaps = [
        all_results[m.replace("\n", "-", 1)]["calibration_gap"] * 100
        for m in models
    ]

    colors = ['#2ecc71' if gap > 0 else '#e74c3c' for gap in calibration_gaps]

    ax2.bar(x, calibration_gaps, color=colors, alpha=0.8)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.axhline(y=30, color='gray', linestyle='--', linewidth=0.5, label='Ideal gap (30%)')
    ax2.set_ylabel('Calibration Gap (Confident - Uncertain) %')
    ax2.set_title('Calibration Gap\n(Positive = Well-calibrated)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=8)
    ax2.legend()

    # Add confidence ratio as text
    for i, (gap, ratio) in enumerate(zip(calibration_gaps, confidence_ratios)):
        y_pos = gap + (3 if gap >= 0 else -8)
        ax2.annotate(f'{ratio:.0f}% conf',
                    xy=(i, gap),
                    xytext=(0, y_pos - gap), textcoords="offset points",
                    ha='center', va='bottom', fontsize=7, color='gray')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved calibration chart to {output_path}")


def generate_markdown_report(all_results: Dict[str, Dict]) -> str:
    """Generate markdown report for calibration analysis."""
    lines = [
        "# MedSafe-Dx Calibration Analysis",
        "",
        "This report assesses how well models' uncertainty estimates align with actual accuracy.",
        "",
        "## Key Metrics",
        "",
        "- **Calibration Gap**: (Confident accuracy) - (Uncertain accuracy)",
        "  - Positive = well-calibrated (confident predictions are more accurate)",
        "  - Negative = poorly calibrated (uncertain predictions are more accurate)",
        "- **Confidence Ratio**: % of predictions marked as CONFIDENT",
        "- **ECE**: Expected Calibration Error (lower is better)",
        "",
        "## Model Calibration Summary",
        "",
        "| Model | Confident N | Confident Acc | Uncertain N | Uncertain Acc | Gap | Conf Ratio |",
        "|-------|-------------|---------------|-------------|---------------|-----|------------|",
    ]

    for model_id, result in sorted(all_results.items()):
        conf = result["confident"]
        unc = result["uncertain"]

        conf_acc = f"{conf['top3_accuracy']*100:.1f}%" if conf["n"] > 0 else "N/A"
        unc_acc = f"{unc['top3_accuracy']*100:.1f}%" if unc["n"] > 0 else "N/A"
        gap = f"{result['calibration_gap']*100:+.1f}%" if result["calibration_gap"] is not None else "N/A"
        ratio = f"{result['confidence_ratio']*100:.0f}%" if result["confidence_ratio"] is not None else "N/A"

        lines.append(f"| {model_id} | {conf['n']} | {conf_acc} | {unc['n']} | {unc_acc} | {gap} | {ratio} |")

    lines.append("")

    # Safety analysis by confidence
    lines.extend([
        "## Safety Pass Rate by Confidence Level",
        "",
        "| Model | Confident Safety | Uncertain Safety |",
        "|-------|------------------|------------------|",
    ])

    for model_id, result in sorted(all_results.items()):
        conf = result["confident"]
        unc = result["uncertain"]

        conf_safety = f"{conf['safety_pass_rate']*100:.1f}%" if conf["n"] > 0 else "N/A"
        unc_safety = f"{unc['safety_pass_rate']*100:.1f}%" if unc["n"] > 0 else "N/A"

        lines.append(f"| {model_id} | {conf_safety} | {unc_safety} |")

    lines.append("")

    # Interpretation
    lines.extend([
        "## Interpretation",
        "",
        "### Well-Calibrated Models",
        "",
        "Models with positive calibration gap (confident predictions are more accurate):",
        "",
    ])

    well_calibrated = [
        (mid, r["calibration_gap"])
        for mid, r in all_results.items()
        if r["calibration_gap"] is not None and r["calibration_gap"] > 0
    ]
    for mid, gap in sorted(well_calibrated, key=lambda x: x[1], reverse=True):
        lines.append(f"- **{mid}**: +{gap*100:.1f}% gap")

    lines.extend([
        "",
        "### Poorly-Calibrated Models",
        "",
        "Models with negative calibration gap (uncertain predictions are more accurate):",
        "",
    ])

    poor_calibrated = [
        (mid, r["calibration_gap"])
        for mid, r in all_results.items()
        if r["calibration_gap"] is not None and r["calibration_gap"] <= 0
    ]
    for mid, gap in sorted(poor_calibrated, key=lambda x: x[1]):
        lines.append(f"- **{mid}**: {gap*100:.1f}% gap")

    lines.append("")

    # Over-confident models
    lines.extend([
        "### Confidence Distribution",
        "",
        "Models that mark most predictions as CONFIDENT (may indicate over-confidence):",
        "",
    ])

    high_confidence = [
        (mid, r["confidence_ratio"])
        for mid, r in all_results.items()
        if r["confidence_ratio"] is not None and r["confidence_ratio"] > 0.8
    ]
    for mid, ratio in sorted(high_confidence, key=lambda x: x[1], reverse=True):
        lines.append(f"- **{mid}**: {ratio*100:.0f}% confident")

    lines.append("")

    return "\n".join(lines)


def main():
    """Run calibration analysis."""
    print("Loading test cases...")
    cases_df = load_test_cases()
    print(f"  Loaded {len(cases_df)} test cases")

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
            result = analyze_model_calibration(predictions_df, cases_df)
            all_results[model_id] = result
        except Exception as e:
            print(f"    Error: {e}")

    if not all_results:
        print("No results to analyze!")
        return

    # Write JSON output
    output_path = PATHS["analysis_output_dir"] / "calibration_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nWrote {output_path}")

    # Generate markdown report
    markdown = generate_markdown_report(all_results)
    md_path = PATHS["analysis_output_dir"] / "calibration_analysis.md"
    with open(md_path, "w") as f:
        f.write(markdown)
    print(f"Wrote {md_path}")

    # Create calibration chart
    if HAS_MATPLOTLIB:
        chart_path = PATHS["analysis_output_dir"] / "calibration_curves.png"
        create_calibration_chart(all_results, chart_path)

    # Print summary
    print("\n" + "=" * 60)
    print("CALIBRATION SUMMARY")
    print("=" * 60)

    print("\nCalibration by model (Top-3 Accuracy):")
    print("-" * 60)

    for model_id, result in sorted(all_results.items()):
        conf = result["confident"]
        unc = result["uncertain"]
        gap = result["calibration_gap"]

        if conf["n"] > 0 and unc["n"] > 0:
            status = "GOOD" if gap and gap > 0 else "POOR"
            print(f"  {model_id}:")
            print(f"    Confident: {conf['top3_accuracy']*100:.1f}% (n={conf['n']})")
            print(f"    Uncertain: {unc['top3_accuracy']*100:.1f}% (n={unc['n']})")
            print(f"    Gap: {gap*100:+.1f}% [{status}]")


if __name__ == "__main__":
    main()
