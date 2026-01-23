#!/usr/bin/env python3
"""
Model Agreement Analysis for MedSafe-Dx.

Analyzes inter-model consistency:
- Agreement on escalation decisions
- Agreement on top-1 diagnosis
- Top-3 diagnosis overlap
- Cohen's kappa for model pairs
- Consensus failures and successes
- Potential case quality issues

Outputs:
- results/analysis/model_agreement.json
- results/analysis/model_agreement_heatmap.png
"""

import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
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


def normalize_icd10(code: str) -> str:
    """Normalize ICD-10 code for comparison."""
    return str(code).upper().replace(".", "").replace(" ", "").strip()


def normalize_escalation(esc: str) -> str:
    """Normalize escalation decision."""
    if not esc:
        return "UNKNOWN"
    e = str(esc).upper().strip()
    if "ESCALATE" in e and "NOW" in e:
        return "ESCALATE_NOW"
    elif "ROUTINE" in e:
        return "ROUTINE_CARE"
    elif "URGENT" in e:
        return "URGENT_CARE"
    return e


def cohens_kappa(labels1: List[str], labels2: List[str]) -> Optional[float]:
    """Calculate Cohen's kappa for two sets of labels."""
    if len(labels1) != len(labels2) or len(labels1) == 0:
        return None

    # Get unique labels
    all_labels = sorted(set(labels1) | set(labels2))
    n = len(labels1)

    if len(all_labels) < 2:
        return None  # Can't compute kappa with only one category

    # Build confusion matrix
    label_to_idx = {l: i for i, l in enumerate(all_labels)}
    k = len(all_labels)
    matrix = np.zeros((k, k))

    for l1, l2 in zip(labels1, labels2):
        i = label_to_idx[l1]
        j = label_to_idx[l2]
        matrix[i, j] += 1

    # Calculate observed agreement
    po = np.trace(matrix) / n

    # Calculate expected agreement
    row_sums = matrix.sum(axis=1)
    col_sums = matrix.sum(axis=0)
    pe = np.sum(row_sums * col_sums) / (n * n)

    # Cohen's kappa
    if pe == 1:
        return 1.0  # Perfect agreement by chance
    kappa = (po - pe) / (1 - pe)

    return kappa


def analyze_case_agreement(
    case_id: str,
    model_results: Dict[str, Dict],
) -> Dict[str, Any]:
    """Analyze agreement for a single case across all models."""

    escalations = []
    top1_codes = []
    top3_codes_sets = []
    safety_results = []

    for model_id, result in model_results.items():
        escalations.append(normalize_escalation(result.get("escalation", "")))

        dx = result.get("differential_diagnoses", [])
        codes = []
        for d in dx:
            if isinstance(d, dict):
                code = d.get("code", "")
            else:
                code = str(d)
            codes.append(normalize_icd10(code))

        if codes:
            top1_codes.append(codes[0])
            top3_codes_sets.append(set(codes[:3]))
        else:
            top1_codes.append("")
            top3_codes_sets.append(set())

        safety_results.append(result.get("safety_pass", False))

    n_models = len(model_results)

    # Escalation agreement
    esc_counts = defaultdict(int)
    for e in escalations:
        esc_counts[e] += 1
    most_common_esc = max(esc_counts.items(), key=lambda x: x[1])
    escalation_agreement = most_common_esc[1] / n_models

    # Top-1 agreement
    top1_counts = defaultdict(int)
    for t in top1_codes:
        if t:
            top1_counts[t] += 1
    if top1_counts:
        most_common_top1 = max(top1_counts.items(), key=lambda x: x[1])
        top1_agreement = most_common_top1[1] / n_models
    else:
        top1_agreement = 0
        most_common_top1 = ("", 0)

    # Top-3 overlap (pairwise Jaccard)
    jaccard_scores = []
    for i, j in combinations(range(len(top3_codes_sets)), 2):
        s1, s2 = top3_codes_sets[i], top3_codes_sets[j]
        if s1 or s2:
            intersection = len(s1 & s2)
            union = len(s1 | s2)
            if union > 0:
                jaccard_scores.append(intersection / union)

    avg_jaccard = np.mean(jaccard_scores) if jaccard_scores else 0

    # Safety consensus
    all_pass = all(safety_results)
    all_fail = not any(safety_results)
    safety_agreement = sum(safety_results) / n_models

    return {
        "case_id": case_id,
        "n_models": n_models,
        "escalation_agreement": escalation_agreement,
        "most_common_escalation": most_common_esc[0],
        "top1_agreement": top1_agreement,
        "most_common_top1": most_common_top1[0],
        "avg_top3_jaccard": avg_jaccard,
        "safety_agreement": safety_agreement,
        "all_pass": all_pass,
        "all_fail": all_fail,
    }


def compute_pairwise_kappa(
    all_results: Dict[str, Dict[str, Dict]],  # model_id -> case_id -> result
    metric: str = "escalation",
) -> Dict[Tuple[str, str], float]:
    """Compute Cohen's kappa for all model pairs."""
    model_ids = sorted(all_results.keys())
    kappa_matrix = {}

    for m1, m2 in combinations(model_ids, 2):
        # Collect paired labels
        labels1 = []
        labels2 = []

        for case_id in all_results[m1]:
            if case_id in all_results[m2]:
                if metric == "escalation":
                    l1 = normalize_escalation(all_results[m1][case_id].get("escalation", ""))
                    l2 = normalize_escalation(all_results[m2][case_id].get("escalation", ""))
                elif metric == "safety":
                    l1 = "PASS" if all_results[m1][case_id].get("safety_pass", False) else "FAIL"
                    l2 = "PASS" if all_results[m2][case_id].get("safety_pass", False) else "FAIL"
                else:
                    continue

                labels1.append(l1)
                labels2.append(l2)

        kappa = cohens_kappa(labels1, labels2)
        kappa_matrix[(m1, m2)] = kappa

    return kappa_matrix


def create_agreement_heatmap(
    kappa_matrix: Dict[Tuple[str, str], float],
    model_ids: List[str],
    output_path: Path,
    title: str = "Inter-Model Agreement (Cohen's Kappa)",
) -> None:
    """Create agreement heatmap."""
    if not HAS_MATPLOTLIB:
        return

    n = len(model_ids)
    matrix = np.zeros((n, n))

    # Fill diagonal with 1s (perfect self-agreement)
    np.fill_diagonal(matrix, 1.0)

    # Fill pairwise kappas
    id_to_idx = {m: i for i, m in enumerate(model_ids)}
    for (m1, m2), kappa in kappa_matrix.items():
        if kappa is not None:
            i, j = id_to_idx[m1], id_to_idx[m2]
            matrix[i, j] = kappa
            matrix[j, i] = kappa

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot heatmap
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=-0.2, vmax=1.0, aspect='auto')

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Cohen's Kappa", rotation=-90, va="bottom")

    # Set ticks
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))

    # Shorten labels
    short_labels = [m.replace("anthropic-", "").replace("openai-", "").replace("google-", "").replace("deepseek-", "")[:15] for m in model_ids]
    ax.set_xticklabels(short_labels, fontsize=8)
    ax.set_yticklabels(short_labels, fontsize=8)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    for i in range(n):
        for j in range(n):
            value = matrix[i, j]
            color = "white" if abs(value) > 0.5 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color, fontsize=8)

    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved agreement heatmap to {output_path}")


def generate_markdown_report(
    case_agreements: List[Dict],
    kappa_escalation: Dict,
    kappa_safety: Dict,
    model_ids: List[str],
) -> str:
    """Generate markdown report for model agreement analysis."""
    lines = [
        "# MedSafe-Dx Model Agreement Analysis",
        "",
        "This report analyzes inter-model consistency and agreement patterns.",
        "",
    ]

    # Summary stats
    n_cases = len(case_agreements)
    consensus_pass = sum(1 for c in case_agreements if c["all_pass"])
    consensus_fail = sum(1 for c in case_agreements if c["all_fail"])
    avg_esc_agreement = np.mean([c["escalation_agreement"] for c in case_agreements])
    avg_top1_agreement = np.mean([c["top1_agreement"] for c in case_agreements])
    avg_jaccard = np.mean([c["avg_top3_jaccard"] for c in case_agreements])

    lines.extend([
        "## Summary Statistics",
        "",
        f"- Total cases analyzed: {n_cases}",
        f"- Models compared: {len(model_ids)}",
        f"- **Consensus passes** (all models pass): {consensus_pass} ({consensus_pass/n_cases*100:.1f}%)",
        f"- **Consensus failures** (all models fail): {consensus_fail} ({consensus_fail/n_cases*100:.1f}%)",
        "",
        "### Average Agreement Metrics",
        "",
        f"- Escalation decision agreement: {avg_esc_agreement*100:.1f}%",
        f"- Top-1 diagnosis agreement: {avg_top1_agreement*100:.1f}%",
        f"- Top-3 diagnosis overlap (Jaccard): {avg_jaccard*100:.1f}%",
        "",
    ])

    # Cohen's Kappa tables
    lines.extend([
        "## Cohen's Kappa: Escalation Decisions",
        "",
        "Kappa interpretation: <0 = poor, 0-0.2 = slight, 0.2-0.4 = fair, 0.4-0.6 = moderate, 0.6-0.8 = substantial, >0.8 = almost perfect",
        "",
    ])

    # Create kappa table
    lines.append("| Model Pair | Kappa | Interpretation |")
    lines.append("|------------|-------|----------------|")

    for (m1, m2), kappa in sorted(kappa_escalation.items(), key=lambda x: x[1] or -2, reverse=True):
        if kappa is None:
            interp = "N/A"
            k_str = "N/A"
        elif kappa < 0:
            interp = "Poor"
            k_str = f"{kappa:.3f}"
        elif kappa < 0.2:
            interp = "Slight"
            k_str = f"{kappa:.3f}"
        elif kappa < 0.4:
            interp = "Fair"
            k_str = f"{kappa:.3f}"
        elif kappa < 0.6:
            interp = "Moderate"
            k_str = f"{kappa:.3f}"
        elif kappa < 0.8:
            interp = "Substantial"
            k_str = f"{kappa:.3f}"
        else:
            interp = "Almost Perfect"
            k_str = f"{kappa:.3f}"

        short_m1 = m1.replace("anthropic-", "").replace("openai-", "").replace("google-", "").replace("deepseek-", "")
        short_m2 = m2.replace("anthropic-", "").replace("openai-", "").replace("google-", "").replace("deepseek-", "")
        lines.append(f"| {short_m1} vs {short_m2} | {k_str} | {interp} |")

    lines.append("")

    # Cases with high disagreement
    lines.extend([
        "## Cases with Low Agreement",
        "",
        "Cases where models significantly disagree (escalation agreement < 60%):",
        "",
        "| Case ID | Esc Agreement | Top-1 Agreement | Most Common Esc |",
        "|---------|---------------|-----------------|-----------------|",
    ])

    low_agreement = [c for c in case_agreements if c["escalation_agreement"] < 0.6]
    for c in sorted(low_agreement, key=lambda x: x["escalation_agreement"])[:20]:
        lines.append(
            f"| {c['case_id']} | {c['escalation_agreement']*100:.0f}% | "
            f"{c['top1_agreement']*100:.0f}% | {c['most_common_escalation']} |"
        )

    lines.append("")

    # Consensus failures
    lines.extend([
        "## Consensus Failures",
        "",
        "Cases where ALL models fail safety checks (potential difficult cases):",
        "",
    ])

    consensus_fails = [c for c in case_agreements if c["all_fail"]]
    if consensus_fails:
        for c in consensus_fails:
            lines.append(f"- {c['case_id']}")
    else:
        lines.append("*No consensus failures found*")

    lines.append("")

    # Consensus successes
    lines.extend([
        "## Consensus Successes",
        "",
        f"Cases where ALL models pass safety checks: {consensus_pass}",
        "",
    ])

    if consensus_pass <= 20:
        consensus_passes = [c for c in case_agreements if c["all_pass"]]
        for c in consensus_passes:
            lines.append(f"- {c['case_id']}")
    else:
        lines.append(f"*{consensus_pass} cases (not listed)*")

    lines.append("")

    # Potential case quality issues
    lines.extend([
        "## Potential Case Quality Issues",
        "",
        "Cases where models strongly agree but all fail may indicate:",
        "- Ambiguous gold standard",
        "- Particularly difficult presentations",
        "- Potential labeling issues",
        "",
    ])

    high_agree_fail = [
        c for c in case_agreements
        if c["escalation_agreement"] >= 0.8 and c["safety_agreement"] < 0.3
    ]
    if high_agree_fail:
        lines.append("| Case ID | Esc Agreement | Safety Rate | Most Common Esc |")
        lines.append("|---------|---------------|-------------|-----------------|")
        for c in high_agree_fail:
            lines.append(
                f"| {c['case_id']} | {c['escalation_agreement']*100:.0f}% | "
                f"{c['safety_agreement']*100:.0f}% | {c['most_common_escalation']} |"
            )
    else:
        lines.append("*No potential issues identified*")

    lines.append("")

    return "\n".join(lines)


def main():
    """Run model agreement analysis."""
    print("Loading test cases...")
    cases_df = load_test_cases()
    print(f"  Loaded {len(cases_df)} test cases")

    # Load all model results
    print("\nLoading model results...")
    model_files = list_model_results()
    print(f"  Found {len(model_files)} models")

    # Collect results by model and case
    all_results = {}  # model_id -> case_id -> result

    for mf in model_files:
        model_id = mf["model_id"]
        print(f"  Loading {model_id}...")

        try:
            predictions_df = load_predictions(mf["predictions_path"])
            merged = cases_df.merge(predictions_df, on="case_id", how="inner")

            model_results = {}
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

                model_results[row["case_id"]] = {
                    "differential_diagnoses": row.get("differential_diagnoses", []),
                    "escalation": row.get("escalation_decision", ""),
                    "safety_pass": eval_result["safety_pass"],
                }

            all_results[model_id] = model_results

        except Exception as e:
            print(f"    Error: {e}")

    if not all_results:
        print("No results to analyze!")
        return

    model_ids = sorted(all_results.keys())

    # Find common cases across all models
    common_cases = set(cases_df["case_id"])
    for model_id, results in all_results.items():
        common_cases &= set(results.keys())
    print(f"\nCommon cases across all models: {len(common_cases)}")

    # Analyze per-case agreement
    print("\nAnalyzing case-level agreement...")
    case_agreements = []

    for case_id in common_cases:
        model_case_results = {mid: all_results[mid][case_id] for mid in model_ids}
        agreement = analyze_case_agreement(case_id, model_case_results)
        case_agreements.append(agreement)

    # Compute pairwise kappa
    print("Computing pairwise Cohen's kappa...")
    kappa_escalation = compute_pairwise_kappa(all_results, "escalation")
    kappa_safety = compute_pairwise_kappa(all_results, "safety")

    # Prepare output
    output = {
        "n_models": len(model_ids),
        "n_common_cases": len(common_cases),
        "model_ids": model_ids,
        "case_agreements": case_agreements,
        "kappa_escalation": {f"{k[0]}|{k[1]}": v for k, v in kappa_escalation.items()},
        "kappa_safety": {f"{k[0]}|{k[1]}": v for k, v in kappa_safety.items()},
    }

    # Write JSON output
    output_path = PATHS["analysis_output_dir"] / "model_agreement.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nWrote {output_path}")

    # Generate markdown report
    markdown = generate_markdown_report(case_agreements, kappa_escalation, kappa_safety, model_ids)
    md_path = PATHS["analysis_output_dir"] / "model_agreement_analysis.md"
    with open(md_path, "w") as f:
        f.write(markdown)
    print(f"Wrote {md_path}")

    # Create heatmaps
    if HAS_MATPLOTLIB:
        heatmap_path = PATHS["analysis_output_dir"] / "model_agreement_heatmap.png"
        create_agreement_heatmap(
            kappa_escalation,
            model_ids,
            heatmap_path,
            "Inter-Model Agreement: Escalation Decisions (Cohen's Kappa)"
        )

    # Print summary
    print("\n" + "=" * 60)
    print("MODEL AGREEMENT SUMMARY")
    print("=" * 60)

    consensus_pass = sum(1 for c in case_agreements if c["all_pass"])
    consensus_fail = sum(1 for c in case_agreements if c["all_fail"])
    avg_esc = np.mean([c["escalation_agreement"] for c in case_agreements])

    print(f"\nCommon cases: {len(common_cases)}")
    print(f"Consensus passes (all models pass): {consensus_pass}")
    print(f"Consensus failures (all models fail): {consensus_fail}")
    print(f"Average escalation agreement: {avg_esc*100:.1f}%")

    print("\nTop 5 most agreed model pairs (escalation kappa):")
    sorted_kappa = sorted(kappa_escalation.items(), key=lambda x: x[1] or -2, reverse=True)
    for (m1, m2), k in sorted_kappa[:5]:
        short_m1 = m1.replace("anthropic-", "").replace("openai-", "").replace("google-", "").replace("deepseek-", "")
        short_m2 = m2.replace("anthropic-", "").replace("openai-", "").replace("google-", "").replace("deepseek-", "")
        print(f"  {short_m1} vs {short_m2}: {k:.3f}" if k else f"  {short_m1} vs {short_m2}: N/A")


if __name__ == "__main__":
    main()
