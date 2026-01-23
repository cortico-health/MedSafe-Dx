#!/usr/bin/env python3
"""
Response Length Analysis for MedSafe-Dx.

Checks for length bias in model responses:
- Calculate response length (chars, tokens) per prediction
- Correlate with safety pass/fail, top-k recall, individual failure types
- Report Pearson r values
- Compare patterns across models

Outputs:
- results/analysis/response_length_analysis.json
- results/analysis/response_length_analysis.md
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from scipy import stats

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


def estimate_tokens(text: str) -> int:
    """Rough token estimate (4 chars per token average)."""
    if not text:
        return 0
    return len(text) // 4


def calculate_response_length(prediction: Dict) -> Dict[str, int]:
    """Calculate response length metrics for a prediction."""
    # Get reasoning text
    reasoning = str(prediction.get("reasoning", ""))
    reasoning_chars = len(reasoning)
    reasoning_tokens = estimate_tokens(reasoning)

    # Get total response
    total_text = reasoning

    # Add diagnoses
    dx = prediction.get("differential_diagnoses", [])
    for d in dx:
        if isinstance(d, dict):
            total_text += " " + str(d.get("code", ""))
            total_text += " " + str(d.get("description", ""))
        else:
            total_text += " " + str(d)

    # Add escalation and uncertainty
    total_text += " " + str(prediction.get("escalation_decision", ""))
    total_text += " " + str(prediction.get("uncertainty", ""))

    total_chars = len(total_text)
    total_tokens = estimate_tokens(total_text)

    return {
        "reasoning_chars": reasoning_chars,
        "reasoning_tokens": reasoning_tokens,
        "total_chars": total_chars,
        "total_tokens": total_tokens,
    }


def pearson_correlation(x: List[float], y: List[float]) -> Tuple[float, float]:
    """Calculate Pearson correlation coefficient and p-value."""
    if len(x) < 3 or len(y) < 3:
        return (None, None)

    # Remove any NaN values
    valid = [(xi, yi) for xi, yi in zip(x, y) if xi is not None and yi is not None]
    if len(valid) < 3:
        return (None, None)

    x_clean, y_clean = zip(*valid)
    try:
        r, p = stats.pearsonr(x_clean, y_clean)
        return (r, p)
    except Exception:
        return (None, None)


def point_biserial_correlation(continuous: List[float], binary: List[bool]) -> Tuple[float, float]:
    """Calculate point-biserial correlation for continuous vs binary variable."""
    if len(continuous) < 3 or len(binary) < 3:
        return (None, None)

    # Convert binary to 0/1
    binary_numeric = [1 if b else 0 for b in binary]

    return pearson_correlation(continuous, binary_numeric)


def analyze_model_length(
    model_id: str,
    predictions_df: pd.DataFrame,
    cases_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Analyze response length patterns for a single model."""

    merged = cases_df.merge(predictions_df, on="case_id", how="inner")

    # Calculate lengths and evaluate each case
    results = []
    for _, row in merged.iterrows():
        prediction = {
            "differential_diagnoses": row.get("differential_diagnoses", []),
            "escalation_decision": row.get("escalation_decision", ""),
            "uncertainty": row.get("uncertainty", ""),
            "reasoning": row.get("reasoning", ""),
        }
        gold = {
            "gold_top3": row.get("gold_top3", []),
            "escalation_required": row.get("escalation_required", False),
            "uncertainty_acceptable": row.get("uncertainty_acceptable", False),
        }

        eval_result = evaluate_case(prediction, gold)
        length_metrics = calculate_response_length(prediction)

        results.append({
            "case_id": row["case_id"],
            **length_metrics,
            **eval_result,
        })

    if not results:
        return {"model_id": model_id, "n_cases": 0}

    # Calculate summary statistics
    total_chars = [r["total_chars"] for r in results]
    reasoning_chars = [r["reasoning_chars"] for r in results]

    length_stats = {
        "mean_total_chars": np.mean(total_chars),
        "median_total_chars": np.median(total_chars),
        "std_total_chars": np.std(total_chars),
        "min_total_chars": min(total_chars),
        "max_total_chars": max(total_chars),
        "mean_reasoning_chars": np.mean(reasoning_chars),
    }

    # Correlations with safety/performance
    safety_pass = [r["safety_pass"] for r in results]
    top1_match = [r["top1_match"] for r in results]
    top3_match = [r["top3_match"] for r in results]
    missed_esc = [r["missed_escalation"] for r in results]
    overconf = [r["overconfident_wrong"] for r in results]

    correlations = {}

    # Length vs safety pass (point-biserial)
    r, p = point_biserial_correlation(total_chars, safety_pass)
    correlations["length_vs_safety_pass"] = {"r": r, "p": p}

    # Length vs top-1 match
    r, p = point_biserial_correlation(total_chars, top1_match)
    correlations["length_vs_top1_match"] = {"r": r, "p": p}

    # Length vs top-3 match
    r, p = point_biserial_correlation(total_chars, top3_match)
    correlations["length_vs_top3_match"] = {"r": r, "p": p}

    # Length vs missed escalation
    r, p = point_biserial_correlation(total_chars, missed_esc)
    correlations["length_vs_missed_escalation"] = {"r": r, "p": p}

    # Length vs overconfident wrong
    r, p = point_biserial_correlation(total_chars, overconf)
    correlations["length_vs_overconfident_wrong"] = {"r": r, "p": p}

    # Reasoning length vs total length
    r, p = pearson_correlation(reasoning_chars, total_chars)
    correlations["reasoning_vs_total_length"] = {"r": r, "p": p}

    # Compare length in pass vs fail cases
    pass_lengths = [r["total_chars"] for r in results if r["safety_pass"]]
    fail_lengths = [r["total_chars"] for r in results if not r["safety_pass"]]

    length_by_outcome = {
        "pass_mean": np.mean(pass_lengths) if pass_lengths else None,
        "pass_n": len(pass_lengths),
        "fail_mean": np.mean(fail_lengths) if fail_lengths else None,
        "fail_n": len(fail_lengths),
    }

    # T-test for difference
    if len(pass_lengths) >= 2 and len(fail_lengths) >= 2:
        try:
            t_stat, t_p = stats.ttest_ind(pass_lengths, fail_lengths)
            length_by_outcome["t_statistic"] = t_stat
            length_by_outcome["t_pvalue"] = t_p
        except Exception:
            pass

    return {
        "model_id": model_id,
        "n_cases": len(results),
        "length_stats": length_stats,
        "correlations": correlations,
        "length_by_outcome": length_by_outcome,
    }


def create_length_distribution_figure(
    all_results: Dict[str, Dict],
    output_path: Path,
) -> None:
    """Create response length distribution figure."""
    if not HAS_MATPLOTLIB:
        return

    models = sorted(all_results.keys())
    n_models = len(models)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Mean length by model
    ax1 = axes[0]
    means = [all_results[m]["length_stats"]["mean_total_chars"] for m in models]
    stds = [all_results[m]["length_stats"]["std_total_chars"] for m in models]

    short_names = [m.replace("anthropic-", "").replace("openai-", "").replace("google-", "").replace("deepseek-", "")[:15] for m in models]

    bars = ax1.bar(range(n_models), means, yerr=stds, capsize=3, alpha=0.8, color='#3498db')
    ax1.set_xticks(range(n_models))
    ax1.set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('Response Length (characters)')
    ax1.set_title('Mean Response Length by Model')
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Length vs safety correlation
    ax2 = axes[1]
    correlations = []
    for m in models:
        r = all_results[m]["correlations"]["length_vs_safety_pass"]["r"]
        correlations.append(r if r is not None else 0)

    colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in correlations]
    bars = ax2.bar(range(n_models), correlations, color=colors, alpha=0.8)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xticks(range(n_models))
    ax2.set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel("Correlation (r)")
    ax2.set_title('Length vs Safety Pass Correlation\n(Positive = longer responses safer)')
    ax2.set_ylim(-0.5, 0.5)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved length distribution figure to {output_path}")


def generate_markdown_report(all_results: Dict[str, Dict]) -> str:
    """Generate markdown report for response length analysis."""
    lines = [
        "# MedSafe-Dx Response Length Analysis",
        "",
        "This report analyzes potential length bias in model responses.",
        "",
        "## Summary Statistics",
        "",
        "| Model | Mean Chars | Median | Std Dev | Min | Max |",
        "|-------|------------|--------|---------|-----|-----|",
    ]

    for model_id, result in sorted(all_results.items()):
        s = result["length_stats"]
        lines.append(
            f"| {model_id} | {s['mean_total_chars']:.0f} | {s['median_total_chars']:.0f} | "
            f"{s['std_total_chars']:.0f} | {s['min_total_chars']} | {s['max_total_chars']} |"
        )

    lines.append("")

    # Correlations
    lines.extend([
        "## Length vs Performance Correlations",
        "",
        "Point-biserial correlations between response length and outcomes.",
        "",
        "| Model | vs Safety Pass | vs Top-1 | vs Top-3 | vs Missed Esc | vs Overconf |",
        "|-------|----------------|----------|----------|---------------|-------------|",
    ])

    for model_id, result in sorted(all_results.items()):
        c = result["correlations"]

        def fmt_corr(corr_data):
            r = corr_data.get("r")
            p = corr_data.get("p")
            if r is None:
                return "N/A"
            sig = "*" if p and p < 0.05 else ""
            return f"{r:+.3f}{sig}"

        lines.append(
            f"| {model_id} | {fmt_corr(c['length_vs_safety_pass'])} | "
            f"{fmt_corr(c['length_vs_top1_match'])} | {fmt_corr(c['length_vs_top3_match'])} | "
            f"{fmt_corr(c['length_vs_missed_escalation'])} | {fmt_corr(c['length_vs_overconfident_wrong'])} |"
        )

    lines.extend([
        "",
        "*Asterisk indicates p < 0.05*",
        "",
    ])

    # Length by outcome
    lines.extend([
        "## Response Length by Safety Outcome",
        "",
        "| Model | Pass Mean (n) | Fail Mean (n) | t-statistic | p-value |",
        "|-------|---------------|---------------|-------------|---------|",
    ])

    for model_id, result in sorted(all_results.items()):
        o = result["length_by_outcome"]
        pass_str = f"{o['pass_mean']:.0f} ({o['pass_n']})" if o.get("pass_mean") else "N/A"
        fail_str = f"{o['fail_mean']:.0f} ({o['fail_n']})" if o.get("fail_mean") else "N/A"
        t_str = f"{o['t_statistic']:.2f}" if o.get("t_statistic") else "N/A"
        p_str = f"{o['t_pvalue']:.4f}" if o.get("t_pvalue") else "N/A"

        lines.append(f"| {model_id} | {pass_str} | {fail_str} | {t_str} | {p_str} |")

    lines.append("")

    # Interpretation
    lines.extend([
        "## Interpretation",
        "",
        "### Key Findings",
        "",
    ])

    # Find significant correlations
    sig_positive = []
    sig_negative = []

    for model_id, result in all_results.items():
        c = result["correlations"]["length_vs_safety_pass"]
        if c.get("r") and c.get("p") and c["p"] < 0.05:
            if c["r"] > 0:
                sig_positive.append((model_id, c["r"]))
            else:
                sig_negative.append((model_id, c["r"]))

    if sig_positive:
        lines.append("**Models where longer responses are significantly safer:**")
        for m, r in sorted(sig_positive, key=lambda x: x[1], reverse=True):
            lines.append(f"- {m}: r = {r:+.3f}")
        lines.append("")

    if sig_negative:
        lines.append("**Models where shorter responses are significantly safer:**")
        for m, r in sorted(sig_negative, key=lambda x: x[1]):
            lines.append(f"- {m}: r = {r:+.3f}")
        lines.append("")

    if not sig_positive and not sig_negative:
        lines.append("*No significant correlations between length and safety pass rate found.*")
        lines.append("")

    lines.extend([
        "### Comparison to HealthBench",
        "",
        "HealthBench found that longer responses correlated with better performance ",
        "for some evaluations but not others. The relationship between response ",
        "length and safety/accuracy is task and model dependent.",
        "",
    ])

    return "\n".join(lines)


def main():
    """Run response length analysis."""
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
            result = analyze_model_length(model_id, predictions_df, cases_df)
            all_results[model_id] = result
        except Exception as e:
            print(f"    Error: {e}")

    if not all_results:
        print("No results to analyze!")
        return

    # Write JSON output
    output_path = PATHS["analysis_output_dir"] / "response_length_analysis.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nWrote {output_path}")

    # Generate markdown report
    markdown = generate_markdown_report(all_results)
    md_path = PATHS["analysis_output_dir"] / "response_length_analysis.md"
    with open(md_path, "w") as f:
        f.write(markdown)
    print(f"Wrote {md_path}")

    # Create figure
    if HAS_MATPLOTLIB:
        fig_path = PATHS["analysis_output_dir"] / "response_length_figure.png"
        create_length_distribution_figure(all_results, fig_path)

    # Print summary
    print("\n" + "=" * 60)
    print("RESPONSE LENGTH SUMMARY")
    print("=" * 60)

    print("\nMean response length by model:")
    for model_id, result in sorted(all_results.items(), key=lambda x: x[1]["length_stats"]["mean_total_chars"]):
        mean = result["length_stats"]["mean_total_chars"]
        r = result["correlations"]["length_vs_safety_pass"]["r"]
        r_str = f"{r:+.3f}" if r else "N/A"
        print(f"  {model_id}: {mean:.0f} chars (r={r_str} with safety)")


if __name__ == "__main__":
    main()
