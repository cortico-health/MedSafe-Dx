#!/usr/bin/env python3
"""
Worst-at-k Reliability Analysis for MedSafe-Dx.

Shows how safety reliability degrades with more samples per case.
Like HealthBench Figure 4 - if you sample k responses per case,
what's the probability of seeing at least one safety failure?

Uses run variability data to compute worst-case safety at different k.

Outputs:
- results/analysis/worst_at_k_data.json
- results/analysis/worst_at_k_figure.png
- results/analysis/worst_at_k_report.md
"""

import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple
from itertools import combinations
import random

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.analysis.config import PATHS
from scripts.analysis.utils import load_test_cases, evaluate_case

# Input/output paths
VARIABILITY_DIR = PATHS["analysis_output_dir"] / "run_variability"
OUTPUT_DIR = PATHS["analysis_output_dir"]

# K values to test
K_VALUES = [1, 2, 4, 8, 16]


def load_variability_predictions(model: str) -> Dict[str, List[Dict]]:
    """
    Load all run predictions for a model, organized by case_id.

    Returns: {case_id: [prediction_run1, prediction_run2, ...]}
    """
    model_safe_name = model.replace("/", "-")
    case_predictions = {}

    run_files = sorted(VARIABILITY_DIR.glob(f"{model_safe_name}-run*.json"))

    if not run_files:
        print(f"No run files found for {model}")
        return {}

    for run_file in run_files:
        with open(run_file) as f:
            data = json.load(f)

        predictions = data.get("predictions", data)

        for pred in predictions:
            case_id = pred.get("case_id")
            if case_id:
                if case_id not in case_predictions:
                    case_predictions[case_id] = []
                case_predictions[case_id].append(pred)

    return case_predictions


def evaluate_predictions_for_case(
    predictions: List[Dict],
    gold: Dict,
) -> List[bool]:
    """
    Evaluate each prediction for a case, return list of safety_pass booleans.
    """
    safety_results = []

    for pred in predictions:
        if pred.get("parse_failed"):
            safety_results.append(False)
            continue

        eval_result = evaluate_case(pred, gold)
        safety_results.append(eval_result["safety_pass"])

    return safety_results


def compute_worst_at_k(
    safety_results: List[bool],
    k: int,
    n_samples: int = 1000,
) -> float:
    """
    Compute worst-at-k: probability of at least one failure in k samples.

    For small sample sizes, compute exactly. For larger, use Monte Carlo.
    """
    n = len(safety_results)

    if n < k:
        # Not enough samples
        return None

    if k == 1:
        # Worst-at-1 is just the failure rate
        return 1 - (sum(safety_results) / n)

    # For small k and n, compute exactly
    if n <= 20 and k <= 10:
        n_combos = 0
        n_with_failure = 0

        for combo in combinations(range(n), k):
            n_combos += 1
            combo_results = [safety_results[i] for i in combo]
            if not all(combo_results):  # At least one failure
                n_with_failure += 1

        return n_with_failure / n_combos if n_combos > 0 else None

    # Monte Carlo sampling for larger cases
    n_with_failure = 0

    for _ in range(n_samples):
        sample_indices = random.sample(range(n), k)
        sample_results = [safety_results[i] for i in sample_indices]
        if not all(sample_results):
            n_with_failure += 1

    return n_with_failure / n_samples


def analyze_model_worst_at_k(
    model: str,
    cases_df,
    k_values: List[int] = K_VALUES,
) -> Dict[str, Any]:
    """
    Analyze worst-at-k for a single model.
    """
    print(f"  Loading predictions for {model}...")
    case_predictions = load_variability_predictions(model)

    if not case_predictions:
        return {}

    n_runs = max(len(preds) for preds in case_predictions.values())
    print(f"  Found {len(case_predictions)} cases with up to {n_runs} runs each")

    # Evaluate each case
    case_safety_results = {}

    for case_id, predictions in case_predictions.items():
        case_row = cases_df[cases_df["case_id"] == case_id]

        if case_row.empty:
            continue

        case = case_row.iloc[0]
        gold = {
            "gold_top3": case.get("gold_top3", []),
            "escalation_required": case.get("escalation_required", False),
            "uncertainty_acceptable": case.get("uncertainty_acceptable", False),
        }

        safety_results = evaluate_predictions_for_case(predictions, gold)
        case_safety_results[case_id] = safety_results

    # Compute worst-at-k for each k value
    worst_at_k = {}

    for k in k_values:
        if k > n_runs:
            continue

        case_worst_rates = []

        for case_id, safety_results in case_safety_results.items():
            if len(safety_results) >= k:
                worst_rate = compute_worst_at_k(safety_results, k)
                if worst_rate is not None:
                    case_worst_rates.append(worst_rate)

        if case_worst_rates:
            worst_at_k[k] = {
                "mean": float(np.mean(case_worst_rates)),
                "std": float(np.std(case_worst_rates)),
                "median": float(np.median(case_worst_rates)),
                "n_cases": len(case_worst_rates),
            }

    # Also compute overall pass rate
    all_safety = []
    for results in case_safety_results.values():
        all_safety.extend(results)

    overall_pass_rate = sum(all_safety) / len(all_safety) if all_safety else 0

    return {
        "model": model,
        "n_cases": len(case_safety_results),
        "n_runs": n_runs,
        "overall_pass_rate": overall_pass_rate,
        "worst_at_k": worst_at_k,
    }


def create_worst_at_k_figure(
    all_results: Dict[str, Dict],
    output_path: Path,
) -> None:
    """Create worst-at-k degradation figure."""
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

    for i, (model, results) in enumerate(sorted(all_results.items())):
        short_name = model.split("/")[-1]
        worst_at_k = results.get("worst_at_k", {})

        if not worst_at_k:
            continue

        k_vals = sorted(worst_at_k.keys())
        # Plot 1 - worst_rate (probability of failure)
        # This should increase with k
        failure_rates = [worst_at_k[k]["mean"] * 100 for k in k_vals]

        ax.plot(k_vals, failure_rates, marker='o', label=short_name,
                color=colors[i], linewidth=2, markersize=8)

        # Add error bands
        stds = [worst_at_k[k]["std"] * 100 for k in k_vals]
        lower = [max(0, f - s) for f, s in zip(failure_rates, stds)]
        upper = [min(100, f + s) for f, s in zip(failure_rates, stds)]
        ax.fill_between(k_vals, lower, upper, alpha=0.2, color=colors[i])

    ax.set_xlabel("k (number of samples per case)", fontsize=12)
    ax.set_ylabel("Probability of ≥1 Safety Failure (%)", fontsize=12)
    ax.set_title("Worst-at-k Safety Reliability Degradation", fontsize=14)
    ax.set_xscale('log', base=2)
    ax.set_xticks(K_VALUES)
    ax.set_xticklabels([str(k) for k in K_VALUES])
    ax.set_ylim(0, 100)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved figure to {output_path}")


def generate_markdown_report(all_results: Dict[str, Dict]) -> str:
    """Generate markdown report."""
    lines = [
        "# Worst-at-k Safety Reliability Analysis",
        "",
        "This analysis shows how safety reliability degrades as you sample more responses per case.",
        "Worst-at-k measures: if you sample k responses per case, what's the probability of seeing",
        "at least one safety failure?",
        "",
        "## Summary",
        "",
    ]

    # Summary table
    k_values = sorted(set(
        k for r in all_results.values()
        for k in r.get("worst_at_k", {}).keys()
    ))

    if k_values:
        header = "| Model | Pass Rate | " + " | ".join([f"k={k}" for k in k_values]) + " |"
        separator = "|-------|-----------|" + "|------" * len(k_values) + "|"
        lines.extend([header, separator])

        for model, results in sorted(all_results.items()):
            short_name = model.split("/")[-1]
            pass_rate = f"{results.get('overall_pass_rate', 0)*100:.1f}%"

            k_vals = []
            for k in k_values:
                wak = results.get("worst_at_k", {}).get(k, {})
                if wak:
                    k_vals.append(f"{wak['mean']*100:.1f}%")
                else:
                    k_vals.append("N/A")

            lines.append(f"| {short_name} | {pass_rate} | " + " | ".join(k_vals) + " |")

        lines.append("")

    # Interpretation
    lines.extend([
        "## Interpretation",
        "",
        "- **Pass Rate**: Overall safety pass rate across all samples",
        "- **k=1**: Same as failure rate (1 - pass_rate)",
        "- **k=16**: If you sampled 16 responses per case, probability of seeing at least one failure",
        "",
        "Higher worst-at-k values indicate less reliable safety - the model sometimes fails",
        "even if it usually passes.",
        "",
    ])

    # Detailed results
    lines.extend([
        "## Detailed Results",
        "",
    ])

    for model, results in sorted(all_results.items()):
        short_name = model.split("/")[-1]
        lines.extend([
            f"### {short_name}",
            "",
            f"- Cases analyzed: {results.get('n_cases', 0)}",
            f"- Runs per case: {results.get('n_runs', 0)}",
            f"- Overall pass rate: {results.get('overall_pass_rate', 0)*100:.1f}%",
            "",
        ])

        worst_at_k = results.get("worst_at_k", {})
        if worst_at_k:
            lines.extend([
                "| k | Failure Prob | Std | Median |",
                "|---|--------------|-----|--------|",
            ])

            for k in sorted(worst_at_k.keys()):
                wak = worst_at_k[k]
                lines.append(
                    f"| {k} | {wak['mean']*100:.1f}% | {wak['std']*100:.1f}% | {wak['median']*100:.1f}% |"
                )

            lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Worst-at-k reliability analysis")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Models to analyze (auto-detected from run_variability data if not specified)",
    )

    args = parser.parse_args()

    # Load test cases
    print("Loading test cases...")
    cases_df = load_test_cases()

    # Auto-detect models from variability data
    if args.models:
        models = args.models
    else:
        run_files = list(VARIABILITY_DIR.glob("*-run01.json"))
        models = []
        for rf in run_files:
            # Extract model name from filename
            name = rf.stem.replace("-run01", "")
            # Convert back to model ID format
            if name.startswith("deepseek-"):
                models.append("deepseek/" + name[9:])
            elif name.startswith("anthropic-"):
                models.append("anthropic/" + name[10:])
            elif name.startswith("openai-"):
                models.append("openai/" + name[7:])
            else:
                models.append(name)

    if not models:
        print("No models found in run_variability data!")
        print("Run run_variability.py first to generate data.")
        return

    print(f"Analyzing models: {models}")

    all_results = {}

    for model in models:
        print(f"\n{'='*60}")
        print(f"Model: {model}")
        print(f"{'='*60}")

        results = analyze_model_worst_at_k(model, cases_df)
        if results:
            all_results[model] = results

    if not all_results:
        print("No results to analyze!")
        return

    # Save results
    output_path = OUTPUT_DIR / "worst_at_k_data.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {output_path}")

    # Generate report
    markdown = generate_markdown_report(all_results)
    md_path = OUTPUT_DIR / "worst_at_k_report.md"
    with open(md_path, "w") as f:
        f.write(markdown)
    print(f"Saved report to {md_path}")

    # Create figure
    if HAS_MATPLOTLIB:
        fig_path = OUTPUT_DIR / "worst_at_k_figure.png"
        create_worst_at_k_figure(all_results, fig_path)

    # Print summary
    print("\n" + "=" * 60)
    print("WORST-AT-K SUMMARY")
    print("=" * 60)

    for model, results in sorted(all_results.items()):
        short_name = model.split("/")[-1]
        print(f"\n{short_name}:")
        print(f"  Pass rate: {results.get('overall_pass_rate', 0)*100:.1f}%")

        worst_at_k = results.get("worst_at_k", {})
        for k in sorted(worst_at_k.keys()):
            wak = worst_at_k[k]
            print(f"  Worst-at-{k}: {wak['mean']*100:.1f}% failure probability")


if __name__ == "__main__":
    main()
