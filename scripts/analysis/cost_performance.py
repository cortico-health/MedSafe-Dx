#!/usr/bin/env python3
"""
Cost vs Performance Analysis for MedSafe-Dx.

Analyzes the cost-effectiveness of different models:
- Define cost table (API pricing per model)
- Calculate average tokens per response
- Compute cost per evaluation
- Plot safety_pass_rate vs cost
- Identify Pareto-optimal models

Outputs:
- results/analysis/cost_performance.json
- results/analysis/cost_performance_figure.png
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    load_evaluation,
    list_model_results,
)


# Extended cost data (prices per 1K tokens, as of late 2024)
# From OpenRouter / official API pricing
MODEL_COSTS = {
    "anthropic-claude-haiku-4.5": {
        "input": 0.0008,
        "output": 0.004,
        "display_name": "Claude Haiku 4.5",
    },
    "anthropic-claude-sonnet-4.5": {
        "input": 0.003,
        "output": 0.015,
        "display_name": "Claude Sonnet 4.5",
    },
    "openai-gpt-4o-mini": {
        "input": 0.00015,
        "output": 0.0006,
        "display_name": "GPT-4o Mini",
    },
    "openai-gpt-4.1": {
        "input": 0.002,
        "output": 0.008,
        "display_name": "GPT-4.1",
    },
    "openai-gpt-5.2": {
        "input": 0.005,
        "output": 0.015,
        "display_name": "GPT-5.2",
    },
    "openai-gpt-oss-120b": {
        "input": 0.001,
        "output": 0.003,
        "display_name": "GPT-OSS 120B",
    },
    "google-gemini-2.5-flash-lite": {
        "input": 0.000075,
        "output": 0.0003,
        "display_name": "Gemini 2.5 Flash Lite",
    },
    "google-gemini-2.5-pro": {
        "input": 0.00125,
        "output": 0.005,
        "display_name": "Gemini 2.5 Pro",
    },
    "google-gemini-3-pro-preview": {
        "input": 0.00175,
        "output": 0.007,
        "display_name": "Gemini 3 Pro Preview",
    },
    "deepseek-deepseek-chat-v3-0324": {
        "input": 0.00014,
        "output": 0.00028,
        "display_name": "DeepSeek Chat V3",
    },
}


def estimate_tokens(text: str) -> int:
    """Rough token estimate (4 chars per token average)."""
    if not text:
        return 0
    return len(text) // 4


def estimate_response_tokens(prediction: Dict) -> int:
    """Estimate tokens in a model response."""
    total = 0

    # Differential diagnoses
    dx = prediction.get("differential_diagnoses", [])
    for d in dx:
        if isinstance(d, dict):
            total += estimate_tokens(str(d.get("code", "")))
            total += estimate_tokens(str(d.get("description", "")))
        else:
            total += estimate_tokens(str(d))

    # Escalation and uncertainty
    total += estimate_tokens(str(prediction.get("escalation_decision", "")))
    total += estimate_tokens(str(prediction.get("uncertainty", "")))

    # Reasoning (usually the longest part)
    total += estimate_tokens(str(prediction.get("reasoning", "")))

    return total


def analyze_model_costs(
    model_id: str,
    predictions_path: Path,
    eval_path: Optional[Path],
) -> Dict[str, Any]:
    """Analyze costs for a single model."""

    # Load predictions
    with open(predictions_path) as f:
        data = json.load(f)

    if isinstance(data, dict) and "predictions" in data:
        predictions = data["predictions"]
    else:
        predictions = data

    # Estimate tokens
    output_tokens = []
    for pred in predictions:
        tokens = estimate_response_tokens(pred)
        output_tokens.append(tokens)

    avg_output_tokens = np.mean(output_tokens) if output_tokens else 0
    total_output_tokens = sum(output_tokens)

    # Estimate input tokens (prompt is roughly constant, ~500 tokens)
    estimated_input_tokens = 500
    total_input_tokens = estimated_input_tokens * len(predictions)

    # Get cost data
    cost_data = MODEL_COSTS.get(model_id, {})
    input_cost = cost_data.get("input", 0)
    output_cost = cost_data.get("output", 0)

    # Calculate costs
    total_cost = (
        (total_input_tokens / 1000) * input_cost +
        (total_output_tokens / 1000) * output_cost
    )
    cost_per_case = total_cost / len(predictions) if predictions else 0

    # Load evaluation metrics if available
    safety_pass_rate = None
    top1_recall = None
    top3_recall = None

    if eval_path and eval_path.exists():
        try:
            eval_data = load_evaluation(eval_path)
            safety_pass_rate = eval_data.get("safety_pass_rate")
            effectiveness = eval_data.get("effectiveness", {})
            top1_recall = effectiveness.get("top1_recall")
            top3_recall = effectiveness.get("top3_recall")
        except Exception:
            pass

    return {
        "model_id": model_id,
        "display_name": cost_data.get("display_name", model_id),
        "n_cases": len(predictions),
        "avg_output_tokens": avg_output_tokens,
        "total_output_tokens": total_output_tokens,
        "estimated_input_tokens_per_case": estimated_input_tokens,
        "cost_per_1k_input": input_cost,
        "cost_per_1k_output": output_cost,
        "total_cost": total_cost,
        "cost_per_case": cost_per_case,
        "safety_pass_rate": safety_pass_rate,
        "top1_recall": top1_recall,
        "top3_recall": top3_recall,
    }


def find_pareto_optimal(
    models: List[Dict],
    x_key: str = "cost_per_case",
    y_key: str = "safety_pass_rate",
    maximize_y: bool = True,
) -> List[str]:
    """
    Find Pareto-optimal models.

    Returns model_ids that are not dominated by any other model.
    Lower cost is better, higher safety (if maximize_y) is better.
    """
    pareto = []

    for m in models:
        if m.get(x_key) is None or m.get(y_key) is None:
            continue

        dominated = False
        for other in models:
            if other["model_id"] == m["model_id"]:
                continue
            if other.get(x_key) is None or other.get(y_key) is None:
                continue

            # Check if 'other' dominates 'm'
            # other dominates if: other.cost <= m.cost AND other.safety >= m.safety
            # AND at least one is strictly better
            cost_better = other[x_key] <= m[x_key]
            perf_better = (other[y_key] >= m[y_key]) if maximize_y else (other[y_key] <= m[y_key])

            cost_strictly = other[x_key] < m[x_key]
            perf_strictly = (other[y_key] > m[y_key]) if maximize_y else (other[y_key] < m[y_key])

            if cost_better and perf_better and (cost_strictly or perf_strictly):
                dominated = True
                break

        if not dominated:
            pareto.append(m["model_id"])

    return pareto


def create_cost_performance_figure(
    models: List[Dict],
    pareto_models: List[str],
    output_path: Path,
) -> None:
    """Create cost vs performance scatter plot."""
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    # Filter models with valid data
    valid_models = [
        m for m in models
        if m.get("cost_per_case") is not None and m.get("safety_pass_rate") is not None
    ]

    if not valid_models:
        print("No valid models for cost-performance plot")
        return

    # Extract data
    costs = [m["cost_per_case"] * 1000 for m in valid_models]  # Cost in millicents
    safety = [m["safety_pass_rate"] * 100 for m in valid_models]
    names = [m["display_name"] for m in valid_models]
    is_pareto = [m["model_id"] in pareto_models for m in valid_models]

    # Color by Pareto optimality
    colors = ['#2ecc71' if p else '#3498db' for p in is_pareto]
    sizes = [150 if p else 100 for p in is_pareto]
    markers = ['*' if p else 'o' for p in is_pareto]

    # Plot
    for i in range(len(valid_models)):
        ax.scatter(
            costs[i], safety[i],
            c=colors[i], s=sizes[i], marker=markers[i],
            alpha=0.8, edgecolors='white', linewidth=1.5
        )

        # Add label
        offset = (5, 5) if i % 2 == 0 else (-5, -10)
        ax.annotate(
            names[i],
            (costs[i], safety[i]),
            xytext=offset,
            textcoords='offset points',
            fontsize=8,
            ha='left' if offset[0] > 0 else 'right',
        )

    # Draw Pareto frontier line
    pareto_points = [(costs[i], safety[i]) for i in range(len(valid_models)) if is_pareto[i]]
    if len(pareto_points) > 1:
        pareto_points.sort(key=lambda x: x[0])
        px, py = zip(*pareto_points)
        ax.plot(px, py, 'g--', alpha=0.5, linewidth=2, label='Pareto Frontier')

    ax.set_xlabel('Cost per Case (millicents)', fontsize=12)
    ax.set_ylabel('Safety Pass Rate (%)', fontsize=12)
    ax.set_title('Cost vs Safety Performance\n(Pareto-optimal models in green)', fontsize=14)

    # Set axis limits with padding
    ax.set_xlim(0, max(costs) * 1.1)
    ax.set_ylim(0, 100)

    # Add grid
    ax.grid(True, alpha=0.3)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#2ecc71',
               markersize=15, label='Pareto-optimal'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db',
               markersize=10, label='Dominated'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved cost-performance figure to {output_path}")


def generate_markdown_report(models: List[Dict], pareto_models: List[str]) -> str:
    """Generate markdown report for cost-performance analysis."""
    lines = [
        "# MedSafe-Dx Cost vs Performance Analysis",
        "",
        "This report analyzes the cost-effectiveness of different models.",
        "",
        "## Cost Breakdown by Model",
        "",
        "| Model | Cases | Avg Output Tokens | Cost/Case | Total Cost | Safety Pass |",
        "|-------|-------|-------------------|-----------|------------|-------------|",
    ]

    for m in sorted(models, key=lambda x: x.get("cost_per_case", 0)):
        safety = f"{m['safety_pass_rate']*100:.1f}%" if m.get("safety_pass_rate") else "N/A"
        cost_case = f"${m['cost_per_case']:.4f}" if m.get("cost_per_case") else "N/A"
        total = f"${m['total_cost']:.2f}" if m.get("total_cost") else "N/A"

        lines.append(
            f"| {m['display_name']} | {m['n_cases']} | {m['avg_output_tokens']:.0f} | "
            f"{cost_case} | {total} | {safety} |"
        )

    lines.append("")

    # Pareto-optimal models
    lines.extend([
        "## Pareto-Optimal Models",
        "",
        "Models that are not dominated by any other model (best cost/safety trade-off):",
        "",
    ])

    pareto_details = [m for m in models if m["model_id"] in pareto_models]
    for m in sorted(pareto_details, key=lambda x: x.get("cost_per_case", 0)):
        safety = f"{m['safety_pass_rate']*100:.1f}%" if m.get("safety_pass_rate") else "N/A"
        cost = f"${m['cost_per_case']*1000:.2f}/1000 cases" if m.get("cost_per_case") else "N/A"
        lines.append(f"- **{m['display_name']}**: {safety} safety at {cost}")

    lines.append("")

    # Cost efficiency ranking
    lines.extend([
        "## Cost Efficiency Ranking",
        "",
        "Safety pass rate per dollar spent (higher is better):",
        "",
        "| Rank | Model | Safety/$ |",
        "|------|-------|----------|",
    ])

    efficiency = []
    for m in models:
        if m.get("safety_pass_rate") and m.get("total_cost") and m["total_cost"] > 0:
            eff = (m["safety_pass_rate"] * m["n_cases"]) / m["total_cost"]
            efficiency.append((m["display_name"], eff))

    for i, (name, eff) in enumerate(sorted(efficiency, key=lambda x: x[1], reverse=True), 1):
        lines.append(f"| {i} | {name} | {eff:.1f} |")

    lines.append("")

    # Pricing reference
    lines.extend([
        "## API Pricing Reference",
        "",
        "| Model | Input ($/1K tokens) | Output ($/1K tokens) |",
        "|-------|---------------------|----------------------|",
    ])

    for model_id, cost_data in sorted(MODEL_COSTS.items()):
        lines.append(
            f"| {cost_data.get('display_name', model_id)} | "
            f"${cost_data['input']:.5f} | ${cost_data['output']:.5f} |"
        )

    lines.append("")

    return "\n".join(lines)


def main():
    """Run cost-performance analysis."""
    print("Loading model results...")
    model_files = list_model_results()
    print(f"  Found {len(model_files)} models")

    all_results = []

    for mf in model_files:
        model_id = mf["model_id"]
        print(f"  Analyzing {model_id}...")

        try:
            result = analyze_model_costs(
                model_id,
                mf["predictions_path"],
                mf["eval_path"],
            )
            all_results.append(result)
        except Exception as e:
            print(f"    Error: {e}")

    if not all_results:
        print("No results to analyze!")
        return

    # Find Pareto-optimal models
    print("\nFinding Pareto-optimal models...")
    pareto_models = find_pareto_optimal(all_results)
    print(f"  Pareto-optimal: {pareto_models}")

    # Write JSON output
    output = {
        "models": all_results,
        "pareto_optimal": pareto_models,
    }

    output_path = PATHS["analysis_output_dir"] / "cost_performance.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nWrote {output_path}")

    # Generate markdown report
    markdown = generate_markdown_report(all_results, pareto_models)
    md_path = PATHS["analysis_output_dir"] / "cost_performance_analysis.md"
    with open(md_path, "w") as f:
        f.write(markdown)
    print(f"Wrote {md_path}")

    # Create figure
    if HAS_MATPLOTLIB:
        fig_path = PATHS["analysis_output_dir"] / "cost_performance_figure.png"
        create_cost_performance_figure(all_results, pareto_models, fig_path)

    # Print summary
    print("\n" + "=" * 60)
    print("COST-PERFORMANCE SUMMARY")
    print("=" * 60)

    print("\nBy cost per case (ascending):")
    for m in sorted(all_results, key=lambda x: x.get("cost_per_case", 999)):
        cost = f"${m['cost_per_case']*1000:.2f}/1K" if m.get("cost_per_case") else "N/A"
        safety = f"{m['safety_pass_rate']*100:.0f}%" if m.get("safety_pass_rate") else "N/A"
        pareto = " [PARETO]" if m["model_id"] in pareto_models else ""
        print(f"  {m['display_name']}: {cost} cases, {safety} safety{pareto}")


if __name__ == "__main__":
    main()
