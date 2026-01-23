#!/usr/bin/env python3
"""
ICD-10 Analysis for MedSafe-Dx.

Analyzes diagnostic patterns:
- Prediction frequency table (all models combined)
- Top-20 most predicted codes
- Top-20 codes with highest accuracy
- Top-20 codes with lowest accuracy
- Confusion analysis by ICD-10 chapter
- Near-miss analysis: correct chapter, wrong specific code
- Confusion heatmap visualization

Outputs:
- results/analysis/icd10_analysis.json
- results/analysis/icd10_confusion_heatmap.png
"""

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping visualizations")

from .config import PATHS
from .utils import (
    load_test_cases,
    load_predictions,
    list_model_results,
    get_icd10_description,
)


# ICD-10 Chapter mapping (first character/digit)
ICD10_CHAPTERS = {
    "A": "Infectious diseases (A00-A99)",
    "B": "Infectious diseases (B00-B99)",
    "C": "Neoplasms (C00-C99)",
    "D": "Blood/Immune disorders (D00-D89)",
    "E": "Endocrine/Metabolic (E00-E89)",
    "F": "Mental disorders (F00-F99)",
    "G": "Nervous system (G00-G99)",
    "H": "Eye/Ear (H00-H95)",
    "I": "Circulatory system (I00-I99)",
    "J": "Respiratory system (J00-J99)",
    "K": "Digestive system (K00-K95)",
    "L": "Skin (L00-L99)",
    "M": "Musculoskeletal (M00-M99)",
    "N": "Genitourinary (N00-N99)",
    "O": "Pregnancy (O00-O99)",
    "P": "Perinatal (P00-P99)",
    "Q": "Congenital (Q00-Q99)",
    "R": "Symptoms/Signs (R00-R99)",
    "S": "Injury (S00-S99)",
    "T": "Injury/Poisoning (T00-T99)",
    "U": "Special codes (U00-U99)",
    "V": "External causes (V00-V99)",
    "W": "External causes (W00-W99)",
    "X": "External causes (X00-X99)",
    "Y": "External causes (Y00-Y99)",
    "Z": "Health status (Z00-Z99)",
}


def normalize_icd10(code: str) -> str:
    """Normalize ICD-10 code for comparison."""
    return str(code).upper().replace(".", "").replace(" ", "").strip()


def get_icd10_chapter(code: str) -> str:
    """Extract the chapter (first character) from an ICD-10 code."""
    normalized = normalize_icd10(code)
    if normalized:
        return normalized[0].upper()
    return "?"


def extract_predictions(predictions_df: pd.DataFrame) -> List[Tuple[str, List[str]]]:
    """Extract (case_id, [predicted_codes]) from predictions DataFrame."""
    results = []
    for _, row in predictions_df.iterrows():
        case_id = row["case_id"]
        dx_list = row.get("differential_diagnoses", [])
        codes = []
        for dx in dx_list:
            if isinstance(dx, dict):
                code = dx.get("code", "")
            else:
                code = str(dx)
            if code:
                codes.append(normalize_icd10(code))
        results.append((case_id, codes))
    return results


def analyze_predictions(
    all_predictions: List[Tuple[str, str, List[str]]],  # (case_id, model_id, codes)
    gold_by_case: Dict[str, List[str]],  # case_id -> [gold_codes]
) -> Dict[str, Any]:
    """Analyze ICD-10 prediction patterns."""

    # Count prediction frequencies
    prediction_counts = Counter()
    code_correct_counts = defaultdict(int)  # code -> times correct
    code_total_counts = defaultdict(int)  # code -> times predicted

    # Chapter confusion matrix
    chapter_predictions = []  # (gold_chapter, pred_chapter)

    # Near-miss tracking
    near_misses = []  # (case_id, gold_code, pred_code)

    for case_id, model_id, pred_codes in all_predictions:
        gold_codes = gold_by_case.get(case_id, [])
        gold_normalized = [normalize_icd10(c) for c in gold_codes]
        gold_chapters = set(get_icd10_chapter(c) for c in gold_codes)

        # Count predictions
        for i, code in enumerate(pred_codes):
            prediction_counts[code] += 1
            code_total_counts[code] += 1

            # Check if correct (top-3 match)
            if code in gold_normalized:
                code_correct_counts[code] += 1

            # Track chapter predictions (top-1 only)
            if i == 0 and gold_codes:
                pred_chapter = get_icd10_chapter(code)
                gold_chapter = get_icd10_chapter(gold_codes[0])
                chapter_predictions.append((gold_chapter, pred_chapter))

                # Near-miss: same chapter but different code
                if pred_chapter == gold_chapter and code not in gold_normalized:
                    near_misses.append((case_id, gold_codes[0], code))

    # Calculate accuracy by code
    code_accuracy = {}
    for code in code_total_counts:
        correct = code_correct_counts[code]
        total = code_total_counts[code]
        code_accuracy[code] = {
            "correct": correct,
            "total": total,
            "accuracy": correct / total if total > 0 else 0,
        }

    # Build chapter confusion matrix
    chapters = sorted(set(c for pair in chapter_predictions for c in pair))
    confusion = defaultdict(lambda: defaultdict(int))
    for gold_ch, pred_ch in chapter_predictions:
        confusion[gold_ch][pred_ch] += 1

    return {
        "prediction_counts": dict(prediction_counts),
        "code_accuracy": code_accuracy,
        "chapter_confusion": {k: dict(v) for k, v in confusion.items()},
        "chapters_seen": chapters,
        "near_misses": near_misses,
        "total_predictions": len(all_predictions),
    }


def format_top_codes(
    code_data: Dict[str, Any],
    n: int = 20,
    sort_by: str = "count",
) -> List[Dict]:
    """Format top N codes for output."""
    if sort_by == "count":
        items = sorted(
            [(code, data["total"]) for code, data in code_data.items()],
            key=lambda x: x[1],
            reverse=True,
        )
    elif sort_by == "accuracy_high":
        # Filter to codes with at least 5 predictions
        items = sorted(
            [(code, data["accuracy"]) for code, data in code_data.items() if data["total"] >= 5],
            key=lambda x: x[1],
            reverse=True,
        )
    elif sort_by == "accuracy_low":
        # Filter to codes with at least 5 predictions
        items = sorted(
            [(code, data["accuracy"]) for code, data in code_data.items() if data["total"] >= 5],
            key=lambda x: x[1],
            reverse=False,
        )
    else:
        items = list(code_data.items())[:n]

    results = []
    for code, value in items[:n]:
        data = code_data[code]
        results.append({
            "code": code,
            "description": get_icd10_description(code),
            "predictions": data["total"],
            "correct": data["correct"],
            "accuracy": data["accuracy"],
        })

    return results


def create_confusion_heatmap(
    confusion: Dict[str, Dict[str, int]],
    chapters: List[str],
    output_path: Path,
) -> None:
    """Create and save confusion heatmap."""
    if not HAS_MATPLOTLIB:
        return

    # Build matrix
    n = len(chapters)
    matrix = np.zeros((n, n))

    ch_to_idx = {ch: i for i, ch in enumerate(chapters)}

    for gold_ch, preds in confusion.items():
        if gold_ch not in ch_to_idx:
            continue
        i = ch_to_idx[gold_ch]
        for pred_ch, count in preds.items():
            if pred_ch not in ch_to_idx:
                continue
            j = ch_to_idx[pred_ch]
            matrix[i, j] = count

    # Normalize by row (gold chapter)
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    matrix_normalized = matrix / row_sums

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot heatmap
    im = ax.imshow(matrix_normalized, cmap='Blues', aspect='auto')

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Proportion of predictions", rotation=-90, va="bottom")

    # Set ticks
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(chapters)
    ax.set_yticklabels(chapters)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add labels
    ax.set_xlabel("Predicted ICD-10 Chapter")
    ax.set_ylabel("Gold ICD-10 Chapter")
    ax.set_title("ICD-10 Chapter Confusion Matrix\n(Row-normalized)")

    # Add text annotations for significant cells
    for i in range(n):
        for j in range(n):
            if matrix[i, j] > 0:
                value = matrix_normalized[i, j]
                color = "white" if value > 0.5 else "black"
                text = f"{value:.2f}" if value < 1 else "1.0"
                if matrix[i, j] >= 3:  # Only annotate cells with 3+ cases
                    ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved heatmap to {output_path}")


def generate_markdown_report(analysis: Dict[str, Any]) -> str:
    """Generate markdown report for ICD-10 analysis."""
    lines = [
        "# MedSafe-Dx ICD-10 Diagnostic Analysis",
        "",
        "This report analyzes ICD-10 prediction patterns across all models.",
        "",
        f"Total predictions analyzed: {analysis['total_predictions']}",
        f"Unique codes predicted: {len(analysis['code_accuracy'])}",
        f"Near-miss cases (correct chapter, wrong code): {len(analysis['near_misses'])}",
        "",
    ]

    # Top 20 most predicted codes
    lines.extend([
        "## Top 20 Most Predicted ICD-10 Codes",
        "",
        "| Rank | ICD-10 | Condition | Predictions | Correct | Accuracy |",
        "|------|--------|-----------|-------------|---------|----------|",
    ])

    top_predicted = format_top_codes(analysis["code_accuracy"], 20, "count")
    for i, code in enumerate(top_predicted, 1):
        desc = code["description"] or "Unknown"
        acc = f"{code['accuracy']*100:.1f}%"
        lines.append(f"| {i} | {code['code']} | {desc[:40]} | {code['predictions']} | {code['correct']} | {acc} |")

    lines.append("")

    # Top 20 highest accuracy codes
    lines.extend([
        "## Top 20 Highest Accuracy Codes (min 5 predictions)",
        "",
        "| Rank | ICD-10 | Condition | Predictions | Correct | Accuracy |",
        "|------|--------|-----------|-------------|---------|----------|",
    ])

    top_accurate = format_top_codes(analysis["code_accuracy"], 20, "accuracy_high")
    for i, code in enumerate(top_accurate, 1):
        desc = code["description"] or "Unknown"
        acc = f"{code['accuracy']*100:.1f}%"
        lines.append(f"| {i} | {code['code']} | {desc[:40]} | {code['predictions']} | {code['correct']} | {acc} |")

    lines.append("")

    # Top 20 lowest accuracy codes
    lines.extend([
        "## Top 20 Lowest Accuracy Codes (min 5 predictions)",
        "",
        "| Rank | ICD-10 | Condition | Predictions | Correct | Accuracy |",
        "|------|--------|-----------|-------------|---------|----------|",
    ])

    low_accurate = format_top_codes(analysis["code_accuracy"], 20, "accuracy_low")
    for i, code in enumerate(low_accurate, 1):
        desc = code["description"] or "Unknown"
        acc = f"{code['accuracy']*100:.1f}%"
        lines.append(f"| {i} | {code['code']} | {desc[:40]} | {code['predictions']} | {code['correct']} | {acc} |")

    lines.append("")

    # Chapter-level analysis
    lines.extend([
        "## ICD-10 Chapter Analysis",
        "",
        "Performance breakdown by ICD-10 chapter (first character of code).",
        "",
    ])

    # Calculate chapter stats
    confusion = analysis["chapter_confusion"]
    chapter_stats = []

    for gold_ch in sorted(confusion.keys()):
        preds = confusion[gold_ch]
        total = sum(preds.values())
        correct = preds.get(gold_ch, 0)
        accuracy = correct / total if total > 0 else 0
        chapter_name = ICD10_CHAPTERS.get(gold_ch, f"Chapter {gold_ch}")

        chapter_stats.append({
            "chapter": gold_ch,
            "name": chapter_name,
            "total": total,
            "correct_chapter": correct,
            "accuracy": accuracy,
        })

    lines.extend([
        "| Chapter | Description | N | Correct Chapter | Chapter Accuracy |",
        "|---------|-------------|---|-----------------|------------------|",
    ])

    for stat in sorted(chapter_stats, key=lambda x: x["total"], reverse=True):
        acc = f"{stat['accuracy']*100:.1f}%"
        lines.append(f"| {stat['chapter']} | {stat['name'][:35]} | {stat['total']} | {stat['correct_chapter']} | {acc} |")

    lines.append("")

    # Near-miss examples
    lines.extend([
        "## Near-Miss Analysis",
        "",
        f"Total near-misses (correct chapter, wrong specific code): {len(analysis['near_misses'])}",
        "",
    ])

    if analysis["near_misses"]:
        # Group by gold code
        near_miss_by_gold = defaultdict(list)
        for case_id, gold, pred in analysis["near_misses"]:
            near_miss_by_gold[gold].append((case_id, pred))

        lines.extend([
            "### Most Common Near-Miss Gold Codes",
            "",
            "| Gold Code | Gold Condition | Near-Miss Count |",
            "|-----------|----------------|-----------------|",
        ])

        for gold, instances in sorted(near_miss_by_gold.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
            desc = get_icd10_description(gold) or "Unknown"
            lines.append(f"| {gold} | {desc[:40]} | {len(instances)} |")

    lines.append("")

    return "\n".join(lines)


def main():
    """Run ICD-10 analysis."""
    print("Loading test cases...")
    cases_df = load_test_cases()
    print(f"  Loaded {len(cases_df)} test cases")

    # Build gold standard mapping
    gold_by_case = {}
    for _, row in cases_df.iterrows():
        gold_by_case[row["case_id"]] = [normalize_icd10(c) for c in row["gold_top3"]]

    # Load all model predictions
    print("\nLoading model predictions...")
    model_files = list_model_results()
    print(f"  Found {len(model_files)} models")

    all_predictions = []  # (case_id, model_id, [codes])

    for mf in model_files:
        model_id = mf["model_id"]
        print(f"  Loading {model_id}...")

        try:
            predictions_df = load_predictions(mf["predictions_path"])
            case_preds = extract_predictions(predictions_df)

            for case_id, codes in case_preds:
                all_predictions.append((case_id, model_id, codes))

        except Exception as e:
            print(f"    Error: {e}")

    if not all_predictions:
        print("No predictions to analyze!")
        return

    # Analyze predictions
    print("\nAnalyzing ICD-10 patterns...")
    analysis = analyze_predictions(all_predictions, gold_by_case)

    # Add formatted top lists to output
    analysis["top_20_most_predicted"] = format_top_codes(analysis["code_accuracy"], 20, "count")
    analysis["top_20_highest_accuracy"] = format_top_codes(analysis["code_accuracy"], 20, "accuracy_high")
    analysis["top_20_lowest_accuracy"] = format_top_codes(analysis["code_accuracy"], 20, "accuracy_low")

    # Truncate near_misses for JSON (keep just count)
    analysis["near_miss_count"] = len(analysis["near_misses"])
    analysis["near_misses"] = analysis["near_misses"][:50]  # Keep first 50 for inspection

    # Write JSON output
    output_path = PATHS["analysis_output_dir"] / "icd10_analysis.json"
    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"\nWrote {output_path}")

    # Generate markdown report
    markdown = generate_markdown_report(analysis)
    md_path = PATHS["analysis_output_dir"] / "icd10_analysis.md"
    with open(md_path, "w") as f:
        f.write(markdown)
    print(f"Wrote {md_path}")

    # Create confusion heatmap
    if HAS_MATPLOTLIB and analysis["chapters_seen"]:
        heatmap_path = PATHS["analysis_output_dir"] / "icd10_confusion_heatmap.png"
        create_confusion_heatmap(
            analysis["chapter_confusion"],
            analysis["chapters_seen"],
            heatmap_path,
        )

    # Print summary
    print("\n" + "=" * 60)
    print("ICD-10 ANALYSIS SUMMARY")
    print("=" * 60)

    print(f"\nTotal predictions analyzed: {analysis['total_predictions']}")
    print(f"Unique codes predicted: {len(analysis['code_accuracy'])}")
    print(f"Near-miss cases: {analysis['near_miss_count']}")

    print("\nTop 5 most predicted codes:")
    for i, code in enumerate(analysis["top_20_most_predicted"][:5], 1):
        desc = code["description"] or "Unknown"
        print(f"  {i}. {code['code']} ({desc[:30]}): {code['predictions']} predictions, {code['accuracy']*100:.1f}% accuracy")

    print("\nTop 5 highest accuracy codes (min 5 predictions):")
    for i, code in enumerate(analysis["top_20_highest_accuracy"][:5], 1):
        desc = code["description"] or "Unknown"
        print(f"  {i}. {code['code']} ({desc[:30]}): {code['accuracy']*100:.1f}% ({code['correct']}/{code['predictions']})")


if __name__ == "__main__":
    main()
