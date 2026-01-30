#!/usr/bin/env python3
"""Analyze cases where consensus got diagnosis right but missed escalation."""

import json
from pathlib import Path
from collections import Counter

# Load results
results_path = Path("results/analysis/moe_panel_results.json")
raw_path = Path("results/analysis/moe_panel/moe_raw_results.json")
test_path = Path("data/test_sets/test-v0.json")

with open(results_path) as f:
    eval_results = json.load(f)

with open(raw_path) as f:
    raw_results = json.load(f)

with open(test_path) as f:
    test_data = json.load(f)
    test_cases = {c["case_id"]: c for c in test_data["cases"]}

# Find cases where consensus got diagnosis but missed escalation
consensus_preds = eval_results["consensus"]["predictions"]

print("=" * 80)
print("CASES: Got diagnosis RIGHT but MISSED ESCALATION")
print("=" * 80)

missed_with_correct_dx = []
for pred in consensus_preds:
    eval_info = pred.get("eval", {})
    if eval_info.get("missed_escalation") and eval_info.get("top3_match"):
        missed_with_correct_dx.append(pred)

print(f"\nFound {len(missed_with_correct_dx)} cases where diagnosis was correct but escalation missed\n")

# Get details for first 5
for pred in missed_with_correct_dx[:5]:
    case_id = pred["case_id"]
    case = test_cases.get(case_id, {})
    eval_info = pred.get("eval", {})
    
    print("-" * 80)
    print(f"Case ID: {case_id}")
    print(f"Gold diagnoses: {case.get('gold_top3', [])}")
    print(f"Escalation required: {case.get('escalation_required')}")
    print(f"\nConsensus prediction:")
    print(f"  Diagnoses: {pred.get('differential_diagnoses', [])[:3]}")
    print(f"  Escalation: {pred.get('escalation_decision')}")
    print(f"  Top-3 match: {eval_info.get('top3_match')}")
    
    # Find raw result to see synthesizer reasoning
    for raw in raw_results:
        if raw["case_id"] == case_id:
            overruled = raw["consensus"].get("synthesizer_overruled_escalation", False)
            overruled_models = raw["consensus"].get("overruled_models", [])
            votes = raw["consensus"].get("panel_escalation_votes", {})
            
            print(f"\n  Panel votes: {votes}")
            print(f"  Synthesizer overruled escalation: {overruled}")
            if overruled_models:
                print(f"  Overruled models: {overruled_models}")
            break
    print()

# Summary stats
print("=" * 80)
print("SUMMARY")
print("=" * 80)

total_missed = sum(1 for p in consensus_preds if p.get("eval", {}).get("missed_escalation"))
missed_with_dx = len(missed_with_correct_dx)

if total_missed > 0:
    print(f"Total missed escalations: {total_missed}")
    print(f"Missed WITH correct diagnosis: {missed_with_dx} ({missed_with_dx/total_missed*100:.1f}%)")
    print(f"\n=> Synthesizer knows the diagnosis but doesn't realize it's severe!")
else:
    print("No missed escalations found")

# Check what gold diagnoses are being missed
print("\n" + "=" * 80)
print("GOLD DIAGNOSES IN MISSED ESCALATION CASES")
print("=" * 80)

missed_dx = []
for pred in consensus_preds:
    if pred.get("eval", {}).get("missed_escalation"):
        case = test_cases.get(pred["case_id"], {})
        missed_dx.extend(case.get("gold_top3", []))

dx_counts = Counter(missed_dx)
print("\nMost common gold diagnoses where escalation was missed:")
for dx, count in dx_counts.most_common(10):
    print(f"  {dx}: {count} cases")
