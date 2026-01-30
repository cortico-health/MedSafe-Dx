#!/usr/bin/env python3
"""Analyze patterns in over-escalation cases."""

import json
from pathlib import Path
from collections import Counter

raw_path = Path("results/analysis/moe_panel/moe_raw_results.json")
results_path = Path("results/analysis/moe_panel_results.json")
test_path = Path("data/test_sets/test-v0.json")

with open(raw_path) as f:
    raw_results = json.load(f)

with open(results_path) as f:
    eval_results = json.load(f)

with open(test_path) as f:
    test_data = json.load(f)
    test_cases = {c["case_id"]: c for c in test_data["cases"]}

# Find over-escalation cases
over_esc_cases = []
consensus_preds = eval_results["consensus"]["predictions"]

for pred in consensus_preds:
    if pred.get("eval", {}).get("over_escalation"):
        case_id = pred["case_id"]
        over_esc_cases.append(case_id)

print(f"Total over-escalation cases: {len(over_esc_cases)}")
print("=" * 80)

# Analyze patterns
gold_diagnoses = []
panel_escalation_patterns = Counter()
escalation_rationales = []

for case_id in over_esc_cases:
    case = test_cases.get(case_id, {})
    gold_dx = case.get("gold_top3", [])
    gold_diagnoses.extend(gold_dx)
    
    # Find raw result
    for result in raw_results:
        if result["case_id"] == case_id:
            # Count how many panelists escalated
            panel = result.get("panel_responses", {})
            n_escalated = 0
            rationales = []
            for model, resp in panel.items():
                parsed = resp.get("parsed", {})
                if parsed and parsed.get("escalation_decision") == "ESCALATE_NOW":
                    n_escalated += 1
                    rationale = parsed.get("escalation_rationale", "")
                    if rationale:
                        rationales.append(f"{model.split('/')[-1]}: {rationale}")
            
            panel_escalation_patterns[f"{n_escalated}/3 panelists escalated"] += 1
            escalation_rationales.extend(rationales)
            break

print("\n### PANEL VOTING PATTERNS IN OVER-ESCALATION CASES ###")
for pattern, count in panel_escalation_patterns.most_common():
    print(f"  {pattern}: {count} cases")

print("\n### GOLD DIAGNOSES IN OVER-ESCALATED CASES ###")
print("(These are conditions that did NOT require escalation)")
dx_counts = Counter(gold_diagnoses)
for dx, count in dx_counts.most_common(15):
    print(f"  {dx}: {count}")

print("\n### SAMPLE ESCALATION RATIONALES (why panelists escalated) ###")
for i, rationale in enumerate(escalation_rationales[:10]):
    print(f"\n{i+1}. {rationale}")

# Show a few detailed examples
print("\n" + "=" * 80)
print("### DETAILED EXAMPLES ###")
print("=" * 80)

for case_id in over_esc_cases[:3]:
    case = test_cases.get(case_id, {})
    print(f"\nCase: {case_id}")
    print(f"Gold Dx: {case.get('gold_top3', [])}")
    print(f"Escalation Required: {case.get('escalation_required')} (False = over-escalation)")
    
    for result in raw_results:
        if result["case_id"] == case_id:
            print("\nPanel rationales:")
            for model, resp in result["panel_responses"].items():
                parsed = resp.get("parsed", {})
                if parsed:
                    esc = parsed.get("escalation_decision")
                    rationale = parsed.get("escalation_rationale", "N/A")
                    print(f"  {model.split('/')[-1]}: {esc}")
                    print(f"    -> {rationale[:100]}...")
            break
    print("-" * 60)
