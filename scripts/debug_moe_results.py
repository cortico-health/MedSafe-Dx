#!/usr/bin/env python3
"""Debug MOE results to see if critical diagnosis overrides are working."""

import json
from pathlib import Path
from collections import Counter

raw_path = Path("results/analysis/moe_panel/moe_raw_results.json")
results_path = Path("results/analysis/moe_panel_results.json")

with open(raw_path) as f:
    raw_results = json.load(f)

with open(results_path) as f:
    eval_results = json.load(f)

# Count critical diagnosis overrides
n_critical_override = 0
critical_codes_triggered = []
for result in raw_results:
    consensus = result.get("consensus", {})
    if consensus.get("critical_diagnosis_override"):
        n_critical_override += 1
        critical_codes_triggered.extend(consensus.get("critical_codes_found", []))

print(f"Critical diagnosis overrides triggered: {n_critical_override}")
print(f"Critical codes found: {Counter(critical_codes_triggered).most_common(10)}")

# Check missed escalations - are they missing critical codes in differential?
consensus_preds = eval_results["consensus"]["predictions"]
missed_esc_cases = [p for p in consensus_preds if p.get("eval", {}).get("missed_escalation")]

print(f"\n=== MISSED ESCALATIONS: {len(missed_esc_cases)} ===")

# Check if the missed cases had critical codes in their differential
for case in missed_esc_cases[:5]:
    case_id = case["case_id"]
    dx_codes = case.get("differential_diagnoses", [])
    print(f"\nCase: {case_id}")
    print(f"  Differential: {dx_codes[:3]}")
    print(f"  Escalation decision: {case.get('escalation_decision')}")
    
    # Find raw result
    for raw in raw_results:
        if raw["case_id"] == case_id:
            consensus = raw.get("consensus", {})
            print(f"  Critical override: {consensus.get('critical_diagnosis_override', False)}")
            print(f"  Critical codes found: {consensus.get('critical_codes_found', [])}")
            break

# Check overconfident wrong cases
overconf_cases = [p for p in consensus_preds if p.get("eval", {}).get("overconfident_wrong")]
print(f"\n=== OVERCONFIDENT WRONG: {len(overconf_cases)} ===")
for case in overconf_cases[:3]:
    print(f"\nCase: {case['case_id']}")
    print(f"  Differential: {case.get('differential_diagnoses', [])[:3]}")
    print(f"  Uncertainty: {case.get('uncertainty')}")
