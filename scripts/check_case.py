#!/usr/bin/env python3
"""Check specific case for critical override."""

import json
from pathlib import Path

raw_path = Path("results/analysis/moe_panel/moe_raw_results.json")
results_path = Path("results/analysis/moe_panel_results.json")

with open(raw_path) as f:
    raw_results = json.load(f)

with open(results_path) as f:
    eval_results = json.load(f)

# Find case ddxplus_103880
for result in raw_results:
    if result["case_id"] == "ddxplus_103880":
        print("=== RAW RESULT ===")
        consensus = result.get("consensus", {})
        parsed = consensus.get("parsed", {})
        print(f"Parsed differential: {parsed.get('differential_diagnoses', [])}")
        print(f"Escalation decision: {parsed.get('escalation_decision')}")
        print(f"Critical override: {consensus.get('critical_diagnosis_override')}")
        print(f"Critical codes found: {consensus.get('critical_codes_found')}")
        break

# Check eval results
for pred in eval_results["consensus"]["predictions"]:
    if pred["case_id"] == "ddxplus_103880":
        print("\n=== EVAL RESULT ===")
        print(f"Differential: {pred.get('differential_diagnoses')}")
        print(f"Escalation: {pred.get('escalation_decision')}")
        print(f"Eval: {pred.get('eval')}")
        break
