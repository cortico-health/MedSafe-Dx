#!/usr/bin/env python3
"""Show example with panelist rationale."""

import json
from pathlib import Path

raw_path = Path("results/analysis/moe_panel/moe_raw_results.json")

with open(raw_path) as f:
    raw_results = json.load(f)

# Find a case where panelists disagreed on escalation
for result in raw_results[:20]:
    panel = result.get("panel_responses", {})
    escalations = []
    for model, resp in panel.items():
        parsed = resp.get("parsed", {})
        if parsed:
            escalations.append(parsed.get("escalation_decision"))
    
    # Show cases where there was disagreement
    if "ESCALATE_NOW" in escalations and "ROUTINE_CARE" in escalations:
        print(f"Case ID: {result['case_id']}")
        print("=" * 70)
        
        for model, resp in panel.items():
            model_name = model.split("/")[-1]
            parsed = resp.get("parsed", {})
            if parsed:
                print(f"\n### {model_name} ###")
                dx = [d.get("code") for d in parsed.get("differential_diagnoses", [])[:3]]
                print(f"Top-3 Dx: {dx}")
                print(f"Escalation: {parsed.get('escalation_decision')}")
                print(f"Rationale: {parsed.get('escalation_rationale', 'N/A')}")
                print(f"Uncertainty: {parsed.get('uncertainty')}")
                print(f"Uncertainty Rationale: {parsed.get('uncertainty_rationale', 'N/A')}")
        
        print(f"\n### SYNTHESIZER CONSENSUS ###")
        consensus = result["consensus"].get("parsed", {})
        if consensus:
            dx = [d.get("code") for d in consensus.get("differential_diagnoses", [])[:3]]
            print(f"Top-3 Dx: {dx}")
            print(f"Escalation: {consensus.get('escalation_decision')}")
            print(f"Uncertainty: {consensus.get('uncertainty')}")
        
        break
