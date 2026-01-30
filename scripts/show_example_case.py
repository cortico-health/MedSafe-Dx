#!/usr/bin/env python3
"""Show detailed example case for review."""

import json
from pathlib import Path

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

# Find a case that shows the issue well:
# - Gold has critical diagnosis (I21, etc)
# - Panel found it
# - Consensus missed escalation OR was overconfident

def find_example_cases():
    examples = {"missed_esc_with_panel_correct": [], "overconfident_wrong": []}
    
    for result in raw_results:
        case_id = result["case_id"]
        case = test_cases.get(case_id, {})
        gold_dx = case.get("gold_top3", [])
        esc_required = case.get("escalation_required", False)
        
        # Get panel responses
        panel = result.get("panel_responses", {})
        consensus = result.get("consensus", {})
        consensus_parsed = consensus.get("parsed", {})
        
        # Find matching eval result
        eval_info = None
        for pred in eval_results["consensus"]["predictions"]:
            if pred["case_id"] == case_id:
                eval_info = pred.get("eval", {})
                break
        
        if not eval_info:
            continue
            
        # Check for missed escalation where panel had right idea
        if eval_info.get("missed_escalation"):
            # Check if any panelist correctly escalated
            panel_escalated = []
            for model, resp in panel.items():
                parsed = resp.get("parsed", {})
                if parsed and parsed.get("escalation_decision") == "ESCALATE_NOW":
                    panel_escalated.append(model.split("/")[-1])
            
            if panel_escalated:
                examples["missed_esc_with_panel_correct"].append({
                    "case_id": case_id,
                    "gold_dx": gold_dx,
                    "panel_escalated": panel_escalated,
                    "result": result,
                    "case": case,
                })
        
        # Check for overconfident wrong
        if eval_info.get("overconfident_wrong"):
            examples["overconfident_wrong"].append({
                "case_id": case_id,
                "gold_dx": gold_dx,
                "result": result,
                "case": case,
            })
    
    return examples

examples = find_example_cases()

# Show best example of missed escalation where panel was right
print("=" * 80)
print("EXAMPLE: MISSED ESCALATION (Panel correctly escalated, synthesizer didn't)")
print("=" * 80)

if examples["missed_esc_with_panel_correct"]:
    ex = examples["missed_esc_with_panel_correct"][0]
    result = ex["result"]
    case = ex["case"]
    
    print(f"\nCase ID: {ex['case_id']}")
    print(f"\n--- PATIENT PRESENTATION ---")
    print(f"Age: {case.get('age')}, Sex: {case.get('sex')}")
    print(f"Symptoms: {case.get('presenting_symptoms', [])[:5]}...")  # First 5 symptom codes
    
    print(f"\n--- GOLD STANDARD ---")
    print(f"Diagnoses: {ex['gold_dx']}")
    print(f"Escalation required: {case.get('escalation_required')}")
    
    print(f"\n--- PANEL RESPONSES ---")
    for model, resp in result["panel_responses"].items():
        model_name = model.split("/")[-1]
        parsed = resp.get("parsed", {})
        if parsed:
            dx = [d.get("code") for d in parsed.get("differential_diagnoses", [])[:3]]
            esc = parsed.get("escalation_decision")
            print(f"\n{model_name}:")
            print(f"  Top-3 Dx: {dx}")
            print(f"  Escalation: {esc}")
    
    print(f"\n--- SYNTHESIZER (GPT-5-mini) CONSENSUS ---")
    consensus = result["consensus"]
    parsed = consensus.get("parsed", {})
    if parsed:
        dx = [d.get("code") for d in parsed.get("differential_diagnoses", [])[:3]]
        print(f"Top-3 Dx: {dx}")
        print(f"Escalation: {parsed.get('escalation_decision')}")
        print(f"Uncertainty: {parsed.get('uncertainty')}")
    
    print(f"\nPanel members who escalated: {ex['panel_escalated']}")
    print(f"Synthesizer overruled: {consensus.get('synthesizer_overruled_escalation', False)}")
    
    # Show raw synthesizer response excerpt
    raw_resp = consensus.get("raw_response", "")
    if raw_resp and len(raw_resp) > 100:
        print(f"\n--- SYNTHESIZER RAW RESPONSE (excerpt) ---")
        print(raw_resp[:500])

else:
    print("No examples found of this type")

# Show overconfident wrong example
print("\n" + "=" * 80)
print("EXAMPLE: OVERCONFIDENT WRONG (Synthesizer confident but diagnosis wrong)")  
print("=" * 80)

if examples["overconfident_wrong"]:
    ex = examples["overconfident_wrong"][0]
    result = ex["result"]
    case = ex["case"]
    
    print(f"\nCase ID: {ex['case_id']}")
    
    print(f"\n--- GOLD STANDARD ---")
    print(f"Diagnoses: {ex['gold_dx']}")
    print(f"Escalation required: {case.get('escalation_required')}")
    
    print(f"\n--- PANEL RESPONSES ---")
    for model, resp in result["panel_responses"].items():
        model_name = model.split("/")[-1]
        parsed = resp.get("parsed", {})
        if parsed:
            dx = [d.get("code") for d in parsed.get("differential_diagnoses", [])[:3]]
            unc = parsed.get("uncertainty")
            print(f"\n{model_name}:")
            print(f"  Top-3 Dx: {dx}")
            print(f"  Uncertainty: {unc}")
    
    print(f"\n--- SYNTHESIZER CONSENSUS ---")
    consensus = result["consensus"]
    parsed = consensus.get("parsed", {})
    if parsed:
        dx = [d.get("code") for d in parsed.get("differential_diagnoses", [])[:3]]
        print(f"Top-3 Dx: {dx}")
        print(f"Uncertainty: {parsed.get('uncertainty')} <-- PROBLEM: Should be UNCERTAIN")
        print(f"Escalation: {parsed.get('escalation_decision')}")
