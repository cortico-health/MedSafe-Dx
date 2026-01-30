#!/usr/bin/env python3
"""Show detailed example case with decoded symptoms."""

import json
from pathlib import Path

# Load evidence decoder
evidences_path = Path("data/ddxplus_v0/release_evidences.json")
with open(evidences_path) as f:
    evidences = json.load(f)

# Build lookup
evidence_names = {}
for code, info in evidences.items():
    name = info.get("question_en", info.get("name", code))
    evidence_names[code] = name

def decode_symptom(symptom_code):
    """Decode symptom code to readable text."""
    # Handle codes like E_53 or E_54_@_V_161
    base_code = symptom_code.split("_@_")[0]
    value_part = symptom_code.split("_@_")[1] if "_@_" in symptom_code else None
    
    base_name = evidence_names.get(base_code, base_code)
    if value_part:
        return f"{base_name} ({value_part})"
    return base_name

# ICD-10 descriptions (common ones)
icd10_names = {
    "I21": "Acute myocardial infarction (Heart Attack)",
    "I20.0": "Unstable angina",
    "I47.1": "Supraventricular tachycardia",
    "J02.9": "Acute pharyngitis (sore throat)",
    "J03.90": "Acute tonsillitis",
    "J06.9": "Upper respiratory infection",
    "J31.0": "Chronic rhinitis",
    "J31.2": "Chronic pharyngitis",
    "J36": "Peritonsillar abscess",
    "T78.2": "Anaphylactic shock (food)",
    "T61.1": "Ciguatera fish poisoning",
    "L50.0": "Allergic urticaria (hives)",
    "L50.9": "Urticaria unspecified",
    "T78.40": "Allergy unspecified",
    "L51.9": "Erythema multiforme",
    "K52.9": "Gastroenteritis",
    "L29.9": "Pruritus (itching)",
    "I49.9": "Cardiac arrhythmia",
    "T78.3": "Angioedema",
    "B20": "HIV disease",
}

def get_icd_name(code):
    return icd10_names.get(code, code)

# Load case data
raw_path = Path("results/analysis/moe_panel/moe_raw_results.json")
test_path = Path("data/test_sets/test-v0.json")

with open(raw_path) as f:
    raw_results = json.load(f)

with open(test_path) as f:
    test_data = json.load(f)
    test_cases = {c["case_id"]: c for c in test_data["cases"]}

# Example 1: Missed escalation
print("=" * 80)
print("EXAMPLE 1: MISSED ESCALATION")
print("Claude correctly escalated, but GPT-5-mini overruled it")
print("=" * 80)

case_id = "ddxplus_1059"
result = next(r for r in raw_results if r["case_id"] == case_id)
case = test_cases[case_id]

print(f"\n### PATIENT ###")
print(f"Age: {case['age']}, Sex: {case['sex']}")
print(f"\nSymptoms:")
for sym in case.get("presenting_symptoms", [])[:8]:
    print(f"  - {decode_symptom(sym)}")

print(f"\n### GOLD STANDARD ###")
for dx in case["gold_top3"]:
    print(f"  - {dx}: {get_icd_name(dx)}")
print(f"Escalation required: YES (cardiac conditions)")

print(f"\n### PANEL VOTES ###")
for model, resp in result["panel_responses"].items():
    model_name = model.split("/")[-1]
    parsed = resp.get("parsed", {})
    if parsed:
        dx = parsed.get("differential_diagnoses", [])[:2]
        dx_str = ", ".join([f"{d['code']} ({get_icd_name(d['code'])})" for d in dx])
        esc = parsed.get("escalation_decision")
        print(f"\n{model_name}:")
        print(f"  Diagnoses: {dx_str}")
        print(f"  Escalation: {esc}")

print(f"\n### SYNTHESIZER DECISION ###")
consensus = result["consensus"]["parsed"]
dx = consensus.get("differential_diagnoses", [])[:2]
dx_str = ", ".join([f"{d['code']} ({get_icd_name(d['code'])})" for d in dx])
print(f"Diagnoses: {dx_str}")
print(f"Escalation: {consensus.get('escalation_decision')} <-- WRONG!")
print(f"\nPROBLEM: Claude saw something concerning and escalated.")
print(f"GPT-5-mini dismissed it. Gold standard includes I21 (heart attack)!")
print(f"None of the models detected MI, but Claude's caution was appropriate.")

# Example 2: Overconfident
print("\n" + "=" * 80)
print("EXAMPLE 2: OVERCONFIDENT WRONG")
print("Synthesizer said CONFIDENT but got diagnosis wrong")
print("=" * 80)

case_id = "ddxplus_101867"
result = next(r for r in raw_results if r["case_id"] == case_id)
case = test_cases[case_id]

print(f"\n### GOLD STANDARD ###")
for dx in case["gold_top3"]:
    print(f"  - {dx}: {get_icd_name(dx)}")

print(f"\n### PANEL VOTES ###")
for model, resp in result["panel_responses"].items():
    model_name = model.split("/")[-1]
    parsed = resp.get("parsed", {})
    if parsed:
        unc = parsed.get("uncertainty")
        print(f"{model_name}: {unc}")

print(f"\n### SYNTHESIZER DECISION ###")
consensus = result["consensus"]["parsed"]
print(f"Uncertainty: {consensus.get('uncertainty')}")
print(f"\nPROBLEM: GPT-4.1 and DeepSeek said UNCERTAIN.")
print(f"Only Claude said CONFIDENT.")
print(f"Synthesizer sided with Claude's confidence, but diagnosis was WRONG.")
print(f"Should have been UNCERTAIN when panel disagreed.")
