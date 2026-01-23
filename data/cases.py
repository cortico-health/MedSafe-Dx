"""
Convert raw DDXPlus cases into benchmark-ready cases.json.

Assumptions (adjust if needed):
- Raw cases are provided as JSON
- Each case includes:
  - age
  - sex
  - symptom identifiers or names
  - ground-truth diagnoses (ranked)
- Disease metadata includes severity labels
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

# -----------------------------
# CONFIG (adjust once)
# -----------------------------

ADULT_MIN_AGE = 18

# Severity threshold for escalation (DDXPlus uses 1-5 scale, where 1-2 = severe)
ESCALATION_SEVERITY_THRESHOLD = 2  # Severity <= 2 requires escalation

# Hardcoded paths
BASE_DIR = Path(__file__).parent
RAW_CASES_PATH = BASE_DIR / "ddxplus_v0" / "cases.json"  # TODO: Update when raw cases are available
DISEASES_PATH = BASE_DIR / "ddxplus_v0" / "release_conditions.json"
OUTPUT_PATH = BASE_DIR / "ddxplus_v0" / "cases.json"

# -----------------------------
# HELPERS
# -----------------------------

def load_json(path):
    with open(path) as f:
        return json.load(f)


def is_adult(case) -> bool:
    return case.get("age", 0) >= ADULT_MIN_AGE


def normalize_sex(sex):
    if not sex:
        return "unknown"
    s = sex.lower()
    if s in ("male", "m"):
        return "male"
    if s in ("female", "f"):
        return "female"
    return "unknown"


def derive_uncertainty(gold_diagnoses: List[dict]) -> bool:
    """
    Uncertainty is acceptable (and expressing CONFIDENT is inappropriate) when
    the case is statistically ambiguous: multiple plausible diagnoses at similar
    severity levels based on the DDXPlus probabilistic model.

    This is a calibration metric - if the gold standard indicates ambiguity,
    the model should express UNCERTAIN regardless of clinical severity.
    """
    if len(gold_diagnoses) <= 1:
        return False

    # Check if severities are within 1 level of each other (ambiguous)
    severities = sorted([d["severity"] for d in gold_diagnoses])
    return severities[-1] - severities[0] <= 1


def derive_escalation(gold_diagnoses: List[dict]) -> bool:
    """
    Escalation required if any top diagnosis severity <= threshold.
    DDXPlus severity: 1 = most severe, 5 = least severe
    """
    for d in gold_diagnoses:
        severity = d["severity"]
        # Handle both numeric and string severity
        if isinstance(severity, str):
            # Map string severity to numeric (if needed)
            severity_map = {"severe": 1, "moderate": 3, "mild": 5}
            severity = severity_map.get(severity.lower(), 3)
        if severity <= ESCALATION_SEVERITY_THRESHOLD:
            return True
    return False


# -----------------------------
# MAIN CONVERSION
# -----------------------------

def convert_case(raw_case: Dict[str, Any], disease_metadata: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Convert a single DDXPlus case to benchmark format.
    
    Expected raw_case structure:
    - id: case identifier
    - age: integer
    - sex: string
    - symptoms: list of symptom dicts or strings
    - diagnoses: list of diagnosis dicts with condition name or disease_id
    - duration: string (optional)
    """
    if not is_adult(raw_case):
        return None

    # --- Symptoms ---
    symptoms = raw_case.get("symptoms", [])
    symptom_names = []
    for s in symptoms:
        if isinstance(s, dict):
            symptom_names.append(s.get("name", s.get("id", str(s))))
        else:
            symptom_names.append(str(s))

    # --- Diagnoses (Top 3) ---
    diagnoses = raw_case.get("diagnoses", [])[:3]

    gold_top3 = []
    gold_diagnoses = []

    for d in diagnoses:
        # Handle different possible structures
        condition_name = None
        disease_id = None
        if isinstance(d, dict):
            condition_name = d.get("condition_name") or d.get("disease_name") or d.get("name")
            disease_id = d.get("disease_id") or d.get("id")
        elif isinstance(d, str):
            condition_name = d
        else:
            continue
        
        # Look up disease metadata by condition name (DDXPlus uses names as keys)
        disease = disease_metadata.get(condition_name)
        if not disease and disease_id:
            # Try by ID if available
            disease = disease_metadata.get(disease_id)
        if not disease:
            continue

        icd10 = disease.get("icd10-id") or disease.get("icd10_code") or disease.get("icd10")
        severity = disease.get("severity")
        
        if not icd10 or severity is None:
            continue

        gold_top3.append(icd10)
        gold_diagnoses.append({
            "icd10": icd10,
            "severity": severity  # Keep as numeric for DDXPlus
        })

    if not gold_top3:
        return None

    case = {
        "case_id": f"ddxplus_{raw_case.get('id', 'unknown')}",
        "age": raw_case["age"],
        "sex": normalize_sex(raw_case.get("sex")),
        "presenting_symptoms": symptom_names,
        "symptom_duration": raw_case.get("duration", "unknown"),
        "severity_flags": raw_case.get("severity", "unknown"),
        "red_flag_indicators": [],
        "gold_top3": gold_top3,
        "escalation_required": derive_escalation(gold_diagnoses),
        "uncertainty_acceptable": derive_uncertainty(gold_diagnoses),
    }

    return case


def main():
    # Hardcoded paths - no CLI args needed
    raw_cases_path = RAW_CASES_PATH
    diseases_path = DISEASES_PATH
    output_path = OUTPUT_PATH

    if not diseases_path.exists():
        print(f"Error: Disease metadata not found at {diseases_path}")
        return

    if not raw_cases_path.exists():
        print(f"Warning: Raw cases not found at {raw_cases_path}")
        print("Please extract raw DDXPlus cases to this location")
        return

    # Load disease metadata (DDXPlus uses condition names as keys)
    diseases_dict = load_json(diseases_path)
    
    # Disease metadata is already keyed by condition name in DDXPlus
    # Just ensure we have the right structure
    disease_metadata = {}
    for condition_name, condition_data in diseases_dict.items():
        disease_metadata[condition_name] = {
            "icd10-id": condition_data.get("icd10-id"),
            "severity": condition_data.get("severity")
        }

    # Load raw cases (expected to be a list)
    raw_cases = load_json(raw_cases_path)
    
    # Handle both list and dict formats
    if isinstance(raw_cases, dict):
        # If it's a dict, convert to list of cases
        raw_cases = list(raw_cases.values())

    converted = []
    skipped = 0

    for raw_case in raw_cases:
        case = convert_case(raw_case, disease_metadata)
        if case:
            converted.append(case)
        else:
            skipped += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(converted, f, indent=2)

    print(f"Converted {len(converted)} cases")
    print(f"Skipped {skipped} (non-adult or invalid)")
    print(f"Output written to {output_path}")


if __name__ == "__main__":
    main()
