"""
Decode DDXPlus symptom codes to human-readable clinical findings.
"""

import json
from pathlib import Path


def load_evidence_data():
    """Load full symptom/evidence data including value meanings."""
    evidence_path = Path(__file__).parent.parent / "data" / "ddxplus_v0" / "release_evidences.json"
    try:
        with open(evidence_path) as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load evidence data: {e}")
        return {}


EVIDENCE_DATA = load_evidence_data()


def _decode_symptom_with_audit(symptom_code):
    """
    Decode symptom code to human-readable clinical text, plus decode fidelity signals.
    Returns: (description, is_antecedent, audit) tuple.
    """
    audit = {
        "input_code": symptom_code,
        "unknown_evidence_code": False,
        "unknown_value_code": False,
    }

    # Handle compound codes like "E_55_@_V_29" or "E_58_@_2"
    parts = symptom_code.split("_@_")
    base_code = parts[0]
    audit["base_code"] = base_code

    # Get base evidence data
    evidence = EVIDENCE_DATA.get(base_code)
    if not evidence:
        audit["unknown_evidence_code"] = True
        evidence = {}

    question = evidence.get("question_en", symptom_code)
    data_type = evidence.get("data_type", "")
    is_antecedent = evidence.get("is_antecedent", False)

    # Convert question to clinical statement
    description = question.replace("Do you have ", "").replace("Have you ", "").replace("?", "")
    description = description.replace("Do you ", "").replace("Are you ", "").replace("Is ", "")
    description = description.replace("Does the ", "").replace("Did the ", "")
    description = description.replace("How ", "").replace("What ", "")
    description = description.strip()

    # Handle specific evidence codes for cleaner presentation
    if base_code == "E_53":
        description = "Pain present"
    elif base_code == "E_57" and not parts[1:]:
        description = "Pain radiation"

    # If there's a value modifier
    if len(parts) > 1:
        value_code = parts[1]

        value_meanings = evidence.get("value_meaning", {})
        if value_code in value_meanings:
            value_text = value_meanings[value_code].get("en", value_code)

            if "feel pain somewhere" in question.lower():
                if value_text.lower() != "nowhere":
                    description = f"pain in {value_text}"
            elif "characterize your pain" in question.lower():
                description = f"pain character {value_text}"
            elif "irradiat" in question.lower() or "radiate" in question.lower():
                description = f"pain radiating to {value_text}"
            elif base_code == "E_204":
                return (f"Recent travel to {value_text}", True, audit)
            else:
                description = f"{description}: {value_text}"
        else:
            audit["unknown_value_code"] = True

            # Check if it's a numeric scale (data_type C with numeric value)
            if data_type == "C" and value_code.isdigit():
                if "intense" in question.lower():
                    description = f"pain intensity {value_code}/10"
                elif "precisely" in question.lower():
                    description = f"pain is diffuse (localization {value_code}/10)"
                elif "fast" in question.lower():
                    description = "sudden pain onset" if int(value_code) >= 7 else "gradual pain onset"
                else:
                    description = f"{description} {value_code}/10"
            else:
                # Try to find value in other evidences (for location codes)
                found = False
                for other_code, other_evidence in EVIDENCE_DATA.items():
                    if value_code in other_evidence.get("value_meaning", {}):
                        value_text = other_evidence["value_meaning"][value_code].get("en", value_code)
                        description = f"{description}: {value_text}"
                        found = True
                        break

                if not found and not value_code.isdigit():
                    description = f"{description}: {value_code}"

    # Capitalize first letter
    if description and description[0].islower():
        description = description[0].upper() + description[1:]

    return (description, is_antecedent, audit)


def decode_symptom(symptom_code):
    """
    Decode symptom code to human-readable clinical text.
    Returns: (description, is_antecedent) tuple
    
    Examples:
        E_19 -> ("Hyperthyroidism", True)
        E_55_@_V_29 -> ("Pain in lower chest", False)
    """
    description, is_antecedent, _audit = _decode_symptom_with_audit(symptom_code)
    return (description, is_antecedent)


def decode_symptoms(symptom_codes):
    """
    Decode a list of symptom codes to human-readable text.
    Returns: (active_symptoms, antecedents) tuple of lists
    """
    active_symptoms = []
    antecedents = []
    
    for code in symptom_codes:
        result = decode_symptom(code)
        if not result:
            continue
            
        text, is_antecedent = result
        if not text:  # Skip None values
            continue
            
        if is_antecedent:
            antecedents.append(text)
        else:
            active_symptoms.append(text)
            
    return active_symptoms, antecedents


def decode_symptoms_with_audit(symptom_codes):
    """
    Decode a list of symptom codes to human-readable text, plus decode fidelity stats.
    Returns: (active_symptoms, antecedents, audit) tuple.
    """
    active_symptoms = []
    antecedents = []
    unknown_evidence_codes: list[str] = []
    unknown_value_codes: list[str] = []

    for code in symptom_codes or []:
        text, is_antecedent, audit = _decode_symptom_with_audit(code)
        if audit.get("unknown_evidence_code"):
            unknown_evidence_codes.append(str(code))
        if audit.get("unknown_value_code"):
            unknown_value_codes.append(str(code))

        if not text:
            continue
        if is_antecedent:
            antecedents.append(text)
        else:
            active_symptoms.append(text)

    return active_symptoms, antecedents, {
        "total_codes": len(symptom_codes or []),
        "unknown_evidence_codes": unknown_evidence_codes[:25],
        "unknown_value_codes": unknown_value_codes[:25],
        "unknown_evidence_count": len(unknown_evidence_codes),
        "unknown_value_count": len(unknown_value_codes),
    }
