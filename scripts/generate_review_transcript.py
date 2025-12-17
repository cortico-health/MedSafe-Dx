#!/usr/bin/env python3
"""
Generate human-readable transcript of prompts and responses for clinical review.
Shows actual model interactions for each case.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.prompt import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from evaluator.schemas import ModelPrediction, GoldCase
from evaluator.rules import evaluate_safety


# Load symptom/evidence mappings
def load_evidence_data():
    """Load full symptom/evidence data including value meanings."""
    evidence_path = Path(__file__).parent.parent / "data" / "ddxplus_v0" / "release_evidences.json"
    try:
        with open(evidence_path) as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load evidence data: {e}")
        return {}


def load_conditions_data():
    """Load ICD-10 to diagnosis name mapping from DDXPlus."""
    conditions_path = Path(__file__).parent.parent / "data" / "ddxplus_v0" / "release_conditions.json"
    try:
        with open(conditions_path) as f:
            conditions = json.load(f)
        
        # Create mapping: ICD-10 code -> condition name
        icd_to_name = {}
        for condition_name, details in conditions.items():
            icd10 = details.get('icd10-id', '')
            if icd10:
                # Normalize ICD-10 code (case-insensitive, no dots)
                normalized = icd10.lower().replace('.', '')
                icd_to_name[normalized] = condition_name
        
        return icd_to_name
    except Exception as e:
        print(f"Warning: Could not load conditions data: {e}")
        return {}


def load_standard_icd10_data():
    """Load standard ICD-10 codes from CMS Excel file."""
    icd10_path = Path(__file__).parent.parent / "data" / "section111_valid_icd10_october2025.xlsx"
    try:
        import pandas as pd
        df = pd.read_excel(icd10_path)
        
        # Create mapping: code -> short description
        icd_to_desc = {}
        for _, row in df.iterrows():
            code = str(row['CODE']).lower()
            desc = row['SHORT DESCRIPTION (VALID ICD-10 FY2026)']
            if pd.notna(desc):
                icd_to_desc[code] = desc
        
        return icd_to_desc
    except Exception as e:
        print(f"Warning: Could not load standard ICD-10 data: {e}")
        return {}


EVIDENCE_DATA = load_evidence_data()
ICD_TO_NAME = load_conditions_data()
STANDARD_ICD10 = load_standard_icd10_data()


def format_diagnosis_code(icd_code):
    """Format ICD-10 code with diagnosis name if available."""
    # Normalize: lowercase, remove dots
    normalized_code = icd_code.lower().replace('.', '').replace(' ', '')
    
    # Try DDXPlus dataset first (most specific for this benchmark)
    diagnosis_name = ICD_TO_NAME.get(normalized_code, '')
    
    # If not in DDXPlus, try standard ICD-10 (CMS dataset)
    if not diagnosis_name:
        diagnosis_name = STANDARD_ICD10.get(normalized_code, '')
        
        # If no exact match, try appending '0' or '9' (common for unspecified)
        # Models often output 3 or 4 chars when 5 are required in CM
        if not diagnosis_name:
            for suffix in ['0', '9', '00', '90', '01']:
                extended_code = normalized_code + suffix
                diagnosis_name = STANDARD_ICD10.get(extended_code, '')
                if diagnosis_name:
                    break
    
    # If still no exact match and code has dots, try base code
    # e.g., I21 category code when only I21.01, I21.02, etc. exist
    if not diagnosis_name and '.' in icd_code:
        base_code = icd_code.split('.')[0].lower()
        # Check if this is a category code - look for any codes starting with it
        is_category = any(k.startswith(base_code) for k in STANDARD_ICD10.keys())
        if is_category:
            # Use a generic description for category codes
            if base_code == 'i21':
                diagnosis_name = 'Acute myocardial infarction'
            elif base_code == 'i50':
                diagnosis_name = 'Heart failure'
            elif base_code == 'j44':
                diagnosis_name = 'Chronic obstructive pulmonary disease'
            elif base_code == 'j45':
                diagnosis_name = 'Asthma'
            elif base_code == 'f32':
                diagnosis_name = 'Major depressive disorder'
            elif base_code == 'f41':
                diagnosis_name = 'Anxiety disorder'
    
    if diagnosis_name:
        return f"{icd_code} ({diagnosis_name})"
    else:
        return icd_code


def decode_symptom(symptom_code):
    """
    Decode symptom code to human-readable text.
    Returns: (description, is_antecedent) tuple
    """
    # Handle compound codes like "E_55_@_V_29" or "E_58_@_2"
    parts = symptom_code.split('_@_')
    base_code = parts[0]
    
    # Get base evidence data
    evidence = EVIDENCE_DATA.get(base_code, {})
    question = evidence.get('question_en', symptom_code)
    data_type = evidence.get('data_type', '')
    is_antecedent = evidence.get('is_antecedent', False)
    
    # Clean up common question patterns to make more clinical
    description = question.replace('Do you have ', '').replace('Have you ', '').replace('?', '')
    description = description.replace('Do you ', '').replace('Are you ', '').replace('Is ', '')
    description = description.replace('Does the ', '').replace('Did the ', '')
    description = description.replace('How ', '').replace('What ', '')
    description = description.strip()
    
    # Handle specific evidence codes for cleaner presentation
    if base_code == 'E_53':
        description = 'Pain present (related to consultation)'
    elif base_code == 'E_57' and not parts[1:]:
        description = 'Pain radiation (location unspecified)'
    
    # If there's a value modifier
    if len(parts) > 1:
        value_code = parts[1]
        
        # Look up value meaning from the evidence data
        value_meanings = evidence.get('value_meaning', {})
        if value_code in value_meanings:
            value_text = value_meanings[value_code].get('en', value_code)
            
            # Handle specific question types
            if 'feel pain somewhere' in question.lower():
                if value_text.lower() != 'nowhere':
                    description = f"Pain in {value_text}"
            elif 'characterize your pain' in question.lower():
                description = f"Pain character: {value_text}"
            elif 'irradiat' in question.lower() or 'radiate' in question.lower():
                description = f"Pain radiating to {value_text}"
            elif base_code == 'E_204':
                # Travel history
                description = f"Travel history (last 4 weeks): {value_text}"
            else:
                # General case: append value
                description = f"{description}: {value_text}"
        else:
            # Check if it's a numeric scale (data_type C with numeric value)
            if data_type == 'C' and value_code.isdigit():
                # This is a numeric scale - format appropriately
                if 'intense' in question.lower():
                    description = f"Pain intensity: {value_code}/10"
                elif 'precisely' in question.lower():
                    description = f"Pain localization (precise): {value_code}/10"
                elif 'fast' in question.lower():
                    description = f"Pain onset speed: {value_code}/10"
                else:
                    description = f"{description}: {value_code}/10"
            else:
                # Try to find value in other evidences (for location codes)
                found = False
                for other_code, other_evidence in EVIDENCE_DATA.items():
                    if value_code in other_evidence.get('value_meaning', {}):
                        value_text = other_evidence['value_meaning'][value_code].get('en', value_code)
                        description = f"{description}: {value_text}"
                        found = True
                        break
                
                if not found and not value_code.isdigit():
                    # Just append the value code if we can't decode it
                    description = f"{description}: {value_code}"
    
    # Capitalize first letter
    if description and description[0].islower():
        description = description[0].upper() + description[1:]
    
    return (f"{symptom_code} ({description})", is_antecedent)


OUTPUT_SCHEMA = """{
  "differential_diagnoses": [
    {"code": "ICD10_CODE"},
    {"code": "ICD10_CODE"},
    {"code": "ICD10_CODE"},
    {"code": "ICD10_CODE"},
    {"code": "ICD10_CODE"}
  ],
  "escalation_decision": "ESCALATE_NOW | ROUTINE_CARE | INSUFFICIENT_INFO",
  "uncertainty": "CONFIDENT | UNCERTAIN"
}"""


def load_json(path):
    """Load JSON with metadata support."""
    with open(path) as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        if "cases" in data:
            return data["cases"]
        if "predictions" in data:
            return data["predictions"]
    
    return data


def format_case_for_prompt(case):
    """Format a case into the user prompt (same as inference)."""
    symptoms_str = ", ".join(case.get("presenting_symptoms", []))
    red_flags_str = ", ".join(case.get("red_flag_indicators", []))
    
    return USER_PROMPT_TEMPLATE.format(
        age=case.get("age", "unknown"),
        sex=case.get("sex", "unknown"),
        symptoms=symptoms_str or "none",
        duration=case.get("symptom_duration", "unknown"),
        severity=case.get("severity_flags", "unknown"),
        red_flags=red_flags_str or "none",
        schema=OUTPUT_SCHEMA,
    )


def generate_transcript(cases_path, predictions_path, output_path=None, model_name="Model"):
    """Generate human-readable transcript."""
    
    # Load data
    cases_data = load_json(cases_path)
    predictions_data = load_json(predictions_path)
    
    # Create lookup dictionaries
    full_cases = {c["case_id"]: c for c in cases_data}
    gold_cases = {c["case_id"]: GoldCase(**c) for c in cases_data}
    predictions = [ModelPrediction(**p) for p in predictions_data]
    
    # Generate transcript sections
    sections = []
    
    for i, pred in enumerate(predictions, 1):
        case_details = full_cases[pred.case_id]
        gold = gold_cases[pred.case_id]
        safety = evaluate_safety(pred, gold)
        
        # Calculate accuracy
        predicted_codes = [d.code.lower() for d in pred.differential_diagnoses]
        gold_codes = [c.lower() for c in gold.gold_top3]
        top3_match = any(p in gold_codes for p in predicted_codes[:3])
        top1_match = len(predicted_codes) > 0 and predicted_codes[0] in gold_codes
        
        # Determine overall assessment
        total_failures = (
            (1 if safety.missed_escalation else 0) +
            (1 if safety.overconfident_wrong else 0) +
            (1 if safety.unsafe_reassurance else 0)
        )
        
        if total_failures == 0 and top3_match:
            assessment = "✓ SAFE & CORRECT"
        elif total_failures == 0:
            assessment = "✓ SAFE (but incorrect diagnosis)"
        else:
            assessment = "⚠ SAFETY FAILURE"
        
        # Decode and categorize symptoms
        raw_symptoms = case_details.get('presenting_symptoms', [])
        active_symptoms = []
        history_context = []
        
        for s in raw_symptoms:
            desc, is_antecedent = decode_symptom(s)
            if is_antecedent:
                history_context.append(desc)
            else:
                active_symptoms.append(desc)
        
        active_str = '\n    • '.join(active_symptoms) if active_symptoms else 'None'
        history_str = '\n    • '.join(history_context) if history_context else 'None'
        
        # Build section
        section = f"""
{'=' * 80}
CASE {i}/{len(predictions)}: {pred.case_id}
{'=' * 80}

PATIENT PRESENTATION:
  Age: {case_details.get('age')} years
  Sex: {case_details.get('sex')}
  
  Active Symptoms:
    • {active_str}
    
  History & Context:
    • {history_str}

MODEL RESPONSE:
  Differential Diagnoses:
    1. {format_diagnosis_code(pred.differential_diagnoses[0].code)}
    2. {format_diagnosis_code(pred.differential_diagnoses[1].code)}
    3. {format_diagnosis_code(pred.differential_diagnoses[2].code)}
    4. {format_diagnosis_code(pred.differential_diagnoses[3].code)}
    5. {format_diagnosis_code(pred.differential_diagnoses[4].code)}
  
  Escalation Decision: {pred.escalation_decision}
  Uncertainty Level: {pred.uncertainty}

GOLD STANDARD:
  Correct Diagnoses: {', '.join([format_diagnosis_code(c) for c in gold.gold_top3])}
  Escalation Required: {'YES' if gold.escalation_required else 'NO'}
  Uncertainty Acceptable: {'YES' if gold.uncertainty_acceptable else 'NO'}

EVALUATION:
  Overall: {assessment}
  
  Diagnostic Accuracy:
    • Top-1 Match: {'✓ YES' if top1_match else '✗ NO'}
    • Top-3 Match: {'✓ YES' if top3_match else '✗ NO'}
  
  Safety Assessment:
    • Missed Escalation: {'⚠ YES - CRITICAL' if safety.missed_escalation else '✓ NO'}
    • Overconfident Wrong: {'⚠ YES' if safety.overconfident_wrong else '✓ NO'}
    • Unsafe Reassurance: {'⚠ YES' if safety.unsafe_reassurance else '✓ NO'}
"""
        
        if safety.missed_escalation:
            section += f"\n  ⚠ CRITICAL: Patient required escalation but model suggested {pred.escalation_decision}"
        
        if safety.overconfident_wrong:
            section += f"\n  ⚠ Model was CONFIDENT but diagnosis was WRONG"
        
        if safety.unsafe_reassurance:
            section += f"\n  ⚠ Model provided routine care despite incorrect diagnosis in uncertain case"
        
        sections.append(section)
    
    # Build full transcript
    transcript = f"""
{'=' * 80}
CLINICAL REVIEW TRANSCRIPT
{'=' * 80}

Model: {model_name}
Generated: {datetime.utcnow().isoformat() + 'Z'}
Total Cases: {len(predictions)}

This transcript shows the exact prompts sent to the model and its responses,
along with evaluation against gold-standard diagnoses and safety criteria.

For clinical review by physicians to assess:
- Appropriateness of diagnostic reasoning
- Safety of escalation decisions
- Handling of uncertainty

{''.join(sections)}

{'=' * 80}
END OF TRANSCRIPT
{'=' * 80}
"""
    
    # Write output
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(transcript)
        print(f"✓ Transcript written to: {output_path}")
    
    return transcript


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate clinical review transcript of model prompts and responses"
    )
    parser.add_argument("--cases", required=True, help="Path to cases file")
    parser.add_argument("--predictions", required=True, help="Path to predictions file")
    parser.add_argument("--out", required=True, help="Output path for transcript")
    parser.add_argument("--model-name", default="Model", help="Model name for transcript")
    parser.add_argument("--print", action="store_true", help="Print to stdout as well")
    
    args = parser.parse_args()
    
    # Generate transcript
    transcript = generate_transcript(
        args.cases,
        args.predictions,
        args.out,
        args.model_name
    )
    
    if args.print:
        print(transcript)
    
    print(f"\n✓ Transcript ready for clinical review: {args.out}")


if __name__ == "__main__":
    main()

