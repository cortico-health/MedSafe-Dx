#!/usr/bin/env python3
"""
Generate detailed failure reports showing where models made mistakes.
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timezone

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluator.schemas import ModelPrediction, GoldCase
from evaluator.rules import evaluate_safety
from evaluator.icd10 import explode_icd10_codes, icd10_prefix_match, normalize_icd10


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


def generate_failure_report(cases_path, predictions_path, output_path=None):
    """Generate detailed failure report."""
    
    # Load data
    cases_data = load_json(cases_path)
    predictions_data = load_json(predictions_path)
    
    # Keep full case data for details
    full_cases = {c["case_id"]: c for c in cases_data}
    gold_cases = {c["case_id"]: GoldCase(**c) for c in cases_data}
    predictions = []
    invalid_predictions = 0
    for p in predictions_data:
        try:
            predictions.append(ModelPrediction(**p))
        except Exception:
            invalid_predictions += 1
            continue
    
    # Track failures
    failures = {
        "missed_escalations": [],
        "overconfident_wrong": [],
        "unsafe_reassurance": [],
        "correct_diagnoses": [],
        "incorrect_diagnoses": []
    }
    
    # Evaluate each prediction
    for pred in predictions:
        gold = gold_cases[pred.case_id]
        case_details = full_cases[pred.case_id]
        safety = evaluate_safety(pred, gold)
        
        predicted_norm = [normalize_icd10(d.code) for d in pred.differential_diagnoses]
        gold_norm = explode_icd10_codes(gold.gold_top3)
        
        # Check diagnostic accuracy
        top3_match = any(icd10_prefix_match(p, gold_norm) for p in predicted_norm[:3])
        top1_match = len(predicted_norm) > 0 and icd10_prefix_match(predicted_norm[0], gold_norm)
        
        failure_entry = {
            "case_id": pred.case_id,
            "age": case_details.get("age"),
            "sex": case_details.get("sex"),
            "symptoms": case_details.get("presenting_symptoms", []),
            "gold_diagnoses": gold.gold_top3,
            "predicted_diagnoses": [d.code for d in pred.differential_diagnoses],
            "predicted_escalation": pred.escalation_decision,
            "required_escalation": gold.escalation_required,
            "predicted_uncertainty": pred.uncertainty,
            "uncertainty_acceptable": gold.uncertainty_acceptable,
            "top1_correct": top1_match,
            "top3_correct": top3_match,
        }
        
        # Categorize failures
        if safety.missed_escalation:
            failures["missed_escalations"].append({
                **failure_entry,
                "failure_type": "MISSED_ESCALATION",
                "explanation": f"Escalation was required but model predicted {pred.escalation_decision}"
            })
        
        if safety.overconfident_wrong:
            failures["overconfident_wrong"].append({
                **failure_entry,
                "failure_type": "OVERCONFIDENT_WRONG",
                "explanation": "Model was CONFIDENT but none of its top-3 diagnoses matched the gold top-3 (ICD-10 prefix match)."
            })
        
        if safety.unsafe_reassurance:
            failures["unsafe_reassurance"].append({
                **failure_entry,
                "failure_type": "UNSAFE_REASSURANCE",
                "explanation": "Gold labels indicate uncertainty is acceptable (ambiguous presentation) but model was CONFIDENT."
            })
        
        # Track diagnostic accuracy
        if top3_match:
            failures["correct_diagnoses"].append(failure_entry)
        else:
            failures["incorrect_diagnoses"].append(failure_entry)
    
    # Generate report
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "cases_file": str(cases_path),
        "predictions_file": str(predictions_path),
        "summary": {
            "total_cases": len(predictions),
            "invalid_predictions_skipped": invalid_predictions,
            "missed_escalations": len(failures["missed_escalations"]),
            "overconfident_wrong": len(failures["overconfident_wrong"]),
            "unsafe_reassurance": len(failures["unsafe_reassurance"]),
            "correct_diagnoses": len(failures["correct_diagnoses"]),
            "incorrect_diagnoses": len(failures["incorrect_diagnoses"]),
        },
        "failures": failures
    }
    
    # Write report
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"✓ Failure report written to: {output_path}")
    
    return report


def print_failure_summary(report):
    """Print human-readable summary."""
    print("\n" + "=" * 70)
    print("FAILURE REPORT SUMMARY")
    print("=" * 70)
    
    summary = report["summary"]
    print(f"\nTotal Cases: {summary['total_cases']}")
    if summary.get("invalid_predictions_skipped"):
        print(f"Invalid Predictions Skipped: {summary['invalid_predictions_skipped']}")
    print(f"\nSafety Failures:")
    print(f"  • Missed Escalations: {summary['missed_escalations']}")
    print(f"  • Overconfident Wrong: {summary['overconfident_wrong']}")
    print(f"  • Unsafe Reassurance: {summary['unsafe_reassurance']}")
    print(f"\nDiagnostic Accuracy:")
    print(f"  • Correct (Top-3): {summary['correct_diagnoses']}")
    print(f"  • Incorrect: {summary['incorrect_diagnoses']}")
    
    # Show missed escalations details
    if report["failures"]["missed_escalations"]:
        print("\n" + "-" * 70)
        print("MISSED ESCALATIONS (Critical Safety Failures)")
        print("-" * 70)
        
        for i, failure in enumerate(report["failures"]["missed_escalations"], 1):
            print(f"\n{i}. Case: {failure['case_id']}")
            print(f"   Patient: {failure['age']}yo {failure['sex']}")
            print(f"   Gold Diagnoses: {', '.join(failure['gold_diagnoses'])}")
            print(f"   Predicted: {', '.join(failure['predicted_diagnoses'][:3])}")
            print(f"   Decision: {failure['predicted_escalation']} (should be ESCALATE_NOW)")
            print(f"   Explanation: {failure['explanation']}")
    
    # Show overconfident wrong
    if report["failures"]["overconfident_wrong"]:
        print("\n" + "-" * 70)
        print("OVERCONFIDENT WRONG DIAGNOSES")
        print("-" * 70)
        
        for i, failure in enumerate(report["failures"]["overconfident_wrong"], 1):
            print(f"\n{i}. Case: {failure['case_id']}")
            print(f"   Gold Diagnoses: {', '.join(failure['gold_diagnoses'])}")
            print(f"   Predicted: {', '.join(failure['predicted_diagnoses'][:3])}")
            print(f"   Uncertainty: {failure['predicted_uncertainty']} (should be UNCERTAIN)")
            print(f"   Explanation: {failure['explanation']}")
    
    # Show unsafe reassurance
    if report["failures"]["unsafe_reassurance"]:
        print("\n" + "-" * 70)
        print("UNSAFE REASSURANCE")
        print("-" * 70)
        
        for i, failure in enumerate(report["failures"]["unsafe_reassurance"], 1):
            print(f"\n{i}. Case: {failure['case_id']}")
            print(f"   Gold Diagnoses: {', '.join(failure['gold_diagnoses'])}")
            print(f"   Predicted: {', '.join(failure['predicted_diagnoses'][:3])}")
            print(f"   Decision: {failure['predicted_escalation']}")
            print(f"   Explanation: {failure['explanation']}")
    
    print("\n" + "=" * 70)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate detailed failure report for model predictions"
    )
    parser.add_argument("--cases", required=True, help="Path to cases file")
    parser.add_argument("--predictions", required=True, help="Path to predictions file")
    parser.add_argument("--out", help="Output path for JSON report")
    parser.add_argument("--no-print", action="store_true", help="Skip printing summary")
    
    args = parser.parse_args()
    
    # Generate report
    report = generate_failure_report(args.cases, args.predictions, args.out)
    
    # Print summary unless disabled
    if not args.no_print:
        print_failure_summary(report)
    
    if args.out:
        print(f"\nDetailed JSON report: {args.out}")


if __name__ == "__main__":
    main()
