import json
import sys
import traceback
from evaluator.schemas import ModelPrediction, GoldCase
from evaluator.rules import evaluate_safety
from evaluator.metrics import MetricsAccumulator


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"CRITICAL ERROR: JSON parse failure in {path}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load {path}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


def evaluate(cases_path, predictions_path, model_name, model_version):
    # Load cases (handle both plain list and metadata format)
    cases_data = load_json(cases_path)
    if isinstance(cases_data, dict) and "cases" in cases_data:
        cases_list = cases_data["cases"]
    else:
        cases_list = cases_data
    
    try:
        gold_cases = {
            c["case_id"]: GoldCase(**c)
            for c in cases_list
        }
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to parse gold cases from {cases_path}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    # Load predictions (handle both plain list and metadata format)
    predictions_data = load_json(predictions_path)
    if isinstance(predictions_data, dict) and "predictions" in predictions_data:
        predictions_list = predictions_data["predictions"]
    else:
        predictions_list = predictions_data
    
    # Parse predictions one by one, tracking format failures
    predictions = []
    metrics = MetricsAccumulator()
    
    for p in predictions_list:
        try:
            predictions.append(ModelPrediction(**p))
        except Exception as e:
            # Track format failure
            metrics.format_failures += 1
            case_id = p.get("case_id", "unknown")
            error_msg = str(e)
            error_trace = traceback.format_exc()
            
            metrics.format_failure_details.append({
                "case_id": case_id,
                "error": error_msg,
                "traceback": error_trace
            })
            
            print(f"WARNING: Format failure for case {case_id}: {error_msg}", file=sys.stderr)
            # Continue processing other predictions

    try:
        for pred in predictions:
            gold = gold_cases[pred.case_id]

            safety = evaluate_safety(pred, gold)
            metrics.add_safety(safety)

            if not safety.failed:
                predicted_codes = [d.code for d in pred.differential_diagnoses]
                metrics.add_effectiveness(predicted_codes, gold.gold_top3)
    except KeyError as e:
        print(f"CRITICAL ERROR: Case ID mismatch - prediction references non-existent case: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"CRITICAL ERROR: Evaluation failed", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    artifact = {
        "model": model_name,
        "version": model_version,
        "cases": len(predictions),  # Successfully parsed cases
        "total_attempted": len(predictions_list),  # Total including format failures
        **metrics.summary(),
    }

    return artifact
