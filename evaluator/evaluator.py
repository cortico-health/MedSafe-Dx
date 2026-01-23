import json
import sys
import traceback
import hashlib
import subprocess
from pathlib import Path
from evaluator.schemas import ModelPrediction, GoldCase
from evaluator.rules import evaluate_safety
from evaluator.metrics import MetricsAccumulator
from evaluator.harm import HarmWeights, harm_for_missing_or_invalid, harm_for_valid_prediction


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

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def try_git_commit() -> str | None:
    try:
        repo_dir = Path(__file__).resolve().parents[1]
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_dir)
        return out.decode("utf-8").strip()
    except Exception:
        return None


def evaluate(
    cases_path,
    predictions_path,
    model_name,
    model_version,
    harm_weights: HarmWeights | None = None,
    strict: bool = False,
):
    cases_sha256 = sha256_file(cases_path)
    predictions_sha256 = sha256_file(predictions_path)

    # Load cases (handle both plain list and metadata format)
    cases_data = load_json(cases_path)
    if isinstance(cases_data, dict) and "cases" in cases_data:
        cases_list = cases_data["cases"]
        cases_metadata = cases_data.get("metadata") if isinstance(cases_data.get("metadata"), dict) else None
    else:
        cases_list = cases_data
        cases_metadata = None
    
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
        predictions_metadata = predictions_data.get("metadata") if isinstance(predictions_data.get("metadata"), dict) else None
    else:
        predictions_list = predictions_data
        predictions_metadata = None
    
    metrics = MetricsAccumulator()
    metrics.total_cases_expected = len(cases_list)
    metrics.total_predictions_in_file = len(predictions_list)
    hw = harm_weights or HarmWeights()
    
    # Index raw predictions by case_id, tracking duplicates and malformed entries
    predictions_by_case_id = {}
    duplicate_predictions = 0

    for p in predictions_list:
        if not isinstance(p, dict):
            metrics.format_failures += 1
            metrics.format_failure_details.append(
                {"case_id": "unknown", "error": "Prediction is not an object", "traceback": ""}
            )
            continue

        case_id = p.get("case_id")
        if not case_id:
            metrics.format_failures += 1
            metrics.format_failure_details.append(
                {"case_id": "unknown", "error": "Missing case_id", "traceback": ""}
            )
            continue

        if case_id in predictions_by_case_id:
            duplicate_predictions += 1
            continue

        predictions_by_case_id[case_id] = p

    metrics.duplicate_predictions = duplicate_predictions

    expected_case_ids = list(gold_cases.keys())
    missing_case_ids = [cid for cid in expected_case_ids if cid not in predictions_by_case_id]
    for cid in missing_case_ids:
        metrics.add_missing_prediction(cid)
        metrics.add_harm(harm_for_missing_or_invalid(gold_cases[cid], hw))

    extra_case_ids = [cid for cid in predictions_by_case_id.keys() if cid not in gold_cases]
    metrics.extra_predictions = len(extra_case_ids)

    try:
        for cid in expected_case_ids:
            raw_pred = predictions_by_case_id.get(cid)
            if not raw_pred:
                continue

            try:
                pred = ModelPrediction(**raw_pred)
            except Exception as e:
                metrics.format_failures += 1
                error_msg = str(e)
                error_trace = traceback.format_exc()
                metrics.format_failure_details.append(
                    {"case_id": cid, "error": error_msg, "traceback": error_trace}
                )
                print(f"WARNING: Format failure for case {cid}: {error_msg}", file=sys.stderr)
                metrics.add_harm(harm_for_missing_or_invalid(gold_cases[cid], hw))
                continue

            gold = gold_cases[pred.case_id]

            safety = evaluate_safety(pred, gold)
            metrics.add_safety(safety)
            metrics.add_harm(harm_for_valid_prediction(pred, gold, safety, hw))

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

    benchmark_version = None
    if cases_metadata and isinstance(cases_metadata.get("test_set_name"), str):
        benchmark_version = cases_metadata.get("test_set_name")
    elif "-v" in str(cases_path):
        benchmark_version = Path(cases_path).name

    artifact = {
        "model": model_name,
        "version": model_version,
        "cases": metrics.total_cases_expected,
        "valid_predictions": metrics.total_evaluated_cases,
        "total_attempted": len(predictions_list),  # Total including malformed entries
        "strict": bool(strict),
        "benchmark_version": benchmark_version,
        "cases_path": str(cases_path),
        "predictions_path": str(predictions_path),
        "cases_sha256": cases_sha256,
        "predictions_sha256": predictions_sha256,
        "cases_metadata": cases_metadata,
        "predictions_metadata": predictions_metadata,
        "git_commit": try_git_commit(),
        **metrics.summary(),
    }

    # Reference baseline for human-friendly presentation:
    # "Always escalate + always uncertain" (valid output). This eliminates missed escalations and
    # overconfidence failures but maximizes over-escalation. It is a good anti-gaming reference.
    n = metrics.total_cases_expected
    nonurgent = sum(1 for cid in expected_case_ids if not gold_cases[cid].escalation_required)
    harm_ref_total = nonurgent * (hw.over_escalation_patient + hw.over_escalation_system)
    harm_ref_mean = (harm_ref_total / n) if n > 0 else None

    expected_harm_mean = artifact.get("expected_harm")
    rel_reduction_pct = None
    if harm_ref_mean is not None and harm_ref_mean > 0 and expected_harm_mean is not None:
        rel_reduction_pct = 100.0 * (harm_ref_mean - expected_harm_mean) / harm_ref_mean

    artifact["harm_weights"] = hw.to_dict()
    artifact["harm_reference_policy"] = "always_escalate_now_and_uncertain"
    artifact["harm_reference_expected_harm"] = harm_ref_mean
    artifact["relative_harm_reduction_pct"] = rel_reduction_pct

    if strict:
        issues = []
        if metrics.missing_predictions:
            issues.append(f"missing_predictions={metrics.missing_predictions}")
        if metrics.format_failures:
            issues.append(f"format_failures={metrics.format_failures}")
        if metrics.extra_predictions:
            issues.append(f"extra_predictions={metrics.extra_predictions}")
        if metrics.duplicate_predictions:
            issues.append(f"duplicate_predictions={metrics.duplicate_predictions}")
        if issues:
            raise ValueError("Strict mode failed: " + ", ".join(issues))

    return artifact
