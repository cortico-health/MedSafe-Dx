#!/usr/bin/env python3
"""
Run inference on benchmark cases using OpenRouter API.
"""

import json
import os
import re
import argparse
from pathlib import Path
from typing import Dict, Any
import time
from concurrent.futures import ThreadPoolExecutor

CHECKPOINT_EVERY = 10


def strip_json_comments(text: str) -> str:
    """
    Strip single-line // comments from a JSON string, respecting string literals.

    This uses a character-level state machine so it never mistakes a ``//``
    inside a quoted string value (e.g. a URL) for a comment.  Handles:
      • ``// comment`` outside strings → removed (up to but not including \\n)
      • ``\\"`` inside strings → treated as an escaped quote, does not toggle state
      • Does NOT attempt to handle block comments (/* … */) because the models
        we are targeting never emit them.
    """
    result = []
    in_string = False
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]

        if in_string:
            result.append(ch)
            if ch == "\\":
                # Escaped character – consume next char verbatim so \" doesn't
                # accidentally close the string.
                i += 1
                if i < n:
                    result.append(text[i])
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
                result.append(ch)
            elif ch == "/" and i + 1 < n and text[i + 1] == "/":
                # Line comment – skip until end-of-line (keep the newline itself
                # so line numbers stay intact for debugging).
                while i < n and text[i] != "\n":
                    i += 1
                continue  # don't increment i again below
            else:
                result.append(ch)

        i += 1
    return "".join(result)


def strip_trailing_commas(text: str) -> str:
    """
    Remove trailing commas before ``]`` or ``}`` in a JSON string.

    Models occasionally emit a trailing comma after the last item in an array
    (e.g. ``{"code": "R51"},  // comment\n  ]``).  After comment stripping the
    comment disappears but the comma remains, producing invalid JSON.  A simple
    regex handles this safely because JSON values cannot themselves start with
    ``,``.
    """
    return re.sub(r",(\s*[}\]])", r"\1", text)


def clean_model_json(response: str) -> str:
    """
    Apply the full cleaning pipeline to a raw model response:
      1. Strip markdown fences (```json … ```)
      2. Fall back to first ``{`` … last ``}`` if no fences
      3. Strip ``//`` line comments
      4. Strip trailing commas before ``]`` / ``}``
    """
    cleaned = response
    if "```json" in response:
        cleaned = response.split("```json")[1].split("```")[0].strip()
    elif "```" in response:
        cleaned = response.split("```")[1].split("```")[0].strip()
    elif "{" in response and "}" in response:
        start = response.find("{")
        end   = response.rfind("}") + 1
        cleaned = response[start:end]

    cleaned = strip_json_comments(cleaned)
    cleaned = strip_trailing_commas(cleaned)
    return cleaned

from inference.openrouter import call_openrouter, load_cases, write_predictions
from inference.codex_cli import call_codex
from inference.prompt import (
    OUTPUT_SCHEMA_V4,
    SYSTEM_PROMPT_CHART_REVIEW_V3,
    SYSTEM_PROMPT_INTAKE_V3,
    USER_PROMPT_TEMPLATE_CHART_REVIEW_V3,
    USER_PROMPT_TEMPLATE_INTAKE_V3,
)
from inference.symptom_decoder import decode_symptoms, decode_symptoms_with_audit


def get_system_prompt(workflow: str) -> str:
    if workflow == "chart_review":
        return SYSTEM_PROMPT_CHART_REVIEW_V3
    return SYSTEM_PROMPT_INTAKE_V3


def get_user_prompt_template(workflow: str) -> str:
    if workflow == "chart_review":
        return USER_PROMPT_TEMPLATE_CHART_REVIEW_V3
    return USER_PROMPT_TEMPLATE_INTAKE_V3


def format_case_for_prompt(case: Dict[str, Any], workflow: str) -> str:
    """Format a case into the user prompt with human-readable symptoms."""
    # Decode symptom codes to readable text
    symptom_codes = case.get("presenting_symptoms", [])
    active_symptoms, antecedents, symptoms_audit = decode_symptoms_with_audit(symptom_codes)
    
    symptoms_str = ", ".join(active_symptoms) if active_symptoms else "none"
    history_str = ", ".join(antecedents) if antecedents else "none"
    
    # Decode red flags (if any)
    red_flag_codes = case.get("red_flag_indicators", [])
    # Red flags are typically active symptoms, so we take the first part of the return tuple
    # Note: decode_symptoms returns (active, antecedents), we just join them all for red flags
    if red_flag_codes:
        rf_active, rf_history, red_flags_audit = decode_symptoms_with_audit(red_flag_codes)
    else:
        rf_active, rf_history, red_flags_audit = ([], [], None)
    decoded_red_flags = rf_active + rf_history
    red_flags_str = ", ".join(decoded_red_flags) if decoded_red_flags else "none"
    
    # Attach decode fidelity for downstream clinician QA/auditing (not used for scoring).
    case["_input_decode_audit"] = {
        "symptoms": symptoms_audit,
        "red_flags": red_flags_audit,
    }

    return get_user_prompt_template(workflow).format(
        age=case.get("age", "unknown"),
        sex=case.get("sex", "unknown"),
        symptoms=symptoms_str,
        history=history_str,
        duration=case.get("symptom_duration", "unknown"),
        severity=case.get("severity_flags", "unknown"),
        red_flags=red_flags_str,
        schema=OUTPUT_SCHEMA_V4,
    )


def run_inference_on_case(
    case: Dict[str, Any],
    model: str,
    workflow: str,
    temperature: float = 0.0,
    max_tokens: int = 2000,
    backend: str = "openrouter",
    reasoning_effort: str = "medium",
) -> Dict[str, Any] | None:
    """Run inference on a single case."""
    
    messages = [
        {"role": "system", "content": get_system_prompt(workflow)},
        {"role": "user", "content": format_case_for_prompt(case, workflow)},
    ]
    
    if backend == "codex":
        response = call_codex(model=model, messages=messages, reasoning_effort=reasoning_effort)
    else:
        response = call_openrouter(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    
    if not response:
        return {
            "case_id": case["case_id"],
            "workflow": workflow,
            "error": "api_failure",
            "raw_response": None,
        }
    
    # Try to parse JSON from response
    try:
        # Fast path: response is already valid JSON.
        try:
             prediction = json.loads(response)
        except json.JSONDecodeError:
            # Full cleaning pipeline: fence stripping → comment stripping →
            # trailing-comma removal.
            cleaned_response = clean_model_json(response)
            prediction = json.loads(cleaned_response)
        
        prediction["case_id"] = case["case_id"]
        # Keep raw text for clinical review / audit. Evaluator ignores extra fields.
        prediction["raw_response"] = response
        prediction["workflow"] = workflow
        if isinstance(case, dict) and case.get("_input_decode_audit"):
            prediction["input_decode_audit"] = case["_input_decode_audit"]
        return prediction
    
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON for case {case['case_id']}: {e}")
        print(f"Response: {response[:200]}")
        return {
            "case_id": case["case_id"],
            "workflow": workflow,
            "error": "json_parse_failure",
            "raw_response": response,
            "input_decode_audit": case.get("_input_decode_audit") if isinstance(case, dict) else None,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        default="data/ddxplus_v0/cases.json",
        help="Path to cases.json",
    )
    parser.add_argument(
        "--model",
        default="anthropic/claude-sonnet-4",
        help="Model name on OpenRouter",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output path for predictions.json",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of cases (for testing)",
    )
    parser.add_argument(
        "--workflow",
        choices=["intake", "chart_review"],
        default="intake",
        help="Workflow context to simulate (affects escalation framing)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2000,
        help="Output token cap. Reasoning models spend this budget on chain-of-thought first; a 2000 cap can starve content and cause spurious format failures (see docs/RUNS.md 2026-09-08).",
    )
    parser.add_argument(
        "--backend",
        choices=["openrouter", "codex"],
        default="openrouter",
        help="openrouter (API, default) or codex (local `codex exec`, ChatGPT plan; see inference/codex_cli.py)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("INFERENCE_WORKERS", "1")),
        help="concurrent cases per model (default 1, or env INFERENCE_WORKERS)",
    )
    parser.add_argument(
        "--reasoning-effort",
        default="medium",
        help="codex backend only: model_reasoning_effort (low|medium|high|xhigh)",
    )
    
    args = parser.parse_args()
    
    # Load cases
    print(f"Loading cases from {args.cases}...")
    cases, metadata = load_cases(args.cases)
    
    if metadata:
        print(f"Test set metadata:")
        if "test_set_name" in metadata:
            print(f"  Name: {metadata['test_set_name']}")
        if "seed" in metadata:
            print(f"  Seed: {metadata['seed']}")
        if "sampled_cases" in metadata:
            print(f"  Sampled: {metadata['sampled_cases']}/{metadata.get('total_available_cases', '?')}")
    
    if args.limit:
        cases = cases[:args.limit]
        print(f"Limited to {args.limit} cases")
    
    print(f"Running inference on {len(cases)} cases...")

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def build_output_metadata(preds, n_successful):
        m = {
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "workflow": args.workflow,
            "prompt_version": "v4",
            "backend": args.backend,
            "reasoning_effort": args.reasoning_effort if args.backend == "codex" else None,
            "total_cases": len(preds),
            "successful_predictions": n_successful,
            "failed_predictions": len(preds) - n_successful,
            "git_commit": os.environ.get("MEDSAFE_GIT_COMMIT") or None,
        }
        if metadata:
            m["test_set_metadata"] = metadata
        return m

    # Resume support: if the output file already exists and matches this
    # model/workflow/test set, keep its predictions and skip those case_ids
    # instead of re-running (and re-billing) them.
    predictions = []
    successful = 0
    existing_case_ids = set()

    if output_path.exists():
        try:
            with open(output_path) as f:
                existing_data = json.load(f)
            if isinstance(existing_data, dict) and "predictions" in existing_data:
                existing_predictions = existing_data["predictions"]
                existing_metadata = existing_data.get("metadata") or {}
            else:
                existing_predictions = existing_data if isinstance(existing_data, list) else []
                existing_metadata = {}

            same_run = (
                existing_metadata.get("model") == args.model
                and existing_metadata.get("workflow") == args.workflow
                and existing_metadata.get("test_set_metadata") == metadata
            )

            if same_run and existing_predictions:
                # Keep only usable predictions: entries that carry an "error"
                # (api_failure, json_parse_failure) are dropped so they get
                # re-run, e.g. after an HTTP 402 credit exhaustion mid-run.
                predictions = [
                    p for p in existing_predictions
                    if isinstance(p, dict) and "error" not in p
                ]
                n_retry = len(existing_predictions) - len(predictions)
                existing_case_ids = {p.get("case_id") for p in predictions}
                successful = len(predictions)
                print(
                    f"Resuming from {output_path}: {len(existing_case_ids)} usable case(s) "
                    f"already present for this model/test set, will skip those"
                    + (f"; {n_retry} errored case(s) will be re-run" if n_retry else "")
                )
            elif existing_predictions:
                print(
                    f"Existing output at {output_path} is for a different "
                    f"model/workflow/test set; starting fresh (will overwrite)"
                )
        except (json.JSONDecodeError, OSError) as e:
            print(f"Could not read existing output {output_path} for resume ({e}); starting fresh")

    since_checkpoint = 0
    pending = [c for c in cases if c.get("case_id") not in existing_case_ids]
    total = len(cases)

    def _infer(case):
        pred = run_inference_on_case(
            case,
            model=args.model,
            workflow=args.workflow,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            backend=args.backend,
            reasoning_effort=args.reasoning_effort,
        )
        # Rate limiting: brief pause per request (per worker)
        time.sleep(0.5)
        return pred

    # --workers N runs N cases concurrently. Reasoning models at max_tokens 16000
    # take 30-90 s per case, so a sequential 250-case pass is hours; 8 workers
    # brings it to minutes. Results are collected in case order per chunk and the
    # final write sorts by test-set order, so output is identical to sequential.
    workers = max(1, int(args.workers))
    done_count = len(existing_case_ids)
    print(f"Progress: {done_count}/{total}")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(pending), CHECKPOINT_EVERY):
            chunk = pending[start:start + CHECKPOINT_EVERY]
            for prediction in pool.map(_infer, chunk):
                predictions.append(prediction)
                if isinstance(prediction, dict) and "error" not in prediction:
                    successful += 1
            done_count += len(chunk)
            print(f"Progress: {done_count}/{total}")
            # Checkpoint every CHECKPOINT_EVERY new predictions so a crash doesn't
            # lose an entire run's worth of API calls.
            write_predictions(output_path, predictions, build_output_metadata(predictions, successful))

    # Final write with metadata, in test-set order (resumed runs append re-run cases at the end)
    order = {c.get("case_id"): i for i, c in enumerate(cases)}
    predictions.sort(key=lambda p: order.get(p.get("case_id") if isinstance(p, dict) else None, len(order)))
    output_metadata = build_output_metadata(predictions, successful)
    write_predictions(output_path, predictions, output_metadata)

    print(f"\nCompleted!")
    print(f"Successful: {successful}")
    print(f"Failed: {len(predictions) - successful}")
    print(f"Predictions written to: {output_path}")


if __name__ == "__main__":
    main()
