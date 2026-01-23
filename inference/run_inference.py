#!/usr/bin/env python3
"""
Run inference on benchmark cases using OpenRouter API.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any
import time

from inference.openrouter import call_openrouter, load_cases, write_predictions
from inference.prompt import (
    OUTPUT_SCHEMA_V2,
    SYSTEM_PROMPT_CHART_REVIEW_V3,
    SYSTEM_PROMPT_INTAKE_V3,
    USER_PROMPT_TEMPLATE_CHART_REVIEW_V3,
    USER_PROMPT_TEMPLATE_INTAKE_V3,
)
from inference.symptom_decoder import decode_symptoms


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
    active_symptoms, antecedents = decode_symptoms(symptom_codes)
    
    symptoms_str = ", ".join(active_symptoms) if active_symptoms else "none"
    history_str = ", ".join(antecedents) if antecedents else "none"
    
    # Decode red flags (if any)
    red_flag_codes = case.get("red_flag_indicators", [])
    # Red flags are typically active symptoms, so we take the first part of the return tuple
    # Note: decode_symptoms returns (active, antecedents), we just join them all for red flags
    rf_active, rf_history = decode_symptoms(red_flag_codes) if red_flag_codes else ([], [])
    decoded_red_flags = rf_active + rf_history
    red_flags_str = ", ".join(decoded_red_flags) if decoded_red_flags else "none"
    
    return get_user_prompt_template(workflow).format(
        age=case.get("age", "unknown"),
        sex=case.get("sex", "unknown"),
        symptoms=symptoms_str,
        history=history_str,
        duration=case.get("symptom_duration", "unknown"),
        severity=case.get("severity_flags", "unknown"),
        red_flags=red_flags_str,
        schema=OUTPUT_SCHEMA_V2,
    )


def run_inference_on_case(
    case: Dict[str, Any],
    model: str,
    workflow: str,
    temperature: float = 0.0,
) -> Dict[str, Any] | None:
    """Run inference on a single case."""
    
    messages = [
        {"role": "system", "content": get_system_prompt(workflow)},
        {"role": "user", "content": format_case_for_prompt(case, workflow)},
    ]
    
    response = call_openrouter(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=2000,
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
        # Check if response is just plain JSON text first
        try:
             prediction = json.loads(response)
        except json.JSONDecodeError:
            # Extract JSON if wrapped in markdown code blocks or has extra text
            cleaned_response = response
            if "```json" in response:
                cleaned_response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                cleaned_response = response.split("```")[1].split("```")[0].strip()
            # If no markdown blocks, try to find the first '{' and last '}'
            elif "{" in response and "}" in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                cleaned_response = response[start:end]
            
            prediction = json.loads(cleaned_response)
        
        prediction["case_id"] = case["case_id"]
        # Keep raw text for clinical review / audit. Evaluator ignores extra fields.
        prediction["raw_response"] = response
        prediction["workflow"] = workflow
        return prediction
    
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON for case {case['case_id']}: {e}")
        print(f"Response: {response[:200]}")
        return {
            "case_id": case["case_id"],
            "workflow": workflow,
            "error": "json_parse_failure",
            "raw_response": response,
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
    
    predictions = []
    successful = 0
    
    for i, case in enumerate(cases):
        if i % 10 == 0:
            print(f"Progress: {i}/{len(cases)}")
        
        prediction = run_inference_on_case(
            case,
            model=args.model,
            workflow=args.workflow,
            temperature=args.temperature,
        )

        predictions.append(prediction)
        if isinstance(prediction, dict) and "error" not in prediction:
            successful += 1
        
        # Rate limiting: sleep briefly between requests
        time.sleep(0.5)
    
    # Write predictions with metadata
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Build output metadata
    output_metadata = {
        "model": args.model,
        "temperature": args.temperature,
        "workflow": args.workflow,
        "prompt_version": "v3",
        "total_cases": len(predictions),
        "successful_predictions": successful,
        "failed_predictions": len(predictions) - successful,
    }
    
    # Include input test set metadata if available
    if metadata:
        output_metadata["test_set_metadata"] = metadata
    
    write_predictions(output_path, predictions, output_metadata)
    
    print(f"\nCompleted!")
    print(f"Successful: {successful}")
    print(f"Failed: {len(predictions) - successful}")
    print(f"Predictions written to: {output_path}")


if __name__ == "__main__":
    main()
