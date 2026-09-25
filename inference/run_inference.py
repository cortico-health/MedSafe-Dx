#!/usr/bin/env python3
"""
Run inference on benchmark cases using OpenRouter API.
"""

import json
import os
import re
import argparse
import copy
from pathlib import Path
from typing import Dict, Any
import fcntl
import sys
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

from inference.openrouter import call_openrouter, call_openrouter_detailed, load_cases, write_predictions
from inference.codex_cli import call_codex
from inference.prompt import (
    OUTPUT_SCHEMA_V4,
    OUTPUT_SCHEMA_V5,
    OUTPUT_SCHEMA_V6,
    SYSTEM_PROMPT_CHART_REVIEW_V3,
    SYSTEM_PROMPT_INTAKE_V3,
    SYSTEM_PROMPT_INTAKE_V5,
    SYSTEM_PROMPT_V6,
    USER_PROMPT_TEMPLATE_CHART_REVIEW_V3,
    USER_PROMPT_TEMPLATE_INTAKE_V3,
    USER_PROMPT_TEMPLATE_INTAKE_V5,
    USER_PROMPT_TEMPLATE_V6,
)
from inference.symptom_decoder import decode_symptoms, decode_symptoms_with_audit, decode_symptoms_versioned

DEFAULT_RUN_CONFIG_V02 = Path(__file__).parent / "run_config_v02.json"
DEFAULT_RUN_CONFIG_V03 = Path(__file__).parent / "run_config_v03.json"
DEFAULT_RUN_CONFIG = {"v5": DEFAULT_RUN_CONFIG_V02, "v6": DEFAULT_RUN_CONFIG_V03}


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


# ---------------------------------------------------------------------------
# v0.2 (prompt v5): prompt, run configuration and per-prediction metadata
# ---------------------------------------------------------------------------


def format_case_for_prompt_v5(case: Dict[str, Any], decoder_version: str = "v02") -> str:
    """Render one case as the v5 user prompt, with the v0.2 decoder by default."""
    active, antecedents, audit = decode_symptoms_versioned(
        case.get("presenting_symptoms", []), version=decoder_version, sex=case.get("sex")
    )
    case["_input_decode_audit"] = {"symptoms": audit}
    return USER_PROMPT_TEMPLATE_INTAKE_V5.format(
        age=case.get("age", "unknown"),
        sex=case.get("sex", "unknown"),
        symptoms=", ".join(active) if active else "none",
        history=", ".join(antecedents) if antecedents else "none",
        schema=OUTPUT_SCHEMA_V5,
    )


def build_messages_v5(case: Dict[str, Any], decoder_version: str = "v02") -> list[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT_INTAKE_V5},
        {"role": "user", "content": format_case_for_prompt_v5(case, decoder_version)},
    ]


def format_case_for_prompt_v6(case: Dict[str, Any], decoder_version: str = "v02") -> str:
    """Render one case as the v6 user prompt: the v5 intake rendering with the v6 schema."""
    active, antecedents, audit = decode_symptoms_versioned(
        case.get("presenting_symptoms", []), version=decoder_version, sex=case.get("sex")
    )
    case["_input_decode_audit"] = {"symptoms": audit}
    return USER_PROMPT_TEMPLATE_V6.format(
        age=case.get("age", "unknown"),
        sex=case.get("sex", "unknown"),
        symptoms=", ".join(active) if active else "none",
        history=", ".join(antecedents) if antecedents else "none",
        schema=OUTPUT_SCHEMA_V6,
    )


def build_messages_v6(case: Dict[str, Any], decoder_version: str = "v02") -> list[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT_V6},
        {"role": "user", "content": format_case_for_prompt_v6(case, decoder_version)},
    ]


def build_messages_for(case: Dict[str, Any], settings: Dict[str, Any]) -> list[Dict[str, str]]:
    """Messages for the run's prompt version (v5 or v6), as the run config names it."""
    build = build_messages_v6 if settings.get("prompt_version") == "v6" else build_messages_v5
    return build(case, settings["decoder_version"])


def load_run_config(path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def resolve_run_settings(
    config: Dict[str, Any],
    model: str,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
    temperature: float | None = None,
) -> Dict[str, Any]:
    """
    Settings for one model under the v0.2 run config. A CLI value that differs from
    the config, or a model the config does not list, sets config_overridden, so the
    row carries a flag (spec section 9: same configuration for every row, or a flag).
    """
    overridden = []
    models = config.get("models", {})
    if model in models:
        effort = models[model].get("reasoning_effort")
    else:
        effort = config.get("default_reasoning_effort")
        overridden.append("model_not_in_config")
    settings = {
        "config_version": config.get("config_version"),
        "prompt_version": config.get("prompt_version", "v5"),
        "decoder_version": config.get("decoder_version", "v02"),
        "temperature": config.get("temperature", 0.0),
        "max_tokens": config["max_tokens"],
        "reasoning_effort": effort,
        "empty_or_truncated_retries": int(config.get("empty_or_truncated_retries", 1)),
    }
    for key, value in (("max_tokens", max_tokens), ("reasoning_effort", reasoning_effort), ("temperature", temperature)):
        if value is not None and value != settings[key]:
            settings[key] = value
            overridden.append(key)
    settings["config_overridden"] = overridden
    return settings


def _needs_retry(meta: Dict[str, Any]) -> str | None:
    """Reason to retry a response under the v0.2 rule, or None."""
    if not (meta.get("content") or "").strip():
        return "empty"
    if meta.get("finish_reason") == "length":
        return "truncated"
    return None


def call_model_v5(messages, model: str, settings: Dict[str, Any], backend: str = "openrouter") -> tuple[Dict[str, Any], list]:
    """
    Call the model with one retry on an empty or truncated response (settings
    empty_or_truncated_retries). Returns (final attempt metadata, all attempts).
    """
    attempts = []
    for i in range(settings["empty_or_truncated_retries"] + 1):
        if backend == "codex":
            text = call_codex(model=model, messages=messages, reasoning_effort=settings["reasoning_effort"] or "medium")
            meta = {"content": text, "finish_reason": None, "native_finish_reason": None, "provider": "codex_cli",
                    "request_id": None, "model_served": None, "usage": None, "error": None if text else "empty_content"}
        else:
            meta = call_openrouter_detailed(
                model=model,
                messages=messages,
                temperature=settings["temperature"],
                max_tokens=settings["max_tokens"],
                reasoning_effort=settings["reasoning_effort"],
                empty_content_retries=0,
            )
        reason = _needs_retry(meta)
        # A failed request (HTTP error after the transport retries) is not a response,
        # so it does not use the empty-or-truncated retry.
        if meta.get("error") not in (None, "empty_content"):
            reason = None
        attempts.append({k: v for k, v in meta.items() if k != "content"} | {"problem": reason})
        if reason is None:
            break
    return meta, attempts


def run_inference_on_case_v5(
    case: Dict[str, Any],
    model: str,
    settings: Dict[str, Any],
    backend: str = "openrouter",
) -> Dict[str, Any]:
    """Run one case with prompt v5 or v6 (settings["prompt_version"]) and record the response metadata beside the prediction."""
    messages = build_messages_for(case, settings)
    meta, attempts = call_model_v5(messages, model, settings, backend)
    record = {
        "finish_reason": meta.get("finish_reason"),
        "native_finish_reason": meta.get("native_finish_reason"),
        "provider": meta.get("provider"),
        "request_id": meta.get("request_id"),
        "model_served": meta.get("model_served"),
        "usage": meta.get("usage"),
        "attempts": attempts,
        "prompt_version": settings["prompt_version"],
        "decoder_version": settings["decoder_version"],
        "input_decode_audit": case.get("_input_decode_audit"),
    }
    response = meta.get("content")
    base = {"case_id": case["case_id"], "workflow": "intake"}
    if not (response or "").strip():
        return base | {"error": meta.get("error") or "api_failure", "raw_response": response} | record
    if meta.get("finish_reason") == "length":
        # Truncated after the retry: keep the text and try to parse it, but mark it.
        record["truncated"] = True
    try:
        try:
            prediction = json.loads(response)
        except json.JSONDecodeError:
            prediction = json.loads(clean_model_json(response))
        if not isinstance(prediction, dict):
            raise json.JSONDecodeError("top-level JSON is not an object", response, 0)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON for case {case['case_id']}: {e}")
        return base | {"error": "json_parse_failure", "raw_response": response} | record
    return prediction | base | {"raw_response": response} | record


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


def write_dry_run(out: Path, cases, model: str, settings: Dict[str, Any], metadata, run_config: str) -> Path:
    """Render each v5 request body exactly as call_openrouter_detailed would send it, with no API call.

    We write <out>.dryrun.json, never <out>, so a dry run cannot be mistaken for
    predictions. The token counts are estimates (characters / 4), for cost planning only.
    """
    from inference.openrouter import build_payload

    requests_out = []
    for case in cases:
        messages = build_messages_for(copy.deepcopy(case), settings)
        payload = build_payload(model, messages, settings["temperature"], settings["max_tokens"],
                                settings["reasoning_effort"])
        chars = sum(len(m["content"]) for m in messages)
        requests_out.append({"case_id": case["case_id"], "prompt_chars": chars,
                             "est_prompt_tokens": round(chars / 4), "payload": payload})
    path = out.with_name(out.stem + ".dryrun.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(requests_out)
    meta = {
        "model": model, "dry_run": True, "cases": n,
        **{k: settings[k] for k in ("prompt_version", "decoder_version", "temperature", "max_tokens",
                                     "reasoning_effort", "empty_or_truncated_retries", "config_version",
                                     "config_overridden")},
        "run_config_path": run_config,
        "est_prompt_tokens_total": sum(r["est_prompt_tokens"] for r in requests_out),
        "est_prompt_tokens_mean": round(sum(r["est_prompt_tokens"] for r in requests_out) / max(n, 1), 1),
        "token_estimate_rule": "characters / 4 over system and user messages",
        "test_set_metadata": metadata,
    }
    with open(path, "w") as f:
        json.dump({"metadata": meta, "requests": requests_out}, f, indent=1)
    print(f"Dry run: rendered {n} request(s) to {path}; no API call made. "
          f"Estimated prompt tokens: {meta['est_prompt_tokens_mean']} per case, {meta['est_prompt_tokens_total']} total")
    return path


def acquire_output_lock(output_path: Path):
    """Take an exclusive lock on <output>.lock, or exit if another run holds it."""
    lock_path = output_path.with_name(output_path.name + ".lock")
    fd = open(lock_path, "w")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(
            f"ERROR: another inference run is writing {output_path} (lock {lock_path} is held). "
            "Wait for it to finish; do not start a second run on the same output.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    fd.write(f"pid {os.getpid()}\n")
    fd.flush()
    return fd


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
        default=None,
        help="Output token cap (v4 default 2000; v5 takes it from the run config, 16000). Reasoning models spend this budget on chain-of-thought first; a 2000 cap can starve content and cause spurious format failures (see docs/RUNS.md 2026-09-08). Under v5 a value that differs from the config flags the row.",
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
        default=None,
        help="v4: codex backend only, model_reasoning_effort (low|medium|high|xhigh; default medium). "
        "v5: taken per model from the run config; a different value here flags the row.",
    )
    parser.add_argument(
        "--prompt-version",
        choices=["v4", "v5", "v6"],
        default="v4",
        help="v4 (v0 benchmark, default), v5 (v0.2: probabilities, p_serious, v0.2 decoder, run config) "
        "or v6 (v0.3: serious_concern, flags, differential, p_serious; run_config_v03.json)",
    )
    parser.add_argument(
        "--run-config",
        default=None,
        help="v5 and v6: run configuration JSON (max_tokens, reasoning effort per model, retry rule); "
        "default run_config_v02.json for v5 and run_config_v03.json for v6",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="v5 only: render every request body to <out>.dryrun.json and exit without calling any API",
    )

    args = parser.parse_args()

    v5_settings = None
    if args.prompt_version in ("v5", "v6"):
        if args.workflow != "intake":
            parser.error(f"--prompt-version {args.prompt_version} supports the intake workflow only")
        if args.run_config is None:
            args.run_config = str(DEFAULT_RUN_CONFIG[args.prompt_version])
        v5_settings = resolve_run_settings(
            load_run_config(args.run_config),
            args.model,
            max_tokens=args.max_tokens,
            reasoning_effort=args.reasoning_effort,
            temperature=args.temperature,
        )
        if v5_settings["prompt_version"] != args.prompt_version:
            parser.error(f"{args.run_config} is for prompt {v5_settings['prompt_version']}, not {args.prompt_version}")
        if v5_settings["config_overridden"]:
            print(f"WARNING: run config overridden ({', '.join(v5_settings['config_overridden'])}); the row will carry a flag")
    else:
        if args.max_tokens is None:
            args.max_tokens = 2000
        if args.reasoning_effort is None:
            args.reasoning_effort = "medium"
    
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
    
    if args.dry_run:
        if v5_settings is None:
            parser.error("--dry-run supports --prompt-version v5 and v6 only")
        write_dry_run(Path(args.out), cases, args.model, v5_settings, metadata, args.run_config)
        return

    print(f"Running inference on {len(cases)} cases...")

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # One writer per output file. Two runs on the same --out each keep their own
    # prediction list and overwrite the file at every checkpoint, so the last one
    # to finish silently replaces the file the other run's evaluator already hashed
    # (GPT-5.4 Mini and GPT-5.6 Luna, run-2026-09). We hold an exclusive lock for
    # the whole run; a second run exits, and evaluator.cli refuses to score while
    # the lock is held.
    lock_fd = acquire_output_lock(output_path)  # noqa: F841 - held until exit

    def build_output_metadata(preds, n_successful):
        if v5_settings is not None:
            m = {
                "model": args.model,
                "workflow": args.workflow,
                "backend": args.backend,
                **{k: v5_settings[k] for k in (
                    "prompt_version", "decoder_version", "temperature", "max_tokens", "reasoning_effort",
                    "empty_or_truncated_retries", "config_version", "config_overridden",
                )},
                "run_config_path": args.run_config,
                "total_cases": len(preds),
                "successful_predictions": n_successful,
                "failed_predictions": len(preds) - n_successful,
                "git_commit": os.environ.get("MEDSAFE_GIT_COMMIT") or None,
            }
            if metadata:
                m["test_set_metadata"] = metadata
            return m
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
                and existing_metadata.get("prompt_version", "v4") == args.prompt_version
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
        if v5_settings is not None:
            pred = run_inference_on_case_v5(case, model=args.model, settings=v5_settings, backend=args.backend)
            time.sleep(0.5)
            return pred
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
