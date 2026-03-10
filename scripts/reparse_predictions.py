#!/usr/bin/env python3
"""
Re-parse prediction files to recover entries that failed due to // comments in JSON.

Usage:
    python3 scripts/reparse_predictions.py \
        --input  results/artifacts/openai-gpt-4o-500cases.json \
        --output results/artifacts/openai-gpt-4o-500cases.json   # overwrite in place

The script attempts to recover ``json_parse_failure`` entries that have a
``raw_response`` by running the same improved parse pipeline used in
``inference/run_inference.py`` (markdown fence stripping + // comment stripping).
``api_failure`` entries (raw_response is None/missing) cannot be recovered and
are left untouched.
"""

import re
import sys
import json
import argparse
from pathlib import Path


def strip_json_comments(text: str) -> str:
    """
    Strip single-line // comments from a JSON string, respecting string literals.

    Character-level state machine so it never mistakes a ``//`` inside a quoted
    string value (e.g. a URL) for a comment.
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
                # Skip until end of line, preserve the newline itself.
                while i < n and text[i] != "\n":
                    i += 1
                continue
            else:
                result.append(ch)

        i += 1
    return "".join(result)


def strip_trailing_commas(text: str) -> str:
    """Remove trailing commas before ] or } (models sometimes emit these)."""
    return re.sub(r",(\s*[}\]])", r"\1", text)


def clean_model_json(response: str) -> str:
    """Full cleaning pipeline: fences → comments → trailing commas."""
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


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_response(raw: str) -> dict | None:
    """
    Attempt to parse a raw model response into a dict.
    Returns None on failure.
    """
    if not raw:
        return None

    # 1. Try verbatim first.
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 2. Full cleaning pipeline: fences → comments → trailing commas.
    cleaned = clean_model_json(raw)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


# ── Main ─────────────────────────────────────────────────────────────────────

def reparse(input_path: str, output_path: str) -> None:
    src = Path(input_path)
    dst = Path(output_path)

    print(f"Loading {src} …")
    with open(src) as f:
        data = json.load(f)

    # Support both plain list and metadata-wrapped format.
    if isinstance(data, dict) and "predictions" in data:
        predictions = data["predictions"]
        metadata = data.get("metadata")
    else:
        predictions = data
        metadata = None

    recovered = 0
    still_failed = 0
    api_failures = 0

    new_predictions = []
    for pred in predictions:
        if not isinstance(pred, dict) or pred.get("error") != "json_parse_failure":
            new_predictions.append(pred)
            continue

        # This entry previously failed JSON parsing.
        raw = pred.get("raw_response")
        if not raw:
            # api_failure or truncated with no raw text — can't recover.
            api_failures += 1
            new_predictions.append(pred)
            continue

        parsed = _parse_response(raw)
        if parsed is None:
            still_failed += 1
            new_predictions.append(pred)
            continue

        # Restore important metadata from original failed entry.
        parsed["case_id"]           = pred["case_id"]
        parsed["workflow"]          = pred.get("workflow", "intake")
        parsed["raw_response"]      = raw
        if "input_decode_audit" in pred:
            parsed["input_decode_audit"] = pred["input_decode_audit"]

        new_predictions.append(parsed)
        recovered += 1

    print(f"  Recovered : {recovered}")
    print(f"  Still failed: {still_failed}")
    print(f"  API failures (no response): {api_failures}")
    print(f"  Total predictions: {len(new_predictions)}")

    # Reassemble the output in the same wrapper format as the input.
    if metadata is not None:
        out_data: dict | list = {"metadata": metadata, "predictions": new_predictions}
    else:
        out_data = new_predictions

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"  Written to {dst}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",  required=True, help="Path to predictions JSON")
    parser.add_argument("--output", required=True, help="Output path (may be same as input to overwrite)")
    args = parser.parse_args()

    reparse(args.input, args.output)


if __name__ == "__main__":
    main()
