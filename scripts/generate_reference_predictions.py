#!/usr/bin/env python3
"""
Generate a deterministic reference predictions file for benchmarking.

This is primarily used to make the "paranoid baseline" policy reproducible:
- always set escalation_decision = ESCALATE_NOW
- always set uncertainty = UNCERTAIN
- always emit a valid JSON prediction with 5 ICD-10 codes

Note: the differential codes are placeholders and are not intended to be clinically meaningful.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


PLACEHOLDER_CODES = ["R69", "Z000", "R6889", "Z711", "Z7189"]


def load_cases(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "cases" in data:
        return data["cases"]
    if isinstance(data, list):
        return data
    raise ValueError("Cases file must be a list or an object with 'cases'")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate reference predictions")
    parser.add_argument(
        "--cases",
        required=True,
        help="Path to cases JSON (e.g., data/test_sets/test-v0.json)",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output path for predictions JSON",
    )
    args = parser.parse_args()

    cases_path = Path(args.cases)
    out_path = Path(args.out)

    cases = load_cases(cases_path)
    predictions = []
    for c in cases:
        case_id = c.get("case_id")
        if not case_id:
            continue
        predictions.append(
            {
                "case_id": case_id,
                "differential_diagnoses": [{"code": code} for code in PLACEHOLDER_CODES],
                "escalation_decision": "ESCALATE_NOW",
                "uncertainty": "UNCERTAIN",
            }
        )

    payload = {
        "metadata": {
            "reference_policy": "always_escalate_now_and_uncertain",
            "placeholder_codes": PLACEHOLDER_CODES,
            "cases_path": str(cases_path),
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        },
        "predictions": predictions,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"✓ Wrote {len(predictions)} predictions to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
