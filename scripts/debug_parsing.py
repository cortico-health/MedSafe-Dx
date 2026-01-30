#!/usr/bin/env python3
"""Debug parsing issues in MOE results."""

import json
from pathlib import Path

raw_path = Path("results/analysis/moe_panel/moe_raw_results.json")

with open(raw_path) as f:
    raw_results = json.load(f)

# Check how many consensus responses failed to parse
n_parsed = 0
n_failed = 0
parse_failures = []

for result in raw_results:
    consensus = result.get("consensus", {})
    parsed = consensus.get("parsed")
    if parsed:
        n_parsed += 1
    else:
        n_failed += 1
        parse_failures.append({
            "case_id": result["case_id"],
            "raw_response": consensus.get("raw_response", "")[:500],
        })

print(f"Parsed successfully: {n_parsed}")
print(f"Failed to parse: {n_failed}")

print("\n=== SAMPLE PARSE FAILURES ===")
for failure in parse_failures[:3]:
    print(f"\nCase: {failure['case_id']}")
    print(f"Raw response:\n{failure['raw_response']}")
    print("-" * 60)
