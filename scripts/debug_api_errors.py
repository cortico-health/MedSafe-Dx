#!/usr/bin/env python3
"""Check for API errors in MOE results."""

import json
from pathlib import Path
from collections import Counter

raw_path = Path("results/analysis/moe_panel/moe_raw_results.json")

with open(raw_path) as f:
    raw_results = json.load(f)

# Check consensus usage/errors
error_types = Counter()
empty_responses = 0
truncated_responses = 0

for result in raw_results:
    consensus = result.get("consensus", {})
    raw = consensus.get("raw_response", "")
    usage = consensus.get("usage", {})
    
    if not raw:
        empty_responses += 1
        if usage and "error" in str(usage):
            error_types[str(usage.get("error", "unknown"))[:50]] += 1
    elif raw.endswith("escalation_decision") or raw.endswith('"'):
        # Truncated mid-JSON
        truncated_responses += 1
    
    # Check completion tokens
    if usage and isinstance(usage, dict):
        completion_tokens = usage.get("completion_tokens", 0)
        if completion_tokens >= 795:  # Near 800 limit
            truncated_responses += 1

print(f"Empty responses: {empty_responses}")
print(f"Truncated responses: {truncated_responses}")
print(f"Error types: {error_types.most_common()}")

# Sample a successful parse to see token usage
for result in raw_results:
    consensus = result.get("consensus", {})
    if consensus.get("parsed"):
        print(f"\nSuccessful parse example:")
        print(f"  Usage: {consensus.get('usage')}")
        break
