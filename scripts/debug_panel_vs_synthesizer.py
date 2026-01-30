#!/usr/bin/env python3
"""Check if panel models are working but synthesizer is failing."""

import json
from pathlib import Path
from collections import Counter

raw_path = Path("results/analysis/moe_panel/moe_raw_results.json")

with open(raw_path) as f:
    raw_results = json.load(f)

# Count success/failure per model
panel_success = Counter()
panel_fail = Counter()
synthesizer_success = 0
synthesizer_fail = 0

for result in raw_results:
    # Check panel models
    for model, resp in result.get("panel_responses", {}).items():
        if resp.get("parsed"):
            panel_success[model] += 1
        else:
            panel_fail[model] += 1
    
    # Check synthesizer
    consensus = result.get("consensus", {})
    if consensus.get("parsed"):
        synthesizer_success += 1
    else:
        synthesizer_fail += 1
        # Check if it's an API error
        usage = consensus.get("usage", {})
        raw = consensus.get("raw_response", "")
        if not raw:
            print(f"Empty synthesizer response for {result['case_id']}: usage={usage}")

print("\n=== PANEL MODEL SUCCESS RATES ===")
for model in panel_success:
    total = panel_success[model] + panel_fail[model]
    print(f"{model}: {panel_success[model]}/{total} ({panel_success[model]/total*100:.1f}%)")

print(f"\n=== SYNTHESIZER SUCCESS RATE ===")
total = synthesizer_success + synthesizer_fail
print(f"GPT-5-mini: {synthesizer_success}/{total} ({synthesizer_success/total*100:.1f}%)")
