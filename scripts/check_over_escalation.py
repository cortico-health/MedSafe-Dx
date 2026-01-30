#!/usr/bin/env python3
"""Check over-escalation rate."""

import json
from pathlib import Path

results_path = Path("results/analysis/moe_panel_results.json")

with open(results_path) as f:
    eval_results = json.load(f)

print("=== OVER-ESCALATION RATES ===\n")

# Check individual models
for model, data in eval_results.get("individual_models", {}).items():
    preds = data.get("predictions", [])
    n_total = len([p for p in preds if not p.get("parse_failed")])
    n_over_esc = sum(1 for p in preds if p.get("eval", {}).get("over_escalation", False))
    rate = n_over_esc / n_total * 100 if n_total > 0 else 0
    print(f"{model.split('/')[-1]}: {n_over_esc}/{n_total} = {rate:.1f}%")

# Check consensus
preds = eval_results.get("consensus", {}).get("predictions", [])
n_total = len([p for p in preds if not p.get("parse_failed")])
n_over_esc = sum(1 for p in preds if p.get("eval", {}).get("over_escalation", False))
rate = n_over_esc / n_total * 100 if n_total > 0 else 0
print(f"\n**Consensus**: {n_over_esc}/{n_total} = {rate:.1f}%")

# Also show breakdown of all safety metrics for consensus
print("\n=== CONSENSUS FULL SAFETY BREAKDOWN ===")
metrics = eval_results.get("consensus", {}).get("metrics", {})
for key, val in metrics.items():
    if isinstance(val, float):
        print(f"{key}: {val*100:.1f}%")
    else:
        print(f"{key}: {val}")
