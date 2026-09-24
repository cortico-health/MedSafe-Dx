#!/usr/bin/env python3
"""Estimate the v0.2 run's tokens and cost per model, with no spend.

Inputs:
1. Prompt size: a dry-run file from `DRY_RUN=1 ./scripts/run_v02.sh` (every model
   gets the same messages, so one file serves all). Tokens = characters / 4;
   real tokenizers differ by roughly 15%.
2. Prices: OpenRouter's public model list (GET /api/v1/models, no key, no charge).
3. Output: v0.2 records usage per prediction, but no stored run does yet, so we
   assume 150 content tokens (the section 3 JSON) plus reasoning tokens at effort
   "medium" in three scenarios: 500 (low), 1,500 (mid) and 4,000 (high) per case.
   Models with no reasoning effort in the run config get none. Retries on empty
   or truncated output are not priced.

Usage:
    python3 scripts/estimate_v02_cost.py [--dryrun results/v02/runs/<model>-v02-470cases.dryrun.json]
"""

from __future__ import annotations

import argparse
import glob
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTENT_TOKENS = 150
REASONING = {"low": 500, "mid": 1500, "high": 4000}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dryrun", default=None)
    ap.add_argument("--run-config", default=str(ROOT / "inference/run_config_v02.json"))
    ap.add_argument("--roster", default=str(ROOT / "scripts/run_v02.sh"))
    args = ap.parse_args()
    path = args.dryrun or sorted(glob.glob(str(ROOT / "results/v02/runs/*.dryrun.json")))[0]
    dr = json.load(open(path))["metadata"]
    n, prompt = dr["cases"], dr["est_prompt_tokens_mean"]
    cfg = json.load(open(args.run_config))["models"]
    roster = []
    in_roster = False
    for line in open(args.roster):
        if line.startswith("ROSTER=("):
            in_roster = True
            continue
        if in_roster:
            if line.strip() == ")":
                break
            roster.append(line.strip().strip('"'))
    with urllib.request.urlopen("https://openrouter.ai/api/v1/models", timeout=30) as r:
        prices = {m["id"]: m["pricing"] for m in json.load(r)["data"]}
    print(f"Prompt: {prompt:.0f} tokens per case (from {Path(path).name}), {n} cases per model.")
    print(f"Output per case: {CONTENT_TOKENS} content + reasoning low/mid/high {REASONING} for reasoning models.\n")
    print(f"| Model | $/M in | $/M out | Input $ | Total $ low | mid | high |")
    print("|---|---|---|---|---|---|---|")
    tot = {k: 0.0 for k in REASONING}
    for m in roster:
        p = prices.get(m)
        if p is None:
            print(f"| {m} | not listed | | | | | |")
            continue
        pin, pout = float(p["prompt"]), float(p["completion"])
        reasons = (cfg.get(m) or {}).get("reasoning_effort") is not None
        inp = n * prompt * pin
        row = []
        for k, rt in REASONING.items():
            out = n * (CONTENT_TOKENS + (rt if reasons else 0)) * pout
            row.append(inp + out)
            tot[k] += inp + out
        print(f"| {m} | {pin * 1e6:.2f} | {pout * 1e6:.2f} | {inp:.2f} | " + " | ".join(f"{x:.2f}" for x in row) + " |")
    print(f"| **Total ({len(roster)} models)** | | | | " + " | ".join(f"**{tot[k]:.0f}**" for k in REASONING) + " |")


if __name__ == "__main__":
    main()
