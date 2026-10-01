#!/usr/bin/env python3
"""
Summarise the settings each model ran under in the v0.3 full run, for the methodology's "Run settings" table.

We read the run config (inference/run_config_v03_abj.json), the two provenance files and every arm-4aj prediction file
under results/v03_full/runs/ (git-ignored), and write results/v03_full/run_settings.md, which is committed so the
methodology can cite it. Per model we report the reasoning effort we sent, the mean reasoning and completion tokens
per case as OpenRouter reported them (usage.completion_tokens_details.reasoning_tokens, usage.completion_tokens), and
the providers OpenRouter routed the requests to, with request counts. Rows follow the arm-4aj score order in
results/v03_full/scores.json; models with no score (the unfinished runs) come last.

Usage: python3 scripts/analysis/v03_run_settings.py
"""
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "results/v03_full/runs"
CONFIG = ROOT / "inference/run_config_v03_abj.json"
SCORES = ROOT / "results/v03_full/scores.json"
OUT = ROOT / "results/v03_full/run_settings.md"


def main():
    config = json.loads(CONFIG.read_text())
    scores = json.loads(SCORES.read_text())["headline_within_condition"]["rows"]
    order = {k.split("|")[0]: -v["score_z_bal"]["value"] if isinstance(v["score_z_bal"], dict) else -v["score_z_bal"]
             for k, v in scores.items() if k.endswith("|4aj")}
    rows = []
    for f in sorted(RUNS.glob("*-v7a4aj.json")):
        preds = json.loads(f.read_text())["predictions"]
        served = collections.Counter()
        reasoning, completion = [], []
        model_id = None
        for p in preds:
            served[p.get("provider") or "unknown"] += 1
            usage = p.get("usage") or {}
            completion.append(usage.get("completion_tokens") or 0)
            reasoning.append((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
            model_id = model_id or p.get("model_served")
        slug = f.name[: -len("-v7a4aj.json")]
        or_id = next((m for m in config["models"] if m.replace("/", "-") == slug), model_id or slug)
        short = or_id.split("/", 1)[1]
        rows.append({
            "id": or_id, "short": short, "n": sum(1 for p in preds if not p.get("error")),
            "effort": config["models"].get(or_id, {}).get("reasoning_effort"),
            "reasoning": sum(reasoning) / len(reasoning), "completion": sum(completion) / len(completion),
            "served": served,
        })
    rows.sort(key=lambda r: (r["short"] not in order, order.get(r["short"], 0)))

    prov = {name: json.loads((RUNS / name).read_text()) for name in ("provenance.json", "provenance-expansion.json")}
    lines = [
        "# v0.3 full run: run settings per model (arm 4aj)",
        "",
        f"Built by `scripts/analysis/v03_run_settings.py` from `{CONFIG.relative_to(ROOT)}`, the provenance files and the",
        "arm-4aj prediction files under `results/v03_full/runs/` (git-ignored). Tokens are means per case as OpenRouter",
        "reported them; \"Served by\" counts requests per provider.",
        "",
        f"- **Shared settings:** prompt {config['prompt_version']}a4aj, decoder {config['decoder_version']}, temperature "
        f"{config['temperature']}, max_tokens {config['max_tokens']}. The OpenRouter client makes up to four attempts on an "
        f"empty response; the runner then retries once on an empty or truncated one (empty_or_truncated_retries "
        f"{config['empty_or_truncated_retries']}); an answer that failed to parse was asked again once.",
    ]
    for name, p in prov.items():
        lines.append(
            f"- **{p['run']}** (`{name}`): launched {p['started']} from commit {p['git_commit'][:7]} with "
            f"git_dirty {str(p['git_dirty']).lower()} (uncommitted changes in the working tree); run config sha256 "
            f"{p['run_config']['sha256'][:12]}.")
    lines += [
        "",
        "| Model | Reasoning effort sent | Reasoning tokens per case | Completion tokens per case | Cases answered | Served by (requests) |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        served = "; ".join(f"{k} {v}" for k, v in r["served"].most_common(4))
        if len(r["served"]) > 4:
            served += f"; {len(r['served']) - 4} more providers ({sum(v for _, v in r['served'].most_common()[4:])} requests)"
        effort = r["effort"] if r["effort"] else "none (no reasoning parameter sent)"
        name = r["id"] + ("" if r["short"] in order else " (unfinished, not scored)")
        lines.append(f"| {name} | {effort} | {r['reasoning']:.0f} | {r['completion']:.0f} | {r['n']} | {served} |")
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(rows)} models)")


if __name__ == "__main__":
    main()
