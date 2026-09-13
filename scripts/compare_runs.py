#!/usr/bin/env python3
"""
Compare two leaderboard rosters and print the deltas that matter for a release note.

Usage:
  python3 scripts/compare_runs.py OLD_JSON NEW_JSON [--pairs new=old ...]

OLD_JSON / NEW_JSON: either a leaderboard-data.json array (as served by web/main.py)
or a directory of *-eval.json files. Prints, as markdown:
  1. roster changes (added / dropped / superseded)
  2. ranking table by Triage Success Rate for the new roster, with old rank
  3. generation deltas for --pairs (default pairs below)

Why: each refresh needs a "what changed" summary for the leaderboard banner and
docs/RUNS.md, and computing it by hand is slow and error-prone.
"""
import argparse
import glob
import json
import os
import sys

DEFAULT_PAIRS = {
    "anthropic-claude-opus-5": "anthropic-claude-opus-4.7",
    "x-ai-grok-4.6": "x-ai-grok-4.20",
    "openai-gpt-5.6-sol": "openai-gpt-5.2",
}


def load(path):
    if os.path.isdir(path):
        rows = []
        for f in sorted(glob.glob(os.path.join(path, "*-eval.json"))):
            with open(f) as fh:
                rows.append(json.load(fh))
        return rows
    with open(path) as fh:
        data = json.load(fh)
    return data if isinstance(data, list) else [data]


def metrics(r):
    cases = r.get("cases_expected") or r.get("cases") or 250
    eff = r.get("effectiveness") or {}
    over = eff.get("over_escalation")
    if over is None:
        over = (r.get("informational") or {}).get("overdiagnosis") or 0
    spr = float(r.get("safety_pass_rate") or 0)
    tsr = spr - float(over) / float(cases)
    return {
        "model": r["model"],
        "version": r.get("version") or r.get("model_version") or "?",
        "spr": spr,
        "over": float(over) / float(cases),
        "tsr": tsr,
        "recall": float(eff.get("top3_recall") or 0),
        "fmt_fail": int(r.get("format_failures") or 0),
        "missed": int((r.get("safety") or {}).get("missed_escalations") or 0),
    }


def pct(x):
    return f"{100 * x:.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--pairs", nargs="*", default=[], help="new=old model slugs")
    a = ap.parse_args()

    pairs = dict(DEFAULT_PAIRS)
    for p in a.pairs:
        n, o = p.split("=", 1)
        pairs[n] = o

    old = {m["model"]: m for m in map(metrics, load(a.old))}
    new = {m["model"]: m for m in map(metrics, load(a.new))}

    added = sorted(set(new) - set(old))
    dropped = sorted(set(old) - set(new))
    print("## Roster")
    print(f"- old: {len(old)} models, new: {len(new)} models")
    print(f"- added: {', '.join(added) or 'none'}")
    print(f"- dropped: {', '.join(dropped) or 'none'}")
    sup = [f"{n} supersedes {o}" for n, o in pairs.items() if n in new and o in new]
    print(f"- superseded (both still listed): {'; '.join(sup) or 'none'}")

    rank_old = {m: i + 1 for i, m in enumerate(sorted(old, key=lambda k: -old[k]["tsr"]))}
    rank_new = sorted(new, key=lambda k: -new[k]["tsr"])
    print("\n## Ranking by Triage Success Rate (new roster)")
    print("| # | Model | Era | TSR | SPR | Over-esc | Top-3 recall | Fmt fail | Missed esc | Old rank |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for i, m in enumerate(rank_new, 1):
        r = new[m]
        print(
            f"| {i} | {m} | {r['version']} | {pct(r['tsr'])} | {pct(r['spr'])} | {pct(r['over'])} | "
            f"{pct(r['recall'])} | {r['fmt_fail']} | {r['missed']} | {rank_old.get(m, 'new')} |"
        )

    top_old = max(old.values(), key=lambda r: r["tsr"]) if old else None
    top_new = max(new.values(), key=lambda r: r["tsr"]) if new else None
    if top_old and top_new:
        print(f"\n- top TSR: {top_old['model']} {pct(top_old['tsr'])} -> {top_new['model']} {pct(top_new['tsr'])}")

    print("\n## Generation deltas (successor vs predecessor)")
    print("| Successor | Predecessor | TSR old -> new | SPR old -> new | Over-esc old -> new |")
    print("|---|---|---|---|---|")
    for n, o in pairs.items():
        if n in new and o in new:
            a_, b_ = new[o], new[n]
            print(
                f"| {n} | {o} | {pct(a_['tsr'])} -> {pct(b_['tsr'])} ({100 * (b_['tsr'] - a_['tsr']):+.1f}) | "
                f"{pct(a_['spr'])} -> {pct(b_['spr'])} | {pct(a_['over'])} -> {pct(b_['over'])} |"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
