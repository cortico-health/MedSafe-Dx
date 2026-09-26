"""Summarise arm 4aj / 4bj prediction files without scoring (spec/v0.3-scoring.md, amendment A1).

Per model and arm we print the parse rate, the flag rate, the split of flags into tier-1 DDXPlus
conditions, other DDXPlus conditions, off-list codes and none, the justification rate, finish
reasons and cost, then a sample of justifications per model. `--json` also writes the table and
every justification, so a reader can check the sample against the whole set.

Usage: python3 scripts/analysis/v03_justification_summary.py FILE... [--sample 5] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluator.answer_key_v03 import load_tiers  # noqa: E402
from evaluator.condition_match import FlagMatcher  # noqa: E402
from evaluator.schemas_v03b import (FLAG_NONE, FLAG_OFFLIST_GROUP, FLAG_OFFLIST_OTHER, FLAG_ONLIST,  # noqa: E402
                                    FLAG_TIER1, flag_kind, load_offlist_groups, parse_v03b)


def summarise(path: Path, matcher, tiers, groups) -> dict:
    d = json.loads(path.read_text())
    meta, preds = d.get("metadata", {}), d.get("predictions", [])
    arm = meta["prompt_version"]
    parsed = [parse_v03b(p, arm) for p in preds]
    readable = [p for p in parsed if p.readable]
    kinds = Counter(flag_kind(p.flag, matcher, tiers, groups) for p in readable)
    flagged = sum(v for k, v in kinds.items() if k != FLAG_NONE)
    usage = [p.get("usage") or {} for p in preds if isinstance(p, dict)]
    n = len(preds)
    pct = lambda a, b: round(100 * a / b, 1) if b else None
    return {
        "file": str(path), "model": meta.get("model"), "arm": arm, "n": n,
        "parse_pct": pct(len(readable), n),
        "flag_pct": pct(flagged, len(readable)),
        "flags": flagged,
        "tier1_pct_of_flags": pct(kinds[FLAG_TIER1], flagged),
        "onlist_other_pct_of_flags": pct(kinds[FLAG_ONLIST], flagged),
        "offlist_pct_of_flags": pct(kinds[FLAG_OFFLIST_GROUP] + kinds[FLAG_OFFLIST_OTHER], flagged),
        "offlist_in_group": kinds[FLAG_OFFLIST_GROUP],
        "none": kinds[FLAG_NONE],
        "invalid_flags": sum(p.flag_status == "invalid" for p in readable),
        "justification_pct": pct(sum(p.justification is not None for p in readable), len(readable)),
        "finish_reasons": dict(Counter(str(p.get("finish_reason")) for p in preds if isinstance(p, dict))),
        "cost_usd": round(sum(u.get("cost") or 0 for u in usage), 4),
        "cost_per_call_usd": round(sum(u.get("cost") or 0 for u in usage) / n, 5) if n else None,
        "justifications": [{"case_id": p.case_id, "flag": p.flag, "kind": flag_kind(p.flag, matcher, tiers, groups),
                            "text": p.justification} for p in readable if p.justification],
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--sample", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260926)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)
    matcher, tiers, groups = FlagMatcher(), load_tiers(), load_offlist_groups()
    rows = [summarise(f, matcher, tiers, groups) for f in args.files]
    cols = ("model", "arm", "n", "parse_pct", "flag_pct", "tier1_pct_of_flags", "onlist_other_pct_of_flags",
            "offlist_pct_of_flags", "none", "justification_pct", "cost_usd", "cost_per_call_usd", "finish_reasons")
    print("\t".join(cols))
    for r in rows:
        print("\t".join(str(r[c]) for c in cols))
    print(f"total cost_usd {round(sum(r['cost_usd'] for r in rows), 4)}")
    rng = random.Random(args.seed)
    for model in dict.fromkeys(r["model"] for r in rows):
        pool = [(r["arm"], j) for r in rows if r["model"] == model for j in r["justifications"]]
        print(f"\n## {model}")
        for arm, j in rng.sample(pool, min(args.sample, len(pool))):
            print(f"- {arm} {j['case_id']} flag={j['flag']} ({j['kind']}): {j['text']}")
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
