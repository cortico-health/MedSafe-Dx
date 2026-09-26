#!/usr/bin/env python3
"""
Audit the arm-4 answers of the v0.3 prompt test (results/v03/ab/runs/*-v7a4[ab].json) and
write results/v03/ab/arm4-audit.json. No model is called.

The audit covers every SERIOUS case whose flag does not name one of the case's R10 targets
(standard map with the family rows), and gives each one class, testing the flag first
because the flag is the arm's decision:

| Class | Rule, in order of testing                                                         |
|-------|------------------------------------------------------------------------------------|
| ii    | the flag names a different tier-1 DDXPlus condition (a partial)                     |
| iii   | the flag names no DDXPlus condition (off-list); split by its off-list tier          |
|       | (spec/offlist_tiers_nhamcs.csv: 1, 2, 3, unscored, unlisted); only tier 1 escalates |
| v     | key artefact: every R10 target is DXA-derived, and the flag or the top diagnosis   |
|       | names the true (not tier-1) condition                                               |
| i     | the target is in the model's own list, not flagged                                  |
| iv-b  | a different serious condition is in the list, not flagged                           |
| iv    | no serious condition anywhere: a genuine miss                                       |

Classes i, iv-b, iv and v hold benign flags (tier 2-3 DDXPlus conditions) and null flags.
For ii and iii we also count how often the target was in the model's list.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluator import v03_valid_reason as vr  # noqa: E402
from evaluator import v03b_score as sb  # noqa: E402

AB = ROOT / "results" / "v03" / "ab"
ORDER = ROOT / "data" / "external" / "icd10cm" / "icd10cm_order_2026.txt"
CLASSES = ("i", "ii", "iii", "iv", "iv-b", "v")
N_EXAMPLES = 5


def icd_names(path: Path = ORDER) -> dict[str, str]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            out[line[6:14].strip()] = line[16:77].strip()
    return out


def decode(code: str, names: dict[str, str]) -> str:
    if not code:
        return "null"
    for n in range(len(code), 2, -1):
        if code[:n] in names:
            suffix = "" if n == len(code) else " (nearest parent)"
            return f"{code} {names[code[:n]]}{suffix}"
    return f"{code} (not an ICD-10-CM code)"


def audit_row(a, ab, rule, names) -> list[dict]:
    k = ab.key
    out = []
    for i, kk in enumerate(k.keys):
        if ab.klass[i] != "serious":
            continue
        p = a.parsed[i]
        flag = p.flag if (p is not None and p.readable) else None
        codes = a.codes[i]
        targets = set(kk.r10)
        fr = vr.reason([flag] if flag else [], kk.r10, ab.matcher, k.tiers, rule)
        if fr.kind == vr.TARGET:
            continue
        flag_hit = ab.matcher.conditions_hit([flag], vr.POLICY) if flag else set()
        list_hits = ab.matcher.conditions_hit(codes, vr.POLICY) if codes else set()
        target_listed = bool(targets & list_hits)
        list_reason = vr.reason(codes, kk.r10, ab.matcher, k.tiers, rule)
        dxa_only = all(t.source == "dxa" for c, t in kk.considered.items() if t.in_r10)
        top1_truth = bool(codes) and ab.matcher.matches(codes[0], kk.truth, vr.POLICY)
        off_tier = None
        if flag and fr.kind == vr.OTHER_TIER1:
            cls = "ii"
        elif flag and not flag_hit:
            cls, off_tier = "iii", rule.tier(flag)[0]
        elif dxa_only and (kk.truth in flag_hit or top1_truth):
            cls = "v"
        elif target_listed:
            cls = "i"
        elif list_reason.kind in (vr.OTHER_TIER1, vr.OFFLIST):
            cls = "iv-b"
        else:
            cls = "iv"
        out.append({"case_id": kk.case_id, "class": cls, "offlist_tier": off_tier, "truth": kk.truth,
                    "truth_tier": kk.truth_tier, "targets": sorted(targets), "dxa_only": dxa_only,
                    "working_diagnosis": ab.design[i].get("working_diagnosis"), "flag": decode(flag, names),
                    "flag_conditions": sorted(flag_hit), "list": [decode(c, names) for c in codes],
                    "target_listed": target_listed, "top1_truth": top1_truth,
                    "escalated_draft3": bool(a.esc[i]), "valid_reason_outcome": "partial" if fr.kind != vr.NONE else "miss"})
    return out


def run(groups_fallback: bool = False) -> dict:
    ab = sb.load_ab()
    if vr.OFFLIST_TIERS_CSV.exists():
        rule, source = vr.TierFileRule(), vr.OFFLIST_TIERS_CSV
    elif groups_fallback:
        rule, source = vr.GroupsRule(vr.load_groups_v2()), vr.OFFLIST_V2_CSV
    else:
        raise SystemExit(f"{vr.OFFLIST_TIERS_CSV} is missing; pass --groups-fallback to use the v2 groups")
    names = icd_names()
    rows, summary = {}, {}
    for f in sorted((AB / "runs").glob("*-v7a4[ab].json")):
        preds, meta = sb.load_predictions(f)
        arm, model = meta["prompt_version"], meta["model"]
        a = sb.row_answers(preds, arm, ab, f"{model}|{arm}")
        cases = audit_row(a, ab, rule, names)
        name = f"{sb.short(model)}|{sb.ARM_LABELS[arm]}"
        rows[name] = cases
        c = Counter(x["class"] for x in cases)
        tiers = Counter(x["offlist_tier"] for x in cases if x["class"] == "iii")
        summary[name] = {"serious": int(ab.serious.sum()), "audited": len(cases),
                         "classes": {k: c.get(k, 0) for k in CLASSES},
                         "iii_by_tier": {t: tiers.get(t, 0) for t in ("1", "2", "3", vr.UNSCORED, vr.UNLISTED)},
                         "target_listed_in_ii": sum(1 for x in cases if x["class"] == "ii" and x["target_listed"]),
                         "target_listed_in_iii": sum(1 for x in cases if x["class"] == "iii" and x["target_listed"]),
                         "under_escalations_draft3": sum(1 for x in cases if not x["escalated_draft3"]),
                         "misses_valid_reason": sum(1 for x in cases if x["valid_reason_outcome"] == "miss")}
    examples: dict[str, list[dict]] = defaultdict(list)
    for cls in CLASSES:  # round-robin over rows, so the examples span models
        pools = [[x | {"row": n} for x in cases if x["class"] == cls] for n, cases in rows.items()]
        while len(examples[cls]) < N_EXAMPLES and any(pools):
            for pool in pools:
                if pool and len(examples[cls]) < N_EXAMPLES:
                    examples[cls].append(pool.pop(0))
    offlist_codes = Counter((x["flag"], x["offlist_tier"]) for cases in rows.values() for x in cases if x["class"] == "iii")
    return {"offlist_source": str(source.relative_to(ROOT)), "summary": summary, "examples": dict(examples),
            "offlist_flags": [{"flag": f, "tier": t, "n": n} for (f, t), n in offlist_codes.most_common()],
            "cases": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(AB / "arm4-audit.json"))
    ap.add_argument("--groups-fallback", action="store_true")
    args = ap.parse_args()
    board = run(args.groups_fallback)
    Path(args.out).write_text(json.dumps(board, indent=1) + "\n")
    for n, s in board["summary"].items():
        print(n, s["audited"], s["classes"], s["iii_by_tier"], "tl_ii", s["target_listed_in_ii"], "tl_iii",
              s["target_listed_in_iii"], "d3U", s["under_escalations_draft3"], "vrMiss", s["misses_valid_reason"])


if __name__ == "__main__":
    main()
