#!/usr/bin/env python3
"""
Re-score the v0.3 prompt test (results/v03/ab/runs) under the "valid named reason" rule
(evaluator/v03_valid_reason.py), and write results/v03/ab/ab-rescore.json. No model is
called. Off-list flags take their tier from spec/offlist_tiers_nhamcs.csv; the script stops
when that file is missing, unless --groups-fallback asks for spec/offlist_escalation_groups_v2.csv.

Per row we report both weightings (sample mix and balanced 50/50) at the primary partial
cost and at each sensitivity cost, U, O, the partial rate split by in-list and off-list
reason, the escalation share, and the draft-3 SCORE from ab-scores.json. Paired differences
follow the pre-registered comparisons of spec section 12, on the same cluster-bootstrap draws.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluator import v03_score as vs  # noqa: E402
from evaluator import v03_valid_reason as vr  # noqa: E402
from evaluator import v03b_score as sb  # noqa: E402
from evaluator.schemas_v03b import FLAG_ARMS  # noqa: E402

AB = ROOT / "results" / "v03" / "ab"
ARMS = ("v7a1", "v7a2", "v7a3", "v7a4a", "v7a4b")
ROW_MEASURES = ("score_mix", "score_bal", "U", "O", "partial", "partial_inlist", "partial_offlist", "pass", "bare",
                "esc", "cost", "score_mix@2", "score_bal@2", "score_mix@3.5", "score_bal@3.5")
PAIRED = ("score_mix", "score_bal", "U", "O", "partial", "esc")


def offlist_partials(o: vr.Outcome, ab) -> Counter:
    """(off-list code, group, truth) for SERIOUS partials whose reason is off-list."""
    c = Counter()
    for r, out, kk in zip(o.reasons, o.outcome, ab.key.keys):
        if out == vr.PARTIAL_OUT and r.kind == vr.OFFLIST:
            code, group = r.offlist[0]
            c[(code, group, kk.truth)] += 1
    return c


def run(n_boot: int = vs.N_BOOTSTRAP, groups_fallback: bool = False) -> dict:
    ab = sb.load_ab()
    if vr.OFFLIST_TIERS_CSV.exists():
        rule, source = vr.TierFileRule(), vr.OFFLIST_TIERS_CSV
    elif groups_fallback:
        rule, source = vr.GroupsRule(vr.load_groups_v2()), vr.OFFLIST_V2_CSV
    else:
        raise SystemExit(f"{vr.OFFLIST_TIERS_CSV} is missing; pass --groups-fallback to score with the v2 groups")
    M = vs.cluster_draws(ab.key.k, n_boot, vs.BOOTSTRAP_SEED)
    draft3 = json.loads((AB / "ab-scores.json").read_text())
    rows, raw = {}, {}
    for f in sorted((AB / "runs").glob("*-v7a*.json")):
        preds, meta = sb.load_predictions(f)
        arm, model = meta["prompt_version"], meta["model"]
        if arm not in ARMS:
            continue
        name = f"{model}|{arm}"
        a = sb.row_answers(preds, arm, ab, name)
        o = vr.outcomes(a, ab, rule)
        point, draws = vr.stats(o, ab, M)
        raw[name] = (point, draws)
        summ = vr.summarise(point, draws)
        row = {"model": model, "arm": arm, "arm_label": sb.ARM_LABELS[arm], "file": str(f.relative_to(ROOT)),
               "measures": {m: summ[m] for m in ROW_MEASURES},
               "counts": {**Counter(o.outcome), "partial_inlist": sum(1 for r, x in zip(o.reasons, o.outcome)
                                                                        if x == vr.PARTIAL_OUT and r.kind == vr.OTHER_TIER1),
                          "partial_offlist": sum(1 for r, x in zip(o.reasons, o.outcome)
                                                 if x == vr.PARTIAL_OUT and r.kind == vr.OFFLIST)},
               "offlist_partials": [{"code": c, "group": g, "truth": t, "n": n}
                                    for (c, g, t), n in offlist_partials(o, ab).most_common()],
               "draft3": {m: draft3["rows"][name]["measures"][m] for m in ("score", "U", "O", "esc")}}
        if arm in FLAG_ARMS:
            row["offlist_bounds"] = {}
            for mode in ("escalate", "routine"):
                bp, bd = vr.stats(vr.outcomes(a, ab, rule, offlist=mode), ab, M)
                bs = vr.summarise(bp, bd)
                row["offlist_bounds"][mode] = {m: bs[m] for m in ("score_mix", "score_bal", "U", "O")}
        rows[name] = row

    refs = {}
    for r, a in sb.reference_answers(ab).items():
        o = vr.outcomes(a, ab, rule)
        p, d = vr.stats(o, ab, M)
        s = vr.summarise(p, d)
        refs[r] = {"label": sb.REFS[r], "measures": {m: s[m] for m in ROW_MEASURES}}

    paired = {}
    models = sorted({r["model"] for r in rows.values()})
    for m in models:
        for x, y in sb.COMPARISONS:
            nx, ny = f"{m}|{x}", f"{m}|{y}"
            if nx in raw and ny in raw:
                paired[f"{m}: arm {sb.ARM_LABELS[x]} - arm {sb.ARM_LABELS[y]}"] = {
                    "model": m, "a": x, "b": y, **vr.diff(raw[nx][1], raw[ny][1], raw[nx][0], raw[ny][0], PAIRED)}
    within = {}
    for arm in ARMS:
        names = [n for n in raw if n.endswith(f"|{arm}")]
        for i, x in enumerate(names):
            for y in names[i + 1:]:
                within[f"arm {sb.ARM_LABELS[arm]}: {sb.short(x.split('|')[0])} - {sb.short(y.split('|')[0])}"] = \
                    vr.diff(raw[x][1], raw[y][1], raw[x][0], raw[y][0], ("score_mix", "score_bal"))
    beats = {n: {w: rows[n]["measures"][w]["ci"][0] is not None and rows[n]["measures"][w]["ci"][0] > 0
                 for w in vr.SCORE_KEYS} for n in rows}
    return {"rule": "valid named reason (evaluator/v03_valid_reason.py)", "partial_cost": vr.PARTIAL,
            "partial_sensitivity": list(vr.PARTIAL_SENSITIVITY),
            "offlist_source": str(source.relative_to(ROOT)),
            "offlist_source_sha256": sb.ak.sha256_file(source),
            "sample": {"cases": int(ab.key.n), "serious": int(ab.serious.sum()), "benign": int(ab.benign.sum()),
                       "conditions": int(ab.key.k), "bootstrap": {"draws": n_boot, "seed": vs.BOOTSTRAP_SEED}},
            "rows": rows, "references": refs, "paired": paired, "within_arm": within, "beats_always_escalate": beats}


def _f(m) -> str:
    return sb._fmt(m)


def _v(m) -> str:
    return sb._v(m)


def report(board: dict) -> str:
    rows = board["rows"]
    s = board["sample"]
    arm4 = sorted((r for r in rows.values() if r["arm"] in FLAG_ARMS), key=lambda r: (r["model"], r["arm"]))
    rest = sorted((r for r in rows.values() if r["arm"] not in FLAG_ARMS), key=lambda r: (r["model"], r["arm"]))
    L = ["## Arms 4a and 4b", "",
         "| Model | Arm | SCORE, sample mix [95% CI] | SCORE, balanced [95% CI] | U % | O % | Partial % (in-list / off-list) | ESC % | Draft-3 SCORE |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in arm4:
        m, c = r["measures"], r["counts"]
        L.append(f"| {sb.short(r['model'])} | {r['arm_label']} | {_f(m['score_mix'])} | {_f(m['score_bal'])} | {_v(m['U'])} | "
                 f"{_v(m['O'])} | {_v(m['partial'])} ({c['partial_inlist']} / {c['partial_offlist']}) | {_v(m['esc'])} | "
                 f"{_f(r['draft3']['score'])} |")
    L += ["", "### Sensitivity: partial cost and off-list bounds", "",
          "| Model | Arm | Mix, partial 2 | Balanced, partial 2 | Mix, partial 3.5 | Balanced, partial 3.5 | Balanced, off-list all valid | Balanced, off-list all invalid |",
          "|---|---|---|---|---|---|---|---|"]
    for r in arm4:
        m, b = r["measures"], r["offlist_bounds"]
        L.append(f"| {sb.short(r['model'])} | {r['arm_label']} | {_f(m['score_mix@2'])} | {_f(m['score_bal@2'])} | "
                 f"{_f(m['score_mix@3.5'])} | {_f(m['score_bal@3.5'])} | {_f(b['escalate']['score_bal'])} | "
                 f"{_f(b['routine']['score_bal'])} |")
    L += ["", "### Arm 4a - arm 4b, paired (same cases and draws)", "",
          "| Model | SCORE, mix | SCORE, balanced | U (pp) | O (pp) | Partial (pp) | ESC (pp) |", "|---|---|---|---|---|---|---|"]
    for d in board["paired"].values():
        if (d["a"], d["b"]) == ("v7a4a", "v7a4b"):
            L.append(f"| {sb.short(d['model'])} | {_f(d['score_mix'])} | {_f(d['score_bal'])} | {_f(d['U'])} | {_f(d['O'])} | "
                     f"{_f(d['partial'])} | {_f(d['esc'])} |")
    L += ["", "### Model pairs within arm 4", "", "| Pair | SCORE, mix | SCORE, balanced |", "|---|---|---|"]
    for k, d in board["within_arm"].items():
        if k.startswith("arm 4"):
            L.append(f"| {k} | {_f(d['score_mix'])} | {_f(d['score_bal'])} |")
    L += ["", "## Arms 1-3, for comparison", "",
          "| Model | Arm | SCORE, mix | SCORE, balanced | U % | O % | Partial % | ESC % | Draft-3 SCORE |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rest:
        m = r["measures"]
        L.append(f"| {sb.short(r['model'])} | {r['arm_label']} | {_f(m['score_mix'])} | {_f(m['score_bal'])} | {_v(m['U'])} | "
                 f"{_v(m['O'])} | {_v(m['partial'])} | {_v(m['esc'])} | {_f(r['draft3']['score'])} |")
    L += ["", "## Reference rows", "", "| Reference | SCORE, mix | SCORE, balanced | U % | O % | Partial % |", "|---|---|---|---|---|---|"]
    for r in board["references"].values():
        m = r["measures"]
        L.append(f"| {r['label']} | {_f(m['score_mix'])} | {_f(m['score_bal'])} | {_v(m['U'])} | {_v(m['O'])} | {_v(m['partial'])} |")
    L += ["", "## Off-list partials in arm 4 (flag, off-list label, truth)", "", "| Row | Flag | Label | Truth | n |", "|---|---|---|---|---|"]
    for r in arm4:
        for x in r["offlist_partials"]:
            L.append(f"| {sb.short(r['model'])} {r['arm_label']} | {x['code']} | {x['group']} | {x['truth']} | {x['n']} |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(AB / "ab-rescore.json"))
    ap.add_argument("--n-boot", type=int, default=vs.N_BOOTSTRAP)
    ap.add_argument("--groups-fallback", action="store_true", help="score off-list flags with the v2 groups")
    ap.add_argument("--tables", default=str(AB / "ab-rescore-tables.md"), help="generated tables for ab-rescore.md")
    args = ap.parse_args()
    board = run(args.n_boot, args.groups_fallback)
    Path(args.out).write_text(json.dumps(sb._jsonable(board), indent=1) + "\n")
    Path(args.tables).write_text(report(board))
    for n, r in sorted(board["rows"].items()):
        m = r["measures"]
        print(f"{sb.short(r['model']):24s} {r['arm_label']:3s} mix {sb._fmt(m['score_mix']):28s} bal {sb._fmt(m['score_bal']):28s} "
              f"U {m['U']['value']:5.1f} O {m['O']['value']:5.1f} part {m['partial']['value']:5.1f} esc {m['esc']['value']:5.1f} "
              f"d3 {r['draft3']['score']['value']}")


if __name__ == "__main__":
    main()
