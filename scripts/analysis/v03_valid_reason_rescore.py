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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(AB / "ab-rescore.json"))
    ap.add_argument("--n-boot", type=int, default=vs.N_BOOTSTRAP)
    ap.add_argument("--groups-fallback", action="store_true", help="score off-list flags with the v2 groups")
    args = ap.parse_args()
    board = run(args.n_boot, args.groups_fallback)
    Path(args.out).write_text(json.dumps(sb._jsonable(board), indent=1) + "\n")
    for n, r in sorted(board["rows"].items()):
        m = r["measures"]
        print(f"{sb.short(r['model']):24s} {r['arm_label']:3s} mix {sb._fmt(m['score_mix']):28s} bal {sb._fmt(m['score_bal']):28s} "
              f"U {m['U']['value']:5.1f} O {m['O']['value']:5.1f} part {m['partial']['value']:5.1f} esc {m['esc']['value']:5.1f} "
              f"d3 {r['draft3']['score']['value']}")


if __name__ == "__main__":
    main()
