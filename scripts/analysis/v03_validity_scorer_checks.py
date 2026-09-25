#!/usr/bin/env python3
"""Scorer checks for the validity review (question 5): GPT-6 Astra's finding 10 and the
DXA coverage discrepancy, tested on the real outputs.

1. D1 Brier on an empty differential. evaluator/v03_score.py `diagnosis` leaves top_p
   at 0 and top1 False when the differential has no codes, so the case scores a Brier
   of 0 (a perfect forecast). We count the empty differentials in the two runs.
2. D2 dangerous confident misdiagnosis needs a tier gap of 2, so a confident MI -> PE
   (both tier 1) or MI -> GERD (gap 1) is not counted. We count every confident wrong
   top diagnosis (top p >= 60/70/80, wrong under the strict map, naming a DDXPlus
   condition) and split it by tier gap.
3. DXA COV: the production reader flags the top 5 tier-1 conditions at DXA p >= 10%
   that are not red herrings (COV 62.2); scripts/analysis/v03_headline_design.py
   `dxa_answers` flags the top 5 tier-1 conditions by DXA p, unfiltered and from any p
   (COV 96.1). We rebuild both flag rules and score COV with the production code.
4. Code-map gaps behind the H' events: for each R10 target the models miss under H',
   which flagged codes have no relation to it in spec/ddxplus_icd10_map.csv.

Outputs: results/analysis/v03_validity/scorer_checks.log, d2_confident_wrong.csv.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v03_validity_common as C  # noqa: E402

from evaluator import v03_score as vs  # noqa: E402


def main() -> None:
    log = C.log_to("scorer_checks.log")
    out = C.ensure_out()
    L = C.load("main")
    key = L.key

    log("== 1. Empty differentials (D1 Brier scores them 0)")
    for m in C.MODELS:
        o = L.outcomes[m]
        empty = sum(1 for p in o.parsed if p.readable and not [e for e in p.differential if e.code])
        log(f"  {m}: {empty} readable cases with no differential codes")
    log("  code: evaluator/v03_score.py diagnosis(): `if not codes: continue` leaves top_p = 0 and top1 = False, so (0 - 0)^2 = 0")

    log("\n== 2. Confident wrong top diagnosis, by tier gap (strict map)")
    rows = []
    for m in C.MODELS:
        o = L.outcomes[m]
        dx = vs.diagnosis(o, key, L.matcher)
        for t in vs.D2_THRESHOLDS:
            c = Counter()
            for i, p in enumerate(o.parsed):
                if not p.readable or not p.differential or dx["top_p"][i] < t or dx["top1"][i]:
                    continue
                owner = L.matcher.cmap.owner(p.differential[0].code)
                if owner is None:
                    c["off-list"] += 1
                    continue
                gap = L.tiers.get(owner, 3) - key.truth_tier[i]
                c[f"gap {gap:+d}"] += 1
                rows.append({"model": m, "threshold": t, "case_id": key.case_ids[i], "truth": key.truth[i],
                             "truth_tier": int(key.truth_tier[i]), "top_code": p.differential[0].code, "top_owner": owner,
                             "owner_tier": L.tiers.get(owner, 3), "top_p": dx["top_p"][i], "gap": gap,
                             "counted_by_D2": abs(gap) >= vs.D2_TIER_GAP})
            total = sum(c.values())
            counted = sum(v for k, v in c.items() if k.startswith("gap") and abs(int(k[4:])) >= 2)
            log(f"  {m} top p >= {t}: {total} confident wrong; D2 counts {counted}; " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())))
    C.write_csv(out / "d2_confident_wrong.csv", rows)
    both_t1 = [r for r in rows if r["threshold"] == 60 and r["truth_tier"] == 1 and r["owner_tier"] == 1]
    log(f"  tier-1 truth confidently (>= 60) called another tier-1 condition: {len(both_t1)}: "
        + "; ".join(f"{r['truth']}->{r['top_owner']}" for r in both_t1[:12]))

    log("\n== 3. DXA reader coverage under the two flag rules")
    canon = L.matcher.cmap.canonical
    prod = [p["flags"] for p in L.rows["dxa"]]
    design = []
    for cid in key.case_ids:
        t1 = sorted(((c, p) for c, p in L.dxa[cid].items() if L.tiers.get(c) == 1), key=lambda kv: -kv[1])[:5]
        design.append([canon[c] for c, _ in t1])
    for name, flags in (("production (top 5 tier-1 at p >= 10%, not red herring)", prod),
                        ("headline design (top 5 tier-1 by p, any p, unfiltered)", design)):
        preds = [{"case_id": c, "serious_concern": p["serious_concern"], "flags": f} for c, p, f in zip(key.case_ids, L.rows["dxa"], flags)]
        o = vs.outcomes(preds, key, L.matcher, policies=("standard",))
        cov = vs.coverage(o, key)
        mt = vs.missed_targets(o, key)
        log(f"  {name}: COV {100 * np.nanmean(cov):.1f}, MT {int((mt['truth_missed'] + mt['dxa_missed']).sum())}/284, "
            f"mean flags {np.mean([len(f) for f in flags]):.2f}")

    log("\n== 4. Map gaps behind the H' events (flag codes with no map relation to the missed target)")
    for m in C.MODELS:
        o = L.outcomes[m]
        masks = C.event_masks(L, m)
        gaps = Counter()
        for i in np.flatnonzero(masks["H_prime"] & o.yes):
            for c in key.keys[i].r10:
                for f in o.flags[i]:
                    if L.matcher.relation(f, c) is None:
                        gaps[(c, f)] += 1
        log(f"  {m}: " + "; ".join(f"{c} <- {f} x{n}" for (c, f), n in gaps.most_common(14)))


if __name__ == "__main__":
    main()
