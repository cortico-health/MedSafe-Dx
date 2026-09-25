"""v0.3 review, part 6: the remaining numbers.

(a) which conditions supply the R5 and R10 targets on the 470 sample, and what the clearly-low-risk set holds;
(b) tier-1 truths that DXA itself puts under 5% or 10% on (the truth-only H events no reader of DXA can avoid);
(c) the 5-10% band with a 2% floor instead of 1%;
(d) H' = NO, unreadable, or YES with no tier-1 flag, on the v0.1 stand-ins;
(e) simulated interval widths for H on the 470 sample at 2, 5 and 10%, uniform and clustered misses.

Run: .venv/bin/python scripts/analysis/v03_review_extra.py
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from v03_review_common import (FLOOR, LENIENT, OUT, RATIO, Matcher, RedHerring, build_key, fmt, load_adults,
                               load_tiers, sample_ids, score)
from v03_review_models import load_rows

BROADER_ONLY = ("equivalent", "narrower", "broader")


def main() -> None:
    df = load_adults()
    tiers = load_tiers()["v03"]
    rh = RedHerring()
    m = Matcher()
    key = build_key(df, sample_ids("470"), tiers, rh)

    # (a)
    r5 = Counter(t for c in key for t in c["R5"])
    r10 = Counter(t for c in key for t in c["R10"])
    print("(a) R5 targets by condition (truth + DXA-derived):")
    for cond, n in r5.most_common():
        truth_n = sum(1 for c in key if c["truth"] == cond and cond in c["R5"])
        print(f"   {cond[:40]:40} R5 {n:3} (truth {truth_n:2}, DXA-derived {n - truth_n:3}); R10 {r10[cond]:3}")
    low = [c for c in key if c["clearly_low"]]
    print("   clearly low-risk by truth:", Counter(c["truth"] for c in low).most_common())
    rf = [c for c in key if c["red_flag"]]
    print(f"   red-flag cases: {len(rf)}; among tier-3 truths with no R5 target: {sum(1 for c in rf if c['tier'] == 3 and not c['R5'])}; names {Counter(n for c in rf for n in c['red_flag_names'])}")
    not_scored = [c for c in key if not c["R10"] and not c["clearly_low"]]
    print(f"   cases in neither the H denominator nor the OC denominator: {len(not_scored)}; by tier {Counter(c['tier'] for c in not_scored)}")

    # (b)
    t1 = [c for c in key if c["tier"] == 1]
    lt5 = [c for c in t1 if c["dxa"].get(c["truth"], 0) < 5]
    lt10 = [c for c in t1 if c["dxa"].get(c["truth"], 0) < 10]
    print(f"\n(b) tier-1 truths: {len(t1)}; DXA p(truth) < 5%: {len(lt5)}; < 10%: {len(lt10)}")
    print("   < 5% by truth:", Counter(c["truth"] for c in lt5).most_common())
    ranks = []
    for c in lt10:
        order = sorted(c["dxa"], key=lambda x: -c["dxa"][x])
        ranks.append(order.index(c["truth"]) + 1 if c["truth"] in c["dxa"] else 99)
    print("   DXA rank of the truth when p < 10%:", Counter(ranks).most_common(8))
    h0 = [c for c in lt10 if rh.hcount(c["truth"], c["evf"]) == 0]
    print(f"   of those, truth has none of its own hallmarks: {len(h0)} ({Counter(c['truth'] for c in h0).most_common(6)})")
    # examples with the case sheet size
    for c in sorted(lt5, key=lambda c: c["dxa"].get(c["truth"], 0))[:5]:
        print(f"   e.g. {c['case_id']} truth {c['truth']} p={c['dxa'].get(c['truth'], 0):.1f} top1 {c['dxa_top1']} ({max(c['dxa'].values()):.0f}%) hallmarks {rh.hcount(c['truth'], c['evf'])} symptoms {len(c['evf'])}")

    # (c) 2% floor in the 5-10% band
    kept1 = kept2 = 0
    cases5_1 = cases5_2 = 0
    for c in key:
        any1 = any2 = False
        for cond, t in c["targets"].items():
            if t["why"] != "dxa" or t["rh"] or t["p"] >= 10:
                continue
            kept1 += 1
            any1 = True
            if t["und"] or t["rate"] >= max(0.02, RATIO * t["p"] / 100):
                kept2 += 1
                any2 = True
        cases5_1 += any1
        cases5_2 += any2
    n5_1 = sum(1 for c in key if c["R5"])
    print(f"\n(c) 5-10% DXA-derived targets kept: floor 1% {kept1} (in {cases5_1} cases), floor 2% {kept2} (in {cases5_2} cases); R5 cases now {n5_1}")

    # (d) H' on the v0.1 stand-ins
    ids = sample_ids("250")
    key250 = build_key(df, ids, tiers, rh)
    rows = load_rows(ids)
    print("\n(d) v0.1 stand-ins on the 250 set: H vs H' (NO, unreadable, or YES without a tier-1 flag under broader-only matching):")
    for name, ans in sorted(rows.items()):
        s = score(key250, ans, m, LENIENT, tiers=tiers)
        ev = den = 0
        for c, a in zip(key250, ans):
            if not c["R10"]:
                continue
            den += 1
            hit_t1 = any(tiers.get(x) == 1 for x in m.conditions_hit(a["flags"][:5], BROADER_ONLY))
            ev += (a["yes"] is not True) or not hit_t1
        print(f"   {name[:16]:16} H {fmt(s['H']):>5}  H' {fmt(ev / den):>5}  YES without tier-1 flag (all cases, broader-only) {fmt(score(key250, ans, m, BROADER_ONLY, tiers=tiers)['YES_no_tier1_flag'])}")

    # (e) simulated interval widths on the 470 sample
    conds = np.array([c["truth"] for c in key if c["R10"]])
    uniq = sorted(set(conds))
    rng = np.random.default_rng(1)
    print(f"\n(e) H interval widths on the 470 sample (R10 denominator {len(conds)}, {len(uniq)} truth clusters), 1,000 condition-bootstrap draws:")
    for h in (0.02, 0.05, 0.10):
        for pattern in ("uniform", "clustered in 3 conditions"):
            widths = []
            for _ in range(20):
                ev = np.zeros(len(conds), bool)
                k = int(round(h * len(conds)))
                if pattern == "uniform":
                    ev[rng.choice(len(conds), k, replace=False)] = True
                else:
                    pool = np.where(np.isin(conds, rng.choice(uniq, 3, replace=False)))[0]
                    ev[rng.choice(pool, min(k, len(pool)), replace=False)] = True
                draws = []
                idx_by = {u: np.where(conds == u)[0] for u in uniq}
                for _ in range(1000):
                    sel = np.concatenate([idx_by[uniq[i]] for i in rng.integers(0, len(uniq), len(uniq))])
                    draws.append(ev[sel].mean())
                widths.append(np.percentile(draws, 97.5) - np.percentile(draws, 2.5))
            print(f"   H={100*h:.0f}% {pattern:26} mean 95% width {100*np.mean(widths):.1f} points")


if __name__ == "__main__":
    main()
