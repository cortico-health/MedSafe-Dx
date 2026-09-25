"""v0.3 review, part 5: pool-subset availability in the DDXPlus test-split adults, main sample excluded.

Atypical serious: truth is tier 1 and DXA's top-1 is tier 3 (we also report ties at the top).
High risk: a tier-1 condition c that is not the truth has DXA p >= 10% and is not a red herring;
counted per c, with the truth mix behind each c (the selection artefact).

Run: .venv/bin/python scripts/analysis/v03_review_pools.py
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from v03_review_common import OUT, RH_DIR, RedHerring, load_adults, load_tiers, sample_ids, write_csv


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_adults()
    tiers = load_tiers()["v03"]
    rh = RedHerring()
    excl = set(sample_ids("470"))
    pool = df[~df.case_id.isin(excl)]
    tier1 = sorted(c for c, t in tiers.items() if t == 1)
    print(f"pool adults: {len(pool)}")

    # atypical serious
    aty = defaultdict(list)
    ties = Counter()
    for r in pool.itertuples():
        if tiers.get(r.PATHOLOGY) != 1:
            continue
        dd = sorted(r.DD, key=lambda x: -x[1])
        top, ptop = dd[0]
        tied = [n for n, p in dd if abs(p - ptop) < 1e-12]
        if len(tied) > 1:
            ties[r.PATHOLOGY] += 1
        top_tier = min(tiers.get(n, 3) for n in tied) if len(tied) > 1 else tiers.get(top, 3)
        if top_tier == 3:
            aty[r.PATHOLOGY].append((r.case_id, tied[0], round(100 * ptop, 1), round(100 * dict(r.DD).get(r.PATHOLOGY, 0), 1)))
    print("\natypical serious (truth tier 1, DXA top-1 tier 3), available per condition:")
    rows = []
    for c in tier1:
        n_truth = int((pool.PATHOLOGY == c).sum())
        lst = aty.get(c, [])
        tops = Counter(t for _, t, _, _ in lst).most_common(3)
        p_truth = np.median([pt for _, _, _, pt in lst]) if lst else float("nan")
        rows.append({"condition": c, "adults_in_pool": n_truth, "atypical_available": len(lst), "share": round(len(lst) / n_truth, 4) if n_truth else "", "median_p_truth": p_truth, "top1_mix": tops, "top1_ties": ties[c]})
        print(f"  {c[:40]:40} truth adults {n_truth:5}  atypical {len(lst):4}  median DXA p(truth) {p_truth}  top-1 {tops}")
    write_csv(OUT / "pool_atypical.csv", rows)
    total = sum(min(10, r["atypical_available"]) for r in rows)
    print(f"  subset size at up to 10 per condition: {total}; conditions with >= 10: {sum(1 for r in rows if r['atypical_available'] >= 10)}; with 0: {sum(1 for r in rows if r['atypical_available'] == 0)}")

    # high risk: per tier-1 condition c, adults whose truth is not c, with DXA p_c >= 10% and not red herring
    # Use pairs.pkl for the 17 severity<=2 conditions (labels at >= 10%), compute the 4 upgraded ones here.
    pairs = pd.read_pickle(RH_DIR / "cache/pairs.pkl")
    pairs = pairs[(pairs.p >= 10) & (~pairs.y) & (~pairs.in470)]
    hi = defaultdict(list)
    for r in pairs.itertuples():
        if r.rh_M1p:
            continue
        hi[r.cname].append((f"ddxplus_{r.row}", r.truth, round(r.p, 1), int(r.hcount), bool(r.und_M1p)))
    upgraded = [c for c in tier1 if c not in set(pairs.cname.unique())]
    for r in pool.itertuples():
        for c, p in r.DD:
            if c in upgraded and c != r.PATHOLOGY and 100 * p >= 10:
                is_rh, und, n, rate, h = rh.label(c, 100 * p, r.EVF)
                if not is_rh:
                    hi[c].append((r.case_id, r.PATHOLOGY, round(100 * p, 1), h, und))
    print("\nhigh risk (non-truth tier-1 c at DXA >= 10%, not red herring), available per condition, with the truth mix:")
    rows = []
    for c in tier1:
        lst = hi.get(c, [])
        mix = Counter(t for _, t, _, _, _ in lst).most_common(4)
        und = sum(1 for x in lst if x[4])
        tier_mix = Counter(tiers.get(t, 3) for _, t, _, _, _ in lst)
        rows.append({"condition": c, "high_risk_available": len(lst), "undetermined": und, "truth_mix": mix, "truth_tier_mix": dict(tier_mix)})
        print(f"  {c[:40]:40} {len(lst):6} (undetermined {und:4}) truth tiers {dict(tier_mix)} top truths {mix}")
    write_csv(OUT / "pool_high_risk.csv", rows)
    print(f"  subset size at up to 10 per condition: {sum(min(10, r['high_risk_available']) for r in rows)}; conditions with 0: {sum(1 for r in rows if r['high_risk_available'] == 0)}")
    # how many high-risk candidates already have a tier-1 truth (so they are in H anyway)
    t1_truth = sum(1 for c in tier1 for x in hi.get(c, []) if tiers.get(x[1]) == 1)
    allc = sum(len(hi.get(c, [])) for c in tier1)
    print(f"  high-risk candidates whose truth is itself tier 1: {t1_truth} of {allc}")


if __name__ == "__main__":
    main()
