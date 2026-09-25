"""v0.3 review, part 1: the answer key on the 470 sample and where it is fragile.

Checks, in order:
1. our M1' recomputation agrees with results/analysis/dxa_red_herrings/sample470_cases.csv;
2. the spec's counts (R10 227, R5 247, clearly low-risk 120) and what they become when the
   four upgraded tier-1 conditions are, or are not, passed through the detector;
3. the 5-10% band: cell sizes, rates and how many R5-only targets survive there;
4. undetermined targets (cells under 30) kept as targets;
5. the kept DXA-derived targets by (truth, target, hallmark count), for the clinical read;
6. denominators under the tier sensitivity rows.

Run: .venv/bin/python scripts/analysis/v03_review_key.py
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from v03_review_common import (OUT, RH_DIR, R10, R5, RedHerring, band_label, band_of, build_key,
                               load_adults, load_tiers, sample_ids, write_csv)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_adults()
    tiers = load_tiers()
    rh = RedHerring()
    ids = sample_ids("470")

    # 1. agreement with the committed per-case file
    ref = pd.read_csv(RH_DIR / "sample470_cases.csv")
    ref = ref[ref.dxa_p * 100 >= R5]
    dis = 0
    for r in ref.itertuples():
        evf = df.loc[r.case_id].EVF
        is_rh, und, n, rate, h = rh.label(r.condition, 100 * r.dxa_p, evf)
        if r.is_truth:
            continue
        if bool(is_rh) != bool(r.red_herring) or bool(und) != bool(r.undetermined) or h != r.hallmarks_present:
            dis += 1
    print(f"[1] recomputed M1' disagrees with sample470_cases.csv on {dis} of {len(ref)} non-truth pairs at >= 5%")

    # 2. counts under the spec (all 21 tier-1 conditions filtered) and with the 4 upgraded ones unfiltered
    key = build_key(df, ids, tiers["v03"], rh)
    key_unf = build_key(df, ids, tiers["v03"], rh, filter_only_severity2=True)
    for name, k in (("spec, all tier-1 filtered", key), ("upgraded 4 unfiltered", key_unf)):
        n10 = sum(1 for c in k if c["R10"])
        n5 = sum(1 for c in k if c["R5"])
        low = sum(1 for c in k if c["clearly_low"])
        n10_dxa_only = sum(1 for c in k if c["R10"] and not c["truth_only"])
        print(f"[2] {name}: R10 cases {n10} (of which truth not tier 1: {n10_dxa_only}); R5 cases {n5}; clearly low-risk {low}")
    # which cases differ
    diff = [(a["case_id"], a["truth"], sorted(b["R10"] - a["R10"])) for a, b in zip(key, key_unf) if a["R10"] != b["R10"]]
    print(f"    cases whose R10 changes when the 4 upgraded conditions skip the detector: {len(diff)}")
    print("    truths:", Counter(t for _, t, _ in diff).most_common(8))
    print("    added targets:", Counter(x for _, _, xs in diff for x in xs).most_common(6))

    # per-target tallies on the spec key
    rows = []
    for c in key:
        for cond, t in c["targets"].items():
            rows.append({"case_id": c["case_id"], "truth": c["truth"], "truth_tier": c["tier"], "target": cond,
                         "p": round(t["p"], 1), "band": band_label(band_of(t["p"])), "why": t["why"],
                         "red_herring": t["rh"], "undetermined": t["und"], "hallmarks": t["h"],
                         "cell_n": t["n"], "cell_rate": t["rate"], "in_R10": cond in c["R10"], "in_R5": cond in c["R5"]})
    T = pd.DataFrame(rows)
    write_csv(OUT / "key470_targets.csv", rows)
    dxa = T[T.why == "dxa"]
    print(f"[2] DXA-derived pairs at >= 5%: {len(dxa)}; red herring {dxa.red_herring.sum()}; undetermined {dxa.undetermined.sum()}; kept {(~dxa.red_herring).sum()}")
    print(f"    kept DXA-derived targets by band:\n{dxa[~dxa.red_herring].groupby('band').size().to_string()}")
    print(f"    kept DXA-derived targets by hallmark count: {dxa[~dxa.red_herring].hallmarks.value_counts().sort_index().to_dict()}")
    print(f"    kept DXA-derived targets by condition (R10 only):")
    print(dxa[(~dxa.red_herring) & dxa.in_R10].groupby("target").size().sort_values(ascending=False).to_string())

    # 3. the 5-10% band
    b5 = dxa[dxa.band == "5-10%"]
    print(f"[3] 5-10% band DXA-derived pairs: {len(b5)}; red herring {b5.red_herring.sum()}; undetermined {b5.undetermined.sum()}; kept {(~b5.red_herring).sum()}")
    kept5 = b5[~b5.red_herring]
    print("    kept by hallmarks:", kept5.hallmarks.value_counts().sort_index().to_dict())
    print("    kept, cell rate summary (truth rate of the reference class):")
    print(kept5.cell_rate.describe().round(3).to_string())
    print("    kept 5-10% targets with cell rate < 2%:", int((kept5.cell_rate < 0.02).sum()), "; < 5%:", int((kept5.cell_rate < 0.05).sum()))
    # 5-10% cells for tier-1 conditions with hallmarks=1: rate vs threshold
    cells = pd.read_csv(RH_DIR / "cells.csv")
    m = cells[(cells.detector == "M1p") & (cells.band == "5-10%")].copy()
    m["h"] = m["sub"].str.split("=").str[1].astype(int)
    m = m[m.condition.isin([c for c, t in tiers["v03"].items() if t == 1])]
    print("    5-10% cells, tier-1 conditions, hallmarks 1 (rate; n): the band's only borderline cells")
    print(m[m.h == 1][["condition", "n", "k", "rate"]].sort_values("rate").to_string(index=False))
    r5only = T[T.in_R5 & ~T.in_R10 & (T.why == "dxa")]
    r5only_cases = r5only.case_id.nunique()
    print(f"    R5-only DXA targets: {len(r5only)} in {r5only_cases} cases; cases with R5 but no R10 target: {sum(1 for c in key if c['R5'] and not c['R10'])}")

    # 4. undetermined
    und = dxa[dxa.undetermined & dxa.in_R10]
    print(f"[4] undetermined pairs kept as R10 targets: {len(und)}")
    print(und[["case_id", "truth", "target", "p", "hallmarks", "cell_n"]].to_string(index=False))

    # 5. kept DXA-derived R10 targets by (truth, target, hallmarks)
    kept = dxa[(~dxa.red_herring) & dxa.in_R10]
    g = kept.groupby(["truth", "target", "hallmarks"]).agg(n=("case_id", "size"), p_mean=("p", "mean"), rate=("cell_rate", "mean")).reset_index()
    g = g.sort_values("n", ascending=False)
    write_csv(OUT / "key470_kept_dxa_targets.csv", g.round(3).to_dict("records"))
    print("[5] kept DXA-derived R10 targets, top (truth, target, hallmarks):")
    print(g.head(30).round(3).to_string(index=False))
    # cases with R10 targets whose truth is tier 3
    t3 = [c for c in key if c["R10"] and c["tier"] == 3]
    print(f"    R10 cases whose truth is tier 3: {len(t3)}; truths: {Counter(c['truth'] for c in t3).most_common(10)}")
    # the hallmark tokens that keep targets alive with hallmarks == 1
    one = kept[kept.hallmarks == 1]
    tok = Counter()
    for r in one.itertuples():
        evf = df.loc[r.case_id].EVF
        for t in rh.hallmarks[r.target]:
            if t in evf:
                tok[(r.target, t)] += 1
    print("    single hallmark tokens that keep an R10 target:", tok.most_common(12))

    # 6. tier sensitivity denominators
    for name in ("severity", "nt", "v03", "one_source"):
        k = build_key(df, ids, tiers[name], rh)
        n10 = sum(1 for c in k if c["R10"])
        n5 = sum(1 for c in k if c["R5"])
        low = sum(1 for c in k if c["clearly_low"])
        nt = sum(len(c["R10"]) for c in k)
        print(f"[6] tiers={name}: R10 cases {n10}, R10 targets {nt}, R5 cases {n5}, clearly low {low}")
    # asthma's own contribution under v03
    asthma_t = [c for c in key if "Bronchospasm / acute asthma exacerbation" in c["R10"]]
    print(f"    asthma in R10: {len(asthma_t)} cases ({sum(1 for c in asthma_t if c['truth'].startswith('Bronchospasm'))} as truth)")
    for cond in ("Pneumonia", "Pulmonary neoplasm", "Pancreatic neoplasm"):
        n = [c for c in key if cond in c["R10"]]
        print(f"    {cond} in R10: {len(n)} cases ({sum(1 for c in n if c['truth'] == cond)} as truth)")


if __name__ == "__main__":
    main()
