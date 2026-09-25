#!/usr/bin/env python3
"""Gaming check for the candidate failure definitions (validity review, question 2).

We score the reference rows and a few uninformed policies under each candidate
definition (H, H', H'', H'' on truth targets) with the same 7:1 cost, on the 470
sample and on both pools, to see which definitions a fixed flag list can satisfy.
Policies beyond the committed references:

    always YES, no flags
    always YES, fixed 5 (the committed always-yes row: PSVT, MI, C34, stable and unstable angina)
    always YES, fixed 5 tuned to cover the most R10 targets on this sample (greedy)
    always YES, flags = DXA's top 5 tier-1 conditions by p (an uninformed reader of the
        differential; it needs DXA, which a model does not have, so it bounds what a
        DXA-shaped guess can earn)
    the DXA reader and naive Bayes with their own flags, and with YES forced everywhere

Outputs: results/analysis/v03_validity/gaming_<set>.csv and gaming.log.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v03_validity_common as C  # noqa: E402

from evaluator import v03_score as vs  # noqa: E402


def greedy_list(L: C.Loaded, k: int = 5) -> list[str]:
    """The k tier-1 conditions that together cover the most R10 cases (greedy)."""
    cover = {c: {i for i, kk in enumerate(L.key.keys) if c in kk.r10} for c in L.tiers if L.tiers[c] == 1}
    chosen: list[str] = []
    covered: set[int] = set()
    for _ in range(k):
        best = max(cover, key=lambda c: (len(cover[c] - covered), c))
        chosen.append(best)
        covered |= cover[best]
    return chosen


def with_answers(L: C.Loaded, name: str, preds: list[dict]) -> None:
    L.rows[name] = preds
    L.outcomes[name] = vs.outcomes(preds, L.key, L.matcher, policies=("standard", "strict", "lenient"))


def policies(L: C.Loaded) -> list[str]:
    canon = L.matcher.cmap.canonical
    ids = L.key.case_ids
    tuned = greedy_list(L)
    with_answers(L, "always-yes-no-flags", [{"case_id": c, "serious_concern": "YES", "flags": []} for c in ids])
    with_answers(L, "always-yes-tuned-5", [{"case_id": c, "serious_concern": "YES", "flags": [canon[x] for x in tuned]} for c in ids])
    dxa_top = []
    for c in ids:
        t1 = sorted(((cond, p) for cond, p in L.dxa[c].items() if L.tiers.get(cond) == 1), key=lambda kv: -kv[1])[:5]
        dxa_top.append({"case_id": c, "serious_concern": "YES", "flags": [canon[cond] for cond, _ in t1]})
    with_answers(L, "always-yes-dxa-top5-tier1", dxa_top)
    for ref in ("dxa", "naive-bayes"):
        forced = [{**p, "serious_concern": "YES"} for p in L.rows[ref]]
        with_answers(L, f"{ref}-forced-yes", forced)
    return ["perfect", "always-yes-no-flags", "always-yes", "always-yes-tuned-5", "always-yes-dxa-top5-tier1",
            "dxa", "dxa-forced-yes", "naive-bayes", "naive-bayes-forced-yes", "always-no"], tuned


def table(L: C.Loaded, rows: list[str], log) -> list[dict]:
    out = []
    n = L.key.n
    hdr = f"  {'row':28s} " + " ".join(f"{c:>8s}" for c in C.CANDIDATES) + "   events(H/H'/H''/Ht)  OC  COV"
    log(hdr)
    for r in rows:
        masks = C.event_masks(L, r)
        cov = vs.coverage(L.outcomes[r], L.key)
        d = {"set": L.name, "row": r}
        for cand in C.CANDIDATES:
            d[f"SC_{cand}"] = round(C.sc(L, r, cand), 1)
            d[f"events_{cand}"] = int(masks[cand].sum())
        d["OC_events"] = int(masks["OC"].sum())
        d["COV"] = round(100 * float(np.nanmean(cov)), 1) if np.isfinite(cov).any() else None
        out.append(d)
        log(f"  {r:28s} " + " ".join(f"{d[f'SC_{c}']:8.1f}" for c in C.CANDIDATES)
            + f"   {d['events_H']}/{d['events_H_prime']}/{d['events_H_dprime']}/{d['events_H_truth']}  {d['OC_events']}  {d['COV']}")
    return out


def main() -> None:
    log = C.log_to("gaming.log")
    out = C.ensure_out()
    for name in ("main", "pool-atypical", "pool-high-risk"):
        L = C.load(name)
        rows, tuned = policies(L)
        log(f"\n== {name}: {L.key.n} cases, {int(L.key.has_r10.sum())} R10, {int(L.key.clearly_low.sum())} clearly low-risk; "
            f"tuned list {tuned}")
        log("   SC under each candidate definition (7:1 cost), models first")
        t = table(L, list(C.MODELS) + rows, log)
        C.write_csv(out / f"gaming_{name}.csv", t)
        if name == "main":
            # How far a fixed list can get under H'': the best k for k = 1..5, greedy
            log("\n   Greedy fixed-list cover of R10 cases (share of the 234), k = 1..5:")
            cover = {c: {i for i, kk in enumerate(L.key.keys) if c in kk.r10} for c in L.tiers if L.tiers[c] == 1}
            covered: set[int] = set()
            for k in range(1, 6):
                best = max(cover, key=lambda c: (len(cover[c] - covered), c))
                covered |= cover[best]
                log(f"     k={k}: + {best:40s} covers {len(covered)}/{int(L.key.has_r10.sum())} = {100 * len(covered) / L.key.has_r10.sum():.0f}%")


if __name__ == "__main__":
    main()
