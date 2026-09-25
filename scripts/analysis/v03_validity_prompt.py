#!/usr/bin/env python3
"""Prompt effect (validity review, question 4): what the models' own p_serious says about
the YES bar in prompt v6.

Prompt v6 defines YES as "at least one potentially serious condition should be brought
to the clinician's attention". Both models answer YES on 95-98% of cases, so the binary
field carries little information. Each answer also carries p_serious, the model's
probability that the patient has a potentially serious condition. If p_serious
separates the key's groups while the YES field does not, the verdict field is the
prompt's bar, not the model's judgement, and a re-worded bar (or a threshold on the
probability) would recover the separation without a new run. We report:

    the p_serious distribution by key group (R10, intermediate, clearly low-risk);
    the AUC of p_serious for R10 against clearly low-risk cases;
    the safety cost if YES were read as p_serious >= t, for t from 10 to 70, under the
    draft-2 cost (H) and under H'' (flags must name a target);
    the share of YES answers with p_serious under 12.5% (the cost ratio's indifference
    point) and under 30%.

Outputs: results/analysis/v03_validity/p_serious_by_group.csv, p_serious_threshold.csv, prompt.log.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v03_validity_common as C  # noqa: E402

from evaluator import v03_score as vs  # noqa: E402


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


def main() -> None:
    log = C.log_to("prompt.log")
    out = C.ensure_out()
    L = C.load("main")
    key = L.key
    groups = {"R10": key.has_r10, "intermediate": ~key.has_r10 & ~key.clearly_low, "clearly_low": key.clearly_low}
    by_group, thr = [], []
    for m in C.MODELS:
        o = L.outcomes[m]
        ps = np.array([p.p_serious if (p.readable and p.p_serious is not None) else np.nan for p in o.parsed])
        yes = o.yes
        log(f"\n== {m}: YES rate {100 * yes.mean():.1f}%; p_serious present {np.isfinite(ps).sum()}/{key.n}")
        for g, mask in groups.items():
            v = ps[mask & np.isfinite(ps)]
            d = {"model": m, "group": g, "cases": int(mask.sum()), "yes_rate": round(100 * yes[mask].mean(), 1),
                 "p_median": float(np.median(v)), "p_q25": float(np.percentile(v, 25)), "p_q75": float(np.percentile(v, 75)),
                 "share_p_under_12.5": round(float((v < 12.5).mean()), 3), "share_p_under_30": round(float((v < 30).mean()), 3),
                 "yes_with_p_under_12.5": int((yes[mask] & (ps[mask] < 12.5)).sum()),
                 "yes_with_p_under_30": int((yes[mask] & (ps[mask] < 30)).sum())}
            by_group.append(d)
            log(f"  {g:13s} n {d['cases']:3d} YES {d['yes_rate']:5.1f}%  p_serious median {d['p_median']:4.0f} IQR {d['p_q25']:.0f}-{d['p_q75']:.0f}; "
                f"YES with p<12.5: {d['yes_with_p_under_12.5']}, p<30: {d['yes_with_p_under_30']}")
        a = auc(ps[key.has_r10 & np.isfinite(ps)], ps[key.clearly_low & np.isfinite(ps)])
        a_truth = auc(ps[(key.truth_tier == 1) & np.isfinite(ps)], ps[key.clearly_low & np.isfinite(ps)])
        log(f"  AUC of p_serious, R10 vs clearly low-risk: {a:.3f}; tier-1 truth vs clearly low-risk: {a_truth:.3f}")
        # YES read as a threshold on p_serious
        masks = C.event_masks(L, m)
        any_r10 = ~masks["H_dprime"] | ~yes  # cases where the flags name a target (when YES)
        hits_target = np.array([any(c in o.hits["standard"][i] for c in k.r10) for i, k in enumerate(key.keys)])
        log("  YES read as p_serious >= t:  t   YES%   H-cost   OC%   H''-cost   misses(H)")
        for t in (0, 10, 12.5, 20, 30, 40, 50, 60, 70):
            y = np.isfinite(ps) & (ps >= t)
            h = key.has_r10 & ~y
            oc = key.clearly_low & y
            cost = 100 * (vs.MISS * h + vs.CONCERN * oc).sum() / key.n
            hd = key.has_r10 & (~y | ~hits_target)
            cost_d = 100 * (vs.MISS * hd + vs.CONCERN * oc).sum() / key.n
            thr.append({"model": m, "t": t, "yes_rate": round(100 * y.mean(), 1), "SC_H": round(cost, 1),
                        "OC": round(100 * oc.sum() / key.clearly_low.sum(), 1), "SC_Hdprime": round(cost_d, 1),
                        "misses": int(h.sum())})
            log(f"    {t:5.1f}  {100 * y.mean():5.1f}  {cost:6.1f}  {100 * oc.sum() / key.clearly_low.sum():5.1f}  {cost_d:7.1f}   {int(h.sum())}")
        # agreement between the YES field and the probability
        disagree = yes & (ps < 12.5)
        log(f"  YES answers with p_serious under 12.5%: {int(disagree.sum())} ({100 * disagree.sum() / yes.sum():.1f}% of YES); "
            f"NO answers with p_serious >= 30%: {int((~yes & (ps >= 30)).sum())}")
    C.write_csv(out / "p_serious_by_group.csv", by_group)
    C.write_csv(out / "p_serious_threshold.csv", thr)


if __name__ == "__main__":
    main()
