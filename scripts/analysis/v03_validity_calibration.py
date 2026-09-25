#!/usr/bin/env python3
"""Calibration of the headline (validity review, question 3).

For each candidate failure definition we put the two models and the reference rows on
one scale, and around each model we print a label-noise range: the model's SC under
every committed key variant (severity-only and one-source tiers, R12.5 and R20 targets,
truth-only targets, the strict and lenient code maps) and under a key relabelled from
the DDXPlus validation split. The relabelled key keeps the committed hallmarks and
re-reads every (case, tier-1 condition) class rate from release_validate_patients
adults, as GPT-6 Astra's review did (docs/v0.3-astra-review.md finding 6), so a target
that owes its status to one split's counts changes status.

Human triage error rates are printed as context from docs/triage-scale-anchor.md
section 4b; the task differs (urgency levels against an expert panel, not a binary
concern against a synthetic truth), and the review says where the comparison holds.

Outputs: results/analysis/v03_validity/calibration_<candidate>.csv,
relabel_validation.csv (the target-status changes), calibration.log.
"""

from __future__ import annotations

import ast
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v03_validity_common as C  # noqa: E402

from evaluator import answer_key_v03 as ak  # noqa: E402
from evaluator import v03_score as vs  # noqa: E402

VALIDATE = C.ROOT / "data/ddxplus_v0/release_validate_patients"
HALLMARKS = C.ROOT / "results/analysis/v03_key/hallmarks.csv"
CACHE = C.OUT / "cache_validate_classes.csv"


def load_hallmarks() -> dict[str, list[str]]:
    hm: dict[str, list[str]] = defaultdict(list)
    with open(HALLMARKS, newline="") as f:
        for r in csv.DictReader(f):
            hm[r["condition"]].append(r["token"])
    return hm


def validate_classes(tier1: list[str], hm: dict[str, list[str]]) -> dict[tuple[str, int, int], tuple[int, int]]:
    """(condition, band, hallmark count) -> (n, k) over validation-split adults, DXA p >= 5%."""
    if CACHE.exists():
        out = {}
        with open(CACHE, newline="") as f:
            for r in csv.DictReader(f):
                out[(r["condition"], int(r["band"]), int(r["hallmarks"]))] = (int(r["n"]), int(r["k"]))
        return out
    n: Counter = Counter()
    k: Counter = Counter()
    t1 = set(tier1)
    with open(VALIDATE, newline="") as f:
        for r in csv.DictReader(f):
            if int(r["AGE"]) < 18:
                continue
            ev = frozenset(ast.literal_eval(r["EVIDENCES"]))
            truth = r["PATHOLOGY"]
            for cond, p in ast.literal_eval(r["DIFFERENTIAL_DIAGNOSIS"]):
                if cond not in t1 or p < 0.05:
                    continue
                h = sum(1 for t in hm.get(cond, []) if t in ev)
                key = (cond, ak.band_of(100 * p), h)
                n[key] += 1
                k[key] += truth == cond
    out = {key: (n[key], k[key]) for key in n}
    C.ensure_out()
    with open(CACHE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "band", "hallmarks", "n", "k"], lineterminator="\n")
        w.writeheader()
        for (cond, b, h), (nn, kk) in sorted(out.items()):
            w.writerow({"condition": cond, "band": b, "hallmarks": h, "n": nn, "k": kk})
    return out


def relabelled_key(L: C.Loaded, classes, hm, log) -> tuple[vs.SetKey, list[dict]]:
    """The main key with every DXA-derived pair's status re-read from the validation classes."""
    import copy

    keys = {}
    changes = []
    for cid, k in zip(L.key.case_ids, L.key.keys):
        ev = frozenset(L.case_by_id[cid]["presenting_symptoms"])
        nk = copy.deepcopy(k)
        for cond, t in nk.considered.items():
            if t.source != "dxa":
                continue
            h = sum(1 for tok in hm.get(cond, []) if tok in ev)
            n, kk = classes.get((cond, ak.band_of(t.dxa_p), h), (0, 0))
            status = ak.rh_status(t.dxa_p, n, kk)
            in_r5 = status != ak.RED_HERRING
            in_r10 = in_r5 and t.dxa_p >= ak.R10
            if (in_r10, in_r5) != (t.in_r10, t.in_r5) or status != t.status:
                changes.append({"case_id": cid, "truth": k.truth, "condition": cond, "dxa_p": round(t.dxa_p, 1),
                                "hallmarks": h, "test_status": t.status, "test_n": t.class_n,
                                "test_rate": t.class_rate, "validate_status": status, "validate_n": n,
                                "validate_rate": round(kk / n, 4) if n else None,
                                "r10_before": t.in_r10, "r10_after": in_r10, "r5_before": t.in_r5, "r5_after": in_r5})
            t.status, t.undetermined, t.in_r10, t.in_r5 = status, status == ak.UNDETERMINED, in_r10, in_r5
            t.class_n, t.class_rate = n, (kk / n if n else None)
        nk.clearly_low_risk = not nk.r5 and nk.truth_tier == 3 and not nk.red_flag
        nk.intermediate = not nk.r10 and not nk.clearly_low_risk
        keys[cid] = nk
    vk = vs.set_key("validate-relabel", L.key.case_ids, keys, tiers=L.tiers)
    log(f"  validation relabel: {len(changes)} pair-status changes; R10 cases {int(vk.has_r10.sum())} (was {int(L.key.has_r10.sum())}), "
        f"clearly low-risk {int(vk.clearly_low.sum())} (was {int(L.key.clearly_low.sum())}); "
        f"R10 gained {sum(1 for c in changes if c['r10_after'] and not c['r10_before'])}, "
        f"lost {sum(1 for c in changes if c['r10_before'] and not c['r10_after'])}")
    return vk, changes


def sc_under(L: C.Loaded, row: str, cand: str, key: vs.SetKey, policy: str = "standard") -> float:
    """SC for one row under a candidate definition on an alternative key (same cases)."""
    o = L.outcomes[row]
    yes = o.yes
    hits = o.hits[policy]
    tgt = key.has_r10
    t1 = np.array([any(key.tiers.get(c) == 1 for c in h) for h in hits])
    any_r10 = np.array([any(c in hits[i] for c in k.r10) for i, k in enumerate(key.keys)])
    truth_t1 = key.truth_tier == 1
    truth_flagged = np.array([k.truth in hits[i] for i, k in enumerate(key.keys)])
    ev = {"H": tgt & ~yes, "H_prime": tgt & (~yes | ~t1), "H_dprime": tgt & (~yes | ~any_r10),
          "H_truth": truth_t1 & (~yes | ~truth_flagged)}[cand]
    cost = vs.MISS * ev + vs.CONCERN * (key.clearly_low & yes)
    return float(100 * cost.sum() / key.n)


def threshold_key(L: C.Loaded, t: float) -> vs.SetKey:
    """Targets at DXA >= t (truth always), clearly low-risk unchanged."""
    import copy

    keys = {}
    for cid, k in zip(L.key.case_ids, L.key.keys):
        nk = copy.deepcopy(k)
        for cond, tt in nk.considered.items():
            if tt.source == "dxa":
                tt.in_r10 = tt.status != ak.RED_HERRING and tt.dxa_p >= t
        keys[cid] = nk
    return vs.set_key(f"R{t:g}", L.key.case_ids, keys, tiers=L.tiers)


def truth_only_key(L: C.Loaded) -> vs.SetKey:
    import copy

    keys = {}
    for cid, k in zip(L.key.case_ids, L.key.keys):
        nk = copy.deepcopy(k)
        for cond, tt in nk.considered.items():
            if tt.source == "dxa":
                tt.in_r10 = False
        keys[cid] = nk
    return vs.set_key("truth-only", L.key.case_ids, keys, tiers=L.tiers)


def main() -> None:
    log = C.log_to("calibration.log")
    out = C.ensure_out()
    L = C.load("main")
    tier1 = [c for c, t in L.tiers.items() if t == 1]
    hm = load_hallmarks()
    log("== Validation-split classes (frozen hallmarks)")
    classes = validate_classes(tier1, hm)
    log(f"  {len(classes)} (condition, band, hallmark) classes from validation adults")
    vkey, changes = relabelled_key(L, classes, hm, log)
    C.write_csv(out / "relabel_validation.csv", changes)
    for c in changes:
        if c["r10_before"] != c["r10_after"]:
            log(f"    R10 change: {c['case_id']} ({c['truth']}) {c['condition']} DXA {c['dxa_p']}%: test {c['test_status']} "
                f"{c['test_n']} n, rate {c['test_rate']}; validate {c['validate_status']} {c['validate_n']} n, rate {c['validate_rate']}")

    variants: dict[str, tuple[vs.SetKey, str]] = {"primary": (L.key, "standard")}
    for v, (kp, sp, _) in vs.TIER_VARIANTS.items():
        variants[f"tiers:{v}"] = (vs.set_key(v, L.key.case_ids, ak.load_key(kp, sp), tiers=vs._variant_tiers(v)), "standard")
    variants["R12.5"] = (threshold_key(L, 12.5), "standard")
    variants["R20"] = (threshold_key(L, 20.0), "standard")
    variants["truth-only"] = (truth_only_key(L), "standard")
    variants["map:strict"] = (L.key, "strict")
    variants["map:lenient"] = (L.key, "lenient")
    variants["validate-relabel"] = (vkey, "standard")

    refs = ["perfect", "always-yes", "always-no", "dxa", "naive-bayes"]
    for cand, desc in C.CANDIDATES.items():
        log(f"\n== {desc}")
        rows = []
        log(f"  {'row':16s} {'SC':>7s} {'95% CI':>15s}  " + " ".join(f"{v[:12]:>12s}" for v in variants if v != "primary") + "   noise range")
        for r in list(C.MODELS) + refs:
            cost = C.candidate_cost(L, r, cand)
            lo, hi = C.cluster_ci(L, cost)
            d = {"candidate": cand, "row": r, "SC": round(C.sc(L, r, cand), 1), "ci_lo": round(lo, 1), "ci_hi": round(hi, 1)}
            vals = {}
            for v, (k, pol) in variants.items():
                if v == "primary":
                    continue
                vals[v] = sc_under(L, r, cand, k, pol)
                d[f"SC_{v}"] = round(vals[v], 1)
            d["noise_lo"], d["noise_hi"] = round(min(vals.values()), 1), round(max(vals.values()), 1)
            rows.append(d)
            log(f"  {r:16s} {d['SC']:7.1f} [{lo:6.1f},{hi:6.1f}]  " + " ".join(f"{vals[v]:12.1f}" for v in vals)
                + f"   {d['noise_lo']:.1f}-{d['noise_hi']:.1f}")
        C.write_csv(out / f"calibration_{cand}.csv", rows)

    # separation between the two models under each candidate, paired condition bootstrap
    log("\n== Paired difference Terra minus OSS (SC), condition bootstrap")
    M = vs.cluster_draws(L.key.k, 2000, vs.BOOTSTRAP_SEED)
    for cand in C.CANDIDATES:
        st = vs.Stats(L.key)
        d = C.candidate_cost(L, "gpt-5.6-terra", cand) - C.candidate_cost(L, "gpt-oss-120b", cand)
        st.add("d", d, np.ones(L.key.n), 100.0)
        pt, dr = st.evaluate(M)
        lo, hi = vs.interval(dr["d"])
        log(f"  {cand:9s} {pt['d']:6.1f} [{lo:6.1f}, {hi:6.1f}]  separated: {'yes' if hi < 0 or lo > 0 else 'no'}")

    log("\n== Human context (docs/triage-scale-anchor.md section 4b; a different task)")
    log("  Smits 2020, NTS, 116 raters x 40 paper cases against an expert panel: 62.3% exact, 17.4% under-triage, 20.2% over-triage, 77% of disagreements one level")
    log("  Giesen/Derkx 2016, Dutch GP telephone urgency, written cases: 63.6% adequate, 17.1% under, 19.3% over")
    log("  Giesen 2007, mystery patients: 69% correct, 19% under; sensitivity 0.76, specificity 0.95")
    log("  Graversen 2020 (Danish out-of-hours) is not in the repository's docs; not cited.")


if __name__ == "__main__":
    main()
