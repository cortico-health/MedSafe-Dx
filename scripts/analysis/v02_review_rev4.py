#!/usr/bin/env python3
"""Final-review checks for revision 4 of spec/v0.2-scoring.md.

Revision 4 keys "serious" to the true condition's DDXPlus severity <= 2, and
counts over-triage (B) only on clearly low-risk patients: not serious, DXA
P(serious risk) < 12.5%, and no off-list red flag. We compute, from
data/test_sets/eval-v02-adult.json only (no inference):

  1. the sizes of A's and B's denominators, by condition;
  2. constant and simple policies on the headline: always, never, DXA at
     several thresholds, red-flag rules, age rules, and noisy DXA readers as
     stand-ins for a typical model;
  3. what the clearly-low-risk set looks like (alarming features it still
     carries, DXA mass distribution), to test the hybrid B for exploits;
  4. cluster (47 conditions) and case bootstrap intervals for the stand-ins;
  5. coherence checks on C and F under the 12.5% rule.

Output: stdout and results/analysis/v02_review/rev4.json.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "analysis"))
from v02_review_gaming import vp_score  # noqa: E402

EVAL = REPO / "data" / "test_sets" / "eval-v02-adult.json"
COND = REPO / "data" / "ddxplus_v0" / "release_conditions.json"
OUT = REPO / "results" / "analysis" / "v02_review" / "rev4.json"
T = 0.125
U, O = 0.05, 0.35
CHEST = {"V_29", "V_101", "V_55", "V_56", "V_159", "V_160", "V_170", "V_171"}
FOCAL = {"E_176", "E_63", "E_156", "E_52", "E_84", "E_83", "E_172", "E_180"}


def feats(c: dict) -> dict:
    codes = c["presenting_symptoms"]
    bases = {x.split("_@_")[0] for x in codes}
    vals = {}
    for x in codes:
        p = x.split("_@_")
        if len(p) == 2 and p[1].isdigit():
            vals[p[0]] = int(p[1])
    chest = any(x.startswith("E_55_@_") and x.split("_@_")[1] in CHEST for x in codes) or "E_14" in bases
    dysp = bool(bases & {"E_66", "E_64", "E_75", "E_67"})
    return {
        "chest_pain": chest, "dyspnea": dysp, "chest_pain_with_dyspnea": chest and dysp,
        "syncope": "E_159" in bases, "stridor": "E_194" in bases, "focal_neuro": bool(bases & FOCAL),
        "confusion": "E_39" in bases, "hemoptysis": "E_45" in bases,
        "gi_bleed": bool(bases & {"E_210", "E_140", "E_179"}), "severe_pain": vals.get("E_56", 0) >= 8,
        "palpitations": bool(bases & {"E_155", "E_164"}), "wheeze": bool(bases & {"E_214", "E_112"}),
        "fever": "E_91" in bases,
    }


NARROW = ("chest_pain_with_dyspnea", "syncope", "stridor", "focal_neuro", "confusion", "hemoptysis", "gi_bleed")
BROAD = NARROW + ("chest_pain", "dyspnea", "severe_pain", "palpitations", "wheeze")


def ab(esc: np.ndarray, serious: np.ndarray, b_den: np.ndarray) -> tuple[float, float]:
    A = float(np.mean(~esc[serious]))
    B = float(np.mean(esc[b_den])) if b_den.any() else 0.0
    return A, B


def boot(esc, serious, b_den, cond_idx, n_cond, by_condition, n_boot=2000, seed=20260923):
    rng = np.random.default_rng(seed)
    n = len(esc)
    out = []
    for _ in range(n_boot):
        if by_condition:
            picked = rng.integers(0, n_cond, size=n_cond)
            idx = np.concatenate([np.flatnonzero(cond_idx == c) for c in picked])
        else:
            idx = rng.integers(0, n, size=n)
        A, B = ab(esc[idx], serious[idx], b_den[idx])
        out.append(vp_score(A, B))
    return np.percentile(out, [2.5, 97.5]).tolist()


def main() -> None:
    cases = json.load(open(EVAL))["cases"]
    cond = json.load(open(COND))
    n = len(cases)
    sev = np.array([c["ddxplus_severity"] for c in cases])
    serious = sev <= 2
    p_risk = np.array([sum(p for name, p in c["ddxplus_differential"] if cond[name]["severity"] <= 2) for c in cases])
    at_risk = p_risk >= T
    red = np.array([bool(c["offlist_red_flag"]) for c in cases])
    low = ~serious & ~at_risk
    b_den = low & ~red
    age = np.array([c["age"] for c in cases])
    truth = [c["true_pathology"] for c in cases]
    conds = sorted(set(truth))
    cond_idx = np.array([conds.index(t) for t in truth])
    F = [feats(c) for c in cases]

    print(f"cases {n}; serious (A denominator) {serious.sum()}; at risk not serious (justified) {(at_risk & ~serious).sum()}; "
          f"clearly low-risk {low.sum()}; red-flagged among them {(low & red).sum()}; B denominator {b_den.sum()}")
    print("  B denominator by condition:", dict(Counter(t for t, m in zip(truth, b_den) if m).most_common(30)))
    print("  conditions with no case in the B denominator (not serious):",
          [c for c in conds if cond[c]["severity"] > 2 and not any(b_den[cond_idx == conds.index(c)])])
    print(f"  P(serious risk) among clearly low-risk: zero {np.mean(p_risk[low] == 0):.2f}, (0, 0.05] {np.mean((p_risk[low] > 0) & (p_risk[low] <= 0.05)):.2f}, "
          f"(0.05, 0.125) {np.mean(p_risk[low] > 0.05):.2f}")
    print(f"  P(serious risk) among serious patients: min {p_risk[serious].min():.3f}, share < 0.125: {np.mean(p_risk[serious] < T):.3f}, "
          f"share where true condition is DXA rank 1: {np.mean([c['ddxplus_differential'][0][0] == c['true_pathology'] for c, s in zip(cases, serious) if s]):.2f}")
    se_b = np.sqrt(0.3 * 0.7 / b_den.sum())
    print(f"  B at 30% on {b_den.sum()} cases: binomial 95% CI +/- {196*se_b:.1f} pts; A at 5% on {serious.sum()}: +/- {196*np.sqrt(0.05*0.95/serious.sum()):.1f} pts")

    # ---- alarming features inside the B denominator
    print("\nAlarming features carried by B-denominator cases (escalating them is a B event):")
    for k in ("chest_pain", "chest_pain_with_dyspnea", "dyspnea", "syncope", "severe_pain", "palpitations", "focal_neuro", "confusion", "fever"):
        cnt = sum(F[i][k] for i in np.flatnonzero(b_den))
        print(f"  {k:24s} {cnt:3d} of {b_den.sum()}  conditions: {dict(Counter(truth[i] for i in np.flatnonzero(b_den) if F[i][k]).most_common(4))}")

    # ---- policies
    print("\nPolicies on the headline")
    pols = {}
    pols["always escalate"] = np.ones(n, bool)
    pols["never escalate"] = np.zeros(n, bool)
    pols["DXA: escalate iff P(risk) >= 0.125 (spec reference)"] = at_risk
    for thr in (0.05, 0.25, 0.5):
        pols[f"DXA threshold {thr}"] = p_risk >= thr
    pols["DXA: any severe entry anywhere"] = p_risk > 0
    pols["DXA: top-1 condition is severe"] = np.array([cond[c["ddxplus_differential"][0][0]]["severity"] <= 2 for c in cases])
    pols["red flags, narrow (any of 7)"] = np.array([any(f[k] for k in NARROW) for f in F])
    pols["red flags, broad (any of 12)"] = np.array([any(f[k] for k in BROAD) for f in F])
    pols["age >= 65"] = age >= 65
    pols["age >= 50"] = age >= 50
    pols["age >= 50 or any broad red flag"] = (age >= 50) | pols["red flags, broad (any of 12)"]
    pols["mimic: escalate iff DXA at risk, else never"] = at_risk
    pols["mimic + escalate every serious patient (oracle A)"] = at_risk | serious
    pols["perfect: escalate iff serious"] = serious
    # noisy DXA readers as stand-ins for a typical model
    rng = np.random.default_rng(7)
    logit = np.log(np.clip(p_risk, 1e-3, 1 - 1e-3) / (1 - np.clip(p_risk, 1e-3, 1 - 1e-3)))
    for sigma in (1.0, 2.0):
        noisy = logit + rng.normal(0, sigma, n)
        pols[f"noisy DXA reader sigma {sigma}, threshold 0.125"] = noisy > np.log(T / (1 - T))
        pols[f"noisy DXA reader sigma {sigma}, threshold 0.5"] = noisy > 0
    res = {}
    print(f"  {'policy':55s} {'esc%':>5s} {'A':>6s} {'B':>6s} {'B_just':>6s} {'score':>6s} {'O=50':>6s}")
    for k, e in pols.items():
        A, B = ab(e, serious, b_den)
        Bj = float(np.mean(e[at_risk & ~serious]))
        res[k] = {"escalates": 100 * float(e.mean()), "A": 100 * A, "B": 100 * B, "B_justified": 100 * Bj,
                  "score": vp_score(A, B), "score_O50": vp_score(A, B, O=0.5)}
        print(f"  {k:55s} {100*e.mean():5.0f} {100*A:6.1f} {100*B:6.1f} {100*Bj:6.1f} {vp_score(A, B):6.1f} {vp_score(A, B, O=0.5):6.1f}")
    print("  A events of the DXA reference by condition:", dict(Counter(t for t, e, s in zip(truth, at_risk, serious) if s and not e)))

    # ---- bootstrap
    print("\nBootstrap 95% intervals for the headline (2000 draws)")
    bres = {}
    for k in ("noisy DXA reader sigma 1.0, threshold 0.125", "noisy DXA reader sigma 2.0, threshold 0.125",
              "red flags, broad (any of 12)", "DXA: escalate iff P(risk) >= 0.125 (spec reference)"):
        e = pols[k]
        ci_case = boot(e, serious, b_den, cond_idx, len(conds), False)
        ci_cond = boot(e, serious, b_den, cond_idx, len(conds), True)
        bres[k] = {"case": ci_case, "condition": ci_cond}
        print(f"  {k:55s} point {res[k]['score']:5.1f}  case {ci_case[0]:5.1f}-{ci_case[1]:5.1f}  condition {ci_cond[0]:5.1f}-{ci_cond[1]:5.1f}")

    # ---- C and F coherence
    print("\nC and F under the 12.5% rule")
    y = serious.astype(float)
    base = y.mean()
    brier_base = float(np.mean((base - y) ** 2))
    for name, p in (("DXA P(risk) as p_serious", p_risk), ("perfect", y), ("constant base rate", np.full(n, base)),
                    ("hedge 50", np.full(n, 0.5)), ("DXA clipped to [0.125, 1] (F-consistent with escalating everyone at risk)", np.maximum(p_risk, T))):
        brier = float(np.mean((p - y) ** 2))
        f_rate = float(np.mean(((p >= T) != at_risk)))  # F if the decision follows DXA
        print(f"  {name:70s} Brier {brier:.3f}  skill {1 - brier/brier_base:5.2f}")
    print(f"  base rate {base:.3f}; a calibrated model escalating at its own p_serious >= 12.5% escalates every case where P(serious | sheet) >= 12.5%")
    # how many serious cases does a model escalating at p_serious >= 12.5% catch if p_serious = DXA risk? = DXA reference A above.

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"sizes": {"serious": int(serious.sum()), "justified": int((at_risk & ~serious).sum()), "clearly_low": int(low.sum()),
                         "red_in_low": int((low & red).sum()), "b_denominator": int(b_den.sum())},
               "policies": res, "bootstrap": bres}, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
