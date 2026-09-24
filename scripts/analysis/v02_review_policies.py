#!/usr/bin/env python3
"""Adversarial checks on the v0.2 scoring design (spec/v0.2-scoring.md).

We compute, from DDXPlus data only and with no inference:

  1. the severity mix of the planned stratified sample (10 per condition)
     against DDXPlus's natural test-split mix, on both severity scales;
  2. what constant-level policies ("always level k") score on measures A
     and B and on the combined cost 7A + B, under each mix and scale;
  3. what the DXA differential, acting as a Bayes decision-maker under the
     section 5 costs, scores, and which levels that Bayes action ever picks;
  4. the Kish effective sample size when the stratified sample is re-weighted
     to the natural mix;
  5. how many sampled cases carry the section 7 red-flag codes;
  6. the expected composition of measure E from the per-condition top-5 rates
     of the 18 stored runs (results/analysis/icd10_map/per_condition_topk.csv);
  7. simple precision figures for A, B and E at the planned sample size.

Outputs go to stdout and results/analysis/v02_review/policies.json. The
evaluator and the spec are not touched.
"""
from __future__ import annotations

import ast
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
TEST_CSV = REPO / "data" / "ddxplus_v0" / "release_test_patients"
CONDITIONS = REPO / "data" / "ddxplus_v0" / "release_conditions.json"
REFERENCE_CSV = REPO / "spec" / "ddxplus_severity_reference.csv"
PER_CONDITION_TOPK = REPO / "results" / "analysis" / "icd10_map" / "per_condition_topk.csv"
OUT_DIR = REPO / "results" / "analysis" / "v02_review"

SEED = 20260923
PER_CONDITION = 10
W_UNDER, W_OVER = 7.0, 1.0  # section 5 combined cost
BLEED = {"E_210", "E_140", "E_179", "E_45"}
IMMUNO = {"E_227", "E_2", "E_44"}
FEVER = "E_91"
LEVELS = np.arange(1, 6)


def load_scales() -> dict[str, dict[str, int]]:
    cond = json.loads(CONDITIONS.read_text())
    with open(REFERENCE_CSV) as f:
        ref = {r["condition"]: int(r["reference_level"]) for r in csv.DictReader(f)}
    return {"ddxplus": {k: int(v["severity"]) for k, v in cond.items()}, "reference": ref}


def read_rows() -> list[dict]:
    """Row index, pathology, age, sex, evidences and the raw differential string (parsed lazily)."""
    rows = []
    with open(TEST_CSV) as f:
        for i, r in enumerate(csv.DictReader(f)):
            rows.append({"i": i, "path": r["PATHOLOGY"], "age": int(r["AGE"]), "sex": r["SEX"],
                         "ev": r["EVIDENCES"], "diff": r["DIFFERENTIAL_DIAGNOSIS"]})
    return rows


def stratified(rows: list[dict], rng: np.random.Generator, adults_only: bool) -> list[dict]:
    by = defaultdict(list)
    for r in rows:
        if adults_only and r["age"] < 18:
            continue
        by[r["path"]].append(r)
    out = []
    for p in sorted(by):
        pool = by[p]
        k = min(PER_CONDITION, len(pool))
        idx = rng.choice(len(pool), size=k, replace=False)
        out.extend(pool[j] for j in sorted(idx))
    return out


def sev_dist_from_diff(diff_str: str, scale: dict[str, int]) -> np.ndarray:
    """Probability mass per level 1..5 from a DDXPlus differential."""
    d = np.zeros(5)
    for name, p in ast.literal_eval(diff_str):
        d[scale[name] - 1] += float(p)
    s = d.sum()
    return d / s if s > 0 else d


def cost_of_level(a: int, dist: np.ndarray) -> float:
    """Expected section 5 cost of stating level a: 7 x P(a - t >= 2) + 1 x P(t - a >= 2)."""
    t = LEVELS
    under = dist[(a - t) >= 2].sum()
    over = dist[(t - a) >= 2].sum()
    return W_UNDER * under + W_OVER * over


def constant_policy_table(mix: np.ndarray) -> list[dict]:
    """A, B and cost per 100 for 'always level k' when the true-level mix is `mix` (levels 1..5)."""
    out = []
    for a in LEVELS:
        A = mix[(a - LEVELS) >= 2].sum()
        B = mix[(LEVELS - a) >= 2].sum()
        out.append({"policy": f"always {a}", "A": 100 * A, "B": 100 * B, "cost": 100 * (W_UNDER * A + W_OVER * B)})
    return out


def rates(a: np.ndarray, t: np.ndarray, w: np.ndarray | None = None) -> dict:
    w = np.ones(len(t)) if w is None else w
    A = float(np.sum(w * ((a - t) >= 2)) / w.sum())
    B = float(np.sum(w * ((t - a) >= 2)) / w.sum())
    return {"A": 100 * A, "B": 100 * B, "cost": 100 * (W_UNDER * A + W_OVER * B)}


def kish_ess(w: np.ndarray) -> float:
    return float(w.sum() ** 2 / np.sum(w ** 2))


def se_rate(p: float, n: int) -> float:
    return math.sqrt(p * (1 - p) / n)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scales = load_scales()
    rows = read_rows()
    n_all = len(rows)
    adults = [r for r in rows if r["age"] >= 18]
    print(f"test split rows: {n_all}; adults (age >= 18): {len(adults)}; under 2: {sum(r['age'] < 2 for r in rows)}")

    # ---- 1. natural mix and stratified mix, both scales
    nat_counts = Counter(r["path"] for r in rows)
    nat_adult_counts = Counter(r["path"] for r in adults)
    conds = sorted(scales["ddxplus"])
    print("\nconditions with fewer than 10 test patients (all ages / adults):",
          [(c, nat_counts[c], nat_adult_counts[c]) for c in conds if min(nat_counts[c], nat_adult_counts[c]) < 10])
    rng = np.random.default_rng(SEED)
    sample = stratified(rows, rng, adults_only=False)
    rng2 = np.random.default_rng(SEED)
    sample_adult = stratified(rows, rng2, adults_only=True)
    print(f"stratified sample: {len(sample)} cases all ages ({sum(r['age'] < 18 for r in sample)} under 18, "
          f"{sum(r['age'] >= 65 for r in sample)} aged 65+); adults-only sample: {len(sample_adult)}")
    for s in sample:
        s["age_band"] = "<18" if s["age"] < 18 else ("65+" if s["age"] >= 65 else "18-64")
    peds = Counter(r["path"] for r in sample if r["age"] < 18)
    print("under-18 cases by condition in the all-ages sample:", dict(peds.most_common(8)))

    mixes = {}
    for scale_name, scale in scales.items():
        strat = np.array([sum(1 for r in sample if scale[r["path"]] == L) for L in LEVELS], float) / len(sample)
        nat = np.array([sum(nat_counts[c] for c in conds if scale[c] == L) for L in LEVELS], float) / n_all
        nat_ad = np.array([sum(nat_adult_counts[c] for c in conds if scale[c] == L) for L in LEVELS], float) / len(adults)
        mixes[scale_name] = {"stratified": strat, "natural": nat, "natural_adults": nat_ad}
    print("\nSeverity mix (share of cases at level 1..5):")
    for scale_name, m in mixes.items():
        for mix_name, v in m.items():
            print(f"  {scale_name:9s} {mix_name:15s} " + " ".join(f"{100*x:5.1f}" for x in v)
                  + f"   P(t<=2) = {100*v[:2].sum():.1f}%")
    top_nat = nat_counts.most_common(8)
    print("most common conditions in the natural test mix:",
          [(c, f"{100*n/n_all:.1f}%") for c, n in top_nat])
    print("rarest:", [(c, f"{100*n/n_all:.2f}%") for c, n in nat_counts.most_common()[-6:]])

    # ---- 2. constant policies
    print("\nConstant-level policies: A, B and combined cost (7A + B) per 100 cases")
    const = {}
    for scale_name, m in mixes.items():
        for mix_name, v in m.items():
            tab = constant_policy_table(v)
            const[f"{scale_name}/{mix_name}"] = tab
            print(f"  [{scale_name} severity, {mix_name} mix]")
            for row in tab:
                print(f"    {row['policy']:9s} A {row['A']:5.1f}  B {row['B']:5.1f}  cost {row['cost']:6.1f}")

    # ---- 3. DXA as a Bayes decision-maker, on the stratified sample
    print("\nDXA differential as a Bayes decision-maker under the section 5 costs (stratified sample):")
    dxa = {}
    nat_weight = np.array([nat_counts[r["path"]] / n_all / (1.0 / len(conds)) for r in sample])  # natural / uniform
    print(f"  Kish effective sample size after re-weighting to the natural mix: "
          f"{kish_ess(nat_weight):.1f} of {len(sample)} (weights range {nat_weight.min():.3f} to {nat_weight.max():.2f})")
    for scale_name, scale in scales.items():
        t = np.array([scale[r["path"]] for r in sample])
        dists = np.array([sev_dist_from_diff(r["diff"], scale) for r in sample])
        bayes = np.array([int(LEVELS[np.argmin([cost_of_level(a, d) for a in LEVELS])]) for d in dists])
        argmax = np.array([int(np.argmax(d)) + 1 for d in dists])
        top1 = np.array([scale[ast.literal_eval(r["diff"])[0][0]] for r in sample])
        in_diff = np.mean([any(n == r["path"] for n, _ in ast.literal_eval(r["diff"])) for r in sample])
        rank_true = [next(k for k, (n, _) in enumerate(ast.literal_eval(r["diff"])) if n == r["path"]) + 1 for r in sample]
        p_sev = dists[:, :2].sum(axis=1)
        y = (t <= 2).astype(float)
        brier = float(np.mean((p_sev - y) ** 2))
        base = y.mean()
        brier_base = float(np.mean((base - y) ** 2))
        bss = 1 - brier / brier_base
        res = {
            "bayes_action": rates(bayes, t), "bayes_action_natural_weights": rates(bayes, t, nat_weight),
            "argmax_level": rates(argmax, t), "top1_severity": rates(top1, t),
            "bayes_action_shares": {int(L): float(np.mean(bayes == L)) for L in LEVELS},
            "true_in_differential": float(in_diff), "true_rank_top5": float(np.mean(np.array(rank_true) <= 5)),
            "true_rank_1": float(np.mean(np.array(rank_true) == 1)),
            "C_brier": brier, "C_base_rate_brier": brier_base, "C_skill": bss,
            "irreducible_A_cases": int(np.sum((bayes - t) >= 2)), "irreducible_B_cases": int(np.sum((t - bayes) >= 2)),
        }
        dxa[scale_name] = res
        print(f"  [{scale_name}] true condition in DXA differential: {100*in_diff:.1f}%; in DXA top 5: "
              f"{100*res['true_rank_top5']:.1f}%; DXA rank 1: {100*res['true_rank_1']:.1f}%")
        for k in ("bayes_action", "bayes_action_natural_weights", "argmax_level", "top1_severity"):
            r = res[k]
            print(f"    {k:30s} A {r['A']:5.1f}  B {r['B']:5.1f}  cost {r['cost']:6.1f}")
        print("    Bayes action shares by level:", {k: f"{100*v:.0f}%" for k, v in res["bayes_action_shares"].items()})
        print(f"    C: DXA Brier {brier:.3f}, base-rate Brier {brier_base:.3f}, skill {bss:.2f}")
        # Under-triage events of the Bayes action, by true condition
        ev = Counter(r["path"] for r, a, tt in zip(sample, bayes, t) if a - tt >= 2)
        print("    Bayes-action A events by true condition:", dict(ev.most_common(8)))
        ev = Counter(r["path"] for r, a, tt in zip(sample, bayes, t) if tt - a >= 2)
        print("    Bayes-action B events by true condition:", dict(ev.most_common(8)))

    # ---- 5. section 7 red flags in the sample
    def flags(r):
        ev = set(e.split("_@_")[0] for e in ast.literal_eval(r["ev"]))
        return {"bleed": bool(ev & BLEED), "fever_immuno": FEVER in ev and bool(ev & IMMUNO)}
    fl = [flags(r) for r in sample]
    t_ddx = np.array([scales["ddxplus"][r["path"]] for r in sample])
    bleed_nonurgent = sum(1 for f, tt in zip(fl, t_ddx) if f["bleed"] and tt >= 4)
    print(f"\nSection 7 red flags in the stratified sample: bleeding codes {sum(f['bleed'] for f in fl)} cases "
          f"({bleed_nonurgent} with DDXPlus severity >= 4, so escalating them is a B event); "
          f"fever + immunosuppression {sum(f['fever_immuno'] for f in fl)}")
    bc = Counter(r["path"] for r, f in zip(sample, fl) if f["bleed"])
    print("  bleeding codes by condition:", dict(bc.most_common(10)))
    # natural prevalence of bleeding codes among severity >= 4 conditions (all rows, cheap)
    nb = 0; nn = 0
    for r in rows:
        if scales["ddxplus"][r["path"]] >= 4:
            nn += 1
            if any(e.split("_@_")[0] in BLEED for e in ast.literal_eval(r["ev"])):
                nb += 1
    print(f"  natural mix: {100*nb/nn:.1f}% of severity >= 4 patients carry a bleeding code")

    # ---- 6. expected composition of E from stored runs
    e_rows = []
    if PER_CONDITION_TOPK.exists():
        with open(PER_CONDITION_TOPK) as f:
            for r in csv.DictReader(f):
                if int(r["severity"]) <= 2:
                    miss = 1 - float(r["top5_map"])
                    e_rows.append((r["condition"], int(r["severity"]), int(r["model_cases"]), miss, PER_CONDITION * miss))
        e_rows.sort(key=lambda x: -x[4])
        tot = sum(x[4] for x in e_rows)
        n_sev = sum(1 for c in conds if scales["ddxplus"][c] <= 2) * PER_CONDITION
        print(f"\nMeasure E, projected from the 18 stored runs (pooled top-5 miss rate under the map, x 10 cases each):")
        print(f"  expected E events per model: {tot:.1f} of {n_sev} severe cases = {100*tot/n_sev:.1f} per 100")
        for c, s, n, miss, exp in e_rows[:8]:
            print(f"    {c:32s} sev {s}  pooled miss {100*miss:5.1f}%  -> {exp:4.1f} events ({100*exp/tot:4.1f}% of E)")
        print("  conditions with severity <= 2 absent from the stored runs' per-condition table:",
              [c for c in conds if scales["ddxplus"][c] <= 2 and c not in {x[0] for x in e_rows}])

    # ---- 7. precision
    print("\nPrecision at the planned sample (binomial SE, no clustering):")
    n = len(sample)
    for name, p, denom in [("A at 2% of 490", 0.02, n), ("A at 5% of 490", 0.05, n), ("A at 10% of the 50 severity-1 cases", 0.10, 50),
                           ("B at 30% of 490", 0.30, n), ("E at 20% of 170 severe cases", 0.20, 170), ("D2 at 3% of 490", 0.03, n)]:
        se = se_rate(p, denom)
        print(f"  {name:38s} SE {100*se:4.1f} pts, 95% CI +/- {196*se:4.1f} pts; "
              f"paired difference vs another model (rho 0.3): +/- {196*se*math.sqrt(2*0.7):4.1f} pts")
    # Cluster effect: 10 cases per condition, condition-level ICC
    for icc in (0.3, 0.5):
        deff = 1 + (PER_CONDITION - 1) * icc
        print(f"  design effect with condition ICC {icc}: {deff:.1f}, so 490 cases act like {n/deff:.0f} independent cases")

    json.dump({"mixes": {s: {m: v.tolist() for m, v in mm.items()} for s, mm in mixes.items()},
               "constant_policies": const, "dxa": dxa,
               "kish_ess_natural_reweight": kish_ess(nat_weight)},
              open(OUT_DIR / "policies.json", "w"), indent=1)
    print(f"\nwrote {OUT_DIR / 'policies.json'}")


if __name__ == "__main__":
    main()
