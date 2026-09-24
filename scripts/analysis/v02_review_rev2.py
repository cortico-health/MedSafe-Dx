#!/usr/bin/env python3
"""Re-review checks for revision 2 of spec/v0.2-scoring.md.

Revision 2 keys the answer to the NTS urgency level of the true condition
(spec/acuity_reference_levels.csv, scale_level) with patient modifiers, draws
adults only (47 conditions, 10 each), scores A and B as crossings of the
urgent line (level <= 3), removes red-flag patients from B's denominator, and
makes the tolerance score the headline. We compute, from DDXPlus data only:

  1. the urgent share under the NTS key, with and without modifiers, and the
     size of B's denominator after the red-flag removal;
  2. how many sampled cases the modifiers change, and how many cross the line;
  3. constant policies (always U1..U5) and the DXA differential under the
     spec's reference rule, an argmax band rule, and a threshold sweep;
  4. cap-79 and hedging effects (they touch D2 and C only);
  5. cluster-bootstrap (47 conditions) against case-bootstrap interval widths;
  6. which off-list categories the D2 rule scores, by share of emitted codes.

Outputs: stdout and results/analysis/v02_review/rev2.json. Nothing in the
spec or evaluator changes.
"""
from __future__ import annotations

import ast
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "analysis"))
import v02_review_policies as vp  # noqa: E402
from v02_review_gaming import qwk, vp_score  # noqa: E402

ACUITY_CSV = REPO / "spec" / "acuity_reference_levels.csv"
OFFLIST_TALLY = REPO / "results" / "analysis" / "icd10_map" / "offlist_categories_tally.csv"
OUT = REPO / "results" / "analysis" / "v02_review" / "rev2.json"

URGENT_MAX = 3
U, O = 0.05, 0.35
BLEED = {"E_210", "E_140", "E_179", "E_45"}
IMMUNO = {"E_227", "E_2", "E_44"}
HEAD_LOCATIONS = {"V_89", "V_125", "V_126", "V_166", "V_167", "V_108", "V_109", "V_25", "V_62", "V_124"}
CHRONIC = {"E_123", "E_31", "E_124", "E_106", "E_69", "E_113", "E_126"}
IMMUNO_MOD = {"E_227", "E_2", "E_34"}
D2_U1_CATEGORIES = {"GI haemorrhage", "Intracranial haemorrhage", "CNS infection", "Aortic aneurysm and dissection", "Sepsis"}


def load_nts() -> dict[str, int]:
    with open(ACUITY_CSV) as f:
        return {r["condition"]: int(r["scale_level"]) for r in csv.DictReader(f)}


def modified_level(path: str, base: int, age: int, ev: set[str]) -> tuple[int, str]:
    """Apply the modifiers column of spec/acuity_reference_levels.csv. Returns (level, rule name or '')."""
    if path == "Bronchospasm / acute asthma exacerbation" and ev & {"E_101", "E_46"}:
        return 1, "asthma near-fatal risk"
    if path == "Pneumonia" and (age >= 65 or ev & {"E_123", "E_31", "E_106"} or ev & IMMUNO_MOD):
        return 2, "pneumonia risk group"
    if path in ("Influenza", "Bronchitis") and (age >= 65 or "E_167" in ev or ev & CHRONIC or ev & IMMUNO_MOD):
        return 3, f"{path.lower()} risk group"
    if path == "Acute otitis media" and ev & {"E_227", "E_69", "E_106", "E_123", "E_31", "E_113"}:
        return 3, "otitis chronic disease"
    if path in ("Acute rhinosinusitis", "Viral pharyngitis") and "E_227" in ev:
        return 3, f"{path.lower()} immunosuppressed"
    if path == "Whooping cough" and age < 1:
        return 3, "pertussis infant"
    if path == "Bronchiolitis" and age < 1 and ev & {"E_160", "E_139"}:
        return 2, "bronchiolitis risk"
    return base, ""


def red_flag(codes: list[str], vals: dict[str, int]) -> str:
    bases = {c.split("_@_")[0] for c in codes}
    if bases & BLEED:
        return "bleeding"
    head = any(c.startswith("E_55_@_") and c.split("_@_")[1] in HEAD_LOCATIONS for c in codes)
    if head and vals.get("E_59", 0) >= 7 and vals.get("E_56", 0) >= 8:
        return "thunderclap"
    if "E_91" in bases and bases & IMMUNO:
        return "fever_immuno"
    return ""


def ab(a_urgent: np.ndarray, t_urgent: np.ndarray, b_mask: np.ndarray) -> tuple[float, float]:
    A = float(np.mean(~a_urgent[t_urgent])) if t_urgent.any() else 0.0
    nonurg = ~t_urgent & b_mask
    B = float(np.mean(a_urgent[nonurg])) if nonurg.any() else 0.0
    return A, B


def bootstrap_ci(a_urgent, t_urgent, b_mask, cond_idx, n_cond, by_condition: bool, n_boot=2000, seed=20260923):
    rng = np.random.default_rng(seed)
    n = len(t_urgent)
    scores = []
    for _ in range(n_boot):
        if by_condition:
            picked = rng.integers(0, n_cond, size=n_cond)
            idx = np.concatenate([np.flatnonzero(cond_idx == c) for c in picked])
        else:
            idx = rng.integers(0, n, size=n)
        A, B = ab(a_urgent[idx], t_urgent[idx], b_mask[idx])
        scores.append(vp_score(A, B))
    return np.percentile(scores, [2.5, 97.5]).tolist()


def main() -> None:
    scales = vp.load_scales()
    nts = load_nts()
    rows = vp.read_rows()
    adults = [r for r in rows if r["age"] >= 18]
    rng = np.random.default_rng(vp.SEED)
    sample = vp.stratified(rows, rng, adults_only=True)
    n = len(sample)
    conds = sorted({r["path"] for r in sample})
    print(f"adult stratified sample: {n} cases over {len(conds)} conditions; age 65+: {sum(r['age'] >= 65 for r in sample)}")

    # ---- per-case keys
    t_base, t_mod, rule, flags, ddx = [], [], [], [], []
    for r in sample:
        codes = ast.literal_eval(r["ev"])
        bases = {c.split("_@_")[0] for c in codes}
        vals = {}
        for c in codes:
            p = c.split("_@_")
            if len(p) == 2 and p[1].isdigit():
                vals[p[0]] = int(p[1])
        b = nts[r["path"]]
        m, why = modified_level(r["path"], b, r["age"], bases)
        t_base.append(b); t_mod.append(m); rule.append(why); flags.append(red_flag(codes, vals)); ddx.append(scales["ddxplus"][r["path"]])
    t_base, t_mod, ddx = np.array(t_base), np.array(t_mod), np.array(ddx)
    urg_mod, urg_base, urg_ddx = t_mod <= URGENT_MAX, t_base <= URGENT_MAX, ddx <= URGENT_MAX
    flags = np.array(flags)
    b_mask = flags == ""

    # ---- 1. urgent share, denominators
    nat_urgent = np.mean([nts[r["path"]] <= URGENT_MAX for r in adults])
    print("\n1. Urgent line (level <= 3)")
    print(f"  NTS key, no modifiers: {urg_base.sum()} urgent of {n} ({100*urg_base.mean():.1f}%); with modifiers: {urg_mod.sum()} ({100*urg_mod.mean():.1f}%)")
    print(f"  DDXPlus severity key at the same line: {urg_ddx.sum()} urgent ({100*urg_ddx.mean():.1f}%); natural adult mix under NTS (no modifiers): {100*nat_urgent:.1f}% urgent")
    print(f"  NTS level mix in sample (with modifiers): {dict(sorted(Counter(t_mod.tolist()).items()))}")
    rf = Counter(flags.tolist()); rf.pop("", None)
    nonurg = ~urg_mod
    print(f"  red flags: {dict(rf)}; among non-urgent cases: {int((~b_mask & nonurg).sum())} removed, "
          f"B denominator = {int((nonurg & b_mask).sum())} of {int(nonurg.sum())} non-urgent cases")
    print("  red-flagged non-urgent cases by condition:", dict(Counter(r["path"] for r, f, u in zip(sample, flags, urg_mod) if f and not u).most_common(8)))
    se_b = np.sqrt(0.3 * 0.7 / (nonurg & b_mask).sum())
    print(f"  B at 30% on that denominator: 95% CI +/- {196*se_b:.1f} pts (binomial)")

    # ---- 2. modifiers
    changed = t_mod != t_base
    crossed = urg_mod != urg_base
    print("\n2. Modifiers")
    print(f"  cases changed: {int(changed.sum())} of {n}; crossing the urgent line: {int(crossed.sum())}")
    print("  by rule:", dict(Counter(w for w in rule if w)))
    # natural prevalence of the flip among adult patients of the affected conditions
    aff = defaultdict(lambda: [0, 0])
    for r in adults:
        if r["path"] in ("Influenza", "Bronchitis", "Pneumonia", "Acute otitis media", "Acute rhinosinusitis", "Viral pharyngitis",
                         "Bronchospasm / acute asthma exacerbation"):
            bases = {c.split("_@_")[0] for c in ast.literal_eval(r["ev"])}
            m, why = modified_level(r["path"], nts[r["path"]], r["age"], bases)
            aff[r["path"]][0] += 1
            aff[r["path"]][1] += bool(why)
    print("  share of adult DDXPlus patients the modifier fires on:", {k: f"{100*v[1]/v[0]:.0f}% of {v[0]}" for k, v in aff.items()})
    cond_idx = np.array([conds.index(r["path"]) for r in sample])
    mixed = [c for c in conds if len(set(urg_mod[cond_idx == conds.index(c)])) > 1]
    print("  conditions whose sampled patients now straddle the urgent line:", mixed)

    # ---- 3. policies
    print("\n3. Policies on the headline (A, B, tolerance score), NTS key with modifiers")
    dists = np.array([vp.sev_dist_from_diff(r["diff"], nts) for r in sample])  # DXA mass per NTS level, no modifiers
    p_urg = dists[:, :URGENT_MAX].sum(axis=1)
    diffs = [ast.literal_eval(r["diff"]) for r in sample]
    top1_lvl = np.array([nts[d[0][0]] for d in diffs])
    truth = [r["path"] for r in sample]

    def band_level(esc: np.ndarray) -> np.ndarray:
        """argmax level inside the chosen band, for Q."""
        out = np.empty(len(esc), int)
        for i, e in enumerate(esc):
            seg = dists[i, :URGENT_MAX] if e else dists[i, URGENT_MAX:]
            out[i] = (1 if e else URGENT_MAX + 1) + int(np.argmax(seg))
        return out

    pols = {}
    for L in range(1, 6):
        pols[f"always U{L}"] = np.full(n, L)
    pols["DXA, spec rule: escalate iff P(urgent) > 1/8"] = band_level(p_urg > 1.0 / 8)
    pols["DXA, argmax band: escalate iff P(urgent) >= 0.5"] = band_level(p_urg >= 0.5)
    pols["DXA, argmax level"] = np.argmax(dists, axis=1) + 1
    pols["DXA, top-1 condition's level"] = top1_lvl
    sweep = {}
    for thr in np.arange(0.1, 0.95, 0.05):
        a = band_level(p_urg >= thr)
        A, B = ab(a <= URGENT_MAX, urg_mod, b_mask)
        sweep[round(float(thr), 2)] = (100 * A, 100 * B, vp_score(A, B))
    best_thr = max(sweep, key=lambda k: sweep[k][2])
    pols[f"DXA, best threshold ({best_thr})"] = band_level(p_urg >= best_thr)
    pols["perfect (a = t)"] = t_mod
    pols["perfect, no modifiers (a = base level)"] = t_base
    pols["DDXPlus-severity key as a model"] = ddx
    results = {}
    print(f"  {'policy':50s} {'A':>6s} {'B':>6s} {'score':>6s} {'QWK':>6s} {'A ddx-key':>10s} {'B ddx-key':>10s}")
    for k, a in pols.items():
        A, B = ab(a <= URGENT_MAX, urg_mod, b_mask)
        A2, B2 = ab(a <= URGENT_MAX, urg_ddx, b_mask)
        s = vp_score(A, B)
        results[k] = {"A": 100 * A, "B": 100 * B, "score": s, "QWK": qwk(a, t_mod), "A_ddx": 100 * A2, "B_ddx": 100 * B2, "score_ddx": vp_score(A2, B2)}
        print(f"  {k:50s} {100*A:6.1f} {100*B:6.1f} {s:6.1f} {results[k]['QWK']:6.2f} {100*A2:10.1f} {100*B2:10.1f}")
    print("  threshold sweep (P(urgent) >= thr -> A, B, score):", {k: tuple(round(x, 1) for x in v) for k, v in sweep.items()})
    # what the spec rule escalates
    print(f"  spec rule escalates {100*np.mean(p_urg > 1/8):.0f}% of cases; P(urgent) distribution: "
          f"<0.125: {np.mean(p_urg < 0.125):.2f}, 0.125-0.5: {np.mean((p_urg >= 0.125) & (p_urg < 0.5)):.2f}, >=0.5: {np.mean(p_urg >= 0.5):.2f}")
    # A events of the best DXA policy by condition
    a_best = pols[f"DXA, best threshold ({best_thr})"]
    print("  best-threshold DXA A events by condition:", dict(Counter(tr for tr, aa, tu in zip(truth, a_best <= 3, urg_mod) if tu and not aa).most_common(8)))
    print("  best-threshold DXA B events by condition:", dict(Counter(tr for tr, aa, tu, m in zip(truth, a_best <= 3, urg_mod, b_mask) if m and not tu and aa).most_common(8)))

    # ---- 4. cap-79 and hedging
    top_p = np.array([float(d[0][1]) for d in diffs])
    correct = np.array([d[0][0] == tr for d, tr in zip(diffs, truth)])
    d2 = {thr: 100 * float(np.mean((top_p >= thr) & ~correct & (np.abs(top1_lvl - t_mod) >= 2))) for thr in (0.6, 0.7, 0.8)}
    d2_cap = {thr: 100 * float(np.mean((np.minimum(top_p, 0.79) >= thr) & ~correct & (np.abs(top1_lvl - t_mod) >= 2))) for thr in (0.6, 0.7, 0.8)}
    y = urg_mod.astype(float)
    base_brier = float(np.mean((y.mean() - y) ** 2))
    c_honest = 1 - float(np.mean((p_urg - y) ** 2)) / base_brier
    c_hedge = 1 - float(np.mean((0.6 - y) ** 2)) / base_brier
    print("\n4. Cap and hedge")
    print(f"  D2 per 100 at 60/70/80, DXA honest: {d2}; capped at 79: {d2_cap}; DXA top-1 p >= 0.8 on {100*np.mean(top_p >= 0.8):.1f}% of cases")
    print(f"  C skill: DXA honest {c_honest:.2f}; hedge (p1+p2+p3 = 60 on every case) {c_hedge:.2f}; neither touches A, B or the headline")

    # ---- 5. bootstrap
    print("\n5. Bootstrap interval for the headline score (2000 draws)")
    boot = {}
    for k in (f"DXA, best threshold ({best_thr})", "DXA, argmax level", "always U3"):
        a = pols[k] <= URGENT_MAX
        ci_case = bootstrap_ci(a, urg_mod, b_mask, cond_idx, len(conds), False)
        ci_cond = bootstrap_ci(a, urg_mod, b_mask, cond_idx, len(conds), True)
        boot[k] = {"case": ci_case, "condition": ci_cond}
        print(f"  {k:45s} point {results[k]['score']:5.1f}  case CI {ci_case[0]:5.1f}-{ci_case[1]:5.1f}  condition CI {ci_cond[0]:5.1f}-{ci_cond[1]:5.1f}")
    # a synthetic model whose A events sit in two conditions
    a_syn = urg_mod.copy()
    for c in ("Pulmonary embolism", "Unstable angina"):
        a_syn[cond_idx == conds.index(c)] = False
    a_syn = np.where(a_syn, 2, 5)
    A, B = ab(a_syn <= 3, urg_mod, b_mask)
    ci_case = bootstrap_ci(a_syn <= 3, urg_mod, b_mask, cond_idx, len(conds), False)
    ci_cond = bootstrap_ci(a_syn <= 3, urg_mod, b_mask, cond_idx, len(conds), True)
    print(f"  synthetic: misses every PE and unstable angina, else perfect: A {100*A:.1f} B {100*B:.1f} score {vp_score(A, B):.1f}  "
          f"case CI {ci_case[0]:.1f}-{ci_case[1]:.1f}  condition CI {ci_cond[0]:.1f}-{ci_cond[1]:.1f}")

    # ---- 6. off-list D2 coverage
    print("\n6. D2 off-list coverage (share of all emitted codes in the 18 stored runs)")
    if OFFLIST_TALLY.exists():
        with open(OFFLIST_TALLY) as f:
            tally = [(r["category"], int(r["count"]), float(r["share_of_all_codes"])) for r in csv.DictReader(f)]
        scored = sum(s for c, _, s in tally if c in D2_U1_CATEGORIES)
        unscored = sum(s for c, _, s in tally if c not in D2_U1_CATEGORIES)
        print(f"  off-list codes with a D2 level (U1 list): {100*scored:.1f}% of all codes; off-list codes left unscored: {100*unscored:.1f}%")
        print("  largest unscored off-list categories:", [(c, f"{100*s:.1f}%") for c, _, s in tally if c not in D2_U1_CATEGORIES][:12])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"policies": results, "sweep": sweep, "bootstrap": boot, "d2": {"honest": d2, "capped": d2_cap},
               "urgent_share": {"nts_mod": float(urg_mod.mean()), "nts_base": float(urg_base.mean()), "ddx": float(urg_ddx.mean()), "natural_adult_nts": float(nat_urgent)},
               "modifiers": {"changed": int(changed.sum()), "crossed": int(crossed.sum()), "rules": dict(Counter(w for w in rule if w))}},
              open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
