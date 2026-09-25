"""v0.3 headline design: a continuous cost with over-concern priced at 1/7 of false reassurance.

We score every policy and every v0.1 stand-in with one per-case cost,

    cost_i = 7 x [case has an R10 target and the answer is not YES]
           + 1 x [case is clearly low-risk and the answer is YES],

and report the headline as cost points per 100 patients (lower is better). The script answers the
design questions in docs/v0.3-headline-design.md:

1. cost function: which cases pay the 1/7, flat against weighted costs, normalisation, unreadable output,
   and the Bayes-optimal threshold (YES when P(target) >= 1/8), checked on the naive-Bayes and DXA readers;
2. gaming: simple policies on the 470 sample against informed readers;
3. coverage: what folding COV into the headline does to the gamer's rank;
4. sensitivity: the ratio at 1/5 and 1/10, truth-only targets, H' (YES with no tier-1 flag = reassurance),
   the rate form 7H + OC, and wider over-concern denominators;
5. stand-ins: the 19 v0.1 rows on the 250 set, with condition-cluster intervals and paired differences.

Key, matching and measures come from v03_review_common (spec text reading: 21 tier-1 conditions, all
filtered). Flags match under equivalent, narrower and broader codes, as the review recommends.
No inference is spent and no existing file is edited.

Run: .venv/bin/python scripts/analysis/v03_headline_design.py   (about two minutes; naive Bayes trains once)
Outputs: results/analysis/v03_headline/
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from v03_review_common import (OUT as REVIEW_OUT, ROOT, Matcher, RedHerring, build_key, load_adults,  # noqa: F401
                               load_tiers, sample_ids, score, write_csv)
from v03_review_gaming import CHEST_PAIN, COUGH, DYSPNOEA, WHEEZE, greedy_cover, has, top_by_target_frequency

sys.path.insert(0, str(ROOT / "scripts"))
from evaluator import v02_references as refs  # noqa: E402
from evaluator.v03_anchors import dxa_reader_conditions  # noqa: E402
from v03_review_models import load_rows  # noqa: E402

OUT = ROOT / "results/analysis/v03_headline"
NB_CACHE = OUT / "nb_counts_holdout470_250.json"
BROADER = ("equivalent", "narrower", "broader")
MISS, OC = 7.0, 1.0
N_BOOT = 2000
SEED = 20260923

log_lines: list[str] = []


def log(*parts) -> None:
    s = " ".join(str(p) for p in parts)
    print(s)
    log_lines.append(s)


# ---------------------------------------------------------------- the cost


def tier1_flagged(flags: list[str], matcher: Matcher, tiers: dict) -> bool:
    return any(tiers.get(c) == 1 for c in matcher.conditions_hit(flags, BROADER))


def case_costs(key: list[dict], answers: list[dict], matcher: Matcher, tiers: dict, miss: float = MISS, oc: float = OC,
               target_field: str = "R10", oc_field: str = "clearly_low", hprime: bool = False,
               weighted: bool = False) -> np.ndarray:
    """Per-case cost. An unreadable answer (yes = None) scores as NO everywhere.

    hprime: a YES whose flags name no tier-1 condition counts as reassurance on target cases.
    weighted: a miss costs `miss` x min(1, evidence / 12.5%), where evidence is 100% for a tier-1 truth and
              the DXA p of the strongest DXA-derived target otherwise; over-concern is unchanged.
    """
    out = np.zeros(len(key))
    for i, (k, a) in enumerate(zip(key, answers)):
        yes = a["yes"] is True
        flags = a["flags"][:5]
        tgt = k[target_field]
        if tgt:
            reassured = not yes or (hprime and not tier1_flagged(flags, matcher, tiers))
            if reassured:
                w = 1.0
                if weighted:
                    ev = 100.0 if k["tier"] == 1 and k["truth"] in tgt else max(k["targets"][c]["p"] for c in tgt)
                    w = min(1.0, ev / 12.5)
                out[i] += miss * w
        if k[oc_field] and yes:
            out[i] += oc
    return out


def per100(costs: np.ndarray) -> float:
    return 100.0 * costs.sum() / len(costs)


def rate_form(key, answers, miss=MISS, oc=OC, target_field="R10", oc_field="clearly_low") -> float:
    """7 x H + OC with H and OC as percentage rates (the L1 tolerance distance x 35)."""
    h_ev = h_den = o_ev = o_den = 0
    for k, a in zip(key, answers):
        if k[target_field]:
            h_den += 1
            h_ev += a["yes"] is not True
        if k[oc_field]:
            o_den += 1
            o_ev += a["yes"] is True
    return miss * 100.0 * h_ev / h_den + oc * 100.0 * o_ev / o_den


def cluster_ci(key: list[dict], costs: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float]:
    conds = sorted({k["truth"] for k in key})
    idx = {c: np.array([i for i, k in enumerate(key) if k["truth"] == c]) for c in conds}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(conds), len(conds))
        sel = np.concatenate([idx[conds[j]] for j in pick])
        draws.append(100.0 * costs[sel].sum() / len(sel))
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


# ---------------------------------------------------------------- readers


def make_nb(ids_holdout: set[str]):
    counts = refs.train_counts(ids_holdout, cache=NB_CACHE)
    return refs.NaiveBayes(counts)


def nb_answers(nb, key, tiers, canon, t: float, cap: int = 5, flag_rule: str = "tier1"):
    """YES when the posterior mass on tier-1 conditions is >= t%. Flags: the top tier-1 conditions by posterior."""
    out = []
    t1 = np.array([tiers.get(c) == 1 for c in nb.conditions])
    for k in key:
        p = nb.posterior(k["evf"])
        mass = 100.0 * p[t1].sum()
        order = np.argsort(-p)
        if flag_rule == "tier1":
            conds = [nb.conditions[j] for j in order if t1[j] and p[j] >= 0.01][:cap]
        else:
            conds = [nb.conditions[j] for j in order[:cap]]
        out.append({"yes": bool(mass >= t), "flags": [canon[c] for c in conds], "mass": float(mass)})
    return out


def dxa_answers(key, tiers, rh, canon, t: float, rule: str = "max", cap: int = 5):
    """rule 'max': the DXA reader of spec section 8 at threshold t (evaluator/v03_anchors.py
    `dxa_reader_conditions`, the one rule every script uses): YES when some tier-1 condition has p >= t and is
    not a red herring; flags = those conditions.
    rule 'mass': YES when the red-herring-filtered tier-1 mass (DXA p >= 5%) is >= t (the v0.2 at-risk rule
    under the v0.3 filter); flags = the reader's conditions at 5%, the ones the mass counts."""
    out = []
    for k in key:
        def red_herring(c, p, evf=k["evf"]):
            return rh.label(c, p, evf)[0]
        live5 = dxa_reader_conditions(k["dxa"], tiers, red_herring, 5.0, cap=len(k["dxa"]))
        mass = sum(k["dxa"][c] for c in live5)
        flags = dxa_reader_conditions(k["dxa"], tiers, red_herring, t, cap) if rule == "max" else live5[:cap]
        yes = bool(flags) if rule == "max" else mass >= t
        out.append({"yes": bool(yes), "flags": [canon[c] for c in flags], "mass": float(mass)})
    return out


# ---------------------------------------------------------------- main


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_adults()
    tiers = load_tiers()["v03"]
    rh = RedHerring()
    m = Matcher()
    canon = m.cmap.canonical
    code = lambda conds: [canon[c] for c in conds]
    ids470 = sample_ids("470")
    ids250 = sample_ids("250")
    key = build_key(df, ids470, tiers, rh)
    key250 = build_key(df, ids250, tiers, rh)
    nb = make_nb(set(ids470) | set(ids250))

    def counts(kk):
        n = len(kk)
        return {"n": n, "R10": sum(1 for k in kk if k["R10"]), "R5": sum(1 for k in kk if k["R5"]),
                "clearly_low": sum(1 for k in kk if k["clearly_low"]),
                "tier2_no_R5": sum(1 for k in kk if not k["R5"] and k["tier"] == 2),
                "tier3_redflag_no_R5": sum(1 for k in kk if not k["R5"] and k["tier"] == 3 and k["red_flag"]),
                "R5_only": sum(1 for k in kk if k["R5"] and not k["R10"]),
                "no_R10": sum(1 for k in kk if not k["R10"])}

    c470, c250 = counts(key), counts(key250)
    log("== case groups ==")
    log("470 sample:", json.dumps(c470))
    log("250 set:   ", json.dumps(c250))
    # wider over-concern denominators, as case fields
    for kk in (key, key250):
        for k in kk:
            k["no_R5"] = not k["R5"]
            k["no_R10"] = not k["R10"]
            k["low_or_tier2"] = k["clearly_low"] or (not k["R5"] and k["tier"] == 2 and not k["red_flag"])

    # ------------------------------------------------------------ policies on the 470
    top5_r5 = top_by_target_frequency(key, "R5", tiers)
    greedy5 = greedy_cover(key, "R5", tiers)
    log("fixed lists: R5 frequency", top5_r5, "| greedy cover", greedy5)

    def fixed(rule, flags, kk=key):
        return [{"yes": rule(c), "flags": code(flags)} for c in kk]

    def policies_for(kk):
        return {
            "always YES, no flags": fixed(lambda c: True, [], kk),
            "always YES, fixed top-5 (R5 frequency)": fixed(lambda c: True, top5_r5, kk),
            "always YES, fixed top-5 (greedy cover, tuned on the 470)": fixed(lambda c: True, greedy5, kk),
            "always NO": fixed(lambda c: False, [], kk),
            "YES if chest pain or dyspnoea; fixed top-5": fixed(lambda c: has(c["evf"], CHEST_PAIN | DYSPNOEA), top5_r5, kk),
            "YES if chest pain, dyspnoea, cough or wheeze; fixed top-5": fixed(lambda c: has(c["evf"], CHEST_PAIN | DYSPNOEA | COUGH | WHEEZE), top5_r5, kk),
            "YES if age >= 40; fixed top-5": fixed(lambda c: c["age"] >= 40, top5_r5, kk),
            "YES if age >= 65; fixed top-5": fixed(lambda c: c["age"] >= 65, top5_r5, kk),
            "YES unless age < 40 and no chest pain or dyspnoea; fixed top-5": fixed(lambda c: not (c["age"] < 40 and not has(c["evf"], CHEST_PAIN | DYSPNOEA)), top5_r5, kk),
            "DXA reader, spec section 8: tier-1 p >= 10%, not a red herring": dxa_answers(kk, tiers, rh, canon, 10.0),
            "DXA reader at 20%": dxa_answers(kk, tiers, rh, canon, 20.0),
            "DXA reader at 5% (the key itself)": dxa_answers(kk, tiers, rh, canon, 5.0),
            "DXA tier-1 mass >= 12.5% (v0.2 rule, red herrings removed)": dxa_answers(kk, tiers, rh, canon, 12.5, rule="mass"),
            "naive Bayes: YES if tier-1 posterior mass >= 12.5%": nb_answers(nb, kk, tiers, canon, 12.5),
            "perfect: YES iff R10; flags = R5 targets": [{"yes": bool(c["R10"]), "flags": code(sorted(c["R5"], key=lambda x: -c["dxa"].get(x, 0))[:5])} for c in kk],
        }

    def row_for(name, kk, ans):
        costs = case_costs(kk, ans, m, tiers)
        lo, hi = cluster_ci(kk, costs)
        s = score(kk, ans, m, BROADER, tiers=tiers)
        fr = sum(1 for k, a in zip(kk, ans) if k["R10"] and a["yes"] is not True)
        ocn = sum(1 for k, a in zip(kk, ans) if k["clearly_low"] and a["yes"] is True)
        n = len(kk)
        return {
            "policy": name, "SC": round(per100(costs), 1), "SC_lo": round(lo, 1), "SC_hi": round(hi, 1),
            "SC_miss_part": round(100.0 * MISS * fr / n, 1), "SC_oc_part": round(100.0 * OC * ocn / n, 1),
            "FR_events": fr, "OC_events": ocn,
            "H": round(100 * s["H"], 1), "OC": round(100 * s["OC"], 1), "COV": round(100 * s["COV"], 1),
            "YES_rate": round(100 * s["YES_rate"], 1),
            "rate_form_7H+OC": round(rate_form(kk, ans), 1),
            "SC_ratio5": round(per100(case_costs(kk, ans, m, tiers, miss=5.0)), 1),
            "SC_ratio10": round(per100(case_costs(kk, ans, m, tiers, miss=10.0)), 1),
            "SC_truth_only": round(per100(case_costs(kk, ans, m, tiers, target_field="truth_only")), 1),
            "SC_Hprime": round(per100(case_costs(kk, ans, m, tiers, hprime=True)), 1),
            "SC_weighted": round(per100(case_costs(kk, ans, m, tiers, weighted=True)), 1),
            "SC_oc_low_or_tier2": round(per100(case_costs(kk, ans, m, tiers, oc_field="low_or_tier2")), 1),
            "SC_oc_no_R5": round(per100(case_costs(kk, ans, m, tiers, oc_field="no_R5")), 1),
            "SC_oc_no_R10": round(per100(case_costs(kk, ans, m, tiers, oc_field="no_R10")), 1),
        }

    log("\n== policies on the 470 sample: cost points per 100 patients (miss 7, unneeded concern 1) ==")
    pol = policies_for(key)
    rows = [row_for(name, key, ans) for name, ans in pol.items()]
    rows.sort(key=lambda r: r["SC"])
    write_csv(OUT / "policies470.csv", rows)
    cols = ["policy", "SC", "SC_lo", "SC_hi", "SC_miss_part", "SC_oc_part", "H", "OC", "COV", "YES_rate", "rate_form_7H+OC"]
    log(" | ".join(cols))
    for r in rows:
        log(" | ".join(str(r[c]) for c in cols))

    # constant-policy arithmetic
    n, nT, nL = c470["n"], c470["R10"], c470["clearly_low"]
    log(f"\nconstant policies on the 470: always NO = 7 x {nT} / {n} x 100 = {700 * nT / n:.1f}; always YES = {nL} / {n} x 100 = {100 * nL / n:.1f}")
    log(f"a model with over-concern rate o beats always YES when H < ({nL} x (1 - o)) / (7 x {nT}): o=0 -> H < {100 * nL / (7 * nT):.1f}%, "
        f"o=0.3 -> H < {100 * 0.7 * nL / (7 * nT):.1f}%, o=0.5 -> H < {100 * 0.5 * nL / (7 * nT):.1f}%")

    # ------------------------------------------------------------ Bayes threshold check
    log("\n== Bayes-optimal threshold: readers by YES threshold (470 sample) ==")
    thr_rows = []
    for t in (2.0, 5.0, 8.0, 10.0, 12.5, 15.0, 20.0, 30.0, 50.0):
        for name, ans in (("naive Bayes tier-1 mass", nb_answers(nb, key, tiers, canon, t)),
                          ("DXA tier-1 mass, red herrings removed", dxa_answers(key, tiers, rh, canon, t, rule="mass")),
                          ("DXA max tier-1 p, red herrings removed", dxa_answers(key, tiers, rh, canon, t, rule="max"))):
            costs = case_costs(key, ans, m, tiers)
            s = score(key, ans, m, BROADER, tiers=tiers)
            thr_rows.append({"reader": name, "threshold": t, "SC": round(per100(costs), 1), "H": round(100 * s["H"], 1),
                             "OC": round(100 * s["OC"], 1), "YES_rate": round(100 * s["YES_rate"], 1)})
    write_csv(OUT / "threshold_sweep.csv", thr_rows)
    for name in ("naive Bayes tier-1 mass", "DXA tier-1 mass, red herrings removed", "DXA max tier-1 p, red herrings removed"):
        log(f"  {name}: " + "; ".join(f"t={r['threshold']:g}: SC {r['SC']} (H {r['H']}, OC {r['OC']})" for r in thr_rows if r["reader"] == name))

    # calibration of the naive-Bayes mass against "has an R10 target"
    masses = np.array([a["mass"] for a in nb_answers(nb, key, tiers, canon, 12.5)])
    has_t = np.array([bool(k["R10"]) for k in key])
    log("  naive-Bayes tier-1 mass bands vs share of cases with an R10 target:")
    for lo_, hi_ in ((0, 5), (5, 10), (10, 12.5), (12.5, 20), (20, 50), (50, 100.01)):
        sel = (masses >= lo_) & (masses < hi_)
        if sel.sum():
            log(f"    mass {lo_:g}-{min(hi_, 100):g}%: n {sel.sum()}, R10 share {100 * has_t[sel].mean():.0f}%")

    # DXA-derived-only R10 cases: cell truth rates (the evidence behind charging 7 for a NO)
    dxa_only = [k for k in key if k["R10"] and k["tier"] != 1]
    rates = [max(k["targets"][c]["rate"] for c in k["R10"]) for k in dxa_only]
    ps = [max(k["targets"][c]["p"] for c in k["R10"]) for k in dxa_only]
    log(f"  DXA-derived-only R10 cases: {len(dxa_only)}; max target DXA p median {np.median(ps):.1f}%; "
        f"reference-class truth rate median {100 * np.median(rates):.1f}% (IQR {100 * np.percentile(rates, 25):.1f}-{100 * np.percentile(rates, 75):.1f}%); "
        f"{sum(1 for r in rates if r >= 0.125)} of {len(rates)} at or above 12.5%")

    # per band of DXA live tier-1 mass: is YES the cheaper call? (7 x P(R10 | band) against P(clearly low | band))
    log("  DXA live tier-1 mass (red herrings removed) by band: cases, share with an R10 target, share clearly low-risk, cheaper call")
    dm = np.array([a["mass"] for a in dxa_answers(key, tiers, rh, canon, 5.0, rule="mass")])
    low = np.array([bool(k["clearly_low"]) for k in key])
    band_rows = []
    for lo_, hi_ in ((0, 5), (5, 10), (10, 12.5), (12.5, 20), (20, 35), (35, 50), (50, 100.01)):
        sel = (dm >= lo_) & (dm < hi_)
        if sel.sum():
            pt, pl = has_t[sel].mean(), low[sel].mean()
            call = "YES" if MISS * pt >= OC * pl else "NO"
            band_rows.append({"band": f"{lo_:g}-{min(hi_, 100):g}%", "n": int(sel.sum()), "R10_share": round(100 * pt, 1),
                              "clearly_low_share": round(100 * pl, 1), "cost_NO": round(MISS * pt, 2), "cost_YES": round(OC * pl, 2), "cheaper": call})
            log(f"    {lo_:g}-{min(hi_, 100):g}%: n {sel.sum()}, R10 {100 * pt:.0f}%, clearly low {100 * pl:.0f}%, expected cost NO {MISS * pt:.2f} vs YES {OC * pl:.2f} -> {call}")
    write_csv(OUT / "dxa_mass_bands.csv", band_rows)

    # interval width on the 470 for a synthetic model at given (H, OC), uniform over cases or clustered
    log("\n== simulated 95% interval width of SC on the 470 (2000 draws each) ==")
    rng = np.random.default_rng(SEED)
    r10_idx = np.array([i for i, k in enumerate(key) if k["R10"]])
    low_idx = np.array([i for i, k in enumerate(key) if k["clearly_low"]])
    conds470 = sorted({k["truth"] for k in key})
    for h, o in ((0.0, 0.35), (0.02, 0.35), (0.05, 0.35), (0.02, 0.60), (0.05, 0.60)):
        widths = []
        for _ in range(20):
            ans = [{"yes": True, "flags": []} for _ in key]
            for i in rng.choice(r10_idx, int(round(h * len(r10_idx))), replace=False):
                ans[i]["yes"] = False
            for i in rng.choice(low_idx, int(round((1 - o) * len(low_idx))), replace=False):
                ans[i]["yes"] = False
            c = case_costs(key, ans, m, tiers)
            lo, hi = cluster_ci(key, c, n_boot=400, seed=int(rng.integers(1e9)))
            widths.append(hi - lo)
        # clustered misses: all H events inside three conditions
        wc = []
        for _ in range(20):
            ans = [{"yes": True, "flags": []} for _ in key]
            picks = rng.choice(len(conds470), 3, replace=False)
            pool = np.array([i for i in r10_idx if key[i]["truth"] in {conds470[j] for j in picks}])
            for i in rng.choice(pool, min(len(pool), int(round(h * len(r10_idx)))), replace=False):
                ans[i]["yes"] = False
            for i in rng.choice(low_idx, int(round((1 - o) * len(low_idx))), replace=False):
                ans[i]["yes"] = False
            c = case_costs(key, ans, m, tiers)
            lo, hi = cluster_ci(key, c, n_boot=400, seed=int(rng.integers(1e9)))
            wc.append(hi - lo)
        sc = 100 * (MISS * h * len(r10_idx) + OC * o * len(low_idx)) / len(key)
        log(f"  H {100 * h:.0f}%, OC {100 * o:.0f}% (SC {sc:.1f}): mean width {np.mean(widths):.1f} points uniform, {np.mean(wc):.1f} with misses in 3 conditions")

    # prevalence re-weighting (v0.2 amendment 4.2): weight each case by share_c / n_c
    import csv as _csv
    mix = {r["condition"]: r for r in _csv.DictReader(open(ROOT / "evaluator/data/v02_condition_mix.csv", newline=""))}
    n_c = Counter(k["truth"] for k in key)
    def weights(col):
        tot = sum(float(mix[c][col]) for c in n_c)
        return np.array([float(mix[k["truth"]][col]) / tot / n_c[k["truth"]] * len(key) for k in key])
    w_ddx, w_nh = weights("ddxplus_adult_rows"), weights("nhamcs_weighted_adult_visits_2016_2022")
    log("\n== SC under the prevalence re-weightings (470 policies): equal (primary) | DDXPlus mix | NHAMCS ED mix ==")
    mix_rows = []
    for name, ans in pol.items():
        c = case_costs(key, ans, m, tiers)
        r = {"policy": name, "SC_equal": round(per100(c), 1), "SC_ddxplus_mix": round(100 * (c * w_ddx).sum() / len(key), 1),
             "SC_nhamcs_mix": round(100 * (c * w_nh).sum() / len(key), 1)}
        mix_rows.append(r)
        log(f"  {name}: {r['SC_equal']} | {r['SC_ddxplus_mix']} | {r['SC_nhamcs_mix']}")
    write_csv(OUT / "policies470_mix.csv", mix_rows)
    tshare = lambda w: 100 * sum(w[i] for i, k in enumerate(key) if k["R10"]) / len(key)
    lshare = lambda w: 100 * sum(w[i] for i, k in enumerate(key) if k["clearly_low"]) / len(key)
    log(f"  share of patients with an R10 target: equal {tshare(np.ones(len(key))):.0f}%, DDXPlus mix {tshare(w_ddx):.0f}%, NHAMCS mix {tshare(w_nh):.0f}%; "
        f"clearly low-risk: {lshare(np.ones(len(key))):.0f}%, {lshare(w_ddx):.0f}%, {lshare(w_nh):.0f}%")

    # ------------------------------------------------------------ stand-ins on the 250 set
    log("\n== v0.1 stand-ins on the 250 set (ESCALATE_NOW = YES, top-5 differential = flags) ==")
    models = load_rows(ids250)
    pol250 = policies_for(key250)
    ref_names = ["always YES, fixed top-5 (R5 frequency)", "always YES, fixed top-5 (greedy cover, tuned on the 470)",
                 "always NO", "YES if chest pain or dyspnoea; fixed top-5", "DXA reader, spec section 8: tier-1 p >= 10%, not a red herring",
                 "DXA tier-1 mass >= 12.5% (v0.2 rule, red herrings removed)", "naive Bayes: YES if tier-1 posterior mass >= 12.5%"]
    entries = [(f"[ref] {nm}", pol250[nm]) for nm in ref_names] + list(models.items())
    rows250 = []
    per_case = {}
    for name, ans in entries:
        r = row_for(name, key250, ans)
        r["unreadable"] = sum(1 for a in ans if a["yes"] is None)
        r["unreadable_in_R10"] = sum(1 for k, a in zip(key250, ans) if k["R10"] and a["yes"] is None)
        readable = [i for i, a in enumerate(ans) if a["yes"] is not None]
        cr = case_costs(key250, ans, m, tiers)
        r["SC_readable_only"] = round(100.0 * cr[readable].sum() / len(readable), 1) if readable else None
        rows250.append(r)
        per_case[name] = case_costs(key250, ans, m, tiers)
    rows250.sort(key=lambda r: r["SC"])
    write_csv(OUT / "standins250.csv", rows250)
    cols = ["policy", "SC", "SC_lo", "SC_hi", "SC_miss_part", "SC_oc_part", "H", "OC", "COV", "YES_rate", "unreadable_in_R10", "SC_readable_only"]
    log(" | ".join(cols))
    for r in rows250:
        log(" | ".join(str(r[c]) for c in cols))

    # paired differences, condition bootstrap, models only
    names = [r["policy"] for r in rows250 if not r["policy"].startswith("[ref]")]
    conds = sorted({k["truth"] for k in key250})
    idx = {c: np.array([i for i, k in enumerate(key250) if k["truth"] == c]) for c in conds}
    rng = np.random.default_rng(SEED)
    draws = [np.concatenate([idx[conds[j]] for j in rng.integers(0, len(conds), len(conds))]) for _ in range(1000)]

    def paired(a, b):
        d = np.array([100.0 * (a[sel].sum() - b[sel].sum()) / len(sel) for sel in draws])
        return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

    sep = pairs = 0
    adj = []
    pair_rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            lo, hi = paired(per_case[names[i]], per_case[names[j]])
            pairs += 1
            s = lo > 0 or hi < 0
            sep += s
            pair_rows.append({"a": names[i], "b": names[j], "diff": round(per100(per_case[names[i]]) - per100(per_case[names[j]]), 1),
                              "lo": round(lo, 1), "hi": round(hi, 1), "separated": s})
            if j == i + 1:
                adj.append(s)
    write_csv(OUT / "standins250_pairs.csv", pair_rows)
    log(f"\nmodel pairs separated at 95% (paired condition bootstrap on the cost): {sep} of {pairs}; adjacent ranks: {sum(adj)} of {len(adj)}")
    # against the always-YES reference
    ay = per_case["[ref] always YES, fixed top-5 (R5 frequency)"]
    beat = []
    for nm in names:
        lo, hi = paired(per_case[nm], ay)
        beat.append((nm, round(per100(per_case[nm]) - per100(ay), 1), round(lo, 1), round(hi, 1)))
    log("model minus always-YES (negative = cheaper than blanket concern), with 95% interval:")
    for b in beat:
        log("   ", b)
    log(f"models cheaper than always YES: {sum(1 for b in beat if b[1] < 0)} of {len(beat)}; separated: {sum(1 for b in beat if b[3] < 0)}")
    write_csv(OUT / "standins250_vs_always_yes.csv", [{"model": b[0], "diff": b[1], "lo": b[2], "hi": b[3]} for b in beat])

    # H-alone ranking against the cost ranking
    h_rank = np.argsort(np.argsort([r["H"] for r in rows250 if r["policy"] in names]))
    c_rank = np.argsort(np.argsort([r["SC"] for r in rows250 if r["policy"] in names]))
    log(f"Spearman(H rank, SC rank) over the 19 stand-ins: {np.corrcoef(h_rank, c_rank)[0, 1]:.2f}")

    # ------------------------------------------------------------ sensitivity: rank stability
    log("\n== sensitivity rows: rank correlation with the primary SC (19 stand-ins; then 470 policies) ==")
    variants = ["SC_ratio5", "SC_ratio10", "SC_truth_only", "SC_Hprime", "SC_weighted", "rate_form_7H+OC",
                "SC_oc_low_or_tier2", "SC_oc_no_R5", "SC_oc_no_R10"]
    sens_rows = []
    for label, rr in (("stand-ins (250)", [r for r in rows250 if r["policy"] in names]), ("policies (470)", rows)):
        base = np.array([r["SC"] for r in rr])
        for v in variants:
            x = np.array([r[v] for r in rr])
            rb, rx = np.argsort(np.argsort(base)), np.argsort(np.argsort(x))
            rho = np.corrcoef(rb, rx)[0, 1]
            top5_base = [rr[i]["policy"] for i in np.argsort(base)[:5]]
            top5_x = [rr[i]["policy"] for i in np.argsort(x)[:5]]
            sens_rows.append({"set": label, "variant": v, "spearman": round(rho, 3), "top5_same": len(set(top5_base) & set(top5_x)),
                              "largest_rank_move": int(np.abs(rb - rx).max())})
            log(f"  {label:16s} {v:20s} rho {rho:.3f}; top-5 overlap {len(set(top5_base) & set(top5_x))}/5; largest rank move {int(np.abs(rb - rx).max())}")
    write_csv(OUT / "sensitivity.csv", sens_rows)

    # where does always YES rank under each variant (470 policies and 250 stand-ins + refs)?
    log("\nrank of 'always YES, fixed top-5 (greedy cover)' among all rows, by variant (1 = cheapest):")
    for label, rr in (("policies (470)", rows), ("stand-ins + refs (250)", rows250)):
        out = []
        for v in ["SC"] + variants:
            order = sorted(rr, key=lambda r: r[v])
            pos = [i for i, r in enumerate(order) if "greedy cover" in r["policy"]][0] + 1
            out.append(f"{v} {pos}/{len(rr)}")
        log(f"  {label}: " + "; ".join(out))

    # targets at DXA >= 12.5% (the Bayes threshold) instead of 10%
    log("\n== sensitivity: DXA-derived targets at >= 12.5% instead of 10% ==")
    key125 = build_key(df, ids470, tiers, rh, t_main=12.5)
    key125_250 = build_key(df, ids250, tiers, rh, t_main=12.5)
    for kk in (key125, key125_250):
        for k in kk:
            k["no_R5"] = not k["R5"]
    dropped = [k for k, k1 in zip(key, key125) if k["R10"] and not k1["R10"]]
    log(f"  470: R10 cases {sum(1 for k in key125 if k['R10'])} (was {c470['R10']}); cases that lose their only target: {len(dropped)} "
        f"({Counter(k['truth'] for k in dropped).most_common(6)})")
    dxa_only_lo = [k for k in dxa_only if max(k["targets"][c]["p"] for c in k["R10"]) < 12.5]
    log(f"  DXA-derived-only R10 cases with every target under 12.5%: {len(dxa_only_lo)} of {len(dxa_only)}")
    r125 = []
    for label, kk, kk125, entries_ in (("policies (470)", key, key125, list(pol.items())), ("stand-ins (250)", key250, key125_250, [(nm, models[nm]) for nm in names])):
        base = np.array([per100(case_costs(kk, ans, m, tiers)) for _, ans in entries_])
        alt = np.array([per100(case_costs(kk125, ans, m, tiers)) for _, ans in entries_])
        rb, ra = np.argsort(np.argsort(base)), np.argsort(np.argsort(alt))
        rho = np.corrcoef(rb, ra)[0, 1]
        log(f"  {label}: rho {rho:.3f}; largest rank move {int(np.abs(rb - ra).max())}; mean SC change {np.mean(alt - base):+.1f}")
        for (nm, _), b, a in zip(entries_, base, alt):
            r125.append({"set": label, "policy": nm, "SC_R10": round(b, 1), "SC_R12.5": round(a, 1)})
    write_csv(OUT / "sensitivity_r125.csv", r125)
    nbk = per100(case_costs(key125, pol["naive Bayes: YES if tier-1 posterior mass >= 12.5%"], m, tiers))
    log(f"  naive Bayes under R12.5: SC {nbk:.1f}; always YES: {per100(case_costs(key125, pol['always YES, no flags'], m, tiers)):.1f}")

    # ------------------------------------------------------------ coverage fold-in
    log("\n== folding coverage into the headline: SC - w x 100 x mean COV credit per patient ==")
    cov_rows = []
    for label, kk, entries_ in (("policies (470)", key, list(pol.items())), ("stand-ins + refs (250)", key250, entries)):
        base = {}
        for name, ans in entries_:
            s = score(kk, ans, m, BROADER, tiers=tiers)
            covsum = 0.0
            for k, a in zip(kk, ans):
                if k["R5"]:
                    covsum += np.mean([m.matched(a["flags"][:5], c, BROADER) for c in k["R5"]])
            base[name] = (per100(case_costs(kk, ans, m, tiers)), 100.0 * covsum / len(kk))
        for w in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0):
            scored = sorted(base.items(), key=lambda kv: kv[1][0] - w * kv[1][1])
            names_ = [nm for nm, _ in scored]
            gamer = [i for i, nm in enumerate(names_) if "greedy cover" in nm][0] + 1
            gamer_val = [v[0] - w * v[1] for nm, v in scored if "greedy cover" in nm][0]
            best_model = next((nm for nm in names_ if not nm.startswith("[ref]") and "always" not in nm and "YES if" not in nm and "DXA" not in nm and "naive" not in nm and "perfect" not in nm), None)
            model_names = [nm for nm in base if nm in names] if label.startswith("stand-ins") else []
            rho = None
            if model_names:
                b0 = np.array([base[nm][0] for nm in model_names]); bw = np.array([base[nm][0] - w * base[nm][1] for nm in model_names])
                rho = round(float(np.corrcoef(np.argsort(np.argsort(b0)), np.argsort(np.argsort(bw)))[0, 1]), 3)
            cov_rows.append({"set": label, "w": w, "gamer_rank": gamer, "of": len(names_), "gamer_value": round(gamer_val, 1),
                             "spearman_models_vs_w0": rho, "top3": "; ".join(names_[:3])})
            log(f"  {label} w={w}: tuned always-YES ranks {gamer}/{len(names_)} at {gamer_val:.1f}; model-rank rho vs w=0 {rho}; top 3: {names_[:3]}")
    write_csv(OUT / "coverage_foldin.csv", cov_rows)
    # what a fixed list earns in COV credit per patient
    for name in ("always YES, fixed top-5 (greedy cover, tuned on the 470)", "always YES, fixed top-5 (R5 frequency)"):
        s = score(key, pol[name], m, BROADER, tiers=tiers)
        log(f"  {name}: COV {100 * s['COV']:.1f} over {s['COV_n']} R5 cases = {100 * s['COV'] * s['COV_n'] / len(key):.1f} COV-case points per 100 patients")

    # ------------------------------------------------------------ unreadable: cost of formats
    log("\n== unreadable output (scored as NO): Gemini 3 Pro stand-in ==")
    for r in rows250:
        if r["policy"] == "Gemini 3 Pro":
            log(f"  SC {r['SC']} (miss part {r['SC_miss_part']}, OC part {r['SC_oc_part']}); unreadable {r['unreadable']} of 250, {r['unreadable_in_R10']} in R10; SC on readable cases {r['SC_readable_only']}")

    # per-condition cost table for the 470 policies (which conditions drive the cost)
    log("\n== where the always-YES cost falls (clearly low-risk cases by truth) ==")
    log("  ", Counter(k["truth"] for k in key if k["clearly_low"]).most_common())
    log("== intermediate cases (no cost either way) by truth and tier ==")
    inter = [k for k in key if not k["R10"] and not k["clearly_low"]]
    log(f"   {len(inter)} cases; tier 2 without R5 target {sum(1 for k in inter if k['tier'] == 2 and not k['R5'])}, "
        f"R5-only {sum(1 for k in inter if k['R5'])}, tier-3 red flag without R5 {sum(1 for k in inter if k['tier'] == 3 and not k['R5'] and k['red_flag'])}")
    log("  ", Counter((k["truth"], k["tier"]) for k in inter).most_common(12))

    (OUT / "headline.log").write_text("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    main()
