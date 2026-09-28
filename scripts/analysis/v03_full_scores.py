#!/usr/bin/env python3
"""
Score the v0.3 full run (docs/v0.3-case-selection-rules.md section 7.4; spec/v0.3-scoring.md record R2).

The seven models x arms 4aj and 4bj on the 900 cases of data/test_sets/eval-v03-full.case_ids.txt, under the
selection rules' classes and credits with amendments A3-A5 (scripts/analysis/v03_phase2_common.py). Arm 4aj is the
headline (decision 18); 4bj is secondary. We report:

1. `score_z_bal` (A2: blanket escalation with one fixed flag scores 0, perfect 100), U, O, the partials and the
   escalation share, with 95% intervals that resample cases within each true condition (the primary interval),
   and the condition bootstrap as the superpopulation sensitivity; 2,000 draws, seed 20260923.
2. Model-pair separation: the paired score difference of every pair within an arm, on the same cases and draws.
3. Reference rows scored by the same code: the zero point (A2), the zero point pinned to I21, naive Bayes (its
   strongest tier-1 flag at a posterior of 10% or more; data/test_sets/eval-v03-full.refs.json) and always-routine.
4. Sensitivity rows: the CCSR off-list tiers included, and every model's score against the I21 zero point.
5. Per-condition tables: full misses on SERIOUS cases and escalations on BENIGN cases, per model.
6. The confirmatory anchor test (record R2): per model, the paired 4aj minus 4bj headline cost per 100 headline
   cases (all penalties), its 95% within-condition interval, a two-sided bootstrap p, and Holm's step-down
   correction across the seven models at a family-wise 5%.

There is no reference review for this set, so no precision is computed.

Outputs: results/v03_full/scores.md and scores.json (aggregates only; the model outputs stay in
results/v03_full/runs/, which git ignores).

Usage: python3 scripts/analysis/v03_full_scores.py
"""

from __future__ import annotations

import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

import v03_phase2_common as pc  # noqa: E402
from v03_phase2_common import cs, vr, vs  # noqa: E402

PHASE = "full"
ALPHA = 0.05
MEASURES = pc.MEASURES


def fmt(x, nd=1) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def ci(m: dict, nd=1) -> str:
    lo, hi = m["ci"]
    return f"{fmt(m['value'], nd)} [{fmt(lo, nd)}, {fmt(hi, nd)}]"


def boot_p(d: np.ndarray) -> float:
    """Two-sided bootstrap p: 2 x the smaller tail share, each counted as (k + 1) / (B + 1)."""
    d = d[np.isfinite(d)]
    B = len(d)
    lo, hi = (np.sum(d <= 0) + 1) / (B + 1), (np.sum(d >= 0) + 1) / (B + 1)
    return float(min(1.0, 2 * min(lo, hi)))


def holm(ps: dict[str, float], alpha: float = ALPHA) -> dict[str, dict]:
    """Holm's step-down procedure: adjusted p (monotone) and the reject decision at family-wise `alpha`."""
    order = sorted(ps, key=lambda k: ps[k])
    m, running, out, stop = len(order), 0.0, {}, False
    for i, k in enumerate(order):
        adj = min(1.0, (m - i) * ps[k])
        running = max(running, adj)
        reject = not stop and ps[k] <= alpha / (m - i)
        stop = stop or not reject
        out[k] = {"p": round(ps[k], 5), "rank": i + 1, "threshold": round(alpha / (m - i), 5),
                  "p_holm": round(running, 5), "reject": reject}
    return out


def policy_outcome(s: pc.ScoredSet, flags: list, esc: list):
    """A fixed policy scored like a model row: per case one flag (or None) and an escalation decision."""
    outs, reasons = [], []
    for k, flag, e in zip(s.ab.key.keys, flags, esc):
        kind = vr.OTHER_TIER1 if (e and flag) else vr.NONE
        o, _, kind_after = cs.verdict_under_rules(s.sel[k.case_id], flag, bool(e), kind, k.truth, s.rm, vr)
        outs.append(o)
        reasons.append(vr.Reason(kind_after if kind_after in vr.REASONS else vr.NONE))
    return vr.Outcome(np.array(esc, bool), reasons, outs)


def reference_rows(s: pc.ScoredSet, M, key, zero) -> dict[str, dict]:
    n = s.ab.key.n
    refs = json.loads((pc.TS / "eval-v03-full.refs.json").read_text())["references"]["naive-bayes"]
    nb = {r["case_id"]: r for r in refs}
    rows_nb = [nb[c] for c in s.ab.key.case_ids]
    _, zi = pc.zero_outcome(s)
    policies = {
        "zero point (A2)": pc.zero_outcome(s)[0],
        "zero point at I21": pc.zero_outcome(s, "I21")[0],
        "naive Bayes": policy_outcome(s, [(r["flags"] or [None])[0] for r in rows_nb], [r["serious_concern"] == "YES" for r in rows_nb]),
        "always-routine": policy_outcome(s, [None] * n, [False] * n),
    }
    out = {}
    for name, o in policies.items():
        point, draws = vr.stats(o, s.ab, M, zero=zero, key=key)
        out[name] = {m: vr.summarise(point, draws)[m] for m in ("score_z_bal", "U", "O", "esc", "cost")}
    return out


def per_condition(s: pc.ScoredSet, recs: list[dict], arm: str) -> dict:
    """Per true condition: SERIOUS cases and full misses (cost 7) per model; BENIGN cases and escalations per model."""
    truth = {k.case_id: k.truth for k in s.ab.key.keys}
    miss, over = defaultdict(Counter), defaultdict(Counter)
    n_s, n_b = Counter(), Counter()
    for c, sl in s.sel.items():
        (n_s if sl["class"] == cs.SERIOUS else n_b)[truth[c]] += 1
    for r in recs:
        if r["arm"] != arm:
            continue
        t = truth[r["case_id"]]
        if r["class"] == cs.SERIOUS and r["cost"] >= 7:
            miss[t][r["model"]] += 1
        if r["class"] == cs.BENIGN and r["esc_after"]:
            over[t][r["model"]] += 1
    return {"serious": {t: {"n": n_s[t], **dict(miss[t])} for t in sorted(n_s)},
            "benign": {t: {"n": n_b[t], **dict(over[t])} for t in sorted(n_b)}}


def main() -> None:
    ph = pc.PHASES[PHASE]
    OUT, RUNS, STEM = ph["out"], ph["runs"], ph["stem"]
    s = pc.load_phase2(phase=PHASE)
    head_rule, ccsr_rule = vr.TierFileRule(), vr.TierFileRule(include_ccsr=True)
    runs = pc.load_runs(s, head_rule, RUNS)
    assert len(runs) == len(pc.MODELS) * len(pc.ARMS), f"{len(runs)} prediction files; expected {len(pc.MODELS) * len(pc.ARMS)}"
    recs = pc.records(s, runs, head_rule)
    head = pc.score(s, recs, within=True)
    cond = pc.score(s, recs)  # the condition bootstrap (sensitivity)
    ccsr = pc.score(s, pc.records(s, pc.load_runs(s, ccsr_rule, RUNS), ccsr_rule), within=True)
    i21 = pc.score(s, recs, zero_code="I21", within=True)
    refs = reference_rows(s, head["M"], head["key"], head["zero_outcome"])
    refs_cond = reference_rows(s, cond["M"], cond["key"], cond["zero_outcome"])

    # The confirmatory anchor test (record R2).
    models = sorted({m for m, _ in head["rows"]})
    anchor, ps = {}, {}
    for m in models:
        (pa, da), (pb, db) = head["raw"][(m, "4aj")], head["raw"][(m, "4bj")]
        d = head["paired_4aj_minus_4bj"][m]["cost"]
        ps[m] = boot_p(np.asarray(da["cost"]) - np.asarray(db["cost"]))
        anchor[m] = {"cost_4aj": round(100 * float(pa["cost"]), 2), "cost_4bj": round(100 * float(pb["cost"]), 2),
                     "diff_per_100": d["value"], "ci": d["ci"]}
    for m, h in holm(ps).items():
        anchor[m].update(h)

    prov = json.loads((RUNS / "provenance.json").read_text())
    parse = {f"{pc.sb.short(m)}|{pc.sb.ARM_LABELS[a]}": p for (m, a), (_, _, p) in runs.items()}
    meta = json.loads((pc.TS / f"{STEM}.json").read_text())["metadata"]
    strata = Counter(c["stratum"] for c in s.ab.cases)
    spend = sum(x["usd"] for x in prov.get("account_spend_usd", []))
    files = sorted(RUNS.glob("*-v7a4?j.json"))
    tokens_cost = sum(((x.get("usage") or {}).get("cost") or 0) for f in files
                      for x in json.loads(f.read_text())["predictions"] if isinstance(x, dict))
    conds = {arm: per_condition(s, recs, arm) for arm in ("4aj", "4bj")}

    rows, rows_c = head["rows"], cond["rows"]
    order = sorted(models, key=lambda m: -(rows[(m, "4aj")]["score_z_bal"]["value"] or -1e9))
    z, zi = head["zero_reference"], i21["zero_reference"]
    L = ["# v0.3 full run: model scores (arms 4aj and 4bj)", "",
         f"- **Cases:** {s.ab.key.n} (seed {meta['seed']}; `data/test_sets/{STEM}.case_ids.txt`, built by "
         "`scripts/build_v03_full_set.py`; design in docs/v0.3-case-selection-rules.md section 7.4, frozen at 7e67e24). Strata: "
         + ", ".join(f"{k} {n}" for k, n in strata.items()) + f". {len(meta['replaced'])} drawn BENIGN cases fell to X9 under the "
         f"key and were replaced by the next cases in their buckets. Cases with an exact public twin: {meta['twins']}.",
         f"- **Classes:** {head['serious']} SERIOUS, {head['benign']} BENIGN; the headline covers {head['headline_cases']} cases.",
         f"- **Scoring:** amendments A3-A5 with the selection rules' classes and credits (A4 truth partials), rule P5 demoted "
         f"(decision 21). Off-list tiers: NHAMCS-rated rows (A5). Zero point (A2): {z['code']} ({z['condition']}, a target on "
         f"{z['serious_cases']} SERIOUS cases). Headline arm 4aj (decision 18); 4bj secondary.",
         "- **Intervals:** 95%, resampling cases within each true condition (the drawn mix is the estimand; spec record R2); "
         "the condition bootstrap, which also varies the mix, is the sensitivity column. 2,000 draws, seed 20260923.",
         f"- **Run:** `inference/run_config_v03_abj.json`, prompts v7a4aj and v7a4bj, via OpenRouter; provenance in "
         f"`{RUNS.relative_to(ROOT)}/provenance.json`. Token cost {tokens_cost:.2f} USD over {len(files)} files (account spend "
         f"delta {spend:.2f} USD). A parse failure left after the one retry is scored as unreadable (routine), as in Phase 2b.",
         "- **No precision:** this set has no reference review.", "",
         "## Scores, arm 4aj (headline)", "",
         "Score is `score_z_bal`. U: SERIOUS cases costing a full miss. O: BENIGN cases escalated. Partial: SERIOUS cases "
         "charged a partial (in-list / off-list / truth). All in %.", ""]

    def score_table(arm):
        T = ["| Model | Score [95% CI, within-condition] | Condition bootstrap | U | O | Partial (in / off / truth) | Escalated |",
             "|---|---|---|---|---|---|---|"]
        for m in order:
            r, rc = rows[(m, arm)], rows_c[(m, arm)]
            T.append(f"| {m} | {ci(r['score_z_bal'])} | [{fmt(rc['score_z_bal']['ci'][0])}, {fmt(rc['score_z_bal']['ci'][1])}] | "
                     f"{ci(r['U'])} | {ci(r['O'])} | {fmt(r['partial']['value'])} ({fmt(r['partial_inlist']['value'])} / "
                     f"{fmt(r['partial_offlist']['value'])} / {fmt(r['partial_truth']['value'])}) | {fmt(r['esc']['value'])} |")
        return T
    L += score_table("4aj") + ["", "## Scores, arm 4bj (secondary)", ""] + score_table("4bj")

    L += ["", "## Reference rows (scored by the same code, within-condition intervals)", "",
          "| Row | Score [95% CI] | Condition bootstrap | U | O | Escalated |", "|---|---|---|---|---|---|"]
    for name, r in refs.items():
        rc = refs_cond[name]
        L.append(f"| {name} | {ci(r['score_z_bal'])} | [{fmt(rc['score_z_bal']['ci'][0])}, {fmt(rc['score_z_bal']['ci'][1])}] | "
                 f"{ci(r['U'])} | {ci(r['O'])} | {fmt(r['esc']['value'])} |")
    same = z["code"] == zi["code"]
    L += ["", "The zero point scores 0 by construction" + (f"; A2's rule picks {z['code']} on this set, so the I21 row and "
          "the I21 sensitivity column repeat the headline" if same else "") + ". Naive Bayes flags its strongest tier-1 condition at a posterior of 10% "
          "or more; it is a dataset-knowledge ceiling, not a clinical target.", ""]

    L += ["## Model-pair separation (paired score difference, same cases and draws)", "",
          "A pair is separated when the 95% interval of the score difference excludes 0. The last column counts it under the "
          "condition bootstrap. Intervals are not corrected for the 21 pairs.", ""]
    sep_summary = {}
    for arm in ("4aj", "4bj"):
        pairs, pairs_c = ({k: d for k, d in sc["model_pairs"].items() if k[0] == arm} for sc in (head, cond))
        sep, sep_c = sum(d["separated"] for d in pairs.values()), sum(d["separated"] for d in pairs_c.values())
        sep_summary[arm] = {"within": sep, "condition_bootstrap": sep_c, "pairs": len(pairs)}
        L += [f"### Arm {arm}: {sep} of {len(pairs)} pairs separated ({sep_c} under the condition bootstrap)", "",
              "| Model A (higher by 4aj score) | Model B | A minus B [95% CI] | Separated | Condition bootstrap |", "|---|---|---|---|---|"]
        for i, a in enumerate(order):
            for b in order[i + 1:]:
                ds = []
                for pp in (pairs, pairs_c):
                    d = pp.get((arm, a, b))
                    if d is None:
                        d0 = pp[(arm, b, a)]
                        d = {"value": -d0["value"], "ci": [-d0["ci"][1], -d0["ci"][0]], "separated": d0["separated"]}
                    ds.append(d)
                L.append(f"| {a} | {b} | {ci(ds[0])} | {'yes' if ds[0]['separated'] else 'no'} | {'yes' if ds[1]['separated'] else 'no'} |")
        L.append("")

    L += ["## Confirmatory anchor test (spec record R2)", "",
          "Arm 4aj minus 4bj headline cost per 100 headline cases (miss 7, partial 1, over-escalation 1), paired on the same "
          "cases and within-condition draws. A positive difference means the benign anchor (arm 4bj) lowers the model's cost. "
          f"Holm's step-down procedure at a family-wise {ALPHA:.0%} across the seven models.", "",
          "| Model | Cost 4aj | Cost 4bj | 4aj minus 4bj [95% CI] | p | Holm threshold | Holm-adjusted p | Effect |",
          "|---|---|---|---|---|---|---|---|"]
    for m in order:
        a = anchor[m]
        eff = ("yes, anchor lowers cost" if a["diff_per_100"] > 0 else "yes, anchor raises cost") if a["reject"] else "no"
        L.append(f"| {m} | {a['cost_4aj']:.1f} | {a['cost_4bj']:.1f} | {fmt(a['diff_per_100'])} [{fmt(a['ci'][0])}, {fmt(a['ci'][1])}] | "
                 f"{a['p']:.4f} | {a['threshold']:.4f} | {a['p_holm']:.4f} | {eff} |")
    L += ["", "## Arm 4aj minus 4bj, per model (paired, within-condition)", "",
          "| Model | Score | U | O | Escalated |", "|---|---|---|---|---|"]
    for m in order:
        d = head["paired_4aj_minus_4bj"][m]
        L.append(f"| {m} | {ci(d['score_z_bal'])} | {ci(d['U'])} | {ci(d['O'])} | {ci(d['esc'])} |")

    L += ["", "## Sensitivity rows (within-condition intervals)", "",
          f"CCSR: the off-list tiers include the CCSR-rated rows. I21: the zero point pinned to {zi['code']} ({zi['condition']}, "
          f"a target on {zi['serious_cases']} SERIOUS cases).", "",
          "| Model | Arm | Headline | CCSR tiers included | Zero at I21 |", "|---|---|---|---|---|"]
    for m in order:
        for arm in ("4aj", "4bj"):
            L.append(f"| {m} | {arm} | {ci(rows[(m, arm)]['score_z_bal'])} | {ci(ccsr['rows'][(m, arm)]['score_z_bal'])} | "
                     f"{ci(i21['rows'][(m, arm)]['score_z_bal'])} |")

    short = [pc.sb.short(m) for m in pc.MODELS]
    order_s = [m for m in order if m in short] or short
    for arm in ("4aj",):
        c = conds[arm]
        L += ["", f"## Full misses per condition, SERIOUS cases (arm {arm})", "",
              "| Condition | n | " + " | ".join(order_s) + " |", "|---|---|" + "---|" * len(order_s)]
        for t, r in sorted(c["serious"].items(), key=lambda x: -sum(v for k, v in x[1].items() if k != "n")):
            L.append(f"| {t} | {r['n']} | " + " | ".join(str(r.get(m, 0)) for m in order_s) + " |")
        L += ["", f"## Escalations per condition, BENIGN cases (arm {arm})", "",
              "| Condition | n | " + " | ".join(order_s) + " |", "|---|---|" + "---|" * len(order_s)]
        for t, r in sorted(c["benign"].items(), key=lambda x: -sum(v for k, v in x[1].items() if k != "n")):
            L.append(f"| {t} | {r['n']} | " + " | ".join(str(r.get(m, 0)) for m in order_s) + " |")
    L += ["", "Arm 4bj's per-condition tables are in scores.json.", "", "## Parsing", "",
          "| Row | Answered | Unreadable | Errored | No justification | Token cost (USD) |", "|---|---|---|---|---|---|"]
    for name, p in sorted(parse.items()):
        L.append(f"| {name} | {p['answered']} of {p['cases']} | {p['unreadable']} | {p['errored']} | {p['no_justification']} | {p['cost_usd']} |")
    L.append("")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scores.md").write_text("\n".join(L))

    def js(sc):
        sc = {k: v for k, v in sc.items() if k not in ("raw", "zero_outcome", "M", "key")}
        return {**sc, "rows": {f"{m}|{a}": r for (m, a), r in sc["rows"].items()},
                "model_pairs": {"|".join(k): d for k, d in sc["model_pairs"].items()}}
    (OUT / "scores.json").write_text(json.dumps({
        "set": {"cases": s.ab.key.n, "seed": meta["seed"], "strata": dict(strata), "twins": meta["twins"],
                "replaced": meta["replaced"], "freeze": "7e67e24"},
        "headline_within_condition": js(head), "condition_bootstrap": js(cond), "sensitivity_ccsr": js(ccsr),
        "sensitivity_zero_I21": js(i21), "reference_rows": refs, "reference_rows_condition_bootstrap": refs_cond,
        "pair_separation": sep_summary, "anchor_test": anchor, "per_condition": conds, "parse": parse,
        "token_cost_usd": round(tokens_cost, 4), "account_spend_usd": spend}, indent=1, default=str) + "\n")
    for m in order:
        print(f"{m:24s} 4aj {ci(rows[(m, '4aj')]['score_z_bal']):24s} 4bj {ci(rows[(m, '4bj')]['score_z_bal']):24s} "
              f"anchor {anchor[m]['diff_per_100']} {anchor[m]['ci']} p_holm {anchor[m]['p_holm']} {anchor[m]['reject']}")
    print(f"pairs separated {sep_summary}; token cost {tokens_cost:.2f}; spend {spend:.2f}; wrote {OUT / 'scores.md'}")


if __name__ == "__main__":
    main()
