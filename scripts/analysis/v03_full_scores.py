#!/usr/bin/env python3
"""
Score the v0.3 full run (docs/v0.3-case-selection-rules.md section 7.4; spec/v0.3-scoring.md record R2).

The original seven models x arms 4aj and 4bj, plus the models of the roster expansion that completed (results/v03_full/
roster_expansion.md; decisions 30-33) in arm 4aj only, on the 900 cases of data/test_sets/eval-v03-full.case_ids.txt,
under the selection rules' classes and credits with amendments A3-A5 (scripts/analysis/v03_phase2_common.py). Arm 4aj
is the headline and the only scored arm (record R3); we keep the seven models' 4bj rows for the methodology's
discussion. We report:

1. `score_z_bal` (A2: blanket escalation with one fixed flag scores 0, perfect 100), U, O, the partials and the
   escalation share, with 95% intervals that resample cases within each true condition (the primary interval),
   and the condition bootstrap as the superpopulation sensitivity; 2,000 draws, seed 20260923.
2. Model-pair separation: the paired score difference of every pair within an arm, on the same cases and draws.
   With more than eight models in an arm, scores.md lists the pairs adjacent in rank and, per model, the models it
   does not separate from; scores.json keeps every pair.
3. Reference rows scored by the same code: the zero point (A2), the zero point pinned to I21, naive Bayes (its
   strongest tier-1 flag at a posterior of 10% or more; data/test_sets/eval-v03-full.refs.json) and always-routine.
4. Sensitivity rows: the CCSR off-list tiers included, and every model's score against the I21 zero point.
5. Per-condition tables: full misses on SERIOUS cases and escalations on BENIGN cases, per model.
6. The confirmatory anchor test (record R2): per model, the paired 4aj minus 4bj headline cost per 100 headline
   cases (all penalties), its 95% within-condition interval, a two-sided bootstrap p, and Holm's step-down
   correction across the seven models that ran both arms, at a family-wise 5%.

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
# The roster expansion of 2026-09-29 (results/v03_full/roster_expansion.md), run in arm 4aj only (record R3).
# We score a model only when its file covers all 900 cases; the others wait in UNFINISHED with the reason.
UNFINISHED = {"x-ai/grok-4.7": "stopped at 800 of 900 cases when the OpenRouter account ran out of credit; to be added later",
              "deepseek/deepseek-v4.1-flash": "stopped at 819 of 900 cases: its providers answered slowly and the account ran out of "
                                              "credit; to be added later"}
EXPANSION = ("anthropic/claude-fable-5.1", "openai/gpt-6-astra", "anthropic/claude-opus-5.5",
             "openai/gpt-5.4-mini", "openai/gpt-6.1-sol", "moonshotai/kimi-k3", "anthropic/claude-sonnet-5.5",
             "google/gemini-3.8-flash", "openai/gpt-6-luna")
MODELS = pc.MODELS + EXPANSION
# The provenance files of the two runs into results/v03_full/runs/.
PROVENANCE = ("provenance.json", "provenance-expansion.json")
# Above this many models in an arm, scores.md lists adjacent pairs and a per-model summary instead of every pair.
FULL_PAIR_LIST_MAX = 8


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
    runs = pc.load_runs(s, head_rule, RUNS, models=MODELS)
    expected = len(pc.MODELS) * len(pc.ARMS) + len(EXPANSION)
    assert len(runs) == expected, f"{len(runs)} prediction files; expected {expected}"
    recs = pc.records(s, runs, head_rule)
    head = pc.score(s, recs, within=True)
    cond = pc.score(s, recs)  # the condition bootstrap (sensitivity)
    ccsr = pc.score(s, pc.records(s, pc.load_runs(s, ccsr_rule, RUNS, models=MODELS), ccsr_rule), within=True)
    i21 = pc.score(s, recs, zero_code="I21", within=True)
    refs = reference_rows(s, head["M"], head["key"], head["zero_outcome"])
    refs_cond = reference_rows(s, cond["M"], cond["key"], cond["zero_outcome"])

    # The confirmatory anchor test (record R2).
    models = sorted({m for m, _ in head["rows"]})
    both = [m for m in models if (m, "4bj") in head["rows"]]  # the models that ran both arms
    anchor, ps = {}, {}
    for m in both:
        (pa, da), (pb, db) = head["raw"][(m, "4aj")], head["raw"][(m, "4bj")]
        d = head["paired_4aj_minus_4bj"][m]["cost"]
        ps[m] = boot_p(np.asarray(da["cost"]) - np.asarray(db["cost"]))
        anchor[m] = {"cost_4aj": round(100 * float(pa["cost"]), 2), "cost_4bj": round(100 * float(pb["cost"]), 2),
                     "diff_per_100": d["value"], "ci": d["ci"]}
    for m, h in holm(ps).items():
        anchor[m].update(h)

    provs = [json.loads((RUNS / f).read_text()) for f in PROVENANCE if (RUNS / f).exists()]
    parse = {f"{pc.sb.short(m)}|{pc.sb.ARM_LABELS[a]}": p for (m, a), (_, _, p) in runs.items()}
    meta = json.loads((pc.TS / f"{STEM}.json").read_text())["metadata"]
    strata = Counter(c["stratum"] for c in s.ab.cases)
    spend_by_run = {p.get("run", f): round(sum(x["usd"] for x in p.get("account_spend_usd", [])), 4)
                    for f, p in zip(PROVENANCE, provs)}
    spend = sum(spend_by_run.values())
    files = sorted(f for f in RUNS.glob("*-v7a4?j.json")
                   if not any(f.name.startswith(m.replace("/", "-") + "-") for m in UNFINISHED))  # scored files only
    tokens_cost = sum(((x.get("usage") or {}).get("cost") or 0) for f in files
                      for x in json.loads(f.read_text())["predictions"] if isinstance(x, dict))
    conds = {arm: per_condition(s, recs, arm) for arm in ("4aj", "4bj")}

    rows, rows_c = head["rows"], cond["rows"]
    order = sorted(models, key=lambda m: -(rows[(m, "4aj")]["score_z_bal"]["value"] or -1e9))
    z, zi = head["zero_reference"], i21["zero_reference"]
    L = ["# v0.3 full run: model scores (arm 4aj; arm 4bj for the original seven)", "",
         f"- **Cases:** {s.ab.key.n} (seed {meta['seed']}; `data/test_sets/{STEM}.case_ids.txt`, built by "
         "`scripts/build_v03_full_set.py`; design in docs/v0.3-case-selection-rules.md section 7.4, frozen at 7e67e24). Strata: "
         + ", ".join(f"{k} {n}" for k, n in strata.items()) + f". {len(meta['replaced'])} drawn BENIGN cases fell to X9 under the "
         f"key and were replaced by the next cases in their buckets. Cases with an exact public twin: {meta['twins']}.",
         f"- **Classes:** {head['serious']} SERIOUS, {head['benign']} BENIGN; the headline covers {head['headline_cases']} cases.",
         f"- **Scoring:** amendments A3-A5 with the selection rules' classes and credits (A4 truth partials), rule P5 demoted "
         f"(decision 21). Off-list tiers: NHAMCS-rated rows (A5). Zero point (A2): {z['code']} ({z['condition']}, a target on "
         f"{z['serious_cases']} SERIOUS cases). Arm 4aj is the headline and the only scored arm (record R3); the original "
         f"seven models also ran 4bj, which the methodology discusses.",
         "- **Intervals:** 95%, resampling cases within each true condition (the drawn mix is the estimand; spec record R2); "
         "the condition bootstrap, which also varies the mix, is the sensitivity column. 2,000 draws, seed 20260923.",
         f"- **Run:** `inference/run_config_v03_abj.json` via OpenRouter under the account's data policy: the original seven "
         f"models in prompts v7a4aj and v7a4bj, the {len(EXPANSION)} added models (results/v03_full/roster_expansion.md) in "
         f"v7a4aj only. Provenance in " + " and ".join(f"`{RUNS.relative_to(ROOT)}/{f}`" for f in PROVENANCE) +
         f". Token cost {tokens_cost:.2f} USD over {len(files)} files (account spend delta {spend:.2f} USD: "
         + "; ".join(f"{k} {v:.2f}" for k, v in spend_by_run.items()) + "). A parse failure left after the one retry is "
         "scored as unreadable (routine), as in Phase 2b." + "".join(
             f" Not scored: {m}, {why}." for m, why in UNFINISHED.items()),
         "- **No precision:** this set has no reference review.", "",
         "## Scores, arm 4aj (headline)", "",
         "Score is `score_z_bal`. U: SERIOUS cases costing a full miss. O: BENIGN cases escalated. Partial: SERIOUS cases "
         "charged a partial (in-list / off-list / truth). All in %.", ""]

    def score_table(arm):
        T = ["| Model | Score [95% CI, within-condition] | Condition bootstrap | U | O | Partial (in / off / truth) | Escalated |",
             "|---|---|---|---|---|---|---|"]
        for m in order:
            if (m, arm) not in rows:
                continue
            r, rc = rows[(m, arm)], rows_c[(m, arm)]
            T.append(f"| {m} | {ci(r['score_z_bal'])} | [{fmt(rc['score_z_bal']['ci'][0])}, {fmt(rc['score_z_bal']['ci'][1])}] | "
                     f"{ci(r['U'])} | {ci(r['O'])} | {fmt(r['partial']['value'])} ({fmt(r['partial_inlist']['value'])} / "
                     f"{fmt(r['partial_offlist']['value'])} / {fmt(r['partial_truth']['value'])}) | {fmt(r['esc']['value'])} |")
        return T
    L += score_table("4aj") + ["", "## Scores, arm 4bj (the original seven; for the methodology's discussion)", ""] + score_table("4bj")

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
          "condition bootstrap. Intervals are not corrected for the number of pairs.", ""]
    sep_summary = {}

    def pair(pp, arm, a, b):
        """The pair's difference as A minus B, whichever order the scorer stored it in."""
        d = pp.get((arm, a, b))
        if d is None:
            d0 = pp[(arm, b, a)]
            d = {"value": -d0["value"], "ci": [-d0["ci"][1], -d0["ci"][0]], "separated": d0["separated"]}
        return d

    yn = lambda d: "yes" if d["separated"] else "no"
    for arm in ("4aj", "4bj"):
        pairs, pairs_c = ({k: d for k, d in sc["model_pairs"].items() if k[0] == arm} for sc in (head, cond))
        sep, sep_c = sum(d["separated"] for d in pairs.values()), sum(d["separated"] for d in pairs_c.values())
        sep_summary[arm] = {"within": sep, "condition_bootstrap": sep_c, "pairs": len(pairs)}
        arm_order = [m for m in order if (m, arm) in rows]
        L += [f"### Arm {arm}: {sep} of {len(pairs)} pairs separated ({sep_c} under the condition bootstrap)", ""]
        if len(arm_order) <= FULL_PAIR_LIST_MAX:
            L += ["| Model A (higher by 4aj score) | Model B | A minus B [95% CI] | Separated | Condition bootstrap |", "|---|---|---|---|---|"]
            for i, a in enumerate(arm_order):
                for b in arm_order[i + 1:]:
                    d, dc = pair(pairs, arm, a, b), pair(pairs_c, arm, a, b)
                    L.append(f"| {a} | {b} | {ci(d)} | {yn(d)} | {yn(dc)} |")
        else:
            L += ["Pairs adjacent in rank (every pair is in scores.json, `headline_within_condition.model_pairs`):", "",
                  "| Rank | Model A | Model B (next in rank) | A minus B [95% CI] | Separated | Condition bootstrap |",
                  "|---|---|---|---|---|---|"]
            for i, (a, b) in enumerate(zip(arm_order, arm_order[1:]), 1):
                d, dc = pair(pairs, arm, a, b), pair(pairs_c, arm, a, b)
                L.append(f"| {i}-{i + 1} | {a} | {b} | {ci(d)} | {yn(d)} | {yn(dc)} |")
            L += ["", "Per model, the models it does not separate from (within-condition intervals), with rank:", "",
                  "| Rank | Model | Separated from | Not separated from |", "|---|---|---|---|"]
            for i, a in enumerate(arm_order, 1):
                tied = [f"{b} ({j})" for j, b in enumerate(arm_order, 1) if b != a and not pair(pairs, arm, a, b)["separated"]]
                L.append(f"| {i} | {a} | {len(arm_order) - 1 - len(tied)} of {len(arm_order) - 1} | {', '.join(tied) or '-'} |")
        L.append("")

    L += ["## Confirmatory anchor test (spec record R2)", "",
          "Arm 4aj minus 4bj headline cost per 100 headline cases (miss 7, partial 1, over-escalation 1), paired on the same "
          "cases and within-condition draws. A positive difference means the benign anchor (arm 4bj) lowers the model's cost. "
          f"Holm's step-down procedure at a family-wise {ALPHA:.0%} across the seven models.", "",
          "| Model | Cost 4aj | Cost 4bj | 4aj minus 4bj [95% CI] | p | Holm threshold | Holm-adjusted p | Effect |",
          "|---|---|---|---|---|---|---|---|"]
    for m in [m for m in order if m in anchor]:
        a = anchor[m]
        eff = ("yes, anchor lowers cost" if a["diff_per_100"] > 0 else "yes, anchor raises cost") if a["reject"] else "no"
        L.append(f"| {m} | {a['cost_4aj']:.1f} | {a['cost_4bj']:.1f} | {fmt(a['diff_per_100'])} [{fmt(a['ci'][0])}, {fmt(a['ci'][1])}] | "
                 f"{a['p']:.4f} | {a['threshold']:.4f} | {a['p_holm']:.4f} | {eff} |")
    L += ["", "## Arm 4aj minus 4bj, per model (paired, within-condition)", "",
          "| Model | Score | U | O | Escalated |", "|---|---|---|---|---|"]
    for m in [m for m in order if m in head["paired_4aj_minus_4bj"]]:
        d = head["paired_4aj_minus_4bj"][m]
        L.append(f"| {m} | {ci(d['score_z_bal'])} | {ci(d['U'])} | {ci(d['O'])} | {ci(d['esc'])} |")

    L += ["", "## Sensitivity rows (within-condition intervals)", "",
          f"CCSR: the off-list tiers include the CCSR-rated rows. I21: the zero point pinned to {zi['code']} ({zi['condition']}, "
          f"a target on {zi['serious_cases']} SERIOUS cases).", "",
          "| Model | Arm | Headline | CCSR tiers included | Zero at I21 |", "|---|---|---|---|---|"]
    for m in order:
        for arm in [a for a in ("4aj", "4bj") if (m, a) in rows]:
            L.append(f"| {m} | {arm} | {ci(rows[(m, arm)]['score_z_bal'])} | {ci(ccsr['rows'][(m, arm)]['score_z_bal'])} | "
                     f"{ci(i21['rows'][(m, arm)]['score_z_bal'])} |")

    short = [pc.sb.short(m) for m in MODELS]
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
        "token_cost_usd": round(tokens_cost, 4), "account_spend_usd": spend, "account_spend_usd_by_run": spend_by_run,
        "not_scored": UNFINISHED},
        indent=1, default=str) + "\n")
    for m in order:
        b = ci(rows[(m, "4bj")]["score_z_bal"]) if (m, "4bj") in rows else "-"
        an = f"anchor {anchor[m]['diff_per_100']} {anchor[m]['ci']} p_holm {anchor[m]['p_holm']} {anchor[m]['reject']}" if m in anchor else ""
        print(f"{m:24s} 4aj {ci(rows[(m, '4aj')]['score_z_bal']):24s} 4bj {b:24s} {an}")
    print(f"pairs separated {sep_summary}; token cost {tokens_cost:.2f}; spend {spend:.2f}; wrote {OUT / 'scores.md'}")


if __name__ == "__main__":
    main()
