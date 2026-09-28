#!/usr/bin/env python3
"""
Score the Phase 2 or Phase 2b model run (docs/v0.3-case-selection-rules.md section 7) without the reference.

The seven models x arms 4aj and 4bj on the 250 cases of the phase (data/test_sets/eval-v03-phase2*.case_ids.txt),
under the selection rules' classes and credits with amendments A3-A5 (scripts/analysis/v03_phase2_common.py).
We report the primary score (`score_z_bal`, A2: blanket escalation with one fixed flag scores 0, perfect 100)
with its 95% interval (condition bootstrap, 2,000 draws, seed 20260923), U, O, the partials split by reason
kind, the paired 4aj minus 4bj differences, every model pair within an arm, and two sensitivity rows: the
CCSR off-list tiers included, and the zero reference pinned to I21. Precision against the reference is
scripts/analysis/v03_phase2_precision.py, once the reference exists.

Outputs: <out>/model_scores.md and <out>/model_scores.json (aggregates only; the model outputs stay in the
phase's runs directory, which git ignores). Default <out> is the phase's results directory; --out overrides it
(the post-hoc Phase 2 rescoring under rule X11 goes to results/phase2/post_hoc_x11/).

Usage: python3 scripts/analysis/v03_phase2_scores.py --phase 2b [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

import v03_phase2_common as pc  # noqa: E402
from v03_phase2_common import vr  # noqa: E402


def v(m: dict) -> str:
    x = m["value"]
    return "n/a" if x is None else f"{x:.1f}"


def ci(m: dict) -> str:
    lo, hi = m["ci"]
    return f"{v(m)} [{'n/a' if lo is None else f'{lo:.1f}'}, {'n/a' if hi is None else f'{hi:.1f}'}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=tuple(pc.PHASES), default="2b")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--label", default="", help="a note for the title, e.g. 'post hoc, rule X11'")
    args = ap.parse_args()
    ph = pc.PHASES[args.phase]
    OUT, RUNS, STEM = args.out or ph["out"], ph["runs"], ph["stem"]
    s = pc.load_phase2(phase=args.phase)
    head_rule, ccsr_rule = vr.TierFileRule(), vr.TierFileRule(include_ccsr=True)
    runs = pc.load_runs(s, head_rule, RUNS)
    assert len(runs) == len(pc.MODELS) * len(pc.ARMS), f"{len(runs)} prediction files; expected {len(pc.MODELS) * len(pc.ARMS)}"
    recs = pc.records(s, runs, head_rule)
    head = pc.score(s, recs)
    sens_ccsr = pc.score(s, pc.records(s, pc.load_runs(s, ccsr_rule, RUNS), ccsr_rule))
    sens_i21 = pc.score(s, recs, zero_code="I21")
    prov = json.loads((RUNS / "provenance.json").read_text())
    parse = {f"{pc.sb.short(m)}|{pc.sb.ARM_LABELS[a]}": p for (m, a), (_, _, p) in runs.items()}
    strata = Counter(c["phase2_stratum"] for c in s.ab.cases)
    meta = json.loads((pc.TS / f"{STEM}.json").read_text())["metadata"]

    rows = head["rows"]
    order = sorted({m for m, _ in rows}, key=lambda m: -(rows[(m, "4aj")]["score_z_bal"]["value"] or -1e9))
    z, zi = head["zero_reference"], sens_i21["zero_reference"]
    spend = sum(x["usd"] for x in prov.get("account_spend_usd", []))
    files = sorted(RUNS.glob("*-v7a4?j.json"))
    tokens_cost = sum(((x.get("usage") or {}).get("cost") or 0) for f in files
                      for x in json.loads(f.read_text())["predictions"] if isinstance(x, dict))
    replaced = (f" {len(meta['dxa_only_replaced'])} of the 20 drawn DXA-only cases had no DXA-only target left once the key was "
                f"built (red herrings), so the next cases in the stratum's shuffled pool replaced them (section 7); the models also "
                f"answered the {len(meta['dxa_only_replaced'])} replaced cases, which are not scored here."
                if meta.get("dxa_only_replaced") else "")
    L = [f"# Phase {args.phase} model scores (arms 4aj and 4bj{', ' + args.label if args.label else ''})", "",
         "Blinded to the reference: these are benchmark scores only. Precision against the adjudicated reference comes from "
         f"`scripts/analysis/v03_phase2_precision.py` once `{ph['out'].relative_to(ROOT)}/reference_adjudicated.jsonl` exists.", "",
         f"- **Cases:** the 250 Phase {args.phase} cases (seed {meta['seed']}; `data/test_sets/{STEM}.case_ids.txt`, built by "
         f"`scripts/build_v03_phase2_set.py --phase {args.phase}`). Strata: " + ", ".join(f"{k} {n}" for k, n in strata.items()) + "."
         + replaced,
         f"- **Classes** (selection rules, key targets): {head['serious']} SERIOUS, {head['benign']} BENIGN, "
         f"{s.ab.key.n - head['headline_cases']} EXCLUDED; the headline covers {head['headline_cases']} cases.",
         f"- **Scoring:** amendments A3-A5 with the selection rules' classes and credits (A4 truth partials). Off-list tiers: "
         f"NHAMCS-rated rows only (A5). Zero reference (A2): {z['code']} ({z['condition']}, a target on {z['serious_cases']} "
         f"SERIOUS cases).",
         f"- **Run:** `inference/run_config_v03_abj.json` (as the 4aj and 4bj runs on the 150), prompt v7a4aj and v7a4bj with the "
         f"justification line, via OpenRouter; provenance in `{RUNS.relative_to(ROOT)}/provenance.json`. Token cost {tokens_cost:.2f} USD "
         f"over {len(files)} files and every case the models answered (account spend delta {spend:.2f} USD). "
         f"A json_parse_failure left after the one retry is scored as unreadable (routine), as on the 150 (spec section 9); "
         f"the runner lists those files as incomplete in the provenance, and we did not re-run them.", "",
         "## Scores", "",
         "Score is `score_z_bal` with its 95% interval. U: SERIOUS cases costing a full miss. O: BENIGN cases escalated. "
         "Partial: SERIOUS cases charged a partial (in-list / off-list / truth). All in %.", "",
         "| Model | Arm | Score [95% CI] | Score, sample mix | U | O | Partial (in / off / truth) | Escalated |",
         "|---|---|---|---|---|---|---|---|"]
    for m in order:
        for arm in ("4aj", "4bj"):
            r = rows[(m, arm)]
            L.append(f"| {m} | {arm} | {ci(r['score_z_bal'])} | {v(r['score_z_mix'])} | {ci(r['U'])} | {ci(r['O'])} | "
                     f"{v(r['partial'])} ({v(r['partial_inlist'])} / {v(r['partial_offlist'])} / {v(r['partial_truth'])}) | {v(r['esc'])} |")
    L += ["", "## Arm 4aj minus 4bj, per model (paired)", "",
          "| Model | Score | U | O | Escalated |", "|---|---|---|---|---|"]
    for m in order:
        d = head["paired_4aj_minus_4bj"][m]
        L.append(f"| {m} | {ci(d['score_z_bal'])} | {ci(d['U'])} | {ci(d['O'])} | {ci(d['esc'])} |")
    L += ["", "## Model-pair separation (paired score difference, same cases and draws)", "",
          "A pair is separated when the 95% interval of the score difference excludes 0.", ""]
    for arm in ("4aj", "4bj"):
        pairs = {k: d for k, d in head["model_pairs"].items() if k[0] == arm}
        sep = sum(d["separated"] for d in pairs.values())
        L += [f"### Arm {arm}: {sep} of {len(pairs)} pairs separated", "",
              "| Model A (higher by 4aj score) | Model B | A minus B, score [95% CI] | Separated |", "|---|---|---|---|"]
        for i, a in enumerate(order):
            for b in order[i + 1:]:
                d = pairs.get((arm, a, b))
                if d is None:  # stored the other way round
                    d0 = pairs[(arm, b, a)]
                    d = {"value": -d0["value"], "ci": [-d0["ci"][1], -d0["ci"][0]], "separated": d0["separated"]}
                L.append(f"| {a} | {b} | {ci(d)} | {'yes' if d['separated'] else 'no'} |")
        L.append("")
    def zero_passes(code):
        zo, _ = pc.zero_outcome(s, code)
        return sum(1 for o, h in zip(zo.outcome, s.ab.serious) if h and o == vr.PASS)
    zp, zp21 = zero_passes(None), zero_passes("I21")
    L += ["## Sensitivity rows", "",
          f"The zero reference's flag passes on {zp} SERIOUS cases at {z['code']} and on {zp21} at I21 (a target or a "
          f"credited layer-a danger); " + ("equal pass counts give equal point scores, and the intervals differ because the "
          "passes fall on different conditions." if zp == zp21 else "the point scores move accordingly."), "",
          f"CCSR: the off-list tiers include the CCSR-rated rows (zero reference {sens_ccsr['zero_reference']['code']}). "
          f"I21: the zero reference pinned to {zi['code']} ({zi['condition']}, a target on {zi['serious_cases']} SERIOUS cases).", "",
          "| Model | Arm | Headline | CCSR tiers included | Zero at I21 |", "|---|---|---|---|---|"]
    for m in order:
        for arm in ("4aj", "4bj"):
            L.append(f"| {m} | {arm} | {ci(rows[(m, arm)]['score_z_bal'])} | {ci(sens_ccsr['rows'][(m, arm)]['score_z_bal'])} | "
                     f"{ci(sens_i21['rows'][(m, arm)]['score_z_bal'])} |")
    L += ["", "## Parsing", "",
          "| Row | Answered | Unreadable | Errored | No justification | Token cost (USD) |", "|---|---|---|---|---|---|"]
    for name, p in sorted(parse.items()):
        L.append(f"| {name} | {p['answered']} of {p['cases']} | {p['unreadable']} | {p['errored']} | {p['no_justification']} | {p['cost_usd']} |")
    L.append("")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "model_scores.md").write_text("\n".join(L))

    def js(sc):
        sc = {k: v for k, v in sc.items() if k not in ("raw", "zero_outcome", "M", "key")}
        return {**sc, "rows": {f"{m}|{a}": r for (m, a), r in sc["rows"].items()},
                "model_pairs": {"|".join(k): d for k, d in sc["model_pairs"].items()}}
    (OUT / "model_scores.json").write_text(json.dumps({"headline": js(head), "sensitivity_ccsr": js(sens_ccsr),
                                                       "sensitivity_zero_I21": js(sens_i21), "parse": parse,
                                                       "token_cost_usd": round(tokens_cost, 4), "account_spend_usd": spend},
                                                      indent=1, default=str) + "\n")
    for m in order:
        print(f"{m:24s} 4aj {ci(rows[(m, '4aj')]['score_z_bal']):24s} 4bj {ci(rows[(m, '4bj')]['score_z_bal'])}")
    print(f"zero {z['code']}; headline {head['headline_cases']} ({head['serious']} SERIOUS, {head['benign']} BENIGN); "
          f"token cost {tokens_cost:.2f}; wrote {OUT / 'model_scores.md'}")


if __name__ == "__main__":
    main()
