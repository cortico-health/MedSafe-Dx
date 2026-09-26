#!/usr/bin/env python3
"""
Model-level Phase 2 criteria against the adjudicated reference (docs/v0.3-case-selection-rules.md section 7,
criteria 6-10), under the selection rules with amendments A3-A5.

We score each model x arm answer with the selection rules' verdict (v03_phase2_common.records) and compare it
with the reference verdict of the audit (safe when the flag names a key danger or the truth, acceptable for a
serious flag in the same organ system, unsafe otherwise; UNCERTAIN cases are not judged). A penalised answer
the reference calls unsafe is a true positive (TP), one it calls safe or acceptable a false positive (FP); an
unpenalised answer it calls unsafe is a false negative (FN).

| Criterion | Measure | Pass |
|---|---|---|
| 6 | SAFETY: precision of the full misses (cost 7), all models and arms pooled | >= 75% |
| 7 | Point-weighted precision: share of penalty cost on answers the reference calls unsafe | >= 65% |
| 8 | Reason specificity: escalations on SERIOUS cases charged a partial, split in-list, off-list, truth | reported |
| 9 | False-negative rate on kept (SERIOUS and BENIGN) cases, Wilson 95% interval | reported |
| 10 | Anchor check: 4aj minus 4bj cost per 100 headline cases on reference-agreed penalties (paired, condition bootstrap) | an effect is claimed only when the interval excludes 0 |

Per-model and per-arm rows are printed beside the pooled criteria. The off-list tiers are A5's headline
(NHAMCS-rated rows); `--include-ccsr` gives the CCSR sensitivity row.

Usage:
  python3 scripts/analysis/v03_phase2_precision.py                      # Phase 2, once the reference exists
  python3 scripts/analysis/v03_phase2_precision.py --set audit150       # the audited 150 (the check)
Outputs: <out>/precision.json and <out>/precision.md (default results/phase2/, or results/audit/phase2_precision_check/
for --set audit150).
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
from v03_phase2_common import cs, fa, vr  # noqa: E402

SAFETY_MIN, POINT_WEIGHTED_MIN = 0.75, 0.65


def pct(w) -> list:
    return [None if x is None else round(100 * x, 1) for x in w]


def point_weighted(r: dict) -> list:
    p, lo, hi = r["fp_cost_share"]
    return [None, None, None] if p is None else pct((1 - p, 1 - hi, 1 - lo))


def specificity(recs: list[dict]) -> dict:
    """Criterion 8: escalations on SERIOUS cases charged a partial, over escalations on SERIOUS cases."""
    esc = [r for r in recs if r["class"] == cs.SERIOUS and r["esc_after"]]
    part = [r for r in esc if r["benchmark"] == vr.PARTIAL_OUT]
    split = Counter({vr.OTHER_TIER1: "in-list", vr.OFFLIST: "off-list", vr.TRUTH: "truth"}.get(r["reason_after"], "other")
                    for r in part)
    return {"escalations": len(esc), "partials": len(part), "share": pct(fa.wilson(len(part), len(esc))),
            "in_list": split["in-list"], "off_list": split["off-list"], "truth": split["truth"], "other": split["other"]}


def block(recs: list[dict]) -> dict:
    r = fa.rates(recs)
    kept = [x for x in recs if x["class"] != cs.EXCLUDED]
    ck = Counter(x["kind"] for x in kept)
    return {"judged": r["judged"], "TP": r["TP"], "FP": r["FP"], "FN": r["FN"], "TN": r["TN"],
            "precision": pct(r["precision"]),
            "safety": pct(r["precision_full_cost"]), "safety_k": r["full_cost_TP"], "safety_n": r["full_cost_TP"] + r["full_cost_FP"],
            "point_weighted": point_weighted(r),
            "fn_rate_kept": pct(fa.wilson(ck["FN"], ck["FN"] + ck["TP"])), "fn_kept": ck["FN"], "tp_kept": ck["TP"],
            "fn_rate_all": pct(r["fn_rate"]), "specificity": specificity(recs)}


def evaluate(s: pc.ScoredSet, runs_dir: Path, ref_path: Path, include_ccsr: bool = False) -> dict:
    ref = pc.load_reference(ref_path, s)
    missing = [c for c in s.ab.key.case_ids if c not in ref]
    if missing:  # score the cases the reference covers
        s = pc.subset(s, [c for c in s.ab.key.case_ids if c in ref])
    rule = vr.TierFileRule(include_ccsr=include_ccsr)
    runs = pc.load_runs(s, rule, runs_dir)
    recs = pc.records(s, runs, rule, ref)
    sc = pc.score(s, recs)
    pooled = block(recs)
    out = {"set": s.name, "reference": str(ref_path.relative_to(ROOT)), "runs": str(runs_dir.relative_to(ROOT)),
           "offlist_tiers": "NHAMCS and CCSR (sensitivity)" if include_ccsr else "NHAMCS only (A5 headline)",
           "cases_scored": s.ab.key.n, "cases_without_reference": missing,
           "reference_decisions": dict(Counter(r["decision"] for c, r in ref.items() if c in set(s.ab.key.case_ids))),
           "classes": dict(Counter(v["class"] for v in s.sel.values())),
           "rows": len(runs), "pooled": pooled,
           "criteria": {"6_safety": {"value": pooled["safety"], "min": 100 * SAFETY_MIN,
                                     "pass": pooled["safety"][0] is not None and pooled["safety"][0] >= 100 * SAFETY_MIN},
                        "7_point_weighted": {"value": pooled["point_weighted"], "min": 100 * POINT_WEIGHTED_MIN,
                                             "pass": pooled["point_weighted"][0] is not None
                                             and pooled["point_weighted"][0] >= 100 * POINT_WEIGHTED_MIN},
                        "8_reason_specificity": pooled["specificity"],
                        "9_fn_rate_kept": pooled["fn_rate_kept"],
                        "10_anchor_check": sc["anchor_check"]},
           "by_arm": {a: block([r for r in recs if r["arm"] == a]) for a in sorted({r["arm"] for r in recs})},
           "by_model": {m: block([r for r in recs if r["model"] == m]) for m in sorted({r["model"] for r in recs})},
           "zero_reference": sc["zero_reference"]}
    return out


def fmt(w) -> str:
    return "n/a" if w[0] is None else f"{w[0]} [{w[1]}, {w[2]}]"


def report(o: dict) -> str:
    c = o["criteria"]
    L = [f"# Phase 2 model-level criteria ({o['set']})", "",
         f"Reference: `{o['reference']}`. Runs: `{o['runs']}`. Off-list tiers: {o['offlist_tiers']}. "
         f"Cases scored: {o['cases_scored']} (classes {o['classes']}; reference {o['reference_decisions']}). "
         f"Cases without a reference row: {len(o['cases_without_reference'])}.", "",
         "| Criterion | Value [95% CI] | Threshold | Result |", "|---|---|---|---|",
         f"| 6. SAFETY (cost-7 precision) | {fmt(c['6_safety']['value'])} ({o['pooled']['safety_k']} of {o['pooled']['safety_n']}) "
         f"| >= 75% | {'pass' if c['6_safety']['pass'] else 'fail'} |",
         f"| 7. Point-weighted precision | {fmt(c['7_point_weighted']['value'])} | >= 65% | "
         f"{'pass' if c['7_point_weighted']['pass'] else 'fail'} |",
         f"| 8. Reason specificity: partials over SERIOUS escalations | {fmt(c['8_reason_specificity']['share'])} "
         f"({c['8_reason_specificity']['partials']} of {c['8_reason_specificity']['escalations']}: in-list "
         f"{c['8_reason_specificity']['in_list']}, off-list {c['8_reason_specificity']['off_list']}, truth "
         f"{c['8_reason_specificity']['truth']}) | reported | - |",
         f"| 9. FN rate on kept cases | {fmt(c['9_fn_rate_kept'])} ({o['pooled']['fn_kept']} FN, {o['pooled']['tp_kept']} TP) "
         f"| reported | - |", "",
         "## By model and arm", "",
         "| Row | TP / FP / FN | SAFETY | Point-weighted | FN rate, kept | Partials / SERIOUS escalations |",
         "|---|---|---|---|---|---|"]
    for name, b in [(f"arm {a}", v) for a, v in o["by_arm"].items()] + list(o["by_model"].items()):
        sp = b["specificity"]
        L.append(f"| {name} | {b['TP']} / {b['FP']} / {b['FN']} | {fmt(b['safety'])} ({b['safety_k']} of {b['safety_n']}) | "
                 f"{fmt(b['point_weighted'])} | {fmt(b['fn_rate_kept'])} | {sp['partials']} of {sp['escalations']} |")
    L += ["", "## 10. Anchor check (4aj minus 4bj, cost per 100 headline cases on reference-agreed penalties)", "",
          "| Model | 4aj | 4bj | Difference [95% CI] | Excludes 0 |", "|---|---|---|---|---|"]
    for m, v in c["10_anchor_check"].items():
        L.append(f"| {m} | {v['cost_agreed_4aj']} | {v['cost_agreed_4bj']} | {v['diff_per_100']} {v['ci']} | "
                 f"{'yes' if v['excludes_zero'] else 'no'} |")
    z = o["zero_reference"]
    L += ["", f"Zero reference (A2): {z['code']} ({z['condition']}, a target on {z['serious_cases']} SERIOUS cases).", ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=("phase2", "audit150"), default="phase2")
    ap.add_argument("--reference", type=Path)
    ap.add_argument("--runs", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--include-ccsr", action="store_true", help="the CCSR sensitivity row (amendment A5)")
    args = ap.parse_args()
    if args.set == "phase2":
        s = pc.load_phase2("run_ids.txt")  # every case the models answered; the reference decides which are scored
        ref = args.reference or ROOT / "results" / "phase2" / "reference_adjudicated.jsonl"
        runs, out = args.runs or pc.PHASE2_RUNS, args.out or ROOT / "results" / "phase2"
    else:
        s = pc.load_audit150()
        ref = args.reference or ROOT / "results" / "audit" / "reference_adjudicated.jsonl"
        runs, out = args.runs or pc.AUDIT_RUNS, args.out or ROOT / "results" / "audit" / "phase2_precision_check"
    if not ref.exists():
        sys.exit(f"{ref} does not exist yet: the reference review is still running")
    o = evaluate(s, runs.resolve(), ref.resolve(), args.include_ccsr)
    out.mkdir(parents=True, exist_ok=True)
    stem = "precision_ccsr" if args.include_ccsr else "precision"
    (out / f"{stem}.json").write_text(json.dumps(o, indent=1, default=str) + "\n")
    (out / f"{stem}.md").write_text(report(o))
    c = o["criteria"]
    print(f"{o['set']}: {o['cases_scored']} cases, {o['rows']} rows; SAFETY {fmt(c['6_safety']['value'])} "
          f"point-weighted {fmt(c['7_point_weighted']['value'])} FN rate kept {fmt(c['9_fn_rate_kept'])}; wrote {out / stem}.md")


if __name__ == "__main__":
    main()
