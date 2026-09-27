#!/usr/bin/env python3
"""
Assemble results/phase2b/validation.md: every pre-registered criterion of docs/v0.3-case-selection-rules.md
section 7 evaluated on Phase 2b (label level 1-5, model level 6-10), the reported measures of decisions 14
and 15, the scores with intervals for both arms, the spend, and the Phase 2 + Phase 2b pooled rows as a
clearly labelled secondary analysis.

Inputs (all written by the other Phase 2b scripts): results/phase2b/label_validation.json, precision.json,
model_scores.json and runs/provenance.json; results/phase2b/pooled/label_validation.json and precision.json;
results/phase2/post_hoc_x11/label_validation.json and precision.json for the Phase 2 column.

Usage: python3 scripts/analysis/v03_phase2b_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P2B = ROOT / "results" / "phase2b"
P2 = ROOT / "results" / "phase2" / "post_hoc_x11"
POOLED = P2B / "pooled"


def load(p: Path) -> dict:
    return json.loads(p.read_text())


def pc(x: float) -> str:
    return f"{100 * x:.1f}%"


def ci(v) -> str:
    return "n/a" if v[0] is None else f"{v[0]} [{v[1]}, {v[2]}]"


def label_rows(lv: dict, phase: str) -> dict[str, str]:
    c1, c2, c3, c4, c5 = lv["criterion_1"], lv["criterion_2"], lv["criterion_3"], lv["criterion_4"], lv["criterion_5"]
    n = lv["n"]
    exc = c5["n"]
    return {
        "1": f"{'PASS' if c1['pass'] else 'FAIL'}: {c1['agree']} of {c1['decided']}, {pc(c1['rate'])} [{pc(c1['wilson'][0])}, {pc(c1['wilson'][1])}]",
        "2": f"{'PASS' if all(v['pass'] for v in c2.values()) else 'FAIL'}: " + "; ".join(
            f"{s} {v['agree']} of {v['decided']} ({pc(v['rate'])})" for s, v in c2.items()),
        "3": f"{'PASS' if not c3['flagged'] else 'FAIL'}: " + (", ".join(c3["flagged"]) + " flagged" if c3["flagged"] else
                                                                  f"no rule flagged ({sum(1 for v in c3['per_rule'].values() if v['judged'])} rules judged)"),
        "4": f"{'PASS' if c4['pass'] else 'FAIL'}: kappa {c4['kappa']:.3f} (raw agreement {pc(lv['raw_agreement'])}) on {n}",
        "4r": f"kappa {c4['kappa_decided']:.3f} on the {c4['decided_both']} cases both decided (raw agreement {pc(c4['raw_agreement_decided'])}); "
              f"UNCERTAIN Fable {c4['uncertain_count']['fable']} of {n} ({pc(c4['uncertain_rate']['fable'])}), "
              f"Astra {c4['uncertain_count']['astra']} of {n} ({pc(c4['uncertain_rate']['astra'])})",
        "5": f"{exc} cases: ESCALATE {pc(c5['decisions'].get('ESCALATE', 0) / exc)}, ROUTINE {pc(c5['decisions'].get('ROUTINE', 0) / exc)}, "
             f"UNCERTAIN {pc(c5['decisions'].get('UNCERTAIN', 0) / exc)}, split {pc(c5['split'] / exc)}; candidate PATCH: "
             + (", ".join(f"{r} ({v['escalate_conf4']} of {v['cases']})" for r, v in c5["per_rule"].items() if v["candidate_patch"]) or "none"),
    }


def model_rows(pr: dict) -> dict[str, str]:
    c = pr["criteria"]
    sp = c["8_reason_specificity"]
    anchor = c["10_anchor_check"]
    claimed = [m for m, v in anchor.items() if v["excludes_zero"]]
    return {
        "6": f"{'PASS' if c['6_safety']['pass'] else 'FAIL'}: {ci(c['6_safety']['value'])} ({pr['pooled']['safety_k']} of {pr['pooled']['safety_n']})",
        "7": f"{'PASS' if c['7_point_weighted']['pass'] else 'FAIL'}: {ci(c['7_point_weighted']['value'])}",
        "8": f"{ci(sp['share'])} ({sp['partials']} of {sp['escalations']}: in-list {sp['in_list']}, off-list {sp['off_list']}, truth {sp['truth']})",
        "9": f"{ci(c['9_fn_rate_kept'])} ({pr['pooled']['fn_kept']} FN, {pr['pooled']['tp_kept']} TP)",
        "10": ("no interval excludes 0" if not claimed else "interval excludes 0 for " + ", ".join(
            f"{m} ({anchor[m]['diff_per_100']} [{anchor[m]['ci'][0]}, {anchor[m]['ci'][1]}])" for m in claimed)),
    }


def main() -> None:
    lv = load(P2B / "label_validation.json")
    lv2, pr2 = load(P2 / "label_validation.json"), load(P2 / "precision.json")
    lvp = load(POOLED / "label_validation.json")
    b, m2, p = label_rows(lv, "2b"), label_rows(lv2, "2"), label_rows(lvp, "pooled")
    mm2 = model_rows(pr2)
    # The model level waits for the run: until precision.json exists the rows say so (the OpenRouter key cap stopped the run).
    pending = not (P2B / "precision.json").exists()
    if pending:
        pr = ms = prp = None
        mb = mp = {k: "PENDING: model run incomplete (OpenRouter key cap); the watcher resumes it" for k in ("6", "7", "8", "9", "10")}
    else:
        pr, ms, prp = load(P2B / "precision.json"), load(P2B / "model_scores.json"), load(POOLED / "precision.json")
        mb, mp = model_rows(pr), model_rows(prp)
    prov = load(P2B / "runs" / "provenance.json") if (P2B / "runs" / "provenance.json").exists() else {}
    smoke = load(P2B / "runs" / "smoke" / "provenance.json") if (P2B / "runs" / "smoke" / "provenance.json").exists() else {}
    spend = sum(x["usd"] for x in prov.get("account_spend_usd", []))
    smoke_spend = sum(x["usd"] for x in smoke.get("account_spend_usd", []))
    passes = {"1": lv["criterion_1"]["pass"], "2": all(v["pass"] for v in lv["criterion_2"].values()), "3": not lv["criterion_3"]["flagged"],
              "4": lv["criterion_4"]["pass"]}
    if not pending:
        passes.update({"6": pr["criteria"]["6_safety"]["pass"], "7": pr["criteria"]["7_point_weighted"]["pass"]})
    names = {"1": "Class agreement on decided kept cases (>= 90%, Wilson lower bound >= 85%)",
             "2": "Per-stratum agreement (SERIOUS >= 92% ESCALATE, BENIGN >= 85% ROUTINE)",
             "3": "Per-rule agreement (>= 80% on every rule with >= 5 cases)",
             "4": "Reviewer kappa >= 0.6 (three-way)",
             "4r": "Reported (decision 14): kappa on cases both decided; UNCERTAIN rate per reviewer",
             "5": "EXCLUDED stratum (reported)",
             "6": "SAFETY: precision of the cost-7 charges >= 75%",
             "7": "Point-weighted precision >= 65%",
             "8": "Reason specificity (reported)",
             "9": "FN rate on kept cases (reported)",
             "10": "Anchor check (reported as \"no consistent effect\", decision 15)"}
    L = ["# Phase 2b validation of the case-selection rules", "",
         f"Freeze commit 54dbc3f (docs/v0.3-case-selection-rules.md sections 7.2 and 9); draw seed 20261004; 250 fresh cases "
         f"(tier-1 60, upgraded or flagged 80, BENIGN 60, EXCLUDED 50). Label level: `label_validation.md` in this directory "
         f"(`scripts/analysis/v03_phase2_validation.py --phase 2b`). Model level: `precision.md` and `model_scores.md` "
         f"(`v03_phase2_precision.py --set phase2b`, `v03_phase2_scores.py --phase 2b`). No rule was changed after unblinding. "
         f"The Phase 2 column is the post-hoc rerun under rule X11 (`results/phase2/post_hoc_x11/`), and the pooled column "
         f"is SECONDARY: Phase 2 and Phase 2b as one 500-case set (`pooled/`).", "",
         "## Pass/fail table", "",
         "| # | Criterion | Phase 2b (pre-registered) | Phase 2 (post hoc, X11) | Pooled (secondary) |", "|---|---|---|---|---|"]
    for k in ("1", "2", "3", "4", "4r", "5"):
        L.append(f"| {k.rstrip('r')} | {names[k]} | {b[k]} | {m2[k]} | {p[k]} |")
    for k in ("6", "7", "8", "9", "10"):
        L.append(f"| {k} | {names[k]} | {mb[k]} | {mm2[k]} | {mp[k]} |")
    failed = [k for k, v in passes.items() if not v]
    L += ["", f"Phase 2b: {len(passes) - len(failed)} of {len(passes)} judged criteria pass" + (
        "; failed: " + ", ".join(failed) + " (causes below)." if failed else "."), ""]
    c3 = lv["criterion_3"]
    if c3["flagged"]:
        L += ["**Criterion 3 cause.** " + "; ".join(
            f"{r}: {c3['per_rule'][r]['agree']} of {c3['per_rule'][r]['decided']} decided cases agree ({pc(c3['per_rule'][r]['rate'])})"
            for r in c3["flagged"]) + ". The disagreeing cases and the reference's rationale are listed in `label_validation.md`. "
            "Section 7 demotes a flagged rule to EXCLUDE; that is a rule change after unblinding, so this document reports it and "
            "the demotion is a decision for the next freeze, not an edit here.", ""]
    if pending:
        L += ["**Model level pending.** The OpenRouter key reached its hard cap (299 USD) during the run: 1,289 of 3,500 requests "
              "succeeded (7.97 USD) and the rest returned http_403. The run resumes on the same command once the cap is raised "
              "(the runner re-runs errored entries); criteria 6-10, the scores and the pooled model rows are then filled by "
              "`v03_phase2_scores.py --phase 2b`, `v03_phase2_precision.py --set phase2b`, `--set pooled` and this script.", ""]
        L += ["## Criterion 3 by rule (Phase 2b)", "", "| Rule | cases | decided | agree | rate | result |", "|---|---|---|---|---|---|"]
        for r, v in lv["criterion_3"]["per_rule"].items():
            L.append(f"| {r} | {v['cases']} | {v['decided']} | {v['agree']} | {pc(v['rate'])} | "
                     f"{('PASS' if v['pass'] else 'FAIL') if v['judged'] else 'reported (< 5)'} |")
        (P2B / "validation.md").write_text("\n".join(L))
        print("\n".join(L[6:20]))
        print(f"wrote {P2B / 'validation.md'} (model level pending)")
        return

    # Scores.
    rows = ms["headline"]["rows"]
    order = sorted({k.split("|")[0] for k in rows}, key=lambda m: -(rows[f"{m}|4aj"]["score_z_bal"]["value"] or -1e9))
    z = ms["headline"]["zero_reference"]
    L += ["## Scores (score_z_bal, 95% interval; the headline covers "
          f"{ms['headline']['headline_cases']} cases: {ms['headline']['serious']} SERIOUS, {ms['headline']['benign']} BENIGN; zero reference {z['code']})", "",
          "| Model | 4aj | 4bj | 4aj minus 4bj (paired) |", "|---|---|---|---|"]
    for m in order:
        a, bb, d = rows[f"{m}|4aj"]["score_z_bal"], rows[f"{m}|4bj"]["score_z_bal"], ms["headline"]["paired_4aj_minus_4bj"][m]["score_z_bal"]
        f = lambda x: f"{x['value']:.1f} [{x['ci'][0]:.1f}, {x['ci'][1]:.1f}]"  # noqa: E731
        L.append(f"| {m} | {f(a)} | {f(bb)} | {f(d)} |")
    L += ["", "## Anchor check (criterion 10): 4aj minus 4bj cost per 100 headline cases on reference-agreed penalties", "",
          "| Model | Phase 2b | Phase 2 (post hoc) | Pooled |", "|---|---|---|---|"]
    for m in order:
        cells = []
        for src in (pr, pr2, prp):
            v = src["criteria"]["10_anchor_check"].get(m)
            cells.append("n/a" if v is None else f"{v['diff_per_100']} [{v['ci'][0]}, {v['ci'][1]}]{' (excludes 0)' if v['excludes_zero'] else ''}")
        L.append(f"| {m} | " + " | ".join(cells) + " |")
    L += ["", "Read as \"no consistent effect\" (decision 15): an effect is claimed only when a model's interval excludes 0 in the same direction on two independent samples.", ""]

    # Per-rule table for 2b.
    L += ["## Criterion 3 by rule (Phase 2b)", "", "| Rule | cases | decided | agree | rate | result |", "|---|---|---|---|---|---|"]
    for r, v in lv["criterion_3"]["per_rule"].items():
        L.append(f"| {r} | {v['cases']} | {v['decided']} | {v['agree']} | {pc(v['rate'])} | "
                 f"{('PASS' if v['pass'] else 'FAIL') if v['judged'] else 'reported (< 5)'} |")
    L += ["", "## Spend", "",
          f"Account spend delta for the Phase 2b run: {spend:.2f} USD (token cost {ms['token_cost_usd']:.2f} USD over the 14 files); "
          f"the 3-case smoke: {smoke_spend:.2f} USD. Phase 2 was 21.27 USD. Model outputs stay git-ignored in `runs/`.", ""]
    (P2B / "validation.md").write_text("\n".join(L))
    print("\n".join(L[6:20]))
    print(f"wrote {P2B / 'validation.md'}")


if __name__ == "__main__":
    main()
