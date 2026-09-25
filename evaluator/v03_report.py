"""
Markdown report for a v0.3 board (the JSON evaluator/v03_score.py writes).

We print the headline beside its parts, as spec/v0.3-scoring.md section 6 requires:
SC with its interval, the miss and concern parts, H and OC, COV, the unreadable share
and SC on readable cases, and the constant rows; then the comparison against always
YES, the paired model differences, every sensitivity row, CON, MT and DX, the pool
subsets, and the harness metadata of each run.
"""

from __future__ import annotations

from typing import Mapping, Optional

from evaluator import v03_anchors


def _f(x: Optional[float], d: int = 1, pct: bool = False) -> str:
    if x is None:
        return "-"
    return f"{100 * x:.{d}f}" if pct else f"{x:.{d}f}"


def _ci(ci, d: int = 1, pct: bool = False) -> str:
    if not ci or ci[0] is None:
        return "-"
    return f"[{_f(ci[0], d, pct)}, {_f(ci[1], d, pct)}]"


def _v(row: Mapping, m: str, d: int = 1, pct: bool = False) -> str:
    return f"{_f(row['point'].get(m), d, pct)} {_ci(row['ci'].get(m), d, pct)}"


def _order(b: Mapping) -> list[str]:
    models = sorted(b["models"], key=lambda r: b["rows"][r]["point"]["SC"])
    refs = sorted(b["references"], key=lambda r: b["rows"][r]["point"]["SC"])
    return models + refs


def _label(b: Mapping, r: str) -> str:
    row = b["rows"][r]
    return f"**{r}**" if row.get("kind") == "model" else f"[{row.get('label', r)}]"


def headline_table(b: Mapping) -> list[str]:
    out = ["| Row | SC [95% CI] | Miss part | Concern part | H % | OC % | COV % | MT truth / DXA | "
           "Unreadable % (in R10) | SC readable | YES % |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in _order(b):
        row = b["rows"][r]
        p, c = row["point"], row["counts"]
        mt = c["MT"]
        out.append(
            f"| {_label(b, r)} | {_v(row, 'SC')} | {_f(p['miss_part'])} | {_f(p['concern_part'])} | "
            f"{_f(p['H'], 1, True)} ({c['misses']}/{c['r10_cases']}) | "
            f"{_f(p.get('OC'), 1, True)} ({c['concerns']}/{c['clearly_low_risk']}) | {_f(p.get('COV'), 1, True)} | "
            f"{mt['truth_missed']}/{mt['truth_targets']} / {mt['dxa_missed']}/{mt['dxa_targets']} | "
            f"{_f(p['unreadable_share'], 1, True)} ({c['unreadable_in_r10']}) | {_f(p.get('SC_readable'))} | "
            f"{_f(p['yes_rate'], 1, True)} |")
    return out


def render_report(board: Mapping) -> str:
    L = ["# MedSafe-Dx v0.3 scores", "",
         f"Spec: {board['spec']}. Scorer: evaluator/v03_score.py"
         + (f" at {board.get('provenance', {}).get('git_commit')}" if board.get("provenance") else "") + ".",
         "", board["reading"], "",
         f"Intervals: {board['constants']['n_boot']} draws. {board['constants']['bootstrap']} "
         "Paired differences use the same draws. Flags match under equivalent, narrower and broader codes; "
         "diagnosis under equivalent and narrower. Unreadable output scores as NO.", ""]
    for s, b in board["sets"].items():
        smp, k = b["sample"], b["constants"]
        L += [f"## {b['label']}", "",
              f"{smp['cases']} cases, {smp['conditions']} conditions: {smp['r10_cases']} with an R10 target "
              f"({smp['r10_targets']} targets, {smp['r10_undetermined']} in undetermined classes), "
              f"{smp['clearly_low_risk']} clearly low-risk, {smp['intermediate']} intermediate (either answer is free); "
              f"{smp['r5_cases']} with an R5 target; {smp['red_flag']} with a section 7 red flag.", "",
              f"Constant rows on these cases: perfect {k['perfect']:.1f}, always YES {k['always_yes']:.1f}, "
              f"always NO {k['always_no']:.1f}.", ""]
        L += headline_table(b) + [""]
        L += ["MT is R10 targets not flagged (true-condition / DXA-derived). "
              "Unreadable % is the share of all cases; the bracket counts unreadable R10 cases.", ""]
        if b.get("vs_always_yes"):
            L += ["### Against blanket concern (row minus always YES, SC)", "",
                  "| Row | Difference [95% CI] | Beats blanket concern |", "|---|---|---|"]
            for r in _order(b):
                d = b["vs_always_yes"].get(r)
                if d:
                    L.append(f"| {_label(b, r)} | {_f(d['SC']['diff'])} {_ci(d['SC']['ci'])} | "
                             f"{'yes' if d['beats_blanket_concern'] else 'no'} |")
            L.append("")
        if b.get("paired"):
            L += ["### Paired model differences (first minus second)", "",
                  "| Pair | SC | Miss part | Concern part | H (pp) | OC (pp) | COV (pp) |", "|---|---|---|---|---|---|---|"]
            for pair, d in b["paired"].items():
                cell = lambda m, pct=False: (f"{_f(d[m]['diff'], 1, pct)} {_ci(d[m]['ci'], 1, pct)}" if m in d else "-")
                L.append(f"| {pair.replace('|', ' vs ')} | {cell('SC')} | {cell('miss_part')} | {cell('concern_part')} | "
                         f"{cell('H', True)} | {cell('OC', True)} | {cell('COV', True)} |")
            L.append("")
        cols = b["models"] + [r for r in ("always-yes", "naive-bayes", "dxa") if r in b["rows"]]
        sens = b["rows"][cols[0]]["counts"].get("sensitivity") if cols else None
        if sens:
            L += ["### Sensitivity rows (SC, cost points per 100 patients)", "",
                  "| Row | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
            L.append("| Primary SC | " + " | ".join(_v(b["rows"][c], "SC") for c in cols) + " |")
            for name, meta in sens.items():
                extra = ""
                if "r10_cases" in meta:
                    extra = f" ({meta['r10_cases']} R10, {meta['clearly_low_risk']} clearly low-risk)"
                L.append(f"| {name}: {meta['description']}{extra} | "
                         + " | ".join(_v(b["rows"][c], f"sens:{name}") for c in cols) + " |")
            L.append("")
            for m, lab in (("always_yes_by_mix", "Always YES under each mix"),
                           ("always_yes_by_tiers", "Always YES under each tier rule")):
                if k.get(m):
                    L.append(f"{lab}: " + ", ".join(f"{n} {v:.1f}" for n, v in k[m].items()) + ".")
            L += ["", "### Code maps (COV, MT, H' SC, CON)", "",
                  "| Row | Map | COV % | MT % | SC under H' | YES no flags | YES no tier-1 flag | NO with tier-1 flag |",
                  "|---|---|---|---|---|---|---|---|"]
            for c in cols:
                row = b["rows"][c]
                con = row["counts"]["CON"]
                L.append(f"| {c} | standard | {_v(row, 'COV', 1, True)} | {_v(row, 'MT', 1, True)} | "
                         f"{_v(row, 'sens:H_prime')} | {con['yes_without_flags']} | {con['yes_without_tier1_flag']} | "
                         f"{con['no_with_tier1_flag']} |")
                for pol in ("strict", "lenient"):
                    con = row["counts"]["maps"][pol]["CON"]
                    L.append(f"| {c} | {pol} | {_v(row, f'map:{pol}:COV', 1, True)} | {_v(row, f'map:{pol}:MT', 1, True)} | "
                             f"{_v(row, f'map:{pol}:SC_H_prime')} | {con['yes_without_flags']} | "
                             f"{con['yes_without_tier1_flag']} | {con['no_with_tier1_flag']} |")
            L.append("")
        L += ["### Diagnosis, consistency and red flags", "",
              "| Row | Top-1 % | Top-5 % | D1 Brier | D2 at 60/70/80 | E | CON (YES no flags / YES no tier-1 / NO with tier-1) | "
              "YES on red-flag cases | p_serious present |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in _order(b):
            row = b["rows"][r]
            c, p = row["counts"], row["point"]
            d2 = "/".join(str(c["DX"]["D2"][t]["events"]) for t in ("60", "70", "80"))
            con = c["CON"]
            L.append(f"| {_label(b, r)} | {_v(row, 'top1', 1, True)} | {_v(row, 'top5', 1, True)} | {_f(p['D1_brier'], 3)} | "
                     f"{d2} | {c['DX']['E_events']}/{c['DX']['E_den']} | {con['yes_without_flags']} / "
                     f"{con['yes_without_tier1_flag']} / {con['no_with_tier1_flag']} | "
                     f"{c['red_flag']['yes']}/{c['red_flag']['cases']} | {c['p_serious_present']}/{c['cases']} |")
        L.append("")
        L += v03_anchors.render_set(b)
        memo = {r: b["rows"][r].get("memorisation") for r in b["models"] if b["rows"][r].get("memorisation")}
        if memo:
            L += ["Memorisation flag (section 11, top-1 diagnosis): "
                  + "; ".join(f"{r}: {'flagged (' + ', '.join(m['reasons']) + ')' if m['flag'] else 'not flagged'}"
                              for r, m in memo.items()) + ".", ""]
        runs = [(r, b["rows"][r]["run"]) for r in b["models"] if b["rows"][r].get("run")]
        if runs:
            L += ["### Runs", "", "| Row | Predictions | Errors | Finish reasons | Retried | Truncated | "
                  "Completion tokens (mean / max) | Reasoning tokens | Cost $ | Parse rules fired |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
            for r, u in runs:
                n = max(u["predictions"], 1)
                rules = b["rows"][r]["counts"]["parse_rules"]
                L.append(f"| {r} | {u['predictions']} | {u['errors'] or 0} | {u['finish_reasons']} | {u['retried']} | "
                         f"{u['truncated']} | {u['completion_tokens'] / n:.0f} / {u['completion_tokens_max']} | "
                         f"{u['reasoning_tokens']} | {u['cost_usd']:.2f} | {rules or 'none'} |")
            L.append("")
    L += v03_anchors.render_board(board)
    L += ["## Limits", "", "Every result carries the limits of spec/v0.3-scoring.md section 11: memorisation (naive Bayes "
          "reads the DDXPlus truth at about 98%), a closed world of 49 conditions, one condition per synthetic patient, "
          "DDXPlus probabilities that are not real-world probabilities, provisional tiers, a provisional 7:1 ratio, and "
          "10 patients per condition.", ""]
    return "\n".join(L)
