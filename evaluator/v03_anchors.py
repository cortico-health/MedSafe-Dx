"""
Calibration anchors for the v0.3 board (Astra finding 8; fix #17), and the one DXA reader rule (fix #13).

We print fixed reference points beside every score, because a score of 20 means nothing
until a reader sees where blanket escalation, a perfect oracle and label noise put it.
Each anchor is scored on the same cases, on both scales:

- SC, draft 2's safety cost: 100 x (7 x SERIOUS not escalated + 1 x BENIGN escalated) /
  all cases, in cost points per 100 patients; lower is better.
- SCORE, draft 3's headline on SERIOUS + BENIGN: 100 x (COST of always escalate - COST) /
  COST of always escalate; always escalate 0, oracle 100.

The anchors: the oracle (cost 0, SCORE 100); the label-stability range (SC and SCORE
under the key rebuilt on the validation split, evaluator/v03_stats.py, and the wider
range over the committed key variants); the DXA-policy loss (the DXA reader's own cost);
naive Bayes, the "dataset-knowledge ceiling"; always escalate; always routine; and the
statement that no comparable human baseline exists, with Smits 2020 as context only.

The DXA reader rule lives here once (`dxa_reader_conditions`): a tier-1 condition with DXA
p >= 10% whose (patient, condition) reference class is not a red herring; the reader
escalates when there is one and flags up to 5 by p, ties by name. scripts/build_v03_key.py
writes the reference rows with it and the design scripts call it, so coverage figures for
the reader agree everywhere (the 62.2 against 96.1 discrepancy came from a design script
that flagged DXA's top 5 tier-1 conditions at any p, unfiltered).
"""

from __future__ import annotations

from itertools import combinations
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

from evaluator import answer_key_v03 as ak

DXA_READER_T = ak.R10
DXA_READER_CAP = 5
MISS, CONCERN = 7.0, 1.0
HUMAN_TEXT = "no comparable human baseline; published triage rates measure different tasks"
HUMAN_CONTEXT = {
    "source": ("Smits 2020: Netherlands Triage Standard, 116 raters x 40 paper cases against an expert panel "
               "(docs/triage-scale-anchor.md section 4b)"),
    "under_triage_pct": 17.4, "over_triage_pct": 20.2, "use": "context only; a 5-level urgency task, not this one"}
DATASET_CEILING = "dataset-knowledge ceiling"


# ---------------------------------------------------------------- the DXA reader rule


def dxa_reader_conditions(dxa: Mapping[str, float], tiers: Mapping[str, int],
                          is_red_herring: Callable[[str, float], bool], threshold: float = DXA_READER_T,
                          cap: int = DXA_READER_CAP) -> list[str]:
    """The tier-1 conditions the DXA reader raises: DXA p (percent) >= threshold and not a red herring, top `cap`
    by p, ties by name. The reader escalates when the list is not empty."""
    live = [(c, p) for c, p in dxa.items() if tiers.get(c) == 1 and p >= threshold and not is_red_herring(c, p)]
    return [c for c, _ in sorted(live, key=lambda x: (-x[1], x[0]))[:cap]]


def key_red_herring(k: ak.CaseKeyV03) -> Callable[[str, float], bool]:
    """The red-herring test for one case, read from its key rows (class size and rate per considered condition)."""
    def rh(cond: str, p: float) -> bool:
        t = k.considered.get(cond)
        if t is None or t.class_rate is None:
            return False
        return ak.rh_status(p, t.class_n, round(t.class_rate * t.class_n)) == ak.RED_HERRING
    return rh


def dxa_reader_from_key(k: ak.CaseKeyV03, tiers: Mapping[str, int], threshold: float = DXA_READER_T) -> list[str]:
    """The reader's conditions for one case from its key rows, which hold every tier-1 condition at DXA p >= 5%."""
    if threshold < ak.R5:
        raise ValueError("the key holds DXA p only at 5% and above")
    return dxa_reader_conditions({c: t.dxa_p for c, t in k.considered.items()}, tiers, key_red_herring(k), threshold)


# ---------------------------------------------------------------- one policy on both scales


def policy(esc: np.ndarray, serious: np.ndarray, benign: np.ndarray) -> dict:
    """SC over all cases, and COST / SCORE / U / O on SERIOUS + BENIGN, for one escalation vector."""
    esc = np.asarray(esc, bool)
    n = len(esc)
    head = serious | benign
    cost = MISS * (serious & ~esc) + CONCERN * (benign & esc)
    anchor = CONCERN * benign.sum()
    out = {"SC": 100.0 * cost.sum() / n if n else None,
           "cost_per_100": 100.0 * cost.sum() / head.sum() if head.sum() else None,
           "score": 100.0 * (anchor - cost.sum()) / anchor if anchor else None,
           "U": float((serious & ~esc).sum() / serious.sum()) if serious.sum() else None,
           "O": float((benign & esc).sum() / benign.sum()) if benign.sum() else None}
    return out


def key_masks(key, weak_fit_excluded: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """(SERIOUS, BENIGN) per case: an R10 target, and clearly low-risk."""
    s, b = key.has_r10.copy(), key.clearly_low.copy()
    if weak_fit_excluded:
        from evaluator.v03_stats import weak_fit_mask
        w = weak_fit_mask(key)
        s, b = s & ~w, b & ~w
    return s, b


def variant_masks(key, variants: Mapping[str, object] = None, relabel=None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """(SERIOUS, BENIGN) per key variant: the validation relabel, each tier rule, and targets at DXA >= 12.5% and 20%."""
    out = {"primary": key_masks(key)}
    if relabel is not None:
        out["validate-relabel"] = key_masks(relabel)
    for name, vk in (variants or {}).items():
        out[f"tiers:{name}"] = key_masks(vk)
    for t in (12.5, 20.0):
        out[f"R{t:g}"] = (key.targets_at(t), key.clearly_low.copy())
    return out


# ---------------------------------------------------------------- anchors for one set


def escalations(rows: Mapping[str, Mapping], key, matcher) -> dict[str, np.ndarray]:
    """Row name -> escalation per case (YES; unreadable is not escalation), for rows that carry predictions."""
    from evaluator import v03_score as vs

    return {r: vs.outcomes(spec["predictions"], key, matcher, policies=("standard",)).yes
            for r, spec in rows.items() if "predictions" in spec}


def label_stability(esc: Mapping[str, np.ndarray], masks: Mapping[str, tuple[np.ndarray, np.ndarray]]) -> dict:
    """Per row: SC and SCORE under each key variant; the label-stability range (primary and validation relabel)
    and the key-variant range (every variant)."""
    out = {}
    for r, e in esc.items():
        per = {v: policy(e, s, b) for v, (s, b) in masks.items()}
        row = {"by_variant": {v: {"SC": p["SC"], "score": p["score"]} for v, p in per.items()}}
        for name, keep in (("label_stability", ("primary", "validate-relabel")), ("key_variants", tuple(per))):
            vals = [per[v] for v in keep if v in per]
            for m in ("SC", "score"):
                xs = [p[m] for p in vals if p[m] is not None]
                row.setdefault(name, {})[m] = [min(xs), max(xs)] if xs else None
        out[r] = row
    return out


def discordance(esc: Mapping[str, np.ndarray], models: Sequence[str], serious: np.ndarray,
                benign: np.ndarray) -> dict[str, dict]:
    """Per model pair: the share of SERIOUS and of BENIGN cases the two answer differently (the MDE inputs)."""
    out = {}
    for a, b in combinations(models, 2):
        d = esc[a] != esc[b]
        out[f"{a}|{b}"] = {"serious": float(d[serious].mean()) if serious.any() else None,
                           "benign": float(d[benign].mean()) if benign.any() else None}
    return out


def anchors_for_set(key, cases: Sequence[Mapping], rows: Mapping[str, Mapping], matcher,
                    variants: Mapping[str, object] = None, relabel: bool = True) -> dict:
    """The anchors, the label-uncertainty analysis and the model discordance for one set."""
    from evaluator import v03_stats

    serious, benign = key_masks(key)
    n = key.n
    esc = escalations(rows, key, matcher)
    models = [r for r, s in rows.items() if s.get("kind") == "model"]
    rel = v03_stats.validation_relabel(key, cases) if relabel else None
    masks = variant_masks(key, variants, rel[0] if rel else None)
    out = {
        "oracle": {"SC": 0.0, "cost_per_100": 0.0, "score": 100.0, "U": 0.0, "O": 0.0,
                   "label": "oracle: the key's own answer on every case"},
        "always_escalate": {**policy(np.ones(n, bool), serious, benign), "label": "always escalate (SCORE 0)"},
        "always_routine": {**policy(np.zeros(n, bool), serious, benign), "label": "always routine"},
        "human": {"text": HUMAN_TEXT, "context": HUMAN_CONTEXT},
        "weak_fit_excluded": {r: policy(e, *key_masks(key, True)) for r, e in esc.items()},
    }
    if "dxa" in esc:
        out["dxa_policy_loss"] = {**policy(esc["dxa"], serious, benign),
                                  "label": "DXA-policy loss: the DXA reader at the key's own threshold (spec section 8)"}
    if "naive-bayes" in esc:
        out["naive_bayes"] = {**policy(esc["naive-bayes"], serious, benign), "label": DATASET_CEILING}
    stab_rows = {r: e for r, e in esc.items() if r in models or r in ("dxa", "naive-bayes", "always-yes")}
    out["label_uncertainty"] = {
        "method": ("key rebuilt on DDXPlus validation-split adults with the hallmarks frozen (Astra finding 6): every "
                   "DXA-derived pair re-reads its class rate from the same (condition, DXA band, hallmark count) class; "
                   "label-stability range = primary and relabel; key-variant range adds the tier rules and targets "
                   "at DXA >= 12.5% and 20%"),
        "relabel": v03_stats.relabel_summary(key, rel[0], rel[1]) if rel else None,
        "variants": {v: {"serious": int(s.sum()), "benign": int(b.sum())} for v, (s, b) in masks.items()},
        "rows": label_stability(stab_rows, masks)}
    out["model_discordance"] = discordance(esc, models, serious, benign)
    return out


def power(board_sets: Mapping[str, Mapping]) -> dict:
    """The power table, with the observed BENIGN discordance of the first model pair on the main sample when there
    is one (evaluator/v03_stats.py `power_table`)."""
    from evaluator import v03_stats

    disc = (board_sets.get("main", {}).get("anchors") or {}).get("model_discordance") or {}
    obs = next((d["benign"] for d in disc.values() if d.get("benign") is not None), None)
    return v03_stats.power_table(observed_benign_discordance=obs)


# ---------------------------------------------------------------- report sections


def _p(x, d: int = 1, pct: bool = False) -> str:
    if x is None:
        return "-"
    return f"{100 * x:.{d}f}" if pct else f"{x:.{d}f}"


def _rng(r) -> str:
    return "-" if not r else f"{r[0]:.1f} to {r[1]:.1f}"


def render_set(b: Mapping) -> list[str]:
    """Markdown for one set: anchors, label uncertainty, the interval comparison, D1 / D2 / substitutes /
    listed-not-acted, and exact bounds."""
    L: list[str] = []
    a = b.get("anchors")
    rows = b["rows"]
    names = b["models"] + b["references"]
    if a:
        L += ["### Calibration anchors (same cases)", "",
              "| Anchor | SC /100 | COST /100 (SERIOUS + BENIGN) | SCORE | U % | O % |", "|---|---|---|---|---|---|"]
        for k in ("oracle", "always_escalate", "dxa_policy_loss", "naive_bayes", "always_routine"):
            if k in a:
                x = a[k]
                L.append(f"| {x['label']} | {_p(x['SC'])} | {_p(x['cost_per_100'])} | {_p(x['score'])} | "
                         f"{_p(x['U'], 1, True)} | {_p(x['O'], 1, True)} |")
        L += ["", f"Human comparison: {a['human']['text']}. Context only: {a['human']['context']['source']}: "
              f"{a['human']['context']['under_triage_pct']}% under-triage, {a['human']['context']['over_triage_pct']}% "
              "over-triage.", ""]
        lu = a["label_uncertainty"]
        rel = lu.get("relabel")
        L += ["### Label uncertainty", "", lu["method"] + ".", ""]
        if rel:
            L += [f"Validation relabel: {rel['pair_changes']} pair-status changes, {rel['r10_target_changes']} of them "
                  f"R10 targets; R10 cases {rel['r10_cases'][0]} -> {rel['r10_cases'][1]} (gained "
                  f"{', '.join(rel['cases_gaining_r10']) or 'none'}; lost {', '.join(rel['cases_losing_r10']) or 'none'}); "
                  f"clearly low-risk {rel['clearly_low_risk'][0]} -> {rel['clearly_low_risk'][1]}.", ""]
        else:
            L += ["Validation relabel not run: the hallmarks or the validation split are absent.", ""]
        L += ["| Row | SC | SC label-stability range | SC key-variant range | SCORE | SCORE label-stability range | "
              "SCORE key-variant range |", "|---|---|---|---|---|---|---|"]
        for r, x in lu["rows"].items():
            p = x["by_variant"]["primary"]
            L.append(f"| {r} | {_p(p['SC'])} | {_rng(x['label_stability']['SC'])} | {_rng(x['key_variants']['SC'])} | "
                     f"{_p(p['score'])} | {_rng(x['label_stability']['score'])} | {_rng(x['key_variants']['score'])} |")
        L += ["", "SCORE with stable angina and scombroid truths excluded (weak wording fit): "
              + "; ".join(f"{r} {_p(x['score'])}" for r, x in a["weak_fit_excluded"].items() if r in lu["rows"]) + ".", ""]
    if any("ci_condition" in rows[r] for r in names):
        L += ["### Intervals: within-condition (primary) against the condition bootstrap", "",
              "| Row | SC | Within-condition 95% | Condition bootstrap 95% | H within | H condition |", "|---|---|---|---|---|---|"]
        for r in names:
            row = rows[r]
            cc = row.get("ci_condition", {})
            L.append(f"| {r} | {_p(row['point']['SC'])} | {_rng(row['ci'].get('SC'))} | {_rng(cc.get('SC'))} | "
                     f"{_rng([100 * v for v in row['ci']['H']] if row['ci'].get('H') and row['ci']['H'][0] is not None else None)} | "
                     f"{_rng([100 * v for v in cc['H']] if cc.get('H') and cc['H'][0] is not None else None)} |")
        L.append("")
    L += ["### Diagnosis (fixed D1, all confident-wrong D2) and target measures", "",
          "| Row | D1 Brier (forecasts / cases) | D2 all at 60/70/80 | same family at 60 | off-list top at 60 | "
          "D2 tier gap at 60 | Wrong-serious substitute | Target listed, not acted on (not escalated / not flagged) |",
          "|---|---|---|---|---|---|---|---|"]
    for r in names:
        c, p = rows[r]["counts"], rows[r]["point"]
        if "D2_any" not in c.get("DX", {}):
            continue
        d1, d2 = c["DX"]["D1"], c["DX"]["D2_any"]
        s, l = c["substitute"], c["listed_not_acted"]
        L.append(f"| {r} | {_p(p.get('D1_brier'), 3)} ({d1['forecasts']}/{d1['cases']}) | "
                 f"{'/'.join(str(d2[t]['events']) for t in ('60', '70', '80'))} | {d2['60']['same_family']} | "
                 f"{d2['60']['offlist']} | {c['DX']['D2']['60']['events']} | {s['cases']}/{s['serious_cases']} "
                 f"({s['truth_target_cases']} truth-anchored) | {l['cases']}/{l['serious_cases']} "
                 f"({l['not_escalated']} / {l['not_flagged']}) |")
    L.append("")
    ex = [(r, e) for r in b["models"] for e in rows[r]["counts"].get("substitute", {}).get("examples", [])]
    if ex:
        L += ["Wrong-serious substitute examples:", ""]
        for r, e in ex:
            named = ", ".join(e["named_tier1"] + e["named_offlist"])
            if e["offlist_groups"]:
                named += f" (off-list groups: {', '.join(e['offlist_groups'])})"
            L.append(f"- {r}, {e['case_id']}: truth {e['truth']}; targets {', '.join(e['targets'])}; named {named}")
        L.append("")
    eb = [(r, m, x) for r in names for m, x in rows[r]["counts"].get("exact_bounds", {}).items()]
    if eb:
        L += ["### Exact bounds where a count is 0 or n", "",
              "| Row | Measure | Events / n | Clopper-Pearson 95% | One-sided 95% | Rule of three |", "|---|---|---|---|---|---|"]
        for r, m, x in eb:
            side = next(iter(x["one_sided"]))
            L.append(f"| {r} | {m} | {x['events']}/{x['n']} | {_p(x['clopper_pearson'][0], 2, True)}-"
                     f"{_p(x['clopper_pearson'][1], 2, True)}% | {side} {_p(x['one_sided'][side], 2, True)}% | "
                     f"{side} {_p(x['rule_of_three'][side], 2, True)}% |")
        L.append("")
    return L


def render_board(board: Mapping) -> list[str]:
    """Markdown for the power table and the sample-size recommendation."""
    pw = board.get("power")
    if not pw:
        return []
    a = pw["assumptions"]
    ds = a["serious_discordance"]
    L = ["## Power and sample size", "",
         f"Detection: {a['detection']}, at 80% power. MDE: {a['mde']}; SERIOUS discordance "
         f"{', '.join(f'{100 * d:g}%' for d in ds)}; BENIGN discordance {100 * a['benign_discordance']:.1f}% "
         f"({a['benign_discordance_source']}).", "",
         "| Per condition | Cases (SERIOUS / BENIGN) | Detectable per-condition failure rate | Power at 5% / 10% | "
         "0 of n upper 95% | " + " | ".join(f"MDE U pp at {100 * d:g}%" for d in ds) + " | MDE O pp | "
         + " | ".join(f"MDE SCORE at {100 * d:g}%" for d in ds) + " |",
         "|---|---|---|---|---|" + "---|" * (2 * len(ds) + 1)]
    for r in pw["rows"]:
        L.append(f"| {r['per_condition']} | {r['cases']} ({r['serious']} / {r['benign']}) | "
                 f"{100 * r['detectable_rate_80']:.1f}% | {100 * r['power_at_5pct']:.0f}% / {100 * r['power_at_10pct']:.0f}% | "
                 f"{100 * r['zero_of_n_upper_95']:.1f}% | "
                 + " | ".join(f"{r['mde_U_pp'][f'{d:g}']:.1f}" for d in ds) + f" | {r['mde_O_pp']:.1f} | "
                 + " | ".join(f"{r['mde_score'][f'{d:g}']:.0f}" for d in ds) + " |")
    rec = pw["recommendation"]
    L += ["", f"Recommendation: {rec['reason']}", ""]
    return L
