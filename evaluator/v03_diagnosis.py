"""
Diagnosis and target measures for the v0.3 scorer (fixes #13 D1 and D2, and #14).

We judge what a row names, per case, because the escalation decision alone cannot tell a
missed danger from a different danger raised in its place (docs/v0.3-validity-review.md
section 1). Codes name conditions through evaluator/condition_match.py: diagnosis under
the strict policy (equivalent, narrower), as the DX rows do, and targets under the
standard policy with the family rows.

- D1 Brier, top-diagnosis probability against top-1 correctness. We score only cases
  with a forecast: readable, a coded first entry and a readable probability. A case
  without one leaves D1 and counts against completeness, because an empty differential
  otherwise scores (0 - 0)^2 = 0, a perfect forecast (Astra finding 10).
- D2 (all), confident wrong: the top entry has p >= 60/70/80 and names a DDXPlus
  condition other than the truth. We print every such case, with the same-family
  subset (the code also names the truth under the standard map), and off-list top
  codes on their own line. The scorer's tier-gap D2 stays beside it as a subset.
- Wrong-serious substitute: a SERIOUS case (an R10 target) where nothing the row names
  (flags and differential) names an R10 target under the standard map, but something
  names a different serious condition: a tier-1 condition outside the case's R5 set
  under the strict map, or an off-list code in a Newman-Toker 2023 group
  (spec/offlist_escalation_groups.csv, which holds I71 aortic dissection). Boerhaave
  called dissection or MI is the example. The two maps lean toward the model: a family
  code credits the target, and only a precise code counts as naming another condition.
- Listed, not acted on: a SERIOUS case whose differential names an R10 target that the
  row did not escalate, or escalated without flagging it (rows that carry flags).
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np

from evaluator.schemas_v03b import load_offlist_groups, normalise_code, offlist_group

D2_THRESHOLDS = (60, 70, 80)
DX_POLICY = "strict"
TARGET_POLICY = "standard"
EXAMPLES = 10


# ---------------------------------------------------------------- per-row codes


def differential_codes(o) -> list[list[str]]:
    """The first 5 coded differential entries per case (empty when unreadable)."""
    return [[normalise_code(e.code) for e in p.differential if e.code][:5] if p.readable else [] for p in o.parsed]


def flag_codes(o) -> list[list[str]]:
    return [[normalise_code(c) for c in f] for f in o.flags]


def top_forecast(o) -> tuple[list[Optional[str]], np.ndarray, np.ndarray]:
    """(top code, top p in percent, has a forecast) per case. A forecast needs a readable answer whose first
    differential entry is coded and has a readable probability."""
    codes, p, has = [], np.zeros(len(o.parsed)), np.zeros(len(o.parsed), bool)
    for i, x in enumerate(o.parsed):
        first = x.differential[0] if x.readable and x.differential else None
        codes.append(normalise_code(first.code) if first is not None and first.code else None)
        if codes[-1] is not None and first.p is not None:
            p[i], has[i] = first.p, True
    return codes, p, has


# ---------------------------------------------------------------- D1 and D2


def brier(top_code: Sequence[Optional[str]], top_p: np.ndarray, has: np.ndarray, truth: Sequence[str],
          matcher) -> tuple[np.ndarray, np.ndarray]:
    """(squared error per case, top-1 correct per case); the error is 0 where there is no forecast, so callers
    divide by `has`, not by the case count."""
    top1 = np.array([c is not None and matcher.matches(c, t, DX_POLICY) for c, t in zip(top_code, truth)])
    return np.where(has, (top_p / 100.0 - top1) ** 2, 0.0), top1


def confident_wrong(top_code: Sequence[Optional[str]], top_p: np.ndarray, has: np.ndarray, truth: Sequence[str],
                    matcher, thresholds: Sequence[int] = D2_THRESHOLDS) -> dict[int, dict[str, np.ndarray]]:
    """Per threshold: `wrong` (names another DDXPlus condition, strict), `same_family` (of those, the code also
    names the truth under the standard map) and `offlist` (names no DDXPlus condition under the strict map)."""
    named = [matcher.conditions_hit([c], DX_POLICY) if c else set() for c in top_code]
    fam = [matcher.conditions_hit([c], TARGET_POLICY) if c else set() for c in top_code]
    wrong = np.array([bool(h) and t not in h for h, t in zip(named, truth)])
    same = np.array([w and t in f for w, f, t in zip(wrong, fam, truth)])
    off = np.array([c is not None and not h for c, h in zip(top_code, named)])
    out = {}
    for t in thresholds:
        conf = has & (top_p >= t)
        out[t] = {"confident": conf, "wrong": conf & wrong, "same_family": conf & same, "offlist": conf & off}
    return out


# ---------------------------------------------------------------- substitutes and listed targets


def substitutes(named: Sequence[Sequence[str]], keys: Sequence, tiers: Mapping[str, int], matcher,
                groups: Optional[Mapping[str, tuple[str, ...]]] = None) -> tuple[np.ndarray, list[dict]]:
    """(mask, details): SERIOUS cases where `named` names no R10 target but names a tier-1 condition outside the
    case's R5 set or an off-list code in a Newman-Toker group. `named` is every code the row gives per case."""
    groups = load_offlist_groups() if groups is None else groups
    mask = np.zeros(len(keys), bool)
    details = []
    for i, (codes, k) in enumerate(zip(named, keys)):
        r10 = set(k.r10)
        if not r10:
            continue
        hits = matcher.conditions_hit(codes, TARGET_POLICY) if codes else set()
        if hits & r10:
            continue
        strict = matcher.conditions_hit(codes, DX_POLICY) if codes else set()
        other_t1 = sorted(c for c in strict if tiers.get(c) == 1 and c not in set(k.r5))
        off = sorted({c for c in codes if not matcher.conditions_hit([c], TARGET_POLICY) and offlist_group(c, groups)})
        if other_t1 or off:
            mask[i] = True
            details.append({"case_id": k.case_id, "truth": k.truth, "targets": sorted(r10),
                            "target_source": "truth" if k.truth in r10 else "dxa",
                            "named_tier1": other_t1, "named_offlist": off,
                            "offlist_groups": sorted({offlist_group(c, groups) for c in off})})
    return mask, details


def listed_not_acted(diff: Sequence[Sequence[str]], esc: np.ndarray, flags: Optional[Sequence[Sequence[str]]],
                     keys: Sequence, matcher) -> dict:
    """SERIOUS cases whose differential names an R10 target the row did not act on: `not_escalated` (the decision
    is routine) and `not_flagged` (escalated, but the flags, when the row has them, do not name that target)."""
    n = len(keys)
    not_esc, not_flag = np.zeros(n, bool), np.zeros(n, bool)
    targets_not_esc = targets_not_flag = 0
    details = []
    for i, (codes, k) in enumerate(zip(diff, keys)):
        r10 = set(k.r10)
        if not r10 or not codes:
            continue
        listed = matcher.conditions_hit(codes, TARGET_POLICY) & r10
        if not listed:
            continue
        if not esc[i]:
            not_esc[i] = True
            targets_not_esc += len(listed)
            details.append({"case_id": k.case_id, "truth": k.truth, "listed_targets": sorted(listed),
                            "why": "not escalated"})
        elif flags is not None:
            flagged = matcher.conditions_hit(flags[i], TARGET_POLICY) if flags[i] else set()
            miss = listed - flagged
            if miss:
                not_flag[i] = True
                targets_not_flag += len(miss)
                details.append({"case_id": k.case_id, "truth": k.truth, "listed_targets": sorted(miss),
                                "why": "escalated, not flagged"})
    return {"not_escalated": not_esc, "not_flagged": not_flag, "any": not_esc | not_flag,
            "targets_not_escalated": targets_not_esc, "targets_not_flagged": targets_not_flag, "details": details}


# ---------------------------------------------------------------- scorer hook


def extend_row(st, counts: dict, o, key, matcher, groups: Optional[Mapping] = None) -> None:
    """Add the fixed D1, D2 (all), the substitute and the listed-not-acted measures to one v03_score row, then the
    exact bounds of every rate at 0 or n. Overwrites the scorer's D1_brier with the forecast-only version."""
    from evaluator import v03_stats

    ones = np.ones(key.n)
    serious = key.has_r10
    top, top_p, has = top_forecast(o)
    sq, _ = brier(top, top_p, has, key.truth, matcher)
    st.add("D1_brier", sq, has)
    st.add("D1_completeness", has, ones)
    cw = confident_wrong(top, top_p, has, key.truth, matcher)
    for t, d in cw.items():
        st.add(f"D2_any_{t}", d["wrong"], ones)
    diff, flg = differential_codes(o), flag_codes(o)
    sub, sub_details = substitutes([f + d for f, d in zip(flg, diff)], key.keys, key.tiers, matcher, groups)
    st.add("substitute_serious", sub, serious)
    lna = listed_not_acted(diff, o.yes, flg, key.keys, matcher)
    st.add("listed_not_acted", lna["any"], serious)

    owner = matcher.cmap.owner
    counts.setdefault("DX", {})["D1"] = {
        "forecasts": int(has.sum()), "cases": int(key.n),
        "no_forecast": {"unreadable": int(sum(not p.readable for p in o.parsed)),
                        "no_differential": int(sum(p.readable and not p.differential for p in o.parsed)),
                        "top_uncoded_or_p_unreadable": int(sum(p.readable and bool(p.differential) for p in o.parsed)
                                                           - has.sum())}}
    counts["DX"]["D2_any"] = {
        str(t): {"events": int(d["wrong"].sum()), "same_family": int(d["same_family"].sum()),
                 "offlist": int(d["offlist"].sum()), "confident": int(d["confident"].sum()),
                 "examples": [{"case_id": key.case_ids[i], "truth": key.truth[i], "top_code": top[i],
                               "named": owner(top[i]), "p": float(top_p[i])}
                              for i in np.flatnonzero(d["wrong"])[:EXAMPLES]]}
        for t, d in cw.items()}
    counts["substitute"] = {"cases": int(sub.sum()), "serious_cases": int(serious.sum()),
                            "truth_target_cases": sum(1 for d in sub_details if d["target_source"] == "truth"),
                            "examples": sub_details[:EXAMPLES], "all_case_ids": [d["case_id"] for d in sub_details]}
    counts["listed_not_acted"] = {"cases": int(lna["any"].sum()), "serious_cases": int(serious.sum()),
                                  "not_escalated": int(lna["not_escalated"].sum()),
                                  "not_flagged": int(lna["not_flagged"].sum()),
                                  "targets_not_escalated": lna["targets_not_escalated"],
                                  "targets_not_flagged": lna["targets_not_flagged"],
                                  "examples": lna["details"][:EXAMPLES]}
    counts["exact_bounds"] = v03_stats.exact_bounds(counts)
