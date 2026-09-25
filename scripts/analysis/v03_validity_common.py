"""Shared pieces for the v0.3 validity review (docs/v0.3-validity-review.md).

We read the committed key, the two test runs (GPT-5.6 Terra, GPT-OSS 120B) and the
reference rows through the production scorer (evaluator/v03_score.py), so every
number in the review comes from the same parsing and matching as the board. On top of
that we build one per-case table per row, with the case's targets, DXA's differential
and the row's flags resolved to DDXPlus conditions, because the validity questions are
about individual cases: which YES answers name none of the case's targets, which
true-condition targets are missed, and why.

No inference is spent, and no existing file is changed. Outputs go to
results/analysis/v03_validity/. Scripts:
    v03_validity_failures.py     the failure-case audit and the precision of each candidate definition
    v03_validity_gaming.py       reference and gaming rows under each candidate definition
    v03_validity_calibration.py  one scale per candidate: models, references, label-noise range
    v03_validity_prompt.py       what the models' own p_serious says about the YES bar
"""

from __future__ import annotations

import ast
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluator import answer_key_v03 as ak  # noqa: E402
from evaluator import v03_score as vs  # noqa: E402
from evaluator.condition_match import FlagMatcher  # noqa: E402
from inference.run_inference import format_case_for_prompt_v6  # noqa: E402

OUT = ROOT / "results/analysis/v03_validity"
RUNS = ROOT / "results/v03/runs"
MODEL_FILES = {
    "main": {"gpt-5.6-terra": RUNS / "openai-gpt-5.6-terra-v03-470cases.json",
             "gpt-oss-120b": RUNS / "openai-gpt-oss-120b-v03-470cases.json"},
    "pool-atypical": {"gpt-5.6-terra": RUNS / "openai-gpt-5.6-terra-v03-pool-atypical.json",
                      "gpt-oss-120b": RUNS / "openai-gpt-oss-120b-v03-pool-atypical.json"},
    "pool-high-risk": {"gpt-5.6-terra": RUNS / "openai-gpt-5.6-terra-v03-pool-high-risk.json",
                       "gpt-oss-120b": RUNS / "openai-gpt-oss-120b-v03-pool-high-risk.json"},
}
MODELS = ("gpt-5.6-terra", "gpt-oss-120b")
REFS = ("always-yes", "always-no", "dxa", "naive-bayes")

# The four candidate failure definitions of the review's question 1.
CANDIDATES = {
    "H": "H: serious_concern is not YES on a case with an R10 target",
    "H_prime": "H': H, or YES whose flags name no tier-1 condition",
    "H_dprime": "H'': H, or YES whose flags name none of the case's R10 targets",
    "H_truth": "H'' on true-condition targets only: the truth is tier 1 and no flag names it",
}


def ensure_out() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    return OUT


def log_to(name: str):
    """A logger that prints and appends to results/analysis/v03_validity/<name>."""
    path = ensure_out() / name
    path.write_text("")

    def log(msg: str = "") -> None:
        print(msg)
        with open(path, "a") as f:
            f.write(msg + "\n")
    return log


def write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    if not rows:
        path.write_text("")
        return
    cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------- loading


@dataclass
class Loaded:
    name: str
    key: vs.SetKey
    cases: list[dict]
    case_by_id: dict[str, dict]
    matcher: FlagMatcher
    tiers: dict[str, int]
    rows: dict[str, list[dict]]  # row name -> predictions
    outcomes: dict[str, vs.Outcomes] = field(default_factory=dict)
    dxa: dict[str, dict[str, float]] = field(default_factory=dict)  # case_id -> condition -> p (percent)


def load(name: str = "main", models: Iterable[str] = MODELS, refs: Iterable[str] = REFS) -> Loaded:
    key, cases = vs.load_set(name)
    matcher = FlagMatcher()
    tiers = ak.load_tiers()
    rows: dict[str, list[dict]] = {}
    for m in models:
        preds, _ = vs.load_predictions(MODEL_FILES[name][m])
        rows[m] = preds
    ref_rows = vs.reference_rows(vs.SETS[name]["refs"])
    for r in refs:
        rows[r] = ref_rows[r]["predictions"]
    rows["perfect"] = vs.perfect_predictions(key, matcher)
    L = Loaded(name=name, key=key, cases=cases, case_by_id={c["case_id"]: c for c in cases}, matcher=matcher,
               tiers=tiers, rows=rows)
    for r, preds in rows.items():
        L.outcomes[r] = vs.outcomes(preds, key, matcher, policies=("standard", "strict", "lenient"))
    for c in cases:
        dd = c["ddxplus_differential"]
        if isinstance(dd, str):
            dd = ast.literal_eval(dd)
        L.dxa[c["case_id"]] = {cond: 100.0 * p for cond, p in dd}
    return L


# ---------------------------------------------------------------- per-case tables


def dx_conditions(o: vs.Outcomes, i: int, matcher: FlagMatcher, policy: str = "strict") -> list[str]:
    """The DDXPlus conditions the differential's codes name, in differential order."""
    p = o.parsed[i]
    if not p.readable:
        return []
    out: list[str] = []
    for e in p.differential[:5]:
        if not e.code:
            continue
        for cond in matcher.conditions_hit([e.code], policy):
            if cond not in out:
                out.append(cond)
    return out


def flag_conditions(o: vs.Outcomes, i: int, policy: str = "standard") -> set[str]:
    return set(o.hits[policy][i])


def case_table(L: Loaded, row: str) -> list[dict]:
    """One dict per case for one row: the key's groups and targets, DXA's view, and the row's answer."""
    o = L.outcomes[row]
    out = []
    for i, (cid, k) in enumerate(zip(L.key.case_ids, L.key.keys)):
        p = o.parsed[i]
        dxa = L.dxa[cid]
        r10 = [c for c, t in k.considered.items() if t.in_r10]
        r5only = [c for c, t in k.considered.items() if t.in_r5 and not t.in_r10]
        removed = [(c, t.dxa_p) for c, t in k.considered.items() if t.status == ak.RED_HERRING]
        hits_std = flag_conditions(o, i, "standard")
        hits_len = flag_conditions(o, i, "lenient")
        hits_strict = flag_conditions(o, i, "strict")
        dxc = dx_conditions(o, i, L.matcher, "strict")
        dxc_len = dx_conditions(o, i, L.matcher, "lenient")
        t1_flags = {c for c in hits_std if L.tiers.get(c) == 1}
        truth_t = k.considered.get(k.truth)
        top5 = sorted(dxa.items(), key=lambda kv: -kv[1])[:5]
        group = "R10" if k.r10 else ("clearly_low" if k.clearly_low_risk else "intermediate")
        d = {
            "case_id": cid, "row": row, "truth": k.truth, "truth_tier": k.truth_tier, "group": group,
            "truth_dxa_p": round(dxa.get(k.truth, 0.0), 2),
            "truth_dxa_rank": 1 + sorted(dxa.values(), reverse=True).index(dxa[k.truth]) if k.truth in dxa else None,
            "r10_targets": "|".join(r10), "r5_only_targets": "|".join(r5only),
            "removed_ge10": "|".join(f"{c}:{p_:.0f}" for c, p_ in removed if p_ >= 10),
            "removed_5to10": "|".join(f"{c}:{p_:.0f}" for c, p_ in removed if 5 <= p_ < 10),
            "n_removed_ge10": sum(1 for _, p_ in removed if p_ >= 10),
            "max_removed_p": round(max((p_ for _, p_ in removed), default=0.0), 1),
            "red_flag": "|".join(k.red_flag_names),
            "dxa_top5": "|".join(f"{c}:{p_:.0f}" for c, p_ in top5),
            "readable": p.readable, "yes": bool(p.yes),
            "flags": "|".join(p.flags) if p.readable else "",
            "flag_conditions": "|".join(sorted(hits_std)),
            "flag_conditions_lenient": "|".join(sorted(hits_len - hits_std)),
            "tier1_flags": "|".join(sorted(t1_flags)),
            "any_tier1_flag": bool(t1_flags),
            "any_r10_flagged": any(c in hits_std for c in r10),
            "all_r10_flagged": all(c in hits_std for c in r10) if r10 else None,
            "any_r10_flagged_lenient": any(c in hits_len for c in r10),
            "truth_flagged": k.truth in hits_std,
            "truth_flagged_strict": k.truth in hits_strict,
            "truth_flagged_lenient": k.truth in hits_len,
            "truth_in_dx": k.truth in dxc,
            "truth_in_dx_lenient": k.truth in dxc_len,
            "dx_conditions": "|".join(dxc),
            "differential": "|".join(f"{e.code}:{e.p if e.p is not None else ''}" for e in p.differential[:5]) if p.readable else "",
            "p_serious": p.p_serious if p.readable else None,
            "r10_missed": "|".join(c for c in r10 if c not in hits_std),
            "r10_missed_lenient": "|".join(c for c in r10 if c not in hits_len),
        }
        for c in r10:
            t = k.considered[c]
            d.setdefault("r10_detail", "")
            d["r10_detail"] += f"{c}[{t.source},p={t.dxa_p:.1f},rate={t.class_rate if t.class_rate is not None else 'na'},hm={len(t.hallmark_tokens)}];"
        out.append(d)
    return out


def intake_text(L: Loaded, cid: str) -> str:
    """The v6 user prompt's intake block, as the model saw it."""
    txt = format_case_for_prompt_v6(L.case_by_id[cid], "v02")
    return txt.split("\n\nReturn ONLY")[0].replace("PATIENT INTAKE:\n\n", "")


# ---------------------------------------------------------------- candidate definitions per case


def event_masks(L: Loaded, row: str, policy: str = "standard") -> dict[str, np.ndarray]:
    """Per case, whether each candidate definition fires, plus the over-concern event."""
    o = L.outcomes[row]
    key = L.key
    yes = o.yes
    tgt = key.has_r10
    t1 = vs.tier1_hit(o, key, policy)
    any_r10 = np.array([any(c in o.hits[policy][i] for c in k.r10) for i, k in enumerate(key.keys)])
    truth_t1 = key.truth_tier == 1
    truth_flagged = np.array([k.truth in o.hits[policy][i] for i, k in enumerate(key.keys)])
    return {
        "H": tgt & ~yes,
        "H_prime": tgt & (~yes | ~t1),
        "H_dprime": tgt & (~yes | ~any_r10),
        "H_truth": truth_t1 & (~yes | ~truth_flagged),
        "OC": key.clearly_low & yes,
    }


def candidate_cost(L: Loaded, row: str, cand: str, policy: str = "standard", miss: float = vs.MISS,
                   concern: float = vs.CONCERN) -> np.ndarray:
    m = event_masks(L, row, policy)
    return miss * m[cand] + concern * m["OC"]


def sc(L: Loaded, row: str, cand: str, policy: str = "standard") -> float:
    return float(100.0 * candidate_cost(L, row, cand, policy).sum() / L.key.n)


def cluster_ci(L: Loaded, cost: np.ndarray, n_boot: int = 2000, seed: int = vs.BOOTSTRAP_SEED) -> tuple[float, float]:
    """95% interval of 100 x mean cost under the condition-cluster bootstrap of the scorer."""
    M = vs.cluster_draws(L.key.k, n_boot, seed)
    st = vs.Stats(L.key)
    st.add("x", cost, np.ones(L.key.n), 100.0)
    _, draws = st.evaluate(M)
    lo, hi = vs.interval(draws["x"])
    return lo, hi


def fmt(x, d: int = 1) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return f"{x:.{d}f}"
