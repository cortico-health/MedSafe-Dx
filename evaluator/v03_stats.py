"""
Statistics for the v0.3 scorer (Astra finding 6, docs/v0.3-astra-review.md; fix #12, #15).

We add four things to the scorer, because the condition bootstrap alone answers a question
the sample does not ask and says nothing when a count sits at 0 or n:

1. Within-condition resampling, the primary interval. The estimand is the rate on the
   fixed mix of 10 patients per condition (spec section 1), so each draw resamples the
   patients within every condition with replacement and keeps each condition's count.
   The condition bootstrap (evaluator/v03_score.py `cluster_draws`) stays as the
   superpopulation sensitivity: it also varies the mix of conditions.
   `case_level_key` turns a SetKey into one cluster per case, so the scorer's `Stats`
   evaluates the same measures on case-level weights (`within_condition_draws`).
2. Exact bounds when a count is 0 or equals n: the two-sided Clopper-Pearson interval,
   the one-sided 95% bound, and the rule of three (3 / n), because a bootstrap of 0
   events returns [0, 0].
3. Label uncertainty: the key rebuilt on the DDXPlus validation split with the hallmarks
   frozen (`validation_relabel`), as Astra finding 6 did. Every DXA-derived (case,
   condition) pair re-reads its class rate from validation-split adults with DXA p >= 5%
   in the same (condition, band, hallmark count) class; the true condition stays a
   target. evaluator/v03_anchors.py reports the score range across key variants.
4. A power table (`power_table`): the per-condition failure rate a sample detects with
   80% power, and the minimum detectable effect (MDE) of a paired model difference.

It also holds the weak-fit sensitivity row (#15): stable angina and scombroid truths
leave the scored cases (spec section 11, "Wording fit").
"""

from __future__ import annotations

import ast
import csv
import dataclasses
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from evaluator import answer_key_v03 as ak

ROOT = Path(__file__).resolve().parent.parent
VALIDATE_CSV = ROOT / "data" / "ddxplus_v0" / "release_validate_patients"
HALLMARKS_CSV = ROOT / "results" / "analysis" / "v03_key" / "hallmarks.csv"
VALIDATE_CLASSES_CACHE = ROOT / "data" / "test_sets" / "eval-v03-validate-classes.csv"

LEVEL = 0.95
POWER = 0.80
Z_ALPHA = 1.959963984540054  # two-sided 5%
Z_BETA = 0.8416212335729143  # 80% power
WEAK_FIT = ("Stable angina", "Scombroid food poisoning")
PRIMARY_INTERVAL = "within-condition"
INTERVAL_NOTE = ("95% intervals resample patients within each true condition (the fixed 10-per-condition mix is the "
                 "estimand); the condition bootstrap, which also varies the mix, is printed as the superpopulation "
                 "sensitivity. Counts of 0 or n carry exact Clopper-Pearson bounds.")


# ---------------------------------------------------------------- within-condition resampling


def within_condition_draws(cond_idx: Sequence[int], n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, cases) multiplicities: each draw resamples every condition's cases with replacement, keeping its count."""
    cond_idx = np.asarray(cond_idx)
    rng = np.random.default_rng(seed)
    W = np.zeros((n_boot, len(cond_idx)))
    for c in np.unique(cond_idx):
        idx = np.flatnonzero(cond_idx == c)
        W[:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1.0 / len(idx)), size=n_boot)
    return W


def case_level_key(key):
    """The same SetKey with one cluster per case, so `Stats` sums per case and takes case-level weights."""
    return dataclasses.replace(key, cond_idx=np.arange(key.n), conditions=list(key.case_ids))


def case_level_mixes(key, mixes: Optional[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Per-condition mix weights spread to the cases, for a case-level key."""
    return {m: np.asarray(w)[key.cond_idx] for m, w in (mixes or {}).items()}


def within_setup(key, mixes, n_boot: int, seed: int):
    """(case-level draws, case-level key, case-level mixes) for one set."""
    return within_condition_draws(key.cond_idx, n_boot, seed), case_level_key(key), case_level_mixes(key, mixes)


# ---------------------------------------------------------------- exact bounds


def _log_pmf(i: int, n: int, p: float) -> float:
    return math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * math.log(p) + (n - i) * math.log1p(-p)


def binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p)."""
    if k < 0:
        return 0.0
    if k >= n or p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    return min(1.0, sum(math.exp(_log_pmf(i, n, p)) for i in range(k + 1)))


def _bisect(f, lo: float = 0.0, hi: float = 1.0, it: int = 200) -> float:
    """Root of an increasing f on [lo, hi]."""
    for _ in range(it):
        mid = (lo + hi) / 2
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def clopper_pearson(k: int, n: int, level: float = LEVEL) -> tuple[float, float]:
    """Exact two-sided interval for k events in n."""
    if n <= 0:
        raise ValueError("n must be positive")
    a = (1 - level) / 2
    lo = 0.0 if k == 0 else _bisect(lambda p: (1 - binom_cdf(k - 1, n, p)) - a)
    hi = 1.0 if k == n else _bisect(lambda p: a - binom_cdf(k, n, p))
    return lo, hi


def one_sided_bound(k: int, n: int, level: float = LEVEL) -> float:
    """Exact one-sided bound at a boundary: the upper bound for 0 events (1 - alpha^(1/n)), the lower for n of n."""
    a = 1 - level
    if k == 0:
        return 1 - a ** (1 / n)
    if k == n:
        return a ** (1 / n)
    raise ValueError("one-sided bound is defined here for 0 or n events only")


def boundary_bounds(k: int, n: int, level: float = LEVEL) -> Optional[dict]:
    """Exact bounds when k is 0 or n (None otherwise, or when n is 0)."""
    if n <= 0 or k not in (0, n):
        return None
    lo, hi = clopper_pearson(k, n, level)
    side = "upper" if k == 0 else "lower"
    r3 = 3.0 / n
    return {"events": int(k), "n": int(n), "clopper_pearson": [lo, hi], "one_sided": {side: one_sided_bound(k, n, level)},
            "rule_of_three": {side: r3 if k == 0 else 1 - r3}}


# measure -> (events, denominator) paths in a v03_score row's counts
BOUNDED_RATES = {
    "H": ("misses", "r10_cases"),
    "OC": ("concerns", "clearly_low_risk"),
    "E": ("DX.E_events", "DX.E_den"),
    "unreadable_share": ("unreadable", "cases"),
    "substitute_serious": ("substitute.cases", "substitute.serious_cases"),
    "listed_not_acted": ("listed_not_acted.cases", "listed_not_acted.serious_cases"),
}


def _get(d: Mapping, path: str):
    for p in path.split("."):
        if not isinstance(d, Mapping) or p not in d:
            return None
        d = d[p]
    return d


def exact_bounds(counts: Mapping, rates: Mapping[str, tuple[str, str]] = BOUNDED_RATES) -> dict:
    """Exact bounds for every rate in `rates` whose count sits at 0 or n."""
    out = {}
    for m, (kp, np_) in rates.items():
        k, n = _get(counts, kp), _get(counts, np_)
        if k is None or n is None:
            continue
        b = boundary_bounds(int(k), int(n))
        if b:
            out[m] = b
    return out


# ---------------------------------------------------------------- weak-fit exclusion (#15)


def weak_fit_mask(key, conditions: Iterable[str] = WEAK_FIT) -> np.ndarray:
    """Cases whose true condition is stable angina or scombroid."""
    return np.isin(key.truth, list(conditions))


def weak_fit_row(yes: np.ndarray, key, concern_mask: np.ndarray, miss: float = 7.0, concern: float = 1.0):
    """(name, description, ratio args) for the scorer's sensitivity table: the weak-fit truths leave the cases."""
    keep = ~weak_fit_mask(key)
    cost = miss * (key.has_r10 & ~yes) + concern * (concern_mask & yes)
    return ("weak_fit_excluded", "stable angina and scombroid truths excluded (weak wording fit)",
            (cost * keep, keep.astype(float), 100.0))


# ---------------------------------------------------------------- label uncertainty (#12)


def load_hallmarks(path: Path = HALLMARKS_CSV) -> dict[str, list[str]]:
    """condition -> its frozen hallmark tokens, from the key build."""
    hm: dict[str, list[str]] = defaultdict(list)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            hm[r["condition"]].append(r["token"])
    return dict(hm)


def count_validate_classes(rows: Iterable[Mapping], tier1: Iterable[str], hallmarks: Mapping[str, list[str]],
                           min_age: int = 18) -> dict[tuple[str, int, int], tuple[int, int]]:
    """(condition, band, hallmark count) -> (n, k) over adults with the tier-1 condition at DXA p >= 5%.
    `rows` are DDXPlus release rows (AGE, PATHOLOGY, EVIDENCES, DIFFERENTIAL_DIAGNOSIS)."""
    n: Counter = Counter()
    k: Counter = Counter()
    t1 = set(tier1)
    for r in rows:
        if int(r["AGE"]) < min_age:
            continue
        ev = r["EVIDENCES"]
        ev = frozenset(ast.literal_eval(ev) if isinstance(ev, str) else ev)
        dd = r["DIFFERENTIAL_DIAGNOSIS"]
        dd = ast.literal_eval(dd) if isinstance(dd, str) else dd
        for cond, p in dd:
            if cond not in t1 or p < ak.R5 / 100:
                continue
            key = (cond, ak.band_of(100 * p), sum(1 for t in hallmarks.get(cond, []) if t in ev))
            n[key] += 1
            k[key] += r["PATHOLOGY"] == cond
    return {c: (n[c], k[c]) for c in n}


def validate_classes(tier1: Iterable[str], hallmarks: Mapping[str, list[str]], validate: Path = VALIDATE_CSV,
                     cache: Optional[Path] = VALIDATE_CLASSES_CACHE) -> dict[tuple[str, int, int], tuple[int, int]]:
    """The validation-split classes, read from `cache` when present, else counted and cached."""
    if cache is not None and Path(cache).exists():
        with open(cache, newline="") as f:
            return {(r["condition"], int(r["band"]), int(r["hallmarks"])): (int(r["n"]), int(r["k"]))
                    for r in csv.DictReader(f)}
    with open(validate, newline="") as f:
        out = count_validate_classes(csv.DictReader(f), tier1, hallmarks)
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["condition", "band", "hallmarks", "n", "k"])
            for (c, b, h), (nn, kk) in sorted(out.items()):
                w.writerow([c, b, h, nn, kk])
    return out


def relabel_keys(keys: Sequence[ak.CaseKeyV03], evidence: Mapping[str, Iterable[str]],
                 classes: Mapping[tuple[str, int, int], tuple[int, int]],
                 hallmarks: Mapping[str, list[str]]) -> tuple[dict[str, ak.CaseKeyV03], list[dict]]:
    """Each case's key with every DXA-derived pair's status re-read from `classes`, and the pairs that changed."""
    import copy

    out, changes = {}, []
    for k in keys:
        ev = frozenset(evidence[k.case_id])
        nk = copy.deepcopy(k)
        for cond, t in nk.considered.items():
            if t.source != "dxa":
                continue
            h = sum(1 for tok in hallmarks.get(cond, []) if tok in ev)
            n, kk = classes.get((cond, ak.band_of(t.dxa_p), h), (0, 0))
            status = ak.rh_status(t.dxa_p, n, kk)
            in_r5 = status != ak.RED_HERRING
            in_r10 = in_r5 and t.dxa_p >= ak.R10
            if (in_r10, in_r5, status) != (t.in_r10, t.in_r5, t.status):
                changes.append({"case_id": k.case_id, "truth": k.truth, "condition": cond, "dxa_p": round(t.dxa_p, 1),
                                "hallmarks": h, "test_status": t.status, "test_n": t.class_n, "test_rate": t.class_rate,
                                "validate_status": status, "validate_n": n,
                                "validate_rate": round(kk / n, 4) if n else None,
                                "r10_before": t.in_r10, "r10_after": in_r10, "r5_before": t.in_r5, "r5_after": in_r5})
            t.status, t.undetermined, t.in_r10, t.in_r5 = status, status == ak.UNDETERMINED, in_r10, in_r5
            t.class_n, t.class_rate = n, (kk / n if n else None)
        nk.clearly_low_risk = not nk.r5 and nk.truth_tier == 3 and not nk.red_flag
        nk.intermediate = not nk.r10 and not nk.clearly_low_risk
        out[k.case_id] = nk
    return out, changes


def validation_relabel(key, cases: Sequence[Mapping], hallmarks_path: Path = HALLMARKS_CSV,
                       validate: Path = VALIDATE_CSV, cache: Optional[Path] = VALIDATE_CLASSES_CACHE):
    """(relabelled SetKey, changes), or None when the hallmarks or the validation split are absent."""
    from evaluator import v03_score as vs

    if not Path(hallmarks_path).exists() or not (Path(validate).exists() or (cache and Path(cache).exists())):
        return None
    hm = load_hallmarks(hallmarks_path)
    classes = validate_classes(ak.tier1_conditions(key.tiers), hm, validate, cache)
    evidence = {c["case_id"]: c["presenting_symptoms"] for c in cases}
    keys, changes = relabel_keys(key.keys, evidence, classes, hm)
    return vs.set_key(f"{key.name}:validate-relabel", key.case_ids, keys, tiers=key.tiers), changes


def relabel_summary(key, vkey, changes: Sequence[Mapping]) -> dict:
    """Counts for the report: pair changes, and the R10 / clearly-low-risk cases gained and lost."""
    return {"pair_changes": len(changes),
            "r10_target_changes": sum(1 for c in changes if c["r10_before"] != c["r10_after"]),
            "r10_cases": [int(key.has_r10.sum()), int(vkey.has_r10.sum())],
            "clearly_low_risk": [int(key.clearly_low.sum()), int(vkey.clearly_low.sum())],
            "cases_gaining_r10": sorted(c for c, a, b in zip(key.case_ids, key.has_r10, vkey.has_r10) if b and not a),
            "cases_losing_r10": sorted(c for c, a, b in zip(key.case_ids, key.has_r10, vkey.has_r10) if a and not b),
            "changes": list(changes)}


# ---------------------------------------------------------------- power (#12)


def detection_power(rate: float, n: int) -> float:
    """Chance that n cases of one condition show at least one failure when its failure rate is `rate`."""
    return 1 - (1 - rate) ** n


def detectable_rate(n: int, power: float = POWER) -> float:
    """The smallest per-condition failure rate that n cases show at least once with probability `power`."""
    return 1 - (1 - power) ** (1 / n)


def cases_to_detect(rate: float, power: float = POWER) -> int:
    """Cases per condition needed to see at least one failure at `rate` with probability `power`."""
    return math.ceil(math.log(1 - power) / math.log(1 - rate))


def mde_paired_rate(n: int, discordance: float) -> float:
    """MDE of a paired difference in a rate (McNemar normal approximation, 5% two-sided, 80% power):
    (z_a + z_b) x sqrt(discordance / n), where `discordance` is the share of cases the two rows answer differently.
    Stratified by condition, the within-condition estimand's variance is at most this."""
    return (Z_ALPHA + Z_BETA) * math.sqrt(discordance / n)


def mde_paired_score(n_serious: int, n_benign: int, disc_serious: float, disc_benign: float, miss: float = 7.0) -> float:
    """MDE of a paired SCORE difference in points. SCORE = 100 x (1 - O - r U) with r = miss x n_serious / n_benign,
    so its paired variance is 100^2 x (disc_benign / n_benign + r^2 x disc_serious / n_serious)."""
    r = miss * n_serious / n_benign
    sd = 100 * math.sqrt(disc_benign / n_benign + r * r * disc_serious / n_serious)
    return (Z_ALPHA + Z_BETA) * sd


DEFAULT_DISCORDANCE = {"serious": (0.05, 0.10, 0.20), "benign": 0.20}


def power_table(serious_per_10: float = 234 / 47, benign_per_10: float = 118 / 47, conditions: int = 47,
                per_condition: Sequence[int] = (10, 20, 30), discordance: Mapping = DEFAULT_DISCORDANCE,
                tier1_conditions: int = 21, observed_benign_discordance: Optional[float] = None) -> dict:
    """Detectable per-condition failure rates and paired MDEs at each sample size.

    `serious_per_10` and `benign_per_10` are SERIOUS and BENIGN cases per condition at 10 per condition (the 470
    sample: 234 and 118 over 47 conditions); larger samples keep that mix. `discordance` names the assumed shares of
    cases on which two models answer differently; the observed v6 BENIGN discordance, when given, replaces the default.
    """
    d_b = observed_benign_discordance if observed_benign_discordance is not None else discordance["benign"]
    rows = []
    for m in per_condition:
        n_s = round(serious_per_10 * conditions * m / 10)
        n_b = round(benign_per_10 * conditions * m / 10)
        rows.append({
            "per_condition": m, "cases": conditions * m, "serious": n_s, "benign": n_b,
            "detectable_rate_80": detectable_rate(m),
            "power_at_5pct": detection_power(0.05, m), "power_at_10pct": detection_power(0.10, m),
            "zero_of_n_upper_95": one_sided_bound(0, m),
            "tier1_zero_of_n_upper_95": one_sided_bound(0, tier1_conditions * m),
            "mde_U_pp": {f"{d:g}": 100 * mde_paired_rate(n_s, d) for d in discordance["serious"]},
            "mde_O_pp": 100 * mde_paired_rate(n_b, d_b),
            "mde_score": {f"{d:g}": mde_paired_score(n_s, n_b, d, d_b) for d in discordance["serious"]},
        })
    return {"rows": rows, "power": POWER, "alpha": 0.05,
            "assumptions": {"serious_discordance": list(discordance["serious"]), "benign_discordance": d_b,
                            "benign_discordance_source": "observed, GPT-5.6 Terra vs GPT-OSS 120B v6 YES on BENIGN"
                            if observed_benign_discordance is not None else "assumed",
                            "detection": "at least one failure among the condition's cases",
                            "mde": "paired, McNemar normal approximation, 5% two-sided, 80% power; independent cases"},
            "cases_for_5pct": cases_to_detect(0.05), "cases_for_10pct": cases_to_detect(0.10),
            "recommendation": recommend(rows)}


RECOMMEND_TARGET_RATE = 0.06  # recommend the smallest size that detects a 6% per-condition failure rate at 80%


def recommend(rows: Sequence[Mapping], target_rate: float = RECOMMEND_TARGET_RATE) -> dict:
    """The smallest tabled size whose detectable per-condition rate is at most `target_rate` (else the largest),
    because a per-condition miss rate of about 5% is the failure Astra finding 6 says 10 per condition cannot see."""
    ok = [r for r in rows if r["detectable_rate_80"] <= target_rate]
    r = ok[0] if ok else rows[-1]
    by = {x["per_condition"]: x for x in rows}
    others = "; ".join(f"{m}: {100 * x['detectable_rate_80']:.1f}%" for m, x in by.items() if m != r["per_condition"])
    return {"per_condition": r["per_condition"], "cases": r["cases"],
            "reason": (f"{r['per_condition']} per condition ({r['cases']:,} cases) shows a "
                       f"{100 * r['detectable_rate_80']:.1f}% condition-specific failure rate with 80% power ({others}), "
                       f"bounds a condition with 0 failures under {100 * r['zero_of_n_upper_95']:.1f}%, and brings the "
                       f"paired U MDE to {r['mde_U_pp']['0.1']:.1f} points at 10% SERIOUS discordance. Paired SCORE "
                       f"differences stay hard to detect at every tabled size (MDE {r['mde_score']['0.1']:.0f} points), "
                       f"because one under-escalation weighs {7 * r['serious'] / r['benign']:.1f} over-escalations, "
                       "so U and O carry model comparisons.")}
