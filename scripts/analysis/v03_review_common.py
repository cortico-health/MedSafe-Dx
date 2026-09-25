"""Shared pieces for the v0.3 adversarial review (spec/v0.3-scoring.md, draft 1).

We rebuild the v0.3 answer key for any DDXPlus adult from committed files only,
so every review script scores the same key:

- tiers from spec/dangerous_if_missed_tiers_v03.csv (and three alternates);
- the red-herring detector M1' from results/analysis/dxa_red_herrings/cells.csv
  (cell rates) and hallmarks.csv (the five symptom hallmarks per condition);
- flag matching from spec/ddxplus_icd10_map.csv through evaluator.v02_score.ConditionMap.

Nothing here spends inference or edits an existing file. Scripts:
    v03_review_key.py      the key on the 470 sample, and where it is fragile
    v03_review_gaming.py   simulated policies (always YES, fixed lists, symptom and age rules)
    v03_review_models.py   the 19 v0.1 rows on the 250 set as stand-ins for a v0.3 run
    v03_review_flags.py    flag leniency: broader and related codes
    v03_review_pools.py    pool-subset availability
"""

from __future__ import annotations

import ast
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluator import answer_key_v02 as ak  # noqa: E402
from evaluator.v02_score import ConditionMap  # noqa: E402

RH_DIR = ROOT / "results/analysis/dxa_red_herrings"
ADULTS_PKL = RH_DIR / "cache/adults.pkl"
CELLS_CSV = RH_DIR / "cells.csv"
HALLMARKS_CSV = RH_DIR / "hallmarks.csv"
TIERS_CSV = ROOT / "spec/dangerous_if_missed_tiers_v03.csv"
COND_JSON = ROOT / "data/ddxplus_v0/release_conditions.json"
SAMPLE470 = ROOT / "data/test_sets/eval-v02-adult.case_ids.txt"
SAMPLE250 = ROOT / "data/test_sets/eval-250-v0.json"
OUT = ROOT / "results/analysis/v03_review"

BANDS = (0.0, 5.0, 10.0, 20.0, 35.0, 50.0, 100.01)  # percent, left-closed, as in dxa_red_herrings.py
FLOOR, RATIO, MIN_N = 0.01, 0.10, 30
R10, R5 = 10.0, 5.0
LENIENT = ("equivalent", "narrower", "broader", "related")
STRICT = ("equivalent", "narrower")
N_BOOT = 2000
SEED = 20260923

# The one-source rule adds these (docs/tier-upgrade-sources.md section 5).
ONE_SOURCE_EXTRA = ("Atrial fibrillation", "Anemia", "HIV (initial infection)", "SLE")
NEWMAN_TOKER = ("Pneumonia", "Pulmonary neoplasm", "Pancreatic neoplasm")


def band_of(p_pct: float) -> int:
    return int(np.searchsorted(np.array(BANDS), p_pct, side="right") - 1)


def band_label(b: int) -> str:
    lo, hi = BANDS[b], BANDS[b + 1]
    return f"{lo:g}-{min(hi, 100):g}%"


# ---------------------------------------------------------------- tiers


def load_tiers() -> dict[str, dict[str, int]]:
    """Four tier tables: v03 (the spec), severity (crosswalk only), nt (Newman-Toker only), one_source."""
    rows = list(csv.DictReader(open(TIERS_CSV, newline="", encoding="utf-8")))
    v03 = {r["condition"]: int(r["final_tier"]) for r in rows}
    sev = {r["condition"]: int(r["base_tier"]) for r in rows}
    nt = dict(sev)
    for c in NEWMAN_TOKER:
        nt[c] = 1
    one = dict(v03)
    for c in ONE_SOURCE_EXTRA:
        one[c] = 1
    return {"v03": v03, "severity": sev, "nt": nt, "one_source": one}


# ---------------------------------------------------------------- red herrings


class RedHerring:
    """M1' from the committed cell table: (condition, band, hallmark count) -> (n, rate)."""

    def __init__(self):
        cells = pd.read_csv(CELLS_CSV)
        m = cells[cells.detector == "M1p"]
        self.cell: dict[tuple[str, str, int], tuple[int, float]] = {}
        for r in m.itertuples():
            h = int(str(r.sub).split("=")[1])
            self.cell[(r.condition, r.band, h)] = (int(r.n), float(r.rate))
        hm = pd.read_csv(HALLMARKS_CSV)
        hm = hm[hm.variant == "symptoms"]
        self.hallmarks: dict[str, list[str]] = defaultdict(list)
        for r in hm.itertuples():
            self.hallmarks[r.condition].append(r.token)

    def hcount(self, cond: str, evf: frozenset) -> int:
        return sum(1 for t in self.hallmarks.get(cond, []) if t in evf)

    def lookup(self, cond: str, p_pct: float, evf: frozenset) -> tuple[int, float, int]:
        """(cell n, cell rate, hallmark count); n = 0 when the cell is absent."""
        h = self.hcount(cond, evf)
        n, rate = self.cell.get((cond, band_label(band_of(p_pct)), h), (0, float("nan")))
        return n, rate, h

    def label(self, cond: str, p_pct: float, evf: frozenset) -> tuple[bool, bool, int, float, int]:
        """(red_herring, undetermined, n, rate, hallmarks)."""
        n, rate, h = self.lookup(cond, p_pct, evf)
        if n < MIN_N or not np.isfinite(rate):
            return False, True, n, rate, h
        return rate < max(FLOOR, RATIO * p_pct / 100.0), False, n, rate, h


# ---------------------------------------------------------------- cases and key


def load_adults() -> pd.DataFrame:
    df = pd.read_pickle(ADULTS_PKL)
    df["case_id"] = "ddxplus_" + df.ROW.astype(str)
    return df.set_index("case_id", drop=False)


def sample_ids(which: str) -> list[str]:
    if which == "470":
        return SAMPLE470.read_text().split()
    if which == "250":
        return [c["case_id"] for c in json.loads(SAMPLE250.read_text())["cases"]]
    raise ValueError(which)


def build_key(df: pd.DataFrame, ids: list[str], tiers: dict[str, int], rh: RedHerring,
              t_main: float = R10, t_cov: float = R5, filter_only_severity2: bool = False,
              conditions: dict | None = None) -> list[dict]:
    """One dict per case: truth, tier, DXA, targets at both thresholds, red herrings, clearly-low-risk.

    filter_only_severity2 reproduces the doc's detector scope (severity <= 2 conditions only), to test
    whether the spec's counts were made with the four upgraded conditions unfiltered.
    """
    conditions = conditions or ak.load_conditions()
    out = []
    for cid in ids:
        r = df.loc[cid]
        truth = r.PATHOLOGY
        dxa = {n: 100.0 * p for n, p in r.DD}
        evf = r.EVF
        targets = {}
        for cond, p in dxa.items():
            if tiers.get(cond) != 1 or p < t_cov:
                continue
            if cond == truth:
                targets[cond] = {"p": p, "why": "truth", "rh": False, "und": False, "h": rh.hcount(cond, evf), "n": 0, "rate": float("nan")}
                continue
            if filter_only_severity2 and conditions[cond]["ddxplus_severity"] > 2:
                targets[cond] = {"p": p, "why": "dxa_unfiltered", "rh": False, "und": False, "h": rh.hcount(cond, evf), "n": 0, "rate": float("nan")}
                continue
            is_rh, und, n, rate, h = rh.label(cond, p, evf)
            targets[cond] = {"p": p, "why": "dxa", "rh": is_rh, "und": und, "h": h, "n": n, "rate": rate}
        if tiers.get(truth) == 1 and truth not in targets:
            targets[truth] = {"p": dxa.get(truth, 0.0), "why": "truth", "rh": False, "und": False, "h": rh.hcount(truth, evf), "n": 0, "rate": float("nan")}
        r10 = {c for c, t in targets.items() if not t["rh"] and (t["why"] == "truth" or t["p"] >= t_main)}
        r5 = {c for c, t in targets.items() if not t["rh"]}
        r20 = {c for c, t in targets.items() if not t["rh"] and (t["why"] == "truth" or t["p"] >= 20.0)}
        rflags = ak.red_flags(evf)
        out.append({
            "case_id": cid, "truth": truth, "tier": tiers.get(truth, 3), "age": int(r.AGE), "sex": r.SEX,
            "dxa": dxa, "evf": evf, "targets": targets, "R10": r10, "R5": r5, "R20": r20,
            "truth_only": {truth} if tiers.get(truth) == 1 else set(),
            "red_flag": bool(rflags), "red_flag_names": rflags,
            "clearly_low": (not r5) and tiers.get(truth, 3) == 3 and not rflags,
            "dxa_top1": max(dxa, key=dxa.get) if dxa else None,
        })
    return out


# ---------------------------------------------------------------- flags and scoring


class Matcher:
    """Flag matching through the map's longest-prefix rule, lenient or strict."""

    def __init__(self):
        self.cmap = ConditionMap()

    def relation(self, code: str, cond: str):
        return self.cmap.relation(code, cond)

    def matched(self, codes: list[str], cond: str, relations=LENIENT) -> bool:
        return any(self.cmap.relation(c, cond) in relations for c in codes if c)

    def conditions_hit(self, codes: list[str], relations=LENIENT) -> set[str]:
        hit = set()
        for c in codes:
            if not c:
                continue
            for cond, (_, rel) in self.cmap.resolve(c).items():
                if rel in relations:
                    hit.add(cond)
        return hit


def score(key: list[dict], answers: list[dict], matcher: Matcher, relations=LENIENT,
          target_field: str = "R10", cov_field: str = "R5", tiers: dict | None = None) -> dict:
    """Spec section 5 on one row.

    answers[i] = {"yes": True/False/None (None = unreadable), "flags": [codes]}.
    """
    h_ev = h_den = 0
    cov = []
    oc_ev = oc_den = 0
    mt_truth = [0, 0]
    mt_dxa = [0, 0]
    con = 0
    yes_no_t1 = 0
    yes_n = 0
    per_case = []
    for k, a in zip(key, answers):
        yes = a["yes"]
        flags = a["flags"][:5]
        tgt = k[target_field]
        h = False
        if tgt:
            h_den += 1
            h = yes is not True
            h_ev += h
        if k[cov_field]:
            cov.append(np.mean([matcher.matched(flags, c, relations) for c in k[cov_field]]))
        if k["clearly_low"]:
            oc_den += 1
            oc_ev += yes is True
        for c in tgt:
            slot = mt_truth if k["targets"][c]["why"] == "truth" else mt_dxa
            slot[1] += 1
            slot[0] += not matcher.matched(flags, c, relations)
        hit_t1 = any((tiers or {}).get(c) == 1 for c in matcher.conditions_hit(flags, relations))
        if yes is True and not flags:
            con += 1
        if yes is False and hit_t1:
            con += 1
        if yes is True:
            yes_n += 1
            if not hit_t1:
                yes_no_t1 += 1
        per_case.append({"case_id": k["case_id"], "truth": k["truth"], "H": h, "in_H": bool(tgt)})
    return {
        "H": h_ev / h_den if h_den else float("nan"), "H_ev": h_ev, "H_den": h_den,
        "COV": float(np.mean(cov)) if cov else float("nan"), "COV_n": len(cov),
        "OC": oc_ev / oc_den if oc_den else float("nan"), "OC_den": oc_den,
        "MT_truth": mt_truth[0] / mt_truth[1] if mt_truth[1] else float("nan"),
        "MT_dxa": mt_dxa[0] / mt_dxa[1] if mt_dxa[1] else float("nan"),
        "MT_n": (mt_truth[1], mt_dxa[1]),
        "CON": con / len(key), "YES_rate": yes_n / len(key),
        "YES_no_tier1_flag": yes_no_t1 / yes_n if yes_n else float("nan"),
        "per_case": per_case,
    }


def cluster_bootstrap_h(key: list[dict], answers: list[dict], target_field: str = "R10",
                        n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float, float]:
    """95% interval for H, resampling truth conditions (clusters). Returns (point, lo, hi)."""
    conds = sorted({k["truth"] for k in key})
    ev = defaultdict(float)
    den = defaultdict(float)
    for k, a in zip(key, answers):
        if k[target_field]:
            den[k["truth"]] += 1
            ev[k["truth"]] += a["yes"] is not True
    e = np.array([ev[c] for c in conds])
    d = np.array([den[c] for c in conds])
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(conds), len(conds))
        dd = d[idx].sum()
        draws.append(e[idx].sum() / dd if dd else np.nan)
    draws = np.array(draws)
    point = e.sum() / d.sum() if d.sum() else float("nan")
    return point, float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def fmt(x, d=1) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return f"{100 * x:.{d}f}"
