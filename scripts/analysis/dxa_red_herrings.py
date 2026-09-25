#!/usr/bin/env python3
"""Red herrings in DXA's differentials, tested against DDXPlus's own ground truth.

A red herring is a (patient, serious condition c) pair where DXA puts p >= X on c, but a
plausibility estimator says c is essentially never the true condition for patients who look
the same. "Never true in DDXPlus's world", not "clinically implausible".

Every estimator gives P(c | patient) for all 49 conditions. The cell-based ones read the
empirical rate of c among reference adults in the same cell:

  DXA    DXA's own p_c                                   the thing under test
  A      cell = (c, p_c band)                            DXA calibration for c alone; fewest assumptions
  M1p    cell = (c, p_c band, count of c's hallmarks)    conditions only on evidence about c
  M1pp   cell = (c, p_c band, which hallmarks present)   finer variant of M1p
  B      cell = (c, p_c band, DXA top-1 condition)       assumes DXA's disease grouping
  M2     cell = (c, DXA top-3 set)                       assumes DXA's grouping, finer
  KNN    200 nearest adults by evidence Jaccard          sample cases only
  NB<T>  tempered naive Bayes over all evidence, likelihoods ^ T; T = 1 is the dataset oracle

Selection: the estimators are scored against real-world pre-test probabilities (the
published rates and NHAMCS cells in results/analysis/risk_proxy/), because DDXPlus's own
truth always favours the sharpest full-evidence model. T is fitted leave-one-pattern-out.

Hallmarks of c: the 5 symptom tokens (evidence with its value, antecedents excluded) with the
largest likelihood ratio P(e|c)/P(e|not c) among tokens present in >= 20% of c's patients,
learned on the reference adults. M1p_ant is the same with antecedents allowed; DDXPlus samples
risk-factor antecedents only for the true condition, so that variant leans to the oracle.

Reference adults: the DDXPlus test split, age >= 18, minus the 470-case eval-v02-adult
sample and the 250-case v0 set, so those cases never see their own truth. Reference rows
are scored leave-one-out. Cells under MIN_N reference patients are suppressed.

Rule: (i, c) is a red herring when DXA p >= X and the estimator's P(c) < max(FLOOR, RATIO * p).

Outputs go to results/analysis/dxa_red_herrings/. No inference spend.
"""

from __future__ import annotations

import ast
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluator.v02_score import ConditionMap  # noqa: E402
from scripts.analysis import dim_pilot as dp  # noqa: E402
from scripts.analysis import risk_proxy_validation as rpv  # noqa: E402

DDX_CSV = ROOT / "data/ddxplus_v0/release_test_patients"
COND_PATH = ROOT / "data/ddxplus_v0/release_conditions.json"
EVID_PATH = ROOT / "data/ddxplus_v0/release_evidences.json"
SAMPLE470 = ROOT / "data/test_sets/eval-v02-adult.case_ids.txt"
SAMPLE250 = ROOT / "data/test_sets/eval-250-v0.json"
NHAMCS_CELLS = ROOT / "results/analysis/risk_proxy/nhamcs_cells.csv"
OUT = ROOT / "results/analysis/dxa_red_herrings"
CACHE = OUT / "cache"

SERIOUS_SEVERITY = 2          # DDXPlus severity <= 2 is "serious" (evaluator/answer_key_v02.py)
X_MAIN = 10.0                 # percent: the DXA probability at which we ask the question
X_GRID = (5.0, 10.0, 20.0)
FLOOR = 0.01                  # rate floor of the rule
RATIO = 0.10                  # rate must be under p * RATIO
MIN_N = 30                    # suppress cells under this many reference patients
BANDS = (0.0, 5.0, 10.0, 20.0, 35.0, 50.0, 100.01)  # p_c bands in percent, left-closed
N_HALLMARKS = 5
HALLMARK_MIN_PREV = 0.20
KNN_K = 200
NB_TEMPS = (0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0)
NB_REPORT = (0.1, 0.25, 0.5, 1.0)
EXCUSE_T = (10.0, 20.0)       # single-condition excuse thresholds, percent
PRIMARY = "M1p"               # the recommended detector (docs/dxa-red-herrings.md); A and B are reported as checks
CHECKS = ("A", "B")
N_BOOT = 2000
SEED = 20260924
SENS_RULES = {  # name -> (floor, ratio, use the Wilson 95% upper bound)
    "max(1%, p/10)": (0.01, 0.10, False),
    "1% absolute": (0.01, 0.0, False),
    "p/10": (0.0, 0.10, False),
    "max(0.5%, p/20)": (0.005, 0.05, False),
    "max(2%, p/5)": (0.02, 0.20, False),
    "Wilson upper 95% < max(1%, p/10)": (0.01, 0.10, True),
}
SENS_MIN_N = (30, 100)

# Known knowledge-base quirks to check (DXA top-1 family, serious condition)
QUIRKS = [
    ("MI in laryngitis / pharyngitis / bronchitis", {"Acute laryngitis", "Viral pharyngitis", "Bronchitis"}, "Possible NSTEMI / STEMI"),
    ("MI in sarcoidosis / SLE", {"Sarcoidosis", "SLE"}, "Possible NSTEMI / STEMI"),
    ("MI in pericarditis", {"Pericarditis"}, "Possible NSTEMI / STEMI"),
    ("MI in pulmonary neoplasm", {"Pulmonary neoplasm"}, "Possible NSTEMI / STEMI"),
    ("MI in GERD", {"GERD"}, "Possible NSTEMI / STEMI"),
    ("Unstable angina in GERD", {"GERD"}, "Unstable angina"),
    ("Anaphylaxis in inguinal hernia", {"Inguinal hernia"}, "Anaphylaxis"),
    ("Anaphylaxis in pancreatic neoplasm", {"Pancreatic neoplasm"}, "Anaphylaxis"),
    ("PSVT in atrial fibrillation", {"Atrial fibrillation"}, "PSVT"),
    ("Pulmonary oedema in atrial fibrillation", {"Atrial fibrillation"}, "Acute pulmonary edema"),
    ("Guillain-Barre in COPD / asthma", {"Acute COPD exacerbation / infection", "Bronchospasm / acute asthma exacerbation"}, "Guillain-Barré syndrome"),
    ("Myocarditis in COPD / asthma", {"Acute COPD exacerbation / infection", "Bronchospasm / acute asthma exacerbation"}, "Myocarditis"),
    ("Dystonic reactions in sarcoidosis / SLE", {"Sarcoidosis", "SLE"}, "Acute dystonic reactions"),
    ("Scombroid in URTI / pharyngitis / sinusitis", {"URTI", "Viral pharyngitis", "Allergic sinusitis", "Acute rhinosinusitis"}, "Scombroid food poisoning"),
    ("PE in bronchitis / pneumonia", {"Bronchitis", "Pneumonia"}, "Pulmonary embolism"),
]


# ---------------------------------------------------------------- data


def load_adults() -> pd.DataFrame:
    """Adults in the test split, with the original row index (case_id = ddxplus_<row>)."""
    cache = CACHE / "adults.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    df = pd.read_csv(DDX_CSV)
    df["ROW"] = np.arange(len(df))
    df = df[df.AGE >= 18].reset_index(drop=True)
    df["EVF"] = df.EVIDENCES.apply(lambda s: frozenset(str(e) for e in ast.literal_eval(s)))
    df["EV"] = df.EVF.apply(lambda ev: {e.split("_@_")[0] for e in ev})
    df["DD"] = df.DIFFERENTIAL_DIAGNOSIS.apply(lambda s: [(n, float(p)) for n, p in ast.literal_eval(s)])
    df = df.drop(columns=["EVIDENCES", "DIFFERENTIAL_DIAGNOSIS"])
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_pickle(cache)
    return df


def sample_rows(path: Path) -> set[int]:
    if path.suffix == ".txt":
        ids = path.read_text().split()
    else:
        ids = [c["case_id"] for c in json.loads(path.read_text())["cases"]]
    return {int(i.split("_")[1]) for i in ids}


def band_of(p_pct: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.array(BANDS), p_pct, side="right") - 1


def band_label(b: int) -> str:
    lo, hi = BANDS[b], BANDS[b + 1]
    return f"{lo:g}-{min(hi, 100):g}%"


def evidence_matrix(df: pd.DataFrame, col: str) -> tuple[np.ndarray, list[str]]:
    """Binary adults x evidence matrix over base codes (col EV) or full tokens with values (col EVF)."""
    codes = sorted({e for ev in df[col] for e in ev})
    idx = {c: j for j, c in enumerate(codes)}
    M = np.zeros((len(df), len(codes)), np.uint8)
    for i, ev in enumerate(df[col]):
        for e in ev:
            M[i, idx[e]] = 1
    return M, codes


# ---------------------------------------------------------------- hallmarks and cells


def hallmarks(M, truth_idx, ref, codes, conds, evid, allowed, variant) -> tuple[list[list[int]], list[dict]]:
    """Per condition: top N_HALLMARKS allowed evidence tokens by likelihood ratio among tokens present in >= 20% of its patients."""
    Mr = M[ref].astype(np.float64)
    tr = truth_idx[ref]
    out, rows = [], []
    for k, c in enumerate(conds):
        pos = tr == k
        if not pos.any():
            out.append([])
            continue
        p1 = Mr[pos].mean(0)
        p0 = Mr[~pos].mean(0)
        lr = (p1 + 1e-6) / (p0 + 1e-6)
        ok = np.where((p1 >= HALLMARK_MIN_PREV) & allowed)[0]
        top = sorted(ok, key=lambda j: -lr[j])[:N_HALLMARKS]
        out.append(top)
        for rank, j in enumerate(top, 1):
            base, _, val = codes[j].partition("_@_")
            e = evid.get(base, {})
            vm = e.get("value_meaning") or {}
            vm = json.loads(vm) if isinstance(vm, str) else vm
            meaning = (vm.get(val) or {}).get("en", val) if val else ""
            rows.append({"variant": variant, "condition": c, "rank": rank, "token": codes[j], "question": e.get("question_en", ""),
                         "value": meaning, "antecedent": e.get("is_antecedent") in (True, "True"),
                         "p_given_c": round(p1[j], 4), "p_given_not_c": round(p0[j], 4), "lr": round(lr[j], 2), "n_c": int(pos.sum())})
    return out, rows


def cell_rates(cell: np.ndarray, y: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Leave-one-out empirical rate of y per row, given an integer cell id per row (one condition)."""
    size = int(cell.max()) + 1
    n = np.bincount(cell[ref], minlength=size).astype(float)
    k = np.bincount(cell[ref], weights=y[ref].astype(float), minlength=size)
    n_row = n[cell] - ref
    k_row = k[cell] - (ref & y)
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(n_row > 0, k_row / n_row, np.nan)
    return rate, n_row, n, k


class Estimators:
    """P(c | patient) matrices (adults x conditions), with cell sizes where a cell exists."""

    def __init__(self, df, conds, M, MF, hm, hm_ant, truth_idx, ref):
        N, C = len(df), len(conds)
        self.conds, self.N, self.C = conds, N, C
        cidx = {c: j for j, c in enumerate(conds)}
        P = np.zeros((N, C))
        top1 = np.zeros(N, int)
        top3 = []
        for i, dd in enumerate(df.DD):
            top1[i] = cidx[dd[0][0]]
            top3.append("|".join(sorted(n for n, _ in dd[:3])))
            for n, p in dd:
                P[i, cidx[n]] = p * 100
        self.P, self.top1 = P, top1
        self.top3_id = pd.factorize(np.array(top3, object))[0]
        self.band = band_of(P)
        Y = np.zeros((N, C), bool)
        Y[np.arange(N), truth_idx] = True
        self.Y = Y
        H = np.zeros((N, C), np.int16)
        HP = np.zeros((N, C), np.int32)
        HA = np.zeros((N, C), np.int16)
        for k in range(C):
            sub = MF[:, hm[k]]
            H[:, k] = sub.sum(1)
            HP[:, k] = (sub * (1 << np.arange(len(hm[k])))).sum(1)
            HA[:, k] = MF[:, hm_ant[k]].sum(1)
        self.H, self.HP, self.HA = H, HP, HA
        self.prior = np.bincount(truth_idx[ref], minlength=C) / ref.sum()
        self.rate, self.n, self.cells = {}, {}, {}
        nb = len(BANDS) - 1
        defs = {
            "A": self.band,
            "M1p": self.band * (N_HALLMARKS + 1) + H,
            "M1pp": self.band * (1 << N_HALLMARKS) + HP,
            "M1p_ant": self.band * (N_HALLMARKS + 1) + HA,
            "B": self.band * C + top1[:, None],
            "M2": np.repeat(self.top3_id[:, None], C, 1),
        }
        for name, cellmat in defs.items():
            R = np.full((N, C), np.nan)
            Nn = np.zeros((N, C))
            cell_rows = []
            for k in range(C):
                r, n_row, n, kk = cell_rates(cellmat[:, k], Y[:, k], ref)
                R[:, k], Nn[:, k] = r, n_row
                if name != "M2":
                    width = {"A": 1, "M1p": N_HALLMARKS + 1, "M1p_ant": N_HALLMARKS + 1, "M1pp": 1 << N_HALLMARKS, "B": C}[name]
                    for cid in np.nonzero(n)[0]:
                        b = cid // width
                        rest = cid - b * width
                        cell_rows.append({"detector": name, "condition": conds[k], "band": band_label(int(b)),
                                          "sub": {"A": "", "M1p": f"hallmarks={rest}", "M1p_ant": f"hallmarks={rest}", "M1pp": f"pattern={rest:05b}", "B": f"top1={conds[rest]}"}[name],
                                          "n": int(n[cid]), "k": int(kk[cid]), "rate": float(kk[cid] / n[cid]), "suppressed": bool(n[cid] < MIN_N)})
            self.rate[name], self.n[name], self.cells[name] = R, Nn, cell_rows
        # tempered naive Bayes
        Mr = M[ref].astype(np.float64)
        tr = truth_idx[ref]
        n = np.bincount(tr, minlength=C).astype(float)
        K = np.stack([Mr[tr == k].sum(0) for k in range(C)])
        Lk = (K + 0.5) / (n[:, None] + 1.0)
        x = M.astype(np.float64)
        self.LL = x @ np.log(Lk).T + (1.0 - x) @ np.log(1.0 - Lk).T
        self.logprior = np.log(n / n.sum())

    def nb(self, T: float) -> np.ndarray:
        lp = self.logprior[None, :] + T * self.LL
        lp -= lp.max(1, keepdims=True)
        p = np.exp(lp)
        return p / p.sum(1, keepdims=True)

    def estimate(self, name: str) -> np.ndarray:
        """P(c | patient) in [0, 1]; cell estimators fall back to the prior where the cell is empty."""
        if name == "DXA":
            return self.P / 100
        if name.startswith("NB"):
            return self.nb(float(name[2:]))
        R = self.rate[name]
        return np.where(np.isnan(R), self.prior[None, :], R)


def knn_rates(M, truth_idx, ref, query_rows, C) -> np.ndarray:
    """Rate of each condition among the KNN_K reference adults nearest by evidence Jaccard; query rows only."""
    Mr = M[ref].astype(np.float32)
    tr = truth_idx[ref]
    size_r = Mr.sum(1)
    R = np.full((M.shape[0], C), np.nan)
    for i in query_rows:
        q = M[i].astype(np.float32)
        inter = Mr @ q
        jac = inter / (size_r + q.sum() - inter)
        nn = np.argpartition(-jac, KNN_K)[:KNN_K]
        R[i] = np.bincount(tr[nn], minlength=C) / KNN_K
    return R


# ---------------------------------------------------------------- selection against real-world rates


def referee_patterns(df) -> list[dict]:
    """Published pre-test probabilities from the risk-proxy analysis, with the DDXPlus adults each speaks to."""
    rows = []
    for name, spec in rpv.PATTERNS.items():
        mask = df.EVF.apply(spec["pred"]).to_numpy()
        for pub in rpv.PUBLISHED.get(name, []):
            if not pub.get("rate"):
                continue
            rows.append({"pattern": name, "conditions": pub["conditions"], "rate": pub["rate"], "primary": bool(pub.get("use")),
                         "setting": pub["setting"], "scope": pub["scope"], "mask": mask, "n": int(mask.sum())})
    return rows


def pattern_means(est: np.ndarray, pats: list[dict], cidx: dict) -> np.ndarray:
    return np.array([est[p["mask"]][:, [cidx[c] for c in p["conditions"]]].sum(1).mean() for p in pats])


def select_estimators(E: Estimators, df, serious_idx, out: dict) -> tuple[pd.DataFrame, pd.DataFrame, float, pd.DataFrame]:
    cidx = {c: j for j, c in enumerate(E.conds)}
    pats = referee_patterns(df)
    prim = [p for p in pats if p["primary"]]
    rates = np.array([p["rate"] for p in pats])
    is_prim = np.array([p["primary"] for p in pats])
    # candidate matrices
    cands = {n: E.estimate(n) for n in ("DXA", "A", "M1p", "M1pp", "M1p_ant", "B")}
    nbm = {T: E.nb(T) for T in NB_TEMPS}
    for T in NB_REPORT:
        cands[f"NB{T:g}"] = nbm[T]
    means = {n: pattern_means(m, pats, cidx) for n, m in cands.items()}
    nb_means = {T: pattern_means(m, pats, cidx) for T, m in nbm.items()}
    # leave-one-pattern-out fit of T on the primary patterns: raw, and shape-only (a free log offset absorbs
    # DDXPlus's fixed prior inflation, which every DDXPlus-calibrated estimator inherits)
    pi = np.where(is_prim)[0]
    folds = []
    lopo = np.full(len(pats), np.nan)
    lopo_shape = np.full(len(pats), np.nan)  # held-out log-ratio error after the offset fitted on the other patterns
    for h in pi:
        train = [j for j in pi if j != h]
        best = min(NB_TEMPS, key=lambda T: np.sum((np.log(nb_means[T][train]) - np.log(rates[train])) ** 2))
        lopo[h] = nb_means[best][h]

        def shape_loss(T):
            e = np.log(nb_means[T][train]) - np.log(rates[train])
            return np.sum((e - e.mean()) ** 2)
        best_s = min(NB_TEMPS, key=shape_loss)
        e_tr = np.log(nb_means[best_s][train]) - np.log(rates[train])
        lopo_shape[h] = np.log(nb_means[best_s][h] / rates[h]) - e_tr.mean()
        folds.append({"held_out": pats[h]["pattern"], "T_fitted": best, "published": rates[h], "nb_fit_mean": lopo[h],
                      "log_ratio": np.log(lopo[h] / rates[h]), "T_fitted_shape": best_s, "shape_error": lopo_shape[h]})
    T_all = min(NB_TEMPS, key=lambda T: np.sum((np.log(nb_means[T][pi]) - np.log(rates[pi])) ** 2))
    means["NBfit"] = nb_means[T_all].copy()
    means["NBfit"][pi] = lopo[pi]  # held-out values on the primary patterns
    cands["NBfit"] = nbm[T_all]
    # per-pattern table
    prow = []
    for j, p in enumerate(pats):
        r = {"pattern": p["pattern"], "conditions": "; ".join(p["conditions"]), "primary": p["primary"], "n_ddx": p["n"],
             "published": p["rate"], "setting": p["setting"], "ddx_true": float(E.Y[p["mask"]][:, [cidx[c] for c in p["conditions"]]].any(1).mean())}
        for n in means:
            r[f"mean_{n}"] = means[n][j]
            r[f"logratio_{n}"] = np.log(means[n][j] / p["rate"])
        prow.append(r)
    # summary per candidate with pattern bootstrap
    rng = np.random.default_rng(SEED)
    boots = rng.integers(0, len(pi), (N_BOOT, len(pi)))
    srow = []
    nh = nhamcs_eval(E, df, serious_idx, cands)
    for n in means:
        e = np.log(means[n] / rates)
        ep = e[pi]
        bs = np.abs(ep[boots]).mean(1)
        if n == "NBfit":
            shape = lopo_shape[pi]
        else:  # held-out offset: the mean log ratio of the other 7 patterns
            shape = np.array([ep[j] - np.delete(ep, j).mean() for j in range(len(pi))])
        bss = np.abs(shape[boots]).mean(1)
        r = {"estimator": n, "mean_abs_logratio_primary": np.abs(ep).mean(), "ci_low": np.percentile(bs, 2.5), "ci_high": np.percentile(bs, 97.5),
             "shape_error_primary": np.abs(shape).mean(), "shape_ci_low": np.percentile(bss, 2.5), "shape_ci_high": np.percentile(bss, 97.5),
             "median_ratio_primary": np.exp(np.median(ep)), "mean_abs_logratio_all": np.abs(e).mean(), "median_ratio_all": np.exp(np.median(e)),
             "patterns_within_2x_primary": int((np.abs(ep) < np.log(2)).sum()), "n_primary": len(pi), "n_all": len(pats)}
        r.update(nh[n])
        srow.append(r)
    out["T_fitted_all_primary"] = T_all
    return pd.DataFrame(prow), pd.DataFrame(srow), T_all, pd.DataFrame(folds)


def nhamcs_eval(E: Estimators, df, serious_idx, cands: dict) -> dict[str, dict]:
    """Per estimator: mean P(any on-list serious) per NHAMCS presentation x age cell, against the ED serious-diagnosis rate."""
    cells = pd.read_csv(NHAMCS_CELLS)
    cells = cells[cells.nhamcs_n >= rpv.MIN_CELL_N]
    masks = {}
    age = df.AGE.to_numpy()
    for cell, pred in rpv.DDX_CELLS.items():
        base = df.EVF.apply(pred).to_numpy()
        for band, lo, hi in rpv.AGE_BANDS + [("18+", 18, 200)]:
            masks[(cell, band)] = base & (age >= lo) & (age <= hi)
    res = {}
    for n, m in cands.items():
        ps = m[:, serious_idx].sum(1)
        est = np.array([ps[masks[(c, b)]].mean() if masks[(c, b)].any() else np.nan for c, b in zip(cells.cell, cells.age_band)])
        banded = (cells.age_band != "18+").to_numpy() & ~np.isnan(est)
        pres = (cells.age_band == "18+").to_numpy() & ~np.isnan(est)
        on = cells.nh_serious_on.to_numpy()
        anyd = cells.nh_serious_dx.to_numpy()
        res[n] = {
            "nh_spearman_on_cells": spearmanr(est[banded], on[banded]).correlation,
            "nh_spearman_any_cells": spearmanr(est[banded], anyd[banded]).correlation,
            "nh_spearman_on_presentations": spearmanr(est[pres], on[pres]).correlation,
            "nh_median_ratio_on_cells": float(np.median(est[banded] / on[banded])),
            "nh_median_ratio_any_cells": float(np.median(est[banded] / anyd[banded])),
            "nh_cells_within_2x_on": int((np.abs(np.log(est[banded] / on[banded])) < np.log(2)).sum()),
            "nh_cells": int(banded.sum()),
        }
    return res


# ---------------------------------------------------------------- rule


def rh_label(p_pct, rate, n, floor=FLOOR, ratio=RATIO, min_n=MIN_N, upper=False):
    """Red herring where the estimator's rate (or its Wilson 95% upper bound) is under max(floor, ratio * p)."""
    thr = np.maximum(floor, ratio * p_pct / 100.0)
    ok = (n >= min_n) & ~np.isnan(rate)
    r = rate
    if upper:
        z = 1.96
        nn = np.where(np.isfinite(n) & (n > 0), n, 1e9)
        r = (rate + z * z / (2 * nn) + z * np.sqrt(rate * (1 - rate) / nn + z * z / (4 * nn * nn))) / (1 + z * z / nn)
    return ok & (r < thr), ~ok


# ---------------------------------------------------------------- helpers


def write_csv(path: Path, rows) -> None:
    if isinstance(rows, pd.DataFrame):
        rows.to_csv(path, index=False)
        return
    rows = list(rows)
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def md_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    def fmt(v):
        if isinstance(v, float):
            return "-" if np.isnan(v) else format(v, floatfmt)
        return str(v)
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def pct(x, d=1) -> str:
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{d}f}%"


def fnum(x, d=4):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), d)


# ---------------------------------------------------------------- main


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    conditions = json.loads(COND_PATH.read_text())
    sev = {n: int(v["severity"]) for n, v in conditions.items()}
    conds = sorted(conditions)
    cidx = {c: j for j, c in enumerate(conds)}
    serious = [c for c in conds if sev[c] <= SERIOUS_SEVERITY]
    serious_idx = np.array([cidx[c] for c in serious])
    evid = json.loads(EVID_PATH.read_text())

    df = load_adults()
    rows470, rows250 = sample_rows(SAMPLE470), sample_rows(SAMPLE250)
    in470 = df.ROW.isin(rows470).values
    in250 = df.ROW.isin(rows250).values
    ref = ~(in470 | in250)
    truth = df.PATHOLOGY.values
    truth_idx = np.array([cidx[t] for t in truth])
    print(f"adults {len(df)}, reference {ref.sum()}, sample470 {in470.sum()}, set250 {in250.sum()}")

    M, codes = evidence_matrix(df, "EV")     # base codes: naive Bayes and k-NN, as in evaluator/v02_references.py
    MF, toks = evidence_matrix(df, "EVF")    # full tokens with values: hallmarks
    is_ant = np.array([evid.get(t.partition("_@_")[0], {}).get("is_antecedent") in (True, "True") for t in toks])
    hm, hm_rows = hallmarks(MF, truth_idx, ref, toks, conds, evid, ~is_ant, "symptoms")
    hm_ant, hm_rows_ant = hallmarks(MF, truth_idx, ref, toks, conds, evid, np.ones(len(toks), bool), "all evidence")
    write_csv(OUT / "hallmarks.csv", hm_rows + hm_rows_ant)
    E = Estimators(df, conds, M, MF, hm, hm_ant, truth_idx, ref)
    cells = pd.DataFrame([r for n in ("A", "M1p", "M1pp", "M1p_ant", "B") for r in E.cells[n]])
    write_csv(OUT / "cells.csv", cells)
    print("estimators built")

    # ---- selection against real-world rates
    sel_out = {}
    pat_tbl, sel_tbl, T_fit, folds = select_estimators(E, df, serious_idx, sel_out)
    write_csv(OUT / "selection_patterns.csv", pat_tbl)
    write_csv(OUT / "selection_summary.csv", sel_tbl)
    write_csv(OUT / "selection_T_folds.csv", folds)
    print(f"T fitted on all primary patterns: {T_fit}")
    nb_deg = []
    nonser = np.array([sev[t] > SERIOUS_SEVERITY for t in truth])
    for T in NB_REPORT + ((T_fit,) if T_fit not in NB_REPORT else ()):
        ps = E.nb(T)[:, serious_idx].sum(1)
        q = np.percentile(ps[nonser], [50, 75, 90, 95, 99])
        nb_deg.append({"T": T, **{f"p{int(x)}": round(float(v), 4) for x, v in zip([50, 75, 90, 95, 99], q)},
                       "mean": round(float(ps[nonser].mean()), 4), "serious_patients_mean": round(float(ps[~nonser].mean()), 4)})
    write_csv(OUT / "nb_degeneracy.csv", nb_deg)

    # ---- long table: (adult, serious c) with p >= 5%
    ii, kk = np.nonzero(E.P[:, serious_idx] >= BANDS[1])
    L = pd.DataFrame({"i": ii, "c": kk})
    L["cname"] = np.array(serious, object)[kk]
    cj = serious_idx[kk]
    L["p"] = E.P[ii, cj]
    L["y"] = E.Y[ii, cj]
    L["band"] = E.band[ii, cj]
    L["hcount"] = E.H[ii, cj]
    L["top1"] = np.array(conds, object)[E.top1[ii]]
    L["truth"] = truth[ii]
    L["ref"], L["in470"], L["in250"] = ref[ii], in470[ii], in250[ii]
    L["row"] = df.ROW.values[ii]
    for name in ("A", "M1p", "M1pp", "M1p_ant", "B", "M2"):
        L[f"rate_{name}"] = E.rate[name][ii, cj]
        L[f"n_{name}"] = E.n[name][ii, cj]
    for T in NB_REPORT:
        L[f"rate_NB{T:g}"] = E.nb(T)[ii, cj]
        L[f"n_NB{T:g}"] = np.inf
    L["rate_NBfit"] = E.nb(T_fit)[ii, cj]
    L["n_NBfit"] = np.inf
    qrows = np.where(in470 | in250)[0]
    RK = knn_rates(M, truth_idx, ref, qrows, len(conds))
    L["rate_KNN"] = RK[ii, cj]
    L["n_KNN"] = np.where(np.isnan(L.rate_KNN), 0, KNN_K)
    print(f"(patient, serious c) pairs with p >= {BANDS[1]:g}%: {len(L)}")

    p = L.p.values
    det_names = ["DXA_self", "A", "M1p", "M1pp", "M1p_ant", "B", "M2", "KNN"] + [f"NB{T:g}" for T in NB_REPORT] + ["NBfit"]
    L["rate_DXA_self"], L["n_DXA_self"] = p / 100, np.inf  # DXA judged by itself: never a red herring
    for name in det_names:
        L[f"rh_{name}"], L[f"und_{name}"] = rh_label(p, L[f"rate_{name}"].values, L[f"n_{name}"].values)
    L["rh_primary"], L["und_primary"] = L[f"rh_{PRIMARY}"], L[f"und_{PRIMARY}"]
    L["rh_M1p_and_A"] = L.rh_M1p & L.rh_A
    L["und_M1p_and_A"] = L.und_M1p | L.und_A
    L["rh_all3"] = L.rh_M1p & L.rh_A & L.rh_NBfit
    L["und_all3"] = L.und_M1p | L.und_A
    L.to_pickle(CACHE / "pairs.pkl")

    # ---- detector comparison
    comp = []
    for name in det_names + ["M1p_and_A", "all3"]:
        rh, und = L[f"rh_{name}"].values, L[f"und_{name}"].values
        for X in X_GRID:
            for pop, mask in (("population", np.ones(len(L), bool)), ("sample470", L.in470.values)):
                if name == "KNN" and pop == "population":
                    continue
                m = mask & (p >= X)
                mass = p[m].sum()
                wrong, right = m & ~L.y.values, m & L.y.values
                comp.append({
                    "detector": name, "X": X, "population": pop, "pairs": int(m.sum()),
                    "mass_share_rh": p[m & rh].sum() / mass if mass else np.nan,
                    "mass_share_undetermined": p[m & und].sum() / mass if mass else np.nan,
                    "wrong_mass_share_rh": p[wrong & rh].sum() / p[wrong].sum() if wrong.any() else np.nan,
                    "true_mass_share_rh": p[right & rh].sum() / p[right].sum() if right.any() else np.nan,
                    "true_pairs_labelled_rh": int((right & rh).sum()), "true_pairs": int(right.sum()),
                    "share_of_all_serious_mass_rh": p[m & rh].sum() / p[mask].sum(),
                    "agree_with_primary": float((rh == L.rh_primary.values)[m].mean()),
                })
    comp = pd.DataFrame(comp)
    write_csv(OUT / "detector_comparison.csv", comp)

    cell_sizes = []
    for name in ("A", "M1p", "M1pp", "M1p_ant", "B"):
        sub = cells[cells.detector == name]
        m = p >= X_MAIN
        cell_sizes.append({"detector": name, "cells": len(sub), "cells_ge_min": int((~sub.suppressed).sum()), "median_n": float(sub.n.median()),
                           "pairs_ge_X_undetermined_share": float(L.loc[m, f"und_{name}"].mean())})
    cell_sizes = pd.DataFrame(cell_sizes)
    write_csv(OUT / "cell_sizes.csv", cell_sizes)

    # ---- per condition (population, X_MAIN)
    m = p >= X_MAIN
    per_cond = []
    for k, c in enumerate(serious):
        mc = L.c.values == k
        allmass = E.P[:, cidx[c]].sum()
        mm = mc & m
        rhm = mm & L.rh_primary.values
        top_prof = Counter()
        for t1, pp in zip(L.top1.values[rhm], p[rhm]):
            top_prof[t1] += pp
        dom = ", ".join(f"{t} {100 * v / p[rhm].sum():.0f}%" for t, v in top_prof.most_common(3)) if rhm.any() else ""
        row = {"condition": c, "severity": sev[c], "true_adults": int((truth == c).sum()),
               "pairs_ge_X": int(mm.sum()), "mass_ge_X": round(p[mm].sum() / 100, 1), "mass_all": round(allmass / 100, 1),
               "true_rate_ge_X": L.y.values[mm].mean() if mm.any() else np.nan, "mean_p_ge_X": p[mm].mean() / 100 if mm.any() else np.nan,
               "rh_pairs": int(rhm.sum()), "rh_mass_share_ge_X": p[rhm].sum() / p[mm].sum() if mm.any() else np.nan,
               "rh_mass_share_all": p[rhm].sum() / allmass if allmass else np.nan,
               "undetermined_share_ge_X": L.und_primary.values[mm].mean() if mm.any() else np.nan}
        for name in ("A", "M1p", "M1p_ant", "B", "NBfit", "NB1", "M1p_and_A"):
            row[f"rh_share_{name}"] = p[mm & L[f"rh_{name}"].values].sum() / p[mm].sum() if mm.any() else np.nan
        row["dominant_dxa_top1_in_rh"] = dom
        per_cond.append(row)
    per_cond = pd.DataFrame(per_cond).sort_values("rh_mass_share_ge_X", ascending=False)
    write_csv(OUT / "per_condition.csv", per_cond)

    prof = L[m & L.rh_primary.values].groupby(["top1", "cname"]).agg(pairs=("p", "size"), mass=("p", "sum"), mean_p=("p", "mean"),
                                                                     true_rate=("y", "mean"), hallmarks_mean=("hcount", "mean")).reset_index()
    prof["mass"] = prof.mass / 100
    prof = prof.sort_values("mass", ascending=False)
    write_csv(OUT / "dominant_profiles.csv", prof)

    # ---- known quirks
    quirk_rows = []
    for label, tops, c in QUIRKS:
        k = serious.index(c)
        mq = (L.c.values == k) & L.top1.isin(tops).values & m
        if not mq.any():
            quirk_rows.append({"quirk": label, "pairs_ge_X": 0})
            continue
        row = {"quirk": label, "pairs_ge_X": int(mq.sum()), "true_count": int(L.y.values[mq].sum()), "true_rate": L.y.values[mq].mean(),
               "mean_p": p[mq].mean() / 100, "hcount_mean": L.hcount.values[mq].mean()}
        for name in ("A", "M1p", "M1p_ant", "B", "NBfit", "NB1"):
            row[f"rate_{name}_mean"] = np.nanmean(L[f"rate_{name}"].values[mq])
        for name in ("primary", "A", "M1p", "M1p_ant", "B", "NBfit", "M1p_and_A"):
            row[f"rh_share_{name}"] = L[f"rh_{name}"].values[mq].mean()
        quirk_rows.append(row)
    write_csv(OUT / "quirks.csv", quirk_rows)

    # ---- sensitivity grid
    sens = []
    for rule, (fl, ra, up) in SENS_RULES.items():
        for mn in SENS_MIN_N:
            for X in X_GRID:
                labels = {n: rh_label(p, L[f"rate_{n}"].values, L[f"n_{n}"].values, fl, ra, mn, up)[0] for n in ("A", "M1p", "M1p_ant", "B", "NBfit")}
                labels["M1p_and_A"] = labels["M1p"] & labels["A"]
                for comb, rh in labels.items():
                    mm = p >= X
                    s4 = mm & L.in470.values
                    right = mm & L.y.values
                    sens.append({"rule": rule, "min_n": mn, "X": X, "detector": comb,
                                 "mass_share_rh_population": p[mm & rh].sum() / p[mm].sum(),
                                 "mass_share_rh_sample470": p[s4 & rh].sum() / p[s4].sum(),
                                 "true_pairs_labelled_rh": int((right & rh).sum()),
                                 "excused_at_20_after": excuse_count(L, sev, 20.0, rh)})
    sens = pd.DataFrame(sens)
    write_csv(OUT / "sensitivity.csv", sens)

    # ---- the 470 sample
    S = L[L.in470.values]
    case_rows = []
    for _, r in S.iterrows():
        case_rows.append({
            "case_id": f"ddxplus_{int(r.row)}", "condition": r.cname, "dxa_p": round(r.p / 100, 4), "truth": r.truth, "is_truth": bool(r.y),
            "band": band_label(int(r.band)), "hallmarks_present": int(r.hcount), "dxa_top1": r.top1,
            "empirical_rate": fnum(r[f"rate_{PRIMARY}"]), "cell_n": int(r[f"n_{PRIMARY}"]) if np.isfinite(r[f"n_{PRIMARY}"]) else "",
            "red_herring": bool(r.rh_primary and r.p >= X_MAIN), "undetermined": bool(r.und_primary and r.p >= X_MAIN),
            "rate_A": fnum(r.rate_A), "n_A": int(r.n_A), "rate_M1p": fnum(r.rate_M1p), "n_M1p": int(r.n_M1p),
            "rate_M1p_ant": fnum(r.rate_M1p_ant), "n_M1p_ant": int(r.n_M1p_ant), "rh_M1p_ant": bool(r.rh_M1p_ant and r.p >= X_MAIN),
            "rate_B": fnum(r.rate_B), "n_B": int(r.n_B), "rate_M2": fnum(r.rate_M2), "n_M2": int(r.n_M2),
            "rate_KNN": fnum(r.rate_KNN), "rate_NBfit": fnum(r.rate_NBfit), "rate_NB1": fnum(r.rate_NB1),
            "rh_A": bool(r.rh_A and r.p >= X_MAIN), "rh_M1p": bool(r.rh_M1p and r.p >= X_MAIN), "rh_B": bool(r.rh_B and r.p >= X_MAIN),
            "rh_NBfit": bool(r.rh_NBfit and r.p >= X_MAIN), "rh_NB1": bool(r.rh_NB1 and r.p >= X_MAIN),
            "rh_M1p_and_A": bool(r.rh_M1p_and_A and r.p >= X_MAIN),
        })
    write_csv(OUT / "sample470_cases.csv", case_rows)

    excuse_rows = []
    for t in EXCUSE_T:
        for name in ["primary", "M1p_and_A", "all3", "A", "M1p", "M1p_ant", "B", "M1pp", "M2", "NBfit", "NB0.25", "NB1"]:
            excuse_rows.append({"threshold": t, "detector": name, "nonserious_excused_before": excuse_count(L, sev, t, np.zeros(len(L), bool)),
                                "nonserious_excused_after": excuse_count(L, sev, t, L[f"rh_{name}"].values)})
    excuse_rows = pd.DataFrame(excuse_rows)
    write_csv(OUT / "excuse_rule.csv", excuse_rows)
    keep = excused_cases(L, sev, 20.0, L.rh_primary.values)
    ex_detail = []
    for i in sorted(excused_cases(L, sev, 20.0, np.zeros(len(L), bool))):
        Si = S[(S.i == i) & (S.p >= 20.0)]
        ex_detail.append({"case_id": f"ddxplus_{df.ROW.values[i]}", "truth": truth[i], "dxa_top1": df.DD.values[i][0][0],
                          "serious_ge_20": "; ".join(f"{c} {pp:.0f}% [{PRIMARY} {rt:.3f}, A {ra:.3f}, NBfit {rn:.3f}]{' RH' if rh else ''}"
                                                     for c, pp, rt, ra, rn, rh in zip(Si.cname, Si.p, Si[f"rate_{PRIMARY}"].fillna(-1), Si.rate_A.fillna(-1), Si.rate_NBfit, Si.rh_primary)),
                          "still_excused": i in keep})
    write_csv(OUT / "excuse_cases_470.csv", ex_detail)

    # atypical serious subset: serious truth, DXA top-1 benign
    n_serious470 = int(sum(sev[truth[i]] <= SERIOUS_SEVERITY for i in np.where(in470)[0]))
    atyp = []
    for i in np.where(in470)[0]:
        t = truth[i]
        top = df.DD.values[i][0][0]
        if sev[t] > SERIOUS_SEVERITY or sev[top] <= SERIOUS_SEVERITY:
            continue
        pt = E.P[i, cidx[t]]
        Si = S[(S.i == i) & (S.cname == t)]
        r = Si.iloc[0] if len(Si) else None
        others = S[(S.i == i) & (S.cname != t) & (S.p >= X_MAIN) & S.rh_primary]
        atyp.append({"case_id": f"ddxplus_{df.ROW.values[i]}", "truth": t, "dxa_top1": top, "p_top1": round(df.DD.values[i][0][1], 3),
                     "p_truth": round(pt / 100, 3), "truth_ge_X": bool(pt >= X_MAIN),
                     "truth_rank": [n for n, _ in df.DD.values[i]].index(t) + 1 if pt > 0 else "",
                     "rate_M1p": fnum(r.rate_M1p) if r is not None else "", "n_M1p": int(r.n_M1p) if r is not None else "",
                     "rate_A": fnum(r.rate_A) if r is not None else "", "rate_NBfit": fnum(r.rate_NBfit) if r is not None else "",
                     "truth_labelled_rh_primary": bool(r is not None and r.rh_primary and pt >= X_MAIN),
                     "truth_labelled_rh_M1p": bool(r is not None and r.rh_M1p and pt >= X_MAIN),
                     "truth_labelled_rh_M1p_ant": bool(r is not None and r.rh_M1p_ant and pt >= X_MAIN),
                     "truth_labelled_rh_A": bool(r is not None and r.rh_A and pt >= X_MAIN),
                     "truth_labelled_rh_B": bool(r is not None and r.rh_B and pt >= X_MAIN),
                     "truth_labelled_rh_NBfit": bool(r is not None and r.rh_NBfit and pt >= X_MAIN),
                     "truth_labelled_rh_NB1": bool(r is not None and r.rh_NB1 and pt >= X_MAIN),
                     "other_serious_rh": "; ".join(f"{c} {pp:.0f}%" for c, pp in zip(others.cname, others.p))})
    write_csv(OUT / "atypical_serious_470.csv", atyp)
    atyp_summary = {"serious_cases": n_serious470, "dxa_top_benign": len(atyp), "truth_ge_X": int(sum(a["truth_ge_X"] for a in atyp)),
                    **{f"truth_labelled_rh_{n}": int(sum(a[f"truth_labelled_rh_{n}"] for a in atyp)) for n in ("primary", "M1p", "A", "B", "NBfit", "NB1")},
                    "with_other_serious_rh": int(sum(bool(a["other_serious_rh"]) for a in atyp))}

    # ---- model flags on the 250-case v0 set
    model_rows, model_pairs = model_flags(L, df, sev, truth, in250)
    write_csv(OUT / "model_flags.csv", model_rows)
    write_csv(OUT / "model_flag_pairs.csv", model_pairs)

    # ---- summary
    prim = comp[(comp.detector == PRIMARY) & (comp.X == X_MAIN)]
    summary = {
        "adults": int(len(df)), "reference": int(ref.sum()), "serious_conditions": serious, "pairs_ge_5": int(len(L)),
        "pairs_ge_X": int((p >= X_MAIN).sum()), "X": X_MAIN, "rule": f"P(c) < max({FLOOR:.0%}, {RATIO:g} p), cell n >= {MIN_N}",
        "primary": PRIMARY, "checks": list(CHECKS), "T_fitted": T_fit,
        "mass_share_rh_population": float(prim[prim.population == "population"].mass_share_rh.iloc[0]),
        "mass_share_rh_sample470": float(prim[prim.population == "sample470"].mass_share_rh.iloc[0]),
        "share_of_all_serious_mass_rh_population": float(prim[prim.population == "population"].share_of_all_serious_mass_rh.iloc[0]),
        "true_pairs_labelled_rh": int(prim[prim.population == "population"].true_pairs_labelled_rh.iloc[0]),
        "selection": sel_tbl.to_dict("records"),
        "excuse": excuse_rows.to_dict("records"), "atypical_serious_470": atyp_summary,
        "nb_degeneracy": nb_deg, "cell_sizes": cell_sizes.to_dict("records"),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    (OUT / "summary.md").write_text(render(summary, comp, sel_tbl, pat_tbl, folds, per_cond, prof, quirk_rows, excuse_rows,
                                           model_rows, hm_rows, cell_sizes, sens, serious))
    print(json.dumps({k: v for k, v in summary.items() if k in ("mass_share_rh_population", "mass_share_rh_sample470",
                                                                 "share_of_all_serious_mass_rh_population", "true_pairs_labelled_rh",
                                                                 "atypical_serious_470", "T_fitted")}, indent=1, default=float))


def excused_cases(L, sev, t, rh) -> set[int]:
    """Non-serious sample-470 patients with a serious condition at DXA >= t that is not a red herring."""
    m = L.in470.values & (L.p.values >= t) & ~rh
    nonser = np.array([sev[x] > SERIOUS_SEVERITY for x in L.truth.values])
    return set(L.i.values[m & nonser].tolist())


def excuse_count(L, sev, t, rh) -> int:
    return len(excused_cases(L, sev, t, rh))


def model_flags(L, df, sev, truth, in250):
    """Per model: where the v0.1 top-5 severe flags land on the 250-case set."""
    cases = dp.load_cases()
    models = dp.load_models(cases, ConditionMap(), dp.load_offlist())
    row_of = {f"ddxplus_{r}": i for i, r in enumerate(df.ROW.values)}
    Lm = L[L.in250.values]
    look = {(int(i), c): (float(pp), bool(rh), bool(und)) for i, c, pp, rh, und in zip(Lm.i, Lm.cname, Lm.p, Lm.rh_primary, Lm.und_primary)}
    rows, pairs = [], []
    pair_counter = Counter()
    for name, mdl in sorted(models.items()):
        if mdl.get("kind") != "model":
            continue
        cnt = Counter()
        for ci, c in enumerate(cases):
            i = row_of.get(c["case_id"])
            if i is None or not mdl["readable"][ci]:
                continue
            for f in mdl["flags"][ci][:dp.DEFAULT_CAP]:
                if f is None or sev.get(f, 9) > SERIOUS_SEVERITY:
                    continue
                cnt["severe_flags"] += 1
                if f == truth[i]:
                    cnt["on_truth"] += 1
                    continue
                pp, rh, und = look.get((i, f), (0.0, False, False))
                if pp < X_MAIN:
                    cnt["dxa_below_X"] += 1
                elif rh:
                    cnt["red_herring"] += 1
                    pair_counter[(truth[i], f)] += 1
                elif und:
                    cnt["undetermined"] += 1
                else:
                    cnt["plausible_not_true"] += 1
        wrong = cnt["severe_flags"] - cnt["on_truth"]
        rows.append({"model": name, **{k: cnt[k] for k in ("severe_flags", "on_truth", "plausible_not_true", "red_herring", "undetermined", "dxa_below_X")},
                     "red_herring_share_of_wrong_severe": cnt["red_herring"] / wrong if wrong else np.nan,
                     "red_herring_share_of_dxa_supported_wrong": cnt["red_herring"] / (wrong - cnt["dxa_below_X"]) if wrong - cnt["dxa_below_X"] else np.nan,
                     "readable_cases": int(mdl["readable"].sum())})
    for (t, f), n in pair_counter.most_common(30):
        pairs.append({"truth": t, "flagged": f, "flags_across_models": n})
    return rows, pairs


def render(summary, comp, sel, pats, folds, per_cond, prof, quirks, excuse, model_rows, hm_rows, cell_sizes, sens, serious) -> str:
    o = ["# DXA red herrings: summary tables", ""]
    o.append(f"Adults {summary['adults']}, reference {summary['reference']}, (patient, serious c) pairs at p >= 5%: {summary['pairs_ge_5']}, at p >= {summary['X']:g}%: {summary['pairs_ge_X']}.")
    o.append(f"Primary detector {summary['primary']}, checks {summary['checks']}, rule {summary['rule']}, T fitted {summary['T_fitted']}.")
    o += ["", "## Estimator selection against real-world rates", "", md_table(sel), "", "### Per pattern", "",
          md_table(pats.drop(columns=[c for c in pats.columns if c.startswith("logratio_")]), ".4f"), "", "### Leave-one-pattern-out folds", "", md_table(folds), ""]
    o += ["## Detector comparison at X = 10%", "", md_table(comp[comp.X == X_MAIN]), ""]
    o += ["## Cell sizes", "", md_table(cell_sizes), ""]
    o += ["## Per condition (population, X = 10%, primary)", "", md_table(per_cond), ""]
    o += ["## Dominant (DXA top-1, condition) profiles among red herrings", "", md_table(prof.head(25)), ""]
    o += ["## Known quirks", "", md_table(pd.DataFrame(quirks)), ""]
    o += ["## Excuse rule on the 470", "", md_table(excuse), ""]
    o += ["## Atypical serious subset (470)", "", json.dumps(summary["atypical_serious_470"], indent=1), ""]
    o += ["## Model flags on the 250 (v0.1 top-5)", "", md_table(pd.DataFrame(model_rows)), ""]
    o += ["## Hallmarks (serious conditions)", "", md_table(pd.DataFrame([r for r in hm_rows if r["condition"] in serious]).drop(columns=["question"])), ""]
    o += ["## Naive-Bayes degeneracy: P(any serious) among non-serious adults", "", md_table(pd.DataFrame(summary["nb_degeneracy"])), ""]
    o += ["## Sensitivity", "", md_table(sens), ""]
    return "\n".join(o)


if __name__ == "__main__":
    main()
