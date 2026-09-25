"""Which DDXPlus-derived proxy tracks a patient's real-world risk of a serious condition?

Two candidate proxies for "flag this patient as possibly having a serious condition":
  (1) the true condition has DDXPlus severity <= 2 (evaluator.answer_key_v02.is_serious);
  (2) DXA's differential puts >= t of its mass on severity <= 2 conditions
      (evaluator.answer_key_v02.p_serious_risk, thresholds 5/10/12.5/20/50%).

We test them against real-world data in two ways, with no inference spend:

Part 1, published pre-test probabilities. We define presentation patterns in DDXPlus
evidence codes (chest pain, dyspnoea, haemoptysis, ...), compute on the full adult
DDXPlus test split (a) DXA's mean probability for the matching serious conditions and
(b) the true-condition rate the generator produced, and compare both with published
real-world rates for the same presentation (PUBLISHED below; every figure was read on
the cited page, see docs/risk-proxy-validation.md).

Part 2, CDC NHAMCS emergency department visits 2016-2022. We map NHAMCS reason-for-visit
codes and DDXPlus evidence codes to the same coarse presentation x age-band cells, then
ask how well each proxy's flag rate per cell tracks the real-world serious-outcome rate
per cell (admission, critical care, death, and a severity <= 2-type discharge diagnosis
built from spec/ddxplus_icd10_map.csv plus off-list serious categories from
spec/ddxplus_offlist_categories.csv).

Inputs:
  data/ddxplus_v0/release_test_patients, release_conditions.json
  data/external/nhamcs/extracted/*.dta, *.sas7bdat   (URLs and checksums: data/external/nhamcs/SHA256SUMS.txt)
  spec/ddxplus_icd10_map.csv, spec/ddxplus_offlist_categories.csv

Outputs, under results/analysis/risk_proxy/:
  patterns.csv           Part 1 per-pattern table (DXA mean, DDXPlus true rate, published rate, factors)
  published_rates.csv    every published rate checked, with DXA and DDXPlus rates on the same conditions
  age_gradient.csv       65+ over 18-39 ratios per presentation, real world vs proxies
  nhamcs_cells.csv       Part 2 per-cell table (NHAMCS outcomes, DDXPlus proxy flag rates)
  cell_correlations.csv  Spearman rank correlation of each proxy with each real-world outcome across cells
  calibration.csv        per-cell comparison of proxy flag rate with NHAMCS serious-diagnosis rate
  summary.md             the tables rendered for the doc
  cache/ddx_adult_test.pkl, cache/nhamcs_rfv.pkl   parsed inputs, safe to delete

Run: .venv/bin/python scripts/analysis/risk_proxy_validation.py
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
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evaluator import answer_key_v02 as ak  # noqa: E402

DDX_CSV = ROOT / "data/ddxplus_v0/release_test_patients"
COND_PATH = ROOT / "data/ddxplus_v0/release_conditions.json"
NHAMCS_RAW = ROOT / "data/external/nhamcs/extracted"
ICD_MAP = ROOT / "spec/ddxplus_icd10_map.csv"
OFFLIST = ROOT / "spec/ddxplus_offlist_categories.csv"
OUT = ROOT / "results/analysis/risk_proxy"
CACHE = OUT / "cache"

THRESHOLDS = [5.0, 10.0, 12.5, 20.0, 50.0]  # percent of DXA mass on severity <= 2
MAX_THRESHOLDS = [5.0, 10.0, 20.0, 50.0]  # percent on a single severity <= 2 condition (proxy 2 as first worded)
AGE_BANDS = [("18-39", 18, 39), ("40-64", 40, 64), ("65+", 65, 200)]
MIN_CELL_N = 30

# --------------------------------------------------------------------------------------
# Part 1: presentation patterns in DDXPlus evidence codes
# --------------------------------------------------------------------------------------
# Pain-location values that mean chest pain (release_evidences.json, E_55 value_meaning):
# V_29 lower chest, V_101 upper chest, V_55/V_56 side of the chest. E_14 is "chest pain even at rest".
CHEST = {"E_55_@_V_29", "E_55_@_V_101", "E_55_@_V_55", "E_55_@_V_56", "E_14"}


def has_rash(ev: set[str]) -> bool:
    """A rash with a colour other than NA (E_130), which DDXPlus only produces with a skin finding."""
    return any(e.startswith("E_130_@") for e in ev) and "E_130_@_V_11" not in ev


# name -> (description, predicate(evidence set), serious target conditions, comparison conditions)
# The targets are the severity <= 2 conditions the published rate speaks to; comparison
# conditions are non-serious conditions reported alongside (stable angina is severity 2
# in DDXPlus but level 3 in the evidence-based reference, so it is listed separately).
PATTERNS: dict[str, dict] = {
    "chest_pain": dict(
        desc="Chest pain (pain at lower/upper/side of chest, or chest pain at rest)",
        codes="E_55 in {V_29, V_101, V_55, V_56} or E_14",
        pred=lambda ev: bool(ev & CHEST),
        targets=["Possible NSTEMI / STEMI", "Unstable angina"],
        compare=["Stable angina"],
    ),
    "heartburn": dict(
        desc="Burning sensation rising from the stomach to the throat (reflux-type pain)",
        codes="E_173",
        pred=lambda ev: "E_173" in ev,
        targets=["Possible NSTEMI / STEMI", "Unstable angina"],
        compare=["GERD"],
    ),
    "pleuritic": dict(
        desc="Pain increased on deep inspiration",
        codes="E_220",
        pred=lambda ev: "E_220" in ev,
        targets=["Pulmonary embolism", "Spontaneous pneumothorax"],
        compare=["Pericarditis"],
    ),
    "dyspnoea": dict(
        desc="Shortness of breath (significant, or with minimal effort)",
        codes="E_66 or E_64",
        pred=lambda ev: "E_66" in ev or "E_64" in ev,
        targets=["Pulmonary embolism", "Acute pulmonary edema"],
        compare=["Pneumonia"],
    ),
    "haemoptysis": dict(
        desc="Coughing up blood",
        codes="E_45",
        pred=lambda ev: "E_45" in ev,
        targets=["Pulmonary embolism"],
        compare=["Pulmonary neoplasm", "Tuberculosis"],
    ),
    "sore_throat": dict(
        desc="Sore throat without chest pain or dyspnoea",
        codes="E_97 and not chest pain and not E_66",
        pred=lambda ev: "E_97" in ev and not (ev & CHEST) and "E_66" not in ev,
        targets=["Epiglottitis", "Possible NSTEMI / STEMI", "Unstable angina"],
        compare=[],
    ),
    "rash": dict(
        desc="Rash with a stated colour (DDXPlus skin finding)",
        codes="E_130 with value != V_11",
        pred=has_rash,
        targets=["Anaphylaxis"],
        compare=[],
    ),
    "fever_cough": dict(
        desc="Fever with cough",
        codes="E_91 and E_201",
        pred=lambda ev: "E_91" in ev and "E_201" in ev,
        targets=["Ebola", "Epiglottitis", "Acute pulmonary edema"],
        compare=["Pneumonia"],
    ),
    "palpitations": dict(
        desc="Palpitations (fast, irregular or missed beats)",
        codes="E_155",
        pred=lambda ev: "E_155" in ev,
        targets=["PSVT", "Possible NSTEMI / STEMI", "Myocarditis"],
        compare=["Atrial fibrillation"],
    ),
    "haematemesis": dict(
        desc="Vomited blood or coffee-ground material",
        codes="E_210",
        pred=lambda ev: "E_210" in ev,
        targets=["Boerhaave"],
        compare=[],
    ),
    "wheeze": dict(
        desc="Wheeze on exhaling, or noisy breathing after coughing",
        codes="E_214 or E_112",
        pred=lambda ev: "E_214" in ev or "E_112" in ev,
        targets=["Anaphylaxis", "Acute pulmonary edema", "Pulmonary embolism"],
        compare=["Bronchospasm / acute asthma exacerbation", "Acute COPD exacerbation / infection"],
    ),
}

# Published real-world rates per pattern and target group. Each entry: rate (fraction),
# setting, age, N, citation, URL. Filled from the literature check recorded in
# docs/risk-proxy-validation.md section 2; a pattern with no verified figure carries None.
PUBLISHED: dict[str, list[dict]] = {}  # populated at the bottom of the file


# --------------------------------------------------------------------------------------
# Part 2: NHAMCS reason-for-visit cells
# --------------------------------------------------------------------------------------
# RFV codes are the 4-digit detailed category plus one decimal digit, stored as 5 digits
# (1050.1 -> 10501). Names from the 2022 ED public-use documentation (data/external/nhamcs/doc/doc22.txt).
RFV_CELLS: dict[str, dict] = {
    "chest_pain": dict(rfv={10500, 10501, 10502, 10503}, label="1050.x Chest pain and related symptoms"),
    "dyspnoea": dict(rfv={14150, 14200}, label="1415.0 Shortness of breath; 1420.0 Labored or difficult breathing"),
    "cough": dict(rfv={14400}, label="1440.0 Cough"),
    "sore_throat": dict(rfv={14550, 14551, 14552, 14553, 14554, 14555, 14556}, label="1455.x Symptoms referable to throat"),
    "haemoptysis": dict(rfv={14701}, label="1470.1 Coughing up blood"),
    "fever": dict(rfv={10100}, label="1010.0 Fever"),
    "palpitations": dict(rfv={12600, 12601, 12602, 12603}, label="1260.x Abnormal pulsations and palpitations"),
    "wheeze": dict(rfv={14250}, label="1425.0 Wheezing"),
    "rash": dict(rfv={18600}, label="1860.0 Skin rash"),
    "heartburn": dict(rfv={15350}, label="1535.0 Heartburn and indigestion"),
    "haematemesis": dict(rfv={15802}, label="1580.2 Vomiting blood"),
    "swelling": dict(rfv={10351}, label="1035.1 Edema"),
}

# The same cells in DDXPlus evidence codes (any-listed, to match RFV1-3 any-listed).
DDX_CELLS: dict[str, callable] = {
    "chest_pain": lambda ev: bool(ev & CHEST),
    "dyspnoea": lambda ev: "E_66" in ev or "E_64" in ev,
    "cough": lambda ev: "E_201" in ev,
    "sore_throat": lambda ev: "E_97" in ev,
    "haemoptysis": lambda ev: "E_45" in ev,
    "fever": lambda ev: "E_91" in ev,
    "palpitations": lambda ev: "E_155" in ev,
    "wheeze": lambda ev: "E_214" in ev or "E_112" in ev,
    "rash": has_rash,
    "heartburn": lambda ev: "E_173" in ev,
    "haematemesis": lambda ev: "E_210" in ev,
    "swelling": lambda ev: "E_151" in ev,
}

# Off-list categories we count as a real-world serious (same-day, life-threatening)
# diagnosis. Chosen by the same standard as DDXPlus severity <= 2: care within hours.
OFFLIST_SERIOUS = {
    "Aortic aneurysm and dissection", "Appendicitis and bowel obstruction", "Aspiration pneumonitis",
    "CNS infection", "Endocarditis", "Foreign body in airway", "GI haemorrhage", "Heart failure",
    "Intracranial haemorrhage", "Ischaemic stroke and TIA", "Other arrhythmia and cardiac arrest",
    "Pancreatitis", "Pericardial effusion and tamponade", "Peritonsillar and deep neck abscess",
    "Pleural effusion and empyema", "Respiratory failure", "Sepsis", "Shock (non-anaphylactic)",
    "Tetanus and botulism", "Venous thromboembolism (DVT)", "Viral haemorrhagic and arboviral fever",
}

NHAMCS_COLS = ["YEAR", "AGE", "IMMEDR", "RFV1", "RFV2", "RFV3", "DIAG1", "DIAG2", "DIAG3", "DIAG4", "DIAG5",
               "ADMITHOS", "OBSHOS", "TRANOTH", "ADMIT", "DIEDED", "DOA", "PATWT"]
DIAGS = ["DIAG1", "DIAG2", "DIAG3", "DIAG4", "DIAG5"]
RFVS = ["RFV1", "RFV2", "RFV3"]


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def load_conditions() -> dict:
    return ak.load_conditions(COND_PATH)


def load_ddx_adults() -> pd.DataFrame:
    """Full DDXPlus test split, adults only, with evidences as a set and DXA differential as a dict."""
    cache = CACHE / "ddx_adult_test.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    df = pd.read_csv(DDX_CSV)
    df = df[df.AGE >= 18].reset_index(drop=True)
    df["EV"] = df.EVIDENCES.apply(lambda s: set(ast.literal_eval(s)))
    df["DD"] = df.DIFFERENTIAL_DIAGNOSIS.apply(lambda s: dict(ast.literal_eval(s)))
    df = df.drop(columns=["EVIDENCES", "DIFFERENTIAL_DIAGNOSIS"])
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_pickle(cache)
    return df


def load_nhamcs() -> pd.DataFrame:
    cache = CACHE / "nhamcs_rfv.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    frames = []
    for p in sorted(NHAMCS_RAW.iterdir()):
        if p.suffix == ".dta":
            df = pd.read_stata(p, columns=NHAMCS_COLS, convert_categoricals=False)
        elif p.suffix == ".sas7bdat":
            df = pd.read_sas(p, format="sas7bdat")[NHAMCS_COLS]
        else:
            continue
        for c in DIAGS:
            df[c] = df[c].apply(lambda v: v.decode() if isinstance(v, bytes) else v).astype(str).str.strip().str.upper()
        for c in NHAMCS_COLS:
            if c not in DIAGS:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["YEAR"] = df["YEAR"].astype(int)
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_pickle(cache)
    return df


def serious_code_sets(conditions: dict) -> tuple[set[str], set[str]]:
    """(on-list serious codes, off-list serious code prefixes), dots removed, upper case."""
    on = set()
    with ICD_MAP.open() as fh:
        for row in csv.DictReader(fh):
            if row["relation"] in ("equivalent", "narrower") and ak.is_serious(row["condition"], conditions):
                on.add(row["code"].replace(".", "").upper())
    off = set()
    with OFFLIST.open() as fh:
        for row in csv.DictReader(fh):
            if row["category"] in OFFLIST_SERIOUS:
                off.add(row["code_prefix"].replace(".", "").upper())
    return on, off


def diag_matches(diag: str, codes: set[str]) -> bool:
    """NHAMCS diagnoses carry 4 characters ('-' pads). A 3-character code matches its category;
    a longer code matches on its first 4 characters (same rule as nhamcs_urgency.py)."""
    if not diag or diag.startswith("ZZZ") or diag == "-9":
        return False
    d3, d4 = diag[:3], diag[:4]
    for c in codes:
        if len(c) == 3:
            if d3 == c:
                return True
        elif d4 == c[:4]:
            return True
    return False


# --------------------------------------------------------------------------------------
# Part 1
# --------------------------------------------------------------------------------------
def p_serious_max(d: dict, conditions: dict) -> float:
    """Largest DXA probability on any single severity <= 2 condition, as a fraction."""
    return max((float(p) for c, p in d.items() if ak.is_serious(c, conditions)), default=0.0)


def part1(ddx: pd.DataFrame, conditions: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, pub_rows = [], []
    for name, spec in PATTERNS.items():
        m = ddx.EV.apply(spec["pred"])
        sub = ddx[m]
        n = len(sub)
        p_serious = sub.DD.apply(lambda d: ak.p_serious_risk(d.items(), conditions) / 100.0)
        row = dict(pattern=name, description=spec["desc"], codes=spec["codes"], n=n,
                   ddx_true_serious=sub.PATHOLOGY.apply(lambda c: ak.is_serious(c, conditions)).mean(),
                   dxa_serious_mean=p_serious.mean())
        for t in THRESHOLDS:
            row[f"p2_flag_{t:g}"] = (p_serious >= t / 100).mean()
        p_max = sub.DD.apply(lambda d: p_serious_max(d, conditions))
        for t in MAX_THRESHOLDS:
            row[f"p2max_flag_{t:g}"] = (p_max >= t / 100).mean()
        tgt = spec["targets"]
        row["targets"] = "; ".join(tgt)
        row["dxa_target_mean"] = sub.DD.apply(lambda d: sum(d.get(t, 0.0) for t in tgt)).mean()
        row["ddx_true_target"] = sub.PATHOLOGY.isin(tgt).mean()
        for t in tgt + spec["compare"]:
            row[f"dxa[{t}]"] = sub.DD.apply(lambda d: d.get(t, 0.0)).mean()
            row[f"true[{t}]"] = (sub.PATHOLOGY == t).mean()
        contrib = defaultdict(float)
        for d in sub.DD:
            for c, p in d.items():
                if ak.is_serious(c, conditions):
                    contrib[c] += p
        top = sorted(contrib.items(), key=lambda kv: -kv[1])[:4]
        row["top_serious_dxa"] = "; ".join(f"{c} {100 * v / n:.1f}%" for c, v in top)
        for pub in PUBLISHED.get(name, []):
            dxa = sub.DD.apply(lambda d: sum(d.get(c, 0.0) for c in pub["conditions"])).mean()
            true = sub.PATHOLOGY.isin(pub["conditions"]).mean()
            pub_rows.append(dict(pattern=name, conditions="; ".join(pub["conditions"]), primary=bool(pub.get("use")),
                                 dxa_mean=dxa, ddx_true=true, published_rate=pub["rate"], scope=pub["scope"], setting=pub["setting"],
                                 cite=pub["cite"], quote=pub.get("quote", ""),
                                 factor_dxa=dxa / pub["rate"] if pub["rate"] else np.nan,
                                 factor_ddx=true / pub["rate"] if pub["rate"] else np.nan))
            if pub.get("use"):
                row["published_rate"] = pub["rate"]
                row["published_setting"] = pub["setting"]
                row["published_cite"] = pub["cite"].split(";")[0].split("(")[0].strip()
                row["published_scope"] = pub["scope"]
                row["published_conditions"] = "; ".join(pub["conditions"])
                row["dxa_on_published_conditions"] = dxa
                row["ddx_true_on_published_conditions"] = true
                row["factor_dxa_vs_published"] = dxa / pub["rate"] if pub["rate"] else np.nan
                row["factor_ddx_vs_published"] = true / pub["rate"] if pub["rate"] else np.nan
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(pub_rows)


# --------------------------------------------------------------------------------------
# Part 2
# --------------------------------------------------------------------------------------
def wmean(mask: np.ndarray, w: np.ndarray) -> float:
    return float(w[mask].sum() / w.sum()) if w.sum() else np.nan


def part2(ddx: pd.DataFrame, nh: pd.DataFrame, conditions: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    on, off = serious_code_sets(conditions)
    uniq = pd.unique(pd.concat([nh[c] for c in DIAGS]))
    on_set = {d for d in uniq if diag_matches(d, on)}
    off_set = {d for d in uniq if diag_matches(d, off)}
    nh = nh[nh.AGE >= 18].copy()
    nh["serious_on"] = False
    nh["serious_off"] = False
    for c in DIAGS:
        nh["serious_on"] |= nh[c].isin(on_set)
        nh["serious_off"] |= nh[c].isin(off_set)
    nh["serious_dx"] = nh.serious_on | nh.serious_off
    nh["admit"] = (nh.ADMITHOS == 1) | (nh.OBSHOS == 1) | (nh.TRANOTH == 1)
    nh["icu"] = nh.ADMIT == 1
    nh["died"] = (nh.DIEDED == 1) | (nh.DOA == 1)
    nh["imm12"] = nh.IMMEDR.between(1, 2)
    nh["triaged"] = nh.IMMEDR.between(1, 5)

    ddx = ddx.copy()
    ddx["serious"] = ddx.PATHOLOGY.apply(lambda c: ak.is_serious(c, conditions))
    ddx["p_serious"] = ddx.DD.apply(lambda d: ak.p_serious_risk(d.items(), conditions) / 100.0)
    ddx["p_max"] = ddx.DD.apply(lambda d: p_serious_max(d, conditions))

    rows = []
    for cell, spec in RFV_CELLS.items():
        nmask = np.zeros(len(nh), dtype=bool)
        for c in RFVS:
            nmask |= nh[c].isin(spec["rfv"]).to_numpy()
        dmask = ddx.EV.apply(DDX_CELLS[cell]).to_numpy()
        for band, lo, hi in AGE_BANDS + [("18+", 18, 200)]:
            nsub = nh[nmask & nh.AGE.between(lo, hi).to_numpy()]
            dsub = ddx[dmask & ddx.AGE.between(lo, hi).to_numpy()]
            w = nsub.PATWT.to_numpy()
            row = dict(cell=cell, rfv=spec["label"], age_band=band, nhamcs_n=len(nsub),
                       nhamcs_weighted_visits=float(w.sum()),
                       nh_admit=wmean(nsub.admit.to_numpy(), w), nh_icu=wmean(nsub.icu.to_numpy(), w),
                       nh_died=wmean(nsub.died.to_numpy(), w), nh_serious_dx=wmean(nsub.serious_dx.to_numpy(), w),
                       nh_serious_on=wmean(nsub.serious_on.to_numpy(), w), nh_serious_off=wmean(nsub.serious_off.to_numpy(), w))
            tri = nsub[nsub.triaged]
            row["nh_immedr12"] = wmean(tri.imm12.to_numpy(), tri.PATWT.to_numpy()) if len(tri) else np.nan
            row["ddx_n"] = len(dsub)
            row["p1_flag"] = dsub.serious.mean() if len(dsub) else np.nan
            row["dxa_serious_mean"] = dsub.p_serious.mean() if len(dsub) else np.nan
            for t in THRESHOLDS:
                row[f"p2_flag_{t:g}"] = (dsub.p_serious >= t / 100).mean() if len(dsub) else np.nan
            for t in MAX_THRESHOLDS:
                row[f"p2max_flag_{t:g}"] = (dsub.p_max >= t / 100).mean() if len(dsub) else np.nan
            rows.append(row)
    cells = pd.DataFrame(rows)

    # Rank correlations across cells (age-banded cells only, both sides with enough visits)
    proxies = ["p1_flag", "dxa_serious_mean"] + [f"p2_flag_{t:g}" for t in THRESHOLDS] + [f"p2max_flag_{t:g}" for t in MAX_THRESHOLDS]
    outcomes = ["nh_admit", "nh_icu", "nh_died", "nh_serious_dx", "nh_immedr12"]
    ok = cells[(cells.age_band != "18+") & (cells.nhamcs_n >= MIN_CELL_N) & (cells.ddx_n >= MIN_CELL_N)]
    corr_rows = []
    for p in proxies:
        for o in outcomes:
            sub = ok[[p, o]].dropna()
            rho, pval = spearmanr(sub[p], sub[o]) if len(sub) >= 4 else (np.nan, np.nan)
            corr_rows.append(dict(proxy=p, outcome=o, n_cells=len(sub), spearman_rho=rho, p_value=pval))
    for p in proxies:
        for o in outcomes:
            sub = cells[(cells.age_band == "18+") & (cells.nhamcs_n >= MIN_CELL_N)][[p, o]].dropna()
            rho, pval = spearmanr(sub[p], sub[o])
            corr_rows.append(dict(proxy=p, outcome=o, n_cells=len(sub), spearman_rho=rho, p_value=pval, scope="presentation (18+)"))
    corr = pd.DataFrame(corr_rows)
    corr["scope"] = corr["scope"].fillna("presentation x age band")

    # Age gradient: ratio of the 65+ cell to the 18-39 cell, per presentation, real world vs proxies
    grad_rows = []
    for cell in RFV_CELLS:
        young = cells[(cells.cell == cell) & (cells.age_band == "18-39")].iloc[0]
        old = cells[(cells.cell == cell) & (cells.age_band == "65+")].iloc[0]
        grad_rows.append(dict(cell=cell, nh_admit_ratio=old.nh_admit / young.nh_admit if young.nh_admit else np.nan,
                              nh_serious_dx_ratio=old.nh_serious_dx / young.nh_serious_dx if young.nh_serious_dx else np.nan,
                              p1_ratio=old.p1_flag / young.p1_flag if young.p1_flag else np.nan,
                              p2_12_5_ratio=old["p2_flag_12.5"] / young["p2_flag_12.5"] if young["p2_flag_12.5"] else np.nan,
                              dxa_mean_ratio=old.dxa_serious_mean / young.dxa_serious_mean))
    gradient = pd.DataFrame(grad_rows)

    # Calibration: how far each proxy's flag rate sits from the real-world serious-diagnosis rate, per cell
    cal_rows = []
    for p in proxies:
        sub = ok[[p, "nh_serious_dx", "nh_admit"]].dropna()
        ratio = sub[p] / sub.nh_serious_dx
        cal_rows.append(dict(proxy=p, n_cells=len(sub), median_ratio_to_serious_dx=float(ratio.median()),
                             cells_within_2x=int(((ratio >= 0.5) & (ratio <= 2)).sum()),
                             cells_over_2x=int((ratio > 2).sum()), cells_under_half=int((ratio < 0.5).sum()),
                             mean_abs_diff_serious_dx=float((sub[p] - sub.nh_serious_dx).abs().mean()),
                             mean_abs_diff_admit=float((sub[p] - sub.nh_admit).abs().mean())))
    calibration = pd.DataFrame(cal_rows)
    return cells, corr, gradient, calibration


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------
def pct(x: float) -> str:
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}%"


def render(patterns: pd.DataFrame, published: pd.DataFrame, cells: pd.DataFrame, corr: pd.DataFrame, gradient: pd.DataFrame, calibration: pd.DataFrame) -> str:
    L = ["# Risk proxy validation: computed tables", ""]
    L += ["## Part 1: DXA vs DDXPlus vs published, per presentation pattern", "",
          "| Pattern | Codes | N adults | Targets | DXA mean P(targets) | DDXPlus true rate (targets) | DDXPlus true serious | DXA mean serious mass | P2 sum @5% | @12.5% | @20% | @50% | P2 max @5% | @10% | @20% | @50% | Published | Factor DXA/pub | Factor DDX/pub |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in patterns.iterrows():
        pub = f"{pct(r['published_rate'])} ({r['published_scope']}; {r['published_cite']})" if pd.notna(r.get("published_rate", np.nan)) else "-"
        fd = f"{r['factor_dxa_vs_published']:.1f}x" if pd.notna(r.get("factor_dxa_vs_published", np.nan)) else "-"
        fx = f"{r['factor_ddx_vs_published']:.1f}x" if pd.notna(r.get("factor_ddx_vs_published", np.nan)) else "-"
        L.append(f"| {r.pattern} | {r.codes} | {r.n} | {r.targets} | {pct(r.dxa_target_mean)} | {pct(r.ddx_true_target)} | "
                 f"{pct(r.ddx_true_serious)} | {pct(r.dxa_serious_mean)} | {pct(r['p2_flag_5'])} | {pct(r['p2_flag_12.5'])} | "
                 f"{pct(r['p2_flag_20'])} | {pct(r['p2_flag_50'])} | {pct(r['p2max_flag_5'])} | {pct(r['p2max_flag_10'])} | "
                 f"{pct(r['p2max_flag_20'])} | {pct(r['p2max_flag_50'])} | {pub} | {fd} | {fx} |")
    L += ["", "### Published rates checked, with DXA and DDXPlus on the same conditions", "",
          "| Pattern | Conditions | DXA mean | DDXPlus true | Published | Scope | Setting | Factor DXA/pub | Factor DDX/pub | Source |", "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in published.iterrows():
        fd = f"{r.factor_dxa:.1f}x" if pd.notna(r.factor_dxa) else "-"
        fx = f"{r.factor_ddx:.1f}x" if pd.notna(r.factor_ddx) else "-"
        L.append(f"| {r.pattern}{' (primary)' if r.primary else ''} | {r.conditions} | {pct(r.dxa_mean)} | {pct(r.ddx_true)} | {pct(r.published_rate) if pd.notna(r.published_rate) else 'none'} | {r.scope} | {r.setting} | {fd} | {fx} | {r.cite} |")
    L += ["", "Top severity <= 2 conditions by mean DXA mass, per pattern:", ""]
    for _, r in patterns.iterrows():
        L.append(f"- {r.pattern}: {r.top_serious_dxa}")
    L += ["", "Per-condition detail (DXA mean / DDXPlus true rate):", ""]
    for _, r in patterns.iterrows():
        parts = [f"{c[4:-1]} {pct(r[c])} / {pct(r['true[' + c[4:-1] + ']'])}" for c in patterns.columns if c.startswith("dxa[") and pd.notna(r[c])]
        L.append(f"- {r.pattern}: " + "; ".join(parts))
    L += ["", "## Part 2: NHAMCS ED cells (adults, 2016-2022, visit-weighted) vs DDXPlus proxies", "",
          "| Cell | Age | NHAMCS n | Admit | ICU | Died | Serious dx (on-list) | Serious dx (any) | Triage 1-2 | DDX n | P1 flag | DXA serious mean | P2 sum @5% | @12.5% | @20% | @50% | P2 max @10% | @20% | @50% |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in cells.iterrows():
        L.append(f"| {r.cell} | {r.age_band} | {r.nhamcs_n} | {pct(r.nh_admit)} | {pct(r.nh_icu)} | {pct(r.nh_died)} | {pct(r.nh_serious_on)} | "
                 f"{pct(r.nh_serious_dx)} | {pct(r.nh_immedr12)} | {r.ddx_n} | {pct(r.p1_flag)} | {pct(r.dxa_serious_mean)} | "
                 f"{pct(r['p2_flag_5'])} | {pct(r['p2_flag_12.5'])} | {pct(r['p2_flag_20'])} | {pct(r['p2_flag_50'])} | "
                 f"{pct(r['p2max_flag_10'])} | {pct(r['p2max_flag_20'])} | {pct(r['p2max_flag_50'])} |")
    for scope in corr.scope.unique():
        L += ["", f"### Spearman rank correlation, {scope}", "",
              "| Proxy | " + " | ".join(corr.outcome.unique()) + " |", "|---|" + "---|" * corr.outcome.nunique()]
        for p in corr.proxy.unique():
            vals = []
            for o in corr.outcome.unique():
                r = corr[(corr.proxy == p) & (corr.outcome == o) & (corr.scope == scope)].iloc[0]
                vals.append("-" if pd.isna(r.spearman_rho) else f"{r.spearman_rho:.2f} (n={r.n_cells}, p={r.p_value:.2f})")
            L.append(f"| {p} | " + " | ".join(vals) + " |")
    L += ["", "### Age gradient: 65+ cell divided by 18-39 cell", "",
          "| Cell | NHAMCS admit | NHAMCS serious dx | P1 flag | P2 @12.5% | DXA serious mean |", "|---|---|---|---|---|---|"]
    for _, r in gradient.iterrows():
        L.append(f"| {r.cell} | {r.nh_admit_ratio:.1f}x | {r.nh_serious_dx_ratio:.1f}x | {r.p1_ratio:.1f}x | {r.p2_12_5_ratio:.1f}x | {r.dxa_mean_ratio:.1f}x |")
    L += ["", "### Calibration against the NHAMCS serious-diagnosis rate, per age-banded cell", "",
          "| Proxy | Cells | Median flag / serious-dx ratio | Within 2x | Over 2x | Under 0.5x | Mean abs diff vs serious dx | vs admit |", "|---|---|---|---|---|---|---|---|"]
    for _, r in calibration.iterrows():
        L.append(f"| {r.proxy} | {r.n_cells} | {r.median_ratio_to_serious_dx:.2f} | {r.cells_within_2x} | {r.cells_over_2x} | {r.cells_under_half} | {100 * r.mean_abs_diff_serious_dx:.1f} pts | {100 * r.mean_abs_diff_admit:.1f} pts |")
    return "\n".join(L) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    conditions = load_conditions()
    ddx = load_ddx_adults()
    print(f"DDXPlus adult test rows: {len(ddx)}")
    patterns, published = part1(ddx, conditions)
    patterns.to_csv(OUT / "patterns.csv", index=False)
    published.to_csv(OUT / "published_rates.csv", index=False)
    nh = load_nhamcs()
    print(f"NHAMCS rows: {len(nh)}")
    cells, corr, gradient, calibration = part2(ddx, nh, conditions)
    cells.to_csv(OUT / "nhamcs_cells.csv", index=False)
    corr.to_csv(OUT / "cell_correlations.csv", index=False)
    gradient.to_csv(OUT / "age_gradient.csv", index=False)
    calibration.to_csv(OUT / "calibration.csv", index=False)
    (OUT / "summary.md").write_text(render(patterns, published, cells, corr, gradient, calibration))
    print(f"wrote {OUT}")


# --------------------------------------------------------------------------------------
# Published rates (fraction of presenting patients with the target condition). One entry
# per pattern carries use=True and feeds the factor columns; the others are context.
# Every figure was read on the cited page during the 2026-09-24 literature check.
# --------------------------------------------------------------------------------------
PUBLISHED.update({
    "chest_pain": [
        dict(conditions=["Possible NSTEMI / STEMI", "Unstable angina"], rate=0.036, use=True, scope="ACS/MI",
             setting="primary care", cite="Haasenritter 2015, Croat Med J, PMID 26526879 (pooled range 1.5-3.6%); 3.6% in Haasenritter 2009, PMID 19883149, Marburg N=1212",
             quote="1.5 to 3.6% (acute coronary syndrome/myocardial infarction)"),
        dict(conditions=["Possible NSTEMI / STEMI", "Unstable angina"], rate=0.015, scope="ACS/MI, low end", setting="primary care",
             cite="Haasenritter 2015 pooled range low end; Klinkman 1994, PMID 8163958, 1.5%"),
        dict(conditions=["Possible NSTEMI / STEMI", "Unstable angina"], rate=0.13, scope="ACS", setting="US ED 2007-2008",
             cite="Bhuiya 2010, NCHS Data Brief 43", quote="from 23.6% in 1999-2000 to 13.0% in 2007-2008"),
        dict(conditions=["Possible NSTEMI / STEMI", "Unstable angina", "Stable angina"], rate=0.147, scope="any CHD", setting="primary care",
             cite="Haasenritter 2009, PMID 19883149: stable IHD 11.1% + ACS 3.6%"),
    ],
    "heartburn": [
        dict(conditions=["Possible NSTEMI / STEMI", "Unstable angina"], rate=None, use=True, scope="ACS", setting="-",
             cite="no published rate for burning-quality pain found; NHAMCS heartburn cell on-list serious dx 6.3% (this analysis)"),
    ],
    "pleuritic": [
        dict(conditions=["Pulmonary embolism"], rate=0.011, use=True, scope="PE", setting="UK ED, N=92",
             cite="Hall 1991, Arch Emerg Med, PMID 1854394", quote="Only one of the patients had a diagnosis of pulmonary embolus"),
        dict(conditions=["Pulmonary embolism"], rate=0.008, scope="PE among all ED chest pain", setting="French ED, N=881",
             cite="Le Gal 2020, Eur J Emerg Med, PMID 32097173", quote="7 cases ultimately diagnosed"),
    ],
    "dyspnoea": [
        dict(conditions=["Pulmonary embolism"], rate=0.012, use=True, scope="PE", setting="ANZ ED, ambulance arrivals, median age 74, N=1007",
             cite="Kelly 2016, Scand J Trauma Resusc Emerg Med, PMID 27658711, table 2", quote="12, 1.2 % (0.7-2.1 %)"),
        dict(conditions=["Pulmonary embolism", "Pulmonary neoplasm"], rate=0.005, scope="PE or neoplasm", setting="primary care",
             cite="Viniol 2015, BMC Fam Pract, PMID 26498502 (citing Okkes)", quote="Other pulmonary diseases (neoplasia, pulmonary embolism) 0.5% (0.3-0.8)"),
        dict(conditions=["Pulmonary embolism"], rate=0.12, scope="PE among patients the GP suspected of PE", setting="primary care, N=598",
             cite="Hendriksen 2015, BMJ, PMID 26349907", quote="72 pulmonary embolism ... (prevalence 12%)"),
        dict(conditions=["Acute pulmonary edema"], rate=0.203, scope="cardiac failure", setting="ANZ ED, ambulance arrivals",
             cite="Kelly 2016, table 2", quote="204, 20.3 % (17.9-22.9 %)"),
        dict(conditions=["Spontaneous pneumothorax"], rate=0.004, scope="pneumothorax", setting="ANZ ED, ambulance arrivals",
             cite="Kelly 2016, table 2", quote="4, 0.4 % (0.2-1 %)"),
    ],
    "haemoptysis": [
        dict(conditions=["Pulmonary embolism"], rate=0.026, use=True, scope="PE", setting="French hospital admissions for haemoptysis, ~15000/yr",
             cite="Abdulmalak 2015, Eur Respir J, PMID 26022949", quote="tuberculosis (2.7%), pulmonary embolism (2.6%)"),
        dict(conditions=["Pulmonary neoplasm"], rate=0.049, scope="lung cancer within 90 days, men", setting="UK primary care, N=4812 first episodes, mean age 54.5",
             cite="Jones 2009, BMJ, PMID 19679615, table 2", quote="4.9 (4.1 to 5.7)"),
        dict(conditions=["Pulmonary neoplasm"], rate=0.028, scope="lung cancer within 90 days, women", setting="UK primary care",
             cite="Jones 2009, BMJ, table 2", quote="2.8 (2.1 to 3.7)"),
        dict(conditions=["Pulmonary neoplasm"], rate=0.174, scope="lung cancer", setting="French hospital admissions",
             cite="Abdulmalak 2015", quote="lung cancer (17.4%)"),
        dict(conditions=["Tuberculosis"], rate=0.027, scope="TB", setting="French hospital admissions", cite="Abdulmalak 2015", quote="tuberculosis (2.7%)"),
    ],
    "sore_throat": [
        dict(conditions=["Epiglottitis"], rate=None, use=True, scope="epiglottitis", setting="-",
             cite="adult incidence 3.1 per 100,000 per year (Berger 2003, Am J Otolaryngol, PMID 14608569, Israel 1996-2000); no consultation denominator found. NHAMCS sore-throat cell on-list serious dx 0.6% (this analysis)"),
    ],
    "rash": [
        dict(conditions=["Anaphylaxis"], rate=0.01, use=True, scope="anaphylaxis among allergy-related ED visits", setting="US ED (NHAMCS 1993-2004), 12.4M visits",
             cite="Gaeta 2007, Ann Allergy Asthma Immunol, PMID 17458433", quote="Anaphylaxis coding was rare (1%)"),
    ],
    "fever_cough": [
        dict(conditions=["Pneumonia"], rate=0.05, use=True, scope="radiographic pneumonia among acute cough (all, fever not split)", setting="European primary care, N=2820, mean age 50",
             cite="van Vugt 2013, BMJ, PMID 23633005", quote="140 (5%) had pneumonia"),
    ],
    "palpitations": [
        dict(conditions=["PSVT"], rate=0.088, use=True, scope="clinically relevant arrhythmia (all types; PSVT is a subset)", setting="Dutch general practice, N=762",
             cite="Zwietering 1998, Fam Pract, PMID 9792350", quote="In 28.3% of the patients, arrhythmias were detected and 8.8% were clinically relevant"),
    ],
    "haematemesis": [
        dict(conditions=["Boerhaave"], rate=0.0, use=True, scope="Boerhaave (oesophageal perforation of any cause is 3.1 per million per year)", setting="population",
             cite="Aburumman 2025, Surg Endosc, PMID 40854992", quote="an incidence of approximately 3.1 per million annually"),
    ],
    "wheeze": [
        dict(conditions=["Pulmonary embolism"], rate=0.059, use=True, scope="PE among hospitalised COPD exacerbations", setting="7 French hospitals, N=740",
             cite="Couturaud 2021, JAMA, PMID 33399840", quote="pulmonary embolism was detected in 5.9% of patients"),
    ],
})


if __name__ == "__main__":
    main()
