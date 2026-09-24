"""Derive a per-condition urgency key for the 49 DDXPlus conditions from CDC NHAMCS ED microdata.

Why: the benchmark's urgency answer key should rest on a third-party, reproducible
source rather than our own condition-by-condition reading of NHS.uk, CTAS and NICE
(docs/third-party-urgency-source.md). NHAMCS records the nurse triage immediacy
(IMMEDR, 1 immediate to 5 nonurgent) and the disposition of a weighted national
sample of US emergency department visits, coded in ICD-10-CM from survey year
2016 onward.

Inputs (all public, downloaded by hand; URLs and checksums in data/external/nhamcs/SHA256SUMS.txt):
  data/external/nhamcs/extracted/ED2016-stata.dta ... ed2021-stata.dta, ed2022_sas.sas7bdat
  spec/ddxplus_icd10_map.csv          condition -> ICD-10 codes (equivalent and narrower only)
  spec/acuity_reference_levels.csv    DDXPlus severity and our NTS translation, for agreement stats
  data/external/nyu_eda/nyu_ed_algorithm_icd10_2025-05-05.xlsx  NYU ED Algorithm, for comparison

Outputs, under results/analysis/nhamcs_urgency/:
  condition_summary.csv   one row per condition x scope (primary/any diagnosis) x age band
  derived_levels.csv      one row per condition: the derived level, its inputs, DDXPlus and NTS levels
  nyu_eda_levels.csv      one row per condition: NYU EDA category shares over the condition's codes
  agreement.json          kappa, exact, within-one and urgent-line agreement
  summary.md              the tables above rendered for the doc

Run: python scripts/analysis/nhamcs_urgency.py
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "external" / "nhamcs" / "extracted"
NYU_XLSX = ROOT / "data" / "external" / "nyu_eda" / "nyu_ed_algorithm_icd10_2025-05-05.xlsx"
MAP = ROOT / "spec" / "ddxplus_icd10_map.csv"
LEVELS = ROOT / "spec" / "acuity_reference_levels.csv"
OUT = ROOT / "results" / "analysis" / "nhamcs_urgency"

COLS = ["YEAR", "AGE", "SEX", "IMMEDR", "DIAG1", "DIAG2", "DIAG3", "DIAG4", "DIAG5",
        "ADMITHOS", "OBSHOS", "TRANOTH", "ADMIT", "DIEDED", "DOA", "PATWT"]
DIAGS = ["DIAG1", "DIAG2", "DIAG3", "DIAG4", "DIAG5"]
MIN_N = 30  # unweighted visits below this are flagged as too few
AGE_BANDS = {"all": (0, 200), "0-17": (0, 17), "18-64": (18, 64), "65+": (65, 200)}
URGENT_MAX_LEVEL = 3  # spec/v0.2-scoring.md: true level 1-3 is urgent

# Derivation rule (one rule, no per-condition judgement): the condition's level is the
# visit-weighted mean triage immediacy of ED visits carrying the condition as the
# primary diagnosis, rounded to the nearest integer. Immediacy 1-5 is NHAMCS IMMEDR,
# collected on the ESI scale by the triage nurse at arrival.
RULE = "round(weighted mean IMMEDR, primary diagnosis, all ages)"


def load_year(path: Path) -> pd.DataFrame:
    if path.suffix == ".dta":
        df = pd.read_stata(path, columns=COLS, convert_categoricals=False)
    else:
        df = pd.read_sas(path, format="sas7bdat")[COLS]
    for c in DIAGS:
        df[c] = df[c].apply(lambda v: v.decode() if isinstance(v, bytes) else v).astype(str).str.strip().str.upper()
    for c in COLS:
        if c not in DIAGS:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_all() -> pd.DataFrame:
    frames = [load_year(p) for p in sorted(RAW.iterdir()) if p.suffix in (".dta", ".sas7bdat")]
    df = pd.concat(frames, ignore_index=True)
    df["YEAR"] = df["YEAR"].astype(int)
    return df


def condition_codes() -> dict[str, list[str]]:
    """Codes per condition, equivalent and narrower relations only (spec/v0.2-scoring.md section on matching)."""
    codes: dict[str, set[str]] = defaultdict(set)
    with MAP.open() as fh:
        for row in csv.DictReader(fh):
            if row["relation"] in ("equivalent", "narrower"):
                codes[row["condition"]].add(row["code"].replace(".", "").upper())
    return {k: sorted(v) for k, v in codes.items()}


def diag_matches(diag: str, codes: list[str]) -> bool:
    """NHAMCS public-use diagnoses carry 4 characters (3-digit category plus first subdigit,
    '-' when masked). A 3-character map code matches on the category; a longer code matches
    on its first 4 characters."""
    if not diag or diag.startswith("ZZZ") or diag == "-9":
        return False
    for c in codes:
        if len(c) == 3:
            if diag[:3] == c:
                return True
        elif diag[:4] == c[:4]:
            return True
    return False


def match_masks(df: pd.DataFrame, codes: dict[str, list[str]]) -> dict[str, dict[str, np.ndarray]]:
    """Per condition: boolean masks for primary (DIAG1) and any-listed (DIAG1-5) matches."""
    uniq = pd.unique(pd.concat([df[c] for c in DIAGS]))
    lookup = {cond: {d for d in uniq if diag_matches(d, cs)} for cond, cs in codes.items()}
    out = {}
    for cond, dset in lookup.items():
        prim = df["DIAG1"].isin(dset).to_numpy()
        anym = prim.copy()
        for c in DIAGS[1:]:
            anym |= df[c].isin(dset).to_numpy()
        out[cond] = {"primary": prim, "any": anym}
    return out


def wshare(values: np.ndarray, w: np.ndarray, level: int) -> float:
    return float(w[values == level].sum() / w.sum()) if w.sum() else float("nan")


def summarise(sub: pd.DataFrame) -> dict:
    """Weighted immediacy distribution and outcome rates for one subset of visits."""
    n = len(sub)
    tri = sub[sub["IMMEDR"].between(1, 5)]
    w = tri["PATWT"].to_numpy()
    imm = tri["IMMEDR"].to_numpy()
    row = {"n_unweighted": n, "n_triaged": len(tri), "weighted_visits": float(sub["PATWT"].sum())}
    shares = {k: wshare(imm, w, k) for k in range(1, 6)}
    for k in range(1, 6):
        row[f"p_immedr_{k}"] = shares[k]
    row["p_immedr_le2"] = shares[1] + shares[2] if len(tri) else float("nan")
    row["p_immedr_le3"] = row["p_immedr_le2"] + shares[3] if len(tri) else float("nan")
    row["wmean_immedr"] = float((imm * w).sum() / w.sum()) if w.sum() else float("nan")
    row["modal_immedr"] = int(max(shares, key=shares.get)) if len(tri) else None
    wa = sub["PATWT"].to_numpy()
    admitted = ((sub["ADMITHOS"] == 1) | (sub["OBSHOS"] == 1) | (sub["TRANOTH"] == 1)).to_numpy()
    icu = (sub["ADMIT"] == 1).to_numpy()
    died = ((sub["DIEDED"] == 1) | (sub["DOA"] == 1)).to_numpy()
    for name, m in (("admit_rate", admitted), ("icu_rate", icu), ("death_rate", died)):
        row[name] = float(wa[m].sum() / wa.sum()) if wa.sum() else float("nan")
    row["too_few"] = n < MIN_N
    return row


def derive_level(wmean: float, n: int) -> int | None:
    if n < MIN_N or not np.isfinite(wmean):
        return None
    return int(np.floor(wmean + 0.5))


def nyu_levels(codes: dict[str, list[str]]) -> list[dict]:
    """Average NYU EDA category shares over each condition's codes (unweighted over codes)."""
    import openpyxl

    wb = openpyxl.load_workbook(NYU_XLSX, read_only=True, data_only=True)
    ws = wb.worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    hdr = [str(h) for h in rows[0]]
    table = {}
    for r in rows[1:]:
        if r[0] is None:
            continue
        vals = [float(v) if isinstance(v, (int, float)) else float("nan") for v in r[2:]]
        table[str(r[0]).replace(".", "").upper()] = vals
    cats = hdr[2:]
    out = []
    for cond, cs in codes.items():
        hits = []
        for c in cs:
            hits += [v for k, v in table.items() if k.startswith(c)]
        row = {"condition": cond, "n_nyu_codes": len(hits)}
        if hits:
            arr = np.nanmean(np.array(hits), axis=0)
            for cat, v in zip(cats, arr):
                row[cat] = round(float(v), 3)
            row["p_ed_care_needed"] = round(float(arr[2] + arr[3]), 3)
            row["p_unclassified_or_special"] = round(float(arr[4:].sum()), 3)
        out.append(row)
    return out


def quadratic_weighted_kappa(a: list[int], b: list[int], k: int = 5) -> float:
    a, b = np.asarray(a) - 1, np.asarray(b) - 1
    o = np.zeros((k, k))
    for i, j in zip(a, b):
        o[i, j] += 1
    w = np.array([[(i - j) ** 2 / (k - 1) ** 2 for j in range(k)] for i in range(k)])
    e = np.outer(o.sum(1), o.sum(0)) / o.sum()
    return float(1 - (w * o).sum() / (w * e).sum())


def agreement(derived: dict[str, int | None], ref: dict[str, int], name: str) -> dict:
    conds = [c for c, v in derived.items() if v is not None and c in ref]
    a = [derived[c] for c in conds]
    b = [ref[c] for c in conds]
    diff = np.abs(np.array(a) - np.array(b))
    ua = np.array(a) <= URGENT_MAX_LEVEL
    ub = np.array(b) <= URGENT_MAX_LEVEL
    return {
        "against": name,
        "n": len(conds),
        "qwk": round(quadratic_weighted_kappa(a, b), 3),
        "exact": int((diff == 0).sum()),
        "within_one": int((diff <= 1).sum()),
        "urgent_line_agree": int((ua == ub).sum()),
        "ref_urgent_kept": int((ua & ub).sum()),
        "ref_urgent_total": int(ub.sum()),
        "ref_nonurgent_marked_urgent": int((ua & ~ub).sum()),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_all()
    codes = condition_codes()
    masks = match_masks(df, codes)
    levels = {r["condition"]: r for r in csv.DictReader(LEVELS.open())}
    # The map spells one condition differently from the levels file.
    alias = {"Larygospasm": "Larygospasm"}

    summary_rows = []
    derived_rows = []
    for cond in sorted(codes):
        for scope in ("primary", "any"):
            for band, (lo, hi) in AGE_BANDS.items():
                m = masks[cond][scope] & df["AGE"].between(lo, hi).to_numpy()
                row = {"condition": cond, "scope": scope, "age_band": band}
                row.update(summarise(df[m]))
                summary_rows.append(row)
        prim = next(r for r in summary_rows if r["condition"] == cond and r["scope"] == "primary" and r["age_band"] == "all")
        adult = next(r for r in summary_rows if r["condition"] == cond and r["scope"] == "primary" and r["age_band"] == "18-64")
        older = next(r for r in summary_rows if r["condition"] == cond and r["scope"] == "primary" and r["age_band"] == "65+")
        anyr = next(r for r in summary_rows if r["condition"] == cond and r["scope"] == "any" and r["age_band"] == "all")
        lv = levels.get(alias.get(cond, cond))
        derived_rows.append({
            "condition": cond,
            "codes_4char": " ".join(sorted({c[:4] for c in codes[cond]})),
            "n_primary": prim["n_unweighted"],
            "n_primary_triaged": prim["n_triaged"],
            "n_any": anyr["n_unweighted"],
            "wmean_immedr_primary": round(prim["wmean_immedr"], 2) if np.isfinite(prim["wmean_immedr"]) else None,
            "p_immedr_le2_primary": round(prim["p_immedr_le2"], 3) if np.isfinite(prim["p_immedr_le2"]) else None,
            "p_immedr_le3_primary": round(prim["p_immedr_le3"], 3) if np.isfinite(prim["p_immedr_le3"]) else None,
            "modal_immedr_primary": prim["modal_immedr"],
            "admit_rate_primary": round(prim["admit_rate"], 3) if np.isfinite(prim["admit_rate"]) else None,
            "icu_rate_primary": round(prim["icu_rate"], 3) if np.isfinite(prim["icu_rate"]) else None,
            "death_rate_primary": round(prim["death_rate"], 4) if np.isfinite(prim["death_rate"]) else None,
            "wmean_immedr_any": round(anyr["wmean_immedr"], 2) if np.isfinite(anyr["wmean_immedr"]) else None,
            "admit_rate_any": round(anyr["admit_rate"], 3) if np.isfinite(anyr["admit_rate"]) else None,
            "level_nhamcs": derive_level(prim["wmean_immedr"], prim["n_unweighted"]),
            "level_nhamcs_any": derive_level(anyr["wmean_immedr"], anyr["n_unweighted"]),
            # Fallback: any-listed diagnosis when the primary-diagnosis count is under MIN_N.
            "level_nhamcs_fallback": derive_level(prim["wmean_immedr"], prim["n_unweighted"])
            if prim["n_unweighted"] >= MIN_N else derive_level(anyr["wmean_immedr"], anyr["n_unweighted"]),
            "n_18_64": adult["n_unweighted"],
            "level_18_64": derive_level(adult["wmean_immedr"], adult["n_unweighted"]),
            "n_65plus": older["n_unweighted"],
            "level_65plus": derive_level(older["wmean_immedr"], older["n_unweighted"]),
            "ddxplus_severity": int(lv["ddxplus_severity"]) if lv else None,
            "nts_level": int(lv["scale_level"]) if lv else None,
            "too_few": prim["n_unweighted"] < MIN_N,
        })

    pd.DataFrame(summary_rows).to_csv(OUT / "condition_summary.csv", index=False)
    derived = pd.DataFrame(derived_rows)
    derived.to_csv(OUT / "derived_levels.csv", index=False)
    nyu = nyu_levels(codes)
    pd.DataFrame(nyu).to_csv(OUT / "nyu_eda_levels.csv", index=False)

    dl = {r["condition"]: r["level_nhamcs"] for r in derived_rows}
    fb = {r["condition"]: r["level_nhamcs_fallback"] for r in derived_rows}
    ddx = {r["condition"]: r["ddxplus_severity"] for r in derived_rows if r["ddxplus_severity"]}
    nts = {r["condition"]: r["nts_level"] for r in derived_rows if r["nts_level"]}
    stats = {
        "rule": RULE,
        "min_n": MIN_N,
        "years": sorted(df["YEAR"].unique().tolist()),
        "visits_total": int(len(df)),
        "visits_triaged": int(df["IMMEDR"].between(1, 5).sum()),
        "covered": int(sum(v is not None for v in dl.values())),
        "not_covered": sorted(c for c, v in dl.items() if v is None),
        "level_counts": {str(k): int(v) for k, v in pd.Series([v for v in dl.values() if v]).value_counts().sort_index().items()},
        "vs_ddxplus_severity": agreement(dl, ddx, "DDXPlus severity"),
        "vs_nts_translation": agreement(dl, nts, "NTS translation (spec/acuity_reference_levels.csv)"),
        "ddxplus_vs_nts": agreement({c: ddx[c] for c in ddx}, nts, "DDXPlus severity vs NTS (reference)"),
        "fallback_covered": int(sum(v is not None for v in fb.values())),
        "fallback_vs_ddxplus_severity": agreement(fb, ddx, "DDXPlus severity (fallback scope)"),
        "fallback_vs_nts_translation": agreement(fb, nts, "NTS translation (fallback scope)"),
    }
    (OUT / "agreement.json").write_text(json.dumps(stats, indent=2))
    write_summary_md(derived, nyu, stats)
    print(json.dumps({k: v for k, v in stats.items() if k != "not_covered"}, indent=1))
    print("not covered:", stats["not_covered"])


def write_summary_md(derived: pd.DataFrame, nyu: list[dict], stats: dict) -> None:
    lines = [f"# NHAMCS-derived urgency levels ({', '.join(map(str, stats['years']))})", "",
             f"Rule: {stats['rule']}. Conditions with fewer than {stats['min_n']} primary-diagnosis visits get no level.", "",
             "| Condition | N primary | N any | Mean IMMEDR | P(<=2) | P(<=3) | Admit | ICU | Died | Level | 18-64 (N) | 65+ (N) | DDX | NTS |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in derived.iterrows():
        f = lambda v, d=2: "" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{v:.{d}f}" if isinstance(v, float) else str(v))
        lv = lambda v: "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(int(v))
        lines.append(f"| {r['condition']} | {r['n_primary']} | {r['n_any']} | {f(r['wmean_immedr_primary'])} | {f(r['p_immedr_le2_primary'])} | {f(r['p_immedr_le3_primary'])} | {f(r['admit_rate_primary'])} | {f(r['icu_rate_primary'])} | {f(r['death_rate_primary'],4)} | {lv(r['level_nhamcs'])} | {lv(r['level_18_64'])} ({r['n_18_64']}) | {lv(r['level_65plus'])} ({r['n_65plus']}) | {lv(r['ddxplus_severity'])} | {lv(r['nts_level'])} |")
    lines += ["", "## Agreement", "", "| Against | N | QWK | Exact | Within one | Urgent-line agree | Ref urgent kept | Ref non-urgent marked urgent |", "|---|---|---|---|---|---|---|---|"]
    for k in ("vs_ddxplus_severity", "vs_nts_translation", "ddxplus_vs_nts", "fallback_vs_ddxplus_severity", "fallback_vs_nts_translation"):
        a = stats[k]
        lines.append(f"| {a['against']} | {a['n']} | {a['qwk']} | {a['exact']} | {a['within_one']} | {a['urgent_line_agree']} | {a['ref_urgent_kept']} of {a['ref_urgent_total']} | {a['ref_nonurgent_marked_urgent']} |")
    lines += ["", "## NYU ED Algorithm category shares (mean over the condition's ICD-10-CM codes)", "",
              "| Condition | Codes | Non-emergent | Emergent, PC treatable | ED needed, preventable | ED needed, not preventable | Injury/Psych/Alcohol/Drug/Unclassified |", "|---|---|---|---|---|---|---|"]
    for r in nyu:
        if r["n_nyu_codes"]:
            lines.append(f"| {r['condition']} | {r['n_nyu_codes']} | {r['Non_Emergent']} | {r['Emergent__PC_Treatable']} | {r['ED_Care_Needed__Preventable_Avoi']} | {r['ED_Care_Needed__not_Preventable']} | {r['p_unclassified_or_special']} |")
        else:
            lines.append(f"| {r['condition']} | 0 | | | | | |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
