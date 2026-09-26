"""Derive "dangerous if missed" tiers for ICD-10-CM codes outside the 49 DDXPlus conditions from CDC NHAMCS.

Why: models flag and list codes the DDXPlus map does not cover (aortic dissection I71, stroke I63,
COVID-19 U07.1, back pain M54, ...). Arm 4 of the v0.3 A/B scorer needs a tier for each such code
that rests on a third-party source, because a per-code judgement by us would be neither
reproducible nor defensible (docs/offlist-severity-nhamcs.md).

Method, in three steps:
  1. From NHAMCS ED visits 2016-2022, adults 18+, we compute weighted admission, critical-care and
     ED-death rates per ICD-10-CM code (4 characters, as the public file carries them) and per
     3-character group, for the code as primary diagnosis and as any listed diagnosis.
  2. On the DDXPlus conditions NHAMCS covers, we pick one threshold rule on those rates that best
     agrees with spec/dangerous_if_missed_tiers_v03b.csv, preferring round thresholds.
  3. We apply that rule to every off-list group, after two overrides: codes in a Newman-Toker 2023
     Table 1 or AHRQ 2022 ED top-15 serious-harm group are tier 1; symptom (R), factor (Z),
     external-cause (V-Y) and under-30-visit codes are "unscored". The rule reads primary-diagnosis
     rates only: a code listed as a secondary diagnosis of an admitted patient says little about the
     code as the reason for the visit, so a prefix under 30 primary visits is unscored.

Inputs:
  data/external/nhamcs/extracted/*.dta, *.sas7bdat   NHAMCS ED microdata (checksums in data/external/nhamcs/SHA256SUMS.txt)
  data/external/icd10cm/icd10cm_order_2026.txt        CMS ICD-10-CM FY2026 order file, for descriptions (optional;
                                                      falls back to the NYU EDA sheet, then to no description)
  spec/ddxplus_icd10_map.csv, spec/dangerous_if_missed_tiers_v03b.csv, spec/offlist_escalation_groups.csv
  results/v03/ab/runs/*v7a*.json, results/v03/runs/*.json   emitted codes, for the coverage report

Outputs:
  results/analysis/nhamcs_offlist/outcomes_by_code.csv   step 1 (rows under 30 unweighted visits are dropped)
  results/analysis/nhamcs_offlist/calibration.json       step 2: rule search, confusion table, leave-one-out
  results/analysis/nhamcs_offlist/overlap_conditions.csv step 2: per-condition rates and tiers
  results/analysis/nhamcs_offlist/coverage.json          step 4: emitted-code coverage and watchlist checks
  spec/offlist_tiers_nhamcs.csv                          step 3: the tier table the scorer reads
  (spec/offlist_tiers_nhamcs_anylisted.csv keeps the first version, which fell back to any-listed rates.)

Run: python3 scripts/analysis/offlist_severity_nhamcs.py
"""
from __future__ import annotations

import csv
import glob
import importlib.util
import itertools
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluator.condition_match import FlagMatcher  # noqa: E402

_spec = importlib.util.spec_from_file_location("nhamcs_urgency", ROOT / "scripts" / "analysis" / "nhamcs_urgency.py")
nu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nu)

OUT = ROOT / "results" / "analysis" / "nhamcs_offlist"
SPEC_OUT = ROOT / "spec" / "offlist_tiers_nhamcs.csv"
TIERS_CSV = ROOT / "spec" / "dangerous_if_missed_tiers_v03b.csv"
NT_GROUPS_CSV = ROOT / "spec" / "offlist_escalation_groups.csv"
ICD_ORDER = ROOT / "data" / "external" / "icd10cm" / "icd10cm_order_2026.txt"
NYU_XLSX = nu.NYU_XLSX
RUN_GLOBS = ("results/v03/ab/runs/*v7a*.json", "results/v03/runs/*.json")

MIN_N = 30
TIER_SCOPE = "primary"  # the tier rule reads primary-diagnosis rates only
WEAK_ICU_VISITS = 5  # a tier-1 row resting on the ICU clause with fewer critical-care visits is weak evidence
ADULT_MIN_AGE = 18
DIAGS = nu.DIAGS

# AHRQ 2022 EPC report 22(23)-EHC043 (Newman-Toker et al.), ED top-15 conditions by serious misdiagnosis-related
# harm. Ten of the fifteen are Newman-Toker 2023 Table 1 rows already in spec/offlist_escalation_groups.csv;
# these are the other five, as ICD-10-CM prefixes. Prefix choices are ours and are listed in the doc.
AHRQ_GROUPS: dict[str, tuple[str, str]] = {
    "Spinal cord compression": ("G95.2", "AHRQ 2022 ED top-15: spinal cord compression (G95.2 cord compression, unspecified)"),
    "Traumatic brain injury and intracranial haemorrhage": ("S06", "AHRQ 2022 ED top-15: TBI and intracranial haemorrhage (S06 intracranial injury; non-traumatic I60-I62 are in the stroke group)"),
    "Cardiac arrhythmia": ("I44 I45 I46 I47 I49", "AHRQ 2022 ED top-15: cardiac arrhythmia (I44-I45 conduction block, I46 cardiac arrest, I47 paroxysmal tachycardia, I49 other arrhythmias; I48 AF is on the DDXPlus list)"),
    "GI perforation and rupture": ("K63.1 K25.1 K25.2 K25.5 K25.6 K26.1 K26.2 K26.5 K26.6 K27.1 K27.2 K27.5 K27.6 K28.1 K28.2 K28.5 K28.6 K35.2 K35.3 K57.0 K57.2 K57.4 K57.8 K65",
                                   "AHRQ 2022 ED top-15: GI perforation and rupture (intestinal perforation K63.1, ulcers with perforation, perforated appendicitis K35.2-3, diverticulitis with perforation and abscess K57.x0, peritonitis K65)"),
    "Intestinal obstruction": ("K56 K31.5 K41.0 K41.3 K42.0 K43.0 K43.3 K43.6 K44.0 K45.0 K46.0",
                               "AHRQ 2022 ED top-15: intestinal obstruction (K56 ileus and obstruction, K31.5 pyloric obstruction, hernias with obstruction; K40 inguinal hernia is on the DDXPlus list)"),
}

# Codes we check by hand in the coverage report: families a clinician would call dangerous (expect tier 1-2)
# and families a clinician would call routine (expect tier 3). A mismatch is reported, never corrected.
WATCH_DANGEROUS = ("T78.0", "T78.2", "T88.6", "I46", "J96", "R57", "K85", "E10.1", "E11.0", "E11.1", "K92.2", "N17",
                   "I50", "G40", "J81", "J80", "I31.4", "L51.1", "T63", "K35", "K81", "N10", "I74", "A41", "G00", "I60",
                   "I61", "I63", "I71", "K92", "E87.1", "G45", "J69", "I49.0", "T81.4", "O00", "N39.0")
WATCH_ROUTINE = ("M54", "M79", "J02", "L50", "G47", "J33", "H10", "L20", "M25", "R51", "J30", "L29", "K59", "B34",
                 "J00", "J20", "M94", "G50", "H81", "M62", "L30", "J31", "L03", "K29", "F43", "H92")


# ---------------------------------------------------------------- descriptions

def load_descriptions() -> dict[str, str]:
    """ICD-10-CM description per dotless code, headers included, from the CMS order file; NYU sheet as fallback."""
    desc: dict[str, str] = {}
    if ICD_ORDER.exists():
        with ICD_ORDER.open(encoding="latin-1") as fh:
            for line in fh:
                code = line[6:13].strip()
                if code:
                    desc[code] = line[77:].strip() or line[16:76].strip()
        return desc
    try:
        import openpyxl

        ws = openpyxl.load_workbook(NYU_XLSX, read_only=True).worksheets[0]
        for r in ws.iter_rows(min_row=2, values_only=True):
            if r[0]:
                desc[str(r[0]).replace(".", "").upper()] = str(r[1] or "")
    except Exception:  # noqa: BLE001 - descriptions are decoration; the tiers do not depend on them
        pass
    return desc


def describe(prefix: str, desc: dict[str, str]) -> str:
    return desc.get(prefix.replace(".", "").upper(), "")


# ---------------------------------------------------------------- step 1: outcomes by code

def visit_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """One row per visit with the outcome flags the rates use."""
    out = pd.DataFrame({
        "w": df["PATWT"].to_numpy(dtype=float),
        "admit": ((df["ADMITHOS"] == 1) | (df["OBSHOS"] == 1) | (df["TRANOTH"] == 1)).to_numpy(),
        "icu": (df["ADMIT"] == 1).to_numpy(),
        "death": ((df["DIEDED"] == 1) | (df["DOA"] == 1)).to_numpy(),
        "immedr": df["IMMEDR"].where(df["IMMEDR"].between(1, 5)).to_numpy(),
    })
    out.index = df.index
    return out


def clean_diag(s: pd.Series) -> pd.Series:
    """NHAMCS diagnosis as a dotless ICD-10-CM prefix of 3 or 4 characters; blank for non-codes."""
    s = s.str.strip().str.upper().str.rstrip("-")
    bad = s.isin(["", "-9", "-8", "-7"]) | s.str.startswith("ZZZ") | s.str.startswith("V99") | ~s.str.match(r"^[A-Z][0-9]{2}")
    return s.where(~bad, "")


def long_diagnoses(df: pd.DataFrame) -> pd.DataFrame:
    """(visit, code4, code3, primary) for every listed diagnosis, one row per visit x distinct code."""
    parts = []
    for i, c in enumerate(DIAGS):
        code = clean_diag(df[c])
        parts.append(pd.DataFrame({"visit": df.index, "code4": code, "primary": i == 0})[code != ""])
    long = pd.concat(parts, ignore_index=True)
    long["code3"] = long["code4"].str[:3]
    return long


def rates(long: pd.DataFrame, vo: pd.DataFrame, key: str, primary_only: bool) -> pd.DataFrame:
    sub = long[long["primary"]] if primary_only else long
    sub = sub.drop_duplicates(["visit", key])
    j = sub.join(vo, on="visit")
    g = j.groupby(key)
    w = g["w"].sum()
    tri = j[j["immedr"].notna()]
    gt = tri.groupby(key)
    wt = gt["w"].sum()
    out = pd.DataFrame({
        "n": g.size(),
        "n_admit": g["admit"].sum().astype(int),
        "n_icu": g["icu"].sum().astype(int),
        "n_death": g["death"].sum().astype(int),
        "weighted_n": w,
        "admission": (j["w"] * j["admit"]).groupby(j[key]).sum() / w,
        "icu": (j["w"] * j["icu"]).groupby(j[key]).sum() / w,
        "death": (j["w"] * j["death"]).groupby(j[key]).sum() / w,
        "n_triaged": gt.size(),
        "wmean_immedr": (tri["w"] * tri["immedr"]).groupby(tri[key]).sum() / wt,
    })
    for k in range(1, 6):
        out[f"p_immedr_{k}"] = (tri["w"] * (tri["immedr"] == k)).groupby(tri[key]).sum() / wt
    out["n_triaged"] = out["n_triaged"].fillna(0).astype(int)
    return out


def outcomes_by_code(df: pd.DataFrame) -> pd.DataFrame:
    vo = visit_outcomes(df)
    long = long_diagnoses(df)
    frames = []
    for level, key in (("code", "code4"), ("group", "code3")):
        for scope, prim in (("primary", True), ("any", False)):
            r = rates(long, vo, key, prim).reset_index().rename(columns={key: "icd10"})
            r.insert(1, "level", level)
            r.insert(2, "scope", scope)
            frames.append(r)
    allr = pd.concat(frames, ignore_index=True)
    allr["icd10"] = allr["icd10"].map(lambda c: c if len(c) == 3 else f"{c[:3]}.{c[3:]}")
    return allr


# ---------------------------------------------------------------- step 2: calibration on the overlap

# Candidate thresholds, round values only. None drops the clause.
GRID = {
    "icu_ge": (0.05, 0.10, 0.15, 0.20, 0.25),
    "death_ge": (None, 0.005, 0.01, 0.02),
    "admit_ge": (None, 0.5, 0.6, 0.7, 0.8, 0.9),
    "admit_lt": (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40),
}


def apply_rule(rule: dict, admission: float, icu: float, death: float) -> int:
    """Tier 1 if any tier-1 clause fires; tier 3 if admission is under the routine line; else tier 2."""
    if icu >= rule["icu_ge"]:
        return 1
    if rule["death_ge"] is not None and death >= rule["death_ge"]:
        return 1
    if rule["admit_ge"] is not None and admission >= rule["admit_ge"]:
        return 1
    if admission < rule["admit_lt"]:
        return 3
    return 2


def rule_text(rule: dict) -> str:
    t1 = [f"ICU >= {rule['icu_ge']:.0%}"]
    if rule["death_ge"] is not None:
        t1.append(f"ED death >= {rule['death_ge']:.1%}")
    if rule["admit_ge"] is not None:
        t1.append(f"admission >= {rule['admit_ge']:.0%}")
    return f"tier 1 if {' or '.join(t1)}; tier 3 if admission < {rule['admit_lt']:.0%}; else tier 2"


def qwk3(a: list[int], b: list[int]) -> float:
    return nu.quadratic_weighted_kappa(a, b, k=3)


def score_rule(rule: dict, rows: list[dict]) -> tuple[float, int]:
    pred = [apply_rule(rule, r["admission"], r["icu"], r["death"]) for r in rows]
    ref = [r["tier_ref"] for r in rows]
    return qwk3(pred, ref), sum(p == q for p, q in zip(pred, ref))


def all_rules() -> list[dict]:
    return [dict(zip(GRID, vals)) for vals in itertools.product(*GRID.values())]


def rank_key(rule: dict, rows: list[dict]) -> tuple:
    """Best kappa first, then most exact matches, then the simpler rule (fewer clauses), then rounder thresholds."""
    k, exact = score_rule(rule, rows)
    clauses = (rule["death_ge"] is not None) + (rule["admit_ge"] is not None)
    roundness = sum(0 if v in (None, 0.05, 0.1, 0.2, 0.3, 0.5, 0.01) else 1 for v in rule.values())
    return (-round(k, 4), -exact, clauses, roundness)


def best_rule(rows: list[dict]) -> dict:
    return min(all_rules(), key=lambda r: rank_key(r, rows))


def confusion(rows: list[dict], rule: dict) -> dict:
    tab = {f"ref{i}": {f"pred{j}": 0 for j in (1, 2, 3)} for i in (1, 2, 3)}
    for r in rows:
        tab[f"ref{r['tier_ref']}"][f"pred{apply_rule(rule, r['admission'], r['icu'], r['death'])}"] += 1
    return tab


def calibrate(df: pd.DataFrame) -> tuple[dict, list[dict]]:
    codes = nu.condition_codes()
    masks = nu.match_masks(df, codes)
    tiers = {r["condition"]: r for r in csv.DictReader(TIERS_CSV.open())}
    vo = visit_outcomes(df)
    adult = (df["AGE"] >= ADULT_MIN_AGE).to_numpy()
    rows = []
    for cond, cs in sorted(codes.items()):
        t = tiers.get(cond)
        if not t:
            continue
        for scope in ("primary", "any"):
            m = masks[cond][scope] & adult
            sub = vo[m]
            w = sub["w"].sum()
            rows.append({
                "condition": cond, "scope": scope, "n": int(m.sum()),
                "admission": float((sub["w"] * sub["admit"]).sum() / w) if w else float("nan"),
                "icu": float((sub["w"] * sub["icu"]).sum() / w) if w else float("nan"),
                "death": float((sub["w"] * sub["death"]).sum() / w) if w else float("nan"),
                "tier_ref": int(t["final_tier"]), "base_tier": int(t["base_tier"]),
                "tier_formal": int(t["final_tier_formal"]),
            })
    overlap = [r for r in rows if r["scope"] == "primary" and r["n"] >= MIN_N]
    rule = best_rule(overlap)
    k, exact = score_rule(rule, overlap)
    ranked = sorted(all_rules(), key=lambda r: rank_key(r, overlap))
    top = [{"rule": rule_text(r), **{k2: v for k2, v in r.items()}, "qwk": round(score_rule(r, overlap)[0], 3),
            "exact": score_rule(r, overlap)[1]} for r in ranked[:10]]
    # Leave-one-out: refit on n-1 conditions, predict the held-out one.
    loo_pred, loo_rules = [], Counter()
    for i, held in enumerate(overlap):
        rest = overlap[:i] + overlap[i + 1:]
        rr = best_rule(rest)
        loo_rules[rule_text(rr)] += 1
        loo_pred.append(apply_rule(rr, held["admission"], held["icu"], held["death"]))
    ref = [r["tier_ref"] for r in overlap]
    for r in overlap:
        r["tier_rule"] = apply_rule(rule, r["admission"], r["icu"], r["death"])
    base_k = qwk3([r["tier_rule"] for r in overlap], [r["base_tier"] for r in overlap])
    formal_k = qwk3([r["tier_rule"] for r in overlap], [r["tier_formal"] for r in overlap])
    calib = {
        "reference": "spec/dangerous_if_missed_tiers_v03b.csv final_tier",
        "scope": f"NHAMCS ED 2016-2022, age >= {ADULT_MIN_AGE}, primary diagnosis, >= {MIN_N} unweighted visits",
        "n_overlap": len(overlap),
        "overlap_conditions": [r["condition"] for r in overlap],
        "rule": rule, "rule_text": rule_text(rule), "qwk": round(k, 3), "exact": exact,
        "within_one": int(sum(abs(r["tier_rule"] - r["tier_ref"]) <= 1 for r in overlap)),
        "confusion": confusion(overlap, rule),
        "qwk_vs_base_tier": round(base_k, 3), "qwk_vs_final_tier_formal": round(formal_k, 3),
        "loo_qwk": round(qwk3(loo_pred, ref), 3), "loo_exact": int(sum(p == q for p, q in zip(loo_pred, ref))),
        "loo_rules": dict(loo_rules.most_common()),
        "top_rules": top,
        "routine_line_sensitivity": [
            {"admit_lt": d, "qwk": round(score_rule({**rule, "admit_lt": d}, overlap)[0], 3),
             "exact": score_rule({**rule, "admit_lt": d}, overlap)[1]} for d in GRID["admit_lt"]],
        "icu_line_sensitivity": [
            {"icu_ge": a, "qwk": round(score_rule({**rule, "icu_ge": a}, overlap)[0], 3),
             "exact": score_rule({**rule, "icu_ge": a}, overlap)[1]} for a in GRID["icu_ge"]],
        "grid": {k2: list(v) for k2, v in GRID.items()},
        "disagreements": [{"condition": r["condition"], "ref": r["tier_ref"], "rule": r["tier_rule"], "n": r["n"],
                           "admission": round(r["admission"], 3), "icu": round(r["icu"], 3), "death": round(r["death"], 4)}
                          for r in overlap if r["tier_rule"] != r["tier_ref"]],
    }
    return calib, rows


# ---------------------------------------------------------------- step 3: tiers for off-list codes

def override_groups() -> list[tuple[str, str, str]]:
    """(prefix, group, source) for Newman-Toker 2023 Table 1 groups (spec CSV) and the AHRQ 2022 additions."""
    out = []
    with NT_GROUPS_CSV.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            for p in r["icd10_prefixes"].split():
                out.append((p.upper(), r["group"], f"override: {r['source_row']}"))
    for group, (prefixes, source) in AHRQ_GROUPS.items():
        for p in prefixes.split():
            out.append((p.upper(), group, f"override: {source}"))
    return out


def override_for(code: str, overrides: list[tuple[str, str, str]]) -> tuple[str, str] | None:
    c = code.replace(".", "").upper()
    hits = [(p, g, s) for p, g, s in overrides if c.startswith(p.replace(".", ""))]
    if not hits:
        return None
    p, g, s = max(hits, key=lambda h: len(h[0]))
    return g, s


def unscored_reason(prefix: str) -> str | None:
    c = prefix.replace(".", "").upper()
    if c[0] == "R":
        return "symptom code (R00-R99)"
    if c[0] == "Z":
        return "factors influencing health status (Z00-Z99)"
    if c[0] in "VWXY":
        return "external cause code (V00-Y99)"
    return None


class Outcomes:
    """Lookup of NHAMCS rates by dotless prefix, in the `TIER_SCOPE` scope (primary diagnosis) only."""

    def __init__(self, table: pd.DataFrame):
        self.t = {(r.icd10.replace(".", ""), r.scope): r for r in table.itertuples()}

    def get(self, prefix: str) -> tuple[object, str] | None:
        c = prefix.replace(".", "").upper()
        r = self.t.get((c, TIER_SCOPE))
        if r is not None and r.n >= MIN_N:
            return r, TIER_SCOPE
        return None

    def n(self, prefix: str) -> int:
        r = self.t.get((prefix.replace(".", "").upper(), TIER_SCOPE))
        return int(r.n) if r is not None else 0


def tier_for(prefix: str, outcomes: Outcomes, rule: dict, overrides: list) -> dict:
    """One spec row for a prefix: override, then unscored classes, then the NHAMCS rule."""
    row = {"icd10_prefix": prefix, "tier": "unscored", "rule_path": "unscored", "admission": "", "icu": "",
           "death": "", "n": "", "n_icu": "", "weak_evidence": False, "source": ""}
    hit = outcomes.get(prefix)
    if hit:
        r, scope = hit
        row.update(admission=f"{r.admission:.3f}", icu=f"{r.icu:.3f}", death=f"{r.death:.4f}", n=int(r.n),
                   n_icu=int(r.n_icu))
        nh = f"NHAMCS ED 2016-2022 adults, {scope} diagnosis, n={int(r.n)}"
    else:
        row["n"] = outcomes.n(prefix)
        nh = f"NHAMCS ED 2016-2022 adults, n={row['n']} (under {MIN_N})"
    ov = override_for(prefix, overrides)
    if ov:
        row.update(tier=1, rule_path="override", source=f"{ov[1]} [{ov[0]}]; {nh}")
        return row
    reason = unscored_reason(prefix)
    if reason:
        row.update(source=f"unscored: {reason}; {nh}")
        return row
    if not hit:
        row.update(source=f"unscored: {nh}")
        return row
    tier = apply_rule(rule, r.admission, r.icu, r.death)
    # Weak evidence: tier 1 by the ICU clause alone (the admission clause does not fire) on few ICU visits.
    weak = (tier == 1 and not (rule.get("admit_ge") is not None and r.admission >= rule["admit_ge"])
            and not (rule.get("death_ge") is not None and r.death >= rule["death_ge"])
            and int(r.n_icu) < WEAK_ICU_VISITS)
    row.update(tier=tier, rule_path="nhamcs", weak_evidence=weak, source=f"nhamcs rule; {nh}")
    return row


DESC_FIX = {"U07": "Emergency use of U07 (U07.1 COVID-19; U07.0 vaping-related disorder)",
            "C": "Malignant neoplasms C00-C97 (Newman-Toker 2023 cancers group)",
            "I64": "Stroke, not specified as haemorrhage or infarction (WHO ICD-10; not an ICD-10-CM code)"}
NOT_CM = "not an ICD-10-CM FY2026 code"


def build_spec(table: pd.DataFrame, rule: dict, emitted: dict[str, int], desc: dict[str, str],
               matcher: FlagMatcher) -> list[dict]:
    """Rows for every 3-character group NHAMCS, the runs or the overrides mention, plus finer prefixes
    whose tier differs from their group's. The scorer matches the longest listed prefix, after the
    DDXPlus map: a group that holds a DDXPlus code says so in `source`, because the on-list tier wins there."""
    outcomes = Outcomes(table)
    overrides = override_groups()
    onlist: dict[str, set[str]] = defaultdict(set)
    for code in matcher.map_codes():
        for cond in matcher.conditions_hit([code]):
            onlist[code[:3]].add(cond)
    groups = {c[:3] for c in emitted} | {c[:3] for c in table["icd10"]} | {p[:3] for p, _, _ in overrides}
    finer = {c for c in table["icd10"] if len(c) > 3} | {p for p, _, _ in overrides if len(p) > 3}
    finer |= {c[:5] for c in emitted if len(c) >= 5}
    rows = []
    for g in sorted(groups):
        grow = tier_for(g, outcomes, rule, overrides)
        grow["description"] = DESC_FIX.get(g) or describe(g, desc) or NOT_CM
        if g in onlist:
            grow["source"] = f"DDXPlus map first ({', '.join(sorted(onlist[g]))}); " + grow["source"]
        rows.append(grow)
        for f in sorted(p for p in finer if p[:3] == g):
            frow = tier_for(f, outcomes, rule, overrides)
            # A subcode with no basis of its own (under 30 visits, no override) inherits the group row.
            if frow["rule_path"] in ("override", "nhamcs") and str(frow["tier"]) != str(grow["tier"]):
                frow["description"] = describe(f, desc) or f"{grow['description']} (subcode)"
                rows.append(frow)
    cols = ["icd10_prefix", "description", "tier", "rule_path", "admission", "icu", "death", "n", "n_icu",
            "weak_evidence", "source"]
    return [{**{c: r[c] for c in cols}, "weak_evidence": "true" if r["weak_evidence"] else "false"} for r in rows]


# ---------------------------------------------------------------- step 4: coverage of emitted codes

_CODE_RE = re.compile(r"^[A-Z][0-9]{2}(\.[0-9A-Z]{1,4})?$")


def norm_emitted(code: str) -> str | None:
    """Emitted code as dotted ICD-10-CM, or None when it is not one code (ranges, ICD-9, junk)."""
    c = str(code).strip().upper().replace(" ", "")
    if "." not in c and len(c) > 3:
        c = f"{c[:3]}.{c[3:]}"
    return c if _CODE_RE.match(c) else None


def emitted_codes(matcher: FlagMatcher) -> dict[str, Counter]:
    """Off-list code mentions per source: A/B differentials, A/B flags, v6 differentials, v6 flags."""
    tallies = {"ab_differential": Counter(), "ab_flag": Counter(), "v6_differential": Counter(), "v6_flag": Counter(),
               "unparseable": Counter(), "onlist_mentions": Counter()}
    for pat in RUN_GLOBS:
        for p in sorted(glob.glob(str(ROOT / pat))):
            if p.endswith(("provenance.json", "scores.json")):
                continue
            src = "ab" if "/ab/" in p else "v6"
            for it in json.load(open(p))["predictions"]:
                ddx = it.get("differential_diagnoses") or it.get("differential") or []
                flags = [it["flag"]] if "flag" in it else (it.get("flags") or [])
                for kind, codes in (("differential", [d.get("code") if isinstance(d, dict) else d for d in ddx]),
                                    ("flag", flags)):
                    for raw in codes:
                        if not raw:
                            continue
                        c = norm_emitted(raw)
                        if c is None:
                            tallies["unparseable"][str(raw)[:20]] += 1
                            continue
                        if matcher.conditions_hit([c]):
                            tallies["onlist_mentions"][f"{src}_{kind}"] += 1
                            continue
                        tallies[f"{src}_{kind}"][c] += 1
    return tallies


def lookup_tier(code: str, spec_rows: list[dict]) -> dict:
    c = code.replace(".", "")
    hits = [r for r in spec_rows if c.startswith(r["icd10_prefix"].replace(".", ""))]
    return max(hits, key=lambda r: len(r["icd10_prefix"])) if hits else {"tier": "unscored", "rule_path": "no row"}


def coverage(tallies: dict[str, Counter], spec_rows: list[dict]) -> dict:
    out = {"by_source": {}, "top30": [], "watch_dangerous_not_tier1_2": [], "watch_routine_not_tier3": []}
    total = Counter()
    for src, tally in tallies.items():
        if src in ("unparseable", "onlist_mentions"):
            continue
        shares = Counter()
        for code, n in tally.items():
            shares[str(lookup_tier(code, spec_rows)["tier"])] += n
        total.update(tally)
        m = sum(tally.values())
        out["by_source"][src] = {"mentions": m, "unique_codes": len(tally),
                                 "share": {t: round(shares[t] / m, 3) if m else 0 for t in ("1", "2", "3", "unscored")}}
    m = sum(total.values())
    shares = Counter()
    for code, n in total.items():
        shares[str(lookup_tier(code, spec_rows)["tier"])] += n
    out["all"] = {"mentions": m, "unique_codes": len(total),
                  "share": {t: round(shares[t] / m, 3) for t in ("1", "2", "3", "unscored")},
                  "count": {t: shares[t] for t in ("1", "2", "3", "unscored")}}
    out["unparseable"] = dict(tallies["unparseable"].most_common(15))
    out["onlist_mentions"] = dict(tallies["onlist_mentions"])
    for code, n in total.most_common(30):
        r = lookup_tier(code, spec_rows)
        out["top30"].append({"code": code, "mentions": n, "tier": r["tier"], "rule_path": r["rule_path"],
                             "description": r.get("description", ""), "matched_prefix": r.get("icd10_prefix", ""),
                             "admission": r.get("admission", ""), "icu": r.get("icu", ""), "n": r.get("n", "")})
    for w in WATCH_DANGEROUS:
        r = lookup_tier(w, spec_rows)
        if str(r["tier"]) not in ("1", "2"):
            out["watch_dangerous_not_tier1_2"].append({"code": w, **{k: r.get(k, "") for k in ("tier", "rule_path", "description", "admission", "icu", "death", "n")}, "mentions": sum(n for c, n in total.items() if c.startswith(w))})
    for w in WATCH_ROUTINE:
        r = lookup_tier(w, spec_rows)
        if str(r["tier"]) != "3":
            out["watch_routine_not_tier3"].append({"code": w, **{k: r.get(k, "") for k in ("tier", "rule_path", "description", "admission", "icu", "death", "n")}, "mentions": sum(n for c, n in total.items() if c.startswith(w))})
    reasons = Counter()
    for code, n in total.items():
        r = lookup_tier(code, spec_rows)
        if str(r["tier"]) == "unscored":
            src = r.get("source", "no row")
            reasons["symptom code (R00-R99)" if "symptom" in src else "Z code" if "Z00-Z99" in src
                    else "external cause" if "V00-Y99" in src else "under 30 visits" if "under" in src else "no row"] += n
    out["unscored_mentions_by_reason"] = dict(reasons.most_common())
    out["tier_scope"] = TIER_SCOPE
    icu_only = [r for r in spec_rows if r["rule_path"] == "nhamcs" and r["tier"] == 1 and float(r["admission"]) < 0.5]
    few = [r for r in spec_rows if r["weak_evidence"] == "true"]
    out["tier1_by_icu_clause_only"] = {"rows": len(icu_only), "rows_under_5_icu_visits": len(few),
                                       "prefixes_under_5_icu_visits": [r["icd10_prefix"] for r in few],
                                       "mentions_under_5_icu_visits": sum(n for c, n in total.items()
                                                                          if lookup_tier(c, spec_rows).get("icd10_prefix") in {r["icd10_prefix"] for r in few})}
    out["tier_counts_in_spec"] = dict(Counter(str(r["tier"]) for r in spec_rows))
    out["rule_path_counts_in_spec"] = dict(Counter(r["rule_path"] for r in spec_rows))
    out["emitted_offlist_prefixes"] = dict(Counter(c[:3] for c in total.elements()).most_common(40))
    return out


# ---------------------------------------------------------------- main

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = nu.load_all()
    adults = df[df["AGE"] >= ADULT_MIN_AGE].copy()
    table = outcomes_by_code(adults)
    published = table[table["n"] >= MIN_N].copy()
    published.to_csv(OUT / "outcomes_by_code.csv", index=False, float_format="%.4f")
    print(f"adult visits {len(adults)} of {len(df)}; outcome rows published {len(published)} "
          f"(suppressed under {MIN_N}: {int((table['n'] < MIN_N).sum())})")

    calib, cond_rows = calibrate(df)
    pd.DataFrame(cond_rows).to_csv(OUT / "overlap_conditions.csv", index=False, float_format="%.4f")
    (OUT / "calibration.json").write_text(json.dumps(calib, indent=1))
    print("rule:", calib["rule_text"], "| qwk", calib["qwk"], "exact", calib["exact"], "of", calib["n_overlap"],
          "| loo qwk", calib["loo_qwk"], "loo exact", calib["loo_exact"])

    matcher = FlagMatcher()
    tallies = emitted_codes(matcher)
    emitted = Counter()
    for k in ("ab_differential", "ab_flag", "v6_differential", "v6_flag"):
        emitted.update(tallies[k])
    desc = load_descriptions()
    spec_rows = build_spec(table, calib["rule"], emitted, desc, matcher)
    with SPEC_OUT.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(spec_rows[0]))
        wr.writeheader()
        wr.writerows(spec_rows)
    cov = coverage(tallies, spec_rows)
    cov["calibration"] = {k: calib[k] for k in ("rule_text", "qwk", "exact", "n_overlap", "loo_qwk", "loo_exact")}
    (OUT / "coverage.json").write_text(json.dumps(cov, indent=1))
    print(f"spec rows {len(spec_rows)}: tiers {cov['tier_counts_in_spec']} paths {cov['rule_path_counts_in_spec']}")
    print("coverage all:", cov["all"])
    for src, v in cov["by_source"].items():
        print(f"  {src}: {v}")


if __name__ == "__main__":
    main()
