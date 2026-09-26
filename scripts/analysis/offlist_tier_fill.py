"""Fill unscored off-list tiers from more third-party data, with the same NHAMCS rule.

Why: 901 of 1,369 prefixes in the NHAMCS tier table are unscored, mostly because they have under 30 adult
primary-diagnosis ED visits in NHAMCS 2016-2022, and 42% of off-list flags in the audited runs land on them.
An unscored flag cannot earn a partial credit, so a model that flags giant cell arteritis or malaria scores as
if it had flagged nothing. We fill those rows from public data with the rule the table already uses, and keep
every scored row as it is (docs/offlist-severity-nhamcs.md, change note 2026-09-26, third).

The rule never changes: primary-diagnosis rates only; tier 1 if ICU >= 5% or admission >= 50%; tier 3 if
admission < 5%; otherwise tier 2; at least 30 primary visits. Symptom (R), factor (Z) and external-cause (V-Y)
codes stay unscored. We try two sources in order for each unscored row, and write the one that scored it to
the new column `tier_source`:

  1. nhamcs_pooled: NHAMCS ED 2011-2022. The 2011-2015 files code diagnoses in ICD-9-CM; we map each code with
     the CMS 2018 ICD-9-CM to ICD-10-CM General Equivalence Mapping (GEM) to the longest ICD-10-CM prefix (4 or
     3 characters) that every GEM target shares, and drop the visit when the targets share no 3-character group.
     A 3-character group then follows the group-tier rule of offlist_severity_nhamcs.py on the pooled rates.
  2. ccsr: the AHRQ CCSR category of the prefix (v2026.1, default outpatient category; for a prefix, the
     category most of its ICD-10-CM codes fall in), rated on the pooled NHAMCS primary-diagnosis visits whose
     code falls in that category, with the same rule and the same 30-visit floor.

HCUP NEDS (HCUPnet), the third source considered, is not used: HCUPnet is an interactive query builder with no
public API, so its rates cannot be pulled by a script.

Rows already scored keep their tier, with tier_source `override` or `nhamcs`. A finer prefix inside a filled
group gets a row of its own when its own basis gives a different tier, as in offlist_severity_nhamcs.py.

Inputs:
  spec/offlist_tiers_nhamcs_pre_fill.csv              the table offlist_severity_nhamcs.py writes
  data/external/nhamcs/extracted/                     NHAMCS ED 2016-2022 (ICD-10-CM)
  data/external/nhamcs/icd9/extracted/*.dta           NHAMCS ED 2011-2015 Stata files (ICD-9-CM), from
                                                      ftp.cdc.gov/pub/Health_Statistics/NCHS/dataset_documentation/nhamcs/stata/
  data/external/gem/cms2018/2018_I9gem.txt            CMS 2018 GEM, ICD-9-CM to ICD-10-CM
  data/external/ccsr/DXCCSR-v2026-1/DXCCSR_v2026-1.csv  AHRQ CCSR for ICD-10-CM diagnoses v2026.1
Outputs:
  spec/offlist_tiers_nhamcs.csv                       the filled table the scorer reads
  results/analysis/nhamcs_offlist/fill_rates.csv      pooled primary-diagnosis rates by code, group and CCSR category
  results/analysis/nhamcs_offlist/fill_report.json    coverage before and after, audited-case moves, example codes

Run: python3 scripts/analysis/offlist_tier_fill.py   (after offlist_severity_nhamcs.py)
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location("offlist_severity_nhamcs", ROOT / "scripts" / "analysis" / "offlist_severity_nhamcs.py")
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
nu = base.nu

PRE_FILL = ROOT / "spec" / "offlist_tiers_nhamcs_pre_fill.csv"
SPEC_OUT = ROOT / "spec" / "offlist_tiers_nhamcs.csv"
ICD9_DIR = ROOT / "data" / "external" / "nhamcs" / "icd9" / "extracted"
GEM = ROOT / "data" / "external" / "gem" / "cms2018" / "2018_I9gem.txt"
CCSR = ROOT / "data" / "external" / "ccsr" / "DXCCSR-v2026-1" / "DXCCSR_v2026-1.csv"
AUDIT = ROOT / "results" / "analysis" / "case_selection" / "audited_150_verdicts.csv"
OUT = base.OUT

RULE = {"icu_ge": 0.05, "death_ge": None, "admit_ge": 0.5, "admit_lt": 0.05}
MIN_N = base.MIN_N
YEARS = "NHAMCS ED 2011-2022 adults"
OUTCOME_COLS = ["AGE", "DIAG1", "ADMITHOS", "OBSHOS", "TRANOTH", "ADMIT", "DIEDED", "DOA", "PATWT", "IMMEDR"]
EXAMPLES = ("M31.6", "B54", "B44", "H40.2", "A98.3", "G12.9", "G21.0", "E84.0", "G73.0", "M47.0", "A35", "I36.9", "J84")
COLS = ["icd10_prefix", "description", "tier", "rule_path", "tier_source", "pooled_tier", "admission", "icu", "death",
        "n", "n_icu", "weak_evidence", "source"]


# ---------------------------------------------------------------- ICD-9-CM visits, 2011-2015

def gem_prefixes(lines) -> dict[str, str]:
    """ICD-9-CM code -> dotless ICD-10-CM prefix every GEM target shares: 4 characters (the public-file length)
    when all targets share them, else 3; codes whose targets share no 3-character group are left out, because
    their visits cannot be put in one group."""
    targets: dict[str, set[str]] = defaultdict(set)
    for line in lines:
        parts = line.split()
        if len(parts) < 3 or parts[1] == "NoDx" or parts[2][1] == "1":  # flag 2nd digit: no map
            continue
        targets[parts[0]].add(parts[1])
    out = {}
    for i9, ts in targets.items():
        for n in (4, 3):
            heads = {t[:n] for t in ts}
            if len(heads) == 1:
                out[i9] = heads.pop()
                break
    return out


def load_icd9_years(gem: dict[str, str]) -> pd.DataFrame:
    """Adult NHAMCS 2011-2015 visits with DIAG1 recoded to a dotless ICD-10-CM prefix (blank when unmapped)."""
    frames = []
    for p in sorted(ICD9_DIR.glob("*.dta")):
        df = pd.read_stata(p, columns=OUTCOME_COLS, convert_categoricals=False)
        d = df["DIAG1"].apply(lambda v: v.decode() if isinstance(v, bytes) else v).astype(str).str.strip().str.upper()
        df["DIAG1"] = d.str.rstrip("-").map(gem).fillna("")
        for c in OUTCOME_COLS:
            if c != "DIAG1":
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["ERA"] = "icd9"
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_pooled() -> pd.DataFrame:
    """Adult primary-diagnosis visits 2011-2022: ERA, DIAG1 (dotless prefix) and the outcome columns."""
    gem = gem_prefixes(GEM.open())
    old = load_icd9_years(gem)
    new = nu.load_all()[OUTCOME_COLS].copy()
    new["DIAG1"] = base.clean_diag(new["DIAG1"])
    new["ERA"] = "icd10"
    df = pd.concat([new, old], ignore_index=True)
    return df[df["AGE"] >= base.ADULT_MIN_AGE].reset_index(drop=True)


def primary_rates(df: pd.DataFrame, keys: pd.Series) -> pd.DataFrame:
    """Rates per key over primary-diagnosis visits, in the columns offlist_severity_nhamcs.rates writes."""
    vo = base.visit_outcomes(df)
    long = pd.DataFrame({"visit": df.index, "key": keys.to_numpy(), "primary": True})[keys.to_numpy() != ""]
    return base.rates(long, vo, "key", primary_only=True)


# ---------------------------------------------------------------- CCSR

def ccsr_category(row: dict) -> str | None:
    """One CCSR row's category: the default outpatient category (an ED visit is an outpatient encounter). AHRQ
    puts codes that cannot be a first-listed diagnosis in placeholder categories XXX000 and XXX111; for those we
    take the code's first clinical category (CCSR CATEGORY 1), because a placeholder groups unrelated diseases."""
    for col in ("'Default CCSR CATEGORY OP'", "'CCSR CATEGORY 1'"):
        cat = (row.get(col) or "").strip("'").strip()
        if cat and not cat.startswith("XXX"):
            return cat
    return None


def load_ccsr() -> tuple[dict[str, str], dict[str, str]]:
    """(full dotless ICD-10-CM code -> CCSR category, category -> description)."""
    codes, names = {}, {}
    with CCSR.open(newline="", encoding="latin-1") as fh:
        for r in csv.DictReader(fh):
            for i in ("1", "2", "3", "4", "5", "6"):
                names[r[f"'CCSR CATEGORY {i}'"].strip("'").strip()] = r[f"'CCSR CATEGORY {i} DESCRIPTION'"].strip()
            code, cat = r["'ICD-10-CM CODE'"].strip("'").strip(), ccsr_category(r)
            if code and cat:
                codes[code] = cat
    return codes, names


class CcsrIndex:
    """Category of an ICD-10-CM prefix: the default outpatient category most of its codes fall in (ties go to the
    alphabetically first category, so the choice is reproducible)."""

    def __init__(self, code_to_cat: dict[str, str]):
        self.codes = code_to_cat
        self.sorted = sorted(code_to_cat)
        self._cache: dict[str, str | None] = {}

    def category(self, prefix: str) -> str | None:
        p = prefix.replace(".", "").upper()
        if p in self._cache:
            return self._cache[p]
        import bisect
        i = bisect.bisect_left(self.sorted, p)
        counts = Counter()
        while i < len(self.sorted) and self.sorted[i].startswith(p):
            counts[self.codes[self.sorted[i]]] += 1
            i += 1
        cat = min(counts, key=lambda c: (-counts[c], c)) if counts else None
        self._cache[p] = cat
        return cat


# ---------------------------------------------------------------- the fill

class Basis:
    """Pooled rate rows by dotless code (the 3- or 4-character public code), by group, and by CCSR category."""

    def __init__(self, code_rates: pd.DataFrame, group_rates: pd.DataFrame, cat_rates: pd.DataFrame, ccsr: CcsrIndex):
        self.code = {k: r for k, r in zip(code_rates.index, code_rates.itertuples())}
        self.group = {k: r for k, r in zip(group_rates.index, group_rates.itertuples())}
        self.cat = {k: r for k, r in zip(cat_rates.index, cat_rates.itertuples())}
        self.ccsr = ccsr

    def nhamcs(self, prefix: str):
        """The pooled row for a prefix: a 3-character prefix reads the group (all its codes), a longer one the code."""
        c = prefix.replace(".", "").upper()
        return (self.group if len(c) == 3 else self.code).get(c[:4])

    def code_table(self, group: str) -> pd.DataFrame:
        """Pooled code rows of one group, in the shape offlist_severity_nhamcs.subcode_tiers reads."""
        rows = [{"icd10": k if len(k) == 3 else f"{k[:3]}.{k[3:]}", "level": "code", "scope": "primary",
                 "n": int(r.n), "admission": r.admission, "icu": r.icu, "death": r.death, "n_icu": int(r.n_icu)}
                for k, r in self.code.items() if k[:3] == group]
        return pd.DataFrame(rows, columns=["icd10", "level", "scope", "n", "admission", "icu", "death", "n_icu"])


def _rates_fields(r) -> dict:
    return {"admission": f"{r.admission:.3f}", "icu": f"{r.icu:.3f}", "death": f"{r.death:.4f}", "n": int(r.n),
            "n_icu": int(r.n_icu)}


def fill_prefix(prefix: str, basis: Basis, overrides: list, names: dict[str, str] | None = None) -> dict:
    """Tier fields for one unscored prefix: pooled NHAMCS first, then its CCSR category, else still unscored."""
    names = names or {}
    r = basis.nhamcs(prefix)
    if r is not None and r.n >= MIN_N:
        tier, weak = base.rule_tier(RULE, r)
        row = {"tier": tier, "rule_path": "nhamcs", "tier_source": "nhamcs_pooled", "weak_evidence": weak,
               "source": f"nhamcs rule; {YEARS}, primary diagnosis, n={int(r.n)}", **_rates_fields(r)}
        if len(prefix.replace(".", "")) == 3:
            subs = base.subcode_tiers(basis.code_table(prefix[:3]), prefix[:3], RULE, overrides)
            row = base.group_tier({**row, "icd10_prefix": prefix}, subs)
            row.pop("icd10_prefix")
        return row
    n_code = int(r.n) if r is not None else 0
    cat = basis.ccsr.category(prefix)
    cr = basis.cat.get(cat) if cat else None
    if cr is not None and cr.n >= MIN_N:
        tier, weak = base.rule_tier(RULE, cr)
        label = f"{cat} {names.get(cat, '')}".strip()
        return {"tier": tier, "rule_path": "ccsr", "tier_source": "ccsr", "weak_evidence": weak,
                "source": f"ccsr rule: category {label}, {YEARS}, primary diagnosis, n={int(cr.n)} "
                          f"(the prefix itself: n={n_code})", **_rates_fields(cr)}
    why = f"CCSR {cat} n={int(cr.n) if cr is not None else 0}" if cat else "no CCSR category"
    return {"tier": "unscored", "rule_path": "unscored", "tier_source": "unscored", "weak_evidence": False,
            "n": n_code, "source": f"unscored: {YEARS}, n={n_code} (under {MIN_N}); {why} (under {MIN_N})"}


def is_fillable(row: dict) -> bool:
    """An unscored row the fill may change: not a symptom, factor or external-cause code."""
    return str(row["tier"]) == "unscored" and base.unscored_reason(row["icd10_prefix"]) is None


def ddx_note(source: str) -> str:
    return source[: source.index("; ") + 2] if source.startswith("DDXPlus map first") else ""


def fill_table(pre: list[dict], basis: Basis, overrides: list, finer: set[str], desc: dict[str, str],
               names: dict[str, str] | None = None) -> list[dict]:
    """The filled table: scored rows unchanged (tier_source = rule_path), unscored rows filled where a source has
    30+ primary visits, and finer prefixes of filled groups added where their own basis gives a different tier."""
    out = []
    for row in pre:
        row = {c: row.get(c, "") for c in COLS}
        if not is_fillable(row):
            row["tier_source"] = row["rule_path"]
            out.append(row)
            continue
        new = fill_prefix(row["icd10_prefix"], basis, overrides, names)
        filled = {**row, "pooled_tier": "", **new, "source": ddx_note(row["source"]) + new["source"]}
        filled["weak_evidence"] = "true" if new["weak_evidence"] else "false"
        out.append(filled)
        g = row["icd10_prefix"]
        if len(g.replace(".", "")) != 3:
            continue
        existing = {r["icd10_prefix"] for r in pre}
        for f in sorted(p for p in finer if p[:3] == g and p not in existing):
            fr = fill_prefix(f, basis, overrides, names)
            if fr["tier_source"] != "unscored" and str(fr["tier"]) != str(filled["tier"]):
                out.append({**{c: "" for c in COLS}, **fr, "icd10_prefix": f,
                            "description": base.describe(f, desc) or f"{row['description']} (subcode)",
                            "weak_evidence": "true" if fr["weak_evidence"] else "false"})
    return out


# ---------------------------------------------------------------- report

def audited_moves(rows: list[dict]) -> dict:
    """Kept audited cases whose flag was off-list unscored: new tier of each disputed (FP) and agreed (TP) miss."""
    from evaluator import v03_valid_reason as vr  # noqa: E402
    tmp = OUT / "_filled_tmp.csv"
    write(rows, tmp)
    rule = vr.TierFileRule(tmp)
    tmp.unlink()
    d = pd.read_csv(AUDIT, dtype=str)
    k = d[(d["bucket"] != "EXCLUDE") & (d["flag_tier"] == "off:unscored")]
    out = {}
    for kind in ("FP", "TP"):
        sub = k[k["kind"] == kind]
        tiers = Counter(rule.tier(f)[0] for f in sub["flag"])
        by_code = defaultdict(Counter)
        for f in sub["flag"]:
            by_code[rule.tier(f)[0]][f] += 1
        out[kind] = {"cases": len(sub), "new_tier": dict(tiers),
                     "to_tier1_partial_cost1": tiers.get("1", 0),
                     "to_tier2_or_3_still_cost7": tiers.get("2", 0) + tiers.get("3", 0),
                     "still_unscored": len(sub) - sum(tiers.get(t, 0) for t in ("1", "2", "3")),
                     "flags_by_new_tier": {t: dict(c.most_common()) for t, c in by_code.items()}}
    return out


def write(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=COLS)
        wr.writeheader()
        wr.writerows({c: r.get(c, "") for c in COLS} for r in rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cal = OUT / "calibration.json"
    if cal.exists():
        assert json.loads(cal.read_text())["rule"] == RULE, "calibrated rule changed; the fill must use the same rule"
    pre = list(csv.DictReader(PRE_FILL.open(newline="", encoding="utf-8")))

    df = load_pooled()
    print(f"adult visits 2011-2022: {len(df)} ({(df['ERA'] == 'icd9').sum()} from 2011-2015); "
          f"2011-2015 primary diagnoses mapped: {((df['ERA'] == 'icd9') & (df['DIAG1'] != '')).sum()}")
    code_to_cat, names = load_ccsr()
    ccsr = CcsrIndex(code_to_cat)
    code_rates = primary_rates(df, df["DIAG1"])
    group_rates = primary_rates(df, df["DIAG1"].str[:3])
    cats = pd.Series([ccsr.category(c) or "" if c else "" for c in df["DIAG1"]], index=df.index)
    cat_rates = primary_rates(df, cats)
    frames = []
    for level, t in (("code", code_rates), ("group", group_rates), ("ccsr", cat_rates)):
        frames.append(t.reset_index().rename(columns={"key": "key"}).assign(level=level))
    pd.concat(frames).to_csv(OUT / "fill_rates.csv", index=False, float_format="%.4f")

    matcher = base.FlagMatcher()
    tallies = base.emitted_codes(matcher)
    emitted = Counter()
    for k in ("ab_differential", "ab_flag", "v6_differential", "v6_flag"):
        emitted.update(tallies[k])
    finer = {c[:5] for c in emitted if len(c) >= 5}
    finer |= {f"{k[:3]}.{k[3:]}" for k in code_rates.index if len(k) == 4}
    basis = Basis(code_rates, group_rates, cat_rates, ccsr)
    rows = fill_table(pre, basis, base.override_groups(), finer, base.load_descriptions(), names)
    write(rows, SPEC_OUT)

    cov_pre, cov_post = base.coverage(tallies, pre), base.coverage(tallies, rows)
    report = {
        "rule": RULE,
        "rows": {"pre": len(pre), "post": len(rows)},
        "tiers_pre": dict(Counter(str(r["tier"]) for r in pre)),
        "tiers_post": dict(Counter(str(r["tier"]) for r in rows)),
        "tier_source_post": dict(Counter(r["tier_source"] for r in rows)),
        "filled_by_source": dict(Counter(f"{r['tier_source']}:{r['tier']}" for r in rows
                                         if r["tier_source"] in ("nhamcs_pooled", "ccsr"))),
        "unscored_prefixes": {"pre": sum(str(r["tier"]) == "unscored" for r in pre),
                              "post": sum(str(r["tier"]) == "unscored" for r in rows)},
        "coverage_pre": {k: v for k, v in cov_pre["by_source"].items()} | {"all": cov_pre["all"]},
        "coverage_post": {k: v for k, v in cov_post["by_source"].items()} | {"all": cov_post["all"]},
        "unscored_mentions_by_reason_post": cov_post["unscored_mentions_by_reason"],
        "audited": audited_moves(rows),
        "examples": {},
    }
    for ex in EXAMPLES:
        r = base.lookup_tier(ex, rows)
        report["examples"][ex] = {k: r.get(k, "") for k in ("icd10_prefix", "tier", "tier_source", "admission", "icu",
                                                            "n", "weak_evidence", "source")}
    (OUT / "fill_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("rows", "tiers_pre", "tiers_post", "filled_by_source", "unscored_prefixes")}))
    for src in ("ab_flag", "all"):
        print(src, "pre", report["coverage_pre"][src]["share"], "post", report["coverage_post"][src]["share"])
    for kind, v in report["audited"].items():
        print(kind, {k: v[k] for k in ("cases", "new_tier", "to_tier1_partial_cost1", "to_tier2_or_3_still_cost7")})
    for ex, v in report["examples"].items():
        print(ex, v["icd10_prefix"], v["tier"], v["tier_source"], v["n"], v["admission"], v["icu"])


if __name__ == "__main__":
    main()
