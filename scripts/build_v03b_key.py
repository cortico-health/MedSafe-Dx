#!/usr/bin/env python3
"""
Build the MedSafe-Dx v0.3b answer key: the v0.3 key with key fixes #7-#10
(docs/v0.3-key-fixes.md). Reads DDXPlus data only; no inference spend.

We reuse scripts/build_v03_key.py for the reference adults, the hallmark learner and
the class counts, and first rebuild the v0.3 key in memory; we refuse to write
anything unless its sha256 equals the pinned data/test_sets/eval-v03-key.sha256,
because every before/after count below compares against that key.

1. Key for the 470 sample under evaluator/answer_key_v03b.py (interval red-herring
   rule, evidence classes, over-concern eligibility), with tiers from
   spec/dangerous_if_missed_tiers_v03b.csv (column final_tier, the v0.3 tiers).
2. Keys for the two committed pools (their ID lists are not redrawn).
3. Matched low-risk controls per pool (#10): chargeable BENIGN reference adults (the
   main sample and both pools excluded), one per pool case, matched on initial evidence
   and age band where possible, drawn with numpy default_rng(CONTROL_SEED); pools in
   the order atypical, high-risk; a control is used once across both pools.
4. Before/after counts, switching cases and the formal-tier sensitivity counts in
   results/analysis/v03b_key/summary.json.

Outputs (data/test_sets/ is git-ignored; we force-add the ID lists, pairs and hashes):
  data/test_sets/eval-v03b-key.csv, .sha256
  data/test_sets/eval-v03b-pool-{atypical,high-risk}.key.csv, .key.sha256
  data/test_sets/eval-v03b-pool-{atypical,high-risk}-controls.{case_ids.txt,pairs.csv,json,key.csv,key.sha256}
  results/analysis/v03b_key/summary.json, switches.csv

Usage: python3 scripts/build_v03b_key.py   (about 30 s)
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluator import answer_key_v02 as ak2  # noqa: E402
from evaluator import answer_key_v03 as ak  # noqa: E402
from evaluator import answer_key_v03b as akb  # noqa: E402
from evaluator import key_views_v03 as kv  # noqa: E402
from build_v03_key import (ADULT_MIN_AGE, COND_JSON, EVID_JSON, SAMPLE_IDS, TEST_CSV, TS, Detector,  # noqa: E402
                           key_counts, key_for, learn_hallmarks, read_rows, write_csv)
from prep_v02_sample import build_case  # noqa: E402

OUT = ROOT / "results/analysis/v03b_key"
CONTROL_SEED = 20260925
AGE_BANDS = ((18, 39), (40, 64), (65, 200))
POOLS = ("atypical", "high-risk")
MATCH_LEVELS = ("initial_evidence+age_band", "initial_evidence", "age_band", "any")


def age_band(age: int) -> str:
    for lo, hi in AGE_BANDS:
        if lo <= age <= hi:
            return f"{lo}-{hi}" if hi < 200 else f"{lo}+"
    raise ValueError(age)


# ---------------------------------------------------------------- class cells


class CellTable:
    """(n, k) per band from 5% up for each (condition, hallmark count), from the v0.3 detector's counts."""

    def __init__(self, det: Detector):
        self.det = det

    def cells(self, c: str, h: int) -> list[list[int]]:
        return [[self.det.n[(c, b, h)], self.det.k[(c, b, h)]] for b in range(akb.FIRST_BAND, akb.N_BANDS)]

    def lookup_for(self, r: dict, is_ref: bool):
        """CellsLookup for one patient; a reference patient's own row is left out of its own cell."""
        def f(c: str, p: float) -> akb.ClassCells:
            h = self.det.present(r, c)
            cells = self.cells(c, len(h))
            b = ak.band_of(p)
            if is_ref and b >= akb.FIRST_BAND:
                cells[b - akb.FIRST_BAND][0] -= 1
                cells[b - akb.FIRST_BAND][1] -= r["path"] == c
            return akb.ClassCells(hallmark_tokens=h, cells=tuple((n, k) for n, k in cells))
        return f


def key_for_b(idx, rows, ref, table, tiers, levels, comps) -> list[dict]:
    out = []
    for i in idx:
        r = rows[i]
        out += akb.case_key_rows(f"ddxplus_{r['i']}", r["path"], r["dxa"], tiers, table.lookup_for(r, bool(ref[i])),
                                 ak2.red_flags(r["evidences"]), levels, comps)
    return out


def case_level(key_rows: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in key_rows:
        out.setdefault(r["case_id"], r)
    return out


def counts_b(key_rows: list[dict]) -> dict:
    cases = case_level(key_rows).values()
    dxa = [r for r in key_rows if r["target_source"] == "dxa"]
    return {
        "cases": len(cases), "r10": sum(c["has_r10"] for c in cases), "r5": sum(c["has_r5"] for c in cases),
        **{f"class_{k}": v for k, v in sorted(Counter(c["evidence_class"] for c in cases).items())},
        **{f"spec_{k}": v for k, v in sorted(Counter(c["spec_class"] for c in cases).items())},
        "benign_chargeable": sum(c["oc_chargeable"] for c in cases),
        "benign_chargeable_no_offlist_complication": sum(c["oc_chargeable"] and not c["offlist_complication"] for c in cases),
        "r10_targets": sum(1 for r in key_rows if r["in_r10"] is True),
        "r10_dxa_targets": sum(1 for r in dxa if r["in_r10"] is True),
        "r5_targets": sum(1 for r in key_rows if r["in_r5"] is True),
        "dxa_concerns": len(dxa), "dxa_status": dict(sorted(Counter(r["status"] for r in dxa).items())),
        "raised_by_monotone": sum(r["raised_by_monotone"] is True for r in dxa),
    }


def monotonicity_report(table: CellTable, t1: list[str]) -> dict:
    """Over every (condition, hallmark count, band from 5% up) class: how often pooling and the
    monotone guarantee act, and where SUPPORTED overrides the p / 10 clause. The v0.3 check
    reads each band at its lower edge, so it undercounts v0.3's within-band reversals."""
    out = Counter()
    for c in t1:
        for h in range(len(table.det.hallmarks[c]) + 1):
            cells = table.cells(c, h)
            st = akb.band_statuses([tuple(x) for x in cells])
            out["classes"] += sum(1 for n, _ in cells if n)
            out["pooled_classes"] += sum(1 for s in st if s.n and len(s.pooled_bands) > 1)
            out["raised_by_monotone"] += sum(1 for s in st if s.n and s.raised)
            out["supported_below_ratio"] += sum(1 for s in st if s.raw == akb.SUPPORTED and s.hi < s.threshold)
            v03 = [ak.rh_status(ak.BANDS[s.band], s.n, s.k) for s in st if s.n >= ak.MIN_N]
            out["v03_nonmonotone_series"] += any(a == ak.KEPT and b == ak.RED_HERRING for a, b in zip(v03, v03[1:]))
            v03b = [akb.RANK[s.status] for s in st]
            out["v03b_nonmonotone_series"] += any(b < a for a, b in zip(v03b, v03b[1:]))
    return dict(out)


# ---------------------------------------------------------------- comparisons


def old_classes(old_rows: list[dict]) -> dict[str, dict]:
    """Per case under v0.3: spec class, the #7 evidence class and #8 chargeability (both on the v0.3 targets)."""
    out = {}
    for cid, c in case_level(old_rows).items():
        conc = [r for r in old_rows if r["case_id"] == cid and r["source"] == "dxa"]
        exempt = any(not r["in_r5"] for r in conc)  # removed; v0.3 kept undetermined concerns as targets
        spec = "SERIOUS" if c["has_r10"] else "BENIGN" if c["clearly_low_risk"] else "MIDDLE" if c["truth_tier"] == 2 else "OTHER"
        ev = (akb.SERIOUS_TRUTH if c["truth_tier"] == 1 else akb.SERIOUS_DXA_ONLY if c["has_r10"]
              else akb.BENIGN if c["clearly_low_risk"] else akb.MIDDLE)
        out[cid] = {"spec": spec, "evidence": ev, "chargeable": c["clearly_low_risk"] and not exempt,
                    "r10": c["r10_targets"], "r5": c["r5_targets"], "truth": c["truth"]}
    return out


def switches(old_rows, new_rows) -> list[dict]:
    old, new = old_classes(old_rows), case_level(new_rows)
    out = []
    for cid, o in old.items():
        n = new[cid]
        if (o["spec"], o["evidence"], o["r10"], o["r5"]) != (n["spec_class"], n["evidence_class"], n["r10_targets"], n["r5_targets"]):
            conc = {r["condition"]: r for r in new_rows if r["case_id"] == cid and r["target_source"] == "dxa"}
            why = [f"{c}@{r['dxa_p']:.1f} v0.3 {r['status_v03']} -> {r['status']} (pooled {r['pooled_k']}/{r['pooled_n']}, "
                   f"95% {100 * r['wilson_lo']:.2f}-{100 * r['wilson_hi']:.2f}%)"
                   for c, r in conc.items() if (r["status_v03"] == ak.RED_HERRING) != (r["status"] == akb.UNSUPPORTED)
                   or (r["status_v03"] != ak.RED_HERRING) != (r["status"] == akb.SUPPORTED)]
            out.append({"case_id": cid, "truth": o["truth"], "spec_v03": o["spec"], "spec_v03b": n["spec_class"],
                        "evidence_v03": o["evidence"], "evidence_v03b": n["evidence_class"],
                        "r10_v03": o["r10"], "r10_v03b": n["r10_targets"], "r5_v03": o["r5"], "r5_v03b": n["r5_targets"],
                        "why": " ; ".join(why)})
    return out


def old_summary(old_rows) -> dict:
    oc = old_classes(old_rows)
    return {**key_counts(old_rows), **{f"class_{k}": v for k, v in sorted(Counter(c["evidence"] for c in oc.values()).items())},
            **{f"spec_{k}": v for k, v in sorted(Counter(c["spec"] for c in oc.values()).items())},
            "benign_chargeable_fix8_only": sum(c["chargeable"] for c in oc.values())}


# ---------------------------------------------------------------- controls (#10)


def draw_controls(pool_idx: dict[str, list[int]], cand: list[int], rows, seed: int) -> dict[str, list[dict]]:
    """One matched control per pool case, pools in POOLS order, cases in ID-list order, no control reused."""
    rng = np.random.default_rng(seed)
    by = {lvl: defaultdict(list) for lvl in MATCH_LEVELS}
    for j in cand:
        r = rows[j]
        by["initial_evidence+age_band"][(r["initial_evidence"], age_band(r["age"]))].append(j)
        by["initial_evidence"][r["initial_evidence"]].append(j)
        by["age_band"][age_band(r["age"])].append(j)
        by["any"][None].append(j)
    taken: set[int] = set()
    out = {}
    for name in POOLS:
        pairs = []
        for i in pool_idx[name]:
            r = rows[i]
            keys = {"initial_evidence+age_band": (r["initial_evidence"], age_band(r["age"])),
                    "initial_evidence": r["initial_evidence"], "age_band": age_band(r["age"]), "any": None}
            for lvl in MATCH_LEVELS:
                free = [j for j in by[lvl].get(keys[lvl], []) if j not in taken]
                if free:
                    j = free[int(rng.integers(len(free)))]
                    taken.add(j)
                    c = rows[j]
                    pairs.append({"pool_case": f"ddxplus_{r['i']}", "control": f"ddxplus_{c['i']}", "match_level": lvl,
                                  "pool_initial_evidence": r["initial_evidence"], "control_initial_evidence": c["initial_evidence"],
                                  "pool_age_band": age_band(r["age"]), "control_age_band": age_band(c["age"]),
                                  "_truth": c["path"], "_j": j})
                    break
        out[name] = pairs
    return out


# ---------------------------------------------------------------- main


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tiers = akb.load_tiers()
    if tiers != ak.load_tiers():
        raise SystemExit("spec/dangerous_if_missed_tiers_v03b.csv final_tier differs from the v0.3 tiers")
    levels = akb.load_evidence_levels()
    comps = akb.load_complications()
    t1 = ak.tier1_conditions(tiers)
    conditions = ak2.load_conditions(COND_JSON)
    conditions_meta = json.loads(COND_JSON.read_text())
    evid = json.loads(EVID_JSON.read_text())

    def is_antecedent(tok: str) -> bool:
        return evid.get(tok.partition("_@_")[0], {}).get("is_antecedent") in (True, "True")

    rows = [r for r in read_rows(TEST_CSV) if r["age"] >= ADULT_MIN_AGE]
    for r in rows:
        r["tokens"] = frozenset(str(e) for e in r["evidences"])
        r["dxa"] = {n: 100.0 * float(p) for n, p in r["differential"]}
    sample_ids = SAMPLE_IDS.read_text().split()
    sample_rows = {int(s.split("_")[1]) for s in sample_ids}
    ref = np.array([r["i"] not in sample_rows for r in rows])
    pos = {r["i"]: j for j, r in enumerate(rows)}
    sample_idx = [pos[int(s.split("_")[1])] for s in sample_ids]

    hallmarks = learn_hallmarks(rows, ref, t1, is_antecedent)
    det = Detector(rows, ref, t1, hallmarks)
    table = CellTable(det)

    # v0.3 key, rebuilt and checked against its pin
    old_rows = key_for(sample_idx, rows, ref, det, tiers)
    tmp = OUT / "v03_rebuilt.csv"
    if ak.write_key(tmp, old_rows) != ak.read_sha256(TS / "eval-v03-key.sha256"):
        raise SystemExit("rebuilt v0.3 key does not match data/test_sets/eval-v03-key.sha256")
    tmp.unlink()

    # 1. the 470 key
    new_rows = key_for_b(sample_idx, rows, ref, table, tiers, levels, comps)
    sha = akb.write_key(TS / "eval-v03b-key.csv", new_rows)
    ak.write_sha256(TS / "eval-v03b-key.sha256", TS / "eval-v03b-key.csv")
    sw = switches(old_rows, new_rows)
    write_csv(OUT / "switches.csv", sw)
    before, after = old_summary(old_rows), counts_b(new_rows)
    print(f"key470 v0.3:  {before}")
    print(f"key470 v0.3b: {after}; sha256 {sha}")

    # 2. pools, from the committed ID lists
    pool_idx, pool_counts = {}, {}
    for name in POOLS:
        ids = (TS / f"eval-v03-pool-{name}.case_ids.txt").read_text().split()
        pool_idx[name] = [pos[int(s.split("_")[1])] for s in ids]
        kr_old = key_for(pool_idx[name], rows, ref, det, tiers)
        if ak.write_key(OUT / "tmp.csv", kr_old) != ak.read_sha256(TS / f"eval-v03-pool-{name}.key.sha256"):
            raise SystemExit(f"rebuilt v0.3 {name} pool key does not match its pin")
        kr = key_for_b(pool_idx[name], rows, ref, table, tiers, levels, comps)
        stem = f"eval-v03b-pool-{name}"
        akb.write_key(TS / f"{stem}.key.csv", kr)
        ak.write_sha256(TS / f"{stem}.key.sha256", TS / f"{stem}.key.csv")
        pool_counts[name] = {"v03": old_summary(kr_old), "v03b": counts_b(kr)}
    (OUT / "tmp.csv").unlink()

    # 3. controls: chargeable BENIGN reference adults, pools excluded
    in_pool = {j for v in pool_idx.values() for j in v}
    cand = [j for j, r in enumerate(rows) if ref[j] and j not in in_pool and tiers.get(r["path"], 3) == 3
            and not ak2.red_flags(r["evidences"]) and not any(r["dxa"].get(c, 0.0) >= akb.R5 for c in t1)]
    pairs = draw_controls(pool_idx, cand, rows, CONTROL_SEED)
    control_counts = {}
    for name in POOLS:
        stem = f"eval-v03b-pool-{name}-controls"
        idx = [p["_j"] for p in pairs[name]]
        kr = key_for_b(idx, rows, ref, table, tiers, levels, comps)
        bad = [r["case_id"] for r in case_level(kr).values() if not r["oc_chargeable"]]
        if bad:
            raise SystemExit(f"{stem}: controls not chargeable BENIGN: {bad[:5]}")
        akb.write_key(TS / f"{stem}.key.csv", kr)
        ak.write_sha256(TS / f"{stem}.key.sha256", TS / f"{stem}.key.csv")
        write_csv(TS / f"{stem}.pairs.csv", [{k: v for k, v in p.items() if not k.startswith("_")} for p in pairs[name]])
        cases = [build_case(rows[j], conditions_meta, conditions) for j in idx]
        ids_text = "".join(f"{c['case_id']}\n" for c in cases)
        (TS / f"{stem}.case_ids.txt").write_text(ids_text)
        meta = {"test_set_name": stem, "source_file": str(TEST_CSV.relative_to(ROOT)),
                "filter": f"age >= {ADULT_MIN_AGE}; main sample and both pools excluded; chargeable BENIGN: truth tier 3, "
                          "no red flag, no tier-1 condition at DXA p >= 5%",
                "matching": f"1:1 per case of eval-v03-pool-{name}, in ID-list order; levels {list(MATCH_LEVELS)}; "
                            f"age bands {[age_band(lo) for lo, _ in AGE_BANDS]}; uniform among the unused candidates at the "
                            f"first level with any; numpy default_rng({CONTROL_SEED}), pools in order {list(POOLS)}, "
                            "no control reused across pools",
                "seed": CONTROL_SEED, "cases": len(cases), "case_ids_sha256": hashlib.sha256(ids_text.encode()).hexdigest(),
                "answer_key": f"{stem}.key.csv (evaluator/answer_key_v03b.py)", "pairs": f"{stem}.pairs.csv"}
        (TS / f"{stem}.json").write_text(json.dumps({"metadata": meta, "cases": cases}, indent=2) + "\n")
        control_counts[name] = {"controls": len(idx), "pool_cases": len(pool_idx[name]),
                                "match_levels": dict(Counter(p["match_level"] for p in pairs[name])),
                                "control_truths": dict(Counter(p["_truth"] for p in pairs[name]).most_common())}
        print(f"{stem}: {control_counts[name]}")

    # 4. sensitivity: the formal category rule's tiers
    formal = akb.load_tiers(column="final_tier_formal")
    formal_counts = None
    if formal != tiers:
        t1f = ak.tier1_conditions(formal)
        hf = learn_hallmarks(rows, ref, t1f, is_antecedent)
        tf = CellTable(Detector(rows, ref, t1f, hf))
        formal_counts = counts_b(key_for_b(sample_idx, rows, ref, tf, formal, levels, comps))
        formal_counts["changed_conditions"] = sorted(c for c in formal if formal[c] != tiers[c])

    old_by, new_by = old_classes(old_rows), case_level(new_rows)
    exempt_now = [c for c in new_by.values() if c["clearly_low_risk"] and not c["oc_chargeable"]]
    summary = {
        "key470_v03": before, "key470_v03b": after, "key470_v03b_sha256": sha,
        "switch_counts": dict(Counter(f"{s['spec_v03']}->{s['spec_v03b']}" for s in sw)),
        "evidence_switch_counts": dict(Counter(f"{s['evidence_v03']}->{s['evidence_v03b']}" for s in sw)),
        "benign_v03_by_truth": dict(Counter(c["truth"] for c in old_by.values() if c["spec"] == "BENIGN").most_common()),
        "benign_exempt_v03b_by_truth": dict(Counter(c["truth"] for c in exempt_now).most_common()),
        "benign_chargeable_v03b_by_truth": dict(Counter(c["truth"] for c in new_by.values() if c["oc_chargeable"]).most_common()),
        "chargeable_with_offlist_complication": dict(Counter(c["truth"] for c in new_by.values()
                                                             if c["oc_chargeable"] and c["offlist_complication"]).most_common()),
        "dxa_only_v03b_by_target": dict(Counter(t for c in new_by.values() if c["evidence_class"] == akb.SERIOUS_DXA_ONLY
                                                for t in c["r10_dxa_targets"].split("|")).most_common()),
        "monotonicity": monotonicity_report(table, t1),
        "pools": pool_counts, "controls": control_counts, "formal_tier_sensitivity": formal_counts,
        "views_check": kv.class_counts(akb.load_key()),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    print(json.dumps({k: summary[k] for k in ("switch_counts", "evidence_switch_counts", "monotonicity", "formal_tier_sensitivity")}, default=str))


if __name__ == "__main__":
    main()
