#!/usr/bin/env python3
"""
Build the v0.3 answer key for the 470 sample under the two alternate tier rules that
spec/v0.3-scoring.md section 6 names as sensitivity rows:

1. severity: tiers from DDXPlus severity only (the `base_tier` column of
   spec/dangerous_if_missed_tiers_v03.csv; 17 tier-1 conditions);
2. one-source: the v0.3 tiers plus the four conditions a single independent source
   names (atrial fibrillation, anemia, HIV, SLE; docs/tier-upgrade-sources.md section 5).

We reuse scripts/build_v03_key.py unchanged: the same reference adults, hallmark
learner, class counts and evaluator/answer_key_v03.py rules. Hallmarks are learned
per condition, so the 21 v0.3 tier-1 conditions keep the classes of the pinned key
and only the added conditions get new ones. As a check we rebuild the v0.3 key the
same way and refuse to write anything unless its sha256 equals the pinned one.

Outputs (data/test_sets/ is git-ignored; the sha256 files are force-added):
  data/test_sets/eval-v03-key.tiers-{severity,one-source}.csv and .sha256

Usage: python3 scripts/build_v03_tier_variants.py   (about a minute; reads DDXPlus only)
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluator import answer_key_v03 as ak  # noqa: E402
from build_v03_key import (EVID_JSON, SAMPLE_IDS, TEST_CSV, TS, ADULT_MIN_AGE, Detector, key_counts,  # noqa: E402
                           key_for, learn_hallmarks, read_rows)

ONE_SOURCE_EXTRA = ("Atrial fibrillation", "Anemia", "HIV (initial infection)", "SLE")
VARIANT_FILES = {"severity": "eval-v03-key.tiers-severity", "one-source": "eval-v03-key.tiers-one-source"}


def tier_tables() -> dict[str, dict[str, int]]:
    with open(ak.TIERS_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    v03 = {r["condition"]: int(r["final_tier"]) for r in rows}
    sev = {r["condition"]: int(r["base_tier"]) for r in rows}
    one = dict(v03)
    for c in ONE_SOURCE_EXTRA:
        one[c] = 1
    return {"v03": v03, "severity": sev, "one-source": one}


def main() -> None:
    tables = tier_tables()
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

    all_t1 = sorted({c for t in tables.values() for c, v in t.items() if v == 1})
    hallmarks = learn_hallmarks(rows, ref, all_t1, is_antecedent)
    det = Detector(rows, ref, all_t1, hallmarks)

    check = TS / "eval-v03-key.rebuild-check.csv"
    got = ak.write_key(check, key_for(sample_idx, rows, ref, det, tables["v03"]))
    check.unlink()
    want = ak.read_sha256(ak.KEY_SHA256)
    if got != want:
        raise SystemExit(f"rebuilt v0.3 key sha256 {got} differs from the pinned {want}; not writing variants")
    print(f"v0.3 key rebuilt identically (sha256 {got[:12]})")

    for name, stem in VARIANT_FILES.items():
        kr = key_for(sample_idx, rows, ref, det, tables[name])
        path = TS / f"{stem}.csv"
        sha = ak.write_key(path, kr)
        ak.write_sha256(TS / f"{stem}.sha256", path)
        n_t1 = sum(1 for v in tables[name].values() if v == 1)
        print(f"{name}: {n_t1} tier-1 conditions; {key_counts(kr)}; sha256 {sha[:12]}")


if __name__ == "__main__":
    main()
