#!/usr/bin/env python3
"""Build evaluator/data/v02_condition_mix.csv: the two re-weighting mixes of spec section 6b.

The v0.2 sample weights every condition equally (10 patients each). Section 6b
asks for A, B and the headline re-weighted to two other condition mixes, so a
reader can see how much the pooled numbers depend on the mix:

1. DDXPlus mix: the condition's count among adult rows (age >= 18) of the DDXPlus
   test split, the population the sample was drawn from.
2. NHAMCS mix: CDC NHAMCS 2016-2022 weighted ED visits with the condition as
   first-listed diagnosis, adults only (18-64 plus 65+), from
   results/analysis/nhamcs_urgency/condition_summary.csv
   (scripts/analysis/nhamcs_urgency.py; raw files and checksums in data/external/nhamcs/).
   A condition is covered when it has at least 30 sampled primary-diagnosis visits
   at all ages, the rule in docs/third-party-urgency-source.md section 3. Uncovered
   conditions get no NHAMCS weight and drop out of the NHAMCS-mix rates.

We commit the output, because the NHAMCS summary sits under the git-ignored
results/ tree and the scorer must reproduce without the 300 MB raw files.

Usage:
    python3 scripts/build_v02_condition_mix.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DDX_CSV = ROOT / "data/ddxplus_v0/release_test_patients"
NHAMCS = ROOT / "results/analysis/nhamcs_urgency/condition_summary.csv"
SAMPLE = ROOT / "data/test_sets/eval-v02-adult.json"
OUT = ROOT / "evaluator/data/v02_condition_mix.csv"
MIN_VISITS = 30


def main() -> None:
    conditions = sorted({c["true_pathology"] for c in json.loads(SAMPLE.read_text())["cases"]})
    adult = Counter()
    with open(DDX_CSV) as f:
        for row in csv.DictReader(f):
            if int(row["AGE"]) >= 18:
                adult[row["PATHOLOGY"]] += 1
    nh: dict[str, dict] = {}
    with open(NHAMCS) as f:
        for r in csv.DictReader(f):
            if r["scope"] != "primary":
                continue
            d = nh.setdefault(r["condition"], {"adult": 0.0, "n_all": 0})
            if r["age_band"] in ("18-64", "65+"):
                d["adult"] += float(r["weighted_visits"] or 0)
            if r["age_band"] == "all":
                d["n_all"] = int(r["n_unweighted"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "ddxplus_adult_rows", "nhamcs_n_primary_all_ages", "nhamcs_covered",
                    "nhamcs_weighted_adult_visits_2016_2022"])
        for c in conditions:
            d = nh.get(c)
            if d is None:
                sys.exit(f"{c} missing from {NHAMCS}")
            covered = d["n_all"] >= MIN_VISITS
            w.writerow([c, adult[c], d["n_all"], int(covered), round(d["adult"], 1) if covered else 0])
    print(f"wrote {OUT} ({len(conditions)} conditions)")


if __name__ == "__main__":
    main()
