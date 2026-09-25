#!/usr/bin/env python3
"""
Write spec/dangerous_if_missed_tiers_v03b.csv (key fix #11): the v0.3 tier table with
evidence strength, the formal category rule and the asthma count-floor flag
(evaluator/tier_evidence_v03b.py). It refuses to write when the recomputed v0.3 tiers
differ from spec/dangerous_if_missed_tiers_v03.csv, because the source rows would then
not reproduce the pre-registered table.

Usage: python3 scripts/build_v03b_tiers.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluator import tier_evidence_v03b as te  # noqa: E402

V03 = ROOT / "spec/dangerous_if_missed_tiers_v03.csv"
V03B = ROOT / "spec/dangerous_if_missed_tiers_v03b.csv"


def build() -> list[dict]:
    with open(V03, newline="", encoding="utf-8") as f:
        old = list(csv.DictReader(f))
    rows = [te.table_row(r["condition"], r["icd10"], int(r["ddxplus_severity"])) for r in old]
    bad = [(r["condition"], o["final_tier"], r["final_tier"]) for r, o in zip(rows, old) if int(o["final_tier"]) != r["final_tier"]]
    if bad:
        raise SystemExit(f"v0.3 tiers not reproduced: {bad}")
    return rows


def main() -> None:
    rows = build()
    with open(V03B, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(te.TABLE_COLUMNS), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    t1 = [r for r in rows if r["final_tier"] == 1]
    print(f"wrote {V03B.relative_to(ROOT)}: {len(rows)} conditions, {len(t1)} tier 1")
    for lvl in (te.LEVEL_NT_NAMED, te.LEVEL_TWO_OF_THREE, te.LEVEL_NT_CATEGORY, te.LEVEL_SEVERITY):
        print(f"  tier 1 by {lvl}: {sorted(r['condition'] for r in t1 if r['evidence_level'] == lvl)}")
    for r in rows:
        if r["tier_change_formal"]:
            print(f"  formal rule changes {r['condition']}: {r['final_tier']} -> {r['final_tier_formal']} ({r['evidence_level_formal']})")
        if r["count_floor_sensitive"]:
            print(f"  count floor 2 changes {r['condition']}: {r['final_tier']} -> {r['final_tier_count_floor']}")


if __name__ == "__main__":
    main()
