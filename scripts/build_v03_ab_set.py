#!/usr/bin/env python3
"""
Build the case file for the v0.3 draft-3 prompt test (spec/v0.3-scoring.md section 12).

We read the 150 seeded case IDs that scripts/analysis/v03_hypothesis_design.py drew
(results/analysis/v03_hypothesis/ablation_cases.txt; strata in ablation_strata.csv),
take each case from the main sample or its pool, and add the working diagnosis and
the case class (evaluator/working_diagnosis.py). The same file feeds all three arms,
so every arm answers the same cases with the same anchor.

Outputs, in data/test_sets/:
1. eval-v03-ab150.json: the cases, each with `working_diagnosis`,
   `working_diagnosis_name` (the prompt rendering) and `ab_stratum`.
2. eval-v03-ab150.case_ids.txt: the IDs in draw order (committed).
3. eval-v03-ab150.design.csv and .sha256: per case, the class, the working diagnosis
   and the R10 / R5 targets; the scorer refuses a design file whose hash differs.

We check each main-sample case against results/analysis/v03_hypothesis/hypothesis_cases.csv,
so the file agrees with the design document's numbers.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluator import answer_key_v03 as ak  # noqa: E402
from evaluator import v03_score as vs  # noqa: E402
from evaluator import working_diagnosis as wd  # noqa: E402

HYP = ROOT / "results/analysis/v03_hypothesis"
TS = ROOT / "data/test_sets"
STEM = "eval-v03-ab150"
SOURCE_SETS = ("main", "pool-atypical", "pool-high-risk")


def dxa_of(case: dict) -> dict[str, float]:
    dd = case["ddxplus_differential"]
    if isinstance(dd, str):
        dd = ast.literal_eval(dd)
    return {c: 100.0 * p for c, p in dd}


def main() -> None:
    ids = (HYP / "ablation_cases.txt").read_text().split()
    with open(HYP / "ablation_strata.csv", newline="") as f:
        strata = [(r["stratum"], int(r["drawn"])) for r in csv.DictReader(f)]
    stratum_of = {}
    pos = 0
    for s, n in strata:
        for c in ids[pos:pos + n]:
            stratum_of[c] = s
        pos += n
    assert pos == len(ids) == 150, (pos, len(ids))

    tiers = ak.load_tiers()
    fallback = wd.load_fallback()
    icd10 = wd.load_icd10()
    cases, keys, source = {}, {}, {}
    for name in SOURCE_SETS:
        spec = vs.SETS[name]
        for c in json.loads(Path(spec["cases"]).read_text())["cases"]:
            cases[c["case_id"]] = c
            source[c["case_id"]] = name
        keys.update(ak.load_key(Path(spec["key"]), Path(spec["sha"])))

    with open(HYP / "hypothesis_cases.csv", newline="") as f:
        expected = {r["case_id"]: r for r in csv.DictReader(f)}

    out_cases, design_rows = [], []
    for cid in ids:
        c, k = cases[cid], keys[cid]
        d = wd.design_for(k, dxa_of(c), c["initial_evidence"], tiers, fallback, icd10)
        if cid in expected:
            e = expected[cid]
            assert (e["hypothesis"], e["class"]) == (d.working_diagnosis, d.klass), (cid, e["hypothesis"], e["class"], d)
        out_cases.append(c | {"working_diagnosis": d.working_diagnosis, "working_diagnosis_name": d.rendering(),
                              "ab_stratum": stratum_of[cid], "source_set": source[cid]})
        design_rows.append({"case_id": cid, "source_set": source[cid], "truth": d.truth, "truth_tier": d.truth_tier,
                            "class": d.klass, "working_diagnosis": d.working_diagnosis,
                            "working_diagnosis_icd10": d.working_diagnosis_icd10,
                            "working_diagnosis_p": round(d.working_diagnosis_p, 2),
                            "working_diagnosis_fallback": d.working_diagnosis_fallback,
                            "working_diagnosis_is_truth": d.working_diagnosis_is_truth,
                            "r10_targets": "|".join(d.r10), "r5_targets": "|".join(d.r5)})

    ids_text = "".join(f"{c}\n" for c in ids)
    (TS / f"{STEM}.case_ids.txt").write_text(ids_text)
    design = TS / f"{STEM}.design.csv"
    with open(design, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=wd.DESIGN_COLUMNS)
        w.writeheader()
        w.writerows(design_rows)
    ak.write_sha256(TS / f"{STEM}.design.sha256", design)
    classes = Counter(r["class"] for r in design_rows)
    meta = {"test_set_name": STEM, "source": "the main sample and both v0.3 pools",
            "selection": "scripts/analysis/v03_hypothesis_design.py section 6, seed 20260923, round-robin over truth "
                         "conditions within strata: " + ", ".join(f"{s} {n}" for s, n in strata),
            "seed": 20260923, "cases": len(ids), "case_ids_sha256": hashlib.sha256(ids_text.encode()).hexdigest(),
            "classes": dict(classes), "conditions": len({r["truth"] for r in design_rows}),
            "design": f"{STEM}.design.csv (evaluator/working_diagnosis.py)", "builder": "scripts/build_v03_ab_set.py"}
    (TS / f"{STEM}.json").write_text(json.dumps({"metadata": meta, "cases": out_cases}, indent=2) + "\n")
    wds = Counter(r["working_diagnosis"] for r in design_rows)
    print(f"{STEM}: {len(ids)} cases, {meta['conditions']} truth conditions, classes {dict(classes)}")
    print(f"  working diagnosis = truth: {sum(r['working_diagnosis_is_truth'] for r in design_rows)}; "
          f"fallback: {sum(r['working_diagnosis_fallback'] for r in design_rows)}")
    print(f"  working diagnoses: {dict(wds.most_common())}")


if __name__ == "__main__":
    main()
