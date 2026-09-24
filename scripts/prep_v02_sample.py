#!/usr/bin/env python3
"""
Build the MedSafe-Dx v0.2 case set (spec/v0.2-scoring.md section 1).

We draw 10 adult patients (age >= 18) per condition from the DDXPlus test split
with a fixed seed, and take every adult patient of a condition that has fewer
than 10. Selection reads DDXPlus fields only, never model outputs.

The draw is the adults-only stratified draw of scripts/analysis/v02_review_policies.py
(numpy default_rng(20260923), conditions in sorted order, rows kept in file order),
so the adversarial review's numbers describe this exact sample.

Each case keeps the v0 case format (data/cases.py convert_case) and adds the
answer-key fields of spec section 4 (evaluator/answer_key_v02.py): DDXPlus
severity of the true condition, `serious`, `p_serious_risk`, `at_risk` and
`clearly_low_risk`. `red_flag_indicators`
stays empty because the prompt shows it to the model; the section 7 flag lives
in `offlist_red_flag`, which the prompt never reads.

Outputs:
  data/test_sets/eval-v02-adult.json
  data/test_sets/eval-v02-adult.case_ids.txt

Usage: .venv/bin/python scripts/prep_v02_sample.py
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.cases import convert_case  # noqa: E402
from evaluator.answer_key_v02 import load_conditions, red_flags, risk_fields  # noqa: E402

SEED = 20260923
PER_CONDITION = 10
ADULT_MIN_AGE = 18
TEST_CSV = ROOT / "data" / "ddxplus_v0" / "release_test_patients"
CONDITIONS_JSON = ROOT / "data" / "ddxplus_v0" / "release_conditions.json"
OUT_JSON = ROOT / "data" / "test_sets" / "eval-v02-adult.json"
OUT_IDS = ROOT / "data" / "test_sets" / "eval-v02-adult.case_ids.txt"


def read_rows(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for i, r in enumerate(csv.DictReader(f)):
            rows.append({
                "i": i,
                "age": int(r["AGE"]),
                "sex": r["SEX"],
                "path": r["PATHOLOGY"],
                "evidences": ast.literal_eval(r["EVIDENCES"]),
                "differential": ast.literal_eval(r["DIFFERENTIAL_DIAGNOSIS"]),
                "initial_evidence": r["INITIAL_EVIDENCE"],
            })
    return rows


def stratified_adults(rows: list[dict], seed: int, per_condition: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for r in rows:
        if r["age"] >= ADULT_MIN_AGE:
            by[r["path"]].append(r)
    out = []
    for cond in sorted(by):
        pool = by[cond]
        k = min(per_condition, len(pool))
        idx = rng.choice(len(pool), size=k, replace=False)
        out.extend(pool[j] for j in sorted(idx))
    return out


def build_case(r: dict, conditions_meta: dict, conditions: dict) -> dict:
    raw = {
        "id": r["i"],
        "age": r["age"],
        "sex": r["sex"],
        "symptoms": r["evidences"],
        "diagnoses": [{"condition_name": n, "probability": p} for n, p in r["differential"][:3]],
        "duration": "unknown",
    }
    diseases = {k: {"icd10-id": v.get("icd10-id"), "severity": v.get("severity")} for k, v in conditions_meta.items()}
    case = convert_case(raw, diseases)
    if case is None:
        raise ValueError(f"row {r['i']} did not convert")
    flags = red_flags(r["evidences"])
    differential = [[n, float(p)] for n, p in r["differential"]]
    case.update({
        "true_pathology": r["path"],
        "true_icd10": conditions[r["path"]]["icd10"],
        "ddxplus_severity": conditions[r["path"]]["ddxplus_severity"],
        **risk_fields(r["path"], differential, conditions),
        "offlist_red_flag": bool(flags),
        "offlist_red_flag_reasons": flags,
        "initial_evidence": r["initial_evidence"],
        "ddxplus_differential": differential,
    })
    return case


def summarise(cases: list[dict]) -> dict:
    n = len(cases)
    per_cond = Counter(c["true_pathology"] for c in cases)
    reasons = Counter(r for c in cases for r in c["offlist_red_flag_reasons"])
    return {
        "cases": n,
        "conditions": len(per_cond),
        "per_condition": dict(sorted(per_cond.items())),
        "serious": sum(c["serious"] for c in cases),
        "at_risk_not_serious": sum(c["at_risk"] and not c["serious"] for c in cases),
        "clearly_low_risk": sum(c["clearly_low_risk"] for c in cases),
        "offlist_red_flag_cases": sum(c["offlist_red_flag"] for c in cases),
        "offlist_red_flag_by_reason": dict(reasons),
        # Measure B's denominator: clearly low-risk, red-flag patients removed (spec section 7).
        "clearly_low_risk_no_red_flag": sum(c["clearly_low_risk"] and not c["offlist_red_flag"] for c in cases),
        "age_65_plus": sum(c["age"] >= 65 for c in cases),
        "ddxplus_severity_counts": dict(sorted(Counter(c["ddxplus_severity"] for c in cases).items())),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--per-condition", type=int, default=PER_CONDITION)
    ap.add_argument("--out", type=Path, default=OUT_JSON)
    ap.add_argument("--ids-out", type=Path, default=OUT_IDS)
    args = ap.parse_args()

    conditions = load_conditions(CONDITIONS_JSON)
    conditions_meta = json.loads(CONDITIONS_JSON.read_text())

    rows = read_rows(TEST_CSV)
    sample = stratified_adults(rows, args.seed, args.per_condition)
    cases = [build_case(r, conditions_meta, conditions) for r in sample]
    summary = summarise(cases)

    ids = [c["case_id"] for c in cases]
    ids_text = "".join(f"{i}\n" for i in ids)
    metadata = {
        "test_set_name": "eval-v02-adult",
        "source_file": str(TEST_CSV.relative_to(ROOT)),
        "source_rows": len(rows),
        "adult_rows": sum(r["age"] >= ADULT_MIN_AGE for r in rows),
        "filter": f"age >= {ADULT_MIN_AGE}",
        "sampling": f"{args.per_condition} per condition (all if fewer), numpy default_rng({args.seed}), conditions sorted by name",
        "seed": args.seed,
        "sampled_cases": len(cases),
        "total_available_cases": sum(r["age"] >= ADULT_MIN_AGE for r in rows),
        "case_ids_sha256": hashlib.sha256(ids_text.encode()).hexdigest(),
        "answer_key": "DDXPlus severity <= 2 of the true condition; DXA differential mass on severity <= 2 (evaluator/answer_key_v02.py)",
        "summary": summary,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"metadata": metadata, "cases": cases}, indent=2) + "\n")
    args.ids_out.write_text(ids_text)

    s = summary
    print(f"{s['cases']} cases over {s['conditions']} conditions -> {args.out.relative_to(ROOT)}")
    short = {c: n for c, n in s["per_condition"].items() if n < args.per_condition}
    print(f"per condition: {args.per_condition} each" + (f"; fewer: {short}" if short else "; none short"))
    print(f"serious {s['serious']}; at risk, not serious {s['at_risk_not_serious']}; "
          f"clearly low-risk {s['clearly_low_risk']} ({s['clearly_low_risk_no_red_flag']} without red flags)")
    print(f"off-list red flag: {s['offlist_red_flag_cases']} cases {s['offlist_red_flag_by_reason']}")
    print(f"DDXPlus severity {s['ddxplus_severity_counts']}; age 65+ {s['age_65_plus']}")


if __name__ == "__main__":
    main()
