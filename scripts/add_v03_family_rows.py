#!/usr/bin/env python3
"""
Add the clinical-family rows of docs/v0.3-validity-review.md section 1.1 to
spec/ddxplus_icd10_map.csv as "broader" relations (spec/v0.3-scoring.md draft 3,
section 4).

A family row lets a code in a condition's clinical family name that condition under
the standard policy: heart failure I50 names acute pulmonary oedema, a tachyarrhythmia
names PSVT, an ischaemia code names each angina / MI condition. We need them because
the validity review found 31 and 52 target misses per model that were a same-family
code the map did not credit. Draft 3 uses the standard policy only for the
descriptive rows (target listed, plausible tier-1 listed), and the strict policy
(equivalent, narrower) for diagnosis, so the family rows never touch the headline or
the top-1 / top-5 rows.

For each condition X in `FAMILY` we take its ICD-10 prefixes and every code that
names one of its family conditions under the standard policy, and then:
1. add a row (X, code, broader) when X has no row for that code;
2. add a "related" row for a code a family condition lists as related under one of
   its covered codes, so X counts exactly the codes that name a family condition;
3. raise X's own row to broader when it is "related", and likewise any longer
   "related" row of X that the family covers, because the longest-prefix rule would
   otherwise let the related row win.
Rows we add or raise carry a note naming the review. The script is idempotent: a
second run changes nothing. It prints what it changed.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluator.v02_score import MAP_CSV, ConditionMap  # noqa: E402

STANDARD = ("equivalent", "narrower", "broader")
NOTE = "family row (docs/v0.3-validity-review.md section 1.1)"
ISCHAEMIA = ("I20", "I21", "I22", "I24", "I25")
AIRWAY = ("J05", "J36", "J38.0", "J38.4", "J38.5", "J38.6", "J38.7", "J39.0", "R06.1", "T17")
# condition -> (ICD-10 prefixes, family conditions). The same table as FAMILY in
# scripts/analysis/v03_validity_failures.py, with the dots written back.
FAMILY: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "Stable angina": (ISCHAEMIA, ("Unstable angina", "Possible NSTEMI / STEMI")),
    "Unstable angina": (ISCHAEMIA, ("Stable angina", "Possible NSTEMI / STEMI")),
    "Possible NSTEMI / STEMI": (ISCHAEMIA, ("Stable angina", "Unstable angina")),
    "Myocarditis": (("I40", "I41", "I51.4", "I30", "I31.3"), ("Pericarditis",)),
    "PSVT": (("I47", "I48", "I49", "R00.0"), ("Atrial fibrillation",)),
    "Acute pulmonary edema": (("I50", "J81"), ()),
    "Scombroid food poisoning": (("T78.0", "T78.1", "T78.2", "T61.1"), ("Anaphylaxis",)),
    "Anaphylaxis": (("T78", "T61.1"), ("Scombroid food poisoning",)),
    "Larygospasm": (AIRWAY, ("Epiglottitis", "Croup")),
    "Epiglottitis": (AIRWAY, ("Larygospasm", "Croup")),
    "Croup": (AIRWAY, ("Epiglottitis", "Larygospasm")),
    "Bronchospasm / acute asthma exacerbation": (("J44", "J45", "J46"), ("Acute COPD exacerbation / infection",)),
    "Acute dystonic reactions": (("G24", "G25"), ()),
    "Guillain-Barré syndrome": (("G61",), ()),
    "Boerhaave": (("K22.3",), ()),
    "Pneumonia": (("J12", "J13", "J14", "J15", "J16", "J17", "J18", "J85", "J86"), ()),
    "Pulmonary neoplasm": (("C34", "C39", "C78.0"), ()),
    "Pancreatic neoplasm": (("C25",), ()),
    "Spontaneous pneumothorax": (("J93",), ()),
    "Pulmonary embolism": (("I26", "I82"), ()),
    "Ebola": (("A98.4",), ()),
}


def dotted(code: str) -> str:
    """Normalised code back to the map's dotted form (I509 -> I50.9)."""
    return code if len(code) <= 3 or "." in code else f"{code[:3]}.{code[3:]}"


def original_rows(rows: list[dict]) -> dict[str, dict[str, str]]:
    """condition -> {normalised code: relation} as the map stood before any family row, so a second run
    computes the same family cover as the first: family rows are left out and raised rows count as related."""
    out: dict[str, dict[str, str]] = {}
    for r in rows:
        if r["system"] == "family":
            continue
        rel = "related" if "relation raised from related" in r["note"] else r["relation"]
        out.setdefault(r["condition"], {})[ConditionMap.norm(r["code"])] = rel
    return out


def main(path: Path = MAP_CSV) -> None:
    norm = ConditionMap.norm
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    before = original_rows(rows)
    ddx_code = {r["condition"]: r["ddxplus_icd10"] for r in rows}
    added = raised = 0
    for cond, (prefixes, family) in FAMILY.items():
        own = {norm(r["code"]) for r in rows if r["condition"] == cond}
        cover = {norm(p): f"clinical-family prefix {p}" for p in prefixes}
        for fc in family:
            for code, rel in before[fc].items():
                if rel in STANDARD:
                    cover.setdefault(code, f"names {fc} ({rel}), a family condition")
        for r in rows:
            if r["condition"] != cond or r["relation"] in STANDARD or r["system"] == "family":
                continue
            c = norm(r["code"])
            why = next((w for p, w in cover.items() if c.startswith(p)), None)
            if why is None:
                continue
            r["relation"] = "broader"
            r["note"] = f"{r['note']}; relation raised from related: {NOTE}, {why}"
            raised += 1
        # A family condition's own "related" rows under a covered code stay non-counting for X too, so X
        # counts exactly the codes that name a family condition (e.g. I31.2 haemopericardium is related to
        # pericarditis, so it stays related to myocarditis under pericarditis's broader I31 row).
        exceptions = {}
        for fc in family:
            for code, rel in before[fc].items():
                if rel not in STANDARD and any(code.startswith(c) for c in cover if not cover[c].startswith("clinical")) \
                        and not any(code.startswith(norm(p)) for p in prefixes):
                    exceptions.setdefault(code, f"{fc} lists it as {rel}, so the family does not cover it")
        for code, why in sorted(cover.items()):
            if code in own:
                continue
            rows.append({"condition": cond, "ddxplus_icd10": ddx_code[cond], "code": dotted(code), "system": "family",
                         "relation": "broader", "source_url": "docs/v0.3-validity-review.md",
                         "note": f"{NOTE}: {why}"})
            added += 1
        for code, why in sorted(exceptions.items()):
            if code in own or code in cover:
                continue
            rows.append({"condition": cond, "ddxplus_icd10": ddx_code[cond], "code": dotted(code), "system": "family",
                         "relation": "related", "source_url": "docs/v0.3-validity-review.md",
                         "note": f"{NOTE}, exception: {why}"})
            added += 1
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"{path.name}: {added} family rows added, {raised} related rows raised to broader")


if __name__ == "__main__":
    main()
