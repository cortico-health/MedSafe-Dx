#!/usr/bin/env python3
"""
Build the case file and answer key for the v0.3 Phase 2 validation (docs/v0.3-case-selection-rules.md section 7).

We read the 250 ids that scripts/analysis/v03_case_selection.py drew with seed 20261003
(results/analysis/case_selection/phase2_candidates.csv), build the v0.3 key for them with the
detector of scripts/build_v03_key.py (reference adults = every adult outside the 470 sample, leave
one out), and render each case the way the 150 were rendered (scripts/build_v03_ab_set.py: the
DDXPlus case plus the working diagnosis of evaluator/working_diagnosis.py), so the models see
exactly what the reviewers see.

The DXA-only stratum needs the key before review (section 7): a case stays in it only when a kept
(not red-herring) DXA-derived R10 target survives the selection rules, so the case is still SERIOUS
by a DXA-only target. We replace every drawn case that fails with the next case in the draw's own
shuffled order of that stratum's pool (seed 20261003), until the stratum holds 20. The other
strata keep their draw; their classes come from the rules applied with the key's targets, and a
case whose class moves is reported, not replaced.

Checks before writing: the rebuilt 470 key must match data/test_sets/eval-v03-key.sha256, and
the reproduced draw must equal phase2_candidates.csv.

Outputs, in data/test_sets/ (git-ignored; the ids, hashes and the stratum file are force-added):
  eval-v03-phase2.json              the 250 cases with `working_diagnosis`, `working_diagnosis_name`,
                                    `phase2_stratum` and `phase2_rules`
  eval-v03-phase2.case_ids.txt      the 250 ids in draw order (the Phase 2 set)
  eval-v03-phase2.run_ids.txt       the 250 plus the replaced DXA-only cases, which the file and key also hold
  eval-v03-phase2.key.csv, .sha256  the v0.3 key rows
  eval-v03-phase2.strata.csv        per case: stratum, the rules that decided it, the class under the
                                    key, and whether it replaced a drawn case

Usage: python3 scripts/build_v03_phase2_set.py   (about 2 minutes: the full-split pass)
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

from evaluator import answer_key_v02 as ak2  # noqa: E402
from evaluator import answer_key_v03 as ak  # noqa: E402
from evaluator import working_diagnosis as wd  # noqa: E402
from build_v03_key import (ADULT_MIN_AGE, COND_JSON, EVID_JSON, SAMPLE_IDS, TEST_CSV, TS, Detector,  # noqa: E402
                           key_for, learn_hallmarks)
from prep_v02_sample import build_case, read_rows  # noqa: E402
import v03_case_selection as cs  # noqa: E402

STEM = "eval-v03-phase2"
CANDIDATES = ROOT / "results" / "analysis" / "case_selection" / "phase2_candidates.csv"
DXA_STRATUM = "serious_dxa_only"
DROPPED = "serious_dxa_only_dropped"  # drawn, then replaced because no DXA-only target survived the key


def detector():
    tiers = ak.load_tiers()
    t1 = ak.tier1_conditions(tiers)
    evid = json.loads(EVID_JSON.read_text())

    def is_antecedent(tok: str) -> bool:
        return evid.get(tok.partition("_@_")[0], {}).get("is_antecedent") in (True, "True")

    rows = [r for r in read_rows(TEST_CSV) if r["age"] >= ADULT_MIN_AGE]
    for r in rows:
        r["tokens"] = frozenset(str(e) for e in r["evidences"])
        r["dxa"] = {n: 100.0 * float(p) for n, p in r["differential"]}
    sample_rows = {int(s.split("_")[1]) for s in SAMPLE_IDS.read_text().split()}
    ref = np.array([r["i"] not in sample_rows for r in rows])
    det = Detector(rows, ref, t1, learn_hallmarks(rows, ref, t1, is_antecedent))
    pos = {r["i"]: j for j, r in enumerate(rows)}
    # The 470 key rebuilt in memory must match the pinned key, so the detector is the one the 150 were keyed with.
    sample_idx = [pos[int(s.split("_")[1])] for s in SAMPLE_IDS.read_text().split()]
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "k.csv"
        ak.write_key(p, key_for(sample_idx, rows, ref, det, tiers))
        got, want = ak.sha256_file(p), ak.read_sha256(TS / "eval-v03-key.sha256")
    assert got == want, f"rebuilt 470 key sha256 {got} differs from the pinned {want}"
    return rows, ref, det, tiers, pos


def draw_order() -> tuple[list[dict], list[str]]:
    """Reproduce the Phase 2 draw (v03_case_selection.main, section 3) and return it with the shuffled order of the
    DXA-only stratum's pool."""
    rules, tiers, codes = cs.load_rules(), cs.load_tiers(), cs.load_evidence_codes()
    reviewed = set()
    for name in cs.REVIEWED_ID_FILES:
        reviewed |= {l.strip() for l in (TS / name).read_text().splitlines() if l.strip()}
    reviewed |= {c["case_id"] for c in json.loads(cs.V0_250.read_text())["cases"]}
    tier1 = sorted(c for c, t in tiers.items() if t == 1)
    pool: dict[str, list[str]] = defaultdict(list)
    tags: dict[str, set[str]] = {}
    for i, row in cs.read_test_split():
        c = cs.parse_row(row)
        cid = f"ddxplus_{i}"
        if cid in reviewed:
            continue
        tier = tiers[c["truth"]]
        ns = cs.namespace(c["evidences"], c["age"], c["sex"], c["truth"], tier, codes)
        s = cs.select(rules, tiers, ns, [(cond, True) for cond in tier1 if cond != c["truth"] and c["dxa"].get(cond, 0.0) >= 10.0])
        if s["class"] == cs.SERIOUS and s["reason"] == "dxa-only":
            pool["serious_dxa_only"].append(cid)
        elif s["class"] == cs.SERIOUS and (s["reason"].startswith("P") or s["upgraded"]):
            pool["serious_upgrade_or_flag"].append(cid)
        elif s["class"] == cs.SERIOUS:
            pool["serious_tier1"].append(cid)
        elif s["class"] == cs.BENIGN:
            pool["benign"].append(cid)
        else:
            pool["excluded"].append(cid)
        t = cs.class_tags(s)
        if t:
            tags[cid] = t
    rng = random.Random(cs.PHASE2_SEED)
    draw, drawn, dxa_order = [], set(), []
    for stratum, n in cs.PHASE2_STRATA.items():
        ids = sorted(pool[stratum])
        rng.shuffle(ids)
        if stratum == DXA_STRATUM:
            dxa_order = list(ids)
        picked: list[str] = []
        for rule_id, m in cs.RULE_MIN.items():
            have = sum(1 for cid in picked if rule_id in tags.get(cid, ()))
            for cid in ids:
                if have >= m:
                    break
                if cid not in drawn and rule_id in tags.get(cid, ()):
                    picked.append(cid)
                    drawn.add(cid)
                    have += 1
        for cid in ids:
            if len(picked) >= n:
                break
            if cid not in drawn:
                picked.append(cid)
                drawn.add(cid)
        draw += [{"stratum": stratum, "case_id": cid, "rules": "|".join(sorted(tags.get(cid, ())))} for cid in picked]
    return draw, dxa_order


def main() -> None:
    committed = list(csv.DictReader(open(CANDIDATES, newline="", encoding="utf-8")))
    draw, dxa_order = draw_order()
    assert [(d["stratum"], d["case_id"], d["rules"]) for d in draw] == \
           [(d["stratum"], d["case_id"], d["rules"]) for d in committed], "reproduced draw differs from phase2_candidates.csv"
    print(f"draw reproduced: {len(draw)} cases; DXA-only pool {len(dxa_order)}")

    rows, ref, det, tiers, pos = detector()
    rules, codes = cs.load_rules(), cs.load_evidence_codes()
    sel_tiers = cs.load_tiers()
    conditions = ak2.load_conditions(COND_JSON)
    conditions_meta = json.loads(COND_JSON.read_text())

    def keyed(cid: str) -> tuple[list[dict], ak.CaseKeyV03, dict]:
        j = pos[int(cid.split("_")[1])]
        kr = key_for([j], rows, ref, det, tiers)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "k.csv"
            ak.write_key(p, kr)
            k = ak.load_key(p, None)[cid]
        r = rows[j]
        targets = [(t.condition, t.in_r10) for t in k.considered.values() if t.source == "dxa" and t.in_r5]
        ns = cs.namespace(r["evidences"], r["age"], r["sex"], r["path"], sel_tiers[r["path"]], codes)
        return kr, k, cs.select(rules, sel_tiers, ns, targets)

    final, key_rows, sel = [], [], {}
    all_ids = {d["case_id"] for d in draw}
    dxa_kept, replaced, dropped_keys = [], [], []
    for d in draw:
        kr, k, s = keyed(d["case_id"])
        if d["stratum"] == DXA_STRATUM:
            if s["class"] == cs.SERIOUS and s["reason"] == "dxa-only":
                dxa_kept.append(d["case_id"])
            else:
                replaced.append((d["case_id"], s["class"], s["reason"]))
                dropped_keys += kr
                sel[d["case_id"]] = s
                continue
        final.append({**d, "replaces": ""})
        key_rows += kr
        sel[d["case_id"]] = s
    need = cs.PHASE2_STRATA[DXA_STRATUM] - len(dxa_kept)
    fills, walked = [], 0
    for cid in dxa_order:
        if need == 0:
            break
        if cid in all_ids:
            continue
        walked += 1
        kr, k, s = keyed(cid)
        if s["class"] == cs.SERIOUS and s["reason"] == "dxa-only":
            fills.append(cid)
            key_rows += kr
            sel[cid] = s
            need -= 1
    assert need == 0, f"DXA-only stratum short by {need}"
    # Replacements take the replaced cases' places in draw order, so the stratum stays contiguous.
    out, fi = [], iter(fills)
    for d in draw:
        if d["stratum"] == DXA_STRATUM and d["case_id"] not in dxa_kept:
            out.append({"stratum": DXA_STRATUM, "case_id": next(fi), "rules": "", "replaces": d["case_id"]})
        else:
            out.append({**d, "replaces": ""})
    ids = [d["case_id"] for d in out]
    assert len(ids) == len(set(ids)) == sum(cs.PHASE2_STRATA.values())
    # The replaced cases stay in the case file and the key after the 250 (stratum DROPPED), so the models answer
    # them too and a reference built on the draw as first written can still be matched.
    out += [{"stratum": DROPPED, "case_id": a, "rules": "", "replaces": ""} for a, _, _ in replaced]
    key_rows += dropped_keys
    all_run = [d["case_id"] for d in out]

    by_id = {}
    for r in key_rows:
        by_id.setdefault(r["case_id"], []).append(r)
    key_rows = [r for cid in all_run for r in by_id[cid]]
    key_path = TS / f"{STEM}.key.csv"
    ak.write_key(key_path, key_rows)
    ak.write_sha256(TS / f"{STEM}.key.sha256", key_path)
    keys = ak.load_key(key_path, TS / f"{STEM}.key.sha256")

    fallback, icd10 = wd.load_fallback(), wd.load_icd10()
    cases, strata_rows = [], []
    for d in out:
        cid = d["case_id"]
        r = rows[pos[int(cid.split("_")[1])]]
        c = build_case(r, conditions_meta, conditions)
        assert c["case_id"] == cid, (c["case_id"], cid)
        wdx = wd.design_for(keys[cid], r["dxa"], c["initial_evidence"], tiers, fallback, icd10)
        cases.append(c | {"working_diagnosis": wdx.working_diagnosis, "working_diagnosis_name": wdx.rendering(),
                          "phase2_stratum": d["stratum"], "phase2_rules": d["rules"]})
        s = sel[cid]
        strata_rows.append({"case_id": cid, "stratum": d["stratum"], "rules": d["rules"], "replaces": d["replaces"],
                            "truth": s.get("truth", r["path"]), "class": s["class"], "reason": s["reason"],
                            "targets": "|".join(s["targets"]), "dangers": "|".join(s["dangers"]),
                            "truth_credit": s["truth_credit"], "a4": s["a4"], "working_diagnosis": wdx.working_diagnosis})
    ids_text = "".join(f"{c}\n" for c in ids)
    (TS / f"{STEM}.case_ids.txt").write_text(ids_text)
    cs.write_csv(TS / f"{STEM}.strata.csv", strata_rows)
    moved = Counter((d["stratum"], sel[d["case_id"]]["class"]) for d in out)
    (TS / f"{STEM}.run_ids.txt").write_text("".join(f"{c}\n" for c in all_run))
    meta = {"test_set_name": STEM, "source_file": str(TEST_CSV.relative_to(ROOT)),
            "selection": "docs/v0.3-case-selection-rules.md section 7: stratified draw, seed 20261003, never-reviewed adults",
            "seed": cs.PHASE2_SEED, "cases": len(ids), "cases_in_file": len(all_run), "case_ids_sha256": hashlib.sha256(ids_text.encode()).hexdigest(),
            "dxa_only_replaced": [{"drawn": a, "class_with_key": b, "reason": c} for a, b, c in replaced],
            "dxa_only_fills": fills, "dxa_only_candidates_walked": walked,
            "stratum_x_class": {f"{a}|{b}": v for (a, b), v in sorted(moved.items())},
            "answer_key": f"{STEM}.key.csv (evaluator/answer_key_v03.py; scripts/build_v03_key.py detector)",
            "builder": "scripts/build_v03_phase2_set.py"}
    (TS / f"{STEM}.json").write_text(json.dumps({"metadata": meta, "cases": cases}, indent=2) + "\n")
    print(f"{STEM}: {len(ids)} cases ({len(all_run)} in the file); DXA-only replaced {len(replaced)} (walked {walked} candidates)")
    print(f"  stratum x class: {meta['stratum_x_class']}")


if __name__ == "__main__":
    main()
