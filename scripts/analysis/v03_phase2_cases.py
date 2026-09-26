#!/usr/bin/env python3
"""
Build the Phase 2 review files for the case-selection rules (docs/v0.3-case-selection-rules.md
section 7). No model is called.

We take the seeded draw in results/analysis/case_selection/phase2_candidates.csv (seed 20261003,
250 never-reviewed adults from the DDXPlus test split), build the v0.3b key for each drawn case so
the DXA-derived targets pass the interval red-herring rule (the full-split pass had no key), rerun
the frozen rules with those targets, and write:

1. results/phase2/cases_blind.json: what a reviewer sees, in the audit's format (results/audit/
   cases_blind.json): case_id, the prompt-v6 intake rendering (the same decoder and text a model
   sees), and the clinician's working diagnosis. Nothing about truth, stratum, rule or class.
2. <unblinded dir>/cases_unblinded.json: the key per case (stratum, rules that decided the class,
   truth, tier, class under the key, targets, dangers). It is written outside the repo by default
   (--unblinded-dir) so a reviewer working in the repo cannot open it; we copy it into
   results/phase2/ after the reviews.

The SERIOUS-by-DXA-only stratum is re-drawn as section 7 requires: we replay the draw with the same
seed and pool order, and skip a candidate whose target does not survive the red-herring rule, so the
20 reviewed cases are the first 20 survivors in the original draw order. The other strata are
checked to reproduce phase2_candidates.csv exactly, and each drawn case is re-classified with the
key; a case whose class moves (a BENIGN candidate that the key's R5-only target excludes under X9)
keeps its drawn stratum in the file and carries `class_key` for the validation to use.

Usage: python3 scripts/analysis/v03_phase2_cases.py [--unblinded-dir DIR]   (about 2 minutes)
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

import v03_case_selection as cs  # noqa: E402
from build_v03_key import Detector, learn_hallmarks  # noqa: E402
from build_v03b_key import CellTable  # noqa: E402
from evaluator import answer_key_v02 as ak2  # noqa: E402
from evaluator import answer_key_v03 as ak  # noqa: E402
from evaluator import answer_key_v03b as ak3b  # noqa: E402
from evaluator import working_diagnosis as wd  # noqa: E402
from inference.run_inference import format_case_for_prompt_v6  # noqa: E402
from prep_v02_sample import build_case, read_rows  # noqa: E402

CAND = ROOT / "results" / "analysis" / "case_selection" / "phase2_candidates.csv"
OUT = ROOT / "results" / "phase2"
COND_JSON = ROOT / "data" / "ddxplus_v0" / "release_conditions.json"
EVID_JSON = ROOT / "data" / "ddxplus_v0" / "release_evidences.json"
ADULT_MIN_AGE = 18


def reviewed_ids() -> set[str]:
    out = set()
    for name in cs.REVIEWED_ID_FILES:
        out |= {l.strip() for l in (ROOT / "data" / "test_sets" / name).read_text().splitlines() if l.strip()}
    out |= {c["case_id"] for c in json.loads(cs.V0_250.read_text())["cases"]}
    return out


def key_targets_for(r: dict, det: CellTable, tiers: dict) -> tuple[list[tuple[str, bool]], list[str], list[str]]:
    """The v0.3b key's DXA-derived tier-1 targets for one never-reviewed adult: [(condition, in_r10)] for the
    supported ones, plus the R10 and R5 lists (truth included when tier 1)."""
    rows = ak3b.case_key_rows(f"ddxplus_{r['i']}", r["path"], r["dxa"], tiers, det.lookup_for(r, True),
                              ak2.red_flags(r["evidences"]))
    dxa = [(k["condition"], bool(k["in_r10"])) for k in rows if k.get("condition") and k["target_source"] == "dxa" and k["in_r5"]]
    r10 = [k["condition"] for k in rows if k.get("condition") and k["in_r10"]]
    r5 = [k["condition"] for k in rows if k.get("condition") and k["in_r5"]]
    return dxa, r10, r5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unblinded-dir", type=Path, default=None,
                    help="where cases_unblinded.json goes (default: results/phase2/unblinded, which reviewers must not open)")
    args = ap.parse_args()
    unblinded_dir = args.unblinded_dir or (OUT / "unblinded")
    OUT.mkdir(parents=True, exist_ok=True)
    unblinded_dir.mkdir(parents=True, exist_ok=True)

    rules, tiers, codes = cs.load_rules(), cs.load_tiers(), cs.load_evidence_codes()
    t1 = ak.tier1_conditions(tiers)
    conditions = ak2.load_conditions(COND_JSON)
    conditions_meta = json.loads(COND_JSON.read_text())
    evid = json.loads(EVID_JSON.read_text())
    fallback, icd10 = wd.load_fallback(), wd.load_icd10()

    def is_antecedent(tok: str) -> bool:
        return evid.get(tok.partition("_@_")[0], {}).get("is_antecedent") in (True, "True")

    # The reference adults and the detector, as scripts/build_v03_key.py builds them (main sample held out).
    all_rows = read_rows(cs.TEST_SPLIT)
    rows = [r for r in all_rows if r["age"] >= ADULT_MIN_AGE]
    for r in rows:
        r["tokens"] = frozenset(str(e) for e in r["evidences"])
        r["dxa"] = {n: 100.0 * float(p) for n, p in r["differential"]}
    main_ids = {int(s.split("_")[1]) for s in cs.MAIN_IDS.read_text().split()}
    ref = np.array([r["i"] not in main_ids for r in rows])
    det = CellTable(Detector(rows, ref, t1, learn_hallmarks(rows, ref, t1, is_antecedent)))
    by_i = {r["i"]: r for r in rows}
    print(f"adults {len(rows)}, reference {int(ref.sum())}")

    # The never-reviewed pools, exactly as the case-selection script builds them (candidates at p >= 10%).
    reviewed = reviewed_ids()
    pool: dict[str, list[str]] = defaultdict(list)
    tags: dict[str, set[str]] = {}
    tier1 = sorted(c for c, t in tiers.items() if t == 1)
    for r in rows:
        cid = f"ddxplus_{r['i']}"
        if cid in reviewed:
            continue
        tier = tiers[r["path"]]
        ns = cs.namespace(r["evidences"], r["age"], r["sex"], r["path"], tier, codes)
        cands = [(cond, True) for cond in tier1 if cond != r["path"] and r["dxa"].get(cond, 0.0) >= 10.0]
        s = cs.select(rules, tiers, ns, cands)
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
    print("never-reviewed pool", {k: len(v) for k, v in pool.items()})

    def with_key(cid: str) -> dict:
        r = by_i[int(cid.split("_")[1])]
        dxa, r10, r5 = key_targets_for(r, det, tiers)
        tier = tiers[r["path"]]
        ns = cs.namespace(r["evidences"], r["age"], r["sex"], r["path"], tier, codes)
        s = cs.select(rules, tiers, ns, dxa)
        return {"case_id": cid, "truth": r["path"], "tier": tier, "age": r["age"], "sex": r["sex"], **s,
                "key_r10_targets": r10, "key_r5_targets": r5}

    # Replay the draw: same seed, same pool order; the DXA-only stratum skips candidates whose target the
    # red-herring rule removes (the class under the key is no longer SERIOUS by dxa-only).
    rng = random.Random(cs.PHASE2_SEED)
    draw, drawn, skipped = [], set(), []
    keyed: dict[str, dict] = {}
    for stratum, n in cs.PHASE2_STRATA.items():
        ids = sorted(pool[stratum])
        rng.shuffle(ids)
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
            if cid in drawn:
                continue
            if stratum == "serious_dxa_only":
                k = with_key(cid)
                keyed[cid] = k
                if not (k["class"] == cs.SERIOUS and k["reason"] == "dxa-only"):
                    skipped.append({"case_id": cid, "truth": k["truth"], "class_key": k["class"], "reason_key": k["reason"],
                                    "dropped": k["dropped"], "key_r5_targets": k["key_r5_targets"]})
                    continue
            picked.append(cid)
            drawn.add(cid)
        draw += [{"stratum": stratum, "case_id": cid, "rules": "|".join(sorted(tags.get(cid, ())))} for cid in picked]
    assert len(draw) == sum(cs.PHASE2_STRATA.values()), len(draw)

    # The other four strata must reproduce the committed candidate file.
    cand = list(csv.DictReader(open(CAND, newline="", encoding="utf-8")))
    for stratum in cs.PHASE2_STRATA:
        if stratum == "serious_dxa_only":
            continue
        a = [d["case_id"] for d in draw if d["stratum"] == stratum]
        b = [d["case_id"] for d in cand if d["stratum"] == stratum]
        assert a == b, f"{stratum}: replayed draw differs from {CAND}"
    orig_dxa = [d["case_id"] for d in cand if d["stratum"] == "serious_dxa_only"]
    new_dxa = [d["case_id"] for d in draw if d["stratum"] == "serious_dxa_only"]
    print(f"dxa-only stratum: {len(set(orig_dxa) & set(new_dxa))} of the {len(orig_dxa)} original candidates survive; "
          f"{len(skipped)} skipped in the replay")

    # Per-case key and blind rendering.
    blind, unblinded = [], []
    moved = Counter()
    for idx, d in enumerate(draw):
        cid = d["case_id"]
        k = keyed.get(cid) or with_key(cid)
        r = by_i[int(cid.split("_")[1])]
        hyp, hp, fb = wd.choose(r["dxa"], tiers, r["initial_evidence"], fallback)
        case = build_case(r, conditions_meta, conditions)
        case["working_diagnosis_name"] = f"{wd.DISPLAY_NAMES[hyp]} ({icd10[hyp]})"
        intake = format_case_for_prompt_v6(case)
        blind.append({"case_id": cid, "intake": intake, "clinician_working_diagnosis": hyp, "working_diagnosis_icd10": icd10[hyp]})
        class_key = k["class"]
        drawn_class = {"benign": cs.BENIGN, "excluded": cs.EXCLUDED}.get(d["stratum"], cs.SERIOUS)
        rule_tags = d["rules"]
        if class_key != drawn_class:
            moved[f"{d['stratum']}>{class_key}"] += 1
            rule_tags = "|".join(sorted(cs.class_tags(k)))  # the rule that decided the class under the key
        unblinded.append({
            "case_id": cid, "index": idx, "stratum": d["stratum"], "rules": rule_tags,
            "true_condition": k["truth"], "truth_tier": str(k["tier"]), "age": k["age"], "sex": k["sex"],
            "class_key": class_key, "bucket": k["bucket"], "reason": k["reason"], "fired": k["fired"],
            "excluded_by": k["excluded_by"], "targets": k["targets"], "dangers": k["dangers"], "dropped": k["dropped"],
            "truth_credit": k["truth_credit"], "a4": k["a4"], "upgraded": k["upgraded"],
            "r10_targets": "|".join(k["key_r10_targets"]), "r5_targets": "|".join(k["key_r5_targets"]),
            "working_diagnosis": hyp, "working_diagnosis_icd10": icd10[hyp], "working_diagnosis_p": round(hp, 2),
            "working_diagnosis_fallback": fb, "working_diagnosis_is_truth": hyp == k["truth"],
        })
    (OUT / "cases_blind.json").write_text(json.dumps(blind, indent=1) + "\n")
    (unblinded_dir / "cases_unblinded.json").write_text(json.dumps(unblinded, indent=1) + "\n")
    (unblinded_dir / "dxa_only_redraw.json").write_text(json.dumps(
        {"original_candidates": orig_dxa, "reviewed": new_dxa, "skipped_in_replay": skipped}, indent=1) + "\n")
    print(f"wrote {len(blind)} blind cases to {OUT / 'cases_blind.json'}; key in {unblinded_dir}")
    print("by stratum", dict(Counter(u["stratum"] for u in unblinded)))
    print("class under the key by stratum", {s: dict(Counter(u["class_key"] for u in unblinded if u["stratum"] == s))
                                            for s in cs.PHASE2_STRATA})
    print("moved", dict(moved))
    print("working diagnosis is the truth:", sum(u["working_diagnosis_is_truth"] for u in unblinded))


if __name__ == "__main__":
    main()
