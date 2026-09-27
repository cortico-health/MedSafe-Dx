#!/usr/bin/env python3
"""
Build the case file, answer key and reference answers for the v0.3 full run (docs/v0.3-case-selection-rules.md
section 7.4).

We read the 900 ids that scripts/analysis/v03_case_selection.py --draw-full wrote (seed 20261005,
results/analysis/case_selection/full_candidates.csv), check that the draw reproduces, build the v0.3 key for each
case with the detector of scripts/build_v03_key.py (as scripts/build_v03_phase2_set.py does), and render each case
with its working diagnosis, so the models see what the Phase 2b models saw.

A drawn case whose class under the key differs from its stratum's class (a BENIGN case the key's DXA targets send
to X9 or X11) is replaced by the next case in its bucket's draw order (v03_case_selection.full_orders) that keeps the
class, so every case is scored. The replaced cases are listed in the metadata; the models do not answer them.

The naive Bayes reference answers (spec section 8) come from a model trained on the DDXPlus test split with the 900
held out (evaluator/v02_references.py), so no case is in its own training data.

Outputs, in data/test_sets/ (git-ignored; the ids and the key hash are force-added):
  eval-v03-full.json              the cases with `working_diagnosis`, `working_diagnosis_name`, `stratum`,
                                  `bucket`, `rules`, `twin`
  eval-v03-full.case_ids.txt      the ids in draw order
  eval-v03-full.key.csv, .sha256  the v0.3 key rows
  eval-v03-full.strata.csv        per case: stratum, bucket, rules, twin, class, reason, targets, dangers,
                                  what it replaces
  eval-v03-full.refs.json         the naive Bayes reference answers

Usage: python3 scripts/build_v03_full_set.py   (a few minutes: the full-split pass and the key)
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

from evaluator import answer_key_v02 as ak2  # noqa: E402
from evaluator import answer_key_v03 as ak  # noqa: E402
from evaluator import working_diagnosis as wd  # noqa: E402
from evaluator.condition_match import FlagMatcher  # noqa: E402
from evaluator.v02_references import NaiveBayes, train_counts  # noqa: E402
from build_v03_key import COND_JSON, TEST_CSV, TS, key_for, reader_answer  # noqa: E402
from build_v03_phase2_set import detector  # noqa: E402
from prep_v02_sample import build_case  # noqa: E402
import v03_case_selection as cs  # noqa: E402

STEM = "eval-v03-full"
CANDIDATES = ROOT / "results" / "analysis" / "case_selection" / "full_candidates.csv"
NB_CACHE = ROOT / "results" / "analysis" / "v03_key" / "nb_counts_holdout_full.json"
STRATUM_CLASS = {"serious_tier1": cs.SERIOUS, "serious_upgrade_or_flag": cs.SERIOUS, "benign": cs.BENIGN}


def reproduce():
    """The draw, its bucket orders and the tags, recomputed from the frozen rules and seed."""
    rules, tiers, codes = cs.load_rules(), cs.load_tiers(), cs.load_evidence_codes()
    truths: dict[str, str] = {}
    _, _, pool, tags = cs.build_pool(rules, tiers, codes, cs.reviewed_ids("full"), truths)
    scored = set(pool["serious_tier1"]) | set(pool["serious_upgrade_or_flag"]) | set(pool["benign"])
    twins = cs.twin_flags(scored)
    buckets = cs.full_orders(pool, truths, twins, tiers)
    draw = cs.draw_full(buckets, tags)
    for d in draw:
        d["twin"] = twins[d["case_id"]]
    return draw, buckets, tags, twins


def main() -> None:
    committed = list(csv.DictReader(open(CANDIDATES, newline="", encoding="utf-8")))
    draw, buckets, tags, twins = reproduce()
    got = [(d["stratum"], d["bucket"], d["case_id"], d["rules"], str(d["twin"])) for d in draw]
    want = [(d["stratum"], d["bucket"], d["case_id"], d["rules"], d["twin"]) for d in committed]
    assert got == want, "reproduced draw differs from full_candidates.csv"
    print(f"draw reproduced: {len(draw)} cases")

    rows, ref, det, tiers, pos = detector()
    rules, codes, sel_tiers = cs.load_rules(), cs.load_evidence_codes(), cs.load_tiers()

    def keyed(cid: str):
        j = pos[int(cid.split("_")[1])]
        kr = key_for([j], rows, ref, det, tiers)
        k = next(iter(_load(kr).values()))
        r = rows[j]
        targets = [(t.condition, t.in_r10) for t in k.considered.values() if t.source == "dxa" and t.in_r5]
        ns = cs.namespace(r["evidences"], r["age"], r["sex"], r["path"], sel_tiers[r["path"]], codes)
        return kr, cs.select(rules, sel_tiers, ns, targets)

    def _load(kr):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "k.csv"
            ak.write_key(p, kr)
            return ak.load_key(p, None)

    drawn_ids = {d["case_id"] for d in draw}
    used = set(drawn_ids)
    out, key_rows, sel, replaced = [], [], {}, []
    for d in draw:
        kr, s = keyed(d["case_id"])
        want_cls = STRATUM_CLASS[d["stratum"]]
        if s["class"] == want_cls:
            out.append({**d, "replaces": ""})
            key_rows += kr
            sel[d["case_id"]] = s
            continue
        # Walk the bucket's order for the next unused case that keeps the class.
        for cid in buckets[(d["stratum"], d["bucket"])]:
            if cid in used:
                continue
            used.add(cid)
            kr2, s2 = keyed(cid)
            if s2["class"] == want_cls:
                out.append({"stratum": d["stratum"], "bucket": d["bucket"], "case_id": cid,
                            "rules": "|".join(sorted(tags.get(cid, ()))), "twin": twins[cid], "replaces": d["case_id"]})
                key_rows += kr2
                sel[cid] = s2
                break
        else:
            raise SystemExit(f"bucket {d['stratum']}|{d['bucket']} ran out while replacing {d['case_id']}")
        replaced.append({"drawn": d["case_id"], "stratum": d["stratum"], "bucket": d["bucket"],
                         "class_with_key": s["class"], "reason": s["reason"], "replacement": out[-1]["case_id"]})
    ids = [d["case_id"] for d in out]
    assert len(ids) == len(set(ids)) == len(draw)

    key_path = TS / f"{STEM}.key.csv"
    ak.write_key(key_path, key_rows)
    ak.write_sha256(TS / f"{STEM}.key.sha256", key_path)
    keys = ak.load_key(key_path, TS / f"{STEM}.key.sha256")

    conditions = ak2.load_conditions(COND_JSON)
    conditions_meta = json.loads(COND_JSON.read_text())
    fallback, icd10 = wd.load_fallback(), wd.load_icd10()
    cases, strata_rows = [], []
    for d in out:
        cid = d["case_id"]
        r = rows[pos[int(cid.split("_")[1])]]
        c = build_case(r, conditions_meta, conditions)
        assert c["case_id"] == cid, (c["case_id"], cid)
        wdx = wd.design_for(keys[cid], r["dxa"], c["initial_evidence"], tiers, fallback, icd10)
        cases.append(c | {"working_diagnosis": wdx.working_diagnosis, "working_diagnosis_name": wdx.rendering(),
                          "stratum": d["stratum"], "bucket": d["bucket"], "rules": d["rules"], "twin": bool(d["twin"])})
        s = sel[cid]
        strata_rows.append({"case_id": cid, "stratum": d["stratum"], "bucket": d["bucket"], "rules": d["rules"],
                            "twin": d["twin"], "replaces": d["replaces"], "truth": r["path"], "class": s["class"],
                            "reason": s["reason"], "targets": "|".join(s["targets"]), "dangers": "|".join(s["dangers"]),
                            "truth_credit": s["truth_credit"], "a4": s["a4"], "working_diagnosis": wdx.working_diagnosis})

    # Naive Bayes reference answers, trained with the 900 held out.
    t1 = ak.tier1_conditions(tiers)
    nb = NaiveBayes(train_counts(set(ids), TEST_CSV, NB_CACHE))
    canonical = FlagMatcher().cmap.canonical
    nb_rows = []
    for cid in ids:
        r = rows[pos[int(cid.split("_")[1])]]
        post = nb.posterior(r["evidences"])
        nbp = {c: 100.0 * float(p) for c, p in zip(nb.conditions, post)}
        ranked = sorted(nbp.items(), key=lambda x: -x[1])
        ok = [(c, nbp.get(c, 0.0)) for c in t1 if nbp.get(c, 0.0) >= ak.R10]
        nb_rows.append(reader_answer(cid, ranked, ok, sum(nbp.get(c, 0.0) for c in t1), canonical))
    (TS / f"{STEM}.refs.json").write_text(json.dumps(
        {"metadata": {"set": STEM, "builder": "scripts/build_v03_full_set.py",
                      "rules": {"naive-bayes": "naive Bayes over base evidence codes (evaluator/v02_references.py), trained on "
                                "the DDXPlus test split with the 900 held out; YES when a tier-1 posterior is >= 10%; flags = "
                                "the top 5 such, strongest first"}},
         "references": {"naive-bayes": nb_rows}}, indent=1) + "\n")

    ids_text = "".join(f"{c}\n" for c in ids)
    (TS / f"{STEM}.case_ids.txt").write_text(ids_text)
    cs.write_csv(TS / f"{STEM}.strata.csv", strata_rows)
    meta = {"test_set_name": STEM, "source_file": str(TEST_CSV.relative_to(ROOT)),
            "selection": "docs/v0.3-case-selection-rules.md section 7.4: full-run stratified draw, seed "
                         f"{cs.FULL_SEED}, never-reviewed adults, cases without an exact public twin first",
            "seed": cs.FULL_SEED, "cases": len(ids), "case_ids_sha256": hashlib.sha256(ids_text.encode()).hexdigest(),
            "replaced": replaced, "twins": sum(bool(d["twin"]) for d in out),
            "stratum_x_class": {f"{a}|{b}": v for (a, b), v in sorted(Counter((d["stratum"], sel[d["case_id"]]["class"]) for d in out).items())},
            "answer_key": f"{STEM}.key.csv (evaluator/answer_key_v03.py; scripts/build_v03_key.py detector)",
            "builder": "scripts/build_v03_full_set.py"}
    (TS / f"{STEM}.json").write_text(json.dumps({"metadata": meta, "cases": cases}, indent=2) + "\n")
    print(f"{STEM}: {len(ids)} cases; replaced {len(replaced)}; twins {meta['twins']}")
    print(f"  stratum x class: {meta['stratum_x_class']}")
    for x in replaced:
        print(f"  replaced {x['drawn']} ({x['bucket']}: {x['class_with_key']} by {x['reason']}) with {x['replacement']}")


if __name__ == "__main__":
    main()
