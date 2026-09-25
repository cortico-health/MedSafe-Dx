#!/usr/bin/env python3
"""
Build the MedSafe-Dx v0.3 answer key, pool subsets and reference predictions
(spec/v0.3-scoring.md sections 1, 4 and 8, with the review fixes 2, 5, 6 and 3 of
docs/v0.3-adversarial-review.md). Reads DDXPlus data only; no inference spend.

1. Red-herring detector M1' for all 21 tier-1 conditions, from DXA p >= 5%
   (scripts/analysis/dxa_red_herrings.py, same method). The reference adults are
   the DDXPlus test split, age >= 18, with the 470-case main sample held out. For
   each condition we learn its five hallmark symptoms: the symptom tokens
   (evidence with its value, antecedents excluded) with the largest likelihood
   ratio among tokens present in >= 20% of its reference patients. A pair's class
   is (condition, DXA band, hallmark count 0-5); a reference patient's own row is
   left out of its class (leave-one-out). evaluator/answer_key_v03.py applies the
   rule, with the 1% floor.
2. The key for the 470 sample: data/test_sets/eval-v03-key.csv, hash-pinned in
   eval-v03-key.sha256.
3. Pool subsets from the reference adults (so the 470 are excluded), up to 10 per
   tier-1 condition, drawn with numpy default_rng(POOL_SEED) per pool, conditions in
   sorted order, candidates in file order; a case drawn for one condition is not
   drawn again:
   a. atypical serious: the truth is tier 1 and DXA's top diagnosis is tier 3 (on a
      tie at the top, the most serious tied condition decides);
   b. high risk: a tier-1 condition that is not the truth has DXA p >= 10%, passes
      the red-herring rule in a class of 30 or more, and the truth is not tier 1.
   Each pool gets a case file, an ID list and a key, like the main sample.
4. Reference predictions in the v0.3 output format, for the 470 and each pool:
   always-yes (flags: the 5 tier-1 conditions with the most R10 targets in the 470
   key; ties by R5 targets, then name), always-no, the DXA reader and naive Bayes.

Outputs (data/test_sets/ is git-ignored; we force-add the ID lists and hashes):
  data/test_sets/eval-v03-key.csv, eval-v03-key.sha256
  data/test_sets/eval-v03-pool-{atypical,high-risk}.{json,case_ids.txt,key.csv,key.sha256}
  data/test_sets/eval-v03-{adult,pool-atypical,pool-high-risk}.refs.json
  results/analysis/v03_key/: summary.json, hallmarks.csv, cells.csv, build.log tables

Usage: python3 scripts/build_v03_key.py
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
from evaluator.v03_anchors import dxa_reader_conditions  # noqa: E402
from evaluator.condition_match import FlagMatcher, multiplicity_summary  # noqa: E402
from evaluator.v02_references import NaiveBayes, train_counts  # noqa: E402
from prep_v02_sample import build_case, read_rows  # noqa: E402

TEST_CSV = ROOT / "data/ddxplus_v0/release_test_patients"
COND_JSON = ROOT / "data/ddxplus_v0/release_conditions.json"
EVID_JSON = ROOT / "data/ddxplus_v0/release_evidences.json"
SAMPLE_IDS = ROOT / "data/test_sets/eval-v02-adult.case_ids.txt"
TS = ROOT / "data/test_sets"
OUT = ROOT / "results/analysis/v03_key"
REVIEW_KEY = ROOT / "results/analysis/v03_review/key470_targets.csv"

ADULT_MIN_AGE = 18
N_HALLMARKS = 5
HALLMARK_MIN_PREV = 0.20
POOL_SEED = 20260925
POOL_PER_CONDITION = 10
N_ALWAYS_YES = 5
READER_T = ak.R10  # the DXA and naive-Bayes readers say YES at a tier-1 p >= 10%
POOLS = ("atypical", "high-risk")


# ---------------------------------------------------------------- detector


def learn_hallmarks(rows, ref, conds, is_antecedent) -> dict[str, list[str]]:
    """Per condition: top N_HALLMARKS non-antecedent tokens by likelihood ratio among tokens in >= 20% of its patients."""
    n_ref = int(ref.sum())
    tok_all = Counter()
    tok_by = defaultdict(Counter)
    n_by = Counter()
    for r, is_ref in zip(rows, ref):
        if not is_ref:
            continue
        n_by[r["path"]] += 1
        tok_by[r["path"]].update(r["tokens"])
        tok_all.update(r["tokens"])
    tokens = sorted(tok_all)
    out = {}
    for c in conds:
        n1, n0 = n_by[c], n_ref - n_by[c]
        cand = []
        for t in tokens:
            if is_antecedent(t):
                continue
            k1 = tok_by[c][t]
            p1 = k1 / n1 if n1 else 0.0
            if p1 < HALLMARK_MIN_PREV:
                continue
            p0 = (tok_all[t] - k1) / n0
            cand.append((t, (p1 + 1e-6) / (p0 + 1e-6)))
        cand.sort(key=lambda x: -x[1])  # stable: ties keep token order
        out[c] = [t for t, _ in cand[:N_HALLMARKS]]
    return out


class Detector:
    """Class counts per (condition, band, hallmark count) over the reference adults, with leave-one-out lookup."""

    def __init__(self, rows, ref, conds, hallmarks):
        self.conds, self.hallmarks = conds, hallmarks
        self.hset = {c: set(hallmarks[c]) for c in conds}
        self.n = Counter()
        self.k = Counter()
        for r, is_ref in zip(rows, ref):
            if not is_ref:
                continue
            for c in conds:
                cell = self.cell(r, c)
                self.n[cell] += 1
                self.k[cell] += r["path"] == c

    def present(self, r, c) -> tuple[str, ...]:
        return tuple(t for t in self.hallmarks[c] if t in r["tokens"])

    def cell(self, r, c) -> tuple[str, int, int]:
        return c, ak.band_of(r["dxa"].get(c, 0.0)), len(self.present(r, c))

    def lookup_for(self, r, is_ref: bool):
        """ClassLookup for one patient; a reference patient's own row is left out."""
        def f(c: str, p: float) -> ak.ClassStats:
            h = self.present(r, c)
            cell = (c, ak.band_of(p), len(h))
            n, k = self.n[cell], self.k[cell]
            if is_ref:
                n -= 1
                k -= r["path"] == c
            return ak.ClassStats(n=n, k=k, hallmark_tokens=h)
        return f

    def cell_rows(self) -> list[dict]:
        out = []
        for (c, b, h), n in sorted(self.n.items()):
            k = self.k[(c, b, h)]
            out.append({"condition": c, "band": ak.band_label(b), "hallmarks": h, "n": n, "k": k,
                        "rate": round(k / n, 6), "undetermined": n < ak.MIN_N})
        return out


# ---------------------------------------------------------------- helpers


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def top_tier(dxa: dict[str, float], tiers) -> int:
    """Tier of DXA's top diagnosis; on a tie at the top, the most serious tied condition."""
    top = max(dxa.values())
    return min(tiers.get(c, 3) for c, p in dxa.items() if abs(p - top) < 1e-9)


def draw(cands: dict[str, list[int]], seed: int, per: int) -> dict[str, list[int]]:
    """Up to `per` row indices per condition (sorted order), without repeats across conditions."""
    rng = np.random.default_rng(seed)
    taken: set[int] = set()
    out = {}
    for c in sorted(cands):
        pool = [i for i in cands[c] if i not in taken]
        k = min(per, len(pool))
        idx = sorted(rng.choice(len(pool), size=k, replace=False)) if k else []
        out[c] = [pool[j] for j in idx]
        taken.update(out[c])
    return out


def key_for(rows_idx, rows, ref, det, tiers) -> list[dict]:
    out = []
    for i in rows_idx:
        r = rows[i]
        out += ak.case_key_rows(f"ddxplus_{r['i']}", r["path"], r["dxa"], tiers, det.lookup_for(r, bool(ref[i])),
                                ak2.red_flags(r["evidences"]))
    return out


def case_level(key_rows: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in key_rows:
        if r["case_id"] not in seen:
            seen.add(r["case_id"])
            out.append(r)
    return out


def key_counts(key_rows: list[dict]) -> dict:
    cases = case_level(key_rows)
    return {"cases": len(cases), "r10": sum(c["has_r10"] for c in cases), "r5": sum(c["has_r5"] for c in cases),
            "clearly_low_risk": sum(c["clearly_low_risk"] for c in cases),
            "intermediate": sum(c["intermediate"] for c in cases),
            "r10_targets": sum(1 for r in key_rows if r["in_r10"] is True),
            "r5_targets": sum(1 for r in key_rows if r["in_r5"] is True),
            "undetermined_r10_targets": sum(1 for r in key_rows if r["in_r10"] is True and r["undetermined"] is True)}


# ---------------------------------------------------------------- references


def reader_answer(case_id, ranked: list[tuple[str, float]], tier1_ok: list[tuple[str, float]], p_serious: float,
                  canonical) -> dict:
    """v0.3 answer from a ranked differential and the tier-1 conditions the reader raises."""
    flags = [c for c, _ in sorted(tier1_ok, key=lambda x: -x[1])[:5]]
    return {"case_id": case_id, "serious_concern": "YES" if flags else "NO",
            "flags": [canonical[c] for c in flags],
            "differential": [{"code": canonical[c], "p": round(p, 4)} for c, p in ranked[:5]],
            "p_serious": round(p_serious, 4)}


def references(rows_idx, rows, ref, det, tiers, nb, always_yes_codes, canonical) -> dict[str, list[dict]]:
    t1 = ak.tier1_conditions(tiers)
    out = {"always-yes": [], "always-no": [], "dxa": [], "naive-bayes": []}
    for i in rows_idx:
        r = rows[i]
        cid = f"ddxplus_{r['i']}"
        out["always-yes"].append({"case_id": cid, "serious_concern": "YES", "flags": list(always_yes_codes)})
        out["always-no"].append({"case_id": cid, "serious_concern": "NO", "flags": []})
        lookup = det.lookup_for(r, bool(ref[i]))
        ranked = sorted(r["dxa"].items(), key=lambda x: -x[1])

        def red_herring(c, p, lookup=lookup):
            cs = lookup(c, p)
            return ak.rh_status(p, cs.n, cs.k) == ak.RED_HERRING

        ok = [(c, r["dxa"][c]) for c in dxa_reader_conditions(r["dxa"], tiers, red_herring, READER_T)]
        out["dxa"].append(reader_answer(cid, ranked, ok, sum(r["dxa"].get(c, 0.0) for c in t1), canonical))
        post = nb.posterior(r["evidences"])
        nbp = {c: 100.0 * float(p) for c, p in zip(nb.conditions, post)}
        ranked_nb = sorted(nbp.items(), key=lambda x: -x[1])
        ok_nb = [(c, nbp.get(c, 0.0)) for c in t1 if nbp.get(c, 0.0) >= READER_T]
        out["naive-bayes"].append(reader_answer(cid, ranked_nb, ok_nb, sum(nbp.get(c, 0.0) for c in t1), canonical))
    return out


REF_RULES = {
    "always-yes": "serious_concern YES for every case; flags = the {n} tier-1 conditions with the most R10 targets in the 470 key "
                  "(ties: more R5 targets, then name), as canonical codes: {codes}",
    "always-no": "serious_concern NO for every case; no flags",
    "dxa": "YES when a tier-1 condition has DXA p >= 10% and is not a red herring (undetermined kept); flags = the top 5 such by p; "
           "differential = DXA's top 5; p_serious = DXA mass on tier-1 conditions. At the key's own threshold, not a ceiling",
    "naive-bayes": "naive Bayes over base evidence codes (evaluator/v02_references.py), trained on the DDXPlus test split with the "
                   "470 sample and both pools held out; YES when a tier-1 posterior is >= 10%; flags = the top 5 such; differential = "
                   "its top 5; p_serious = posterior mass on tier-1 conditions. Dataset-knowledge ceiling, not a clinical target",
}


# ---------------------------------------------------------------- main


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TS.mkdir(parents=True, exist_ok=True)
    tiers = ak.load_tiers()
    t1 = ak.tier1_conditions(tiers)
    conditions = ak2.load_conditions(COND_JSON)
    conditions_meta = json.loads(COND_JSON.read_text())
    evid = json.loads(EVID_JSON.read_text())

    def is_antecedent(tok: str) -> bool:
        return evid.get(tok.partition("_@_")[0], {}).get("is_antecedent") in (True, "True")

    all_rows = read_rows(TEST_CSV)
    rows = [r for r in all_rows if r["age"] >= ADULT_MIN_AGE]
    for r in rows:
        r["tokens"] = frozenset(str(e) for e in r["evidences"])
        r["dxa"] = {n: 100.0 * float(p) for n, p in r["differential"]}
    sample_ids = SAMPLE_IDS.read_text().split()
    sample_rows = {int(s.split("_")[1]) for s in sample_ids}
    ref = np.array([r["i"] not in sample_rows for r in rows])
    pos = {r["i"]: j for j, r in enumerate(rows)}
    sample_idx = [pos[int(s.split("_")[1])] for s in sample_ids]
    print(f"adults {len(rows)}, reference {int(ref.sum())}, sample {len(sample_idx)}; tier-1 conditions {len(t1)}")

    # 1. detector
    hallmarks = learn_hallmarks(rows, ref, t1, is_antecedent)
    det = Detector(rows, ref, t1, hallmarks)
    write_csv(OUT / "hallmarks.csv", [{"condition": c, "rank": j + 1, "token": t} for c in t1 for j, t in enumerate(hallmarks[c])])
    write_csv(OUT / "cells.csv", det.cell_rows())
    checks = compare_committed(hallmarks, t1)

    # 2. the 470 key
    key_rows = key_for(sample_idx, rows, ref, det, tiers)
    sha = ak.write_key(TS / "eval-v03-key.csv", key_rows)
    ak.write_sha256(TS / "eval-v03-key.sha256", TS / "eval-v03-key.csv")
    counts = key_counts(key_rows)
    print(f"key470: {counts}; sha256 {sha}")
    rh_table = red_herring_table(key_rows, t1)
    floor = floor_effect(key_rows)
    checks["review_key_disagreements"] = compare_review_key(key_rows)

    # 3. pools
    cands = {"atypical": defaultdict(list), "high-risk": defaultdict(list)}
    for j, r in enumerate(rows):
        if not ref[j]:
            continue
        truth_tier = tiers.get(r["path"], 3)
        if truth_tier == 1:
            if top_tier(r["dxa"], tiers) == 3:
                cands["atypical"][r["path"]].append(j)
            continue
        lookup = det.lookup_for(r, True)
        for c in t1:
            p = r["dxa"].get(c, 0.0)
            if p >= ak.R10:
                cs = lookup(c, p)
                if ak.rh_status(p, cs.n, cs.k) == ak.KEPT:
                    cands["high-risk"][c].append(j)
    drawn = {name: draw({c: cands[name].get(c, []) for c in t1}, POOL_SEED, POOL_PER_CONDITION) for name in POOLS}
    pool_idx = {name: sorted({j for lst in drawn[name].values() for j in lst}) for name in POOLS}
    avail = pool_availability(cands, drawn, rows, t1)

    # 4. references
    always_yes = always_yes_list(key_rows, t1)
    cmap = FlagMatcher().cmap
    holdout = set(sample_ids) | {f"ddxplus_{rows[j]['i']}" for name in POOLS for j in pool_idx[name]}
    nb = NaiveBayes(train_counts(holdout, TEST_CSV, OUT / "nb_counts_holdout_v03.json"))
    ay_codes = [cmap.canonical[c] for c in always_yes]
    rules = dict(REF_RULES)
    rules["always-yes"] = rules["always-yes"].format(n=N_ALWAYS_YES, codes=", ".join(f"{c} ({k})" for c, k in zip(ay_codes, always_yes)))

    sets = {"adult": sample_idx, **{f"pool-{n}": pool_idx[n] for n in POOLS}}
    pool_counts = {}
    for name, idx in sets.items():
        refs = references(idx, rows, ref, det, tiers, nb, ay_codes, cmap.canonical)
        (TS / f"eval-v03-{name}.refs.json").write_text(json.dumps(
            {"metadata": {"set": name, "rules": rules, "builder": "scripts/build_v03_key.py"}, "references": refs}, indent=1) + "\n")
        if name == "adult":
            continue
        kr = key_for(idx, rows, ref, det, tiers)
        stem = f"eval-v03-{name}"
        ak.write_key(TS / f"{stem}.key.csv", kr)
        ak.write_sha256(TS / f"{stem}.key.sha256", TS / f"{stem}.key.csv")
        cases = [build_case(rows[j], conditions_meta, conditions) for j in idx]
        ids_text = "".join(f"{c['case_id']}\n" for c in cases)
        (TS / f"{stem}.case_ids.txt").write_text(ids_text)
        meta = {"test_set_name": stem, "source_file": str(TEST_CSV.relative_to(ROOT)), "filter": f"age >= {ADULT_MIN_AGE}, main sample excluded",
                "selection": ("truth tier 1 and DXA top diagnosis tier 3" if name == "pool-atypical" else
                              "a non-truth tier-1 condition at DXA p >= 10% that passes the red-herring rule in a class of 30 or more; truth not tier 1"),
                "sampling": f"up to {POOL_PER_CONDITION} per tier-1 condition, numpy default_rng({POOL_SEED}), conditions sorted, no repeats",
                "seed": POOL_SEED, "cases": len(cases), "case_ids_sha256": hashlib.sha256(ids_text.encode()).hexdigest(),
                "per_condition": {c: [f"ddxplus_{rows[j]['i']}" for j in lst] for c, lst in drawn[name.removeprefix("pool-")].items()},
                "answer_key": f"{stem}.key.csv (evaluator/answer_key_v03.py)"}
        (TS / f"{stem}.json").write_text(json.dumps({"metadata": meta, "cases": cases}, indent=2) + "\n")
        pool_counts[name] = key_counts(kr)
        print(f"{stem}: {len(cases)} cases; key {pool_counts[name]}")

    mult = multiplicity_summary(FlagMatcher(), tiers)
    summary = {"key470": counts, "key470_sha256": sha, "red_herrings_per_condition": rh_table, "floor_effect": floor,
               "always_yes": dict(zip(always_yes, ay_codes)), "pool_availability": avail, "pool_key_counts": pool_counts,
               "flag_multiplicity": mult, "checks": checks, "hallmarks": hallmarks}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    report(summary)


# ---------------------------------------------------------------- reporting


def red_herring_table(key_rows, t1) -> list[dict]:
    out = []
    for c in t1:
        d = [r for r in key_rows if r["condition"] == c and r["source"] == "dxa"]
        row = {"condition": c}
        for lab, lo, hi in (("5-10", 5.0, 10.0), ("10+", 10.0, 101.0)):
            b = [r for r in d if lo <= r["dxa_p"] < hi]
            row[f"pairs_{lab}"] = len(b)
            row[f"removed_{lab}"] = sum(r["status"] == ak.RED_HERRING for r in b)
            row[f"undetermined_{lab}"] = sum(r["status"] == ak.UNDETERMINED for r in b)
            row[f"kept_{lab}"] = sum(r["status"] == ak.KEPT for r in b)
        row["truth_cases"] = sum(1 for r in key_rows if r["condition"] == c and r["source"] == "truth")
        out.append(row)
    return out


def floor_effect(key_rows) -> dict:
    """What the 1% floor does, against the ratio alone (p / 10) and a 2% floor."""
    d = [r for r in key_rows if r["source"] == "dxa" and r["status"] != ak.UNDETERMINED]
    by_floor = [r for r in d if r["status"] == ak.RED_HERRING and r["class_rate"] >= r["dxa_p"] / 1000.0]
    kept = [r for r in d if r["status"] == ak.KEPT]
    kept_1_2 = [r for r in kept if r["class_rate"] < 0.02]
    ratio_only = [r for r in d if r["status"] == ak.RED_HERRING and r["class_rate"] >= 0.01]
    cases_lost = {r["case_id"] for r in by_floor}
    return {
        "removed_only_by_floor": len(by_floor), "removed_only_by_floor_cases": len(cases_lost),
        "removed_only_by_floor_bands": dict(Counter(r["band"] for r in by_floor)),
        "kept_with_rate_1_to_2pct": len(kept_1_2), "kept_with_rate_1_to_2pct_cases": len({r["case_id"] for r in kept_1_2}),
        "removed_by_ratio_above_floor": len(ratio_only),
        "dxa_pairs_considered": len([r for r in key_rows if r["source"] == "dxa"]),
        "removed_total": sum(r["status"] == ak.RED_HERRING for r in d), "kept_total": len(kept),
        "undetermined_total": sum(1 for r in key_rows if r["source"] == "dxa" and r["status"] == ak.UNDETERMINED),
    }


def always_yes_list(key_rows, t1) -> list[str]:
    r10 = Counter(r["condition"] for r in key_rows if r["in_r10"] is True)
    r5 = Counter(r["condition"] for r in key_rows if r["in_r5"] is True)
    return sorted(t1, key=lambda c: (-r10[c], -r5[c], c))[:N_ALWAYS_YES]


def pool_availability(cands, drawn, rows, t1) -> list[dict]:
    out = []
    for c in t1:
        row = {"condition": c}
        for name in POOLS:
            lst = cands[name].get(c, [])
            row[f"{name}_available"] = len(lst)
            row[f"{name}_drawn"] = len(drawn[name][c])
            if name == "atypical":
                mix = Counter(max(rows[j]["dxa"], key=rows[j]["dxa"].get) for j in lst)
            else:
                mix = Counter(rows[j]["path"] for j in lst)
            row[f"{name}_mix"] = "; ".join(f"{k} {v}" for k, v in mix.most_common(3))
        out.append(row)
    return out


def compare_committed(hallmarks, t1) -> dict:
    """Hallmarks against results/analysis/dxa_red_herrings/hallmarks.csv (reference there also held out the 250 set)."""
    p = ROOT / "results/analysis/dxa_red_herrings/hallmarks.csv"
    if not p.exists():
        return {"hallmarks_committed": "absent"}
    old = defaultdict(list)
    for r in csv.DictReader(open(p, encoding="utf-8")):
        if r["variant"] == "symptoms":
            old[r["condition"]].append(r["token"])
    diff = {c: {"now": hallmarks[c], "before": old[c]} for c in t1 if set(hallmarks[c]) != set(old[c])}
    return {"hallmark_sets_differing": diff}


def compare_review_key(key_rows) -> dict:
    """Labels against the review's rebuild (results/analysis/v03_review/key470_targets.csv), which used the committed cells."""
    if not REVIEW_KEY.exists():
        return {"review_key": "absent"}
    old = {}
    for r in csv.DictReader(open(REVIEW_KEY, encoding="utf-8")):
        if r["why"] == "dxa":
            st = ak.RED_HERRING if r["red_herring"] == "True" else ak.UNDETERMINED if r["undetermined"] == "True" else ak.KEPT
            old[(r["case_id"], r["target"])] = st
    new = {(r["case_id"], r["condition"]): r["status"] for r in key_rows if r["source"] == "dxa"}
    both = set(old) & set(new)
    diff = sorted((k, old[k], new[k]) for k in both if old[k] != new[k])
    return {"pairs_compared": len(both), "only_in_review": len(set(old) - set(new)), "only_now": len(set(new) - set(old)),
            "label_differences": len(diff), "examples": [f"{a} {b}: {o} -> {n}" for (a, b), o, n in diff[:10]]}


def report(s: dict) -> None:
    k = s["key470"]
    print(f"\nKEY 470: R10 {k['r10']}, R5 {k['r5']}, clearly low-risk {k['clearly_low_risk']}, intermediate {k['intermediate']}; "
          f"R10 targets {k['r10_targets']} (undetermined {k['undetermined_r10_targets']}), R5 targets {k['r5_targets']}")
    print("\nRed herrings per tier-1 condition (DXA-derived pairs on the 470; removed / undetermined / kept):")
    for r in s["red_herrings_per_condition"]:
        print(f"  {r['condition'][:40]:40} 5-10%: {r['pairs_5-10']:3} ({r['removed_5-10']}/{r['undetermined_5-10']}/{r['kept_5-10']})"
              f"   >=10%: {r['pairs_10+']:3} ({r['removed_10+']}/{r['undetermined_10+']}/{r['kept_10+']})")
    print(f"\nFloor effect: {s['floor_effect']}")
    print(f"Always-YES list: {s['always_yes']}")
    print("\nFlag multiplicity (map codes matching > 1 tier-1 condition):")
    for p, v in s["flag_multiplicity"].items():
        print(f"  {p:8} {v['codes_matching_more_than_one_tier1']:4} of {v['codes_matching_tier1']} (max {v['max_tier1_per_code']})")
    print("\nPools (available / drawn):")
    for r in s["pool_availability"]:
        print(f"  {r['condition'][:40]:40} atypical {r['atypical_available']:5}/{r['atypical_drawn']:2}  high-risk {r['high-risk_available']:5}/{r['high-risk_drawn']:2}")
    print(f"\nChecks: {json.dumps(s['checks'], default=str)[:1500]}")


if __name__ == "__main__":
    main()
