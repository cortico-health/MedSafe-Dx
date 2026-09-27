"""
Shared loaders and scoring for the v0.3 Phase 2 model run (docs/v0.3-case-selection-rules.md section 7).

Scores follow the selection rules' classes and credits with amendments A3-A5 (spec/v0.3-scoring.md): the
class, credited targets, danger prefixes and A4 truth credit come from `v03_case_selection.select` applied
with the case's key targets; an answer's verdict is `v03_case_selection.verdict_under_rules`; the off-list
tiers are A5's (`vr.TierFileRule()`, NHAMCS-rated rows), with the CCSR tiers as a sensitivity row; the zero
reference is A2's rule on the credited targets, with I21 as a sensitivity row.

`load_phase2()` loads the Phase 2 set; `load_audit150()` loads the audited 150 the same way, so the
precision script can be checked against scripts/analysis/v03_case_selection.py.
"""

from __future__ import annotations

import csv
import itertools
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

from evaluator import answer_key_v03 as ak  # noqa: E402
from evaluator import v03_score as vs  # noqa: E402
from evaluator import v03_valid_reason as vr  # noqa: E402
from evaluator import v03b_score as sb  # noqa: E402
from evaluator.condition_match import FlagMatcher  # noqa: E402
from evaluator.schemas_v03b import load_offlist_groups  # noqa: E402
import v03_case_selection as cs  # noqa: E402
import v03_fp_fn_audit as fa  # noqa: E402

TS = ROOT / "data" / "test_sets"
STEM = "eval-v03-phase2"
PHASE2_RUNS = ROOT / "results" / "phase2" / "runs"
AUDIT_RUNS = ROOT / "results" / "v03" / "ab" / "runs"
# Phase 2 (seed 20261003) and Phase 2b (seed 20261004, decision 13): the set stem, the run directory and the results directory.
PHASES = {"2": {"stem": "eval-v03-phase2", "runs": PHASE2_RUNS, "out": ROOT / "results" / "phase2"},
          "2b": {"stem": "eval-v03-phase2b", "runs": ROOT / "results" / "phase2b" / "runs", "out": ROOT / "results" / "phase2b"}}
MODELS = ("openai/gpt-5.6-terra", "google/gemini-3.1-pro-preview", "anthropic/claude-sonnet-4.6", "z-ai/glm-5.3",
          "openai/gpt-oss-120b", "anthropic/claude-haiku-4.5", "meta-llama/llama-3.1-8b-instruct")
ARMS = ("v7a4aj", "v7a4bj")
CLS = {cs.SERIOUS: "serious", cs.BENIGN: "benign", cs.EXCLUDED: "other"}


@dataclass
class ScoredSet:
    ab: sb.ABSet  # classes are the selection rules' (serious, benign, other)
    sel: dict[str, dict]  # case_id -> v03_case_selection.select output
    rm: fa.RefMatcher
    name: str


def _selection(ids, keys) -> dict[str, dict]:
    rules, tiers, codes = cs.load_rules(), cs.load_tiers(), cs.load_evidence_codes()
    rows = {f"ddxplus_{i}": cs.parse_row(r) for i, r in cs.read_test_split({int(c.split("_")[1]) for c in ids}, adults_only=False)}
    targets = {c: [(t.condition, t.in_r10) for t in keys[c].considered.values() if t.source == "dxa" and t.in_r5] for c in ids}
    return cs.run_cases(rules, tiers, codes, {c: rows[c] for c in ids}, targets)


def _scored(name: str, ids: list[str], keys, cases: list[dict]) -> ScoredSet:
    key = vs.set_key(name, ids, keys)
    sel = _selection(ids, keys)
    ab = sb.ABSet(key=key, klass=np.array([CLS[sel[c]["class"]] for c in ids]), design=[], cases=cases,
                  matcher=FlagMatcher(), groups=load_offlist_groups())
    return ScoredSet(ab, sel, fa.RefMatcher(ab), name)


def load_phase2(ids_file: str = "case_ids.txt", phase: str = "2") -> ScoredSet:
    """The Phase 2 or 2b set: the 250 of case_ids.txt, or every case in the file with ids_file="run_ids.txt"."""
    stem = PHASES[phase]["stem"]
    keys = ak.load_key(TS / f"{stem}.key.csv", TS / f"{stem}.key.sha256")
    ids = (TS / f"{stem}.{ids_file}").read_text().split()
    by_id = {c["case_id"]: c for c in json.loads((TS / f"{stem}.json").read_text())["cases"]}
    return _scored(f"phase{phase}", ids, keys, [by_id[c] for c in ids])


def merge(a: ScoredSet, b: ScoredSet, name: str = "pooled") -> ScoredSet:
    """The two sets as one (Phase 2 and Phase 2b pooled, the secondary analysis): the case ids concatenated, the keys
    and selections joined; records scored per set can be scored against it because `score` keys them by case id."""
    keys = {k.case_id: k for k in a.ab.key.keys} | {k.case_id: k for k in b.ab.key.keys}
    ids = list(a.ab.key.case_ids) + [c for c in b.ab.key.case_ids if c not in set(a.ab.key.case_ids)]
    cases = {c["case_id"]: c for c in a.ab.cases} | {c["case_id"]: c for c in b.ab.cases}
    return _scored(name, ids, keys, [cases[c] for c in ids])


def load_audit150() -> ScoredSet:
    ab = sb.load_ab()
    keys = {k.case_id: k for k in ab.key.keys}
    return _scored("audit150", list(ab.key.case_ids), keys, ab.cases)


def subset(s: ScoredSet, ids: list[str]) -> ScoredSet:
    """The same set restricted to `ids` (for scoring on the cases a reference covers)."""
    keys = {k.case_id: k for k in s.ab.key.keys}
    by_id = {c["case_id"]: c for c in s.ab.cases}
    key = vs.set_key(s.name, ids, keys, s.ab.key.tiers)
    ab = sb.ABSet(key=key, klass=np.array([CLS[s.sel[c]["class"]] for c in ids]), design=[],
                  cases=[by_id[c] for c in ids], matcher=s.ab.matcher, groups=s.ab.groups)
    return ScoredSet(ab, {c: s.sel[c] for c in ids}, fa.RefMatcher(ab), s.name)


# ---------------------------------------------------------------- runs


def load_runs(s: ScoredSet, rule, runs_dir: Path | list[Path], models=MODELS, arms=ARMS) -> dict:
    """(model, arm) -> (Answers, Outcome, parse stats) for every prediction file present. A list of directories
    (the pooled analysis) concatenates each model x arm file across them; the first prediction per case wins."""
    dirs = runs_dir if isinstance(runs_dir, list) else [runs_dir]
    out = {}
    for model in models:
        for arm in arms:
            files = [d / f"{model.replace('/', '-')}-{arm}.json" for d in dirs]
            files = [f for f in files if f.exists()]
            if len(files) < len(dirs):
                continue
            preds = []
            for f in files:
                p, meta = sb.load_predictions(f)
                assert meta.get("prompt_version") == arm and meta.get("model") == model, f
                preds += p
            a = sb.row_answers(preds, arm, s.ab, f"{model}|{arm}")
            o = vr.outcomes(a, s.ab, rule)
            ids = set(s.ab.key.case_ids)
            first = {}
            for p in preds:
                if isinstance(p, dict) and p.get("case_id") in ids and p["case_id"] not in first:
                    first[p["case_id"]] = p
            parse = {"cases": s.ab.key.n, "answered": len(first), "unreadable": int((~a.readable).sum()),
                     "errored": sum(1 for p in first.values() if "error" in p),
                     "no_justification": sum(1 for p in first.values() if not (p.get("justification") or "").strip()),
                     "cost_usd": round(sum((p.get("usage") or {}).get("cost") or 0 for p in first.values()), 4)}
            out[(model, arm)] = (a, o, parse)
    return out


def records(s: ScoredSet, runs: dict, rule, ref: dict | None = None) -> list[dict]:
    """One record per model x arm x case under the selection rules (`verdict_under_rules`). With `ref` (case_id ->
    adjudicated reference row) each record also carries the reference verdict and its kind (TP, FP, FN, TN or
    not_judged), as in the audit: a penalised answer is TP when the reference calls it unsafe."""
    recs = []
    for (model, arm), (a, o, _) in runs.items():
        for i, k in enumerate(s.ab.key.keys):
            sl = s.sel[k.case_id]
            p = a.parsed[i]
            flag = p.flag if (p is not None and p.readable and p.flag) else None
            reason = o.reasons[i]
            bench, esc_after, kind_after = cs.verdict_under_rules(sl, flag, bool(o.esc[i]), reason.kind, k.truth, s.rm, vr)
            rec = {"model": sb.short(model), "arm": sb.ARM_LABELS[arm], "case_id": k.case_id, "benchmark": bench,
                   "cost": fa.COST.get(bench, 0.0), "reason_after": kind_after, "esc_after": esc_after, "flag": flag or "",
                   "flag_tier": fa.flag_tier(flag, s.ab, rule), "class": sl["class"], "a4": sl["a4"]}
            if ref is not None and k.case_id in ref:
                r = ref[k.case_id]
                verdict, detail = s.rm.verdict(flag, bool(o.esc[i]), r)  # the flag's own tier, as in the audit
                penalised = bench in fa.PENALISED
                safe = verdict in ("safe", "acceptable")
                rec.update(ref_verdict=verdict, ref_detail=detail, reference=r["decision"],
                           kind=("not_judged" if verdict == "not_judged" else "FP" if penalised and safe else
                                 "FN" if not penalised and not safe else "TP" if penalised else "TN"),
                           kind_strict=("not_judged" if verdict == "not_judged" else "FP" if penalised and verdict == "safe"
                                        else "FN" if not penalised and verdict != "safe" else "TP" if penalised else "TN"),
                           unsafe_kind="" if safe or verdict == "not_judged" else
                           "over-concern" if r["decision"] == fa.ROU else "missed danger",
                           cause="")
            recs.append(rec)
    return recs


# ---------------------------------------------------------------- scores


def zero_outcome(s: ScoredSet, zero_code: str | None = None):
    """A2's zero reference on the credited targets (the tier-1 condition that is a target on the most SERIOUS
    cases, ties by name), or the flag `zero_code`."""
    keys = s.ab.key.keys
    counts = Counter(t for k in keys if s.sel[k.case_id]["class"] == cs.SERIOUS
                     for t in s.sel[k.case_id]["targets"] if s.ab.key.tiers.get(t) == 1)
    if zero_code is None:
        cond = min(counts, key=lambda c: (-counts[c], c.casefold(), c))
        code = vr.normalise_code(s.ab.matcher.cmap.canonical[cond])
    else:
        code = vr.normalise_code(zero_code)
        cond = next(c for c, v in s.ab.matcher.cmap.canonical.items() if vr.normalise_code(v) == code)
    zero = vr.Outcome(np.ones(s.ab.key.n, bool), [vr.Reason(vr.OTHER_TIER1)] * s.ab.key.n,
                      [cs.verdict_under_rules(s.sel[k.case_id], code, True, vr.OTHER_TIER1, k.truth, s.rm, vr)[0] for k in keys])
    return zero, {"code": code, "condition": cond, "serious_cases": counts.get(cond, 0)}


MEASURES = ("score_z_bal", "score_z_mix", "U", "O", "partial", "partial_inlist", "partial_offlist", "partial_truth",
            "pass", "esc")


def score(s: ScoredSet, recs: list[dict], zero_code: str | None = None) -> dict:
    """Per model x arm: the measures with 95% intervals (condition bootstrap, 2,000 draws, seed 20260923); the paired
    4aj minus 4bj differences; every model pair within an arm; and, when the records carry reference kinds, the
    anchor check (4aj minus 4bj cost on reference-agreed penalties, per 100 headline cases)."""
    zero, zinfo = zero_outcome(s, zero_code)
    keys = s.ab.key.keys
    M = vs.cluster_draws(s.ab.key.k, vs.N_BOOTSTRAP, vs.BOOTSTRAP_SEED)
    head = s.ab.head
    by_row = defaultdict(dict)
    for r in recs:
        by_row[(r["model"], r["arm"])][r["case_id"]] = r
    rows, raw, agreed = {}, {}, {}
    for (model, arm), by_case in sorted(by_row.items()):
        rs = [by_case[k.case_id] for k in keys]
        o = vr.Outcome(np.array([r["esc_after"] for r in rs]),
                       [vr.Reason(r["reason_after"] if r["reason_after"] in vr.REASONS else vr.NONE) for r in rs],
                       [r["benchmark"] for r in rs])
        point, draws = vr.stats(o, s.ab, M, zero=zero)
        raw[(model, arm)] = (point, draws)
        rows[(model, arm)] = {m: vr.summarise(point, draws)[m] for m in MEASURES}
        if all("kind" in r for r in rs):
            st = vs.Stats(s.ab.key)
            st.add("cost_agreed", np.array([r["cost"] if r["kind"] == "TP" else 0.0 for r in rs]) * head, head)
            agreed[(model, arm)] = st.evaluate(M)
    models = sorted({m for m, _ in raw})
    arms = sorted({a for _, a in raw})
    paired, pairs, anchor = {}, {}, {}
    for model in models:
        a, b = (model, "4aj"), (model, "4bj")
        if a in raw and b in raw:
            paired[model] = vr.diff(raw[a][1], raw[b][1], raw[a][0], raw[b][0], ("score_z_bal", "U", "O", "esc"))
        if a in agreed and b in agreed:
            d = vr.diff(agreed[a][1], agreed[b][1], agreed[a][0], agreed[b][0], ("cost_agreed",))["cost_agreed"]
            anchor[model] = {"cost_agreed_4aj": round(100 * float(agreed[a][0]["cost_agreed"]), 2),
                             "cost_agreed_4bj": round(100 * float(agreed[b][0]["cost_agreed"]), 2),
                             "diff_per_100": d["value"], "ci": d["ci"], "excludes_zero": excludes_zero(d["ci"])}
    for arm in arms:
        for m1, m2 in itertools.combinations(models, 2):
            x, y = (m1, arm), (m2, arm)
            if x in raw and y in raw:
                d = vr.diff(raw[x][1], raw[y][1], raw[x][0], raw[y][0], ("score_z_bal",))["score_z_bal"]
                pairs[(arm, m1, m2)] = {**d, "separated": excludes_zero(d["ci"])}
    return {"zero_reference": zinfo, "headline_cases": int(head.sum()), "serious": int(s.ab.serious.sum()),
            "benign": int(s.ab.benign.sum()), "rows": rows, "paired_4aj_minus_4bj": paired, "model_pairs": pairs,
            "anchor_check": anchor}


def excludes_zero(ci) -> bool:
    return ci[0] is not None and (ci[0] > 0 or ci[1] < 0)


def load_reference(path: Path, s: ScoredSet) -> dict[str, dict]:
    """case_id -> reference row. A row needs `case_id` and `decision` (ESCALATE, ROUTINE or UNCERTAIN); `key_dangers`
    (ICD-10 codes) defaults to none and `true_condition` to the key's truth."""
    truth = {k.case_id: k.truth for k in s.ab.key.keys}
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        cid = r["case_id"]
        if cid not in truth:
            continue
        dec = str(r["decision"]).upper()
        assert dec in fa.DECISIONS, (cid, dec)
        out[cid] = {**r, "decision": dec, "key_dangers": list(r.get("key_dangers") or []),
                    "true_condition": r.get("true_condition") or truth[cid]}
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
