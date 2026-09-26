#!/usr/bin/env python3
"""
Phase 2 label validation of the frozen case-selection rules (docs/v0.3-case-selection-rules.md
section 7; freeze commit 45a7599). No model is called here; the reviews were produced separately.

Inputs, under results/phase2/: cases_blind.json (scripts/analysis/v03_phase2_cases.py), the two
blind Fable reviews (reference_fable_A.jsonl for cases 0-124, reference_fable_B.jsonl for 125-249,
each written in two parts), the Astra review (reference_astra.jsonl, five codex chunks), the
adjudications (adjudications.json: one entry per case where Fable and Astra disagreed or either said
UNCERTAIN) and the unblinded key (cases_unblinded.json, kept outside the repo during the review).

Two stages:

  collect   merge the review parts, parse Astra's JSON Lines into reference_astra.json, report the
            Fable x Astra agreement and kappa, and write adjudication_queue.json: the intake and both
            reviews for every case the adjudicator must decide (no truth, stratum or rule in it).
  finalize  build reference_adjudicated.jsonl the way the audit did (agreed cases adopt the agreed
            decision, the union of dangers and the lower confidence; queued cases take the
            adjudication), draw the seeded spot check, evaluate the pre-registered label-level
            criteria 1-5 and write label_validation.md and label_validation.json.

The criteria (section 7, fixed before the review):
1. class agreement on decided kept cases >= 90%, Wilson 95% lower bound >= 85%;
2. BENIGN stratum >= 85% ROUTINE among decided; each SERIOUS stratum >= 92% ESCALATE among decided;
3. each rule with >= 5 fresh cases it decided: >= 80% agreement, else flagged for demotion;
4. Cohen's kappa between the two reviewers >= 0.6 on the decision;
5. EXCLUDED stratum, reported only: ESCALATE / ROUTINE / UNCERTAIN / split shares, and any exclusion
   rule whose cases are ESCALATE in >= 80% of draws at confidence >= 4 (a candidate PATCH later).

A case's class is the rules' class under the v0.3b key (`class_key`): two BENIGN candidates whose
key carries an R5-only target fall to EXCLUDED under X9 and are evaluated there.

Usage: python3 scripts/analysis/v03_phase2_validation.py collect|finalize [--unblinded PATH]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P2 = ROOT / "results" / "phase2"
ESC, ROU, UNC = "ESCALATE", "ROUTINE", "UNCERTAIN"
DECISIONS = (ESC, ROU, UNC)
SPOT_SEED = 20261003
SPOT_N = 10
STRATA = ("serious_tier1", "serious_upgrade_or_flag", "serious_dxa_only", "benign", "excluded")


# ---------------------------------------------------------------- statistics


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def kappa(pairs) -> tuple[float, float]:
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[d] * cb[d] for d in DECISIONS) / n / n
    return (po - pe) / (1 - pe), po


def pct(k: int, n: int) -> str:
    return f"{100 * k / n:.1f}%" if n else "n/a"


# ---------------------------------------------------------------- reviews


def read_jsonl(path: Path) -> list[dict]:
    out = []
    for line in path.read_text().splitlines():
        line = line.strip().strip("`")
        if not line or line.startswith("```"):
            continue
        # A chunk file without a trailing newline joins two objects on one line: decode every object on the line.
        dec = json.JSONDecoder()
        pos = line.find("{")
        while pos >= 0:
            try:
                obj, end = dec.raw_decode(line, pos)
            except json.JSONDecodeError:
                break
            out.append(obj)
            pos = line.find("{", end)
    return out


def norm_decision(d: str) -> str:
    d = (d or "").strip().upper()
    if d.startswith("ESC"):
        return ESC
    if d.startswith("ROU"):
        return ROU
    return UNC


def merge_fable(name: str) -> dict[str, dict]:
    parts = sorted(P2.glob(f"reference_fable_{name}.part*.jsonl"))
    rows: dict[str, dict] = {}
    for p in parts:
        for r in read_jsonl(p):
            rows[r["case_id"]] = r
    merged = P2 / f"reference_fable_{name}.jsonl"
    merged.write_text("".join(json.dumps(rows[c]) + "\n" for c in sorted(rows, key=lambda c: rows[c]["index"])))
    return rows


def load_astra() -> dict[str, dict]:
    src = P2 / "reference_astra.jsonl"
    rows = {}
    for r in read_jsonl(src):
        r["decision"] = norm_decision(r.get("decision"))
        r.setdefault("dangers_to_consider", [])
        r.setdefault("red_flags", [])
        r["confidence"] = int(r.get("confidence") or 3)
        rows[r["case_id"]] = r
    return rows


def load_blind() -> list[dict]:
    return json.loads((P2 / "cases_blind.json").read_text())


def fable_decision(f: dict) -> str:
    return norm_decision(f["partA"]["decision"])


def agreement_table(fable: dict, astra: dict, ids: list[str]):
    table = Counter()
    pairs = []
    for cid in ids:
        a, b = fable_decision(fable[cid]), astra[cid]["decision"]
        table[(a, b)] += 1
        pairs.append((a, b))
    k, po = kappa(pairs)
    return table, k, po


def table_md(table: Counter, row_label: str, col_label: str) -> str:
    L = [f"| {row_label} \\ {col_label} | " + " | ".join(DECISIONS) + " |", "|---|---|---|---|"]
    for a in DECISIONS:
        L.append(f"| {a} | " + " | ".join(str(table[(a, b)]) for b in DECISIONS) + " |")
    return "\n".join(L)


# ---------------------------------------------------------------- stage: collect


def collect(args) -> None:
    blind = load_blind()
    ids = [c["case_id"] for c in blind]
    fable = {**merge_fable("A"), **merge_fable("B")}
    astra = load_astra()
    (P2 / "reference_astra.json").write_text(json.dumps([astra[c] for c in ids if c in astra], indent=1) + "\n")
    missing_f = [c for c in ids if c not in fable]
    missing_a = [c for c in ids if c not in astra]
    print(f"fable {len(fable)} cases (missing {len(missing_f)}), astra {len(astra)} (missing {len(missing_a)})")
    if missing_f or missing_a:
        print("  missing fable:", missing_f[:10], " missing astra:", missing_a[:10])
    both = [c for c in ids if c in fable and c in astra]
    table, k, po = agreement_table(fable, astra, both)
    print(table_md(table, "Fable", "Astra"))
    print(f"kappa {k:.3f}, raw agreement {po:.3f} on {len(both)} cases")
    queue = []
    for c in blind:
        cid = c["case_id"]
        if cid not in fable or cid not in astra:
            continue
        fd, ad = fable_decision(fable[cid]), astra[cid]["decision"]
        if fd == ad != UNC:
            continue
        fa = fable[cid]["partA"]
        queue.append({"case_id": cid, "index": fable[cid]["index"], "intake": c["intake"],
                      "clinician_working_diagnosis": c["clinician_working_diagnosis"],
                      "fable": {"decision": fd, "dangers_to_consider": fa["dangers_to_consider"], "red_flags": fa["red_flags"],
                                "evidence": fa.get("evidence", []), "confidence": fa["confidence"], "reasoning": fa["reasoning"]},
                      "astra": {k2: astra[cid].get(k2) for k2 in ("decision", "dangers_to_consider", "red_flags", "evidence",
                                                                 "confidence", "reasoning")}})
    (P2 / "adjudication_queue.json").write_text(json.dumps(queue, indent=1) + "\n")
    kinds = Counter("split" if q["fable"]["decision"] != q["astra"]["decision"] and UNC not in (q["fable"]["decision"], q["astra"]["decision"])
                    else "uncertain" for q in queue)
    print(f"adjudication queue: {len(queue)} cases ({dict(kinds)}) -> {P2 / 'adjudication_queue.json'}")


# ---------------------------------------------------------------- stage: finalize


def build_reference(blind, fable, astra, adjudicated, unblinded, spot: set[str]) -> list[dict]:
    rows = []
    for c in blind:
        cid = c["case_id"]
        f, a, u = fable[cid], astra[cid], unblinded[cid]
        fa = f["partA"]
        fd = fable_decision(f)
        entry = {"case_id": cid, "index": u["index"], "fable": fd, "astra": a["decision"],
                 "fable_confidence": fa["confidence"], "astra_confidence": a["confidence"],
                 "true_condition": u["true_condition"], "truth_tier": u["truth_tier"],
                 "benchmark_class": u["class_key"].lower(), "stratum": u["stratum"], "rules": u["rules"],
                 "r10_targets": u["r10_targets"]}
        if cid in adjudicated:
            adj = adjudicated[cid]
            entry.update(decision=norm_decision(adj["decision"]), key_dangers=adj["dangers"], confidence=int(adj["confidence"]),
                         rationale=adj["rationale"], source="spot-check overturned" if cid in spot else "adjudicated")
        else:
            assert fd == a["decision"] != UNC, cid
            dangers = list(dict.fromkeys(list(fa["dangers_to_consider"]) + list(a["dangers_to_consider"])))
            entry.update(decision=fd, key_dangers=dangers, confidence=min(fa["confidence"], a["confidence"]),
                         rationale=fa["reasoning"].split(". ")[0].rstrip(".") + ".",
                         source="spot-checked" if cid in spot else "agreed")
        entry["fable_evidence"] = [e.get("citation") for e in fa.get("evidence", []) if isinstance(e, dict)]
        rows.append(entry)
    return rows


def agrees(r: dict) -> bool | None:
    """True/False for a decided kept case, None for UNCERTAIN or EXCLUDED."""
    if r["benchmark_class"] not in ("serious", "benign") or r["decision"] == UNC:
        return None
    return (r["benchmark_class"] == "serious") == (r["decision"] == ESC)


def finalize(args) -> None:
    blind = load_blind()
    ids = [c["case_id"] for c in blind]
    fable = {**{r["case_id"]: r for r in read_jsonl(P2 / "reference_fable_A.jsonl")},
             **{r["case_id"]: r for r in read_jsonl(P2 / "reference_fable_B.jsonl")}}
    astra = {r["case_id"]: r for r in json.loads((P2 / "reference_astra.json").read_text())}
    unblinded = {u["case_id"]: u for u in json.loads(Path(args.unblinded).read_text())}
    adjudicated = json.loads((P2 / "adjudications.json").read_text())
    if isinstance(adjudicated, list):
        adjudicated = {a["case_id"]: a for a in adjudicated}
    table, k, po = agreement_table(fable, astra, ids)
    queue_ids = [cid for cid in ids if not (fable_decision(fable[cid]) == astra[cid]["decision"] != UNC)]
    missing_adj = [cid for cid in queue_ids if cid not in adjudicated]
    assert not missing_adj, f"adjudications missing for {missing_adj}"
    agreed_ids = [cid for cid in ids if cid not in queue_ids]
    rng = random.Random(SPOT_SEED)
    spot = set(rng.sample(agreed_ids, min(SPOT_N, len(agreed_ids))))
    ref = build_reference(blind, fable, astra, adjudicated, unblinded, spot)
    (P2 / "reference_adjudicated.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ref))

    out: dict = {"n": len(ref), "kappa": k, "raw_agreement": po, "confusion": {f"{a}|{b}": v for (a, b), v in table.items()},
                 "adjudicated": len(queue_ids), "spot_check": sorted(spot), "decisions": dict(Counter(r["decision"] for r in ref))}
    L = ["# Phase 2 label validation of the frozen case-selection rules", "",
         f"Freeze commit 45a7599; draw seed 20261003 (docs/v0.3-case-selection-rules.md section 7). Script: "
         f"`scripts/analysis/v03_phase2_validation.py`; cases from `scripts/analysis/v03_phase2_cases.py`. "
         f"Reference: two blind Fable reviews with verified citations (A on cases 0-124, B on 125-249), a blind Astra "
         f"review from knowledge (all 250), and a blind adjudication of every disagreement and every UNCERTAIN "
         f"({len(queue_ids)} cases); the {len(agreed_ids)} agreed cases adopt the agreed decision, and {len(spot)} of them "
         f"were spot-checked (seed {SPOT_SEED}). No reviewer saw the truth, the stratum, the rules or any model output.", ""]

    # Criterion 4: reviewer agreement.
    L += ["## Reviewer agreement (criterion 4: kappa >= 0.6)", "", table_md(table, "Fable", "Astra"), "",
          f"Cohen's kappa {k:.3f}, raw agreement {po * 100:.1f}% ({int(round(po * len(ids)))} of {len(ids)}). "
          f"**{'PASS' if k >= 0.6 else 'FAIL'}.**", ""]
    out["criterion_4"] = {"kappa": k, "pass": k >= 0.6}

    # Reference by stratum.
    by_stratum = defaultdict(Counter)
    for r in ref:
        by_stratum[r["stratum"]][r["decision"]] += 1
    L += ["## The reference by stratum", "", "| Stratum | n | ESCALATE | ROUTINE | UNCERTAIN | class under the key |", "|---|---|---|---|---|---|"]
    for s in STRATA:
        rows = [r for r in ref if r["stratum"] == s]
        cls = Counter(r["benchmark_class"] for r in rows)
        L.append(f"| {s} | {len(rows)} | {by_stratum[s][ESC]} | {by_stratum[s][ROU]} | {by_stratum[s][UNC]} | "
                 + ", ".join(f"{c} {n}" for c, n in sorted(cls.items())) + " |")
    L.append("")
    out["by_stratum"] = {s: dict(by_stratum[s]) for s in STRATA}

    # Criterion 1: class agreement on decided kept cases.
    kept = [r for r in ref if r["benchmark_class"] in ("serious", "benign")]
    decided = [r for r in kept if r["decision"] != UNC]
    agree = [r for r in decided if agrees(r)]
    lo, hi = wilson(len(agree), len(decided))
    c1 = len(agree) / len(decided) >= 0.90 and lo >= 0.85
    L += ["## Criterion 1: class agreement on decided kept cases (>= 90%, Wilson lower bound >= 85%)", "",
          f"{len(agree)} of {len(decided)} decided kept cases agree: {pct(len(agree), len(decided))} "
          f"[{100 * lo:.1f}, {100 * hi:.1f}]; {len(kept) - len(decided)} kept cases are UNCERTAIN. **{'PASS' if c1 else 'FAIL'}.**", "",
          "| Class | ESCALATE | ROUTINE | UNCERTAIN | total |", "|---|---|---|---|---|"]
    for cls in ("serious", "benign", "excluded"):
        rows = [r for r in ref if r["benchmark_class"] == cls]
        cnt = Counter(r["decision"] for r in rows)
        L.append(f"| {cls.upper()} | {cnt[ESC]} | {cnt[ROU]} | {cnt[UNC]} | {len(rows)} |")
    L.append("")
    out["criterion_1"] = {"agree": len(agree), "decided": len(decided), "rate": len(agree) / len(decided), "wilson": [lo, hi], "pass": c1}

    # Criterion 2: per stratum.
    L += ["## Criterion 2: BENIGN >= 85% ROUTINE; each SERIOUS stratum >= 92% ESCALATE (among decided cases)", "",
          "| Stratum | decided | agree | rate [Wilson 95%] | target | result |", "|---|---|---|---|---|---|"]
    c2 = {}
    for s in STRATA[:4]:
        rows = [r for r in ref if r["stratum"] == s and r["benchmark_class"] in ("serious", "benign")]
        dec = [r for r in rows if r["decision"] != UNC]
        ag = [r for r in dec if agrees(r)]
        target = 0.85 if s == "benign" else 0.92
        rate = len(ag) / len(dec) if dec else 0.0
        lo2, hi2 = wilson(len(ag), len(dec))
        ok = rate >= target
        c2[s] = {"decided": len(dec), "agree": len(ag), "rate": rate, "wilson": [lo2, hi2], "target": target, "pass": ok}
        L.append(f"| {s} | {len(dec)} | {len(ag)} | {pct(len(ag), len(dec))} [{100 * lo2:.1f}, {100 * hi2:.1f}] | >= {int(target * 100)}% | {'PASS' if ok else 'FAIL'} |")
    L += ["", f"Two BENIGN candidates (ddxplus_23603, ddxplus_68301) carry an R5-only target under the key and fall to EXCLUDED "
              f"under X9; they are counted in the EXCLUDED stratum below, not here.", ""]
    out["criterion_2"] = c2

    # Criterion 3: per rule.
    per_rule: dict[str, list[dict]] = defaultdict(list)
    for r in ref:
        for t in r["rules"].split("|"):
            if t:
                per_rule[t].append(r)
    L += ["## Criterion 3: each rule with >= 5 fresh cases it decided, >= 80% agreement", "",
          "| Rule | cases | decided | agree | rate | ESCALATE / ROUTINE / UNCERTAIN | result |", "|---|---|---|---|---|---|---|"]
    c3 = {}
    flagged = []
    for rule in sorted(per_rule, key=lambda x: (x[0] != "P", x)):
        rows = per_rule[rule]
        if rows[0]["benchmark_class"] == "excluded":
            continue
        dec = [r for r in rows if r["decision"] != UNC]
        ag = [r for r in dec if agrees(r)]
        cnt = Counter(r["decision"] for r in rows)
        rate = len(ag) / len(dec) if dec else 0.0
        judged = len(rows) >= 5
        ok = rate >= 0.80
        if judged and not ok:
            flagged.append(rule)
        c3[rule] = {"cases": len(rows), "decided": len(dec), "agree": len(ag), "rate": rate, "judged": judged, "pass": ok,
                    "decisions": dict(cnt)}
        L.append(f"| {rule} | {len(rows)} | {len(dec)} | {len(ag)} | {pct(len(ag), len(dec))} | {cnt[ESC]} / {cnt[ROU]} / {cnt[UNC]} | "
                 f"{('PASS' if ok else 'FAIL, flagged for demotion') if judged else 'reported only (< 5 cases)'} |")
    L += ["", ("Rules flagged for demotion to EXCLUDE: " + ", ".join(flagged) + ".") if flagged else
          "No rule with 5 or more fresh cases falls below 80% agreement.", ""]
    out["criterion_3"] = {"per_rule": c3, "flagged": flagged}

    # Criterion 5: the EXCLUDED stratum.
    exc = [r for r in ref if r["benchmark_class"] == "excluded"]
    cnt = Counter(r["decision"] for r in exc)
    split = [r for r in exc if r["fable"] != r["astra"]]
    L += ["## Criterion 5 (reported only): the EXCLUDED stratum", "",
          f"{len(exc)} cases: ESCALATE {cnt[ESC]} ({pct(cnt[ESC], len(exc))}), ROUTINE {cnt[ROU]} ({pct(cnt[ROU], len(exc))}), "
          f"UNCERTAIN {cnt[UNC]} ({pct(cnt[UNC], len(exc))}); the two reviewers split on {len(split)} ({pct(len(split), len(exc))}).", "",
          "| Exclusion rule | cases | ESCALATE | ROUTINE | UNCERTAIN | split | ESCALATE at confidence >= 4 | candidate PATCH (>= 80%) |",
          "|---|---|---|---|---|---|---|---|"]
    c5 = {"n": len(exc), "decisions": dict(cnt), "split": len(split), "per_rule": {}}
    for rule in sorted(per_rule):
        rows = [r for r in per_rule[rule] if r["benchmark_class"] == "excluded"]
        if not rows:
            continue
        rc = Counter(r["decision"] for r in rows)
        hi_esc = sum(1 for r in rows if r["decision"] == ESC and r["confidence"] >= 4)
        sp = sum(1 for r in rows if r["fable"] != r["astra"])
        cand = hi_esc / len(rows) >= 0.80
        c5["per_rule"][rule] = {"cases": len(rows), "decisions": dict(rc), "split": sp, "escalate_conf4": hi_esc, "candidate_patch": cand}
        L.append(f"| {rule} | {len(rows)} | {rc[ESC]} | {rc[ROU]} | {rc[UNC]} | {sp} | {hi_esc} ({pct(hi_esc, len(rows))}) | {'yes' if cand else 'no'} |")
    L.append("")
    out["criterion_5"] = c5

    # Per-condition view of the kept strata, for reading the failures.
    L += ["## Kept cases by condition", "", "| Condition | tier | class | n | ESCALATE | ROUTINE | UNCERTAIN | agree |", "|---|---|---|---|---|---|---|---|"]
    by_cond = defaultdict(list)
    for r in kept:
        by_cond[(r["true_condition"], r["truth_tier"], r["benchmark_class"])].append(r)
    for (cond, tier, cls), rows in sorted(by_cond.items(), key=lambda kv: (kv[0][2], kv[0][1], kv[0][0])):
        cc = Counter(r["decision"] for r in rows)
        ag = sum(1 for r in rows if agrees(r))
        L.append(f"| {cond} | {tier} | {cls} | {len(rows)} | {cc[ESC]} | {cc[ROU]} | {cc[UNC]} | {ag} |")
    L.append("")

    # The disagreeing kept cases.
    bad = [r for r in kept if agrees(r) is False]
    L += [f"## Kept cases where the reference disagrees with the class ({len(bad)})", "",
          "| Idx | Case | Truth (tier) | Stratum | Rules | Class | Reference (conf.) | Source | Rationale |", "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(bad, key=lambda r: r["index"]):
        L.append(f"| {r['index']} | {r['case_id']} | {r['true_condition']} ({r['truth_tier']}) | {r['stratum']} | {r['rules'] or '-'} | "
                 f"{r['benchmark_class'].upper()} | {r['decision']} ({r['confidence']}) | {r['source']} | {r['rationale']} |")
    L.append("")
    unc = [r for r in kept if r["decision"] == UNC]
    L += [f"## Kept cases left UNCERTAIN ({len(unc)})", "", "| Idx | Case | Truth (tier) | Stratum | Rules | Class | Rationale |", "|---|---|---|---|---|---|---|"]
    for r in sorted(unc, key=lambda r: r["index"]):
        L.append(f"| {r['index']} | {r['case_id']} | {r['true_condition']} ({r['truth_tier']}) | {r['stratum']} | {r['rules'] or '-'} | "
                 f"{r['benchmark_class'].upper()} | {r['rationale']} |")
    L.append("")

    # Verdict.
    passes = {"1": c1, "2": all(v["pass"] for v in c2.values()), "3": not flagged, "4": k >= 0.6}
    L += ["## Verdict", "", "| Criterion | Result |", "|---|---|",
          f"| 1. Class agreement on decided kept cases | {'PASS' if passes['1'] else 'FAIL'}: {pct(len(agree), len(decided))} [{100 * lo:.1f}, {100 * hi:.1f}] |",
          f"| 2. Per-stratum agreement | {'PASS' if passes['2'] else 'FAIL'}: " + "; ".join(f"{s} {pct(v['agree'], v['decided'])}" for s, v in c2.items()) + " |",
          f"| 3. Per-rule agreement (>= 5 cases) | {'PASS' if passes['3'] else 'FAIL'}: " + (", ".join(flagged) + " flagged" if flagged else "no rule flagged") + " |",
          f"| 4. Reviewer kappa | {'PASS' if passes['4'] else 'FAIL'}: {k:.3f} |",
          f"| 5. EXCLUDED stratum (reported) | ESCALATE {pct(cnt[ESC], len(exc))}, ROUTINE {pct(cnt[ROU], len(exc))}, UNCERTAIN {pct(cnt[UNC], len(exc))}, split {pct(len(split), len(exc))} |",
          "", "Section 7 says failing criterion 1 or 2 means the slice is not certified and Phase 2 repeats on seed 20261004; "
              "a rule flagged under criterion 3 is demoted to EXCLUDE. This document reports; it changes no rule.", ""]
    out["passes"] = passes

    # Summary, from the numbers above, placed after the heading paragraph.
    dxa = c2["serious_dxa_only"]
    dxa_rows = [r for r in ref if r["stratum"] == "serious_dxa_only"]
    dxa_truths = Counter(r["true_condition"] for r in dxa_rows)
    dxa_targets = Counter(r["r10_targets"] for r in dxa_rows)
    tier1_rou = Counter(r["true_condition"] for r in ref if r["stratum"] == "serious_tier1" and r["decision"] == ROU)
    splits = table[(ESC, ROU)] + table[(ROU, ESC)]
    f_unc = sum(table[(UNC, b)] for b in DECISIONS)
    a_unc = sum(table[(a, UNC)] for a in DECISIONS)
    exc_conf4 = [rule for rule, v in c5["per_rule"].items() if v["candidate_patch"]]
    ben_esc = [r for r in ref if r["benchmark_class"] == "benign" and r["decision"] == ESC]
    summary = [
        "## Summary", "",
        f"1. **Criterion 1 passes: the rules' class agrees with the blind reference on {len(agree)} of {len(decided)} decided kept cases "
        f"({pct(len(agree), len(decided))}, Wilson lower bound {100 * lo:.1f}%).** The layer-a and UPGRADE rules hold on fresh cases: "
        f"every rule with 5 or more cases agrees on 100% of them (criterion 3 passes, no rule flagged), and the BENIGN stratum is "
        f"{pct(c2['benign']['agree'], c2['benign']['decided'])} ROUTINE.",
        f"2. **Criterion 2 fails on one stratum: SERIOUS by a DXA-only target is {pct(dxa['agree'], dxa['decided'])} ESCALATE against the 92% target.** "
        f"After the red-herring rule the stratum is " + ", ".join(f"{c} {n}" for c, n in dxa_truths.most_common()) + " with targets "
        + ", ".join(f"{t} {n}" for t, n in dxa_targets.most_common()) + f"; the reference keeps {dxa['decided'] - dxa['agree']} routine, because a "
        f"panic attack in a young adult with chronic anxiety and no cardiac risk factor is closed by an office ECG and vitals (HEART age 0; NICE CG113). "
        f"Section 7 says a failing stratum's rules go back to EXCLUDE and Phase 2 repeats on seed 20261004; this document changes nothing.",
        f"3. **The tier-1 stratum sits exactly on its 92% target ({pct(c2['serious_tier1']['agree'], c2['serious_tier1']['decided'])}).** "
        "The reference keeps routine " + ", ".join(f"{n} {c}" for c, n in tier1_rou.most_common()) + " cases: palpitations with light-headedness "
        "on caffeine, energy drinks, stimulants or decongestants, without chest pain or syncope, which the audit also kept routine (case 36). "
        "The condition table lists PSVT as INCLUDE; this is the one tier-1 condition the reference does not treat as an escalation by default.",
        f"4. **Criterion 4 fails: Cohen's kappa is {k:.3f} (raw agreement {100 * po:.1f}%), below 0.6.** The disagreement is almost entirely one "
        f"reviewer's UNCERTAIN against the other's decision (Fable {f_unc} UNCERTAIN, Astra {a_unc}); only {splits} of {len(ids)} cases are outright "
        f"ESCALATE/ROUTINE splits. The adjudication decided {len(queue_ids) - out['decisions'].get(UNC, 0)} of the {len(queue_ids)} queued cases; "
        f"{out['decisions'].get(UNC, 0)} stay UNCERTAIN.",
        f"5. **The EXCLUDED stratum is {pct(cnt[ESC], len(exc))} ESCALATE, {pct(cnt[ROU], len(exc))} ROUTINE, {pct(cnt[UNC], len(exc))} UNCERTAIN, "
        f"and the reviewers split on {pct(len(split), len(exc))}.** " + (
            "Exclusion rules whose cases are ESCALATE at confidence 4 or more in at least 80% of draws (candidate PATCH for a later phase): "
            + ", ".join(f"{r} ({c5['per_rule'][r]['escalate_conf4']} of {c5['per_rule'][r]['cases']})" for r in exc_conf4) + "."
            if exc_conf4 else "No exclusion rule reaches the 80% ESCALATE-at-confidence-4 mark.")
        + f" Among BENIGN cases the reference escalates {len(ben_esc)} (listed under the disagreements below): " + ", ".join(
            f"{r['true_condition']} at index {r['index']}" for r in ben_esc) + ".",
        "",
    ]
    L[4:4] = summary
    (P2 / "label_validation.md").write_text("\n".join(L))
    (P2 / "label_validation.json").write_text(json.dumps(out, indent=1) + "\n")
    print("\n".join(L[-9:-2]))
    print(f"wrote {P2 / 'label_validation.md'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("collect", "finalize"))
    ap.add_argument("--unblinded", default=str(P2 / "cases_unblinded.json"))
    args = ap.parse_args()
    (collect if args.stage == "collect" else finalize)(args)


if __name__ == "__main__":
    main()
