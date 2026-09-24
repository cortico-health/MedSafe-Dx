#!/usr/bin/env python3
"""Measure what the condition-level ICD-10 map (spec/ddxplus_icd10_map.csv)
changes, against the codes models actually emit.

We tally every code in every leaderboard prediction file, then score each
model's top-3 and top-5 under four matchers for the true DDXPlus pathology:

  prefix    the current evaluator rule (evaluator.icd10.icd10_prefix_match):
            the model code and the DDXPlus code are prefixes of each other.
  category  what MetricsAccumulator.top3_hits counts: prefix, or the same
            3-character category.
  map       the new map: the model code resolves, by longest matching map
            prefix, to an equivalent or narrower code of the true condition.
  map+broad map, plus broader codes (reported, never counted).

Off-list codes (no equivalent, narrower or broader mapping to any of the 49
conditions) are grouped by spec/ddxplus_offlist_categories.csv for reporting.
No severity is assigned to them.

Outputs go to results/analysis/icd10_map/. The evaluator is not touched.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from evaluator.icd10 import explode_icd10_codes, icd10_prefix_match, normalize_icd10  # noqa: E402

MAP_CSV = REPO / "spec" / "ddxplus_icd10_map.csv"
OFFLIST_CSV = REPO / "spec" / "ddxplus_offlist_categories.csv"
CONDITIONS = REPO / "data" / "ddxplus_v0" / "release_conditions.json"
PER_CASE = REPO / "results" / "analysis" / "failure_shape" / "per_case.csv"
LEADERBOARD = REPO / "leaderboard"
OUT = REPO / "results" / "analysis" / "icd10_map"
COUNTED = {"equivalent", "narrower"}


def norm(code: str) -> str:
    return normalize_icd10(code).upper()


class ConditionMap:
    """Longest-prefix lookup from a model code to (condition, relation)."""

    def __init__(self, path: Path):
        self.entries: dict[str, dict[str, str]] = defaultdict(dict)  # cond -> norm code -> relation
        self.owner: dict[str, str] = {}
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                c = norm(row["code"])
                self.entries[row["condition"]][c] = row["relation"]
                if row["relation"] in COUNTED:
                    prev = self.owner.get(c)
                    if prev and prev != row["condition"]:
                        raise SystemExit(f"map ownership conflict: {c} counted for {prev} and {row['condition']}")
                    self.owner[c] = row["condition"]
        self.conditions = list(self.entries)

    def relation(self, code: str, condition: str) -> str | None:
        """Relation of `code` to `condition` by longest map prefix, or None."""
        c = norm(code)
        best = None
        for mcode, rel in self.entries[condition].items():
            if c.startswith(mcode) and (best is None or len(mcode) > len(best[0])):
                best = (mcode, rel)
        return best[1] if best else None

    def resolve(self, code: str) -> dict[str, str]:
        """All conditions this code maps to, with relation."""
        return {cond: rel for cond in self.conditions if (rel := self.relation(code, cond))}


class OffList:
    def __init__(self, path: Path):
        self.rows: list[tuple[str, str]] = []
        if path.exists():
            with open(path, newline="", encoding="utf-8") as f:
                self.rows = [(norm(r["code_prefix"]), r["category"]) for r in csv.DictReader(f)]

    def category(self, code: str) -> str:
        c = norm(code)
        best = None
        for prefix, cat in self.rows:
            if c.startswith(prefix) and (best is None or len(prefix) > len(best[0])):
                best = (prefix, cat)
        return best[1] if best else "other"


def load_predictions() -> list[tuple[str, str, list[dict]]]:
    """(model, path, predictions) for every leaderboard eval file whose predictions exist."""
    out = []
    for f in sorted(LEADERBOARD.glob("*-eval.json")):
        meta = json.load(open(f))
        p = meta.get("predictions_path")
        if not p:
            continue
        cands = [REPO / p, LEADERBOARD / Path(p).name]
        path = next((c for c in cands if c.exists()), None)
        if path is None:
            out.append((meta["model"], p, []))
            continue
        out.append((meta["model"], str(path.relative_to(REPO)), json.load(open(path))["predictions"]))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cmap = ConditionMap(MAP_CSV)
    offlist = OffList(OFFLIST_CSV)
    conds = json.load(open(CONDITIONS))
    ddx_codes = {name: explode_icd10_codes([v["icd10-id"]]) for name, v in conds.items()}
    ddx_cats = {name: {c[:3] for c in codes} for name, codes in ddx_codes.items()}
    severity = {name: int(v["severity"]) for name, v in conds.items()}
    all_ddx = set().union(*ddx_codes.values())
    per_case = {r["case_id"]: r for r in csv.DictReader(open(PER_CASE))}

    def matchers(code: str, truth: str) -> dict[str, bool]:
        n = normalize_icd10(code)
        gold = ddx_codes[truth]
        prefix = icd10_prefix_match(n, gold)
        rel = cmap.relation(code, truth)
        return {
            "prefix": prefix,
            "category": prefix or (n[:3] in ddx_cats[truth]),
            "map": rel in COUNTED,
            "map+broad": rel in COUNTED or rel == "broader",
        }

    tally: Counter[str] = Counter()
    per_model_rows = []
    per_cond = defaultdict(lambda: Counter())
    skipped = []
    models_used = 0
    for model, path, preds in load_predictions():
        if not preds:
            skipped.append((model, path))
            continue
        models_used += 1
        stats = Counter()
        for r in preds:
            dx = r.get("differential_diagnoses") or []
            if not dx or r["case_id"] not in per_case:
                continue
            truth = per_case[r["case_id"]]["pathology"]
            sev = severity[truth]
            codes = [d.get("code") or "" for d in dx[:5]]
            for c in codes:
                tally[c.strip().upper()] += 1
            stats["cases"] += 1
            if sev <= 2:
                stats["severe_cases"] += 1
            for k in (3, 5):
                hit = defaultdict(bool)
                for c in codes[:k]:
                    for name, ok in matchers(c, truth).items():
                        hit[name] |= ok
                for name, ok in hit.items():
                    if ok:
                        stats[f"top{k}_{name}"] += 1
                        if sev <= 2:
                            stats[f"severe_top{k}_{name}"] += 1
                        per_cond[truth][f"top{k}_{name}"] += 1
            per_cond[truth]["cases"] += 1
        row = {"model": model, "cases": stats["cases"], "severe_cases": stats["severe_cases"]}
        for k in (3, 5):
            for name in ("prefix", "category", "map", "map+broad"):
                row[f"top{k}_{name}"] = round(stats[f"top{k}_{name}"] / stats["cases"], 4)
                row[f"severe_top{k}_{name}"] = round(stats[f"severe_top{k}_{name}"] / max(stats["severe_cases"], 1), 4)
        per_model_rows.append(row)

    # Coverage of every emitted code.
    total = sum(tally.values())
    cov = Counter()
    code_rows = []
    unmapped: Counter[str] = Counter()
    offlist_tally: Counter[str] = Counter()
    for code, n in tally.most_common():
        nn = normalize_icd10(code)
        res = cmap.resolve(code)
        counted = [c for c, r in res.items() if r in COUNTED]
        broad = [c for c, r in res.items() if r == "broader"]
        related = [c for c, r in res.items() if r == "related"]
        prefix_any = icd10_prefix_match(nn, all_ddx)
        cat_any = prefix_any or any(nn[:3] in cats for cats in ddx_cats.values())
        if prefix_any:
            cov["prefix"] += n
        if cat_any:
            cov["category"] += n
        if counted:
            cov["map"] += n
        if counted or broad:
            cov["map+broad"] += n
        if not counted and not broad:
            unmapped[code] += n
            offlist_tally[offlist.category(code)] += n
        code_rows.append({
            "code": code, "count": n,
            "prefix_match": prefix_any, "category_match": cat_any,
            "map_condition": counted[0] if counted else "",
            "map_relation": res[counted[0]] if counted else "",
            "broader_for": "; ".join(broad), "related_to": "; ".join(related),
            "offlist_category": offlist.category(code) if not counted and not broad else "",
        })

    with open(OUT / "code_tally.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(code_rows[0]))
        w.writeheader(); w.writerows(code_rows)
    with open(OUT / "unmapped_codes.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["code", "count", "offlist_category"])
        for code, n in unmapped.most_common():
            w.writerow([code, n, offlist.category(code)])
    with open(OUT / "offlist_categories_tally.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["category", "count", "share_of_all_codes"])
        for cat, n in offlist_tally.most_common():
            w.writerow([cat, n, round(n / total, 4)])
    with open(OUT / "per_model_topk.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_model_rows[0]))
        w.writeheader(); w.writerows(per_model_rows)
    cond_rows = []
    for name, c in sorted(per_cond.items(), key=lambda kv: -(kv[1]["top5_map"] - kv[1]["top5_prefix"])):
        row = {"condition": name, "ddxplus_icd10": conds[name]["icd10-id"], "severity": severity[name],
               "model_cases": c["cases"]}
        for k in (3, 5):
            for m in ("prefix", "category", "map", "map+broad"):
                row[f"top{k}_{m}"] = round(c[f"top{k}_{m}"] / c["cases"], 4)
        row["top5_gain_vs_prefix"] = round(row["top5_map"] - row["top5_prefix"], 4)
        cond_rows.append(row)
    with open(OUT / "per_condition_topk.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cond_rows[0]))
        w.writeheader(); w.writerows(cond_rows)

    summary = {
        "models_used": models_used,
        "models_skipped_missing_predictions": [m for m, _ in skipped],
        "total_codes": total, "distinct_codes": len(tally),
        "coverage": {k: {"count": cov[k], "share": round(cov[k] / total, 4)} for k in ("prefix", "category", "map", "map+broad")},
        "unmapped_top30": unmapped.most_common(30),
        "offlist_categories": offlist_tally.most_common(),
        "mean_topk": {
            name: round(sum(r[name] for r in per_model_rows) / len(per_model_rows), 4)
            for name in per_model_rows[0] if name.startswith(("top", "severe_top"))
        },
    }
    json.dump(summary, open(OUT / "coverage_summary.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "unmapped_top30"}, indent=1))
    print("unmapped top 30:", unmapped.most_common(30))


if __name__ == "__main__":
    main()
