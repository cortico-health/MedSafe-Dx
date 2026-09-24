"""
Reference rows for the v0.2 board (spec/v0.2-scoring.md section 8).

Each reference writes predictions in the model output format of section 3, so
evaluator/v02_score.py scores it with the same code as a model:

1. always-escalate: ESCALATE_NOW for everyone; no p_serious, no differential.
2. never-escalate: ROUTINE_CARE for everyone; no p_serious, no differential.
3. base-rate: p_serious = the sample share of serious patients (34%); it escalates
   everyone, because that share is above 12.5%.
4. dxa: p_serious = DXA's P(serious risk); escalates when it is >= 12.5%; its
   differential is DXA's top 5, each condition written as its DDXPlus ICD-10 code.
   It scores 100 on the headline by construction, so it is a reference for C, D1
   and E only.
5. naive-bayes: a naive-Bayes reader of the case's evidence codes, trained on the
   DDXPlus test split with the 470 sample rows held out (the train split is not in
   this checkout). Features are the presence and absence of each base evidence
   code; the prior is the training share of each condition. p_serious is its
   posterior mass on severity 1-2 conditions, it escalates at 12.5%, and its
   differential is its top 5. Labelled "dataset-knowledge ceiling, not a clinical target".

Always and never answer no p_serious, so C scores them at the base-rate forecast
(skill 0), and they carry no differential, so D1 and E score them as blank.
"""

from __future__ import annotations

import ast
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from evaluator import answer_key_v02 as ak
from evaluator.v02_score import T, T_SENSITIVITY, ConditionMap

ROOT = Path(__file__).resolve().parent.parent
DDX_CSV = ROOT / "data" / "ddxplus_v0" / "release_test_patients"
NB_CACHE = ROOT / "results" / "analysis" / "v02_references" / "nb_counts_holdout470.json"

LABELS = {
    "always-escalate": "Always escalate",
    "never-escalate": "Never escalate",
    "base-rate": "Base rate",
    "dxa": "DXA (DDXPlus's differential)",
    "naive-bayes": "Naive Bayes: dataset-knowledge ceiling, not a clinical target",
}


def _decision(p: float, t: float = T) -> str:
    return "ESCALATE_NOW" if p >= t else "ROUTINE_CARE"


def constant_rows(cases: Sequence[dict]) -> dict[str, list[dict]]:
    base = 100.0 * float(np.mean([c["serious"] for c in cases]))
    return {
        "always-escalate": [{"case_id": c["case_id"], "escalation_decision": "ESCALATE_NOW"} for c in cases],
        "never-escalate": [{"case_id": c["case_id"], "escalation_decision": "ROUTINE_CARE"} for c in cases],
        "base-rate": [{"case_id": c["case_id"], "p_serious": base, "escalation_decision": _decision(base)}
                      for c in cases],
    }


def dxa_rows(cases: Sequence[dict], cmap: ConditionMap, t: float = T) -> list[dict]:
    """DXA's answer. `t` is its escalation threshold: the at-risk threshold T, so the
    sensitivity rows at T = 5% and 25% re-derive it (spec section 6: "any T from 12.5%
    to 32.8% leaves the DXA reference unchanged")."""
    out = []
    for c in cases:
        diff = sorted(c["ddxplus_differential"], key=lambda x: -float(x[1]))[:5]
        out.append({
            "case_id": c["case_id"],
            "differential": [{"code": cmap.canonical[name], "p": 100.0 * float(p)} for name, p in diff],
            "p_serious": float(c["p_serious_risk"]),
            "escalation_decision": _decision(float(c["p_serious_risk"]), t),
        })
    return out


# ---------------------------------------------------------------- naive Bayes


def _codes(evidences) -> set[str]:
    return {str(e).split("_@_")[0] for e in evidences}


def train_counts(holdout_ids: set[str], csv_path: Path = DDX_CSV, cache: Path = NB_CACHE) -> dict:
    """Per-condition row counts and evidence-code counts over the test split minus the held-out rows."""
    if cache.exists():
        d = json.loads(cache.read_text())
        if set(d["holdout"]) == holdout_ids:
            return d
    hold = {int(h.split("_")[1]) for h in holdout_ids}
    n = Counter()
    codes: dict[str, Counter] = defaultdict(Counter)
    with open(csv_path) as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i in hold:
                continue
            c = row["PATHOLOGY"]
            n[c] += 1
            codes[c].update(_codes(ast.literal_eval(row["EVIDENCES"])))
    d = {"holdout": sorted(holdout_ids), "n": dict(n), "codes": {k: dict(v) for k, v in codes.items()},
         "source": str(csv_path.relative_to(ROOT)), "rows": int(sum(n.values()))}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(d))
    return d


class NaiveBayes:
    def __init__(self, counts: Mapping):
        self.conditions = sorted(counts["n"])
        self.vocab = sorted({q for v in counts["codes"].values() for q in v})
        vi = {q: j for j, q in enumerate(self.vocab)}
        n = np.array([counts["n"][c] for c in self.conditions], float)
        K = np.zeros((len(self.conditions), len(self.vocab)))
        for i, c in enumerate(self.conditions):
            for q, k in counts["codes"].get(c, {}).items():
                K[i, vi[q]] = k
        L = (K + 0.5) / (n[:, None] + 1.0)
        self.log1, self.log0 = np.log(L), np.log(1.0 - L)
        self.logprior = np.log(n / n.sum())
        self.vi = vi

    def posterior(self, evidences) -> np.ndarray:
        x = np.zeros(len(self.vocab))
        for q in _codes(evidences):
            j = self.vi.get(q)
            if j is not None:
                x[j] = 1.0
        lp = self.logprior + self.log1 @ x + self.log0 @ (1.0 - x)
        lp -= lp.max()
        p = np.exp(lp)
        return p / p.sum()


def nb_rows(cases: Sequence[dict], cmap: ConditionMap, conditions: Mapping[str, dict] | None = None) -> tuple[list[dict], dict]:
    conditions = conditions or ak.load_conditions()
    counts = train_counts({c["case_id"] for c in cases})
    nb = NaiveBayes(counts)
    serious = np.array([ak.is_serious(c, conditions) for c in nb.conditions])
    out = []
    for c in cases:
        p = nb.posterior(c["presenting_symptoms"])
        top = np.argsort(-p)[:5]
        ps = 100.0 * float(p[serious].sum())
        out.append({
            "case_id": c["case_id"],
            "differential": [{"code": cmap.canonical[nb.conditions[j]], "p": round(100.0 * float(p[j]), 4)} for j in top],
            "p_serious": ps,
            "escalation_decision": _decision(ps),
        })
    meta = {"training": f"DDXPlus test split ({counts['source']}), {counts['rows']} rows, the 470 sample rows held out",
            "features": "presence and absence of each base evidence code", "prior": "training share of each condition"}
    return out, meta


def reference_rows(cases: Sequence[dict], cmap: ConditionMap | None = None, with_nb: bool = True) -> dict[str, dict]:
    """{name: {"predictions", "kind": "reference", "label", ...}} for every section 8 reference."""
    cmap = cmap or ConditionMap()
    rows = {k: {"predictions": v} for k, v in constant_rows(cases).items()}
    rows["dxa"] = {"predictions": dxa_rows(cases, cmap),
                   "sensitivity_predictions": {t: dxa_rows(cases, cmap, t) for t in T_SENSITIVITY}}
    if with_nb:
        preds, meta = nb_rows(cases, cmap)
        rows["naive-bayes"] = {"predictions": preds, "method": meta}
    for k, v in rows.items():
        v.update({"kind": "reference", "label": LABELS[k]})
    return rows
