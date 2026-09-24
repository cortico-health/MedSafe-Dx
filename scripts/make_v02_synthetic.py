#!/usr/bin/env python3
"""Write SYNTHETIC v0.2 prediction files, so the scorer and board can be seen working with no API spend.

No model is called. Each simulated model answers the 470-case sample from the
answer key plus seeded noise, in the section 3 output format. Every file is
named SYNTHETIC-*.json and carries "synthetic": true in its metadata, and the
board labels the rows SYNTHETIC, because nobody may mistake them for results.

Simulated models (escalation probability by answer-key group; top-1 accuracy):

| Name | P(esc) serious | P(esc) at risk | P(esc) clearly low-risk | Top-1 | Other behaviour |
|---|---|---|---|---|---|
| careful | 0.995 | 0.90 | 0.04 | 0.80 | top p 40-65 |
| cold-escalator | 1.00 | 0.98 | 0.65 | 0.75 | |
| misser | 0.90 | 0.60 | 0.03 | 0.70 | |
| overconfident | 0.98 | 0.90 | 0.05 | 0.60 | top p 80-97, wrong answers taken from DXA's differential |
| format-failing | 0.98 | 0.90 | 0.05 | 0.78 | 5% of cases unreadable |
| real-world-prior | 0.99 (0.60 on rare serious conditions) | 0.90 | 0.04 | 0.80 | under-calls conditions NHAMCS barely sees |

Usage:
    python3 scripts/make_v02_synthetic.py            # writes results/v02/synthetic/SYNTHETIC-*.json
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluator.v02_score import ConditionMap, load_sample  # noqa: E402

OUT = ROOT / "results" / "v02" / "synthetic"
SEED = 20260923
OFFLIST_CODES = ["R07.9", "R06.02", "K92.2", "I50.9", "R51.9", "I71.00"]

MODELS = {
    "careful": dict(esc=(0.995, 0.90, 0.04), top1=0.80, top_p=(40, 65),
                    about="escalates at-risk and serious patients, with small noise"),
    "cold-escalator": dict(esc=(1.00, 0.98, 0.65), top1=0.75, top_p=(40, 65),
                           about="also escalates many clearly low-risk patients"),
    "misser": dict(esc=(0.90, 0.60, 0.03), top1=0.70, top_p=(40, 65),
                   about="routes some serious patients to routine care"),
    "overconfident": dict(esc=(0.98, 0.90, 0.05), top1=0.60, top_p=(80, 97), wrong_from_dxa=True,
                          about="states a top diagnosis at 80-97% and is often wrong"),
    "format-failing": dict(esc=(0.98, 0.90, 0.05), top1=0.78, top_p=(40, 65), unreadable=0.05,
                           about="careful, but 5% of outputs are unreadable"),
    "real-world-prior": dict(esc=(0.99, 0.90, 0.04), top1=0.80, top_p=(40, 65), rare_serious_esc=0.60,
                             about="careful on common conditions, under-calls serious conditions that US EDs rarely see"),
}


def narrower_codes(cmap: ConditionMap) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    with open(ROOT / "spec/ddxplus_icd10_map.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["relation"] == "narrower":
                out.setdefault(r["condition"], []).append(r["code"])
    return out


def rare_conditions() -> set[str]:
    with open(ROOT / "evaluator/data/v02_condition_mix.csv", newline="") as f:
        return {r["condition"] for r in csv.DictReader(f) if r["nhamcs_covered"] != "1"}


def simulate(name: str, cfg: dict, cases: list[dict], cmap: ConditionMap, rng: np.random.Generator) -> list[dict]:
    narrow = narrower_codes(cmap)
    rare = rare_conditions()

    def code_for(cond: str) -> str:
        if narrow.get(cond) and rng.random() < 0.2:
            return str(rng.choice(narrow[cond]))
        return cmap.canonical[cond]

    preds = []
    for c in cases:
        truth = c["true_pathology"]
        if c["serious"]:
            pe = cfg["esc"][0]
            if cfg.get("rare_serious_esc") and truth in rare:
                pe = cfg["rare_serious_esc"]
        elif c["at_risk"]:
            pe = cfg["esc"][1]
        else:
            pe = cfg["esc"][2]
        escalate = rng.random() < pe
        # p_serious agrees with the decision 97% of the time (F measures the rest).
        consistent = rng.random() < 0.97
        high = escalate == consistent
        if high:
            ps = float(rng.uniform(30, 90) if c["serious"] else rng.uniform(13, 60))
        else:
            ps = float(rng.uniform(1, 12))
        # Differential: top diagnosis right with probability top1, else a DXA alternative or an off-list code.
        others = [n for n, _ in sorted(c["ddxplus_differential"], key=lambda x: -x[1]) if n != truth]
        right = rng.random() < cfg["top1"]
        if right:
            top = code_for(truth)
        elif cfg.get("wrong_from_dxa") and others:
            top = code_for(others[0])
        elif rng.random() < 0.25 or not others:
            top = str(rng.choice(OFFLIST_CODES))
        else:
            top = code_for(others[int(rng.integers(0, min(3, len(others))))])
        rest = [code_for(n) for n in others[:6]]
        if not right and rng.random() < 0.8:
            rest.insert(int(rng.integers(0, 3)), code_for(truth))
        codes = [top] + [x for x in rest if x != top][:4]
        top_p = float(rng.integers(cfg["top_p"][0], cfg["top_p"][1] + 1))
        remaining = (100 - top_p) * 0.9
        shares = np.array([0.4, 0.3, 0.2, 0.1])[: len(codes) - 1]
        ps_rest = (remaining * shares / shares.sum()).round().tolist() if len(codes) > 1 else []
        pred = {
            "case_id": c["case_id"],
            "differential": [{"code": k, "p": p} for k, p in zip(codes, [top_p] + ps_rest)],
            "p_serious": round(ps),
            "escalation_decision": "ESCALATE_NOW" if escalate else "ROUTINE_CARE",
        }
        if cfg.get("unreadable") and rng.random() < cfg["unreadable"]:
            if rng.random() < 0.5:
                pred = {"case_id": c["case_id"], "error": "json_parse_failure", "raw_response": "{\"differential\": [ ..."}
            else:
                pred.pop("escalation_decision")
        pred["raw_response"] = pred.get("raw_response", "SYNTHETIC: no model was called")
        preds.append(pred)
    return preds


def main() -> None:
    cases = load_sample()
    cmap = ConditionMap()
    OUT.mkdir(parents=True, exist_ok=True)
    for k, (name, cfg) in enumerate(MODELS.items()):
        rng = np.random.default_rng(SEED + k)
        preds = simulate(name, cfg, cases, cmap, rng)
        meta = {
            "model": f"SYNTHETIC-{name}", "synthetic": True, "description": cfg["about"],
            "note": "SYNTHETIC preview row generated by scripts/make_v02_synthetic.py; no model was called.",
            "seed": SEED + k, "prompt_version": "v5 (simulated)", "total_cases": len(preds),
        }
        path = OUT / f"SYNTHETIC-{name}.json"
        path.write_text(json.dumps({"metadata": meta, "predictions": preds}, indent=1))
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
