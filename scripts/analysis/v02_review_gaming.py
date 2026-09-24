#!/usr/bin/env python3
"""Gaming simulations for the v0.2 scoring design (spec/v0.2-scoring.md).

We take the DXA differential of each sampled case as the best available
state of knowledge and score, with the spec's own definitions, an honest
policy next to policies that game one measure at a time:

  honest          a = Bayes action under the section 5 costs; acuity
                  probabilities = DXA mass per level; differential = DXA top 5
                  with DXA probabilities; escalate iff a <= 2.
  always-2        a = 2 and ESCALATE_NOW on every case, everything else honest.
  cap-79          honest, but the top diagnosis's p is min(p, 79).
  pad-severe      honest top diagnosis, then four fixed severe conditions at
                  p = 1 each in slots 2-5.
  gamed           always-2 + cap-79 + pad-severe together.
  hedge-acuity    honest, but acuity probabilities = 20 each.
  always-1        the spec's own reference: a = 1, p1 = 100.

We also score alternative headline definitions that a constant policy cannot
win: quadratic weighted kappa, and a graded asymmetric level cost (under by
k levels costs 7k, over by k costs k). Output: stdout and
results/analysis/v02_review/gaming.json. Nothing in the evaluator changes.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "analysis"))
import v02_review_policies as vp  # noqa: E402

OUT = REPO / "results" / "analysis" / "v02_review" / "gaming.json"
LEVELS = vp.LEVELS
PAD = ["Possible NSTEMI / STEMI", "Anaphylaxis", "Pulmonary embolism", "Spontaneous pneumothorax"]


def qwk(a: np.ndarray, t: np.ndarray, k: int = 5) -> float:
    a = np.asarray(a) - 1
    t = np.asarray(t) - 1
    conf = np.zeros((k, k))
    for x, y in zip(a, t):
        conf[x, y] += 1
    conf /= conf.sum()
    i, j = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")
    W = ((i - j) / (k - 1)) ** 2
    exp = np.outer(conf.sum(1), conf.sum(0))
    return float(1 - (W * conf).sum() / (W * exp).sum())


def graded_cost(a: np.ndarray, t: np.ndarray) -> float:
    """Mean per-case cost: under-triage by k levels costs 7k, over-triage by k levels costs k."""
    d = np.asarray(a) - np.asarray(t)
    return float(np.mean(np.where(d > 0, 7 * d, -d)))


def score(policy: dict, t: np.ndarray, scale: dict[str, int], truth: list[str]) -> dict:
    a = policy["a"]
    n = len(t)
    A = float(np.mean((a - t) >= 2))
    B = float(np.mean((t - a) >= 2))
    y = (t <= 2).astype(float)
    p12 = policy["acuity"][:, :2].sum(axis=1)
    brier = float(np.mean((p12 - y) ** 2))
    base = float(np.mean((y.mean() - y) ** 2))
    top_name, top_p = policy["top"]
    correct = np.array([tn == tr for tn, tr in zip(top_name, truth)], float)
    d1 = float(np.mean((np.asarray(top_p) - correct) ** 2))
    top_sev = np.array([scale[tn] for tn in top_name])
    d2 = float(np.mean((np.asarray(top_p) >= 0.8) & (correct == 0) & (np.abs(top_sev - t) >= 2)))
    in5 = np.array([tr in lst for lst, tr in zip(policy["top5"], truth)])
    e = float(np.mean(~in5[t <= 2]))
    esc = policy["escalate"]
    f = float(np.mean((esc & (a >= 4)) | (~esc & (a <= 2))))
    A_strict = float(np.mean((t <= 2) & (a >= 3)))  # urgent patient not given same-day care
    B_strict = float(np.mean((t >= 3) & (a <= 2)))  # non-urgent patient given same-day care
    return {"A": 100 * A, "B": 100 * B, "cost_7A+B": 100 * (7 * A + B),
            "triage_score(U=5,O=35)": float(vp_score(A, B)),
            "A_strict": 100 * A_strict, "B_strict": 100 * B_strict,
            "strict_triage_score": float(vp_score(A_strict, B_strict)),
            "C_brier": brier, "C_skill": 1 - brier / base, "D1_brier": d1, "D2": 100 * d2, "E": 100 * e, "F": 100 * f,
            "QWK": qwk(a, t), "graded_cost": graded_cost(a, t),
            "level_shares": {int(L): round(float(np.mean(a == L)), 2) for L in LEVELS}}


def vp_score(A: float, B: float, U: float = 0.05, O: float = 0.35) -> float:
    d2 = (A / U) ** 2 + (B / O) ** 2
    return 100.0 / (1.0 + d2 / 2.0)


def main() -> None:
    scales = vp.load_scales()
    rows = vp.read_rows()
    rng = np.random.default_rng(vp.SEED)
    sample = vp.stratified(rows, rng, adults_only=False)
    truth = [r["path"] for r in sample]
    diffs = [ast.literal_eval(r["diff"]) for r in sample]
    results = {}
    for scale_name, scale in scales.items():
        t = np.array([scale[p] for p in truth])
        dists = np.array([vp.sev_dist_from_diff(r["diff"], scale) for r in sample])
        bayes = np.array([int(LEVELS[np.argmin([vp.cost_of_level(a, d) for a in LEVELS])]) for d in dists])
        top_name = [d[0][0] for d in diffs]
        top_p = np.array([float(d[0][1]) for d in diffs])
        top5 = [[n for n, _ in d[:5]] for d in diffs]
        padded = [[d[0][0]] + [c for c in PAD if c != d[0][0]][:4] for d in diffs]
        n = len(sample)
        honest = {"a": bayes, "acuity": dists, "top": (top_name, top_p), "top5": top5, "escalate": bayes <= 2}
        pols = {
            "honest (DXA Bayes action)": honest,
            "always-2, escalate all": {**honest, "a": np.full(n, 2), "escalate": np.ones(n, bool)},
            "cap top p at 79": {**honest, "top": (top_name, np.minimum(top_p, 0.79))},
            "pad slots 2-5 with 4 severe codes": {**honest, "top5": padded},
            "gamed (all three)": {**honest, "a": np.full(n, 2), "escalate": np.ones(n, bool),
                                  "top": (top_name, np.minimum(top_p, 0.79)), "top5": padded},
            "hedge acuity probabilities (20 each)": {**honest, "acuity": np.full((n, 5), 0.2)},
            "always-1 (spec reference)": {**honest, "a": np.ones(n, int), "acuity": np.eye(5)[np.zeros(n, int)],
                                         "escalate": np.ones(n, bool)},
            "always-3": {**honest, "a": np.full(n, 3), "escalate": np.zeros(n, bool)},
            "DXA argmax level": {**honest, "a": np.argmax(dists, axis=1) + 1,
                                 "escalate": (np.argmax(dists, axis=1) + 1) <= 2},
            "perfect (a = t)": {**honest, "a": t, "escalate": t <= 2, "acuity": np.eye(5)[t - 1]},
            "DXA top 3 only": {**honest, "top5": [l[:3] for l in top5]},
            "DXA top 3 + 2 severe pads from its tail": {**honest, "top5": [
                l[:3] + [n for n, _ in d[3:] if scale[n] <= 2][:2] for l, d in zip(top5, diffs)]},
        }
        results[scale_name] = {k: score(v, t, scale, truth) for k, v in pols.items()}
        print(f"\n[{scale_name} severity, stratified sample n={n}]")
        hdr = ["A", "B", "cost_7A+B", "triage_score(U=5,O=35)", "A_strict", "B_strict", "strict_triage_score",
               "C_skill", "D1_brier", "D2", "E", "F", "QWK", "graded_cost"]
        print(f"  {'policy':42s} " + " ".join(f"{h[:8]:>8s}" for h in hdr))
        for k, v in results[scale_name].items():
            print(f"  {k:42s} " + " ".join(f"{v[h]:8.2f}" for h in hdr))
        print("  level shares:", {k: v["level_shares"] for k, v in results[scale_name].items()
                                  if k in ("honest (DXA Bayes action)", "DXA argmax level")})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
