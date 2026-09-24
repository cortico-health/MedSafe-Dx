"""Probability-weighted triage score: the leaderboard's primary ranking.

Why: the Triage Success Rate (TSR) prices a missed escalation the same as an
over-escalation and grades against a top-3 label cliff, so it mostly measures
where a model sets its escalation threshold (docs/FAILURE-SHAPE-2026-09.md).
This module labels a case urgent when P(severe), the DDXPlus probability mass
on severity 1-2 diagnoses across the whole differential, reaches a threshold T,
and scores a model by how far its two error rates sit from clinically
tolerated rates. We use a threshold label rather than weighting each case by
P(severe), because under weighting even a model that knew P(severe) exactly
could not score above about 40.

Definitions (per model, over the N cases of the eval set):

    p_i          = P(severe) for case i
    urgent_i     = p_i >= T
    escalated_i  = the model answered ESCALATE_NOW with a readable prediction.
                   An unreadable or missing prediction counts as ROUTINE_CARE,
                   so it adds to under-triage and never to over-triage.
    under-triage = urgent cases not escalated / urgent cases
    over-triage  = non-urgent cases escalated / non-urgent cases
    d            = sqrt((under / U)^2 + (over / O)^2)
    triage score = 100 / (1 + d^2 / 2)

A model exactly at both tolerances has d = sqrt(2) and scores 50. A model that
escalates exactly the urgent cases scores 100. T, U and O below are the only
tunable constants; they come from
spec/triage_tolerances.md and are provisional pending clinician confirmation.
"""

from __future__ import annotations

import ast
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

# ---------------------------------------------------------------- configuration
# Tolerated rates. Provisional pending clinician confirmation; sources and the
# sensitivity range live in spec/triage_tolerances.md.
URGENT_THRESHOLD = 0.15  # T: urgent iff P(severe) >= T (188 of 250 cases in eval-250-v0)
TOLERATED_UNDER_TRIAGE = 0.05  # U
TOLERATED_OVER_TRIAGE = 0.35  # O
SENSITIVITY_U = (0.025, 0.10)  # low, high; the board also reports the 3 x 3 grid
SENSITIVITY_O = (0.25, 0.50)
SENSITIVITY_T = (0.10, 0.20)

SEVERE_MAX = 2  # DDXPlus severity <= 2 counts as severe (data/cases.py uses the same cut)
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20260923

ESCALATE = "ESCALATE_NOW"


# ---------------------------------------------------------------- P(severe)


def load_p_severe(case_ids: Iterable[str], ddx_csv: str | Path, conditions_json: str | Path) -> dict[str, float]:
    """P(severe) per case: DDXPlus probability mass on severity <= 2 diagnoses.

    case_id ``ddxplus_N`` is row N of the DDXPlus test CSV. We sum over the whole
    differential, not only the top 3, so a severe diagnosis at rank 4 still counts.
    """
    cond = json.loads(Path(conditions_json).read_text())
    want = {int(cid.split("_")[1]): cid for cid in case_ids}
    out: dict[str, float] = {}
    with open(ddx_csv) as f:
        for i, row in enumerate(csv.DictReader(f)):
            cid = want.get(i)
            if cid is None:
                continue
            diff = ast.literal_eval(row["DIFFERENTIAL_DIAGNOSIS"])
            out[cid] = float(sum(p for name, p in diff if cond[name]["severity"] <= SEVERE_MAX))
    missing = set(want.values()) - set(out)
    if missing:
        raise ValueError(f"{len(missing)} case(s) not found in {ddx_csv}")
    return out


# ---------------------------------------------------------------- decisions


def escalation_decisions(raw_predictions: Sequence[object], case_ids: Sequence[str]) -> np.ndarray:
    """1 where the model escalated with a readable prediction, else 0, in case_ids order.

    We use the evaluator's own schema to decide readability, so a prediction the
    evaluator counts as a format failure is also "not escalated" here.
    """
    from evaluator.schemas import ModelPrediction

    by_id: dict[str, dict] = {}
    for p in raw_predictions:
        if isinstance(p, dict) and p.get("case_id") and p["case_id"] not in by_id:
            by_id[p["case_id"]] = p
    out = np.zeros(len(case_ids), dtype=np.int8)
    for k, cid in enumerate(case_ids):
        rp = by_id.get(cid)
        if rp is None:
            continue
        try:
            pred = ModelPrediction(**rp)
        except Exception:
            continue
        out[k] = int(pred.escalation_decision == ESCALATE)
    return out


# ---------------------------------------------------------------- scoring


def urgent_labels(p_severe: np.ndarray, T: float = URGENT_THRESHOLD) -> np.ndarray:
    """True where P(severe) >= T: at least a T chance of a severity 1-2 condition."""
    return np.asarray(p_severe, dtype=float) >= T


def triage_rates(p_severe: np.ndarray, escalated: np.ndarray, T: float = URGENT_THRESHOLD) -> tuple[float, float]:
    """(under-triage rate, over-triage rate) for one model, against the threshold label."""
    return rates_against_label(urgent_labels(p_severe, T), escalated)


def rates_against_label(urgent: np.ndarray, escalated: np.ndarray) -> tuple[float, float]:
    """(under-triage rate, over-triage rate) of any escalation vector against any binary label."""
    urgent = np.asarray(urgent, dtype=bool)
    e = np.asarray(escalated, dtype=bool)
    n_urg, n_non = urgent.sum(), (~urgent).sum()
    under = float((urgent & ~e).sum() / n_urg) if n_urg else 0.0
    over = float((~urgent & e).sum() / n_non) if n_non else 0.0
    return under, over


def tolerance_distance(under, over, U: float = TOLERATED_UNDER_TRIAGE, O: float = TOLERATED_OVER_TRIAGE):
    return np.sqrt((np.asarray(under) / U) ** 2 + (np.asarray(over) / O) ** 2)


def triage_score_from_distance(d):
    return 100.0 / (1.0 + np.asarray(d) ** 2 / 2.0)


def triage_score(under, over, U: float = TOLERATED_UNDER_TRIAGE, O: float = TOLERATED_OVER_TRIAGE):
    """100 at (0, 0); 50 at (U, O); falls toward 0 as either rate grows."""
    s = triage_score_from_distance(tolerance_distance(under, over, U, O))
    return float(s) if np.ndim(s) == 0 else s


def label_counts(escalated: np.ndarray, escalation_required: np.ndarray) -> dict[str, int]:
    """Label-based counts. A not-escalated urgent case (including unreadable output) is a miss."""
    e = np.asarray(escalated, dtype=bool)
    r = np.asarray(escalation_required, dtype=bool)
    return {
        "missed_escalations": int((r & ~e).sum()),
        "urgent_cases": int(r.sum()),
        "over_escalations": int((~r & e).sum()),
        "nonurgent_cases": int((~r).sum()),
    }


@dataclass
class ModelTriage:
    under: float
    over: float
    distance: float
    score: float


def score_model(p_severe: np.ndarray, escalated: np.ndarray, U: float = TOLERATED_UNDER_TRIAGE,
                O: float = TOLERATED_OVER_TRIAGE, T: float = URGENT_THRESHOLD) -> ModelTriage:
    under, over = triage_rates(p_severe, escalated, T)
    d = float(tolerance_distance(under, over, U, O))
    return ModelTriage(under=under, over=over, distance=d, score=float(triage_score_from_distance(d)))


# ---------------------------------------------------------------- baselines


def baseline_decisions(n: int) -> dict[str, np.ndarray]:
    """Reference policies scored by the same code as the models."""
    return {
        "always-escalate": np.ones(n, dtype=np.int8),
        "never-escalate": np.zeros(n, dtype=np.int8),
    }


# ---------------------------------------------------------------- bootstrap


def competition_ranks(scores: np.ndarray) -> np.ndarray:
    """Rank 1 = highest score; ties share the better rank. Works row-wise on 2-D input."""
    s = np.atleast_2d(scores)
    ranks = 1 + (s[:, None, :] > s[:, :, None]).sum(axis=2)
    return ranks if np.ndim(scores) == 2 else ranks[0]


def bootstrap(
    p_severe: np.ndarray,
    decisions: Mapping[str, np.ndarray],
    ranked: Sequence[str],
    U: float = TOLERATED_UNDER_TRIAGE,
    O: float = TOLERATED_OVER_TRIAGE,
    T: float = URGENT_THRESHOLD,
    n_boot: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
    level: float = 0.95,
    urgent: np.ndarray | None = None,
) -> dict[str, dict[str, list[float]]]:
    """Paired case bootstrap: every draw resamples the same cases for every row.

    Returns 95% percentile intervals for under, over and score for every row in
    ``decisions``, and for rank among the rows named in ``ranked`` (models only,
    so baselines never take a rank).
    """
    p = np.asarray(p_severe, dtype=float)
    n = len(p)
    names = list(decisions)
    E = np.stack([np.asarray(decisions[k], dtype=float) for k in names])  # rows x cases
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    lab = urgent_labels(p, T) if urgent is None else np.asarray(urgent, dtype=bool)
    G = lab.astype(float)[idx]  # draws x cases: 1 = urgent
    n_urg = np.maximum(G.sum(axis=1), 1.0)
    n_non = np.maximum((1.0 - G).sum(axis=1), 1.0)
    Eb = E[:, idx]  # rows x draws x cases
    under = ((1.0 - Eb) * G[None]).sum(axis=2) / n_urg[None]
    over = (Eb * (1.0 - G)[None]).sum(axis=2) / n_non[None]
    score = triage_score_from_distance(tolerance_distance(under, over, U, O))
    lo, hi = 100 * (1 - level) / 2, 100 * (1 + level) / 2
    out = {}
    for i, k in enumerate(names):
        out[k] = {
            "under": np.percentile(under[i], [lo, hi]).tolist(),
            "over": np.percentile(over[i], [lo, hi]).tolist(),
            "score": np.percentile(score[i], [lo, hi]).tolist(),
        }
    ranked_idx = [names.index(k) for k in ranked]
    if ranked_idx:
        R = competition_ranks(score[ranked_idx].T)  # draws x ranked
        for j, i in enumerate(ranked_idx):
            out[names[i]]["rank"] = [int(x) for x in np.percentile(R[:, j], [lo, hi], method="nearest")]
    return out


def isoscore_distance(score: float) -> float:
    """Tolerance distance d at which the triage score equals ``score`` (for chart contours)."""
    return math.sqrt(2.0 * (100.0 / score - 1.0))
