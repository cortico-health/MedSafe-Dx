"""Measurement ceiling from label noise.

Why: DDXPlus severity and an evidence-based severity reference
(spec/ddxplus_severity_reference.csv, column reference_level) agree only at
quadratic weighted kappa 0.77 (docs/ddxplus-severity-validation.md). When a
model agrees with our label about as well as two defensible labels agree with
each other, the label can no longer tell models apart. We measure that limit by
scoring an alternate label as if it were a model, at the same granularity as
the model metrics.

Label variants (config below, never hard-coded in callers):

    differential / <scale>  urgent iff the probability mass on diagnoses rated
                            <= SEVERE_MAX on <scale>, over the whole DDXPlus
                            differential, is >= T. "differential/ddxplus" is the
                            primary label in evaluator/triage_score.py.
    pathology / <scale>     urgent iff the case's true DDXPlus pathology is
                            rated <= SEVERE_MAX on <scale>.

The agreement code takes integer codes and an optional weighting, so the same
functions serve the binary label now and 5-level acuity with quadratic weighted
kappa in v0.2.
"""

from __future__ import annotations

import ast
import csv
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from evaluator import triage_score as ts

# ---------------------------------------------------------------- configuration
PRIMARY_LABEL = "differential/ddxplus"
CEILING_LABEL = "differential/reference"  # the alternate label scored as a model
LABEL_VARIANTS = {
    "differential/ddxplus": {"basis": "differential", "scale": "ddxplus"},
    "differential/reference": {"basis": "differential", "scale": "reference"},
    "pathology/ddxplus": {"basis": "pathology", "scale": "ddxplus"},
    "pathology/reference": {"basis": "pathology", "scale": "reference"},
}
# Inter-label agreement of the two severity scales on the 49 conditions, for the
# v0.2 ordinal display (docs/ddxplus-severity-validation.md, section 5).
ORDINAL_INTER_LABEL_QWK = 0.766
REFERENCE_CSV = "spec/ddxplus_severity_reference.csv"


# ---------------------------------------------------------------- severity scales


def load_scales(conditions_json: str | Path, reference_csv: str | Path) -> dict[str, dict[str, int]]:
    """Severity per DDXPlus condition name, on each scale."""
    cond = json.loads(Path(conditions_json).read_text())
    with open(reference_csv) as f:
        ref = {r["condition"]: int(r["reference_level"]) for r in csv.DictReader(f)}
    missing = set(cond) - set(ref)
    if missing:
        raise ValueError(f"{reference_csv} lacks {sorted(missing)}")
    return {"ddxplus": {k: int(v["severity"]) for k, v in cond.items()}, "reference": ref}


def load_case_differentials(case_ids: Iterable[str], ddx_csv: str | Path) -> dict[str, dict]:
    """{case_id: {"diff": [(condition, p), ...], "pathology": condition}}; ddxplus_N is CSV row N."""
    want = {int(cid.split("_")[1]): cid for cid in case_ids}
    out = {}
    with open(ddx_csv) as f:
        for i, row in enumerate(csv.DictReader(f)):
            cid = want.get(i)
            if cid is not None:
                out[cid] = {"diff": [(n, float(p)) for n, p in ast.literal_eval(row["DIFFERENTIAL_DIAGNOSIS"])],
                            "pathology": row["PATHOLOGY"]}
    if len(out) != len(want):
        raise ValueError(f"{len(want) - len(out)} case(s) not found in {ddx_csv}")
    return out


def p_severe(diff: list[tuple[str, float]], scale: Mapping[str, int], severe_max: int = ts.SEVERE_MAX) -> float:
    return float(sum(p for name, p in diff if scale[name] <= severe_max))


def build_labels(
    case_ids: list[str],
    cases: Mapping[str, dict],
    scales: Mapping[str, Mapping[str, int]],
    T: float = ts.URGENT_THRESHOLD,
    variants: Mapping[str, dict] = LABEL_VARIANTS,
) -> dict[str, np.ndarray]:
    """Binary urgent label per variant, in case_ids order."""
    out = {}
    for name, v in variants.items():
        scale = scales[v["scale"]]
        if v["basis"] == "differential":
            out[name] = np.array([p_severe(cases[c]["diff"], scale) >= T for c in case_ids])
        elif v["basis"] == "pathology":
            out[name] = np.array([scale[cases[c]["pathology"]] <= ts.SEVERE_MAX for c in case_ids])
        else:
            raise ValueError(f"unknown basis {v['basis']!r}")
    return out


# ---------------------------------------------------------------- agreement


def weight_matrix(k: int, weights: str | None) -> np.ndarray:
    """Disagreement weights: None = unweighted (0/1), 'linear' or 'quadratic' for ordinal levels."""
    i, j = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")
    if weights is None:
        return (i != j).astype(float)
    d = np.abs(i - j) / max(k - 1, 1)
    if weights == "linear":
        return d
    if weights == "quadratic":
        return d ** 2
    raise ValueError(f"unknown weights {weights!r}")


def cohen_kappa(a, b, n_levels: int = 2, weights: str | None = None) -> float:
    """Cohen's kappa between two integer-coded raters (codes 0..n_levels-1).

    Binary labels use weights=None. Ordinal acuity levels use weights='quadratic',
    which is the statistic behind the 0.77 inter-label figure.
    """
    return float(kappa_from_draws(np.asarray(a, int)[None], np.asarray(b, int)[None], n_levels, weights)[0])


def kappa_from_draws(A: np.ndarray, B: np.ndarray, n_levels: int = 2, weights: str | None = None) -> np.ndarray:
    """Kappa per row of two (draws x cases) integer arrays. Returns nan where expected disagreement is 0."""
    W = weight_matrix(n_levels, weights)
    k = n_levels
    n = A.shape[1]
    cells = A * k + B  # draws x cases
    conf = np.stack([np.bincount(r, minlength=k * k) for r in cells]).reshape(-1, k, k) / n
    pa, pb = conf.sum(axis=2), conf.sum(axis=1)
    expected = pa[:, :, None] * pb[:, None, :]
    obs_d = (conf * W).sum(axis=(1, 2))
    exp_d = (expected * W).sum(axis=(1, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(exp_d > 0, 1.0 - obs_d / exp_d, np.nan)


def bootstrap_kappa(
    reference: np.ndarray,
    raters: Mapping[str, np.ndarray],
    n_levels: int = 2,
    weights: str | None = None,
    n_boot: int = ts.N_BOOTSTRAP,
    seed: int = ts.BOOTSTRAP_SEED,
    level: float = 0.95,
) -> dict[str, dict]:
    """Kappa of each rater against ``reference`` with a paired case-bootstrap interval.

    The resampling indices match evaluator.triage_score.bootstrap (same seed and
    case count), so every interval on the board comes from the same draws.
    """
    ref = np.asarray(reference, int)
    n = len(ref)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    lo, hi = 100 * (1 - level) / 2, 100 * (1 + level) / 2
    out = {}
    for name, r in raters.items():
        r = np.asarray(r, int)
        draws = kappa_from_draws(r[idx], ref[idx], n_levels, weights)
        out[name] = {
            "kappa": cohen_kappa(r, ref, n_levels, weights),
            "ci": np.nanpercentile(draws, [lo, hi]).tolist(),
        }
    return out


def at_measurement_limit(model_ci: list[float], ceiling_ci: list[float]) -> bool:
    """A model is at the limit when its kappa interval reaches the ceiling's interval."""
    return model_ci[1] >= ceiling_ci[0]


def label_ceiling(
    p_primary: np.ndarray,
    labels: Mapping[str, np.ndarray],
    decisions: Mapping[str, np.ndarray],
    U: float = ts.TOLERATED_UNDER_TRIAGE,
    O: float = ts.TOLERATED_OVER_TRIAGE,
    T: float = ts.URGENT_THRESHOLD,
) -> dict:
    """Score every alternate label as a model against the primary label, plus kappas.

    Returns {"labels": {variant: {...}}, "models": {name: kappa + ci}, "ceiling": {...}}.
    """
    primary = ts.urgent_labels(p_primary, T)
    if not np.array_equal(primary, labels[PRIMARY_LABEL]):
        raise ValueError("labels[PRIMARY_LABEL] does not match the primary P(severe) >= T label")
    alt = {k: v.astype(int) for k, v in labels.items() if k != PRIMARY_LABEL}
    boot = ts.bootstrap(p_primary, alt, ranked=[], U=U, O=O, T=T)
    kap = bootstrap_kappa(primary.astype(int), {**alt, **decisions})
    out_labels = {}
    for k, e in alt.items():
        m = ts.score_model(p_primary, e, U, O, T)
        out_labels[k] = {
            "urgent_cases": int(e.sum()),
            "under": m.under, "over": m.over, "score": m.score,
            "ci": boot[k],
            "kappa": kap[k]["kappa"], "kappa_ci": kap[k]["ci"],
            "counts": ts.label_counts(e, primary),
        }
    c = out_labels[CEILING_LABEL]
    models = {}
    for k in decisions:
        models[k] = {**kap[k], "at_limit": at_measurement_limit(kap[k]["ci"], c["kappa_ci"])}
    return {"labels": out_labels, "models": models, "ceiling": {"variant": CEILING_LABEL, **c}}
