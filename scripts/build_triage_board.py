"""Build leaderboard/triage-scores.json: the triage score, rates and intervals per row.

The web board ranks by the probability-weighted triage score in
evaluator/triage_score.py. This script computes it from stored predictions (no
inference) and writes one JSON file that web/main.py serves beside the eval rows.

Usage:
    .venv/bin/python scripts/build_triage_board.py

Rules:
1. We score a row only from the prediction file whose sha256 matches its eval
   JSON, so the score and the published counts describe the same answers. A
   mismatch stops the build.
2. Rows with no per-case predictions get label-based counts only, marked
   "per-case data unavailable".
3. Sonnet 4.6 is scored from its N=500 run restricted to these 250 cases and
   marked as a proxy, because its 250-case file is lost but the proxy reproduces
   the published missed (11) and over-escalation (63) counts exactly.
"""

from __future__ import annotations

import glob
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluator import label_ceiling as lc  # noqa: E402
from evaluator import triage_score as ts  # noqa: E402

CASES_PATH = ROOT / "data/test_sets/eval-250-v0.json"
DDX_CSV = ROOT / "data/ddxplus_v0/release_test_patients"
COND_PATH = ROOT / "data/ddxplus_v0/release_conditions.json"
OUT = ROOT / "leaderboard/triage-scores.json"

# Rows whose eval JSON points at a file that is gone: (path, status, note).
FALLBACK = {
    "anthropic-claude-haiku-4.5": ("leaderboard/anthropic-claude-haiku-4.5-250cases.json", "scored", None),
    "anthropic-claude-sonnet-4.6": (
        "results/artifacts/anthropic-claude-sonnet-4.6-500cases.json",
        "proxy",
        "Proxy: scored from the N=500 run (2026-03) restricted to these 250 cases. "
        "It reproduces the published missed (11) and over-escalation (63) counts.",
    ),
}

BASELINES = {
    "always-escalate": "Always escalate",
    "never-escalate": "Never escalate",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(ev: dict) -> tuple[Path | None, str, str | None]:
    p = ROOT / ev["predictions_path"]
    if p.exists():
        if sha256(p) != ev["predictions_sha256"]:
            raise SystemExit(
                f"{ev['model']}: {p} does not match the sha256 in its eval JSON; rescore it before building the board"
            )
        return p, "scored", None
    fb = FALLBACK.get(ev["model"])
    if fb:
        path = ROOT / fb[0]
        if fb[1] == "scored" and sha256(path) != ev["predictions_sha256"]:
            raise SystemExit(f"{ev['model']}: fallback {path} does not match the eval JSON sha256")
        return path, fb[1], fb[2]
    return None, "unavailable", "Per-case data unavailable: no prediction file on disk or in git history."


def main() -> None:
    cases = json.loads(CASES_PATH.read_text())["cases"]
    case_ids = [c["case_id"] for c in cases]
    required = np.array([bool(c["escalation_required"]) for c in cases])
    p_map = ts.load_p_severe(case_ids, DDX_CSV, COND_PATH)
    p = np.array([p_map[c] for c in case_ids])
    urgent = ts.urgent_labels(p)

    rows, decisions = {}, {}
    for f in sorted(glob.glob(str(ROOT / "leaderboard/*-eval.json"))):
        ev = json.loads(Path(f).read_text())
        path, status, note = resolve(ev)
        hw = ev.get("harm_weights") or {}
        missed_incl_unreadable = round(
            (ev.get("expected_harm_breakdown_total") or {}).get("missed_escalation", 0.0)
            / float(hw.get("missed_escalation") or 100.0)
        )
        row = {
            "status": status,
            "note": note,
            "label": {
                "missed_escalations": missed_incl_unreadable,
                "urgent_cases": int(required.sum()),
                "over_escalations": int(ev["effectiveness"]["over_escalation"]),
                "nonurgent_cases": int((~required).sum()),
            },
        }
        if path is not None:
            raw = json.loads(path.read_text())
            preds = raw["predictions"] if isinstance(raw, dict) else raw
            e = ts.escalation_decisions(preds, case_ids)
            decisions[ev["model"]] = e
            lab = ts.label_counts(e, required)
            if status == "scored" and (
                lab["missed_escalations"] != missed_incl_unreadable
                or lab["over_escalations"] != row["label"]["over_escalations"]
            ):
                raise SystemExit(f"{ev['model']}: per-case counts {lab} disagree with the eval JSON {row['label']}")
            row["label"] = lab
            row["threshold"] = ts.label_counts(e, urgent)
            row["predictions_file"] = str(path.relative_to(ROOT))
            row["escalate_rate"] = float(e.mean())
            m = ts.score_model(p, e)
            row.update(under=m.under, over=m.over, distance=m.distance, score=m.score)
        rows[ev["model"]] = row

    for key, e in ts.baseline_decisions(len(p)).items():
        decisions[key] = e

    models = [k for k in decisions if k not in BASELINES]
    ci = ts.bootstrap(p, decisions, ranked=models)

    def rank_of(U, O, T=ts.URGENT_THRESHOLD):
        s = np.array([ts.score_model(p, decisions[k], U, O, T).score for k in models])
        return dict(zip(models, ts.competition_ranks(s).tolist())), s

    ranks, base_scores = rank_of(ts.TOLERATED_UNDER_TRIAGE, ts.TOLERATED_OVER_TRIAGE)
    sens = {}
    lowU, highU = ts.SENSITIVITY_U
    lowO, highO = ts.SENSITIVITY_O
    for U in (lowU, ts.TOLERATED_UNDER_TRIAGE, highU):
        for O in (lowO, ts.TOLERATED_OVER_TRIAGE, highO):
            r, s = rank_of(U, O)
            sens[f"U={U:g},O={O:g},T={ts.URGENT_THRESHOLD:g}"] = {
                "spearman_vs_default": float(spearmanr(base_scores, s).statistic),
                "ranks": r,
                "always_escalate_score": ts.score_model(p, decisions["always-escalate"], U, O).score,
                "always_escalate_would_rank": int(1 + (s > ts.score_model(p, decisions["always-escalate"], U, O).score).sum()),
            }

    # Threshold sensitivity at the default U and O: how ranks move with T.
    sens_t = {}
    lowT, highT = ts.SENSITIVITY_T
    for T in (lowT, ts.URGENT_THRESHOLD, highT):
        r, s = rank_of(ts.TOLERATED_UNDER_TRIAGE, ts.TOLERATED_OVER_TRIAGE, T)
        ae = ts.score_model(p, decisions["always-escalate"], T=T).score
        sens_t[f"T={T:g}"] = {
            "urgent_cases": int(ts.urgent_labels(p, T).sum()),
            "spearman_vs_default": float(spearmanr(base_scores, s).statistic),
            "max_rank_move": int(max(abs(r[k] - ranks[k]) for k in models)),
            "ranks": r,
            "always_escalate_score": ae,
            "always_escalate_would_rank": int(1 + (s > ae).sum()),
        }
        for U in (lowU, highU):
            for O in (lowO, highO):
                if T == ts.URGENT_THRESHOLD:
                    continue
                r2, _ = rank_of(U, O, T)
                sens[f"U={U:g},O={O:g},T={T:g}"] = {"ranks": r2}
        if T != ts.URGENT_THRESHOLD:
            sens[f"U={ts.TOLERATED_UNDER_TRIAGE:g},O={ts.TOLERATED_OVER_TRIAGE:g},T={T:g}"] = {"ranks": r}

    for k in models:
        rows[k]["rank"] = ranks[k]
        rows[k]["ci"] = ci[k]
        rows[k]["rank_range_sensitivity"] = [min(v["ranks"][k] for v in sens.values()),
                                             max(v["ranks"][k] for v in sens.values())]

    # Measurement ceiling: the evidence-reference label scored as a model against the primary label.
    scales = lc.load_scales(COND_PATH, ROOT / lc.REFERENCE_CSV)
    labels = lc.build_labels(case_ids, lc.load_case_differentials(case_ids, DDX_CSV), scales)
    ceiling = lc.label_ceiling(p, labels, decisions)
    for k in models:
        rows[k]["kappa"] = ceiling["models"][k]

    baselines = []
    for key, name in BASELINES.items():
        m = ts.score_model(p, decisions[key])
        s_models = np.array([rows[k]["score"] for k in models])
        baselines.append({
            "id": key,
            "name": name,
            "under": m.under,
            "over": m.over,
            "distance": m.distance,
            "score": m.score,
            "ci": ci[key],
            "would_rank": int(1 + (s_models > m.score).sum()),
            "label": ts.label_counts(decisions[key], required),
            "threshold": ts.label_counts(decisions[key], urgent),
            "kappa": ceiling["models"][key],
        })

    out = {
        "meta": {
            "generated": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "eval_set": "eval-250-v0",
            "cases": len(p),
            "U": ts.TOLERATED_UNDER_TRIAGE,
            "O": ts.TOLERATED_OVER_TRIAGE,
            "sensitivity_U": list(ts.SENSITIVITY_U),
            "sensitivity_O": list(ts.SENSITIVITY_O),
            "tolerances_status": "provisional pending clinician confirmation",
            "tolerances_source": "spec/triage_tolerances.md",
            "severe_max": ts.SEVERE_MAX,
            "T": ts.URGENT_THRESHOLD,
            "sensitivity_T": list(ts.SENSITIVITY_T),
            "urgent_cases": int(urgent.sum()),
            "nonurgent_cases": int((~urgent).sum()),
            "urgent_rule": "urgent = P(severe) >= T, P(severe) = DDXPlus probability mass on severity 1-2 diagnoses across the whole differential",
            "v0_label": "escalation_required in the eval set: a severity 1-2 diagnosis in the DDXPlus top 3",
            "sensitivity_T_grid": sens_t,
            "p_severe_sum": float(p.sum()),
            "p_severe_mean": float(p.mean()),
            "n_bootstrap": ts.N_BOOTSTRAP,
            "bootstrap_seed": ts.BOOTSTRAP_SEED,
            "scored_models": len(models),
            "sensitivity": sens,
        },
        "ceiling": {
            **ceiling["ceiling"],
            "all_labels": ceiling["labels"],
            "ordinal_inter_label_qwk": lc.ORDINAL_INTER_LABEL_QWK,
            "source": "docs/ddxplus-severity-validation.md",
            "at_limit_rule": "model kappa 95% interval upper bound >= label-vs-label kappa 95% interval lower bound",
        },
        "rows": rows,
        "baselines": baselines,
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT.relative_to(ROOT)}: {len(models)} scored rows, {len(rows) - len(models)} without per-case data")
    order = sorted(models, key=lambda k: -rows[k]["score"])
    for k in order:
        r = rows[k]
        print(f"{r['rank']:>2} {k:34s} {r['status']:6s} score {r['score']:5.1f} [{r['ci']['score'][0]:4.1f}-{r['ci']['score'][1]:4.1f}]"
              f" under {100*r['under']:4.1f}% over {100*r['over']:4.1f}% rank CI {r['ci']['rank']}"
              f" sens {r['rank_range_sensitivity']}")
    for b in baselines:
        print(f"-- {b['name']:32s} score {b['score']:5.1f} under {100*b['under']:5.1f}% over {100*b['over']:5.1f}% would rank {b['would_rank']}")
    for k, v in sens.items():
        if "spearman_vs_default" in v:
            print(f"sens {k}: rho {v['spearman_vs_default']:.2f}, always-escalate would rank {v['always_escalate_would_rank']}")
    for k, v in ceiling["labels"].items():
        print(f"label {k:24s} urgent {v['urgent_cases']:3d} under {100*v['under']:4.1f}% {[round(100*x,1) for x in v['ci']['under']]}"
              f" over {100*v['over']:4.1f}% {[round(100*x,1) for x in v['ci']['over']]} score {v['score']:4.1f} {[round(x,1) for x in v['ci']['score']]}"
              f" kappa {v['kappa']:.2f} {[round(x,2) for x in v['kappa_ci']]}")
    for k in order:
        kk = ceiling["models"][k]
        print(f"kappa {k:34s} {kk['kappa']:.2f} {[round(x,2) for x in kk['ci']]} {'AT LIMIT' if kk['at_limit'] else ''}")
    for k, v in sens_t.items():
        print(f"sens {k}: urgent {v['urgent_cases']}, rho {v['spearman_vs_default']:.2f}, max rank move {v['max_rank_move']}, always-escalate would rank {v['always_escalate_would_rank']}")


if __name__ == "__main__":
    main()
