import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluator import triage_score as ts

ROOT = Path(__file__).resolve().parents[2]


def pred(cid, decision="ESCALATE_NOW", n_dx=5):
    return {
        "case_id": cid,
        "differential_diagnoses": [{"code": "I21.9"}] * n_dx,
        "escalation_decision": decision,
        "uncertainty": "UNCERTAIN",
    }


class TestTriageRates(unittest.TestCase):
    # With T = 0.15: cases 0, 1 and 4 are urgent; 2 and 3 are not.
    p = np.array([0.9, 0.5, 0.1, 0.0, 0.15])
    T = 0.15

    def test_threshold_label(self):
        self.assertEqual(ts.urgent_labels(self.p, self.T).tolist(), [True, True, False, False, True])

    def test_rates_are_plain_counts(self):
        under, over = ts.triage_rates(self.p, np.array([1, 0, 1, 0, 0]), self.T)
        self.assertAlmostEqual(under, 2 / 3)
        self.assertAlmostEqual(over, 1 / 2)

    def test_oracle_scores_100(self):
        # A model that escalates exactly when P(severe) >= T makes no error of either kind.
        e = (self.p >= ts.URGENT_THRESHOLD).astype(int)
        self.assertEqual(ts.triage_rates(self.p, e), (0.0, 0.0))
        self.assertAlmostEqual(ts.score_model(self.p, e).score, 100.0)

    def test_always_escalate_baseline(self):
        e = ts.baseline_decisions(len(self.p))["always-escalate"]
        under, over = ts.triage_rates(self.p, e, self.T)
        self.assertEqual((under, over), (0.0, 1.0))
        # d = 1/O, so the score is 100 / (1 + 1/(2 O^2)) whatever the cases are.
        O = ts.TOLERATED_OVER_TRIAGE
        self.assertAlmostEqual(ts.score_model(self.p, e).score, 100 / (1 + 1 / (2 * O * O)))

    def test_never_escalate_baseline(self):
        e = ts.baseline_decisions(len(self.p))["never-escalate"]
        self.assertEqual(ts.triage_rates(self.p, e, self.T), (1.0, 0.0))


class TestTriageScore(unittest.TestCase):
    def test_anchor_points(self):
        U, O = 0.05, 0.5
        self.assertAlmostEqual(ts.triage_score(0, 0, U, O), 100.0)
        self.assertAlmostEqual(ts.triage_score(U, O, U, O), 50.0)
        self.assertAlmostEqual(ts.triage_score(U, 0, U, O), 100 / 1.5)

    def test_monotone_in_each_rate(self):
        self.assertGreater(ts.triage_score(0.02, 0.3), ts.triage_score(0.03, 0.3))
        self.assertGreater(ts.triage_score(0.02, 0.3), ts.triage_score(0.02, 0.4))

    def test_isoscore_distance_inverts_score(self):
        for s in (25, 50, 75):
            self.assertAlmostEqual(float(ts.triage_score_from_distance(ts.isoscore_distance(s))), s)


class TestDecisions(unittest.TestCase):
    ids = ["a", "b", "c", "d"]

    def test_unreadable_and_missing_count_as_not_escalated(self):
        raw = [
            pred("a", "ESCALATE_NOW"),
            pred("b", "ESCALATE_NOW", n_dx=2),  # fewer than 5 diagnoses: format failure
            {"case_id": "c", "error": "api_failure", "raw_response": None},
            # "d" is missing entirely
        ]
        e = ts.escalation_decisions(raw, self.ids)
        self.assertEqual(e.tolist(), [1, 0, 0, 0])
        p = np.array([0.2, 0.6, 0.4, 0.0])  # a, b, c urgent at T = 0.15; d not
        under, over = ts.triage_rates(p, e, 0.15)
        # Unreadable (b), errored (c) and missing (d) output count as not escalated:
        # they add to under-triage and never to over-triage.
        self.assertAlmostEqual(under, 2 / 3)
        self.assertAlmostEqual(over, 0.0)

    def test_label_counts_treat_unreadable_urgent_as_miss(self):
        raw = [pred("a", "ROUTINE_CARE"), pred("b", n_dx=1), pred("c"), pred("d")]
        e = ts.escalation_decisions(raw, self.ids)
        c = ts.label_counts(e, np.array([True, True, False, False]))
        self.assertEqual(c["missed_escalations"], 2)
        self.assertEqual(c["over_escalations"], 2)

    def test_first_prediction_per_case_wins(self):
        raw = [pred("a", "ROUTINE_CARE"), pred("a", "ESCALATE_NOW")]
        self.assertEqual(ts.escalation_decisions(raw, ["a"]).tolist(), [0])


class TestBootstrap(unittest.TestCase):
    def test_intervals_cover_point_estimate_and_baselines_get_no_rank(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0, 0.6, 200)
        good = (p >= ts.URGENT_THRESHOLD).astype(int)
        noisy = ((p + rng.normal(0, 0.15, 200)) >= ts.URGENT_THRESHOLD).astype(int)
        dec = {"good": good, "noisy": noisy, "always-escalate": np.ones(200, int)}
        out = ts.bootstrap(p, dec, ranked=["good", "noisy"], n_boot=500)
        for k in ("good", "noisy"):
            s = ts.score_model(p, dec[k]).score
            lo, hi = out[k]["score"]
            self.assertLessEqual(lo, s)
            self.assertGreaterEqual(hi, s)
            self.assertIn("rank", out[k])
        self.assertNotIn("rank", out["always-escalate"])
        self.assertEqual(out["good"]["rank"][0], 1)

    def test_competition_ranks_share_ties(self):
        self.assertEqual(ts.competition_ranks(np.array([50.0, 70.0, 50.0])).tolist(), [2, 1, 2])


class TestEvaluatorRefusesLockedPredictions(unittest.TestCase):
    """evaluator.cli must not score a file an inference run still holds (run-2026-09 Mini/Luna)."""

    def test_refuses_while_locked(self):
        import fcntl

        with tempfile.TemporaryDirectory() as d:
            cases = Path(d) / "cases.json"
            cases.write_text(json.dumps([{"case_id": "a", "gold_top3": ["I21"], "escalation_required": True,
                                          "uncertainty_acceptable": False}]))
            preds = Path(d) / "p.json"
            preds.write_text(json.dumps([pred("a")]))
            cmd = [sys.executable, "-m", "evaluator.cli", "--cases", str(cases), "--predictions", str(preds),
                   "--model-name", "x", "--model-version", "x", "--out", str(Path(d) / "e.json")]
            with open(str(preds) + ".lock", "w") as fd:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
                self.assertNotEqual(r.returncode, 0)
                self.assertIn("still writing", r.stderr)
            r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)


class TestLeaderboardPredictionHashes(unittest.TestCase):
    """Every leaderboard row whose prediction file is on disk must match its recorded sha256.

    results/ is gitignored, so on a fresh clone the files are absent and the check skips them.
    """

    def test_sha256_matches(self):
        checked = 0
        for f in sorted((ROOT / "leaderboard").glob("*-eval.json")):
            ev = json.loads(f.read_text())
            p = ROOT / ev["predictions_path"]
            if not p.exists():
                continue
            checked += 1
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), ev["predictions_sha256"], f.name)
        if checked == 0:
            self.skipTest("no prediction files on disk")


class TestBoardData(unittest.TestCase):
    """leaderboard/triage-scores.json must agree with the scoring code and constants."""

    def test_board_matches_config(self):
        f = ROOT / "leaderboard/triage-scores.json"
        if not f.exists():
            self.skipTest("triage-scores.json not built")
        d = json.loads(f.read_text())
        self.assertEqual(d["meta"]["U"], ts.TOLERATED_UNDER_TRIAGE)
        self.assertEqual(d["meta"]["O"], ts.TOLERATED_OVER_TRIAGE)
        self.assertEqual(d["meta"]["T"], ts.URGENT_THRESHOLD)
        for r in d["rows"].values():
            if "score" in r:
                self.assertAlmostEqual(r["score"], ts.triage_score(r["under"], r["over"]), places=9)
        always = next(b for b in d["baselines"] if b["id"] == "always-escalate")
        self.assertEqual((always["under"], always["over"]), (0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
