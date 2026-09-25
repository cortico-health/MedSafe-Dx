import json
import unittest

import numpy as np

from evaluator import answer_key_v03 as ak
from evaluator import v03_anchors as an
from evaluator import v03_score as vs
from evaluator.condition_match import FlagMatcher

MI = "Possible NSTEMI / STEMI"
PE = "Pulmonary embolism"
PSVT = "PSVT"
TIERS = {MI: 1, PE: 1, PSVT: 1, "COPD": 2, "URTI": 3}


def target(cond, source, p, n=100, rate=0.1, in_r10=True, in_r5=True, status=ak.KEPT):
    return ak.TargetV03(condition=cond, source=source, dxa_p=p, status=status, undetermined=False,
                        in_r10=in_r10, in_r5=in_r5, class_n=n, class_rate=rate, hallmark_tokens=())


def case(cid, truth, tier, clearly_low=False, targets=()):
    k = ak.CaseKeyV03(case_id=cid, truth=truth, truth_tier=tier, clearly_low_risk=clearly_low,
                      intermediate=False, red_flag=False, red_flag_names=())
    for t in targets:
        k.considered[t.condition] = t
    k.intermediate = not k.r10 and not clearly_low
    return k


class TestDxaReaderRule(unittest.TestCase):
    def test_threshold_filter_cap_and_ties(self):
        dxa = {MI: 30.0, PE: 30.0, PSVT: 9.9, "URTI": 60.0}
        never = lambda c, p: False  # noqa: E731
        self.assertEqual(an.dxa_reader_conditions(dxa, TIERS, never), [MI, PE])  # ties by name; URTI is not tier 1
        self.assertEqual(an.dxa_reader_conditions(dxa, TIERS, never, threshold=5.0), [MI, PE, PSVT])
        self.assertEqual(an.dxa_reader_conditions(dxa, TIERS, never, cap=1), [MI])
        self.assertEqual(an.dxa_reader_conditions(dxa, TIERS, lambda c, p: c == MI), [PE])

    def test_red_herring_from_key_rows(self):
        # PE at 30% in a class of 400 with rate 0.5%: under max(1%, 3%), a red herring. MI's class passes.
        k = case("c1", "COPD", 2, targets=[target(PE, "dxa", 30.0, n=400, rate=0.005, in_r10=False, in_r5=False,
                                                  status=ak.RED_HERRING),
                                           target(MI, "dxa", 12.0, n=400, rate=0.05)])
        self.assertEqual(an.dxa_reader_from_key(k, TIERS), [MI])
        self.assertTrue(an.key_red_herring(k)(PE, 30.0))
        # an undetermined class (under 30 patients) is kept
        k2 = case("c2", "COPD", 2, targets=[target(PE, "dxa", 30.0, n=10, rate=0.0)])
        self.assertEqual(an.dxa_reader_from_key(k2, TIERS), [PE])
        with self.assertRaises(ValueError):
            an.dxa_reader_from_key(k, TIERS, threshold=2.0)


@unittest.skipUnless(all(vs.SETS[s]["refs"].exists() for s in vs.SETS), "v0.3 reference rows not built")
class TestDxaReaderEverywhere(unittest.TestCase):
    def test_committed_reference_rows_follow_the_rule(self):
        """The builder's DXA rows and the rule applied to the key give the same flags on every set, so every
        coverage figure for the reader comes from one rule."""
        m = FlagMatcher()
        canon = m.cmap.canonical
        for s in vs.SETS:
            key, _ = vs.load_set(s)
            refs = {p["case_id"]: p for p in vs.reference_rows(vs.SETS[s]["refs"])["dxa"]["predictions"]}
            for cid, k in zip(key.case_ids, key.keys):
                want = [canon[c] for c in an.dxa_reader_from_key(k, key.tiers)]
                self.assertEqual(refs[cid]["flags"], want, (s, cid))
                self.assertEqual(refs[cid]["serious_concern"] == "YES", bool(want), (s, cid))


class TestPolicyAndAnchors(unittest.TestCase):
    def setUp(self):
        self.keys = [case("s1", MI, 1, targets=[target(MI, "truth", 30)]),
                     case("s2", "Stable angina", 1, targets=[target("Stable angina", "truth", 30)]),
                     case("b1", "URTI", 3, clearly_low=True), case("b2", "URTI", 3, clearly_low=True),
                     case("m1", "COPD", 2)]
        self.key = vs.set_key("toy", [k.case_id for k in self.keys], {k.case_id: k for k in self.keys})
        self.s, self.b = an.key_masks(self.key)

    def test_both_scales(self):
        ae = an.policy(np.ones(5, bool), self.s, self.b)
        self.assertEqual(ae["score"], 0.0)
        self.assertAlmostEqual(ae["SC"], 100 * 2 / 5)
        self.assertAlmostEqual(ae["cost_per_100"], 100 * 2 / 4)
        ar = an.policy(np.zeros(5, bool), self.s, self.b)
        self.assertAlmostEqual(ar["score"], 100 * (2 - 14) / 2)
        self.assertEqual((ar["U"], ar["O"]), (1.0, 0.0))
        oracle = an.policy(np.array([True, True, False, False, True]), self.s, self.b)
        self.assertEqual((oracle["score"], oracle["SC"]), (100.0, 0.0))  # MIDDLE is free

    def test_weak_fit_masks(self):
        s, b = an.key_masks(self.key, weak_fit_excluded=True)
        np.testing.assert_array_equal(s, [True, False, False, False, False])

    def test_label_stability_ranges(self):
        esc = {"m": np.array([True, False, True, False, True])}
        relabelled = (self.s & np.array([True, False, True, True, True]), self.b)
        out = an.label_stability(esc, {"primary": (self.s, self.b), "validate-relabel": relabelled,
                                       "R20": (np.zeros(5, bool), self.b)})
        r = out["m"]
        self.assertEqual(r["label_stability"]["SC"], [100 * 1 / 5, 100 * 8 / 5])
        self.assertEqual(r["key_variants"]["SC"], [100 * 1 / 5, 100 * 8 / 5])
        self.assertEqual(r["by_variant"]["R20"]["score"], 50.0)

    def test_anchors_for_set(self):
        preds = [{"case_id": k.case_id, "serious_concern": "YES", "flags": []} for k in self.keys]
        rows = {"m": {"predictions": preds, "kind": "model"}, "n": {"predictions": preds[:2], "kind": "model"},
                "dxa": {"predictions": [], "kind": "reference"}, "naive-bayes": {"predictions": [], "kind": "reference"}}
        a = an.anchors_for_set(self.key, [], rows, FlagMatcher(), relabel=False)
        self.assertEqual(a["human"]["text"], "no comparable human baseline; published triage rates measure different tasks")
        self.assertEqual((a["human"]["context"]["under_triage_pct"], a["human"]["context"]["over_triage_pct"]), (17.4, 20.2))
        self.assertEqual(a["naive_bayes"]["label"], "dataset-knowledge ceiling")
        self.assertEqual((a["oracle"]["SC"], a["oracle"]["score"]), (0.0, 100.0))
        self.assertEqual(a["dxa_policy_loss"]["score"], an.policy(np.zeros(5, bool), self.s, self.b)["score"])
        self.assertIsNone(a["label_uncertainty"]["relabel"])
        self.assertEqual(a["model_discordance"]["m|n"], {"serious": 0.0, "benign": 1.0})
        self.assertIn("m", a["weak_fit_excluded"])


@unittest.skipUnless(vs.SETS["main"]["key"].exists(), "v0.3 key not built")
class TestBoard(unittest.TestCase):
    def test_board_carries_anchors_power_and_report_sections(self):
        from evaluator.v03_report import render_report

        board = vs.score_board([], n_boot=100)
        a = board["sets"]["main"]["anchors"]
        self.assertAlmostEqual(round(a["always_escalate"]["cost_per_100"], 1), 33.5)
        self.assertAlmostEqual(round(a["always_routine"]["score"], 1), -1288.1)
        self.assertAlmostEqual(round(a["dxa_policy_loss"]["score"], 1), -166.9)
        self.assertAlmostEqual(round(a["naive_bayes"]["score"], 1), -101.7)
        self.assertEqual(board["power"]["recommendation"]["per_condition"], 30)
        self.assertIn("within", board["constants"]["bootstrap"])
        rows = board["sets"]["main"]["rows"]
        self.assertIn("sens:weak_fit_excluded", rows["dxa"]["point"])
        self.assertIn("ci_condition", rows["dxa"])
        md = render_report(json.loads(json.dumps(vs._jsonable(board))))
        for h in ("### Calibration anchors", "### Label uncertainty", "no comparable human baseline",
                  "### Exact bounds", "## Power and sample size", "Wrong-serious substitute"):
            self.assertIn(h, md)


if __name__ == "__main__":
    unittest.main()
