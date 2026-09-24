import unittest

import numpy as np

from evaluator import label_ceiling as lc
from evaluator import triage_score as ts


class TestKappa(unittest.TestCase):
    def test_perfect_and_chance(self):
        a = np.array([0, 1, 0, 1, 1, 0])
        self.assertAlmostEqual(lc.cohen_kappa(a, a), 1.0)
        # Independent marginals (each 50/50, crossed evenly) give kappa 0.
        self.assertAlmostEqual(lc.cohen_kappa([0, 0, 1, 1], [0, 1, 0, 1]), 0.0)

    def test_known_binary_value(self):
        # 2x2 table [[20, 5], [10, 15]]: po = 0.7, pe = 0.5*0.6 + 0.5*0.4 = 0.5, kappa = 0.4
        a = np.array([0] * 25 + [1] * 25)
        b = np.array([0] * 20 + [1] * 5 + [0] * 10 + [1] * 15)
        self.assertAlmostEqual(lc.cohen_kappa(a, b), 0.4)

    def test_constant_rater_has_zero_kappa(self):
        # Always-escalate agrees with the urgent cases only by chance.
        self.assertAlmostEqual(lc.cohen_kappa(np.ones(6, int), [0, 1, 0, 1, 1, 0]), 0.0)

    def test_quadratic_weighted_ordinal(self):
        # One-step errors cost less than a two-step error under quadratic weights.
        ref = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        near = np.array([1, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        far = np.array([4, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        q_near = lc.cohen_kappa(near, ref, n_levels=5, weights="quadratic")
        q_far = lc.cohen_kappa(far, ref, n_levels=5, weights="quadratic")
        self.assertGreater(q_near, q_far)
        self.assertGreater(q_near, lc.cohen_kappa(near, ref, n_levels=5))  # weighted > unweighted for near misses

    def test_bootstrap_interval_contains_estimate(self):
        rng = np.random.default_rng(1)
        ref = rng.integers(0, 2, 300)
        rater = np.where(rng.uniform(size=300) < 0.85, ref, 1 - ref)
        out = lc.bootstrap_kappa(ref, {"r": rater}, n_boot=400)
        lo, hi = out["r"]["ci"]
        self.assertLessEqual(lo, out["r"]["kappa"])
        self.assertGreaterEqual(hi, out["r"]["kappa"])


class TestLabels(unittest.TestCase):
    scales = {"ddxplus": {"A": 1, "B": 3, "C": 4}, "reference": {"A": 1, "B": 2, "C": 4}}
    cases = {
        "x": {"diff": [("B", 0.9), ("A", 0.1)], "pathology": "B"},   # P = 0.1 on DDXPlus, 1.0 on reference
        "y": {"diff": [("C", 0.5), ("A", 0.5)], "pathology": "C"},   # P = 0.5 on both
        "z": {"diff": [("C", 1.0)], "pathology": "A"},                # P = 0 on both; true condition severe
    }

    def test_variants(self):
        L = lc.build_labels(["x", "y", "z"], self.cases, self.scales, T=0.15)
        self.assertEqual(L["differential/ddxplus"].tolist(), [False, True, False])
        self.assertEqual(L["differential/reference"].tolist(), [True, True, False])
        self.assertEqual(L["pathology/ddxplus"].tolist(), [False, False, True])
        self.assertEqual(L["pathology/reference"].tolist(), [True, False, True])

    def test_primary_variant_matches_triage_score(self):
        p = np.array([lc.p_severe(self.cases[c]["diff"], self.scales["ddxplus"]) for c in "xyz"])
        L = lc.build_labels(list("xyz"), self.cases, self.scales)
        self.assertEqual(ts.urgent_labels(p).tolist(), L[lc.PRIMARY_LABEL].tolist())


class TestCeiling(unittest.TestCase):
    def test_label_scored_as_model_and_limit_flag(self):
        rng = np.random.default_rng(3)
        p = rng.uniform(0, 0.5, 300)
        primary = ts.urgent_labels(p)
        alt = primary.copy()
        alt[:6] = ~alt[:6]  # a close second label
        labels = {lc.PRIMARY_LABEL: primary, lc.CEILING_LABEL: alt}
        good = primary.astype(int)
        poor = rng.integers(0, 2, 300)
        out = lc.label_ceiling(p, labels, {"good": good, "poor": poor})
        c = out["ceiling"]
        u, o = ts.rates_against_label(primary, alt)
        self.assertAlmostEqual(c["under"], u)
        self.assertAlmostEqual(c["over"], o)
        self.assertAlmostEqual(c["score"], ts.triage_score(u, o))
        self.assertTrue(out["models"]["good"]["at_limit"])
        self.assertFalse(out["models"]["poor"]["at_limit"])

    def test_rejects_mismatched_primary(self):
        p = np.array([0.1, 0.5])
        with self.assertRaises(ValueError):
            lc.label_ceiling(p, {lc.PRIMARY_LABEL: np.array([True, True]), lc.CEILING_LABEL: np.array([1, 1])}, {})


if __name__ == "__main__":
    unittest.main()
