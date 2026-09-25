import math
import unittest

import numpy as np

from evaluator import answer_key_v03 as ak
from evaluator import v03_score as vs
from evaluator import v03_stats as st
from evaluator.condition_match import FlagMatcher

MI = "Possible NSTEMI / STEMI"
PE = "Pulmonary embolism"


def target(cond, source, p, in_r10=True, in_r5=True, status=ak.KEPT, n=100, rate=0.1):
    return ak.TargetV03(condition=cond, source=source, dxa_p=p, status=status, undetermined=False,
                        in_r10=in_r10, in_r5=in_r5, class_n=n, class_rate=rate, hallmark_tokens=())


def case(cid, truth, tier, clearly_low=False, targets=()):
    k = ak.CaseKeyV03(case_id=cid, truth=truth, truth_tier=tier, clearly_low_risk=clearly_low,
                      intermediate=False, red_flag=False, red_flag_names=())
    for t in targets:
        k.considered[t.condition] = t
    k.intermediate = not k.r10 and not clearly_low
    return k


class TestExactBounds(unittest.TestCase):
    def test_clopper_pearson_known_values(self):
        lo, hi = st.clopper_pearson(0, 10)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 1 - 0.025 ** 0.1, places=6)  # 0.3085
        lo, hi = st.clopper_pearson(10, 10)
        self.assertAlmostEqual(lo, 0.025 ** 0.1, places=6)
        self.assertEqual(hi, 1.0)
        lo, hi = st.clopper_pearson(5, 10)  # textbook: 0.1871-0.8129
        self.assertAlmostEqual(lo, 0.1871, places=4)
        self.assertAlmostEqual(hi, 0.8129, places=4)

    def test_one_sided_and_rule_of_three(self):
        self.assertAlmostEqual(st.one_sided_bound(0, 234), 0.0127, places=4)  # Astra finding 6: 1.27%
        self.assertAlmostEqual(st.one_sided_bound(0, 30), 0.0950, places=4)  # Astra finding 3: 9.5%
        self.assertAlmostEqual(st.one_sided_bound(30, 30), 0.05 ** (1 / 30))
        with self.assertRaises(ValueError):
            st.one_sided_bound(3, 30)

    def test_boundary_bounds_only_at_0_or_n(self):
        self.assertIsNone(st.boundary_bounds(3, 10))
        self.assertIsNone(st.boundary_bounds(0, 0))
        b = st.boundary_bounds(0, 100)
        self.assertEqual(b["rule_of_three"], {"upper": 0.03})
        self.assertIn("upper", b["one_sided"])
        b = st.boundary_bounds(100, 100)
        self.assertAlmostEqual(b["rule_of_three"]["lower"], 0.97)
        self.assertEqual(b["clopper_pearson"][1], 1.0)

    def test_exact_bounds_reads_row_counts(self):
        counts = {"misses": 0, "r10_cases": 234, "concerns": 50, "clearly_low_risk": 118, "unreadable": 0, "cases": 470,
                  "DX": {"E_events": 200, "E_den": 200}}
        b = st.exact_bounds(counts)
        self.assertEqual(sorted(b), ["E", "H", "unreadable_share"])
        self.assertEqual(b["E"]["events"], 200)


class TestWithinConditionResampling(unittest.TestCase):
    def test_draws_keep_each_condition_count(self):
        cond = np.array([0, 0, 0, 1, 1, 2])
        W = st.within_condition_draws(cond, 500, 7)
        self.assertEqual(W.shape, (500, 6))
        np.testing.assert_array_equal(W[:, :3].sum(1), 3)
        np.testing.assert_array_equal(W[:, 3:5].sum(1), 2)
        np.testing.assert_array_equal(W[:, 5], 1)  # a one-case condition never varies
        np.testing.assert_array_equal(W, st.within_condition_draws(cond, 500, 7))

    def test_fixed_mix_estimand_against_condition_bootstrap(self):
        # Condition A: both cases fail; condition B: neither. On the fixed 2 + 2 mix the rate is exactly 1/2, so
        # within-condition resampling gives [0.5, 0.5]; the condition bootstrap also varies the mix.
        keys = [case("a1", MI, 1, targets=[target(MI, "truth", 30)]), case("a2", MI, 1, targets=[target(MI, "truth", 30)]),
                case("b1", PE, 1, targets=[target(PE, "truth", 30)]), case("b2", PE, 1, targets=[target(PE, "truth", 30)])]
        key = vs.set_key("toy", [k.case_id for k in keys], {k.case_id: k for k in keys})
        fail = np.array([1.0, 1.0, 0.0, 0.0])
        s = vs.Stats(key)
        s.add("rate", fail, np.ones(4))
        _, dc = s.evaluate(vs.cluster_draws(key.k, 400, 1))
        ck = st.case_level_key(key)
        s2 = vs.Stats(ck)
        s2.add("rate", fail, np.ones(4))
        point, dw = s2.evaluate(st.within_condition_draws(key.cond_idx, 400, 1))
        self.assertEqual(point["rate"], 0.5)
        self.assertEqual(vs.interval(dw["rate"]), [0.5, 0.5])
        self.assertEqual(vs.interval(dc["rate"]), [0.0, 1.0])

    def test_case_level_key_keeps_points_and_mixes(self):
        keys = [case(f"c{i}", MI if i < 3 else PE, 1, targets=[target(MI if i < 3 else PE, "truth", 30)]) for i in range(5)]
        key = vs.set_key("toy", [k.case_id for k in keys], {k.case_id: k for k in keys})
        v = np.array([1, 0, 1, 1, 0], float)
        w = np.array([2.0, 0.5])
        a, b = vs.Stats(key), vs.Stats(st.case_level_key(key))
        a.add("x", v, np.ones(5), 1.0, w)
        b.add("x", v, np.ones(5), 1.0, st.case_level_mixes(key, {"m": w})["m"])
        self.assertAlmostEqual(a.evaluate(np.ones((1, 2)))[0]["x"], b.evaluate(np.ones((1, 5)))[0]["x"])

    def test_score_set_primary_is_within_condition(self):
        keys = [case("a1", MI, 1, targets=[target(MI, "truth", 30)]), case("a2", MI, 1, targets=[target(MI, "truth", 30)]),
                case("b1", "URTI", 3, clearly_low=True), case("b2", "URTI", 3, clearly_low=True)]
        key = vs.set_key("toy", [k.case_id for k in keys], {k.case_id: k for k in keys})
        preds = [{"case_id": c, "serious_concern": a, "flags": []} for c, a in
                 (("a1", "NO"), ("a2", "NO"), ("b1", "YES"), ("b2", "YES"))]
        rows = {"m": {"predictions": preds, "kind": "model"}, "n": {"predictions": preds, "kind": "model"}}
        b = vs.score_set("toy", key, rows, FlagMatcher(), n_boot=200, seed=1, sensitivity=False)
        r = b["rows"]["m"]
        self.assertEqual(r["ci"]["H"], [1.0, 1.0])  # fixed mix: every MI case is missed
        self.assertIn("ci_condition", r)
        self.assertIn("condition_bootstrap", b["paired"]["m|n"])
        self.assertEqual(r["counts"]["exact_bounds"]["H"]["events"], 2)


class TestWeakFit(unittest.TestCase):
    def test_row_drops_stable_angina_and_scombroid_truths(self):
        keys = [case("s1", "Stable angina", 1, targets=[target("Stable angina", "truth", 30)]),
                case("s2", "Scombroid food poisoning", 1, targets=[target("Scombroid food poisoning", "truth", 30)]),
                case("m1", MI, 1, targets=[target(MI, "truth", 30)]),
                case("u1", "URTI", 3, clearly_low=True)]
        key = vs.set_key("toy", [k.case_id for k in keys], {k.case_id: k for k in keys})
        np.testing.assert_array_equal(st.weak_fit_mask(key), [True, True, False, False])
        yes = np.array([False, False, True, True])
        name, _, (num, den, scale) = st.weak_fit_row(yes, key, key.clearly_low)
        self.assertEqual(name, "weak_fit_excluded")
        s = vs.Stats(key)
        s.add("x", num, den, scale)
        self.assertAlmostEqual(s.evaluate(np.ones((1, key.k)))[0]["x"], 100 * 1 / 2)  # m1 free, u1 costs 1

    def test_scorer_prints_the_row(self):
        keys = [case("s1", "Stable angina", 1, targets=[target("Stable angina", "truth", 30)]),
                case("u1", "URTI", 3, clearly_low=True)]
        key = vs.set_key("toy", [k.case_id for k in keys], {k.case_id: k for k in keys})
        o = vs.outcomes([{"case_id": "s1", "serious_concern": "NO", "flags": []}], key, FlagMatcher())
        _, counts = vs.row_stats(o, key, FlagMatcher())
        self.assertIn("weak_fit_excluded", counts["sensitivity"])


class TestRelabel(unittest.TestCase):
    def test_count_validate_classes(self):
        rows = [{"AGE": "40", "PATHOLOGY": PE, "EVIDENCES": "['E_1', 'E_2']",
                 "DIFFERENTIAL_DIAGNOSIS": f"[['{PE}', 0.3], ['URTI', 0.7]]"},
                {"AGE": "50", "PATHOLOGY": "URTI", "EVIDENCES": "['E_1']",
                 "DIFFERENTIAL_DIAGNOSIS": f"[['{PE}', 0.12], ['{MI}', 0.02]]"},
                {"AGE": "10", "PATHOLOGY": PE, "EVIDENCES": "['E_1']", "DIFFERENTIAL_DIAGNOSIS": f"[['{PE}', 0.3]]"}]
        c = st.count_validate_classes(rows, [PE, MI], {PE: ["E_1", "E_2"]})
        self.assertEqual(c[(PE, ak.band_of(30), 2)], (1, 1))
        self.assertEqual(c[(PE, ak.band_of(12), 1)], (1, 0))
        self.assertNotIn((MI, ak.band_of(2), 0), c)  # under 5%; the child is not counted

    def test_relabel_changes_status_and_class(self):
        k = case("c1", "COPD", 3, targets=[target(PE, "dxa", 15.0)])
        classes = {(PE, ak.band_of(15.0), 1): (100, 0)}  # the validation class never has PE: a red herring
        keys, changes = st.relabel_keys([k], {"c1": ["E_9"]}, classes, {PE: ["E_9"]})
        nk = keys["c1"]
        self.assertFalse(nk.considered[PE].in_r10)
        self.assertEqual(nk.considered[PE].status, ak.RED_HERRING)
        self.assertTrue(nk.clearly_low_risk)  # no R5 target left, tier-3 truth, no red flag
        self.assertEqual(len(changes), 1)
        self.assertEqual((changes[0]["r10_before"], changes[0]["r10_after"]), (True, False))
        self.assertTrue(k.considered[PE].in_r10)  # the input key is untouched

    def test_truth_target_is_never_relabelled(self):
        k = case("c1", MI, 1, targets=[target(MI, "truth", 2.0)])
        keys, changes = st.relabel_keys([k], {"c1": []}, {}, {})
        self.assertTrue(keys["c1"].considered[MI].in_r10)
        self.assertEqual(changes, [])


@unittest.skipUnless(vs.SETS["main"]["key"].exists() and st.HALLMARKS_CSV.exists()
                     and (st.VALIDATE_CSV.exists() or st.VALIDATE_CLASSES_CACHE.exists()), "validation split not present")
class TestRealRelabel(unittest.TestCase):
    def test_reproduces_astra_finding_6(self):
        key, cases = vs.load_set("main")
        vk, changes = st.validation_relabel(key, cases)
        s = st.relabel_summary(key, vk, changes)
        self.assertEqual(s["pair_changes"], 15)
        self.assertEqual(s["r10_target_changes"], 5)
        self.assertEqual(s["r10_cases"], [234, 234])
        self.assertEqual(s["cases_losing_r10"], ["ddxplus_116571"])
        self.assertEqual(s["cases_gaining_r10"], ["ddxplus_51583"])


class TestPower(unittest.TestCase):
    def test_detection(self):
        self.assertAlmostEqual(st.detection_power(0.05, 10), 0.4013, places=4)  # Astra finding 6: 40%
        self.assertAlmostEqual(st.detectable_rate(10), 1 - 0.2 ** 0.1)
        self.assertAlmostEqual(st.detection_power(st.detectable_rate(30), 30), 0.8)
        self.assertEqual(st.cases_to_detect(0.05), 32)
        self.assertEqual(st.cases_to_detect(0.10), 16)

    def test_mde(self):
        z = st.Z_ALPHA + st.Z_BETA
        self.assertAlmostEqual(st.mde_paired_rate(100, 0.1), z * math.sqrt(0.001))
        # With no SERIOUS discordance the SCORE MDE is the O MDE in points.
        self.assertAlmostEqual(st.mde_paired_score(100, 50, 0.0, 0.2), 100 * st.mde_paired_rate(50, 0.2))
        self.assertGreater(st.mde_paired_score(100, 50, 0.1, 0.2), st.mde_paired_score(100, 50, 0.05, 0.2))

    def test_table_and_recommendation(self):
        t = st.power_table(observed_benign_discordance=0.127)
        self.assertEqual([r["per_condition"] for r in t["rows"]], [10, 20, 30])
        self.assertEqual([(r["serious"], r["benign"]) for r in t["rows"]], [(234, 118), (468, 236), (702, 354)])
        self.assertAlmostEqual(t["rows"][0]["detectable_rate_80"], 0.1487, places=4)
        self.assertEqual(t["assumptions"]["benign_discordance"], 0.127)
        self.assertEqual(t["recommendation"]["per_condition"], 30)
        self.assertEqual(st.recommend(t["rows"], target_rate=0.08)["per_condition"], 20)
        self.assertEqual(st.recommend(t["rows"], target_rate=0.01)["per_condition"], 30)  # none qualifies: largest


if __name__ == "__main__":
    unittest.main()
