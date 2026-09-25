import unittest

import numpy as np

from evaluator import answer_key_v03 as ak
from evaluator import v03_diagnosis as dg
from evaluator import v03_score as vs
from evaluator.condition_match import FlagMatcher

MI = "Possible NSTEMI / STEMI"
UA = "Unstable angina"
PE = "Pulmonary embolism"
BOER = "Boerhaave"
PTX = "Spontaneous pneumothorax"


def target(cond, source, p, in_r10=True, in_r5=True):
    return ak.TargetV03(condition=cond, source=source, dxa_p=p, status=ak.KEPT, undetermined=False,
                        in_r10=in_r10, in_r5=in_r5, class_n=100, class_rate=0.1, hallmark_tokens=())


def case(cid, truth, tier, clearly_low=False, targets=()):
    k = ak.CaseKeyV03(case_id=cid, truth=truth, truth_tier=tier, clearly_low_risk=clearly_low,
                      intermediate=False, red_flag=False, red_flag_names=())
    for t in targets:
        k.considered[t.condition] = t
    k.intermediate = not k.r10 and not clearly_low
    return k


def diff(*pairs):
    return [{"code": c, "p": p} for c, p in pairs]


# b1-b5: Boerhaave truths (tier 1); b6: COPD truth with an R10 PE target and an R5-only pneumothorax target;
# u1-u3: URTI, clearly low-risk; a1 unstable angina truth.
KEYS = [
    case("b1", BOER, 1, targets=[target(BOER, "truth", 30)]),
    case("b2", BOER, 1, targets=[target(BOER, "truth", 30)]),
    case("b3", BOER, 1, targets=[target(BOER, "truth", 30)]),
    case("b4", BOER, 1, targets=[target(BOER, "truth", 30)]),
    case("b5", BOER, 1, targets=[target(BOER, "truth", 30)]),
    case("b6", "COPD", 2, targets=[target(PE, "dxa", 15), target(PTX, "dxa", 7, in_r10=False)]),
    case("u1", "URTI", 3, clearly_low=True),
    case("u2", "URTI", 3, clearly_low=True),
    case("u3", "URTI", 3, clearly_low=True),
    case("a1", UA, 1, targets=[target(UA, "truth", 30)]),
]
PREDS = [
    # Boerhaave called aortic dissection (off-list, Newman-Toker group) and MI: a substitute, confident wrong at 70.
    {"case_id": "b1", "serious_concern": "YES", "flags": ["I71.0"], "differential": diff(("I21.4", 70), ("I71.0", 20))},
    # Boerhaave named in the differential but not flagged: listed, not acted on; no substitute.
    {"case_id": "b2", "serious_concern": "YES", "flags": ["I21.4"], "differential": diff(("K22.3", 40), ("I21.4", 30))},
    # Boerhaave named in the differential, answered NO: listed, not escalated.
    {"case_id": "b3", "serious_concern": "NO", "flags": [], "differential": diff(("K22.3", 40), ("K21.9", 60))},
    # Nothing serious named: a miss, not a substitute.
    {"case_id": "b4", "serious_concern": "YES", "flags": ["K21.9"], "differential": diff(("K21.9", 90))},
    # Empty differential: no forecast, so D1 skips it.
    {"case_id": "b5", "serious_concern": "YES", "flags": [], "differential": []},
    # Names the R5-only pneumothorax, not the R10 PE: not a substitute (it is one of the case's own concerns).
    {"case_id": "b6", "serious_concern": "YES", "flags": ["J93.1"], "differential": diff(("J93.1", 50))},
    # Off-list top code at 80.
    {"case_id": "u1", "serious_concern": "YES", "flags": [], "differential": diff(("Z99", 80))},
    # Probabilities unreadable (sum over 105): no forecast.
    {"case_id": "u2", "serious_concern": "NO", "flags": [], "differential": diff(("J06.9", 90), ("J20.9", 90))},
    {"case_id": "u3", "serious_concern": "NO", "flags": [], "differential": diff(("J06.9", 90))},
    # Unstable angina called MI at 85: confident wrong, same family.
    {"case_id": "a1", "serious_concern": "YES", "flags": ["I21.4"], "differential": diff(("I21.4", 85))},
]


class ToyBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = FlagMatcher()
        cls.key = vs.set_key("toy", [k.case_id for k in KEYS], {k.case_id: k for k in KEYS})
        cls.o = vs.outcomes(PREDS, cls.key, cls.m)
        cls.st, cls.counts = vs.row_stats(cls.o, cls.key, cls.m, sensitivity=False)
        cls.point = cls.st.evaluate(np.ones((1, cls.key.k)))[0]

    def idx(self, cid):
        return self.key.case_ids.index(cid)


class TestD1(ToyBase):
    def test_forecasts_exclude_empty_and_unreadable(self):
        _, _, has = dg.top_forecast(self.o)
        self.assertFalse(has[self.idx("b5")])  # empty differential
        self.assertFalse(has[self.idx("u2")])  # unreadable probabilities
        self.assertEqual(int(has.sum()), 8)
        d1 = self.counts["DX"]["D1"]
        self.assertEqual((d1["forecasts"], d1["cases"]), (8, 10))
        self.assertEqual(d1["no_forecast"]["no_differential"], 1)
        self.assertAlmostEqual(self.point["D1_completeness"], 0.8)

    def test_brier_is_the_mean_over_forecasts(self):
        top, p, has = dg.top_forecast(self.o)
        sq, top1 = dg.brier(top, p, has, self.key.truth, self.m)
        self.assertAlmostEqual(self.point["D1_brier"], sq[has].mean())
        self.assertTrue(top1[self.idx("b2")])  # K22.3 is Boerhaave
        self.assertTrue(top1[self.idx("u3")])
        self.assertEqual(sq[self.idx("b5")], 0.0)  # present in the vector, but outside the denominator


class TestD2All(ToyBase):
    def test_every_confident_wrong_counts(self):
        d2 = self.counts["DX"]["D2_any"]
        # 60: b1 (MI 70), b4 (GERD 90), a1 (MI 85); u1 is off-list; b2 at 40 is under the bar.
        self.assertEqual(d2["60"]["events"], 3)
        self.assertEqual(d2["70"]["events"], 3)
        self.assertEqual(d2["80"]["events"], 2)
        self.assertEqual(d2["60"]["same_family"], 1)  # a1: MI is in unstable angina's ischaemia family
        self.assertEqual(d2["80"]["offlist"], 1)
        self.assertEqual({e["case_id"] for e in d2["60"]["examples"]}, {"b1", "b4", "a1"})
        # The tier-gap D2 sees none of them, Astra finding 10's complaint: GERD (tier 2) for a tier-1 truth is a
        # gap of 1, and MI for Boerhaave or unstable angina a gap of 0.
        self.assertEqual(self.counts["DX"]["D2"]["60"]["events"], 0)


class TestSubstitute(ToyBase):
    def test_substitute_cases(self):
        s = self.counts["substitute"]
        self.assertEqual(s["all_case_ids"], ["b1"])
        self.assertEqual(s["serious_cases"], 7)
        ex = s["examples"][0]
        self.assertEqual(ex["named_tier1"], [MI])
        self.assertEqual(ex["named_offlist"], ["I710"])
        self.assertEqual(ex["offlist_groups"], ["Aortic aneurysm and dissection"])
        self.assertEqual(ex["target_source"], "truth")
        self.assertAlmostEqual(self.point["substitute_serious"], 1 / 7)

    def test_family_code_credits_the_target(self):
        # Unstable angina named by an MI code: the standard map credits the target, so no substitute.
        mask, _ = dg.substitutes([["I21.4"]], [KEYS[-1]], self.key.tiers, self.m)
        self.assertFalse(mask[0])

    def test_broader_code_does_not_count_as_another_condition(self):
        # I24.9 names MI only as a family ("broader") code: not a precise naming of another condition.
        mask, _ = dg.substitutes([["I24.9"]], [KEYS[0]], self.key.tiers, self.m)
        self.assertFalse(mask[0])


class TestListedNotActed(ToyBase):
    def test_listed_targets(self):
        c = self.counts["listed_not_acted"]
        self.assertEqual((c["cases"], c["not_escalated"], c["not_flagged"]), (2, 1, 1))
        self.assertEqual({e["case_id"]: e["why"] for e in c["examples"]},
                         {"b2": "escalated, not flagged", "b3": "not escalated"})
        self.assertAlmostEqual(self.point["listed_not_acted"], 2 / 7)

    def test_without_flags_only_the_decision_counts(self):
        d = dg.listed_not_acted([["K22.3"], ["K22.3"]], np.array([True, False]), None, KEYS[:2], self.m)
        np.testing.assert_array_equal(d["any"], [False, True])


class TestExactBoundsHook(ToyBase):
    def test_bounds_follow_the_counts(self):
        eb = self.counts["exact_bounds"]
        self.assertIn("unreadable_share", eb)  # 0 of 10
        self.assertNotIn("substitute_serious", eb)  # 1 of 7


if __name__ == "__main__":
    unittest.main()
