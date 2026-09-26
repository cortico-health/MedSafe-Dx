import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluator import v03_score as vs
from evaluator import v03_valid_reason as vr
from evaluator import v03b_score as sb
from evaluator.answer_key_v03 import load_tiers
from evaluator.condition_match import FlagMatcher

MI = "Possible NSTEMI / STEMI"
PE = "Pulmonary embolism"


def tier_file(rows: str) -> Path:
    f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
    f.write(rows)
    f.close()
    return Path(f.name)


class TestReason(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = FlagMatcher()
        cls.tiers = load_tiers()
        cls.groups = vr.GroupsRule(vr.load_groups_v2())

    def kind(self, codes, targets=(MI,), rule=None, **kw):
        return vr.reason(codes, list(targets), self.m, self.tiers, rule or self.groups, **kw).kind

    def test_target_beats_other_tier1(self):
        self.assertEqual(self.kind(["I26", "I21"]), vr.TARGET)
        self.assertEqual(self.kind(["I26"]), vr.OTHER_TIER1)

    def test_offlist_needs_tier_1(self):
        self.assertEqual(self.kind(["I63.9"]), vr.OFFLIST)  # stroke, a group
        self.assertEqual(self.kind(["K56.609"]), vr.OFFLIST)  # bowel obstruction, an AHRQ group in v2
        self.assertEqual(self.kind(["M31.6"]), vr.NONE)  # giant-cell arteritis: in no group
        self.assertEqual(self.kind([]), vr.NONE)

    def test_bounding_modes(self):
        self.assertEqual(self.kind(["M31.6"], offlist="escalate"), vr.OFFLIST)
        self.assertEqual(self.kind(["I63.9"], offlist="routine"), vr.NONE)

    def test_tier_file_longest_prefix(self):
        p = tier_file("icd10_prefix,tier,label\nK92,2,GI haemorrhage\nK92.2,1,GI haemorrhage unspecified\n"
                      "R07,unscored,chest pain\nM31.6,1,giant-cell arteritis\n")
        rule = vr.TierFileRule(p)
        self.assertEqual(rule.tier("K92.2")[0], "1")
        self.assertEqual(rule.tier("K92.0")[0], "2")
        self.assertEqual(rule.tier("R07.9")[0], vr.UNSCORED)
        self.assertEqual(rule.tier("L50.0")[0], vr.UNLISTED)
        self.assertEqual(self.kind(["M31.6"], rule=rule), vr.OFFLIST)
        self.assertEqual(self.kind(["R07.9"], rule=rule), vr.NONE)
        self.assertEqual(self.kind(["K92.0"], rule=rule), vr.NONE)

    def test_tier_file_needs_columns(self):
        with self.assertRaises(ValueError):
            vr.TierFileRule(tier_file("code_x,level\nA00,1\n"))


class TestCosts(unittest.TestCase):
    def test_case_cost(self):
        o = vr.Outcome(np.array([True] * 6), [vr.Reason(vr.NONE)] * 6,
                       [vr.PASS, vr.PARTIAL_OUT, vr.MISS_OUT, vr.BARE, vr.BENIGN_ESC, vr.NOT_SCORED])
        self.assertEqual(vr.case_cost(o).tolist(), [0, 1, 7, 7, 1, 0])
        self.assertEqual(vr.case_cost(o, 3.5).tolist(), [0, 3.5, 7, 7, 1, 0])


class TestOnTheSet(unittest.TestCase):
    """The anchors on the 150-case set: always escalate with a target named = 0 under both weightings;
    always routine = -600 balanced."""

    @classmethod
    def setUpClass(cls):
        cls.ab = sb.load_ab()
        cls.rule = vr.GroupsRule(vr.load_groups_v2())
        cls.M = vs.cluster_draws(cls.ab.key.k, 50, 1)
        cls.code_of = {}
        for code in cls.ab.matcher.map_codes():
            for c in cls.ab.matcher.conditions_hit([code], "strict"):
                cls.code_of.setdefault(c, code)

    def score(self, esc, codes):
        a = sb.fixed_answers("x", self.ab, np.asarray(esc), codes)
        return vr.stats(vr.outcomes(a, self.ab, self.rule), self.ab, self.M)[0]

    def test_ideal_blanket_escalation_is_zero(self):
        codes = [[self.code_of[k.r10[0]]] if k.r10 else [] for k in self.ab.key.keys]
        p = self.score(np.ones(self.ab.key.n, bool), codes)
        self.assertAlmostEqual(p["score_mix"], 0.0)
        self.assertAlmostEqual(p["score_bal"], 0.0)
        self.assertEqual(p["partial"], 0.0)

    def test_always_routine(self):
        p = self.score(np.zeros(self.ab.key.n, bool), [[]] * self.ab.key.n)
        self.assertAlmostEqual(p["score_bal"], -600.0)
        self.assertAlmostEqual(p["U"], 1.0)

    def test_bare_escalation_is_a_miss(self):
        p = self.score(np.ones(self.ab.key.n, bool), [[]] * self.ab.key.n)
        self.assertAlmostEqual(p["U"], 1.0)
        self.assertAlmostEqual(p["bare"], 1.0)
        self.assertAlmostEqual(p["O"], 1.0)

    def test_partial_everywhere(self):
        # PE named on every case; the cases whose target is PE pass, the rest of SERIOUS are partials
        codes = [[self.code_of[PE]]] * self.ab.key.n
        p = self.score(np.ones(self.ab.key.n, bool), codes)
        pe = np.array([PE in k.r10 for k in self.ab.key.keys]) & self.ab.serious
        self.assertAlmostEqual(p["partial"], 1 - pe.sum() / self.ab.serious.sum())
        u_eff = p["partial"] * 1 / 7
        self.assertAlmostEqual(p["score_bal"], 100 * (1 - 1 - 7 * u_eff))
        self.assertAlmostEqual(p["score_bal@3.5"], 100 * (1 - 1 - 3.5 * p["partial"]))

    def test_arm4_flag_decides(self):
        preds = [{"case_id": c, "differential_diagnoses": [{"code": "I21", "p": 50}], "flag": "I63.9"}
                 for c in self.ab.key.case_ids]
        a = sb.row_answers(preds, "v7a4a", self.ab, "x")
        o = vr.outcomes(a, self.ab, self.rule)
        self.assertTrue(o.esc.all())  # stroke is a valid off-list reason
        self.assertNotIn(vr.PASS, o.outcome)  # the listed MI does not count: the flag is the reason
        o = vr.outcomes(a, self.ab, self.rule, offlist="routine")
        self.assertFalse(o.esc.any())


if __name__ == "__main__":
    unittest.main()
