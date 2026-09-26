import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

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

    def test_weak_evidence_rows(self):
        p = tier_file("icd10_prefix,tier,weak_evidence\nK86,1,true\nK85,1,false\n")
        self.assertEqual(vr.TierFileRule(p).tier("K86.1")[0], "1")
        weak = vr.TierFileRule(p, exclude_weak=True)
        self.assertEqual((weak.tier("K86.1")[0], weak.tier("K85.9")[0]), (vr.WEAK, "1"))
        self.assertEqual(self.kind(["K86.1"], rule=weak), vr.NONE)
        self.assertEqual(self.kind(["K85.9"], rule=weak), vr.OFFLIST)

    def test_tier_file_needs_columns(self):
        with self.assertRaises(ValueError):
            vr.TierFileRule(tier_file("code_x,level\nA00,1\n"))

    def test_truth_credit_a4(self):
        # SLE (tier 3) named on a case a selection rule made SERIOUS with a PE target
        sle = ["M32.9"]
        self.assertEqual(self.kind(sle, targets=(PE,)), vr.NONE)  # no credit: the pre-A4 reading
        self.assertEqual(self.kind(sle, targets=(PE,), truth="SLE", truth_credit=vr.PARTIAL_OUT), vr.TRUTH)
        self.assertEqual(self.kind(sle, targets=(PE,), truth="SLE", truth_credit=vr.PASS), vr.TARGET)
        self.assertEqual(self.kind(sle, targets=(PE,), truth="SLE", truth_credit=None), vr.NONE)
        # the credit is for the truth only, and a stronger reason still wins
        self.assertEqual(self.kind(["J40"], targets=(PE,), truth="SLE", truth_credit=vr.PARTIAL_OUT), vr.NONE)
        self.assertEqual(self.kind(["I26"], targets=(PE,), truth="SLE", truth_credit=vr.PARTIAL_OUT), vr.TARGET)
        self.assertEqual(self.kind(["I21"], targets=(PE,), truth="SLE", truth_credit=vr.PARTIAL_OUT), vr.OTHER_TIER1)
        r = vr.reason(sle, [PE], self.m, self.tiers, self.groups, truth="SLE", truth_credit=vr.PASS)
        self.assertEqual(r.tier1, ("SLE",))


class TestCosts(unittest.TestCase):
    def test_case_cost(self):
        o = vr.Outcome(np.array([True] * 6), [vr.Reason(vr.NONE)] * 6,
                       [vr.PASS, vr.PARTIAL_OUT, vr.MISS_OUT, vr.BARE, vr.BENIGN_ESC, vr.NOT_SCORED])
        self.assertEqual(vr.case_cost(o).tolist(), [0, 1, 7, 7, 1, 0])
        self.assertEqual(vr.case_cost(o, 3.5).tolist(), [0, 3.5, 7, 7, 1, 0])
        self.assertEqual(vr.case_cost(o, np.array([9, 2, 9, 9, 9, 9])).tolist(), [0, 2, 7, 7, 1, 0])

    def test_boerhaave_pair_row(self):
        mi = vr.Reason(vr.OTHER_TIER1, (MI,))
        dissection = vr.Reason(vr.OFFLIST, (), (("I7100", "Aortic aneurysm and dissection"),))
        stroke = vr.Reason(vr.OFFLIST, (), (("I639", "Cerebral infarction"),))
        o = vr.Outcome(np.array([True] * 5), [mi, dissection, stroke, mi, vr.Reason(vr.TARGET)],
                       [vr.PARTIAL_OUT, vr.PARTIAL_OUT, vr.PARTIAL_OUT, vr.PARTIAL_OUT, vr.PASS])
        truths = [vr.BOERHAAVE, vr.BOERHAAVE, vr.BOERHAAVE, PE, vr.BOERHAAVE]
        self.assertEqual(vr.pair_partials(o, truths).tolist(), [3.5, 1, 3.5, 1, 1])
        self.assertEqual(vr.case_cost(o, vr.pair_partials(o, truths)).tolist(), [3.5, 1, 3.5, 1, 0])


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

    def test_a4_truth_flag_on_a_promoted_case(self):
        # A SERIOUS case with a tier-3 truth (a DXA-derived target under A3) stands in for a case a selection
        # rule promoted; the flag names the truth. Without A4 that is a miss; with PARTIAL_OUT credit it is an
        # escalation costing a partial; with PASS credit it passes.
        n = self.ab.key.n
        for i, k in enumerate(self.ab.key.keys):
            if self.ab.klass[i] != "serious" or k.truth_tier != 3:
                continue
            preds = [{"case_id": c, "differential_diagnoses": [{"code": "J40", "p": 50}], "flag": self.code_of[k.truth]}
                     for c in self.ab.key.case_ids]
            a = sb.row_answers(preds, "v7a4a", self.ab, "x")
            base = vr.outcomes(a, self.ab, self.rule)
            if base.outcome[i] == vr.MISS_OUT:  # the truth's code is no reason under the base rule
                break
        else:
            self.fail("no SERIOUS tier-3 case whose truth flag is a miss")
        self.assertFalse(base.esc[i])
        credit = [None] * n
        credit[i] = vr.PARTIAL_OUT
        o = vr.outcomes(a, self.ab, self.rule, truth_credit=credit)
        self.assertEqual(o.outcome[i], vr.PARTIAL_OUT)
        self.assertEqual(o.reasons[i].kind, vr.TRUTH)
        self.assertTrue(o.esc[i])
        self.assertEqual(vr.case_cost(o)[i], 1.0)
        self.assertEqual([x for j, x in enumerate(o.outcome) if j != i], [x for j, x in enumerate(base.outcome) if j != i])
        p = vr.stats(o, self.ab, self.M)[0]
        self.assertAlmostEqual(p["partial_truth"], 1 / self.ab.serious.sum())
        credit[i] = vr.PASS
        o = vr.outcomes(a, self.ab, self.rule, truth_credit=credit)
        self.assertEqual(o.outcome[i], vr.PASS)
        self.assertEqual(o.reasons[i].kind, vr.TARGET)


def stub_set(r10s, serious, canonical, tiers):
    """The fields of an `ABSet` that `zero_reference` reads."""
    key = SimpleNamespace(keys=[SimpleNamespace(r10=list(r)) for r in r10s], tiers=tiers)
    return SimpleNamespace(key=key, serious=np.array(serious), matcher=SimpleNamespace(cmap=SimpleNamespace(canonical=canonical)))


class TestZeroReference(unittest.TestCase):
    """Amendment A2: 0 is always escalate with one fixed flag on the most common tier-1 target; 100 is perfect."""

    @classmethod
    def setUpClass(cls):
        cls.ab = sb.load_ab()
        cls.M = vs.cluster_draws(cls.ab.key.k, 50, 1)
        cls.code_of = {}
        for code in cls.ab.matcher.map_codes():
            for c in cls.ab.matcher.conditions_hit([code], "strict"):
                cls.code_of.setdefault(c, code)

    def score(self, esc, codes):
        a = sb.fixed_answers("x", self.ab, np.asarray(esc), codes)
        return vr.stats(vr.outcomes(a, self.ab, vr.default_rule()), self.ab, self.M)

    def test_chosen_from_the_key(self):
        # under amendment A3's classes MI and PSVT tie at 8 SERIOUS cases; "Possible NSTEMI / STEMI" sorts first
        # by name with case ignored ("PSVT" would sort first case-sensitively)
        self.assertEqual(vr.zero_reference(self.ab), ("I21", MI, 8))

    def test_most_cases_then_name_ignoring_case(self):
        tiers = {"b": 1, "A": 1, "C": 1, "t2": 2}
        canon = {"b": "x01", "A": "y02", "C": "z03", "t2": "w04"}
        most = stub_set([["C"], ["C", "b"], ["A", "t2"], ["t2"], ["t2"]], [1, 1, 1, 1, 1], canon, tiers)
        self.assertEqual(vr.zero_reference(most), ("Z03", "C", 2))  # tier-2 targets are never chosen
        tie = stub_set([["C"], ["b"], ["A"], ["b"]], [1, 1, 1, 0], canon, tiers)
        self.assertEqual(vr.zero_reference(tie)[1], "A")  # A, b, C: case ignored
        with self.assertRaises(ValueError):
            vr.zero_reference(stub_set([["b"]], [0], canon, tiers))

    def test_zero_reference_scores_zero(self):
        code = vr.zero_reference(self.ab)[0]
        p, d = self.score(np.ones(self.ab.key.n, bool), [[code]] * self.ab.key.n)
        for m in ("score_z_bal", "score_z_mix", "score_z_bal@3.5", "score_z_mix@2"):
            self.assertAlmostEqual(p[m], 0.0)
            self.assertTrue(np.allclose(d[m][np.isfinite(d[m])], 0.0))
        self.assertLess(p["score_bal"], 0)  # below draft 3's zero: it names few SERIOUS cases' targets

    def test_perfect_scores_100(self):
        s = self.ab.serious
        codes = [[self.code_of[k.r10[0]]] if (x and k.r10) else [] for k, x in zip(self.ab.key.keys, s)]
        p, _ = self.score(s, codes)
        self.assertAlmostEqual(p["score_z_bal"], 100.0)
        self.assertAlmostEqual(p["score_z_mix"], 100.0)

    def test_rescaling_is_affine_in_cost(self):
        z, _ = self.score(np.ones(self.ab.key.n, bool), [[vr.zero_reference(self.ab)[0]]] * self.ab.key.n)
        p, _ = self.score(np.zeros(self.ab.key.n, bool), [[]] * self.ab.key.n)
        for new, old in (("score_z_bal", "score_bal"), ("score_z_mix", "score_mix")):
            c, c0 = 1 - p[old] / 100, 1 - z[old] / 100
            self.assertAlmostEqual(p[new], 100 * (c0 - c) / c0)

    def test_extra_function_applies_to_the_zero_reference(self):
        code = vr.zero_reference(self.ab)[0]
        a = sb.fixed_answers("x", self.ab, np.ones(self.ab.key.n, bool), [[code]] * self.ab.key.n)
        o = vr.outcomes(a, self.ab, vr.default_rule())
        p, _ = vr.stats(o, self.ab, self.M, extra={"pair": lambda x: vr.pair_partials(x, self.ab.key.truth)})
        self.assertAlmostEqual(p["score_z_bal@pair"], 0.0)


class TestJustifiedArmLabels(unittest.TestCase):
    """Arms 4aj and 4bj (amendment A1) carry labels and paired comparisons in the scorer."""

    def test_labels_and_comparisons(self):
        self.assertEqual((sb.ARM_LABELS["v7a4aj"], sb.ARM_LABELS["v7a4bj"]), ("4aj", "4bj"))
        for pair in (("v7a4aj", "v7a4bj"), ("v7a4aj", "v7a4a"), ("v7a4bj", "v7a4b")):
            self.assertIn(pair, sb.COMPARISONS)
        from evaluator.schemas_v03b import ARMS, FLAG_ARMS
        self.assertTrue(set(sb.ARM_LABELS) >= set(ARMS))
        self.assertIn("v7a4aj", FLAG_ARMS)


class TestArm4cAnd2j(unittest.TestCase):
    """Arm 4c and variant 2j are kept in the repo, unused; their parser still reads them."""

    def test_parse(self):
        from evaluator.schemas_v03b import parse_v03b
        d = [{"code": "J20.9", "p": 60}]
        p = parse_v03b({"case_id": "c", "differential_diagnoses": d, "flag": "I26.99",
                        "justification": " PE must be excluded. "}, "v7a4c")
        self.assertEqual((p.flag, p.justification, p.flag_in_list), ("I2699", "PE must be excluded.", False))
        p = parse_v03b({"case_id": "c", "differential_diagnoses": d, "escalation_decision": "ROUTINE_CARE"}, "v7a2j")
        self.assertFalse(p.escalate)
        self.assertIn("justification:missing", p.rule_log)


if __name__ == "__main__":
    unittest.main()
