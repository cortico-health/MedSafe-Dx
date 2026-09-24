import unittest

from evaluator.schemas_v02 import parse_v02, scale_fractions


def pred(**over):
    base = {
        "case_id": "c1",
        "differential": [{"code": "I21.4", "p": 40}, {"code": "I20.0", "p": 30}, {"code": "K21.9", "p": 10}],
        "p_serious": 70,
        "escalation_decision": "ESCALATE_NOW",
    }
    base.update(over)
    return base


class TestWellFormed(unittest.TestCase):
    def test_clean_prediction(self):
        r = parse_v02(pred())
        self.assertTrue(r.readable)
        self.assertEqual((r.p_serious, r.p_serious_status), (70, "ok"))
        self.assertEqual(r.differential_p_status, "ok")
        self.assertEqual([e.p for e in r.differential], [40, 30, 10])  # remainder 20 means "other"
        self.assertEqual(r.differential[0].code, "I21.4")  # the top diagnosis is the first entry
        self.assertEqual(r.escalation_decision, "ESCALATE_NOW")
        self.assertEqual(r.rule_log, [])


class TestProbabilityValues(unittest.TestCase):
    def test_range_0_to_100(self):
        for bad in (-5, 105):
            r = parse_v02(pred(p_serious=bad))
            self.assertEqual(r.p_serious_status, "unreadable:out_of_range", bad)
        r = parse_v02(pred(differential=[{"code": "I21", "p": 120}]))
        self.assertEqual(r.differential_p_status, "unreadable:out_of_range")
        for bad in (True, "high", [50], {"p": 50}):
            r = parse_v02(pred(p_serious=bad))
            self.assertEqual(r.p_serious_status, "unreadable:not_a_number", bad)

    def test_numeric_strings_and_non_integers_are_accepted_and_logged(self):
        r = parse_v02(pred(p_serious="15%"))
        self.assertEqual(r.p_serious, 15.0)
        r = parse_v02(pred(p_serious=33.3))
        self.assertEqual(r.p_serious_status, "ok")
        self.assertIn("p_serious:non_integer", r.rule_log)


class TestFractionScaling(unittest.TestCase):
    def test_scale_rule(self):
        self.assertEqual(scale_fractions([0.5, 0.5]), ([50.0, 50.0], True))
        self.assertEqual(scale_fractions([0.5, 0.55])[1], True)  # 1.05, the limit
        self.assertEqual(scale_fractions([0.5, 0.56])[1], False)  # 1.06, over
        self.assertEqual(scale_fractions([0.5, 0.3])[1], True)  # below 1: the remainder is "other"
        self.assertEqual(scale_fractions([1, 0, 0])[1], True)  # 1 = 100%
        self.assertEqual(scale_fractions([2, 0.0])[1], False)  # a value above 1 means percentages

    def test_p_serious_fraction(self):
        r = parse_v02(pred(p_serious=0.3))
        self.assertAlmostEqual(r.p_serious, 30.0)
        self.assertIn("p_serious:fraction_scaled", r.rule_log)
        r = parse_v02(pred(p_serious=1))
        self.assertEqual(r.p_serious, 100.0)
        r = parse_v02(pred(p_serious=5))
        self.assertEqual(r.p_serious, 5.0)

    def test_differential_fractions(self):
        r = parse_v02(pred(differential=[{"code": "I21", "p": 0.85}, {"code": "I20.0", "p": 0.15}]))
        self.assertIn("differential:fraction_scaled", r.rule_log)
        self.assertAlmostEqual(r.differential[0].p, 85.0)
        r = parse_v02(pred(differential=[{"code": "I21", "p": 0.6}, {"code": "I20.0", "p": 0.3}]))
        self.assertIn("differential:fraction_scaled", r.rule_log)
        self.assertAlmostEqual(r.differential[0].p, 60.0)


class TestPSerious(unittest.TestCase):
    def test_missing_keeps_case(self):
        p = pred()
        del p["p_serious"]
        r = parse_v02(p)
        self.assertTrue(r.readable)
        self.assertIsNone(r.p_serious)
        self.assertEqual(r.p_serious_status, "missing")
        self.assertIn("p_serious:missing", r.rule_log)

    def test_unreadable_keeps_case(self):
        r = parse_v02(pred(p_serious="likely"))
        self.assertTrue(r.readable)
        self.assertIsNone(r.p_serious)
        self.assertEqual(r.escalation_decision, "ESCALATE_NOW")

    def test_no_urgency_fields(self):
        r = parse_v02(pred())
        self.assertFalse(hasattr(r, "urgency_level"))
        self.assertFalse(hasattr(r, "urgency_probabilities"))


class TestDifferential(unittest.TestCase):
    def _d(self, ps):
        return parse_v02(pred(differential=[{"code": c, "p": p} for c, p in zip(["I21", "I20.0", "K21.9", "J18.9", "R07.9"], ps)]))

    def test_sum_limit(self):
        self.assertEqual(self._d([60, 40]).differential_p_status, "ok")
        r = self._d([60, 45])  # 105: allowed, rescaled to 100
        self.assertEqual(r.differential_p_status, "rescaled")
        self.assertAlmostEqual(sum(e.p for e in r.differential), 100.0)
        r = self._d([60, 46])  # 106: probabilities unreadable, codes kept
        self.assertEqual(r.differential_p_status, "unreadable:sum_over_105")
        self.assertEqual([e.code for e in r.differential], ["I21", "I20.0"])
        self.assertTrue(all(e.p is None for e in r.differential))
        self.assertTrue(r.readable)

    def test_truncates_to_five(self):
        r = self._d([20, 20, 20, 20, 10])
        self.assertEqual(len(r.differential), 5)
        r = parse_v02(pred(differential=[{"code": "I21", "p": 10}] * 7))
        self.assertEqual(len(r.differential), 5)
        self.assertIn("differential:truncated", r.rule_log)

    def test_missing_p(self):
        r = parse_v02(pred(differential=[{"code": "I21", "p": 50}, {"code": "I20.0"}]))
        self.assertEqual(r.differential_p_status, "unreadable:missing_p")
        self.assertEqual(len(r.differential), 2)

    def test_invalid_code_is_kept_and_logged(self):
        r = parse_v02(pred(differential=[{"code": "heart attack", "p": 50}]))
        self.assertFalse(r.differential[0].code_valid)
        self.assertIn("differential:invalid_code", r.rule_log)

    def test_v4_field_name(self):
        p = pred()
        p["differential_diagnoses"] = p.pop("differential")
        r = parse_v02(p)
        self.assertEqual(len(r.differential), 3)
        self.assertIn("differential:v4_field_name", r.rule_log)

    def test_missing_differential_keeps_case(self):
        p = pred()
        del p["differential"]
        r = parse_v02(p)
        self.assertTrue(r.readable)
        self.assertEqual(r.differential_p_status, "missing")


class TestEscalationAndHarness(unittest.TestCase):
    def test_missing_escalation_makes_case_unreadable(self):
        p = pred()
        del p["escalation_decision"]
        r = parse_v02(p)
        self.assertFalse(r.readable)
        self.assertIsNone(r.escalation_decision)
        self.assertEqual(r.unreadable_reason, "escalation_decision:missing")
        for bad in ("MAYBE", "", 1, True, ["ESCALATE_NOW"]):
            r = parse_v02(pred(escalation_decision=bad))
            self.assertFalse(r.readable, bad)
            self.assertEqual(r.unreadable_reason, "escalation_decision:unparseable")

    def test_accepted_escalation_forms(self):
        r = parse_v02(pred(escalation_decision=" routine_care "))
        self.assertTrue(r.readable)
        self.assertEqual(r.escalation_decision, "ROUTINE_CARE")

    def test_unreadable_case_keeps_other_fields(self):
        r = parse_v02(pred(escalation_decision=None))
        self.assertEqual(r.p_serious, 70)
        self.assertEqual(len(r.differential), 3)

    def test_harness_error_is_unreadable(self):
        r = parse_v02({"case_id": "c1", "error": "json_parse_failure", "raw_response": "{"})
        self.assertFalse(r.readable)
        self.assertEqual(r.unreadable_reason, "harness:json_parse_failure")


if __name__ == "__main__":
    unittest.main()
