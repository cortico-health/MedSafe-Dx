import unittest

from evaluator.schemas_v03 import parse_flags, parse_v03


def pred(**over):
    base = {
        "case_id": "c1",
        "serious_concern": "YES",
        "flags": ["I21.4", "I26.99"],
        "differential": [{"code": "I21.4", "p": 40}, {"code": "I20.0", "p": 30}, {"code": "K21.9", "p": 10}],
        "p_serious": 70,
    }
    base.update(over)
    return base


class TestWellFormed(unittest.TestCase):
    def test_clean_prediction(self):
        r = parse_v03(pred())
        self.assertTrue(r.readable)
        self.assertEqual(r.serious_concern, "YES")
        self.assertIs(r.yes, True)
        self.assertEqual(r.flags, ["I214", "I2699"])
        self.assertEqual(r.flags_status, "ok")
        self.assertEqual([e.p for e in r.differential], [40, 30, 10])
        self.assertEqual((r.p_serious, r.p_serious_status), (70, "ok"))
        self.assertEqual(r.rule_log, [])

    def test_no_answer(self):
        r = parse_v03(pred(serious_concern="NO", flags=[]))
        self.assertIs(r.yes, False)
        self.assertEqual(r.flags_status, "empty")


class TestSeriousConcern(unittest.TestCase):
    def test_case_and_spaces_are_ignored(self):
        for v, want in ((" yes ", "YES"), ("No", "NO")):
            self.assertEqual(parse_v03(pred(serious_concern=v)).serious_concern, want)

    def test_missing_makes_case_unreadable(self):
        p = pred()
        del p["serious_concern"]
        r = parse_v03(p)
        self.assertFalse(r.readable)
        self.assertEqual(r.unreadable_reason, "serious_concern:missing")
        self.assertIsNone(r.yes)
        self.assertEqual(r.flags, ["I214", "I2699"])  # other fields still parse

    def test_unparseable_makes_case_unreadable(self):
        for bad in ("MAYBE", "YES/NO", True, 1, "", ["YES"], "ESCALATE_NOW"):
            r = parse_v03(pred(serious_concern=bad))
            self.assertFalse(r.readable, bad)
            self.assertEqual(r.unreadable_reason, "serious_concern:unparseable", bad)

    def test_harness_error_and_non_object(self):
        self.assertEqual(parse_v03(pred(error="timeout")).unreadable_reason, "harness:timeout")
        self.assertEqual(parse_v03("YES").unreadable_reason, "not_an_object")


class TestFlags(unittest.TestCase):
    def test_only_first_five_read(self):
        codes = ["I21.4", "I26.99", "J93.0", "T78.0", "I47.1", "J81.0", "A98.4"]
        r = parse_v03(pred(flags=codes))
        self.assertEqual(r.flags, ["I214", "I2699", "J930", "T780", "I471"])
        self.assertIn("flags:truncated", r.rule_log)

    def test_truncation_happens_before_cleaning(self):
        # the sixth entry is not promoted when an earlier one is dropped
        flags, _ = parse_flags(["I21.4", "junk", "I21.4", "J93.0", "T78.0", "I47.1"], log := [])
        self.assertEqual(flags, ["I214", "J930", "T780"])
        self.assertIn("flags:invalid_code", log)
        self.assertIn("flags:duplicate", log)

    def test_normalisation_and_text(self):
        flags, log = parse_flags(["i21.9 - MI", " J45 ", {"code": "I26.9"}, "I21.9 (NSTEMI)"], [])
        self.assertEqual(flags, ["I219", "J45", "I269"])
        flags, log = [], []
        parse_flags(["I21.9 - MI"], log)
        self.assertIn("flags:text_stripped", log)

    def test_invalid_entries_dropped(self):
        flags, status = parse_flags(["myocardial infarction", 42, None, "", "12.3"], [])
        self.assertEqual((flags, status), ([], "empty"))

    def test_missing_or_not_list_means_no_flags_but_readable(self):
        p = pred()
        del p["flags"]
        r = parse_v03(p)
        self.assertTrue(r.readable)
        self.assertEqual((r.flags, r.flags_status), ([], "missing"))
        r = parse_v03(pred(flags="I21.9, I26.9"))
        self.assertTrue(r.readable)
        self.assertEqual((r.flags, r.flags_status), ([], "unreadable:not_a_list"))

    def test_differential_never_substitutes(self):
        r = parse_v03(pred(flags=None))
        self.assertEqual(r.flags, [])
        self.assertEqual(len(r.differential), 3)


class TestV02RulesReused(unittest.TestCase):
    def test_fractions_scaled(self):
        r = parse_v03(pred(differential=[{"code": "I21", "p": 0.6}, {"code": "I20.0", "p": 0.4}], p_serious=0.3))
        self.assertEqual([e.p for e in r.differential], [60, 40])
        self.assertAlmostEqual(r.p_serious, 30.0)

    def test_sum_rules(self):
        r = parse_v03(pred(differential=[{"code": "I21", "p": 52}, {"code": "I20.0", "p": 52}]))
        self.assertEqual(r.differential_p_status, "rescaled")
        r = parse_v03(pred(differential=[{"code": "I21", "p": 60}, {"code": "I20.0", "p": 50}]))
        self.assertEqual(r.differential_p_status, "unreadable:sum_over_105")
        self.assertEqual([e.code for e in r.differential], ["I21", "I20.0"])  # codes kept

    def test_missing_p_serious_stays_readable(self):
        p = pred()
        del p["p_serious"]
        r = parse_v03(p)
        self.assertTrue(r.readable)
        self.assertEqual(r.p_serious_status, "missing")


if __name__ == "__main__":
    unittest.main()
