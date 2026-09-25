import csv
import tempfile
import unittest
from pathlib import Path

from evaluator.condition_match import POLICIES, FlagMatcher, multiplicity_summary, relations_for

ROOT = Path(__file__).resolve().parents[2]
TIERS = {r["condition"]: int(r["final_tier"])
         for r in csv.DictReader(open(ROOT / "spec/dangerous_if_missed_tiers_v03.csv", encoding="utf-8"))}
MI = "Possible NSTEMI / STEMI"

TOY_MAP = """condition,ddxplus_icd10,code,system,relation,source_url,note
Alpha,A10,A10,WHO,equivalent,,DDXPlus code
Alpha,A10,A10.1,WHO,narrower,,
Alpha,A10,X1,WHO,broader,,
Alpha,A10,B20,WHO,related,,
Beta,B20,B20,WHO,equivalent,,DDXPlus code
Beta,B20,X1,WHO,broader,,
Gamma,C30,C30,WHO,equivalent,,DDXPlus code
Gamma,C30,A10,WHO,related,,
"""
TOY_TIERS = {"Alpha": 1, "Beta": 1, "Gamma": 3}


class TestToyMap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
        cls.tmp.write(TOY_MAP)
        cls.tmp.close()
        cls.m = FlagMatcher(path=Path(cls.tmp.name))

    def test_policies_nest(self):
        self.assertEqual(relations_for("strict"), ("equivalent", "narrower"))
        self.assertLess(set(POLICIES["strict"]), set(POLICIES["standard"]))
        self.assertLess(set(POLICIES["standard"]), set(POLICIES["lenient"]))
        with self.assertRaises(ValueError):
            relations_for("loose")

    def test_equivalent_narrower_broader_count_related_does_not(self):
        m = self.m
        self.assertTrue(m.matches("A10", "Alpha"))
        self.assertTrue(m.matches("A10.19", "Alpha"))  # longest prefix A10.1: narrower
        self.assertTrue(m.matches("X1", "Alpha"))  # broader
        self.assertFalse(m.matches("B20", "Alpha"))  # related
        self.assertTrue(m.matches("B20", "Alpha", "lenient"))
        self.assertFalse(m.matches("X1", "Alpha", "strict"))

    def test_off_map_and_empty_codes_match_nothing(self):
        for policy in POLICIES:
            self.assertEqual(self.m.conditions_hit(["Z99.9", "", None], policy), set())

    def test_conditions_hit_and_tier1(self):
        self.assertEqual(self.m.conditions_hit(["A10"]), {"Alpha"})
        self.assertEqual(self.m.conditions_hit(["A10"], "lenient"), {"Alpha", "Gamma"})
        self.assertEqual(self.m.tier1_hit(["A10"], TOY_TIERS, "lenient"), {"Alpha"})
        self.assertEqual(self.m.tier1_hit(["X1.5"], TOY_TIERS), {"Alpha", "Beta"})  # broader for both

    def test_multiplicity(self):
        s = multiplicity_summary(self.m, TOY_TIERS)
        self.assertEqual(s["strict"]["codes_matching_more_than_one_tier1"], 0)
        self.assertEqual(s["standard"]["codes_matching_more_than_one_tier1"], 1)  # X1
        self.assertEqual(s["lenient"]["codes_matching_more_than_one_tier1"], 2)  # X1, B20
        self.assertEqual(s["lenient"]["max_tier1_per_code"], 2)


class TestRealMap(unittest.TestCase):
    """The policy calls named in docs/v0.3-adversarial-review.md section 3."""

    @classmethod
    def setUpClass(cls):
        cls.m = FlagMatcher()

    def test_related_codes_stop_counting(self):
        self.assertFalse(self.m.matches("K21.9", MI))  # GERD is related to MI
        self.assertTrue(self.m.matches("K21.9", MI, "lenient"))
        self.assertFalse(self.m.matches("I48.91", "PSVT"))  # AF
        self.assertFalse(self.m.matches("F41.0", "PSVT"))  # panic attack

    def test_broader_calls_count(self):
        self.assertTrue(self.m.matches("I49.9", "PSVT"))
        self.assertTrue(self.m.matches("R00.0", "PSVT"))
        self.assertTrue(self.m.matches("I20.9", "Unstable angina"))
        self.assertFalse(self.m.matches("I20.9", "Unstable angina", "strict"))

    def test_mi_code_matches_only_mi_under_standard(self):
        self.assertEqual(self.m.tier1_hit(["I21.9"], TIERS), {MI})
        self.assertEqual(len(self.m.tier1_hit(["I21.9"], TIERS, "lenient")), 8)

    def test_strict_codes_are_unambiguous(self):
        s = multiplicity_summary(self.m, TIERS)
        self.assertEqual(s["strict"]["codes_matching_more_than_one_tier1"], 0)
        self.assertLess(s["standard"]["codes_matching_more_than_one_tier1"], s["lenient"]["codes_matching_more_than_one_tier1"])


if __name__ == "__main__":
    unittest.main()
