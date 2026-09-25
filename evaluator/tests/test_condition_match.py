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
        self.assertFalse(self.m.matches("F41.0", "PSVT"))  # panic attack
        self.assertFalse(self.m.matches("K21.9", "Stable angina"))

    def test_broader_calls_count(self):
        self.assertTrue(self.m.matches("I49.9", "PSVT"))
        self.assertTrue(self.m.matches("R00.0", "PSVT"))
        self.assertTrue(self.m.matches("I20.9", "Unstable angina"))
        self.assertFalse(self.m.matches("I20.9", "Unstable angina", "strict"))

    def test_mi_code_matches_the_ischaemia_family_under_standard(self):
        # draft 3 family rows: an MI code names the three ischaemia conditions; strict still names MI alone
        self.assertEqual(self.m.tier1_hit(["I21.9"], TIERS), {MI, "Stable angina", "Unstable angina"})
        self.assertEqual(self.m.tier1_hit(["I21.9"], TIERS, "strict"), {MI})
        self.assertEqual(len(self.m.tier1_hit(["I21.9"], TIERS, "lenient")), 8)

    def test_family_rows(self):
        """docs/v0.3-validity-review.md section 1.1, added as broader relations (spec draft 3 section 4)."""
        cases = [("I50.9", "Acute pulmonary edema"), ("I48.91", "PSVT"), ("R00.0", "PSVT"),
                 ("T78.2", "Scombroid food poisoning"), ("T61.1", "Anaphylaxis"), ("R06.1", "Larygospasm"),
                 ("J38.4", "Larygospasm"), ("J05.1", "Croup"), ("J44.1", "Bronchospasm / acute asthma exacerbation"),
                 ("I30.9", "Myocarditis"), ("G24.01", "Acute dystonic reactions"), ("I82.4", "Pulmonary embolism")]
        for code, cond in cases:
            with self.subTest(code=code, cond=cond):
                self.assertEqual(self.m.relation(code, cond), "broader")
                self.assertTrue(self.m.matches(code, cond))
                self.assertFalse(self.m.matches(code, cond, "strict"))

    def test_family_exceptions_stay_related(self):
        # haemopericardium is related to pericarditis, so pericarditis's broader I31 row does not carry it to myocarditis
        self.assertEqual(self.m.relation("I31.2", "Myocarditis"), "related")
        self.assertFalse(self.m.matches("I31.2", "Myocarditis"))

    def test_family_rows_are_idempotent(self):
        """Running the builder on the committed map (in a temporary copy) changes nothing."""
        import importlib.util
        import shutil
        spec = importlib.util.spec_from_file_location("add_fam", ROOT / "scripts/add_v03_family_rows.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as d:
            copy = Path(d) / "map.csv"
            shutil.copy(ROOT / "spec/ddxplus_icd10_map.csv", copy)
            mod.main(copy)
            self.assertEqual(copy.read_bytes(), (ROOT / "spec/ddxplus_icd10_map.csv").read_bytes())

    def test_strict_codes_are_unambiguous(self):
        s = multiplicity_summary(self.m, TIERS)
        self.assertEqual(s["strict"]["codes_matching_more_than_one_tier1"], 0)
        self.assertLess(s["standard"]["codes_matching_more_than_one_tier1"], s["lenient"]["codes_matching_more_than_one_tier1"])


if __name__ == "__main__":
    unittest.main()
