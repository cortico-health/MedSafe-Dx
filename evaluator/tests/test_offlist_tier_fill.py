"""The fill of unscored off-list tiers (scripts/analysis/offlist_tier_fill.py) and the tier file it writes."""

import importlib.util
import unittest
from pathlib import Path

import pandas as pd

from evaluator import v03_valid_reason as vr

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "analysis" / "offlist_tier_fill.py"


def load_script():
    spec = importlib.util.spec_from_file_location("offlist_tier_fill", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rates(rows):
    """Rate rows keyed by dotless code or category: (key, n, admission, icu, n_icu)."""
    return pd.DataFrame([{"key": k, "n": n, "admission": a, "icu": i, "death": 0.0, "n_icu": ni}
                         for k, n, a, i, ni in rows], columns=["key", "n", "admission", "icu", "death", "n_icu"]
                        ).set_index("key")


def pre_row(prefix, tier="unscored", rule_path="unscored", source="unscored: NHAMCS ED 2016-2022 adults, n=0"):
    return {"icd10_prefix": prefix, "description": prefix, "tier": tier, "rule_path": rule_path, "pooled_tier": "",
            "admission": "", "icu": "", "death": "", "n": "0", "n_icu": "", "weak_evidence": "false", "source": source}


class TestFill(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = load_script()

    def basis(self, codes=(), groups=(), cats=(), ccsr=None):
        return self.m.Basis(rates(codes), rates(groups), rates(cats), self.m.CcsrIndex(ccsr or {}))

    def test_gem_keeps_the_prefix_every_target_shares(self):
        gem = self.m.gem_prefixes([
            "4465  M316    00000",                        # one target: 4 characters
            "2500  E119    10000", "2500  E118    10000",  # two targets in E11: the group only
            "0781  A630    10000", "0781  B078    10000",  # targets in two groups: dropped
            "V700  NoDx    11000",                         # no map
        ])
        self.assertEqual(gem, {"4465": "M316", "2500": "E11"})

    def test_ccsr_placeholder_category_falls_back_to_the_first_clinical_one(self):
        row = {"'Default CCSR CATEGORY OP'": "'XXX111'", "'CCSR CATEGORY 1'": "'NVS006'"}
        self.assertEqual(self.m.ccsr_category(row), "NVS006")
        self.assertEqual(self.m.ccsr_category({"'Default CCSR CATEGORY OP'": "'MUS024'"}), "MUS024")

    def test_prefix_category_is_the_plurality_of_its_codes(self):
        idx = self.m.CcsrIndex({"B440": "RSP016", "B441": "RSP016", "B442": "INF004", "B447": "INF004",
                                "B449": "INF004"})
        self.assertEqual(idx.category("B44"), "INF004")
        self.assertEqual(idx.category("B44.1"), "RSP016")
        self.assertIsNone(idx.category("B45"))

    def test_pooled_nhamcs_comes_first(self):
        b = self.basis(groups=[("M47", 33, 0.036, 0.0, 0)], cats=[("MUS010", 500, 0.6, 0.1, 50)],
                       ccsr={"M470": "MUS010"})
        r = self.m.fill_prefix("M47", b, [])
        self.assertEqual((r["tier"], r["tier_source"], r["n"]), (3, "nhamcs_pooled", 33))

    def test_ccsr_category_when_the_code_has_under_30_visits(self):
        b = self.basis(groups=[("M31", 12, 0.9, 0.5, 6)], cats=[("MUS024", 46, 0.179, 0.023, 1)],
                       ccsr={"M316": "MUS024", "M310": "MUS024"})
        r = self.m.fill_prefix("M31", b, [])
        self.assertEqual((r["tier"], r["tier_source"], r["n"]), (2, "ccsr", 46))
        self.assertIn("the prefix itself: n=12", r["source"])

    def test_category_under_30_visits_stays_unscored(self):
        b = self.basis(cats=[("CIR003", 29, 0.6, 0.1, 3)], ccsr={"I369": "CIR003"})
        r = self.m.fill_prefix("I36", b, [])
        self.assertEqual((r["tier"], r["tier_source"]), ("unscored", "unscored"))

    def test_weak_evidence_uses_the_same_definition(self):
        # tier 1 by the ICU clause alone on under 5 critical-care visits
        b = self.basis(cats=[("NVS006", 40, 0.3, 0.06, 3)], ccsr={"G129": "NVS006"})
        r = self.m.fill_prefix("G12", b, [])
        self.assertEqual((r["tier"], r["weak_evidence"]), (1, True))

    def test_only_fillable_rows_change(self):
        pre = [pre_row("I71", 1, "override", "Aortic [I71]; NHAMCS"),
               pre_row("M54", 3, "nhamcs", "nhamcs rule; NHAMCS"),
               pre_row("R57"),
               pre_row("A98", source="DDXPlus map first (Ebola); unscored: n=0")]
        b = self.basis(groups=[("R57", 300, 0.8, 0.4, 100), ("I71", 300, 0.0, 0.0, 0)],
                       cats=[("INF008", 900, 0.02, 0.0, 0)], ccsr={"A983": "INF008"})
        out = {r["icd10_prefix"]: r for r in self.m.fill_table(pre, b, [], set(), {})}
        self.assertEqual((out["I71"]["tier"], out["I71"]["tier_source"]), (1, "override"))
        self.assertEqual((out["M54"]["tier"], out["M54"]["tier_source"]), (3, "nhamcs"))
        self.assertEqual((out["R57"]["tier"], out["R57"]["tier_source"]), ("unscored", "unscored"))
        self.assertEqual((out["A98"]["tier"], out["A98"]["tier_source"]), (3, "ccsr"))
        self.assertTrue(out["A98"]["source"].startswith("DDXPlus map first (Ebola); ccsr rule"))

    def test_finer_prefix_with_a_different_tier_gets_its_own_row(self):
        pre = [pre_row("B44")]
        b = self.basis(cats=[("INF004", 212, 0.007, 0.0, 0), ("RSP016", 215, 0.314, 0.06, 13)],
                       ccsr={"B440": "RSP016", "B441": "RSP016", "B442": "INF004", "B447": "INF004", "B449": "INF004"})
        out = self.m.fill_table(pre, b, [], {"B44.1", "B44.9"}, {})
        self.assertEqual([(r["icd10_prefix"], r["tier"]) for r in out], [("B44", 3), ("B44.1", 1)])

    def test_pooled_group_follows_the_group_tier_rule(self):
        # the pooled group is tier 1, but tier-2 sub-codes hold most of its visits
        b = self.basis(codes=[("Q501", 20, 0.9, 0.5, 10), ("Q509", 40, 0.2, 0.0, 0)],
                       groups=[("Q50", 60, 0.45, 0.17, 10)])
        r = self.m.fill_prefix("Q50", b, [])
        self.assertEqual((r["tier"], r["pooled_tier"], r["tier_source"]), (2, 1, "nhamcs_pooled"))


class TestFilledTierFile(unittest.TestCase):
    """spec/offlist_tiers_nhamcs.csv after the fill, and the kept pre-fill table."""

    def test_scored_rows_keep_their_tier(self):
        pre, post = (pd.read_csv(ROOT / "spec" / f, dtype=str).set_index("icd10_prefix")
                     for f in ("offlist_tiers_nhamcs_pre_fill.csv", "offlist_tiers_nhamcs.csv"))
        scored = pre[pre["tier"] != "unscored"]
        self.assertTrue((post.loc[scored.index, "tier"] == scored["tier"]).all())
        self.assertTrue(set(post["tier_source"]) <= {"override", "nhamcs", "nhamcs_pooled", "ccsr", "unscored"})

    def test_symptom_codes_stay_unscored(self):
        rule = vr.TierFileRule()
        self.assertEqual(rule.tier("R57.0")[0], vr.UNSCORED)
        self.assertEqual(rule.tier("R07.9")[0], vr.UNSCORED)

    def test_examples(self):
        rule, pre = vr.TierFileRule(), vr.TierFileRule(ROOT / "spec" / "offlist_tiers_nhamcs_pre_fill.csv")
        self.assertEqual(pre.tier("M31.6")[0], vr.UNSCORED)
        self.assertEqual(rule.tier("M31.6")[0], "2")
        self.assertEqual(rule.tier("M47.0")[0], "3")
        self.assertEqual(rule.tier("B44.1")[0], "1")
        self.assertEqual(rule.tier("E11.9")[0], "2")


if __name__ == "__main__":
    unittest.main()
