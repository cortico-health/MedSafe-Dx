"""The group-tier rule of scripts/analysis/offlist_severity_nhamcs.py and the tier file it writes."""

import importlib.util
import unittest
from pathlib import Path

import pandas as pd

from evaluator import v03_valid_reason as vr

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "analysis" / "offlist_severity_nhamcs.py"
RULE = {"icu_ge": 0.05, "death_ge": None, "admit_ge": 0.5, "admit_lt": 0.05}


def load_script():
    spec = importlib.util.spec_from_file_location("offlist_severity_nhamcs", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def table(rows):
    """NHAMCS primary-diagnosis code rows: (code, n, admission, icu, n_icu)."""
    return pd.DataFrame([{"icd10": c, "level": "code", "scope": "primary", "n": n, "admission": a, "icu": i,
                          "death": 0.0, "n_icu": ni} for c, n, a, i, ni in rows])


def group_row(n, tier):
    return {"icd10_prefix": "E11", "tier": tier, "rule_path": "nhamcs", "n": n, "weak_evidence": False,
            "source": "nhamcs rule; NHAMCS ED 2016-2022 adults, primary diagnosis, n=%d" % n}


class TestGroupTier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = load_script()

    def subs(self, rows, group="E11"):
        return self.m.subcode_tiers(table(rows), group, RULE, [])

    def test_majority_of_visits_sets_the_tier(self):
        # E11 as in NHAMCS: E11.1 ketoacidosis is severe, but E11, E11.6 and E11.9 hold 396 of 483 visits
        subs = self.subs([("E11", 51, 0.19, 0.041, 2), ("E11.1", 54, 0.87, 0.446, 23),
                          ("E11.6", 198, 0.13, 0.031, 4), ("E11.9", 147, 0.15, 0.039, 4)])
        self.assertEqual([(c, t) for c, t, _, _ in subs], [("E11", 2), ("E11.1", 1), ("E11.6", 2), ("E11.9", 2)])
        g = self.m.group_tier(group_row(483, 1), subs)
        self.assertEqual((g["tier"], g["pooled_tier"]), (2, 1))
        self.assertIn("hold 396 of 483", g["source"])

    def test_no_majority_keeps_the_pooled_tier(self):
        # the rated sub-codes hold only 90 of 200 visits; the rest sit in sub-codes under 30 visits
        subs = self.subs([("E11.6", 60, 0.13, 0.031, 2), ("E11.9", 30, 0.15, 0.039, 1), ("E11.0", 20, 0.9, 0.5, 10)])
        self.assertEqual([c for c, *_ in subs], ["E11.6", "E11.9"])  # under 30 visits: no tier of its own
        g = self.m.group_tier(group_row(200, 1), subs)
        self.assertEqual((g["tier"], g["pooled_tier"]), (1, 1))

    def test_only_rule_rows_change(self):
        subs = [("K25.1", 1, 400, False)]
        for path in ("override", "unscored"):
            g = {**group_row(500, 1), "rule_path": path}
            self.assertEqual(self.m.group_tier(g, subs), g)

    def test_weak_group_when_every_majority_subcode_is_weak(self):
        # tier 1 by the ICU clause alone on under 5 critical-care visits
        subs = self.subs([("S22.0", 70, 0.3, 0.06, 4), ("S22.4", 70, 0.3, 0.06, 4), ("S22.3", 116, 0.06, 0.0, 0)], "S22")
        g = self.m.group_tier(group_row(271, 2), subs)
        self.assertEqual(g["tier"], 1)
        self.assertTrue(g["weak_evidence"])


class TestTierFile(unittest.TestCase):
    """spec/offlist_tiers_nhamcs.csv under the group-tier rule."""

    def test_e11_group_is_tier_2_ketoacidosis_tier_1(self):
        rule = vr.TierFileRule()
        self.assertEqual(rule.tier("E11")[0], "2")
        self.assertEqual(rule.tier("E11.9")[0], "2")
        self.assertEqual(rule.tier("E11.65")[0], "2")
        self.assertEqual(rule.tier("E11.10")[0], "1")

    def test_pooled_version_is_kept(self):
        pooled = vr.TierFileRule(ROOT / "spec" / "offlist_tiers_nhamcs_pooled.csv")
        self.assertEqual(pooled.tier("E11")[0], "1")
        self.assertEqual(pooled.tier("E11.9")[0], "2")


if __name__ == "__main__":
    unittest.main()
