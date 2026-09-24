"""Checks on the committed v0.2 case set (data/test_sets/eval-v02-adult.json)."""

import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path

from evaluator import answer_key_v02 as ak

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "data/test_sets/eval-v02-adult.json"
IDS = ROOT / "data/test_sets/eval-v02-adult.case_ids.txt"


@unittest.skipUnless(SAMPLE.exists(), "run scripts/prep_v02_sample.py first (data/test_sets/ is git-ignored)")
class TestV02Sample(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(SAMPLE.read_text())
        cls.cases = cls.data["cases"]

    def test_design(self):
        self.assertEqual(len(self.cases), 470)
        per = Counter(c["true_pathology"] for c in self.cases)
        self.assertEqual(len(per), 47)
        self.assertEqual(set(per.values()), {10})
        self.assertNotIn("Bronchiolitis", per)
        self.assertNotIn("Croup", per)
        self.assertTrue(all(c["age"] >= 18 for c in self.cases))
        self.assertEqual(len({c["case_id"] for c in self.cases}), 470)

    def test_id_list_matches(self):
        text = IDS.read_text()
        self.assertEqual(text.split(), [c["case_id"] for c in self.cases])
        self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), self.data["metadata"]["case_ids_sha256"])

    def test_keeps_v0_case_format(self):
        for key in ("case_id", "age", "sex", "presenting_symptoms", "symptom_duration", "severity_flags",
                    "red_flag_indicators", "gold_top3", "escalation_required", "uncertainty_acceptable"):
            self.assertIn(key, self.cases[0])
        # The prompt shows red_flag_indicators, so the section 7 flag must not leak into it.
        self.assertTrue(all(c["red_flag_indicators"] == [] for c in self.cases))

    def test_answer_key_fields_recompute(self):
        conds = ak.load_conditions()
        for c in self.cases:
            self.assertEqual(c["ddxplus_severity"], conds[c["true_pathology"]]["ddxplus_severity"])
            fields = ak.risk_fields(c["true_pathology"], c["ddxplus_differential"], conds)
            self.assertEqual({k: c[k] for k in fields}, fields, c["case_id"])
            self.assertEqual(c["offlist_red_flag_reasons"], ak.red_flags(c["presenting_symptoms"]))

    def test_no_nts_fields(self):
        # v0.2 scores against DDXPlus severity only (spec revision 4).
        for c in self.cases:
            self.assertFalse([k for k in c if k.startswith("nts_") or "urgency" in k], c["case_id"])

    def test_risk_groups_partition(self):
        for c in self.cases:
            self.assertEqual(c["serious"], c["ddxplus_severity"] <= 2)
            self.assertEqual(c["at_risk"], c["p_serious_risk"] >= 12.5)
            self.assertEqual(c["clearly_low_risk"], not c["serious"] and not c["at_risk"])


if __name__ == "__main__":
    unittest.main()
