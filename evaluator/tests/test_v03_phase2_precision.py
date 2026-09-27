"""scripts/analysis/v03_phase2_precision.py on the audited 150: it must reproduce the in-sample numbers of
scripts/analysis/v03_case_selection.py under amendment A5 (spec/v0.3-scoring.md) and rule X11 (docs/v0.3-case-selection-rules.md
section 4), so the Phase 2b criteria are computed by code known to work before the Phase 2b reference exists."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
REF = ROOT / "results" / "audit" / "reference_adjudicated.jsonl"
RUNS = ROOT / "results" / "v03" / "ab" / "runs"
DATA = ROOT / "data" / "ddxplus_v0" / "release_test_patients"


@unittest.skipUnless(REF.exists() and RUNS.exists() and DATA.exists(), "needs the audit reference, the 4aj/4bj runs and DDXPlus")
class TestPrecisionOnAudit150(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import v03_phase2_common as pc
        import v03_phase2_precision as pp
        s = pc.load_audit150()
        cls.head = pp.evaluate(s, RUNS, REF)
        cls.ccsr = pp.evaluate(s, RUNS, REF, include_ccsr=True)

    def test_counts_and_criteria_match_the_case_selection_script(self):
        p = self.head["pooled"]
        self.assertEqual((p["TP"], p["FP"], p["FN"], p["judged"]), (247, 221, 78, 1988))
        self.assertEqual((p["safety_k"], p["safety_n"]), (149, 184))
        self.assertEqual(p["safety"][0], 81.0)
        self.assertEqual(p["point_weighted"][0], 72.6)
        self.assertTrue(self.head["criteria"]["6_safety"]["pass"])
        self.assertTrue(self.head["criteria"]["7_point_weighted"]["pass"])

    def test_ccsr_row_reproduces_the_frozen_document(self):
        # the CCSR sensitivity row of scripts/analysis/v03_case_selection.py (summary.json, audited_150.sensitivity.ccsr)
        p = self.ccsr["pooled"]
        self.assertEqual((p["TP"], p["FP"], p["FN"]), (240, 229, 78))
        self.assertEqual((p["safety"][0], p["point_weighted"][0]), (83.4, 73.2))

    def test_anchor_check(self):
        a = self.head["criteria"]["10_anchor_check"]
        self.assertEqual(len(a), 7)
        self.assertEqual(a["claude-sonnet-4.6"]["diff_per_100"], -3.2)
        self.assertFalse(any(v["excludes_zero"] for v in a.values()))

    def test_reason_specificity_and_fn_rate_kept(self):
        sp = self.head["criteria"]["8_reason_specificity"]
        self.assertEqual(sp["partials"], sp["in_list"] + sp["off_list"] + sp["truth"] + sp["other"])
        self.assertLess(self.head["pooled"]["fn_kept"], self.head["pooled"]["FN"])  # most FNs sit on excluded cases


if __name__ == "__main__":
    unittest.main()
