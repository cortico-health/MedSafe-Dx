import unittest

from evaluator import working_diagnosis as wd
from evaluator.answer_key_v03 import CaseKeyV03, TargetV03
from inference.prompt import V7_ARMS
from inference.run_inference import build_messages_v7

TIERS = {"Bronchitis": 3, "Anemia": 3, "URTI": 3, "Pneumonia": 1, "GERD": 2}


def key(truth, tier, r10=(), r5=(), low=False):
    k = CaseKeyV03(case_id="c", truth=truth, truth_tier=tier, clearly_low_risk=low, intermediate=False,
                   red_flag=False, red_flag_names=())
    for c in set(r10) | set(r5):
        k.considered[c] = TargetV03(c, "dxa", 12.0, "kept", False, c in r10, True, 50, 0.1, ())
    return k


class TestChoose(unittest.TestCase):
    def test_highest_tier3_with_ties_by_name(self):
        self.assertEqual(wd.choose({"Pneumonia": 60, "URTI": 20, "Anemia": 20}, TIERS, "E_1", {}), ("Anemia", 20, False))

    def test_fallback_then_default(self):
        self.assertEqual(wd.choose({"Pneumonia": 90}, TIERS, "E_1", {"E_1": "URTI"}), ("URTI", 0.0, True))
        self.assertEqual(wd.choose({"Pneumonia": 90}, TIERS, "E_2", {"E_1": "URTI"}), ("Bronchitis", 0.0, True))

    def test_committed_fallback_file(self):
        fb = wd.load_fallback()
        self.assertEqual(len(fb), 93)
        self.assertTrue(set(fb.values()) <= set(wd.DISPLAY_NAMES))


class TestClasses(unittest.TestCase):
    def test_classes(self):
        self.assertEqual(wd.case_class(key("Pneumonia", 1, r10=["Pneumonia"])), "serious")
        self.assertEqual(wd.case_class(key("URTI", 3, r10=["Pneumonia"])), "serious")
        self.assertEqual(wd.case_class(key("GERD", 2, r5=["Pneumonia"])), "middle")
        self.assertEqual(wd.case_class(key("URTI", 3, low=True)), "benign")
        self.assertEqual(wd.case_class(key("URTI", 3, r5=["Pneumonia"])), "other")

    def test_a3_tier2_truth_is_middle(self):
        # amendment A3: a tier-2 truth leaves the headline even with a DXA-derived R10 target
        k = key("GERD", 2, r10=["Possible NSTEMI / STEMI"])
        self.assertEqual(wd.case_class(k), "middle")
        self.assertEqual(wd.case_class(k, wd.DRAFT3), "serious")
        self.assertEqual(wd.case_class(key("URTI", 3, r10=["Pneumonia"]), wd.DRAFT3), "serious")
        with self.assertRaises(ValueError):
            wd.case_class(k, "draft2")

    def test_scorer_uses_a3_design_keeps_draft3(self):
        from evaluator import v03b_score as sb
        ab = sb.load_ab()
        a3 = [wd.case_class(k) for k in ab.key.keys]
        self.assertEqual(list(ab.klass), a3)
        self.assertEqual((a3.count("serious"), a3.count("benign"), a3.count("middle")), (71, 40, 39))
        self.assertEqual([r["class"] for r in ab.design].count("serious"), 90)
        self.assertFalse(any(k.truth_tier == 2 for k, c in zip(ab.key.keys, a3) if c == "serious"))


class TestPromptV7(unittest.TestCase):
    CASE = {"case_id": "c", "age": 40, "sex": "female", "presenting_symptoms": [],
            "working_diagnosis_name": "acute bronchitis (J40)"}

    def test_anchor_line_per_arm(self):
        line = "The clinician's working diagnosis is acute bronchitis (J40)."
        hint = "Note if there may be any patient safety flags with this case."
        for arm, (_, anchor) in V7_ARMS.items():
            with self.subTest(arm=arm):
                system, user = (m["content"] for m in build_messages_v7(dict(self.CASE), arm))
                self.assertEqual(line in user, anchor is not None)
                self.assertEqual(hint in user, arm == "v7a3")
                self.assertIn("Symptoms and history not listed were asked about and are absent.", system)
                self.assertIn('"differential_diagnoses"', user)
                for word in ("7 times", "urgent", "minutes", "hours"):
                    self.assertNotIn(word, system + user)

    def test_decision_fields(self):
        fields = {"v7a1": "serious_condition", "v7a2": "escalation_decision", "v7a3": "safety_flag",
                  "v7a4a": '"flag"', "v7a4b": '"flag"'}
        for arm, f in fields.items():
            self.assertIn(f, build_messages_v7(dict(self.CASE), arm)[1]["content"])

    def test_anchored_arm_needs_working_diagnosis(self):
        with self.assertRaises(ValueError):
            build_messages_v7({"case_id": "c", "presenting_symptoms": []}, "v7a2")


if __name__ == "__main__":
    unittest.main()
