"""Arms 4aj and 4bj (spec/v0.3-scoring.md section 12, amendment A1): arms 4a and 4b plus a last,
unscored "justification" field."""

import unittest

from evaluator.schemas_v03b import parse_v03b
from inference.prompt import V7_ARMS, V7_FIELDS, V7_SCHEMAS
from inference.run_inference import build_messages_v7

CASE = {"case_id": "c", "age": 40, "sex": "female", "presenting_symptoms": [],
        "working_diagnosis_name": "acute bronchitis (J40)"}
LINE = ('- "justification": One sentence explaining whether this patient needs escalation to the clinician '
        "and why.")
DIFF = [{"code": "J20.9", "p": 60}, {"code": "I21.9", "p": 20}]


class TestPrompt(unittest.TestCase):
    def test_same_as_base_arm_plus_last_field(self):
        for arm in ("v7a4a", "v7a4b"):
            with self.subTest(arm=arm):
                base_sys, base_user = (m["content"] for m in build_messages_v7(dict(CASE), arm))
                sys_, user = (m["content"] for m in build_messages_v7(dict(CASE), arm + "j"))
                self.assertEqual(sys_, base_sys.replace(V7_FIELDS[arm], V7_FIELDS[arm] + "\n" + LINE))
                self.assertEqual(user, base_user.replace('"flag": "ICD10_CODE | null"\n}',
                                                         '"flag": "ICD10_CODE | null",\n  "justification": "STRING"\n}'))
                self.assertEqual(V7_ARMS[arm + "j"], V7_ARMS[arm])
                self.assertTrue(V7_SCHEMAS[arm + "j"].rstrip("}\n ").endswith('"justification": "STRING"'))


class TestParse(unittest.TestCase):
    def test_justification_kept_flag_parsed(self):
        for arm in ("v7a4aj", "v7a4bj"):
            p = parse_v03b({"case_id": "c", "differential_diagnoses": DIFF, "flag": "I21.9",
                            "justification": "  Chest pain needs an ECG today. "}, arm)
            self.assertTrue(p.readable)
            self.assertEqual((p.flag, p.flag_status), ("I219", "ok"))
            self.assertEqual(p.justification, "Chest pain needs an ECG today.")
            self.assertNotIn("justification:missing", p.rule_log)

    def test_missing_justification_is_logged_not_unreadable(self):
        p = parse_v03b({"case_id": "c", "differential_diagnoses": DIFF, "flag": None}, "v7a4aj")
        self.assertTrue(p.readable)
        self.assertIsNone(p.justification)
        self.assertEqual(p.flag_status, "none")
        self.assertIn("justification:missing", p.rule_log)

    def test_base_arms_ignore_justification(self):
        p = parse_v03b({"case_id": "c", "differential_diagnoses": DIFF, "flag": None, "justification": "x"}, "v7a4a")
        self.assertIsNone(p.justification)
        self.assertNotIn("justification:missing", p.rule_log)


if __name__ == "__main__":
    unittest.main()
