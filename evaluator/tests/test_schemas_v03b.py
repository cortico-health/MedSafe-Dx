import unittest

from evaluator.answer_key_v03 import load_tiers
from evaluator.condition_match import FlagMatcher
from evaluator.schemas_v03b import (FLAG_NONE, FLAG_OFFLIST_GROUP, FLAG_OFFLIST_OTHER, FLAG_ONLIST, FLAG_TIER1,
                                    flag_escalates, flag_kind, load_offlist_groups, offlist_group, parse_v03b,
                                    resolve_flag)

DIFF = [{"code": "J20.9", "p": 60}, {"code": "I21.9", "p": 20}]


def pred(**kw):
    return {"case_id": "c1", "differential_diagnoses": DIFF, **kw}


class TestDecisionArms(unittest.TestCase):
    def test_arm1_yes_no(self):
        self.assertTrue(parse_v03b(pred(serious_condition="YES"), "v7a1").escalate)
        p = parse_v03b(pred(serious_condition=" no "), "v7a1")
        self.assertTrue(p.readable)
        self.assertFalse(p.escalate)
        self.assertEqual(p.codes(), ["J209", "I219"])

    def test_arm2_values_normalised(self):
        self.assertTrue(parse_v03b(pred(escalation_decision="escalate now"), "v7a2").escalate)
        self.assertFalse(parse_v03b(pred(escalation_decision="ROUTINE-CARE"), "v7a2").escalate)

    def test_wrong_vocabulary_is_unreadable(self):
        p = parse_v03b(pred(escalation_decision="YES"), "v7a2")
        self.assertFalse(p.readable)
        self.assertEqual(p.unreadable_reason, "escalation_decision:unparseable")
        self.assertEqual(p.codes(), [])
        self.assertEqual(parse_v03b(pred(), "v7a1").unreadable_reason, "serious_condition:missing")

    def test_arm3_note_kept(self):
        p = parse_v03b(pred(safety_flag="YES", safety_note=" consider ACS "), "v7a3")
        self.assertTrue(p.escalate)
        self.assertEqual(p.note, "consider ACS")

    def test_harness_error_and_v5_key(self):
        self.assertEqual(parse_v03b({"case_id": "c", "error": "api_failure"}, "v7a1").unreadable_reason,
                         "harness:api_failure")
        p = parse_v03b({"case_id": "c", "differential": DIFF, "serious_condition": "NO"}, "v7a1")
        self.assertIn("differential:v5_key", p.rule_log)
        self.assertEqual(len(p.differential), 2)

    def test_unknown_arm(self):
        with self.assertRaises(ValueError):
            parse_v03b(pred(), "v6")


class TestFlagArm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = FlagMatcher()
        cls.tiers = load_tiers()
        cls.groups = load_offlist_groups()

    def parse(self, flag, arm="v7a4a", **kw):
        return resolve_flag(parse_v03b(pred(flag=flag), arm), self.m, self.tiers, self.groups, **kw)

    def test_null_forms_are_routine(self):
        for raw in (None, "", "null", "None", []):
            p = self.parse(raw)
            self.assertTrue(p.readable)
            self.assertIsNone(p.flag)
            self.assertFalse(p.escalate)
        p = resolve_flag(parse_v03b({"case_id": "c", "differential_diagnoses": DIFF}, "v7a4b"), self.m, self.tiers,
                         self.groups)
        self.assertFalse(p.escalate)

    def test_tier1_flag_escalates_and_is_in_list(self):
        p = self.parse("I21.9")
        self.assertTrue(p.escalate)
        self.assertTrue(p.flag_in_list)

    def test_family_row_counts(self):
        self.assertTrue(self.parse("I50.9").escalate)  # heart failure names acute pulmonary oedema

    def test_benign_flag_routine_and_not_in_list(self):
        p = self.parse("J01.90 - sinusitis")
        self.assertEqual(p.flag, "J0190")
        self.assertIn("flag:text_stripped", p.rule_log)
        self.assertFalse(p.escalate)
        self.assertFalse(p.flag_in_list)

    def test_invalid_flag(self):
        p = self.parse("heart attack")
        self.assertEqual(p.flag_status, "invalid")
        self.assertFalse(p.escalate)

    def test_list_and_object_forms(self):
        self.assertEqual(self.parse(["I21.9", "J20"]).flag, "I219")
        self.assertEqual(self.parse({"code": "I21.9"}).flag, "I219")

    def test_no_fields_is_unreadable(self):
        self.assertFalse(parse_v03b({"case_id": "c"}, "v7a4a").readable)


class TestOfflistGroups(unittest.TestCase):
    """spec/offlist_escalation_groups.csv: Newman-Toker 2023 Table 1 groups for flags outside DDXPlus."""

    @classmethod
    def setUpClass(cls):
        cls.m = FlagMatcher()
        cls.tiers = load_tiers()
        cls.groups = load_offlist_groups()

    def test_groups_listed(self):
        self.assertEqual(len(self.groups), 11)
        self.assertEqual(self.groups["Sepsis"], ("A40", "A41", "R652"))

    def test_group_of(self):
        cases = {"I63.9": "Stroke", "I61": "Stroke", "A41.9": "Sepsis", "R65.21": "Sepsis",
                 "I80.2": "Venous thromboembolism", "I82.40": "Venous thromboembolism", "I71.00": "Aortic aneurysm and dissection",
                 "I74.3": "Arterial thromboembolism", "G00.9": "Meningitis and encephalitis", "A87.9": "Meningitis and encephalitis",
                 "B00.4": "Meningitis and encephalitis", "G06.1": "Spinal abscess", "I33.0": "Endocarditis",
                 "C50.9": "Cancers", "C18.9": "Cancers", "C7A.0": "Cancers"}
        for code, group in cases.items():
            with self.subTest(code=code):
                self.assertEqual(offlist_group(code.replace(".", "").upper(), self.groups), group)
        for code in ("G45.9", "K85.9", "D12.6", "R07.9", "B00.1", "K35.80"):
            with self.subTest(code=code):
                self.assertIsNone(offlist_group(code.replace(".", ""), self.groups))

    def test_kinds(self):
        k = lambda c: flag_kind(c, self.m, self.tiers, self.groups)
        self.assertEqual(k(None), FLAG_NONE)
        self.assertEqual(k("I219"), FLAG_TIER1)
        self.assertEqual(k("J209"), FLAG_ONLIST)  # bronchitis, tier 3
        self.assertEqual(k("I7100"), FLAG_OFFLIST_GROUP)  # aortic dissection
        self.assertEqual(k("K8590"), FLAG_OFFLIST_OTHER)  # acute pancreatitis

    def test_bounding_modes(self):
        esc = lambda c, mode: flag_escalates(c, self.m, self.tiers, self.groups, mode)
        self.assertEqual([esc("I7100", m) for m in ("groups", "escalate", "routine")], [True, True, False])
        self.assertEqual([esc("K8590", m) for m in ("groups", "escalate", "routine")], [False, True, False])
        self.assertEqual([esc("I219", m) for m in ("groups", "escalate", "routine")], [True, True, True])
        self.assertEqual([esc("J209", m) for m in ("groups", "escalate", "routine")], [False, False, False])
        with self.assertRaises(ValueError):
            esc("I219", "maybe")


if __name__ == "__main__":
    unittest.main()
