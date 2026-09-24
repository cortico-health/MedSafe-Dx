import unittest

import acuity_key as ak

REF = ak.load_reference()


def level(cond, age=40, ev=()):
    return ak.nts_level(cond, age, list(ev), REF)


class TestModifierRules(unittest.TestCase):
    """One test per modifier rule in docs/triage-scale-anchor.md section 3."""

    def test_rules_match_csv(self):
        ak.check_modifiers_match_csv(REF)

    def test_csv_drift_is_caught(self):
        broken = {k: dict(v) for k, v in REF.items()}
        broken["Pneumonia"]["modifiers"] = "age >= 65 -> 1 (changed)"
        with self.assertRaises(ValueError):
            ak.check_modifiers_match_csv(broken)
        broken = {k: dict(v) for k, v in REF.items()}
        broken["URTI"]["modifiers"] = "age >= 65 -> 3"
        with self.assertRaises(ValueError):
            ak.check_modifiers_match_csv(broken)

    def test_pneumonia(self):
        self.assertEqual(level("Pneumonia", 40), (3, 3, None))
        self.assertEqual(level("Pneumonia", 64), (3, 3, None))
        self.assertEqual(level("Pneumonia", 65), (3, 2, "pneumonia_risk"))
        for code in ["E_123", "E_31", "E_106", "E_227", "E_2", "E_34"]:
            self.assertEqual(level("Pneumonia", 30, [code])[1], 2, code)
        # Diabetes is not a pneumonia trigger.
        self.assertEqual(level("Pneumonia", 30, ["E_69"])[1], 3)

    def test_asthma(self):
        c = "Bronchospasm / acute asthma exacerbation"
        self.assertEqual(level(c, 30), (2, 2, None))
        self.assertEqual(level(c, 30, ["E_101"]), (2, 1, "asthma_near_fatal_risk"))
        self.assertEqual(level(c, 30, ["E_46"])[1], 1)
        self.assertEqual(level(c, 80)[1], 2)  # age alone does not trigger

    def test_bronchiolitis(self):
        self.assertEqual(level("Bronchiolitis", 0), (3, 3, None))
        self.assertEqual(level("Bronchiolitis", 0, ["E_160"]), (3, 2, "bronchiolitis_infant_risk"))
        self.assertEqual(level("Bronchiolitis", 0, ["E_139"])[1], 2)
        self.assertEqual(level("Bronchiolitis", 1, ["E_160"])[1], 3)  # needs age < 1

    def _risk_group(self, cond, rule_id):
        self.assertEqual(level(cond, 40), (5, 5, None))
        self.assertEqual(level(cond, 65), (5, 3, rule_id))
        for code in ["E_167", "E_123", "E_31", "E_124", "E_106", "E_69", "E_113", "E_126", "E_227", "E_2", "E_34"]:
            self.assertEqual(level(cond, 30, [code])[1], 3, code)
        self.assertEqual(level(cond, 30, ["E_79"])[1], 5)  # smoking is not in the set

    def test_influenza(self):
        self._risk_group("Influenza", "influenza_risk_group")

    def test_bronchitis(self):
        self._risk_group("Bronchitis", "bronchitis_risk_group")

    def test_otitis_media(self):
        c = "Acute otitis media"
        self.assertEqual(level(c, 70), (5, 5, None))  # age alone does not trigger
        for code in ["E_227", "E_69", "E_106", "E_123", "E_31", "E_113"]:
            self.assertEqual(level(c, 30, [code]), (5, 3, "otitis_media_risk_group"), code)
        self.assertEqual(level(c, 30, ["E_2"])[1], 5)  # HIV is not listed for otitis

    def test_rhinosinusitis(self):
        c = "Acute rhinosinusitis"
        self.assertEqual(level(c, 30, ["E_227"]), (5, 3, "rhinosinusitis_immunosuppressed"))
        self.assertEqual(level(c, 70, ["E_2", "E_34"])[1], 5)

    def test_pharyngitis(self):
        c = "Viral pharyngitis"
        self.assertEqual(level(c, 30, ["E_227"]), (5, 3, "pharyngitis_immunosuppressed"))
        self.assertEqual(level(c, 70, ["E_2"])[1], 5)

    def test_whooping_cough(self):
        self.assertEqual(level("Whooping cough", 0), (5, 3, "whooping_cough_infant"))
        self.assertEqual(level("Whooping cough", 1)[1], 5)

    def test_compound_codes_count_by_base(self):
        # Antecedents arrive as plain codes, but a value-coded evidence must still match its base.
        self.assertEqual(level("Pneumonia", 30, ["E_204_@_V_10", "E_123"])[1], 2)

    def test_condition_without_modifier(self):
        self.assertEqual(level("URTI", 90, ["E_227", "E_123"]), (5, 5, None))


class TestRedFlags(unittest.TestCase):
    def test_bleeding_codes(self):
        for code in ["E_210", "E_140", "E_179", "E_45"]:
            self.assertEqual(ak.red_flags([code]), ["bleeding"], code)
        self.assertEqual(ak.red_flags(["E_178"]), [])  # unusual bruising is not in the list

    def test_thunderclap(self):
        head = ["E_53", "E_55_@_V_89", "E_59_@_7", "E_56_@_8"]
        self.assertEqual(ak.red_flags(head), ["thunderclap_headache"])
        self.assertEqual(ak.red_flags(["E_55_@_V_89", "E_59_@_6", "E_56_@_9"]), [])  # onset < 7
        self.assertEqual(ak.red_flags(["E_55_@_V_89", "E_59_@_9", "E_56_@_7"]), [])  # intensity < 8
        self.assertEqual(ak.red_flags(["E_55_@_V_29", "E_59_@_10", "E_56_@_10"]), [])  # chest, not head
        self.assertEqual(ak.red_flags(["E_55_@_V_124", "E_59_@_10", "E_56_@_10"]), ["thunderclap_headache"])  # occiput
        self.assertEqual(ak.red_flags(["E_55_@_V_89", "E_56_@_10"]), [])  # onset missing

    def test_fever_immunosuppressed(self):
        for code in ["E_227", "E_2", "E_44"]:
            self.assertEqual(ak.red_flags(["E_91", code]), ["fever_immunosuppressed"], code)
        self.assertEqual(ak.red_flags(["E_227"]), [])  # no fever
        self.assertEqual(ak.red_flags(["E_91", "E_61"]), [])  # IV drug use is TIER2, not a red flag
        self.assertEqual(ak.red_flags(["E_94", "E_227"]), [])  # chills alone are not fever

    def test_several(self):
        self.assertEqual(ak.red_flags(["E_45", "E_91", "E_2"]), ["bleeding", "fever_immunosuppressed"])


if __name__ == "__main__":
    unittest.main()
