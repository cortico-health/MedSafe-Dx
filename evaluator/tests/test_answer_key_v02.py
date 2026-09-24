import unittest

from evaluator import answer_key_v02 as ak

CONDS = ak.load_conditions()


class TestRiskFields(unittest.TestCase):
    def test_serious_is_severity_1_or_2(self):
        self.assertTrue(ak.is_serious("Pulmonary embolism", CONDS))  # severity 2
        self.assertTrue(ak.is_serious("Anaphylaxis", CONDS))  # severity 1
        self.assertFalse(ak.is_serious("Pneumonia", CONDS))  # severity 3
        self.assertFalse(ak.is_serious("URTI", CONDS))  # severity 5

    def test_icd10_normalised(self):
        self.assertEqual(CONDS["Pneumonia"]["icd10"], "J17/J18")
        self.assertEqual(CONDS["Pulmonary embolism"]["icd10"], "I26")

    def test_at_risk_threshold(self):
        at = ak.risk_fields("URTI", [["URTI", 0.875], ["Pulmonary embolism", 0.125]], CONDS)
        self.assertEqual((at["serious"], at["at_risk"], at["clearly_low_risk"]), (False, True, False))
        self.assertAlmostEqual(at["p_serious_risk"], 12.5)
        below = ak.risk_fields("URTI", [["URTI", 0.88], ["Pulmonary embolism", 0.12]], CONDS)
        self.assertEqual((below["at_risk"], below["clearly_low_risk"]), (False, True))

    def test_serious_patient_is_never_clearly_low_risk(self):
        f = ak.risk_fields("Pulmonary embolism", [["URTI", 1.0]], CONDS)
        self.assertEqual((f["serious"], f["at_risk"], f["clearly_low_risk"]), (True, False, False))


class TestRedFlags(unittest.TestCase):
    def test_bleeding_codes(self):
        for code in ("E_210", "E_140", "E_179", "E_45"):
            self.assertEqual(ak.red_flags([code]), ["bleeding"], code)
        self.assertEqual(ak.red_flags(["E_178"]), [])  # unusual bruising is not in the list

    def test_thunderclap(self):
        self.assertEqual(ak.red_flags(["E_55_@_V_89", "E_59_@_7", "E_56_@_8"]), ["thunderclap_headache"])
        self.assertEqual(ak.red_flags(["E_55_@_V_89", "E_59_@_6", "E_56_@_9"]), [])  # onset < 7
        self.assertEqual(ak.red_flags(["E_55_@_V_89", "E_59_@_9", "E_56_@_7"]), [])  # intensity < 8
        self.assertEqual(ak.red_flags(["E_55_@_V_29", "E_59_@_10", "E_56_@_10"]), [])  # chest, not head
        self.assertEqual(ak.red_flags(["E_55_@_V_89", "E_56_@_10"]), [])  # onset missing

    def test_fever_immunosuppressed(self):
        for code in ("E_227", "E_2", "E_44"):
            self.assertEqual(ak.red_flags(["E_91", code]), ["fever_immunosuppressed"], code)
        self.assertEqual(ak.red_flags(["E_227"]), [])  # no fever
        self.assertEqual(ak.red_flags(["E_94", "E_227"]), [])  # chills alone are not fever

    def test_several(self):
        self.assertEqual(ak.red_flags(["E_45", "E_91", "E_2"]), ["bleeding", "fever_immunosuppressed"])


if __name__ == "__main__":
    unittest.main()
