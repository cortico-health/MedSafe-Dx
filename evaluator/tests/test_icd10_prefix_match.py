import unittest


class TestICD10PrefixMatch(unittest.TestCase):
    def test_rules_empty_predicted_never_matches(self):
        from evaluator.rules import _icd10_prefix_match

        self.assertFalse(_icd10_prefix_match("", {"i219"}))
        self.assertFalse(_icd10_prefix_match("", {"i21"}))

    def test_metrics_empty_predicted_never_matches(self):
        from evaluator.metrics import _icd10_prefix_match

        self.assertFalse(_icd10_prefix_match("", {"i219"}))
        self.assertFalse(_icd10_prefix_match("", {"i21"}))

    def test_prefix_matching_is_bidirectional(self):
        from evaluator.rules import _icd10_prefix_match

        self.assertTrue(_icd10_prefix_match("i21", {"i219"}))
        self.assertTrue(_icd10_prefix_match("i219", {"i21"}))


if __name__ == "__main__":
    unittest.main()
