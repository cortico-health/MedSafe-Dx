import unittest

from evaluator.answer_key_v03 import load_tiers
from evaluator.condition_match import FlagMatcher
from evaluator.schemas_v03 import parse_v03
from evaluator.v03_measures import (NO_WITH_TIER1_FLAG, YES_WITHOUT_FLAGS, YES_WITHOUT_TIER1_FLAG,
                                    consistency_issues, reassures, reassures_prime, tier1_flagged)

MI = "Possible NSTEMI / STEMI"


class TestHPrimeAndConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = FlagMatcher()
        cls.tiers = load_tiers()

    def judge(self, concern, flags, policy="standard"):
        p = parse_v03({"serious_concern": concern, "flags": flags})
        hit = tier1_flagged(p, self.m, self.tiers, policy)
        return p, hit, reassures(p), reassures_prime(p, hit), consistency_issues(p, hit)

    def test_yes_with_tier1_flag(self):
        _, hit, h, hp, con = self.judge("YES", ["I21.9", "J01.90"])
        self.assertEqual(hit, {MI})
        self.assertEqual((h, hp, con), (False, False, []))

    def test_yes_with_only_benign_flags_counts_in_h_prime(self):
        _, hit, h, hp, con = self.judge("YES", ["J01.90", "K21.9"])  # sinusitis, GERD
        self.assertEqual(hit, set())
        self.assertEqual((h, hp), (False, True))
        self.assertEqual(con, [YES_WITHOUT_TIER1_FLAG])

    def test_related_code_is_a_tier1_flag_only_when_lenient(self):
        _, hit, _, hp, _ = self.judge("YES", ["K21.9"], "lenient")
        self.assertIn(MI, hit)
        self.assertFalse(hp)

    def test_yes_without_flags(self):
        _, _, h, hp, con = self.judge("YES", [])
        self.assertEqual((h, hp), (False, True))
        self.assertEqual(con, [YES_WITHOUT_FLAGS, YES_WITHOUT_TIER1_FLAG])

    def test_no_while_flagging_tier1(self):
        _, _, h, hp, con = self.judge("NO", ["I26.99"])
        self.assertEqual((h, hp), (True, True))
        self.assertEqual(con, [NO_WITH_TIER1_FLAG])

    def test_plain_no(self):
        _, _, h, hp, con = self.judge("NO", ["J06.9"])
        self.assertEqual((h, hp, con), (True, True, []))

    def test_unreadable(self):
        _, _, h, hp, con = self.judge("maybe", ["I21.9"])
        self.assertEqual((h, hp, con), (True, True, []))


if __name__ == "__main__":
    unittest.main()
