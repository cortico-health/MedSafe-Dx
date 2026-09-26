import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluator import v03_score as vs
from evaluator import v03b_score as sb
from evaluator.answer_key_v03 import CaseKeyV03, TargetV03, load_tiers
from evaluator.condition_match import FlagMatcher
from evaluator.schemas_v03b import load_offlist_groups

MI, BRONCHITIS, URTI, PNEUMONIA = "Possible NSTEMI / STEMI", "Bronchitis", "URTI", "Pneumonia"


def key(cid, truth, tier, r10=(), low=False):
    k = CaseKeyV03(case_id=cid, truth=truth, truth_tier=tier, clearly_low_risk=low, intermediate=False,
                   red_flag=False, red_flag_names=())
    for c in r10:
        k.considered[c] = TargetV03(c, "truth" if c == truth else "dxa", 50.0, "kept", False, True, True, 50, 0.2, ())
    return k


def toy() -> sb.ABSet:
    """Two SERIOUS (MI truth; bronchitis truth with a pneumonia target), two BENIGN, one MIDDLE."""
    keys = [key("s1", MI, 1, [MI]), key("s2", BRONCHITIS, 3, [PNEUMONIA]), key("b1", URTI, 3, low=True),
            key("b2", BRONCHITIS, 3, low=True), key("m1", "GERD", 2)]
    k = vs.set_key("toy", [x.case_id for x in keys], {x.case_id: x for x in keys}, load_tiers())
    return sb.ABSet(key=k, klass=np.array(["serious", "serious", "benign", "benign", "middle"]), design=[{}] * 5,
                    cases=[], matcher=FlagMatcher(), groups=load_offlist_groups())


def preds(field, values, diffs=None):
    diffs = diffs or [["J20.9"]] * len(values)
    return [{"case_id": c, "differential_diagnoses": [{"code": d, "p": 30} for d in ds], field: v}
            for c, v, ds in zip(("s1", "s2", "b1", "b2", "m1"), values, diffs)]


class TestHeadline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ab = toy()
        cls.M = vs.cluster_draws(cls.ab.key.k, 50, 1)

    def stats(self, arm, field, values, diffs=None):
        a = sb.row_answers(preds(field, values, diffs), arm, self.ab, "row")
        return sb.row_stats(a, self.ab, self.M)[0]

    def test_perfect_and_always_escalate(self):
        p = self.stats("v7a1", "serious_condition", ["YES", "YES", "NO", "NO", "NO"])
        self.assertEqual((p["score"], p["U"], p["O"]), (100.0, 0.0, 0.0))
        p = self.stats("v7a2", "escalation_decision", ["ESCALATE_NOW"] * 5)
        self.assertAlmostEqual(p["score"], 0.0)
        self.assertEqual(p["esc"], 1.0)
        self.assertEqual(p["mid_esc"], 1.0)

    def test_one_miss(self):
        # cost 7 over 4 headline cases, always escalate costs 2 / 4: SCORE = 100 x (0.5 - 1.75) / 0.5
        p = self.stats("v7a3", "safety_flag", ["YES", "NO", "NO", "NO", "NO"])
        self.assertAlmostEqual(p["score"], -250.0)
        self.assertEqual((p["U"], p["O"]), (0.5, 0.0))

    def test_unreadable_is_routine(self):
        p = self.stats("v7a1", "serious_condition", ["maybe", "YES", "NO", "NO", "NO"])
        self.assertEqual((p["U"], p["unreadable"]), (0.5, 0.2))

    def test_target_listed_and_top1(self):
        p = self.stats("v7a1", "serious_condition", ["YES"] * 5,
                       [["I21.9"], ["J20.9", "J18.9"], ["J06.9"], ["J20.9"], ["K21.9"]])
        self.assertEqual((p["TL"], p["TL_truth"], p["TL_dxa"]), (1.0, 1.0, 1.0))
        self.assertEqual(p["top1"], 1.0)  # every case lists its truth first

    def test_arm4_flags(self):
        ab = self.ab
        rows = preds("flag", ["I21.9", "I71.00", None, "K85.9", "J20.9"],
                     [["I21.9"], ["J20.9"], ["J06.9"], ["J20.9"], ["K21.9"]])
        a = sb.row_answers(rows, "v7a4a", ab, "row")
        self.assertEqual(a.esc.tolist(), [True, True, False, False, False])  # tier 1; aortic dissection group
        c = sb.counts(a, ab)["flags"]
        self.assertEqual((c["tier1"], c["offlist_escalating"], c["offlist_not_escalating"], c["onlist_not_tier1"],
                          c["null_or_invalid"], c["not_in_own_list"]), (1, 1, 1, 1, 1, 3))
        self.assertEqual(sb.row_answers(rows, "v7a4a", ab, "r", offlist="routine").esc.tolist(), [True, False, False, False, False])
        self.assertEqual(sb.row_answers(rows, "v7a4a", ab, "r", offlist="escalate").esc.tolist(), [True, True, False, True, False])


class TestBoard(unittest.TestCase):
    def test_paired_comparisons_and_checks(self):
        ab = toy()
        with tempfile.TemporaryDirectory() as d:
            files = []
            for arm, field, vals in (("v7a1", "serious_condition", ["YES"] * 5),
                                     ("v7a2", "escalation_decision", ["ESCALATE_NOW", "ESCALATE_NOW", "ROUTINE_CARE", "ROUTINE_CARE", "ROUTINE_CARE"]),
                                     ("v7a3", "safety_flag", ["YES", "YES", "NO", "YES", "YES"])):
                f = Path(d) / f"m-{arm}.json"
                f.write_text(json.dumps({"metadata": {"model": "x/m", "prompt_version": arm}, "predictions": preds(field, vals)}))
                files.append(f)
            board = sb.score_board(files, n_boot=50, ab=ab, references=False)
        pr = board["paired"]
        self.assertEqual(set(pr), {"x/m: arm 1 - arm 2", "x/m: arm 2 - arm 3"})
        self.assertAlmostEqual(pr["x/m: arm 1 - arm 2"]["score"]["value"], -100.0)
        self.assertAlmostEqual(pr["x/m: arm 1 - arm 2"]["O"]["value"], 100.0)
        self.assertAlmostEqual(pr["x/m: arm 2 - arm 3"]["O"]["value"], -50.0)
        self.assertIn("| m | 2 | 100.0", sb.report(board))

    def test_justified_arms_score_and_pair(self):
        ab = toy()
        with tempfile.TemporaryDirectory() as d:
            files = []
            for arm, flags in (("v7a4aj", ["I21.9", None, None, None, None]), ("v7a4bj", ["I21.9", "J18.9", None, None, None])):
                rows = preds("flag", flags)
                for r in rows:
                    r["justification"] = "One sentence."
                f = Path(d) / f"m-{arm}.json"
                f.write_text(json.dumps({"metadata": {"model": "x/m", "prompt_version": arm}, "predictions": rows}))
                files.append(f)
            board = sb.score_board(files, n_boot=50, ab=ab, references=False)
        self.assertEqual({r["arm_label"] for r in board["rows"].values()}, {"4aj", "4bj"})
        pr = board["paired"]["x/m: arm 4aj - arm 4bj"]
        self.assertAlmostEqual(pr["U"]["value"], 50.0)  # 4aj misses the pneumonia target that 4bj flags
        self.assertIn("| m | 4aj |", sb.report(board))


if __name__ == "__main__":
    unittest.main()
