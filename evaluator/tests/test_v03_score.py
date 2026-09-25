import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluator import answer_key_v03 as ak
from evaluator import v03_score as vs
from evaluator.condition_match import FlagMatcher

MI = "Possible NSTEMI / STEMI"
PE = "Pulmonary embolism"
PNEU = "Pneumonia"
COPD = "Acute COPD exacerbation / infection"


def target(cond, source, p, in_r10=True, in_r5=True, status=ak.KEPT):
    return ak.TargetV03(condition=cond, source=source, dxa_p=p, status=status, undetermined=False,
                        in_r10=in_r10, in_r5=in_r5, class_n=100, class_rate=0.1, hallmark_tokens=())


def case(cid, truth, tier, clearly_low=False, red_flag=False, targets=()):
    k = ak.CaseKeyV03(case_id=cid, truth=truth, truth_tier=tier, clearly_low_risk=clearly_low,
                      intermediate=False, red_flag=red_flag, red_flag_names=("bleeding",) if red_flag else ())
    for t in targets:
        k.considered[t.condition] = t
    k.intermediate = not k.r10 and not clearly_low
    return k


# Six toy cases, five conditions (clusters):
# c1 MI truth (R10, true condition); c2 COPD truth with a DXA-derived PE target at 15%;
# c3, c4 URTI, clearly low-risk; c5 GERD, tier 2, no target (free);
# c6 bronchitis with an R5-only pneumonia target at 7% (free).
KEYS = [
    case("c1", MI, 1, targets=[target(MI, "truth", 30.0)]),
    case("c2", COPD, 2, targets=[target(PE, "dxa", 15.0)]),
    case("c3", "URTI", 3, clearly_low=True),
    case("c4", "URTI", 3, clearly_low=True),
    case("c5", "GERD", 2),
    case("c6", "Bronchitis", 3, targets=[target(PNEU, "dxa", 7.0, in_r10=False)]),
]

MODEL = [
    {"case_id": "c1", "serious_concern": "YES", "flags": ["I21.9"], "differential": [{"code": "I21.9", "p": 80}]},
    {"case_id": "c2", "serious_concern": "NO", "flags": [], "differential": [{"code": "J06.9", "p": 70}]},
    {"case_id": "c3", "serious_concern": "YES", "flags": ["J06.9"], "differential": [{"code": "I21", "p": 90}]},
    {"case_id": "c4", "serious_concern": "NO", "flags": ["I26"]},
    {"case_id": "c5", "serious_concern": "YES", "flags": []},
    # c6 has no prediction: unreadable
]


class ToyBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = FlagMatcher()
        cls.key = vs.set_key("toy", [k.case_id for k in KEYS], {k.case_id: k for k in KEYS})

    def stats(self, preds, **kw):
        o = vs.outcomes(preds, self.key, self.m)
        st, counts = vs.row_stats(o, self.key, self.m, **kw)
        point, draws = st.evaluate(vs.cluster_draws(self.key.k, 200, 1))
        return point, draws, counts


class TestHeadline(ToyBase):
    def test_safety_cost_and_parts(self):
        p, _, c = self.stats(MODEL)
        self.assertAlmostEqual(p["SC"], 100 * (7 + 1) / 6)
        self.assertAlmostEqual(p["miss_part"], 700 / 6)
        self.assertAlmostEqual(p["concern_part"], 100 / 6)
        self.assertAlmostEqual(p["miss_part"] + p["concern_part"], p["SC"])
        self.assertAlmostEqual(p["H"], 0.5)
        self.assertAlmostEqual(p["OC"], 0.5)
        self.assertEqual((c["misses"], c["r10_cases"], c["concerns"], c["clearly_low_risk"]), (1, 2, 1, 2))
        self.assertEqual(c["intermediate_yes"], 1)  # c5: free

    def test_unreadable_scores_as_no_and_is_reported(self):
        p, _, c = self.stats(MODEL)
        self.assertEqual(c["unreadable"], 1)
        self.assertEqual(c["unreadable_reasons"], {"missing": 1})
        self.assertAlmostEqual(p["unreadable_share"], 1 / 6)
        self.assertAlmostEqual(p["SC_readable"], 100 * 8 / 5)
        # An unreadable answer on a target case costs 7, like a NO.
        preds = [dict(x) for x in MODEL]
        preds[0] = {"case_id": "c1", "serious_concern": "maybe", "flags": ["I21.9"]}
        p2, _, c2 = self.stats(preds)
        self.assertAlmostEqual(p2["SC"], 100 * (7 + 7 + 1) / 6)
        self.assertEqual(c2["unreadable_in_r10"], 1)
        self.assertEqual(c2["MT"]["truth_missed"], 1)  # an unreadable case's flags do not count

    def test_constants(self):
        k = vs.constants(self.key)
        self.assertAlmostEqual(k["always_yes"], 100 * 2 / 6)
        self.assertAlmostEqual(k["always_no"], 700 * 2 / 6)
        yes = [{"case_id": x.case_id, "serious_concern": "YES", "flags": []} for x in KEYS]
        no = [{"case_id": x.case_id, "serious_concern": "NO", "flags": []} for x in KEYS]
        self.assertAlmostEqual(self.stats(yes)[0]["SC"], k["always_yes"])
        self.assertAlmostEqual(self.stats(no)[0]["SC"], k["always_no"])
        self.assertAlmostEqual(self.stats(vs.perfect_predictions(self.key, self.m))[0]["SC"], 0.0)

    def test_coverage_and_missed_targets(self):
        p, _, c = self.stats(MODEL)
        # R5 cases: c1 (MI flagged), c2 (PE not flagged), c6 (unreadable) -> mean 1/3.
        self.assertAlmostEqual(p["COV"], 1 / 3)
        self.assertEqual(c["MT"], {"truth_targets": 1, "truth_missed": 0, "dxa_targets": 1, "dxa_missed": 1})
        self.assertAlmostEqual(p["MT_truth"], 0.0)
        self.assertAlmostEqual(p["MT_dxa"], 1.0)

    def test_consistency(self):
        _, _, c = self.stats(MODEL)
        # c3 YES with only a URTI flag; c5 YES with no flags (both tags); c4 NO flagging PE.
        self.assertEqual(c["CON"], {"yes_without_flags": 1, "yes_without_tier1_flag": 2, "no_with_tier1_flag": 1})

    def test_diagnosis(self):
        p, _, c = self.stats(MODEL)
        self.assertEqual(c["DX"]["top1"], 1)  # c1 I21.9 is narrower than MI's I21
        self.assertEqual((c["DX"]["E_events"], c["DX"]["E_den"]), (0, 1))
        # c3: URTI (tier 3) called MI (tier 1) at 90%: a tier gap of 2 at every threshold, naming a graver condition.
        for t in ("60", "70", "80"):
            self.assertEqual(c["DX"]["D2"][t]["events"], 1)
            self.assertEqual(c["DX"]["D2"][t]["milder_named"], 0)
        # D1 covers the 3 cases with a forecast (c1-c3); the 3 without one count against completeness.
        brier = ((0.8 - 1) ** 2 + 0.7 ** 2 + 0.9 ** 2) / 3
        self.assertAlmostEqual(p["D1_brier"], brier)
        self.assertAlmostEqual(p["D1_completeness"], 3 / 6)


class TestSensitivity(ToyBase):
    def test_rows(self):
        p, _, c = self.stats(MODEL)
        self.assertAlmostEqual(p["sens:ratio_1/10"], 100 * (10 + 1) / 6)
        self.assertAlmostEqual(p["sens:ratio_1/5"], 100 * (5 + 1) / 6)
        self.assertAlmostEqual(p["sens:truth_only"], 100 * 1 / 6)  # c2 is not a target
        self.assertAlmostEqual(p["sens:R12.5"], p["SC"])  # PE at 15% stays
        self.assertAlmostEqual(p["sens:R20"], 100 * 1 / 6)  # PE at 15% drops
        self.assertAlmostEqual(p["sens:rate_form"], 100 * (7 * 0.5 + 0.5))
        self.assertAlmostEqual(p["sens:concern_tier2"], 100 * 9 / 6)  # c5 GERD YES now costs 1
        self.assertAlmostEqual(p["sens:H_prime"], p["SC"])  # c1's YES flags MI

    def test_h_prime_counts_yes_without_tier1_flag(self):
        preds = [dict(x) for x in MODEL]
        preds[0] = {"case_id": "c1", "serious_concern": "YES", "flags": ["K21.9"]}  # GERD only
        p, _, _ = self.stats(preds)
        self.assertAlmostEqual(p["SC"], 100 * 8 / 6)
        self.assertAlmostEqual(p["sens:H_prime"], 100 * 15 / 6)

    def test_code_maps(self):
        preds = [dict(x) for x in MODEL]
        preds[0] = {"case_id": "c1", "serious_concern": "YES", "flags": ["I21.9"]}
        p, _, c = self.stats(preds)
        for pol in ("strict", "lenient"):
            self.assertIn(f"map:{pol}:COV", p)
            self.assertIn("CON", c["maps"][pol])

    def test_tier_variant_and_mix(self):
        # A variant where c2 has no target and is clearly low-risk: its NO is now free, and not a concern.
        alt = [case(k.case_id, k.truth, k.truth_tier, k.clearly_low_risk, k.red_flag, k.considered.values()) for k in KEYS]
        alt[1] = case("c2", COPD, 3, clearly_low=True)
        vk = vs.set_key("alt", [k.case_id for k in alt], {k.case_id: k for k in alt})
        w = np.ones(self.key.k)
        w[self.key.conditions.index("URTI")] = 2.0
        p, _, c = self.stats(MODEL, variants={"alt": vk}, mixes={"double-urti": w})
        self.assertAlmostEqual(p["sens:tiers:alt"], 100 * 1 / 6)
        # URTI's two cases count double: cost sum 7 + 2 x 1 over 6 + 2 weighted cases.
        self.assertAlmostEqual(p["sens:mix:double-urti"], 100 * 9 / 8)
        self.assertEqual(c["sensitivity"]["tiers:alt"]["r10_cases"], 1)


class TestBootstrapAndBoard(ToyBase):
    def test_interval_brackets_point_and_is_deterministic(self):
        p, d, _ = self.stats(MODEL)
        lo, hi = vs.interval(d["SC"])
        self.assertLessEqual(lo, p["SC"])
        self.assertGreaterEqual(hi, p["SC"])
        _, d2, _ = self.stats(MODEL)
        np.testing.assert_array_equal(d["SC"], d2["SC"])

    def test_score_set_paired_and_blanket_concern(self):
        yes = [{"case_id": x.case_id, "serious_concern": "YES", "flags": []} for x in KEYS]
        rows = {"a": {"predictions": MODEL, "kind": "model"}, "b": {"predictions": MODEL, "kind": "model"},
                vs.ALWAYS_YES: {"predictions": yes, "kind": "reference"},
                "perfect": {"predictions": vs.perfect_predictions(self.key, self.m), "kind": "constant"}}
        b = vs.score_set("toy", self.key, rows, self.m, n_boot=300, seed=3, sensitivity=False)
        self.assertEqual(b["models"], ["a", "b"])
        d = b["paired"]["a|b"]["SC"]
        self.assertEqual((d["diff"], d["ci"]), (0.0, [0.0, 0.0]))
        va = b["vs_always_yes"]["a"]
        self.assertAlmostEqual(va["SC"]["diff"], 100 * 8 / 6 - 100 * 2 / 6)
        self.assertFalse(va["beats_blanket_concern"])
        self.assertEqual(va["beats_blanket_concern"], va["SC"]["ci"][1] < 0)
        self.assertEqual(sum(r["cost"] for r in b["per_condition"]["a"]), 8.0)

    def test_first_prediction_wins_and_strangers_are_counted(self):
        preds = MODEL + [{"case_id": "c2", "serious_concern": "YES", "flags": []}, {"case_id": "zz", "serious_concern": "YES"}]
        o = vs.outcomes(preds, self.key, self.m)
        self.assertEqual((o.duplicates, o.unknown_cases), (1, 1))
        self.assertFalse(o.yes[1])

    def test_set_detection(self):
        ids = {"main": {"c1", "c2"}, "pool": {"p1"}}
        self.assertEqual(vs.set_of({"test_set_metadata": {"test_set_name": "eval-v03-pool-atypical"}}, [], ids),
                         "pool-atypical")
        self.assertEqual(vs.set_of({}, [{"case_id": "p1"}], ids), "pool")


@unittest.skipUnless(vs.SETS["main"]["key"].exists(), "v0.3 key not built")
class TestRealKeyReferences(unittest.TestCase):
    """The reference rows reproduce docs/v0.3-headline-design.md section 3 on the 470 sample."""

    def test_reference_rows(self):
        board = vs.score_board([], n_boot=400)
        rows = board["sets"]["main"]["rows"]
        self.assertAlmostEqual(rows["perfect"]["point"]["SC"], 0.0)
        self.assertAlmostEqual(round(rows["always-yes"]["point"]["SC"], 1), 25.1)
        self.assertAlmostEqual(round(rows["always-no"]["point"]["SC"], 1), 348.5)
        self.assertAlmostEqual(round(rows["dxa"]["point"]["SC"], 1), 67.0)
        self.assertAlmostEqual(round(rows["naive-bayes"]["point"]["SC"], 1), 50.6)
        self.assertAlmostEqual(round(rows["naive-bayes"]["point"]["sens:R12.5"], 1), 31.3)
        k = board["sets"]["main"]["constants"]
        self.assertAlmostEqual(round(k["always_yes_by_mix"]["nhamcs"], 1), 35.8)

    def test_cli_writes_board_and_report(self):
        from evaluator.v03_score import main
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "b.json"
            main(["--out", str(out), "--n-boot", "50"])
            self.assertIn("main", json.loads(out.read_text())["sets"])
            self.assertIn("## Main sample", out.with_suffix(".md").read_text())


if __name__ == "__main__":
    unittest.main()
