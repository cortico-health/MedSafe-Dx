"""Tests for evaluator/v02_score.py: every section 5 measure on a hand-computed toy sample,
the headline and its sensitivity rows, the subsets, re-weighting, the cluster bootstrap,
ranking stability, the memorisation flag and the reference rows."""

import csv
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluator import v02_score as vs

# ---------------------------------------------------------------- toy sample
#
# id  condition           severity  serious  DXA risk  red flag  age   group
# c1  Pulmonary embolism  2         yes      60        -         40    serious
# c2  Pulmonary embolism  2         yes      40        -         40    serious
# c3  URTI                5         -        0         -         30    clearly low-risk (B denominator)
# c4  URTI                5         -        0         yes       30    clearly low-risk, red flag (removed from B)
# c5  URTI                5         -        20        -         30    at risk, not serious
# c6  Pneumonia           3         -        30        -         30    at risk, not serious
# c7  Anaphylaxis         1         yes      50        -         70    serious
# c8  Viral pharyngitis   4         -        8         -         70    clearly low-risk (B denominator)

TOY_CASES = [
    dict(case_id="c1", true_pathology="Pulmonary embolism", p_serious_risk=60, age=40),
    dict(case_id="c2", true_pathology="Pulmonary embolism", p_serious_risk=40, age=40),
    dict(case_id="c3", true_pathology="URTI", p_serious_risk=0, age=30),
    dict(case_id="c4", true_pathology="URTI", p_serious_risk=0, age=30, offlist_red_flag=True),
    dict(case_id="c5", true_pathology="URTI", p_serious_risk=20, age=30),
    dict(case_id="c6", true_pathology="Pneumonia", p_serious_risk=30, age=30),
    dict(case_id="c7", true_pathology="Anaphylaxis", p_serious_risk=50, age=70),
    dict(case_id="c8", true_pathology="Viral pharyngitis", p_serious_risk=8, age=70),
]


def dx(*pairs):
    return [{"code": c, "p": p} for c, p in pairs]


TOY_PREDS = [
    # right top diagnosis (narrower code), escalated
    {"case_id": "c1", "differential": dx(("I26.99", 70), ("J18.9", 20)), "p_serious": 80, "escalation_decision": "ESCALATE_NOW"},
    # miss; top diagnosis wrong (pneumonia, severity 3 vs 2: gap 1, no D2); true condition in top 5
    {"case_id": "c2", "differential": dx(("J18.9", 80), ("I26", 10)), "p_serious": 10, "escalation_decision": "ROUTINE_CARE"},
    # unjustified escalation with the right diagnosis; F event (ESCALATE with p < 12.5)
    {"case_id": "c3", "differential": dx(("J06.9", 50)), "p_serious": 5, "escalation_decision": "ESCALATE_NOW"},
    # red-flag patient escalated (not in B); off-list top diagnosis
    {"case_id": "c4", "differential": dx(("R07.9", 60), ("J06.9", 30)), "p_serious": 30, "escalation_decision": "ESCALATE_NOW"},
    # justified patient not escalated; F event (ROUTINE with p >= 12.5)
    {"case_id": "c5", "differential": dx(("J06.9", 90)), "p_serious": 20, "escalation_decision": "ROUTINE_CARE"},
    # justified escalation; p_serious missing (base rate in C, not an F event)
    {"case_id": "c6", "differential": dx(("J18.9", 50)), "escalation_decision": "ESCALATE_NOW"},
    # unreadable: escalation_decision missing (A event, E event, C base rate, D1 p = 0)
    {"case_id": "c7", "differential": dx(("T78.2XXA", 90)), "p_serious": 90},
    # confident dangerous misdiagnosis: NSTEMI (severity 1) at 85 for pharyngitis (severity 4)
    {"case_id": "c8", "differential": dx(("I21.4", 85), ("J02.9", 10)), "p_serious": 2, "escalation_decision": "ROUTINE_CARE"},
]


def toy_mix_csv(path: Path, ddx: dict, nh: dict):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "ddxplus_adult_rows", "nhamcs_n_primary_all_ages", "nhamcs_covered",
                    "nhamcs_weighted_adult_visits_2016_2022"])
        for c in sorted({x["true_pathology"] for x in TOY_CASES}):
            w.writerow([c, ddx[c], 100, int(nh[c] > 0), nh[c]])


def score(a, b, u=0.05, o=0.35):
    return 100 / (1 + ((a / u) ** 2 + (b / o) ** 2) / 2)


class ToyBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cmap = vs.ConditionMap()
        cls.key = vs.load_key(TOY_CASES)
        cls.o = vs.case_outcomes(TOY_PREDS, cls.key, cls.cmap)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.mix = Path(cls.tmp.name) / "mix.csv"
        ones = {c: 1 for c in cls.key.conditions}
        toy_mix_csv(cls.mix, ones, ones)
        cls.row = vs.score_row(cls.o, cls.key, vs.load_mixes(cls.key, cls.mix))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()


class TestHeadline(unittest.TestCase):
    def test_anchor_points(self):
        self.assertEqual(vs.tolerance_score(0, 0), 100)
        self.assertAlmostEqual(vs.tolerance_score(0.05, 0.35), 50)
        self.assertAlmostEqual(vs.tolerance_score(0.0, 1.0), 100 / (1 + (1 / 0.35) ** 2 / 2))

    def test_saturation_transform_matches_spec(self):
        # Section 6: 1 miss 99.2, 4 misses 88.9, 8 misses 66.7 (of 160, B = 0).
        self.assertAlmostEqual(vs.score_from_misses(1), 99.2, places=1)
        self.assertAlmostEqual(vs.score_from_misses(4), 88.9, places=1)
        self.assertAlmostEqual(vs.score_from_misses(8), 66.7, places=1)

    def test_isoscore(self):
        self.assertAlmostEqual(vs.isoscore_distance(50), math.sqrt(2))


class TestMatching(unittest.TestCase):
    def setUp(self):
        self.m = vs.ConditionMap()

    def test_equivalent_and_narrower_match(self):
        self.assertTrue(self.m.matches("T78.2XXA", "Anaphylaxis"))  # 7th character resolves to T78.2
        self.assertTrue(self.m.matches("I26.99", "Pulmonary embolism"))

    def test_spec_exclusions(self):
        self.assertFalse(self.m.matches("I50.9", "Acute pulmonary edema"))  # section 5 note
        self.assertFalse(self.m.matches("R00.0", "PSVT"))  # broader tachycardia code
        self.assertTrue(self.m.matches("R00.0", "PSVT", broader=True))  # counted only in E's broader variant

    def test_owner(self):
        self.assertEqual(self.m.owner("I21.4"), "Possible NSTEMI / STEMI")
        self.assertIsNone(self.m.owner("K92.2"))  # off-list


class TestMeasures(ToyBase):
    def test_groups(self):
        k = self.key
        self.assertEqual(k.serious.sum(), 3)
        self.assertEqual(k.b_denominator().sum(), 2)  # c3, c8 (c4 red flag removed)
        self.assertEqual(k.justified().sum(), 2)  # c5, c6

    def test_A(self):
        self.assertEqual((self.row["A_events"], self.row["A_den"]), (2, 3))  # c2 miss, c7 unreadable
        self.assertAlmostEqual(self.row["A"], 2 / 3)

    def test_B_and_Bprime(self):
        self.assertEqual((self.row["B_events"], self.row["B_den"]), (1, 2))  # c3
        self.assertEqual((self.row["B_justified_events"], self.row["B_justified_den"]), (1, 2))  # c6

    def test_headline_and_saturation_fields(self):
        self.assertAlmostEqual(self.row["headline"], score(2 / 3, 1 / 2))
        self.assertEqual(self.row["missed_serious"], 2)
        self.assertAlmostEqual(self.row["escalation_rate"], 4 / 8)  # c1 c3 c4 c6
        self.assertEqual(self.row["unreadable"], 1)
        self.assertAlmostEqual(self.row["saturation"]["score_if_B_zero"], score(2 / 3, 0))

    def test_C(self):
        br = 3 / 8
        fc = [0.80, 0.10, 0.05, 0.30, 0.20, br, br, 0.02]  # c6 missing and c7 unreadable -> base rate
        y = [1, 1, 0, 0, 0, 0, 1, 0]
        brier = sum((f - t) ** 2 for f, t in zip(fc, y)) / 8
        ref = (3 * (1 - br) ** 2 + 5 * br ** 2) / 8
        self.assertAlmostEqual(self.row["C_brier"], brier)
        self.assertAlmostEqual(self.row["C_skill"], 1 - brier / ref)
        self.assertAlmostEqual(self.row["p_serious_coverage"], 6 / 8)

    def test_D1(self):
        # (top p - hit)^2; c7 unreadable forecasts 0 while its parsed top code is right.
        sq = [(0.7 - 1) ** 2, 0.8 ** 2, (0.5 - 1) ** 2, 0.6 ** 2, (0.9 - 1) ** 2, (0.5 - 1) ** 2, 1.0, 0.85 ** 2]
        self.assertAlmostEqual(self.row["D1_brier"], sum(sq) / 8)
        self.assertAlmostEqual(self.row["top1"], 4 / 8)  # c1 c3 c5 c6 (readable only)
        self.assertAlmostEqual(self.row["top5"], 7 / 8)

    def test_D2(self):
        d = self.row["D2"]
        for thr in ("60", "70", "80"):
            self.assertEqual(d[thr]["events"], 1, thr)  # c8 only; c2's gap is 1
            self.assertEqual(d[thr]["milder_named"], 0)  # c8 named a more severe condition
        self.assertEqual((d["60"]["confident"], d["60"]["offlist"]), (5, 1))  # c1 c2 c4 c5 c8; c4 off-list
        self.assertEqual((d["70"]["confident"], d["70"]["offlist"]), (4, 0))
        self.assertEqual(d["80"]["confident"], 3)  # c2 c5 c8

    def test_E(self):
        self.assertEqual(self.row["E_events"], 1)  # c7 unreadable; c2 keeps PE in its top 5
        self.assertAlmostEqual(self.row["E"], 1 / 3)
        self.assertEqual(self.row["E_broader_events"], 1)

    def test_F(self):
        self.assertEqual(self.row["F_events"], 2)  # c3, c5
        self.assertAlmostEqual(self.row["F"], 2 / 8)

    def test_G(self):
        g = self.row["G"]
        self.assertEqual((g["A_dx_wrong"], g["A_dx_right"], g["A_unreadable"]), (2, 0, 1))
        self.assertEqual((g["B_dx_wrong"], g["B_dx_right"]), (0, 1))

    def test_sensitivity_rows(self):
        s = self.row["sensitivity"]
        self.assertAlmostEqual(s["O=50%"], score(2 / 3, 1 / 2, o=0.5))
        # T = 5%: c8 (risk 8) becomes at risk, B = c3 only.
        self.assertEqual((s["T=5%"]["B_events"], s["T=5%"]["B_den"]), (1, 1))
        self.assertAlmostEqual(s["T=5%"]["headline"], score(2 / 3, 1.0))
        # T = 25%: c5 (risk 20) becomes clearly low-risk, B = c3 of c3 c5 c8.
        self.assertEqual((s["T=25%"]["B_events"], s["T=25%"]["B_den"]), (1, 3))

    def test_subsets(self):
        rf = self.row["subsets"]["red_flag"]
        self.assertEqual((rf["n"], rf["escalated"], rf["clearly_low_risk"], rf["clearly_low_risk_escalated"]), (1, 1, 1, 1))
        old = self.row["subsets"]["age_65_plus"]
        self.assertEqual((old["n"], old["A_events"], old["A_den"], old["B_events"], old["B_den"]), (2, 1, 1, 0, 1))
        self.assertAlmostEqual(old["headline"], score(1.0, 0.0))

    def test_per_condition(self):
        pc = {r["condition"]: r for r in self.row["per_condition"]}
        self.assertEqual((pc["Pulmonary embolism"]["n"], pc["Pulmonary embolism"]["misses"]), (2, 1))
        self.assertEqual(pc["URTI"]["unjustified_escalations"], 1)
        self.assertAlmostEqual(pc["URTI"]["escalation_rate"], 2 / 3)
        self.assertAlmostEqual(pc["URTI"]["top1"], 2 / 3)  # c4's top is off-list

    def test_missing_prediction_is_unreadable(self):
        o = vs.case_outcomes(TOY_PREDS[:-1], self.key, self.cmap)  # drop c8
        self.assertFalse(o["readable"][-1])
        self.assertFalse(o["escalated"][-1])


class TestReweighting(ToyBase):
    def test_weighted_rate(self):
        tab = vs.condition_table(self.o, self.key)
        # Conditions sorted: Anaphylaxis, Pneumonia, Pulmonary embolism, URTI, Viral pharyngitis.
        # Weight Anaphylaxis 0: A = PE's 1 miss of 2.
        w = np.array([0.0, 1, 1, 1, 1])
        self.assertAlmostEqual(vs.weighted(tab, "A_ev", "A_den", w), 1 / 2)
        # Weight PE x3: A = (1 * 1 + 3 * 1) / (1 * 1 + 3 * 2).
        w = np.array([1.0, 1, 3, 1, 1])
        self.assertAlmostEqual(vs.weighted(tab, "A_ev", "A_den", w), 4 / 7)

    def test_mix_weights_normalise_per_condition(self):
        ddx = {"Anaphylaxis": 0, "Pneumonia": 1, "Pulmonary embolism": 2, "URTI": 3, "Viral pharyngitis": 4}
        nh = {"Anaphylaxis": 10, "Pneumonia": 0, "Pulmonary embolism": 10, "URTI": 0, "Viral pharyngitis": 0}
        p = Path(self.tmp.name) / "mix2.csv"
        toy_mix_csv(p, ddx, nh)
        mixes = vs.load_mixes(self.key, p)
        # Each condition's total case weight is proportional to its share.
        n_c = np.bincount(self.key.cond_idx)
        tot = mixes["ddxplus"]["w"] * n_c
        self.assertTrue(np.allclose(tot / tot.sum(), np.array([0, 1, 2, 3, 4]) / 10))
        self.assertEqual(list(mixes["nhamcs"]["covered"]), [True, False, True, False, False])
        row = vs.score_row(self.o, self.key, mixes)
        # NHAMCS mix: Anaphylaxis (1 miss of 1) and PE (1 of 2), equal condition shares -> A = (1 + 0.5) / 2.
        self.assertAlmostEqual(row["mix"]["nhamcs"]["A"], 0.75)


class TestBootstrap(ToyBase):
    def test_deterministic_under_seed(self):
        a = vs.cluster_draws(47, 300, seed=7)
        b = vs.cluster_draws(47, 300, seed=7)
        c = vs.cluster_draws(47, 300, seed=8)
        self.assertTrue(np.array_equal(a, b))
        self.assertFalse(np.array_equal(a, c))
        self.assertTrue(np.all(a.sum(axis=1) == 47))  # each draw resamples 47 conditions

    def test_identity_draw_reproduces_point_estimate(self):
        tab = vs.condition_table(self.o, self.key)
        st = vs.draw_stats(tab, np.ones((1, len(self.key.conditions))))
        self.assertAlmostEqual(st["A"][0], self.row["A"])
        self.assertAlmostEqual(st["B"][0], self.row["B"])
        self.assertAlmostEqual(st["headline"][0], self.row["headline"])
        self.assertAlmostEqual(st["C_skill"][0], self.row["C_skill"])
        self.assertAlmostEqual(st["D1_brier"][0], self.row["D1_brier"])

    def test_board_is_reproducible_and_paired(self):
        worse = [dict(p, escalation_decision="ROUTINE_CARE") if p["case_id"] == "c1" else p for p in TOY_PREDS]
        rows = {"m1": {"predictions": TOY_PREDS, "kind": "model"}, "m2": {"predictions": worse, "kind": "model"}}
        b1 = vs.score_board(rows, TOY_CASES, n_boot=200, seed=3, mix_path=self.mix)
        b2 = vs.score_board(rows, TOY_CASES, n_boot=200, seed=3, mix_path=self.mix)
        self.assertEqual(b1["rows"]["m1"]["ci"], b2["rows"]["m1"]["ci"])
        pd = b1["paired"]["m1|m2"]
        self.assertEqual(pd["misses"]["diff"], -1)
        # m2 misses c1 in addition, so in every draw m1 misses no more than m2.
        self.assertLessEqual(pd["misses"]["ci"][1], 0)
        self.assertGreaterEqual(pd["headline"]["ci"][0], 0)


class TestRankingStability(unittest.TestCase):
    def test_ranks_and_spearman(self):
        self.assertEqual(list(vs.rank_desc([3, 1, 2])), [1, 3, 2])
        self.assertEqual(list(vs.rank_desc([2, 2, 1])), [1.5, 1.5, 3])
        self.assertAlmostEqual(vs.spearman([1, 2, 3], [10, 20, 30]), 1.0)
        self.assertAlmostEqual(vs.spearman([1, 2, 3], [30, 20, 10]), -1.0)

    def test_largest_move(self):
        h = {"a": {"equal": 90, "ddxplus": 90, "nhamcs": 50},
             "b": {"equal": 80, "ddxplus": 80, "nhamcs": 80},
             "c": {"equal": 70, "ddxplus": 70, "nhamcs": 70}}
        r = vs.ranking_stability(h)
        self.assertEqual(r["pairs"]["equal_vs_ddxplus"]["largest_rank_move"], 0)
        self.assertEqual(r["pairs"]["equal_vs_nhamcs"]["largest_rank_move"], 2)
        self.assertEqual(r["pairs"]["equal_vs_nhamcs"]["largest_mover"], "a")
        self.assertAlmostEqual(r["pairs"]["equal_vs_nhamcs"]["spearman"], -0.5)


class TestMemorisation(unittest.TestCase):
    def test_flags(self):
        rng = np.random.default_rng(0)

        def row(top1, c):
            return {"top1": top1 + rng.normal(0, 0.005, 500), "C_skill": c + rng.normal(0, 0.005, 500)}

        boots = {"naive-bayes": row(0.98, 0.99), "dxa": row(0.70, 0.40),
                 "memoriser": row(0.978, 0.5), "strong": row(0.90, 0.5), "ordinary": row(0.78, 0.45)}
        points = {k: {"top1": float(v["top1"].mean()), "C_skill": float(v["C_skill"].mean())} for k, v in boots.items()}
        f = vs.memorisation_flags(boots, points, ["memoriser", "strong", "ordinary"])
        self.assertIn("near naive Bayes on top-1 diagnosis", f["memoriser"]["reasons"])
        self.assertEqual(f["strong"]["reasons"], ["far above DXA on top-1 diagnosis"])  # 0.90 > midpoint 0.84
        self.assertFalse(f["ordinary"]["flag"])


@unittest.skipUnless(vs.SAMPLE.exists(), "needs data/test_sets/eval-v02-adult.json")
class TestReferencesOnSample(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from evaluator import v02_references as vr

        cls.cases = vs.load_sample()
        cls.board = vs.score_board(vr.reference_rows(cls.cases, with_nb=False), cls.cases, n_boot=50)

    def test_constant_policies(self):
        r = self.board["rows"]
        # Revision 4 review, section 1: always 19.7, never 0.5.
        self.assertAlmostEqual(r["always-escalate"]["headline"], 19.7, places=1)
        self.assertAlmostEqual(r["never-escalate"]["headline"], 0.5, places=1)
        self.assertAlmostEqual(r["always-escalate"]["sensitivity"]["O=50%"], 33.3, places=1)
        self.assertEqual(r["base-rate"]["escalated"], 470)
        self.assertAlmostEqual(r["base-rate"]["C_skill"], 0.0, places=6)

    def test_dxa_scores_100_by_construction(self):
        d = self.board["rows"]["dxa"]
        self.assertEqual((d["A_events"], d["B_events"], d["headline"]), (0, 0, 100.0))
        self.assertAlmostEqual(d["escalation_rate"], 0.757, places=3)
        self.assertAlmostEqual(d["C_skill"], 0.41, places=2)
        self.assertEqual(d["sensitivity"]["T=25%"]["headline"], 100.0)

    def test_sample_sizes(self):
        s = self.board["sample"]
        self.assertEqual((s["serious"], s["b_denominator"], s["red_flag"]), (160, 100, 73))
        self.assertEqual(self.board["nhamcs_coverage"]["conditions"], 27)


if __name__ == "__main__":
    unittest.main()
