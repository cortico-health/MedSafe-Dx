import csv
import hashlib
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

from evaluator import answer_key_v03 as ak
from evaluator import answer_key_v03b as akb
from evaluator import key_views_v03 as kv
from evaluator import tier_evidence_v03b as te

ROOT = Path(__file__).resolve().parents[2]
TS = ROOT / "data/test_sets"
TIERS = {"T1a": 1, "T1b": 1, "T1c": 1, "T2": 2, "T3": 3}
N = akb.N_BANDS - akb.FIRST_BAND  # bands from 5% up


def lookup_from(table):
    """CellsLookup from {condition: [(n, k) per band from 5% up]}; missing conditions get empty classes."""
    def f(cond, p):
        return akb.ClassCells(hallmark_tokens=("h1",), cells=tuple(table.get(cond, [(0, 0)] * N)))
    return f


def flat(n, k):
    return [(n, k)] * N


SUPPORTED_CELLS = flat(1000, 50)  # 5%, lower bound about 3.8%
UNSUPPORTED_CELLS = flat(5000, 0)  # upper bound about 0.08%
UNCERTAIN_CELLS = flat(20, 0)  # upper bound about 16%


# ---------------------------------------------------------------- #9 interval rule


class TestWilson(unittest.TestCase):
    def test_zero_events(self):
        lo, hi = akb.wilson(0, 30)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 3.8415 / 33.8415, places=4)

    def test_known_value(self):
        lo, hi = akb.wilson(10, 100)
        self.assertAlmostEqual(lo, 0.0552, places=4)
        self.assertAlmostEqual(hi, 0.1744, places=4)

    def test_empty_class(self):
        self.assertEqual(akb.wilson(0, 0), (0.0, 1.0))


class TestRawStatus(unittest.TestCase):
    def test_band_thresholds(self):
        self.assertEqual([round(akb.band_threshold(b), 4) for b in range(akb.FIRST_BAND, akb.N_BANDS)],
                         [0.01, 0.01, 0.02, 0.035, 0.05])

    def test_three_outcomes(self):
        b = akb.FIRST_BAND
        self.assertEqual(akb.raw_status(50, 1000, b), akb.SUPPORTED)
        self.assertEqual(akb.raw_status(0, 5000, b), akb.UNSUPPORTED)
        self.assertEqual(akb.raw_status(0, 20, b), akb.UNCERTAIN)
        self.assertEqual(akb.raw_status(0, 0, b), akb.UNCERTAIN)

    def test_supported_needs_lower_bound_at_one_percent(self):
        b = akb.FIRST_BAND
        self.assertEqual(akb.raw_status(12, 1000, b), akb.UNCERTAIN)  # 1.2%, lower bound 0.69%
        self.assertEqual(akb.raw_status(120, 10000, b), akb.SUPPORTED)  # 1.2%, lower bound 1.004%

    def test_unsupported_threshold_rises_with_band(self):
        # 6 of 400 = 1.5%: 95% 0.69-3.2%; under the 50-100% band's 5% threshold but not the 5-10% band's 1%
        top = akb.N_BANDS - 1
        self.assertEqual(akb.raw_status(6, 400, akb.FIRST_BAND), akb.UNCERTAIN)
        self.assertEqual(akb.raw_status(6, 400, top), akb.UNSUPPORTED)

    def test_supported_is_tested_first(self):
        # 3% of 10000: lower bound 2.7% >= 1%, upper 3.3% < the top band's 5%: still supported
        self.assertEqual(akb.raw_status(300, 10000, akb.N_BANDS - 1), akb.SUPPORTED)


class TestPooling(unittest.TestCase):
    def test_falling_rates_pool(self):
        blocks = akb.pool_adjacent([(100, 10), (100, 2), (100, 30)])
        self.assertEqual(blocks[0], [0, 1])
        self.assertEqual(blocks[1], [0, 1])
        self.assertEqual(blocks[2], [2])

    def test_empty_cells_stay_alone(self):
        blocks = akb.pool_adjacent([(100, 10), (0, 0), (100, 2)])
        self.assertEqual(blocks[1], [1])
        self.assertEqual(blocks[0], [0, 2])

    def test_pooled_rates_never_fall(self):
        rng = random.Random(7)
        for _ in range(500):
            cells = [(n, rng.randint(0, n)) if n else (0, 0) for n in (rng.choice([0, 5, 40, 300, 3000]) for _ in range(N))]
            blocks = akb.pool_adjacent(cells)
            rates = []
            for i, (n, _) in enumerate(cells):
                if n:
                    pn = sum(cells[j][0] for j in blocks[i])
                    pk = sum(cells[j][1] for j in blocks[i])
                    rates.append(pk / pn)
            self.assertTrue(all(b >= a - 1e-12 for a, b in zip(rates, rates[1:])), cells)


class TestMonotone(unittest.TestCase):
    def test_status_never_falls_with_band(self):
        rng = random.Random(11)
        for _ in range(2000):
            cells = [(n, rng.randint(0, max(0, n // rng.choice([1, 5, 50, 1000])))) if n else (0, 0)
                     for n in (rng.choice([0, 3, 30, 200, 2000, 20000]) for _ in range(N))]
            ranks = [akb.RANK[s.status] for s in akb.band_statuses(cells)]
            self.assertEqual(ranks, sorted(ranks), cells)

    def test_guarantee_raises_a_later_band(self):
        # band 1: 1.2% of 100000, supported; the top band: 2% of 100, a higher rate (no pooling)
        # but too few patients for a lower bound of 1%, so alone it would be uncertain
        cells = [(100000, 1200), (0, 0), (0, 0), (0, 0), (100, 2)]
        st = akb.band_statuses(cells)
        self.assertEqual(st[-1].pooled_bands, (akb.N_BANDS - 1,))
        self.assertEqual(st[-1].raw, akb.UNCERTAIN)
        self.assertEqual(st[-1].status, akb.SUPPORTED)
        self.assertTrue(st[-1].raised)

    def test_same_class_same_status_at_any_p(self):
        # Astra finding 3: one class, DXA 10.97% and 15.71%; the v0.3 rule keeps one and removes the other
        n, k = 2431, 30  # 1.23%
        self.assertEqual(ak.rh_status(10.97, n, k), ak.KEPT)
        self.assertEqual(ak.rh_status(15.71, n, k), ak.RED_HERRING)
        cells = [(0, 0)] * N
        cells[ak.band_of(10.97) - akb.FIRST_BAND] = (n, k)
        tab = {"T1b": cells}
        a = {r["condition"]: r for r in akb.case_key_rows("c", "T3", {"T1b": 10.97}, TIERS, lookup_from(tab), [])}
        b = {r["condition"]: r for r in akb.case_key_rows("c", "T3", {"T1b": 15.71}, TIERS, lookup_from(tab), [])}
        self.assertEqual(a["T1b"]["status"], b["T1b"]["status"])


# ---------------------------------------------------------------- #7 and #8 case classes


class TestCaseKey(unittest.TestCase):
    def case(self, truth, dxa, table, flags=(), comps={}):
        rows = akb.case_key_rows("c", truth, dxa, TIERS, lookup_from(table), flags, {"T1a": "L"}, complications=comps)
        return rows[0], {r["condition"]: r for r in rows if r["condition"]}

    def test_truth_target(self):
        c, t = self.case("T1a", {"T1a": 1.0}, {})
        self.assertEqual((t["T1a"]["target_source"], t["T1a"]["status"], t["T1a"]["in_r10"]), ("truth", "truth", True))
        self.assertEqual(c["evidence_class"], akb.SERIOUS_TRUTH)
        self.assertEqual(c["r10_truth_targets"], "T1a")
        self.assertEqual(c["truth_evidence_level"], "L")

    def test_supported_dxa_target_makes_dxa_only_serious(self):
        c, t = self.case("T2", {"T1b": 12.0}, {"T1b": SUPPORTED_CELLS})
        self.assertEqual((t["T1b"]["target_source"], t["T1b"]["in_r10"], t["T1b"]["in_r5"]), ("dxa", True, True))
        self.assertEqual(c["evidence_class"], akb.SERIOUS_DXA_ONLY)
        self.assertEqual(c["r10_dxa_targets"], "T1b")
        self.assertEqual(c["spec_class"], "SERIOUS")

    def test_supported_at_5_to_10_is_r5_only(self):
        c, t = self.case("T3", {"T1b": 7.0}, {"T1b": SUPPORTED_CELLS})
        self.assertEqual((t["T1b"]["in_r10"], t["T1b"]["in_r5"]), (False, True))
        self.assertEqual((c["evidence_class"], c["spec_class"], c["clearly_low_risk"]), (akb.MIDDLE, "OTHER", False))

    def test_uncertain_is_no_target_and_exempts_over_concern(self):
        c, t = self.case("T3", {"T1b": 30.0}, {"T1b": UNCERTAIN_CELLS})
        self.assertEqual((t["T1b"]["status"], t["T1b"]["in_r5"], t["T1b"]["oc_exempting"]), (akb.UNCERTAIN, False, True))
        self.assertEqual((c["evidence_class"], c["clearly_low_risk"], c["oc_chargeable"]), (akb.BENIGN, True, False))
        self.assertEqual(c["oc_exempt_concerns"], "T1b@30.0:uncertain")

    def test_removed_concern_exempts_over_concern(self):
        c, t = self.case("T3", {"T1b": 55.0}, {"T1b": UNSUPPORTED_CELLS})
        self.assertEqual(t["T1b"]["status"], akb.UNSUPPORTED)
        self.assertEqual(t["T1b"]["status_v03"], ak.RED_HERRING)
        self.assertFalse(c["oc_chargeable"])

    def test_concern_under_5_is_ignored(self):
        c, t = self.case("T3", {"T1b": 4.9}, {"T1b": UNCERTAIN_CELLS})
        self.assertEqual(t, {})
        self.assertTrue(c["oc_chargeable"])

    def test_chargeable_benign(self):
        c, _ = self.case("T3", {}, {})
        self.assertEqual((c["evidence_class"], c["spec_class"], c["oc_chargeable"]), (akb.BENIGN, "BENIGN", True))

    def test_red_flag_and_tier2_are_middle(self):
        c, _ = self.case("T3", {}, {}, flags=["bleeding"])
        self.assertEqual((c["evidence_class"], c["spec_class"], c["oc_chargeable"]), (akb.MIDDLE, "OTHER", False))
        c, _ = self.case("T2", {}, {})
        self.assertEqual((c["evidence_class"], c["spec_class"]), (akb.MIDDLE, "MIDDLE"))

    def test_offlist_complication_flags_but_stays_chargeable(self):
        c, _ = self.case("T3", {}, {}, comps={"T3": "orbital complication (1/1000; src)"})
        self.assertTrue(c["oc_chargeable"])
        self.assertEqual(c["offlist_complication"], "orbital complication (1/1000; src)")


def sample_key():
    rows = []
    rows += akb.case_key_rows("a", "T1a", {"T1a": 1.0}, TIERS, lookup_from({}), [])
    rows += akb.case_key_rows("b", "T2", {"T1b": 12.0}, TIERS, lookup_from({"T1b": SUPPORTED_CELLS}), [])
    rows += akb.case_key_rows("c", "T3", {}, TIERS, lookup_from({}), [])
    rows += akb.case_key_rows("d", "T3", {"T1b": 30.0}, TIERS, lookup_from({"T1b": UNCERTAIN_CELLS}), [])
    rows += akb.case_key_rows("e", "T3", {}, TIERS, lookup_from({}), [], complications={"T3": "x"})
    rows += akb.case_key_rows("f", "T2", {}, TIERS, lookup_from({}), [])
    return rows


class TestFileAndViews(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        akb.write_key(d / "k.csv", sample_key())
        ak.write_sha256(d / "k.sha256", d / "k.csv")
        self.d = d
        self.key = akb.load_key(d / "k.csv", d / "k.sha256")

    def tearDown(self):
        self.tmp.cleanup()

    def test_round_trip(self):
        self.assertEqual(list(self.key), list("abcdef"))
        self.assertEqual(self.key["b"].considered["T1b"].source, "dxa")
        self.assertEqual(self.key["b"].r10, ["T1b"])
        self.assertEqual(self.key["d"].oc_exempt_concerns, ("T1b@30.0:uncertain",))
        self.assertEqual(self.key["d"].targets_at(5.0), [])
        self.assertEqual(self.key["b"].targets_at(12.5), [])

    def test_hash_mismatch_raises(self):
        (self.d / "k.sha256").write_text("0" * 64 + "  k.csv\n")
        with self.assertRaises(ValueError):
            akb.load_key(self.d / "k.csv", self.d / "k.sha256")

    def test_views(self):
        self.assertEqual(kv.headline(self.key), (["a", "b"], ["c", "e"]))
        self.assertEqual(kv.truth_only_headline(self.key), (["a"], ["c", "e"]))
        self.assertEqual(kv.chargeable_benign(self.key, strict=True), ["c"])
        self.assertEqual(kv.exempt_benign(self.key), ["d"])
        self.assertEqual(kv.pool_with_controls(self.key, self.key), (["a", "b"], ["c", "e"]))
        self.assertEqual(kv.class_counts(self.key)["MIDDLE"], 1)

    def test_dxa_supported_risk_ignored(self):
        line = kv.dxa_supported_risk_ignored(self.key, {"b": True})
        self.assertEqual((line["n"], line["not_escalated"]), (1, 0))
        for esc in ({}, {"b": None}, {"b": False}):
            self.assertEqual(kv.dxa_supported_risk_ignored(self.key, esc)["not_escalated"], 1)


# ---------------------------------------------------------------- #11 tier evidence


class TestTierEvidence(unittest.TestCase):
    def v03(self):
        with open(ROOT / "spec/dangerous_if_missed_tiers_v03.csv", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def test_reproduces_v03_tiers(self):
        for r in self.v03():
            self.assertEqual(te.tier_under(r["condition"], int(r["ddxplus_severity"]), te.counts_v03)[0],
                             int(r["final_tier"]), r["condition"])

    def test_formal_rule_changes_only_rib_fracture(self):
        changed = {r["condition"]: te.tier_under(r["condition"], int(r["ddxplus_severity"]), te.counts_formal)
                   for r in self.v03()
                   if te.tier_under(r["condition"], int(r["ddxplus_severity"]), te.counts_formal)[0] != int(r["final_tier"])}
        self.assertEqual(changed, {"Spontaneous rib fracture": (1, te.LEVEL_TWO_OF_THREE)})

    def test_category_rule(self):
        rows = te.SOURCE_ROWS
        other_cancers = rows["Pancreatic neoplasm"][0]
        self.assertTrue(te.counts_formal("Pancreatic neoplasm", other_cancers, rows["Pancreatic neoplasm"]))
        # a source with a condition-level row does not also count its category row
        af = rows["Atrial fibrillation"]
        self.assertFalse(te.counts_formal("Atrial fibrillation", af[1], af))
        # rows that name no family never count
        self.assertFalse(te.counts_formal("URTI", rows["URTI"][0], rows["URTI"]))
        # v0.3 counts only the listed exception
        self.assertFalse(te.counts_v03("Spontaneous rib fracture", rows["Spontaneous rib fracture"][0],
                                       rows["Spontaneous rib fracture"]))

    def test_count_floor_moves_only_asthma(self):
        moved = [r["condition"] for r in self.v03()
                 if te.tier_under(r["condition"], int(r["ddxplus_severity"]), te.counts_v03, count_floor=2)[0]
                 != int(r["final_tier"])]
        self.assertEqual(moved, ["Bronchospasm / acute asthma exacerbation"])

    def test_evidence_levels(self):
        lv = {r["condition"]: te.tier_under(r["condition"], int(r["ddxplus_severity"]), te.counts_v03)[1] for r in self.v03()}
        self.assertEqual(lv["Pancreatic neoplasm"], te.LEVEL_NT_CATEGORY)
        self.assertEqual(lv["Bronchospasm / acute asthma exacerbation"], te.LEVEL_TWO_OF_THREE)
        self.assertEqual(lv["Pneumonia"], te.LEVEL_NT_NAMED)
        self.assertEqual(lv["Stable angina"], te.LEVEL_SEVERITY)

    def test_committed_table_is_current(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import build_v03b_tiers
        with open(ROOT / "spec/dangerous_if_missed_tiers_v03b.csv", newline="", encoding="utf-8") as f:
            committed = list(csv.DictReader(f))
        self.assertEqual(committed, [{k: str(v) for k, v in r.items()} for r in build_v03b_tiers.build()])
        self.assertEqual(akb.load_tiers(), ak.load_tiers())


# ---------------------------------------------------------------- built files


@unittest.skipUnless((TS / "eval-v03b-key.csv").exists(), "run scripts/build_v03b_key.py first")
class TestBuiltKey(unittest.TestCase):
    def test_counts(self):
        key = akb.load_key()
        c = kv.class_counts(key)
        self.assertEqual((c["SERIOUS_TRUTH"], c["SERIOUS_DXA_ONLY"], c["chargeable_benign"]), (200, 34, 52))
        self.assertEqual(sum(bool(k.r10) for k in key.values()), 234)
        for k in key.values():
            if k.oc_chargeable:
                self.assertFalse(any(t.source == "dxa" and t.dxa_p >= 5 for t in k.considered.values()))
            for t in k.considered.values():
                if t.source == "dxa":
                    self.assertEqual(t.in_r5, t.status == akb.SUPPORTED)

    def test_controls(self):
        main = set((TS / "eval-v02-adult.case_ids.txt").read_text().split())
        pools = {n: (TS / f"eval-v03-pool-{n}.case_ids.txt").read_text().split() for n in ("atypical", "high-risk")}
        used = set()
        for name, pool in pools.items():
            stem = f"eval-v03b-pool-{name}-controls"
            ids = (TS / f"{stem}.case_ids.txt").read_text().split()
            with open(TS / f"{stem}.pairs.csv", newline="", encoding="utf-8") as f:
                pairs = list(csv.DictReader(f))
            self.assertEqual([p["pool_case"] for p in pairs], pool)
            self.assertEqual([p["control"] for p in pairs], ids)
            self.assertEqual(len(set(ids)), len(ids))
            self.assertFalse(set(ids) & (main | set(pools["atypical"]) | set(pools["high-risk"]) | used))
            used |= set(ids)
            key = akb.load_key(TS / f"{stem}.key.csv", TS / f"{stem}.key.sha256")
            self.assertTrue(all(key[i].oc_chargeable for i in ids))
            for p in pairs:
                if p["match_level"].startswith("initial_evidence"):
                    self.assertEqual(p["pool_initial_evidence"], p["control_initial_evidence"])
                if p["match_level"] in ("initial_evidence+age_band", "age_band"):
                    self.assertEqual(p["pool_age_band"], p["control_age_band"])
            meta = json.loads((TS / f"{stem}.json").read_text())["metadata"]
            self.assertEqual(meta["case_ids_sha256"], hashlib.sha256((TS / f"{stem}.case_ids.txt").read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
