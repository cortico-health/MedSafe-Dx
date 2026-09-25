import csv
import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from evaluator import answer_key_v03 as ak
from evaluator.schemas_v03 import parse_v03

ROOT = Path(__file__).resolve().parents[2]
TS = ROOT / "data/test_sets"
TIERS = {"T1a": 1, "T1b": 1, "T1c": 1, "T2": 2, "T3": 3}


def lookup_from(table):
    """ClassLookup from {condition: (n, k, hallmark tokens)}."""
    def f(cond, p):
        n, k, h = table.get(cond, (0, 0, ()))
        return ak.ClassStats(n=n, k=k, hallmark_tokens=tuple(h))
    return f


class TestRule(unittest.TestCase):
    def test_bands(self):
        self.assertEqual(ak.band_label(ak.band_of(4.99)), "0-5%")
        self.assertEqual(ak.band_label(ak.band_of(5.0)), "5-10%")
        self.assertEqual(ak.band_label(ak.band_of(10.0)), "10-20%")
        self.assertEqual(ak.band_label(ak.band_of(100.0)), "50-100%")

    def test_floor_is_one_percent_at_every_threshold(self):
        self.assertEqual(ak.rh_threshold(5.0), 0.01)
        self.assertEqual(ak.rh_threshold(9.9), 0.01)
        self.assertAlmostEqual(ak.rh_threshold(30.0), 0.03)
        # 5-10% band: kept only at a class rate of 1% or more
        self.assertEqual(ak.rh_status(7.0, 1000, 9), ak.RED_HERRING)  # 0.9%
        self.assertEqual(ak.rh_status(7.0, 1000, 10), ak.KEPT)  # 1.0%
        # above 10%, the ratio p / 10 governs
        self.assertEqual(ak.rh_status(30.0, 1000, 29), ak.RED_HERRING)
        self.assertEqual(ak.rh_status(30.0, 1000, 30), ak.KEPT)

    def test_small_classes_are_undetermined(self):
        self.assertEqual(ak.rh_status(50.0, 29, 0), ak.UNDETERMINED)
        self.assertEqual(ak.rh_status(50.0, 30, 0), ak.RED_HERRING)
        self.assertEqual(ak.rh_status(50.0, 0, 0), ak.UNDETERMINED)


class TestCaseKey(unittest.TestCase):
    def rows(self, truth, dxa, table, flags=()):
        return ak.case_key_rows("c", truth, dxa, TIERS, lookup_from(table), flags)

    def by_cond(self, rows):
        return {r["condition"]: r for r in rows}

    def test_truth_is_always_a_target(self):
        rows = self.rows("T1a", {"T1a": 1.0}, {"T1a": (1000, 0, ())})
        r = self.by_cond(rows)["T1a"]
        self.assertEqual((r["source"], r["in_r10"], r["in_r5"], r["status"]), ("truth", True, True, ak.KEPT))
        self.assertEqual(rows[0]["r10_targets"], "T1a")

    def test_dxa_targets_by_threshold_and_rule(self):
        dxa = {"T1a": 12.0, "T1b": 6.0, "T1c": 4.0, "T2": 40.0}
        table = {"T1a": (500, 100, ("E_1",)), "T1b": (500, 10, ("E_2", "E_3")), "T1c": (500, 400, ())}
        rows = self.rows("T3", dxa, table)
        d = self.by_cond(rows)
        self.assertNotIn("T1c", d)  # under 5%: not considered
        self.assertNotIn("T2", d)  # not tier 1
        self.assertEqual((d["T1a"]["in_r10"], d["T1a"]["in_r5"]), (True, True))
        self.assertEqual((d["T1b"]["in_r10"], d["T1b"]["in_r5"]), (False, True))  # 2% class, 6% p: R5 only
        self.assertEqual(d["T1b"]["hallmark_tokens"], "E_2|E_3")
        self.assertEqual(rows[0]["r5_targets"], "T1a|T1b")
        self.assertFalse(rows[0]["clearly_low_risk"])
        self.assertFalse(rows[0]["intermediate"])

    def test_red_herring_removed_and_undetermined_kept(self):
        rows = self.rows("T2", {"T1a": 30.0, "T1b": 30.0}, {"T1a": (5000, 1, ()), "T1b": (12, 0, ())})
        d = self.by_cond(rows)
        self.assertEqual(d["T1a"]["status"], ak.RED_HERRING)
        self.assertFalse(d["T1a"]["in_r5"])
        self.assertEqual(d["T1b"]["status"], ak.UNDETERMINED)
        self.assertTrue(d["T1b"]["undetermined"])
        self.assertTrue(d["T1b"]["in_r10"])

    def test_clearly_low_and_intermediate(self):
        low = self.rows("T3", {"T3": 90.0}, {})
        self.assertEqual(len(low), 1)
        self.assertEqual(low[0]["condition"], "")
        self.assertTrue(low[0]["clearly_low_risk"])
        flagged = self.rows("T3", {"T3": 90.0}, {}, ["bleeding"])
        self.assertFalse(flagged[0]["clearly_low_risk"])
        self.assertTrue(flagged[0]["intermediate"])
        tier2 = self.rows("T2", {"T2": 90.0}, {})
        self.assertTrue(tier2[0]["intermediate"])
        r5only = self.rows("T3", {"T1a": 6.0}, {"T1a": (500, 50, ())})
        self.assertTrue(r5only[0]["intermediate"])

    def test_roundtrip_and_hash_pin(self):
        rows = (self.rows("T1a", {"T1a": 20.0, "T1b": 7.0}, {"T1b": (100, 5, ("E_9",))})
                + self.rows("T3", {"T3": 90.0}, {}))
        with tempfile.TemporaryDirectory() as d:
            path, sha = Path(d) / "k.csv", Path(d) / "k.sha256"
            ak.write_key(path, rows)
            ak.write_sha256(sha, path)
            key = ak.load_key(path, sha)
            self.assertEqual(list(key), ["c"])  # same case id twice collapses to one case
            k = key["c"]
            self.assertEqual(k.r10, ["T1a"])
            self.assertEqual(k.r5, ["T1a", "T1b"])
            self.assertEqual(k.targets_at(5.0), ["T1a", "T1b"])
            self.assertEqual(k.considered["T1b"].hallmark_tokens, ("E_9",))
            sha.write_text("0" * 64 + "  k.csv\n")
            with self.assertRaises(ValueError):
                ak.load_key(path, sha)


@unittest.skipUnless((TS / "eval-v03-key.csv").exists(), "run scripts/build_v03_key.py first (data/test_sets/ is git-ignored)")
class TestBuiltKey(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.key = ak.load_key()  # checks the pinned hash
        cls.tiers = ak.load_tiers()

    def test_sample_and_counts(self):
        ids = (TS / "eval-v02-adult.case_ids.txt").read_text().split()
        self.assertEqual(list(self.key), ids)
        k = self.key.values()
        self.assertEqual(sum(bool(c.r10) for c in k), 234)
        self.assertEqual(sum(bool(c.r5) for c in k), 269)
        self.assertEqual(sum(c.clearly_low_risk for c in k), 118)
        self.assertEqual(sum(c.intermediate for c in k), 118)

    def test_invariants(self):
        for c in self.key.values():
            self.assertLessEqual(set(c.r10), set(c.r5))
            self.assertEqual(c.truth_tier, self.tiers[c.truth])
            if self.tiers[c.truth] == 1:
                self.assertIn(c.truth, c.r10)
            self.assertEqual(c.clearly_low_risk, not c.r5 and c.truth_tier == 3 and not c.red_flag)
            self.assertEqual(c.intermediate, not c.r10 and not c.clearly_low_risk)
            for t in c.considered.values():
                self.assertEqual(self.tiers[t.condition], 1)
                if t.source == "dxa":
                    self.assertGreaterEqual(t.dxa_p, 5.0)
                    if t.in_r5 and not t.undetermined:
                        self.assertGreaterEqual(t.class_rate, 0.01)  # the 1% floor
                    self.assertEqual(t.undetermined, t.class_n < 30)


@unittest.skipUnless((TS / "eval-v03-pool-atypical.json").exists(), "run scripts/build_v03_key.py first")
class TestPools(unittest.TestCase):
    def test_pools(self):
        tiers = ak.load_tiers()
        main = set((TS / "eval-v02-adult.case_ids.txt").read_text().split())
        seen = set()
        for name in ("atypical", "high-risk"):
            stem = f"eval-v03-pool-{name}"
            data = json.loads((TS / f"{stem}.json").read_text())
            text = (TS / f"{stem}.case_ids.txt").read_text()
            ids = text.split()
            self.assertEqual(ids, [c["case_id"] for c in data["cases"]])
            self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), data["metadata"]["case_ids_sha256"])
            self.assertFalse(set(ids) & main)
            self.assertFalse(set(ids) & seen)
            seen |= set(ids)
            self.assertTrue(all(int(c["age"]) >= 18 for c in data["cases"]))
            per = data["metadata"]["per_condition"]
            self.assertTrue(all(len(v) <= 10 for v in per.values()))
            self.assertEqual(sorted(i for v in per.values() for i in v), sorted(ids))
            key = ak.load_key(TS / f"{stem}.key.csv", TS / f"{stem}.key.sha256")
            self.assertEqual(sorted(key), sorted(ids))
            for cond, lst in per.items():
                for cid in lst:
                    k = key[cid]
                    if name == "atypical":
                        self.assertEqual(k.truth, cond)
                        self.assertEqual(tiers[k.truth], 1)
                    else:
                        self.assertNotEqual(tiers[k.truth], 1)
                        t = k.considered[cond]
                        self.assertTrue(t.in_r10 and not t.undetermined)

    def test_atypical_top_diagnosis_is_tier3(self):
        tiers = ak.load_tiers()
        data = json.loads((TS / "eval-v03-pool-atypical.json").read_text())
        for c in data["cases"]:
            dd = c["ddxplus_differential"]
            top = max(p for _, p in dd)
            self.assertEqual(min(tiers[n] for n, p in dd if abs(p - top) < 1e-12), 3)


@unittest.skipUnless((TS / "eval-v03-adult.refs.json").exists(), "run scripts/build_v03_key.py first")
class TestReferences(unittest.TestCase):
    def test_every_case_every_reference_parses(self):
        for name in ("adult", "pool-atypical", "pool-high-risk"):
            ids = (TS / (f"eval-v03-{name}.case_ids.txt" if name != "adult" else "eval-v02-adult.case_ids.txt")).read_text().split()
            refs = json.loads((TS / f"eval-v03-{name}.refs.json").read_text())["references"]
            self.assertEqual(set(refs), {"always-yes", "always-no", "dxa", "naive-bayes"})
            for rname, preds in refs.items():
                self.assertEqual([p["case_id"] for p in preds], ids, (name, rname))
                for p in preds:
                    parsed = parse_v03(p)
                    self.assertTrue(parsed.readable)
                    self.assertEqual(parsed.yes, bool(parsed.flags), (name, rname))

    def test_always_yes_list(self):
        refs = json.loads((TS / "eval-v03-adult.refs.json").read_text())["references"]
        key = ak.load_key()
        r10 = Counter(c for k in key.values() for c in k.r10)
        flags = refs["always-yes"][0]["flags"]
        self.assertEqual(len(flags), 5)
        top5 = [c for c, _ in r10.most_common()][:5]
        self.assertGreaterEqual(min(r10[c] for c in top5), max((r10[c] for c in r10 if c not in top5), default=0))


if __name__ == "__main__":
    unittest.main()
