"""The rendering variants (inference/rendering.py) change the surface of the intake and nothing
else: every finding survives each rendering, and a rerun renders a case the same way."""

import collections
import json
import re
import unittest
from pathlib import Path

from inference import rendering as R
from inference import run_inference as RI
from inference.symptom_decoder import decode_symptoms_versioned

ROOT = Path(__file__).resolve().parents[2]
TS = ROOT / "data" / "test_sets"
SETS = [TS / "eval-v02-adult.json", TS / "eval-v03-pool-atypical.json", TS / "eval-v03-pool-high-risk.json"]

CASE = {
    "case_id": "ddxplus_0", "age": 49, "sex": "female", "working_diagnosis_name": "GERD (K21)",
    "presenting_symptoms": ["E_53", "E_54_@_V_112", "E_54_@_V_161", "E_54_@_V_180", "E_54_@_V_181", "E_55_@_V_29",
                            "E_55_@_V_101", "E_55_@_V_103", "E_56_@_6", "E_57_@_V_29", "E_57_@_V_101", "E_58_@_3",
                            "E_59_@_2", "E_70", "E_78", "E_98", "E_140", "E_167", "E_173", "E_201", "E_204_@_V_10", "E_217"],
}


def prompt_items(text: str) -> tuple[list[str], list[str]]:
    """The comma-separated findings on the Chief Complaints and Medical History lines."""
    symptoms = re.search(r"^Chief Complaints: (.*)$", text, re.M).group(1)
    history = re.search(r"^Medical History / Context: (.*)$", text, re.M).group(1)
    return symptoms.split(", "), history.split(", ")


class TestShuffled(unittest.TestCase):
    def test_same_strings_different_order(self):
        active, history, _ = decode_symptoms_versioned(CASE["presenting_symptoms"], "v02", CASE["sex"])
        a, h = R.render_strings(active, history, CASE["case_id"], "shuffled")
        self.assertEqual(sorted(a), sorted(active))
        self.assertEqual(sorted(h), sorted(history))
        self.assertGreaterEqual(len(active), 10)
        self.assertNotEqual(a, active)  # 10 or more strings: the same order has probability under 1e-6

    def test_deterministic_per_case(self):
        strings = [f"s{i}" for i in range(12)]
        first = R.render_strings(strings, ["h1", "h2", "h3"], "ddxplus_123", "shuffled")
        self.assertEqual(first, R.render_strings(strings, ["h1", "h2", "h3"], "ddxplus_123", "shuffled"))
        self.assertNotEqual(first, R.render_strings(strings, ["h1", "h2", "h3"], "ddxplus_124", "shuffled"))
        # The order is fixed by the seed, not by the run: this is what a rerun must reproduce.
        self.assertEqual(first[0], ["s7", "s10", "s8", "s5", "s1", "s0", "s6", "s4", "s2", "s9", "s11", "s3"])

    def test_standard_is_identity(self):
        self.assertEqual(R.render_strings(["a", "b"], ["c"], "x", "standard"), (["a", "b"], ["c"]))


class TestParaphrased(unittest.TestCase):
    def test_table_is_a_bijection_of_40_changed_strings(self):
        self.assertEqual(len(R.PARAPHRASE), 40)
        for k, v in R.PARAPHRASE.items():
            self.assertNotEqual(k, v)
            self.assertNotIn(v, R.PARAPHRASE, f"{v!r} is both an alternative and an original")
        self.assertEqual(len(R.inverse_table()), 40)

    def test_inverse_restores_the_findings(self):
        active, history, _ = decode_symptoms_versioned(CASE["presenting_symptoms"], "v02", CASE["sex"])
        a, h = R.render_strings(active, history, CASE["case_id"], "paraphrased")
        inv = R.inverse_table()
        self.assertEqual([inv.get(s, s) for s in a], active)
        self.assertEqual([inv.get(s, s) for s in h], history)
        self.assertNotEqual(a, active)
        self.assertIn("Reports pain", a)
        self.assertEqual(h[-1], "Has not travelled abroad in the past 4 weeks")
        self.assertEqual(h[2], "A hiatal hernia")  # outside the table: unchanged

    def test_unknown_string_passes_through(self):
        self.assertEqual(R.paraphrase_strings(["Not in the table"]), ["Not in the table"])

    def test_duplicate_alternative_is_refused(self):
        with self.assertRaises(ValueError):
            R.inverse_table({"a": "x", "b": "x"})

    def test_unknown_rendering(self):
        with self.assertRaises(ValueError):
            R.render_strings(["a"], [], "x", "reversed")


class TestPromptContent(unittest.TestCase):
    """The rendered prompts carry the same findings under every rendering, for every prompt version."""

    def check(self, build):
        base = build(dict(CASE), "standard")
        s0, h0 = prompt_items(base)
        for rendering in ("shuffled", "paraphrased"):
            text = build(dict(CASE), rendering)
            s, h = prompt_items(text)
            if rendering == "paraphrased":
                inv = R.inverse_table()
                s, h = [inv.get(x, x) for x in s], [inv.get(x, x) for x in h]
            self.assertEqual(sorted(s), sorted(s0), rendering)
            self.assertEqual(sorted(h), sorted(h0), rendering)
            # Only the two finding lines change.
            strip = lambda t: re.sub(r"^(Chief Complaints|Medical History / Context): .*$", "", t, flags=re.M)
            self.assertEqual(strip(text), strip(base), rendering)

    def test_v5(self):
        self.check(lambda c, r: RI.format_case_for_prompt_v5(c, "v02", r))

    def test_v6(self):
        self.check(lambda c, r: RI.format_case_for_prompt_v6(c, "v02", r))

    def test_v7_arms(self):
        for arm in RI.V7_ARMS:
            self.check(lambda c, r, arm=arm: RI.format_case_for_prompt_v7(c, arm, "v02", r))

    def test_audit_records_the_rendering(self):
        case = dict(CASE)
        RI.format_case_for_prompt_v7(case, "v7a1", "v02", "paraphrased")
        self.assertEqual(case["_input_decode_audit"]["rendering"], "paraphrased")

    def test_settings_carry_the_rendering(self):
        cfg = RI.load_run_config(RI.DEFAULT_RUN_CONFIG_V03_AB)
        s = RI.resolve_run_settings(cfg, "anthropic/claude-haiku-4.5")
        self.assertEqual(s["rendering"], "standard")
        self.assertEqual(s["config_overridden"], [])
        s = RI.resolve_run_settings(cfg, "anthropic/claude-haiku-4.5", rendering="shuffled")
        self.assertEqual(s["rendering"], "shuffled")
        self.assertEqual(s["config_overridden"], ["rendering"])
        s["prompt_version"] = "v7a1"
        msgs = RI.build_messages_for(dict(CASE), s)
        self.assertEqual(sorted(prompt_items(msgs[1]["content"])[0]),
                         sorted(prompt_items(RI.format_case_for_prompt_v7(dict(CASE), "v7a1"))[0]))


@unittest.skipUnless(all(p.exists() for p in SETS), "needs the v0.3 sample and pools under data/test_sets")
class TestTableCoversTheSample(unittest.TestCase):
    def test_keys_are_the_40_most_frequent_strings(self):
        counts = collections.Counter()
        for path in SETS:
            for c in json.loads(path.read_text())["cases"]:
                a, h, _ = decode_symptoms_versioned(c["presenting_symptoms"], "v02", c["sex"])
                counts.update(a)
                counts.update(h)
        top = {s for s, _ in counts.most_common(40)}
        self.assertEqual(top, set(R.PARAPHRASE))


if __name__ == "__main__":
    unittest.main()
