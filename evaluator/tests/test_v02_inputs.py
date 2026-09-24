"""Tests for the v0.2 model input: decoder, prompt v5 and run configuration."""

import json
import re
import unittest
from pathlib import Path
from unittest import mock

from inference import prompt as P
from inference import run_inference as RI
from inference.symptom_decoder import decode_symptoms_versioned, decode_symptoms_with_audit

ROOT = Path(__file__).resolve().parents[2]


def v02(codes, sex="male"):
    active, history, audit = decode_symptoms_versioned(codes, version="v02", sex=sex)
    return active, history, audit


class TestDecoderV0Kept(unittest.TestCase):
    """v0 output must not change, so the v0 test sets reproduce the published prompts."""

    def test_v0_strings(self):
        active, history, _ = decode_symptoms_with_audit(["E_53", "E_59_@_8", "E_204_@_V_10", "E_54_@_V_71"])
        self.assertIn("Sudden pain onset", active)
        self.assertIn("Pain character heartbreaking", active)
        self.assertEqual(history, ["Recent travel to N"])

    def test_versioned_v0_matches_legacy(self):
        codes = ["E_53", "E_55_@_V_123", "E_59_@_3", "E_204_@_V_10", "E_130_@_V_11", "E_91"]
        self.assertEqual(decode_symptoms_versioned(codes, "v0")[:2], decode_symptoms_with_audit(codes)[:2])

    def test_v4_prompt_unchanged(self):
        case = {"case_id": "x", "age": 40, "sex": "male", "presenting_symptoms": ["E_91", "E_204_@_V_10"],
                "symptom_duration": "unknown", "severity_flags": "unknown", "red_flag_indicators": []}
        text = RI.format_case_for_prompt(case, "intake")
        self.assertIn("Medical History / Context: Recent travel to N", text)
        self.assertIn("Noted Red Flags: none", text)
        self.assertIn('"uncertainty": "CONFIDENT | UNCERTAIN"', text)

    def test_unknown_version(self):
        with self.assertRaises(ValueError):
            decode_symptoms_versioned(["E_91"], "v9")


class TestDecoderV02(unittest.TestCase):
    def test_no_travel(self):
        _, history, _ = v02(["E_204_@_V_10"])
        self.assertEqual(history, ["No travel outside the country in the last 4 weeks"])

    def test_travel_destination(self):
        _, history, _ = v02(["E_204_@_V_1"])
        self.assertEqual(history, ["Travel outside the country in the last 4 weeks: West Africa"])
        _, history, _ = v02(["E_204_@_V_8"])
        self.assertEqual(history, ["Travel outside the country in the last 4 weeks: the Caribbean"])

    def test_onset_is_its_scale_value(self):
        for n in (0, 6, 7, 10):
            active, _, _ = v02(["E_53", f"E_59_@_{n}"])
            self.assertIn(f"How fast the pain appeared: {n}/10 (10 = fastest)", active)
            self.assertFalse(any("udden" in a or "radual" in a for a in active))

    def test_localisation_is_not_called_diffuse(self):
        active, _, _ = v02(["E_53", "E_58_@_9"])
        self.assertEqual(active[1], "How precisely the pain can be located: 9/10 (10 = most precise)")

    def test_mistranslations(self):
        active, _, _ = v02(["E_53", "E_54_@_V_71", "E_54_@_V_179", "E_54_@_V_184", "E_55_@_V_137", "E_202"])
        self.assertEqual(active[1:], ["Pain character: tearing", "Pain character: stabbing", "Pain character: throbbing",
                                      "Pain location: palate", "Barking cough"])

    def test_empty_answers_are_dropped(self):
        active, _, audit = v02(["E_53", "E_54_@_V_11", "E_55_@_V_89", "E_57_@_V_123",
                                "E_129", "E_130_@_V_11", "E_133_@_V_123", "E_151", "E_152_@_V_123"])
        self.assertEqual(active, ["Pain present", "Pain location: forehead", "Pain does not radiate",
                                  "Any lesions, redness or problems on your skin that you believe are related to the condition you are consulting for",
                                  "Swelling in one or more areas of your body"])
        self.assertEqual(len(audit["dropped_empty_codes"]), 4)

    def test_detail_without_opening_question_is_dropped(self):
        # DDXPlus fills pain and skin details with defaults when the patient has neither.
        codes = ["E_54_@_V_11", "E_55_@_V_123", "E_56_@_0", "E_57_@_V_123", "E_58_@_0", "E_59_@_0",
                 "E_130_@_V_11", "E_131_@_V_10", "E_132_@_0", "E_135_@_V_10", "E_136_@_0", "E_152_@_V_123", "E_155"]
        active, _, audit = v02(codes)
        self.assertEqual(active, ["Feel your heart is beating fast (racing), irregularly (missing a beat) or do you feel palpitations"])
        self.assertEqual(len(audit["dropped_orphan_detail_codes"]), 12)

    def test_skin_lesion_fields(self):
        active, _, _ = v02(["E_129", "E_130_@_V_107", "E_131_@_V_12", "E_132_@_3", "E_134_@_2", "E_135_@_V_10", "E_136_@_4"])
        self.assertEqual(active[1:], ["Skin lesion colour: yellow", "Skin lesions peel off: yes", "Skin lesions raised: 3/10",
                                      "Pain caused by the skin lesions: 2/10", "Skin lesion larger than 1 cm: no",
                                      "Itching of the skin lesions: 4/10"])

    def test_rash_pain_is_not_a_second_pain_intensity(self):
        active, _, _ = v02(["E_53", "E_56_@_7", "E_129", "E_134_@_2"])
        self.assertEqual(sum(a.startswith("Pain intensity") for a in active), 1)

    def test_sex_specific_sites(self):
        active, _, _ = v02(["E_53", "E_55_@_V_168"], sex="female")
        self.assertEqual(active[1], "Pain location: genital area (R)")
        active, _, _ = v02(["E_53", "E_55_@_V_168"], sex="male")
        self.assertEqual(active[1], "Pain location: testicle (R)")
        active, _, _ = v02(["E_129", "E_133_@_V_95"], sex="male")
        self.assertEqual(active[1], "Skin lesion location: genitals (R)")
        active, _, _ = v02(["E_129", "E_133_@_V_95"], sex="female")
        self.assertEqual(active[1], "Skin lesion location: labia majora (R)")

    @unittest.skipUnless((ROOT / "data/test_sets/eval-v02-adult.json").exists(), "run scripts/prep_v02_sample.py first")
    def test_every_test_split_code_decodes(self):
        # Every code in the v0.2 sample renders to text or is dropped on purpose.
        data = json.loads((ROOT / "data/test_sets/eval-v02-adult.json").read_text())
        for case in data["cases"]:
            active, history, audit = v02(case["presenting_symptoms"], case["sex"])
            self.assertEqual(audit["unknown_evidence_count"] + audit["unknown_value_count"], 0, case["case_id"])
            text = " ".join(active + history)
            for bad in ("Recent travel to N", "Sudden pain onset", "Gradual pain onset", " NA", "nowhere",
                        "palace", "heartbreaking", "whooping", "Feel pain somewhere"):
                self.assertNotIn(bad, text, case["case_id"])


class TestPromptV5(unittest.TestCase):
    """Prompt v5 follows spec/v0.2-scoring.md section 2."""

    def test_escalation_definitions(self):
        self.assertIn("ESCALATE_NOW: the patient needs same-day urgent clinical assessment.", P.SYSTEM_PROMPT_INTAKE_V5)
        self.assertIn("ROUTINE_CARE: the patient can wait for a routine appointment or self-care.", P.SYSTEM_PROMPT_INTAKE_V5)

    def test_harm_ratio_line(self):
        self.assertIn("Under-triage is judged about 7 times as harmful as over-triage.", P.SYSTEM_PROMPT_INTAKE_V5)

    def test_neutral_wording(self):
        text = P.SYSTEM_PROMPT_INTAKE_V5.lower()
        for banned in ("most dangerous condition", "cannot exclude", "cannot reasonably exclude", "most urgent plausible",
                       "netherlands", "urgency level", "urgency_level", "urgency_probabilities"):
            self.assertNotIn(banned, text)
        self.assertNotRegex(P.SYSTEM_PROMPT_INTAKE_V5, r"\bNTS\b")
        self.assertFalse(hasattr(P, "NTS_LEVEL_DEFINITIONS"))

    def test_absent_line(self):
        self.assertIn("Symptoms and history not listed were asked about and are absent.", P.SYSTEM_PROMPT_INTAKE_V5)

    def test_keeps_v4_framing(self):
        for line in ("You are assisting at a GP clinic while processing patient intake.",
                     "PRIMARY TASK - ROUTING / DIFFERENTIAL (ICD-10):",
                     "You cannot interactively ask follow-up questions; make the best decision from what is provided."):
            self.assertIn(line, P.SYSTEM_PROMPT_INTAKE_V5)

    def test_schema_fields(self):
        for key in ('"differential"', '"code"', '"p"', '"p_serious"', '"escalation_decision"'):
            self.assertIn(key, P.OUTPUT_SCHEMA_V5)
        for gone in ("uncertainty", "urgency"):
            self.assertNotIn(gone, P.OUTPUT_SCHEMA_V5)

    def test_v4_constants_intact(self):
        self.assertNotIn("Netherlands", P.SYSTEM_PROMPT_INTAKE_V3)
        self.assertIn('"differential_diagnoses"', P.OUTPUT_SCHEMA_V4)

    def test_rendered_case(self):
        case = {"case_id": "x", "age": 52, "sex": "female", "presenting_symptoms": ["E_53", "E_59_@_9", "E_204_@_V_10"]}
        msgs = RI.build_messages_v5(case)
        self.assertEqual(msgs[0]["content"], P.SYSTEM_PROMPT_INTAKE_V5)
        user = msgs[1]["content"]
        self.assertIn("Age: 52", user)
        self.assertIn("How fast the pain appeared: 9/10", user)
        self.assertIn("No travel outside the country in the last 4 weeks", user)
        for gone in ("Symptom Duration", "Patient-Reported Severity", "Noted Red Flags"):
            self.assertNotIn(gone, user)


class TestRunConfig(unittest.TestCase):
    def setUp(self):
        self.config = RI.load_run_config(RI.DEFAULT_RUN_CONFIG_V02)

    def test_pinned_values(self):
        self.assertEqual(self.config["max_tokens"], 16000)
        self.assertEqual(self.config["empty_or_truncated_retries"], 1)
        self.assertEqual(self.config["prompt_version"], "v5")
        for model, entry in self.config["models"].items():
            self.assertIn("reasoning_effort", entry, model)

    def test_resolve_and_flag_overrides(self):
        s = RI.resolve_run_settings(self.config, "anthropic/claude-opus-5")
        self.assertEqual((s["max_tokens"], s["reasoning_effort"], s["config_overridden"]), (16000, "medium", []))
        s = RI.resolve_run_settings(self.config, "anthropic/claude-opus-5", max_tokens=16000, reasoning_effort="medium")
        self.assertEqual(s["config_overridden"], [])
        s = RI.resolve_run_settings(self.config, "anthropic/claude-opus-5", max_tokens=2000)
        self.assertEqual(s["config_overridden"], ["max_tokens"])
        s = RI.resolve_run_settings(self.config, "someone/new-model")
        self.assertIn("model_not_in_config", s["config_overridden"])

    def _meta(self, content, finish="stop", error=None):
        return {"content": content, "finish_reason": finish, "native_finish_reason": finish, "provider": "Prov",
                "request_id": "gen-1", "model_served": "m", "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                "error": error}

    def _run(self, responses):
        settings = RI.resolve_run_settings(self.config, "anthropic/claude-opus-5")
        case = {"case_id": "c1", "age": 40, "sex": "male", "presenting_symptoms": ["E_91"]}
        with mock.patch.object(RI, "call_openrouter_detailed", side_effect=responses) as m, \
                mock.patch("time.sleep"):
            pred = RI.run_inference_on_case_v5(case, "anthropic/claude-opus-5", settings)
        return pred, m

    GOOD = '{"differential": [{"code": "J11.1", "p": 60}], "p_serious": 5, "escalation_decision": "ROUTINE_CARE"}'

    def test_records_metadata(self):
        pred, m = self._run([self._meta(self.GOOD)])
        self.assertEqual(m.call_count, 1)
        kwargs = m.call_args.kwargs
        self.assertEqual((kwargs["max_tokens"], kwargs["reasoning_effort"], kwargs["empty_content_retries"]), (16000, "medium", 0))
        self.assertEqual(pred["p_serious"], 5)
        for key, want in (("finish_reason", "stop"), ("provider", "Prov"), ("request_id", "gen-1")):
            self.assertEqual(pred[key], want)
        self.assertEqual(pred["usage"]["completion_tokens"], 5)
        self.assertEqual(pred["prompt_version"], "v5")
        self.assertEqual(len(pred["attempts"]), 1)

    def test_one_retry_on_empty(self):
        pred, m = self._run([self._meta(None, error="empty_content"), self._meta(self.GOOD)])
        self.assertEqual(m.call_count, 2)
        self.assertEqual([a["problem"] for a in pred["attempts"]], ["empty", None])
        self.assertNotIn("error", pred)

    def test_one_retry_on_truncation_only(self):
        truncated = self._meta('{"differential": [', finish="length")
        pred, m = self._run([truncated, truncated, self._meta(self.GOOD)])
        self.assertEqual(m.call_count, 2)  # one retry, not two
        self.assertEqual(pred["error"], "json_parse_failure")
        self.assertTrue(pred["truncated"])

    def test_no_retry_on_http_error(self):
        pred, m = self._run([self._meta(None, finish=None, error="http_400"), self._meta(self.GOOD)])
        self.assertEqual(m.call_count, 1)
        self.assertEqual(pred["error"], "http_400")


class TestOutputLockKept(unittest.TestCase):
    def test_second_writer_exits(self):
        import contextlib
        import io
        import tempfile
        import warnings
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "preds.json"
            fd = RI.acquire_output_lock(out)
            try:
                with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()), \
                        warnings.catch_warnings():
                    warnings.simplefilter("ignore", ResourceWarning)
                    RI.acquire_output_lock(out)
            finally:
                fd.close()


if __name__ == "__main__":
    unittest.main()
