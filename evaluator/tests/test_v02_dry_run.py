"""The v0.2 dry run renders request bodies and never touches the network."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from evaluator import v02_score as vs


@unittest.skipUnless(vs.SAMPLE.exists(), "needs data/test_sets/eval-v02-adult.json")
class TestDryRun(unittest.TestCase):
    def test_renders_without_sending(self):
        from inference import openrouter, run_inference as ri

        cases = vs.load_sample()[:3]
        settings = ri.resolve_run_settings(ri.load_run_config(ri.DEFAULT_RUN_CONFIG_V02), "anthropic/claude-opus-5")
        with tempfile.TemporaryDirectory() as d, mock.patch.object(openrouter.requests, "post",
                                                                    side_effect=AssertionError("network call")):
            path = ri.write_dry_run(Path(d) / "x.json", cases, "anthropic/claude-opus-5", settings, {}, "cfg")
            self.assertEqual(path.name, "x.dryrun.json")
            self.assertFalse((Path(d) / "x.json").exists())  # never the predictions path
            out = json.loads(path.read_text())
        req = out["requests"][0]["payload"]
        self.assertEqual(req, openrouter.build_payload("anthropic/claude-opus-5", req["messages"], 0.0, 16000, "medium"))
        self.assertEqual(req["reasoning"], {"effort": "medium"})
        self.assertIn("Under-triage is judged about 7 times as harmful", req["messages"][0]["content"] + req["messages"][1]["content"])
        self.assertEqual(out["metadata"]["cases"], 3)
        self.assertEqual(out["metadata"]["config_overridden"], [])


if __name__ == "__main__":
    unittest.main()
