import unittest
from unittest import mock

import requests

from inference import openrouter


def response(status, body="", headers=None, json_data=None):
    r = mock.Mock()
    r.status_code = status
    r.text = body
    r.headers = headers or {}
    r.json.return_value = json_data
    if status >= 400:
        r.raise_for_status.side_effect = requests.exceptions.HTTPError(f"{status}", response=r)
    else:
        r.raise_for_status.return_value = None
    return r


OK = {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}], "id": "x", "model": "m"}
IN_FLIGHT = '{"error":{"code":402,"metadata":{"reason":"in_flight_budget_exhausted"}}}'


class TestInFlightBudget(unittest.TestCase):
    def call(self, responses):
        with mock.patch.object(openrouter.requests, "post", side_effect=responses), \
                mock.patch.object(openrouter.time, "sleep") as sleep, \
                mock.patch.object(openrouter, "OPENROUTER_API_KEY", "k"):  # read once at import, not from os.environ
            out = openrouter.call_openrouter_detailed(model="m", messages=[{"role": "user", "content": "x"}])
        return out, sleep

    def test_waits_and_succeeds(self):
        out, sleep = self.call([response(402, IN_FLIGHT, {"Retry-After": "5"}), response(200, json_data=OK)])
        self.assertIsNone(out["error"])
        self.assertEqual(out["content"], "{}")
        self.assertGreaterEqual(sleep.call_args[0][0], 5)

    def test_other_402_fails_fast(self):
        out, sleep = self.call([response(402, '{"error":{"message":"Insufficient credits"}}')])
        self.assertEqual(out["error"], "http_402")
        sleep.assert_not_called()

    def test_gives_up_after_the_wait_budget(self):
        n = openrouter.IN_FLIGHT_WAITS + 1
        out, sleep = self.call([response(402, IN_FLIGHT) for _ in range(n)])
        self.assertEqual(out["error"], "http_402")
        self.assertEqual(sleep.call_count, openrouter.IN_FLIGHT_WAITS)


if __name__ == "__main__":
    unittest.main()
