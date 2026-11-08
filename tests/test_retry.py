import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tradebot.retry import RetryPolicy, backoff_delay, call_with_retry, is_rate_limit

FAST = RetryPolicy(attempts=4, base_delay=1, rate_limit_delay=5, max_delay=60)


class HTTPErr(Exception):
    def __init__(self, status, headers=None):
        super().__init__(f"HTTP {status}")
        self.response = SimpleNamespace(status_code=status, headers=headers or {})


class YFRateLimitError(Exception):
    pass


def flaky(errors, result="ok"):
    calls = []

    def fn():
        calls.append(1)
        if len(calls) <= len(errors):
            raise errors[len(calls) - 1]
        return result
    return fn, calls


def test_retries_rate_limit_then_succeeds():
    fn, calls = flaky([HTTPErr(429), YFRateLimitError("Too Many Requests")])
    sleeps = []
    assert call_with_retry(fn, policy=FAST, sleep=sleeps.append) == "ok"
    assert len(calls) == 3 and len(sleeps) == 2
    assert all(s >= 2.5 for s in sleeps)  # rate-limit backoff, not the 1s transient base


def test_does_not_retry_client_errors():
    fn, calls = flaky([HTTPErr(403)])
    with pytest.raises(HTTPErr):
        call_with_retry(fn, policy=FAST, sleep=lambda s: None)
    assert len(calls) == 1


def test_gives_up_after_max_attempts():
    fn, calls = flaky([HTTPErr(503)] * 10)
    with pytest.raises(HTTPErr):
        call_with_retry(fn, policy=FAST, sleep=lambda s: None)
    assert len(calls) == FAST.attempts


def test_honours_retry_after_and_caps_backoff():
    assert backoff_delay(1, HTTPErr(429, {"Retry-After": "12"}), FAST) == 12
    assert backoff_delay(10, HTTPErr(503), FAST) <= FAST.max_delay
    assert is_rate_limit(YFRateLimitError("x"))


def test_submit_does_not_double_order_after_timeout():
    from tradebot.broker import AlpacaBroker

    accepted = SimpleNamespace(id="abc")
    submits = []

    class Client:
        def submit_order(self, req):
            submits.append(req.client_order_id)
            raise TimeoutError("read timed out")  # but the server got it

        def get_order_by_client_id(self, coid):
            return accepted

    b = AlpacaBroker.__new__(AlpacaBroker)
    b.client, b.paper = Client(), True
    assert b.submit_notional("AAPL", "buy", 100) is accepted
    assert len(submits) == 1
