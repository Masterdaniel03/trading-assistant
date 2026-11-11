"""Gemini client behaviour with a fake API: pacing, fallback, quota handling. No network."""
import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from tradebot.llm import GeminiLLM, LLMError, is_daily_quota, short_error


class Answer(BaseModel):
    ok: bool


class ApiError(Exception):
    """Shaped like google-genai's errors: the HTTP status is in .code."""
    def __init__(self, code, message=""):
        super().__init__(f"{code} {message}")
        self.code, self.response = code, None


OVERLOADED = ApiError(503, "UNAVAILABLE high demand")
PER_MINUTE = ApiError(429, "RESOURCE_EXHAUSTED quotaId GenerateRequestsPerMinutePerProjectPerModel-FreeTier")
PER_DAY = ApiError(429, "RESOURCE_EXHAUSTED quotaId GenerateRequestsPerDayPerProjectPerModel-FreeTier")


class FakeClient:
    """`script[model]` is a list of outcomes, used in order: an exception to raise, or 'ok'."""
    def __init__(self, script):
        self.script, self.calls = {m: list(v) for m, v in script.items()}, []
        self.models = SimpleNamespace(generate_content=self._generate)

    def _generate(self, model, contents, config):
        self.calls.append(model)
        outcome = self.script[model].pop(0) if self.script[model] else "ok"
        if isinstance(outcome, Exception):
            raise outcome
        usage = SimpleNamespace(prompt_token_count=10, candidates_token_count=5)
        return SimpleNamespace(text=json.dumps({"ok": True}), usage_metadata=usage)


class Clock:
    def __init__(self):
        self.now, self.sleeps = 1000.0, []

    def __call__(self):
        return self.now

    def sleep(self, s):
        self.sleeps.append(s)
        self.now += s


def make(script, models=("a", "b"), rpm=5):
    clock = Clock()
    client = FakeClient(script)
    llm = GeminiLLM(None, list(models), requests_per_minute=rpm, client=client,
                    sleep=clock.sleep, clock=clock)
    return llm, client, clock


def test_answer_reports_which_model_answered():
    llm, client, _ = make({"a": [], "b": []})
    reply = llm.generate_json("p", Answer)
    assert (reply.model, reply.input_tokens, reply.output_tokens) == ("a", 10, 5)
    assert client.calls == ["a"] and llm.model == "a"


def test_overloaded_model_is_retried_once_then_the_next_model_answers():
    llm, client, _ = make({"a": [OVERLOADED, OVERLOADED], "b": []})
    assert llm.generate_json("p", Answer).model == "b"
    assert client.calls == ["a", "a", "b"]                 # 2 tries on a, not 5


def test_used_up_daily_quota_is_not_retried_and_the_model_is_skipped_afterwards():
    llm, client, _ = make({"a": [PER_DAY], "b": []})
    assert llm.generate_json("p", Answer).model == "b"
    assert llm.generate_json("p", Answer).model == "b"
    assert client.calls == ["a", "b", "b"]                 # a asked once, then left alone


def test_per_minute_limit_is_retried_on_the_same_model():
    llm, client, _ = make({"a": [PER_MINUTE], "b": []})
    assert llm.generate_json("p", Answer).model == "a"
    assert client.calls == ["a", "a"]


def test_all_models_failing_raises_a_readable_error():
    llm, client, _ = make({"a": [PER_DAY], "b": [OVERLOADED, OVERLOADED]})
    with pytest.raises(LLMError) as err:
        llm.generate_json("p", Answer)
    assert str(err.value) == ("no model could answer: a: daily free quota used up "
                              "(it resets at midnight Pacific time); b: model overloaded (503)")
    assert len(client.calls) == 3


def test_bad_request_is_not_passed_to_other_models():
    llm, client, _ = make({"a": [ApiError(400, "INVALID_ARGUMENT")], "b": []})
    with pytest.raises(ApiError):
        llm.generate_json("p", Answer)
    assert client.calls == ["a"]


def test_calls_to_one_model_are_spaced_out():
    llm, client, clock = make({"a": []}, models=("a",), rpm=5)
    for _ in range(3):
        llm.generate_json("p", Answer)
    assert clock.sleeps == [12.0, 12.0]                     # 60 s / 5 requests


def test_error_classification():
    assert is_daily_quota(PER_DAY) and not is_daily_quota(PER_MINUTE)
    assert not is_daily_quota(OVERLOADED)
    assert short_error(OVERLOADED) == "model overloaded (503)"
    assert short_error(PER_MINUTE) == "per-minute rate limit hit"
    assert short_error(ApiError(504)) == "model timed out (504)"


def test_missing_key_is_a_clear_error():
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        GeminiLLM(None, ["a"])
