"""The LLM behind the research layer. Gemini for now (its free tier is enough).

Research code only needs an object with a `model` name and a
`generate_json(prompt, schema)` method, so another provider can be dropped in
later without touching anything else.

Free-tier limits are per model (at the time of writing: 5 requests a minute,
20 a day), and failed attempts count too. So this client
  * spaces requests out to stay under the per-minute limit,
  * retries an overloaded model once, not four times,
  * never retries a used-up daily quota (it would only wait hours), and
  * falls back to the next model in the list, since each has its own quota.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from tradebot.retry import RetryPolicy, call_with_retry, is_rate_limit, is_retryable

DEFAULT_MODELS = ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite"]
DEFAULT_MODEL = DEFAULT_MODELS[0]
TIMEOUT_SECONDS = 60
GEMINI_POLICY = RetryPolicy(attempts=2, base_delay=5.0, rate_limit_delay=15.0, max_delay=60.0)


class LLMError(Exception):
    pass


@dataclass(frozen=True)
class LLMReply:
    text: str            # the JSON the model wrote
    input_tokens: int
    output_tokens: int
    model: str = ""      # which model answered


def is_daily_quota(exc: BaseException) -> bool:
    """A used-up daily quota. Retrying can't help until it resets."""
    text = str(exc)
    return is_rate_limit(exc) and ("PerDay" in text or "per day" in text.lower())


def short_error(exc: BaseException) -> str:
    """A one-line, readable reason for a failed call."""
    code = getattr(exc, "code", None)
    if is_daily_quota(exc):
        return "daily free quota used up (it resets at midnight Pacific time)"
    if is_rate_limit(exc):
        return "per-minute rate limit hit"
    if code == 503:
        return "model overloaded (503)"
    if code == 504:
        return "model timed out (504)"
    return f"{type(exc).__name__}: {str(exc)[:150]}"


class GeminiLLM:
    def __init__(self, api_key: str | None, models: list[str] | str = DEFAULT_MODELS,
                 requests_per_minute: float = 5, timeout_s: int = TIMEOUT_SECONDS,
                 client=None, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic):
        self.models = [models] if isinstance(models, str) else list(models)
        if not self.models:
            raise ValueError("At least one model is needed")
        self.model = self.models[0]           # names the setup, e.g. for reusing stored answers
        self._min_gap = 60.0 / requests_per_minute if requests_per_minute else 0.0
        self._sleep, self._clock = sleep, clock
        self._last_call: dict[str, float] = {}
        self._out_of_quota: set[str] = set()

        from google.genai import types
        self._types = types
        if client is None:
            if not api_key:
                raise RuntimeError("Missing GEMINI_API_KEY in .env (the research command needs it)")
            from google import genai
            client = genai.Client(api_key=api_key,
                                  http_options=types.HttpOptions(timeout=timeout_s * 1000))
        self._client = client

    def _paced_call(self, model: str, **kwargs):
        last = self._last_call.get(model)
        if last is not None:
            wait = self._min_gap - (self._clock() - last)
            if wait > 0:
                self._sleep(wait)
        self._last_call[model] = self._clock()
        return self._client.models.generate_content(model=model, **kwargs)

    def generate_json(self, prompt: str, schema) -> LLMReply:
        """Ask for an answer shaped like `schema` (a pydantic model class),
        trying each model in turn until one answers."""
        config = self._types.GenerateContentConfig(
            response_mime_type="application/json", response_schema=schema)
        failures = []
        for model in self.models:
            if model in self._out_of_quota:
                failures.append(f"{model}: daily free quota used up")
                continue
            try:
                response = call_with_retry(
                    self._paced_call, model, contents=prompt, config=config,
                    policy=GEMINI_POLICY, what=f"gemini {model}", sleep=self._sleep,
                    retry_if=lambda e: is_retryable(e) and not is_daily_quota(e))
            except Exception as exc:  # noqa: BLE001 - classified below
                if not is_retryable(exc):
                    raise                     # a bad request fails the same way on every model
                if is_daily_quota(exc):
                    self._out_of_quota.add(model)
                failures.append(f"{model}: {short_error(exc)}")
                continue
            text = (response.text or "").strip()
            if not text:
                raise LLMError(f"{model} returned an empty answer")
            usage = response.usage_metadata
            return LLMReply(text, (usage.prompt_token_count or 0) if usage else 0,
                            (usage.candidates_token_count or 0) if usage else 0, model)
        raise LLMError("no model could answer: " + "; ".join(failures))
