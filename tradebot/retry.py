"""Retry with exponential backoff + jitter, tuned for free-tier data APIs.

Two classes of failure are retried:
  * rate limits (HTTP 429, yfinance YFRateLimitError) -> longer backoff,
    honouring a Retry-After header when the server sends one
  * transient errors (5xx, timeouts, dropped connections) -> normal backoff

Everything else (bad symbol, auth failure, 4xx validation) is raised
immediately, because retrying it would only burn quota.
"""
from __future__ import annotations

import functools
import logging
import random
import time
from dataclasses import dataclass
from typing import Callable, TypeVar

log = logging.getLogger(__name__)
T = TypeVar("T")

TRANSIENT_STATUS = {500, 502, 503, 504}


@dataclass(frozen=True)
class RetryPolicy:
    attempts: int = 5             # total tries, including the first
    base_delay: float = 1.0       # seconds, transient errors
    rate_limit_delay: float = 5.0  # seconds, first wait after a 429
    max_delay: float = 60.0


DEFAULT_POLICY = RetryPolicy()


def _status_code(exc: BaseException) -> int | None:
    for obj in (exc, getattr(exc, "response", None)):
        code = getattr(obj, "status_code", None)
        if isinstance(code, int):
            return code
    return None


def _retry_after(exc: BaseException) -> float | None:
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None) or {}
    try:
        value = headers.get("Retry-After")
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def is_rate_limit(exc: BaseException) -> bool:
    if type(exc).__name__ == "YFRateLimitError":
        return True
    if _status_code(exc) == 429:
        return True
    return "too many requests" in str(exc).lower() or "rate limit" in str(exc).lower()


def is_transient(exc: BaseException) -> bool:
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    if _status_code(exc) in TRANSIENT_STATUS:
        return True
    # requests / curl_cffi (used by yfinance) don't subclass the builtins
    name = type(exc).__name__
    return name in {"ConnectionError", "Timeout", "ReadTimeout", "ConnectTimeout",
                    "ChunkedEncodingError", "RequestsError", "CurlError"}


def is_retryable(exc: BaseException) -> bool:
    return is_rate_limit(exc) or is_transient(exc)


def backoff_delay(attempt: int, exc: BaseException, policy: RetryPolicy = DEFAULT_POLICY) -> float:
    """Delay before retry number `attempt` (1-based). Full jitter, capped."""
    if is_rate_limit(exc):
        hinted = _retry_after(exc)
        if hinted is not None:
            return min(hinted, policy.max_delay)
        base = policy.rate_limit_delay
    else:
        base = policy.base_delay
    ceiling = min(policy.max_delay, base * 2 ** (attempt - 1))
    return random.uniform(ceiling / 2, ceiling)


def call_with_retry(fn: Callable[..., T], *args,
                    policy: RetryPolicy = DEFAULT_POLICY,
                    retry_if: Callable[[BaseException], bool] = is_retryable,
                    sleep: Callable[[float], None] = time.sleep,
                    what: str | None = None,
                    **kwargs) -> T:
    label = what or getattr(fn, "__name__", "call")
    for attempt in range(1, policy.attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - classified below
            if attempt == policy.attempts or not retry_if(exc):
                raise
            delay = backoff_delay(attempt, exc, policy)
            kind = "rate-limited" if is_rate_limit(exc) else "transient error"
            log.warning("%s %s (%s: %s); retry %d/%d in %.1fs", label, kind,
                        type(exc).__name__, exc, attempt, policy.attempts - 1, delay)
            sleep(delay)
    raise AssertionError("unreachable")


def with_retry(policy: RetryPolicy = DEFAULT_POLICY,
               retry_if: Callable[[BaseException], bool] = is_retryable):
    """Decorator form of call_with_retry."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return call_with_retry(fn, *args, policy=policy, retry_if=retry_if,
                                   what=fn.__qualname__, **kwargs)
        return wrapper
    return deco
