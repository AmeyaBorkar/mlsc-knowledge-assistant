"""Shared provider machinery: result types, error mapping and retry.

Adapters implement transport; everything here is common. Keeping the retry policy and
the error taxonomy in one place means "Gemini is throttling us" and "no key configured"
surface identically no matter which backend is selected.
"""

from __future__ import annotations

import json
import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from mlsc_assistant.core.errors import (
    ProviderRateLimitedError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    StructuredOutputError,
)


@dataclass(frozen=True, slots=True)
class LLMResult:
    """Implements ``core.ports.LLMResponse``."""

    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class StructuredResult:
    """Implements ``core.ports.StructuredResponse``."""

    data: dict[str, Any]
    raw_text: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Retry
# ---------------------------------------------------------------------------

# Measured during Phase 4: gemini-3.7-flash returned 503 UNAVAILABLE ("high demand")
# on a trivial request and took 151s on another. Transient upstream congestion is the
# normal case, not the exception, so retrying is part of the design rather than defensive
# padding — a 40-question evaluation run would otherwise fail partway through routinely.
_RETRYABLE_MARKERS = ("503", "429", "unavailable", "resource_exhausted", "overloaded", "timeout")


def is_retryable(exc: BaseException) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in _RETRYABLE_MARKERS)


def with_retry[T](
    call: Callable[[], T],
    *,
    max_retries: int,
    provider: str,
    base_delay: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Retry transient upstream failures, preferring the server's own advice.

    When a provider says "retry in 6.9s" it knows something we do not — the exact
    remaining window on a quota. Guessing with exponential backoff instead is how a
    40-question evaluation run dies partway through against a 5-requests-per-minute free
    tier: three blind retries total roughly 7 seconds against a 60-second window.

    Falls back to exponential backoff with jitter when no delay is advertised. Jitter
    matters because the harness issues many similar requests, and identical backoff
    across them would resynchronise every retry into the spike that caused the failure.
    """
    last: BaseException | None = None
    for attempt in range(max_retries + 1):
        try:
            return call()
        except Exception as exc:  # re-raised below as a typed domain error
            last = exc
            if not is_retryable(exc) or attempt == max_retries:
                raise translate_error(exc, provider) from exc
            suggested = retry_after_seconds(str(exc))
            backoff = base_delay * (2**attempt) * (0.5 + random.random())
            # A small margin over the server's figure: retrying at the exact boundary
            # tends to land just inside the window and burn an attempt.
            sleep(max(suggested + 0.5, backoff) if suggested is not None else backoff)

    raise translate_error(last or RuntimeError("unknown provider failure"), provider)


def translate_error(exc: BaseException, provider: str) -> Exception:
    """Map an SDK exception onto the typed errors the HTTP edge understands.

    Three distinct statuses rather than one 500, because "you have not set a key",
    "you are being throttled" and "the upstream is down" need three different reactions
    from a caller (DECISIONS.md, errors table in docs/API.md).
    """
    text = str(exc)
    lowered = text.lower()

    if "429" in text or "resource_exhausted" in lowered or "rate limit" in lowered:
        return ProviderRateLimitedError(
            f"{provider} rate-limited the request. Wait and retry, or lower the "
            "evaluation concurrency.",
            retry_after=_retry_after(text),
        )
    if "timeout" in lowered or "deadline" in lowered:
        return ProviderTimeoutError(
            f"{provider} did not respond within the configured timeout. "
            "Raise `llm.timeout_s` or try a smaller model."
        )
    if "api key" in lowered or "401" in text or "403" in text or "permission" in lowered:
        return ProviderUnavailableError(
            f"{provider} rejected the credentials. Check the API key in your .env file."
        )
    if "404" in text and "model" in lowered:
        return ProviderUnavailableError(
            f"{provider} does not offer the configured model: {text.strip()[:200]} "
            "Update `llm.models` in config.yaml."
        )
    return ProviderUnavailableError(f"{provider} request failed: {text.strip()[:300]}")


# Providers advertise the wait in more than one shape. Gemini returns both
# `'retryDelay': '6s'` in the error details and "Please retry in 6.882579384s" in the
# message; the structured field is preferred because it is not prose.
_RETRY_PATTERNS = (
    re.compile(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s?", re.IGNORECASE),
    re.compile(r"retry[- ]after['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"retry\s+in\s+(\d+(?:\.\d+)?)\s*s", re.IGNORECASE),
)

_MAX_ADVERTISED_WAIT = 120.0


def retry_after_seconds(text: str) -> float | None:
    """Extract a provider-advertised retry delay, if it gave one."""
    for pattern in _RETRY_PATTERNS:
        match = pattern.search(text)
        if match:
            # Cap it: an absurd advertised wait should surface as a failure rather than
            # silently hanging an evaluation run for an hour.
            return min(float(match.group(1)), _MAX_ADVERTISED_WAIT)
    return None


def _retry_after(text: str) -> int | None:
    seconds = retry_after_seconds(text)
    return None if seconds is None else int(seconds)


# ---------------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------------

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_structured(raw: str, *, provider: str, required: tuple[str, ...] = ()) -> dict[str, Any]:
    """Parse a structured response, repairing the common wrappers.

    Providers with native schema support return clean JSON and this is a plain
    ``json.loads``. The salvage path exists for backends that only emulate structured
    output — Ollama being the motivating case — where the model wraps JSON in prose or a
    fenced code block. A malformed response raises rather than returning a partial dict,
    because the answerer treats missing fields as a reason to abstain and silently
    defaulting them would turn a provider failure into a confident wrong answer.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_BLOCK.search(text)
        if not match:
            raise StructuredOutputError(
                f"{provider} returned no parseable JSON object. First 200 characters: {raw[:200]!r}"
            ) from None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise StructuredOutputError(
                f"{provider} returned malformed JSON: {exc}. First 200 characters: {raw[:200]!r}"
            ) from exc

    if not isinstance(data, dict):
        raise StructuredOutputError(
            f"{provider} returned a {type(data).__name__}, expected a JSON object."
        )

    missing = [key for key in required if key not in data]
    if missing:
        raise StructuredOutputError(
            f"{provider} response is missing required field(s): {', '.join(missing)}."
        )
    return data


class RateLimiter:
    """Minimum-interval pacing for a provider client.

    Measured: the Gemini free tier allows **5 requests per minute** for the configured
    model. Firing a 40-question evaluation run at full speed exhausts that in seconds and
    then spends the run in retry, which is slower than simply pacing and produces
    misleading latency numbers along the way.

    Deliberately a simple minimum interval rather than a token bucket. A bucket would let
    the harness burst through the allowance and then stall, which is exactly the
    behaviour that made the run fail.
    """

    def __init__(
        self,
        requests_per_minute: float | None,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.min_interval = 60.0 / requests_per_minute if requests_per_minute else 0.0
        self._monotonic = monotonic
        self._sleep = sleep
        self._last: float | None = None

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        now = self._monotonic()
        if self._last is not None:
            remaining = self.min_interval - (now - self._last)
            if remaining > 0:
                self._sleep(remaining)
                now = self._monotonic()
        self._last = now
