"""Google Gemini adapter.

The default provider: the most usable free tier for repeated evaluation runs, with
native JSON-schema output (DECISIONS.md D8).

Two things measured in Phase 4 that shape this adapter:

**Thinking is off by default.** Gemini's flash models reason before answering, which for
grounded extraction bought nothing here and cost a great deal — a trivial abstention
took 39s with thinking and 1.18s without, both returning the same verdict. Across a
40-question evaluation run repeated per ablation, that is the difference between 26
minutes and under a minute. It is a config value, so the assumption is testable rather
than baked in.

**Transient 503s are normal, and the free tier is 5 requests per minute.** Both
``high demand`` 503s and 429 quota errors appeared within a handful of smoke calls, so
retry lives in ``base.with_retry`` and outbound calls are paced by ``base.RateLimiter``.
Without pacing, an evaluation run exhausts the per-minute allowance in seconds and then
spends itself in backoff.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

from mlsc_assistant.core.errors import ConfigurationError
from mlsc_assistant.generation.providers.base import (
    LLMResult,
    RateLimiter,
    StructuredResult,
    parse_structured,
    translate_error,
    with_retry,
)


class GeminiProvider:
    """Implements ``core.ports.StreamingLLMProvider``."""

    name = "gemini"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None,
        timeout_s: float = 30.0,
        max_retries: int = 3,
        thinking_budget: int | None = 0,
        requests_per_minute: float | None = None,
    ) -> None:
        if not api_key:
            raise ConfigurationError(
                "No Gemini API key. Set GOOGLE_API_KEY in your .env file "
                "(free key at https://aistudio.google.com/apikey), or switch provider "
                "with MLSC_LLM__PROVIDER."
            )
        self._model = model
        self._api_key = api_key
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.thinking_budget = thinking_budget
        self._limiter = RateLimiter(requests_per_minute)
        self._client: Any | None = None

    @property
    def model(self) -> str:
        return self._model

    # -- client --------------------------------------------------------------

    def _get_client(self) -> Any:
        """Construct lazily so importing the CLI does not build an HTTP client."""
        if self._client is not None:
            return self._client
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - declared dependency
            raise ConfigurationError(
                'google-genai is not installed. Run `pip install -e "."`.'
            ) from exc
        self._client = genai.Client(api_key=self._api_key)
        return self._client

    def _config(
        self, *, system: str, temperature: float, max_output_tokens: int | None, **extra: Any
    ) -> Any:
        from google.genai import types

        kwargs: dict[str, Any] = {
            "system_instruction": system,
            "temperature": temperature,
            "http_options": types.HttpOptions(timeout=int(self.timeout_s * 1000)),
            **extra,
        }
        if max_output_tokens:
            kwargs["max_output_tokens"] = max_output_tokens
        if self.thinking_budget is not None:
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=self.thinking_budget)
        return types.GenerateContentConfig(**kwargs)

    # -- generation ----------------------------------------------------------

    def complete(
        self,
        *,
        system: str,
        prompt: str,
        temperature: float = 0.0,
        max_output_tokens: int | None = None,
    ) -> LLMResult:
        self._limiter.wait()
        started = time.perf_counter()
        response = with_retry(
            lambda: self._get_client().models.generate_content(
                model=self._model,
                contents=prompt,
                config=self._config(
                    system=system, temperature=temperature, max_output_tokens=max_output_tokens
                ),
            ),
            max_retries=self.max_retries,
            provider=self.name,
        )
        tokens_in, tokens_out = _usage(response)
        return LLMResult(
            text=response.text or "",
            input_tokens=tokens_in,
            output_tokens=tokens_out,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    def complete_structured(
        self,
        *,
        system: str,
        prompt: str,
        schema: dict[str, Any],
        temperature: float = 0.0,
        max_output_tokens: int | None = None,
    ) -> StructuredResult:
        self._limiter.wait()
        started = time.perf_counter()
        response = with_retry(
            lambda: self._get_client().models.generate_content(
                model=self._model,
                contents=prompt,
                config=self._config(
                    system=system,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    response_mime_type="application/json",
                    response_schema=schema,
                ),
            ),
            max_retries=self.max_retries,
            provider=self.name,
        )
        raw = response.text or ""
        tokens_in, tokens_out = _usage(response)
        return StructuredResult(
            data=parse_structured(
                raw, provider=self.name, required=tuple(schema.get("required", ()))
            ),
            raw_text=raw,
            input_tokens=tokens_in,
            output_tokens=tokens_out,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    def stream(
        self,
        *,
        system: str,
        prompt: str,
        temperature: float = 0.0,
        max_output_tokens: int | None = None,
    ) -> Iterator[str]:
        self._limiter.wait()
        try:
            chunks = self._get_client().models.generate_content_stream(
                model=self._model,
                contents=prompt,
                config=self._config(
                    system=system, temperature=temperature, max_output_tokens=max_output_tokens
                ),
            )
            for chunk in chunks:
                if chunk.text:
                    yield chunk.text
        except Exception as exc:  # translated to a typed domain error
            # Streaming is not retried: tokens may already have reached the client, and
            # replaying the call would duplicate them.
            raise translate_error(exc, self.name) from exc


def _usage(response: Any) -> tuple[int | None, int | None]:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return None, None
    output = getattr(usage, "candidates_token_count", None) or 0
    # Thinking tokens are billed and are invisible in candidates_token_count. Folding
    # them in keeps reported cost honest when thinking is enabled.
    thoughts = getattr(usage, "thoughts_token_count", None) or 0
    return getattr(usage, "prompt_token_count", None), (output + thoughts) or None
