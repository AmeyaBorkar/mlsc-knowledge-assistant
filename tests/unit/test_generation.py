"""Generation tests: gates, citation binding, retry, pacing and parsing.

All offline. A stub provider stands in for the LLM so the gate logic is tested against
responses chosen to be awkward, including the ones a real model produces rarely enough
that waiting to observe them would be unreliable.
"""

from __future__ import annotations

from typing import Any

import pytest

from mlsc_assistant.config import Settings
from mlsc_assistant.core.errors import (
    ProviderRateLimitedError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    StructuredOutputError,
)
from mlsc_assistant.core.models import (
    AbstentionReason,
    Chunk,
    ChunkKind,
    RetrievalResult,
    RetrievalStrategy,
    ScoredChunk,
)
from mlsc_assistant.generation.answerer import GroundedAnswerer
from mlsc_assistant.generation.providers.base import (
    RateLimiter,
    StructuredResult,
    is_retryable,
    parse_structured,
    retry_after_seconds,
    translate_error,
    with_retry,
)
from mlsc_assistant.generation.verifier import verify_answer

# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------


def _chunk(chunk_id: str, doc_id: str = "leadership", text: str = "Some passage text.") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        doc_title="MLSC Leadership Structure",
        source_file=f"{doc_id}.txt",
        text=text,
        embed_text=f"MLSC Leadership Structure - {text}",
        char_range=(0, len(text)),
        index=0,
        kind=ChunkKind.PARAGRAPH,
        token_estimate=8,
        checksum="x",
    )


class StubRetriever:
    """Stands in for HybridRetriever with a fixed result."""

    def __init__(
        self, chunks: tuple[ScoredChunk, ...], *, dense_score: float | None = 0.85
    ) -> None:
        self._chunks = chunks
        self._dense = dense_score
        self.corpus_size = max(len(chunks), 1)

    def retrieve(self, query: str, **_: Any) -> RetrievalResult:
        scored = tuple(
            ScoredChunk(
                chunk=sc.chunk,
                score=sc.score,
                rank=i,
                dense_score=self._dense,
                dense_rank=i,
            )
            for i, sc in enumerate(self._chunks, start=1)
        )
        return RetrievalResult(
            query=query,
            strategy=RetrievalStrategy.HYBRID,
            chunks=scored,
            candidates_considered=len(scored),
            timings_ms={"dense": 1.0, "lexical": 1.0, "fusion": 0.1},
        )


class StubProvider:
    """Returns queued responses. Records every call for assertions."""

    name = "stub"
    model = "stub-1"

    def __init__(self, *responses: dict[str, Any] | Exception) -> None:
        self._queue = list(responses)
        self.calls: list[dict[str, Any]] = []

    def _next(self) -> dict[str, Any]:
        item = self._queue.pop(0) if self._queue else {}
        if isinstance(item, Exception):
            raise item
        return item

    def complete(self, **kwargs: Any) -> Any:  # pragma: no cover - unused here
        raise NotImplementedError

    def complete_structured(self, **kwargs: Any) -> StructuredResult:
        self.calls.append(kwargs)
        return StructuredResult(data=self._next(), raw_text="", latency_ms=1.0)


def _answerer(
    provider: StubProvider,
    chunks: tuple[ScoredChunk, ...],
    *,
    dense_score: float | None = 0.85,
    **overrides: Any,
) -> GroundedAnswerer:
    settings = Settings(**overrides)
    return GroundedAnswerer(
        retriever=StubRetriever(chunks, dense_score=dense_score),  # type: ignore[arg-type]
        provider=provider,  # type: ignore[arg-type]
        settings=settings,
    )


def _scored(*chunk_ids: str) -> tuple[ScoredChunk, ...]:
    return tuple(
        ScoredChunk(chunk=_chunk(cid), score=0.9 - 0.1 * i, rank=i + 1)
        for i, cid in enumerate(chunk_ids)
    )


GOOD = {
    "sufficient_context": True,
    "answer": "Domain leads plan the roadmap.",
    "cited_chunk_ids": ["leadership::c01"],
    "confidence": "high",
}


# ---------------------------------------------------------------------------
# Gate 1
# ---------------------------------------------------------------------------


def test_gate_one_abstains_without_calling_the_model() -> None:
    """The whole point of a pre-LLM gate: an off-domain question costs nothing.

    It also removes any opportunity to hallucinate, since no generation happens.
    """
    provider = StubProvider(GOOD)
    answer = _answerer(provider, _scored("leadership::c01"), dense_score=0.30).answer("q")

    assert not answer.answered
    assert answer.abstention_reason is AbstentionReason.NO_RELEVANT_CONTEXT
    assert provider.calls == [], "no model call should have been made"
    assert "generation" not in answer.diagnostics


def test_gate_one_passes_above_the_calibrated_threshold() -> None:
    provider = StubProvider(GOOD)
    answer = _answerer(provider, _scored("leadership::c01"), dense_score=0.85).answer("q")
    assert answer.answered
    assert answer.diagnostics["gates"]["retrieval_gate"] == "pass"


def test_gate_one_abstains_when_retrieval_is_empty() -> None:
    provider = StubProvider(GOOD)
    answer = _answerer(provider, ()).answer("q")
    assert not answer.answered
    assert provider.calls == []


def test_gate_one_defers_when_there_is_no_dense_score() -> None:
    """Lexical-only retrieval produces no cosine to threshold.

    Gate 1 cannot apply, so it must defer to gate 2 rather than guessing — abstaining
    here would make `--strategy lexical` refuse everything.
    """
    provider = StubProvider(GOOD)
    answer = _answerer(provider, _scored("leadership::c01"), dense_score=None).answer("q")
    assert answer.answered
    assert len(provider.calls) == 1


# ---------------------------------------------------------------------------
# Gate 2
# ---------------------------------------------------------------------------


def test_gate_two_abstains_on_insufficient_context() -> None:
    """The near-miss case, which no threshold can catch."""
    provider = StubProvider(
        {
            "sufficient_context": False,
            "answer": "The knowledge base describes the role but does not name anyone.",
            "cited_chunk_ids": [],
            "confidence": "high",
        }
    )
    answer = _answerer(provider, _scored("leadership::c01")).answer("Who is the Technical Head?")

    assert not answer.answered
    assert answer.abstention_reason is AbstentionReason.INSUFFICIENT_CONTEXT
    assert answer.diagnostics["gates"] == {
        "retrieval_gate": "pass",
        "context_sufficiency": "fail",
        "citation_binding": "skipped",
        "faithfulness_check": "skipped",
    }


def test_refusal_keeps_the_models_explanation() -> None:
    """A refusal that says what the knowledge base *does* cover is the useful kind."""
    explanation = "The knowledge base describes the role but does not name the holder."
    provider = StubProvider(
        {
            "sufficient_context": False,
            "answer": explanation,
            "cited_chunk_ids": [],
            "confidence": "high",
        }
    )
    assert _answerer(provider, _scored("leadership::c01")).answer("q").text == explanation


def test_gate_two_can_be_disabled() -> None:
    """Needed for the ablation that measures how much gate 2 is actually doing."""
    provider = StubProvider(
        {
            "sufficient_context": False,
            "answer": "Something.",
            "cited_chunk_ids": ["leadership::c01"],
            "confidence": "low",
        }
    )
    answerer = _answerer(
        provider,
        _scored("leadership::c01"),
        abstention={"require_sufficient_context": False},
    )
    assert answerer.answer("q").answered


# ---------------------------------------------------------------------------
# Citation binding
# ---------------------------------------------------------------------------


def test_fabricated_citations_are_dropped() -> None:
    """A cited id that was never retrieved is caught mechanically, not trusted."""
    provider = StubProvider(
        {
            "sufficient_context": True,
            "answer": "An answer.",
            "cited_chunk_ids": ["leadership::c01", "invented::c99"],
            "confidence": "high",
        }
    )
    answer = _answerer(provider, _scored("leadership::c01", "leadership::c02")).answer("q")

    assert [c.chunk_id for c in answer.citations] == ["leadership::c01"]
    assert answer.diagnostics["invalid_citations"] == ["invented::c99"]
    assert answer.diagnostics["gates"]["citation_binding"] == "repaired"


def test_an_answer_citing_nothing_valid_is_refused() -> None:
    """R3 says every answer carries its sources.

    An answer whose citations are all fabricated cannot be traced to the knowledge base,
    and there is no way to reconstruct what it actually used — so it is not reported.
    """
    provider = StubProvider(
        {
            "sufficient_context": True,
            "answer": "Confident but untraceable.",
            "cited_chunk_ids": ["ghost::c00"],
            "confidence": "high",
        }
    )
    answer = _answerer(provider, _scored("leadership::c01")).answer("q")

    assert not answer.answered
    assert answer.abstention_reason is AbstentionReason.UNFAITHFUL_ANSWER
    assert answer.diagnostics["gates"]["citation_binding"] == "fail"


def test_duplicate_citations_are_collapsed() -> None:
    provider = StubProvider(
        {
            "sufficient_context": True,
            "answer": "An answer.",
            "cited_chunk_ids": ["leadership::c01", "leadership::c01"],
            "confidence": "high",
        }
    )
    assert len(_answerer(provider, _scored("leadership::c01")).answer("q").citations) == 1


def test_citations_can_span_documents() -> None:
    """Multi-document answers must report every source file they used."""
    chunks = (
        ScoredChunk(chunk=_chunk("leadership::c01", "leadership"), score=0.9, rank=1),
        ScoredChunk(chunk=_chunk("hackathons::c01", "hackathons"), score=0.8, rank=2),
    )
    provider = StubProvider(
        {
            "sufficient_context": True,
            "answer": "Spans both.",
            "cited_chunk_ids": ["leadership::c01", "hackathons::c01"],
            "confidence": "high",
        }
    )
    answer = _answerer(provider, chunks).answer("q")
    assert set(answer.sources) == {"leadership.txt", "hackathons.txt"}


def test_malformed_structured_output_abstains_rather_than_guessing() -> None:
    provider = StubProvider(StructuredOutputError("bad json"))
    answer = _answerer(provider, _scored("leadership::c01")).answer("q")
    assert not answer.answered
    assert answer.abstention_reason is AbstentionReason.PROVIDER_UNAVAILABLE


# ---------------------------------------------------------------------------
# Gate 3
# ---------------------------------------------------------------------------


def test_gate_three_is_off_by_default() -> None:
    provider = StubProvider(GOOD)
    answer = _answerer(provider, _scored("leadership::c01")).answer("q")
    assert answer.diagnostics["gates"]["faithfulness_check"] == "skipped"
    assert len(provider.calls) == 1, "verification must not run unless requested"


def test_gate_three_refuses_an_unsupported_answer() -> None:
    provider = StubProvider(
        GOOD,
        {
            "supported": False,
            "unsupported_claims": ["Domain leads earn a stipend."],
            "reason": "The passages never mention pay.",
        },
    )
    answer = _answerer(provider, _scored("leadership::c01")).answer("q", verify_faithfulness=True)

    assert not answer.answered
    assert answer.abstention_reason is AbstentionReason.UNFAITHFUL_ANSWER
    assert answer.diagnostics["gates"]["faithfulness_check"] == "fail"
    assert answer.diagnostics["verification"]["unsupported_claims"]


def test_gate_three_passes_a_supported_answer() -> None:
    provider = StubProvider(
        GOOD, {"supported": True, "unsupported_claims": [], "reason": "All supported."}
    )
    answer = _answerer(provider, _scored("leadership::c01")).answer("q", verify_faithfulness=True)
    assert answer.answered
    assert answer.diagnostics["gates"]["faithfulness_check"] == "pass"


def test_verifier_trusts_the_claim_list_over_the_flag() -> None:
    """When the model contradicts itself, the enumerated work beats the summary."""
    provider = StubProvider({"supported": True, "unsupported_claims": ["x"], "reason": "r"})
    result = verify_answer(
        provider=provider,  # type: ignore[arg-type]
        question="q",
        answer="a",
        citations=_answerer(StubProvider(GOOD), _scored("leadership::c01")).answer("q").citations,
    )
    assert not result.supported


def test_verifier_refuses_when_there_is_nothing_to_check() -> None:
    provider = StubProvider()
    result = verify_answer(provider=provider, question="q", answer="a", citations=())  # type: ignore[arg-type]
    assert not result.supported
    assert provider.calls == [], "no call needed when there are no citations"


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def test_every_answer_reports_all_four_gates() -> None:
    """Diagnostics are part of the contract, so a wrong refusal is attributable."""
    answer = _answerer(StubProvider(GOOD), _scored("leadership::c01")).answer("q")
    assert set(answer.diagnostics["gates"]) == {
        "retrieval_gate",
        "context_sufficiency",
        "citation_binding",
        "faithfulness_check",
    }
    assert answer.diagnostics["trace_id"]
    assert answer.diagnostics["abstention_threshold"] == 0.55


def test_prompt_carries_chunk_ids_the_model_must_cite() -> None:
    provider = StubProvider(GOOD)
    _answerer(provider, _scored("leadership::c01")).answer("q")
    assert "[leadership::c01]" in provider.calls[0]["prompt"]


# ---------------------------------------------------------------------------
# Provider plumbing
# ---------------------------------------------------------------------------


def test_retry_prefers_the_servers_advertised_delay() -> None:
    """Guessing beats nothing, but the server knows the exact quota window."""
    slept: list[float] = []
    attempts = {"n": 0}

    def flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("429 RESOURCE_EXHAUSTED. 'retryDelay': '6s'")
        return "ok"

    assert with_retry(flaky, max_retries=2, provider="stub", sleep=slept.append) == "ok"
    assert slept and slept[0] >= 6.0, f"should honour the 6s the server asked for, slept {slept}"


def test_retry_falls_back_to_backoff_without_advice() -> None:
    slept: list[float] = []
    attempts = {"n": 0}

    def flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("503 UNAVAILABLE high demand")
        return "ok"

    with_retry(flaky, max_retries=2, provider="stub", sleep=slept.append)
    assert slept and 0 < slept[0] < 6.0


def test_non_retryable_errors_fail_immediately() -> None:
    slept: list[float] = []

    def bad() -> str:
        raise RuntimeError("400 INVALID_ARGUMENT")

    with pytest.raises(ProviderUnavailableError):
        with_retry(bad, max_retries=3, provider="stub", sleep=slept.append)
    assert slept == [], "a malformed request must not be retried"


def test_retries_are_bounded() -> None:
    slept: list[float] = []

    def always_busy() -> str:
        raise RuntimeError("503 UNAVAILABLE")

    with pytest.raises(ProviderUnavailableError):
        with_retry(always_busy, max_retries=2, provider="stub", sleep=slept.append)
    assert len(slept) == 2


def test_is_retryable_distinguishes_transient_from_permanent() -> None:
    assert is_retryable(RuntimeError("503 UNAVAILABLE"))
    assert is_retryable(RuntimeError("429 RESOURCE_EXHAUSTED"))
    assert not is_retryable(RuntimeError("401 UNAUTHENTICATED"))


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("429 RESOURCE_EXHAUSTED quota", ProviderRateLimitedError),
        ("504 deadline exceeded", ProviderTimeoutError),
        ("401 invalid api key", ProviderUnavailableError),
        ("404 NOT_FOUND model gone", ProviderUnavailableError),
    ],
)
def test_errors_map_to_distinct_types(message: str, expected: type) -> None:
    """Three statuses rather than one 500: "no key", "throttled" and "upstream down"
    each need a different reaction from a caller."""
    assert isinstance(translate_error(RuntimeError(message), "stub"), expected)


def test_rate_limiter_paces_calls() -> None:
    """Measured constraint: the free tier allows 5 requests per minute."""
    now = {"t": 0.0}
    slept: list[float] = []

    limiter = RateLimiter(
        60.0,
        monotonic=lambda: now["t"],
        sleep=lambda s: (slept.append(s), now.__setitem__("t", now["t"] + s))[0],
    )
    limiter.wait()
    limiter.wait()
    assert slept == [1.0], "second call should wait a full interval"


def test_rate_limiter_disabled_never_sleeps() -> None:
    slept: list[float] = []
    limiter = RateLimiter(None, monotonic=lambda: 0.0, sleep=slept.append)
    limiter.wait()
    limiter.wait()
    assert slept == []


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"a": 1}', {"a": 1}),
        ('```json\n{"a": 1}\n```', {"a": 1}),
        ('Here you go: {"a": 1}', {"a": 1}),
    ],
)
def test_structured_parsing_survives_common_wrappers(raw: str, expected: dict[str, Any]) -> None:
    """Native schema output needs none of this; backends that only emulate it do."""
    assert parse_structured(raw, provider="stub") == expected


def test_structured_parsing_rejects_junk() -> None:
    with pytest.raises(StructuredOutputError, match="no parseable JSON"):
        parse_structured("not json at all", provider="stub")


def test_structured_parsing_requires_declared_fields() -> None:
    """Silently defaulting a missing field would turn a provider fault into a confident
    wrong answer, since the answerer reads `sufficient_context` as a boolean."""
    with pytest.raises(StructuredOutputError, match="sufficient_context"):
        parse_structured('{"answer": "x"}', provider="stub", required=("sufficient_context",))


def test_retry_after_extraction() -> None:
    assert retry_after_seconds("'retryDelay': '6s'") == 6.0
    assert retry_after_seconds("Please retry in 6.88s") == pytest.approx(6.88)
    assert retry_after_seconds("nothing here") is None
    assert retry_after_seconds("'retryDelay': '99999s'") == 120.0


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_unknown_provider_lists_the_valid_ones() -> None:
    from mlsc_assistant.core.errors import ConfigurationError
    from mlsc_assistant.generation.providers.registry import build_provider

    config = Settings(llm={"provider": "gemini", "models": {"gemini": "m"}}).llm
    object.__setattr__(config, "provider", "mystery-llm")

    with pytest.raises(ConfigurationError, match="Choose one of"):
        build_provider(config)


def test_missing_optional_provider_names_the_extra() -> None:
    """A user switching to Anthropic should be told the install command, not shown a
    raw ImportError."""
    from mlsc_assistant.core.errors import ConfigurationError
    from mlsc_assistant.generation.providers.registry import build_provider

    config = Settings(llm={"provider": "anthropic", "models": {"anthropic": "claude-x"}}).llm

    with pytest.raises(ConfigurationError, match=r"\[anthropic\]"):
        build_provider(config)


def test_gemini_without_a_key_says_where_to_put_one(monkeypatch: pytest.MonkeyPatch) -> None:
    from mlsc_assistant.core.errors import ConfigurationError
    from mlsc_assistant.generation.providers.registry import build_provider

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    config = Settings(llm={"provider": "gemini", "models": {"gemini": "gemini-3.5-flash"}}).llm

    with pytest.raises(ConfigurationError, match=r"\.env"):
        build_provider(config)
