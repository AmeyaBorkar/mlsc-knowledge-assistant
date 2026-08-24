"""End-to-end generation against the real provider.

Skipped automatically when no API key is configured, which is how CI stays green without
a secret — the retrieval integration tests still run there, because retrieval needs no
key (DECISIONS.md D4).

Deliberately few tests. The free tier allows 5 requests per minute, so each test here
costs roughly 12 seconds of wall clock; the exhaustive behaviour is covered offline in
``tests/unit/test_generation.py`` with a stub provider. What these add is the one thing a
stub cannot: evidence that a real model, given this prompt, actually behaves as designed.
"""

from __future__ import annotations

import pytest

from mlsc_assistant.config import Settings, get_settings
from mlsc_assistant.core.models import AbstentionReason
from mlsc_assistant.factories import load_store, make_answerer, make_embedder, make_retriever

_settings = get_settings()

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(
        not _settings.llm.is_configured,
        reason="no LLM API key configured; generation tests need one",
    ),
]


@pytest.fixture(scope="module")
def answerer():  # type: ignore[no-untyped-def]
    """One answerer for the module.

    Shared so the rate limiter paces the whole module rather than resetting per test,
    and so the embedding model loads once.
    """
    settings = Settings()
    store, _ = load_store(settings)
    retriever = make_retriever(settings, embedder=make_embedder(settings), store=store)
    return make_answerer(settings, retriever=retriever)


def test_answers_a_direct_question_with_citations(answerer) -> None:  # type: ignore[no-untyped-def]
    """The core demo case."""
    answer = answerer.answer("What are the responsibilities of a domain lead?")

    assert answer.answered
    assert answer.citations, "an answered question must carry its sources (R3)"
    assert "leadership.txt" in answer.sources
    # Content check kept loose: the exact wording is the model's, and asserting on it
    # would make this a change-detector rather than a behaviour test.
    assert "roadmap" in answer.text.lower() or "coordinator" in answer.text.lower()


def test_every_citation_resolves_to_a_real_passage(answerer) -> None:  # type: ignore[no-untyped-def]
    """Citation binding, verified against the live index rather than a stub."""
    answer = answerer.answer("What technical domains exist in MLSC?")
    known = {c.chunk_id for c in answerer.retriever.chunks_by_id.values()}

    assert answer.answered
    for citation in answer.citations:
        assert citation.chunk_id in known
        assert citation.snippet


def test_refuses_a_near_miss_that_no_threshold_could_catch(answerer) -> None:  # type: ignore[no-untyped-def]
    """The question the three-gate design exists for.

    The knowledge base describes the Technical Head role in detail and never names the
    holder, so retrieval scores high and gate 1 correctly lets it through. Only the model
    reading the context can tell the fact is absent.
    """
    answer = answerer.answer("Who is the current Technical Head of MLSC?")

    assert not answer.answered
    assert answer.abstention_reason is AbstentionReason.INSUFFICIENT_CONTEXT
    gates = answer.diagnostics["gates"]
    assert gates["retrieval_gate"] == "pass", "gate 1 cannot catch this, and should not try"
    assert gates["context_sufficiency"] == "fail", "gate 2 is what catches it"
    # A refusal that names what the knowledge base does cover is the useful kind.
    assert "knowledge base" in answer.text.lower()


def test_refuses_a_fact_it_knows_but_the_corpus_does_not_contain(answerer) -> None:  # type: ignore[no-untyped-def]
    """The sharpest test of grounding as opposed to knowledge.

    The model certainly knows the capital of France. Answering would be correct in the
    world and wrong for this system, which must answer only from the knowledge base.
    """
    answer = answerer.answer("What is the capital of France?")

    assert not answer.answered
    assert "paris" not in answer.text.lower()


def test_does_not_borrow_an_adjacent_number(answerer) -> None:  # type: ignore[no-untyped-def]
    """Adversarial by construction.

    The retrieved passage states that each domain has *two domain leads*. Nothing states
    a coordinator count, so reporting "two" would be a hallucination assembled from a
    real nearby fact — the most plausible-looking failure mode this system has.
    """
    answer = answerer.answer("How many coordinators does each domain have?")

    assert not answer.answered, f"should abstain, got: {answer.text}"
    assert "two" not in answer.text.lower().split("domain lead")[0]


def test_multi_document_answer_cites_multiple_sources(answerer) -> None:  # type: ignore[no-untyped-def]
    answer = answerer.answer("How do MLSC hackathons connect to the technical domains?")

    assert answer.answered
    assert len(answer.sources) >= 2, f"expected several sources, got {answer.sources}"


def test_diagnostics_are_complete_on_a_live_call(answerer) -> None:  # type: ignore[no-untyped-def]
    answer = answerer.answer("How are hackathon projects evaluated?")
    diagnostics = answer.diagnostics

    assert diagnostics["trace_id"]
    assert diagnostics["retrieval"]["top_dense_score"] is not None
    generation = diagnostics["generation"]
    assert generation["prompt_version"] == "grounded-v1"
    assert generation["input_tokens"] and generation["output_tokens"]
    assert set(diagnostics["gates"]) == {
        "retrieval_gate",
        "context_sufficiency",
        "citation_binding",
        "faithfulness_check",
    }
