"""HTTP API tests against the contract in docs/API.md.

The provider is stubbed, so `/v1/ask` is tested without spending quota — the free tier
allows 20 requests per day, which is far too few to cover a route properly. The live
model is exercised separately in ``tests/integration/test_generation_live.py``.

Retrieval and document routes run against the real index, since they need no key.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from mlsc_assistant.api.app import create_app
from mlsc_assistant.config import Settings
from mlsc_assistant.generation.providers.base import StructuredResult
from mlsc_assistant.stores.numpy_store import NumpyVectorStore

# These exercise the API against real retrieval, which needs a built index. The index is
# a build artefact and is gitignored, so a fresh clone has none — skip with a message
# naming the fix rather than erroring twenty-one times. CI builds the index first.
pytestmark = pytest.mark.skipif(
    NumpyVectorStore.read_manifest(Settings().index_path) is None,
    reason="no index built; run `mlsc index` first",
)

ANSWER = {
    "sufficient_context": True,
    "answer": "Domain leads plan the learning roadmap and mentor coordinators.",
    "cited_chunk_ids": ["leadership::c01"],
    "confidence": "high",
}
REFUSAL = {
    "sufficient_context": False,
    "answer": "The knowledge base describes the role but does not name the holder.",
    "cited_chunk_ids": [],
    "confidence": "high",
}


class StubProvider:
    name = "stub"
    model = "stub-1"

    def __init__(self, payload: dict[str, Any] | Exception = ANSWER) -> None:
        self.payload = payload

    def complete(self, **kwargs: Any) -> Any:  # pragma: no cover - unused
        raise NotImplementedError

    def complete_structured(self, **kwargs: Any) -> StructuredResult:
        if isinstance(self.payload, Exception):
            raise self.payload
        return StructuredResult(
            data=self.payload, raw_text="", input_tokens=700, output_tokens=90, latency_ms=42.0
        )


@pytest.fixture(scope="module")
def client():  # type: ignore[no-untyped-def]
    settings = Settings()
    app = create_app(settings)
    with TestClient(app) as c:
        c.app.state.mlsc._provider = StubProvider()  # type: ignore[attr-defined]
        yield c


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def test_health_reports_the_index_manifest(client) -> None:  # type: ignore[no-untyped-def]
    body = client.get("/v1/health").json()

    assert body["index"]["chunks"] == 18
    assert body["index"]["documents"] == 6
    assert body["index"]["embedder"] == "BAAI/bge-small-en-v1.5"
    assert body["index"]["stale"] is False


def test_health_never_returns_the_api_key(client) -> None:  # type: ignore[no-untyped-def]
    """`configured` reports presence only. The key must not appear anywhere."""
    response = client.get("/v1/health")
    assert isinstance(response.json()["generation"]["configured"], bool)
    assert "AIza" not in response.text


def test_every_response_carries_a_trace_id(client) -> None:  # type: ignore[no-untyped-def]
    assert client.get("/v1/health").headers["X-Trace-Id"]


def test_a_supplied_trace_id_is_echoed(client) -> None:  # type: ignore[no-untyped-def]
    """Lets a caller correlate its own logs with ours."""
    response = client.get("/v1/health", headers={"X-Trace-Id": "caller-123"})
    assert response.headers["X-Trace-Id"] == "caller-123"


# ---------------------------------------------------------------------------
# Search — no API key required
# ---------------------------------------------------------------------------


def test_search_returns_ranked_results_with_explain(client) -> None:  # type: ignore[no-untyped-def]
    body = client.post("/v1/search", json={"query": "Web3 blockchain", "top_k": 3}).json()

    assert body["strategy"] == "hybrid"
    assert 1 <= len(body["results"]) <= 3
    top = body["results"][0]
    assert top["rank"] == 1
    assert top["doc_id"] == "domains"
    assert top["explain"] is not None
    assert "timings_ms" in body


def test_search_explain_can_be_switched_off(client) -> None:  # type: ignore[no-untyped-def]
    body = client.post("/v1/search", json={"query": "hackathons", "explain": False}).json()
    assert all(r["explain"] is None for r in body["results"])


def test_search_honours_the_strategy_switch(client) -> None:  # type: ignore[no-untyped-def]
    """The ablation is exposed over HTTP so a demo can compare strategies live."""
    dense = client.post("/v1/search", json={"query": "What is MLSC?", "strategy": "dense"}).json()
    lexical = client.post(
        "/v1/search", json={"query": "What is MLSC?", "strategy": "lexical"}
    ).json()

    assert dense["strategy"] == "dense"
    assert dense["results"]
    # Measured in Phase 3: this query has no discriminative term, so BM25 abstains.
    assert lexical["results"] == []


def test_search_rejects_an_empty_query(client) -> None:  # type: ignore[no-untyped-def]
    response = client.post("/v1/search", json={"query": ""})
    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/problem+json")


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


def test_documents_lists_the_whole_knowledge_base(client) -> None:  # type: ignore[no-untyped-def]
    body = client.get("/v1/documents").json()
    assert len(body) == 6
    assert sum(d["chunk_count"] for d in body) == 18


def test_document_detail_returns_full_text(client) -> None:  # type: ignore[no-untyped-def]
    body = client.get("/v1/documents/leadership").json()
    assert body["title"] == "MLSC Leadership Structure"
    assert "Technical Head" in body["text"]


def test_chunks_expose_offsets_for_highlighting(client) -> None:  # type: ignore[no-untyped-def]
    """A client highlights the cited passage using char_range against the document."""
    document = client.get("/v1/documents/domains").json()
    chunks = client.get("/v1/documents/domains/chunks").json()

    assert chunks
    for chunk in chunks:
        start, end = chunk["char_range"]
        slice_ = document["text"][start:end]
        assert " ".join(slice_.split()) == " ".join(chunk["text"].split())


def test_unknown_document_lists_what_exists(client) -> None:  # type: ignore[no-untyped-def]
    response = client.get("/v1/documents/nope")
    assert response.status_code == 404
    body = response.json()
    assert body["type"].endswith("document-not-found")
    assert "leadership" in body["detail"], "the error should name the available documents"
    assert body["trace_id"]


# ---------------------------------------------------------------------------
# Ask
# ---------------------------------------------------------------------------


def test_ask_returns_an_answer_with_citations(client) -> None:  # type: ignore[no-untyped-def]
    body = client.post(
        "/v1/ask", json={"question": "What are the responsibilities of a domain lead?"}
    ).json()

    assert body["answered"] is True
    assert body["abstained"] is False
    assert body["abstention_reason"] is None
    assert body["citations"]
    assert body["sources"] == ["leadership.txt"]
    citation = body["citations"][0]
    assert citation["chunk_id"] == "leadership::c01"
    assert citation["char_range"]


def test_abstention_is_a_200_not_an_error() -> None:
    """Refusing correctly is a successful outcome (D10).

    A 4xx would make a correct refusal indistinguishable from a malformed request, and
    client retry logic would treat it as something to retry.
    """
    app = create_app(Settings())
    with TestClient(app) as client:
        client.app.state.mlsc._provider = StubProvider(REFUSAL)  # type: ignore[attr-defined]
        response = client.post("/v1/ask", json={"question": "Who is the current Technical Head?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answered"] is False
    assert body["abstained"] is True
    assert body["abstention_reason"] == "insufficient_context"
    assert body["citations"] == []


def test_ask_reports_which_gates_fired(client) -> None:  # type: ignore[no-untyped-def]
    body = client.post("/v1/ask", json={"question": "What do domain leads do?"}).json()
    gates = body["diagnostics"]["gates"]
    assert set(gates) == {
        "retrieval_gate",
        "context_sufficiency",
        "citation_binding",
        "faithfulness_check",
    }


def test_diagnostics_can_be_suppressed(client) -> None:  # type: ignore[no-untyped-def]
    body = client.post(
        "/v1/ask",
        json={"question": "What do domain leads do?", "include_diagnostics": False},
    ).json()
    assert body["diagnostics"] is None


def test_ask_rejects_an_overlong_question(client) -> None:  # type: ignore[no-untyped-def]
    response = client.post("/v1/ask", json={"question": "x" * 5000})
    assert response.status_code == 422


def test_provider_failure_maps_to_a_typed_problem() -> None:
    """ "No key", "throttled" and "upstream down" need different reactions, so they get
    different statuses rather than a generic 500."""
    from mlsc_assistant.core.errors import ProviderRateLimitedError

    app = create_app(Settings())
    with TestClient(app) as client:
        client.app.state.mlsc._provider = StubProvider(  # type: ignore[attr-defined]
            ProviderRateLimitedError("quota exhausted", retry_after=30)
        )
        # An on-topic question, so gate 1 passes and the provider is actually reached.
        # "anything" would be caught by the retrieval gate before any call is made,
        # which is correct behaviour but tests nothing about provider errors.
        response = client.post(
            "/v1/ask", json={"question": "What are the responsibilities of a domain lead?"}
        )

    assert response.status_code == 429
    assert response.headers["Retry-After"] == "30"
    assert response.json()["type"].endswith("provider-rate-limited")


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def test_stream_emits_retrieval_before_the_answer(client) -> None:  # type: ignore[no-untyped-def]
    """The reason streaming exists here: sources appear before the answer does."""
    with client.stream(
        "POST", "/v1/ask/stream", json={"question": "What do domain leads do?"}
    ) as response:
        assert response.status_code == 200
        events = [
            line[len("event: ") :] for line in response.iter_lines() if line.startswith("event: ")
        ]

    assert events[0] == "retrieval"
    assert events == ["retrieval", "answer", "citations", "done"]


def test_stream_retrieval_event_carries_sources(client) -> None:  # type: ignore[no-untyped-def]
    with client.stream("POST", "/v1/ask/stream", json={"question": "Web3"}) as response:
        payloads = [
            json.loads(line[len("data: ") :])
            for line in response.iter_lines()
            if line.startswith("data: ")
        ]

    assert payloads[0]["chunks"]
    assert payloads[0]["documents"]


# ---------------------------------------------------------------------------
# Evaluation jobs
# ---------------------------------------------------------------------------


def test_eval_run_is_a_job_not_a_blocking_request(client) -> None:  # type: ignore[no-untyped-def]
    response = client.post("/v1/eval/runs", json={"strategy": "dense"})

    assert response.status_code == 202
    assert response.headers["Location"].startswith("/v1/eval/runs/")
    run_id = response.json()["run_id"]

    # TestClient runs background tasks synchronously on response close.
    result = client.get(f"/v1/eval/runs/{run_id}").json()
    assert result["status"] == "completed", result.get("error")
    assert result["metrics"]["retrieval"]["6"]["recall_at_k"] > 0.9


def test_unknown_eval_run_explains_that_runs_are_in_memory(client) -> None:  # type: ignore[no-untyped-def]
    response = client.get("/v1/eval/runs/does-not-exist")
    assert response.status_code == 404
    assert "evaluation/runs" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Contract surface
# ---------------------------------------------------------------------------


def test_openapi_documents_every_endpoint_the_contract_promises(client) -> None:  # type: ignore[no-untyped-def]
    paths = client.get("/openapi.json").json()["paths"]
    for path in (
        "/v1/health",
        "/v1/ask",
        "/v1/ask/stream",
        "/v1/search",
        "/v1/documents",
        "/v1/documents/{doc_id}",
        "/v1/documents/{doc_id}/chunks",
        "/v1/eval/runs",
        "/v1/eval/runs/{run_id}",
    ):
        assert path in paths, f"{path} is documented in docs/API.md but not served"
