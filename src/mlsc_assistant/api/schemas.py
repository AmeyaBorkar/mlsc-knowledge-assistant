"""Request and response DTOs.

Deliberately separate from ``core.models``. Domain objects carry things the wire should
not — raw vectors, internal scores, provider detail — and the public contract should be
free to stay stable while internals move. The translation functions below are the only
place the two shapes meet.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

from mlsc_assistant.core.models import Answer, IndexManifest, RetrievalResult

Strategy = Literal["hybrid", "dense", "lexical"]


class MetricFamily(StrEnum):
    RETRIEVAL = "retrieval"


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    strategy: Strategy | None = None
    include_diagnostics: bool = True
    verify_faithfulness: bool | None = Field(
        default=None,
        description="Enable abstention gate 3. Costs a second model call.",
    )


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    strategy: Strategy | None = None
    explain: bool = True
    max_chunks_per_document: int | None = Field(default=None, ge=1)


class EvalRunRequest(BaseModel):
    dataset: str | None = None
    strategy: Strategy | None = None
    metrics: list[MetricFamily] = Field(default_factory=lambda: [MetricFamily.RETRIEVAL])
    """Only the retrieval family is implemented; generation metrics arrive in Phase 7."""


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


class CitationOut(BaseModel):
    chunk_id: str
    doc_id: str
    doc_title: str
    source_file: str
    snippet: str
    char_range: tuple[int, int]
    score: float


class AskResponse(BaseModel):
    question: str
    answer: str
    answered: bool
    abstained: bool
    abstention_reason: str | None
    confidence: str
    citations: list[CitationOut]
    sources: list[str]
    diagnostics: dict[str, Any] | None = None


class ExplainOut(BaseModel):
    """Each retriever's independent verdict on one chunk.

    Exposed so a caller can see *why* a chunk ranked where it did without re-running
    anything — the single most useful debugging affordance in the system.
    """

    dense_rank: int | None = None
    dense_score: float | None = None
    lexical_rank: int | None = None
    lexical_score: float | None = None
    rrf_score: float | None = None
    matched_terms: list[str] = Field(default_factory=list)


class SearchResultOut(BaseModel):
    chunk_id: str
    doc_id: str
    doc_title: str
    source_file: str
    text: str
    char_range: tuple[int, int]
    score: float
    rank: int
    explain: ExplainOut | None = None


class SearchResponse(BaseModel):
    query: str
    strategy: str
    results: list[SearchResultOut]
    candidates_considered: int
    top_dense_score: float | None
    documents_represented: list[str]
    timings_ms: dict[str, float]


class DocumentSummary(BaseModel):
    doc_id: str
    title: str
    source_file: str
    chunk_count: int
    characters: int


class DocumentDetail(DocumentSummary):
    text: str


class ChunkOut(BaseModel):
    chunk_id: str
    doc_id: str
    index: int
    kind: str
    text: str
    char_range: tuple[int, int]
    token_estimate: int


class IndexInfo(BaseModel):
    built_at: str
    documents: int
    chunks: int
    embedder: str
    dimension: int
    chunker_version: str
    stale: bool


class GenerationInfo(BaseModel):
    provider: str
    model: str
    configured: bool
    """Whether a key is present. The key itself is never returned."""


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    index: IndexInfo | None
    generation: GenerationInfo


class EvalRunSummary(BaseModel):
    run_id: str
    status: Literal["queued", "running", "completed", "failed"]
    dataset: str | None = None
    strategy: str | None = None
    metrics: dict[str, Any] | None = None
    error: str | None = None
    report_path: str | None = None


class Problem(BaseModel):
    """RFC 9457 problem detail. `detail` always names the action that fixes it."""

    type: str
    title: str
    status: int
    detail: str
    trace_id: str | None = None


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------


def to_ask_response(answer: Answer, *, include_diagnostics: bool) -> AskResponse:
    return AskResponse(
        question=answer.question,
        answer=answer.text,
        answered=answer.answered,
        abstained=answer.abstained,
        abstention_reason=(answer.abstention_reason.value if answer.abstention_reason else None),
        confidence=answer.confidence.value,
        citations=[
            CitationOut(
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                doc_title=c.doc_title,
                source_file=c.source_file,
                snippet=c.snippet,
                char_range=c.char_range,
                score=round(c.score, 5),
            )
            for c in answer.citations
        ],
        sources=list(answer.sources),
        diagnostics=answer.diagnostics if include_diagnostics else None,
    )


def to_search_response(result: RetrievalResult, *, explain: bool) -> SearchResponse:
    return SearchResponse(
        query=result.query,
        strategy=result.strategy.value,
        results=[
            SearchResultOut(
                chunk_id=sc.chunk.chunk_id,
                doc_id=sc.chunk.doc_id,
                doc_title=sc.chunk.doc_title,
                source_file=sc.chunk.source_file,
                text=sc.chunk.text,
                char_range=sc.chunk.char_range,
                score=round(sc.score, 5),
                rank=sc.rank,
                explain=(
                    ExplainOut(
                        dense_rank=sc.dense_rank,
                        dense_score=None if sc.dense_score is None else round(sc.dense_score, 5),
                        lexical_rank=sc.lexical_rank,
                        lexical_score=(
                            None if sc.lexical_score is None else round(sc.lexical_score, 5)
                        ),
                        rrf_score=None if sc.rrf_score is None else round(sc.rrf_score, 6),
                        matched_terms=list(sc.matched_terms),
                    )
                    if explain
                    else None
                ),
            )
            for sc in result.chunks
        ],
        candidates_considered=result.candidates_considered,
        top_dense_score=result.top_dense_score,
        documents_represented=list(result.documents_represented),
        timings_ms=result.timings_ms,
    )


def to_index_info(manifest: IndexManifest, *, stale: bool) -> IndexInfo:
    return IndexInfo(
        built_at=manifest.built_at.isoformat(),
        documents=manifest.document_count,
        chunks=manifest.chunk_count,
        embedder=manifest.embedder,
        dimension=manifest.dimension,
        chunker_version=manifest.chunker_version,
        stale=stale,
    )
