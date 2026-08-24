"""Evaluation orchestration.

Runs an evaluation set through retrieval, scores it, and writes a run directory that
carries enough context to be interpreted months later: the config that produced it, the
index manifest, and a per-question trace.

A metric without its config is not a result. Two runs of "recall 0.86" mean nothing next
to each other unless you know which strategy, k, embedder and chunker produced each.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mlsc_assistant.config import Settings
from mlsc_assistant.core.models import IndexManifest, RetrievalStrategy
from mlsc_assistant.evaluation.dataset import EvalDataset, EvalQuestion
from mlsc_assistant.evaluation.metrics.abstention import (
    AbstentionOutcome,
    ThresholdPoint,
    sweep_threshold,
)
from mlsc_assistant.evaluation.metrics.retrieval import (
    RetrievalScores,
    aggregate,
    score_question,
)
from mlsc_assistant.retrieval.retriever import HybridRetriever

DEFAULT_THRESHOLDS = tuple(round(0.30 + 0.025 * i, 4) for i in range(29))  # 0.300 - 1.000


@dataclass(frozen=True, slots=True)
class QuestionTrace:
    """Everything needed to explain one question's score without re-running it."""

    question_id: str
    question: str
    type: str
    answerable: bool
    subtype: str | None
    gold_chunks: list[str]
    retrieved_chunks: list[str]
    retrieved_documents: list[str]
    top_dense_score: float | None
    scores: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class EvalRun:
    run_id: str
    dataset: str
    config: dict[str, Any]
    metrics: dict[str, Any]
    traces: list[QuestionTrace]
    thresholds: list[ThresholdPoint] = field(default_factory=list)


def run_retrieval_evaluation(
    dataset: EvalDataset,
    retriever: HybridRetriever,
    settings: Settings,
    *,
    manifest: IndexManifest | None = None,
    strategy: RetrievalStrategy | None = None,
    k_values: Sequence[int] | None = None,
    run_id: str | None = None,
    timestamp: datetime | None = None,
) -> EvalRun:
    """Score an evaluation set on retrieval only. No LLM, no API key."""
    strategy = strategy or RetrievalStrategy(settings.retrieval.strategy)
    k_values = list(k_values or settings.evaluation.k_values)
    primary_k = settings.retrieval.top_k
    # Metrics are reported at several k so nothing is tuned to one value, but the report
    # has to lead with a single number, and top_k is what the system actually serves.
    if primary_k not in k_values:
        k_values = sorted({*k_values, primary_k})

    # Retrieve once at the widest k and slice for the rest. Retrieval is deterministic,
    # so re-querying per k would only add latency and a chance of drift.
    widest = max(k_values)

    traces: list[QuestionTrace] = []
    scores_by_k: dict[int, list[RetrievalScores]] = {k: [] for k in k_values}
    per_type: dict[str, list[RetrievalScores]] = {}
    multi_doc_hits: list[bool] = []
    abstention_inputs: list[AbstentionOutcome] = []

    for question in dataset.questions:
        result = retriever.retrieve(question.question, strategy=strategy, top_k=widest)
        retrieved = [sc.chunk.chunk_id for sc in result.chunks]
        retrieved_docs = list(result.documents_represented)

        abstention_inputs.append(
            AbstentionOutcome(
                question_id=question.id,
                answerable=question.answerable,
                # Gate 1 has not been applied yet; the sweep decides per threshold.
                abstained=False,
                subtype=question.subtype.value if question.subtype else None,
                score=result.top_dense_score,
            )
        )

        # Unanswerable questions have no gold passage by definition, so scoring them on
        # retrieval would be meaningless. They are measured by the abstention family.
        if not (question.answerable and question.has_chunk_labels):
            traces.append(_trace(question, retrieved, retrieved_docs, result.top_dense_score, None))
            continue

        primary: RetrievalScores | None = None
        for k in k_values:
            scored = score_question(
                retrieved, retrieved_docs, question.gold_chunks, question.gold_documents, k
            )
            scores_by_k[k].append(scored)
            if k == primary_k:
                primary = scored

        assert primary is not None
        per_type.setdefault(question.type.value, []).append(primary)
        if question.is_multi_document:
            multi_doc_hits.append(primary.all_docs_hit)

        traces.append(
            _trace(question, retrieved, retrieved_docs, result.top_dense_score, asdict(primary))
        )

    metrics: dict[str, Any] = {
        "primary_k": primary_k,
        "retrieval": {str(k): aggregate(scores_by_k[k]) for k in k_values},
        "by_question_type": {t: aggregate(s) for t, s in sorted(per_type.items())},
        "multi_doc_coverage": (
            sum(multi_doc_hits) / len(multi_doc_hits) if multi_doc_hits else None
        ),
        "counts": {
            "questions": len(dataset),
            "scored": len(scores_by_k[primary_k]),
            "unanswerable": len(dataset.unanswerable),
            "multi_document": len(multi_doc_hits),
        },
    }

    now = timestamp or datetime.now(UTC)
    return EvalRun(
        run_id=run_id or f"{now:%Y%m%d-%H%M%S}-{strategy.value}",
        dataset=dataset.name,
        config=_config_snapshot(settings, strategy, k_values, manifest),
        metrics=metrics,
        traces=traces,
        thresholds=sweep_threshold(abstention_inputs, DEFAULT_THRESHOLDS),
    )


def _trace(
    question: EvalQuestion,
    retrieved: list[str],
    retrieved_docs: list[str],
    top_dense_score: float | None,
    scores: dict[str, Any] | None,
) -> QuestionTrace:
    return QuestionTrace(
        question_id=question.id,
        question=question.question,
        type=question.type.value,
        answerable=question.answerable,
        subtype=question.subtype.value if question.subtype else None,
        gold_chunks=list(question.gold_chunks),
        retrieved_chunks=retrieved,
        retrieved_documents=retrieved_docs,
        top_dense_score=top_dense_score,
        scores=scores,
    )


def _config_snapshot(
    settings: Settings,
    strategy: RetrievalStrategy,
    k_values: Sequence[int],
    manifest: IndexManifest | None,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "strategy": strategy.value,
        "top_k": settings.retrieval.top_k,
        "candidate_k": settings.retrieval.candidate_k,
        "rrf_k": settings.retrieval.rrf_k,
        "max_chunks_per_document": settings.retrieval.max_chunks_per_document,
        "bm25": {
            "k1": settings.retrieval.bm25.k1,
            "b": settings.retrieval.bm25.b,
            "index_title": settings.retrieval.bm25.index_title,
        },
        "embedder": settings.embedding.model,
        "chunking": {
            "version": settings.chunking.version,
            "min_tokens": settings.chunking.min_tokens,
            "prepend_doc_title": settings.chunking.prepend_doc_title,
        },
        "k_values": list(k_values),
    }
    if manifest is not None:
        config["index"] = {
            "built_at": manifest.built_at.isoformat(),
            "chunks": manifest.chunk_count,
            "documents": manifest.document_count,
            "chunker_version": manifest.chunker_version,
        }
    return config


def run_directory(settings: Settings, run: EvalRun) -> Path:
    return settings.runs_path / run.run_id
