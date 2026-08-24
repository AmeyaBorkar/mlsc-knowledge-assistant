"""Full evaluation: answer every question, then score retrieval, generation and abstention.

Separate from ``runner.py`` deliberately. That one is the key-free retrieval harness that
runs in CI; this one spends quota. Keeping them apart means the CI gate can never
accidentally start needing a secret.

Cost is the dominant design constraint here. Each question costs one answering call plus
up to three judging calls, and the free tier allows 20 requests per day. So:

- judge verdicts are cached by content, making a re-run nearly free
- abstained questions skip judging entirely, since there is no answer to score
- ``--limit`` and per-family selection exist so a partial run is possible
- a quota failure partway through is recorded as an **incomplete run**, not silently
  scored as bad answers
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from mlsc_assistant.config import Settings
from mlsc_assistant.core.errors import MLSCError
from mlsc_assistant.core.models import IndexManifest, RetrievalStrategy
from mlsc_assistant.core.ports import Embedder
from mlsc_assistant.evaluation.dataset import EvalDataset, EvalQuestion
from mlsc_assistant.evaluation.judge import Judge
from mlsc_assistant.evaluation.metrics.abstention import (
    AbstentionOutcome,
    score_abstention,
)
from mlsc_assistant.evaluation.metrics.generation import (
    aggregate_generation,
    score_answer_correctness,
    score_answer_relevancy,
    score_faithfulness,
)
from mlsc_assistant.evaluation.metrics.retrieval import aggregate, score_question
from mlsc_assistant.generation.answerer import GroundedAnswerer


@dataclass(frozen=True, slots=True)
class FullQuestionTrace:
    question_id: str
    question: str
    type: str
    answerable: bool
    subtype: str | None
    answered: bool
    abstention_reason: str | None
    answer: str
    reference_answer: str
    citations: list[str]
    retrieved_chunks: list[str]
    gold_chunks: list[str]
    gates: dict[str, str]
    retrieval_scores: dict[str, Any] | None = None
    generation_scores: dict[str, float] = field(default_factory=dict)
    judge_detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class FullEvalRun:
    run_id: str
    dataset: str
    config: dict[str, Any]
    metrics: dict[str, Any]
    traces: list[FullQuestionTrace]
    complete: bool = True
    error: str | None = None


def run_full_evaluation(
    dataset: EvalDataset,
    answerer: GroundedAnswerer,
    embedder: Embedder,
    judge: Judge,
    settings: Settings,
    *,
    manifest: IndexManifest | None = None,
    strategy: RetrievalStrategy | None = None,
    families: Sequence[str] = ("retrieval", "generation", "abstention"),
    limit: int | None = None,
    progress: Any = None,
) -> FullEvalRun:
    strategy = strategy or RetrievalStrategy(settings.retrieval.strategy)
    k = settings.retrieval.top_k
    questions = list(dataset.questions)[: limit or len(dataset.questions)]

    traces: list[FullQuestionTrace] = []
    retrieval_scores = []
    generation_scores: list[dict[str, float]] = []
    per_type_generation: dict[str, list[dict[str, float]]] = {}
    abstention_outcomes: list[AbstentionOutcome] = []
    complete = True
    error: str | None = None

    for question in questions:
        try:
            trace, retrieval_score, gen_score = _evaluate_one(
                question, answerer, embedder, judge, settings, strategy, k, families
            )
        except MLSCError as exc:
            # Stop rather than continue: every remaining question would fail the same
            # way, and a run half-scored by quota exhaustion must not look like a result.
            complete = False
            error = f"{exc.title}: {exc.detail}"
            break

        traces.append(trace)
        if retrieval_score is not None:
            retrieval_scores.append(retrieval_score)
        if gen_score:
            generation_scores.append(gen_score)
            per_type_generation.setdefault(question.type.value, []).append(gen_score)

        abstention_outcomes.append(
            AbstentionOutcome(
                question_id=question.id,
                answerable=question.answerable,
                abstained=not trace.answered,
                subtype=question.subtype.value if question.subtype else None,
                score=(trace.retrieval_scores or {}).get("top_dense_score"),
            )
        )
        if progress is not None:
            progress(question, trace)

    judge.cache.flush()

    metrics: dict[str, Any] = {"primary_k": k, "counts": {"evaluated": len(traces)}}
    if "retrieval" in families:
        metrics["retrieval"] = {str(k): aggregate(retrieval_scores)}
    if "generation" in families:
        metrics["generation"] = aggregate_generation(generation_scores)
        metrics["generation_by_type"] = {
            t: aggregate_generation(s) for t, s in sorted(per_type_generation.items())
        }
        metrics["counts"]["judged"] = len(generation_scores)
    if "abstention" in families:
        metrics["abstention"] = score_abstention(abstention_outcomes).as_dict()
        metrics["abstention_by_subtype"] = _subtype_breakdown(abstention_outcomes)

    metrics["judge"] = {
        "calls": judge.calls,
        "cache_hits": judge.cache_hits,
        "model": judge.provider.model,
    }

    now = datetime.now(UTC)
    return FullEvalRun(
        run_id=f"{now:%Y%m%d-%H%M%S}-full-{strategy.value}",
        dataset=dataset.name,
        config={
            "strategy": strategy.value,
            "top_k": k,
            "embedder": settings.embedding.model,
            "provider": answerer.provider.name,
            "model": answerer.provider.model,
            "judge_model": judge.provider.model,
            "prompt_version": settings.generation.prompt_version,
            "abstention_threshold": settings.abstention.min_dense_score,
            "families": list(families),
            "index": (
                {"chunks": manifest.chunk_count, "built_at": manifest.built_at.isoformat()}
                if manifest
                else None
            ),
        },
        metrics=metrics,
        traces=traces,
        complete=complete,
        error=error,
    )


def _evaluate_one(
    question: EvalQuestion,
    answerer: GroundedAnswerer,
    embedder: Embedder,
    judge: Judge,
    settings: Settings,
    strategy: RetrievalStrategy,
    k: int,
    families: Sequence[str],
) -> tuple[FullQuestionTrace, Any, dict[str, float]]:
    answer = answerer.answer(question.question, strategy=strategy)
    diagnostics = answer.diagnostics
    retrieval_diag = diagnostics.get("retrieval", {})
    retrieved = [c["chunk_id"] for c in retrieval_diag.get("chunks", [])]

    retrieval_score = None
    if "retrieval" in families and question.answerable and question.has_chunk_labels:
        retrieval_score = score_question(
            retrieved,
            retrieval_diag.get("documents_represented", []),
            question.gold_chunks,
            question.gold_documents,
            k,
        )

    generation_scores: dict[str, float] = {}
    judge_detail: dict[str, Any] = {}

    # Only answered questions are judged. An abstention cites nothing and claims nothing,
    # so it is trivially faithful — averaging refusals in would let a system that refuses
    # everything post a perfect faithfulness score.
    if "generation" in families and answer.answered:
        # Full chunk text, not citation.snippet. Snippets are truncated to 240
        # characters for display; judging against them measures truncation, not
        # faithfulness, and produced a confident 0.00 on an answer quoted verbatim
        # from its own source.
        contexts = [
            f"[{chunk_id}] {text}" for chunk_id, text in answerer.cited_passages(answer.citations)
        ]

        faithfulness = score_faithfulness(
            judge, question=question.question, answer=answer.text, contexts=contexts
        )
        generation_scores["faithfulness"] = faithfulness.score
        judge_detail["unsupported_claims"] = list(faithfulness.unsupported)

        relevancy = score_answer_relevancy(
            judge,
            embedder,
            question=question.question,
            answer=answer.text,
            n_questions=settings.evaluation.relevancy_questions,
        )
        generation_scores["answer_relevancy"] = relevancy.score
        judge_detail["generated_questions"] = list(relevancy.generated_questions)

        if question.reference_answer:
            correctness = score_answer_correctness(
                judge,
                question=question.question,
                answer=answer.text,
                reference=question.reference_answer,
            )
            generation_scores["answer_correctness"] = correctness.score
            judge_detail["correctness"] = {
                "verdict": correctness.verdict,
                "missing": list(correctness.missing),
                "contradictions": list(correctness.contradictions),
                "reason": correctness.reason,
            }

    trace = FullQuestionTrace(
        question_id=question.id,
        question=question.question,
        type=question.type.value,
        answerable=question.answerable,
        subtype=question.subtype.value if question.subtype else None,
        answered=answer.answered,
        abstention_reason=(answer.abstention_reason.value if answer.abstention_reason else None),
        answer=answer.text,
        reference_answer=question.reference_answer,
        citations=[c.chunk_id for c in answer.citations],
        retrieved_chunks=retrieved,
        gold_chunks=list(question.gold_chunks),
        gates=diagnostics.get("gates", {}),
        retrieval_scores=(
            {**asdict(retrieval_score), "top_dense_score": retrieval_diag.get("top_dense_score")}
            if retrieval_score
            else {"top_dense_score": retrieval_diag.get("top_dense_score")}
        ),
        generation_scores=generation_scores,
        judge_detail=judge_detail,
    )
    return trace, retrieval_score, generation_scores


def _subtype_breakdown(outcomes: Sequence[AbstentionOutcome]) -> dict[str, Any]:
    """Abstention split by *why* a question is unanswerable.

    An aggregate can look healthy while every near miss slips through, and only the
    split makes that visible.
    """
    groups: dict[str, list[AbstentionOutcome]] = {}
    for outcome in outcomes:
        if not outcome.answerable:
            groups.setdefault(outcome.subtype or "unspecified", []).append(outcome)
    return {
        name: {
            "caught": sum(1 for o in items if o.abstained),
            "total": len(items),
            "recall": sum(1 for o in items if o.abstained) / len(items),
        }
        for name, items in sorted(groups.items())
    }
