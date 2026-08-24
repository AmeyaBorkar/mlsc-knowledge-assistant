"""Retrieval metrics. No LLM, fully deterministic, runnable in CI without a secret.

Each metric answers a different question, which is why several are reported rather than
one headline number:

``recall@k``
    Did we fetch the facts at all? The hard ceiling on the whole system — a passage that
    was never retrieved cannot be answered from, however good the model is.
``precision@k``
    How much of the context window is noise? Predicts faithfulness: noise is what drags
    a model off-source.
``r_precision``
    Precision at k = |gold|. Reported because most questions in this set have a single
    gold chunk, so precision@6 is capped at 0.167 for them — a property of the labelling,
    not of retrieval. R-precision removes that artefact and is comparable across
    questions with different numbers of gold chunks.
``average_precision``
    Rank-weighted precision. The closest deterministic analogue to RAGAS's
    ``context_precision``, which is otherwise LLM-judged.
``mrr``
    How high the *first* relevant chunk lands. Models attend unevenly across a context
    window, so a gold chunk at rank 6 is worth less than the same chunk at rank 1.
``ndcg@k``
    Rank-discounted gain over all relevant chunks. The metric that moves when fusion
    reorders results without changing which chunks are present.
``document`` family
    What the user actually sees is a source document, so document-level coverage is
    measured separately from chunk-level. ``multi_doc_coverage`` restricts the strict
    "every gold document present" measure to genuinely multi-document questions, where
    it is the number the diversification step exists to move.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetrievalScores:
    """Per-question retrieval metrics at one k."""

    k: int
    precision: float
    recall: float
    r_precision: float
    average_precision: float
    reciprocal_rank: float
    ndcg: float
    doc_recall: float
    all_docs_hit: bool
    first_relevant_rank: int | None


def precision_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    """Fraction of the returned chunks that are gold.

    Divided by the number actually returned, not by ``k``. On an 18-chunk corpus a
    request for k=10 can legitimately come back with fewer, and dividing by k would
    penalise the retriever for the corpus being small.
    """
    top = retrieved[:k]
    if not top:
        return 0.0
    relevant = sum(1 for chunk_id in top if chunk_id in set(gold))
    return relevant / len(top)


def recall_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    if not gold:
        return 0.0
    found = sum(1 for chunk_id in set(gold) if chunk_id in set(retrieved[:k]))
    return found / len(set(gold))


def r_precision(retrieved: Sequence[str], gold: Sequence[str]) -> float:
    """Precision at k = |gold|, so the ceiling is 1.0 regardless of gold-set size."""
    if not gold:
        return 0.0
    return precision_at_k(retrieved, gold, len(set(gold)))


def average_precision(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    """Mean of precision@i taken at each rank i holding a gold chunk.

    Rewards putting relevant chunks early rather than merely including them.
    """
    gold_set = set(gold)
    if not gold_set:
        return 0.0

    hits = 0
    total = 0.0
    for i, chunk_id in enumerate(retrieved[:k], start=1):
        if chunk_id in gold_set:
            hits += 1
            total += hits / i
    return total / min(len(gold_set), k)


def reciprocal_rank(retrieved: Sequence[str], gold: Sequence[str]) -> float:
    gold_set = set(gold)
    for i, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in gold_set:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    """Binary-relevance nDCG.

    Relevance is binary because the labels are: a chunk either carries a fact needed for
    the answer or it does not. Graded relevance would need a judgement the labelling rule
    deliberately avoids making.
    """
    gold_set = set(gold)
    if not gold_set:
        return 0.0

    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, chunk_id in enumerate(retrieved[:k], start=1)
        if chunk_id in gold_set
    )
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold_set), k) + 1))
    return dcg / ideal if ideal else 0.0


def document_recall(retrieved_docs: Sequence[str], gold_docs: Sequence[str]) -> float:
    if not gold_docs:
        return 0.0
    gold_set = set(gold_docs)
    return len(gold_set & set(retrieved_docs)) / len(gold_set)


def score_question(
    retrieved_chunks: Sequence[str],
    retrieved_docs: Sequence[str],
    gold_chunks: Sequence[str],
    gold_docs: Sequence[str],
    k: int,
) -> RetrievalScores:
    top_docs = _docs_of(retrieved_chunks[:k]) or list(retrieved_docs)
    first = next(
        (i for i, c in enumerate(retrieved_chunks, start=1) if c in set(gold_chunks)), None
    )
    return RetrievalScores(
        k=k,
        precision=precision_at_k(retrieved_chunks, gold_chunks, k),
        recall=recall_at_k(retrieved_chunks, gold_chunks, k),
        r_precision=r_precision(retrieved_chunks, gold_chunks),
        average_precision=average_precision(retrieved_chunks, gold_chunks, k),
        reciprocal_rank=reciprocal_rank(retrieved_chunks[:k], gold_chunks),
        ndcg=ndcg_at_k(retrieved_chunks, gold_chunks, k),
        doc_recall=document_recall(top_docs, gold_docs),
        all_docs_hit=bool(gold_docs) and set(gold_docs) <= set(top_docs),
        first_relevant_rank=first,
    )


def _docs_of(chunk_ids: Sequence[str]) -> list[str]:
    seen: dict[str, None] = {}
    for chunk_id in chunk_ids:
        seen.setdefault(chunk_id.split("::", 1)[0], None)
    return list(seen)


def aggregate(scores: Sequence[RetrievalScores]) -> dict[str, float]:
    """Mean each metric across questions.

    A macro average — every question counts equally — rather than pooling all retrieved
    chunks together. With 28 answerable questions, pooling would let the few questions
    with four gold chunks dominate the result.
    """
    if not scores:
        return {}
    n = len(scores)
    return {
        "precision_at_k": sum(s.precision for s in scores) / n,
        "recall_at_k": sum(s.recall for s in scores) / n,
        "r_precision": sum(s.r_precision for s in scores) / n,
        "average_precision": sum(s.average_precision for s in scores) / n,
        "mrr": sum(s.reciprocal_rank for s in scores) / n,
        "ndcg_at_k": sum(s.ndcg for s in scores) / n,
        "doc_recall": sum(s.doc_recall for s in scores) / n,
        "all_docs_hit_rate": sum(1.0 for s in scores if s.all_docs_hit) / n,
    }
