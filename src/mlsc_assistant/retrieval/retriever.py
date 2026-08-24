"""The retriever: runs the strategies, fuses, diversifies, and explains itself.

Pipeline for ``hybrid``::

    query ─┬─▶ dense  (cosine over the chunk matrix)  ─┐
           └─▶ lexical (BM25 over stemmed tokens)     ─┤
                                                       ├─▶ RRF ─▶ per-doc cap ─▶ top_k
                                                      ─┘

``dense`` and ``lexical`` run the same pipeline with one arm disabled, so the evaluation
can *demonstrate* that fusion beats its components rather than asserting it.

Every result carries where it came from — each retriever's independent rank and score,
the fused score, and which query terms matched. That is what makes a wrong answer
diagnosable: it separates "retrieval never found it" from "the model mishandled good
context" without re-running anything.
"""

from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter

from mlsc_assistant.core.models import Chunk, RetrievalResult, RetrievalStrategy, ScoredChunk
from mlsc_assistant.core.ports import Embedder, Reranker, VectorStore
from mlsc_assistant.retrieval.dense import DenseRetriever
from mlsc_assistant.retrieval.diversify import cap_per_document
from mlsc_assistant.retrieval.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from mlsc_assistant.retrieval.lexical import BM25Retriever


class HybridRetriever:
    """Orchestrates dense and lexical retrieval. Constructed from a loaded store."""

    def __init__(
        self,
        *,
        embedder: Embedder,
        store: VectorStore,
        strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
        top_k: int = 6,
        candidate_k: int = 15,
        rrf_k: int = DEFAULT_RRF_K,
        max_chunks_per_document: int = 3,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        bm25_index_title: bool = True,
        bm25_max_document_frequency: float = 0.5,
        reranker: Reranker | None = None,
    ) -> None:
        chunks = store.all_chunks()
        self.chunks_by_id: dict[str, Chunk] = {c.chunk_id: c for c in chunks}
        self.corpus_size = len(chunks)

        self.dense = DenseRetriever(embedder, store)
        # BM25 builds its index once at construction, not per query. The API holds one
        # retriever for the process lifetime, so per-request rebuilding would be pure waste.
        self.lexical = BM25Retriever(
            chunks,
            k1=bm25_k1,
            b=bm25_b,
            index_title=bm25_index_title,
            max_document_frequency=bm25_max_document_frequency,
        )

        self.strategy = strategy
        self.top_k = top_k
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.max_chunks_per_document = max_chunks_per_document
        self.reranker = reranker

    # -----------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        strategy: RetrievalStrategy | None = None,
        max_chunks_per_document: int | None = None,
    ) -> RetrievalResult:
        strategy = strategy or self.strategy
        top_k = top_k or self.top_k
        max_per_doc = (
            max_chunks_per_document
            if max_chunks_per_document is not None
            else self.max_chunks_per_document
        )
        # Each arm must offer at least top_k candidates or fusion has nothing to work with.
        candidate_k = max(self.candidate_k, top_k)

        timings: dict[str, float] = {}
        dense_hits: list[tuple[Chunk, float]] = []
        lexical_hits: list[tuple[Chunk, float]] = []

        if strategy in (RetrievalStrategy.HYBRID, RetrievalStrategy.DENSE):
            start = perf_counter()
            dense_hits = self.dense.search(query, candidate_k)
            timings["dense"] = (perf_counter() - start) * 1000

        if strategy in (RetrievalStrategy.HYBRID, RetrievalStrategy.LEXICAL):
            start = perf_counter()
            lexical_hits = self.lexical.search(query, candidate_k)
            timings["lexical"] = (perf_counter() - start) * 1000

        dense_rank = {c.chunk_id: i for i, (c, _) in enumerate(dense_hits, start=1)}
        dense_score = {c.chunk_id: s for c, s in dense_hits}
        lexical_rank = {c.chunk_id: i for i, (c, _) in enumerate(lexical_hits, start=1)}
        lexical_score = {c.chunk_id: s for c, s in lexical_hits}

        start = perf_counter()
        ordered_ids, primary_score, rrf_scores = self._order(
            strategy, dense_hits, lexical_hits, dense_score, lexical_score
        )
        timings["fusion"] = (perf_counter() - start) * 1000

        kept = cap_per_document(
            ordered_ids,
            doc_id_of=lambda cid: self.chunks_by_id[cid].doc_id,
            max_per_document=max_per_doc,
            limit=top_k,
        )

        scored = tuple(
            ScoredChunk(
                chunk=self.chunks_by_id[cid],
                score=primary_score[cid],
                rank=rank,
                dense_score=dense_score.get(cid),
                dense_rank=dense_rank.get(cid),
                lexical_score=lexical_score.get(cid),
                lexical_rank=lexical_rank.get(cid),
                rrf_score=rrf_scores.get(cid),
                matched_terms=self.lexical.matched_terms(query, cid),
            )
            for rank, cid in enumerate(kept, start=1)
        )

        return RetrievalResult(
            query=query,
            strategy=strategy,
            chunks=scored,
            candidates_considered=len(set(dense_rank) | set(lexical_rank)),
            timings_ms={k: round(v, 3) for k, v in timings.items()},
        )

    # -----------------------------------------------------------------------

    def _order(
        self,
        strategy: RetrievalStrategy,
        dense_hits: Sequence[tuple[Chunk, float]],
        lexical_hits: Sequence[tuple[Chunk, float]],
        dense_score: dict[str, float],
        lexical_score: dict[str, float],
    ) -> tuple[list[str], dict[str, float], dict[str, float]]:
        """Produce the ranked id list plus the score each result is ordered by."""
        if strategy is RetrievalStrategy.DENSE:
            return [c.chunk_id for c, _ in dense_hits], dict(dense_score), {}

        if strategy is RetrievalStrategy.LEXICAL:
            return [c.chunk_id for c, _ in lexical_hits], dict(lexical_score), {}

        rrf_scores = reciprocal_rank_fusion(
            [[c.chunk_id for c, _ in dense_hits], [c.chunk_id for c, _ in lexical_hits]],
            k=self.rrf_k,
        )
        # Exact RRF ties are common here, since two retrievers over 18 chunks produce a
        # coarse score space. Breaking ties by dense rank (then id) keeps ordering
        # deterministic across runs, which evaluation comparisons depend on.
        ordered = sorted(
            rrf_scores,
            key=lambda cid: (
                -rrf_scores[cid],
                -dense_score.get(cid, 0.0),
                cid,
            ),
        )
        return ordered, rrf_scores, rrf_scores
