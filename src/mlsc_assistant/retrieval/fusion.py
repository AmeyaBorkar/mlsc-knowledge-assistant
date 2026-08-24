"""Reciprocal Rank Fusion.

    RRF(d) = sum over retrievers of  1 / (k + rank(d))

Chosen over a weighted score blend because cosine similarity and BM25 scores live on
incompatible, corpus-dependent scales: BM25 is unbounded and depends on corpus
statistics, while this embedder's cosine scores are compressed into roughly [0.6, 1.0]
(two entirely unrelated passages here score 0.65). Any fixed ``alpha * dense +
(1-alpha) * bm25`` would need re-tuning whenever the corpus changes, and an 18-chunk
corpus cannot honestly support fitting that weight.

RRF consumes only ranks, so it is scale-free with one interpretable constant
(DECISIONS.md D1). The cost is that it discards score *magnitude* — which is why the
abstention gate reads raw dense scores rather than fused ones.
"""

from __future__ import annotations

from collections.abc import Sequence

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], *, k: int = DEFAULT_RRF_K
) -> dict[str, float]:
    """Fuse ranked id lists into one score map.

    ``k`` damps the influence of top ranks: at k=60 the gap between rank 1 and rank 2 is
    small, so a document both retrievers rank highly outranks one that a single
    retriever puts first. That is the property we want — agreement should beat a single
    confident opinion.

    Empty rankings contribute nothing, so a strategy that ran only one retriever still
    produces sensible output.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for position, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + position)
    return scores
