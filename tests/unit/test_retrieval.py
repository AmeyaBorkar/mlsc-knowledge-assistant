"""Retrieval unit tests: tokenisation, BM25, RRF and diversification.

These run without the real embedding model. Behaviour against the actual corpus and a
real embedder is covered in ``tests/integration/test_retrieval_quality.py``.
"""

from __future__ import annotations

import pytest

from mlsc_assistant.core.models import Chunk, ChunkKind
from mlsc_assistant.retrieval.diversify import cap_per_document
from mlsc_assistant.retrieval.fusion import reciprocal_rank_fusion
from mlsc_assistant.retrieval.lexical import BM25Retriever, tokenize

# --- tokenisation -----------------------------------------------------------


def test_web3_survives_tokenisation() -> None:
    """The single term that uniquely identifies a domain must not be split.

    ``web`` plus ``3`` would turn a rare, highly discriminative token into a common one
    plus a digit.
    """
    assert "web3" in tokenize("The Web3 domain focuses on blockchain")


def test_stemming_collapses_inflections() -> None:
    """Without this, a query for "hackathon" simply does not match "hackathons"."""
    assert tokenize("hackathons")[0] == tokenize("hackathon")[0]
    assert tokenize("coordinators")[0] == tokenize("coordinating")[0]
    assert tokenize("judged")[0] == tokenize("judging")[0]


def test_stemming_does_not_collapse_genuine_synonyms() -> None:
    """Documents the limit of lexical matching, and why dense retrieval is needed.

    "judged" and "evaluated" mean the same thing here and stem differently, so BM25 can
    never connect the question to the passage. Only the embedding bridges that.
    """
    assert tokenize("judged")[0] != tokenize("evaluated")[0]


def test_stopwords_are_removed() -> None:
    assert tokenize("what are the domains of the community") == tokenize("domains community")


def test_punctuation_and_case_are_normalised() -> None:
    assert tokenize("AI/ML, second-year!") == tokenize("ai ml second year")


def test_stemming_can_be_disabled() -> None:
    """Needed for the stemming ablation."""
    assert tokenize("hackathons", stem=False) == ["hackathons"]


# --- BM25 -------------------------------------------------------------------


def _chunk(chunk_id: str, doc_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        doc_title=f"{doc_id.title()} Doc",
        source_file=f"{doc_id}.txt",
        text=text,
        embed_text=f"{doc_id.title()} Doc - {text}",
        char_range=(0, len(text)),
        index=int(chunk_id[-1]),
        kind=ChunkKind.PARAGRAPH,
        token_estimate=len(text.split()),
        checksum="x",
    )


@pytest.fixture
def corpus() -> list[Chunk]:
    return [
        _chunk("a::c0", "a", "The Web3 domain focuses on blockchain and smart contracts"),
        _chunk("a::c1", "a", "The Cloud Computing domain focuses on deployment and containers"),
        _chunk("b::c0", "b", "Hackathon participants are evaluated on innovation and feasibility"),
        _chunk("b::c1", "b", "Mentors provide technical guidance to participating teams"),
    ]


def test_bm25_ranks_the_exact_term_first(corpus: list[Chunk]) -> None:
    retriever = BM25Retriever(corpus)
    assert retriever.search("Web3 blockchain", 3)[0][0].chunk_id == "a::c0"


def test_bm25_matches_across_inflection(corpus: list[Chunk]) -> None:
    """ "hackathons" (plural) must find a chunk that says "Hackathon" (singular)."""
    assert BM25Retriever(corpus).search("hackathons", 3)[0][0].chunk_id == "b::c0"


def test_bm25_returns_nothing_for_an_unrelated_query(corpus: list[Chunk]) -> None:
    """Chunks scoring zero share no term and are not hits.

    This is what lets the caller distinguish "nothing matched" from "everything matched
    weakly", which the abstention gate depends on.
    """
    assert BM25Retriever(corpus).search("photosynthesis chlorophyll", 5) == []


def test_bm25_handles_an_all_stopword_query(corpus: list[Chunk]) -> None:
    assert BM25Retriever(corpus).search("what is the of", 5) == []


def test_bm25_on_an_empty_corpus_is_not_an_error() -> None:
    assert BM25Retriever([]).search("anything", 3) == []


def test_bm25_ordering_is_deterministic(corpus: list[Chunk]) -> None:
    """Score ties between chunks are common at this corpus size; evaluation comparisons
    across runs depend on ties breaking the same way every time."""
    retriever = BM25Retriever(corpus)
    runs = [[c.chunk_id for c, _ in retriever.search("domain focuses", 4)] for _ in range(5)]
    assert all(r == runs[0] for r in runs)


def test_matched_terms_reports_stems_actually_present(corpus: list[Chunk]) -> None:
    retriever = BM25Retriever(corpus)
    terms = retriever.matched_terms("Web3 blockchain contracts", "a::c0")
    assert set(terms) == {"web3", "blockchain", "contract"}
    assert retriever.matched_terms("Web3", "b::c1") == ()


def test_title_indexing_is_switchable(corpus: list[Chunk]) -> None:
    """The ablation knob must genuinely change what is indexed."""
    with_title = BM25Retriever(corpus, index_title=True)
    without = BM25Retriever(corpus, index_title=False)
    assert with_title.matched_terms("doc", "a::c0") == ("doc",)
    assert without.matched_terms("doc", "a::c0") == ()


# --- RRF --------------------------------------------------------------------


def test_rrf_rewards_agreement_over_a_single_strong_opinion() -> None:
    """The defining property of RRF at k=60.

    ``b`` is second on both lists; ``a`` is first on one and absent from the other.
    Agreement should win, because two independent retrievers both liking something is
    stronger evidence than one liking it a lot.
    """
    scores = reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "d"]], k=60)
    assert scores["b"] > scores["a"]


def test_rrf_ignores_score_magnitude() -> None:
    """Only ranks are consumed, which is exactly why it is scale-free."""
    assert reciprocal_rank_fusion([["x", "y"]], k=60) == pytest.approx({"x": 1 / 61, "y": 1 / 62})


def test_rrf_handles_an_empty_arm() -> None:
    """A single-strategy run must still produce sensible output."""
    scores = reciprocal_rank_fusion([["a", "b"], []], k=60)
    assert scores == pytest.approx({"a": 1 / 61, "b": 1 / 62})


def test_rrf_is_flat_on_a_small_corpus() -> None:
    """Documents a real limitation rather than hiding it.

    k=60 comes from TREC experiments over large candidate pools. Across 18 candidates
    the whole score range spans less than 30%, so RRF barely discriminates between rank
    1 and the tail. `rrf_k` is therefore on the Phase 3 ablation list.
    """
    scores = reciprocal_rank_fusion([[str(i) for i in range(18)]], k=60)
    spread = max(scores.values()) / min(scores.values())
    assert spread < 1.3


# --- diversification --------------------------------------------------------


def test_cap_forces_a_second_document_into_the_window() -> None:
    """The multi-document lever: without the cap, one document takes every slot."""
    ranked = ["a1", "a2", "a3", "a4", "b1"]
    kept = cap_per_document(ranked, doc_id_of=lambda x: x[0], max_per_document=2, limit=3)
    assert kept == ["a1", "a2", "b1"]


def test_cap_backfills_rather_than_returning_less() -> None:
    """On a genuinely single-document question, a strict cap would discard relevant
    context to satisfy a diversity rule nobody asked for."""
    ranked = ["a1", "a2", "a3", "a4"]
    kept = cap_per_document(ranked, doc_id_of=lambda x: x[0], max_per_document=2, limit=4)
    assert kept == ["a1", "a2", "a3", "a4"]


def test_backfill_preserves_relative_order() -> None:
    ranked = ["a1", "a2", "a3", "b1", "a4"]
    kept = cap_per_document(ranked, doc_id_of=lambda x: x[0], max_per_document=1, limit=4)
    assert kept == ["a1", "b1", "a2", "a3"]


def test_cap_at_corpus_size_is_a_no_op() -> None:
    """How `--no-diversify` is implemented."""
    ranked = ["a1", "a2", "a3"]
    assert (
        cap_per_document(ranked, doc_id_of=lambda x: x[0], max_per_document=99, limit=3) == ranked
    )


def test_cap_below_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        cap_per_document(["a1"], doc_id_of=lambda x: x[0], max_per_document=0, limit=1)
