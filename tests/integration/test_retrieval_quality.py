"""Retrieval quality against the real corpus and the real embedding model.

No API key required — this is the point of D4, and it is why CI runs these.

These are *characterisation* tests: they pin measured behaviour so that a regression, or
a shift in the embedding model, shows up as a failure rather than as a quietly worse
answer. They are not a substitute for the evaluation harness in Phase 3, which measures
the same properties across a real question set instead of a handful of chosen examples.
"""

from __future__ import annotations

import pytest

from mlsc_assistant.config import Settings
from mlsc_assistant.core.models import RetrievalStrategy
from mlsc_assistant.factories import make_retriever

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture(scope="module")
def retriever():  # type: ignore[no-untyped-def]
    return make_retriever(Settings())


def ranks(retriever, query, strategy, k=6):  # type: ignore[no-untyped-def]
    result = retriever.retrieve(query, strategy=strategy, top_k=k)
    return [sc.chunk.chunk_id for sc in result.chunks]


# --- the questions the brief itself names ------------------------------------


def test_domain_list_ranks_first_for_the_briefs_example(retriever) -> None:  # type: ignore[no-untyped-def]
    """ "What technical domains exist in MLSC?" must surface the chunk listing all five.

    Phase 2 could only assert presence, because fusing a noisy BM25 opinion pushed this
    to rank 3. The low-IDF filter removed that noise — the query's only surviving terms
    are `domain` (at exactly 50% document frequency, so IDF is zero) and `exist` (absent
    from the corpus), so BM25 correctly abstains and hybrid defers to dense.
    """
    assert ranks(retriever, "What technical domains exist in MLSC?", None)[0] == "domains::c01"


def test_rare_exact_term_is_found(retriever) -> None:  # type: ignore[no-untyped-def]
    """`Web3` is the term hybrid retrieval exists for: rare, discriminative, and the
    kind of token a small dense embedder blurs into neighbouring concepts."""
    assert ranks(retriever, "Web3", None)[0].startswith("domains::")


def test_synonym_gap_is_bridged_by_dense_retrieval(retriever) -> None:  # type: ignore[no-untyped-def]
    """The corpus says "evaluated"; the question says "judged".

    These stem differently, so BM25 alone cannot connect them on wording — the
    embedding is what closes the gap.
    """
    assert "hackathons::c01" in ranks(retriever, "How are hackathon projects judged?", None)


# --- hybrid versus its components --------------------------------------------


def test_lexical_rescues_a_dense_near_miss(retriever) -> None:  # type: ignore[no-untyped-def]
    """The clearest measured case for fusion.

    "Each technical domain has two domain leads." merged into leadership::c00. Dense
    retrieval ranks it fifth — nearly out of the window — because the phrasing is
    generic; BM25 ranks it first on the exact terms. Hybrid lands in between.

    If this ever inverts, D1 needs revisiting.
    """
    query = "How many leads does each domain have?"
    gold = "leadership::c00"

    dense = ranks(retriever, query, RetrievalStrategy.DENSE)
    lexical = ranks(retriever, query, RetrievalStrategy.LEXICAL)
    hybrid = ranks(retriever, query, RetrievalStrategy.HYBRID)

    assert lexical.index(gold) < dense.index(gold), "BM25 should beat dense here"
    assert hybrid.index(gold) <= dense.index(gold), "fusion should not lose ground to dense"


def test_all_three_strategies_run_and_differ(retriever) -> None:  # type: ignore[no-untyped-def]
    """The ablation needs three genuinely distinct strategies, not three aliases."""
    query = "How many domain leads does each technical domain have?"
    dense = ranks(retriever, query, RetrievalStrategy.DENSE)
    lexical = ranks(retriever, query, RetrievalStrategy.LEXICAL)
    assert dense and lexical
    assert dense != lexical


def test_lexical_abstains_when_no_query_term_discriminates(retriever) -> None:
    """ "What is MLSC?" reduces to a single term present in all 18 chunks.

    BM25 would then rank by little more than chunk length. Measured: fusing that opinion
    pushed the correct chunk out of the results entirely on a question dense answered
    perfectly. Abstaining is the correct behaviour, and hybrid falls back to dense.
    """
    query = "What is MLSC?"
    assert ranks(retriever, query, RetrievalStrategy.LEXICAL) == []
    assert ranks(retriever, query, RetrievalStrategy.HYBRID) == ranks(
        retriever, query, RetrievalStrategy.DENSE
    )
    assert "about_mlsc::c00" in ranks(retriever, query, RetrievalStrategy.HYBRID)


def test_only_the_requested_arm_runs(retriever) -> None:  # type: ignore[no-untyped-def]
    """A dense-only run must not pay for BM25, and vice versa."""
    dense_only = retriever.retrieve("domains", strategy=RetrievalStrategy.DENSE)
    lexical_only = retriever.retrieve("domains", strategy=RetrievalStrategy.LEXICAL)

    assert "lexical" not in dense_only.timings_ms
    assert "dense" not in lexical_only.timings_ms
    assert lexical_only.top_dense_score is None, "no dense scores when dense did not run"


# --- the numbers the abstention gate will be calibrated against ---------------


def test_off_domain_questions_score_far_below_on_topic_ones(retriever) -> None:  # type: ignore[no-untyped-def]
    """Gate 1's entire basis, measured.

    A question about cricket has nothing in this corpus, and its best cosine sits well
    below anything genuinely on-topic. This gap is what a calibrated threshold exploits.
    """
    off_domain = retriever.retrieve("Who won the IPL final in 2024?").top_dense_score
    on_topic = retriever.retrieve("What technical domains exist in MLSC?").top_dense_score

    assert off_domain is not None and on_topic is not None
    assert off_domain < 0.55 < on_topic
    assert on_topic - off_domain > 0.3


def test_near_miss_unanswerables_score_as_high_as_real_hits(retriever) -> None:  # type: ignore[no-untyped-def]
    """Why gate 1 alone cannot be the abstention mechanism.

    The knowledge base describes the Technical Head role in detail and never names the
    person holding it. Retrieval is *correct* and scores high; only the model reading
    the context can tell that the answer is not in it. Same for a membership fee, which
    the membership document never mentions.

    Any threshold high enough to refuse these would refuse most answerable questions
    too — which is precisely the trade-off `mlsc calibrate` will quantify.
    """
    technical_head = retriever.retrieve("Who is the current Technical Head of MLSC?")
    fee = retriever.retrieve("What is the MLSC membership fee?")
    answerable = retriever.retrieve("What are the responsibilities of a domain lead?")

    assert technical_head.top_dense_score is not None
    assert fee.top_dense_score is not None
    assert answerable.top_dense_score is not None

    assert technical_head.top_dense_score > 0.7
    assert fee.top_dense_score > 0.7
    # Overlapping with answerable questions is the whole point of the finding.
    assert abs(technical_head.top_dense_score - answerable.top_dense_score) < 0.15


# --- diversification ---------------------------------------------------------


def test_multi_document_question_draws_on_multiple_documents(retriever) -> None:  # type: ignore[no-untyped-def]
    result = retriever.retrieve("How do domain leads relate to hackathons?", top_k=6)
    assert len(result.documents_represented) >= 2


def test_diversification_caps_a_dominant_document(retriever) -> None:  # type: ignore[no-untyped-def]
    """With the cap lifted, one document can take every slot; with it, it cannot."""
    query = "What is expected of someone leading a technical domain?"

    capped = retriever.retrieve(query, top_k=6, max_chunks_per_document=2)
    per_doc: dict[str, int] = {}
    for sc in capped.chunks:
        per_doc[sc.doc_id] = per_doc.get(sc.doc_id, 0) + 1
    assert max(per_doc.values()) <= 2

    uncapped = retriever.retrieve(query, top_k=6, max_chunks_per_document=retriever.corpus_size)
    assert len(uncapped.documents_represented) <= len(capped.documents_represented)


# --- contract ----------------------------------------------------------------


def test_results_carry_full_provenance(retriever) -> None:  # type: ignore[no-untyped-def]
    """Every result must explain itself, since that is what makes a bad answer
    diagnosable without re-running anything."""
    result = retriever.retrieve("hackathon judging criteria", top_k=3)
    top = result.chunks[0]

    assert top.rank == 1
    assert top.rrf_score is not None
    assert top.dense_rank is not None or top.lexical_rank is not None
    assert result.candidates_considered > 0
    assert {"dense", "lexical", "fusion"} <= set(result.timings_ms)


def test_retrieval_is_deterministic(retriever) -> None:  # type: ignore[no-untyped-def]
    """Evaluation runs compared weeks apart depend on this."""
    query = "coordinator responsibilities"
    first = ranks(retriever, query, None)
    assert all(ranks(retriever, query, None) == first for _ in range(3))


def test_empty_and_whitespace_queries_do_not_crash(retriever) -> None:  # type: ignore[no-untyped-def]
    for query in ("", "   ", "the of and"):
        result = retriever.retrieve(query, top_k=3)
        assert isinstance(result.chunks, tuple)
