"""Evaluation metric tests.

The metrics are hand-implemented precisely so they can be justified (DECISIONS.md D7),
which means they have to be verifiable by hand too. Expected values below are worked out
in the assertions rather than copied from a run.
"""

from __future__ import annotations

import pytest

from mlsc_assistant.evaluation.metrics.abstention import (
    AbstentionOutcome,
    best_operating_point,
    score_abstention,
    sweep_threshold,
)
from mlsc_assistant.evaluation.metrics.retrieval import (
    aggregate,
    average_precision,
    document_recall,
    ndcg_at_k,
    precision_at_k,
    r_precision,
    recall_at_k,
    reciprocal_rank,
    score_question,
)

# --- retrieval --------------------------------------------------------------


def test_precision_counts_only_gold_chunks() -> None:
    assert precision_at_k(["a", "b", "c", "d"], ["a", "c"], 4) == 0.5


def test_precision_divides_by_what_was_returned() -> None:
    """An 18-chunk corpus can legitimately return fewer than k.

    Dividing by k would penalise the retriever for the corpus being small rather than
    for ranking badly.
    """
    assert precision_at_k(["a", "b"], ["a"], 10) == 0.5


def test_precision_of_nothing_is_zero() -> None:
    assert precision_at_k([], ["a"], 5) == 0.0


def test_recall_is_the_share_of_gold_found() -> None:
    assert recall_at_k(["a", "x", "y"], ["a", "b"], 3) == 0.5
    assert recall_at_k(["a", "b"], ["a", "b"], 3) == 1.0


def test_recall_respects_the_cutoff() -> None:
    """A gold chunk below k was not in the context window and does not count."""
    assert recall_at_k(["x", "y", "a"], ["a"], 2) == 0.0


def test_r_precision_is_not_capped_by_gold_set_size() -> None:
    """Why it is reported alongside precision@k.

    With one gold chunk and k=6, precision@6 cannot exceed 0.167 — an artefact of the
    labelling, not a retrieval failure. R-precision reaches 1.0 for a perfect result
    regardless of how many gold chunks there are.
    """
    assert precision_at_k(["a", "x", "y", "z", "w", "v"], ["a"], 6) == pytest.approx(1 / 6)
    assert r_precision(["a", "x", "y", "z", "w", "v"], ["a"]) == 1.0


def test_average_precision_rewards_early_relevance() -> None:
    """Both lists contain the same gold chunks; only the ordering differs."""
    early = average_precision(["a", "b", "x", "y"], ["a", "b"], 4)
    late = average_precision(["x", "y", "a", "b"], ["a", "b"], 4)
    assert early == 1.0
    assert late < early


def test_average_precision_matches_hand_calculation() -> None:
    # gold at ranks 1 and 3: (1/1 + 2/3) / 2
    assert average_precision(["a", "x", "b"], ["a", "b"], 3) == pytest.approx((1 + 2 / 3) / 2)


def test_reciprocal_rank_uses_the_first_hit() -> None:
    assert reciprocal_rank(["x", "a", "b"], ["a", "b"]) == 0.5
    assert reciprocal_rank(["x", "y"], ["a"]) == 0.0


def test_ndcg_is_one_for_a_perfect_ranking() -> None:
    assert ndcg_at_k(["a", "b", "x"], ["a", "b"], 3) == pytest.approx(1.0)


def test_ndcg_discounts_by_rank() -> None:
    assert ndcg_at_k(["x", "a"], ["a"], 2) < ndcg_at_k(["a", "x"], ["a"], 2)


def test_ndcg_ideal_accounts_for_k_smaller_than_gold() -> None:
    """With more gold chunks than slots, retrieving every slot correctly is still 1.0."""
    assert ndcg_at_k(["a", "b"], ["a", "b", "c", "d"], 2) == pytest.approx(1.0)


def test_metrics_are_zero_without_gold() -> None:
    """Unanswerable questions are excluded from retrieval scoring rather than scored;
    these guards ensure a labelling mistake cannot silently produce a division error."""
    assert recall_at_k(["a"], [], 5) == 0.0
    assert ndcg_at_k(["a"], [], 5) == 0.0
    assert average_precision(["a"], [], 5) == 0.0
    assert document_recall(["d"], []) == 0.0


def test_document_recall_is_measured_over_documents() -> None:
    assert document_recall(["a", "b"], ["a", "c"]) == 0.5


def test_score_question_derives_documents_from_chunk_ids() -> None:
    scored = score_question(
        retrieved_chunks=["docA::c01", "docB::c00", "docA::c02"],
        retrieved_docs=[],
        gold_chunks=["docA::c01"],
        gold_docs=["docA"],
        k=3,
    )
    assert scored.recall == 1.0
    assert scored.all_docs_hit
    assert scored.first_relevant_rank == 1
    assert scored.doc_recall == 1.0


def test_all_docs_hit_requires_every_gold_document() -> None:
    """Multi-document coverage is strict on purpose: half the answer is still wrong."""
    scored = score_question(
        retrieved_chunks=["docA::c01", "docA::c02"],
        retrieved_docs=[],
        gold_chunks=["docA::c01", "docB::c00"],
        gold_docs=["docA", "docB"],
        k=6,
    )
    assert not scored.all_docs_hit
    assert scored.recall == 0.5


def test_aggregate_is_a_macro_average() -> None:
    """Every question counts equally.

    Pooling retrieved chunks instead would let the few questions with four gold chunks
    dominate the headline number.
    """
    perfect = score_question(["a"], [], ["a"], ["a"], 1)
    missed = score_question(["x"], [], ["a"], ["a"], 1)
    assert aggregate([perfect, missed])["recall_at_k"] == 0.5


def test_aggregate_of_nothing_is_empty() -> None:
    assert aggregate([]) == {}


# --- abstention -------------------------------------------------------------


def _outcome(qid, answerable, abstained, subtype=None, score=None):  # type: ignore[no-untyped-def]
    return AbstentionOutcome(qid, answerable, abstained, subtype, score)


def test_refusing_everything_scores_perfect_recall_and_terrible_over_refusal() -> None:
    """The failure mode this metric family exists to expose.

    A system that refuses every question is perfectly faithful and completely useless,
    which is why hallucination rate and over-refusal rate are always reported together.
    """
    outcomes = [
        _outcome("a", answerable=True, abstained=True),
        _outcome("b", answerable=False, abstained=True),
    ]
    m = score_abstention(outcomes)
    assert m.recall == 1.0
    assert m.hallucination_rate == 0.0
    assert m.over_refusal_rate == 1.0
    assert m.precision == 0.5


def test_answering_everything_hides_behind_zero_over_refusal() -> None:
    outcomes = [
        _outcome("a", answerable=True, abstained=False),
        _outcome("b", answerable=False, abstained=False),
    ]
    m = score_abstention(outcomes)
    assert m.over_refusal_rate == 0.0
    assert m.hallucination_rate == 1.0
    assert m.recall == 0.0


def test_perfect_abstention() -> None:
    outcomes = [
        _outcome("a", answerable=True, abstained=False),
        _outcome("b", answerable=False, abstained=True),
    ]
    m = score_abstention(outcomes)
    assert (m.precision, m.recall, m.f1) == (1.0, 1.0, 1.0)
    assert m.hallucination_rate == 0.0
    assert m.over_refusal_rate == 0.0


def test_sweep_refuses_below_the_threshold() -> None:
    outcomes = [
        _outcome("low", answerable=False, abstained=False, subtype="off_domain", score=0.40),
        _outcome("high", answerable=True, abstained=False, score=0.90),
    ]
    at_half = next(p for p in sweep_threshold(outcomes, [0.5]) if p.threshold == 0.5)
    assert at_half.metrics.recall == 1.0
    assert at_half.metrics.over_refusal_rate == 0.0
    assert at_half.off_domain_recall == 1.0


def test_sweep_treats_a_missing_score_as_no_evidence() -> None:
    """Retrieval returning nothing means there was nothing to answer from."""
    outcomes = [_outcome("none", answerable=False, abstained=False, score=None)]
    assert sweep_threshold(outcomes, [0.0])[0].metrics.recall == 1.0


def test_sweep_separates_near_miss_from_off_domain() -> None:
    """The split that makes a threshold's real ceiling visible.

    An aggregate recall of 0.5 here would look reasonable while the mechanism catches
    none of the hard cases.
    """
    outcomes = [
        _outcome("od", answerable=False, abstained=False, subtype="off_domain", score=0.40),
        _outcome("nm", answerable=False, abstained=False, subtype="near_miss", score=0.75),
    ]
    point = sweep_threshold(outcomes, [0.55])[0]
    assert point.off_domain_recall == 1.0
    assert point.near_miss_recall == 0.0
    assert point.metrics.recall == 0.5


def test_best_operating_point_respects_the_over_refusal_budget() -> None:
    outcomes = [
        _outcome("ans", answerable=True, abstained=False, score=0.70),
        _outcome("un", answerable=False, abstained=False, subtype="off_domain", score=0.40),
    ]
    points = sweep_threshold(outcomes, [0.5, 0.8])
    best = best_operating_point(points, max_over_refusal=0.0)
    assert best is not None
    assert best.threshold == 0.5, "0.8 would refuse the answerable question too"


def test_best_operating_point_returns_none_when_distributions_overlap() -> None:
    """A real outcome worth reporting, not a reason to relax the constraint."""
    outcomes = [
        _outcome("ans", answerable=True, abstained=False, score=0.70),
        _outcome("un", answerable=False, abstained=False, subtype="near_miss", score=0.75),
    ]
    points = sweep_threshold(outcomes, [0.72])
    assert best_operating_point(points, max_over_refusal=0.0) is None
