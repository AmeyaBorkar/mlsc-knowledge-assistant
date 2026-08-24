"""Abstention metrics, and the calibration sweep for gate 1.

Treats "should this question be answered?" as binary classification. This family matters
more than any other here, because **a system that refuses everything scores a perfect
1.0 on faithfulness** — only these metrics expose that.

Hallucination rate and over-refusal rate are always reported as a pair. They trade off
directly, and a threshold can be moved to make either look excellent on its own.

What this module measures in Phase 3 is **gate 1 alone** — a threshold on the retrieval
score, no LLM involved, so it runs without an API key. That is deliberately a partial
picture: the sweep quantifies exactly how much of the abstention problem a threshold can
possibly solve, which is the evidence for or against needing gate 2 at all.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AbstentionOutcome:
    """One question's abstention decision against ground truth."""

    question_id: str
    answerable: bool
    abstained: bool
    subtype: str | None = None
    score: float | None = None

    @property
    def correct(self) -> bool:
        return self.abstained != self.answerable


@dataclass(frozen=True, slots=True)
class AbstentionMetrics:
    precision: float
    recall: float
    f1: float
    hallucination_rate: float
    over_refusal_rate: float
    answered_count: int
    abstained_count: int

    def as_dict(self) -> dict[str, float]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "hallucination_rate": self.hallucination_rate,
            "over_refusal_rate": self.over_refusal_rate,
        }


def score_abstention(outcomes: Sequence[AbstentionOutcome]) -> AbstentionMetrics:
    """Refusal is the positive class.

    - true positive: unanswerable, and refused
    - false positive: answerable, but refused (over-refusal)
    - false negative: unanswerable, but answered (hallucination risk)
    """
    refused_correctly = sum(1 for o in outcomes if o.abstained and not o.answerable)
    refused_wrongly = sum(1 for o in outcomes if o.abstained and o.answerable)
    answered_wrongly = sum(1 for o in outcomes if not o.abstained and not o.answerable)

    unanswerable = sum(1 for o in outcomes if not o.answerable)
    answerable = sum(1 for o in outcomes if o.answerable)

    precision = _ratio(refused_correctly, refused_correctly + refused_wrongly)
    recall = _ratio(refused_correctly, unanswerable)
    f1 = _ratio(2 * precision * recall, precision + recall)

    return AbstentionMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        hallucination_rate=_ratio(answered_wrongly, unanswerable),
        over_refusal_rate=_ratio(refused_wrongly, answerable),
        answered_count=sum(1 for o in outcomes if not o.abstained),
        abstained_count=sum(1 for o in outcomes if o.abstained),
    )


@dataclass(frozen=True, slots=True)
class ThresholdPoint:
    threshold: float
    metrics: AbstentionMetrics
    near_miss_recall: float
    """Share of near-miss unanswerables caught. Expected to stay near zero — near
    misses score like answerable questions, which is the whole argument for gate 2."""

    off_domain_recall: float
    """Share of off-domain unanswerables caught. This is what a threshold can do."""


def sweep_threshold(
    outcomes: Sequence[AbstentionOutcome], thresholds: Sequence[float]
) -> list[ThresholdPoint]:
    """Recompute abstention metrics across candidate gate-1 thresholds.

    A question is refused when its best retrieval score falls *below* the threshold.
    Questions with no score (retrieval returned nothing) count as refused, since there
    was nothing to answer from.

    The per-subtype recalls are the point of the sweep: an operating point that looks
    respectable in aggregate while catching zero near misses is not a working abstention
    mechanism, and only the split makes that visible.
    """
    points: list[ThresholdPoint] = []
    for threshold in thresholds:
        decided = tuple(
            AbstentionOutcome(
                question_id=o.question_id,
                answerable=o.answerable,
                abstained=o.score is None or o.score < threshold,
                subtype=o.subtype,
                score=o.score,
            )
            for o in outcomes
        )
        points.append(
            ThresholdPoint(
                threshold=threshold,
                metrics=score_abstention(decided),
                near_miss_recall=_subtype_recall(decided, "near_miss"),
                off_domain_recall=_subtype_recall(decided, "off_domain"),
            )
        )
    return points


def best_operating_point(
    points: Sequence[ThresholdPoint], *, max_over_refusal: float = 0.0
) -> ThresholdPoint | None:
    """Highest-F1 threshold that keeps over-refusal within budget.

    The constraint comes first on purpose. Refusing an answerable question is a visible,
    immediate failure for a user, so the sensible framing is "catch as much as possible
    without refusing real questions" rather than maximising F1 outright.

    Returns None when no threshold satisfies the budget — a real outcome worth reporting
    rather than silently relaxing the constraint.
    """
    eligible = [p for p in points if p.metrics.over_refusal_rate <= max_over_refusal]
    if not eligible:
        return None
    return max(eligible, key=lambda p: (p.metrics.f1, p.threshold))


def _subtype_recall(outcomes: Sequence[AbstentionOutcome], subtype: str) -> float:
    matching = [o for o in outcomes if not o.answerable and o.subtype == subtype]
    if not matching:
        return 0.0
    return sum(1 for o in matching if o.abstained) / len(matching)


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0
