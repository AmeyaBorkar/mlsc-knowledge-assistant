"""Generation metric and judge tests.

The judge is stubbed. These verify the *arithmetic and edge handling* around it — the
part that is mine to get right — rather than whether a real model judges well, which the
human spot-check in docs/EVALUATION.md covers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mlsc_assistant.core.errors import ProviderRateLimitedError
from mlsc_assistant.evaluation.judge import Judge, VerdictCache, cache_key
from mlsc_assistant.evaluation.metrics.generation import (
    aggregate_generation,
    score_answer_correctness,
    score_answer_relevancy,
    score_faithfulness,
)
from mlsc_assistant.generation.providers.base import StructuredResult


class StubProvider:
    name = "stub"
    model = "judge-1"

    def __init__(self, *payloads: dict[str, Any] | Exception) -> None:
        self._queue = list(payloads)
        self.calls: list[dict[str, Any]] = []

    def complete(self, **kwargs: Any) -> Any:  # pragma: no cover - unused
        raise NotImplementedError

    def complete_structured(self, **kwargs: Any) -> StructuredResult:
        self.calls.append(kwargs)
        item = self._queue.pop(0) if self._queue else {}
        if isinstance(item, Exception):
            raise item
        return StructuredResult(data=item, raw_text="", latency_ms=1.0)


def _judge(tmp_path: Path, *payloads: dict[str, Any] | Exception) -> Judge:
    return Judge(
        StubProvider(*payloads),  # type: ignore[arg-type]
        cache=VerdictCache(tmp_path / "cache.json"),
    )


class FakeEmbedder:
    """Returns fixed vectors so relevancy arithmetic is checkable by hand."""

    def __init__(self, query_vec: list[float], doc_vecs: list[list[float]]) -> None:
        self._query = query_vec
        self._docs = doc_vecs

    name = "fake"
    dimension = 2

    def embed_query(self, text: str) -> list[float]:
        return self._query

    def embed_documents(self, texts: Any) -> list[list[float]]:
        return self._docs


# ---------------------------------------------------------------------------
# Faithfulness
# ---------------------------------------------------------------------------


def test_faithfulness_is_claim_level_not_answer_level(tmp_path: Path) -> None:
    """A four-claim answer with one invention should score 0.75, not 0 or 1.

    Answer-level judging cannot express that, which is why the schema decomposes.
    """
    judge = _judge(
        tmp_path,
        {
            "claims": [
                {"claim": "a", "supported": True, "reason": ""},
                {"claim": "b", "supported": True, "reason": ""},
                {"claim": "c", "supported": True, "reason": ""},
                {"claim": "d", "supported": False, "reason": "not in passages"},
            ]
        },
    )
    result = score_faithfulness(judge, question="q", answer="four claims", contexts=["ctx"])

    assert result.score == 0.75
    assert result.unsupported == ("d",)


def test_faithfulness_of_an_empty_answer_is_zero(tmp_path: Path) -> None:
    judge = _judge(tmp_path)
    assert score_faithfulness(judge, question="q", answer="   ", contexts=["ctx"]).score == 0.0
    assert judge.provider.calls == []  # type: ignore[attr-defined]


def test_faithfulness_without_context_is_zero(tmp_path: Path) -> None:
    """An answer with no cited passages has nothing supporting it."""
    judge = _judge(tmp_path)
    assert score_faithfulness(judge, question="q", answer="text", contexts=[]).score == 0.0


def test_a_judge_returning_no_claims_scores_zero_not_one(tmp_path: Path) -> None:
    """Extracting no claims from a real answer is a judge failure, not perfection.

    Scoring 1.0 here would silently inflate the headline metric on exactly the cases
    where the instrument broke.
    """
    judge = _judge(tmp_path, {"claims": []})
    assert score_faithfulness(judge, question="q", answer="real text", contexts=["c"]).score == 0.0


def test_faithfulness_passes_full_passages_to_the_judge(tmp_path: Path) -> None:
    """Regression: passing display snippets made this measure truncation."""
    long_passage = "x" * 400 + " Each technical domain has two domain leads."
    judge = _judge(tmp_path, {"claims": [{"claim": "a", "supported": True, "reason": ""}]})
    score_faithfulness(judge, question="q", answer="a", contexts=[long_passage])

    prompt = judge.provider.calls[0]["prompt"]  # type: ignore[attr-defined]
    assert "two domain leads" in prompt


# ---------------------------------------------------------------------------
# Answer relevancy
# ---------------------------------------------------------------------------


def test_relevancy_is_mean_cosine_to_reverse_engineered_questions(tmp_path: Path) -> None:
    judge = _judge(tmp_path, {"questions": ["q1", "q2"], "noncommittal": False})
    embedder = FakeEmbedder([1.0, 0.0], [[1.0, 0.0], [0.0, 1.0]])

    result = score_answer_relevancy(
        judge,
        embedder,
        question="q",
        answer="a",
        n_questions=2,  # type: ignore[arg-type]
    )
    assert result.score == pytest.approx(0.5), "mean of cosine 1.0 and 0.0"


def test_a_noncommittal_answer_scores_zero(tmp_path: Path) -> None:
    """Catches answers that are grounded and cited but evasive."""
    judge = _judge(tmp_path, {"questions": ["q1"], "noncommittal": True})
    embedder = FakeEmbedder([1.0, 0.0], [[1.0, 0.0]])

    result = score_answer_relevancy(judge, embedder, question="q", answer="hedge")  # type: ignore[arg-type]
    assert result.score == 0.0
    assert result.noncommittal


def test_relevancy_respects_the_question_count(tmp_path: Path) -> None:
    judge = _judge(tmp_path, {"questions": ["a", "b", "c", "d", "e"], "noncommittal": False})
    embedder = FakeEmbedder([1.0, 0.0], [[1.0, 0.0]] * 2)

    result = score_answer_relevancy(
        judge,
        embedder,
        question="q",
        answer="a",
        n_questions=2,  # type: ignore[arg-type]
    )
    assert len(result.generated_questions) == 2


# ---------------------------------------------------------------------------
# Answer correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("verdict", "expected"),
    [("correct", 1.0), ("partially_correct", 0.5), ("incorrect", 0.0)],
)
def test_correctness_maps_verdicts_to_scores(tmp_path: Path, verdict: str, expected: float) -> None:
    judge = _judge(
        tmp_path,
        {"verdict": verdict, "missing": [], "contradictions": [], "reason": "r"},
    )
    result = score_answer_correctness(judge, question="q", answer="a", reference="ref")
    assert result.score == expected


def test_correctness_records_what_was_missing(tmp_path: Path) -> None:
    """Partial credit is only useful if the report says what was omitted."""
    judge = _judge(
        tmp_path,
        {
            "verdict": "partially_correct",
            "missing": ["mentoring coordinators"],
            "contradictions": [],
            "reason": "omits one responsibility",
        },
    )
    result = score_answer_correctness(judge, question="q", answer="a", reference="ref")
    assert result.missing == ("mentoring coordinators",)


def test_correctness_without_a_reference_is_zero(tmp_path: Path) -> None:
    judge = _judge(tmp_path)
    assert score_answer_correctness(judge, question="q", answer="a", reference="").score == 0.0


def test_an_unknown_verdict_does_not_award_credit(tmp_path: Path) -> None:
    judge = _judge(
        tmp_path, {"verdict": "spectacular", "missing": [], "contradictions": [], "reason": ""}
    )
    assert score_answer_correctness(judge, question="q", answer="a", reference="r").score == 0.0


# ---------------------------------------------------------------------------
# Judge caching
# ---------------------------------------------------------------------------


def test_identical_inputs_hit_the_cache(tmp_path: Path) -> None:
    """The reason a report can be re-run against a 20-per-day quota."""
    judge = _judge(tmp_path, {"claims": [{"claim": "a", "supported": True, "reason": ""}]})

    first = score_faithfulness(judge, question="q", answer="a", contexts=["c"])
    second = score_faithfulness(judge, question="q", answer="a", contexts=["c"])

    assert first.score == second.score
    assert judge.calls == 1
    assert judge.cache_hits == 1


def test_the_cache_survives_a_restart(tmp_path: Path) -> None:
    judge = _judge(tmp_path, {"claims": [{"claim": "a", "supported": True, "reason": ""}]})
    score_faithfulness(judge, question="q", answer="a", contexts=["c"])
    judge.cache.flush()

    reopened = Judge(StubProvider(), cache=VerdictCache(tmp_path / "cache.json"))  # type: ignore[arg-type]
    score_faithfulness(reopened, question="q", answer="a", contexts=["c"])
    assert reopened.calls == 0, "a fresh process should not re-pay for judged content"


def test_cache_keys_separate_models() -> None:
    """A verdict from a different model is a different measurement.

    Reusing it silently would make a model comparison compare nothing.
    """
    payload = {"question": "q", "answer": "a"}
    assert cache_key("faithfulness", "model-a", payload) != cache_key(
        "faithfulness", "model-b", payload
    )
    assert cache_key("faithfulness", "m", payload) != cache_key("relevancy", "m", payload)


def test_a_corrupt_cache_degrades_to_a_miss(tmp_path: Path) -> None:
    (tmp_path / "cache.json").write_text("{ not json", encoding="utf-8")
    cache = VerdictCache(tmp_path / "cache.json")
    assert len(cache) == 0


def test_caching_can_be_disabled(tmp_path: Path) -> None:
    cache = VerdictCache(tmp_path / "cache.json", enabled=False)
    cache.put("k", {"v": 1})
    assert cache.get("k") is None


def test_quota_failures_propagate_rather_than_scoring_zero(tmp_path: Path) -> None:
    """A run stopped by quota must be visibly incomplete.

    Swallowing the error would score the remaining questions as bad answers, which is
    the difference between "we ran out of quota" and "the system is broken".
    """
    judge = _judge(tmp_path, ProviderRateLimitedError("quota exhausted"))
    with pytest.raises(ProviderRateLimitedError):
        score_faithfulness(judge, question="q", answer="a", contexts=["c"])


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_aggregate_is_a_macro_average() -> None:
    result = aggregate_generation([{"faithfulness": 1.0}, {"faithfulness": 0.0}])
    assert result["faithfulness"] == 0.5


def test_aggregate_skips_metrics_a_question_did_not_produce() -> None:
    """Answer correctness is only scored where a reference answer exists; questions
    without one must not be counted as zeros."""
    result = aggregate_generation(
        [{"faithfulness": 1.0, "answer_correctness": 1.0}, {"faithfulness": 0.5}]
    )
    assert result["faithfulness"] == 0.75
    assert result["answer_correctness"] == 1.0


def test_aggregate_of_nothing_is_empty() -> None:
    assert aggregate_generation([]) == {}
