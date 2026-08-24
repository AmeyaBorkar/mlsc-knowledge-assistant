"""Generation metrics: faithfulness, answer relevancy, answer correctness.

The three the brief asks for, each measuring something the others cannot:

``faithfulness``
    Is every claim supported by the cited context? **Claim-level, not answer-level** — a
    four-sentence answer with one invented sentence should score 0.75, not 0 or 1, and
    only decomposition gives that resolution. Judged against the *cited* passages rather
    than everything retrieved, which is stricter and matches what the user is shown.

``answer_relevancy``
    Does the answer actually address the question? Computed the RAGAS way and
    deliberately **not** LLM-judged at the scoring step: the model generates hypothetical
    questions the answer would answer, and those are embedded and compared to the real
    question with the fastembed model already loaded. Cheap, and the arithmetic is
    reproducible even though the generated questions are not.

``answer_correctness``
    Does it agree with the reference answer? The evaluation set ships reference answers
    and ignoring them would waste the strongest available signal. Kept separate from
    faithfulness because an answer can be perfectly grounded and still incomplete.

A note on what faithfulness alone would reward: an abstention cites nothing and claims
nothing, so it is trivially faithful. Abstained questions are therefore **excluded** from
these metrics and measured by the abstention family instead. Averaging them in would let
a system that refuses everything post a perfect score.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from mlsc_assistant.core.ports import Embedder
from mlsc_assistant.evaluation.judge import Judge

# ---------------------------------------------------------------------------
# Faithfulness
# ---------------------------------------------------------------------------

FAITHFULNESS_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "claims": {
            "type": "ARRAY",
            "description": "Each atomic factual claim in the answer, with its verdict.",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "claim": {"type": "STRING"},
                    "supported": {"type": "BOOLEAN"},
                    "reason": {"type": "STRING"},
                },
                "required": ["claim", "supported", "reason"],
            },
        }
    },
    "required": ["claims"],
}

FAITHFULNESS_SYSTEM = """\
You decompose an answer into atomic factual claims and judge each against the passages \
provided.

An atomic claim is one assertion. "Domain leads plan the roadmap and mentor coordinators" \
is two claims, not one.

For each claim, decide whether the passages state it or directly entail it. Judge support \
only, never truth in the wider world: an accurate statement the passages do not contain is \
unsupported, and finding those is the entire point.

Paraphrase and reordering are fine. Added specifics are not — a number, name, date or \
qualifier absent from the passages is unsupported however plausible it sounds.\
"""


@dataclass(frozen=True, slots=True)
class FaithfulnessResult:
    score: float
    claims: tuple[dict[str, Any], ...]

    @property
    def unsupported(self) -> tuple[str, ...]:
        return tuple(str(c["claim"]) for c in self.claims if not c.get("supported"))


def score_faithfulness(
    judge: Judge, *, question: str, answer: str, contexts: Sequence[str]
) -> FaithfulnessResult:
    if not answer.strip() or not contexts:
        return FaithfulnessResult(score=0.0, claims=())

    passages = "\n\n".join(contexts)
    verdict = judge.judge(
        task="faithfulness",
        system=FAITHFULNESS_SYSTEM,
        prompt=f"Passages:\n\n{passages}\n\nQuestion: {question}\n\nAnswer:\n{answer}",
        schema=FAITHFULNESS_SCHEMA,
        payload={"question": question, "answer": answer, "contexts": list(contexts)},
    )

    raw = verdict.data.get("claims")
    claims = tuple(c for c in raw if isinstance(c, dict)) if isinstance(raw, list) else ()
    if not claims:
        # No claims extracted from a non-empty answer means the judge failed, not that
        # the answer was perfect. Scoring 1.0 here would quietly inflate the metric.
        return FaithfulnessResult(score=0.0, claims=())

    supported = sum(1 for c in claims if c.get("supported"))
    return FaithfulnessResult(score=supported / len(claims), claims=claims)


# ---------------------------------------------------------------------------
# Answer relevancy
# ---------------------------------------------------------------------------

RELEVANCY_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "questions": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": "Questions this answer would be a direct response to.",
        },
        "noncommittal": {
            "type": "BOOLEAN",
            "description": "True if the answer evades rather than answers.",
        },
    },
    "required": ["questions", "noncommittal"],
}

RELEVANCY_SYSTEM = """\
Given an answer, write the questions it would be a direct and complete response to. \
Base them only on what the answer says — do not use outside knowledge, and do not try to \
guess the original question.

Mark the answer noncommittal if it evades, hedges without content, or says it cannot \
answer.\
"""


@dataclass(frozen=True, slots=True)
class RelevancyResult:
    score: float
    generated_questions: tuple[str, ...]
    noncommittal: bool


def score_answer_relevancy(
    judge: Judge,
    embedder: Embedder,
    *,
    question: str,
    answer: str,
    n_questions: int = 3,
) -> RelevancyResult:
    """Embed reverse-engineered questions and compare them to the real one.

    Catches answers that are true and cited but off-target: if the answer only supports
    questions unlike the one asked, it answered something else.
    """
    if not answer.strip():
        return RelevancyResult(score=0.0, generated_questions=(), noncommittal=True)

    verdict = judge.judge(
        task="relevancy",
        system=RELEVANCY_SYSTEM,
        prompt=f"Write {n_questions} such questions.\n\nAnswer:\n{answer}",
        schema=RELEVANCY_SCHEMA,
        payload={"answer": answer, "n": n_questions},
    )

    raw = verdict.data.get("questions")
    generated = tuple(str(q) for q in raw)[:n_questions] if isinstance(raw, list) else ()
    noncommittal = bool(verdict.data.get("noncommittal"))

    if not generated or noncommittal:
        return RelevancyResult(score=0.0, generated_questions=generated, noncommittal=noncommittal)

    original = embedder.embed_query(question)
    others = embedder.embed_documents(list(generated))
    # Vectors are L2-normalised at embed time, so a dot product is cosine similarity.
    similarities = [sum(a * b for a, b in zip(original, other, strict=True)) for other in others]
    return RelevancyResult(
        score=max(0.0, sum(similarities) / len(similarities)),
        generated_questions=generated,
        noncommittal=False,
    )


# ---------------------------------------------------------------------------
# Answer correctness
# ---------------------------------------------------------------------------

CORRECTNESS_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "verdict": {
            "type": "STRING",
            "enum": ["correct", "partially_correct", "incorrect"],
        },
        "missing": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": "Facts in the reference answer that the answer omits.",
        },
        "contradictions": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": "Statements that conflict with the reference answer.",
        },
        "reason": {"type": "STRING"},
    },
    "required": ["verdict", "missing", "contradictions", "reason"],
}

CORRECTNESS_SYSTEM = """\
Compare an answer against a reference answer and judge factual agreement.

Wording does not matter; facts do. An answer that says the same things differently is \
correct. An answer that omits something the reference states is partially correct. An \
answer that contradicts the reference is incorrect.

Extra detail is not a fault as long as it does not contradict the reference.\
"""

_CORRECTNESS_SCORES = {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}


@dataclass(frozen=True, slots=True)
class CorrectnessResult:
    score: float
    verdict: str
    missing: tuple[str, ...]
    contradictions: tuple[str, ...]
    reason: str


def score_answer_correctness(
    judge: Judge, *, question: str, answer: str, reference: str
) -> CorrectnessResult:
    if not answer.strip() or not reference.strip():
        return CorrectnessResult(0.0, "incorrect", (), (), "No answer or no reference.")

    verdict = judge.judge(
        task="correctness",
        system=CORRECTNESS_SYSTEM,
        prompt=(
            f"Question: {question}\n\nReference answer:\n{reference}\n\nAnswer to judge:\n{answer}"
        ),
        schema=CORRECTNESS_SCHEMA,
        payload={"question": question, "answer": answer, "reference": reference},
    )

    data = verdict.data
    label = str(data.get("verdict", "incorrect"))
    return CorrectnessResult(
        score=_CORRECTNESS_SCORES.get(label, 0.0),
        verdict=label,
        missing=tuple(str(m) for m in data.get("missing", []) or ()),
        contradictions=tuple(str(c) for c in data.get("contradictions", []) or ()),
        reason=str(data.get("reason", "")),
    )


def aggregate_generation(scores: Sequence[dict[str, float]]) -> dict[str, float]:
    """Macro-average each metric over the questions that produced one."""
    if not scores:
        return {}
    keys = {k for s in scores for k in s}
    out: dict[str, float] = {}
    for key in sorted(keys):
        values = [s[key] for s in scores if key in s]
        if values:
            out[key] = sum(values) / len(values)
    return out
