"""Evaluation set schema and loading.

Two things this module takes seriously.

**Gold labels are validated against the live index.** A chunk id that does not exist —
a typo, or a label written before the chunker changed — would silently make a question
unanswerable and quietly deflate recall across every run. Failing loudly on load is the
difference between a wrong number and a visible error.

**The format MLSC supplies is not assumed.** ``load_dataset`` reads our schema;
``from_records`` adapts a plain list of question/answer mappings, which is what a handed
-over evaluation set usually looks like. Gold labels are optional there, because an
external set may only provide reference answers.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

from mlsc_assistant.core.errors import InvalidRequestError


class QuestionType(StrEnum):
    DIRECT = "direct"
    MULTI_DOCUMENT = "multi_document"
    REASONING = "reasoning"
    UNANSWERABLE = "unanswerable"
    AMBIGUOUS = "ambiguous"


class UnanswerableKind(StrEnum):
    """Why a question cannot be answered.

    The distinction is load-bearing: ``OFF_DOMAIN`` questions are what a similarity
    threshold can catch, while ``NEAR_MISS`` questions score as high as answerable ones
    and can only be caught by the model reading the context. Reporting abstention
    without this split hides which gate is actually doing the work.
    """

    NEAR_MISS = "near_miss"
    OFF_DOMAIN = "off_domain"


@dataclass(frozen=True, slots=True)
class EvalQuestion:
    id: str
    question: str
    type: QuestionType
    answerable: bool
    gold_documents: tuple[str, ...] = ()
    gold_chunks: tuple[str, ...] = ()
    reference_answer: str = ""
    subtype: UnanswerableKind | None = None
    notes: str = ""

    @property
    def is_multi_document(self) -> bool:
        return len(self.gold_documents) > 1

    @property
    def has_chunk_labels(self) -> bool:
        """Whether chunk-level metrics can be computed for this question.

        An externally supplied set may label documents but not chunks, in which case
        chunk-level metrics are skipped rather than silently scored as zero.
        """
        return bool(self.gold_chunks)


@dataclass(frozen=True, slots=True)
class EvalDataset:
    name: str
    questions: tuple[EvalQuestion, ...]
    description: str = ""
    labelling_rule: str = ""
    source_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.questions)

    def __iter__(self) -> Iterator[EvalQuestion]:
        return iter(self.questions)

    @property
    def answerable(self) -> tuple[EvalQuestion, ...]:
        return tuple(q for q in self.questions if q.answerable)

    @property
    def unanswerable(self) -> tuple[EvalQuestion, ...]:
        return tuple(q for q in self.questions if not q.answerable)

    def by_type(self, question_type: QuestionType) -> tuple[EvalQuestion, ...]:
        return tuple(q for q in self.questions if q.type is question_type)

    def composition(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for q in self.questions:
            counts[q.type.value] = counts.get(q.type.value, 0) + 1
        return dict(sorted(counts.items()))


# ---------------------------------------------------------------------------


def _question_from_mapping(raw: dict[str, Any], index: int) -> EvalQuestion:
    try:
        question_text = str(raw["question"]).strip()
    except KeyError as exc:
        raise InvalidRequestError(f"Question {index} has no 'question' field.") from exc

    question_id = str(raw.get("id") or f"q{index + 1:02d}")
    raw_type = str(raw.get("type", "direct"))
    try:
        question_type = QuestionType(raw_type)
    except ValueError as exc:
        valid = ", ".join(t.value for t in QuestionType)
        raise InvalidRequestError(
            f"Question {question_id!r} has unknown type {raw_type!r}. Expected one of: {valid}."
        ) from exc

    # Default answerability from the type rather than requiring both, so an adapted
    # external set cannot end up with a question typed unanswerable but flagged answerable.
    answerable = bool(raw.get("answerable", question_type is not QuestionType.UNANSWERABLE))

    subtype_raw = raw.get("subtype")
    subtype = UnanswerableKind(str(subtype_raw)) if subtype_raw else None

    if not answerable and (raw.get("gold_chunks") or raw.get("gold_documents")):
        raise InvalidRequestError(
            f"Question {question_id!r} is marked unanswerable but carries gold labels. "
            "An unanswerable question has no correct source passage by definition."
        )
    if answerable and not raw.get("reference_answer"):
        raise InvalidRequestError(
            f"Question {question_id!r} is answerable but has no reference_answer, so "
            "answer correctness cannot be scored for it."
        )

    return EvalQuestion(
        id=question_id,
        question=question_text,
        type=question_type,
        answerable=answerable,
        gold_documents=tuple(raw.get("gold_documents") or ()),
        gold_chunks=tuple(raw.get("gold_chunks") or ()),
        reference_answer=" ".join(str(raw.get("reference_answer", "")).split()),
        subtype=subtype,
        notes=" ".join(str(raw.get("notes", "")).split()),
    )


def from_records(records: Sequence[dict[str, Any]], *, name: str = "external") -> EvalDataset:
    """Adapt a plain list of question mappings.

    The entry point for whatever format MLSC supplies. Anything beyond ``question`` is
    optional; missing gold labels mean retrieval metrics are skipped for that question
    rather than scored as failures.
    """
    questions = tuple(_question_from_mapping(r, i) for i, r in enumerate(records))
    _assert_unique_ids(questions)
    return EvalDataset(name=name, questions=questions)


def load_dataset(path: Path) -> EvalDataset:
    if not path.is_file():
        raise InvalidRequestError(
            f"Evaluation set not found: {path}. "
            "Check `evaluation.dataset_dir` and the dataset name."
        )

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict) or "questions" not in raw:
        raise InvalidRequestError(f"{path} must be a mapping containing a 'questions' list.")

    questions = tuple(_question_from_mapping(r, i) for i, r in enumerate(raw["questions"]))
    _assert_unique_ids(questions)

    return EvalDataset(
        name=str(raw.get("name") or path.stem),
        questions=questions,
        description=" ".join(str(raw.get("description", "")).split()),
        labelling_rule=" ".join(str(raw.get("labelling_rule", "")).split()),
        source_path=path,
        metadata={"version": raw.get("version")},
    )


def resolve_dataset(dataset_dir: Path, name: str) -> EvalDataset:
    """Load by bare name, accepting either extension."""
    for suffix in (".yaml", ".yml"):
        candidate = dataset_dir / f"{name}{suffix}"
        if candidate.is_file():
            return load_dataset(candidate)
    available = sorted(p.stem for p in dataset_dir.glob("*.y*ml")) if dataset_dir.is_dir() else []
    raise InvalidRequestError(
        f"No evaluation set named {name!r} in {dataset_dir}. "
        f"Available: {', '.join(available) or 'none'}."
    )


def _assert_unique_ids(questions: Sequence[EvalQuestion]) -> None:
    seen: set[str] = set()
    for q in questions:
        if q.id in seen:
            raise InvalidRequestError(
                f"Duplicate question id {q.id!r}. Ids key per-question traces, so they "
                "must be unique or results become impossible to attribute."
            )
        seen.add(q.id)


def validate_against_index(
    dataset: EvalDataset, known_chunk_ids: Iterable[str], known_doc_ids: Iterable[str]
) -> None:
    """Fail loudly if a gold label does not exist in the current index.

    Without this, a stale or mistyped label makes a question permanently unrecoverable
    and drags recall down for a reason nobody can see in the report.
    """
    chunks = set(known_chunk_ids)
    docs = set(known_doc_ids)
    problems: list[str] = []

    for q in dataset.questions:
        for chunk_id in q.gold_chunks:
            if chunk_id not in chunks:
                problems.append(f"{q.id}: gold chunk {chunk_id!r} is not in the index")
        for doc_id in q.gold_documents:
            if doc_id not in docs:
                problems.append(f"{q.id}: gold document {doc_id!r} is not in the index")
        # A chunk id implies its document; catching the mismatch here beats debugging a
        # document-recall number that disagrees with chunk recall for no visible reason.
        implied = {c.split("::", 1)[0] for c in q.gold_chunks}
        missing = implied - set(q.gold_documents)
        if missing:
            problems.append(
                f"{q.id}: gold chunks imply document(s) {sorted(missing)} "
                "that are not listed in gold_documents"
            )

    if problems:
        listed = "\n  ".join(problems)
        raise InvalidRequestError(
            f"Evaluation set {dataset.name!r} does not match the current index:\n  {listed}\n"
            "Rebuild the index with `mlsc index`, or correct the labels."
        )
