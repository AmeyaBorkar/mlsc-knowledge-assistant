"""Evaluation-set loading, validation and the anti-hard-coding guard."""

from __future__ import annotations

from pathlib import Path

import pytest

from mlsc_assistant.core.errors import InvalidRequestError
from mlsc_assistant.evaluation.dataset import (
    QuestionType,
    UnanswerableKind,
    from_records,
    load_dataset,
    resolve_dataset,
    validate_against_index,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "evaluation" / "datasets"


@pytest.fixture(scope="module")
def dev_set():  # type: ignore[no-untyped-def]
    return load_dataset(DATASET_DIR / "dev_set.yaml")


# --- the committed dev set --------------------------------------------------


def test_dev_set_loads(dev_set) -> None:  # type: ignore[no-untyped-def]
    assert len(dev_set) == 40
    assert dev_set.labelling_rule, "the labelling rule must be recorded with the data"


def test_dev_set_covers_every_question_type(dev_set) -> None:  # type: ignore[no-untyped-def]
    """All five types named in the brief must be represented."""
    assert set(dev_set.composition()) == {t.value for t in QuestionType}


def test_at_least_a_third_of_questions_are_unanswerable(dev_set) -> None:  # type: ignore[no-untyped-def]
    """Abstention is the hardest requirement, so it needs enough samples for its
    metrics to mean anything."""
    assert len(dev_set.unanswerable) / len(dev_set) >= 0.30


def test_unanswerables_are_mostly_near_misses(dev_set) -> None:  # type: ignore[no-untyped-def]
    """A set of only obvious off-domain questions would flatter the system badly.

    Near misses — topic present, fact absent — are the cases no threshold can catch.
    """
    near = [q for q in dev_set.unanswerable if q.subtype is UnanswerableKind.NEAR_MISS]
    assert len(near) > len(dev_set.unanswerable) / 2


def test_answerable_questions_all_carry_gold_and_reference(dev_set) -> None:  # type: ignore[no-untyped-def]
    for q in dev_set.answerable:
        assert q.gold_chunks, f"{q.id} has no gold chunks"
        assert q.gold_documents, f"{q.id} has no gold documents"
        assert q.reference_answer, f"{q.id} has no reference answer"


def test_unanswerable_questions_carry_no_gold(dev_set) -> None:  # type: ignore[no-untyped-def]
    for q in dev_set.unanswerable:
        assert not q.gold_chunks and not q.gold_documents


def test_multi_document_questions_actually_span_documents(dev_set) -> None:  # type: ignore[no-untyped-def]
    """q20 is deliberately typed multi_document while living in one file — it tests
    multi-chunk recall — so the check is that most of them genuinely span documents."""
    multi = dev_set.by_type(QuestionType.MULTI_DOCUMENT)
    spanning = [q for q in multi if q.is_multi_document]
    assert len(spanning) >= len(multi) - 1


def test_gold_chunks_match_the_committed_index(dev_set) -> None:  # type: ignore[no-untyped-def]
    """A mistyped gold id would silently deflate recall on every run forever."""
    import json

    index = REPO_ROOT / "data" / "index" / "chunks.json"
    if not index.is_file():
        pytest.skip("no index built; run `mlsc index`")
    chunks = json.loads(index.read_text(encoding="utf-8"))
    validate_against_index(
        dev_set,
        known_chunk_ids=[c["chunk_id"] for c in chunks],
        known_doc_ids={c["doc_id"] for c in chunks},
    )


def _executable_source(path: Path) -> str:
    """Source with docstrings and comments removed.

    The brief bans hard-coded *answers* — data the system could serve instead of
    retrieving. Prose that happens to quote the corpus while explaining a design
    decision is documentation, not an answer lookup, and the chunker's docstring
    legitimately quotes a sentence to explain why the merge policy changed.

    Stripping every bare string-literal statement covers module, class, function and
    attribute docstrings; ``ast.unparse`` drops comments. What remains is the code that
    could actually produce a response.
    """
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list):
            continue
        kept = [
            stmt
            for stmt in body
            if not (
                isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, str)
            )
        ]
        node.body = kept or [ast.Pass()]  # type: ignore[attr-defined]
    return ast.unparse(ast.fix_missing_locations(tree))


def test_no_reference_answer_leaks_into_the_source_tree(dev_set) -> None:  # type: ignore[no-untyped-def]
    """Enforces the brief's ban on hard-coded answers.

    The only path from a question to an answer must run through retrieval and the model,
    so no reference answer may appear verbatim in executable source under src/.
    """
    sources = " ".join(_executable_source(p) for p in (REPO_ROOT / "src").rglob("*.py"))
    for q in dev_set.answerable:
        sentence = q.reference_answer.split(".")[0].strip()
        if len(sentence) > 40:
            assert sentence not in sources, f"{q.id}'s reference answer appears in src/"


def test_the_leak_guard_would_actually_catch_a_leak(tmp_path: Path) -> None:
    """A guard that cannot fail is not a guard.

    Confirms the docstring-stripping did not defang the check: the same sentence is
    caught in a real assignment and ignored in a docstring.
    """
    secret = "Each technical domain has two domain leads"

    as_code = tmp_path / "leak.py"
    as_code.write_text(f'ANSWERS = {{"q3": "{secret}"}}\n', encoding="utf-8")
    assert secret in _executable_source(as_code)

    as_prose = tmp_path / "prose.py"
    as_prose.write_text(f'"""Explains why: {secret}."""\nX = 1\n', encoding="utf-8")
    assert secret not in _executable_source(as_prose)


# --- loading and validation -------------------------------------------------


def test_resolve_by_name() -> None:
    assert resolve_dataset(DATASET_DIR, "dev_set").name == "dev_set"


def test_unknown_dataset_lists_what_is_available() -> None:
    with pytest.raises(InvalidRequestError, match="Available"):
        resolve_dataset(DATASET_DIR, "no_such_set")


def test_missing_file_names_the_setting(tmp_path: Path) -> None:
    with pytest.raises(InvalidRequestError, match="dataset_dir"):
        load_dataset(tmp_path / "absent.yaml")


def test_records_adapter_defaults_answerability_from_type() -> None:
    """The path an externally supplied evaluation set takes."""
    data = from_records(
        [
            {"question": "Answerable?", "reference_answer": "Yes."},
            {"question": "Not in the KB?", "type": "unanswerable"},
        ]
    )
    assert data.questions[0].answerable
    assert not data.questions[1].answerable
    assert data.questions[0].id == "q01"


def test_unanswerable_with_gold_labels_is_rejected() -> None:
    with pytest.raises(InvalidRequestError, match="no correct source passage"):
        from_records([{"question": "x", "type": "unanswerable", "gold_chunks": ["a::c0"]}])


def test_answerable_without_reference_answer_is_rejected() -> None:
    with pytest.raises(InvalidRequestError, match="reference_answer"):
        from_records([{"question": "x", "type": "direct"}])


def test_unknown_type_lists_the_valid_ones() -> None:
    with pytest.raises(InvalidRequestError, match="Expected one of"):
        from_records([{"question": "x", "type": "trick", "reference_answer": "y"}])


def test_duplicate_ids_are_rejected() -> None:
    with pytest.raises(InvalidRequestError, match="Duplicate question id"):
        from_records(
            [
                {"id": "q1", "question": "a", "reference_answer": "x"},
                {"id": "q1", "question": "b", "reference_answer": "y"},
            ]
        )


def test_validation_catches_a_gold_chunk_not_in_the_index() -> None:
    data = from_records(
        [
            {
                "question": "x",
                "reference_answer": "y",
                "gold_chunks": ["ghost::c99"],
                "gold_documents": ["ghost"],
            }
        ]
    )
    with pytest.raises(InvalidRequestError, match="not in the index"):
        validate_against_index(data, known_chunk_ids=["real::c00"], known_doc_ids=["real"])


def test_validation_catches_chunk_and_document_labels_disagreeing() -> None:
    """A chunk id implies its document; a mismatch would make document recall and chunk
    recall disagree for a reason invisible in the report."""
    data = from_records(
        [
            {
                "question": "x",
                "reference_answer": "y",
                "gold_chunks": ["real::c00"],
                "gold_documents": ["other"],
            }
        ]
    )
    with pytest.raises(InvalidRequestError, match="imply document"):
        validate_against_index(data, known_chunk_ids=["real::c00"], known_doc_ids=["real", "other"])
