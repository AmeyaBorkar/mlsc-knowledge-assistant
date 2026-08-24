"""Chunker tests.

The list-atomicity tests are the important ones. If ``domains::c01`` ever gets split,
the brief's own example question — "What technical domains exist in MLSC?" — stops
having a single chunk that answers it, and no amount of prompt work recovers that.
"""

from __future__ import annotations

import pytest

from mlsc_assistant.core.models import ChunkKind, Document
from mlsc_assistant.ingestion.chunker import StructuralChunker, chunk_documents, estimate_tokens

DOMAIN_NAMES = [
    "Artificial Intelligence and Machine Learning",
    "Web Development",
    "App Development",
    "Cloud Computing",
    "Web3",
]

LEAD_RESPONSIBILITIES = [
    "Planning the learning roadmap",
    "Assigning tasks to coordinators",
    "Conducting knowledge-sharing sessions",
    "Reviewing technical projects",
    "Mentoring coordinators",
    "Encouraging participation in hackathons",
    "Coordinating cross-domain projects",
]


@pytest.fixture(scope="module")
def chunker() -> StructuralChunker:
    return StructuralChunker()


@pytest.fixture(scope="module")
def chunks(documents, chunker):  # type: ignore[no-untyped-def]
    return chunk_documents(documents, chunker)


# --- list atomicity ---------------------------------------------------------


def test_all_five_domains_live_in_one_chunk(chunks) -> None:  # type: ignore[no-untyped-def]
    """The domain list must never be split across chunks."""
    matches = [c for c in chunks if all(name in c.text for name in DOMAIN_NAMES)]
    assert len(matches) == 1, (
        "Expected exactly one chunk containing all five domains; "
        f"found {len(matches)}. Splitting this list breaks 'What domains exist in MLSC?'"
    )
    assert matches[0].kind is ChunkKind.LIST_BLOCK


def test_domain_list_keeps_its_introducing_sentence(chunks) -> None:  # type: ignore[no-untyped-def]
    """Without the lead-in, the chunk is an unattributed list of bare nouns."""
    chunk = next(c for c in chunks if "Web3" in c.text and c.kind is ChunkKind.LIST_BLOCK)
    assert "The major domains include:" in chunk.text


def test_all_lead_responsibilities_live_in_one_chunk(chunks) -> None:  # type: ignore[no-untyped-def]
    matches = [c for c in chunks if all(r in c.text for r in LEAD_RESPONSIBILITIES)]
    assert len(matches) == 1
    assert matches[0].kind is ChunkKind.LIST_BLOCK
    assert "Domain leads are responsible for:" in matches[0].text


# --- merge policy -----------------------------------------------------------


def test_no_document_collapses_into_a_single_chunk(chunks, documents) -> None:  # type: ignore[no-untyped-def]
    """Regression: an earlier bidirectional merge cascaded.

    Because every paragraph in this corpus is individually under the token floor, a
    rule that merged whenever *either* side was short absorbed entire documents into one
    chunk. The accumulator stops growing once the buffer reaches the floor.
    """
    multi_paragraph = {"about_mlsc", "domains", "leadership", "membership"}
    for doc in documents:
        if doc.doc_id in multi_paragraph:
            count = len([c for c in chunks if c.doc_id == doc.doc_id])
            assert count > 1, f"{doc.doc_id} collapsed to a single chunk"


def test_merging_reaches_the_token_floor_where_possible(chunks) -> None:  # type: ignore[no-untyped-def]
    """Only genuinely unmergeable blocks may sit under the floor.

    ``domains::c00`` is the one: a document's opening paragraph followed immediately by
    a list block, so it has no mergeable neighbour on either side.
    """
    floor = StructuralChunker().min_tokens
    undersized = [
        c.chunk_id
        for c in chunks
        if c.kind is not ChunkKind.LIST_BLOCK and c.token_estimate < floor
    ]
    assert undersized == ["domains::c00"]


def test_chunks_stay_under_the_max(chunks) -> None:  # type: ignore[no-untyped-def]
    limit = StructuralChunker().max_tokens
    assert all(c.token_estimate <= limit for c in chunks)


# --- identity and offsets ---------------------------------------------------


def test_chunk_ids_are_deterministic_and_unique(chunks) -> None:  # type: ignore[no-untyped-def]
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert all(c.chunk_id == f"{c.doc_id}::c{c.index:02d}" for c in chunks)


def test_chunking_is_reproducible(documents, chunker) -> None:  # type: ignore[no-untyped-def]
    """Same input, same chunk ids and text — otherwise eval sets rot between runs."""
    first = chunk_documents(documents, chunker)
    second = chunk_documents(documents, chunker)
    assert [(c.chunk_id, c.checksum) for c in first] == [(c.chunk_id, c.checksum) for c in second]


def test_char_ranges_resolve_back_to_the_source(documents, chunks) -> None:  # type: ignore[no-untyped-def]
    """Every citation offset must locate real text in the original document.

    Merged chunks join paragraphs with a newline, so the slice is compared on
    whitespace-normalised content rather than byte-for-byte.
    """
    by_id = {d.doc_id: d for d in documents}
    for chunk in chunks:
        start, end = chunk.char_range
        source = by_id[chunk.doc_id].text[start:end]
        assert " ".join(source.split()) == " ".join(chunk.text.split()), chunk.chunk_id


def test_title_line_is_not_indexed_on_its_own(chunks, documents) -> None:  # type: ignore[no-untyped-def]
    """The title is prepended to every chunk, so indexing it alone adds a
    topic-matching chunk with no facts in it."""
    titles = {d.title for d in documents}
    assert not [c for c in chunks if c.text.strip() in titles]


# --- contextual headers -----------------------------------------------------


def test_embed_text_carries_the_document_title(chunks) -> None:  # type: ignore[no-untyped-def]
    """The fix for pronoun-headed paragraphs ("Each domain has domain leads...")."""
    for chunk in chunks:
        assert chunk.embed_text.startswith(f"{chunk.doc_title} - ")
        assert chunk.text in chunk.embed_text


def test_stored_text_stays_clean(chunks) -> None:  # type: ignore[no-untyped-def]
    """Displayed text and citation snippets must not carry the embedding prefix."""
    assert not [c for c in chunks if c.text.startswith(f"{c.doc_title} - ")]


def test_title_prefix_can_be_disabled(documents) -> None:  # type: ignore[no-untyped-def]
    """Needed for the contextual-header ablation in docs/EVALUATION.md."""
    plain = StructuralChunker(prepend_doc_title=False)
    chunks = chunk_documents(documents, plain)
    assert all(c.embed_text == c.text for c in chunks)


# --- synthetic edge cases ---------------------------------------------------


def _doc(body: str) -> Document:
    text = f"Test Title\n\n{body}"
    return Document(
        doc_id="test",
        title="Test Title",
        text=text,
        source_file="test.txt",
        checksum=Document.compute_checksum(text),
    )


def test_numbered_and_bulleted_lists_are_both_recognised() -> None:
    for marker in ("1.", "1)", "-", "*", "•"):
        body = f"Things to note:\n\n{marker} first item here\n{marker} second item here\n"
        chunks = StructuralChunker().chunk(_doc(body))
        kinds = [c.kind for c in chunks]
        assert ChunkKind.LIST_BLOCK in kinds, f"marker {marker!r} not detected"


def test_colon_line_without_a_list_is_a_plain_paragraph() -> None:
    body = "This sentence ends with a colon:\n\nAnd this is ordinary prose that follows it.\n"
    chunks = StructuralChunker().chunk(_doc(body))
    assert all(c.kind is not ChunkKind.LIST_BLOCK for c in chunks)


def test_blank_lines_inside_a_list_do_not_end_it() -> None:
    """The real files separate every line with a blank one, list items included."""
    body = "The items are:\n\n1. first item\n\n2. second item\n\n3. third item\n"
    chunks = StructuralChunker().chunk(_doc(body))
    list_chunks = [c for c in chunks if c.kind is ChunkKind.LIST_BLOCK]
    assert len(list_chunks) == 1
    assert all(item in list_chunks[0].text for item in ("first", "second", "third"))


def test_empty_document_yields_no_chunks() -> None:
    assert StructuralChunker().chunk(_doc("")) == []


def test_estimate_tokens_grows_with_length() -> None:
    assert estimate_tokens("one two three") < estimate_tokens("one two three four five six")
