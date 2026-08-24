"""Loader tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from mlsc_assistant.core.errors import KnowledgeBaseEmptyError
from mlsc_assistant.ingestion.loader import checksums, load_document, load_documents

EXPECTED_FILES = {
    "about_mlsc.txt",
    "code_of_conduct.txt",
    "domains.txt",
    "hackathons.txt",
    "leadership.txt",
    "membership.txt",
}


def test_loads_every_knowledge_base_document(documents) -> None:  # type: ignore[no-untyped-def]
    assert {d.source_file for d in documents} == EXPECTED_FILES


def test_documents_load_in_a_stable_order(kb_path: Path) -> None:
    """Chunk ids and vector row order derive from this.

    An index whose ordering depends on filesystem iteration is one whose metrics cannot
    be compared across machines.
    """
    assert [d.doc_id for d in load_documents(kb_path)] == [
        d.doc_id for d in load_documents(kb_path)
    ]
    assert [d.doc_id for d in load_documents(kb_path)] == sorted(
        d.doc_id for d in load_documents(kb_path)
    )


def test_titles_come_from_the_first_line(documents) -> None:  # type: ignore[no-untyped-def]
    titles = {d.doc_id: d.title for d in documents}
    assert titles["leadership"] == "MLSC Leadership Structure"
    assert titles["domains"] == "MLSC Technical Domains"
    assert titles["code_of_conduct"] == "MLSC Code of Conduct"


def test_unicode_survives_loading(documents) -> None:  # type: ignore[no-untyped-def]
    """about_mlsc.txt has an em dash in its title, which is prepended to every one of
    its chunks at embed time — mangling it would degrade that whole document."""
    about = next(d for d in documents if d.doc_id == "about_mlsc")
    assert "—" in about.title
    assert "﻿" not in about.text


def test_readme_is_not_treated_as_knowledge(kb_path: Path) -> None:
    """data/knowledge_base/README.md documents provenance; it is not a source."""
    assert "README.md" not in {d.source_file for d in load_documents(kb_path, "*")}


def test_checksums_track_content(tmp_path: Path) -> None:
    path = tmp_path / "doc.txt"
    path.write_text("Title\n\nOriginal body.\n", encoding="utf-8")
    before = load_document(path).checksum

    path.write_text("Title\n\nEdited body.\n", encoding="utf-8")
    assert load_document(path).checksum != before


def test_checksum_map_is_keyed_by_doc_id(documents) -> None:  # type: ignore[no-untyped-def]
    result = checksums(documents)
    assert set(result) == {d.doc_id for d in documents}
    assert all(len(v) == 64 for v in result.values())


def test_missing_directory_names_the_setting(tmp_path: Path) -> None:
    with pytest.raises(KnowledgeBaseEmptyError, match=r"config\.yaml"):
        load_documents(tmp_path / "nope")


def test_empty_directory_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(KnowledgeBaseEmptyError, match="Nothing to index"):
        load_documents(tmp_path)


def test_macos_resource_forks_are_ignored(tmp_path: Path) -> None:
    """The supplied zip carries `._name` files alongside the real ones."""
    (tmp_path / "real.txt").write_text("Real Title\n\nBody.\n", encoding="utf-8")
    (tmp_path / "._real.txt").write_bytes(b"\x00\x05\x16\x07binary junk")

    documents = load_documents(tmp_path)
    assert [d.source_file for d in documents] == ["real.txt"]


def test_bom_is_stripped(tmp_path: Path) -> None:
    """A Windows-edited knowledge base would otherwise put U+FEFF into the title, and
    from there into every embedded chunk of that document."""
    path = tmp_path / "bom.txt"
    path.write_text("BOM Title\n\nBody text.\n", encoding="utf-8-sig")
    assert load_document(path).title == "BOM Title"
