"""Loading knowledge-base documents from disk."""

from __future__ import annotations

from pathlib import Path

from mlsc_assistant.core.errors import KnowledgeBaseEmptyError
from mlsc_assistant.core.models import Document


def _title_from(text: str, fallback: str) -> str:
    """Take the first non-empty line as the title.

    Every document in this knowledge base opens with one (``MLSC Leadership
    Structure``, ``MLSC Code of Conduct``). The title matters more than it looks: it is
    prepended to each chunk at embed time so that pronoun-headed paragraphs keep their
    subject, so a bad title degrades retrieval across the whole document.
    """
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return fallback


def load_document(path: Path) -> Document:
    # utf-8-sig strips a BOM if one is present. The sample files came from a macOS
    # zip, but a Windows-edited knowledge base would otherwise put U+FEFF at the front
    # of the title and thus into every embedded chunk of that document.
    text = path.read_text(encoding="utf-8-sig")
    doc_id = path.stem
    return Document(
        doc_id=doc_id,
        title=_title_from(text, fallback=doc_id.replace("_", " ").title()),
        text=text,
        source_file=path.name,
        checksum=Document.compute_checksum(text),
    )


def load_documents(kb_path: Path, glob: str = "*.txt") -> list[Document]:
    """Load every knowledge-base document, sorted by filename.

    Sorted so that chunk ordering — and therefore the row order of the vector matrix —
    is reproducible across machines and filesystems. An index that depends on
    directory iteration order is an index whose metrics cannot be compared.
    """
    if not kb_path.is_dir():
        raise KnowledgeBaseEmptyError(
            f"Knowledge base directory not found: {kb_path}. "
            "Check `knowledge_base.path` in config.yaml."
        )

    paths = sorted(p for p in kb_path.glob(glob) if p.is_file() and not p.name.startswith("."))
    # macOS zips carry `__MACOSX/._name` resource forks; the dotfile filter above drops
    # them, and README.md is documentation rather than knowledge.
    paths = [p for p in paths if p.name.lower() != "readme.md"]

    if not paths:
        raise KnowledgeBaseEmptyError(
            f"No documents matching {glob!r} in {kb_path}. Nothing to index."
        )

    return [load_document(p) for p in paths]


def checksums(documents: list[Document]) -> dict[str, str]:
    """Map of ``doc_id`` -> checksum, for the index manifest's staleness check."""
    return {d.doc_id: d.checksum for d in documents}
