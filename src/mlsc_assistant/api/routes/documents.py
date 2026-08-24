"""The knowledge base as a browsable resource.

Exists so a citation in a UI is clickable through to its source, and so a reviewer can
confirm the system reads the documents it was given and nothing else. No API key needed.
"""

from __future__ import annotations

from fastapi import APIRouter

from mlsc_assistant.api.deps import StateDep
from mlsc_assistant.api.schemas import ChunkOut, DocumentDetail, DocumentSummary
from mlsc_assistant.core.errors import DocumentNotFoundError

router = APIRouter(prefix="/documents", tags=["documents"])


def _require(state: StateDep, doc_id: str):  # type: ignore[no-untyped-def]
    document = state.documents.get(doc_id)
    if document is None:
        known = ", ".join(sorted(state.documents))
        raise DocumentNotFoundError(f"No document {doc_id!r}. Available: {known}.")
    return document


@router.get("", response_model=list[DocumentSummary])
def list_documents(state: StateDep) -> list[DocumentSummary]:
    counts: dict[str, int] = {}
    for chunk in state.store.all_chunks():
        counts[chunk.doc_id] = counts.get(chunk.doc_id, 0) + 1
    return [
        DocumentSummary(
            doc_id=d.doc_id,
            title=d.title,
            source_file=d.source_file,
            chunk_count=counts.get(d.doc_id, 0),
            characters=len(d.text),
        )
        for d in sorted(state.documents.values(), key=lambda d: d.doc_id)
    ]


@router.get("/{doc_id}", response_model=DocumentDetail)
def get_document(doc_id: str, state: StateDep) -> DocumentDetail:
    document = _require(state, doc_id)
    chunks = [c for c in state.store.all_chunks() if c.doc_id == doc_id]
    return DocumentDetail(
        doc_id=document.doc_id,
        title=document.title,
        source_file=document.source_file,
        chunk_count=len(chunks),
        characters=len(document.text),
        text=document.text,
    )


@router.get("/{doc_id}/chunks", response_model=list[ChunkOut])
def get_chunks(doc_id: str, state: StateDep) -> list[ChunkOut]:
    """Chunks with their character offsets, so a client can highlight the exact
    passage an answer rests on."""
    _require(state, doc_id)
    return [
        ChunkOut(
            chunk_id=c.chunk_id,
            doc_id=c.doc_id,
            index=c.index,
            kind=c.kind.value,
            text=c.text,
            char_range=c.char_range,
            token_estimate=c.token_estimate,
        )
        for c in state.store.all_chunks()
        if c.doc_id == doc_id
    ]
