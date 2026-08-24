"""Index construction: load -> chunk -> embed -> persist."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from mlsc_assistant.config import Settings
from mlsc_assistant.core.models import Chunk, Document, IndexManifest
from mlsc_assistant.core.ports import Chunker, Embedder, VectorStore
from mlsc_assistant.ingestion.chunker import chunk_documents
from mlsc_assistant.ingestion.loader import checksums, load_documents


@dataclass(frozen=True, slots=True)
class IndexBuildResult:
    manifest: IndexManifest
    documents: list[Document]
    chunks: list[Chunk]
    cache_hits: int
    elapsed_s: float


def build_index(
    settings: Settings,
    *,
    embedder: Embedder,
    chunker: Chunker,
    store: VectorStore,
) -> IndexBuildResult:
    """Build and persist the index.

    Takes its collaborators as arguments rather than constructing them, so the
    evaluation harness can build an index with a stub embedder without touching disk or
    downloading a model.
    """
    started = datetime.now(UTC)

    documents = load_documents(settings.kb_path, settings.knowledge_base.glob)
    chunks = chunk_documents(documents, chunker)  # type: ignore[arg-type]

    cache_before = _cache_size(embedder)
    # Embed `embed_text`, not `text`: the former carries the document-title prefix that
    # keeps pronoun-headed paragraphs anchored to their subject (ARCHITECTURE.md s4).
    vectors = embedder.embed_documents([c.embed_text for c in chunks])
    cache_hits = max(0, cache_before + len(chunks) - _cache_size(embedder)) if cache_before else 0

    store.add(chunks, vectors)

    manifest = IndexManifest(
        built_at=started,
        embedder=embedder.name,
        dimension=embedder.dimension,
        chunker_version=chunker.version,
        document_count=len(documents),
        chunk_count=len(chunks),
        document_checksums=checksums(documents),
    )
    store.persist(manifest)
    _flush_cache(embedder)

    elapsed = (datetime.now(UTC) - started).total_seconds()
    return IndexBuildResult(
        manifest=manifest,
        documents=documents,
        chunks=chunks,
        cache_hits=cache_hits,
        elapsed_s=elapsed,
    )


def current_checksums(settings: Settings) -> dict[str, str]:
    """Checksums of the knowledge base as it is on disk right now.

    Compared against the manifest to detect a knowledge base edited without a rebuild,
    which would otherwise silently serve answers from stale content.
    """
    return checksums(load_documents(settings.kb_path, settings.knowledge_base.glob))


def _cache_size(embedder: Embedder) -> int:
    cache = getattr(embedder, "cache", None)
    return len(cache) if cache is not None else 0


def _flush_cache(embedder: Embedder) -> None:
    cache = getattr(embedder, "cache", None)
    flush = getattr(cache, "flush", None)
    if callable(flush):
        flush()
