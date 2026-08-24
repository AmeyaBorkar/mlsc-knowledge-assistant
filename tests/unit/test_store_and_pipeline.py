"""Vector store, embedding cache and index pipeline tests."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from mlsc_assistant.config import Settings
from mlsc_assistant.core.errors import IndexNotBuiltError
from mlsc_assistant.core.models import Chunk, ChunkKind, IndexManifest
from mlsc_assistant.embeddings.cache import FileEmbeddingCache, cache_key
from mlsc_assistant.factories import make_chunker
from mlsc_assistant.ingestion.pipeline import build_index, current_checksums
from mlsc_assistant.stores.numpy_store import NumpyVectorStore


def _chunk(i: int, doc: str = "doc") -> Chunk:
    text = f"chunk number {i}"
    return Chunk(
        chunk_id=f"{doc}::c{i:02d}",
        doc_id=doc,
        doc_title="Doc Title",
        source_file=f"{doc}.txt",
        text=text,
        embed_text=f"Doc Title - {text}",
        char_range=(i * 10, i * 10 + len(text)),
        index=i,
        kind=ChunkKind.PARAGRAPH,
        token_estimate=4,
        checksum=f"sum{i}",
    )


def _manifest(chunk_count: int = 3) -> IndexManifest:
    return IndexManifest(
        built_at=datetime.now(UTC),
        embedder="fake-embedder",
        dimension=16,
        chunker_version="structural-v1",
        document_count=1,
        chunk_count=chunk_count,
        document_checksums={"doc": "abc"},
    )


# --- store ------------------------------------------------------------------


def test_search_ranks_by_cosine_similarity(tmp_path: Path, fake_embedder) -> None:  # type: ignore[no-untyped-def]
    store = NumpyVectorStore(tmp_path)
    chunks = [_chunk(i) for i in range(5)]
    store.add(chunks, fake_embedder.embed_documents([c.embed_text for c in chunks]))

    results = store.search(fake_embedder.embed_query(chunks[2].embed_text), k=3)

    assert len(results) == 3
    assert results[0][0].chunk_id == chunks[2].chunk_id, "exact match should rank first"
    assert results[0][1] == pytest.approx(1.0, abs=1e-5), "normalised vectors: self-similarity is 1"
    assert [s for _, s in results] == sorted((s for _, s in results), reverse=True)


def test_search_before_load_names_the_fix(tmp_path: Path, fake_embedder) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(IndexNotBuiltError, match="mlsc index"):
        NumpyVectorStore(tmp_path).search(fake_embedder.embed_query("anything"), k=3)


def test_k_larger_than_the_corpus_is_clamped(tmp_path: Path, fake_embedder) -> None:  # type: ignore[no-untyped-def]
    store = NumpyVectorStore(tmp_path)
    chunks = [_chunk(i) for i in range(3)]
    store.add(chunks, fake_embedder.embed_documents([c.embed_text for c in chunks]))
    assert len(store.search(fake_embedder.embed_query("q"), k=50)) == 3


def test_dimension_mismatch_tells_you_to_rebuild(tmp_path: Path, fake_embedder) -> None:  # type: ignore[no-untyped-def]
    """Switching embedder without re-indexing is a real mistake, and a bare NumPy
    broadcast error would not explain it."""
    store = NumpyVectorStore(tmp_path)
    chunks = [_chunk(0)]
    store.add(chunks, fake_embedder.embed_documents([chunks[0].embed_text]))

    with pytest.raises(ValueError, match="rebuild"):
        store.search([0.1] * 384, k=1)


def test_mismatched_chunk_and_vector_counts_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mismatch"):
        NumpyVectorStore(tmp_path).add([_chunk(0), _chunk(1)], [[0.1] * 4])


def test_round_trip_preserves_chunks_and_scores(tmp_path: Path, fake_embedder) -> None:  # type: ignore[no-untyped-def]
    chunks = [_chunk(i) for i in range(4)]
    vectors = fake_embedder.embed_documents([c.embed_text for c in chunks])

    original = NumpyVectorStore(tmp_path)
    original.add(chunks, vectors)
    original.persist(_manifest(chunk_count=4))
    before = original.search(fake_embedder.embed_query("chunk number 1"), k=4)

    reloaded = NumpyVectorStore(tmp_path)
    manifest = reloaded.load()

    assert manifest.embedder == "fake-embedder"
    assert len(reloaded) == 4
    after = reloaded.search(fake_embedder.embed_query("chunk number 1"), k=4)
    assert [(c.chunk_id, round(s, 6)) for c, s in before] == [
        (c.chunk_id, round(s, 6)) for c, s in after
    ]


def test_reloaded_chunks_keep_every_field(tmp_path: Path, fake_embedder) -> None:  # type: ignore[no-untyped-def]
    """Citation rendering depends on char_range, kind and titles surviving a round trip."""
    chunk = _chunk(7)
    store = NumpyVectorStore(tmp_path)
    store.add([chunk], fake_embedder.embed_documents([chunk.embed_text]))
    store.persist(_manifest(chunk_count=1))

    restored = NumpyVectorStore(tmp_path)
    restored.load()
    assert restored.all_chunks()[0] == chunk


def test_load_without_an_index_names_the_fix(tmp_path: Path) -> None:
    with pytest.raises(IndexNotBuiltError, match="mlsc index"):
        NumpyVectorStore(tmp_path / "missing").load()


def test_read_manifest_does_not_need_the_vectors(tmp_path: Path, fake_embedder) -> None:  # type: ignore[no-untyped-def]
    """`GET /v1/health` reads the manifest on every call and should not load the matrix."""
    store = NumpyVectorStore(tmp_path)
    store.add([_chunk(0)], fake_embedder.embed_documents(["x"]))
    store.persist(_manifest(chunk_count=1))

    assert NumpyVectorStore.read_manifest(tmp_path) is not None
    assert NumpyVectorStore.read_manifest(tmp_path / "elsewhere") is None


def test_corrupt_index_is_detected(tmp_path: Path, fake_embedder) -> None:  # type: ignore[no-untyped-def]
    import json

    store = NumpyVectorStore(tmp_path)
    chunks = [_chunk(i) for i in range(3)]
    store.add(chunks, fake_embedder.embed_documents([c.embed_text for c in chunks]))
    store.persist(_manifest(chunk_count=3))

    data = json.loads((tmp_path / "chunks.json").read_text(encoding="utf-8"))
    (tmp_path / "chunks.json").write_text(json.dumps(data[:2]), encoding="utf-8")

    with pytest.raises(IndexNotBuiltError, match="Corrupt index"):
        NumpyVectorStore(tmp_path).load()


# --- manifest ---------------------------------------------------------------


def test_manifest_detects_an_edited_knowledge_base() -> None:
    manifest = _manifest()
    assert not manifest.is_stale({"doc": "abc"})
    assert manifest.is_stale({"doc": "different"})
    assert manifest.is_stale({"doc": "abc", "extra": "new"}), "a new document is also stale"


# --- embedding cache --------------------------------------------------------


def test_cache_survives_a_restart(tmp_path: Path) -> None:
    key = cache_key("model", 4, "hello")
    cache = FileEmbeddingCache(tmp_path)
    cache.put(key, [0.1, 0.2, 0.3, 0.4])
    cache.flush()

    reopened = FileEmbeddingCache(tmp_path)
    assert reopened.get(key) == pytest.approx([0.1, 0.2, 0.3, 0.4], abs=1e-6)


def test_cache_key_separates_models() -> None:
    """Switching embedder must never serve a vector produced by the previous one."""
    assert cache_key("model-a", 384, "text") != cache_key("model-b", 384, "text")
    assert cache_key("model-a", 384, "text") != cache_key("model-a", 768, "text")


def test_corrupt_cache_degrades_to_a_miss(tmp_path: Path) -> None:
    """A damaged cache is a performance problem, never a correctness one."""
    cache = FileEmbeddingCache(tmp_path)
    cache.put(cache_key("m", 4, "a"), [1.0, 0.0, 0.0, 0.0])
    cache.flush()
    (tmp_path / "keys.json").write_text("{ not json", encoding="utf-8")

    assert FileEmbeddingCache(tmp_path).get(cache_key("m", 4, "a")) is None


# --- pipeline ---------------------------------------------------------------


def test_build_index_produces_a_loadable_index(settings: Settings, fake_embedder) -> None:  # type: ignore[no-untyped-def]
    result = build_index(
        settings,
        embedder=fake_embedder,
        chunker=make_chunker(settings),
        store=NumpyVectorStore(settings.index_path),
    )

    assert result.manifest.document_count == 6
    assert result.manifest.chunk_count == len(result.chunks) == 18
    assert result.manifest.embedder == "fake-embedder"
    assert result.manifest.chunker_version == "structural-v1"

    reloaded = NumpyVectorStore(settings.index_path)
    assert reloaded.load().chunk_count == 18
    assert len(reloaded) == 18


def test_index_embeds_the_prefixed_text(settings: Settings, fake_embedder) -> None:  # type: ignore[no-untyped-def]
    """Embedding `text` instead of `embed_text` would silently drop the contextual
    header and quietly cost recall on pronoun-headed paragraphs."""
    captured: list[str] = []
    original = fake_embedder.embed_documents

    def spy(texts):  # type: ignore[no-untyped-def]
        captured.extend(texts)
        return original(texts)

    fake_embedder.embed_documents = spy  # type: ignore[method-assign]
    build_index(
        settings,
        embedder=fake_embedder,
        chunker=make_chunker(settings),
        store=NumpyVectorStore(settings.index_path),
    )

    assert captured
    assert all(" - " in t for t in captured)
    assert any(t.startswith("MLSC Technical Domains - ") for t in captured)


def test_manifest_matches_the_knowledge_base_on_disk(settings: Settings, fake_embedder) -> None:  # type: ignore[no-untyped-def]
    result = build_index(
        settings,
        embedder=fake_embedder,
        chunker=make_chunker(settings),
        store=NumpyVectorStore(settings.index_path),
    )
    assert not result.manifest.is_stale(current_checksums(settings))
