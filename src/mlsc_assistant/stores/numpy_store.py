"""NumPy vector store — the default.

The entire corpus is an ``18 x 384`` float32 matrix, about 28 KB. Search is one matrix
multiply. A vector database would add a service, a schema and a failure mode to make a
microsecond operation slightly different (DECISIONS.md D2).

Because every vector is L2-normalised at embed time, ``matrix @ query`` *is* cosine
similarity — no per-query normalisation, no distance-to-score conversion.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from mlsc_assistant.core.errors import IndexNotBuiltError
from mlsc_assistant.core.models import Chunk, ChunkKind, IndexManifest

_VECTORS_FILE = "vectors.npz"
_CHUNKS_FILE = "chunks.json"
_MANIFEST_FILE = "manifest.json"


class NumpyVectorStore:
    """Implements ``core.ports.VectorStore``."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None

    # -- writing -------------------------------------------------------------

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunk/vector count mismatch: {len(chunks)} chunks, {len(vectors)} vectors"
            )
        if not chunks:
            return

        new = np.asarray(vectors, dtype=np.float32)
        if new.ndim != 2:
            raise ValueError(f"vectors must be 2-dimensional, got shape {new.shape}")

        self._chunks.extend(chunks)
        self._matrix = new if self._matrix is None else np.vstack([self._matrix, new])

    # -- reading -------------------------------------------------------------

    def search(self, vector: Sequence[float], k: int) -> list[tuple[Chunk, float]]:
        if self._matrix is None or not self._chunks:
            raise IndexNotBuiltError(f"No index loaded from {self.path}. Run `mlsc index` first.")

        query = np.asarray(vector, dtype=np.float32)
        if query.shape[0] != self._matrix.shape[1]:
            raise ValueError(
                f"Query dimension {query.shape[0]} does not match index dimension "
                f"{self._matrix.shape[1]}. The index was built with a different embedder — "
                "rebuild it with `mlsc index --force`."
            )

        scores = self._matrix @ query
        k = min(k, len(self._chunks))
        # argpartition finds the top k without sorting all n; the slice is then sorted.
        # At n=23 this is irrelevant, but it costs nothing and does not mislead a reader
        # into thinking the store is O(n log n) by necessity.
        top = np.argpartition(-scores, k - 1)[:k] if k < len(scores) else np.arange(len(scores))
        top = top[np.argsort(-scores[top])]
        return [(self._chunks[i], float(scores[i])) for i in top]

    def all_chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def __len__(self) -> int:
        return len(self._chunks)

    # -- persistence ---------------------------------------------------------

    def persist(self, manifest: IndexManifest) -> None:
        if self._matrix is None:
            raise IndexNotBuiltError("Nothing to persist: no vectors have been added.")

        self.path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.path / _VECTORS_FILE, vectors=self._matrix)
        (self.path / _CHUNKS_FILE).write_text(
            json.dumps([_chunk_to_dict(c) for c in self._chunks], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (self.path / _MANIFEST_FILE).write_text(
            json.dumps(_manifest_to_dict(manifest), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def load(self) -> IndexManifest:
        manifest_path = self.path / _MANIFEST_FILE
        if not manifest_path.is_file():
            raise IndexNotBuiltError(f"No index found at {self.path}. Run `mlsc index` first.")

        with np.load(self.path / _VECTORS_FILE) as data:
            self._matrix = data["vectors"]
        self._chunks = [
            _chunk_from_dict(d)
            for d in json.loads((self.path / _CHUNKS_FILE).read_text(encoding="utf-8"))
        ]
        manifest = _manifest_from_dict(json.loads(manifest_path.read_text(encoding="utf-8")))

        assert self._matrix is not None
        if self._matrix.shape[0] != len(self._chunks):
            raise IndexNotBuiltError(
                f"Corrupt index at {self.path}: {self._matrix.shape[0]} vectors for "
                f"{len(self._chunks)} chunks. Rebuild with `mlsc index --force`."
            )
        return manifest

    @staticmethod
    def read_manifest(path: Path) -> IndexManifest | None:
        """Read the manifest without loading vectors.

        ``GET /v1/health`` needs the manifest to report staleness, and paying for the
        matrix on a liveness check would be silly.
        """
        manifest_path = path / _MANIFEST_FILE
        if not manifest_path.is_file():
            return None
        return _manifest_from_dict(json.loads(manifest_path.read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# Serialisation
#
# Hand-written rather than pickled: the index is a build artefact that should be
# inspectable with a text editor, diffable, and safe to load from disk.
# ---------------------------------------------------------------------------


def _chunk_to_dict(chunk: Chunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "doc_title": chunk.doc_title,
        "source_file": chunk.source_file,
        "text": chunk.text,
        "embed_text": chunk.embed_text,
        "char_range": list(chunk.char_range),
        "index": chunk.index,
        "kind": chunk.kind.value,
        "token_estimate": chunk.token_estimate,
        "checksum": chunk.checksum,
    }


def _chunk_from_dict(data: dict[str, Any]) -> Chunk:
    """Rebuild a chunk from its JSON form.

    Typed ``Any`` rather than ``object``: decoded JSON genuinely is dynamic, and the
    constructor calls below are the point where it gets validated back into real types.
    Pretending otherwise only buys a row of ``type: ignore`` comments.
    """
    start, end = data["char_range"]
    return Chunk(
        chunk_id=str(data["chunk_id"]),
        doc_id=str(data["doc_id"]),
        doc_title=str(data["doc_title"]),
        source_file=str(data["source_file"]),
        text=str(data["text"]),
        embed_text=str(data["embed_text"]),
        char_range=(int(start), int(end)),
        index=int(data["index"]),
        kind=ChunkKind(str(data["kind"])),
        token_estimate=int(data["token_estimate"]),
        checksum=str(data["checksum"]),
    )


def _manifest_to_dict(manifest: IndexManifest) -> dict[str, Any]:
    return {
        "built_at": manifest.built_at.isoformat(),
        "embedder": manifest.embedder,
        "dimension": manifest.dimension,
        "chunker_version": manifest.chunker_version,
        "document_count": manifest.document_count,
        "chunk_count": manifest.chunk_count,
        "document_checksums": manifest.document_checksums,
    }


def _manifest_from_dict(data: dict[str, Any]) -> IndexManifest:
    return IndexManifest(
        built_at=datetime.fromisoformat(str(data["built_at"])),
        embedder=str(data["embedder"]),
        dimension=int(data["dimension"]),
        chunker_version=str(data["chunker_version"]),
        document_count=int(data["document_count"]),
        chunk_count=int(data["chunk_count"]),
        document_checksums={str(k): str(v) for k, v in data["document_checksums"].items()},
    )
