"""Content-hash keyed embedding cache.

Re-indexing an unchanged knowledge base should cost nothing. This matters less for the
6 KB corpus than for the evaluation harness, which rebuilds the index once per ablation
run — without a cache, the chunking ablations would re-embed everything each time.

The key includes the model name and dimension, so switching embedders can never serve a
vector produced by a different model.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np


def cache_key(model: str, dimension: int, text: str) -> str:
    payload = f"{model}|{dimension}|{text}".encode()
    return hashlib.sha256(payload).hexdigest()


class FileEmbeddingCache:
    """A single ``.npz`` of vectors plus a JSON key index.

    One file rather than one-file-per-vector: at a few thousand entries the whole thing
    loads in milliseconds, and it avoids scattering thousands of tiny files across a
    Windows filesystem.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._vectors_file = path / "vectors.npz"
        self._keys_file = path / "keys.json"
        self._keys: dict[str, int] = {}
        self._vectors: list[np.ndarray] = []
        self._dirty = False
        self._load()

    def _load(self) -> None:
        if not (self._vectors_file.is_file() and self._keys_file.is_file()):
            return
        try:
            with np.load(self._vectors_file) as data:
                matrix = data["vectors"]
            keys = json.loads(self._keys_file.read_text(encoding="utf-8"))
        except (OSError, ValueError, KeyError):
            # A corrupt or half-written cache is a performance problem, never a
            # correctness one: drop it and re-embed rather than failing the build.
            return
        if len(keys) != matrix.shape[0]:
            return
        self._keys = {k: i for i, k in enumerate(keys)}
        self._vectors = [matrix[i] for i in range(matrix.shape[0])]

    def get(self, key: str) -> list[float] | None:
        index = self._keys.get(key)
        return None if index is None else self._vectors[index].tolist()

    def put(self, key: str, vector: Sequence[float]) -> None:
        if key in self._keys:
            return
        self._keys[key] = len(self._vectors)
        self._vectors.append(np.asarray(vector, dtype=np.float32))
        self._dirty = True

    def put_many(self, items: Iterable[tuple[str, Sequence[float]]]) -> None:
        for key, vector in items:
            self.put(key, vector)

    def flush(self) -> None:
        if not self._dirty or not self._vectors:
            return
        self.path.mkdir(parents=True, exist_ok=True)
        matrix = np.vstack(self._vectors).astype(np.float32)
        np.savez_compressed(self._vectors_file, vectors=matrix)
        ordered = sorted(self._keys, key=lambda k: self._keys[k])
        self._keys_file.write_text(json.dumps(ordered), encoding="utf-8")
        self._dirty = False

    def __len__(self) -> int:
        return len(self._keys)


class NullEmbeddingCache:
    """Disabled cache. Used in tests where caching would mask a real embedding call."""

    def get(self, key: str) -> list[float] | None:  # noqa: ARG002
        return None

    def put(self, key: str, vector: Sequence[float]) -> None: ...

    def put_many(self, items: Iterable[tuple[str, Sequence[float]]]) -> None: ...

    def flush(self) -> None: ...

    def __len__(self) -> int:
        return 0
