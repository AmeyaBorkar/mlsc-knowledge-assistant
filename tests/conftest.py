"""Shared fixtures.

Nothing here touches the network or loads an embedding model. The real embedder is
exercised in ``tests/integration``; unit tests use ``FakeEmbedder`` so they stay fast
and deterministic.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import pytest

from mlsc_assistant.config import Settings
from mlsc_assistant.core.models import Document
from mlsc_assistant.ingestion.loader import load_documents

REPO_ROOT = Path(__file__).resolve().parents[1]
KB_PATH = REPO_ROOT / "data" / "knowledge_base"


@pytest.fixture(scope="session")
def kb_path() -> Path:
    return KB_PATH


@pytest.fixture(scope="session")
def documents() -> list[Document]:
    """The real knowledge base.

    Tests run against the actual corpus rather than fixtures: the chunker's job is to
    handle *these* documents correctly, and a synthetic stand-in would not catch the
    cases that matter (the domain list, the responsibilities list).
    """
    return load_documents(KB_PATH)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings pointed at a temporary index directory, never the real one."""
    return Settings(
        repo_root=REPO_ROOT,
        store={"backend": "numpy", "path": tmp_path / "index"},  # type: ignore[arg-type]
        embedding={"cache_dir": tmp_path / "cache"},  # type: ignore[arg-type]
    )


class FakeEmbedder:
    """Deterministic hash-based embedder. Implements ``core.ports.Embedder``.

    Vectors are meaningless as semantics but stable across runs and L2-normalised, which
    is all the store and pipeline tests need. Using the real model here would make the
    unit suite depend on a 130 MB download.
    """

    def __init__(self, dimension: int = 16) -> None:
        self._dimension = dimension
        self.embed_calls = 0

    @property
    def name(self) -> str:
        return "fake-embedder"

    @property
    def dimension(self) -> int:
        return self._dimension

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [digest[i % len(digest)] / 255.0 for i in range(self._dimension)]
        norm = sum(v * v for v in raw) ** 0.5 or 1.0
        return [v / norm for v in raw]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.embed_calls += 1
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()
