"""Ports — the interfaces the core pipeline is written against.

Nothing in this module imports a provider SDK, and nothing in ``retrieval/``,
``generation/`` or ``evaluation/`` imports a concrete adapter. Adapters are chosen in
exactly one place, ``api/deps.py``, from configuration.

These are ``Protocol`` classes rather than ABCs deliberately: adapters do not inherit
from them, so a third-party object that happens to have the right shape can be dropped
in during a test without a wrapper.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import Any, Protocol, runtime_checkable

from mlsc_assistant.core.models import Chunk, Document, IndexManifest

# ---------------------------------------------------------------------------


@runtime_checkable
class Embedder(Protocol):
    """Turns text into vectors.

    Implementations must return L2-normalised vectors so that a dot product *is*
    cosine similarity. Normalising once at embed time keeps the store trivial and
    means retrieval scores are directly comparable to the calibrated abstention
    threshold.
    """

    @property
    def name(self) -> str:
        """Model identifier, recorded in the index manifest (e.g. ``BAAI/bge-small-en-v1.5``)."""
        ...

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed corpus text. Batched; may be cached by content hash."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a query.

        Separate from ``embed_documents`` because asymmetric models (bge among them)
        prepend an instruction prefix to queries but not to passages.
        """
        ...


@runtime_checkable
class Chunker(Protocol):
    """Splits a document into retrievable units."""

    @property
    def version(self) -> str:
        """Recorded in the manifest so chunk ids can be traced to the logic that made them."""
        ...

    def chunk(self, document: Document) -> list[Chunk]: ...


@runtime_checkable
class VectorStore(Protocol):
    """Persists chunks and their vectors, and searches them.

    The default implementation is a NumPy matrix (DECISIONS.md D2). This port exists so
    "what if the knowledge base grows" is answered by a config change rather than a
    rewrite — not because 18 chunks need a database.
    """

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None: ...

    def search(self, vector: Sequence[float], k: int) -> list[tuple[Chunk, float]]:
        """Return the ``k`` nearest chunks with their similarity scores, best first."""
        ...

    def all_chunks(self) -> list[Chunk]:
        """Every chunk in insertion order.

        Needed because the lexical retriever builds its own BM25 index over the same
        corpus, and because ``GET /v1/documents/{id}/chunks`` browses it.
        """
        ...

    def persist(self, manifest: IndexManifest) -> None: ...

    def load(self) -> IndexManifest:
        """Restore a persisted index. Raises ``IndexNotBuiltError`` if there is none."""
        ...

    def __len__(self) -> int: ...


@runtime_checkable
class Reranker(Protocol):
    """Reorders candidates after fusion.

    A no-op by default: an 18-chunk corpus cannot justify the latency of a
    cross-encoder. The port exists so the evaluation can measure whether one would help
    before any is adopted.
    """

    def rerank(
        self, query: str, chunks: Sequence[Chunk], top_k: int
    ) -> list[tuple[Chunk, float]]: ...


# ---------------------------------------------------------------------------


@runtime_checkable
class LLMProvider(Protocol):
    """A text-generation backend.

    Kept to the lowest common denominator across Gemini, Anthropic, OpenAI, Groq and
    Ollama. Provider-specific features (prompt caching, extended thinking) are
    deliberately not exposed — none of them matter for a ~900-token grounded prompt,
    and exposing them would make the adapters non-interchangeable.
    """

    @property
    def name(self) -> str:
        """Provider key, e.g. ``gemini``."""
        ...

    @property
    def model(self) -> str: ...

    def complete(
        self,
        *,
        system: str,
        prompt: str,
        temperature: float = 0.0,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        """Free-form completion. Used by the judge and the query decomposer."""
        ...

    def complete_structured(
        self,
        *,
        system: str,
        prompt: str,
        schema: dict[str, Any],
        temperature: float = 0.0,
        max_output_tokens: int | None = None,
    ) -> StructuredResponse:
        """Completion constrained to a JSON schema.

        This is what makes abstention a parsed field rather than a string match
        (DECISIONS.md D5). Adapters whose backend lacks native schema support must
        emulate it — prompted JSON plus a repair pass — rather than silently returning
        prose, because callers rely on the shape.
        """
        ...


@runtime_checkable
class StreamingLLMProvider(LLMProvider, Protocol):
    """Optional capability, probed with ``isinstance``.

    Split out so that ``/v1/ask/stream`` can degrade to a single buffered chunk on
    providers that cannot stream, instead of every adapter having to implement it.
    """

    def stream(
        self,
        *,
        system: str,
        prompt: str,
        temperature: float = 0.0,
        max_output_tokens: int | None = None,
    ) -> Iterator[str]: ...


class LLMResponse(Protocol):
    text: str
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: float


class StructuredResponse(Protocol):
    data: dict[str, Any]
    raw_text: str
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: float


# ---------------------------------------------------------------------------


@runtime_checkable
class EmbeddingCache(Protocol):
    """Content-hash keyed vector cache.

    Re-indexing an unchanged knowledge base should cost nothing, which matters because
    the evaluation harness rebuilds the index across ablation runs.
    """

    def get(self, key: str) -> list[float] | None: ...

    def put(self, key: str, vector: Sequence[float]) -> None: ...

    def put_many(self, items: Iterable[tuple[str, Sequence[float]]]) -> None: ...
