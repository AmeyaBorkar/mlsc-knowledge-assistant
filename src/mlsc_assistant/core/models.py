"""Domain models.

Plain data structures shared by every layer. These are deliberately *not* the API
schemas (see ``api/schemas.py``): domain objects carry internals the wire should not,
and the public contract should be free to stay stable while these move.

Everything here is frozen. Values flow one way through the pipeline
(load -> chunk -> embed -> retrieve -> answer) and nothing downstream should be able
to mutate what an earlier stage produced.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Document:
    """One source file from the knowledge base."""

    doc_id: str
    """Stable identifier, derived from the filename stem (e.g. ``leadership``)."""

    title: str
    """Human-readable title, taken from the first line of the file."""

    text: str
    """Full document text, verbatim, including the title line.

    Kept whole so that chunk character offsets index into *this* string and a
    citation can always be resolved back to the original file.
    """

    source_file: str
    """Filename as it appears in the knowledge base directory."""

    checksum: str
    """SHA-256 of ``text``. Lets the index manifest detect edited documents."""

    @staticmethod
    def compute_checksum(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ChunkKind(StrEnum):
    """What structural element a chunk came from.

    Recorded because the chunker treats lists atomically (see DECISIONS.md D9), and
    when retrieval behaves oddly it matters whether the chunk was a paragraph or a
    whole list block.
    """

    PARAGRAPH = "paragraph"
    LIST_BLOCK = "list_block"
    HEADING = "heading"
    MERGED = "merged"


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable unit of text.

    Two text views exist on purpose:

    ``text``
        Clean text as it appears in the document. This is what gets displayed, what
        citations quote, and what ``char_range`` indexes.
    ``embed_text``
        What actually gets embedded, which prefixes the document title so that
        pronoun-headed paragraphs ("Each domain has domain leads...") keep their
        subject in vector space.

    Conflating the two would put the title into every quoted snippet.
    """

    chunk_id: str
    """``{doc_id}::c{index:02d}`` — deterministic, so evaluation sets can name chunks."""

    doc_id: str
    doc_title: str
    source_file: str
    text: str
    embed_text: str
    char_range: tuple[int, int]
    """``[start, end)`` offsets of ``text`` within ``Document.text``."""

    index: int
    """Position of this chunk within its document, zero-based."""

    kind: ChunkKind = ChunkKind.PARAGRAPH
    token_estimate: int = 0
    checksum: str = ""

    @property
    def citation_label(self) -> str:
        return f"[{self.chunk_id}]"


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


class RetrievalStrategy(StrEnum):
    HYBRID = "hybrid"
    DENSE = "dense"
    LEXICAL = "lexical"


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    """A chunk with the scores that got it here.

    Per-retriever ranks and scores are kept alongside the fused score rather than
    collapsed into one number. When hybrid beats dense-only on the evaluation set,
    this is the object that shows *why* on any individual query.
    """

    chunk: Chunk
    score: float
    """The score used for ordering — fused for hybrid, raw for single-strategy runs."""

    rank: int
    dense_score: float | None = None
    dense_rank: int | None = None
    lexical_score: float | None = None
    lexical_rank: int | None = None
    rrf_score: float | None = None
    matched_terms: tuple[str, ...] = ()

    @property
    def doc_id(self) -> str:
        return self.chunk.doc_id


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """The outcome of one retrieval call, with the diagnostics the API returns."""

    query: str
    strategy: RetrievalStrategy
    chunks: tuple[ScoredChunk, ...]
    candidates_considered: int
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def top_score(self) -> float:
        return self.chunks[0].score if self.chunks else 0.0

    @property
    def score_margin(self) -> float:
        """Gap between rank 1 and rank 2.

        A high top score with a negligible margin means several chunks look equally
        plausible, which is weak evidence even when the absolute score is high.
        """
        if len(self.chunks) < 2:
            return 0.0
        return self.chunks[0].score - self.chunks[1].score

    @property
    def top_dense_score(self) -> float | None:
        """Best raw cosine score, or None if dense retrieval did not run.

        Abstention gate 1 reads *this*, never ``top_score``. Under the hybrid strategy
        ``top_score`` is an RRF score (~0.03, scale-free by construction), so comparing
        a calibrated cosine threshold against it would be meaningless.
        """
        scores = [sc.dense_score for sc in self.chunks if sc.dense_score is not None]
        return max(scores) if scores else None

    @property
    def documents_represented(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for sc in self.chunks:
            seen.setdefault(sc.doc_id, None)
        return tuple(seen)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


class AbstentionReason(StrEnum):
    """Why the system declined to answer.

    An enum rather than a string so clients can branch, and so the evaluation harness
    can attribute a refusal to the gate that caused it.
    """

    NO_RELEVANT_CONTEXT = "no_relevant_context"
    """Gate 1: retrieval found nothing above the calibrated threshold."""

    INSUFFICIENT_CONTEXT = "insufficient_context"
    """Gate 2: context was retrieved but does not contain the answer."""

    UNFAITHFUL_ANSWER = "unfaithful_answer"
    """Gate 3: the drafted answer made claims its citations do not support."""

    PROVIDER_UNAVAILABLE = "provider_unavailable"


class Confidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True, slots=True)
class Citation:
    """A pointer from an answer back to the passage that supports it."""

    chunk_id: str
    doc_id: str
    doc_title: str
    source_file: str
    snippet: str
    char_range: tuple[int, int]
    score: float


@dataclass(frozen=True, slots=True)
class Answer:
    """The system's response to a question.

    ``answered=False`` is a *successful* outcome, not an error — refusing correctly is
    a feature here (DECISIONS.md D10). Clients read ``answered``, not the HTTP status.
    """

    question: str
    text: str
    answered: bool
    citations: tuple[Citation, ...] = ()
    abstention_reason: AbstentionReason | None = None
    confidence: Confidence = Confidence.MEDIUM
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def abstained(self) -> bool:
        return not self.answered

    @property
    def sources(self) -> tuple[str, ...]:
        """Distinct source files backing this answer, in citation order."""
        seen: dict[str, None] = {}
        for c in self.citations:
            seen.setdefault(c.source_file, None)
        return tuple(seen)


# ---------------------------------------------------------------------------
# Index metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IndexManifest:
    """What produced an index.

    Written on every build and echoed by ``GET /v1/health`` and every evaluation run.
    A metric without one of these is not a reproducible result: it is impossible to
    say later which embedder, chunker or corpus version produced a number.
    """

    built_at: datetime
    embedder: str
    dimension: int
    chunker_version: str
    document_count: int
    chunk_count: int
    document_checksums: dict[str, str]
    """``doc_id`` -> SHA-256, so a knowledge base edited without re-indexing is detectable."""

    def is_stale(self, current: dict[str, str]) -> bool:
        return self.document_checksums != current
