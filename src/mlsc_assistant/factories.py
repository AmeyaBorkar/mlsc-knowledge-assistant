"""Composition root — the only module that maps configuration to concrete adapters.

Everything else in the codebase depends on ``core.ports`` and never learns which
implementation it is talking to. Keeping the ``if backend == ...`` decisions in one
place is what makes "swap fastembed for sentence-transformers" or "swap Gemini for
Claude" a config change rather than a search-and-replace.

``api/deps.py`` and ``cli.py`` both call these; neither imports an adapter directly.
"""

from __future__ import annotations

from typing import cast

from mlsc_assistant.config import Settings
from mlsc_assistant.core.errors import ConfigurationError
from mlsc_assistant.core.models import IndexManifest, RetrievalStrategy
from mlsc_assistant.core.ports import Chunker, Embedder, LLMProvider, VectorStore
from mlsc_assistant.embeddings.cache import FileEmbeddingCache, NullEmbeddingCache
from mlsc_assistant.embeddings.fastembed_embedder import FastEmbedEmbedder
from mlsc_assistant.generation.answerer import GroundedAnswerer
from mlsc_assistant.generation.providers.registry import build_provider
from mlsc_assistant.ingestion.chunker import StructuralChunker
from mlsc_assistant.retrieval.retriever import HybridRetriever
from mlsc_assistant.stores.numpy_store import NumpyVectorStore


def make_chunker(settings: Settings) -> Chunker:
    cfg = settings.chunking
    if cfg.strategy != "structural":
        raise ConfigurationError(
            f"Unknown chunking strategy {cfg.strategy!r}. Only 'structural' is implemented; "
            "'fixed' is reserved for the chunking ablation."
        )
    return StructuralChunker(
        version=cfg.version,
        min_tokens=cfg.min_tokens,
        max_tokens=cfg.max_tokens,
        keep_lists_atomic=cfg.keep_lists_atomic,
        prepend_doc_title=cfg.prepend_doc_title,
    )


def make_embedder(settings: Settings, *, use_cache: bool = True) -> Embedder:
    cfg = settings.embedding
    cache = FileEmbeddingCache(settings.embedding_cache_path) if use_cache else NullEmbeddingCache()

    if cfg.backend == "fastembed":
        return FastEmbedEmbedder(
            cfg.model,
            dimension=cfg.dimension,
            batch_size=cfg.batch_size,
            models_dir=settings.models_path,
            cache=cache,
        )

    if cfg.backend == "sbert":
        # Optional extra; imported lazily so the default install never pays for torch.
        try:
            from mlsc_assistant.embeddings.sbert_embedder import SBertEmbedder
        except ImportError as exc:
            raise ConfigurationError(
                "embedding.backend is 'sbert' but sentence-transformers is not installed. "
                'Run `pip install -e ".[sbert]"`, or switch back to `fastembed`.'
            ) from exc
        # cast: the module is an optional extra, absent from a default install, so
        # mypy resolves it to Any and cannot verify it satisfies Embedder.
        return cast(
            "Embedder",
            SBertEmbedder(
                cfg.model, dimension=cfg.dimension, batch_size=cfg.batch_size, cache=cache
            ),
        )

    raise ConfigurationError(f"Unknown embedding backend {cfg.backend!r}.")


def make_store(settings: Settings) -> VectorStore:
    backend = settings.store.backend

    if backend == "numpy":
        return NumpyVectorStore(settings.index_path)

    if backend == "chroma":
        try:
            from mlsc_assistant.stores.chroma_store import ChromaVectorStore
        except ImportError as exc:
            raise ConfigurationError(
                "store.backend is 'chroma' but chromadb is not installed. "
                'Run `pip install -e ".[chroma]"`, or switch back to `numpy`.'
            ) from exc
        return cast("VectorStore", ChromaVectorStore(settings.index_path))

    raise ConfigurationError(f"Unknown vector store backend {backend!r}.")


def load_store(settings: Settings) -> tuple[VectorStore, IndexManifest]:
    """Build a store and load the persisted index into it.

    Raises ``IndexNotBuiltError`` if there is no index, with a message naming the
    command that fixes it.
    """
    store = make_store(settings)
    manifest = store.load()
    return store, manifest


def make_retriever(
    settings: Settings,
    *,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
) -> HybridRetriever:
    """Build a retriever over an already-loaded store.

    Collaborators are injectable so the evaluation harness can sweep strategies and
    chunking variants without re-reading config or reloading the model each time.
    """
    if store is None:
        store, _ = load_store(settings)
    if embedder is None:
        embedder = make_embedder(settings)

    cfg = settings.retrieval
    return HybridRetriever(
        embedder=embedder,
        store=store,
        strategy=RetrievalStrategy(cfg.strategy),
        top_k=cfg.top_k,
        candidate_k=cfg.candidate_k,
        rrf_k=cfg.rrf_k,
        max_chunks_per_document=cfg.max_chunks_per_document,
        bm25_k1=cfg.bm25.k1,
        bm25_b=cfg.bm25.b,
        bm25_index_title=cfg.bm25.index_title,
        bm25_max_document_frequency=cfg.bm25.max_document_frequency,
    )


def make_provider(settings: Settings) -> LLMProvider:
    """Construct the configured generation backend."""
    return build_provider(settings.llm)


def make_answerer(
    settings: Settings,
    *,
    retriever: HybridRetriever | None = None,
    provider: LLMProvider | None = None,
) -> GroundedAnswerer:
    """Assemble the full question-answering pipeline.

    Collaborators are injectable so the evaluation harness can hold one retriever and
    one provider across a whole run instead of rebuilding them per question.
    """
    return GroundedAnswerer(
        retriever=retriever if retriever is not None else make_retriever(settings),
        provider=provider if provider is not None else make_provider(settings),
        settings=settings,
    )
