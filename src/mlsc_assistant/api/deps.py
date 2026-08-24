"""Dependency wiring — the API's composition root.

Expensive collaborators are built **once at startup**, not per request: the embedding
model takes seconds to load and BM25 indexes the corpus on construction. Building them
per request would make every call pay a cold start.

The provider is the exception. It is built lazily and cached, so an API with no key
configured still serves `/v1/health`, `/v1/search` and the document endpoints — the
key-free half of the system stays available rather than the whole app refusing to start
(DECISIONS.md D4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any

from fastapi import Depends, Request

from mlsc_assistant.config import Settings
from mlsc_assistant.core.errors import ConfigurationError, MLSCError
from mlsc_assistant.core.models import Document, IndexManifest
from mlsc_assistant.core.ports import LLMProvider, VectorStore
from mlsc_assistant.factories import (
    load_store,
    make_answerer,
    make_embedder,
    make_provider,
    make_retriever,
)
from mlsc_assistant.generation.answerer import GroundedAnswerer
from mlsc_assistant.ingestion.loader import load_documents
from mlsc_assistant.retrieval.retriever import HybridRetriever


@dataclass
class AppState:
    """Everything built once at startup and shared across requests."""

    settings: Settings
    store: VectorStore
    manifest: IndexManifest
    retriever: HybridRetriever
    documents: dict[str, Document]
    _provider: LLMProvider | None = None
    _provider_error: str | None = None
    eval_runs: dict[str, dict[str, Any]] = field(default_factory=dict)

    # -- generation ----------------------------------------------------------

    @property
    def generation_configured(self) -> bool:
        return self.settings.llm.is_configured

    def provider(self) -> LLMProvider:
        """Build the provider on first use, then reuse it.

        Caching matters beyond startup cost: the rate limiter lives on the provider, so
        a fresh instance per request would reset pacing and walk straight into the quota.
        """
        if self._provider is not None:
            return self._provider
        if self._provider_error is not None:
            raise ConfigurationError(self._provider_error)
        try:
            self._provider = make_provider(self.settings)
        except MLSCError as exc:
            self._provider_error = exc.detail
            raise
        return self._provider

    def answerer(self) -> GroundedAnswerer:
        return make_answerer(self.settings, retriever=self.retriever, provider=self.provider())

    # -- knowledge base ------------------------------------------------------

    def current_checksums(self) -> dict[str, str]:
        return {d.doc_id: d.checksum for d in self.documents.values()}

    def index_is_stale(self) -> bool:
        """Whether the knowledge base changed since the index was built.

        Reported by `/v1/health` so a knowledge base edited without re-indexing is
        visible rather than silently serving stale content.
        """
        return self.manifest.is_stale(self.current_checksums())


def build_state(settings: Settings) -> AppState:
    """Load the index and construct the shared collaborators."""
    store, manifest = load_store(settings)
    embedder = make_embedder(settings)
    retriever = make_retriever(settings, embedder=embedder, store=store)
    documents = {
        d.doc_id: d for d in load_documents(settings.kb_path, settings.knowledge_base.glob)
    }
    return AppState(
        settings=settings,
        store=store,
        manifest=manifest,
        retriever=retriever,
        documents=documents,
    )


def get_state(request: Request) -> AppState:
    state: AppState = request.app.state.mlsc
    return state


StateDep = Annotated[AppState, Depends(get_state)]
