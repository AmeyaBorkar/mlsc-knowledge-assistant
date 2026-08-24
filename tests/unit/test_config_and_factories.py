"""Configuration layering and adapter selection."""

from __future__ import annotations

from pathlib import Path

import pytest

from mlsc_assistant.config import Settings, get_settings
from mlsc_assistant.core.errors import ConfigurationError
from mlsc_assistant.core.ports import Chunker, Embedder, VectorStore
from mlsc_assistant.embeddings.cache import NullEmbeddingCache
from mlsc_assistant.factories import make_chunker, make_embedder, make_store
from mlsc_assistant.ingestion.chunker import StructuralChunker
from mlsc_assistant.stores.numpy_store import NumpyVectorStore

# --- layering ---------------------------------------------------------------


def test_config_yaml_is_loaded() -> None:
    """The committed config.yaml should be what the app actually runs on."""
    settings = get_settings(reload=True)
    assert settings.embedding.model == "BAAI/bge-small-en-v1.5"
    assert settings.retrieval.strategy == "hybrid"
    assert settings.llm.provider == "gemini"


def test_environment_overrides_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    """An operator must always be able to override a committed value."""
    monkeypatch.setenv("MLSC_RETRIEVAL__TOP_K", "11")
    assert get_settings(reload=True).retrieval.top_k == 11


def test_relative_paths_resolve_against_the_repo_root(tmp_path: Path) -> None:
    """`mlsc index` must behave the same regardless of the working directory."""
    settings = Settings(repo_root=tmp_path)
    assert settings.kb_path == tmp_path / "data" / "knowledge_base"
    assert settings.index_path == tmp_path / "data" / "index"


def test_absolute_paths_are_left_alone(tmp_path: Path) -> None:
    settings = Settings(repo_root=Path("/repo"), store={"path": tmp_path})  # type: ignore[arg-type]
    assert settings.index_path == tmp_path


# --- validation -------------------------------------------------------------


def test_candidate_k_must_cover_top_k() -> None:
    """Fusion cannot return more results than each retriever supplied."""
    with pytest.raises(ValueError, match="candidate_k"):
        Settings(retrieval={"top_k": 20, "candidate_k": 5})  # type: ignore[arg-type]


def test_max_tokens_must_exceed_min_tokens() -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        Settings(chunking={"min_tokens": 100, "max_tokens": 50})  # type: ignore[arg-type]


# --- LLM configuration ------------------------------------------------------


def test_model_falls_back_to_the_per_provider_default() -> None:
    settings = Settings(llm={"provider": "anthropic", "models": {"anthropic": "claude-x"}})  # type: ignore[arg-type]
    assert settings.llm.resolved_model() == "claude-x"


def test_explicit_model_wins() -> None:
    settings = Settings(llm={"provider": "gemini", "model": "chosen", "models": {"gemini": "d"}})  # type: ignore[arg-type]
    assert settings.llm.resolved_model() == "chosen"


def test_is_configured_reports_key_presence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Health reports whether generation can run; it must never echo the key."""
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    settings = Settings(llm={"provider": "gemini", "models": {"gemini": "m"}})  # type: ignore[arg-type]
    assert not settings.llm.is_configured

    monkeypatch.setenv("GOOGLE_API_KEY", "secret-value")
    assert settings.llm.is_configured


def test_ollama_needs_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    settings = Settings(llm={"provider": "ollama", "models": {"ollama": "llama3.1"}})  # type: ignore[arg-type]
    assert settings.llm.is_configured
    assert settings.llm.api_key() is None


def test_blank_key_counts_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """`.env.example` ships `GOOGLE_API_KEY=` — copying it should not read as configured."""
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    settings = Settings(llm={"provider": "gemini", "models": {"gemini": "m"}})  # type: ignore[arg-type]
    assert not settings.llm.is_configured


# --- factories --------------------------------------------------------------


def test_factories_return_port_implementations(settings: Settings) -> None:
    """The point of the ports: nothing downstream needs the concrete class."""
    assert isinstance(make_chunker(settings), Chunker)
    assert isinstance(make_store(settings), VectorStore)
    assert isinstance(make_embedder(settings, use_cache=False), Embedder)


def test_chunker_is_built_from_config() -> None:
    settings = Settings(chunking={"min_tokens": 12, "max_tokens": 99, "prepend_doc_title": False})  # type: ignore[arg-type]
    chunker = make_chunker(settings)
    assert isinstance(chunker, StructuralChunker)
    assert (chunker.min_tokens, chunker.max_tokens, chunker.prepend_doc_title) == (12, 99, False)


def test_store_backend_is_selected_by_config(settings: Settings) -> None:
    assert isinstance(make_store(settings), NumpyVectorStore)


def test_embedder_caching_can_be_disabled(settings: Settings) -> None:
    """`mlsc index --no-cache` must genuinely bypass the cache."""
    embedder = make_embedder(settings, use_cache=False)
    assert isinstance(embedder.cache, NullEmbeddingCache)  # type: ignore[attr-defined]


def test_unknown_store_backend_is_rejected() -> None:
    settings = Settings(store={"backend": "numpy"})  # type: ignore[arg-type]
    object.__setattr__(settings.store, "backend", "redis")
    with pytest.raises(ConfigurationError, match="Unknown vector store backend"):
        make_store(settings)


def test_missing_optional_extra_names_the_install_command(settings: Settings) -> None:
    """A missing optional adapter should say how to fix it, not raise ImportError."""
    object.__setattr__(settings.embedding, "backend", "sbert")
    with pytest.raises(ConfigurationError, match=r"\[sbert\]"):
        make_embedder(settings, use_cache=False)
