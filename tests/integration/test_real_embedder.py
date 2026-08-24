"""Tests that load the real embedding model.

Marked ``integration`` and ``slow`` because the first run downloads ~130 MB. They need
no API key — the whole point of D4 is that retrieval is key-free — so CI can and does
run them.

Run just these with:  pytest -m integration
Skip them with:       pytest -m "not integration"
"""

from __future__ import annotations

import numpy as np
import pytest

from mlsc_assistant.config import Settings
from mlsc_assistant.embeddings.cache import FileEmbeddingCache
from mlsc_assistant.factories import make_chunker, make_embedder, make_store
from mlsc_assistant.ingestion.pipeline import build_index

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture(scope="module")
def embedder():  # type: ignore[no-untyped-def]
    return make_embedder(Settings(), use_cache=False)


def test_vectors_are_normalised(embedder) -> None:  # type: ignore[no-untyped-def]
    """The Embedder contract promises unit vectors, which is what lets the store treat
    a dot product as cosine similarity."""
    vectors = embedder.embed_documents(["MLSC Hackathons - Participants are evaluated."])
    assert float(np.linalg.norm(vectors[0])) == pytest.approx(1.0, abs=1e-5)


def test_dimension_matches_the_declared_config(embedder) -> None:  # type: ignore[no-untyped-def]
    assert len(embedder.embed_query("anything")) == embedder.dimension == 384


def test_queries_are_prefixed_differently_from_passages(embedder) -> None:  # type: ignore[no-untyped-def]
    """bge is asymmetric: dropping the query instruction costs real recall, so the two
    paths must not produce identical vectors for identical text."""
    text = "What technical domains exist in MLSC?"
    assert embedder.embed_query(text) != embedder.embed_documents([text])[0]


def test_unrelated_passages_still_score_high(embedder) -> None:
    """Documents the measured similarity floor that shapes the abstention design.

    bge-small maps two entirely unrelated passages from this corpus to cosine ~0.65.
    Any absolute retrieval threshold must therefore be calibrated: a hand-picked 0.35
    would never fire, and the gate would be decorative. If this assertion ever fails
    because the floor moved, the calibrated threshold needs revisiting too.
    """
    vectors = embedder.embed_documents(
        [
            "MLSC Technical Domains - The major domains include Artificial Intelligence "
            "and Machine Learning, Web Development, App Development, Cloud Computing and Web3.",
            "MLSC Hackathons - Participants are generally evaluated based on innovation, "
            "technical implementation, problem relevance, feasibility and scalability.",
        ]
    )
    similarity = float(np.dot(vectors[0], vectors[1]))
    assert 0.5 < similarity < 0.8, (
        f"Unrelated-passage similarity is {similarity:.3f}. The abstention threshold is "
        "calibrated against this floor; a large shift invalidates it."
    )


def test_semantically_closer_text_scores_higher(embedder) -> None:  # type: ignore[no-untyped-def]
    """Sanity check that the model orders things sensibly at all.

    Absolute scores sit on a compressed scale, so only the *ordering* is asserted —
    which is exactly the property RRF relies on.
    """
    query = embedder.embed_query("How are hackathon projects judged?")
    passages = embedder.embed_documents(
        [
            "MLSC Hackathons - Participants are generally evaluated based on factors such "
            "as innovation, technical implementation, feasibility and quality of presentation.",
            "MLSC Membership - Students can participate in MLSC activities as community members.",
        ]
    )
    assert np.dot(query, passages[0]) > np.dot(query, passages[1])


def test_full_index_build_against_the_real_corpus(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """End-to-end: the real knowledge base, the real model, a reloadable index."""
    settings = Settings(
        store={"path": tmp_path / "index"},  # type: ignore[arg-type]
        embedding={"cache_dir": tmp_path / "cache"},  # type: ignore[arg-type]
    )
    result = build_index(
        settings,
        embedder=make_embedder(settings),
        chunker=make_chunker(settings),
        store=make_store(settings),
    )

    assert result.manifest.chunk_count == 18
    assert result.manifest.dimension == 384

    reloaded = make_store(settings)
    reloaded.load()
    hits = reloaded.search(make_embedder(settings).embed_query("Web3 blockchain"), k=3)
    assert hits[0][0].doc_id == "domains"


def test_cache_makes_a_rebuild_free(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The evaluation harness rebuilds the index once per ablation; without a working
    cache those runs would re-embed the corpus every time."""
    settings = Settings(
        store={"path": tmp_path / "index"},  # type: ignore[arg-type]
        embedding={"cache_dir": tmp_path / "cache"},  # type: ignore[arg-type]
    )
    build_index(
        settings,
        embedder=make_embedder(settings),
        chunker=make_chunker(settings),
        store=make_store(settings),
    )
    assert len(FileEmbeddingCache(settings.embedding_cache_path)) == 18
