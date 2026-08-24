"""Dense retrieval: cosine similarity over the embedded corpus."""

from __future__ import annotations

from mlsc_assistant.core.models import Chunk
from mlsc_assistant.core.ports import Embedder, VectorStore


class DenseRetriever:
    """Embeds the query and searches the vector store.

    Thin by design. Vectors are L2-normalised at embed time, so the store's dot product
    is already cosine similarity and there is nothing to convert or rescale here.
    """

    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self.embedder = embedder
        self.store = store

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        # embed_query, not embed_documents: bge is asymmetric and prepends a retrieval
        # instruction to queries only. Using the passage path here costs real recall.
        return self.store.search(self.embedder.embed_query(query), k)
