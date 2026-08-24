"""fastembed (ONNX) embedder — the default.

``BAAI/bge-small-en-v1.5`` through ONNX runtime: ~150 MB and no PyTorch, versus roughly
1 GB and a 5-10 s cold start for the same model under sentence-transformers
(DECISIONS.md D3).

One measured property worth stating, because it shapes the abstention design: this
model has a **high similarity floor**. Two entirely unrelated passages from this corpus
(the domain list and the hackathon judging paragraph) embed to cosine 0.65, and even an
off-domain *query* only falls to about 0.43. Scores are therefore meaningful mainly
relative to each other, and any absolute threshold must be calibrated rather than
guessed.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from mlsc_assistant.core.errors import ConfigurationError
from mlsc_assistant.embeddings.cache import NullEmbeddingCache, cache_key

if TYPE_CHECKING:
    from mlsc_assistant.core.ports import EmbeddingCache

# bge models are trained asymmetrically: queries carry a retrieval instruction, passages
# do not. Omitting this costs real recall, and it is the reason `Embedder` has separate
# document and query methods rather than one `embed()`.
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class FastEmbedEmbedder:
    """Implements ``core.ports.Embedder``."""

    def __init__(
        self,
        model: str = "BAAI/bge-small-en-v1.5",
        *,
        dimension: int = 384,
        batch_size: int = 32,
        models_dir: Path | None = None,
        cache: EmbeddingCache | None = None,
        query_prefix: str = _QUERY_PREFIX,
    ) -> None:
        self._model_name = model
        self._dimension = dimension
        self.batch_size = batch_size
        self.query_prefix = query_prefix
        # `cache is None`, not `or`: FileEmbeddingCache defines __len__, so an empty
        # one is falsy and `cache or NullEmbeddingCache()` would silently discard it —
        # disabling caching on exactly the cold-start run that needed it most.
        self.cache: EmbeddingCache = cache if cache is not None else NullEmbeddingCache()
        self._models_dir = models_dir
        self._model: Any | None = None

    @property
    def name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def _ensure_model(self) -> Any:
        """Load lazily.

        Importing fastembed costs a couple of seconds and loading the model more. The
        CLI constructs an embedder for commands that may never embed anything, and
        `--help` should not pay for an ONNX session.
        """
        if self._model is not None:
            return self._model
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - install-time failure
            raise ConfigurationError(
                "fastembed is not installed. Run `pip install -e .`, or set "
                "`embedding.backend: sbert` in config.yaml to use sentence-transformers."
            ) from exc

        kwargs: dict[str, Any] = {"model_name": self._model_name}
        if self._models_dir is not None:
            self._models_dir.mkdir(parents=True, exist_ok=True)
            kwargs["cache_dir"] = str(self._models_dir)

        # Quiet the HF symlink warning on Windows, where symlinks need developer mode.
        # The cache degrades to copies, which is fine for one 130 MB model.
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

        self._model = TextEmbedding(**kwargs)
        actual = self._probe_dimension(self._model)
        if actual != self._dimension:
            raise ConfigurationError(
                f"Embedding model {self._model_name!r} produces {actual}-dimensional "
                f"vectors but config declares {self._dimension}. Fix `embedding.dimension`."
            )
        return self._model

    @staticmethod
    def _probe_dimension(model: Any) -> int:
        return len(next(iter(model.embed(["probe"]))))

    # -- embedding -----------------------------------------------------------

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed passages, consulting the cache first.

        Only cache misses reach the model, and the returned list preserves input order
        regardless of how the misses were batched.
        """
        if not texts:
            return []

        results: list[list[float] | None] = [None] * len(texts)
        misses: list[int] = []

        for i, text in enumerate(texts):
            hit = self.cache.get(cache_key(self._model_name, self._dimension, text))
            if hit is None:
                misses.append(i)
            else:
                results[i] = hit

        if misses:
            model = self._ensure_model()
            missing_texts = [texts[i] for i in misses]
            vectors = list(model.embed(missing_texts, batch_size=self.batch_size))
            for i, vector in zip(misses, vectors, strict=True):
                normalised = _normalise(vector)
                results[i] = normalised
                self.cache.put(cache_key(self._model_name, self._dimension, texts[i]), normalised)

        return [r for r in results if r is not None]

    def embed_query(self, text: str) -> list[float]:
        model = self._ensure_model()
        prefixed = f"{self.query_prefix}{text}"
        vector = next(iter(model.embed([prefixed])))
        return _normalise(vector)


def _normalise(vector: Any) -> list[float]:
    """L2-normalise so that a dot product is cosine similarity.

    fastembed already returns unit vectors for this model, but the ``Embedder`` contract
    promises normalisation and other backends do not all honour it. Normalising here
    keeps the store a plain matrix multiply and keeps scores comparable to the
    calibrated abstention threshold.
    """
    arr = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm > 0:
        arr = arr / norm
    return [float(x) for x in arr]
