"""Lexical retrieval: BM25 over stemmed tokens.

This half of the hybrid is load-bearing rather than legacy. The corpus contains rare
exact terms — ``Web3``, ``Technical Head``, ``second-year coordinators`` — that a small
dense embedder blurs into neighbouring concepts, and with 18 chunks there is no
redundancy to recover a miss (DECISIONS.md D1).

Tokenisation choices, all of which matter on a corpus this small:

``[a-z0-9]+``
    Keeps ``Web3`` as one token. Splitting on digit boundaries would turn the one term
    that uniquely identifies a whole domain into the stopword-ish ``web`` plus ``3``.
Snowball stemming
    ``hackathons`` and ``hackathon`` must match; so must ``coordinators`` and
    ``coordinating``. Without stemming these are simply different terms.
Stopword removal
    IDF already flattens corpus-ubiquitous words, so this is mostly redundant for
    *scoring*. It is kept for ``matched_terms``: without it every query's explain
    output is dominated by "the", "of", "a", which makes the diagnostic useless.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from functools import cache, lru_cache
from typing import Any

from rank_bm25 import BM25Okapi

from mlsc_assistant.core.models import Chunk

_TOKEN = re.compile(r"[a-z0-9]+")

# Deliberately small and generic. A larger, hand-curated list risks dropping something
# meaningful from an 18-chunk corpus where almost every content word carries signal.
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "may",
        "might",
        "must",
        "of",
        "on",
        "or",
        "should",
        "so",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
    ]
)


@cache
def _stemmer() -> Any:
    """The Snowball stemmer, or None if it is unavailable.

    Returning None rather than raising means lexical retrieval degrades to unstemmed
    matching instead of taking the whole system down over an optional-feeling detail.
    """
    try:
        from py_rust_stemmers import SnowballStemmer
    except ImportError:  # pragma: no cover - the dependency is declared in pyproject
        return None
    return SnowballStemmer("english")


@lru_cache(maxsize=8192)
def _stem(token: str) -> str:
    """Snowball stem, cached.

    The corpus is stemmed once at construction and every query re-stems its terms, so
    the cache makes repeated queries free.
    """
    stemmer = _stemmer()
    return token if stemmer is None else str(stemmer.stem_word(token))


def tokenize(text: str, *, remove_stopwords: bool = True, stem: bool = True) -> list[str]:
    """Normalise text into BM25 terms."""
    tokens = _TOKEN.findall(text.lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in _STOPWORDS]
    if stem:
        tokens = [_stem(t) for t in tokens]
    return tokens


class BM25Retriever:
    """BM25 over the same text the dense retriever embeds.

    Indexing ``embed_text`` (document title + chunk) rather than bare ``text`` keeps the
    two retrievers looking at identical input, so any difference in their rankings is
    attributable to the matching algorithm alone — which is what makes the ``explain``
    output and the hybrid-vs-dense ablation interpretable.

    Measured rather than assumed: indexing bare ``text`` instead drops the domain-list
    chunk out of the results entirely for "What technical domains exist in MLSC?" and
    changes nothing else, so the prefix stays. ``index_title=False`` is kept as an
    ablation knob rather than removed.
    """

    def __init__(
        self,
        chunks: Sequence[Chunk],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        stem: bool = True,
        index_title: bool = True,
        max_document_frequency: float = 0.5,
    ) -> None:
        self.chunks = list(chunks)
        self.stem = stem
        self.index_title = index_title
        self.max_document_frequency = max_document_frequency
        self._tokenised = [
            tokenize(c.embed_text if index_title else c.text, stem=stem) for c in self.chunks
        ]
        self._index: BM25Okapi | None = (
            BM25Okapi(self._tokenised, k1=k1, b=b) if self._tokenised else None
        )
        self._term_sets = [set(t) for t in self._tokenised]
        self._doc_freq = Counter(term for terms in self._term_sets for term in terms)

    def discriminative_terms(self, terms: Sequence[str]) -> list[str]:
        """Keep only query terms that actually narrow the corpus.

        A term appearing in more than ``max_document_frequency`` of the chunks carries
        almost no information about *which* chunk is relevant. BM25's IDF down-weights
        such terms but does not eliminate them: Okapi floors negative IDF at a small
        positive epsilon, so a ubiquitous term still contributes a score driven by term
        frequency and length normalisation — that is, by noise.

        Measured consequence of not doing this: "What is MLSC?" reduces to the single
        term ``mlsc``, which appears in 18 of 18 chunks. BM25 then ranks by little more
        than chunk length, and fusing that opinion with dense retrieval's real signal
        pushed the correct chunk out of the results entirely on a question dense had
        answered perfectly.
        """
        ceiling = max(1.0, len(self.chunks) * self.max_document_frequency)
        return [t for t in terms if self._doc_freq.get(t, 0) <= ceiling]

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        if self._index is None:
            return []

        terms = self.discriminative_terms(tokenize(query, stem=self.stem))
        if not terms:
            # No discriminative term: BM25 has no opinion worth fusing, so it abstains
            # and hybrid retrieval degrades to dense for this query.
            return []

        scores = self._index.get_scores(terms)
        ranked = sorted(
            range(len(self.chunks)),
            # chunk_id as the tie-break keeps ordering deterministic; on a corpus this
            # small, exact score ties between chunks are common rather than exotic.
            key=lambda i: (-float(scores[i]), self.chunks[i].chunk_id),
        )
        # BM25 assigns a non-zero score to anything sharing a term, so results are
        # filtered to positives; a zero-score chunk matched nothing and is not a "hit".
        return [(self.chunks[i], float(scores[i])) for i in ranked[:k] if scores[i] > 0]

    def matched_terms(self, query: str, chunk_id: str) -> tuple[str, ...]:
        """Query terms present in a chunk, for the explain output.

        Returns stems, which is what actually matched — reporting the surface form would
        misrepresent why ``hackathons`` scored against a query for ``hackathon``.
        """
        terms = tokenize(query, stem=self.stem)
        for i, chunk in enumerate(self.chunks):
            if chunk.chunk_id == chunk_id:
                seen = self._term_sets[i]
                # dict.fromkeys preserves query order while de-duplicating.
                return tuple(dict.fromkeys(t for t in terms if t in seen))
        return ()
