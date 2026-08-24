"""Per-document diversification.

The lever for multi-document questions (requirement R4). Without it, a question like
"how do domain leads relate to hackathons?" can fill every context slot from
``leadership.txt`` — the document whose vocabulary dominates the query — and silently
drop the hackathon half of the answer. The model then produces a confident, fully
grounded, half-wrong answer, which is the worst failure mode available.

Capping how many chunks one document may contribute forces the second document into the
window. The effect is measured directly by the multi-document coverage metric rather
than assumed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence


def cap_per_document[T](
    ranked: Sequence[T],
    *,
    doc_id_of: Callable[[T], str],
    max_per_document: int,
    limit: int,
) -> list[T]:
    """Take the best ``limit`` items, allowing at most ``max_per_document`` per document.

    Items skipped by the cap are held back and used to backfill if the cap leaves fewer
    than ``limit`` results. This matters: on a question that genuinely concerns one
    document, a strict cap would return three chunks where six were asked for and throw
    away relevant context to satisfy a diversity rule nobody wanted. Diversity is a
    tie-breaker among equally good options, not a reason to return less.

    Relative order is preserved in both passes, so a promoted item never outranks a
    higher-scoring one.
    """
    if max_per_document < 1:
        raise ValueError("max_per_document must be at least 1")

    selected: list[T] = []
    deferred: list[T] = []
    counts: dict[str, int] = {}

    for item in ranked:
        if len(selected) >= limit:
            break
        doc_id = doc_id_of(item)
        if counts.get(doc_id, 0) >= max_per_document:
            deferred.append(item)
            continue
        counts[doc_id] = counts.get(doc_id, 0) + 1
        selected.append(item)

    if len(selected) < limit:
        selected.extend(deferred[: limit - len(selected)])

    return selected
