"""Structure-aware chunking.

The knowledge base is prose: a title on line 1, then one paragraph per non-empty line,
with two documents containing lists. The chunker follows that structure instead of
imposing a fixed window (DECISIONS.md D9).

The rule that earns its keep is list atomicity. ``domains.txt`` reads:

    The major domains include:

    1. Artificial Intelligence and Machine Learning
    2. Web Development
    ...

A 512-token sliding window would happily cut that list in half, and the brief's own
example question — "What technical domains exist in MLSC?" — would then have no single
chunk that answers it. Keeping the list with its introducing sentence is the difference
between answering that question and not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from mlsc_assistant.core.models import Chunk, ChunkKind, Document

# A list item is a line starting with "1." / "1)" / "-" / "*" / "•".
_LIST_ITEM = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s+")

# A line ending in a colon introduces what follows ("Domain leads are responsible for:").
_INTRODUCES_LIST = re.compile(r":\s*$")

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def estimate_tokens(text: str) -> int:
    """Rough token count.

    Whitespace words times 1.3 approximates subword tokenisation closely enough to
    drive merge and split decisions. A real tokeniser would be more accurate and would
    couple the chunker to a specific model for no benefit at these sizes.
    """
    return int(len(text.split()) * 1.3) + 1


@dataclass(frozen=True, slots=True)
class _Block:
    """A structural unit of the document, before merge/split policy is applied."""

    text: str
    start: int
    end: int
    kind: ChunkKind


def _split_blocks(text: str, *, keep_lists_atomic: bool) -> list[_Block]:
    """Split raw document text into blocks, preserving character offsets.

    Offsets are tracked against the original string throughout rather than
    recomputed later with ``str.find``, which would mis-locate any paragraph whose text
    happens to repeat elsewhere in the document.
    """
    blocks: list[_Block] = []
    lines = text.splitlines(keepends=True)

    offset = 0
    line_spans: list[tuple[str, int, int]] = []
    for line in lines:
        line_spans.append((line, offset, offset + len(line)))
        offset += len(line)

    i = 0
    n = len(line_spans)
    while i < n:
        line, start, end = line_spans[i]
        if not line.strip():
            i += 1
            continue

        is_list_item = bool(_LIST_ITEM.match(line))
        introduces = bool(_INTRODUCES_LIST.search(line.strip()))

        if keep_lists_atomic and (is_list_item or introduces):
            # Absorb the run of list items, plus the sentence introducing it, into one
            # block. Blank lines *between* items are part of the list, so the scan
            # looks past them rather than stopping at the first one.
            group_start = start
            group_end = end
            j = i + 1
            saw_item = is_list_item
            while j < n:
                nxt, _nstart, nend = line_spans[j]
                if not nxt.strip():
                    # A blank line only continues the list if a list item follows it.
                    k = j
                    while k < n and not line_spans[k][0].strip():
                        k += 1
                    if k < n and _LIST_ITEM.match(line_spans[k][0]):
                        j = k
                        continue
                    break
                if _LIST_ITEM.match(nxt):
                    saw_item = True
                    group_end = nend
                    j += 1
                    continue
                break

            if saw_item:
                blocks.append(
                    _Block(
                        text=text[group_start:group_end].strip(),
                        start=group_start,
                        end=group_end,
                        kind=ChunkKind.LIST_BLOCK,
                    )
                )
                i = j
                continue
            # A colon line with no list after it is just a paragraph.

        blocks.append(_Block(text=line.strip(), start=start, end=end, kind=ChunkKind.PARAGRAPH))
        i += 1

    return blocks


_MERGEABLE = (ChunkKind.PARAGRAPH, ChunkKind.MERGED)


def _merge_short(blocks: list[_Block], min_tokens: int) -> list[_Block]:
    """Bring undersized paragraphs up to the token floor.

    A six-word paragraph embeds to a vague, low-information vector that matches
    everything weakly. List blocks are never merged: they are atomic by intent.

    Merging runs in both directions on purpose. A backward-only rule makes the outcome
    depend on where a short block happens to sit rather than on its size — in this
    corpus, "Each technical domain has two domain leads." (10 tokens) survived intact
    only because the paragraph before it was already above the floor, while other short
    paragraphs were merged. That is an accident, not a policy, and inconsistent chunk
    sizing is not something worth defending in an ablation.
    """
    if not blocks:
        return []

    def _join(prev: _Block, nxt: _Block) -> _Block:
        # Span from the first block's start to the second's end, so the combined text
        # is exactly the source slice between them and citation offsets stay valid.
        return _Block(
            text=prev.text + "\n" + nxt.text,
            start=prev.start,
            end=nxt.end,
            kind=ChunkKind.MERGED,
        )

    merged: list[_Block] = []
    buffer: _Block | None = None

    def _close(current: _Block | None) -> None:
        """Emit the buffer, folding it backward if it never reached the floor."""
        if current is None:
            return
        if (
            estimate_tokens(current.text) < min_tokens
            and merged
            and merged[-1].kind in _MERGEABLE
            and current.kind in _MERGEABLE
        ):
            # Happens when a list block interrupts accumulation. Folding into the
            # previous chunk is a single one-off merge, not a new accumulation, so it
            # cannot cascade.
            merged.append(_join(merged.pop(), current))
        else:
            merged.append(current)

    for block in blocks:
        if block.kind is ChunkKind.LIST_BLOCK:
            _close(buffer)
            buffer = None
            merged.append(block)
            continue

        if buffer is None:
            buffer = block
        elif estimate_tokens(buffer.text) < min_tokens:
            # Grow only while still under the floor. Testing the *accumulated* size
            # rather than the incoming block's is what stops a document of uniformly
            # short paragraphs from collapsing into one chunk.
            buffer = _join(buffer, block)
        else:
            merged.append(buffer)
            buffer = block

    _close(buffer)

    # Anything still under the floor could not merge at all: a document's opening
    # paragraph followed immediately by a list, or a lone paragraph between two lists.
    # Left as-is deliberately.
    return merged


def _split_long(block: _Block, max_tokens: int) -> list[_Block]:
    """Split an oversized block at sentence boundaries.

    No block in the supplied corpus is anywhere near ``max_tokens``, so this never fires
    today. It exists so the chunker degrades sensibly rather than emitting a chunk that
    overruns the embedder's 512-token window if the knowledge base grows.
    """
    if estimate_tokens(block.text) <= max_tokens or block.kind is ChunkKind.LIST_BLOCK:
        return [block]

    parts: list[_Block] = []
    cursor = block.start
    buffer: list[str] = []

    for sentence in _SENTENCE_END.split(block.text):
        candidate = " ".join([*buffer, sentence])
        if buffer and estimate_tokens(candidate) > max_tokens:
            joined = " ".join(buffer)
            parts.append(_Block(joined, cursor, cursor + len(joined), block.kind))
            cursor += len(joined) + 1
            buffer = [sentence]
        else:
            buffer.append(sentence)

    if buffer:
        joined = " ".join(buffer)
        parts.append(_Block(joined, cursor, min(cursor + len(joined), block.end), block.kind))

    return parts


class StructuralChunker:
    """Paragraph- and list-aware chunker. Implements ``core.ports.Chunker``."""

    def __init__(
        self,
        *,
        version: str = "structural-v1",
        min_tokens: int = 40,
        max_tokens: int = 320,
        keep_lists_atomic: bool = True,
        prepend_doc_title: bool = True,
    ) -> None:
        self._version = version
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.keep_lists_atomic = keep_lists_atomic
        self.prepend_doc_title = prepend_doc_title

    @property
    def version(self) -> str:
        return self._version

    def chunk(self, document: Document) -> list[Chunk]:
        body_start = self._body_start(document)
        blocks = _split_blocks(document.text, keep_lists_atomic=self.keep_lists_atomic)
        # Drop the title line itself: it is prepended to every chunk anyway, so
        # indexing it alone would add a chunk that matches the document's topic
        # generally while containing no facts.
        blocks = [b for b in blocks if b.start >= body_start]

        blocks = _merge_short(blocks, self.min_tokens)
        expanded: list[_Block] = []
        for block in blocks:
            expanded.extend(_split_long(block, self.max_tokens))

        chunks: list[Chunk] = []
        for index, block in enumerate(expanded):
            text = block.text.strip()
            if not text:
                continue
            embed_text = f"{document.title} - {text}" if self.prepend_doc_title else text
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}::c{index:02d}",
                    doc_id=document.doc_id,
                    doc_title=document.title,
                    source_file=document.source_file,
                    text=text,
                    embed_text=embed_text,
                    char_range=(block.start, block.end),
                    index=index,
                    kind=block.kind,
                    token_estimate=estimate_tokens(text),
                    checksum=Document.compute_checksum(text),
                )
            )
        return chunks

    @staticmethod
    def _body_start(document: Document) -> int:
        """Offset just past the title line."""
        newline = document.text.find("\n")
        return 0 if newline == -1 else newline + 1


def chunk_documents(documents: list[Document], chunker: StructuralChunker) -> list[Chunk]:
    return [chunk for document in documents for chunk in chunker.chunk(document)]
