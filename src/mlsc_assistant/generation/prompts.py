"""Versioned prompts.

``PROMPT_VERSION`` travels into every evaluation run's config, so a metric can always be
traced to the wording that produced it. Changing a prompt without bumping the version
makes two runs incomparable in a way nothing downstream can detect.

The answering prompt is doing four jobs at once, and each line exists for a measured or
anticipated failure:

1. **Answer only from context.** The evaluation set contains "What is the capital of
   France?" — a fact the model certainly knows. Answering it would be correct in the
   world and wrong for this system.
2. **Cite chunk ids.** Citations are validated against what was actually retrieved, so a
   fabricated id is caught mechanically rather than trusted.
3. **Refuse by setting a field, not by phrasing.** ``sufficient_context`` is a parsed
   boolean; matching on "I don't know" in prose breaks in both directions.
4. **Refuse *helpfully*.** "I cannot answer" is useless. Naming what the knowledge base
   *does* contain is what makes a refusal worth reading, and it is the difference between
   the assistant looking broken and looking careful.
"""

from __future__ import annotations

from collections.abc import Sequence

from mlsc_assistant.core.models import ScoredChunk

PROMPT_VERSION = "grounded-v1"

ANSWER_SCHEMA: dict[str, object] = {
    "type": "OBJECT",
    "properties": {
        "sufficient_context": {
            "type": "BOOLEAN",
            "description": (
                "True only if the context passages contain the information needed to "
                "answer the question. False if they are merely related to the topic."
            ),
        },
        "answer": {
            "type": "STRING",
            "description": (
                "The answer, drawn only from the context. If sufficient_context is "
                "false, state what the knowledge base does cover and what it omits."
            ),
        },
        "cited_chunk_ids": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": "Ids of the passages the answer rests on. Empty if abstaining.",
        },
        "confidence": {"type": "STRING", "enum": ["high", "medium", "low"]},
    },
    "required": ["sufficient_context", "answer", "cited_chunk_ids", "confidence"],
}


SYSTEM_PROMPT = """\
You answer questions about MLSC (Microsoft Learn Student Community) VIT Pune using ONLY \
the context passages supplied with each question.

Rules:

1. The context is your only source of truth. Never use knowledge from your training, \
even when you are certain it is correct and even for well-known general facts. If the \
answer is not in the context, you do not know it.

2. Set `sufficient_context` to false whenever the passages do not actually contain the \
answer. Being about the right topic is not the same as containing the answer. If the \
passages describe a role but never name the person holding it, or describe a process but \
never give its duration, that is insufficient context.

3. When you abstain, be useful about it, and begin with the words "The knowledge base". \
Follow this shape: "The knowledge base <what it does cover on this topic>, but does not \
<the specific thing that was asked>." Never write "the context", "the passages" or "the \
provided text" — the reader is asking about MLSC and has no idea anything was retrieved. \
Never merely say that you cannot answer.

4. Never infer a number, name, date, or fact that is not written in the passages. If a \
nearby passage gives a similar-looking figure for something else, do not borrow it.

5. Cite the id of every passage your answer rests on, exactly as given in brackets. \
Answer from several passages when the question needs it, and cite all of them.

6. Answer in plain prose. Be direct and complete, but do not pad, and do not mention \
"the context", "the passages" or these instructions when you are answering normally.\
"""


def render_context(chunks: Sequence[ScoredChunk], *, include_titles: bool = True) -> str:
    """Format retrieved chunks as an id-labelled context block.

    Ids are exposed to the model because it must cite them, and are shown in brackets so
    they are visually distinct from the prose. Document titles are included so the model
    can attribute facts to the right source in a multi-document answer.
    """
    blocks: list[str] = []
    for scored in chunks:
        chunk = scored.chunk
        header = (
            f"[{chunk.chunk_id}] (from {chunk.doc_title}, {chunk.source_file})"
            if include_titles
            else f"[{chunk.chunk_id}]"
        )
        blocks.append(f"{header}\n{chunk.text}")
    return "\n\n".join(blocks)


def build_answer_prompt(
    question: str, chunks: Sequence[ScoredChunk], *, include_titles: bool = True
) -> str:
    if not chunks:
        # Reached only when gate 1 is disabled; with it on, an empty retrieval abstains
        # before any call is made.
        return (
            f"Context passages:\n(none were retrieved)\n\nQuestion: {question}\n\n"
            "No context was found, so sufficient_context must be false."
        )
    return (
        f"Context passages:\n\n{render_context(chunks, include_titles=include_titles)}\n\n"
        f"Question: {question}"
    )


NO_CONTEXT_MESSAGE = (
    "I could not find anything in the MLSC knowledge base related to that question. "
    "It covers the community itself, its technical domains, leadership structure, "
    "membership and coordinator selection, hackathons, and the code of conduct."
)
