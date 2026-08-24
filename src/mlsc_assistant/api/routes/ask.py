"""Grounded answering over HTTP.

**Abstention returns 200, not an error.** Refusing correctly is a successful outcome of
this system, and a 4xx would make it indistinguishable from a malformed request — client
retry logic would then treat a correct refusal as something to retry (DECISIONS.md D10).
Clients read ``answered``, never the status code.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from mlsc_assistant.api.deps import StateDep
from mlsc_assistant.api.schemas import AskRequest, AskResponse, to_ask_response
from mlsc_assistant.core.errors import MLSCError, QuestionTooLongError
from mlsc_assistant.core.models import RetrievalStrategy

router = APIRouter(tags=["answering"])


def _validate(request: AskRequest, state: StateDep) -> None:
    limit = state.settings.api.max_question_length
    if len(request.question) > limit:
        raise QuestionTooLongError(
            f"Question is {len(request.question)} characters; the limit is {limit}. "
            "Shorten it, or raise `api.max_question_length`."
        )


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, state: StateDep) -> AskResponse:
    _validate(request, state)
    answer = state.answerer().answer(
        request.question,
        top_k=request.top_k,
        strategy=RetrievalStrategy(request.strategy) if request.strategy else None,
        verify_faithfulness=request.verify_faithfulness,
    )
    return to_ask_response(answer, include_diagnostics=request.include_diagnostics)


@router.post("/ask/stream")
async def ask_stream(request: AskRequest, state: StateDep) -> EventSourceResponse:
    """Server-sent events, so a client can show its sources before the answer exists.

    An honest note on what streams and what does not. Answering is a single
    schema-constrained call (D5) whose fields — ``sufficient_context``,
    ``cited_chunk_ids`` — are what make abstention a parsed decision rather than a string
    match. A partially-received JSON object cannot be interpreted: until the response is
    complete there is no way to know whether the system is answering or refusing, and
    emitting prose from a half-parsed object would mean streaming text that the gates
    might then retract.

    So the ``answer`` event arrives whole. What *does* stream early is ``retrieval``,
    emitted the moment retrieval finishes and before any model call — which is the part
    a user actually benefits from seeing early, since it shows which sources are about to
    be used while the answer is still being produced.

    Streaming genuine tokens would mean a second, unconstrained call after the structured
    one: double the cost and the quota, to stream an answer that already exists.
    """
    _validate(request, state)

    async def events() -> AsyncIterator[dict[str, Any]]:
        try:
            retrieval = state.retriever.retrieve(
                request.question,
                top_k=request.top_k,
                strategy=(RetrievalStrategy(request.strategy) if request.strategy else None),
            )
            yield {
                "event": "retrieval",
                "data": json.dumps(
                    {
                        "strategy": retrieval.strategy.value,
                        "top_dense_score": retrieval.top_dense_score,
                        "documents": list(retrieval.documents_represented),
                        "chunks": [
                            {
                                "chunk_id": sc.chunk.chunk_id,
                                "doc_title": sc.chunk.doc_title,
                                "source_file": sc.chunk.source_file,
                                "score": round(sc.score, 5),
                            }
                            for sc in retrieval.chunks
                        ],
                    }
                ),
            }

            answer = state.answerer().answer(
                request.question,
                top_k=request.top_k,
                strategy=(RetrievalStrategy(request.strategy) if request.strategy else None),
                verify_faithfulness=request.verify_faithfulness,
            )
            payload = to_ask_response(answer, include_diagnostics=request.include_diagnostics)

            yield {
                "event": "answer",
                "data": json.dumps(
                    {
                        "answer": payload.answer,
                        "answered": payload.answered,
                        "abstention_reason": payload.abstention_reason,
                        "confidence": payload.confidence,
                    }
                ),
            }
            yield {
                "event": "citations",
                "data": json.dumps(
                    {
                        "citations": [c.model_dump() for c in payload.citations],
                        "sources": payload.sources,
                    }
                ),
            }
            yield {
                "event": "done",
                "data": json.dumps(
                    {"answered": payload.answered, "diagnostics": payload.diagnostics},
                    default=str,
                ),
            }
        except MLSCError as exc:
            # The stream has already started, so the status code is committed. An error
            # event is the only way left to tell the client what happened.
            yield {"event": "error", "data": json.dumps(exc.to_problem())}

    return EventSourceResponse(events())
