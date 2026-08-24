"""Retrieval without generation.

**No API key required.** This is the endpoint that makes the system inspectable: when an
answer looks wrong, it separates "retrieval never found it" from "the model mishandled
good context" without re-running anything.
"""

from __future__ import annotations

from fastapi import APIRouter

from mlsc_assistant.api.deps import StateDep
from mlsc_assistant.api.schemas import SearchRequest, SearchResponse, to_search_response
from mlsc_assistant.core.models import RetrievalStrategy

router = APIRouter(tags=["retrieval"])


@router.post("/search", response_model=SearchResponse)
def search(request: SearchRequest, state: StateDep) -> SearchResponse:
    result = state.retriever.retrieve(
        request.query,
        top_k=request.top_k,
        strategy=RetrievalStrategy(request.strategy) if request.strategy else None,
        max_chunks_per_document=request.max_chunks_per_document,
    )
    return to_search_response(result, explain=request.explain)
