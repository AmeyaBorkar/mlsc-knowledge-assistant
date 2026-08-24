"""Liveness and index status."""

from __future__ import annotations

from fastapi import APIRouter

from mlsc_assistant import __version__
from mlsc_assistant.api.deps import StateDep
from mlsc_assistant.api.schemas import GenerationInfo, HealthResponse, to_index_info

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(state: StateDep) -> HealthResponse:
    """Report readiness, the index manifest, and whether generation is configured.

    `status` is "degraded" rather than "ok" when the knowledge base has changed since
    the index was built, or when no generation key is set — both are states where part
    of the system works and part does not, which a boolean would hide.
    """
    stale = state.index_is_stale()
    configured = state.generation_configured
    return HealthResponse(
        status="ok" if (configured and not stale) else "degraded",
        version=__version__,
        index=to_index_info(state.manifest, stale=stale),
        generation=GenerationInfo(
            provider=state.settings.llm.provider,
            model=state.settings.llm.resolved_model(),
            # Presence only. The key itself is never returned, logged or echoed.
            configured=configured,
        ),
    )
