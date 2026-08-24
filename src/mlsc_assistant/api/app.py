"""FastAPI application factory.

Two things worth pointing at.

**The index loads once, in the lifespan.** Building the embedder and BM25 index per
request would make every call pay a multi-second cold start.

**Errors are RFC 9457 `application/problem+json`.** Domain errors already carry the
status and a `detail` naming the action that fixes them, so the handler is a translation
rather than a place where status codes get invented.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from mlsc_assistant import __version__
from mlsc_assistant.api.deps import build_state
from mlsc_assistant.api.routes import ask, documents, evaluation, health, search
from mlsc_assistant.config import Settings, get_settings
from mlsc_assistant.core.errors import IndexNotBuiltError, MLSCError

logger = logging.getLogger("mlsc")

PROBLEM_MEDIA_TYPE = "application/problem+json"

DESCRIPTION = """\
A grounded, citation-first assistant over the MLSC knowledge base.

Every answer is traceable to the passage it came from, and the system refuses when the
knowledge base does not contain the answer rather than inventing something plausible.

**Abstention is a `200`, not an error.** Read `answered`, not the status code.

**Retrieval needs no API key.** `/v1/search`, `/v1/documents` and `/v1/health` work
without one; only `/v1/ask` requires generation credentials.\
"""


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            app.state.mlsc = build_state(resolved)
            logger.info(
                "index loaded: %s chunks from %s documents",
                app.state.mlsc.manifest.chunk_count,
                app.state.mlsc.manifest.document_count,
            )
        except IndexNotBuiltError:
            # Fail loudly at startup rather than returning a confusing 500 on the first
            # request. The message already names the command that fixes it.
            logger.error("no index found; run `mlsc index` before serving")
            raise
        yield

    app = FastAPI(
        title="MLSC Knowledge Assistant",
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
        openapi_tags=[
            {"name": "answering", "description": "Grounded answers with citations."},
            {"name": "retrieval", "description": "Retrieval only. No API key required."},
            {"name": "documents", "description": "The knowledge base, browsable."},
            {"name": "evaluation", "description": "Evaluation runs."},
            {"name": "health", "description": "Liveness and index status."},
        ],
    )

    if resolved.api.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=resolved.api.cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )

    @app.middleware("http")
    async def attach_trace_id(request: Request, call_next: Any) -> Any:
        """One id per request, echoed on responses and errors.

        Makes a problem response tie back to the log line that produced it, which is the
        difference between a reported bug being reproducible and being a story.
        """
        trace_id = request.headers.get("X-Trace-Id") or uuid.uuid4().hex[:16]
        request.state.trace_id = trace_id
        response = await call_next(request)
        response.headers["X-Trace-Id"] = trace_id
        return response

    @app.exception_handler(MLSCError)
    async def handle_domain_error(request: Request, exc: MLSCError) -> JSONResponse:
        trace_id = getattr(request.state, "trace_id", None)
        headers: dict[str, str] = {}
        retry_after = getattr(exc, "retry_after", None)
        if retry_after:
            headers["Retry-After"] = str(retry_after)
        logger.warning("%s: %s [%s]", exc.title, exc.detail, trace_id)
        return JSONResponse(
            status_code=exc.status,
            content=exc.to_problem(trace_id),
            media_type=PROBLEM_MEDIA_TYPE,
            headers=headers,
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Render FastAPI's validation failures in the same problem+json shape.

        Without this a client would meet two different error formats depending on
        whether the request was malformed or the system was.
        """
        return JSONResponse(
            status_code=422,
            content={
                "type": "https://mlsc-assistant/errors/invalid-request",
                "title": "Invalid request",
                "status": 422,
                "detail": "The request body did not match the expected schema.",
                "errors": [
                    {"field": ".".join(str(p) for p in e["loc"][1:]), "message": e["msg"]}
                    for e in exc.errors()
                ],
                "trace_id": getattr(request.state, "trace_id", None),
            },
            media_type=PROBLEM_MEDIA_TYPE,
        )

    for module in (health, ask, search, documents, evaluation):
        app.include_router(module.router, prefix="/v1")

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {
            "name": "MLSC Knowledge Assistant",
            "version": __version__,
            "docs": "/docs",
            "health": "/v1/health",
        }

    return app


# Module-level instance for `uvicorn mlsc_assistant.api.app:app`. Constructing the
# FastAPI object is cheap; the index is loaded by the lifespan at startup, not here, so
# importing this module does not read the disk or load a model.
app = create_app()
