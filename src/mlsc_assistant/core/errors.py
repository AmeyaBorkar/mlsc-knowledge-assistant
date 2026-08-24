"""Typed domain errors.

Each error carries what the HTTP edge needs to render an RFC 9457 ``problem+json``
response, so ``api/`` never has to guess a status code from an exception type.

``detail`` should always state the *action* that fixes the problem. "Index not built"
is a diagnosis; "run ``mlsc index`` first" is useful.
"""

from __future__ import annotations

from typing import Any


class MLSCError(Exception):
    """Base class. Everything raised deliberately by this system inherits from it."""

    status: int = 500
    type_slug: str = "internal-error"
    title: str = "Internal error"

    def __init__(self, detail: str, **extra: Any) -> None:
        super().__init__(detail)
        self.detail = detail
        self.extra = extra

    def to_problem(self, trace_id: str | None = None) -> dict[str, Any]:
        problem: dict[str, Any] = {
            "type": f"https://mlsc-assistant/errors/{self.type_slug}",
            "title": self.title,
            "status": self.status,
            "detail": self.detail,
        }
        if trace_id:
            problem["trace_id"] = trace_id
        problem.update(self.extra)
        return problem


# --- configuration and startup ---------------------------------------------


class ConfigurationError(MLSCError):
    status = 500
    type_slug = "configuration-error"
    title = "Configuration error"


class IndexNotBuiltError(MLSCError):
    status = 409
    type_slug = "index-not-built"
    title = "Index not built"


class KnowledgeBaseEmptyError(MLSCError):
    status = 409
    type_slug = "knowledge-base-empty"
    title = "Knowledge base is empty"


# --- request validation -----------------------------------------------------


class InvalidRequestError(MLSCError):
    status = 400
    type_slug = "invalid-request"
    title = "Invalid request"


class QuestionTooLongError(MLSCError):
    status = 422
    type_slug = "question-too-long"
    title = "Question too long"


class DocumentNotFoundError(MLSCError):
    status = 404
    type_slug = "document-not-found"
    title = "Document not found"


# --- provider failures ------------------------------------------------------
# Split into three statuses rather than one generic 500 because "you have not set a
# key", "you are being throttled" and "the upstream is down" each need a different
# reaction from a caller.


class ProviderUnavailableError(MLSCError):
    status = 503
    type_slug = "provider-unavailable"
    title = "Generation provider unavailable"


class ProviderRateLimitedError(MLSCError):
    status = 429
    type_slug = "provider-rate-limited"
    title = "Generation provider rate limited"

    def __init__(self, detail: str, retry_after: int | None = None, **extra: Any) -> None:
        super().__init__(detail, **extra)
        self.retry_after = retry_after


class ProviderTimeoutError(MLSCError):
    status = 504
    type_slug = "provider-timeout"
    title = "Generation provider timed out"


class StructuredOutputError(MLSCError):
    """The provider returned something that does not match the requested schema.

    Distinct from a transport failure: it means the model misbehaved, and the answerer
    should abstain rather than pass malformed data downstream.
    """

    status = 502
    type_slug = "structured-output-error"
    title = "Malformed structured output"
