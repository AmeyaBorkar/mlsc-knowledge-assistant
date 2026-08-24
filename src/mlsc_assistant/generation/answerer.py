"""Grounded answering: retrieve, gate, generate, gate again, bind citations.

The pipeline, with the abstention gates marked:

    question
       │
       ├─▶ retrieve (hybrid)
       │
       ├─▶ [gate 1] best cosine below the calibrated threshold?   → abstain, no LLM call
       │
       ├─▶ one schema-constrained call
       │
       ├─▶ [gate 2] model reports sufficient_context = false?     → abstain, helpfully
       │
       ├─▶ citation binding: every cited id must be one we retrieved
       │
       └─▶ [gate 3, optional] post-hoc faithfulness verification
              → answer + citations + diagnostics

Gate 1 is calibrated at 0.55 from the Phase 3 sweep and catches off-domain questions
only. Measured: near-miss unanswerables score 0.71 to 0.78 against answerable
questions' 0.67 to 0.90, so a threshold catching them would refuse 39% of real questions. Gate 2 is
therefore the load-bearing one, and gate 1's job is narrower than it looks — remove
obvious noise for free, never harm a real question.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from mlsc_assistant.config import Settings
from mlsc_assistant.core.errors import MLSCError, StructuredOutputError
from mlsc_assistant.core.models import (
    AbstentionReason,
    Answer,
    Citation,
    Confidence,
    RetrievalResult,
    RetrievalStrategy,
    ScoredChunk,
)
from mlsc_assistant.core.ports import LLMProvider
from mlsc_assistant.generation.prompts import (
    ANSWER_SCHEMA,
    NO_CONTEXT_MESSAGE,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    build_answer_prompt,
)
from mlsc_assistant.generation.verifier import verify_answer
from mlsc_assistant.retrieval.retriever import HybridRetriever

SNIPPET_CHARS = 240


@dataclass(frozen=True, slots=True)
class GateOutcome:
    """What each gate decided, reported on every answer.

    Kept as explicit per-gate results rather than a single verdict so a wrong refusal is
    attributable to one gate instead of to "the system".
    """

    retrieval_gate: str = "skipped"
    context_sufficiency: str = "skipped"
    citation_binding: str = "skipped"
    faithfulness_check: str = "skipped"

    def as_dict(self) -> dict[str, str]:
        return {
            "retrieval_gate": self.retrieval_gate,
            "context_sufficiency": self.context_sufficiency,
            "citation_binding": self.citation_binding,
            "faithfulness_check": self.faithfulness_check,
        }


class GroundedAnswerer:
    """Turns a question into a cited, grounded answer — or a reasoned refusal."""

    def __init__(
        self,
        *,
        retriever: HybridRetriever,
        provider: LLMProvider,
        settings: Settings,
    ) -> None:
        self.retriever = retriever
        self.provider = provider
        self.settings = settings

    # -----------------------------------------------------------------------

    def answer(
        self,
        question: str,
        *,
        top_k: int | None = None,
        strategy: RetrievalStrategy | None = None,
        verify_faithfulness: bool | None = None,
    ) -> Answer:
        started = perf_counter()
        trace_id = uuid.uuid4().hex[:16]
        abstention = self.settings.abstention

        retrieval = self.retriever.retrieve(question, top_k=top_k, strategy=strategy)

        # --- gate 1: free, pre-LLM ------------------------------------------
        gate1 = self._retrieval_gate(retrieval)
        if gate1 is not None:
            return self._abstain(
                question,
                text=NO_CONTEXT_MESSAGE,
                reason=AbstentionReason.NO_RELEVANT_CONTEXT,
                confidence=Confidence.HIGH,
                gates=GateOutcome(retrieval_gate=gate1),
                retrieval=retrieval,
                trace_id=trace_id,
                started=started,
            )

        context = retrieval.chunks[: self.settings.generation.max_context_chunks]

        # --- the single schema-constrained call ------------------------------
        try:
            response = self.provider.complete_structured(
                system=SYSTEM_PROMPT,
                prompt=build_answer_prompt(
                    question, context, include_titles=self.settings.generation.include_doc_titles
                ),
                schema=ANSWER_SCHEMA,
                temperature=self.settings.llm.temperature,
                max_output_tokens=self.settings.llm.max_output_tokens,
            )
        except StructuredOutputError as exc:
            # The model misbehaved rather than the transport failing. Abstaining is the
            # safe reading: passing malformed data downstream risks a confident answer
            # assembled from fields that were never validated.
            return self._abstain(
                question,
                text=(
                    "The answer could not be produced reliably: the model returned a "
                    "malformed response. Please retry."
                ),
                reason=AbstentionReason.PROVIDER_UNAVAILABLE,
                confidence=Confidence.LOW,
                gates=GateOutcome(retrieval_gate="pass", context_sufficiency="error"),
                retrieval=retrieval,
                trace_id=trace_id,
                started=started,
                extra={"error": exc.detail},
            )

        data = response.data
        generation_diag = {
            "provider": self.provider.name,
            "model": self.provider.model,
            "prompt_version": PROMPT_VERSION,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "latency_ms": round(response.latency_ms, 1),
        }

        # --- gate 2: the load-bearing one -----------------------------------
        if abstention.require_sufficient_context and not bool(data.get("sufficient_context")):
            return self._abstain(
                question,
                text=str(data.get("answer") or "").strip() or NO_CONTEXT_MESSAGE,
                reason=AbstentionReason.INSUFFICIENT_CONTEXT,
                confidence=_confidence(data.get("confidence")),
                gates=GateOutcome(retrieval_gate="pass", context_sufficiency="fail"),
                retrieval=retrieval,
                trace_id=trace_id,
                started=started,
                generation=generation_diag,
            )

        # --- citation binding ------------------------------------------------
        citations, invalid = self._bind_citations(data.get("cited_chunk_ids"), context)

        if not citations:
            # An answer claiming sufficient context while citing nothing verifiable
            # cannot satisfy R3 ("provide the relevant source documents"), and there is
            # no way to reconstruct which passages it used. Refusing is the honest
            # outcome; how often this fires is measured rather than assumed.
            return self._abstain(
                question,
                text=(
                    "An answer was drafted but could not be traced to any retrieved "
                    "passage, so it is not reported."
                ),
                reason=AbstentionReason.UNFAITHFUL_ANSWER,
                confidence=Confidence.LOW,
                gates=GateOutcome(
                    retrieval_gate="pass",
                    context_sufficiency="pass",
                    citation_binding="fail",
                ),
                retrieval=retrieval,
                trace_id=trace_id,
                started=started,
                generation=generation_diag,
                extra={"invalid_citations": invalid},
            )

        binding = "pass" if not invalid else "repaired"
        text = str(data.get("answer", "")).strip()
        extra: dict[str, Any] = {"invalid_citations": invalid} if invalid else {}

        # --- gate 3: optional, costs a second call ---------------------------
        wants_verify = (
            verify_faithfulness
            if verify_faithfulness is not None
            else abstention.verify_faithfulness
        )
        verdict = None
        if wants_verify:
            verdict = verify_answer(
                provider=self.provider,
                question=question,
                answer=text,
                passages=self.cited_passages(citations),
                temperature=self.settings.llm.temperature,
            )
            extra["verification"] = verdict.as_dict()

            if not verdict.supported:
                return self._abstain(
                    question,
                    text=(
                        "A draft answer was produced but could not be fully supported by "
                        "the cited passages, so it is not reported."
                    ),
                    reason=AbstentionReason.UNFAITHFUL_ANSWER,
                    confidence=Confidence.LOW,
                    gates=GateOutcome(
                        retrieval_gate="pass",
                        context_sufficiency="pass",
                        citation_binding=binding,
                        faithfulness_check="fail",
                    ),
                    retrieval=retrieval,
                    trace_id=trace_id,
                    started=started,
                    generation=generation_diag,
                    extra=extra,
                )

        gates = GateOutcome(
            retrieval_gate="pass",
            context_sufficiency="pass",
            citation_binding=binding,
            faithfulness_check="pass" if verdict is not None else "skipped",
        )

        return Answer(
            question=question,
            text=text,
            answered=True,
            citations=citations,
            confidence=_confidence(data.get("confidence")),
            diagnostics=self._diagnostics(
                retrieval=retrieval,
                gates=gates,
                trace_id=trace_id,
                started=started,
                generation=generation_diag,
                extra=extra or None,
            ),
        )

    # -----------------------------------------------------------------------

    def _retrieval_gate(self, retrieval: RetrievalResult) -> str | None:
        """Return a failure label, or None to continue.

        Reads ``top_dense_score`` rather than ``score``: under the hybrid strategy the
        latter is an RRF value around 0.03 and scale-free by construction, so comparing
        a calibrated cosine threshold against it would be meaningless.
        """
        cfg = self.settings.abstention

        if not retrieval.chunks:
            return "fail_no_results"

        best = retrieval.top_dense_score
        if best is None:
            # Lexical-only retrieval produces no cosine to threshold. Gate 1 cannot
            # apply, so it defers to gate 2 rather than guessing.
            return None
        if best < cfg.min_dense_score:
            return "fail_below_threshold"
        if cfg.min_score_margin > 0 and retrieval.score_margin < cfg.min_score_margin:
            return "fail_low_margin"
        return None

    def cited_passages(self, citations: Sequence[Citation]) -> list[tuple[str, str]]:
        """Full text of each cited chunk, for anything that judges the answer.

        Citations carry a truncated ``snippet`` for display. Judging against that would
        measure truncation rather than faithfulness, so grading paths resolve the chunk
        and use its whole text.
        """
        return [
            (c.chunk_id, chunk.text)
            for c in citations
            if (chunk := self.retriever.chunks_by_id.get(c.chunk_id)) is not None
        ]

    def _bind_citations(
        self, cited: Any, context: tuple[ScoredChunk, ...]
    ) -> tuple[tuple[Citation, ...], list[str]]:
        """Validate cited ids against what was actually retrieved.

        This is what makes a fabricated citation a mechanical failure rather than
        something a reader has to catch. Unknown ids are dropped and reported in
        diagnostics instead of being silently ignored.
        """
        by_id = {sc.chunk.chunk_id: sc for sc in context}
        wanted = [str(c) for c in cited] if isinstance(cited, list) else []

        citations: list[Citation] = []
        invalid: list[str] = []
        seen: set[str] = set()

        for chunk_id in wanted:
            scored = by_id.get(chunk_id)
            if scored is None:
                invalid.append(chunk_id)
                continue
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            chunk = scored.chunk
            citations.append(
                Citation(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    doc_title=chunk.doc_title,
                    source_file=chunk.source_file,
                    snippet=_snippet(chunk.text),
                    char_range=chunk.char_range,
                    score=scored.score,
                )
            )
        return tuple(citations), invalid

    # -----------------------------------------------------------------------

    def _abstain(
        self,
        question: str,
        *,
        text: str,
        reason: AbstentionReason,
        confidence: Confidence,
        gates: GateOutcome,
        retrieval: RetrievalResult,
        trace_id: str,
        started: float,
        generation: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Answer:
        return Answer(
            question=question,
            text=text,
            answered=False,
            citations=(),
            abstention_reason=reason,
            confidence=confidence,
            diagnostics=self._diagnostics(
                retrieval=retrieval,
                gates=gates,
                trace_id=trace_id,
                started=started,
                generation=generation,
                extra=extra,
            ),
        )

    def _diagnostics(
        self,
        *,
        retrieval: RetrievalResult,
        gates: GateOutcome,
        trace_id: str,
        started: float,
        generation: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        diagnostics: dict[str, Any] = {
            "trace_id": trace_id,
            "retrieval": {
                "strategy": retrieval.strategy.value,
                "top_k": len(retrieval.chunks),
                "candidates_considered": retrieval.candidates_considered,
                "top_dense_score": retrieval.top_dense_score,
                "score_margin": round(retrieval.score_margin, 5),
                "documents_represented": list(retrieval.documents_represented),
                "chunks": [
                    {"chunk_id": sc.chunk.chunk_id, "score": round(sc.score, 5)}
                    for sc in retrieval.chunks
                ],
                "timings_ms": retrieval.timings_ms,
            },
            "gates": gates.as_dict(),
            "abstention_threshold": self.settings.abstention.min_dense_score,
            "total_ms": round((perf_counter() - started) * 1000, 1),
        }
        if generation:
            diagnostics["generation"] = generation
        if extra:
            diagnostics.update(extra)
        return diagnostics


def _confidence(raw: Any) -> Confidence:
    try:
        return Confidence(str(raw).lower())
    except ValueError:
        return Confidence.MEDIUM


def _snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else collapsed[:limit].rstrip() + "..."


def provider_unavailable_answer(question: str, exc: MLSCError) -> Answer:
    """Turn a provider failure into an abstention rather than an exception.

    Used where a caller wants the system to degrade instead of erroring — the CLI and
    the evaluation harness both prefer a recorded refusal to a crashed run.
    """
    return Answer(
        question=question,
        text=f"Answer generation is unavailable: {exc.detail}",
        answered=False,
        abstention_reason=AbstentionReason.PROVIDER_UNAVAILABLE,
        confidence=Confidence.LOW,
        diagnostics={"error": exc.detail, "gates": GateOutcome().as_dict()},
    )
