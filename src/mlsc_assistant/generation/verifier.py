"""Abstention gate 3: post-hoc faithfulness verification.

Gates 1 and 2 both ask questions *before* the answer exists — is anything relevant here,
and does the context suffice. Neither can catch an answer that had good context and
drifted beyond it anyway: a correct-looking sentence built from the model's own
knowledge, or a number borrowed from an adjacent fact.

This gate re-reads the finished answer against only the passages it cited and asks
whether each claim is actually supported. It costs a second call, so it is off by
default and enabled per request or via ``abstention.verify_faithfulness``.

Deliberately separate from the faithfulness *metric* in Phase 7. The metric scores a run
after the fact; this changes what the user is shown. Same idea, different job — and
keeping them apart means the metric can grade this gate rather than agreeing with itself
by construction.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from mlsc_assistant.core.errors import StructuredOutputError
from mlsc_assistant.core.ports import LLMProvider

VERIFIER_PROMPT_VERSION = "verify-v1"

VERIFY_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "unsupported_claims": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": (
                "Claims in the answer that the passages do not support. Empty if every "
                "claim is supported."
            ),
        },
        "supported": {
            "type": "BOOLEAN",
            "description": "True only if every claim in the answer is supported.",
        },
        "reason": {"type": "STRING", "description": "One sentence explaining the verdict."},
    },
    "required": ["unsupported_claims", "supported", "reason"],
}

VERIFY_SYSTEM = """\
You check whether an answer is fully supported by the passages it cites.

Split the answer into individual factual claims. For each, decide whether the passages \
state it or directly entail it. Judge only support, never whether the claim is true in \
the wider world — an accurate statement that the passages do not contain is unsupported, \
and that is exactly what you are looking for.

Paraphrase is fine. Added specifics are not: a number, name, date or qualifier that does \
not appear in the passages is unsupported even if it sounds plausible.

List every unsupported claim. Set supported to true only when the list is empty.\
"""


@dataclass(frozen=True, slots=True)
class VerificationResult:
    supported: bool
    unsupported_claims: tuple[str, ...]
    reason: str
    latency_ms: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "supported": self.supported,
            "unsupported_claims": list(self.unsupported_claims),
            "reason": self.reason,
            "prompt_version": VERIFIER_PROMPT_VERSION,
            "latency_ms": round(self.latency_ms, 1),
        }


def verify_answer(
    *,
    provider: LLMProvider,
    question: str,
    answer: str,
    passages: Sequence[tuple[str, str]],
    temperature: float = 0.0,
) -> VerificationResult:
    """Check the answer against its own cited passages.

    Checks against *cited* passages rather than everything retrieved. That is the
    stricter reading and the honest one: the citations are what the system claims the
    answer rests on and what the user is shown, so grading against unrelated retrieved
    context would let an answer pass on evidence it never pointed to.

    ``passages`` carries **full chunk text**, not the display snippet. This is not a
    detail: snippets are truncated to 240 characters for rendering, and an earlier
    version passed those here. The verifier then dutifully reported the truncated tail as
    unsupported — "Each technical domain has two domain leads" was judged unsupported by
    the very chunk that ends with that sentence. Verifying against what the user sees is
    the wrong instinct; verify against what the answer was actually built from.
    """
    if not passages:
        return VerificationResult(
            supported=False,
            unsupported_claims=(answer,),
            reason="The answer cites no passages, so nothing supports it.",
        )

    rendered = "\n\n".join(f"[{chunk_id}] {text}" for chunk_id, text in passages)
    prompt = f"Cited passages:\n\n{rendered}\n\nQuestion: {question}\n\nAnswer to check:\n{answer}"

    try:
        response = provider.complete_structured(
            system=VERIFY_SYSTEM,
            prompt=prompt,
            schema=VERIFY_SCHEMA,
            temperature=temperature,
        )
    except StructuredOutputError as exc:
        # A broken verifier must not silently pass the answer through. Failing closed
        # would also be wrong — it would refuse a possibly-fine answer over a transport
        # problem — so the outcome is reported as inconclusive and the caller decides.
        return VerificationResult(
            supported=True,
            unsupported_claims=(),
            reason=f"Verification was inconclusive: {exc.detail}",
        )

    data = response.data
    claims = data.get("unsupported_claims")
    unsupported = tuple(str(c) for c in claims) if isinstance(claims, list) else ()
    return VerificationResult(
        # Trust the claim list over the boolean when they disagree: the list is the
        # model's actual work and the flag is a summary of it.
        supported=bool(data.get("supported")) and not unsupported,
        unsupported_claims=unsupported,
        reason=str(data.get("reason", "")).strip(),
        latency_ms=response.latency_ms,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
    )
