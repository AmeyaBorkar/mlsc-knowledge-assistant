"""LLM-as-judge, treated as a measuring instrument.

An instrument needs controls, so this wrapper enforces them in one place rather than
leaving each metric to remember:

**Structured verdicts.** The judge returns a schema, never prose to be regex-matched,
and every verdict carries its reason into the run trace — so a surprising score can be
read rather than guessed at.

**Cached by content.** Keyed on ``(task, prompt_version, inputs)``. Re-running a report
costs nothing, which matters enormously against a free tier of 20 requests per day: a
metric you cannot afford to re-run is a metric you stop checking.

**Temperature 0 and a pinned model**, both recorded in the run config.

The honest caveat, stated where the code is rather than only in the write-up: the same
model family generates and judges, which risks self-preference bias. Mitigations are the
human spot-check in docs/EVALUATION.md and the option to point ``evaluation.judge`` at a
different provider entirely — a one-line config change, because the judge talks to the
``LLMProvider`` port like everything else.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlsc_assistant.core.errors import MLSCError
from mlsc_assistant.core.ports import LLMProvider

JUDGE_PROMPT_VERSION = "judge-v1"


@dataclass(frozen=True, slots=True)
class JudgeVerdict:
    data: dict[str, Any]
    cached: bool = False
    input_tokens: int | None = None
    output_tokens: int | None = None


class VerdictCache:
    """Content-addressed JSON cache of judge verdicts."""

    def __init__(self, path: Path, *, enabled: bool = True) -> None:
        self.path = path
        self.enabled = enabled
        self._entries: dict[str, dict[str, Any]] = {}
        self._dirty = False
        if enabled and path.is_file():
            try:
                self._entries = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                # A corrupt cache is a cost problem, never a correctness one: drop it
                # and re-judge rather than failing the run.
                self._entries = {}

    def get(self, key: str) -> dict[str, Any] | None:
        return self._entries.get(key) if self.enabled else None

    def put(self, key: str, value: dict[str, Any]) -> None:
        if not self.enabled:
            return
        self._entries[key] = value
        self._dirty = True

    def flush(self) -> None:
        if not (self.enabled and self._dirty):
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._entries, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        self._dirty = False

    def __len__(self) -> int:
        return len(self._entries)


def cache_key(task: str, model: str, payload: dict[str, Any]) -> str:
    """Key on everything that could change the verdict.

    The model and prompt version are included deliberately: a cached verdict from a
    different model is not the same measurement, and silently reusing it would make a
    model comparison compare nothing.
    """
    blob = json.dumps(
        {"task": task, "model": model, "version": JUDGE_PROMPT_VERSION, **payload},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


class Judge:
    """Runs judging tasks through a provider, with caching and a single retry policy."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        cache: VerdictCache,
        temperature: float = 0.0,
    ) -> None:
        self.provider = provider
        self.cache = cache
        self.temperature = temperature
        self.calls = 0
        self.cache_hits = 0

    def judge(
        self,
        *,
        task: str,
        system: str,
        prompt: str,
        schema: dict[str, Any],
        payload: dict[str, Any],
    ) -> JudgeVerdict:
        key = cache_key(task, self.provider.model, payload)
        cached = self.cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return JudgeVerdict(data=cached, cached=True)

        try:
            response = self.provider.complete_structured(
                system=system, prompt=prompt, schema=schema, temperature=self.temperature
            )
        except MLSCError:
            # Let the runner decide: a quota failure partway through a run should be
            # visible as an incomplete run, not silently scored as a bad answer.
            raise

        self.calls += 1
        self.cache.put(key, response.data)
        return JudgeVerdict(
            data=response.data,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )
