"""Reporting for the full evaluation run."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from mlsc_assistant.evaluation.full_runner import FullEvalRun


def write_full_run(directory: Path, run: FullEvalRun) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(
        json.dumps(run.config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (directory / "metrics.json").write_text(
        json.dumps(run.metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (directory / "per_question.jsonl").open("w", encoding="utf-8") as fh:
        for trace in run.traces:
            fh.write(json.dumps(asdict(trace), ensure_ascii=False) + "\n")

    path = directory / "report.md"
    path.write_text(render_full_markdown(run), encoding="utf-8")
    return path


def render_full_markdown(run: FullEvalRun) -> str:
    m = run.metrics
    lines: list[str] = [f"# Full evaluation — `{run.run_id}`", ""]

    if not run.complete:
        # An incomplete run must never read as a result. Say so before any number.
        lines += [
            f"> **INCOMPLETE RUN.** Stopped after {len(run.traces)} questions: {run.error}",
            "> The metrics below cover only the questions that ran and are not comparable",
            "> with a full run.",
            "",
        ]

    lines += [
        f"Dataset **{run.dataset}** · {m['counts']['evaluated']} questions evaluated"
        + (f", {m['counts'].get('judged', 0)} judged" if "judged" in m["counts"] else ""),
        "",
        "```json",
        json.dumps(run.config, indent=2, ensure_ascii=False),
        "```",
        "",
    ]

    if gen := m.get("generation"):
        lines += [
            "## Generation quality",
            "",
            "Scored on **answered questions only** — an abstention cites nothing and claims",
            "nothing, so averaging refusals in would let a system that refuses everything",
            "post a perfect faithfulness score.",
            "",
            "| Metric | Score |",
            "|---|---|",
            *(f"| {k.replace('_', ' ').title()} | {v:.3f} |" for k, v in sorted(gen.items())),
            "",
        ]

    if by_type := m.get("generation_by_type"):
        keys = sorted({k for v in by_type.values() for k in v})
        lines += [
            "### By question type",
            "",
            "| Type | " + " | ".join(k.replace("_", " ") for k in keys) + " |",
            "|---" * (len(keys) + 1) + "|",
            *(
                f"| {name} | " + " | ".join(f"{vals.get(k, 0.0):.3f}" for k in keys) + " |"
                for name, vals in sorted(by_type.items())
            ),
            "",
        ]

    if abst := m.get("abstention"):
        lines += [
            "## Abstention",
            "",
            "| Metric | Score |",
            "|---|---|",
            *(f"| {k.replace('_', ' ').title()} | {v:.3f} |" for k, v in sorted(abst.items())),
            "",
        ]

    if sub := m.get("abstention_by_subtype"):
        lines += [
            "### By unanswerable type",
            "",
            "| Kind | Caught | Recall |",
            "|---|---|---|",
            *(
                f"| {name} | {d['caught']}/{d['total']} | {d['recall']:.3f} |"
                for name, d in sorted(sub.items())
            ),
            "",
        ]

    if retr := m.get("retrieval"):
        primary = retr.get(str(m["primary_k"]), {})
        lines += [
            f"## Retrieval (k = {m['primary_k']})",
            "",
            "| Metric | Score |",
            "|---|---|",
            *(f"| {k.replace('_', ' ').title()} | {v:.3f} |" for k, v in sorted(primary.items())),
            "",
        ]

    judge = m.get("judge", {})
    lines += [
        "## Judge",
        "",
        f"Model `{judge.get('model')}` · {judge.get('calls')} calls · "
        f"{judge.get('cache_hits')} cache hits.",
        "",
        "The same model family generates and judges, which risks self-preference bias.",
        "Mitigations: the human spot-check below, and `evaluation.judge.provider`, which",
        "points the judge at a different backend in one line.",
        "",
    ]

    lines += ["## Questions the system got wrong", "", *_failures(run)]
    return "\n".join(lines) + "\n"


def _failures(run: FullEvalRun) -> list[str]:
    rows: list[str] = []
    for t in run.traces:
        wrong_abstention = t.answered != t.answerable
        weak = any(v < 0.7 for v in t.generation_scores.values())
        if not (wrong_abstention or weak):
            continue
        problem = (
            "refused an answerable question"
            if (not t.answered and t.answerable)
            else "answered an unanswerable question"
            if (t.answered and not t.answerable)
            else ", ".join(
                f"{k}={v:.2f}" for k, v in sorted(t.generation_scores.items()) if v < 0.7
            )
        )
        rows.append(f"| `{t.question_id}` | {t.type} | {problem} | {t.question[:60]} |")

    if not rows:
        return ["_Nothing failed._"]
    return ["| Question | Type | Problem | Text |", "|---|---|---|---|", *rows]
