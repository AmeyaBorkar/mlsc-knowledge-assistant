"""Run persistence and Markdown reporting.

Every run writes four files:

``config.json``       what produced the numbers
``metrics.json``      the numbers
``per_question.jsonl`` one trace per question, so any figure is attributable
``report.md``         the human-readable summary

The report always breaks results down **by question type**, never only in aggregate. An
overall recall of 0.9 hiding a collapse on multi-document questions is exactly the
failure this project is judged on, and only the breakdown reveals it.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mlsc_assistant.evaluation.metrics.abstention import ThresholdPoint, best_operating_point
from mlsc_assistant.evaluation.runner import EvalRun

_METRIC_LABELS = {
    "recall_at_k": "Recall@k",
    "precision_at_k": "Precision@k",
    "r_precision": "R-Precision",
    "average_precision": "Avg Precision",
    "mrr": "MRR",
    "ndcg_at_k": "nDCG@k",
    "doc_recall": "Doc recall",
    "all_docs_hit_rate": "All-docs hit",
}


def write_run(directory: Path, run: EvalRun) -> Path:
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

    report_path = directory / "report.md"
    report_path.write_text(render_markdown(run), encoding="utf-8")
    return report_path


def render_markdown(run: EvalRun) -> str:
    m = run.metrics
    primary_k = int(m["primary_k"])
    primary = m["retrieval"][str(primary_k)]
    counts = m["counts"]

    lines: list[str] = [
        f"# Retrieval evaluation — `{run.run_id}`",
        "",
        f"Dataset **{run.dataset}** · {counts['questions']} questions "
        f"({counts['scored']} scored on retrieval, {counts['unanswerable']} unanswerable)",
        "",
        "Retrieval metrics only — no LLM was involved, so these numbers are exactly "
        "reproducible and require no API key.",
        "",
        "## Configuration",
        "",
        "```json",
        json.dumps(run.config, indent=2, ensure_ascii=False),
        "```",
        "",
        f"## Headline (k = {primary_k})",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    for key, label in _METRIC_LABELS.items():
        if key in primary:
            lines.append(f"| {label} | {primary[key]:.3f} |")

    coverage = m.get("multi_doc_coverage")
    if coverage is not None:
        lines.append(
            f"| Multi-doc coverage | {coverage:.3f} ({counts['multi_document']} questions) |"
        )

    lines += ["", "## Across k", "", _k_table(m["retrieval"])]

    lines += [
        "",
        "## By question type",
        "",
        "Aggregates hide the cases that matter. Unanswerable questions are absent here "
        "by design: they have no gold passage, and are measured by abstention instead.",
        "",
        _type_table(m["by_question_type"]),
    ]

    if run.thresholds:
        lines += ["", *_threshold_section(run.thresholds)]

    lines += ["", "## Weakest questions", "", *_weakest(run)]
    return "\n".join(lines) + "\n"


def _k_table(by_k: dict[str, Any]) -> str:
    ks = sorted(by_k, key=int)
    header = "| Metric | " + " | ".join(f"k={k}" for k in ks) + " |"
    sep = "|---" * (len(ks) + 1) + "|"
    rows = [
        "| " + label + " | " + " | ".join(f"{by_k[k].get(key, 0.0):.3f}" for k in ks) + " |"
        for key, label in _METRIC_LABELS.items()
        if any(key in by_k[k] for k in ks)
    ]
    return "\n".join([header, sep, *rows])


def _type_table(by_type: dict[str, Any]) -> str:
    if not by_type:
        return "_No scored questions._"
    keys = ["recall_at_k", "precision_at_k", "r_precision", "mrr", "ndcg_at_k", "all_docs_hit_rate"]
    header = "| Type | " + " | ".join(_METRIC_LABELS[k] for k in keys) + " |"
    sep = "|---" * (len(keys) + 1) + "|"
    rows = [
        f"| {name} | " + " | ".join(f"{values.get(k, 0.0):.3f}" for k in keys) + " |"
        for name, values in by_type.items()
    ]
    return "\n".join([header, sep, *rows])


def _threshold_section(points: list[ThresholdPoint]) -> list[str]:
    """The gate-1 calibration curve.

    Deliberately split by unanswerable subtype: an operating point can look respectable
    in aggregate while catching none of the near misses, and the split is what makes
    that visible rather than flattering.
    """
    lines = [
        "## Abstention gate 1 — threshold sweep",
        "",
        "Gate 1 only: refuse when the best retrieval score falls below the threshold. "
        "No LLM involved, so this measures the *ceiling* of what a similarity threshold "
        "can achieve on its own.",
        "",
        "| Threshold | Abst. P | Abst. R | F1 | Halluc. | Over-refuse | Near-miss R | Off-domain R |",
        "|---|---|---|---|---|---|---|---|",
    ]
    # Only inflection points are listed; 29 near-identical rows would bury the finding.
    shown = [p for i, p in enumerate(points) if i % 2 == 0]
    for p in shown:
        mt = p.metrics
        lines.append(
            f"| {p.threshold:.3f} | {mt.precision:.2f} | {mt.recall:.2f} | {mt.f1:.2f} "
            f"| {mt.hallucination_rate:.2f} | {mt.over_refusal_rate:.2f} "
            f"| {p.near_miss_recall:.2f} | {p.off_domain_recall:.2f} |"
        )

    best = best_operating_point(points, max_over_refusal=0.0)
    lines += ["", "**Best threshold with zero over-refusal:** "]
    if best is None:
        lines[-1] += (
            "none. Every threshold that refuses any unanswerable question also refuses "
            "at least one answerable one — the score distributions overlap completely."
        )
    else:
        mt = best.metrics
        lines[-1] += (
            f"`{best.threshold:.3f}` — catches {mt.recall:.0%} of unanswerable questions "
            f"({best.off_domain_recall:.0%} of off-domain, {best.near_miss_recall:.0%} of "
            f"near-miss) while refusing no answerable question."
        )
    return lines


def _weakest(run: EvalRun, limit: int = 5) -> list[str]:
    scored = [t for t in run.traces if t.scores]
    if not scored:
        return ["_No scored questions._"]

    worst = sorted(scored, key=lambda t: (t.scores or {}).get("recall", 0.0))[:limit]
    lines = ["| Question | Type | Recall | Gold | First relevant rank |", "|---|---|---|---|---|"]
    for t in worst:
        s = t.scores or {}
        rank = s.get("first_relevant_rank")
        lines.append(
            f"| `{t.question_id}` {t.question[:52]} | {t.type} | {s.get('recall', 0.0):.2f} "
            f"| {', '.join(t.gold_chunks) or '—'} | {rank if rank else 'not retrieved'} |"
        )
    return lines
