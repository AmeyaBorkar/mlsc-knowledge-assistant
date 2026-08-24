"""The `mlsc eval` command.

Kept out of ``cli.py`` so the evaluation harness — the heaviest part of the tree — is
not imported by `mlsc index` or `mlsc search`.
"""

from __future__ import annotations

from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from mlsc_assistant.config import Settings, get_settings
from mlsc_assistant.core.errors import InvalidRequestError, MLSCError
from mlsc_assistant.core.models import RetrievalStrategy
from mlsc_assistant.evaluation.dataset import resolve_dataset, validate_against_index
from mlsc_assistant.evaluation.metrics.abstention import best_operating_point
from mlsc_assistant.evaluation.report import write_run
from mlsc_assistant.evaluation.runner import run_retrieval_evaluation
from mlsc_assistant.factories import load_store, make_embedder, make_retriever

console = Console()
err_console = Console(stderr=True)


def _parse_fail_under(raw: list[str]) -> dict[str, float]:
    """Parse ``--fail-under recall_at_k=0.80`` pairs into a threshold map."""
    thresholds: dict[str, float] = {}
    for item in raw:
        if "=" not in item:
            raise InvalidRequestError(
                f"--fail-under expects metric=value, got {item!r} (for example: recall_at_k=0.80)."
            )
        metric, _, value = item.partition("=")
        try:
            thresholds[metric.strip()] = float(value)
        except ValueError as exc:
            raise InvalidRequestError(
                f"--fail-under value for {metric!r} is not a number: {value!r}."
            ) from exc
    return thresholds


def eval_command(
    dataset: Annotated[
        str, typer.Option("--dataset", "-d", help="Evaluation set name under evaluation/datasets.")
    ] = "",
    strategy: Annotated[
        str, typer.Option("--strategy", "-s", help="hybrid | dense | lexical")
    ] = "",
    metrics: Annotated[
        str,
        typer.Option(
            "--metrics",
            help="retrieval | all | comma-separated subset. Beyond retrieval needs a key.",
        ),
    ] = "retrieval",
    limit: Annotated[
        int, typer.Option("--limit", help="Evaluate only the first N questions. Saves quota.")
    ] = 0,
    compare: Annotated[
        bool, typer.Option("--compare", help="Run all three strategies and tabulate them.")
    ] = False,
    fail_under: Annotated[
        list[str] | None,
        typer.Option(
            "--fail-under", help="Exit non-zero if a metric falls below, e.g. recall_at_k=0.8"
        ),
    ] = None,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Print only the summary.")] = False,
) -> None:
    """Evaluate the system against an evaluation set.

    `--metrics retrieval` (the default) runs entirely offline and is what CI gates on.
    `--metrics all` additionally answers every question and judges the answers, costing
    roughly one call per question plus three per answered question. Judge verdicts are
    cached by content, so a repeat run is nearly free.
    """
    settings = get_settings()

    families = _parse_families(metrics)
    if families is None:
        err_console.print(
            f"[bold red]Invalid request[/bold red]: --metrics {metrics!r} is not recognised. "
            "Use 'retrieval', 'all', or a comma-separated subset of "
            "retrieval,generation,abstention."
        )
        raise typer.Exit(code=1)

    if "generation" in families:
        _run_full(settings, dataset, strategy, families, limit, fail_under or [])
        return

    try:
        thresholds = _parse_fail_under(fail_under or [])
        data = resolve_dataset(
            settings.dataset_path, dataset or settings.evaluation.default_dataset
        )
        store, manifest = load_store(settings)
        validate_against_index(
            data,
            known_chunk_ids=[c.chunk_id for c in store.all_chunks()],
            known_doc_ids={c.doc_id for c in store.all_chunks()},
        )
        embedder = make_embedder(settings)
        retriever = make_retriever(settings, embedder=embedder, store=store)

        chosen: list[RetrievalStrategy | None] = (
            [RetrievalStrategy(s) for s in ("dense", "lexical", "hybrid")]
            if compare
            else [RetrievalStrategy(strategy) if strategy else None]
        )
    except MLSCError as exc:
        err_console.print(f"[bold red]{exc.title}[/bold red]: {exc.detail}")
        raise typer.Exit(code=1) from exc
    except ValueError as exc:
        err_console.print(f"[bold red]Invalid request[/bold red]: {exc}")
        raise typer.Exit(code=1) from exc

    if not quiet:
        console.print(
            f"[bold]{data.name}[/bold] · {len(data)} questions · "
            f"{', '.join(f'{k}:{v}' for k, v in data.composition().items())}"
        )
        console.print(f"[dim]index: {manifest.chunk_count} chunks, {manifest.embedder}[/dim]\n")

    runs = []
    for strat in chosen:
        with console.status(
            f"Evaluating {strat.value if strat else settings.retrieval.strategy}..."
        ):
            run = run_retrieval_evaluation(
                data, retriever, settings, manifest=manifest, strategy=strat
            )
        report_path = write_run(settings.runs_path / run.run_id, run)
        runs.append((run, report_path))

    _render(runs, quiet=quiet)

    last_run = runs[-1][0]
    primary = last_run.metrics["retrieval"][str(last_run.metrics["primary_k"])]
    failures = [
        f"{metric} = {primary.get(metric, 0.0):.3f} < {minimum:.3f}"
        for metric, minimum in thresholds.items()
        if primary.get(metric, 0.0) < minimum
    ]
    if failures:
        err_console.print("\n[bold red]Quality gate failed[/bold red]:")
        for failure in failures:
            err_console.print(f"  {failure}")
        raise typer.Exit(code=1)


def _render(runs: list, *, quiet: bool) -> None:  # type: ignore[type-arg]
    keys = [
        ("recall_at_k", "Recall@k"),
        ("precision_at_k", "Prec@k"),
        ("r_precision", "R-Prec"),
        ("mrr", "MRR"),
        ("ndcg_at_k", "nDCG@k"),
        ("doc_recall", "DocRec"),
    ]

    table = Table("Strategy", *[label for _, label in keys], "MultiDoc", box=None, padding=(0, 2))
    for run, _ in runs:
        primary = run.metrics["retrieval"][str(run.metrics["primary_k"])]
        coverage = run.metrics.get("multi_doc_coverage")
        table.add_row(
            run.config["strategy"],
            *[f"{primary.get(key, 0.0):.3f}" for key, _ in keys],
            "n/a" if coverage is None else f"{coverage:.3f}",
        )
    console.print(table)

    if quiet:
        return

    run = runs[-1][0]
    by_type = run.metrics["by_question_type"]
    if by_type:
        console.print("\n[bold]By question type[/bold] (last strategy)")
        type_table = Table(
            "Type", "Recall@k", "Prec@k", "MRR", "All-docs", box=None, padding=(0, 2)
        )
        for name, values in by_type.items():
            type_table.add_row(
                name,
                f"{values.get('recall_at_k', 0.0):.3f}",
                f"{values.get('precision_at_k', 0.0):.3f}",
                f"{values.get('mrr', 0.0):.3f}",
                f"{values.get('all_docs_hit_rate', 0.0):.3f}",
            )
        console.print(type_table)

    best = best_operating_point(run.thresholds, max_over_refusal=0.0)
    console.print("\n[bold]Abstention gate 1[/bold] (threshold only, no LLM)")
    if best is None:
        console.print(
            "  [yellow]No threshold refuses any unanswerable question without also "
            "refusing an answerable one.[/yellow]"
        )
    else:
        m = best.metrics
        console.print(
            f"  best zero-over-refusal threshold [bold]{best.threshold:.3f}[/bold] — "
            f"catches {m.recall:.0%} of unanswerable "
            f"([green]{best.off_domain_recall:.0%}[/green] off-domain, "
            f"[yellow]{best.near_miss_recall:.0%}[/yellow] near-miss)"
        )

    for _, path in runs:
        console.print(f"[dim]report: {path}[/dim]")


_VALID_FAMILIES = ("retrieval", "generation", "abstention")


def _parse_families(raw: str) -> list[str] | None:
    """Resolve --metrics into a family list, or None if unrecognised."""
    if raw.strip() == "all":
        return list(_VALID_FAMILIES)
    families = [f.strip() for f in raw.split(",") if f.strip()]
    if not families or any(f not in _VALID_FAMILIES for f in families):
        return None
    return families


def _run_full(
    settings: Settings,
    dataset: str,
    strategy: str,
    families: list[str],
    limit: int,
    fail_under: list[str],
) -> None:
    """The quota-spending path: answer every question, then judge the answers."""
    from mlsc_assistant.evaluation.full_report import write_full_run
    from mlsc_assistant.evaluation.full_runner import run_full_evaluation
    from mlsc_assistant.evaluation.judge import Judge, VerdictCache
    from mlsc_assistant.factories import make_answerer, make_provider

    try:
        thresholds = _parse_fail_under(fail_under)
        data = resolve_dataset(
            settings.dataset_path, dataset or settings.evaluation.default_dataset
        )
        store, manifest = load_store(settings)
        validate_against_index(
            data,
            known_chunk_ids=[c.chunk_id for c in store.all_chunks()],
            known_doc_ids={c.doc_id for c in store.all_chunks()},
        )
        embedder = make_embedder(settings)
        retriever = make_retriever(settings, embedder=embedder, store=store)
        provider = make_provider(settings)
        answerer = make_answerer(settings, retriever=retriever, provider=provider)
        judge = Judge(
            # The judge shares the answering provider, so it shares the rate limiter too.
            # Two independently paced clients would blow the quota together while each
            # believed it was behaving.
            provider,
            cache=VerdictCache(
                settings.runs_path / "judge-cache.json",
                enabled=settings.evaluation.judge.cache,
            ),
            temperature=settings.evaluation.judge.temperature,
        )
    except MLSCError as exc:
        err_console.print(f"[bold red]{exc.title}[/bold red]: {exc.detail}")
        raise typer.Exit(code=1) from exc

    total = min(limit, len(data)) if limit else len(data)
    console.print(f"[bold]{data.name}[/bold] - {total} questions - families: {', '.join(families)}")
    console.print(
        f"[dim]answering with {provider.model}, judging with {judge.provider.model}[/dim]"
    )
    console.print()

    seen = {"n": 0}

    def on_question(question: Any, trace: Any) -> None:
        seen["n"] += 1
        mark = "ok " if (trace.answered == question.answerable) else "BAD"
        # Distinct labels: a 4-character truncation rendered answer_correctness and
        # answer_relevancy identically as "answ".
        labels = {
            "faithfulness": "faith",
            "answer_relevancy": "relev",
            "answer_correctness": "correct",
        }
        scores = " ".join(
            f"{labels.get(k, k)}={v:.2f}" for k, v in sorted(trace.generation_scores.items())
        )
        console.print(
            f"  [{seen['n']:>2}/{total}] {mark} {question.id} "
            f"[dim]{question.type.value:<15}[/dim] {scores}"
        )

    run = run_full_evaluation(
        data,
        answerer,
        embedder,
        judge,
        settings,
        manifest=manifest,
        strategy=RetrievalStrategy(strategy) if strategy else None,
        families=families,
        limit=limit or None,
        progress=on_question,
    )

    report = write_full_run(settings.runs_path / run.run_id, run)
    _render_full(run, report)

    if not run.complete:
        # An incomplete run is neither a pass nor a metric failure - it is a run that did
        # not happen. A distinct exit code stops it being mistaken for either.
        raise typer.Exit(code=2)

    generation = run.metrics.get("generation", {})
    failures = [
        f"{metric} = {generation.get(metric, 0.0):.3f} < {minimum:.3f}"
        for metric, minimum in thresholds.items()
        if generation.get(metric, 0.0) < minimum
    ]
    if failures:
        err_console.print()
        err_console.print("[bold red]Quality gate failed[/bold red]:")
        for failure in failures:
            err_console.print(f"  {failure}")
        raise typer.Exit(code=1)


def _render_full(run: Any, report: Any) -> None:
    if not run.complete:
        console.print()
        console.print(
            f"[bold yellow]Incomplete run[/bold yellow] - stopped after "
            f"{len(run.traces)} questions."
        )
        console.print(f"  {run.error}")

    for label, key in (("Generation metric", "generation"), ("Abstention metric", "abstention")):
        values = run.metrics.get(key)
        if not values:
            continue
        table = Table(label, "Score", box=None, padding=(0, 2))
        for name, value in sorted(values.items()):
            table.add_row(name.replace("_", " "), f"{value:.3f}")
        console.print()
        console.print(table)

    if sub := run.metrics.get("abstention_by_subtype"):
        console.print()
        for name, detail in sorted(sub.items()):
            console.print(f"  {name}: caught {detail['caught']}/{detail['total']}")

    stats = run.metrics.get("judge", {})
    console.print()
    console.print(
        f"[dim]judge: {stats.get('calls')} calls, {stats.get('cache_hits')} cache hits[/dim]"
    )
    console.print(f"[dim]report: {report}[/dim]")
