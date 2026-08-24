"""Command-line interface.

The CLI is a first-class entry point, not a debug harness: `index` and `search` are the
two commands that work with no API key, which makes them the fastest way for a reviewer
to confirm the system reads the knowledge base and retrieves sensibly.
"""

from __future__ import annotations

import sys
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from mlsc_assistant.config import get_settings
from mlsc_assistant.core.errors import MLSCError
from mlsc_assistant.factories import make_chunker, make_embedder, make_store
from mlsc_assistant.ingestion.pipeline import build_index, current_checksums
from mlsc_assistant.stores.numpy_store import NumpyVectorStore

app = typer.Typer(
    name="mlsc",
    help="MLSC Knowledge Assistant — grounded question answering over the MLSC knowledge base.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
err_console = Console(stderr=True)


def _fail(exc: MLSCError) -> None:
    """Render a domain error the way the API would, and exit non-zero.

    `detail` always names the action that fixes the problem, so this stays a one-liner
    rather than a stack trace.
    """
    err_console.print(f"[bold red]{exc.title}[/bold red]: {exc.detail}")
    raise typer.Exit(code=1)


# ---------------------------------------------------------------------------


@app.command()
def index(
    force: Annotated[
        bool, typer.Option("--force", help="Rebuild even if the index is already current.")
    ] = False,
    no_cache: Annotated[
        bool, typer.Option("--no-cache", help="Ignore the embedding cache and re-embed everything.")
    ] = False,
    show_chunks: Annotated[
        bool, typer.Option("--show-chunks", help="Print every chunk after building.")
    ] = False,
) -> None:
    """Build the vector index from the knowledge base. No API key required."""
    settings = get_settings()

    try:
        existing = NumpyVectorStore.read_manifest(settings.index_path)
        if existing is not None and not force:
            if not existing.is_stale(current_checksums(settings)):
                console.print(
                    f"[green]Index is current[/green] — {existing.chunk_count} chunks from "
                    f"{existing.document_count} documents, built {existing.built_at:%Y-%m-%d %H:%M}."
                )
                console.print("[dim]Use --force to rebuild anyway.[/dim]")
                return
            console.print("[yellow]Knowledge base has changed since the last build.[/yellow]")

        with console.status("Loading model and embedding chunks..."):
            result = build_index(
                settings,
                embedder=make_embedder(settings, use_cache=not no_cache),
                chunker=make_chunker(settings),
                store=make_store(settings),
            )
    except MLSCError as exc:
        _fail(exc)
        return

    m = result.manifest
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row("Documents", str(m.document_count))
    table.add_row("Chunks", str(m.chunk_count))
    table.add_row("Embedder", f"{m.embedder} ({m.dimension}d)")
    table.add_row("Chunker", m.chunker_version)
    table.add_row("Elapsed", f"{result.elapsed_s:.2f}s")
    table.add_row("Written to", str(settings.index_path))
    console.print(Panel(table, title="[bold]Index built[/bold]", border_style="green"))

    by_doc: dict[str, int] = {}
    for chunk in result.chunks:
        by_doc[chunk.source_file] = by_doc.get(chunk.source_file, 0) + 1
    breakdown = Table("Document", "Chunks", "Kinds", box=None, padding=(0, 2))
    for doc in result.documents:
        kinds = sorted({c.kind.value for c in result.chunks if c.doc_id == doc.doc_id})
        breakdown.add_row(doc.source_file, str(by_doc.get(doc.source_file, 0)), ", ".join(kinds))
    console.print(breakdown)

    if show_chunks:
        console.print()
        for chunk in result.chunks:
            console.print(
                f"[cyan]{chunk.chunk_id}[/cyan] "
                f"[dim]{chunk.kind.value}, ~{chunk.token_estimate} tokens, "
                f"chars {chunk.char_range[0]}-{chunk.char_range[1]}[/dim]"
            )
            console.print(f"  {chunk.text[:200]}{'...' if len(chunk.text) > 200 else ''}\n")


@app.command()
def info() -> None:
    """Show the resolved configuration and index status."""
    settings = get_settings()

    config_table = Table("Setting", "Value", box=None, padding=(0, 2))
    config_table.add_row("Knowledge base", str(settings.kb_path))
    config_table.add_row("Index", str(settings.index_path))
    config_table.add_row("Embedder", f"{settings.embedding.backend}: {settings.embedding.model}")
    config_table.add_row("Store", settings.store.backend)
    config_table.add_row(
        "Retrieval", f"{settings.retrieval.strategy}, top_k={settings.retrieval.top_k}"
    )
    config_table.add_row("LLM provider", settings.llm.provider)
    config_table.add_row("LLM model", settings.llm.resolved_model())
    # Reports whether generation can run without ever echoing the key.
    configured = "[green]yes[/green]" if settings.llm.is_configured else "[red]no key set[/red]"
    config_table.add_row("LLM configured", configured)
    console.print(Panel(config_table, title="[bold]Configuration[/bold]", border_style="blue"))

    manifest = NumpyVectorStore.read_manifest(settings.index_path)
    if manifest is None:
        console.print("[yellow]No index built yet.[/yellow] Run `mlsc index`.")
        return

    try:
        stale = manifest.is_stale(current_checksums(settings))
    except MLSCError as exc:
        _fail(exc)
        return

    index_table = Table("Field", "Value", box=None, padding=(0, 2))
    index_table.add_row("Built at", f"{manifest.built_at:%Y-%m-%d %H:%M:%S} UTC")
    index_table.add_row("Documents", str(manifest.document_count))
    index_table.add_row("Chunks", str(manifest.chunk_count))
    index_table.add_row("Embedder", f"{manifest.embedder} ({manifest.dimension}d)")
    index_table.add_row("Chunker", manifest.chunker_version)
    index_table.add_row(
        "Status",
        "[red]stale — knowledge base changed, run `mlsc index`[/red]"
        if stale
        else "[green]current[/green]",
    )
    console.print(Panel(index_table, title="[bold]Index[/bold]", border_style="blue"))


def main() -> None:  # pragma: no cover - entry point
    try:
        app()
    except KeyboardInterrupt:
        err_console.print("[dim]Interrupted.[/dim]")
        sys.exit(130)


if __name__ == "__main__":  # pragma: no cover
    main()
