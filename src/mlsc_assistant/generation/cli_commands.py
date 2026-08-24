"""The `mlsc ask` command.

Kept out of ``cli.py`` so `mlsc index` and `mlsc search` never import a provider SDK.
"""

from __future__ import annotations

import json
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from mlsc_assistant.config import get_settings
from mlsc_assistant.core.errors import MLSCError
from mlsc_assistant.core.models import Answer, RetrievalStrategy
from mlsc_assistant.factories import make_answerer

console = Console()
err_console = Console(stderr=True)


def ask_command(
    question: Annotated[str, typer.Argument(help="The question to answer.")],
    top_k: Annotated[int, typer.Option("--top-k", "-k", help="Chunks to pass to the model.")] = 0,
    strategy: Annotated[
        str, typer.Option("--strategy", "-s", help="hybrid | dense | lexical")
    ] = "",
    verify: Annotated[
        bool, typer.Option("--verify", help="Enable gate 3, post-hoc faithfulness check.")
    ] = False,
    show_context: Annotated[
        bool, typer.Option("--show-context", help="Print the passages the model was given.")
    ] = False,
    diagnostics: Annotated[
        bool, typer.Option("--diagnostics", "-d", help="Print the full diagnostics block.")
    ] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Emit the raw response object.")] = False,
) -> None:
    """Answer a question from the knowledge base, with citations. Requires an API key."""
    settings = get_settings()

    try:
        chosen = RetrievalStrategy(strategy) if strategy else None
    except ValueError:
        err_console.print(
            f"[bold red]Invalid request[/bold red]: unknown strategy {strategy!r}. "
            "Choose hybrid, dense or lexical."
        )
        raise typer.Exit(code=1) from None

    try:
        answerer = make_answerer(settings)
        answer = answerer.answer(
            question,
            top_k=top_k or None,
            strategy=chosen,
            verify_faithfulness=True if verify else None,
        )
    except MLSCError as exc:
        err_console.print(f"[bold red]{exc.title}[/bold red]: {exc.detail}")
        raise typer.Exit(code=1) from exc

    if as_json:
        console.print_json(json.dumps(_serialise(answer)))
        return

    _render(answer, show_context=show_context, diagnostics=diagnostics)


def _render(answer: Answer, *, show_context: bool, diagnostics: bool) -> None:
    if answer.answered:
        console.print(Panel(answer.text, border_style="green", title="[bold]Answer[/bold]"))
        console.print("[bold]Sources[/bold]  " + ", ".join(answer.sources))
        for citation in answer.citations:
            console.print(f"  [cyan]{citation.chunk_id}[/cyan] [dim]{citation.doc_title}[/dim]")
            console.print(f'    [dim]"{citation.snippet}"[/dim]')
    else:
        # Abstention is a successful outcome, not an error, so it is not styled as one.
        console.print(Panel(answer.text, border_style="yellow", title="[bold]No answer[/bold]"))
        reason = answer.abstention_reason.value if answer.abstention_reason else "unknown"
        console.print(f"[bold]Answered[/bold] no  ·  [bold]Reason[/bold] {reason}")

    console.print(f"[dim]confidence: {answer.confidence.value}[/dim]")

    gen = answer.diagnostics.get("generation") or {}
    retrieval = answer.diagnostics.get("retrieval") or {}
    if gen:
        console.print(
            f"[dim]{gen.get('model')} · {gen.get('input_tokens')} in / "
            f"{gen.get('output_tokens')} out · {answer.diagnostics.get('total_ms')}ms total[/dim]"
        )
    else:
        console.print(
            f"[dim]no model call was made · {answer.diagnostics.get('total_ms')}ms total[/dim]"
        )

    gates = answer.diagnostics.get("gates") or {}
    if gates:
        console.print("[dim]gates: " + "  ".join(f"{k}={v}" for k, v in gates.items()) + "[/dim]")

    if show_context:
        console.print("\n[bold]Context given to the model[/bold]")
        for entry in retrieval.get("chunks", []):
            console.print(f"  [cyan]{entry['chunk_id']}[/cyan] [dim]score {entry['score']}[/dim]")

    if diagnostics:
        console.print("\n[bold]Diagnostics[/bold]")
        console.print_json(json.dumps(answer.diagnostics, default=str))


def _serialise(answer: Answer) -> dict[str, object]:
    """The same shape `POST /v1/ask` returns, so the CLI and API stay honest with each
    other before the HTTP layer exists."""
    return {
        "question": answer.question,
        "answer": answer.text,
        "answered": answer.answered,
        "abstained": answer.abstained,
        "abstention_reason": (answer.abstention_reason.value if answer.abstention_reason else None),
        "confidence": answer.confidence.value,
        "citations": [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "doc_title": c.doc_title,
                "source_file": c.source_file,
                "snippet": c.snippet,
                "char_range": list(c.char_range),
                "score": round(c.score, 5),
            }
            for c in answer.citations
        ],
        "sources": list(answer.sources),
        "diagnostics": answer.diagnostics,
    }
