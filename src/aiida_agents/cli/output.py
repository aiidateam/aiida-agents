"""Shared console and reply rendering for the CLI.

The lone ``Console`` plus the reply/duration formatting and tool-call trace
rendering, kept here so both the REPL loop and the write-approval flow render
consistently without importing each other.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import rich_click as click
from pydantic_ai.messages import ModelMessage, ToolCallPart
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.text import Text

from aiida_agents._logging import ToolPart, trace_response, trace_tool_part

console = Console()


def _print_agent(text: str) -> None:
    """Print an agent reply, blank-line padded so it stands clear of the ``You:``
    turns on either side: a highlighted label, then the body as markdown so
    tables and formatting render.
    """
    console.print()
    console.print("Agent:", style="bold green")
    console.print(Markdown(text))
    console.print()
    trace_response(text)


def _print_reply(text: str, *, raw: bool = False) -> None:
    """Render a one-shot ``ask`` reply and record it to the trace log.

    Pretty Markdown on an interactive terminal; the raw Markdown source when the
    output is piped or redirected, so ``aiida-agents ask ... > answer.md`` and
    shell pipelines get clean, re-renderable text rather than box-drawing tables
    reflowed to the terminal width. ``raw=True`` forces the raw source even on a
    terminal (``--raw``), for copy-paste or when the auto-detection guesses wrong.
    """
    if raw or not console.is_terminal:
        click.echo(text)
    else:
        console.print(Markdown(text))
    trace_response(text)


def _tool_parts(messages: list[ModelMessage]) -> Iterator[ToolPart]:
    """All tool call/return parts of ``messages``, in message order."""
    return (
        part for msg in messages for part in msg.parts if isinstance(part, ToolPart)
    )


def _render_part(part: ToolPart, console: Console) -> None:
    """Render one tool call/return on the console with rich formatting."""
    console.print()
    if isinstance(part, ToolCallPart):
        console.print(
            f"[bold cyan]→ TOOL CALLED:[/bold cyan] [yellow]{part.tool_name}[/yellow]"
        )
        console.print(f"  [dim]ID:[/dim] {part.tool_call_id}")
        # escape(): part.args is a dict repr that can contain bracketed values or
        # stray [/tag]-shaped text; without escaping, rich would swallow it as
        # markup or raise MarkupError (the tool-return Panel below uses Text() for
        # the same reason).
        console.print(f"  [dim]Args:[/dim] {escape(str(part.args))}")
    else:
        console.print(
            f"[bold green]← TOOL RETURNED:[/bold green] [yellow]{part.tool_name}[/yellow]"
        )
        console.print(f"  [dim]ID:[/dim] {part.tool_call_id}")
        console.print(
            Panel(
                # Text() renders the content literally: tool returns contain
                # bracketed [source § section] headers that rich's markup
                # parser would otherwise swallow as style tags.
                Text(str(part.content)),
                title=f"Tool Return: {part.tool_name}",
                border_style="green",
            )
        )
    console.print()


def _trace_tool_calls(messages: list[ModelMessage]) -> None:
    """Record every tool call/return to the trace log (level-independent).

    The log file's content must not depend on the console log level, so this
    always records; rendering to the console is a separate, debug-gated concern
    (:func:`_render_tool_calls`).
    """
    for part in _tool_parts(messages):
        trace_tool_part(part)


def _render_tool_calls(messages: list[ModelMessage], console: Console) -> None:
    """Render tool calls/returns on the console, but only at DEBUG.

    Kept separate from :func:`_trace_tool_calls` so a caller with a live spinner
    or progress bar can record during the run and render *after* the live region
    is torn down, rather than printing into it.
    """
    if logging.getLogger().getEffectiveLevel() > logging.DEBUG:
        return
    for part in _tool_parts(messages):
        _render_part(part, console)


def _log_tool_calls_debug(messages: list[ModelMessage], console: Console) -> None:
    """Record to the trace log (always) and render on the console at DEBUG.

    Convenience for callers with no live region to coordinate around; the REPL
    splits these so it can render after its spinner stops.
    """
    _trace_tool_calls(messages)
    _render_tool_calls(messages, console)


def _format_duration(seconds: float) -> str:
    """Human-readable elapsed time: ``12.3s`` under a minute, ``2m 12s`` above."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m {secs}s"
