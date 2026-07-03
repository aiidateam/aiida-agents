"""Shared console and reply rendering for the CLI.

The lone ``Console`` plus the reply/duration formatting and tool-call trace
rendering, kept here so both the REPL loop and the write-approval flow render
consistently without importing each other.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

from pydantic_ai.messages import ModelMessage, ToolCallPart
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from aiida_agents._logging import ToolPart, trace_response, trace_tool_part

console = Console()


def _print_agent(text: str) -> None:  # pragma: no cover
    """Print an agent reply, blank-line padded so it stands clear of the ``You:``
    turns on either side: a highlighted label, then the body as markdown so
    tables and formatting render.
    """
    console.print()
    console.print("Agent:", style="bold green")
    console.print(Markdown(text))
    console.print()
    trace_response(text)


def _tool_parts(messages: list[ModelMessage]) -> Iterator[ToolPart]:
    """All tool call/return parts of ``messages``, in message order."""
    for msg in messages:
        for part in msg.parts:
            if isinstance(part, ToolPart):
                yield part


def _render_part(part: ToolPart, console: Console) -> None:
    """Render one tool call/return on the console with rich formatting."""
    console.print()
    if isinstance(part, ToolCallPart):
        console.print(
            f"[bold cyan]→ TOOL CALLED:[/bold cyan] [yellow]{part.tool_name}[/yellow]"
        )
        console.print(f"  [dim]ID:[/dim] {part.tool_call_id}")
        console.print(f"  [dim]Args:[/dim] {part.args}")
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


def _log_tool_calls_debug(messages: list[ModelMessage], console: Console) -> None:
    """Record tool calls/returns to the trace log; render on the console at DEBUG.

    The trace log always records: the log file's content must not depend on
    the console log level. Only the console rendering is debug-gated.
    """
    render = logging.getLogger().getEffectiveLevel() <= logging.DEBUG
    for part in _tool_parts(messages):
        trace_tool_part(part)
        if render:
            _render_part(part, console)


def _format_duration(seconds: float) -> str:
    """Human-readable elapsed time: ``12.3s`` under a minute, ``2m 12s`` above."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m {secs}s"
