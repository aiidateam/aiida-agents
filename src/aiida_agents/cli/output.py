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

logger = logging.getLogger(__name__)

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


def _warn_ungrounded(text: str, messages: list[ModelMessage], question: str) -> None:
    """Flag any physical quantity in a reply that no tool produced.

    The agents are asked what cutoff to use, so a wrong number is not a wrong
    answer -- it configures a calculation. The failure that occurs is a
    plausible value, often attached to a real label the model did retrieve, and
    the user cannot tell it apart from a sourced one by reading.

    This runs on every reply because the prompt-level version of the same rule
    has a measured failure rate: an explicit instruction not to do this was
    ignored in five test runs out of five. A check that reads the answer
    afterwards does not depend on the model having complied.

    Warns rather than blocks: the detector is deliberately narrow (see
    ``aiida_agents.grounding``) but a false positive must cost a line of output,
    never a withheld answer.
    """
    from aiida_agents.grounding import (
        syntax_errors,
        tool_output_text,
        ungrounded_quantities,
        ungrounded_symbols,
    )

    evidence = tool_output_text(messages)

    invented = ungrounded_quantities(text, evidence, question)
    if invented:
        values = ", ".join(sorted(invented))
        console.print(
            f"[yellow]⚠ Not found in any tool output: {values}. "
            "Verify before using these values.[/yellow]"
        )
        logger.warning("ungrounded quantities in reply: %s", values)

    # The same question asked of generated code. Separate messages because the
    # actions differ: an unsourced number wants checking, a name that exists in
    # no example wants not being run.
    unknown = ungrounded_symbols(text, evidence)
    if unknown:
        names = ", ".join(sorted(unknown))
        console.print(
            f"[yellow]⚠ Imported from AiiDA but in no documentation this run "
            f"retrieved: {names}. Check these exist before running the "
            "code.[/yellow]"
        )
        logger.warning("ungrounded symbols in reply: %s", names)

    for problem in syntax_errors(text):
        console.print(f"[yellow]⚠ The Python above does not parse: {problem}[/yellow]")
        logger.warning("syntax error in generated code: %s", problem)
