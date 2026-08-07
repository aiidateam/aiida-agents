"""Shared console and reply rendering for the CLI.

The lone ``Console`` plus the reply/duration formatting and tool-call trace
rendering, kept here so both the REPL loop and the write-approval flow render
consistently without importing each other.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import rich_click as click
from pydantic_ai.messages import ModelMessage, ToolCallPart, ToolReturnPart
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.syntax import Syntax
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


def _tool_call_args(part: ToolCallPart) -> dict[str, object]:
    """A tool call's args as a dict, whether they arrived as a dict or as JSON."""
    if isinstance(part.args, dict):
        return part.args
    if isinstance(part.args, str):
        try:
            parsed = json.loads(part.args)
        except (json.JSONDecodeError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _snippet_outcomes(messages: list[ModelMessage]) -> list[tuple[str, bool]]:
    """Each ``run_aiida_code`` snippet this turn, paired with whether it ran cleanly.

    A snippet ran cleanly if its return is not a refusal, timeout or traceback
    (:func:`~aiida_agents.sandbox.runner.summary_is_failure`). Calls are matched
    to their returns by ``tool_call_id``; retried attempts are all included, in
    order. The last attempt is not necessarily the right one, so callers that
    keep code keep every clean run rather than guessing.
    """
    from aiida_agents.sandbox.runner import summary_is_failure

    returns = {
        part.tool_call_id: str(part.content)
        for part in _tool_parts(messages)
        if isinstance(part, ToolReturnPart) and part.tool_name == "run_aiida_code"
    }
    outcomes: list[tuple[str, bool]] = []
    for part in _tool_parts(messages):
        if isinstance(part, ToolCallPart) and part.tool_name == "run_aiida_code":
            code = _tool_call_args(part).get("code")
            if isinstance(code, str) and code.strip():
                summary = returns.get(part.tool_call_id, "")
                ran_cleanly = bool(summary) and not summary_is_failure(summary)
                outcomes.append((code, ran_cleanly))
    return outcomes


def _generated_code(messages: list[ModelMessage]) -> list[str]:
    """The Python the codegen agent ran this turn, in order (clean or not)."""
    return [code for code, _ in _snippet_outcomes(messages)]


def _save_generated_code(snippets: list[str], save_dir: Path) -> list[Path]:
    """Write each snippet to its own timestamped ``.py`` file under ``save_dir``.

    Returns the paths written, so a user can read, edit, and re-run them.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for code in snippets:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        path = save_dir / f"{stamp}.py"
        header = (
            f"# Generated by aiida-agents (codegen) at {stamp}.\n"
            "# Ran against a read-only sandbox profile; review before running it "
            "against a writable one.\n\n"
        )
        path.write_text(header + code.rstrip() + "\n", encoding="utf-8")
        paths.append(path)
    return paths


def _code_syntax(code: str) -> Syntax:
    """Syntax-highlight a snippet for the console, on the terminal's own theme."""
    return Syntax(
        code, "python", theme="ansi_dark", background_color="default", word_wrap=True
    )


def announce_saved_code(paths: list[Path], console: Console) -> None:
    """Note where this turn's generated code was saved, without printing it.

    The code is not echoed inline: it reads badly in the middle of an answer,
    and the model's own reply already interprets the result. It is saved and, in
    the REPL, revealed on demand with Ctrl+O or ``/code``. This is the
    breadcrumb that always shows, one line naming the file(s) written. A no-op
    when nothing ran cleanly, since nothing was saved.
    """
    if not paths:
        return
    if len(paths) == 1:
        console.print(f"[dim]Generated code saved to {paths[0]}[/dim]")
        return
    console.print(f"[dim]Generated code saved ({len(paths)} snippets):[/dim]")
    for path in paths:
        console.print(f"[dim]  {path}[/dim]")


def render_all_generated_code(messages: list[ModelMessage], console: Console) -> None:
    """Print every snippet the codegen agent ran this turn, each marked ran/failed.

    Generated code is not shown inline (only its save location is, see
    :func:`announce_saved_code`); a terminal cannot fold already-printed output,
    so this prints the full set on demand instead (the REPL binds it to Ctrl+O
    and ``/code``). Prints a short note rather than nothing on an empty turn, so
    the key press has feedback.
    """
    outcomes = _snippet_outcomes(messages)
    if not outcomes:
        console.print("[dim]No code was generated in the last turn.[/dim]")
        return
    for index, (code, ok) in enumerate(outcomes, start=1):
        marker = "[green]✓ ran[/green]" if ok else "[red]✗ failed[/red]"
        console.print()
        console.print(
            f"[bold cyan]Snippet {index}/{len(outcomes)}[/bold cyan] {marker}"
        )
        console.print(_code_syntax(code))


def save_generated_code(messages: list[ModelMessage]) -> list[Path]:
    """Save every snippet that ran cleanly this turn to its own ``.py`` file the
    user can read, edit, and re-run.

    One file per distinct clean snippet: the last attempt is not always the
    right one, so keeping only it would drop earlier runs that worked, and a
    refused or failed try is not worth keeping at all. Identical snippets are
    written once. Returns the paths written (empty when nothing ran cleanly);
    the directory is ``SandboxSettings.codegen_save_dir``.
    """
    clean = [code for code, ok in _snippet_outcomes(messages) if ok]
    unique = list(dict.fromkeys(clean))
    if not unique:
        return []

    from aiida_agents._settings import SandboxSettings

    return _save_generated_code(unique, SandboxSettings().codegen_save_dir)


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
