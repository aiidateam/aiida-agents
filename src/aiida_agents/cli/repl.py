"""Interactive REPL loop and its prompt_toolkit session."""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
import rich_click as click
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.tools import DeferredToolRequests

from aiida_agents._settings import ModelSettings, ReplSettings
from aiida_agents.cli.agent import _AGENT_CHOICES, _build_agent, ask
from aiida_agents.cli.hitl import _handle_deferred
from aiida_agents.cli.output import (
    _format_duration,
    _print_agent,
    _render_tool_calls,
    console,
)


def _cap_history(messages: list[ModelMessage], max_turns: int) -> list[ModelMessage]:
    """Trim ``messages`` to the last ``max_turns`` user turns.

    A user turn starts with a ``ModelRequest`` carrying a ``UserPromptPart``;
    tool call/return rounds live inside a turn, so cutting on these boundaries
    never splits a tool-call/return pair. A raw ``messages[-N:]`` slice can,
    and providers then reject the unpaired ``tool_use``/``tool_result``.
    """
    starts = [
        i
        for i, m in enumerate(messages)
        if isinstance(m, ModelRequest)
        and any(isinstance(p, UserPromptPart) for p in m.parts)
    ]
    return messages[starts[-max_turns] :] if len(starts) > max_turns else messages


def _history_file() -> Path:
    """Persistent location for the REPL's input history.

    Follows the XDG base-directory spec so recalled prompts survive across
    sessions without cluttering ``$HOME``.
    """
    data_home = os.environ.get("XDG_DATA_HOME")
    base = Path(data_home) if data_home else Path.home() / ".local" / "share"
    return base / "aiida-agents" / "repl-history"


def _key_bindings() -> KeyBindings:
    """Bind Enter to submit; Esc+Enter or Ctrl+J to insert a newline.

    prompt_toolkit's multiline default is the reverse (Enter inserts a
    newline, Meta+Enter submits), which is wrong for a chat REPL where
    single-line turns dominate. Flipping it keeps the common case a single
    keystroke while still allowing a multi-line turn on demand.

    Ctrl+J is bound as a second newline key: it sends ``\\n`` (``ControlJ``),
    distinct from Enter's ``\\r`` (``ControlM``), and most terminals (Windows
    Terminal included) emit that same ``\\n`` for Ctrl+Enter, so Ctrl+Enter
    works too. It also sidesteps Esc+Enter's awkwardness in vi mode, where Esc
    is the mode switch. (Shift+Enter can't be bound: terminals don't send a
    distinct byte for it.)
    """
    bindings = KeyBindings()

    # ``# pyright: ignore[reportUnusedFunction]``: the ``@bindings.add`` decorator
    # registers each handler, so it is used, but a static analyser sees no
    # reference to the name.
    @bindings.add("enter")
    def _submit(
        event: KeyPressEvent,
    ) -> None:  # pragma: no cover  # pyright: ignore[reportUnusedFunction]
        event.current_buffer.validate_and_handle()

    @bindings.add("escape", "enter")
    @bindings.add("c-j")
    def _newline(
        event: KeyPressEvent,
    ) -> None:  # pragma: no cover  # pyright: ignore[reportUnusedFunction]
        event.current_buffer.insert_text("\n")

    return bindings


def _prompt_continuation(width: int, _line_number: int, _wrap_count: int) -> str:
    """Continuation prefix for a multi-line turn, padded to the prompt width.

    prompt_toolkit passes the width of the main prompt (``You: ``), so the
    dotted marker lines wrapped input up under the first line's text.
    """
    return "." * (width - 1) + " "


def _make_session(repl_cfg: ReplSettings) -> PromptSession[str]:  # pragma: no cover
    """Build the prompt session: persistent history plus the REPL key bindings."""
    history_file = _history_file()
    history_file.parent.mkdir(parents=True, exist_ok=True)
    return PromptSession(
        history=FileHistory(str(history_file)),
        key_bindings=_key_bindings(),
        multiline=True,
        vi_mode=repl_cfg.vi_mode,
    )


def _run_turn(
    agent: Agent,
    question: str,
    history: list[ModelMessage],
    repl_cfg: ReplSettings,
) -> list[ModelMessage]:  # pragma: no cover
    """Run one query, render its reply, and return the updated history.

    Returns ``history`` unchanged if the run is interrupted (Ctrl-C) or errors,
    so a failed turn never corrupts the conversation.
    """
    start = time.monotonic()
    try:
        with console.status("[dim]thinking...[/]", spinner="dots"):
            result = asyncio.run(
                ask(
                    agent,
                    question,
                    _cap_history(history, repl_cfg.history_max_turns) or None,
                )
            )
    except KeyboardInterrupt:
        click.echo("(interrupted)")
        return history
    except Exception as exc:
        click.echo(f"❌ Error: {exc}")
        return history
    elapsed = time.monotonic() - start

    # Render the run's tool-call trace now that the spinner has stopped: the
    # traces are a post-run dump, so printing them into the still-live spinner
    # region above would fight its redraws. Debug-gated inside.
    _render_tool_calls(result.new_messages(), console)

    if isinstance(result.output, DeferredToolRequests):
        history = _handle_deferred(agent, result, history)
    else:
        _print_agent(result.output)
        history = result.all_messages()
    console.print(f"[dim]⏱ {_format_duration(elapsed)}[/]")
    return history


def _parse_agent_switch(question: str, current: str) -> str | None:
    """Parse a ``/agent`` REPL command into the requested agent name.

    Returns the requested agent name for a valid ``/agent <name>``, or ``None``
    for a bare ``/agent`` or an unknown name (after echoing the current agent and
    usage). The caller rebuilds only when the returned name differs from
    ``current``, so re-selecting the active agent is a no-op.
    """
    parts = question.split()
    if len(parts) >= 2 and parts[1].lower() in _AGENT_CHOICES:
        return parts[1].lower()
    click.echo(
        f"Current agent: {current.capitalize()} Agent. "
        f"Usage: /agent {' | '.join(_AGENT_CHOICES)}\n"
    )
    return None


def _run_repl(
    agent: Agent,
    settings: ModelSettings,
    profile: str | None = None,
    agent_type: str = "analysis",
) -> None:  # pragma: no cover
    """Drive the interactive REPL: read a line, dispatch commands, run the turn.

    ``profile`` and ``agent_type`` are kept so ``/agent`` can rebuild the agent in
    place (same profile, different agent) and reset the conversation.
    """
    repl_cfg = ReplSettings()
    session = _make_session(repl_cfg)

    click.echo(
        f"AiiDA {agent_type.capitalize()} Agent [{settings.provider}:{settings.model}] - "
        "type 'quit' to exit, '/clear' to start a new conversation, "
        "'/agent [analysis|execution]' to switch agent, "
        "Ctrl+Enter (or Esc then Enter) for a new line\n"
    )

    history: list[ModelMessage] = []
    while True:
        # Ctrl-C aborts the current line (like a shell); Ctrl-D at an empty
        # prompt exits. prompt_toolkit raises KeyboardInterrupt / EOFError.
        try:
            question = session.prompt(
                HTML("<ansicyan><b>You:</b></ansicyan> "),
                prompt_continuation=_prompt_continuation,
            ).strip()
        except KeyboardInterrupt:
            continue
        except EOFError:
            break

        # Empty input re-prompts (shell-like); only an explicit word or Ctrl-D exits.
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            break
        if question.lower() == "/clear":
            history = []
            click.echo("Conversation cleared.\n")
            continue
        if question.lower().startswith("/agent"):
            requested = _parse_agent_switch(question, agent_type)
            if requested is not None and requested != agent_type:
                agent_type = requested
                agent = _build_agent(settings, profile, agent_type)
                history = []
                click.echo(
                    f"Switched to {agent_type.capitalize()} Agent. "
                    "Conversation cleared.\n"
                )
            continue

        history = _run_turn(agent, question, history, repl_cfg)
