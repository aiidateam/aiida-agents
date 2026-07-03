"""Interactive REPL loop and its prompt_toolkit session."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.tools import DeferredToolRequests

from aiida_agents._settings import ModelSettings, ReplSettings
from aiida_agents.cli.session import ask
from aiida_agents.cli.hitl import _handle_deferred
from aiida_agents.cli.output import _format_duration, _print_agent, console


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
    """Bind Enter to submit and Alt/Esc+Enter to insert a newline.

    prompt_toolkit's multiline default is the reverse (Enter inserts a
    newline, Meta+Enter submits), which is wrong for a chat REPL where
    single-line turns dominate. Flipping it keeps the common case a single
    keystroke while still allowing a multi-line turn on demand.
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


def _run_repl(agent: Agent, settings: ModelSettings) -> None:  # pragma: no cover
    """Drive the interactive REPL over an already-built agent."""
    repl_cfg = ReplSettings()
    history_file = _history_file()
    history_file.parent.mkdir(parents=True, exist_ok=True)
    session: PromptSession[str] = PromptSession(
        history=FileHistory(str(history_file)),
        key_bindings=_key_bindings(),
        multiline=True,
        vi_mode=repl_cfg.vi_mode,
    )

    print(
        f"AiiDA Agent [{settings.provider}:{settings.model}] - "
        "type 'quit' to exit, '/clear' to start a new conversation, "
        "Esc then Enter (Alt+Enter) for a new line\n"
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
            print("Conversation cleared.\n")
            continue

        start = time.monotonic()
        try:
            # A live spinner fights the interleaved tool-call trace that renders
            # to the console in debug mode, so skip it and let the trace flow.
            status_ctx: AbstractContextManager[object]
            if logging.getLogger().getEffectiveLevel() > logging.DEBUG:
                status_ctx = console.status("[dim]thinking…[/]", spinner="dots")
            else:
                status_ctx = nullcontext()

            with status_ctx:
                result = asyncio.run(
                    ask(
                        agent,
                        question,
                        _cap_history(history, repl_cfg.history_max_turns) or None,
                    )
                )
        except KeyboardInterrupt:
            print("(interrupted)")
            continue
        except Exception as exc:
            print(f"❌ Error: {exc}")
            continue
        elapsed = time.monotonic() - start

        if isinstance(result.output, DeferredToolRequests):
            history = _handle_deferred(agent, result, history)
        else:
            _print_agent(result.output)
            history = result.all_messages()
        console.print(f"[dim]⏱ {_format_duration(elapsed)}[/]")
