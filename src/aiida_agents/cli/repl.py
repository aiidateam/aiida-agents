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
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.tools import DeferredToolRequests

from aiida_agents._settings import ModelSettings, ReplSettings
from aiida_agents.agents.handoff import NodeReference, node_references_from_messages
from aiida_agents.cli.agent import (
    _AGENT_CHOICES,
    _build_agent,
    _resolve_plan,
    _step_prompt,
    _StepResult,
    ask,
)
from aiida_agents.cli.hitl import _handle_deferred
from aiida_agents.cli.output import (
    _format_duration,
    _print_agent,
    _warn_ungrounded,
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


#: User turns the planner's transcript keeps.
#:
#: Deliberately smaller than ``ReplSettings.history_max_turns``, which sizes
#: what a *specialist* needs to keep working. The planner only has to resolve a
#: reference back to a recent turn ("the former", "that one"), and its own
#: docstrings sell it as one cheap round-trip -- putting a session's worth of
#: specialist prose in front of a call whose entire output is one line would
#: stop that being true.
_PLANNER_HISTORY_MAX_TURNS = 3


def _record_planner_turn(
    history: list[ModelMessage], question: str, answer: str
) -> None:
    """Append one user/assistant pair, dropping the oldest turn past the cap.

    Trimmed on write rather than on send, so the list itself stays bounded over
    a long session. A plain slice suffices: this transcript strictly alternates
    and holds no tools, whereas :func:`_cap_history` has to search for turn
    boundaries to avoid splitting a specialist's tool-call/return pair.
    """
    history.append(ModelRequest(parts=[UserPromptPart(content=question)]))
    history.append(ModelResponse(parts=[TextPart(content=answer)]))
    del history[: -2 * _PLANNER_HISTORY_MAX_TURNS]


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
) -> tuple[
    list[ModelMessage], str | None, tuple[NodeReference, ...]
]:  # pragma: no cover
    """Run one query, render its reply, and return history, answer and references.

    Returns ``history`` unchanged if the run is interrupted (Ctrl-C) or errors,
    so a failed turn never corrupts the conversation.

    The answer comes back alongside it because a multi-step plan feeds each
    step's answer into the next. ``None`` means this turn produced no text to
    pass on -- it errored, was interrupted, or ended in an approval flow -- and
    a plan stops there rather than continuing on nothing.
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
        return history, None, ()
    except Exception as exc:
        click.echo(f"❌ Error: {exc}")
        return history, None, ()
    elapsed = time.monotonic() - start

    # Render the run's tool-call trace now that the spinner has stopped: the
    # traces are a post-run dump, so printing them into the still-live spinner
    # region above would fight its redraws. Debug-gated inside.
    _render_tool_calls(result.new_messages(), console)

    answer: str | None = None
    # From this turn's messages only: the accumulated history carries pks from
    # earlier questions, and handing those to a later step as "what this step
    # found" would be false.
    references = node_references_from_messages(result.new_messages())
    if isinstance(result.output, DeferredToolRequests):
        history = _handle_deferred(agent, result, history)
    else:
        _print_agent(result.output)
        _warn_ungrounded(result.output, result.all_messages(), question)
        history = result.all_messages()
        answer = result.output
    console.print(f"[dim]⏱ {_format_duration(elapsed)}[/]")
    return history, answer, references


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
    agent: Agent | None,
    settings: ModelSettings,
    profile: str | None = None,
    agent_type: str = "analysis",
) -> None:  # pragma: no cover
    """Drive the interactive REPL: read a line, dispatch commands, run the turn.

    ``agent`` is the pre-built specialist to start from, or ``None`` in auto
    mode, where the specialist is not known until a question has been routed.
    Either way the loop builds what it needs on first use.

    ``profile`` and ``agent_type`` are kept so ``/agent`` can rebuild the agent in
    place (same profile, different agent) and reset the conversation.
    """
    repl_cfg = ReplSettings()
    session = _make_session(repl_cfg)

    banner = (
        "AiiDA Agents (auto-routing)"
        if agent_type == "auto"
        else f"AiiDA {agent_type.capitalize()} Agent"
    )
    click.echo(
        f"{banner} [{settings.provider}:{settings.model}] - "
        "type 'quit' to exit, '/clear' to start a new conversation, "
        "'/agent [auto|analysis|execution]' to switch agent, "
        "Ctrl+Enter (or Esc then Enter) for a new line\n"
    )

    # In auto mode each specialist keeps its own conversation and its own agent,
    # built on first use. Their tool sets differ, so replaying one's history to
    # the other would reference tools it does not have. Carrying context *across*
    # specialists is the next increment (ADR-09); until then a routed switch
    # starts that specialist where it last left off, not mid-thread on another.
    agents: dict[str, Agent] = {} if agent is None else {agent_type: agent}
    histories: dict[str, list[ModelMessage]] = {}
    # The planner is tool-less and called fresh each turn, so a follow-up ("the
    # former") cannot be routed without the prior turns. This clean transcript is
    # replayed to it, kept separate from the specialists' tool-carrying
    # histories, which the planner cannot read. Only turns that ended in an
    # answer are recorded: one that errored, was interrupted, or ended in an
    # approval flow leaves no text, and is dropped along with its question.
    planner_history: list[ModelMessage] = []
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
            histories.clear()
            planner_history.clear()
            click.echo("Conversation cleared.\n")
            continue
        if question.lower().startswith("/agent"):
            requested = _parse_agent_switch(question, agent_type)
            if requested is not None and requested != agent_type:
                agent_type = requested
                agents.clear()
                histories.clear()
                planner_history.clear()
                if agent_type != "auto":
                    agents[agent_type] = _build_agent(settings, profile, agent_type)
                label = (
                    "auto-routing"
                    if agent_type == "auto"
                    else f"{agent_type.capitalize()} Agent"
                )
                click.echo(f"Switched to {label}. Conversation cleared.\n")
            continue

        previous: _StepResult | None = None
        steps = _resolve_plan(
            agent_type,
            question,
            settings,
            message_history=planner_history or None,
        )
        for index, step in enumerate(steps, start=1):
            active = step.specialist
            if active not in agents:
                agents[active] = _build_agent(settings, profile, active)
            if len(steps) > 1:
                click.echo(f"— step {index}/{len(steps)} ({active}) —")

            histories[active], answer, references = _run_turn(
                agents[active],
                _step_prompt(step, question, previous),
                histories.get(active, []),
                repl_cfg,
            )
            if answer is None:
                # The step errored, was interrupted, or ended in an approval
                # flow. Continuing would hand the next specialist a premise
                # nobody established -- a resubmission built on a diagnosis
                # that never completed is worse than no resubmission.
                if index < len(steps):
                    click.echo(
                        "Plan stopped: this step produced nothing to build on.\n"
                    )
                break
            previous = _StepResult(active, answer, references)

        # Record this turn for the planner's next routing decision, so a
        # reference like "the former" resolves against what was actually said.
        # A multi-step plan records its last step's answer: that is the answer
        # the request ended on, and the earlier steps were its working.
        if previous is not None:
            _record_planner_turn(planner_history, question, previous.answer)
