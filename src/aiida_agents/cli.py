"""Command-line entry point for the AiiDA agent."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.tools import DeferredToolRequests
from rich.console import Console
from rich.markdown import Markdown

logger = logging.getLogger(__name__)
console = Console()


async def ask(
    agent: Agent,
    question: str,
    message_history: list[ModelMessage] | None = None,
) -> Any:  # pragma: no cover
    """Run a single query through the agent, returning the result."""
    logger.info("agent query: %s", question)
    return await agent.run(question, message_history=message_history)


def _parse_args(args: str | dict[str, Any] | None) -> dict[str, Any]:
    """Safely parse tool call args to a dict regardless of whether they arrived as JSON or a dict."""
    if args is None:
        return {}
    if isinstance(args, dict):
        return args
    try:
        parsed = json.loads(args)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    return {}


# Bound the propose -> deny -> retry loop so a model that keeps emitting bad
# inputs cannot spin forever.
_MAX_APPROVAL_ROUNDS = 10

# A pending submission awaiting the user's decision: the original tool call, its
# loaded process class, and its resolved inputs. The latter two are None for any
# non-submit approval tool (shown with raw args, not executable out of band).
_Preview = tuple[Any, Any, dict[str, Any] | None]


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


def _triage_submissions(
    pending: DeferredToolRequests,
) -> tuple[dict[str, Any], list[_Preview]]:
    """Resolve and validate each pending approval before the user sees it.

    Returns ``(auto_denials, previews)``:

    * ``auto_denials`` maps a tool-call id to a ``ToolDenied`` for any
      ``submit_workflow`` whose inputs fail resolution or validation. These go
      straight back to the model so it can correct its own mistakes without
      bothering the user.
    * ``previews`` lists ``(call, process_class, resolved)`` for the calls the
      user must decide on: ``process_class`` / ``resolved`` are the loaded
      process class and resolved-inputs dict for a valid ``submit_workflow``
      (so the caller can submit on the main thread), or ``None`` for any other
      approval-gated tool.
    """
    from pydantic_ai.tools import ToolDenied

    from aiida_agents.tools.submit import SubmissionInputError, _prepare_submission

    auto: dict[str, Any] = {}
    previews: list[_Preview] = []
    for call in pending.approvals:
        if call.tool_name != "submit_workflow":
            previews.append((call, None, None))
            continue
        args = _parse_args(call.args)
        try:
            process_class, resolved = _prepare_submission(
                args.get("entry_point", ""), args.get("inputs", {})
            )
        except SubmissionInputError as exc:
            auto[call.tool_call_id] = ToolDenied(
                f"Submission rejected before reaching the user: {exc} "
                "Correct the inputs and call submit_workflow again."
            )
            continue
        previews.append((call, process_class, resolved))
    return auto, previews


def _print_previews(previews: list[_Preview]) -> None:  # pragma: no cover
    """Print the resolved submissions awaiting the user's confirmation."""
    from aiida_agents.tools.submit import _format_resolved_inputs

    print("\n⚠️  The agent wants to perform the following submission(s):")
    for call, _, resolved in previews:
        print(f"   Tool  : {call.tool_name}")
        if resolved is None:
            print(f"   Inputs: {_parse_args(call.args)}")
            continue
        args = _parse_args(call.args)
        print(f"   Entry : {args.get('entry_point', '<unknown>')}")
        print(f"   Inputs (resolved):\n{_format_resolved_inputs(resolved)}")


def _handle_deferred(
    agent: Agent,
    result: Any,
    history: list[ModelMessage],
) -> list[ModelMessage]:  # pragma: no cover
    """Confirm and run pending submissions, denying invalid ones to the model.

    Each round: invalid submissions are denied straight back to the model so it
    retries with corrected inputs; valid ones are previewed for the user, who
    approves or cancels. Approved submissions are executed *here, on the main
    thread*, not by re-running the agent: pydantic-ai runs sync tools on a worker
    thread and AiiDA's storage is thread-bound, so writing from the worker thread
    (reusing the default user / nodes the preview bound to the main-thread
    session) raises a cross-thread SQLAlchemy error. Only confirmed, valid inputs
    reach the database (ADR-08, docs/adr/08-human-in-the-loop-before-writes.md).

    Returns the message history to carry into the next turn. Submissions run out
    of band (not through pydantic-ai), so it never records their tool returns; we
    splice each approval's outcome back in as a ``ToolReturnPart`` before
    returning, which keeps the submission in context and leaves no unanswered
    tool call for pydantic-ai to reject next turn. Cancelling or exhausting the
    retry budget returns the pre-turn ``history`` unchanged.
    """
    from aiida_agents.tools.submit import _run_submission

    for _ in range(_MAX_APPROVAL_ROUNDS):
        pending = result.output
        auto, previews = _triage_submissions(pending)

        if previews:
            _print_previews(previews)
            if input("\nProceed? [y/N]: ").strip().lower() != "y":
                print("Cancelled - nothing was submitted.")
                return history

            # Outcome per approval tool-call id. Auto-denied invalid submissions
            # were never executed, so they carry their denial message.
            outcomes: dict[str, Any] = {
                call_id: {"rejected": denied.message}
                for call_id, denied in auto.items()
            }
            for call, process_class, resolved in previews:
                if process_class is None or resolved is None:
                    print(
                        f"   Skipping {call.tool_name}: not an executable submission."
                    )
                    outcomes[call.tool_call_id] = {"skipped": call.tool_name}
                    continue
                entry_point = _parse_args(call.args).get("entry_point", "")
                try:
                    res = _run_submission(entry_point, process_class, resolved)
                except Exception as exc:
                    print(f"\n❌ Submission failed: {exc}")
                    outcomes[call.tool_call_id] = {"error": str(exc)}
                    continue
                print(
                    f"\n✅ Submitted {res['workflow']}: "
                    f"pk={res['pk']}, state={res['state']}"
                )
                outcomes[call.tool_call_id] = res

            print()  # separate the submission summary from the next prompt

            # Splice each approval's outcome back as its tool return so the
            # submission survives in history and no unanswered tool call is left
            # to reject the next turn (the calls ran out of band, so pydantic-ai
            # never recorded returns itself).
            updated: list[ModelMessage] = result.all_messages()
            updated.append(
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name=call.tool_name,
                            content=outcomes[call.tool_call_id],
                            tool_call_id=call.tool_call_id,
                        )
                        for call in pending.approvals
                    ]
                )
            )
            return updated

        if not auto:
            return history

        # Only invalid submissions this round: deny them back to the model so it
        # corrects its own inputs, then re-run. No DB write happens on the worker
        # thread here (denied calls are never executed), so this is thread-safe.
        print("\n⚠️  Inputs were invalid; asking the agent to correct them.")
        try:
            result = asyncio.run(
                agent.run(
                    None,
                    message_history=result.all_messages(),
                    deferred_tool_results=pending.build_results(approvals=auto),
                )
            )
        except Exception as exc:
            print(f"\n❌ Error: {exc}")
            return history

        if not isinstance(result.output, DeferredToolRequests):
            _print_agent(result.output)
            messages: list[ModelMessage] = result.all_messages()
            return messages

    print("\n⚠️  Too many correction rounds; stopping without submitting.")
    return history


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


def _print_agent(text: str) -> None:  # pragma: no cover
    """Print an agent reply, blank-line padded so it stands clear of the ``You:``
    turns on either side: a highlighted label, then the body as markdown so
    tables and formatting render.
    """
    console.print()
    console.print("Agent:", style="bold green")
    console.print(Markdown(text))
    console.print()


def main() -> None:  # pragma: no cover
    """Interactive REPL for the AiiDA agent."""
    from aiida import load_profile
    from aiida_agents.agents import get_agent
    from aiida_agents._settings import (
        ModelSettings,
        ReplSettings,
        warn_on_unrecognized_settings,
    )

    warn_on_unrecognized_settings()
    settings = ModelSettings()
    repl_cfg = ReplSettings()
    load_profile()
    agent = get_agent()

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

        try:
            with console.status("[dim]thinking…[/]", spinner="dots"):
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

        if isinstance(result.output, DeferredToolRequests):
            history = _handle_deferred(agent, result, history)
        else:
            _print_agent(result.output)
            history = result.all_messages()
