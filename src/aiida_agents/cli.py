"""Command-line entry point for the AiiDA agent."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.tools import DeferredToolRequests
from rich.console import Console

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
            print(f"Agent: {result.output}")
            messages: list[ModelMessage] = result.all_messages()
            return messages

    print("\n⚠️  Too many correction rounds; stopping without submitting.")
    return history


def _read_input() -> str:  # pragma: no cover
    """Read one user turn, allowing line continuation with a trailing backslash."""
    lines = []
    prompt = "You: "
    while True:
        line = input(prompt)
        if line.endswith("\\"):
            lines.append(line[:-1])
            prompt = "... "
        else:
            lines.append(line)
            break
    return "\n".join(lines)


def main() -> None:  # pragma: no cover
    """Interactive REPL for the AiiDA agent."""
    from aiida import load_profile
    from aiida_agents.agents import get_agent
    from aiida_agents._settings import (
        AgentSettings,
        ModelSettings,
        warn_on_unrecognized_settings,
    )

    warn_on_unrecognized_settings()
    settings = ModelSettings()
    agent_cfg = AgentSettings()
    load_profile()
    agent = get_agent()

    print(
        f"AiiDA Agent [{settings.provider}:{settings.model}] - "
        "type 'quit' to exit, '/clear' to reset memory\n"
    )

    history: list[ModelMessage] = []

    while True:
        try:
            question = _read_input().strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not question or question.lower() in ("quit", "exit", "q"):
            break

        if question.lower() == "/clear":
            history = []
            print("Memory cleared.\n")
            continue

        try:
            with console.status("[dim]thinking…[/]", spinner="dots"):
                result = asyncio.run(
                    ask(
                        agent,
                        question,
                        _cap_history(history, agent_cfg.history_max_turns) or None,
                    )
                )
        except Exception as exc:
            print(f"❌ Error: {exc}")
            continue

        if isinstance(result.output, DeferredToolRequests):
            history = _handle_deferred(agent, result, history)
        else:
            print(f"Agent: {result.output}")
            history = result.all_messages()
