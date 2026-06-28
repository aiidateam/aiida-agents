"""Command-line entry point for the AiiDA agent."""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import sys
import threading
import time
from typing import Any, cast

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.tools import DeferredToolRequests

logger = logging.getLogger(__name__)


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

# A pending submission awaiting the user's decision: the original tool call and
# its resolved inputs (None for non-submit approval tools, shown with raw args).
_Preview = tuple[Any, dict[str, Any] | None]

# Keep the last N messages in the history to avoid filling the context window.
_MAX_HISTORY = 20


def _triage_submissions(
    pending: DeferredToolRequests,
) -> tuple[dict[str, Any], list[_Preview]]:
    """Resolve and validate each pending approval before the user sees it.

    Returns ``(auto_denials, previews)``:

    * ``auto_denials`` maps a tool-call id to a ``ToolDenied`` for any
      ``submit_workflow`` whose inputs fail resolution or validation. These go
      straight back to the model so it can correct its own mistakes without
      bothering the user.
    * ``previews`` lists ``(call, resolved)`` for the calls the user must
      decide on: ``resolved`` is the resolved-inputs dict for a valid
      ``submit_workflow``, or ``None`` for any other approval-gated tool.
    """
    from pydantic_ai.tools import ToolDenied

    from aiida_agents.mcp.tools.submit import SubmissionInputError, _prepare_submission

    auto: dict[str, Any] = {}
    previews: list[_Preview] = []
    for call in pending.approvals:
        if call.tool_name != "submit_workflow":
            previews.append((call, None))
            continue
        args = _parse_args(call.args)
        try:
            _, resolved = _prepare_submission(
                args.get("entry_point", ""), args.get("inputs", {})
            )
        except SubmissionInputError as exc:
            auto[call.tool_call_id] = ToolDenied(
                f"Submission rejected before reaching the user: {exc} "
                "Correct the inputs and call submit_workflow again."
            )
            continue
        previews.append((call, resolved))
    return auto, previews


def _print_previews(previews: list[_Preview]) -> None:  # pragma: no cover
    """Print the resolved submissions awaiting the user's confirmation."""
    from aiida_agents.mcp.tools.submit import _format_resolved_inputs

    print("\n⚠️  The agent wants to perform the following submission(s):")
    for call, resolved in previews:
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
    """Resolve, validate, and confirm pending submissions, then continue the run.

    Each round: invalid submissions are denied straight back to the model so it
    retries with corrected inputs; valid ones are shown to the user, who
    approves or cancels. Only confirmed, valid inputs reach the database
    (ADR-08, docs/adr/08-human-in-the-loop-before-writes.md). The loop repeats
    while the model keeps proposing submissions in response.

    Returns the updated message history after the deferred path completes.
    """
    from pydantic_ai.tools import ToolApproved, ToolDenied

    for _ in range(_MAX_APPROVAL_ROUNDS):
        pending = result.output
        auto, previews = _triage_submissions(pending)
        approvals: dict[str, Any] = dict(auto)

        if previews:
            _print_previews(previews)
            approved = input("\nProceed? [y/N]: ").strip().lower() == "y"
            if approved:
                decision: Any = ToolApproved()
            else:
                decision = ToolDenied("Cancelled by the user.")
                print("Cancelled — nothing was submitted.")
            for call, _ in previews:
                approvals[call.tool_call_id] = decision
        elif auto:
            print("\n⚠️  Inputs were invalid; asking the agent to correct them.")

        if not approvals:
            return cast(list[ModelMessage], result.all_messages())

        try:
            result = asyncio.run(
                agent.run(
                    None,
                    message_history=result.all_messages(),
                    deferred_tool_results=pending.build_results(approvals=approvals),
                )
            )
        except Exception as exc:
            print(f"\n❌ Error during submission: {exc}")
            return history

        if not isinstance(result.output, DeferredToolRequests):
            print(f"Agent: {result.output}")
            return cast(list[ModelMessage], result.all_messages())

    print("\n⚠️  Too many correction rounds; stopping without submitting.")
    return history


def _spinner(stop: threading.Event) -> None:  # pragma: no cover
    """Animate a spinner on stdout until stop is set."""
    for frame in itertools.cycle(["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]):
        if stop.is_set():
            break
        sys.stdout.write(f"\r{frame} thinking...")
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write("\r" + " " * 20 + "\r")
    sys.stdout.flush()


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
    from aiida_agents._settings import ModelSettings, warn_on_unrecognized_settings

    warn_on_unrecognized_settings()
    settings = ModelSettings()
    load_profile()
    agent = get_agent()

    print(
        f"AiiDA Agent [{settings.provider}:{settings.model}] — "
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

        stop = threading.Event()
        spinner_thread = threading.Thread(target=_spinner, args=(stop,), daemon=True)
        spinner_thread.start()

        try:
            result = asyncio.run(ask(agent, question, history[-_MAX_HISTORY:] or None))
        except Exception as exc:
            stop.set()
            spinner_thread.join()
            print(f"❌ Error: {exc}")
            continue

        stop.set()
        spinner_thread.join()

        if isinstance(result.output, DeferredToolRequests):
            history = _handle_deferred(agent, result, history)
        else:
            print(f"Agent: {result.output}")
            history = result.all_messages()
