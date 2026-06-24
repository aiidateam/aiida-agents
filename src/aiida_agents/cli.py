"""Command-line entry point for the AiiDA agent."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.tools import DeferredToolRequests

logger = logging.getLogger(__name__)


async def ask(agent: Agent, question: str) -> Any:  # pragma: no cover
    """Run a single query through the agent, returning the result."""
    logger.info("agent query: %s", question)
    return await agent.run(question)


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


def _handle_deferred(agent: Agent, result: Any) -> None:  # pragma: no cover
    """Show pending approval requests and ask the user to confirm or cancel.

    For ``submit_workflow`` calls, inputs are resolved (without storing) before
    the prompt so the user sees the actual node types and values they are
    approving, not the raw agent arguments (ADR-08).
    """
    from aiida_agents.mcp.tools.submit import _resolve_inputs, _format_resolved_inputs
    from fastmcp.exceptions import ToolError

    pending = result.output

    print("\n⚠️  The agent wants to perform the following submission(s):")
    for call in pending.approvals:
        print(f"   Tool  : {call.tool_name}")

        args = _parse_args(call.args)

        # For submit_workflow, resolve inputs first so the user sees the real
        # node types/values — not the raw primitives or reference dicts.
        if call.tool_name == "submit_workflow":
            entry_point = args.get("entry_point", "<unknown>")
            raw_inputs = args.get("inputs", {})
            print(f"   Entry : {entry_point}")
            try:
                resolved = _resolve_inputs(entry_point, raw_inputs)
                print(f"   Inputs (resolved):\n{_format_resolved_inputs(resolved)}")
            except ToolError as exc:
                # Resolution failed — show raw args and the error so the user
                # can make an informed decision to cancel.
                print(f"   Inputs (raw)   : {raw_inputs}")
                print(f"   ⚠️  Could not resolve inputs: {exc}")
        else:
            print(f"   Inputs: {args}")

    answer = input("\nProceed? [y/N]: ").strip().lower()
    if answer != "y":
        print("Cancelled — nothing was submitted.")
        return

    deferred_results = pending.build_results(approve_all=True)
    try:
        followup = asyncio.run(
            agent.run(
                None,
                message_history=result.all_messages(),
                deferred_tool_results=deferred_results,
            )
        )
        print(f"Agent: {followup.output}")
    except Exception as exc:
        print(f"\n❌ Error during submission: {exc}")


def main() -> None:  # pragma: no cover
    """Interactive REPL for the AiiDA agent."""
    from aiida import load_profile
    from aiida_agents.agents import get_agent
    from aiida_agents._settings import ModelSettings, warn_on_unrecognized_settings

    warn_on_unrecognized_settings()
    settings = ModelSettings()
    load_profile()
    agent = get_agent()

    print(f"AiiDA Agent [{settings.provider}:{settings.model}] - type 'quit' to exit\n")

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not question or question.lower() in ("quit", "exit", "q"):
            break

        try:
            result = asyncio.run(ask(agent, question))
            if isinstance(result.output, DeferredToolRequests):
                _handle_deferred(agent, result)
            else:
                print(f"Agent: {result.output}")
        except Exception as exc:
            print(f"❌ Error: {exc}")
