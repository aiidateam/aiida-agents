"""Command-line entry point for the AiiDA agent."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.tools import DeferredToolRequests

logger = logging.getLogger(__name__)


async def ask(agent: Agent, question: str) -> Any:  # pragma: no cover
    """Run a single query through the agent, returning the result."""
    logger.info("agent query: %s", question)
    return await agent.run(question)


def _handle_deferred(agent: Agent, result: Any) -> None:  # pragma: no cover
    """Show pending approval requests and ask the user to confirm or cancel."""
    pending = result.output

    print("\n⚠️  The agent wants to perform the following submission(s):")
    for call in pending.approvals:
        print(f"   Tool  : {call.tool_name}")
        print(f"   Inputs: {call.args}")

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
