"""Command-line entry point for the AiiDA agent."""

from __future__ import annotations

import asyncio
import logging

from pydantic_ai import Agent

logger = logging.getLogger(__name__)


async def ask(agent: Agent, question: str) -> None:  # pragma: no cover
    """Run a single query through the agent and stream the response to stdout."""
    logger.info("agent query: %s", question)
    async with agent.run_stream(question) as result:
        print("Agent: ", end="", flush=True)
        printed_len = 0
        async for chunk in result.stream_text(debounce_by=None):
            print(chunk[printed_len:], end="", flush=True)
            printed_len = len(chunk)
        print()


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
        asyncio.run(ask(agent, question))
