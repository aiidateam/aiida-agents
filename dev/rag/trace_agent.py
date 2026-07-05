"""Trace agent tool calls and outputs for a single query.

Prints the full sequence of tool calls (name, args) and tool returns
(content) the model produced while answering `query`, followed by the
final output. Useful for diagnosing cases where the agent's final
answer doesn't match what a direct tool call would return — e.g.
checking whether search_aiida_docs was actually invoked, or whether
submit_workflow was called with fabricated arguments.

Usage:
    uv run python dev/rag/trace_agent.py "How do I submit a workflow using the builder?"
"""

from __future__ import annotations

import asyncio
import logging
import sys

from pydantic_ai.messages import ToolCallPart, ToolReturnPart

from aiida_agents.agents import get_agent


async def trace(query: str) -> None:
    agent = get_agent()
    result = await agent.run(query)

    for msg in result.all_messages():
        for part in msg.parts:
            if isinstance(part, ToolCallPart):
                print("=== TOOL CALLED ===")
                print("tool_name:", part.tool_name)
                print("args:", part.args)
                print("tool_call_id:", part.tool_call_id)
                print()
            elif isinstance(part, ToolReturnPart):
                print("=== TOOL RETURNED ===")
                print("tool_name:", part.tool_name)
                print("tool_call_id:", part.tool_call_id)
                print("content:", part.content)
                print()

    print("=== FINAL OUTPUT ===")
    print(result.output)


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: uv run python dev/rag/trace_agent.py "your query here"')
        sys.exit(1)

    # Match the REPL's environment: DB tools need a loaded profile, and
    # AIIDA_AGENTS_LOG_LEVEL=DEBUG streams tool/library internals live
    # (the tool-call dump below still prints after the run).
    from aiida import load_profile
    from aiida_agents._logging import suppress_noisy_loggers
    from aiida_agents._settings import LoggingSettings

    logging.basicConfig(level=LoggingSettings().log_level, stream=sys.stderr)
    if logging.getLogger().getEffectiveLevel() <= logging.DEBUG:
        suppress_noisy_loggers()
    load_profile()
    query = sys.argv[1]
    asyncio.run(trace(query))


if __name__ == "__main__":
    main()
