"""Trace agent tool calls and outputs for a single query.

Prints the full sequence of tool calls (name, args) and tool returns
(content) the model produced while answering `query`, followed by the
final output. Useful for diagnosing cases where the agent's final
answer doesn't match what a direct tool call would return — e.g.
checking whether search_aiida_docs was actually invoked, or whether
submit_workflow was called with fabricated arguments.

Usage:
    uv run python dev/trace_agent.py "How do I submit a workflow using the builder?"
"""

from __future__ import annotations

import asyncio
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
        print('Usage: uv run python dev/trace_agent.py "your query here"')
        sys.exit(1)
    query = sys.argv[1]
    asyncio.run(trace(query))


if __name__ == "__main__":
    main()
