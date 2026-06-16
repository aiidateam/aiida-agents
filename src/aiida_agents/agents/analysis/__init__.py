"""Analysis agent — read-only exploration of an AiiDA profile.

This is the first concrete agent (ADR-04). It exposes read-only MCP tools
for querying processes, nodes, and crystal structures, and answers
conceptual questions from its system prompt.

Public API
----------
get_agent()
    Build and return a ready-to-use Analysis agent instance.
"""

from __future__ import annotations

from importlib.resources import files

from typing import Any

from pydantic_ai import Agent

from aiida_agents.agents._models import get_model
from aiida_agents.mcp.tools.nodes import get_node_inputs, get_node_outputs, query_nodes
from aiida_agents.mcp.tools.processes import get_process_status, list_processes
from aiida_agents.mcp.tools.structures import search_structures
from aiida_agents.rag import search_aiida_docs

_TOOLS: list[Any] = [
    get_process_status,
    list_processes,
    query_nodes,
    get_node_inputs,
    get_node_outputs,
    search_structures,
    search_aiida_docs,
]

_SYSTEM_PROMPT = (
    files(__package__).joinpath("prompt.md").read_text(encoding="utf-8").strip()
)


def get_agent() -> Agent:
    """Build and return the Analysis agent.

    Called from the CLI after environment variables are loaded, so model
    construction always sees a fully populated environment.
    """
    return Agent(get_model(), tools=_TOOLS, system_prompt=_SYSTEM_PROMPT)
