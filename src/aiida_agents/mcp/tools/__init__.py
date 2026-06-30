"""Register the surface-agnostic tool functions onto a fastmcp server.

Each read tool is wrapped with the adapter from ``aiida_agents.mcp._errors`` so it
cannot reach the client with an uncaught ``AiidaException``.
"""

from __future__ import annotations

from collections.abc import Callable

from fastmcp import FastMCP

from aiida_agents.mcp._errors import to_mcp_tool_error
from aiida_agents.tools import (
    get_node_inputs,
    get_node_outputs,
    get_process_status,
    list_processes,
    query_nodes,
    search_structures,
)

# submit is intentionally NOT registered: it writes, so it goes only through the
# HITL-gated agent (ADR-08). The agent imports it from aiida_agents.tools.submit.


def register_tool(mcp: FastMCP, func: Callable[..., object]) -> None:
    """Register a tool with the AiiDA-exception adapter applied.

    Routing every tool through here keeps a newly added one from reaching the
    client with an uncaught ``AiidaException``.
    """
    mcp.tool()(to_mcp_tool_error(func))


def register_all(mcp: FastMCP) -> None:
    """Register the read-only tools on the MCP server."""
    register_tool(mcp, get_process_status)
    register_tool(mcp, list_processes)
    register_tool(mcp, get_node_inputs)
    register_tool(mcp, get_node_outputs)
    register_tool(mcp, query_nodes)
    register_tool(mcp, search_structures)
