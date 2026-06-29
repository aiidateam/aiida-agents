"""Tool registration for the aiida-agents MCP server.

This module is the only place that depends on both ``aiida_agents.tools``
(the surface-agnostic tool functions) and ``fastmcp`` (the MCP server
framework). Tool functions live in ``aiida_agents.tools``; registration
lives here.
"""

from __future__ import annotations

from fastmcp import FastMCP

from aiida_agents.mcp._errors import register_tool
from aiida_agents.tools.nodes import get_node_inputs, get_node_outputs, query_nodes
from aiida_agents.tools.processes import get_process_status, list_processes
from aiida_agents.tools.structures import search_structures

# submit is intentionally NOT registered here. submit_workflow writes to the
# database and must be reached only through the HITL-gated agent (ADR-08),
# never this unauthenticated server. The agent imports it directly from
# aiida_agents.mcp.tools.submit.


def register_all(mcp: FastMCP) -> None:
    """Register the read-only tools on the MCP server.

    Read tools are wrapped with ``register_tool`` so AiiDA exceptions surface
    as a clean ``ToolError`` for the MCP client.
    """
    register_tool(mcp, get_process_status)
    register_tool(mcp, list_processes)
    register_tool(mcp, get_node_inputs)
    register_tool(mcp, get_node_outputs)
    register_tool(mcp, query_nodes)
    register_tool(mcp, search_structures)
