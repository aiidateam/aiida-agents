"""Tool registration for aiida-agents MCP server."""

from __future__ import annotations

from fastmcp import FastMCP

from . import nodes, processes, structures

# submit is intentionally NOT registered here. submit_workflow writes to the
# database and must be reached only through the HITL-gated agent (ADR-08),
# never this unauthenticated server. The agent imports it directly from
# aiida_agents.mcp.tools.submit.


def register_all(mcp: FastMCP) -> None:
    """Register the read-only tools on the MCP server."""
    processes.register(mcp)
    nodes.register(mcp)
    structures.register(mcp)
