"""Tool registration for aiida-agents MCP server."""

from __future__ import annotations
from fastmcp import FastMCP
from . import nodes, processes, structures


def register_all(mcp: FastMCP) -> None:
    """Register all tools on the MCP server."""
    processes.register(mcp)
    nodes.register(mcp)
    structures.register(mcp)
