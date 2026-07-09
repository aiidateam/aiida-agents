"""AiiDA Agents MCP package.

Public API
----------
mcp
    The MCP server instance (run it via ``aiida-agents mcp serve`` or
    ``mcp.run()``).
"""

from __future__ import annotations
from .server import mcp

__all__ = ["mcp"]
