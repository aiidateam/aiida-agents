"""Tests for ``aiida_agents.mcp.server``."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import pkgutil

import pytest
from aiida.common.exceptions import NotExistent
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError

from aiida_agents import tools
from aiida_agents.mcp.server import mcp
from aiida_agents.mcp.tools import register_tool


def _tool_functions() -> set[str]:
    """Public tool functions defined across every ``aiida_agents.tools`` module.

    Discovers both the tool modules and their functions, so neither a new tool
    nor a whole new tool module needs to be listed by hand.
    """
    names: set[str] = set()
    for _, mod_name, _ in pkgutil.iter_modules(tools.__path__):
        if mod_name.startswith("_"):
            continue
        module = importlib.import_module(f"{tools.__name__}.{mod_name}")
        for name, func in inspect.getmembers(module, inspect.isfunction):
            if (
                func.__module__ == module.__name__  # defined here, not imported
                and not name.startswith("_")  # exclude private helpers
            ):
                names.add(name)
    return names


def test_server_registers_read_tools_only() -> None:
    """The server exposes exactly the read tools, and never ``submit_workflow``.

    ``submit_workflow`` is a surface-agnostic tool like the others (it lives in
    ``aiida_agents.tools.submit``, so it *is* discovered), but it writes to the
    database, so it must be reached only through the HITL-gated agent (ADR-08),
    never the unauthenticated MCP server. The server therefore registers every
    discovered tool *except* the write one.
    """
    from aiida_agents.tools import submit  # the write tool exists, kept separate

    registered = {tool.name for tool in asyncio.run(mcp.list_tools())}
    discovered = _tool_functions()
    assert hasattr(submit, "submit_workflow")
    assert "submit_workflow" in discovered  # a surface-agnostic tool, so discovered
    assert "submit_workflow" not in registered  # but never exposed on the server
    assert registered == discovered - {"submit_workflow"}  # exactly the read tools


def test_register_tool_surfaces_tool_error() -> None:
    """A tool registered via ``register_tool`` reports a ``ToolError`` to the client.

    Proves registration applies the adapter and fastmcp surfaces it over the wire
    (the adapter itself is unit-tested in ``test_errors``).
    """
    server = FastMCP(name="test")

    def boom(identifier: str) -> str:
        raise NotExistent("no node 987654321")

    register_tool(server, boom)

    async def _call() -> None:
        async with Client(server) as client:
            await client.call_tool("boom", {"identifier": "987654321"})

    with pytest.raises(ToolError, match="987654321"):
        asyncio.run(_call())
