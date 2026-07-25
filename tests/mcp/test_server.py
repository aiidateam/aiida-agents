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


# The tools that reach the database. Kept off the MCP server (see the test
# below); ``execute_workflow_spec`` delegates to ``submit_workflow``, so both
# are writes even though only the latter calls the engine directly.
_WRITE_TOOLS = {"submit_workflow", "execute_workflow_spec", "import_structure"}


def _tool_functions() -> set[str]:
    """Public tool functions defined across every ``aiida_agents.tools`` module.

    Walks the per-agent subpackages (``tools/analysis/``, ``tools/execution/``,
    ...) recursively, so neither a new tool, a new tool module, nor a whole new
    agent's tool package needs to be listed by hand.
    """
    names: set[str] = set()
    for mod_info in pkgutil.walk_packages(tools.__path__, prefix=f"{tools.__name__}."):
        # Skip private modules and anything inside a private subpackage.
        if any(part.startswith("_") for part in mod_info.name.split(".")):
            continue
        module = importlib.import_module(mod_info.name)
        for name, func in inspect.getmembers(module, inspect.isfunction):
            if (
                func.__module__ == module.__name__  # defined here, not imported
                and not name.startswith("_")  # exclude private helpers
            ):
                names.add(name)
    return names


def test_server_registers_read_tools_only() -> None:
    """The server exposes exactly the read tools, and never a write tool.

    The write tools are surface-agnostic like the others (they live under
    ``aiida_agents.tools.execution``, so they *are* discovered), but they reach
    the database, so they must go only through the HITL-gated agents (ADR-08),
    never the unauthenticated MCP server. The server therefore registers every
    discovered tool *except* the writes.
    """
    from aiida_agents.tools.execution import submit  # kept separate, not re-exported

    registered = {tool.name for tool in asyncio.run(mcp.list_tools())}
    discovered = _tool_functions()
    assert hasattr(submit, "submit_workflow")
    # Surface-agnostic tools, so discovered...
    assert _WRITE_TOOLS <= discovered
    # ...but never exposed on the server.
    assert not (_WRITE_TOOLS & registered)
    assert registered == discovered - _WRITE_TOOLS  # exactly the read tools


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
