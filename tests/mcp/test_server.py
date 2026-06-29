"""Tests for ``aiida_agents.mcp.server``."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import pkgutil

from aiida_agents import tools
from aiida_agents.mcp.server import mcp


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
                and name != "register"  # exclude the registration hook
            ):
                names.add(name)
    return names


def test_server_registers_read_tools_only() -> None:
    """The server exposes exactly the read tools, and never ``submit_workflow``.

    ``submit_workflow`` writes to the database, so it must be reached only
    through the HITL-gated agent (ADR-08), never the unauthenticated MCP
    server. It lives in ``mcp/tools/submit.py`` and is imported directly by the
    agent, deliberately kept out of the surface-agnostic ``aiida_agents.tools``
    package, so the server neither discovers nor registers it.
    """
    from aiida_agents.mcp.tools import submit  # the write tool exists, kept separate

    registered = {tool.name for tool in asyncio.run(mcp.list_tools())}
    discovered = _tool_functions()
    assert hasattr(submit, "submit_workflow")
    assert registered == discovered  # the server exposes exactly the read tools
    assert "submit_workflow" not in discovered  # not in the agnostic tools package
    assert "submit_workflow" not in registered
