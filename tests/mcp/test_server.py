"""Tests for ``aiida_agents.mcp.server``."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import logging
import pkgutil

import pytest
from aiida.common.exceptions import IncompatibleStorageSchema, NotExistent
from aiida.manage import get_manager
from aiida.manage.manager import Manager
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError

from aiida_agents import tools
from aiida_agents.mcp.server import mcp
from aiida_agents.mcp.tools import register_tool


# The tools that reach the database. Kept off the MCP server (see the test
# below); ``submit_process_spec`` delegates to ``submit_workflow``, so both
# are writes even though only the latter calls the engine directly.
_WRITE_TOOLS = {
    "submit_workflow",
    "submit_process_spec",
    "submit_process_batch",
    "import_structure",
}

# Read-only, and still not exported. ``run_python_snippet`` executes arbitrary
# Python; it is safe in the agents because it runs against a disposable copy
# of the user's storage, and that guarantee rests entirely on
# ``AIIDA_AGENTS_SANDBOX_PROFILE`` naming a profile someone verified with
# ``aiida-agents sandbox check``. An MCP client cannot verify that and we
# cannot see whether it holds, so the honest export is none: a client that
# wants to run code can run it itself, with its own consent.
_UNEXPORTED_READ_TOOLS = {"run_python_snippet"}


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
    never the unauthenticated MCP server.

    ``_UNEXPORTED_READ_TOOLS`` is the third case: read-only, but withheld for
    the reason recorded beside it. The server registers every discovered tool
    except those two sets, so adding a tool to either is a deliberate edit
    rather than something that can happen by forgetting.
    """
    from aiida_agents.tools.execution import submit  # kept separate, not re-exported

    registered = {tool.name for tool in asyncio.run(mcp.list_tools())}
    discovered = _tool_functions()
    assert hasattr(submit, "submit_workflow")
    # Surface-agnostic tools, so discovered...
    assert _WRITE_TOOLS <= discovered
    # ...but never exposed on the server.
    assert not (_WRITE_TOOLS & registered)
    assert not (_UNEXPORTED_READ_TOOLS & registered)
    assert registered == discovered - _WRITE_TOOLS - _UNEXPORTED_READ_TOOLS


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


def test_lifespan_opens_the_storage_before_serving_tools(
    unopened_profile_storage: Manager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No tool may be the first thing to open the storage.

    fastmcp runs a sync tool on a worker thread and AiiDA opens storage lazily,
    so two tool threads taking that first open together race the PID-named temp
    move in ``ProfileAccessManager`` and one raises ``FileNotFoundError``. The
    assertion is *inside* the context manager: opening the storage on the way
    out would leave the race exactly as it was.
    """
    from aiida_agents.mcp import server as server_mod

    # The lifespan resets root logging handlers; keep that out of the test session.
    monkeypatch.setattr(server_mod, "_configure_logging", lambda *_a, **_k: None)

    manager = unopened_profile_storage
    assert not manager.profile_storage_loaded

    async def _enter() -> None:
        async with server_mod._lifespan(server_mod.mcp):
            assert manager.profile_storage_loaded, (
                "the lifespan must open the profile storage before serving tools"
            )

    asyncio.run(_enter())


def test_a_storage_that_cannot_be_opened_is_reported_before_the_traceback(
    unmigrated_storage_error: IncompatibleStorageSchema,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Startup fails, but the operator reads the fix instead of hunting for it.

    An unmigrated storage is the ordinary case here (any ``aiida-core`` upgrade
    puts a profile there), and AiiDA's own message ends with the ``verdi``
    command that resolves it.
    """
    from aiida_agents.mcp import server as server_mod

    monkeypatch.setattr(server_mod, "_configure_logging", lambda *_a, **_k: None)

    def _unmigrated() -> None:
        raise unmigrated_storage_error

    # Injected where AiiDA opens the storage, so the code under test runs whole.
    monkeypatch.setattr(get_manager(), "get_profile_storage", _unmigrated)

    async def _enter() -> None:
        async with server_mod._lifespan(server_mod.mcp):
            pass

    with (
        caplog.at_level(logging.ERROR, logger=server_mod.__name__),
        pytest.raises(IncompatibleStorageSchema),
    ):
        asyncio.run(_enter())

    assert "verdi -p test storage migrate" in caplog.text
    # Not just present: on the record's first line, so a log search finds the
    # failure rather than a prefix followed by a blank.
    assert (
        caplog.records[0]
        .getMessage()
        .startswith("cannot open the profile storage: Database schema version")
    )
