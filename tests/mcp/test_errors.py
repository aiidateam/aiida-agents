"""Tests for the MCP-surface error adapters in ``aiida_agents.mcp._errors``.

The wrappers are surface-agnostic, so these exercise them against tiny local
functions that raise the relevant exceptions, rather than through a real tool and
a database. The shared ``describe_aiida_error`` message builder (reused by the
agent boundary too) is unit-tested here as the single source of that wording.
"""

from __future__ import annotations

import asyncio

import pytest
from aiida.common.exceptions import AiidaException, MultipleObjectsError, NotExistent
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError

from aiida_agents.mcp._errors import (
    describe_aiida_error,
    register_tool,
    to_mcp_tool_error,
)


@pytest.mark.parametrize(
    ("exc", "present", "absent"),
    [
        pytest.param(
            NotExistent("no node 42"),
            ["no node 42", "list_processes()", "query_nodes()"],
            [],
            id="not-found-points-at-listing-tools",
        ),
        pytest.param(
            MultipleObjectsError("ambiguous uuid 'ab'"),
            ["ambiguous uuid 'ab'", "full uuid"],
            ["list_processes()"],
            id="ambiguous-asks-for-full-identifier-not-listing",
        ),
        pytest.param(
            AiidaException("some other failure"),
            ["some other failure"],
            ["list_processes()", "full uuid"],
            id="other-reported-as-is-without-a-misfit-hint",
        ),
    ],
)
def test_describe_aiida_error(
    exc: AiidaException, present: list[str], absent: list[str]
) -> None:
    """The message keeps the original error and adds guidance fitted to its type."""
    message = describe_aiida_error(exc)
    assert all(substring in message for substring in present)
    assert not any(substring in message for substring in absent)


@pytest.mark.parametrize(
    ("raised", "surfaced"),
    [
        pytest.param(
            NotExistent("no node 42"), ToolError, id="aiida-becomes-toolerror"
        ),
        pytest.param(ValueError("nope"), ValueError, id="non-aiida-propagates"),
    ],
)
def test_to_mcp_tool_error_boundary(
    raised: Exception, surfaced: type[Exception]
) -> None:
    """AiiDA exceptions become a ToolError; everything else propagates unchanged."""

    @to_mcp_tool_error
    def tool() -> None:
        raise raised

    with pytest.raises(surfaced, match=str(raised)):
        tool()


def test_to_mcp_tool_error_passes_success_through() -> None:
    """A successful call returns its value unchanged; only failures are adapted."""

    @to_mcp_tool_error
    def tool(value: int) -> int:
        return value + 1

    assert tool(41) == 42


def test_registered_tool_surfaces_tool_error() -> None:
    """A tool registered via ``register_tool`` reports a ``ToolError`` to a client.

    End-to-end: proves ``register_tool`` applies the adapter and that fastmcp
    surfaces the converted error over the wire (the wrapper itself is unit-tested
    above; this pins the registration wiring).
    """
    mcp = FastMCP(name="test")

    def boom(identifier: str) -> str:
        """A tool that always fails."""
        raise NotExistent("no node 987654321")

    register_tool(mcp, boom)

    async def _call() -> None:
        async with Client(mcp) as client:
            await client.call_tool("boom", {"identifier": "987654321"})

    with pytest.raises(ToolError, match="987654321"):
        asyncio.run(_call())
