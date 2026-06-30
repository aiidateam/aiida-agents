"""Tests for the MCP-surface error adapters in ``aiida_agents.mcp._errors``.

The wrappers are surface-agnostic, so these exercise them against tiny local
functions that raise the relevant exceptions, rather than through a real tool and
a database. The shared ``describe_aiida_error`` wording they reuse is unit-tested
where it is defined, in ``tests/tools/test_errors.py``.
"""

from __future__ import annotations

import pytest
from aiida.common.exceptions import NotExistent
from fastmcp.exceptions import ToolError

from aiida_agents.mcp._errors import to_mcp_tool_error


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
