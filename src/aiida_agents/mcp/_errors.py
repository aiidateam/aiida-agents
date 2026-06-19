"""MCP-surface adaptation of AiiDA exceptions raised by the shared tool functions.

Tool functions (in nodes.py, processes.py, structures.py) call into shared
helpers like ``load_node()`` and let AiiDA's native exceptions (``NotExistent``,
...) propagate unchanged. Each consumer surface adapts them at its own boundary.

This module owns the MCP-server boundary:

- ``describe_aiida_error`` turns an AiiDA exception into a user-facing message
  with recovery guidance fitted to the failure. It is shared: the agent boundary
  (``aiida_agents.agents._errors``) reuses it so both surfaces speak the same
  way. It depends only on AiiDA, never on a surface framework.
- ``to_mcp_tool_error`` is the adapter: AiiDA exceptions become a clean
  ``ToolError`` for the MCP client instead of an uncaught 500.
- ``register_tool`` registers a tool with the adapter always applied, so a tool
  cannot be wired up with an uncaught ``AiidaException`` by mistake.

The agent boundary lives in ``aiida_agents.agents._errors`` (not here) because it
depends on pydantic-ai, which the MCP server surface must not import.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import ParamSpec, TypeVar

from aiida.common.exceptions import AiidaException, MultipleObjectsError, NotExistent
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

P = ParamSpec("P")
R = TypeVar("R")

_FIND_IDENTIFIERS = "Use list_processes() or query_nodes() to find valid identifiers."


def describe_aiida_error(exc: AiidaException) -> str:
    """Build a user-facing message from an AiiDA exception, with recovery guidance.

    The guidance is fitted to the failure: a missing identifier points at the
    listing tools, an ambiguous uuid prefix asks for a longer identifier, and any
    other AiiDA error is reported as-is (without a hint that would misdescribe it).

    :param exc: The AiiDA exception a tool function let propagate.
    :return: The exception text plus what to do next.
    """
    if isinstance(exc, NotExistent):
        return f"{exc} {_FIND_IDENTIFIERS}"
    if isinstance(exc, MultipleObjectsError):
        return f"{exc} The identifier is ambiguous; give the full uuid or the pk."
    return str(exc)


def to_mcp_tool_error(func: Callable[P, R]) -> Callable[P, R]:
    """Wrap a tool function so AiiDA exceptions surface as a ``ToolError``.

    :param func: A tool function that may let an ``AiidaException`` propagate.
    :return: The same function, with AiiDA exceptions converted to ``ToolError``.
    """

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return func(*args, **kwargs)
        except AiidaException as exc:
            msg = describe_aiida_error(exc)
            raise ToolError(msg) from exc

    return wrapper


def register_tool(mcp: FastMCP, func: Callable[..., object]) -> None:
    """Register a tool on the MCP server with the AiiDA-exception adapter applied.

    Registering through this helper, rather than ``mcp.tool()`` directly, keeps
    every tool on the same boundary, so a newly added tool cannot reach the client
    with an uncaught ``AiidaException``.

    :param mcp: The MCP server to register the tool on.
    :param func: The tool function to wrap and register.
    """
    mcp.tool()(to_mcp_tool_error(func))
