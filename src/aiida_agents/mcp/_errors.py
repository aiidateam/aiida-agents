"""MCP-surface adaptation of AiiDA exceptions raised by the shared tool functions.

Tool functions (in nodes.py, processes.py, structures.py) call into shared
helpers like ``load_node()`` and let AiiDA's native exceptions (``NotExistent``,
...) propagate unchanged. Each consumer surface adapts them at its own boundary.

This module owns the MCP-server boundary:

- ``to_mcp_tool_error`` is the adapter: AiiDA exceptions become a clean
  ``ToolError`` for the MCP client instead of an uncaught 500.
- ``register_tool`` registers a tool with the adapter always applied, so a tool
  cannot be wired up with an uncaught ``AiidaException`` by mistake.

``describe_aiida_error`` has moved to ``aiida_agents.tools._errors`` — it depends
only on AiiDA and is shared by both surfaces. Import it from there.

The agent boundary lives in ``aiida_agents.agents._errors`` (not here) because it
depends on pydantic-ai, which the MCP server surface must not import.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import ParamSpec, TypeVar

from aiida.common.exceptions import AiidaException
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from aiida_agents.tools._errors import describe_aiida_error

__all__ = ["describe_aiida_error", "to_mcp_tool_error", "register_tool"]

P = ParamSpec("P")
R = TypeVar("R")


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
