"""MCP-surface adapter: turn AiiDA exceptions from the tool functions into a
clean ``ToolError`` for the client.

``register_tool`` (which applies this adapter to every registered tool) lives in
``aiida_agents.mcp.tools``; ``describe_aiida_error`` in ``aiida_agents.tools._errors``.
The agent's adapter is ``aiida_agents.agents._errors`` (it needs pydantic-ai,
which this surface must not import).
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import ParamSpec, TypeVar

from aiida.common.exceptions import AiidaException
from fastmcp.exceptions import ToolError

from aiida_agents.tools._errors import describe_aiida_error  # used by the wrapper

__all__ = ["to_mcp_tool_error"]

P = ParamSpec("P")
R = TypeVar("R")


def to_mcp_tool_error(func: Callable[P, R]) -> Callable[P, R]:
    """Wrap a tool so AiiDA exceptions surface as a ``ToolError``."""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return func(*args, **kwargs)
        except AiidaException as exc:
            msg = describe_aiida_error(exc)
            raise ToolError(msg) from exc

    return wrapper
