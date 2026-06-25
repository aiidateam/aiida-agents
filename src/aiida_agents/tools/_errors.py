"""Surface-agnostic AiiDA exception helpers.

This module owns the shared error-description logic used by both the MCP
server surface and the pydantic-ai agent surface. It depends only on AiiDA,
never on a surface framework (fastmcp, pydantic-ai, etc.).

Surface-specific adapters live in their own modules:
- MCP server: ``aiida_agents.mcp._errors`` (``to_mcp_tool_error``, ``register_tool``)
- Agent:       ``aiida_agents.agents._errors`` (``RetryOnToolError``)
"""

from __future__ import annotations

from aiida.common.exceptions import AiidaException, MultipleObjectsError, NotExistent

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
