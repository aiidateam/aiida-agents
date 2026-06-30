"""Surface-agnostic AiiDA exception helpers, shared by both surfaces.

``describe_aiida_error`` depends only on AiiDA. The surface adapters that use it
live in ``aiida_agents.mcp._errors`` and ``aiida_agents.agents._errors``.
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
