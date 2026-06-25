"""Shared ORM helpers for the MCP tools."""

from __future__ import annotations

from aiida import orm
from aiida.common.exceptions import AiidaException

from ._types import Identifier


class WrongNodeType(AiidaException):
    """A node exists for the identifier but is not the type a tool requires.

    Subclasses ``AiidaException`` so the surfaces adapt it like any other AiiDA
    error (a ``ToolError`` on the MCP server, a ``ModelRetry`` for the agent).
    ``NotExistent`` would misdescribe it: the node does exist, it is just the
    wrong kind for this tool.
    """


def load_node(identifier: Identifier) -> orm.Node:
    """Load a node by pk or uuid.

    A purely numeric identifier is treated as an integer pk; anything else is
    passed through to ``orm.load_node`` as a uuid (or uuid prefix). Accepting a
    bare string for both is what lets the MCP Inspector send a uuid without
    JSON-quoting it.

    Raises ``aiida.common.exceptions.NotExistent`` unchanged if no matching
    node exists; callers adapt it at their own boundary (MCP server raises
    ``ToolError``, the agent raises ``ModelRetry``).
    """
    resolved: int | str = int(identifier) if identifier.isdigit() else identifier
    return orm.load_node(resolved)
