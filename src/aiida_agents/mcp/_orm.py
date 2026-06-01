"""Shared ORM helpers for the MCP tools."""

from __future__ import annotations

from aiida import orm
from aiida.common.exceptions import NotExistent
from fastmcp.exceptions import ToolError

from ._types import Identifier


def load_node(identifier: Identifier) -> orm.Node:
    """Load a node by pk or uuid, raising ``ToolError`` if it does not exist.

    A purely numeric identifier is treated as an integer pk; anything else is
    passed through to ``orm.load_node`` as a uuid (or uuid prefix). Accepting a
    bare string for both is what lets the MCP Inspector send a uuid without
    JSON-quoting it.
    """
    resolved: int | str = int(identifier) if identifier.isdigit() else identifier
    try:
        return orm.load_node(resolved)
    except NotExistent as exc:
        raise ToolError(
            f"No node found with identifier={identifier!r}. "
            "Use list_processes() or query_nodes() to find valid identifiers."
        ) from exc
