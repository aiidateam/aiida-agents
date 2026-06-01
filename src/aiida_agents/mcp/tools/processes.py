"""MCP tools for AiiDA process nodes."""

from __future__ import annotations

import logging

from aiida import orm
from fastmcp import FastMCP

from .._orm import load_node
from .._types import Identifier

logger = logging.getLogger(__name__)


def get_process_status(identifier: Identifier) -> dict[str, str | int | None]:
    """Get the status and exit code of an AiiDA process by its pk or uuid."""
    logger.debug("get_process_status(identifier=%r)", identifier)
    node = load_node(identifier)
    return {
        "pk": node.pk,
        "process_label": node.process_label,
        "process_type": node.process_type,
        "state": node.process_state.value if node.process_state else None,
        "exit_status": node.exit_status,
        "exit_message": node.exit_message,
    }


def list_processes(limit: int = 10) -> list[dict[str, str | int | None]]:
    """List recent AiiDA processes, newest first."""
    logger.debug("list_processes(limit=%d)", limit)

    # One query projecting the state/exit_status attributes, rather than a
    # follow-up attribute lookup per process.
    qb = orm.QueryBuilder()
    qb.append(
        orm.ProcessNode,
        tag="process",
        project=[
            "id",
            "uuid",
            "node_type",
            "process_type",
            "attributes.process_state",
            "attributes.exit_status",
        ],
    )
    qb.order_by({"process": {"ctime": "desc"}})
    qb.limit(limit)

    records: list[dict[str, str | int | None]] = [
        {
            "pk": pk,
            "uuid": uuid,
            "node_type": node_type,
            "process_type": process_type,
            "state": state,
            "exit_status": exit_status,
        }
        for pk, uuid, node_type, process_type, state, exit_status in qb.iterall()
    ]
    logger.debug("list_processes: returned %d records", len(records))
    return records


def register(mcp: FastMCP) -> None:
    """Register process tools on the MCP server."""
    mcp.tool()(get_process_status)
    mcp.tool()(list_processes)
