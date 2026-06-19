"""MCP tools for AiiDA process nodes."""

from __future__ import annotations

import logging
import typing as t

from aiida import orm
from fastmcp import FastMCP

from .._orm import WrongNodeType, load_node
from .._types import Identifier, ProcessRecord, ProcessStatus
from .._errors import register_tool

logger = logging.getLogger(__name__)


def get_process_status(identifier: Identifier) -> ProcessStatus:
    """Get the status and exit code of an AiiDA process by its pk or uuid."""
    logger.debug("get_process_status(identifier=%r)", identifier)
    node = load_node(identifier)
    # A valid identifier for a *data* node would otherwise hit AttributeError on
    # node.process_label below; raise WrongNodeType (an AiidaException the
    # surfaces adapt) so the model/client gets a clear message, not a crash.
    if not isinstance(node, orm.ProcessNode):
        msg = (
            f"Node {identifier} is not a process node (type {type(node).__name__}). "
            "Use query_nodes() to explore data nodes."
        )
        raise WrongNodeType(msg)
    return {
        "pk": t.cast(int, node.pk),  # a loaded node is always stored
        "process_label": t.cast(str, node.process_label),  # always set on a process
        "process_type": t.cast(str, node.process_type),  # a process always has one
        "state": node.process_state.value if node.process_state else None,
        "exit_status": node.exit_status,
        "exit_message": node.exit_message,
    }


def list_processes(limit: int = 10) -> list[ProcessRecord]:
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

    records: list[ProcessRecord] = [
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
    register_tool(mcp, get_process_status)
    register_tool(mcp, list_processes)
