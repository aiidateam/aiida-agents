"""Surface-agnostic tools for inspecting AiiDA process nodes."""

from __future__ import annotations

import logging
import typing as t

from aiida import orm
from aiida.cmdline.utils.common import (
    get_calcjob_report,
    get_process_function_report,
    get_workchain_report,
)
from aiida.common.log import LOG_LEVELS
from pydantic import Field

from ._orm import WrongNodeType, load_node
from ._types import Identifier, ProcessRecord, ProcessReport, ProcessStatus


logger = logging.getLogger(__name__)

_DEFAULT_LEVELNAME = "REPORT"


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


def get_process_report(
    identifier: Identifier,
    levelname: t.Annotated[
        str,
        Field(
            description=(
                "Minimum log level to include for a WorkChain's report (one of "
                f"{sorted(LOG_LEVELS)}). Ignored for a CalcJob or calc/work "
                "function, whose reports are always unfiltered."
            )
        ),
    ] = _DEFAULT_LEVELNAME,
) -> ProcessReport:
    """Get the log report for an AiiDA process -- why it failed, warned, or what it logged.

    Mirrors `verdi process report`: a CalcJob's scheduler stdout/stderr plus its
    log messages, a WorkChain's (and its sub-workchains') log messages down to
    `levelname`, or a calcfunction/workfunction's log messages. Use this to see
    *why* a process behaved the way it did -- not just its exit code from
    `get_process_status`.

    Args:
        identifier: The process's pk or uuid.
        levelname: Minimum log level for a WorkChain's report.

    Returns:
        The process's identity/status plus its formatted report text.
    """
    logger.debug(
        "get_process_report(identifier=%r, levelname=%r)", identifier, levelname
    )
    if levelname not in LOG_LEVELS:
        msg = (
            f"levelname {levelname!r} is not a recognised log level. "
            f"Use one of: {sorted(LOG_LEVELS)}."
        )
        raise ValueError(msg)

    node = load_node(identifier)
    # Same rationale as get_process_status: a data-node pk would otherwise hit
    # an AttributeError deep inside the report helpers below.
    if not isinstance(node, orm.ProcessNode):
        msg = (
            f"Node {identifier} is not a process node (type {type(node).__name__}). "
            "Use query_nodes() to explore data nodes."
        )
        raise WrongNodeType(msg)

    if isinstance(node, orm.CalcJobNode):
        report_text = get_calcjob_report(node)
    elif isinstance(node, orm.WorkChainNode):
        report_text = get_workchain_report(node, levelname)
    elif isinstance(node, (orm.CalcFunctionNode, orm.WorkFunctionNode)):
        report_text = get_process_function_report(node)
    else:
        report_text = (
            f"No log report available for process node type {type(node).__name__}."
        )

    return {
        "pk": t.cast(int, node.pk),
        "process_label": t.cast(str, node.process_label),
        "node_type": type(node).__name__,
        "state": node.process_state.value if node.process_state else None,
        "exit_status": node.exit_status,
        "exit_message": node.exit_message,
        "report": report_text,
    }
