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
from ._types import (
    Identifier,
    ProcessRecord,
    ProcessReport,
    ProcessStatus,
    RetrievedFileContent,
    RetrievedFileInfo,
    RetrievedListing,
)


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


#: Cap on the text returned from one file. A converged Quantum ESPRESSO run
#: can emit tens of megabytes; handing that to a model would blow the context
#: window and cost a fortune to learn one line. Callers who need more can page
#: with `tail_lines` or read a specific file.
_MAX_CHARS = 20_000


def _retrieved_folder(identifier: Identifier) -> tuple[orm.CalcJobNode, orm.FolderData]:
    """Resolve an identifier to a CalcJob and the folder of files it brought back.

    Only a CalcJob retrieves files: a WorkChain orchestrates calculations but
    stores no output of its own. Rather than guessing which of a WorkChain's
    calculations was meant, this says so and points at the tools that find it,
    since "the failed one" is a question about the provenance graph.
    """
    node = load_node(identifier)

    if not isinstance(node, orm.CalcJobNode):
        if isinstance(node, orm.WorkChainNode):
            msg = (
                f"Node {identifier} is a WorkChain, which retrieves no files of its "
                "own -- its calculations do. Find the CalcJob you want with "
                "get_node_outputs() or query_nodes(), then call this on that pk."
            )
        else:
            msg = (
                f"Node {identifier} is a {type(node).__name__}, not a CalcJob, so it "
                "has no retrieved files."
            )
        raise WrongNodeType(msg)

    try:
        retrieved = node.outputs.retrieved
    except AttributeError:
        msg = (
            f"CalcJob {identifier} has no `retrieved` output. It may not have "
            "finished, or its files may not have been retrieved yet -- check "
            "get_process_status()."
        )
        raise WrongNodeType(msg) from None

    return node, retrieved


def list_retrieved_files(identifier: Identifier) -> RetrievedListing:
    """List the files a calculation brought back from the machine it ran on.

    These are the code's own outputs -- for Quantum ESPRESSO the main output
    file, plus the scheduler's stdout/stderr. `get_process_report` shows
    AiiDA's view of a run; these files are the *code's* view, which is where a
    physics failure ("convergence NOT achieved", "too many bands") is actually
    written.

    Call this first to see what is available, then read one with
    `get_retrieved_file`.

    Args:
        identifier: The CalcJob's pk or uuid.

    Returns:
        The calculation's identity and each retrieved file's name and size.
    """
    logger.debug("list_retrieved_files(identifier=%r)", identifier)
    node, retrieved = _retrieved_folder(identifier)

    files: list[RetrievedFileInfo] = []
    for obj in retrieved.base.repository.list_objects():
        try:
            size = len(
                retrieved.base.repository.get_object_content(obj.name, mode="rb")
            )
        except (OSError, ValueError, TypeError):  # a directory, or unreadable
            continue
        files.append({"name": obj.name, "size_bytes": size})

    return {
        "pk": t.cast(int, node.pk),
        "process_label": t.cast(str, node.process_label),
        "retrieved_pk": t.cast(int, retrieved.pk),
        "files": sorted(files, key=lambda f: f["name"]),
    }


def get_retrieved_file(
    identifier: Identifier,
    filename: t.Annotated[
        str,
        Field(
            description=(
                "Name of the file to read, exactly as list_retrieved_files "
                "reported it (e.g. 'aiida.out')."
            )
        ),
    ],
    tail_lines: t.Annotated[
        int | None,
        Field(
            description=(
                "Return only the last N lines. A code that fails usually says why "
                "at the end, so this is the cheap way to read a large output."
            )
        ),
    ] = None,
) -> RetrievedFileContent:
    """Read one file a calculation brought back -- the code's own output.

    Use this to find out *why* a calculation produced the result it did, when
    `get_process_status` gives only an exit code and `get_process_report` gives
    only AiiDA's log. The error a simulation code reports about its own physics
    is written here and nowhere else.

    Output is capped, and the result says whether it was truncated and how many
    lines the file actually has -- do not report a conclusion drawn from a
    truncated file without saying it was truncated.

    Args:
        identifier: The CalcJob's pk or uuid.
        filename: Which retrieved file to read.
        tail_lines: Return only the final N lines, where errors usually are.

    Returns:
        The file's text, with truncation flagged explicitly.
    """
    logger.debug(
        "get_retrieved_file(identifier=%r, filename=%r, tail_lines=%r)",
        identifier,
        filename,
        tail_lines,
    )
    node, retrieved = _retrieved_folder(identifier)

    available = sorted(obj.name for obj in retrieved.base.repository.list_objects())
    if filename not in available:
        msg = (
            f"{filename!r} is not among the retrieved files of {identifier}. "
            f"Available: {available}."
        )
        raise FileNotFoundError(msg)

    # Read bytes and decode leniently rather than asking for text: a code's
    # output can carry stray non-UTF-8 bytes, and a diagnostic tool that raises
    # UnicodeDecodeError on the file holding the error message is useless
    # exactly when it is needed.
    raw = retrieved.base.repository.get_object_content(filename, mode="rb").decode(
        "utf-8", errors="replace"
    )

    lines = raw.splitlines()
    lines_total = len(lines)

    selected = lines[-tail_lines:] if tail_lines is not None else lines
    content = "\n".join(selected)

    truncated = tail_lines is not None and lines_total > len(selected)
    if len(content) > _MAX_CHARS:
        # Keep the end: a code that fails says why in its final lines.
        content = content[-_MAX_CHARS:]
        selected = content.splitlines()
        truncated = True

    return {
        "pk": t.cast(int, node.pk),
        "filename": filename,
        "content": content,
        "truncated": truncated,
        "lines_returned": len(selected),
        "lines_total": lines_total,
    }
