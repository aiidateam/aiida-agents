"""Shared return-shape types for the surface-agnostic tool functions.

Both surfaces turn these ``TypedDict``s into the tool's output schema, so the
client sees named, typed fields instead of an opaque blob.
"""

from __future__ import annotations

import typing as t

from pydantic import Field

# Pydantic requires ``typing_extensions.TypedDict`` (not ``typing.TypedDict``)
# on Python < 3.12 to build a schema from these, so FastMCP can derive the
# tools' ``outputSchema``.
from typing_extensions import TypedDict

__all__ = [
    "CodeRecord",
    "DaemonStatus",
    "FailedStep",
    "FailureDiagnosis",
    "HandlerAttempt",
    "Identifier",
    "NodeLink",
    "ProcessCompletion",
    "ProcessRecord",
    "ProcessReport",
    "ProcessStatus",
    "QueryResult",
    "RegisteredHandler",
    "StructureImportResult",
    "StructureRecord",
    "SubmitResult",
]

# A node identifier: a pk or a uuid, both as a plain string. Using ``str``
# (rather than ``int | str``) means the MCP Inspector sends the value as-is
# without requiring JSON quotes; the loader coerces a purely numeric
# identifier back to an integer pk.
Identifier = t.Annotated[
    str, Field(description="Node pk or uuid (e.g. '42' or '0cef...')")
]


class ProcessStatus(TypedDict):
    """Return shape of ``get_process_status``."""

    pk: int
    process_label: str
    process_type: str
    state: str | None
    exit_status: int | None
    exit_message: str | None


class CodeRecord(TypedDict):
    """A row returned by ``list_codes``.

    ``full_label`` is the ``name@computer`` string a submission's ``code`` input
    takes as ``{"label": ...}``, so a caller can hand it straight back without
    reassembling it from the label and computer.
    """

    pk: int
    uuid: str
    label: str
    full_label: str
    computer: str | None
    default_calc_job_plugin: str | None
    node_type: str


class StructureImportResult(TypedDict):
    """Return shape of ``import_structure``.

    ``pk`` is the point of the tool: it is what a submission's ``structure``
    input takes as ``{"pk": ...}``.
    """

    pk: int
    uuid: str
    formula: str
    num_sites: int
    label: str


class ProcessRecord(TypedDict):
    """A row returned by ``list_recent_processes``."""

    pk: int
    uuid: str
    node_type: str
    process_type: str
    state: str | None
    exit_status: int | None


class DaemonStatus(TypedDict):
    """Return shape of ``get_daemon_status``.

    ``pending`` counts processes the daemon still owes work to. Read it
    together with ``running``: a stopped daemon and a non-zero ``pending`` is
    the state where nothing will ever progress on its own.
    """

    running: bool
    workers: int | None
    profile: str
    process_control: str | None
    pending: int
    oldest_pending_pk: int | None
    diagnosis: str


class ProcessReport(TypedDict):
    """Return shape of ``get_process_report``."""

    pk: int
    process_label: str
    node_type: str
    state: str | None
    exit_status: int | None
    exit_message: str | None
    report: str


class RetrievedFileInfo(TypedDict):
    """One file in a calculation's retrieved folder."""

    name: str
    size_bytes: int


class RetrievedListing(TypedDict):
    """Return shape of ``list_retrieved_files``."""

    pk: int
    process_label: str
    retrieved_pk: int
    files: list[RetrievedFileInfo]


class RetrievedFileContent(TypedDict):
    """Return shape of ``get_retrieved_file``.

    ``truncated`` and the two line counts are part of the contract rather than
    a detail: a silently shortened file reads exactly like a complete one, and
    a model drawing conclusions from the wrong half of an output has no way to
    know it is missing the part that mattered.
    """

    pk: int
    filename: str
    content: str
    truncated: bool
    lines_returned: int
    lines_total: int


class NodeLink(TypedDict):
    """A link returned by ``get_node_inputs`` / ``get_node_outputs``."""

    pk: int
    uuid: str
    node_type: str
    link_label: str
    link_type: str


class QueryResult(TypedDict):
    """What ``query_nodes`` returns.

    ``total`` counts every match, while ``records`` holds at most ``limit`` of
    them (and is empty for a count-only query), so a caller can report how many
    there are without fetching them. The record keys are whatever the query
    projected, hence the open value type.
    """

    total: int
    records: list[dict[str, t.Any]]


class StructureRecord(TypedDict):
    """A row returned by ``search_structures``."""

    pk: int
    uuid: str
    formula: str
    num_sites: int
    ctime: str


class SubmitResult(TypedDict):
    """Return shape of ``submit_workflow``."""

    pk: int
    uuid: str
    entry_point: str
    state: str


class FailedStep(TypedDict):
    """One process on the path from a reported failure down to its cause.

    ``exit_message`` is what the run stored; ``exit_code_meaning`` is what the
    process class *declares* that status to mean. They usually agree, but a
    templated exit code stores the formatted message while the declaration
    carries the general one, and the general one is often the more diagnostic
    of the two.
    """

    pk: int
    process_label: str
    node_type: str
    state: str | None
    exit_status: int | None
    exit_message: str | None
    exit_code_meaning: str | None


class HandlerAttempt(TypedDict):
    """One registered handler's verdict on one iteration of a restart loop.

    ``applied`` false means the handler was consulted and declined --- it does
    not recognise this failure --- which is as informative as a handler that
    fired, because it rules that remedy out.
    """

    iteration: int
    handler: str
    applied: bool
    do_break: bool | None
    exit_status: int | None
    description: str | None


class RegisteredHandler(TypedDict):
    """A remedy a restart work chain knows how to apply.

    ``handles_exit_statuses`` is ``None`` when the handler declares no exit
    codes (it is consulted for every failure) *or* when the declaration could
    not be read; both mean the same thing to a caller --- it might apply.
    """

    name: str
    priority: int | None
    enabled: bool | None
    handles_exit_statuses: list[int] | None
    description: str | None


class FailureDiagnosis(TypedDict):
    """Return shape of ``diagnose_process_failure``.

    ``failed`` is stated rather than implied, so a process that finished fine
    cannot be read as a failure with nothing found. When it is false there is no
    chain, cause or remedy to report --- but ``handling_attempted`` may still be
    populated, because a run that recovered from its own trouble is not the same
    as one that never had any.
    """

    pk: int
    process_label: str
    state: str | None
    failed: bool
    exit_status: int | None
    exit_message: str | None
    failure_chain: list[FailedStep]
    root_cause: FailedStep | None
    handling_attempted: list[HandlerAttempt]
    known_remedies: list[RegisteredHandler]
    retrieved_files_pk: int | None


class ProcessCompletion(TypedDict):
    """Return shape of ``wait_for_process``.

    ``terminated`` is the field to branch on. False means the wait ran out
    before the process finished --- a fact about the wait, not about the
    process, which is very likely still running perfectly well.

    ``outputs`` is populated only once the process has terminated, since a
    running process's outputs are incomplete and reading one as final is how a
    chained submission ends up consuming a structure that is about to be
    superseded.
    """

    pk: int
    process_label: str
    state: str | None
    exit_status: int | None
    exit_message: str | None
    terminated: bool
    waited_seconds: float
    outputs: list[NodeLink]
