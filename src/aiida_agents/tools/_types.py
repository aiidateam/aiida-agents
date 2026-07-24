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
    "Identifier",
    "NodeLink",
    "ProcessRecord",
    "ProcessReport",
    "ProcessStatus",
    "QueryResult",
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
    """A row returned by ``list_processes``."""

    pk: int
    uuid: str
    node_type: str
    process_type: str
    state: str | None
    exit_status: int | None


class ProcessReport(TypedDict):
    """Return shape of ``get_process_report``."""

    pk: int
    process_label: str
    node_type: str
    state: str | None
    exit_status: int | None
    exit_message: str | None
    report: str


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
    workflow: str
    state: str
