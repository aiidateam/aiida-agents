"""Shared types for the MCP layer (tools, resources, ...).

The ``TypedDict``s below are the tools' return shapes. FastMCP turns them into
each tool's ``outputSchema``, so the agent sees named, typed fields rather than
an opaque object of ``str | int | null`` values.
"""

from __future__ import annotations

import typing as t

from pydantic import Field

__all__ = [
    "Identifier",
    "NodeLink",
    "NodeRecord",
    "ProcessRecord",
    "ProcessStatus",
    "StructureRecord",
]

# A node identifier: a pk or a uuid, both as a plain string. Using ``str``
# (rather than ``int | str``) means the MCP Inspector sends the value as-is
# without requiring JSON quotes; the loader coerces a purely numeric
# identifier back to an integer pk.
Identifier = t.Annotated[
    str, Field(description="Node pk or uuid (e.g. '42' or '0cef…')")
]


class ProcessStatus(t.TypedDict):
    """Return shape of ``get_process_status``."""

    pk: int
    process_label: str
    process_type: str
    state: str | None
    exit_status: int | None
    exit_message: str | None


class ProcessRecord(t.TypedDict):
    """A row returned by ``list_processes``."""

    pk: int
    uuid: str
    node_type: str
    process_type: str
    state: str | None
    exit_status: int | None


class NodeRecord(t.TypedDict):
    """A row returned by ``query_nodes``."""

    pk: int
    uuid: str
    node_type: str
    ctime: str


class NodeLink(t.TypedDict):
    """A link returned by ``get_node_inputs`` / ``get_node_outputs``."""

    pk: int
    uuid: str
    node_type: str
    link_label: str
    link_type: str


class StructureRecord(t.TypedDict):
    """A row returned by ``search_structures``."""

    pk: int
    uuid: str
    formula: str
    num_sites: int
    ctime: str
