"""MCP tools for generic AiiDA node queries."""

from __future__ import annotations

import logging
import typing as t
from functools import lru_cache

from aiida import orm
from aiida.plugins.entry_point import get_entry_point_names, load_entry_point
from fastmcp import FastMCP
from aiida_restapi.services.node import NodeService
from aiida_restapi.common.query import QueryBuilderParams

from .._orm import load_node
from .._types import Identifier, NodeLink, NodeRecord
from .._errors import register_tool

logger = logging.getLogger(__name__)

# Abstract hierarchy levels are node_type prefixes, not entry points, so they
# can't be derived from the registry. "like <prefix>%" selects the whole subtree.
_SUBTREE_PREFIXES: dict[str, str] = {
    "node": "%",
    "data": "data.%",
    "process": "process.%",
    "processnode": "process.%",
    "calculation": "process.calculation.%",
    "calculationnode": "process.calculation.%",
    "workflow": "process.workflow.%",
    "workflownode": "process.workflow.%",
}

# Built once at module level; all three tools share the same service instance.
_node_service: NodeService[orm.Node, t.Any] = NodeService(orm.Node)


@lru_cache(maxsize=1)
def _node_type_index() -> dict[str, str]:
    """Lowercased class name and entry-point name -> node_type, from the entry points."""
    index: dict[str, str] = {}
    for group in ("aiida.data", "aiida.node"):
        for name in get_entry_point_names(group):
            try:
                cls = load_entry_point(group, name)
            except Exception:
                continue
            if node_type := getattr(cls, "class_node_type", None):
                index[name.lower()] = node_type
                index[cls.__name__.lower()] = node_type
    return index


def _node_type_for(name: str) -> str | None:
    """Resolve a class or entry-point name to a node_type string."""
    if name.endswith("."):  # already a fully-qualified node_type
        return name
    return _node_type_index().get(name.lower())


def query_nodes(
    node_type: str = "process",
    limit: int = 10,
) -> list[NodeRecord]:
    """Query AiiDA nodes by type.

    ``node_type`` accepts an abstract hierarchy level (``node``, ``data``,
    ``process``, ``calculation``, ``workflow``, or their ``...Node`` class
    names) which matches the whole subtree, a concrete class or entry-point name
    (``StructureData``, ``Int``, ``CalcJobNode``, ...) resolved to an exact
    ``node_type`` via AiiDA's entry points, or, as a last resort, an arbitrary
    substring of the ``node_type``.
    """
    logger.debug("query_nodes(node_type=%r, limit=%d)", node_type, limit)

    normalized = node_type.lower()
    filters: dict[str, t.Any]
    if normalized in _SUBTREE_PREFIXES:
        filters = {"node_type": {"like": _SUBTREE_PREFIXES[normalized]}}
    elif (node_type_string := _node_type_for(node_type)) is not None:
        filters = {"node_type": node_type_string}
    else:
        filters = {"node_type": {"like": f"%{node_type}%"}}

    params = QueryBuilderParams(
        page_size=limit, filters=filters, order_by={"ctime": "desc"}
    )
    res = _node_service.get_many(params)
    records: list[NodeRecord] = [
        {
            "pk": item["pk"],
            "uuid": item["uuid"],
            "node_type": item["node_type"],
            "ctime": str(item.get("ctime")),
        }
        for item in res.data
    ]
    logger.debug("query_nodes: returned %d nodes", len(records))
    return records


def _node_links(
    identifier: Identifier, direction: t.Literal["incoming", "outgoing"]
) -> list[NodeLink]:
    """Return a node's incoming or outgoing links as serialisable dicts."""
    node = load_node(identifier)
    links = (
        node.base.links.get_incoming()
        if direction == "incoming"
        else node.base.links.get_outgoing()
    )
    return [
        {
            "pk": t.cast(int, entry.node.pk),  # a linked node is always stored
            "uuid": entry.node.uuid,
            "node_type": entry.node.node_type,
            "link_label": entry.link_label,
            "link_type": entry.link_type.value,
        }
        for entry in links.all()
    ]


def get_node_inputs(identifier: Identifier) -> list[NodeLink]:
    """Get the incoming links of any AiiDA node by its pk or uuid.

    Works for data and processes alike: a data node's incoming link is the
    process that created it; a process's incoming links are its input data.
    """
    logger.debug("get_node_inputs(identifier=%r)", identifier)
    results = _node_links(identifier, "incoming")
    logger.debug("get_node_inputs: found %d incoming links", len(results))
    return results


def get_node_outputs(identifier: Identifier) -> list[NodeLink]:
    """Get the outgoing links of any AiiDA node by its pk or uuid.

    Works for data and processes alike: a data node's outgoing links are the
    processes that consumed it; a process's outgoing links are the data it
    produced (and any sub-processes it called).
    """
    logger.debug("get_node_outputs(identifier=%r)", identifier)
    results = _node_links(identifier, "outgoing")
    logger.debug("get_node_outputs: found %d outgoing links", len(results))
    return results


def register(mcp: FastMCP) -> None:
    """Register node tools on the MCP server."""
    register_tool(mcp, query_nodes)
    register_tool(mcp, get_node_inputs)
    register_tool(mcp, get_node_outputs)
