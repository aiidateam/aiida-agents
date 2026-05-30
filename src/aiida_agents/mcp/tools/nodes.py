"""MCP tools for AiiDA generic node queries."""

from __future__ import annotations
import typing as t
from aiida import orm
from fastmcp import FastMCP
from aiida_restapi.services.node import NodeService
from aiida_restapi.common.query import QueryBuilderParams
from .._types import Identifier


def query_nodes(
    node_type: str = "ProcessNode",
    limit: int = 10,
) -> list[dict[str, t.Any]]:
    """Query AiiDA nodes by type."""
    print(
        f"\n🔍 [Agent invoking tool] query_nodes(node_type='{node_type}', limit={limit})..."
    )

    # Standard common maps for direct, fast, indexed queries
    type_map: dict[str, str] = {
        "processnode": "process.calculation.calcjob.CalcJobNode.",
        "structuredata": "data.core.structure.StructureData.",
        "structure": "data.core.structure.StructureData.",
        "dict": "data.core.dict.Dict.",
        "calcjobnode": "process.calculation.calcjob.CalcJobNode.",
        "calcjob": "process.calculation.calcjob.CalcJobNode.",
        "pwcalculation": "process.calculation.calcjob.CalcJobNode.",
        "calculation": "process.calculation.calcjob.CalcJobNode.",
        "workchainnode": "process.workflow.workchain.WorkChainNode.",
        "workchain": "process.workflow.workchain.WorkChainNode.",
        "pwbaseworkchain": "process.workflow.workchain.WorkChainNode.",
    }

    normalized_type = node_type.lower()

    if normalized_type in type_map:
        filter_type = type_map[normalized_type]
        filters = {"node_type": {"like": f"{filter_type}%"}}
    else:
        filters = {"node_type": {"like": f"%{node_type}%"}}

    try:
        node_service: NodeService[orm.Node, t.Any] = NodeService(orm.Node)
        params = QueryBuilderParams(
            page_size=limit, filters=filters, order_by={"ctime": "desc"}
        )
        res = node_service.get_many(params)
        records = [
            {
                "pk": item.get("pk"),
                "uuid": item.get("uuid"),
                "node_type": item.get("node_type"),
                "created": str(item.get("ctime")),
            }
            for item in res.data
        ]
        print(f"✅ Tool output: Returned {len(records)} nodes.")
        return records
    except Exception as e:
        print(f"❌ Tool error: {e}")
        return [{"error": str(e)}]


def get_node_inputs(identifier: Identifier) -> list[dict[str, t.Any]]:
    """Get all input nodes of an AiiDA node by its pk or uuid."""
    print(f"\n🔍 [Agent invoking tool] get_node_inputs(identifier={identifier})...")
    try:
        # Single-node traversal: plain ORM gives the linked nodes directly.
        node = orm.load_node(identifier)
        results = [
            {
                "pk": entry.node.pk,
                "uuid": entry.node.uuid,
                "node_type": entry.node.node_type,
                "link_label": entry.link_label,
                "link_type": entry.link_type.value,
            }
            for entry in node.base.links.get_incoming().all()
        ]
        print(f"✅ Tool output: Found {len(results)} incoming links.")
        return results
    except Exception as e:
        print(f"❌ Tool error: {e}")
        return [{"error": str(e)}]


def get_node_outputs(identifier: Identifier) -> list[dict[str, t.Any]]:
    """Get all output nodes of an AiiDA node by its pk or uuid."""
    print(f"\n🔍 [Agent invoking tool] get_node_outputs(identifier={identifier})...")
    try:
        # Single-node traversal: plain ORM gives the linked nodes directly.
        node = orm.load_node(identifier)
        results = [
            {
                "pk": entry.node.pk,
                "uuid": entry.node.uuid,
                "node_type": entry.node.node_type,
                "link_label": entry.link_label,
                "link_type": entry.link_type.value,
            }
            for entry in node.base.links.get_outgoing().all()
        ]
        print(f"✅ Tool output: Found {len(results)} outgoing links.")
        return results
    except Exception as e:
        print(f"❌ Tool error: {e}")
        return [{"error": str(e)}]


def register(mcp: FastMCP) -> None:
    """Register node tools on the MCP server."""
    mcp.tool()(query_nodes)
    mcp.tool()(get_node_inputs)
    mcp.tool()(get_node_outputs)
