"""MCP tools for AiiDA generic node queries."""

from __future__ import annotations
from aiida import orm
from fastmcp import FastMCP


def query_nodes(
    node_type: str = "ProcessNode",
    limit: int = 10,
) -> list[dict[str, str | int]]:
    """Query AiiDA nodes by type."""
    print(
        f"\n🔍 [Agent invoking tool] query_nodes(node_type='{node_type}', limit={limit})..."
    )
    type_map: dict[str, type] = {
        "ProcessNode": orm.ProcessNode,
        "StructureData": orm.StructureData,
        "Dict": orm.Dict,
        "CalcJobNode": orm.CalcJobNode,
        "WorkChainNode": orm.WorkChainNode,
    }
    node_class = type_map.get(node_type, orm.Node)
    qb = orm.QueryBuilder()
    qb.append(node_class, project=["id", "uuid", "node_type", "ctime"])
    qb.order_by({node_class: {"ctime": "desc"}})
    qb.limit(limit)
    res = [
        {"pk": r[0], "uuid": r[1], "node_type": r[2], "created": str(r[3])}
        for r in qb.all()
    ]
    print(f"✅ Tool output: Returned {len(res)} nodes.")
    return res


def get_node_inputs(pk: int) -> list[dict[str, str | int | None]]:
    """Get all input nodes of an AiiDA node by its primary key."""
    print(f"\n🔍 [Agent invoking tool] get_node_inputs(pk={pk})...")
    try:
        node = orm.load_node(pk=pk)
        results = []
        for entry in node.base.links.get_incoming().all():
            results.append(
                {
                    "pk": entry.node.pk,
                    "uuid": str(entry.node.uuid),
                    "node_type": entry.node.node_type,
                    "link_label": entry.link_label,
                    "link_type": str(entry.link_type),
                }
            )
        print(f"✅ Tool output: Found {len(results)} incoming links.")
        return results
    except Exception as e:
        print(f"❌ Tool error: {e}")
        return [{"error": str(e)}]


def get_node_outputs(pk: int) -> list[dict[str, str | int | None]]:
    """Get all output nodes of an AiiDA node by its primary key."""
    print(f"\n🔍 [Agent invoking tool] get_node_outputs(pk={pk})...")
    try:
        node = orm.load_node(pk=pk)
        results = []
        for entry in node.base.links.get_outgoing().all():
            results.append(
                {
                    "pk": entry.node.pk,
                    "uuid": str(entry.node.uuid),
                    "node_type": entry.node.node_type,
                    "link_label": entry.link_label,
                    "link_type": str(entry.link_type),
                }
            )
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
