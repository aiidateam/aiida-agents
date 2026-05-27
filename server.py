"""AiiDA Agents MCP Server - read-only exploration tools."""

from __future__ import annotations


import uvicorn.config  # noqa: E402
from typing import cast, MutableMapping, Any  # noqa: E402

cast(MutableMapping[str, Any], uvicorn.config.WS_PROTOCOLS).setdefault(
    "websockets-sansio",
    "uvicorn.protocols.websockets.wsproto_impl:WSProtocol",
)

from fastmcp import FastMCP  # noqa: E402
from aiida import load_profile, orm  # noqa: E402

mcp = FastMCP(
    "aiida-agents",
    instructions="Tools for exploring an AiiDA database using natural language.",
)


@mcp.tool
def get_process_status(pk: int) -> dict[str, str | int | None]:
    """Get the status and exit code of an AiiDA process by its primary key."""
    try:
        node = orm.load_node(pk=pk)
        return {
            "pk": node.pk,
            "process_label": node.process_label,
            "process_type": node.process_type,
            "state": str(node.process_state),
            "exit_status": node.exit_status,
            "exit_message": node.exit_message,
        }
    except Exception as e:
        return {"error": str(e)}


@mcp.tool
def list_processes(limit: int = 10) -> list[dict[str, str | int | None]]:
    """List recent AiiDA processes, newest first."""
    qb = orm.QueryBuilder()
    qb.append(
        orm.ProcessNode,
        project=[
            "id",
            "uuid",
            "node_type",
            "process_type",
            "attributes.process_state",
            "attributes.exit_status",
        ],
    )
    qb.order_by({orm.ProcessNode: {"ctime": "desc"}})
    qb.limit(limit)
    return [
        {
            "pk": row[0],
            "uuid": row[1],
            "node_type": row[2],
            "process_type": row[3],
            "state": row[4],
            "exit_status": row[5],
        }
        for row in qb.all()
    ]


@mcp.tool
def query_nodes(
    node_type: str = "ProcessNode",
    limit: int = 10,
) -> list[dict[str, str | int]]:
    """Query AiiDA nodes by type.

    Args:
        node_type: Type of node — ProcessNode, StructureData, Dict, CalcJobNode, WorkChainNode.
        limit: Maximum number of results to return.
    """
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
    return [
        {"pk": r[0], "uuid": r[1], "node_type": r[2], "created": str(r[3])}
        for r in qb.all()
    ]


@mcp.tool
def get_node_inputs(pk: int) -> list[dict[str, str | int | None]]:
    """Get all input nodes of an AiiDA node by its primary key."""
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
        return results
    except Exception as e:
        return [{"error": str(e)}]


@mcp.tool
def get_node_outputs(pk: int) -> list[dict[str, str | int | None]]:
    """Get all output nodes of an AiiDA node by its primary key."""
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
        return results
    except Exception as e:
        return [{"error": str(e)}]


@mcp.tool
def search_structures(
    formula: str | None = None,
    limit: int = 10,
) -> list[dict[str, str | int | None]]:
    """Search for crystal structures in the AiiDA database.

    Args:
        formula: Element symbol to search for (e.g. 'Fe', 'Li', 'Si'). Optional.
        limit: Maximum number of results to return.
    """
    qb = orm.QueryBuilder()
    qb.append(
        orm.StructureData,
        project=["id", "uuid", "ctime"],
    )
    qb.order_by({orm.StructureData: {"ctime": "desc"}})
    qb.limit(limit * 5 if formula else limit)  # fetch more if filtering

    results = []
    for r in qb.all():
        try:
            node = orm.load_node(pk=r[0])
            node_formula = node.get_formula()
            if formula and formula not in node_formula:
                continue
            results.append(
                {
                    "pk": r[0],
                    "uuid": r[1],
                    "formula": node_formula,
                    "num_sites": len(node.sites),
                    "created": str(r[2]),
                }
            )
            if len(results) >= limit:
                break
        except Exception:
            continue

    return results


if __name__ == "__main__":
    load_profile()

    mcp.run(transport="streamable-http", port=8000)
