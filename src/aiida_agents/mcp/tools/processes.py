"""MCP tools for AiiDA process nodes."""

from __future__ import annotations
from aiida import orm
from fastmcp import FastMCP


def get_process_status(pk: int) -> dict[str, str | int | None]:
    """Get the status and exit code of an AiiDA process by its primary key."""
    print(f"\n🔍 [Agent invoking tool] get_process_status(pk={pk})...")
    try:
        node = orm.load_node(pk=pk)
        res = {
            "pk": node.pk,
            "process_label": node.process_label,
            "process_type": node.process_type,
            "state": str(node.process_state),
            "exit_status": node.exit_status,
            "exit_message": node.exit_message,
        }
        print(f"✅ Tool output: {res}")
        return res
    except Exception as e:
        print(f"❌ Tool error: {e}")
        return {"error": str(e)}


def list_processes(limit: int = 10) -> list[dict[str, str | int | None]]:
    """List recent AiiDA processes, newest first."""
    print(f"\n🔍 [Agent invoking tool] list_processes(limit={limit})...")
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
    res = [
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
    print(f"✅ Tool output: Returned {len(res)} process records.")
    return res


def register(mcp: FastMCP) -> None:
    """Register process tools on the MCP server."""
    mcp.tool()(get_process_status)
    mcp.tool()(list_processes)
