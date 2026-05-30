"""MCP tools for AiiDA process nodes."""

from __future__ import annotations
import typing as t
from aiida import orm
from fastmcp import FastMCP
from aiida_restapi.services.node import NodeService
from aiida_restapi.common.query import QueryBuilderParams
from .._types import Identifier


def get_process_status(identifier: Identifier) -> dict[str, str | int | None]:
    """Get the status and exit code of an AiiDA process by its pk or uuid."""
    print(f"\n🔍 [Agent invoking tool] get_process_status(identifier={identifier})...")
    try:
        # Single-node lookup: plain ORM is simpler than the service layer here.
        node = orm.load_node(identifier)
        res = {
            "pk": node.pk,
            "process_label": node.process_label,
            "process_type": node.process_type,
            "state": node.process_state.value if node.process_state else None,
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
    try:
        node_service: NodeService[orm.Node, t.Any] = NodeService(orm.Node)
        params = QueryBuilderParams(
            page_size=limit,
            filters={"node_type": {"like": "%process%"}},
            order_by={"ctime": "desc"},
        )
        res = node_service.get_many(params)
        records = []
        for item in res.data:
            uuid = item.get("uuid")
            if uuid is None:
                continue
            # Pull process details from attributes if possible (get_field is by uuid)
            attrs: dict[str, t.Any] = {}
            try:
                attrs = node_service.get_field(uuid, "attributes") or {}
            except Exception:
                pass
            records.append(
                {
                    "pk": item.get("pk"),
                    "uuid": uuid,
                    "node_type": item.get("node_type"),
                    "process_type": item.get("process_type"),
                    "state": attrs.get("process_state"),
                    "exit_status": attrs.get("exit_status"),
                }
            )
        print(f"✅ Tool output: Returned {len(records)} process records.")
        return records
    except Exception as e:
        print(f"❌ Tool error: {e}")
        return [{"error": str(e)}]


def register(mcp: FastMCP) -> None:
    """Register process tools on the MCP server."""
    mcp.tool()(get_process_status)
    mcp.tool()(list_processes)
