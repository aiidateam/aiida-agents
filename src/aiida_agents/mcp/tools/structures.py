"""MCP tools for AiiDA structure data queries."""

from __future__ import annotations
from aiida import orm
from fastmcp import FastMCP


def search_structures(
    formula: str | None = None,
    limit: int = 10,
) -> list[dict[str, str | int | None]]:
    """Search for crystal structures in the AiiDA database."""
    print(
        f"\n🔍 [Agent invoking tool] search_structures(formula='{formula}', limit={limit})..."
    )
    qb = orm.QueryBuilder()
    qb.append(orm.StructureData, project=["id", "uuid", "ctime"])
    qb.order_by({orm.StructureData: {"ctime": "desc"}})
    qb.limit(limit * 5 if formula else limit)
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
    print(f"✅ Tool output: Found {len(results)} matching structures.")
    return results


def register(mcp: FastMCP) -> None:
    """Register structure tools on the MCP server."""
    mcp.tool()(search_structures)
