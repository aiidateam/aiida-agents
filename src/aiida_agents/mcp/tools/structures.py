"""MCP tools for AiiDA structure data queries."""

from __future__ import annotations
import re
from aiida import orm
from aiida.common.constants import elements
from fastmcp import FastMCP


# Element name -> symbol, derived from aiida-core's periodic table (all 115
# elements) rather than hand-maintained. aiida uses the IUPAC spellings
# (aluminium, sulfur, caesium), so add the common alternates it doesn't carry.
ELEMENT_NAME_TO_SYMBOL: dict[str, str] = {
    str(data["name"]).lower(): str(data["symbol"]) for data in elements.values()
}
ELEMENT_NAME_TO_SYMBOL.update({"aluminum": "Al", "sulphur": "S", "cesium": "Cs"})


def search_structures(
    formula: str | None = None,
    limit: int = 10,
) -> list[dict[str, str | int | None]]:
    """Search for crystal structures in the AiiDA database.

    Supports chemical formulas (e.g. 'SiO2'), element abbreviations (e.g. 'Si'),
    multi-element queries (e.g. 'Si, O', 'Silicon and Oxygen'), or full element names.
    """
    print(
        f"\n🔍 [Agent invoking tool] search_structures(formula='{formula}', limit={limit})..."
    )

    target_elements: list[str] = []

    if formula and formula.strip().lower() != "none":
        # Normalize formula string and split it
        # Extract word/chemical tokens
        normalized = formula.replace(" and ", " ").replace(",", " ").replace("-", " ")
        tokens = re.findall(r"[A-Za-z]+", normalized)

        for token in tokens:
            token_lower = token.lower()
            if token_lower in ELEMENT_NAME_TO_SYMBOL:
                target_elements.append(ELEMENT_NAME_TO_SYMBOL[token_lower])
            else:
                # E.g. 'SiO2' -> split into 'Si', 'O' using regex for chemical elements
                sub_tokens = re.findall(r"[A-Z][a-z]?", token)
                if sub_tokens:
                    target_elements.extend(sub_tokens)
                else:
                    target_elements.append(token)

        print(f"Parsed search elements: {target_elements}")

    qb = orm.QueryBuilder()
    qb.append(orm.StructureData, project=["id", "uuid", "ctime", "attributes.kinds"])
    qb.order_by({orm.StructureData: {"ctime": "desc"}})

    # We retrieve all structure records kinds attributes in one fast query
    # and perform filtering directly in python to avoid slow SQL joints.
    results = []
    for row in qb.all():
        pk, uuid, ctime, kinds = row
        if not kinds:
            continue

        # Get all element symbols present in this structure node
        symbols = set()
        for kind in kinds:
            if "symbols" in kind:
                symbols.update(kind["symbols"])

        # Check if all target elements are matched in this structure
        if target_elements:
            match = True
            for element in target_elements:
                if element not in symbols:
                    match = False
                    break
            if not match:
                continue

        # If it matched, load the node to obtain the clean stoichiometric formula
        try:
            node = orm.load_node(pk=pk)
            node_formula = node.get_formula()
            results.append(
                {
                    "pk": pk,
                    "uuid": uuid,
                    "formula": node_formula,
                    "num_sites": len(node.sites),
                    "created": str(ctime),
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
