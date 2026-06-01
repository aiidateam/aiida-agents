"""MCP tools for AiiDA structure data queries."""

from __future__ import annotations

import logging
import re

from aiida import orm
from aiida.common.constants import elements
from aiida.common.exceptions import NotExistent
from fastmcp import FastMCP

logger = logging.getLogger(__name__)

# Element name -> symbol, derived from aiida-core's periodic table.
# AiiDA uses IUPAC spellings, so common alternates are added manually.
ELEMENT_NAME_TO_SYMBOL: dict[str, str] = {
    str(data["name"]).lower(): str(data["symbol"]) for data in elements.values()
}
ELEMENT_NAME_TO_SYMBOL.update({"aluminum": "Al", "sulphur": "S", "cesium": "Cs"})


def search_structures(
    formula: str | None = None,
    limit: int = 10,
) -> list[dict[str, str | int | None]]:
    """Search for crystal structures in the AiiDA database.

    Supports chemical formulas (e.g. 'SiO2'), element symbols (e.g. 'Si'),
    multi-element queries (e.g. 'Si, O', 'Silicon and Oxygen'), or full element names.
    """
    logger.debug("search_structures(formula=%r, limit=%d)", formula, limit)

    target_elements: list[str] = []

    if formula and formula.strip().lower() != "none":
        # Normalise separators, then tokenise into chemical words.
        normalised = formula.replace(" and ", " ").replace(",", " ").replace("-", " ")
        tokens = re.findall(r"[A-Za-z]+", normalised)

        for token in tokens:
            if token.lower() in ELEMENT_NAME_TO_SYMBOL:
                target_elements.append(ELEMENT_NAME_TO_SYMBOL[token.lower()])
            else:
                # Split a formula token like 'SiO2' into ['Si', 'O'].
                sub_tokens = re.findall(r"[A-Z][a-z]?", token)
                target_elements.extend(sub_tokens if sub_tokens else [token])

        logger.debug("search_structures: parsed elements %s", target_elements)

    qb = orm.QueryBuilder()
    qb.append(orm.StructureData, project=["id", "uuid", "ctime", "attributes.kinds"])
    qb.order_by({orm.StructureData: {"ctime": "desc"}})

    # Fetch kinds in one query and filter in Python to avoid slow SQL joins.
    results = []
    for pk, uuid, ctime, kinds in qb.all():
        if not kinds:
            continue

        symbols = {s for kind in kinds for s in kind.get("symbols", [])}

        if target_elements and not all(e in symbols for e in target_elements):
            continue

        try:
            node = orm.load_node(pk=pk)
            results.append(
                {
                    "pk": pk,
                    "uuid": uuid,
                    "formula": node.get_formula(),
                    "num_sites": len(node.sites),
                    "ctime": str(ctime),
                }
            )
        except NotExistent:
            continue

        if len(results) >= limit:
            break

    logger.debug("search_structures: found %d matching structures", len(results))
    return results


def register(mcp: FastMCP) -> None:
    """Register structure tools on the MCP server."""
    mcp.tool()(search_structures)
