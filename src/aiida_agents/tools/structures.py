"""Surface-agnostic tools for AiiDA structure data queries."""

from __future__ import annotations

import logging
import re
import typing as t

from aiida import orm
from aiida.common.constants import elements

from ._types import StructureRecord

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
) -> list[StructureRecord]:
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

    # Element-set matching pushed to the database: each target element must
    # appear in some kind's ``symbols``, AND-combined across elements. The
    # QueryBuilder ``contains`` operator works on both sqlite and psql backends.
    filters: dict[str, t.Any] = {}
    if target_elements:
        filters = {
            "and": [
                {"attributes.kinds": {"contains": [{"symbols": [element]}]}}
                for element in target_elements
            ]
        }

    qb = orm.QueryBuilder()
    qb.append(orm.StructureData, filters=filters, project=["id", "uuid", "ctime"])
    qb.order_by({orm.StructureData: {"ctime": "desc"}})
    qb.limit(limit)

    # The node is loaded only for the matched, limited rows (for formula and
    # site count, which aren't plain projectable attributes).
    results: list[StructureRecord] = []
    for pk, uuid, ctime in qb.iterall():
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

    logger.debug("search_structures: found %d matching structures", len(results))
    return results
