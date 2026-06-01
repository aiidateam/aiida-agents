"""Tests for ``aiida_agents.mcp.tools.processes``.

These tools wrap the ORM / ``aiida-restapi`` thinly, so the tests target only
what is *ours*: the output-dict contract, pk/uuid acceptance, the ``ToolError``
on a missing node, and the ``%process%`` filter plus state-from-attributes
assembly. See ``tests/conftest.py`` for the real, session-scoped process fixtures.
"""

from __future__ import annotations

import pytest
from aiida import orm
from fastmcp.exceptions import ToolError

from aiida_agents.mcp.tools.processes import get_process_status, list_processes


@pytest.mark.parametrize("by", ["pk", "uuid"])
def test_get_process_status(add_calc: orm.CalcJobNode, by: str) -> None:
    """A finished process's full status is returned, by pk or uuid alike.

    The pk/uuid axis is the regression guard for the identifier handling; the
    rest pins the output-dict contract the tool exposes to the agent.
    """
    identifier = add_calc.pk if by == "pk" else add_calc.uuid

    assert get_process_status(identifier) == {
        "pk": add_calc.pk,
        "process_label": "ArithmeticAddCalculation",
        "process_type": "aiida.calculations:core.arithmetic.add",
        "state": "finished",
        "exit_status": 0,
        "exit_message": None,
    }


@pytest.mark.usefixtures("aiida_profile")
def test_get_process_status_not_found() -> None:
    """An unknown identifier raises a ``ToolError`` naming the identifier."""
    with pytest.raises(ToolError, match="987654321"):
        get_process_status(987654321)


def test_list_processes(add_calc: orm.CalcJobNode) -> None:
    """The process filter selects process nodes and pulls state from attributes."""
    records = list_processes(limit=50)

    # The ``%process%`` node-type filter excludes plain data nodes.
    assert all("process" in record["node_type"] for record in records)

    # State and exit status come from the per-node attributes lookup (the N+1).
    calc = next((r for r in records if r["pk"] == add_calc.pk), None)
    assert calc is not None
    assert calc["state"] == "finished"
    assert calc["exit_status"] == 0
