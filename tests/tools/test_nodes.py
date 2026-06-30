"""Tests for ``aiida_agents.tools.nodes``.

These tools wrap the ORM thinly, so the tests target only what is *ours*: the
``node_type`` mapping (both the alias and substring-fallback branches), the
output-dict contract, and the ``link_type`` stringification. See
``tests/conftest.py`` for the real, session-scoped node fixtures.
"""

from __future__ import annotations

import pytest
from aiida import orm

from aiida_agents.tools.nodes import get_node_inputs, get_node_outputs, query_nodes


@pytest.mark.usefixtures("add_calc", "multiply_add_workchain")
def test_query_nodes_abstract_subtree() -> None:
    """An abstract level matches the whole node_type subtree, not a single leaf.

    This is the fix for the old ``ProcessNode`` mislabeling: ``"process"`` now
    spans calculations *and* workflows, not just calcjobs.
    """
    types = {
        record["node_type"] for record in query_nodes(node_type="process", limit=50)
    }
    # The filter has no false positives: every match is a process node...
    assert types
    assert all(nt.startswith("process.") for nt in types)
    # ...and the subtree spans calculations and workflows, not just calcjobs.
    assert "process.calculation.calcjob.CalcJobNode." in types  # from add_calc
    assert "process.workflow.workchain.WorkChainNode." in types  # multiply_add


@pytest.mark.usefixtures("add_calc")
@pytest.mark.parametrize(
    "node_type,expected",
    [
        pytest.param(
            "CalcJobNode",
            "process.calculation.calcjob.CalcJobNode.",
            id="process-class",
        ),
        pytest.param("Int", "data.core.int.Int.", id="data-class"),
    ],
)
def test_query_nodes_concrete_class(node_type: str, expected: str) -> None:
    """A concrete class name resolves to an exact node_type via the registry.

    Asserting every result has that exact type proves the resolved filter has no
    false positives.
    """
    results = query_nodes(node_type=node_type, limit=50)
    assert results
    assert all(record["node_type"] == expected for record in results)


@pytest.mark.usefixtures("add_calc")
def test_query_nodes_substring_fallback() -> None:
    """An unresolvable name falls back to a ``node_type`` substring match."""
    results = query_nodes(node_type="calcjob", limit=50)
    assert results
    assert all("calcjob" in record["node_type"].lower() for record in results)


def test_get_node_inputs(add_calc: orm.CalcJobNode) -> None:
    """Incoming links are returned with their labels and stringified link types."""
    links = {
        (r["link_label"], r["link_type"]) for r in get_node_inputs(str(add_calc.pk))
    }
    assert links == {
        ("x", "input_calc"),
        ("y", "input_calc"),
        ("code", "input_calc"),
    }


def test_get_node_outputs(add_calc: orm.CalcJobNode) -> None:
    """A calculation's outgoing links are its created data nodes."""
    links = {
        (r["link_label"], r["link_type"]) for r in get_node_outputs(str(add_calc.pk))
    }
    assert links == {
        ("sum", "create"),
        ("remote_folder", "create"),
        ("retrieved", "create"),
    }


def test_get_node_outputs_workchain(multiply_add_workchain: orm.WorkChainNode) -> None:
    """A work chain's outgoing links include its sub-process calls, not just data.

    This pins a surprising-but-real behavior: ``get_node_outputs`` surfaces the
    ``call_calc`` links to the sub-processes alongside the ``return`` outputs.
    If that ever changes (e.g. filtering to returns only), this fails loudly.
    """
    outputs = get_node_outputs(str(multiply_add_workchain.pk))
    calls = [r for r in outputs if r["link_type"] == "call_calc"]
    returns = [r for r in outputs if r["link_type"] == "return"]

    # Two sub-processes called: the multiply calcfunction and the add calcjob.
    assert len(calls) == 2
    assert [r["link_label"] for r in returns] == ["result"]
