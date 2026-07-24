"""Tests for ``aiida_agents.tools.nodes``.

These tools wrap the ORM thinly, so the tests target only what is *ours*: the
output-dict contract and the ``link_type`` stringification. See
``tests/conftest.py`` for the real, session-scoped node fixtures.
"""

from __future__ import annotations

from aiida import orm

from aiida_agents.tools.nodes import get_node_inputs, get_node_outputs


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
