"""Tests for ``aiida_agents.tools.execution.codes``.

The tool exists so the agent never has to guess a ``name@computer`` label for a
submission's ``code`` input, so what is pinned here is the ``full_label`` it
reports, the entry-point filter, and the hidden-code default. See
``tests/conftest.py`` for the session-scoped ``arithmetic_add_code`` fixture.
"""

from __future__ import annotations

import pytest
from aiida import orm

from aiida_agents.tools.execution.codes import list_codes


def test_lists_the_configured_code(arithmetic_add_code: orm.InstalledCode) -> None:
    """A configured code is reported with the full_label a submission needs."""
    records = list_codes()

    code = next((r for r in records if r["pk"] == arithmetic_add_code.pk), None)
    assert code is not None
    assert code["label"] == "bash"
    assert code["full_label"] == arithmetic_add_code.full_label
    assert code["computer"] == "localhost-agents-tests"
    assert code["default_calc_job_plugin"] == "core.arithmetic.add"
    assert code["node_type"] == "InstalledCode"


def test_lists_only_codes(
    arithmetic_add_code: orm.InstalledCode, silicon_structure: orm.StructureData
) -> None:
    """Every record is a real code, never some other Data node.

    Regression: querying the abstract ``AbstractCode`` base (which has no entry
    point of its own) silently matched unrelated Data nodes, so a StructureData
    in the profile came back as a "code" and blew up on attribute access.
    """
    pks = {record["pk"] for record in list_codes()}

    assert arithmetic_add_code.pk in pks
    assert silicon_structure.pk not in pks
    assert all(isinstance(orm.load_node(pk), orm.AbstractCode) for pk in pks)


def test_full_label_round_trips_as_a_code_reference(
    arithmetic_add_code: orm.InstalledCode,
) -> None:
    """The reported full_label resolves back to the same code.

    This is the tool's whole point: the label it hands the agent must be usable
    verbatim as ``{"label": ...}`` in a submission, so load it the way
    ``submit._resolve_node_reference`` does and check it comes back.
    """
    record = next(r for r in list_codes() if r["pk"] == arithmetic_add_code.pk)

    assert orm.load_code(record["full_label"]).pk == arithmetic_add_code.pk


def test_entry_point_filter_selects_matching_codes(
    arithmetic_add_code: orm.InstalledCode,
) -> None:
    """Filtering by entry point keeps codes for that plugin and drops the rest."""
    matching = list_codes(entry_point="core.arithmetic.add")
    assert any(r["pk"] == arithmetic_add_code.pk for r in matching)
    assert all(r["default_calc_job_plugin"] == "core.arithmetic.add" for r in matching)

    assert list_codes(entry_point="quantumespresso.pw") == []


@pytest.mark.usefixtures("aiida_profile")
def test_hidden_codes_are_excluded_unless_requested(
    arithmetic_add_code: orm.InstalledCode,
) -> None:
    """A hidden code is omitted by default and included on request."""
    hidden = orm.InstalledCode(
        label="hidden-bash",
        computer=arithmetic_add_code.computer,
        filepath_executable="/bin/bash",
        default_calc_job_plugin="core.arithmetic.add",
    ).store()
    hidden.is_hidden = True

    visible_pks = {r["pk"] for r in list_codes()}
    assert hidden.pk not in visible_pks

    all_pks = {r["pk"] for r in list_codes(include_hidden=True)}
    assert hidden.pk in all_pks
