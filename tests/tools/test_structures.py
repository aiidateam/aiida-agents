"""Tests for ``aiida_agents.tools.structures``.

Unlike the other tools, ``search_structures`` carries real logic of its own:
formula parsing, element-symbol mapping and Python-side filtering. The tests
exercise the match / no-match / no-filter branches of that logic. See
``tests/conftest.py`` for the session-scoped ``silicon_structure`` fixture.
"""

from __future__ import annotations

import pytest
from aiida import orm

from aiida_agents.tools.structures import search_structures


@pytest.mark.parametrize(
    "formula,should_match",
    [
        pytest.param("Si", True, id="match"),
        pytest.param("Fe", False, id="no-match"),
        pytest.param(None, True, id="no-filter"),
    ],
)
def test_search_structures(
    silicon_structure: orm.StructureData,
    formula: str | None,
    should_match: bool,
) -> None:
    """Structures are filtered by element; a non-matching element excludes them."""
    results = search_structures(formula=formula, limit=10)
    pks = [record["pk"] for record in results]

    if not should_match:
        assert silicon_structure.pk not in pks
        return

    assert silicon_structure.pk in pks
    record = next(r for r in results if r["pk"] == silicon_structure.pk)
    assert record["formula"] == "Si2"
    assert record["num_sites"] == 2


@pytest.mark.parametrize(
    "formula",
    [
        pytest.param("MultiplyAdd", id="workflow-name"),
        pytest.param("Xx", id="nonexistent-symbol"),
    ],
)
def test_search_structures_rejects_non_elements(formula: str) -> None:
    """A query that parses to non-elements raises, so the caller can correct it
    instead of getting a silent empty result (needs no DB: the raise precedes it).
    """
    with pytest.raises(ValueError, match="valid chemical elements"):
        search_structures(formula=formula)
