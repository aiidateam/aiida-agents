"""Tests for ``describe_aiida_error`` in ``aiida_agents.tools._errors``.

The shared recovery-message builder, unit-tested next to its definition.
"""

from __future__ import annotations

import pytest
from aiida.common.exceptions import AiidaException, MultipleObjectsError, NotExistent

from aiida_agents.tools._errors import describe_aiida_error


@pytest.mark.parametrize(
    ("exc", "present", "absent"),
    [
        pytest.param(
            NotExistent("no node 42"),
            ["no node 42", "list_processes()", "query_nodes()"],
            [],
            id="not-found-points-at-listing-tools",
        ),
        pytest.param(
            MultipleObjectsError("ambiguous uuid 'ab'"),
            ["ambiguous uuid 'ab'", "full uuid"],
            ["list_processes()"],
            id="ambiguous-asks-for-full-identifier-not-listing",
        ),
        pytest.param(
            AiidaException("some other failure"),
            ["some other failure"],
            ["list_processes()", "full uuid"],
            id="other-reported-as-is-without-a-misfit-hint",
        ),
    ],
)
def test_describe_aiida_error(
    exc: AiidaException, present: list[str], absent: list[str]
) -> None:
    """The message keeps the original error and adds guidance fitted to its type."""
    message = describe_aiida_error(exc)
    assert all(substring in message for substring in present)
    assert not any(substring in message for substring in absent)
