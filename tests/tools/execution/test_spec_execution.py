"""Tests for ``submit_process_spec`` and the ``_validate_spec`` guard it shares.

``submit_workflow`` is submit-only and the session profile is brokerless, so
the happy path is proved by how far it gets: a spec whose ``entry_point`` and
``inputs`` were read correctly reaches the engine and stops at the missing
broker, where a spec read wrongly would have been turned back by
``_validate_spec`` several steps earlier. Same reasoning as
``test_submit.py::TestSubmitWorkflow::test_submit_requires_a_broker``.

The rejection messages are asserted by the key they name, because that key is
the tool's contract with the model: a message pointing at a field the schema
does not declare sends it looking for a mistake it did not make.
"""

from __future__ import annotations

import typing as t

import pytest
from aiida import orm

from aiida_agents.tools.execution.schemas import SubmissionSpec
from aiida_agents.tools.execution.spec_execution import (
    _validate_spec,
    submit_process_spec,
)
from aiida_agents.tools.execution.submit import SubmissionError

MULTIPLY_ADD = "core.arithmetic.multiply_add"
ADD = "core.arithmetic.add"


class TestValidateSpec:
    """The shape check, which a batch runs over every spec before submitting any."""

    def test_a_workflow_spec_comes_back_as_its_two_halves(self) -> None:
        spec: SubmissionSpec = {"entry_point": MULTIPLY_ADD, "inputs": {"x": 1}}
        assert _validate_spec(spec) == (MULTIPLY_ADD, {"x": 1})

    def test_a_calculation_entry_point_is_accepted_too(self) -> None:
        """The reason the tool is not called ``submit_workflow_spec``: it takes
        anything in ``aiida.calculations`` as readily as anything in
        ``aiida.workflows``.
        """
        spec: SubmissionSpec = {"entry_point": ADD, "inputs": {"x": 1}}
        assert _validate_spec(spec) == (ADD, {"x": 1})

    def test_a_non_dict_spec_is_a_type_error(self) -> None:
        with pytest.raises(TypeError, match="spec must be a dictionary"):
            _validate_spec(t.cast(SubmissionSpec, [MULTIPLY_ADD]))

    @pytest.mark.parametrize(
        ("spec", "missing"),
        [
            ({"inputs": {"x": 1}}, "entry_point"),
            ({"entry_point": "", "inputs": {"x": 1}}, "entry_point"),
            ({"entry_point": 42, "inputs": {"x": 1}}, "entry_point"),
            ({"entry_point": MULTIPLY_ADD}, "inputs"),
            ({"entry_point": MULTIPLY_ADD, "inputs": {}}, "inputs"),
            ({"entry_point": MULTIPLY_ADD, "inputs": "x=1"}, "inputs"),
        ],
    )
    def test_a_missing_half_is_refused_naming_the_key_the_schema_declares(
        self, spec: t.Any, missing: str
    ) -> None:
        with pytest.raises(ValueError, match=f"required '{missing}'"):
            _validate_spec(spec)

    def test_an_unregistered_entry_point_is_refused(self) -> None:
        spec: SubmissionSpec = {
            "entry_point": "nonexistent.fake.process",
            "inputs": {"x": 1},
        }
        with pytest.raises(ValueError, match="is not a known AiiDA entry point"):
            _validate_spec(spec)


class TestSubmitProcessSpec:
    def test_a_valid_spec_reaches_the_engine(
        self, arithmetic_add_code: orm.InstalledCode
    ) -> None:
        """Getting as far as the broker is the assertion: the spec's own
        ``entry_point`` and ``inputs`` were unpacked and handed to
        ``submit_workflow``, rather than rejected on the way.
        """
        spec: SubmissionSpec = {
            "entry_point": MULTIPLY_ADD,
            "inputs": {
                "x": 2,
                "y": 3,
                "z": 4,
                "code": {"pk": arithmetic_add_code.pk},
            },
        }
        with pytest.raises(SubmissionError, match=r"no broker"):
            submit_process_spec(spec)

    def test_an_invalid_spec_is_refused_before_the_engine(self) -> None:
        """The guard runs first, so a malformed spec never reaches submission."""
        with pytest.raises(ValueError, match="is not a known AiiDA entry point"):
            submit_process_spec({"entry_point": "nope.not.real", "inputs": {"x": 1}})
