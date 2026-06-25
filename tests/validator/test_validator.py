"""Tests for the Validator layer.

Three layers tested here:
- _schema.py: type and presence checks against AiiDA port specs
- validator/__init__.py: ValidationError raised when schema checks fail
- submit_workflow: ToolError raised when validation fails, proving the
  write path cannot be reached without passing validation
"""

from __future__ import annotations

import pytest
from aiida import orm

from aiida_agents.agents.validator import ValidationError, validate
from aiida_agents.agents.validator._schema import validate_schema


class TestValidateSchema:
    def test_valid_inputs_return_no_errors(self, add_calc: orm.CalcJobNode) -> None:
        """Well-formed inputs produce an empty error list."""
        errors = validate_schema(
            "core.arithmetic.add",
            {
                "x": orm.Int(2).store(),
                "y": orm.Int(3).store(),
                "code": add_calc.inputs.code,
            },
        )
        assert errors == []

    def test_missing_required_input(self) -> None:
        """Omitting a required port is reported as an error."""
        errors = validate_schema(
            "core.arithmetic.add",
            {"x": orm.Int(2).store()},  # y and code missing
        )
        assert any("y" in e for e in errors)

    def test_calcjob_requires_code(self) -> None:
        """CalcJob processes must require a 'code' input."""
        errors = validate_schema(
            "core.arithmetic.add",
            {
                "x": orm.Int(2).store(),
                "y": orm.Int(3).store(),
            },
        )
        assert any("code" in e for e in errors)

    def test_wrong_type_reported(self) -> None:
        """Passing a Str where Int is expected is caught."""
        errors = validate_schema(
            "core.arithmetic.add",
            {
                "x": orm.Str("not-a-number").store(),
                "y": orm.Int(3).store(),
            },
        )
        assert any("x" in e for e in errors)

    def test_unknown_entry_point_returns_error(self) -> None:
        """A non-existent entry point returns an error rather than raising."""
        errors = validate_schema("core.nonexistent.workflow", {})
        assert len(errors) == 1
        assert "core.nonexistent.workflow" in errors[0]

    def test_multiply_add_valid_inputs(
        self, multiply_add_workchain: orm.WorkChainNode
    ) -> None:
        """MultiplyAddWorkChain accepts Int nodes for x, y, z."""
        errors = validate_schema(
            "core.arithmetic.multiply_add",
            {
                "x": orm.Int(2).store(),
                "y": orm.Int(3).store(),
                "z": orm.Int(4).store(),
                "code": multiply_add_workchain.inputs.code,
            },
        )
        assert errors == []


class TestValidate:
    def test_raises_validation_error_on_missing_input(self) -> None:
        """validate() raises ValidationError, not a bare list."""
        with pytest.raises(ValidationError) as exc_info:
            validate("core.arithmetic.add", {})
        assert exc_info.value.errors
        assert any("x" in e for e in exc_info.value.errors)

    def test_passes_silently_for_valid_inputs(self, add_calc: orm.CalcJobNode) -> None:
        """validate() returns None and does not raise for valid inputs."""
        validate(
            "core.arithmetic.add",
            {
                "x": orm.Int(1).store(),
                "y": orm.Int(2).store(),
                "code": add_calc.inputs.code,
            },
        )

    def test_error_message_lists_all_failures(self) -> None:
        """ValidationError.errors contains one entry per failing port."""
        with pytest.raises(ValidationError) as exc_info:
            validate("core.arithmetic.add", {})
        # Both x and y are required
        errors = exc_info.value.errors
        assert any("x" in e for e in errors)
        assert any("y" in e for e in errors)


class TestSubmitWorkflowValidationGate:
    def test_raises_tool_error_when_validation_fails(self) -> None:
        """submit_workflow must not reach orm.submit if validation fails."""
        from fastmcp.exceptions import ToolError
        from aiida_agents.mcp.tools.submit import submit_workflow

        with pytest.raises(ToolError, match="Validation failed"):
            submit_workflow("core.arithmetic.add", {})

    def test_raises_tool_error_for_unknown_entry_point(self) -> None:
        """Unknown entry points are caught before any DB write."""
        from fastmcp.exceptions import ToolError
        from aiida_agents.mcp.tools.submit import submit_workflow

        with pytest.raises(ToolError):
            submit_workflow("core.nonexistent.workflow", {})

    def test_no_engine_call_without_valid_inputs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Neither engine entry point runs when validation fails.

        Both branches are patched: a broker profile reaches ``submit`` and a
        brokerless one (like the test profile) reaches ``run_get_node``, so
        guarding only one would pass for the wrong reason.
        """
        from fastmcp.exceptions import ToolError
        from aiida_agents.mcp.tools import submit as submit_mod

        called: list[str] = []
        monkeypatch.setattr(
            submit_mod, "submit", lambda *a, **k: called.append("submit")
        )
        monkeypatch.setattr(
            submit_mod, "run_get_node", lambda *a, **k: called.append("run_get_node")
        )

        with pytest.raises(ToolError, match="Validation failed"):
            submit_mod.submit_workflow("core.arithmetic.add", {})

        assert called == [], "engine was invoked despite validation failure"
