"""Test suite for channel-1 workflow and calculation introspection tools (`list_process_entry_points`, `describe_process`)."""

from __future__ import annotations

import pytest

from aiida_agents.tools.execution.introspection import (
    describe_process,
    list_process_entry_points,
)


class TestIntrospectionTools:
    """Test `list_process_entry_points` and `describe_process`."""

    def test_list_process_entry_points_all_groups(self) -> None:
        """Ensure `list_process_entry_points` returns entries for both workflows and calculations when group=None."""
        res = list_process_entry_points()
        assert res["query_type"] == "list_process_entry_points"
        assert res["group"] is None
        assert isinstance(res["total_entry_points"], int)
        assert res["total_entry_points"] > 0
        assert "aiida.workflows" in res["entry_points"]
        assert "aiida.calculations" in res["entry_points"]
        assert "core.arithmetic.multiply_add" in res["entry_points"]["aiida.workflows"]

    def test_list_process_entry_points_specific_group(self) -> None:
        """Ensure `list_process_entry_points` filters by specific group when requested."""
        res = list_process_entry_points(group="aiida.workflows")
        assert res["group"] == "aiida.workflows"
        assert "aiida.workflows" in res["entry_points"]
        assert "aiida.calculations" not in res["entry_points"]
        assert "core.arithmetic.multiply_add" in res["entry_points"]["aiida.workflows"]

    def test_describe_process_valid_workflow(self) -> None:
        """Ensure `describe_process` extracts required/optional ports and exit codes correctly."""
        res = describe_process("core.arithmetic.multiply_add")
        assert res["query_type"] == "describe_process"
        assert res["entry_point"] == "core.arithmetic.multiply_add"
        assert res["group"] == "aiida.workflows"
        assert (
            res["process_class"]
            == "aiida.workflows.arithmetic.multiply_add:MultiplyAddWorkChain"
        )
        assert isinstance(res["has_protocol_builder"], bool)
        assert "x" in res["required_inputs"]
        assert "y" in res["required_inputs"]
        assert "code" in res["required_inputs"]
        assert isinstance(res["exit_codes"], list)

    def test_describe_process_valid_calculation(self) -> None:
        """Ensure `describe_process` works for calculations as well as workchains."""
        res = describe_process("core.arithmetic.add")
        assert res["group"] == "aiida.calculations"
        assert res["entry_point"] == "core.arithmetic.add"
        assert "x" in res["required_inputs"]
        assert "y" in res["required_inputs"]

    def test_describe_process_missing_entry_point(self) -> None:
        """Ensure `describe_process` raises ValueError when entry point is not found."""
        with pytest.raises(
            ValueError, match="not found in 'aiida.workflows' or 'aiida.calculations'"
        ):
            describe_process("nonexistent.workflow.entry_point")

    def test_describe_process_invalid_input(self) -> None:
        """Ensure `describe_process` raises ValueError when entry point is empty or not string."""
        with pytest.raises(ValueError, match="must be a non-empty string"):
            describe_process("")
