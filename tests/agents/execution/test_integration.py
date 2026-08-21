"""Test suite for full workflow integration and execution pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import DeferredToolRequests

from aiida_agents.agents.execution import get_agent
from aiida_agents.tools.run_context import query_run_context
from aiida_agents.tools.execution.introspection import (
    describe_process,
    list_process_entry_points,
)
from aiida_agents.tools.execution.schemas import SubmissionSpec
from aiida_agents.tools.execution.spec_execution import submit_process_spec


class TestFullWorkflowIntegration:
    """Test full workflow progression: query → list/describe → execute."""

    def test_full_successful_workflow_discovery_and_describe(self) -> None:
        """Agent discovery workflow: query context → list workflows → describe specific workflow."""
        # Step 1: Query for context
        context = query_run_context(
            query_type="past_successful_workflows",
            filters={
                "workflow_type": "core.arithmetic.multiply_add",
            },
        )
        assert isinstance(context["count"], int)
        assert context["count"] >= 0
        assert "success_rate" in context

        # Step 2: List workflows to discover available entry points
        available = list_process_entry_points(group="aiida.workflows")
        assert (
            "core.arithmetic.multiply_add"
            in available["entry_points"]["aiida.workflows"]
        )

        # Step 3: Describe specific workflow schema
        schema = describe_process("core.arithmetic.multiply_add")
        assert schema["entry_point"] == "core.arithmetic.multiply_add"
        assert "x" in schema["required_inputs"]
        assert "y" in schema["required_inputs"]
        assert "code" in schema["required_inputs"]

    def test_submit_process_spec_unknown_entry_point(self) -> None:
        """Submitting a spec whose entry_point is not registered raises ValueError."""
        spec: SubmissionSpec = {
            "entry_point": "nonexistent.fake.workflow",
            "inputs": {"x": 1},
        }
        with pytest.raises(ValueError, match="is not a known AiiDA entry point"):
            submit_process_spec(spec)

    def test_verify_submit_workflow_invoked(self) -> None:
        """Verify that calling submit_workflow actually invokes the submission tool/function."""
        from aiida_agents.tools.execution.submit import submit_workflow

        with patch(
            "aiida_agents.tools.execution.submit._prepare_submission"
        ) as mock_prep:
            mock_process = MagicMock()
            mock_prep.return_value = (mock_process, {"x": 2, "y": 3})

            with patch(
                "aiida_agents.tools.execution.submit._run_submission"
            ) as mock_run:
                mock_run.return_value = {
                    "pk": 999,
                    "uuid": "1234-5678-90ab",
                    "entry_point": "core.arithmetic.add",
                    "state": "created",
                }

                res = submit_workflow("core.arithmetic.add", {"x": 2, "y": 3})
                mock_prep.assert_called_once_with(
                    "core.arithmetic.add", {"x": 2, "y": 3}
                )
                mock_run.assert_called_once_with(
                    "core.arithmetic.add", mock_process, {"x": 2, "y": 3}
                )
                assert res["pk"] == 999
                assert res["state"] == "created"

    def test_full_execution_preview_requires_approval(self) -> None:
        """Test execution gating: agent invokes submit_process_spec which requires HITL approval."""
        agent = get_agent()
        with agent.override(model=TestModel(call_tools=["submit_process_spec"])):
            res = agent.run_sync("Please submit the calculation.")
            assert isinstance(res.output, DeferredToolRequests)
            assert len(res.output.approvals) == 1
            assert res.output.approvals[0].tool_name == "submit_process_spec"


EXPECTED_EXECUTION_TOOLS = {
    "query_run_context",
    "list_process_entry_points",
    "describe_process",
    "build_process_inputs",
    "draft_process_inputs",
    "check_cutoffs_against_pseudos",
    "build_resubmission_spec",
    "submit_process_batch",
    "list_codes",
    "get_process_status",
    "wait_for_process",
    "get_daemon_status",
    "search_aiida_docs",
    "submit_process_spec",
    "import_structure",
    "hand_off_to",  # the cross-agent hand-off signal (see agents/reroute.py)
}


def test_execution_agent_exposes_expected_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """get_agent wires exactly the execution tools onto the agent.

    Pins the tool surface the way the Analysis agent's own test does, so a tool
    added or dropped here is a deliberate edit rather than a silent drift.
    ``get_process_status`` is shared with the Analysis agent on purpose: it is
    how this agent follows up on what it just submitted.
    """
    monkeypatch.setenv("AIIDA_AGENTS_PROVIDER", "ollama")
    agent = get_agent()
    fake = TestModel(call_tools=[])  # register only; don't invoke
    with agent.override(model=fake):
        agent.run_sync("ping")
    params = fake.last_model_request_parameters
    assert params is not None
    assert {t.name for t in params.function_tools} == EXPECTED_EXECUTION_TOOLS
