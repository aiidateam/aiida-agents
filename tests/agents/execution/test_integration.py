"""Test suite for full workflow integration and execution pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import DeferredToolRequests

from aiida_agents.agents.execution import get_agent
from aiida_agents.tools.execution.analysis_queries import query_analysis_agent
from aiida_agents.tools.workflows.introspection import describe_workflow, list_workflows
from aiida_agents.tools.workflows.schemas import WorkflowSpec
from aiida_agents.tools.workflows.spec_execution import execute_workflow_spec


class TestFullWorkflowIntegration:
    """Test full workflow progression: query → list/describe → execute."""

    def test_full_successful_workflow_discovery_and_describe(self) -> None:
        """Agent discovery workflow: query context → list workflows → describe specific workflow."""
        # Step 1: Query for context
        context = query_analysis_agent(
            query_type="past_successful_workflows",
            filters={
                "workflow_type": "core.arithmetic.multiply_add",
            },
        )
        assert isinstance(context["count"], int)
        assert context["count"] >= 0
        assert "success_rate" in context

        # Step 2: List workflows to discover available entry points
        available = list_workflows(group="aiida.workflows")
        assert (
            "core.arithmetic.multiply_add"
            in available["entry_points"]["aiida.workflows"]
        )

        # Step 3: Describe specific workflow schema
        schema = describe_workflow("core.arithmetic.multiply_add")
        assert schema["entry_point"] == "core.arithmetic.multiply_add"
        assert "x" in schema["required_inputs"]
        assert "y" in schema["required_inputs"]
        assert "code" in schema["required_inputs"]

    def test_execute_workflow_spec_unknown_workflow(self) -> None:
        """Execute tool should raise ValueError when workflow_type is not registered."""
        spec: WorkflowSpec = {
            "workflow_type": "nonexistent.fake.workflow",
            "inputs": {"x": 1},
        }
        with pytest.raises(ValueError, match="is not a known AiiDA entry point"):
            execute_workflow_spec(spec)

    def test_verify_submit_workflow_invoked(self) -> None:
        """Verify that calling submit_workflow actually invokes the submission tool/function."""
        from aiida_agents.tools.submit import submit_workflow

        with patch("aiida_agents.tools.submit._prepare_submission") as mock_prep:
            mock_process = MagicMock()
            mock_prep.return_value = (mock_process, {"x": 2, "y": 3})

            with patch("aiida_agents.tools.submit._run_submission") as mock_run:
                mock_run.return_value = {
                    "pk": 999,
                    "uuid": "1234-5678-90ab",
                    "workflow": "core.arithmetic.add",
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
        """Test execution gating: agent invokes execute_workflow_spec which requires HITL approval."""
        agent = get_agent()
        with agent.override(model=TestModel(call_tools=["execute_workflow_spec"])):
            res = agent.run_sync("Please submit the calculation.")
            assert isinstance(res.output, DeferredToolRequests)
            assert len(res.output.approvals) == 1
            assert res.output.approvals[0].tool_name == "execute_workflow_spec"
