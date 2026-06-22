"""Regression tests for the HITL enforcement on submit_workflow.

These tests prove the structural guarantee from ADR-08: there is no code
path that submits to AiiDA without passing through human confirmation.

Two invariants are tested:
1. submit_workflow is registered with requires_approval=True — the agent
   framework will never execute it inline; it always pauses for approval.
2. When the agent run returns a DeferredToolRequests, the CLI's
   _handle_deferred path calls build_results only after user confirms,
   and does nothing if the user declines.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from pydantic_ai.tools import DeferredToolRequests

from aiida_agents.agents import get_agent


# ---------------------------------------------------------------------------
# Structural: submit_workflow must be approval-required
# ---------------------------------------------------------------------------


class TestSubmitWorkflowRequiresApproval:
    def test_submit_workflow_has_requires_approval(self) -> None:
        """submit_workflow must be registered with requires_approval=True."""
        agent = get_agent()
        toolset = agent._function_toolset

        # Pydantic AI stores approval-required tools separately from plain tools
        approval_tool_names = {
            t.name
            for t in toolset.tools.values()
            if getattr(t, "requires_approval", False)
        }
        assert "submit_workflow" in approval_tool_names, (
            "submit_workflow must be registered with requires_approval=True (ADR-08). "
            "Without this, the agent can submit without user confirmation."
        )

    def test_read_tools_do_not_require_approval(self) -> None:
        """Read-only tools must not require approval — they should be frictionless."""
        agent = get_agent()
        toolset = agent._function_toolset

        read_tools = {
            "get_process_status",
            "list_processes",
            "query_nodes",
            "get_node_inputs",
            "get_node_outputs",
            "search_structures",
            "search_aiida_docs",
        }
        for name, tool in toolset.tools.items():
            if name in read_tools:
                assert not getattr(tool, "requires_approval", False), (
                    f"Read tool '{name}' must not require approval."
                )


# ---------------------------------------------------------------------------
# Behavioural: CLI HITL path
# ---------------------------------------------------------------------------


class TestHITLConfirmationPath:
    def test_declined_confirmation_does_not_call_build_results(self) -> None:
        """When the user declines, build_results is never called."""
        from aiida_agents.cli import _handle_deferred

        pending = MagicMock(spec=DeferredToolRequests)
        pending.approvals = [MagicMock(tool_name="submit_workflow", args={})]
        result = MagicMock()
        result.output = pending

        agent = MagicMock()

        with patch("builtins.input", return_value="n"):
            _handle_deferred(agent, result)

        pending.build_results.assert_not_called()
        agent.run.assert_not_called()

    def test_confirmed_calls_build_results_with_approve_all(self) -> None:
        """When the user confirms, build_results(approve_all=True) is called."""
        from aiida_agents.cli import _handle_deferred

        pending = MagicMock(spec=DeferredToolRequests)
        pending.approvals = [MagicMock(tool_name="submit_workflow", args={})]
        deferred_results = MagicMock()
        pending.build_results.return_value = deferred_results

        result = MagicMock()
        result.output = pending
        result.all_messages.return_value = []

        followup = MagicMock()
        followup.output = "Submitted successfully."
        agent = MagicMock()
        agent.run = MagicMock(return_value=followup)

        with (
            patch("builtins.input", return_value="y"),
            patch("asyncio.run", return_value=followup),
        ):
            _handle_deferred(agent, result)

        pending.build_results.assert_called_once_with(approve_all=True)

    def test_empty_input_treated_as_decline(self) -> None:
        """Pressing Enter without typing 'y' must not proceed."""
        from aiida_agents.cli import _handle_deferred

        pending = MagicMock(spec=DeferredToolRequests)
        pending.approvals = [MagicMock(tool_name="submit_workflow", args={})]
        result = MagicMock()
        result.output = pending
        agent = MagicMock()

        with patch("builtins.input", return_value=""):
            _handle_deferred(agent, result)

        pending.build_results.assert_not_called()
