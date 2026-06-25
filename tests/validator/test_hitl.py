"""Regression tests for the HITL enforcement on submit_workflow.

These tests prove the structural guarantee from ADR-08
(docs/adr/08-human-in-the-loop-before-writes.md): there is no code path
that submits to AiiDA without passing through human confirmation.

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
from aiida_agents.agents._errors import RetryOnToolError
from aiida_agents.agents.analysis import _READ_TOOLS

# Derived from the single source of truth in analysis.get_agent, so this tracks
# the registered read tools instead of duplicating their names by hand.
READ_TOOL_NAMES = frozenset(tool.__name__ for tool in _READ_TOOLS)


class TestSubmitWorkflowRequiresApproval:
    def test_submit_workflow_is_the_only_approval_tool(self) -> None:
        """submit_workflow is registered approval-gated, and nothing else is.

        Approval-capable tools live in the agent's function toolset (populated
        by ``tool_plain``); read tools sit in a separate plain toolset with no
        approval mechanism. Asserting the whole set, not just membership, means
        a second write tool added without ``requires_approval`` fails here too.
        """
        function_toolset = get_agent()._function_toolset
        assert set(function_toolset.tools) == {"submit_workflow"}
        assert function_toolset.tools["submit_workflow"].requires_approval is True

    def test_read_tools_match_the_registered_set(self) -> None:
        """The read toolset is exactly ``_READ_TOOLS``, and the write tool never
        leaks into it, an ungated submit_workflow here would bypass approval.
        """
        agent = get_agent()
        retry = next(ts for ts in agent.toolsets if isinstance(ts, RetryOnToolError))
        assert set(retry.wrapped.tools) == READ_TOOL_NAMES
        assert "submit_workflow" not in retry.wrapped.tools


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
