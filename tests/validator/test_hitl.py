"""Regression tests for the HITL enforcement on submit_workflow.

These tests prove the agent half of the structural guarantee from ADR-08
(docs/adr/08-human-in-the-loop-before-writes.md): on the agent path there is
no way to submit to AiiDA without passing through human confirmation. The
other half -- that the standalone MCP server never exposes the write tool at
all -- is covered in tests/mcp/test_server.py.

Two invariants are tested:
1. submit_workflow is registered with requires_approval=True — the agent
   framework will never execute it inline; it always pauses for approval.
2. submit_workflow inputs are resolved and validated before the user is
   asked: invalid submissions are denied straight back to the model, only
   valid ones reach the confirmation prompt (via _triage_submissions).
"""

from __future__ import annotations

from aiida import orm
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.tools import DeferredToolRequests, ToolDenied

from aiida_agents.agents import get_agent
from aiida_agents.agents._errors import RetryOnToolError
from aiida_agents.agents.analysis import _READ_TOOLS
from aiida_agents.cli import _triage_submissions

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
        read_toolset = retry.wrapped
        assert isinstance(read_toolset, FunctionToolset)
        assert set(read_toolset.tools) == READ_TOOL_NAMES
        assert "submit_workflow" not in read_toolset.tools


MULTIPLY_ADD = "core.arithmetic.multiply_add"


class TestTriageSubmissions:
    """Inputs are resolved and validated before the user is asked: invalid
    submissions are denied straight back to the model, valid ones are queued
    for confirmation. This is the decision the deferred path used to skip.
    """

    @staticmethod
    def _pending(*calls: ToolCallPart) -> DeferredToolRequests:
        return DeferredToolRequests(approvals=list(calls))

    def test_invalid_submission_is_denied_without_prompting(self) -> None:
        call = ToolCallPart(
            tool_name="submit_workflow",
            args={"entry_point": MULTIPLY_ADD, "inputs": {"x": 1, "y": 2}},
            tool_call_id="c1",
        )
        auto, previews = _triage_submissions(self._pending(call))

        assert previews == []  # the user is never bothered with invalid inputs
        assert set(auto) == {"c1"}
        assert isinstance(auto["c1"], ToolDenied)
        assert "submit_workflow again" in auto["c1"].message

    def test_valid_submission_is_queued_for_the_user(
        self, arithmetic_add_code: orm.InstalledCode
    ) -> None:
        call = ToolCallPart(
            tool_name="submit_workflow",
            args={
                "entry_point": MULTIPLY_ADD,
                "inputs": {
                    "x": 2,
                    "y": 3,
                    "z": 4,
                    "code": {"pk": arithmetic_add_code.pk},
                },
            },
            tool_call_id="c1",
        )
        auto, previews = _triage_submissions(self._pending(call))

        assert auto == {}
        assert len(previews) == 1
        preview_call, resolved = previews[0]
        assert preview_call.tool_call_id == "c1"
        assert resolved is not None
        assert isinstance(resolved["x"], orm.Int) and resolved["x"].value == 2

    def test_non_submit_approval_falls_through_to_the_user(self) -> None:
        """Any other approval-gated tool is shown to the user with raw args."""
        call = ToolCallPart(tool_name="other_tool", args={}, tool_call_id="c2")
        auto, previews = _triage_submissions(self._pending(call))

        assert auto == {}
        assert previews == [(call, None)]
