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

from collections.abc import Callable
from typing import Any

import pytest
from aiida import orm
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.tools import DeferredToolRequests, ToolDenied

from aiida_agents.agents import get_agent
from aiida_agents.agents._errors import RetryOnToolError
from aiida_agents.agents.analysis import _READ_TOOLS
from aiida_agents.cli.hitl import _triage_submissions

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
        preview_call, process_class, resolved = previews[0]
        assert preview_call.tool_call_id == "c1"
        from aiida.plugins import WorkflowFactory

        assert process_class is WorkflowFactory(MULTIPLY_ADD)
        assert resolved is not None
        assert isinstance(resolved["x"], orm.Int) and resolved["x"].value == 2

    def test_non_submit_approval_falls_through_to_the_user(self) -> None:
        """Any other approval-gated tool is shown to the user with raw args."""
        call = ToolCallPart(tool_name="other_tool", args={}, tool_call_id="c2")
        auto, previews = _triage_submissions(self._pending(call))

        assert auto == {}
        assert previews == [(call, None, None)]


def _approval_agent(*fns: Callable[..., Any]) -> Agent:
    """A real agent with ``fns`` registered as approval-gated write tools."""
    agent: Agent = Agent(TestModel(), output_type=(str, DeferredToolRequests))
    for fn in fns:
        agent.tool_plain(requires_approval=True)(fn)
    return agent


def test_run_approvals_records_one_outcome_per_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each approval gets exactly one outcome, keyed by its tool-call id: an
    auto-denied input carries its denial, a tool the agent has no function for is
    skipped, a raising submission records its error, and a successful one records
    the run result.
    """
    from aiida_agents.cli.hitl import _Preview, _run_approvals

    def _fake_run_submission(
        entry_point: str, process_class: object, resolved: object
    ) -> dict[str, object]:
        if entry_point == "boom":
            raise RuntimeError("submit exploded")
        return {"workflow": entry_point, "pk": 7, "state": "created"}

    monkeypatch.setattr(
        "aiida_agents.tools.submit._run_submission", _fake_run_submission
    )

    ok = ToolCallPart(
        tool_name="submit_workflow",
        args={"entry_point": "core.arithmetic.add"},
        tool_call_id="ok",
    )
    err = ToolCallPart(
        tool_name="submit_workflow", args={"entry_point": "boom"}, tool_call_id="err"
    )
    other = ToolCallPart(tool_name="other", args={}, tool_call_id="skip")
    previews = [
        _Preview(ok, object(), {"x": 1}),
        _Preview(err, object(), {"x": 1}),
        _Preview(other, None, None),  # the agent has no such tool -> unrunnable
    ]

    outcomes = _run_approvals(
        _approval_agent(), previews, {"denied": ToolDenied("bad inputs")}
    )

    assert outcomes == {
        "denied": {"rejected": "bad inputs"},
        "ok": {"workflow": "core.arithmetic.add", "pk": 7, "state": "created"},
        "err": {"error": "submit exploded"},
        "skip": {"skipped": "other"},
    }


class TestNonSubmitWriteToolsRun:
    """A write tool that isn't a submission must actually run once approved.

    Regression: the approval flow only knew how to execute ``submit_workflow``,
    so any other approval-gated tool (a plugin's write tool) was previewed, the
    user approved it, and it was then silently dropped as "not an executable
    submission" -- the worst outcome, since the user believes they approved
    something that never ran.
    """

    def test_approved_plugin_write_tool_is_executed(self) -> None:
        from aiida_agents.cli.hitl import _Preview, _run_approvals

        ran: list[int] = []

        def plugin_write(x: int) -> str:
            """A plugin's write tool."""
            ran.append(x)
            return f"wrote {x}"

        call = ToolCallPart(tool_name="plugin_write", args={"x": 7}, tool_call_id="c1")
        outcomes = _run_approvals(
            _approval_agent(plugin_write), [_Preview(call, None, None)], {}
        )

        assert ran == [7]  # it really ran, with the model's arguments
        assert outcomes == {"c1": "wrote 7"}

    def test_raising_plugin_write_tool_records_its_error(self) -> None:
        """A failing plugin tool is reported back to the model, not fatal."""
        from aiida_agents.cli.hitl import _Preview, _run_approvals

        def plugin_write() -> str:
            """A plugin's write tool."""
            raise RuntimeError("plugin exploded")

        call = ToolCallPart(tool_name="plugin_write", args={}, tool_call_id="c1")
        outcomes = _run_approvals(
            _approval_agent(plugin_write), [_Preview(call, None, None)], {}
        )

        assert outcomes == {"c1": {"error": "plugin exploded"}}


def test_splice_outcomes_appends_one_tool_return_per_approval() -> None:
    """Every approval's outcome returns as its own ToolReturnPart, so the next
    turn has no unanswered tool call for pydantic-ai to reject.
    """
    from aiida_agents.cli.hitl import _splice_outcomes

    prior = ModelRequest(parts=[])
    call_a = ToolCallPart(tool_name="submit_workflow", args={}, tool_call_id="a")
    call_b = ToolCallPart(tool_name="submit_workflow", args={}, tool_call_id="b")
    pending = DeferredToolRequests(approvals=[call_a, call_b])

    class _Result:
        def all_messages(self) -> list[ModelMessage]:
            return [prior]

    spliced = _splice_outcomes(
        _Result(), pending, {"a": {"pk": 1}, "b": {"skipped": "x"}}
    )

    assert spliced[0] is prior
    added = spliced[1]
    assert isinstance(added, ModelRequest)
    returns = [part for part in added.parts if isinstance(part, ToolReturnPart)]
    assert len(returns) == len(added.parts)  # every part is a tool return
    assert [(part.tool_call_id, part.content) for part in returns] == [
        ("a", {"pk": 1}),
        ("b", {"skipped": "x"}),
    ]


class TestConfirmAndRun:
    @pytest.mark.parametrize("mode", ["declined", "ctrl-c-abort"])
    def test_cancel_runs_nothing(
        self, monkeypatch: pytest.MonkeyPatch, mode: str
    ) -> None:
        """Declining, or Ctrl-C/Ctrl-D at the prompt (click.Abort), returns the
        pre-turn history unchanged and never runs a submission.
        """
        import rich_click

        from aiida_agents.cli import hitl

        def _confirm(*args: object, **kwargs: object) -> bool:
            if mode == "ctrl-c-abort":
                raise rich_click.Abort
            return False

        submitted: list[int] = []

        def _record(previews: object, auto: object) -> dict[str, object]:
            submitted.append(1)
            return {}

        monkeypatch.setattr(hitl, "_print_previews", lambda previews: None)
        monkeypatch.setattr(rich_click, "confirm", _confirm)
        monkeypatch.setattr(hitl, "_run_approvals", _record)

        history: list[ModelMessage] = []
        out = hitl._confirm_and_run(
            _approval_agent(), None, DeferredToolRequests(approvals=[]), {}, [], history
        )

        assert out is history
        assert submitted == []

    def test_proceed_runs_then_splices(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """On approval the outcomes are computed then spliced into history."""
        import rich_click

        from aiida_agents.cli import hitl

        spliced: list[ModelMessage] = [ModelRequest(parts=[])]
        monkeypatch.setattr(hitl, "_print_previews", lambda previews: None)
        monkeypatch.setattr(rich_click, "confirm", lambda *a, **k: True)
        monkeypatch.setattr(
            hitl, "_run_approvals", lambda agent, previews, auto: {"id": {"pk": 1}}
        )
        monkeypatch.setattr(
            hitl, "_splice_outcomes", lambda result, pending, outcomes: spliced
        )

        out = hitl._confirm_and_run(
            _approval_agent(), None, DeferredToolRequests(approvals=[]), {}, [], []
        )

        assert out is spliced


def test_print_previews_shows_resolved_submission_and_raw_fallback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A resolved submission shows its entry point and formatted inputs; any other
    approval-gated tool falls back to its raw args.
    """
    from aiida_agents.cli import hitl
    from aiida_agents.cli.hitl import _Preview

    monkeypatch.setattr(
        "aiida_agents.tools.submit._format_resolved_inputs", lambda resolved: "INPUTS"
    )

    submit = ToolCallPart(
        tool_name="submit_workflow",
        args={"entry_point": "core.arithmetic.add"},
        tool_call_id="s",
    )
    other = ToolCallPart(tool_name="other", args={"k": "v"}, tool_call_id="o")
    hitl._print_previews(
        [_Preview(submit, object(), {"x": 1}), _Preview(other, None, None)]
    )

    out = capsys.readouterr().out
    assert "core.arithmetic.add" in out
    assert "INPUTS" in out
    assert "other" in out
    assert "{'k': 'v'}" in out  # raw args shown for the non-submit tool
