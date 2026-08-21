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
from pathlib import Path
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
    def test_every_execution_write_tool_is_approval_gated(self) -> None:
        """The Execution agent's write tools are approval-gated, and nothing else is.

        Approval-capable tools live in the agent's function toolset (populated
        by ``tool_plain``); read tools sit in a separate plain toolset with no
        approval mechanism. Asserting the whole set, not just membership, means
        a further write tool added without ``requires_approval`` fails here too.
        """
        function_toolset = get_agent("execution")._function_toolset
        # Both tools that touch the database: the submission itself, and the
        # structure import that writes a StructureData from a file on disk.
        assert set(function_toolset.tools) == {
            "submit_process_spec",
            "submit_process_batch",
            "import_structure",
        }
        for name in (
            "submit_process_spec",
            "submit_process_batch",
            "import_structure",
        ):
            assert function_toolset.tools[name].requires_approval is True

    def test_analysis_agent_holds_no_write_tool(self) -> None:
        """The Analysis agent is read-only: submission is the Execution agent's.

        Nothing is registered via ``tool_plain`` at all, so no write tool can
        reach the database from this agent, gated or otherwise.
        """
        assert set(get_agent("analysis")._function_toolset.tools) == set()

    def test_read_tools_match_the_registered_set(self, without_plugins: None) -> None:
        """The read toolset is exactly ``_READ_TOOLS``, and no write tool leaks
        into it, an ungated write tool here would bypass approval.
        """
        agent = get_agent()
        retry = next(ts for ts in agent.toolsets if isinstance(ts, RetryOnToolError))
        read_toolset = retry.wrapped
        assert isinstance(read_toolset, FunctionToolset)
        assert set(read_toolset.tools) == READ_TOOL_NAMES
        assert "submit_workflow" not in read_toolset.tools
        assert "submit_process_spec" not in read_toolset.tools


MULTIPLY_ADD = "core.arithmetic.multiply_add"

# A minimal 2-atom silicon CIF, enough for ASE to parse into a structure.
_SILICON_CIF = """data_silicon
_cell_length_a 3.8
_cell_length_b 3.8
_cell_length_c 3.8
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Si1 0.0 0.0 0.0
Si2 0.5 0.5 0.5
"""


class TestSubmissionArgs:
    """Both write tools reduce to the same ``(entry_point, inputs)`` pair, but
    ``submit_process_spec`` nests them under ``spec`` while
    ``submit_workflow`` carries them at the top level.
    """

    def test_submit_workflow_reads_top_level_args(self) -> None:
        from aiida_agents.cli.hitl import _submission_args

        call = ToolCallPart(
            tool_name="submit_workflow",
            args={"entry_point": MULTIPLY_ADD, "inputs": {"x": 1}},
            tool_call_id="c1",
        )
        assert _submission_args(call) == (MULTIPLY_ADD, {"x": 1})

    def test_submit_process_spec_reads_nested_spec(self) -> None:
        from aiida_agents.cli.hitl import _submission_args

        call = ToolCallPart(
            tool_name="submit_process_spec",
            args={
                "spec": {
                    "entry_point": "aiida.workflows:PwRelaxWorkChain",
                    "inputs": {"structure": 1},
                }
            },
            tool_call_id="c1",
        )
        assert _submission_args(call) == (
            "aiida.workflows:PwRelaxWorkChain",
            {"structure": 1},
        )

    def test_submit_process_spec_missing_spec_yields_empty(self) -> None:
        """A malformed payload yields empty values, which _prepare_submission
        then rejects with an actionable message (rather than raising here).
        """
        from aiida_agents.cli.hitl import _submission_args

        call = ToolCallPart(tool_name="submit_process_spec", args={}, tool_call_id="c1")
        assert _submission_args(call) == ("", {})


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
        preview_call, process_class, resolved, _batch = previews[0]
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
        from aiida_agents.cli.hitl import _Preview

        assert previews == [_Preview(call, None, None)]

    def test_invalid_submit_process_spec_is_denied_naming_its_tool(self) -> None:
        """The execution agent's write tool goes through the same triage: an
        invalid one is denied without prompting, and the denial names the tool
        the model must retry (submit_process_spec, not submit_workflow).
        """
        call = ToolCallPart(
            tool_name="submit_process_spec",
            args={
                "spec": {
                    "entry_point": MULTIPLY_ADD,
                    "inputs": {"x": 1, "y": 2},  # missing z / code
                }
            },
            tool_call_id="c1",
        )
        auto, previews = _triage_submissions(self._pending(call))

        assert previews == []
        assert isinstance(auto["c1"], ToolDenied)
        assert "submit_process_spec again" in auto["c1"].message

    def test_valid_submit_process_spec_is_queued_for_the_user(
        self, arithmetic_add_code: orm.InstalledCode
    ) -> None:
        """A valid submit_process_spec resolves through its nested spec and
        reaches the confirmation prompt, exactly like submit_workflow.
        """
        call = ToolCallPart(
            tool_name="submit_process_spec",
            args={
                "spec": {
                    "entry_point": MULTIPLY_ADD,
                    "inputs": {
                        "x": 2,
                        "y": 3,
                        "z": 4,
                        "code": {"pk": arithmetic_add_code.pk},
                    },
                }
            },
            tool_call_id="c1",
        )
        auto, previews = _triage_submissions(self._pending(call))

        assert auto == {}
        assert len(previews) == 1
        _, process_class, resolved, _batch = previews[0]
        from aiida.plugins import WorkflowFactory

        assert process_class is WorkflowFactory(MULTIPLY_ADD)
        assert resolved is not None
        assert isinstance(resolved["x"], orm.Int) and resolved["x"].value == 2


class TestBatchTriage:
    """A batch is triaged as a set, on its own branch of ``_triage_submissions``.

    That branch reads each spec's entry point by key rather than through
    ``_submission_args``, so nothing else in this file exercises it: a spec key
    that stops matching resolves to ``""``, every ``_prepare_submission`` then
    fails, and the whole batch is auto-denied with a message naming a spec the
    model wrote correctly.
    """

    @staticmethod
    def _pending(*calls: ToolCallPart) -> DeferredToolRequests:
        return DeferredToolRequests(approvals=list(calls))

    @staticmethod
    def _spec(code: orm.InstalledCode, x: int) -> dict[str, Any]:
        return {
            "entry_point": MULTIPLY_ADD,
            "inputs": {"x": x, "y": 3, "z": 4, "code": {"pk": code.pk}},
        }

    def test_a_valid_batch_is_resolved_and_queued_as_one_preview(
        self, arithmetic_add_code: orm.InstalledCode
    ) -> None:
        """One approval covers the set, and it carries every spec resolved."""
        call = ToolCallPart(
            tool_name="submit_process_batch",
            args={
                "specs": [
                    self._spec(arithmetic_add_code, 2),
                    self._spec(arithmetic_add_code, 5),
                ]
            },
            tool_call_id="c1",
        )
        auto, previews = _triage_submissions(self._pending(call))

        assert auto == {}
        assert len(previews) == 1
        _, process_class, resolved, batch = previews[0]
        # The batch enrichment lives in ``batch``, not in the single-spec fields.
        assert process_class is None and resolved is None
        assert batch is not None
        assert [entry_point for entry_point, _ in batch] == [MULTIPLY_ADD] * 2
        assert [inputs["x"].value for _, inputs in batch] == [2, 5]

    def test_one_bad_spec_refuses_the_whole_batch_naming_its_position(
        self, arithmetic_add_code: orm.InstalledCode
    ) -> None:
        """All are approved together or none are, so a bad spec at position N
        stops the batch before the user sees it, and the denial says which.
        """
        bad = {"entry_point": MULTIPLY_ADD, "inputs": {"x": 1, "y": 2}}  # no z/code
        call = ToolCallPart(
            tool_name="submit_process_batch",
            args={"specs": [self._spec(arithmetic_add_code, 2), bad]},
            tool_call_id="c1",
        )
        auto, previews = _triage_submissions(self._pending(call))

        assert previews == []
        assert isinstance(auto["c1"], ToolDenied)
        assert "spec 2 of 2" in auto["c1"].message
        assert "submit_process_batch again" in auto["c1"].message

    @pytest.mark.parametrize("specs", [[], "not a list", None])
    def test_a_batch_that_is_not_a_non_empty_list_is_refused(self, specs: Any) -> None:
        call = ToolCallPart(
            tool_name="submit_process_batch",
            args={"specs": specs},
            tool_call_id="c1",
        )
        auto, previews = _triage_submissions(self._pending(call))

        assert previews == []
        assert isinstance(auto["c1"], ToolDenied)
        assert "non-empty list" in auto["c1"].message


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
        return {"entry_point": entry_point, "pk": 7, "state": "created"}

    monkeypatch.setattr(
        "aiida_agents.tools.execution.submit._run_submission", _fake_run_submission
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
        "ok": {"entry_point": "core.arithmetic.add", "pk": 7, "state": "created"},
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

    def test_async_write_tool_is_awaited_not_left_as_a_coroutine(self) -> None:
        """An async write tool runs to completion, rather than the approval flow
        storing its un-awaited coroutine as the "result".

        ``AgentTool.fn`` is any callable and pydantic-ai supports coroutines, so
        a plugin may well contribute an async write tool. Calling it without
        awaiting reproduces the very bug this module fixes: approved by the
        user, never actually run.
        """
        from aiida_agents.cli.hitl import _Preview, _run_approvals

        ran: list[int] = []

        async def plugin_write_async(x: int) -> str:
            """A plugin's async write tool."""
            ran.append(x)
            return f"wrote {x}"

        call = ToolCallPart(
            tool_name="plugin_write_async", args={"x": 9}, tool_call_id="c1"
        )
        outcomes = _run_approvals(
            _approval_agent(plugin_write_async), [_Preview(call, None, None)], {}
        )

        assert ran == [9]
        assert outcomes == {"c1": "wrote 9"}


class TestImportStructureRunsOnApproval:
    """The concrete case that made this bug reachable in shipped code.

    ``import_structure`` is an approval-gated write tool that is *not* a
    submission, so before this fix it was previewed, approved, and then dropped
    with ``{"skipped": "import_structure"}`` -- the user was told nothing, and
    no structure was ever created.
    """

    def test_approved_import_structure_actually_imports(self, tmp_path: Path) -> None:
        from aiida_agents.cli.hitl import _Preview, _run_approvals, _triage_submissions

        pytest.importorskip("ase", reason="needs ASE to parse the structure file")

        path = tmp_path / "si.cif"
        path.write_text(_SILICON_CIF)

        call = ToolCallPart(
            tool_name="import_structure",
            args={"filepath": str(path)},
            tool_call_id="c1",
        )
        # Triage leaves a non-submission unenriched: this is exactly the shape
        # that used to fall through to "not an executable submission".
        auto, previews = _triage_submissions(DeferredToolRequests(approvals=[call]))

        assert previews == [_Preview(call, None, None)]

        outcomes = _run_approvals(get_agent("execution"), previews, auto)

        result = outcomes["c1"]
        assert "skipped" not in result, "the approved import was silently dropped"
        # It really wrote a node, not just reported success.
        loaded = orm.load_node(result["pk"])
        assert isinstance(loaded, orm.StructureData)
        assert loaded.is_stored


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
        "aiida_agents.tools.execution.submit._format_resolved_inputs",
        lambda resolved: "INPUTS",
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
