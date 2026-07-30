"""Human-in-the-loop approval flow for write tool calls.

A write tool is registered with ``requires_approval=True`` (ADR-08), so a run
that wants to write pauses and returns a ``DeferredToolRequests``. This module
previews each proposed call, gets the user's decision, runs the approved ones on
the main thread, and splices the outcomes back into the message history.

Approved calls run *here* rather than by handing them back to pydantic-ai, which
would execute them on a worker thread: AiiDA's storage is thread-bound, so a
write from the worker thread raises a cross-thread error. Every approval-gated
tool therefore needs a main-thread executor, not just the submissions
(``submit_workflow`` / ``execute_workflow_spec``): ``_run_approvals`` calls the
tool's own registered function, and submissions get an extra resolve-and-preview
pass on top (see ``_triage_submissions``). Without that, a non-submission write
tool -- ``import_structure``, or any write tool a plugin contributes -- is
previewed, approved, and then silently dropped.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any, NamedTuple

import rich_click as click
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ToolReturnPart
from pydantic_ai.tools import DeferredToolRequests

from aiida_agents.cli.output import _log_tool_calls_debug, _print_agent, console

logger = logging.getLogger(__name__)


# Bound the propose -> deny -> retry loop so a model that keeps emitting bad
# inputs cannot spin forever.
_MAX_APPROVAL_ROUNDS = 10

# The approval-gated write tools that resolve to an AiiDA submission. Both funnel
# through the same resolve/validate/submit path; they differ only in where the
# entry point and inputs sit in the tool-call args (see ``_submission_args``).
_SUBMIT_TOOLS = ("submit_workflow", "execute_workflow_spec")


def _submission_args(call: Any) -> tuple[str, dict[str, Any]]:
    """Extract ``(entry_point, inputs)`` from a submit tool call.

    ``submit_workflow`` carries them at the top level; ``execute_workflow_spec``
    nests them under ``validated_spec`` as ``workflow_type`` / ``inputs``. A
    malformed ``execute_workflow_spec`` payload yields empty values, which
    ``_prepare_submission`` then rejects with an actionable message.
    """
    args = call.args_as_dict()
    if call.tool_name == "execute_workflow_spec":
        spec = args.get("validated_spec", {})
        if isinstance(spec, dict):
            return spec.get("workflow_type", ""), spec.get("inputs", {})
        return "", {}
    return args.get("entry_point", ""), args.get("inputs", {})


class _Preview(NamedTuple):
    """A pending tool call awaiting the user's decision.

    ``process_class`` and ``resolved`` carry the submission-specific enrichment
    (the loaded process class and its resolved inputs) so the preview can show
    what will actually be created. They are None for any other approval-gated
    tool, which is previewed with its raw args and executed by calling its own
    registered function (see ``_run_approvals``).
    """

    call: Any
    process_class: Any
    resolved: dict[str, Any] | None


def _tool_function(agent: Agent, name: str) -> Any:
    """The function registered behind an approval-gated tool, or None.

    Approved calls are executed here on the main thread, so we need the tool's
    own callable. pydantic-ai keeps it on the agent's function toolset, which is
    where ``tool_plain(requires_approval=True)`` puts it.
    """
    tool = agent._function_toolset.tools.get(name)
    return getattr(tool, "function", None)


def _triage_submissions(
    pending: DeferredToolRequests,
) -> tuple[dict[str, Any], list[_Preview]]:
    """Resolve and validate each pending approval before the user sees it.

    Returns ``(auto_denials, previews)``:

    * ``auto_denials`` maps a tool-call id to a ``ToolDenied`` for any submit
      tool call (``submit_workflow`` / ``execute_workflow_spec``) whose inputs
      fail resolution or validation. These go straight back to the model so it
      can correct its own mistakes without bothering the user.
    * ``previews`` lists ``(call, process_class, resolved)`` for the calls the
      user must decide on: ``process_class`` / ``resolved`` are the loaded
      process class and resolved-inputs dict for a valid submission (so the
      caller can submit on the main thread), or ``None`` for any other
      approval-gated tool.
    """
    from pydantic_ai.tools import ToolDenied

    from aiida_agents.tools.execution.submit import (
        SubmissionInputError,
        _prepare_submission,
    )

    auto: dict[str, Any] = {}
    previews: list[_Preview] = []
    for call in pending.approvals:
        if call.tool_name not in _SUBMIT_TOOLS:
            previews.append(_Preview(call, None, None))
            continue
        entry_point, inputs = _submission_args(call)
        try:
            process_class, resolved = _prepare_submission(entry_point, inputs)
        except SubmissionInputError as exc:
            auto[call.tool_call_id] = ToolDenied(
                f"Submission rejected before reaching the user: {exc} "
                f"Correct the inputs and call {call.tool_name} again."
            )
            continue
        previews.append(_Preview(call, process_class, resolved))
    return auto, previews


def _range_warnings(call: Any) -> list[str]:
    """Physics findings to show beside a submission, if there are any.

    Run here rather than only offered as a tool, for the same reason the
    grounding check runs on every reply: an instruction to check can be
    ignored, and this is the last moment before core-hours are spent. A
    failure inside the check is swallowed --- an approval prompt that raises
    instead of asking is worse than one that shows nothing.
    """
    from aiida_agents.tools.execution.ranges import check_input_ranges

    entry_point, inputs = _submission_args(call)
    if not entry_point or not isinstance(inputs, dict):
        return []
    try:
        findings = check_input_ranges({"workflow_type": entry_point, "inputs": inputs})
    except Exception:
        logger.debug("range check failed for %s", entry_point, exc_info=True)
        return []
    return [finding["message"] for finding in findings]


def _print_previews(previews: list[_Preview]) -> None:
    """Print the calls awaiting the user's confirmation.

    A submission shows its entry point and fully resolved inputs, so the user can
    see what will be created before anything is written; any other approval-gated
    tool shows the raw arguments the model supplied, plus any cutoff that
    disagrees with what its pseudopotentials were converged for.
    """
    from aiida_agents.tools.execution.submit import _format_resolved_inputs

    click.echo("\n⚠️  The agent wants to perform the following action(s):")
    for call, _, resolved in previews:
        click.echo(f"   Tool  : {call.tool_name}")
        if resolved is None:
            click.echo(f"   Inputs: {call.args_as_dict()}")
            continue
        entry_point, _ = _submission_args(call)
        click.echo(f"   Entry : {entry_point or '<unknown>'}")
        click.echo(f"   Inputs (resolved):\n{_format_resolved_inputs(resolved)}")
        for message in _range_warnings(call):
            click.echo(f"   ⚠️  {message}")


def _run_approvals(
    agent: Agent,
    previews: list[_Preview],
    auto: dict[str, Any],
) -> dict[str, Any]:
    """Execute each approved call on the main thread, one outcome per tool-call id.

    Runs *here*, not by re-running the agent: pydantic-ai runs sync tools on a
    worker thread and AiiDA's storage is thread-bound, so writing from the worker
    thread (reusing the default user / nodes the preview bound to the main-thread
    session) raises a cross-thread SQLAlchemy error (ADR-08). Auto-denied invalid
    submissions never ran, so they carry their denial message straight through.

    A submission runs through ``_run_submission`` with the inputs already resolved
    at triage; any other approval-gated tool is executed by calling its own
    registered function with the arguments the model supplied. A tool the agent
    has no function for cannot be run at all, so it is reported as skipped rather
    than silently dropped.
    """
    from aiida_agents.tools.execution.submit import _run_submission

    outcomes: dict[str, Any] = {
        call_id: {"rejected": denied.message} for call_id, denied in auto.items()
    }
    for call, process_class, resolved in previews:
        if process_class is not None and resolved is not None:
            # Not args_as_dict()["entry_point"]: execute_workflow_spec nests it
            # under validated_spec, so only _submission_args reads both shapes.
            entry_point, _ = _submission_args(call)
            try:
                res = _run_submission(entry_point, process_class, resolved)
            except Exception as exc:
                click.echo(f"\n❌ Submission failed: {exc}")
                outcomes[call.tool_call_id] = {"error": str(exc)}
                continue
            click.echo(
                f"\n✅ Submitted {res['workflow']}: pk={res['pk']}, state={res['state']}"
            )
            outcomes[call.tool_call_id] = res
            continue

        fn = _tool_function(agent, call.tool_name)
        if fn is None:
            click.echo(f"   Skipping {call.tool_name}: no registered function to run.")
            outcomes[call.tool_call_id] = {"skipped": call.tool_name}
            continue
        try:
            result = fn(**call.args_as_dict())
            # A plugin may contribute an async write tool (AgentTool.fn is any
            # callable, and pydantic-ai supports coroutines). Calling one just
            # returns a coroutine, so without this the tool never runs -- the
            # same silent no-op this function exists to fix.
            if inspect.iscoroutine(result):
                result = asyncio.run(result)
        except Exception as exc:
            click.echo(f"\n❌ {call.tool_name} failed: {exc}")
            outcomes[call.tool_call_id] = {"error": str(exc)}
            continue
        click.echo(f"\n✅ Ran {call.tool_name}.")
        outcomes[call.tool_call_id] = result
    return outcomes


def _splice_outcomes(
    result: Any,
    pending: DeferredToolRequests,
    outcomes: dict[str, Any],
) -> list[ModelMessage]:
    """Append each approval's outcome as its ``ToolReturnPart``.

    The submissions ran out of band (not through pydantic-ai), so it never
    recorded their tool returns; splicing them back keeps the submission in
    context and leaves no unanswered tool call for pydantic-ai to reject next
    turn.
    """
    updated: list[ModelMessage] = result.all_messages()
    updated.append(
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name=call.tool_name,
                    content=outcomes[call.tool_call_id],
                    tool_call_id=call.tool_call_id,
                )
                for call in pending.approvals
            ]
        )
    )
    return updated


def _confirm_and_run(
    agent: Agent,
    result: Any,
    pending: DeferredToolRequests,
    auto: dict[str, Any],
    previews: list[_Preview],
    history: list[ModelMessage],
) -> list[ModelMessage]:
    """Preview the pending calls, get the user's go/no-go, run the approved ones,
    and splice their outcomes into history.

    Cancelling (an explicit no, or Ctrl-C / Ctrl-D, which raise ``click.Abort``)
    returns the pre-turn ``history`` unchanged and runs nothing, staying in the
    REPL rather than tearing down the session.
    """
    _print_previews(previews)
    try:
        proceed = click.confirm("\nProceed?", default=False)
    except click.Abort:
        proceed = False
    if not proceed:
        click.echo("Cancelled - nothing was run.")
        return history

    outcomes = _run_approvals(agent, previews, auto)
    click.echo()  # separate the summary from the next prompt
    return _splice_outcomes(result, pending, outcomes)


def _rerun_with_denials(
    agent: Agent,
    result: Any,
    pending: DeferredToolRequests,
    auto: dict[str, Any],
) -> Any:  # pragma: no cover
    """Deny the invalid submissions back to the model and re-run so it corrects
    its own inputs.

    No DB write happens on the worker thread here (denied calls are never
    executed), so this is thread-safe. Returns the new agent result, which may
    itself be a ``DeferredToolRequests`` if the correction proposed fresh
    submissions.
    """
    click.echo("\n⚠️  Inputs were invalid; asking the agent to correct them.")
    rerun = asyncio.run(
        agent.run(
            None,
            message_history=result.all_messages(),
            deferred_tool_results=pending.build_results(approvals=auto),
        )
    )
    _log_tool_calls_debug(rerun.new_messages(), console)
    return rerun


def _handle_deferred(
    agent: Agent,
    result: Any,
    history: list[ModelMessage],
) -> list[ModelMessage]:  # pragma: no cover
    """Confirm and run pending submissions, denying invalid ones to the model.

    Each round triages the pending approvals (:func:`_triage_submissions`): if any
    reach the user they are confirmed and submitted (terminal); otherwise the
    invalid ones are denied back to the model to correct and the agent re-runs,
    looping until it stops proposing submissions or the retry budget is spent.
    Cancelling or exhausting the budget returns the pre-turn ``history`` unchanged.
    """
    for _ in range(_MAX_APPROVAL_ROUNDS):
        pending = result.output
        auto, previews = _triage_submissions(pending)

        if previews:
            return _confirm_and_run(agent, result, pending, auto, previews, history)

        if not auto:
            return history

        try:
            result = _rerun_with_denials(agent, result, pending, auto)
        except Exception as exc:
            click.echo(f"\n❌ Error: {exc}")
            return history

        if not isinstance(result.output, DeferredToolRequests):
            _print_agent(result.output)
            messages: list[ModelMessage] = result.all_messages()
            return messages

    click.echo("\n⚠️  Too many correction rounds; stopping without submitting.")
    return history
