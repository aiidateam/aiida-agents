"""Tests for the cross-agent hand-off tool and its detection signal.

Whether a *real* model hands off at the right moment is a routing-quality
question for the eval tier. These pin the parts that must hold regardless of the
model: that a ``hand_off_to`` call parses to a request the REPL can route on, and
that anything else, or a target the REPL cannot run, leaves it on the current
agent rather than crashing.
"""

from __future__ import annotations

from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
)

from aiida_agents.agents.reroute import HandoffRequest, hand_off_to, handoff_request


def test_hand_off_to_names_the_target_and_reason() -> None:
    """The tool's return goes back to the model and the REPL routes on the call,
    so the string only has to name where it went and why.
    """
    out = hand_off_to("execution", "resubmit pk 1234 with a longer wallclock")
    assert "execution" in out
    assert "resubmit pk 1234 with a longer wallclock" in out


def test_handoff_request_reads_a_hand_off_call() -> None:
    """A ``hand_off_to`` call parses to which specialist to run next, and why."""
    messages: list[ModelMessage] = [
        ModelResponse(
            parts=[
                ToolCallPart(
                    "hand_off_to",
                    {"specialist": "codegen", "reason": "tabulate every relaxation"},
                )
            ]
        )
    ]
    assert handoff_request(messages) == HandoffRequest(
        "codegen", "tabulate every relaxation"
    )


def test_handoff_request_is_none_without_a_call() -> None:
    """A turn that only read or answered in prose is not re-routed."""
    messages: list[ModelMessage] = [
        ModelResponse(parts=[ToolCallPart("query_nodes", {"entity_type": "node"})]),
        ModelResponse(parts=[TextPart("343254 nodes.")]),
    ]
    assert handoff_request(messages) is None


def test_handoff_request_ignores_an_unknown_specialist() -> None:
    """A router must not dispatch on a target it cannot run: an out-of-set name
    is read as no hand-off rather than a crash.
    """
    messages: list[ModelMessage] = [
        ModelResponse(
            parts=[ToolCallPart("hand_off_to", {"specialist": "wizard", "reason": "x"})]
        )
    ]
    assert handoff_request(messages) is None
