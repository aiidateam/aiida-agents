"""Unit tests for cli helpers that need no live model or database."""

from __future__ import annotations

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from aiida_agents.cli import _cap_history


def _turn(i: int) -> list[ModelMessage]:
    """One user turn with a tool round: a call/return pair lives inside it."""
    return [
        ModelRequest(parts=[UserPromptPart(content=f"q{i}")]),
        ModelResponse(
            parts=[ToolCallPart(tool_name="t", args={}, tool_call_id=f"c{i}")]
        ),
        ModelRequest(
            parts=[ToolReturnPart(tool_name="t", content="r", tool_call_id=f"c{i}")]
        ),
        ModelResponse(parts=[TextPart(content=f"a{i}")]),
    ]


def _user_turns(messages: list[ModelMessage]) -> int:
    return sum(
        isinstance(m, ModelRequest)
        and any(isinstance(p, UserPromptPart) for p in m.parts)
        for m in messages
    )


@pytest.mark.parametrize("max_turns", [1, 2, 3])
def test_cap_history_keeps_last_n_whole_turns(max_turns: int) -> None:
    """The window holds the last ``max_turns`` turns and starts on a boundary."""
    messages = [m for i in range(3) for m in _turn(i)]
    capped = _cap_history(messages, max_turns)

    assert _user_turns(capped) == min(max_turns, 3)
    # The window starts on a user-turn boundary, never mid tool round.
    assert isinstance(capped[0], ModelRequest)
    assert any(isinstance(p, UserPromptPart) for p in capped[0].parts)


def test_cap_history_never_orphans_a_tool_return() -> None:
    """Every tool return kept in the window still has its matching tool call.

    Regression for the count-based ``history[-N:]`` slice, which could start
    mid-pair and leave an unpaired ``tool_use``/``tool_result`` the provider
    rejects.
    """
    messages = [m for i in range(4) for m in _turn(i)]
    capped = _cap_history(messages, max_turns=2)

    call_ids = {
        p.tool_call_id for m in capped for p in m.parts if isinstance(p, ToolCallPart)
    }
    return_ids = {
        p.tool_call_id for m in capped for p in m.parts if isinstance(p, ToolReturnPart)
    }
    assert return_ids and return_ids <= call_ids


def test_cap_history_returns_input_when_within_budget() -> None:
    """Under the turn budget, the history is returned untouched."""
    messages = [m for i in range(2) for m in _turn(i)]
    assert _cap_history(messages, max_turns=5) is messages
