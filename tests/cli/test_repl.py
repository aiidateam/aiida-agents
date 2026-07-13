"""Tests for cli/repl.py: history windowing and the prompt session."""

from __future__ import annotations

from pathlib import Path

import pytest
from prompt_toolkit.keys import Keys
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from aiida_agents.cli.repl import (
    _cap_history,
    _history_file,
    _key_bindings,
    _parse_agent_switch,
    _prompt_continuation,
)


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


@pytest.mark.parametrize(
    "xdg, expected",
    [
        pytest.param(
            "/custom/data",
            Path("/custom/data/aiida-agents/repl-history"),
            id="xdg-set",
        ),
        pytest.param(
            None,
            Path.home() / ".local" / "share" / "aiida-agents" / "repl-history",
            id="xdg-unset",
        ),
    ],
)
def test_history_file_location(
    monkeypatch: pytest.MonkeyPatch, xdg: str | None, expected: Path
) -> None:
    """History lives under ``$XDG_DATA_HOME``, falling back to ``~/.local/share``."""
    if xdg is None:
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    else:
        monkeypatch.setenv("XDG_DATA_HOME", xdg)
    assert _history_file() == expected


def test_key_bindings_flip_enter_and_newline() -> None:
    """Enter submits; Esc+Enter and Ctrl+J (``ControlJ``, which most terminals
    also send for Ctrl+Enter) insert a newline: the deliberate flip of
    prompt_toolkit's multiline default (see ``_key_bindings``). ``Keys.Enter`` is
    the ``ControlM`` alias prompt_toolkit stores for a bare ``"enter"``.
    """
    bound = {binding.keys for binding in _key_bindings().bindings}
    assert bound == {
        (Keys.Enter,),
        (Keys.Escape, Keys.Enter),
        (Keys.ControlJ,),
    }


def test_prompt_continuation_aligns_under_prompt() -> None:
    """The continuation fills the prompt width so wrapped lines line up under it."""
    assert _prompt_continuation(len("You: "), 0, 0) == ".... "


@pytest.mark.parametrize(
    "question, expected",
    [
        pytest.param("/agent execution", "execution", id="switch"),
        pytest.param("/agent ANALYSIS", "analysis", id="case-insensitive"),
        pytest.param("/agent", None, id="bare-shows-usage"),
        pytest.param("/agent bogus", None, id="unknown-name-shows-usage"),
    ],
)
def test_parse_agent_switch(
    capsys: pytest.CaptureFixture[str], question: str, expected: str | None
) -> None:
    """``/agent <name>`` returns the requested agent; a bare or unknown name
    returns ``None`` and echoes the current agent plus usage.
    """
    assert _parse_agent_switch(question, current="analysis") == expected
    if expected is None:
        assert "Usage: /agent" in capsys.readouterr().out
