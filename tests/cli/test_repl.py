"""Tests for cli/repl.py: history windowing, the planner transcript, and the
prompt session."""

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

from aiida_agents._settings import ReplSettings
from aiida_agents.cli.repl import (
    _PLANNER_HISTORY_MAX_TURNS,
    _cap_history,
    _history_file,
    _key_bindings,
    _parse_agent_switch,
    _prompt_continuation,
    _record_planner_turn,
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


class TestThePlannerTranscript:
    """What the REPL hands the planner so a follow-up can refer back a turn.

    Deliberately not the specialists' histories: those carry tool calls the
    tool-less planner cannot read. This is a plain alternating user/assistant
    transcript, and these pin the three things the routing depends on -- the
    user's own words, the order, and the bound.
    """

    @staticmethod
    def _texts(history: list[ModelMessage]) -> list[str]:
        """The transcript as plain strings, in the order the planner reads it."""
        return [
            part.content
            for message in history
            for part in message.parts
            if isinstance(part, UserPromptPart | TextPart)
            and isinstance(part.content, str)
        ]

    def test_a_turn_is_recorded_as_the_question_then_the_answer(self) -> None:
        """Order is the whole point: "the former" resolves against the turn before."""
        history: list[ModelMessage] = []

        _record_planner_turn(history, "search for silicon structures", "PK 105, PK 150")
        _record_planner_turn(history, "relax the former", "submitted PK 161")

        assert self._texts(history) == [
            "search for silicon structures",
            "PK 105, PK 150",
            "relax the former",
            "submitted PK 161",
        ]
        assert [isinstance(message, ModelRequest) for message in history] == [
            True,
            False,
            True,
            False,
        ], "the planner reads a user turn and an assistant turn, alternating"

    def test_the_user_words_are_recorded_not_a_rendered_step_prompt(self) -> None:
        """A multi-step turn runs a handoff prompt; the planner must see the request.

        Routing the *next* turn works off what the user actually asked, so
        recording the generated step prompt instead would have the planner
        resolving references against words the user never wrote.
        """
        history: list[ModelMessage] = []

        _record_planner_turn(history, "why did pk 1234 fail", "the wallclock ran out")

        assert self._texts(history)[0] == "why did pk 1234 fail"

    def test_the_transcript_is_bounded_as_it_is_written(self) -> None:
        """Trimmed on write, so a long session cannot grow it without bound.

        The oldest turns go first: a demonstrative refers back a turn or two,
        and keeping every answer ever produced would put a session's worth of
        prose in front of a routing call whose entire output is one line.
        """
        history: list[ModelMessage] = []
        total = _PLANNER_HISTORY_MAX_TURNS + 3

        for i in range(total):
            _record_planner_turn(history, f"q{i}", f"a{i}")

        assert len(history) == 2 * _PLANNER_HISTORY_MAX_TURNS
        assert self._texts(history)[0] == f"q{total - _PLANNER_HISTORY_MAX_TURNS}"
        assert self._texts(history)[-1] == f"a{total - 1}"

    def test_a_short_session_keeps_every_turn(self) -> None:
        history: list[ModelMessage] = []

        for i in range(_PLANNER_HISTORY_MAX_TURNS):
            _record_planner_turn(history, f"q{i}", f"a{i}")

        assert len(history) == 2 * _PLANNER_HISTORY_MAX_TURNS
        assert self._texts(history)[0] == "q0"

    def test_the_planner_window_is_tighter_than_a_specialist_conversation(self) -> None:
        """The two caps size different needs and must not silently converge.

        ``history_max_turns`` sizes what a specialist needs to keep working;
        the planner only has to resolve a reference back a turn or two, and its
        docstrings sell it as one cheap round-trip.
        """
        assert _PLANNER_HISTORY_MAX_TURNS < ReplSettings().history_max_turns


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
