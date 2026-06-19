"""Tests for the agent-side retry boundary in ``aiida_agents.agents._errors``.

These run the real ``RetryOnToolError`` toolset against a scripted model (no
network), proving the wiring that #9 is about: a tool failure must come back to
the model as a recoverable ``ModelRetry``, not abort the run. Tool-selection and
answer quality are a real-model concern, not tested here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from aiida.common.exceptions import NotExistent
from pydantic_ai import Agent, ModelRetry, models
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.toolsets import FunctionToolset

from aiida_agents.agents._errors import _RECHECK_HINT, RetryOnToolError

# Fail loudly if a test would hit a real endpoint (FunctionModel is exempt).
models.ALLOW_MODEL_REQUESTS = False

Script = Callable[[list[Any], Any], ModelResponse]


def _call_then_answer(
    tool_name: str, captured: dict[str, str], identifier: str = "42"
) -> Script:
    """Script a model that calls ``tool_name`` until it returns, then echoes it.

    A turn carrying a tool return is answered with that content; any other turn
    (the initial prompt, or a retry prompt) drives another call to the tool. The
    last retry-prompt content seen is written to ``captured["retry"]`` so a test
    can assert what guidance reached the model.
    """

    def script(messages: list[Any], info: Any) -> ModelResponse:
        last = messages[-1]
        for part in last.parts:
            if part.part_kind == "retry-prompt":
                captured["retry"] = str(part.content)
        tool_return = next(
            (p for p in last.parts if p.part_kind == "tool-return"), None
        )
        if tool_return is not None:
            return ModelResponse(parts=[TextPart(str(tool_return.content))])
        return ModelResponse(
            parts=[ToolCallPart(tool_name, {"identifier": identifier})]
        )

    return script


def test_aiida_error_recovers_and_carries_guidance() -> None:
    """A ``NotExistent`` from a tool becomes a retry the model recovers from."""
    calls = {"n": 0}

    def flaky(identifier: str) -> str:
        """Fail once with NotExistent, then succeed (a corrected identifier)."""
        calls["n"] += 1
        if calls["n"] == 1:
            raise NotExistent("no node 42")
        return "loaded ok"

    captured: dict[str, str] = {"retry": ""}
    toolset = RetryOnToolError(FunctionToolset([flaky]))
    agent = Agent(
        FunctionModel(_call_then_answer("flaky", captured)),
        toolsets=[toolset],
        retries=3,
    )

    result = agent.run_sync("go")

    assert calls["n"] == 2  # failed once, retried, then succeeded
    assert "loaded ok" in result.output
    # The guidance from describe_aiida_error is what the model saw on the retry.
    assert "list_processes()" in captured["retry"]


def test_non_aiida_error_is_converted_not_fatal() -> None:
    """A non-AiiDA tool error (the wrong-type-node case) does not crash the run.

    ``get_process_status`` on a data-node pk raises ``AttributeError``, which is
    not an ``AiidaException``. The boundary still converts it to a retry, so the
    raw error never escapes; it is bounded by ``retries`` and surfaces as
    ``UnexpectedModelBehavior`` once the model runs out of attempts.
    """

    def wrong_type(identifier: str) -> str:
        raise AttributeError("process_label")

    captured: dict[str, str] = {"retry": ""}
    toolset = RetryOnToolError(FunctionToolset([wrong_type]))
    agent = Agent(
        FunctionModel(_call_then_answer("wrong_type", captured)),
        toolsets=[toolset],
        retries=1,
    )

    with pytest.raises(UnexpectedModelBehavior):
        agent.run_sync("go")
    assert _RECHECK_HINT in captured["retry"]


def test_tool_model_retry_passes_through_unchanged() -> None:
    """A ``ModelRetry`` a tool raises itself is forwarded verbatim, not re-wrapped."""

    def asks_for_better_input(identifier: str) -> str:
        raise ModelRetry("give a real identifier, not 42")

    captured: dict[str, str] = {"retry": ""}
    toolset = RetryOnToolError(FunctionToolset([asks_for_better_input]))
    agent = Agent(
        FunctionModel(_call_then_answer("asks_for_better_input", captured)),
        toolsets=[toolset],
        retries=1,
    )

    with pytest.raises(UnexpectedModelBehavior):
        agent.run_sync("go")
    assert captured["retry"] == "give a real identifier, not 42"
    assert _RECHECK_HINT not in captured["retry"]
