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


@pytest.mark.parametrize(
    ("raised", "expected_retry"),
    [
        pytest.param(
            AttributeError("process_label"),
            f"process_label {_RECHECK_HINT}",
            id="non-aiida-error-gets-recheck-hint",
        ),
        pytest.param(
            ModelRetry("give a real identifier, not 42"),
            "give a real identifier, not 42",
            id="tool-model-retry-forwarded-verbatim",
        ),
    ],
)
def test_unrecoverable_tool_error_is_bounded_not_fatal(
    raised: Exception, expected_retry: str
) -> None:
    """An always-failing tool is bounded by ``retries``, never escaping raw.

    Two boundary behaviours, same shape: a non-AiiDA error (the wrong-type-node
    case, an ``AttributeError`` that is *not* an ``AiidaException``) is converted
    to a retry carrying the recheck hint, and a tool's own ``ModelRetry`` is
    forwarded verbatim (no hint appended). Either way the raw error never
    escapes; once the retry budget is spent the run ends in
    ``UnexpectedModelBehavior``, not the original exception.
    """

    def always_fails(identifier: str) -> str:
        raise raised

    captured: dict[str, str] = {"retry": ""}
    toolset = RetryOnToolError(FunctionToolset([always_fails]))
    agent = Agent(
        FunctionModel(_call_then_answer("always_fails", captured)),
        toolsets=[toolset],
        retries=1,
    )

    with pytest.raises(UnexpectedModelBehavior):
        agent.run_sync("go")
    assert captured["retry"] == expected_retry
