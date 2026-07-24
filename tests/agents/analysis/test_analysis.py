"""Tests for the AiiDA exploration agent (``get_agent``).

Two layers, no real LLM:
  * tool wiring (``get_agent`` exposes exactly our tools, via ``TestModel``);
  * one tool dispatched through the agent against a real fixture node (``FunctionModel``).

Provider selection in ``get_model`` is covered separately in
``tests/agents/test_models.py``.

What can't be unit-tested, and isn't here: whether the model picks the *right* tool
for a phrasing, or answer quality. That is a real-model evaluation, not a mock.
"""

from typing import Any

import pytest
from aiida import orm
from pydantic_ai import ModelResponse, TextPart, ToolCallPart, models
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from aiida_agents.agents import get_agent

# Fail loudly if a test would hit a real endpoint (TestModel/FunctionModel are exempt).
models.ALLOW_MODEL_REQUESTS = False

EXPECTED_TOOLS = {
    "get_process_status",
    "list_processes",
    "query_nodes",
    "get_node_inputs",
    "get_node_outputs",
    "search_structures",
    "search_aiida_docs",
    "submit_workflow",
}


# agent wiring (no real LLM)


def test_agent_exposes_expected_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_agent wires exactly our tools onto the agent (no more, no fewer)."""
    monkeypatch.setenv("AIIDA_AGENTS_PROVIDER", "ollama")
    agent = get_agent()
    fake = TestModel(call_tools=[])  # don't invoke tools; just inspect registration
    with agent.override(model=fake):
        agent.run_sync("ping")
    params = fake.last_model_request_parameters
    assert params is not None
    registered = {t.name for t in params.function_tools}
    assert registered == EXPECTED_TOOLS


def test_tool_runs_through_agent_against_real_node(
    add_calc: orm.CalcJobNode,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tool dispatched through the agent executes against the real fixture node.

    The model is scripted (FunctionModel), so this tests the agent -> tool -> DB
    wiring, not tool selection. One such test covers the wiring for all tools, which
    pydantic-ai registers and marshals the same way; per-tool query behaviour lives
    in tests/mcp/.
    """
    pk = add_calc.pk

    def script(messages: list[Any], info: Any) -> ModelResponse:
        if len(messages) == 1:  # drive the tool with the real pk
            return ModelResponse(
                parts=[ToolCallPart("get_process_status", {"identifier": str(pk)})]
            )
        tool_return = next(
            p for p in messages[-1].parts if p.part_kind == "tool-return"
        )
        return ModelResponse(parts=[TextPart(str(tool_return.content))])

    monkeypatch.setenv("AIIDA_AGENTS_PROVIDER", "ollama")
    agent = get_agent()
    with agent.override(model=FunctionModel(script)):
        result = agent.run_sync(f"status of {pk}")

    assert str(pk) in result.output
    assert "finished" in result.output
