"""Tests for the AiiDA exploration agent (pairs with the get_model/get_agent refactor).

Three layers, no real LLM:
  * provider selection in ``get_model`` (which model/provider per env);
  * tool wiring (``get_agent`` exposes exactly our tools, via ``TestModel``);
  * one tool dispatched through the agent against a real fixture node (``FunctionModel``).

What can't be unit-tested, and isn't here: whether the model picks the *right* tool
for a phrasing, or answer quality. That is a real-model evaluation, not a mock.
"""

from __future__ import annotations

import pytest
from aiida import orm
from pydantic_ai import ModelResponse, TextPart, ToolCallPart, models
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.test import TestModel

from aiida_agents.agents import get_agent
from aiida_agents.agents._models import get_model

# Fail loudly if a test would hit a real endpoint (TestModel/FunctionModel are exempt).
models.ALLOW_MODEL_REQUESTS = False

EXPECTED_TOOLS = {
    "get_process_status",
    "list_processes",
    "query_nodes",
    "get_node_inputs",
    "get_node_outputs",
    "search_structures",
}


# get_model: provider selection (pure unit, no model run)


@pytest.mark.parametrize(
    ("provider", "env", "model_cls"),
    [
        ("ollama", {}, OpenAIChatModel),
        ("openai", {"OPENAI_API_KEY": "x"}, OpenAIChatModel),
        ("anthropic", {"ANTHROPIC_API_KEY": "x"}, AnthropicModel),
        (
            "openai-compatible",
            {"AIIDA_AGENT_BASE_URL": "https://api.deepseek.com/v1", "AIIDA_AGENT_API_KEY": "x"},
            OpenAIChatModel,
        ),
    ],
)
def test_get_model_builds_expected_model(provider, env, model_cls, monkeypatch):
    monkeypatch.setenv("AIIDA_AGENT_PROVIDER", provider)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert isinstance(get_model(), model_cls)


@pytest.mark.parametrize(
    ("provider", "env", "expected_base_url"),
    [
        ("ollama", {}, "http://localhost:11434/v1"),
        ("ollama", {"OLLAMA_BASE_URL": "http://remote:11434/v1"}, "http://remote:11434/v1"),
        (
            "openai-compatible",
            {"AIIDA_AGENT_BASE_URL": "https://api.deepseek.com/v1", "AIIDA_AGENT_API_KEY": "x"},
            "https://api.deepseek.com/v1",
        ),
    ],
)
def test_get_model_resolves_base_url(provider, env, expected_base_url, monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("AIIDA_AGENT_BASE_URL", raising=False)
    monkeypatch.setenv("AIIDA_AGENT_PROVIDER", provider)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert str(get_model().provider.base_url).rstrip("/") == expected_base_url


@pytest.mark.parametrize(
    ("env", "match"),
    [
        ({"AIIDA_AGENT_PROVIDER": "no-such-provider"}, "Unsupported"),
        ({"AIIDA_AGENT_PROVIDER": "openai-compatible"}, "AIIDA_AGENT_BASE_URL"),
    ],
)
def test_get_model_rejects_bad_config(env, match, monkeypatch):
    monkeypatch.delenv("AIIDA_AGENT_BASE_URL", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match=match):
        get_model()


# agent wiring (no real LLM)


def test_agent_exposes_expected_tools(monkeypatch):
    """get_agent wires exactly our tools onto the agent (no more, no fewer)."""
    monkeypatch.setenv("AIIDA_AGENT_PROVIDER", "ollama")
    agent = get_agent()
    fake = TestModel(call_tools=[])  # don't invoke tools; just inspect registration
    with agent.override(model=fake):
        agent.run_sync("ping")
    registered = {t.name for t in fake.last_model_request_parameters.function_tools}
    assert registered == EXPECTED_TOOLS


def test_tool_runs_through_agent_against_real_node(add_calc: orm.CalcJobNode, monkeypatch):
    """A tool dispatched through the agent executes against the real fixture node.

    The model is scripted (FunctionModel), so this tests the agent -> tool -> DB
    wiring, not tool selection. One such test covers the wiring for all tools, which
    pydantic-ai registers and marshals the same way; per-tool query behaviour lives
    in tests/mcp/.
    """
    pk = add_calc.pk

    def script(messages, info):
        if len(messages) == 1:  # drive the tool with the real pk
            return ModelResponse(
                parts=[ToolCallPart("get_process_status", {"identifier": str(pk)})]
            )
        tool_return = next(p for p in messages[-1].parts if p.part_kind == "tool-return")
        return ModelResponse(parts=[TextPart(str(tool_return.content))])

    monkeypatch.setenv("AIIDA_AGENT_PROVIDER", "ollama")
    agent = get_agent()
    with agent.override(model=FunctionModel(script)):
        result = agent.run_sync(f"status of {pk}")

    assert str(pk) in result.output
    assert "finished" in result.output
