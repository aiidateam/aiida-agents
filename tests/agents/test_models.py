"""Tests for the shared model factory (agents/_models.py).

Pure unit tests — no agent construction, no fixtures, no LLM calls.
Parametrized over all supported providers so adding a new provider
means adding one row to the table, not a new test function.
"""

from __future__ import annotations

import pytest
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel

from aiida_agents.agents._models import get_model


# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("provider", "env", "model_cls"),
    [
        ("ollama", {}, OpenAIChatModel),
        ("openai", {"OPENAI_API_KEY": "x"}, OpenAIChatModel),
        ("anthropic", {"ANTHROPIC_API_KEY": "x"}, AnthropicModel),
        (
            "openai-compatible",
            {
                "AIIDA_AGENT_BASE_URL": "https://api.deepseek.com/v1",
                "AIIDA_AGENT_API_KEY": "x",
            },
            OpenAIChatModel,
        ),
    ],
)
def test_get_model_builds_expected_model(
    provider: str,
    env: dict[str, str],
    model_cls: type,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each supported provider builds the correct model class."""
    monkeypatch.setenv("AIIDA_AGENT_PROVIDER", provider)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert isinstance(get_model(), model_cls)


# ---------------------------------------------------------------------------
# Base URL resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("provider", "env", "expected_base_url"),
    [
        ("ollama", {}, "http://localhost:11434/v1"),
        (
            "ollama",
            {"OLLAMA_BASE_URL": "http://remote:11434/v1"},
            "http://remote:11434/v1",
        ),
        (
            "openai-compatible",
            {
                "AIIDA_AGENT_BASE_URL": "https://api.deepseek.com/v1",
                "AIIDA_AGENT_API_KEY": "x",
            },
            "https://api.deepseek.com/v1",
        ),
    ],
)
def test_get_model_resolves_base_url(
    provider: str,
    env: dict[str, str],
    expected_base_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Base URL is read from the correct environment variable per provider."""
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("AIIDA_AGENT_BASE_URL", raising=False)
    monkeypatch.setenv("AIIDA_AGENT_PROVIDER", provider)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    model = get_model()
    assert isinstance(model, OpenAIChatModel)
    provider_obj = model.provider
    assert provider_obj is not None
    assert str(getattr(provider_obj, "base_url")).rstrip("/") == expected_base_url


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("env", "match"),
    [
        ({"AIIDA_AGENT_PROVIDER": "no-such-provider"}, "Unsupported"),
        ({"AIIDA_AGENT_PROVIDER": "openai-compatible"}, "AIIDA_AGENT_BASE_URL"),
    ],
)
def test_get_model_rejects_bad_config(
    env: dict[str, str],
    match: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown provider and missing base URL raise ValueError with a clear message."""
    monkeypatch.delenv("AIIDA_AGENT_BASE_URL", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match=match):
        get_model()
