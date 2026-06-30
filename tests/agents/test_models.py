"""Tests for the shared model factory (agents/_models.py).

Pure unit tests — no agent construction, no fixtures, no LLM calls.
Parametrized over all supported providers so adding a new provider
means adding one row to the table, not a new test function.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel

from aiida_agents._settings import ModelSettings, OllamaSettings
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
                "AIIDA_AGENTS_BASE_URL": "https://api.deepseek.com/v1",
                "AIIDA_AGENTS_API_KEY": "x",
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
    monkeypatch.setenv("AIIDA_AGENTS_PROVIDER", provider)
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
                "AIIDA_AGENTS_BASE_URL": "https://api.deepseek.com/v1",
                "AIIDA_AGENTS_API_KEY": "x",
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
    monkeypatch.delenv("AIIDA_AGENTS_BASE_URL", raising=False)
    monkeypatch.setenv("AIIDA_AGENTS_PROVIDER", provider)
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
    ("env", "exc_type", "match"),
    [
        # Bad provider is caught at settings load time by the Literal validation.
        (
            {"AIIDA_AGENTS_PROVIDER": "no-such-provider"},
            ValidationError,
            "literal_error",
        ),
        # Missing base URL is caught inside get_model.
        (
            {"AIIDA_AGENTS_PROVIDER": "openai-compatible"},
            ValueError,
            "AIIDA_AGENTS_BASE_URL",
        ),
    ],
)
def test_get_model_rejects_bad_config(
    env: dict[str, str],
    exc_type: type[Exception],
    match: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bad provider fails at settings load; missing base URL fails in get_model."""
    monkeypatch.delenv("AIIDA_AGENTS_BASE_URL", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(exc_type, match=match):
        get_model()


# ---------------------------------------------------------------------------
# Token budget: max_tokens (all providers) and context_length (Ollama num_ctx)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("provider", "env"),
    [
        ("ollama", {}),
        ("openai", {"OPENAI_API_KEY": "x"}),
        ("anthropic", {"ANTHROPIC_API_KEY": "x"}),
        (
            "openai-compatible",
            {
                "AIIDA_AGENTS_BASE_URL": "https://api.deepseek.com/v1",
                "AIIDA_AGENTS_API_KEY": "x",
            },
        ),
    ],
)
def test_get_model_applies_max_tokens(
    provider: str,
    env: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The output cap reaches every provider's model settings (all 4 branches)."""
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.setenv("AIIDA_AGENTS_PROVIDER", provider)
    monkeypatch.setenv("AIIDA_AGENTS_MAX_TOKENS", "4242")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    model = get_model()
    assert model.settings is not None
    assert model.settings["max_tokens"] == 4242


def test_get_model_sets_ollama_num_ctx_from_context_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``context_length`` is sent to Ollama as the ``num_ctx`` body field."""
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    model = get_model(
        model_settings=ModelSettings(
            provider="ollama", context_length=16384, max_tokens=8192
        )
    )
    assert isinstance(model, OpenAIChatModel)
    assert model.settings is not None
    assert model.settings["extra_body"] == {"num_ctx": 16384}


def test_get_model_omits_num_ctx_when_context_length_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no ``context_length``, no ``num_ctx`` is sent (Ollama's default holds)."""
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    model = get_model(model_settings=ModelSettings(provider="ollama"))
    assert isinstance(model, OpenAIChatModel)
    assert model.settings is not None
    assert "extra_body" not in model.settings


# ---------------------------------------------------------------------------
# Dependency injection
# ---------------------------------------------------------------------------


def test_get_model_uses_injected_ollama_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Ollama endpoint is taken from an injected ``OllamaSettings``.

    Regression for the half-DI gap: the ``ollama`` branch used to hard-read
    ``OllamaSettings()`` from the environment even when ``get_model`` was given
    explicit configuration, so the endpoint could not be injected.
    """
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    model = get_model(
        model_settings=ModelSettings(provider="ollama"),
        ollama_settings=OllamaSettings(base_url="http://injected:9999/v1"),
    )
    assert isinstance(model, OpenAIChatModel)
    provider_obj = model.provider
    assert provider_obj is not None
    assert (
        str(getattr(provider_obj, "base_url")).rstrip("/") == "http://injected:9999/v1"
    )
