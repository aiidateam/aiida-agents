"""Tests for the shared model factory (agents/_models.py).

Pure unit tests — no agent construction, no AiiDA fixtures, no LLM calls.
One parametrized row per provider, so adding a provider is a one-line
change rather than a new test function.
"""

from __future__ import annotations

import pathlib
from collections.abc import Callable
from typing import Any, Literal

import pytest
from pydantic import ValidationError
from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel

from aiida_agents._settings import ModelSettings, OllamaSettings
from aiida_agents.agents._models import get_model

# OpenRouter model ids are ``vendor/model``; its provider splits on ``/`` to pick
# a model profile, so any test that builds an OpenRouter model needs a real id,
# not the ollama-style default (``qwen3.5:2b``), which fails to unpack.
_OPENROUTER_MODEL = "anthropic/claude-sonnet-4-6"


@pytest.fixture(autouse=True)
def _isolate_dotenv(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run each test in an empty dir so a stray repo ``.env`` can't leak in.

    Otherwise a dogfooding ``.env`` at the repo root (e.g. one setting a valid
    ``AIIDA_AGENTS_MODEL``) silently supplies config and masks defaults-based
    failures that CI, running with no ``.env``, would hit.
    """
    monkeypatch.chdir(tmp_path)


# ---------------------------------------------------------------------------
# Provider selection
#
# One row per provider: the model class it builds and, where the endpoint is
# ours to wire, the base URL it resolves to. ``base_url=None`` marks the cloud
# SDKs whose endpoint is the SDK's own default (asserting it would test the
# library, not our factory).
# ---------------------------------------------------------------------------

_PROVIDER_CASES = [
    pytest.param(
        "ollama", {}, OpenAIChatModel, "http://localhost:11434/v1", id="ollama"
    ),
    pytest.param("openai", {"OPENAI_API_KEY": "x"}, OpenAIChatModel, None, id="openai"),
    pytest.param(
        "anthropic", {"ANTHROPIC_API_KEY": "x"}, AnthropicModel, None, id="anthropic"
    ),
    pytest.param(
        "openrouter",
        {"OPENROUTER_API_KEY": "x", "AIIDA_AGENTS_MODEL": _OPENROUTER_MODEL},
        OpenAIChatModel,
        "https://openrouter.ai/api/v1",
        id="openrouter",
    ),
    pytest.param(
        "openai-compatible",
        {
            "AIIDA_AGENTS_BASE_URL": "https://api.deepseek.com/v1",
            "AIIDA_AGENTS_API_KEY": "x",
        },
        OpenAIChatModel,
        "https://api.deepseek.com/v1",
        id="openai-compatible",
    ),
]


@pytest.mark.parametrize(("provider", "env", "model_cls", "base_url"), _PROVIDER_CASES)
def test_get_model_builds_expected_model(
    provider: str,
    env: dict[str, str],
    model_cls: type,
    base_url: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each provider builds the correct model class wired to the right endpoint."""
    # Clean slate so a developer's own OLLAMA_BASE_URL / AIIDA_AGENTS_BASE_URL
    # can't leak into the default-endpoint rows.
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("AIIDA_AGENTS_BASE_URL", raising=False)
    monkeypatch.setenv("AIIDA_AGENTS_PROVIDER", provider)
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    model = get_model()

    assert isinstance(model, model_cls)
    if base_url is not None:
        # All endpoint-bearing providers wrap OpenAIChatModel; narrow for ``.provider``.
        assert isinstance(model, OpenAIChatModel)
        assert model.provider is not None
        assert str(getattr(model.provider, "base_url")).rstrip("/") == base_url


# ---------------------------------------------------------------------------
# Ollama endpoint configuration
# ---------------------------------------------------------------------------


def _ollama_from_env(monkeypatch: pytest.MonkeyPatch, base_url: str) -> Model:
    monkeypatch.setenv("AIIDA_AGENTS_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_BASE_URL", base_url)
    return get_model()


def _ollama_from_injection(monkeypatch: pytest.MonkeyPatch, base_url: str) -> Model:
    return get_model(
        model_settings=ModelSettings(provider="ollama"),
        ollama_settings=OllamaSettings(base_url=base_url),
    )


@pytest.mark.parametrize(
    "build_model",
    [
        pytest.param(_ollama_from_env, id="from-env"),
        pytest.param(_ollama_from_injection, id="from-injection"),
    ],
)
def test_ollama_endpoint_is_configurable(
    build_model: Callable[[pytest.MonkeyPatch, str], Model],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Ollama endpoint comes from ``OllamaSettings``, via env var or injection.

    Regression guard: the ``ollama`` branch must honor an injected
    ``OllamaSettings`` rather than re-reading ``OllamaSettings()`` from the
    environment, so explicit configuration is never silently ignored.
    """
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    endpoint = "http://configured:9999/v1"

    model = build_model(monkeypatch, endpoint)

    assert isinstance(model, OpenAIChatModel)
    assert model.provider is not None
    assert str(getattr(model.provider, "base_url")).rstrip("/") == endpoint


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
            "openrouter",
            {"OPENROUTER_API_KEY": "x", "AIIDA_AGENTS_MODEL": _OPENROUTER_MODEL},
        ),
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
    """The output cap reaches every provider's model settings."""
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
# Cloud credential plumbing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("provider", "key_field", "env_var", "model_id"),
    [
        pytest.param(
            "openai", "openai_api_key", "OPENAI_API_KEY", "gpt-4o", id="openai"
        ),
        pytest.param(
            "anthropic",
            "anthropic_api_key",
            "ANTHROPIC_API_KEY",
            "claude-sonnet-4-6",
            id="anthropic",
        ),
        pytest.param(
            "openrouter",
            "openrouter_api_key",
            "OPENROUTER_API_KEY",
            _OPENROUTER_MODEL,
            id="openrouter",
        ),
    ],
)
def test_cloud_key_comes_from_settings(
    provider: Literal["openai", "anthropic", "openrouter"],
    key_field: str,
    env_var: str,
    model_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The provider's key is taken from ModelSettings, not the SDK's env fallback.

    The matching env var is removed first, so a client that still carries the
    key proves get_model passed it through rather than the SDK reading it from
    the environment itself.
    """
    monkeypatch.delenv(env_var, raising=False)
    # ``Any`` value type: the single dynamic key is unpacked into ModelSettings,
    # which also has int-typed fields, so a ``dict[str, str]`` would not type-check.
    key_override: dict[str, Any] = {key_field: "secret"}
    settings = ModelSettings(provider=provider, model=model_id, **key_override)
    model = get_model(model_settings=settings)
    assert getattr(model, "provider").client.api_key == "secret"
