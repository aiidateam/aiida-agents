"""Shared model factory for all AiiDA agents.

All agents share one model (ADR-04). Configuration is read from
``ModelSettings``, a typed, validated object that fails fast on bad
config rather than blowing up deep inside a tool call.
"""

from __future__ import annotations

from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openai import OpenAIProvider

from aiida_agents._settings import ModelSettings, OllamaSettings
from pydantic_ai.settings import ModelSettings as PydanticModelSettings


def get_model(
    model_settings: ModelSettings | None = None,
    ollama_settings: OllamaSettings | None = None,
) -> Model:
    ...
    cfg = model_settings if model_settings is not None else ModelSettings()
    pai_settings = PydanticModelSettings(max_tokens=cfg.max_tokens)

    if cfg.provider == "ollama":
        ollama_cfg = (
            ollama_settings if ollama_settings is not None else OllamaSettings()
        )
        return OpenAIChatModel(
            cfg.model,
            provider=OllamaProvider(base_url=ollama_cfg.base_url),
            settings=pai_settings,
        )

    if cfg.provider == "openai":
        return OpenAIChatModel(cfg.model, settings=pai_settings)

    if cfg.provider == "anthropic":
        from pydantic_ai.models.anthropic import AnthropicModel

        return AnthropicModel(cfg.model, settings=pai_settings)

    if cfg.provider == "openai-compatible":
        if not cfg.base_url:
            msg = (
                "AIIDA_AGENTS_PROVIDER='openai-compatible' requires "
                "AIIDA_AGENTS_BASE_URL (e.g. https://api.deepseek.com/v1)."
            )
            raise ValueError(msg)
        return OpenAIChatModel(
            cfg.model,
            provider=OpenAIProvider(base_url=cfg.base_url, api_key=cfg.api_key),
            settings=pai_settings,
        )

    msg = f"Unsupported provider {cfg.provider!r}"  # pragma: no cover
    raise ValueError(msg)  # pragma: no cover
