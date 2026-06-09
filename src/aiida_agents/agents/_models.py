"""Shared model factory for all AiiDA agents.

All agents share one model, selected from environment variables (ADR-04).
This module is the single place to change provider or model configuration.

Environment variables
---------------------
AIIDA_AGENT_PROVIDER
    ``ollama`` (default) | ``openai`` | ``anthropic`` | ``openai-compatible``
AIIDA_AGENT_MODEL
    Model name. Default: ``qwen3.5:2b``.
OLLAMA_BASE_URL
    Ollama endpoint. Default: ``http://localhost:11434/v1``.
AIIDA_AGENT_BASE_URL
    Base URL for ``openai-compatible`` providers.
AIIDA_AGENT_API_KEY
    API key for ``openai-compatible`` providers (optional for keyless servers).
"""

from __future__ import annotations

import os

from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openai import OpenAIProvider


def get_model() -> Model:
    """Return the configured model from environment variables.

    Providers:

    * ``ollama`` — local Ollama server; ``OLLAMA_BASE_URL`` sets the endpoint.
    * ``openai`` — OpenAI cloud; reads ``OPENAI_API_KEY``.
    * ``anthropic`` — Anthropic cloud; reads ``ANTHROPIC_API_KEY``.
    * ``openai-compatible`` — any OpenAI-compatible endpoint (DeepSeek, Together,
      vLLM, etc.); requires ``AIIDA_AGENT_BASE_URL``.

    Raises:
        ValueError: For an unrecognised provider, or if ``openai-compatible``
            is selected without ``AIIDA_AGENT_BASE_URL``.
    """
    provider = os.getenv("AIIDA_AGENT_PROVIDER", "ollama").lower()
    model_name = os.getenv("AIIDA_AGENT_MODEL", "qwen3.5:2b")

    if provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        return OpenAIChatModel(model_name, provider=OllamaProvider(base_url=base_url))

    if provider == "openai":
        return OpenAIChatModel(model_name)

    if provider == "anthropic":
        from pydantic_ai.models.anthropic import AnthropicModel

        return AnthropicModel(model_name)

    if provider == "openai-compatible":
        compat_base_url = os.getenv("AIIDA_AGENT_BASE_URL")
        if not compat_base_url:
            msg = (
                "AIIDA_AGENT_PROVIDER='openai-compatible' requires AIIDA_AGENT_BASE_URL "
                "(e.g. https://api.deepseek.com/v1)."
            )
            raise ValueError(msg)
        api_key = os.getenv("AIIDA_AGENT_API_KEY", "api-key-not-set")
        return OpenAIChatModel(
            model_name,
            provider=OpenAIProvider(base_url=compat_base_url, api_key=api_key),
        )

    msg = (
        f"Unsupported AIIDA_AGENT_PROVIDER {provider!r}; "
        "use 'ollama', 'openai', 'anthropic', or 'openai-compatible'."
    )
    raise ValueError(msg)
