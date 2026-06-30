"""Shared model factory for all AiiDA agents.

All agents share one model (ADR-04). Configuration is read from
``ModelSettings``, a typed, validated object that fails fast on bad
config rather than blowing up deep inside a tool call.
"""

from __future__ import annotations

from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings as PydanticModelSettings

from aiida_agents._settings import ModelSettings, OllamaSettings


def get_model(
    model_settings: ModelSettings | None = None,
    ollama_settings: OllamaSettings | None = None,
) -> Model:
    """Return the configured model.

    Args:
        model_settings: Model configuration. Read from env / ``.env`` if
            not given.
        ollama_settings: Ollama endpoint configuration, consulted only for
            ``provider='ollama'``. Read from env / ``.env`` if not given.

    Providers:

    * ``ollama``: local Ollama server; ``OLLAMA_BASE_URL`` sets the endpoint.
    * ``openai``: OpenAI cloud; reads ``OPENAI_API_KEY``.
    * ``anthropic``: Anthropic cloud; reads ``ANTHROPIC_API_KEY``.
    * ``openai-compatible``: any OpenAI-compatible endpoint (DeepSeek, Together,
      vLLM, etc.); requires ``AIIDA_AGENTS_BASE_URL``.

    Every model gets ``max_tokens`` as its output cap; for ``ollama``,
    ``context_length`` (if set) is sent as the per-request ``num_ctx``.

    Raises:
        ValidationError: When called without ``model_settings``, an
            unsupported provider fails here as ``ModelSettings()`` is
            constructed; a pre-built ``model_settings`` would already have
            failed on construction upstream.
        ValueError: If ``openai-compatible`` is selected without ``base_url``.
    """
    cfg = model_settings if model_settings is not None else ModelSettings()

    if cfg.provider == "ollama":
        ollama_cfg = (
            ollama_settings if ollama_settings is not None else OllamaSettings()
        )
        request_settings = OpenAIChatModelSettings(max_tokens=cfg.max_tokens)
        if cfg.context_length is not None:
            # Ollama reads num_ctx as a top-level body field; extra_body sends it.
            request_settings["extra_body"] = {"num_ctx": cfg.context_length}
        return OpenAIChatModel(
            cfg.model,
            provider=OllamaProvider(base_url=ollama_cfg.base_url),
            settings=request_settings,
        )

    if cfg.provider == "openai":
        return OpenAIChatModel(
            cfg.model, settings=OpenAIChatModelSettings(max_tokens=cfg.max_tokens)
        )

    if cfg.provider == "anthropic":
        from pydantic_ai.models.anthropic import AnthropicModel

        return AnthropicModel(
            cfg.model, settings=PydanticModelSettings(max_tokens=cfg.max_tokens)
        )

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
            settings=OpenAIChatModelSettings(max_tokens=cfg.max_tokens),
        )

    # Unreachable: Literal validation catches bad providers at settings load.
    # Included for type checker completeness.
    msg = f"Unsupported provider {cfg.provider!r}"  # pragma: no cover
    raise ValueError(msg)  # pragma: no cover
