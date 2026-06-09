"""AiiDA exploration agent using Pydantic AI.

Model selection is driven by environment variables:

    AIIDA_AGENT_PROVIDER     ollama (default) | openai | anthropic | openai-compatible
    AIIDA_AGENT_MODEL        model name, default qwen3.5:2b
    OLLAMA_BASE_URL          Ollama endpoint, default http://localhost:11434/v1
    AIIDA_AGENT_BASE_URL     base URL for openai-compatible providers
    AIIDA_AGENT_API_KEY      API key for openai-compatible providers (optional)
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openai import OpenAIProvider

from aiida_agents.mcp.tools.nodes import get_node_inputs, get_node_outputs, query_nodes
from aiida_agents.mcp.tools.processes import get_process_status, list_processes
from aiida_agents.mcp.tools.structures import search_structures
from aiida_agents.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

_TOOLS: list[Any] = [
    get_process_status,
    list_processes,
    query_nodes,
    get_node_inputs,
    get_node_outputs,
    search_structures,
]


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


def get_model() -> Model:
    """Return the configured model from environment variables.

    Providers (``AIIDA_AGENT_PROVIDER``):

    * ``ollama`` (default) — any local model served by Ollama.
      ``OLLAMA_BASE_URL`` overrides the ``http://localhost:11434/v1`` default.
    * ``openai`` — hosted OpenAI models; reads ``OPENAI_API_KEY``.
    * ``anthropic`` — hosted Anthropic models; reads ``ANTHROPIC_API_KEY``.
    * ``openai-compatible`` — any OpenAI-compatible endpoint (DeepSeek, Together,
      Fireworks, Perplexity, Azure, OpenRouter, vLLM, ...).
      Requires ``AIIDA_AGENT_BASE_URL``; ``AIIDA_AGENT_API_KEY`` is optional.

    Raises:
        ValueError: If an unsupported provider is given, or if
            ``openai-compatible`` is used without ``AIIDA_AGENT_BASE_URL``.
    """
    provider = os.getenv("AIIDA_AGENT_PROVIDER", "ollama").lower()
    model_name = os.getenv("AIIDA_AGENT_MODEL", "qwen3.5:2b")

    if provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        return OpenAIChatModel(model_name, provider=OllamaProvider(base_url=base_url))

    if provider == "openai":
        return OpenAIChatModel(model_name)  # reads OPENAI_API_KEY

    if provider == "anthropic":
        from pydantic_ai.models.anthropic import AnthropicModel

        return AnthropicModel(model_name)  # reads ANTHROPIC_API_KEY

    if provider == "openai-compatible":
        openai_base_url = os.getenv("AIIDA_AGENT_BASE_URL")
        if not openai_base_url:
            msg = (
                "AIIDA_AGENT_PROVIDER='openai-compatible' requires AIIDA_AGENT_BASE_URL "
                "(e.g. https://api.deepseek.com/v1)."
            )
            raise ValueError(msg)
        api_key = os.getenv("AIIDA_AGENT_API_KEY", "api-key-not-set")
        return OpenAIChatModel(
            model_name,
            provider=OpenAIProvider(base_url=openai_base_url, api_key=api_key),
        )

    msg = (
        f"Unsupported AIIDA_AGENT_PROVIDER {provider!r}; "
        "use 'ollama', 'openai', 'anthropic', or 'openai-compatible'."
    )
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------


def get_agent() -> Agent:
    """Build and return the AiiDA exploration agent.

    Called from ``main()`` after ``load_dotenv()`` so environment variables
    are populated before model construction.
    """
    return Agent(get_model(), tools=_TOOLS, system_prompt=SYSTEM_PROMPT)


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


async def ask(agent: Agent, question: str) -> None:  # pragma: no cover
    """Run a single query through the agent and stream the response to stdout."""
    logger.info("agent query: %s", question)
    async with agent.run_stream(question) as result:
        print("Agent: ", end="", flush=True)
        printed_len = 0
        async for chunk in result.stream_text(debounce_by=None):
            print(chunk[printed_len:], end="", flush=True)
            printed_len = len(chunk)
        print()


def main() -> None:  # pragma: no cover
    """Interactive REPL for the AiiDA agent."""
    from dotenv import load_dotenv
    from aiida import load_profile

    load_dotenv(".env")
    load_profile()

    agent = get_agent()

    provider = os.getenv("AIIDA_AGENT_PROVIDER", "ollama")
    model = os.getenv("AIIDA_AGENT_MODEL", "qwen3.5:2b")
    print(f"AiiDA Agent — {provider}:{model} — type 'quit' to exit\n")

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not question or question.lower() in ("quit", "exit", "q"):
            break
        asyncio.run(ask(agent, question))


if __name__ == "__main__":  # pragma: no cover
    main()
