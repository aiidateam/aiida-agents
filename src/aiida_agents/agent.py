"""AiiDA exploration agent using Pydantic AI."""

from __future__ import annotations

import asyncio
import logging
import os

from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel

from aiida_agents.mcp.tools.nodes import get_node_inputs, get_node_outputs, query_nodes
from aiida_agents.mcp.tools.processes import get_process_status, list_processes
from aiida_agents.mcp.tools.structures import search_structures
from aiida_agents.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def _load_env(env_path: str = ".env") -> None:  # pragma: no cover
    """Load environment variables from a .env file without overwriting existing ones."""
    if not os.path.exists(env_path):
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                if key.strip() not in os.environ:
                    os.environ[key.strip()] = val.strip().strip('"').strip("'")


_load_env()

if "OLLAMA_BASE_URL" not in os.environ:  # pragma: no cover
    os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434/v1"


def get_model() -> Model:
    """Return the configured AI model based on environment variables.

    AIIDA_AGENT_PROVIDER: ollama (default), openai, or any pydantic-ai provider.
    AIIDA_AGENT_MODEL: model name, defaults to qwen3.5:2b.
    """
    provider = os.getenv("AIIDA_AGENT_PROVIDER", "ollama").lower()
    model_name = os.getenv("AIIDA_AGENT_MODEL", "qwen3.5:2b")

    if provider == "ollama":
        return OpenAIChatModel(model_name=model_name, provider="ollama")
    elif provider == "openai":  # pragma: no cover
        return OpenAIChatModel(model_name=model_name)
    else:  # pragma: no cover
        from pydantic_ai.models import infer_model
        return infer_model(f"{provider}:{model_name}")


agent = Agent(
    get_model(),
    tools=[
        get_process_status,
        list_processes,
        query_nodes,
        get_node_inputs,
        get_node_outputs,
        search_structures,
    ],
    system_prompt=SYSTEM_PROMPT,
)


async def ask(question: str) -> None:  # pragma: no cover
    """Run a query through the agent and stream the response."""
    logger.info("Running agent query: %s", question)
    async with agent.run_stream(question) as result:
        print("Agent: ", end="", flush=True)
        printed_len = 0
        async for text in result.stream_text(debounce_by=None):
            print(text[printed_len:], end="", flush=True)
            printed_len = len(text)
        print()


def main() -> None:  # pragma: no cover
    """Interactive loop for the AiiDA agent."""
    print(
        f"AiiDA Agent — {os.getenv('AIIDA_AGENT_PROVIDER', 'ollama')}:"
        f"{os.getenv('AIIDA_AGENT_MODEL', 'qwen3.5:2b')} — type 'quit' to exit\n"
    )
    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if question.lower() in ("quit", "exit", "q") or not question:
            break
        asyncio.run(ask(question))


if __name__ == "__main__":  # pragma: no cover
    from aiida import load_profile
    load_profile()
    main()