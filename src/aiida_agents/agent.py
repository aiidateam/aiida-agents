"""AiiDA exploration agent using Pydantic AI."""

from __future__ import annotations
import asyncio
import os
from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel

# Import our refactored MCP tool functions directly
from aiida_agents.mcp.tools.processes import get_process_status, list_processes
from aiida_agents.mcp.tools.nodes import query_nodes, get_node_inputs, get_node_outputs
from aiida_agents.mcp.tools.structures import search_structures

# Automatically set OLLAMA_BASE_URL to point to Windows/WSL localhost if not set
if "OLLAMA_BASE_URL" not in os.environ:
    os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434/v1"


def get_model() -> Model:
    """Get the configured AI model dynamically from environment variables.

    This ensures the agent is fully model-agnostic and modular:
    1. By default, it runs your lightweight local 'qwen3.5:2b' on your 8GB laptop.
    2. In production/servers, the AiiDA team can point it to cloud models or larger local
       models simply by setting environment variables in their shell.
    """
    # Allowed providers: 'ollama', 'openai', 'anthropic', 'gemini'
    model_provider = os.getenv("AIIDA_AGENT_PROVIDER", "ollama").lower()
    model_name = os.getenv("AIIDA_AGENT_MODEL", "qwen3.5:2b")

    if model_provider == "ollama":
        # Use Pydantic AI's native OpenAI-compatible Ollama provider
        return OpenAIChatModel(
            model_name=model_name,
            provider="ollama",
        )
    elif model_provider == "openai":
        # Standard OpenAI cloud model (looks up OPENAI_API_KEY environment variable)
        return OpenAIChatModel(model_name=model_name)
    else:
        # Fallback to Pydantic AI's dynamic inference (handles 'anthropic:claude-3-5-sonnet', etc.)
        from pydantic_ai.models import infer_model

        return infer_model(f"{model_provider}:{model_name}")


# Instantiate our modular model
model = get_model()

# Create the AiiDA Agent with our local tools!
agent = Agent(
    model,
    tools=[
        get_process_status,
        list_processes,
        query_nodes,
        get_node_inputs,
        get_node_outputs,
        search_structures,
    ],
    system_prompt=(
        "You are an expert assistant for the AiiDA (Automated Interactive Infrastructure "
        "for Database Applications) materials science database. You help materials scientists "
        "explore their calculations, structures, and process provenance records by querying "
        "the database provenance graph. Always use the available tools to fetch real data "
        "before answering. Be concise, precise, and professional."
    ),
)


async def ask(question: str) -> None:
    """Run a user query through the agent and stream the response in real-time."""
    print("Agent is thinking and querying tools...\n")
    try:
        # Use run_stream to stream the response as it is being generated
        async with agent.run_stream(question) as result:
            print("🤖 Agent: ", end="", flush=True)
            printed_len = 0
            async for text in result.stream_text(debounce_by=None):
                new_text = text[printed_len:]
                print(new_text, end="", flush=True)
                printed_len = len(text)
            print("\n")
    except Exception as e:
        print(f"\n❌ Error running agent: {e}\n")


def main() -> None:
    """Interactive loop to talk to the AiiDA Agent."""
    print("=" * 60)
    print("AiiDA Exploration Agent (Pydantic AI) - Ready!")
    print(
        f"Active Model: {os.getenv('AIIDA_AGENT_PROVIDER', 'ollama')}:{os.getenv('AIIDA_AGENT_MODEL', 'qwen3.5:2b')}"
    )
    print("Type your question or 'quit'/'exit' to exit.")
    print("=" * 60 + "\n")

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not question:
            continue

        asyncio.run(ask(question))


if __name__ == "__main__":
    from aiida import load_profile

    # Load the default active AiiDA database profile inside the process
    load_profile()
    main()
