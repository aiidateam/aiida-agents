"""Shared assembly for specialist agents (analysis, execution, codegen).

Centralises the pydantic-ai ``Agent`` construction and the two invariants
every specialist must satisfy:

* Read tools are wrapped in ``RetryOnToolError`` so a tool failure becomes
  a ``ModelRetry`` the model can recover from, not a fatal crash.
* Write tools, when present, are registered with ``requires_approval=True``
  so the CLI can gate them behind human confirmation (ADR-08).

The Planner is deliberately excluded: it has no tools, uses ``instructions``
instead of ``system_prompt``, and shares nothing with this assembly.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.tools import DeferredToolRequests
from pydantic_ai.toolsets import FunctionToolset

from aiida_agents._settings import AgentSettings, ModelSettings, OllamaSettings
from aiida_agents.agents._errors import RetryOnToolError
from aiida_agents.agents._models import get_model


def build_agent(
    *,
    system_prompt: str,
    read_tools: Sequence[Any],
    write_tools: Sequence[Any] = (),
    output_type: Any = str,
    model_settings: ModelSettings | None = None,
    ollama_settings: OllamaSettings | None = None,
    agent_settings: AgentSettings | None = None,
) -> Agent:
    """Build a specialist agent from its declared parts.

    Each specialist defines *what* it needs (prompt, tools, output type);
    this function handles *how* those are assembled into a ``pydantic-ai``
    ``Agent``, so the invariants live in exactly one place.

    Args:
        system_prompt: The specialist's system prompt text.
        read_tools: Read-only tool functions, wrapped in RetryOnToolError.
        write_tools: Write tool functions, each registered with
            ``requires_approval=True``.
        output_type: The agent's output type. Defaults to ``str``.
            Agents with write tools typically pass
            ``(str, DeferredToolRequests)``.
        model_settings: Model/provider config. Read from env if not given.
        ollama_settings: Ollama endpoint config. Read from env if not given.
        agent_settings: Agent behaviour (retry budget). Read from env if
            not given.
    """
    cfg = agent_settings if agent_settings is not None else AgentSettings()

    # Wrap all read tools so failures become recoverable ModelRetry.
    toolset = RetryOnToolError(FunctionToolset(read_tools))

    agent: Agent = Agent(  # pyright: ignore[reportAssignmentType]
        get_model(model_settings=model_settings, ollama_settings=ollama_settings),
        toolsets=[toolset],
        retries=cfg.tool_retries,
        system_prompt=system_prompt,
        output_type=output_type,
    )

    # Every write tool is HITL-gated (ADR-08).
    for fn in write_tools:
        agent.tool_plain(requires_approval=True)(fn)

    return agent