"""Execution Agent — workflow generation, validation, and submission."""

from __future__ import annotations

from importlib.resources import files
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.tools import DeferredToolRequests
from pydantic_ai.toolsets import FunctionToolset

from aiida_agents._settings import AgentSettings, ModelSettings, OllamaSettings
from aiida_agents.agents._errors import RetryOnToolError
from aiida_agents.agents._models import get_model

from aiida_agents.tools.workflows.introspection import describe_workflow, list_workflows
from aiida_agents.tools.workflows.protocol import build_workflow_inputs
from aiida_agents.tools.workflows.spec_execution import execute_workflow_spec
from aiida_agents.tools.execution.analysis_queries import query_analysis_agent

# Read-only tools: wrapped by RetryOnToolError so tool failures become ModelRetry
_READ_TOOLS: list[Any] = [
    query_analysis_agent,  # Query the AiiDA database for context (read-only)
    list_workflows,  # Discover registered workflow and calculation entry points (read-only)
    describe_workflow,  # Inspect process schema, ports, defaults, and exit codes (read-only)
    build_workflow_inputs,  # Pre-populate inputs from a protocol builder (read-only)
]

# Load system prompt
_SYSTEM_PROMPT = (
    files(__package__).joinpath("prompt.md").read_text(encoding="utf-8").strip()
)


def get_agent(
    model_settings: ModelSettings | None = None,
    ollama_settings: OllamaSettings | None = None,
    agent_settings: AgentSettings | None = None,
) -> Agent:
    """Build and return the Execution Agent.

    The Execution Agent:
    1. Queries the AiiDA database for context on past runs (query_analysis_agent)
    2. Discovers installed workflow and calculation entry points (list_workflows)
    3. Introspects process schemas, required/optional ports, and exit codes (describe_workflow)
    4. Pre-populates inputs from a protocol builder when one exists (build_workflow_inputs)
    5. Submits workflow specs (execute_workflow_spec, requires HITL approval)

    All read tools are wrapped by RetryOnToolError so tool failures
    (e.g., hallucinated parameters) become recoverable retries instead of crashes.

    ``execute_workflow_spec`` is registered with ``requires_approval=True`` so the
    agent pauses for human confirmation before anything is written to the database.

    Args:
        model_settings: Model/provider config. Read from env/.env if not given.
        ollama_settings: Ollama endpoint config. Read from env/.env if not given.
        agent_settings: Agent behaviour (retry budget). Read from env/.env if not given.

    Returns:
        Agent: Ready-to-use Execution Agent instance.
    """
    cfg = agent_settings if agent_settings is not None else AgentSettings()

    # Wrap read tools with RetryOnToolError for automatic retry on failures
    toolset = RetryOnToolError(FunctionToolset(_READ_TOOLS))

    agent: Agent = Agent(
        get_model(model_settings=model_settings, ollama_settings=ollama_settings),
        toolsets=[toolset],
        retries=cfg.tool_retries,
        system_prompt=_SYSTEM_PROMPT,
        output_type=(str, DeferredToolRequests),
    )

    # execute_workflow_spec is the single HITL-gated write tool.
    # It delegates to submit_workflow internally, so submit_workflow is NOT
    # registered separately — doing so would expose it twice and confuse the model.
    agent.tool_plain(requires_approval=True)(execute_workflow_spec)

    return agent
