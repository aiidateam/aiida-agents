"""Execution Agent — workflow generation, validation, and submission.

FIXED VERSION:
- All tools properly imported and wrapped
- Uses existing submit_workflow from aiida_agents.tools.submit
- Complete tool list with query_analysis_agent included
"""

from __future__ import annotations

from importlib.resources import files
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.tools import DeferredToolRequests
from pydantic_ai.toolsets import FunctionToolset

from aiida_agents._settings import AgentSettings, ModelSettings, OllamaSettings
from aiida_agents.agents._errors import RetryOnToolError
from aiida_agents.agents._models import get_model

# Import all Execution Agent tools
from aiida_agents.tools.workflows.retrieval import get_workflow_templates
from aiida_agents.tools.workflows.spec_execution import execute_workflow_spec
from aiida_agents.tools.workflows.spec_generation import generate_workflow_spec
from aiida_agents.tools.workflows.spec_validation import validate_workflow_spec
from aiida_agents.tools.execution.analysis_queries import query_analysis_agent

# Reuse existing submit tool from Analysis Agent
from aiida_agents.tools.submit import submit_workflow

logger_setup = True  # Placeholder for logging setup if needed

# Read-only tools: wrapped by RetryOnToolError so tool failures become ModelRetry
# All four are read-only or query-only, not write operations
_READ_TOOLS: list[Any] = [
    query_analysis_agent,  # Query Analysis Agent for context (read-only)
    get_workflow_templates,  # Retrieve workflow templates and parameter schemas (read-only)
    generate_workflow_spec,  # Generate spec (read-only, no AiiDA writes)
    validate_workflow_spec,  # Validate spec (read-only, no AiiDA writes)
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
    1. Queries Analysis Agent for context on past runs (query_analysis_agent)
    2. Inspects workflow schemas and parameter bounds (get_workflow_templates)
    3. Generates workflow specs (generate_workflow_spec)
    4. Validates specs (validate_workflow_spec)
    5. Submits validated specs (execute_workflow_spec or submit_workflow, requires HITL approval)

    All read tools are wrapped by RetryOnToolError so tool failures
    (e.g., hallucinated parameters) become recoverable retries instead of crashes.

    submit_workflow and execute_workflow_spec are registered with requires_approval=True
    so the agent pauses for human confirmation before executing.

    Args:
        model_settings: Model/provider config. Read from env/.env if not given.
        ollama_settings: Ollama endpoint config. Read from env/.env if not given.
        agent_settings: Agent behavior (retry budget). Read from env/.env if not given.

    Returns:
        Agent: Ready-to-use Execution Agent instance.
    """
    cfg = agent_settings if agent_settings is not None else AgentSettings()

    # Wrap read tools with RetryOnToolError for automatic retry on failures
    toolset = RetryOnToolError(FunctionToolset(_READ_TOOLS))

    # Build agent with wrapped tools
    agent: Agent = Agent(
        get_model(model_settings=model_settings, ollama_settings=ollama_settings),
        toolsets=[toolset],
        retries=cfg.tool_retries,
        system_prompt=_SYSTEM_PROMPT,
        output_type=(str, DeferredToolRequests),
    )

    # Register the write tools with HITL approval required
    # This pauses the agent run and waits for human confirmation before execution
    agent.tool_plain(requires_approval=True)(submit_workflow)
    agent.tool_plain(requires_approval=True)(execute_workflow_spec)

    return agent
