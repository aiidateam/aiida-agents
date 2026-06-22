"""Analysis agent — read-only exploration of an AiiDA profile.

This is the first concrete agent (ADR-04). It exposes read-only MCP tools
for querying processes, nodes, and crystal structures, and a write tool
(submit_workflow) that requires explicit human confirmation before execution.

Public API
----------
get_agent()
    Build and return a ready-to-use Analysis agent instance.
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
from aiida_agents.mcp.tools.nodes import get_node_inputs, get_node_outputs, query_nodes
from aiida_agents.mcp.tools.processes import get_process_status, list_processes
from aiida_agents.mcp.tools.structures import search_structures
from aiida_agents.mcp.tools.submit import submit_workflow
from aiida_agents.rag import search_aiida_docs

# Every read tool is exposed through RetryOnToolError (see get_agent), so a
# tool that raises -- e.g. on a hallucinated or wrong-type identifier -- comes
# back to the model as a recoverable ModelRetry instead of crashing the agent
# run. submit_workflow is registered separately with requires_approval=True
# (ADR-08) and is not part of this toolset.
_READ_TOOLS: list[Any] = [
    get_process_status,
    list_processes,
    query_nodes,
    get_node_inputs,
    get_node_outputs,
    search_structures,
    search_aiida_docs,
]

_SYSTEM_PROMPT = (
    files(__package__).joinpath("prompt.md").read_text(encoding="utf-8").strip()
)


def get_agent(
    model_settings: ModelSettings | None = None,
    ollama_settings: OllamaSettings | None = None,
    agent_settings: AgentSettings | None = None,
) -> Agent:
    """Build and return the Analysis agent.

    submit_workflow is registered with ``requires_approval=True`` so the
    agent run pauses and returns a ``DeferredToolRequests`` object whenever
    the model wants to submit — the CLI must obtain user confirmation before
    re-running with ``DeferredToolResults``.

    The read tools are wrapped once, at the toolset boundary, by
    ``RetryOnToolError``: a tool failure becomes a ``ModelRetry`` the model
    can recover from rather than a fatal error that aborts the run, bounded
    by ``tool_retries``.

    Called from the CLI after environment variables are loaded, so model
    construction always sees a fully populated environment.

    :param model_settings: Model/provider configuration, forwarded to
        ``get_model``. Read from env / ``.env`` if not given.
    :param ollama_settings: Ollama endpoint configuration, forwarded to
        ``get_model``. Read from env / ``.env`` if not given.
    :param agent_settings: Agent behaviour configuration (the per-tool retry
        budget). Read from env / ``.env`` if not given.
    """
    cfg = agent_settings if agent_settings is not None else AgentSettings()
    toolset = RetryOnToolError(FunctionToolset(_READ_TOOLS))

    agent: Agent = Agent(
        get_model(model_settings=model_settings, ollama_settings=ollama_settings),
        toolsets=[toolset],
        retries=cfg.tool_retries,
        system_prompt=_SYSTEM_PROMPT,
        output_type=(str, DeferredToolRequests),
    )

    # Register the write tool with approval required
    agent.tool_plain(requires_approval=True)(submit_workflow)
    return agent
