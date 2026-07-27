"""Analysis agent — read-only exploration of an AiiDA profile.

This is the first concrete agent (ADR-04,
docs/adr/04-multi-agent-architecture.md). It exposes read-only tools for
querying processes, nodes, and crystal structures, plus documentation search.

It holds no write tool: submitting is the Execution agent's responsibility
(``agents.execution``), which reaches the database through its own HITL-gated
``execute_workflow_spec`` after discovering and validating a spec. An installed
plugin may still contribute a write tool, which is registered here behind the
same approval gate (ADR-08).
"""

from __future__ import annotations

from collections.abc import Sequence
from importlib.resources import files
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.tools import DeferredToolRequests
from pydantic_ai.toolsets import FunctionToolset

from aiida_agents._settings import AgentSettings, ModelSettings, OllamaSettings
from aiida_agents.agents._errors import RetryOnToolError
from aiida_agents.agents._models import get_model
from aiida_agents.plugins import LoadedPlugin, discover_plugins
from aiida_agents.tools import (
    get_node_inputs,
    get_node_outputs,
    get_process_report,
    get_process_status,
    get_retrieved_file,
    list_processes,
    list_retrieved_files,
    query_nodes,
    search_structures,
)
from aiida_agents.rag import search_aiida_docs

# Every read tool is exposed through RetryOnToolError (see get_agent), so a
# tool that raises -- e.g. on a hallucinated or wrong-type identifier -- comes
# back to the model as a recoverable ModelRetry instead of crashing the agent
# run. This agent registers no write tool of its own; a plugin-contributed one
# is gated with requires_approval=True (ADR-08,
# docs/adr/08-human-in-the-loop-before-writes.md) and is not part of this toolset.
_READ_TOOLS: list[Any] = [
    get_process_status,
    get_process_report,
    list_retrieved_files,
    get_retrieved_file,
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


def _system_prompt(plugins: Sequence[LoadedPlugin]) -> str:
    """The base prompt plus each plugin's domain guidance, in plugin-name order.

    A fragment is appended under its plugin's name, after the core prompt and
    clearly attributed, so the core instructions come first and the model can
    tell whose convention it is reading. Fragments are ordered and budgeted at
    discovery, so this never depends on entry-point iteration order.
    """
    fragments = [(p.name, p.prompt_fragment) for p in plugins if p.prompt_fragment]
    if not fragments:
        return _SYSTEM_PROMPT
    sections = "\n\n".join(
        f"### {name}\n\n{fragment}" for name, fragment in sorted(fragments)
    )
    return (
        f"{_SYSTEM_PROMPT}\n\n"
        "## Plugin-specific guidance\n\n"
        "The following notes come from installed plugins and apply only to their "
        "own workflows. Where they conflict with the instructions above, the "
        "instructions above win.\n\n"
        f"{sections}"
    )


def get_agent(
    model_settings: ModelSettings | None = None,
    ollama_settings: OllamaSettings | None = None,
    agent_settings: AgentSettings | None = None,
) -> Agent:
    """Build and return the Analysis agent.

    This agent is read-only: submission belongs to the Execution agent. A
    plugin-contributed write tool is still registered with
    ``requires_approval=True``, so the agent run pauses and returns a
    ``DeferredToolRequests`` object whenever the model wants to call one — the
    CLI must obtain user confirmation before re-running with
    ``DeferredToolResults``.

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

    # Installed plugins may contribute tools and domain guidance (see
    # aiida_agents.plugins). Read tools join the retry-wrapped toolset like the
    # built-ins; write tools are registered behind the approval gate below.
    plugins = discover_plugins()
    plugin_reads = [tool.fn for p in plugins for tool in p.tools if not tool.writes]
    plugin_writes = [tool.fn for p in plugins for tool in p.tools if tool.writes]
    toolset = RetryOnToolError(FunctionToolset(_READ_TOOLS + plugin_reads))

    # ``output_type=(str, DeferredToolRequests)`` makes the real type
    # ``Agent[None, str | DeferredToolRequests]``, but agents are annotated as the
    # bare ``Agent`` throughout (mypy takes that as-is; basedpyright resolves bare
    # ``Agent`` to its PEP 696 default ``Agent[None, str]`` and flags the mismatch).
    # Keep it bare until the multi-agent architecture (Orchestrator/Workflow)
    # settles how agent types are shared; this ignore is basedpyright-only.
    agent: Agent = Agent(  # pyright: ignore[reportAssignmentType]
        get_model(model_settings=model_settings, ollama_settings=ollama_settings),
        toolsets=[toolset],
        retries=cfg.tool_retries,
        system_prompt=_system_prompt(plugins),
        output_type=(str, DeferredToolRequests),
    )

    # A plugin-contributed write tool is gated the same way the Execution
    # agent's own write tool is: the CLI previews it and runs it on the main
    # thread only after the user approves (ADR-08).
    for fn in plugin_writes:
        agent.tool_plain(requires_approval=True)(fn)
    return agent
