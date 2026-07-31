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

from aiida_agents.tools.run_context import query_run_context
from aiida_agents.tools.execution.codes import list_codes
from aiida_agents.tools.execution.introspection import describe_workflow, list_workflows
from aiida_agents.tools.execution.drafting import draft_workflow_inputs
from aiida_agents.tools.execution.protocol import build_workflow_inputs
from aiida_agents.tools.execution.ranges import check_input_ranges
from aiida_agents.tools.execution.spec_execution import execute_workflow_spec
from aiida_agents.tools.execution.structures import import_structure
from aiida_agents.tools.execution.waiting import wait_for_process
from aiida_agents.tools.processes import get_process_status
from aiida_agents.rag import search_aiida_docs

# Read-only tools: wrapped by RetryOnToolError so tool failures become ModelRetry
_READ_TOOLS: list[Any] = [
    query_run_context,  # Query the AiiDA database for context (read-only)
    list_workflows,  # Discover registered workflow and calculation entry points (read-only)
    describe_workflow,  # Inspect process schema, ports, defaults, and exit codes (read-only)
    build_workflow_inputs,  # Pre-populate inputs from a protocol builder (read-only)
    draft_workflow_inputs,  # Draft inputs from the process spec, when it has no protocol builder (read-only)
    check_input_ranges,  # Compare a spec's cutoffs against its pseudopotentials (read-only)
    list_codes,  # Discover the configured codes to submit against (read-only)
    get_process_status,  # Follow up on what was just submitted (read-only)
    wait_for_process,  # Wait for a submission to finish, to run the next one on its output (read-only)
    search_aiida_docs,  # Look up how a workflow/input actually works (read-only)
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
    1. Queries the AiiDA database for context on past runs (query_run_context)
    2. Discovers installed workflow and calculation entry points (list_workflows)
    3. Introspects process schemas, required/optional ports, and exit codes (describe_workflow)
    4. Pre-populates inputs from a protocol builder when one exists (build_workflow_inputs),
       and drafts them from the process spec itself when one does not (draft_workflow_inputs)
    5. Discovers the configured codes a calculation can run on (list_codes)
    6. Submits workflow specs (execute_workflow_spec, requires HITL approval)
    7. Follows up on what it submitted (get_process_status)

    Step 4's two tools are a pair, not alternatives to weigh: a protocol builder
    carries physics defaults tuned on real runs and is always preferred, but only
    a minority of processes ship one. For the rest --- most calculations, and any
    plugin that never adopted the convention --- ``draft_workflow_inputs`` reads
    the same ``Process.spec()`` that validates the submission, so the agent never
    has to reconstruct a port tree from prose.

    It can also read the AiiDA (and any installed plugin's) documentation
    via search_aiida_docs. describe_workflow gives a workflow's input
    *schema*; the docs are what explain what those inputs mean, which is
    exactly what configuring a real simulation needs.

    It also imports a structure file into the profile (import_structure), for
    the common case where the structure to run on is a CIF/POSCAR on disk
    rather than a node that already exists. Like execute_workflow_spec, it is
    HITL-gated.

    Step 7 is why ``get_process_status`` is shared rather than Analysis-owned:
    "did the job I just started actually launch?" is part of submitting, and
    routing a single pk lookup through ``query_run_context`` would spend a
    whole extra agent run on it.

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

    # Both write tools are HITL-gated (ADR-08). execute_workflow_spec delegates
    # to submit_workflow internally, so submit_workflow is NOT registered
    # separately — doing so would expose it twice and confuse the model.
    # import_structure writes a single StructureData; it is gated too, so a file
    # read off the user's disk still needs their explicit approval.
    agent.tool_plain(requires_approval=True)(execute_workflow_spec)
    agent.tool_plain(requires_approval=True)(import_structure)

    return agent
