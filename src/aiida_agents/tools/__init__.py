"""Surface-agnostic AiiDA tools, shared by the MCP server and the agents.

Tools are grouped by the agent that owns them, mirroring ``agents/``: each
agent's tools live in ``tools/<agent>/``, and anything shared by more than one
agent stays at this top level (``_orm``, ``_types``, ``_errors``). A new agent
adds a new sibling package here rather than more flat modules.

Read tools are re-exported flat (``from aiida_agents.tools import query_nodes``)
for convenience; the write tool ``submit_workflow`` stays an explicit
``tools.execution.submit`` import, so the database-writing tool is not grabbed as
casually as a read (HITL-gated, ADR-08).

Public API
----------
get_process_status, get_process_report, list_processes
    Process inspection, shared by both agents (``tools/processes.py``): the
    Analysis agent reports on runs and explains failures, the Execution agent
    follows up on what it just submitted.
query_nodes, get_node_inputs, get_node_outputs, search_structures
    Analysis agent's read-only AiiDA query tools (``tools/analysis/``), safe to
    register on an agent or expose over MCP.
list_workflows, describe_workflow, build_workflow_inputs,
draft_workflow_inputs, list_codes, query_run_context
    Execution agent's read-only tools (``tools/execution/``): entry-point and
    ``Process.spec()`` introspection, input pre-population from a protocol
    builder or (for a process without one) from the spec itself,
    installed-code discovery, and historical context from the Analysis agent.
    The write path (``execute_workflow_spec``/``submit_workflow``) is
    intentionally not re-exported here.
"""

from __future__ import annotations

from aiida_agents.tools.analysis import (
    get_node_inputs,
    get_node_outputs,
    query_nodes,
    search_structures,
)
from aiida_agents.tools.processes import (
    get_process_report,
    get_retrieved_file,
    get_process_status,
    list_processes,
    list_retrieved_files,
)
from aiida_agents.tools.execution import (
    build_workflow_inputs,
    describe_workflow,
    draft_workflow_inputs,
    list_workflows,
    query_run_context,
)
from aiida_agents.tools.execution.codes import list_codes

__all__ = [
    "build_workflow_inputs",
    "describe_workflow",
    "draft_workflow_inputs",
    "get_node_inputs",
    "get_node_outputs",
    "get_process_report",
    "get_retrieved_file",
    "get_process_status",
    "list_codes",
    "list_processes",
    "list_retrieved_files",
    "list_workflows",
    "query_run_context",
    "query_nodes",
    "search_structures",
]
