"""Surface-agnostic AiiDA tools, shared by the MCP server and the agent.

Read tools are re-exported flat (``from aiida_agents.tools import query``);
the write tool ``submit_workflow`` stays an explicit ``tools.submit`` import, so
the database-writing tool is not grabbed as casually as a read (HITL-gated, ADR-08).

Public API
----------
query, get_node_inputs, get_node_outputs, list_processes,
get_process_status, search_structures, list_workflows,
describe_workflow, query_analysis_agent
    Read-only AiiDA query and workflow-introspection tools, safe to register on an agent
    or expose over MCP. The write tool ``submit_workflow`` is intentionally not
    re-exported here; import it from ``tools.submit`` (HITL-gated, ADR-08).
"""

from __future__ import annotations

from aiida_agents.tools.nodes import get_node_inputs, get_node_outputs
from aiida_agents.tools.processes import get_process_status, list_processes
from aiida_agents.tools.structures import search_structures
from aiida_agents.tools.querybuilder import query
from aiida_agents.tools.workflows.introspection import describe_workflow, list_workflows
from aiida_agents.tools.execution.analysis_queries import query_analysis_agent

__all__ = [
    "get_node_inputs",
    "get_node_outputs",
    "get_process_status",
    "list_processes",
    "query",
    "search_structures",
    "describe_workflow",
    "list_workflows",
    "query_analysis_agent",
]
