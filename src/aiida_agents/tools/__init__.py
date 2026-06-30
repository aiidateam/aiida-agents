"""Surface-agnostic AiiDA tools, shared by the MCP server and the agent.

Read tools are re-exported flat (``from aiida_agents.tools import query_nodes``);
the write tool ``submit_workflow`` stays an explicit ``tools.submit`` import, so
the database-writing tool is not grabbed as casually as a read (HITL-gated, ADR-08).
"""

from __future__ import annotations

from aiida_agents.tools.nodes import get_node_inputs, get_node_outputs, query_nodes
from aiida_agents.tools.processes import get_process_status, list_processes
from aiida_agents.tools.structures import search_structures

__all__ = [
    "get_node_inputs",
    "get_node_outputs",
    "get_process_status",
    "list_processes",
    "query_nodes",
    "search_structures",
]
