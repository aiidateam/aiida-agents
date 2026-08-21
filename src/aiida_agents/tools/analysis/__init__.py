"""Read-only tools owned by the Analysis agent.

Process inspection (``get_process_status``, ``list_recent_processes``) deliberately
does *not* live here: the Execution agent needs it too, to follow up on what it
submitted, so it sits at the shared ``tools`` top level (see that package's
docstring for the rule).

Public API
----------
query_nodes
    Generic node-query tool: a structured spec lowered to AiiDA's QueryBuilder.
get_node_inputs, get_node_outputs
    Provenance links (incoming/outgoing) for a node.
search_structures
    Crystal structure search by formula/elements.
diagnose_process_failure
    Why a process failed: the calculation that actually broke, what its exit
    code means, and which of the workflow's own remedies were tried or remain.
"""

from __future__ import annotations

from aiida_agents.tools.analysis.diagnostics import diagnose_process_failure
from aiida_agents.tools.analysis.nodes import get_node_inputs, get_node_outputs
from aiida_agents.tools.analysis.query_builder import query_nodes
from aiida_agents.tools.analysis.structures import search_structures

__all__ = [
    "diagnose_process_failure",
    "get_node_inputs",
    "get_node_outputs",
    "query_nodes",
    "search_structures",
]
