"""Read-only tools owned by the Analysis agent.

Process inspection (``get_process_status``, ``list_processes``) deliberately
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
"""

from __future__ import annotations

from aiida_agents.tools.analysis.nodes import get_node_inputs, get_node_outputs
from aiida_agents.tools.analysis.query_builder import query_nodes
from aiida_agents.tools.analysis.structures import search_structures

__all__ = [
    "get_node_inputs",
    "get_node_outputs",
    "query_nodes",
    "search_structures",
]
