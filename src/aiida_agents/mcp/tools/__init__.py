"""Register the surface-agnostic tool functions onto a fastmcp server.

Each read tool is wrapped with the adapter from ``aiida_agents.mcp._errors`` so it
cannot reach the client with an uncaught ``AiidaException``.
"""

from __future__ import annotations

from collections.abc import Callable

from fastmcp import FastMCP

from aiida_agents.mcp._errors import to_mcp_tool_error
from aiida_agents.tools import (
    build_process_inputs,
    build_resubmission_spec,
    check_cutoffs_against_pseudos,
    describe_process,
    diagnose_process_failure,
    draft_process_inputs,
    get_daemon_status,
    get_node_inputs,
    get_node_outputs,
    get_process_report,
    get_process_status,
    get_retrieved_file,
    list_codes,
    list_process_entry_points,
    list_recent_processes,
    list_retrieved_files,
    query_run_context,
    query_nodes,
    wait_for_process,
    search_structures,
)

# The write tools are intentionally NOT registered: they reach the database, so
# they go only through the HITL-gated agents (ADR-08). ``submit_workflow`` lives
# in ``aiida_agents.tools.execution.submit`` and ``submit_process_spec`` (which
# delegates to it) in ``...execution.spec_execution``; the Execution agent
# imports the latter directly.


def register_tool(mcp: FastMCP, func: Callable[..., object]) -> None:
    """Register a tool with the AiiDA-exception adapter applied.

    Routing every tool through here keeps a newly added one from reaching the
    client with an uncaught ``AiidaException``.
    """
    mcp.tool()(to_mcp_tool_error(func))


def register_all(mcp: FastMCP) -> None:
    """Register the read-only tools on the MCP server."""
    # Analysis tools
    register_tool(mcp, get_process_status)
    register_tool(mcp, get_process_report)
    register_tool(mcp, diagnose_process_failure)
    register_tool(mcp, get_daemon_status)
    register_tool(mcp, list_recent_processes)
    register_tool(mcp, list_retrieved_files)
    register_tool(mcp, get_retrieved_file)
    register_tool(mcp, get_node_inputs)
    register_tool(mcp, get_node_outputs)
    register_tool(mcp, query_nodes)
    register_tool(mcp, search_structures)
    # Execution tools (read-only half: discovery and input building)
    register_tool(mcp, list_process_entry_points)
    register_tool(mcp, describe_process)
    register_tool(mcp, build_process_inputs)
    register_tool(mcp, draft_process_inputs)
    register_tool(mcp, check_cutoffs_against_pseudos)
    register_tool(mcp, build_resubmission_spec)
    register_tool(mcp, list_codes)
    register_tool(mcp, query_run_context)
    register_tool(mcp, wait_for_process)
