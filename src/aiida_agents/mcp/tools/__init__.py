"""Register the surface-agnostic tool functions onto a fastmcp server.

Each read tool is wrapped with the adapter from ``aiida_agents.mcp._errors`` so it
cannot reach the client with an uncaught ``AiidaException``.
"""

from __future__ import annotations

from collections.abc import Callable
from aiida_agents.rag import search_aiida_docs
from aiida_agents.plugins import discover_plugins
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

    builtin_tools = (
        # Analysis tools
        get_process_status,
        get_process_report,
        diagnose_process_failure,
        get_daemon_status,
        list_recent_processes,
        list_retrieved_files,
        get_retrieved_file,
        get_node_inputs,
        get_node_outputs,
        query_nodes,
        search_structures,
        # Execution tools (read-only half: discovery and input building)
        list_process_entry_points,
        describe_process,
        build_process_inputs,
        draft_process_inputs,
        check_cutoffs_against_pseudos,
        build_resubmission_spec,
        list_codes,
        query_run_context,
        wait_for_process,
        # Documentation search (RAG)
        search_aiida_docs,
    )
    for func in builtin_tools:
        register_tool(mcp, func)
    # Keep built-in tools authoritative if a plugin uses the same name.
    registered_names = {func.__name__ for func in builtin_tools}
    # Register plugin-contributed read-only tools.
    for plugin in discover_plugins():
        for tool in plugin.tools:
            if tool.writes:
                continue

            name = tool.fn.__name__
            if name in registered_names:
                continue

            register_tool(mcp, tool.fn)
            registered_names.add(name)
