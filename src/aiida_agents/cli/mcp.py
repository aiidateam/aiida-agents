"""``aiida-agents mcp`` command group: run the MCP server from the CLI."""

from __future__ import annotations

import rich_click as click

from aiida_agents.cli._guards import _needs_recognized_settings


@click.group("mcp")
def mcp() -> None:
    """Run the Model Context Protocol (MCP) server."""


@mcp.command("serve")
@click.option(
    "--port",
    type=int,
    default=None,
    help="Port to serve on (overrides AIIDA_AGENTS_PORT).",
)
@click.pass_context
@_needs_recognized_settings
def mcp_serve(ctx: click.Context, port: int | None) -> None:  # pragma: no cover
    """Start the MCP server (streamable-http transport).

    The same server as ``python -m aiida_agents.mcp.server``, surfaced here so
    both entry points live under one command. Its lifespan loads the AiiDA
    profile and configures logging.
    """
    from aiida import load_profile

    from aiida_agents._settings import ServerSettings
    from aiida_agents.mcp import server

    # Honour the global --profile: load it here so the server's lifespan (which
    # only loads a profile when none is active) serves this one, not the default.
    load_profile(ctx.obj["profile"])
    resolved = port if port is not None else ServerSettings().port
    click.echo(f"Starting MCP server on port {resolved} (Ctrl-C to stop) ...")
    server.mcp.run(transport="streamable-http", port=resolved)
