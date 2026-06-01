"""AiiDA Agents MCP Server."""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from aiida import load_profile
from aiida.manage import get_manager
from fastmcp import FastMCP
from aiida_agents.mcp.tools import register_all

logging.basicConfig(
    level=os.getenv("AIIDA_AGENTS_LOG_LEVEL", "INFO"), stream=sys.stderr
)


@asynccontextmanager
async def _lifespan(_: FastMCP) -> AsyncGenerator[None]:
    """Load the default AiiDA profile when the server starts.

    Runs on every launch path (``main()``, ``fastmcp run``, ``fastmcp dev``),
    unlike a ``main()``-only load. Skipped if a profile is already loaded (e.g.
    the test fixture's), so it never clobbers an active profile.
    """
    if get_manager().get_profile() is None:
        load_profile()
    yield


def get_mcp() -> FastMCP:
    """Build the MCP server and register its tools. No profile needed here."""
    mcp = FastMCP(
        name="aiida-agents",
        instructions="Tools for exploring an AiiDA database using natural language.",
        lifespan=_lifespan,
    )
    register_all(mcp)
    return mcp


mcp = get_mcp()


def main() -> None:  # pragma: no cover
    """Run the MCP server (the profile is loaded by the lifespan on startup)."""
    port = int(os.getenv("AIIDA_AGENTS_PORT", "8000"))
    mcp.run(transport="streamable-http", port=port)


if __name__ == "__main__":  # pragma: no cover
    main()
