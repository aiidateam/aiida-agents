"""AiiDA Agents MCP Server."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from aiida import load_profile
from aiida.common.exceptions import AiidaException
from aiida.manage import get_manager
from fastmcp import FastMCP

from aiida_agents._logging import _configure_logging
from aiida_agents._profile import open_profile_storage
from aiida_agents._settings import (
    LoggingSettings,
    ServerSettings,
    warn_on_unrecognized_settings,
)
from aiida_agents.mcp.tools import register_all

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(_: FastMCP) -> AsyncGenerator[None]:
    """Configure logging, load the default AiiDA profile and open its storage.

    Runs on every launch path (``main()``, ``fastmcp run``, ``fastmcp dev``),
    unlike a ``main()``-only setup. Logging is configured here rather than at
    import time so importing the module stays side-effect-free.
    ``_configure_logging`` applies the REPL's policy: it resets the root
    handlers (``force=True``) and quiets the noisy third-party loggers, so even
    at ``DEBUG`` the server no longer surfaces per-request httpx/httpcore lines.
    The profile load is skipped if one is already active (e.g. the test
    fixture's), so it never clobbers a loaded profile.
    """
    _configure_logging(LoggingSettings())
    warn_on_unrecognized_settings()
    if get_manager().get_profile() is None:
        load_profile()
    try:
        # The tools run on worker threads, so the storage has to be open before
        # the first request does; see ``open_profile_storage``.
        open_profile_storage()
    except AiidaException as exc:
        # Refusing to start beats serving tools that all fail the same way. Log
        # the message first: AiiDA's names the command that fixes it, and the
        # startup traceback would otherwise bury it. Stripped so the record's
        # first line carries the failure -- AiiDA's open with a blank one.
        logger.error("cannot open the profile storage: %s", str(exc).strip())
        raise
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
    """Run the MCP server (logging and profile are set up by the lifespan)."""
    mcp.run(transport="streamable-http", port=ServerSettings().port)


if __name__ == "__main__":  # pragma: no cover
    main()
