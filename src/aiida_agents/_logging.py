"""Trace logging for agent tool calls and replies.

The trace loggers record what the agent did (tool calls, tool returns, final
replies) as DEBUG records. They are ordinary propagating loggers: whether the
records surface anywhere is decided entirely by the handlers the entry point
installs. ``_configure_logging`` (below) writes them to the optional log file
and keeps them out of the console handler via ``TRACE_LOGGER_NAMES``; how the
same events are *rendered* on screen is the REPL's concern, not this module's.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, TypeAlias

from pydantic_ai.messages import ToolCallPart, ToolReturnPart

if TYPE_CHECKING:
    from aiida_agents._settings import LoggingSettings

file_logger = logging.getLogger("aiida_agents.file_debug")
file_logger.propagate = True
file_logger.setLevel(logging.DEBUG)

responses_logger = logging.getLogger("aiida_agents.responses_debug")
responses_logger.propagate = True
responses_logger.setLevel(logging.DEBUG)

# For handler-side filtering (keeping trace records out of a console handler)
# without consumers hardcoding the logger names.
TRACE_LOGGER_NAMES: tuple[str, ...] = (file_logger.name, responses_logger.name)

# Not a ``type`` statement alias: the union object doubles as an isinstance
# check at the call sites, and isinstance() rejects TypeAliasType.
ToolPart: TypeAlias = ToolCallPart | ToolReturnPart


def trace_tool_part(part: ToolPart) -> None:
    """Record one tool call/return to the trace log."""
    if isinstance(part, ToolCallPart):
        file_logger.debug(
            "→ TOOL CALLED: %s with args %s (id: %s)",
            part.tool_name,
            part.args,
            part.tool_call_id,
        )
    else:
        file_logger.debug("← TOOL RETURNED: %s -> %s", part.tool_name, part.content)


def trace_response(text: str) -> None:
    """Record the agent's final reply to the trace log."""
    responses_logger.debug("Agent Response: %s", text)


def suppress_noisy_loggers() -> None:
    """Reduce verbosity of third-party debug logs."""
    # httpx logs one INFO line per HTTP request (unlike httpcore, which is
    # only chatty at DEBUG), so it must be in this list to keep per-turn
    # request lines out of the REPL.
    noisy = ["asyncio", "httpcore", "httpx", "openai", "chromadb", "markdown_it"]
    for logger_name in noisy:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


class _ConsoleFilter(logging.Filter):
    """Keep trace records (tool calls, agent replies) out of the console handler."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not record.name.startswith(TRACE_LOGGER_NAMES)


def _configure_logging(log_cfg: LoggingSettings) -> None:
    """Configure console (and optional file) logging for any entry point.

    Shared by the REPL and the MCP server so file logging behaves the same on
    both. ``force=True`` removes and closes any pre-existing root handlers. The
    console handler filters out trace records; the file handler, when enabled,
    receives everything (console records plus the full tool-call/agent-reply
    traces, independent of ``log_level``).
    """
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.addFilter(_ConsoleFilter())
    # The console honours log_level directly. The default (WARNING) keeps an
    # interactive session uncluttered; INFO-level operational logs would both
    # duplicate the rich output and collide with its live regions, so raise the
    # level explicitly to surface them. The file handler, when enabled, still
    # captures the full tool-call/agent-reply traces regardless: the trace
    # loggers emit at DEBUG independently of this level.
    console_handler.setLevel(log_cfg.log_level)
    handlers: list[logging.Handler] = [console_handler]
    if log_cfg.log_file is not None:
        # Create the parent dir when it doesn't exist yet, so a log_file in a
        # new directory enables file logging rather than crashing the entry
        # point with FileNotFoundError. (A log_file that is itself a directory,
        # or an unwritable path, still raises: those are fatal misconfigs.)
        log_cfg.log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_cfg.log_file))

    logging.basicConfig(
        level=log_cfg.log_level,
        format="%(levelname)s:%(name)s:%(message)s",
        handlers=handlers,
        force=True,
    )

    # Unconditional: third-party request/debug chatter (one httpx INFO line
    # per model call, for instance) drowns the conversation at any level.
    suppress_noisy_loggers()
