"""Trace logging for agent tool calls and replies.

The trace loggers record what the agent did (tool calls, tool returns, final
replies) as DEBUG records. They are ordinary propagating loggers: whether the
records surface anywhere is decided entirely by the handlers the entry point
installs. The REPL writes them to the optional log file and keeps them out of
the console handler via ``TRACE_LOGGER_NAMES`` (see ``cli._configure_logging``);
how the same events are *rendered* on screen is the REPL's concern, not this
module's.
"""

from __future__ import annotations

import logging
from typing import TypeAlias

from pydantic_ai.messages import ToolCallPart, ToolReturnPart

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
