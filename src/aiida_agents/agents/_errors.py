"""Agent-side adaptation of tool failures into recoverable retries.

On the pydantic-ai agent surface, any exception other than ``ModelRetry`` aborts
the whole run. The shared tool functions (``aiida_agents.mcp.tools``) instead let
exceptions propagate, and small local models routinely call tools with
hallucinated or wrong-type identifiers. ``RetryOnToolError`` therefore wraps the
agent's toolset and converts *any* tool failure into a ``ModelRetry`` carrying
recovery guidance, so the model can correct itself instead of crashing the run.

This lives in the agent layer, not ``aiida_agents.mcp``: it depends on
pydantic-ai, which the MCP server surface must not import. It reuses
``describe_aiida_error`` so an AiiDA failure reads the same on both surfaces.
"""

from __future__ import annotations

import logging
from typing import Any

from aiida.common.exceptions import AiidaException
from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.toolsets import ToolsetTool, WrapperToolset

from aiida_agents.tools._errors import describe_aiida_error

logger = logging.getLogger(__name__)

_RECHECK_HINT = "Re-check the arguments, or try a different tool."


class RetryOnToolError(WrapperToolset[Any]):
    """Convert any tool exception into a ``ModelRetry`` so the agent can recover.

    Wrapping the toolset covers every tool at the single ``call_tool`` boundary, so
    a newly added tool is protected automatically with no per-tool decoration to
    forget (the manual, per-surface wrapping it replaces is what let a bad
    identifier crash the run in the first place). A ``ModelRetry`` raised by a tool
    itself passes through unchanged; any other exception is logged and re-raised as
    a ``ModelRetry`` whose message tells the model how to recover.
    """

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[Any],
        tool: ToolsetTool[Any],
    ) -> Any:
        try:
            return await super().call_tool(
                name=name, tool_args=tool_args, ctx=ctx, tool=tool
            )
        except ModelRetry:
            # A tool asked for a retry deliberately; pass its message through.
            raise
        except AiidaException as exc:
            logger.info(
                "tool %r raised %s; asking the model to retry", name, type(exc).__name__
            )
            msg = describe_aiida_error(exc)
            raise ModelRetry(msg) from exc
        except Exception as exc:
            # Deliberately broad: this is the agent's recover-or-crash boundary.
            # pydantic-ai treats anything but ModelRetry as fatal, and small models
            # pass bad or wrong-type arguments routinely (e.g. a data-node pk to
            # get_process_status, which raises AttributeError, not an
            # AiidaException). Converting every failure into a retry keeps the run
            # alive; pydantic-ai bounds the attempts via ``Agent(retries=...)``.
            # Logged at exception level so genuine tool bugs stay visible in logs.
            logger.exception("tool %r failed; asking the model to retry", name)
            msg = f"{exc} {_RECHECK_HINT}"
            raise ModelRetry(msg) from exc
