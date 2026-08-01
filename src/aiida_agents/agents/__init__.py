"""AiiDA agent subpackage.

Public API
----------
get_agent(agent_type: str = "analysis", **kwargs)
    Build and return the active agent ("analysis", "execution" or "codegen").
"""

from __future__ import annotations

import typing as t

from pydantic_ai import Agent

from aiida_agents.agents.analysis import get_agent as get_analysis_agent
from aiida_agents.agents.codegen import get_agent as get_codegen_agent
from aiida_agents.agents.execution import get_agent as get_execution_agent


def get_agent(agent_type: str = "analysis", **kwargs: t.Any) -> Agent:
    """Build and return the specified AiiDA agent.

    One of ``"analysis"``, ``"execution"`` or ``"codegen"``.
    """
    builders = {
        "analysis": get_analysis_agent,
        "execution": get_execution_agent,
        "codegen": get_codegen_agent,
    }
    builder = builders.get(agent_type.lower())
    if builder is None:
        supported = ", ".join(repr(name) for name in builders)
        msg = f"Unknown agent_type: {agent_type!r}. Supported options: {supported}"
        raise ValueError(msg)
    return builder(**kwargs)


__all__ = [
    "get_agent",
    "get_analysis_agent",
    "get_codegen_agent",
    "get_execution_agent",
]
