"""AiiDA agent subpackage.

Public API
----------
get_agent(agent_type: str = "analysis", **kwargs)
    Build and return the active agent ("analysis" or "execution").
"""

from __future__ import annotations

import typing as t

from pydantic_ai import Agent

from aiida_agents.agents.analysis import get_agent as get_analysis_agent
from aiida_agents.agents.execution import get_agent as get_execution_agent


def get_agent(agent_type: str = "analysis", **kwargs: t.Any) -> Agent:
    """Build and return the specified AiiDA agent (`"analysis"` or `"execution"`)."""
    if agent_type.lower() == "execution":
        return get_execution_agent(**kwargs)
    elif agent_type.lower() == "analysis":
        return get_analysis_agent(**kwargs)
    else:
        msg = f"Unknown agent_type: {agent_type!r}. Supported options: 'analysis', 'execution'"
        raise ValueError(msg)


__all__ = ["get_agent", "get_analysis_agent", "get_execution_agent"]
