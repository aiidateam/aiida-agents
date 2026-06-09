"""AiiDA agent subpackage.

Public API
----------
get_agent()
    Build and return the active agent (currently the Analysis agent).
"""

from __future__ import annotations

from aiida_agents.agents.analysis import get_agent

__all__ = ["get_agent"]
