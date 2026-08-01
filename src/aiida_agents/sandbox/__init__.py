"""Running model-written Python against AiiDA without risking the database.

Two pieces, layered:

:mod:`~aiida_agents.sandbox.guard`
    A static pass that refuses code which imports something it should not,
    calls something that writes, or reaches for a builtin that would step
    around either rule. Cheap, runs first, and explicitly *not* containment.

:mod:`~aiida_agents.sandbox.runner`
    Runs what survives in a subprocess, under a profile the caller nominates,
    with a timeout. The containment is that profile being backed by a
    read-only database role --- Postgres refusing the write, rather than us
    hoping the guard caught everything.

Nothing here is wired to an agent. Executing generated code is a write action
by nature and belongs behind the approval gate, which is a separate change; a
library nobody calls yet is the safe order to build this in.
"""

from __future__ import annotations

from aiida_agents.sandbox.guard import Rejection, check_code
from aiida_agents.sandbox.runner import SandboxResult, run_in_sandbox

__all__ = [
    "Rejection",
    "SandboxResult",
    "check_code",
    "run_in_sandbox",
]
