"""Tools owned by the Codegen agent: running the Python it just wrote.

One tool, and the interesting thing about it is what it is *not*. It is not
approval-gated, because nothing it does to the profile reaches the user's own:
it runs against a disposable copy of their storage, in a subprocess, behind a
static guard (see :mod:`aiida_agents.sandbox`). That containment covers the
AiiDA storage and stops there --- the filesystem and the network the subprocess
sees are the real ones.

That is the point of the whole arrangement. An approval prompt showing twenty
lines of unexecuted Python asks the user to review something they cannot
usefully judge under time pressure. Letting the code run where it can do no
harm, and showing them what it *actually returned*, turns the same decision
into one they can make from evidence --- and lets the agent fix its own
mistakes from a real traceback before the user is ever shown them.
"""

from __future__ import annotations

from aiida_agents.tools.codegen.execution import run_python_snippet

__all__ = ["run_python_snippet"]
