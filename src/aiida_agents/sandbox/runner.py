"""Run generated Python against AiiDA, contained.

Three layers, and only the second and third are containment:

1. :mod:`aiida_agents.sandbox.guard` refuses the obvious before anything runs.
2. The code runs in a **subprocess**, so an infinite loop is a timeout rather
   than a hung CLI, and a crash takes nothing with it.
3. That subprocess loads a **profile the caller nominates**, which is meant to
   be one pointing at the same database through a read-only Postgres role. The
   containment is Postgres refusing the write, not us noticing it --- and that
   is the only layer here a determined generation cannot talk its way past.

Layer 3 is the one that matters and the one this module cannot enforce: it can
load whichever profile it is given, and whether that profile is read-only is a
fact about the database. :func:`run_in_sandbox` therefore never guesses a
profile. A caller that passes the user's own writable profile gets exactly what
it asked for, which is why the tool layer above must not do that.

A note on what "sandbox" does not mean here. The profile is not a scratch copy
of the user's data --- an empty database cannot answer "which structures did I
relax last month", which is most of what anyone wants to ask. Same data,
refused writes.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field

from aiida_agents.sandbox.guard import Rejection, check_code

__all__ = ["SandboxResult", "run_in_sandbox"]

#: Longest a snippet may run. A query over a large provenance graph is slow;
#: an accidental ``while True`` is forever. Thirty seconds separates them
#: without making the caller wait on a mistake.
DEFAULT_TIMEOUT_SECONDS = 30.0

#: Ceiling on what a caller may ask for, so a generated argument cannot turn
#: into an unbounded wait.
MAX_TIMEOUT_SECONDS = 300.0

#: Output kept from a run. A query printing ten thousand rows would otherwise
#: land whole in a model's context and crowd out everything else.
MAX_OUTPUT_CHARS = 20_000


@dataclass(frozen=True)
class SandboxResult:
    """What running a snippet produced.

    ``ok`` is the only field worth branching on: it means the code was allowed
    to run, ran, and exited cleanly. Everything else describes why not.
    """

    ok: bool
    stdout: str = ""
    stderr: str = ""
    rejections: tuple[Rejection, ...] = field(default_factory=tuple)
    timed_out: bool = False
    duration_seconds: float = 0.0

    @property
    def refused(self) -> bool:
        """True if the guard stopped this before it ran."""
        return bool(self.rejections)

    def summary(self) -> str:
        """One paragraph a model can act on, or a user can read.

        Deliberately explicit about *which* of the three outcomes occurred:
        "refused", "timed out" and "raised" call for different next moves, and
        collapsing them into "it didn't work" costs a retry.
        """
        if self.refused:
            reasons = "\n".join(f"  - {r}" for r in self.rejections)
            return f"Refused before running:\n{reasons}"
        if self.timed_out:
            return (
                f"Timed out after {self.duration_seconds:.0f}s. The query is "
                "probably unbounded -- add a filter or a limit."
            )
        if not self.ok:
            return f"Raised:\n{self.stderr.strip()}"
        return self.stdout.strip() or "Ran cleanly, but printed nothing."


def _clip(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return text[:MAX_OUTPUT_CHARS] + f"\n... [truncated at {MAX_OUTPUT_CHARS} chars]"


def _script(code: str, profile: str | None) -> str:
    """The snippet, with the profile loaded ahead of it.

    ``load_profile`` is called by name rather than by letting AiiDA pick the
    default: the whole safety argument rests on which profile this is, and a
    default resolved inside the subprocess is one the caller never chose.
    """
    if profile is None:
        return code
    preamble = f"""\
        from aiida import load_profile
        load_profile({profile!r})
    """
    return textwrap.dedent(preamble) + "\n" + code


def run_in_sandbox(
    code: str,
    profile: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> SandboxResult:
    """Check ``code``, then run it in a subprocess under ``profile``.

    Args:
        code: The Python to run. Refused outright if
            :func:`~aiida_agents.sandbox.guard.check_code` objects to it.
        profile: AiiDA profile to load first. **Pass a read-only one**: this
            function loads what it is given and cannot tell the difference.
            ``None`` loads no profile at all, which is what a snippet touching
            no database wants.
        timeout: Seconds before the subprocess is killed, capped at
            :data:`MAX_TIMEOUT_SECONDS`.

    Returns:
        A :class:`SandboxResult`. Never raises for a failure *in* the code ---
        a traceback is a result here, and usually the useful one, since it is
        what a model needs in order to fix its own snippet.
    """
    rejections = check_code(code)
    if rejections:
        return SandboxResult(ok=False, rejections=tuple(rejections))

    limit = max(1.0, min(float(timeout), MAX_TIMEOUT_SECONDS))
    started = time.monotonic()
    try:
        completed = subprocess.run(  # noqa: S603
            [sys.executable, "-c", _script(code, profile)],
            capture_output=True,
            text=True,
            timeout=limit,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return SandboxResult(
            ok=False, timed_out=True, duration_seconds=time.monotonic() - started
        )

    return SandboxResult(
        ok=completed.returncode == 0,
        stdout=_clip(completed.stdout),
        stderr=_clip(completed.stderr),
        duration_seconds=time.monotonic() - started,
    )
