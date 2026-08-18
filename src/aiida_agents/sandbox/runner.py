"""Run generated Python against AiiDA, contained.

Three layers, and only the second and third are containment:

1. :mod:`aiida_agents.sandbox.guard` refuses the obvious before anything runs.
   A pre-check, not a boundary --- assume it is porous.
2. The code runs in a **subprocess**, with a scrubbed environment, resource
   limits, its own session, and no inherited credentials. An infinite loop is a
   timeout rather than a hung CLI, and a crash takes nothing with it.
3. That subprocess loads a **profile the caller nominates**, which is meant to
   be a disposable copy of the user's storage (see
   :mod:`aiida_agents.sandbox.copy`). The containment is that a write which
   gets past layer 1 lands somewhere nobody reads, not that it is refused.

Layer 3 is the one this module cannot enforce: it loads whichever profile it is
given, and whether that profile is a copy is a fact about the configuration.
:func:`run_in_sandbox` therefore never guesses a profile. A caller that passes
the user's own profile gets exactly what it asked for, which is why the tool
layer above must not do that.

What this does **not** contain:

*Writes that layer 1 misses.* On PostgreSQL a read-only role can be put under
the copy as well, and ``sandbox check`` verifies it where it is. On SQLite
there is no such role to have, so a write the guard does not recognise
succeeds against the copy. It costs the user nothing, and costs the sandbox
its accuracy until it is rebuilt, but it means layer 1's list of forbidden
names is doing more work than a pre-check should.

*Anything outside the database.* Generated code that gets past layer 1 can
still read the filesystem and reach the network. The environment scrub and the
rlimits here narrow that, but narrowing is not closing: OS-level isolation
(bwrap, nsjail, a container) is the fix, and it is not here yet.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field

from aiida_agents.sandbox.guard import Rejection, check_code

__all__ = ["SandboxResult", "run_in_sandbox"]

#: Environment variables the subprocess is allowed to see. An allowlist, not a
#: denylist: the process inherited the parent's whole environment until now,
#: which handed generated code every API key the user had exported
#: (``OPENAI_API_KEY`` and friends), plus anything else in the shell.
#:
#: A snippet's legitimate job is to query the user's AiiDA profile. That needs
#: a home directory to find ``~/.aiida``, a PATH, and an encoding. It does not
#: need credentials for a model provider.
#:
#: ``PYTHONPATH`` and ``PYTHONHOME`` are deliberately absent, and the child runs
#: under ``-I``, so the interpreter resolves its own site-packages and ignores
#: the user's site directory and ``PYTHON*`` settings.
_ENV_ALLOWLIST = (
    "AIIDA_PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "PATH",
    "TMPDIR",
    "TZ",
    "USER",
)

#: Address space ceiling for the subprocess. Generous, because a QueryBuilder
#: result set over a real provenance graph is genuinely large; the point is that
#: a runaway allocation dies instead of taking the machine's memory with it.
_MAX_ADDRESS_SPACE_BYTES = 4 * 1024**3

#: Largest file the subprocess may write. Not zero: Python writes ``__pycache__``
#: entries and libraries make temporary files, and a hard zero turns ordinary
#: imports into confusing failures. Small enough that generated code cannot fill
#: a disk.
_MAX_FILE_SIZE_BYTES = 10 * 1024**2

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


def sandbox_env() -> dict[str, str]:
    """The environment the subprocess gets: :data:`_ENV_ALLOWLIST` and nothing else.

    Built from scratch rather than by removing keys from ``os.environ``, so a
    variable nobody anticipated is absent by default instead of leaked by
    oversight --- the same reasoning as :data:`~.guard.ALLOWED_IMPORTS`.
    """
    return {name: os.environ[name] for name in _ENV_ALLOWLIST if name in os.environ}


def _apply_limits() -> None:  # pragma: no cover - runs in the forked child
    """Cap what the child may consume, between ``fork`` and ``exec``.

    POSIX only, and best-effort: a limit the platform refuses is skipped rather
    than turned into a failure to run anything at all.

    ``RLIMIT_NPROC`` is deliberately not set. It counts processes per *user*,
    not per process tree, so a value low enough to be meaningful here would
    collide with whatever else the user already has running.
    """
    import resource

    limits = (
        (resource.RLIMIT_AS, _MAX_ADDRESS_SPACE_BYTES),
        (resource.RLIMIT_FSIZE, _MAX_FILE_SIZE_BYTES),
        (resource.RLIMIT_CORE, 0),
    )
    for which, value in limits:
        try:
            soft, hard = resource.getrlimit(which)
            ceiling = value if hard == resource.RLIM_INFINITY else min(value, hard)
            resource.setrlimit(which, (ceiling, hard))
        except (ValueError, OSError):
            continue


def _terminate_group(process: subprocess.Popen[str]) -> None:
    """Kill the whole process group, not just the child we know about.

    The child is a session leader (``start_new_session``), so anything it
    spawned shares its group id. Killing only the direct child would leave a
    grandchild holding the pipes open and running past its timeout.
    """
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        process.kill()


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
    posix = os.name == "posix"

    # -I (isolated): ignore PYTHON* variables and the user site directory, and
    # keep the working directory off sys.path.
    process = subprocess.Popen(  # noqa: S603
        [sys.executable, "-I", "-c", _script(code, profile)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=sandbox_env(),
        # Its own session, so a timeout can take the whole process group with
        # it rather than orphaning whatever the snippet spawned.
        start_new_session=True,
        preexec_fn=_apply_limits if posix else None,  # noqa: PLW1509
    )
    try:
        stdout, stderr = process.communicate(timeout=limit)
    except subprocess.TimeoutExpired:
        _terminate_group(process)
        # Reap it, so the timeout path leaves no zombie behind.
        process.communicate()
        return SandboxResult(
            ok=False, timed_out=True, duration_seconds=time.monotonic() - started
        )

    return SandboxResult(
        ok=process.returncode == 0,
        stdout=_clip(stdout),
        stderr=_clip(stderr),
        duration_seconds=time.monotonic() - started,
    )
