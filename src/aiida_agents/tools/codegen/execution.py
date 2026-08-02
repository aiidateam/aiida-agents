"""Run generated Python against the read-only sandbox profile."""

from __future__ import annotations

import logging

from aiida_agents._settings import SandboxSettings

logger = logging.getLogger(__name__)

__all__ = ["run_aiida_code"]

_NOT_CONFIGURED = (
    "No sandbox profile is configured, so this code cannot be run. Tell the "
    "user to run `aiida-agents sandbox init` once to create a read-only "
    "profile, then `aiida-agents sandbox check` to confirm it. Do NOT run the "
    "code any other way, and do NOT claim to have run it: show them the "
    "snippet and say it is unverified."
)


def run_aiida_code(code: str) -> str:
    """Run Python against the user's AiiDA data and return what it printed.

    Use this to **check the code you just wrote before showing it to the
    user**. It runs against their real database, so the output is a real
    answer --- and if the snippet is wrong you get the traceback instead of
    them getting a broken snippet.

    Always ``print()`` what you want to see; a bare expression on the last line
    produces nothing, exactly as in a script.

    The profile this runs against **cannot write**. Its database role has no
    INSERT privilege, so anything that stores, deletes or submits will be
    refused --- by Postgres, not by politeness. Do not attempt writes here: to
    submit a workflow or import a structure, say so and let the Execution agent
    do it behind its approval prompt.

    If the code is refused or raises, read the reason, fix the snippet and try
    again. Show the user code that has run, not code you hope works.

    Args:
        code: Python to run. Imports are restricted to ``aiida`` and a small
            set of standard-library modules.

    Returns:
        What the code printed, or the reason it was refused, timed out or
        raised.
    """
    from aiida_agents.sandbox import run_in_sandbox
    from aiida_agents.sandbox.setup import sandbox_profile_exists

    settings = SandboxSettings()
    profile = settings.sandbox_profile

    # Checked before running rather than letting ``load_profile`` fail inside
    # the subprocess, so an unset sandbox does not reach the model as a
    # traceback it will try to debug.
    if not sandbox_profile_exists(profile):
        logger.warning("sandbox profile %r is not configured", profile)
        return _NOT_CONFIGURED

    result = run_in_sandbox(code, profile=profile, timeout=settings.sandbox_timeout)
    logger.info(
        "sandbox run: ok=%s refused=%s timed_out=%s in %.1fs",
        result.ok,
        result.refused,
        result.timed_out,
        result.duration_seconds,
    )
    return result.summary()
