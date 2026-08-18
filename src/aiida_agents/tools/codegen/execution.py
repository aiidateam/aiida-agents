"""Run generated Python against the read-only sandbox profile."""

from __future__ import annotations

import logging

from aiida_agents._settings import SandboxSettings

logger = logging.getLogger(__name__)

__all__ = ["run_aiida_code"]

_NOT_CONFIGURED = (
    "No sandbox profile is configured, so this code cannot be run. Tell the "
    "user to run `aiida-agents sandbox init` once to copy their storage into "
    "one, then `aiida-agents sandbox check` to confirm it. Do NOT run the "
    "code any other way, and do NOT claim to have run it: show them the "
    "snippet and say it is unverified."
)

_NOT_SEPARATE = (
    "The configured sandbox profile shares storage with another profile, so "
    "this code was not run: anything it did would land in data that is not the "
    "sandbox's. Tell the user to run `aiida-agents sandbox check`, which names "
    "the profile it overlaps with. Do NOT run the code any other way, and do "
    "NOT claim to have run it: show them the snippet and say it is unverified."
)


def run_aiida_code(code: str) -> str:
    """Run Python against a copy of the user's AiiDA data and return what it printed.

    Use this to **check the code you just wrote before showing it to the
    user**. It runs against a copy of their database, made when they set the
    sandbox up, so the output is a real answer over their real provenance ---
    and if the snippet is wrong you get the traceback instead of them getting a
    broken snippet.

    Always ``print()`` what you want to see; a bare expression on the last line
    produces nothing, exactly as in a script.

    **Nothing you do to this profile reaches the user's own.** Writes are
    refused before the code runs, and one that slips past that check lands in
    the copy, where nobody but you will ever see it. So never report having
    stored, submitted, deleted or changed anything, however plainly the output
    says you did: to change the user's data, say so and let the Execution agent
    do it behind its approval prompt.

    That covers the AiiDA profile and nothing else. The filesystem and the
    network are the user's real ones, so code that reads their files or reaches
    out over the network is not made harmless by the copy.

    If the code is refused or raises, read the reason, fix the snippet and try
    again. Show the user code that has run, not code you hope works.

    Args:
        code: Python to run. Imports are restricted to ``aiida`` and a small
            set of standard-library modules.

    Returns:
        What the code printed, or the reason it was refused, timed out or
        raised.
    """
    from aiida.manage.configuration import get_config

    from aiida_agents.sandbox import run_in_sandbox
    from aiida_agents.sandbox.copy import (
        profiles_sharing_storage,
        sandbox_profile_exists,
    )

    settings = SandboxSettings()
    profile = settings.sandbox_profile

    # Checked before running rather than letting ``load_profile`` fail inside
    # the subprocess, so an unset sandbox does not reach the model as a
    # traceback it will try to debug.
    if not sandbox_profile_exists(profile):
        logger.warning("sandbox profile %r is not configured", profile)
        return _NOT_CONFIGURED

    # `sandbox check` proves separation; this proves it again at the moment it
    # matters. The setting is a profile name, and nothing stops it naming the
    # user's own profile --- which is issue #73 with an extra step.
    try:
        sharing = profiles_sharing_storage(get_config(), profile)
    except Exception:
        # Deliberately broad, and deliberately fails closed: whatever went
        # wrong, separation was not proved, and the answer to "I could not
        # tell" here has to be the same as the answer to "they share".
        logger.warning("could not verify sandbox profile %r", profile, exc_info=True)
        return _NOT_SEPARATE
    if sharing:
        logger.warning(
            "sandbox profile %r is not separate: %s",
            profile,
            ", ".join(entry.name for entry in sharing),
        )
        return _NOT_SEPARATE

    result = run_in_sandbox(code, profile=profile, timeout=settings.sandbox_timeout)
    logger.info(
        "sandbox run: ok=%s refused=%s timed_out=%s in %.1fs",
        result.ok,
        result.refused,
        result.timed_out,
        result.duration_seconds,
    )
    return result.summary()
