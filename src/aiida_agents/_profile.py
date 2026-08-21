"""Opening the loaded profile's storage before anything starts running tools."""

from __future__ import annotations

from aiida.manage import get_manager

__all__ = ["open_profile_storage"]


def open_profile_storage() -> None:
    """Open the loaded profile's storage on the calling thread.

    Every entry point that is about to serve tools calls this from its main
    thread, right after loading the profile. Both pydantic-ai and fastmcp hand a
    sync tool to a thread pool, and AiiDA opens storage lazily on first access:
    two threads taking that first open together race the PID-named temp move in
    ``ProfileAccessManager.request_access``, and the loser raises
    ``FileNotFoundError``. Calling this again is a no-op.

    :raises aiida.common.exceptions.AiidaException: if the storage cannot be
        opened -- no profile loaded, an unmigrated schema, an unreachable
        database, or a profile locked by a maintenance operation.
    """
    get_manager().get_profile_storage()
