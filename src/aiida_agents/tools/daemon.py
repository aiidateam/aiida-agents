"""Whether the daemon is running, and whether anything is waiting on it.

Every other tool reports what a process *is* ("waiting", "created"); none could
say why it is not moving. A stopped daemon leaves a submitted process in a
non-terminal state indefinitely, and ``get_process_status``,
``get_process_report``, ``diagnose_process_failure`` and ``wait_for_process``
all describe that correctly and unhelpfully -- the process has not failed, so
there is no exit code to explain and no report line to quote.

This tool supplies the missing fact. It is read-only: it reports the daemon's
state and never starts, stops or resizes it, because a daemon a user did not
start is not one an agent should start for them.
"""

from __future__ import annotations

import logging

from aiida import orm
from aiida.engine import ProcessState
from aiida.engine.daemon.client import get_daemon_client
from aiida.manage import get_manager

from ._types import DaemonStatus

logger = logging.getLogger(__name__)

# The states in which a process is still owed work by the daemon. `EXCEPTED` and
# `KILLED` are terminal, and `FINISHED` is terminal whatever the exit code, so a
# process in any of those is not waiting on anything.
_PENDING_STATES = (
    ProcessState.CREATED.value,
    ProcessState.WAITING.value,
    ProcessState.RUNNING.value,
)


def _pending_processes() -> tuple[int, int | None]:
    """Count processes in a non-terminal state, and find the oldest one's pk."""
    builder = orm.QueryBuilder()
    builder.append(
        orm.ProcessNode,
        filters={"attributes.process_state": {"in": list(_PENDING_STATES)}},
        project=["id"],
    )
    builder.order_by({orm.ProcessNode: {"ctime": "asc"}})
    rows = builder.all()
    return len(rows), (rows[0][0] if rows else None)


def _diagnose(running: bool, workers: int | None, pending: int) -> str:
    """The sentence the other tools cannot produce, given the two facts above."""
    if not running:
        if pending:
            return (
                f"The daemon is NOT running and {pending} process(es) are still in a "
                "non-terminal state, so nothing will progress until it is started. "
                "Run `verdi daemon start`. Those processes resume on their own "
                "afterwards -- they do not need resubmitting."
            )
        return (
            "The daemon is NOT running. Nothing is waiting on it, but any new "
            "submission will sit unprocessed until `verdi daemon start`."
        )
    if workers == 0:
        return (
            "The daemon is running but has 0 workers, so no process can be picked "
            "up. Run `verdi daemon incr 1`."
        )
    if pending:
        return (
            f"The daemon is running with {workers} worker(s) and {pending} "
            "process(es) are in a non-terminal state. That is the normal picture "
            "for work in flight; a process that stays put across several minutes "
            "is worth looking at with get_process_report()."
        )
    return f"The daemon is running with {workers} worker(s) and nothing is pending."


def get_daemon_status() -> DaemonStatus:
    """Check whether the AiiDA daemon is running and what is waiting on it.

    Call this whenever a process is not progressing -- stuck in 'created',
    'waiting' or 'running', or when wait_for_process() timed out. A stopped
    daemon is the most common cause and is invisible to every other tool: the
    process has not failed, so nothing about it looks wrong.
    """
    logger.debug("get_daemon_status()")
    client = get_daemon_client()
    running = client.is_daemon_running

    workers: int | None = None
    if running:
        try:
            response = client.get_numprocesses()
            workers = response.get("numprocesses")
        except Exception:  # noqa: BLE001 - worker count is a detail, not the answer
            # The daemon can be up while the circus endpoint is briefly
            # unreachable (during a restart). `running` is the load-bearing
            # fact, so report it rather than failing the whole call.
            logger.debug("get_daemon_status: worker count unavailable", exc_info=True)

    pending, oldest = _pending_processes()
    profile = get_manager().get_profile()

    return {
        "running": running,
        "workers": workers,
        "profile": client.profile.name,
        "process_control": profile.process_control_backend if profile else None,
        "pending": pending,
        "oldest_pending_pk": oldest,
        "diagnosis": _diagnose(running, workers, pending),
    }
