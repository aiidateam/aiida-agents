"""Tests for ``aiida_agents.tools.daemon``.

The daemon client is not ours, so it is stubbed: what these pin is the part
that *is* ours -- the pending-process count read from the live test profile,
and the ``diagnosis`` sentence, which is the whole reason the tool exists. Every
other tool can already report that a process is "waiting"; only this one says
why it will stay that way.
"""

from __future__ import annotations

import typing as t

import pytest
from aiida import orm

from aiida_agents.tools import daemon as daemon_tool
from aiida_agents.tools.daemon import get_daemon_status


class _FakeClient:
    """Stands in for ``DaemonClient``: only the three members the tool reads."""

    def __init__(self, running: bool, workers: int | None = 1) -> None:
        self.is_daemon_running = running
        self._workers = workers
        self.profile = t.cast(t.Any, type("_P", (), {"name": "test-profile"})())

    def get_numprocesses(self) -> dict[str, int | None]:
        if self._workers is None:
            msg = "circus endpoint unreachable"
            raise RuntimeError(msg)
        return {"numprocesses": self._workers}


@pytest.fixture
def fake_daemon(monkeypatch: pytest.MonkeyPatch) -> t.Callable[..., None]:
    """Install a stub daemon client with the given running/worker state."""

    def install(running: bool, workers: int | None = 1) -> None:
        monkeypatch.setattr(
            daemon_tool, "get_daemon_client", lambda: _FakeClient(running, workers)
        )

    return install


def test_a_stopped_daemon_is_reported_with_the_fix(
    fake_daemon: t.Callable[..., None],
) -> None:
    """The answer names the command, because that is what the user needs next."""
    fake_daemon(running=False)

    status = get_daemon_status()

    assert status["running"] is False
    assert status["profile"] == "test-profile"
    assert "verdi daemon start" in status["diagnosis"]


def test_a_running_daemon_reports_its_worker_count(
    fake_daemon: t.Callable[..., None],
) -> None:
    """The healthy case still answers, rather than only speaking up on trouble."""
    fake_daemon(running=True, workers=2)

    status = get_daemon_status()

    assert status["running"] is True
    assert status["workers"] == 2
    assert "2 worker(s)" in status["diagnosis"]


def test_zero_workers_is_distinguished_from_a_stopped_daemon(
    fake_daemon: t.Callable[..., None],
) -> None:
    """A running daemon with no workers processes nothing, and needs a different fix."""
    fake_daemon(running=True, workers=0)

    status = get_daemon_status()

    assert status["running"] is True
    assert "verdi daemon incr" in status["diagnosis"]
    assert "verdi daemon start" not in status["diagnosis"]


def test_an_unreachable_worker_count_does_not_fail_the_call(
    fake_daemon: t.Callable[..., None],
) -> None:
    """``running`` is the load-bearing fact; the worker count is a detail.

    The circus endpoint is briefly unreachable during a daemon restart. Failing
    the whole call there would withhold the one fact the caller came for.
    """
    fake_daemon(running=True, workers=None)

    status = get_daemon_status()

    assert status["running"] is True
    assert status["workers"] is None


def test_a_pending_process_is_counted_and_named(
    fake_daemon: t.Callable[..., None],
) -> None:
    """A stopped daemon plus pending work is the state that strands a user.

    Built against the live profile rather than mocked, so the query that finds
    non-terminal processes is exercised for real -- that query is the half of
    this tool that could silently return nothing.
    """
    fake_daemon(running=False)
    before = get_daemon_status()["pending"]

    stuck = orm.WorkChainNode()
    stuck.set_process_state("waiting")
    stuck.store()

    status = get_daemon_status()

    assert status["pending"] == before + 1
    assert status["oldest_pending_pk"] is not None
    assert f"{status['pending']} process(es)" in status["diagnosis"]
    assert "resume on their own" in status["diagnosis"]


def test_a_terminated_process_is_not_counted_as_pending(
    fake_daemon: t.Callable[..., None],
    add_calc: orm.CalcJobNode,
) -> None:
    """A finished process is owed nothing, whatever its exit code.

    Guards the inverse error: counting every process would make ``pending``
    always non-zero, and the diagnosis permanently alarming.
    """
    fake_daemon(running=True, workers=1)
    assert add_calc.is_terminated

    pending_pks = _pending_pks()

    assert add_calc.pk not in pending_pks


def _pending_pks() -> set[int]:
    """The pks the tool would count, via the same filter it uses."""
    builder = orm.QueryBuilder()
    builder.append(
        orm.ProcessNode,
        filters={"attributes.process_state": {"in": list(daemon_tool._PENDING_STATES)}},
        project=["id"],
    )
    return {row[0] for row in builder.all()}
