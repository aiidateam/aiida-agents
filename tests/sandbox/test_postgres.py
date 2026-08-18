"""Tests for making the PostgreSQL copy rather than printing how to make it.

The interesting half is what happens when it *cannot*: a managed server, a
remote host, no `sudo`. Every one of those has to land back on the printed
commands rather than on a traceback, because the printed commands were the
whole feature until now and are still the only road on those machines.
"""

from __future__ import annotations

import subprocess
import sys
import typing as t

import pytest

from aiida_agents.sandbox.postgres import (
    PostgresUnavailableError,
    copy_database,
    create_database,
)


def _pg(database: str = "aiida_db") -> dict[str, object]:
    return {
        "database_name": database,
        "database_hostname": "localhost",
        "database_port": 5432,
        "database_username": "aiida",
        "database_password": "pw",
    }


class _FakePostgres:
    """`aiida.manage.external.postgres.Postgres`, as far as this uses it."""

    def __init__(self, *, connected: bool = True, existing: bool = False) -> None:
        self.is_connected = connected
        self._existing = existing
        self.created: list[tuple[str, str]] = []

    def determine_setup(self) -> None:
        return None

    def db_exists(self, name: str) -> bool:
        return self._existing

    def create_db(self, owner: str, name: str) -> None:
        self.created.append((owner, name))


class TestCreatingTheDatabase:
    def test_it_creates_one_owned_by_the_profile_s_user(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Owned by them, because they are who fills and reads it afterwards."""
        postgres = _FakePostgres()
        monkeypatch.setattr(
            "aiida.manage.external.postgres.Postgres", lambda **kwargs: postgres
        )

        create_database(_pg(), "aiida_db_agents_sandbox")

        assert postgres.created == [("aiida", "aiida_db_agents_sandbox")]

    def test_an_existing_database_is_reused_rather_than_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`init` is rerun after a half-finished setup more often than not."""
        postgres = _FakePostgres(existing=True)
        monkeypatch.setattr(
            "aiida.manage.external.postgres.Postgres", lambda **kwargs: postgres
        )

        create_database(_pg(), "aiida_db_agents_sandbox")

        assert postgres.created == []

    @pytest.mark.parametrize(
        "postgres, reason",
        [
            pytest.param(_FakePostgres(connected=False), "no", id="no-connection"),
            pytest.param(None, "boom", id="construction-raised"),
        ],
    )
    def test_no_privileged_connection_is_reported_not_raised_raw(
        self, monkeypatch: pytest.MonkeyPatch, postgres: object, reason: str
    ) -> None:
        """The caller's fallback is to print the commands, and it needs a
        sentence saying why it came to that."""

        def _build(**kwargs: object) -> object:
            if postgres is None:
                raise RuntimeError("boom")
            return postgres

        monkeypatch.setattr("aiida.manage.external.postgres.Postgres", _build)

        with pytest.raises(PostgresUnavailableError, match=reason):
            create_database(_pg(), "sandbox_db")


def _spy_popen(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[subprocess.Popen[bytes]], dict[str, str]]:
    """Record the processes `copy_database` starts and the environment it gives them.

    The signature matches how it calls `Popen`, so dropping `env` there fails
    here rather than being swallowed by a `**kwargs` spy.
    """
    started: list[subprocess.Popen[bytes]] = []
    environment: dict[str, str] = {}
    real = subprocess.Popen

    def _spy(
        argv: t.Sequence[str],
        *,
        env: dict[str, str],
        stdin: int | t.IO[bytes] | None = None,
        stdout: int | None = None,
        stderr: int | None = None,
    ) -> subprocess.Popen[bytes]:
        environment.update(env)
        process = real(argv, env=env, stdin=stdin, stdout=stdout, stderr=stderr)
        started.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", _spy)
    return started, environment


class TestCopyingTheData:
    @staticmethod
    def _stub_argv(monkeypatch: pytest.MonkeyPatch, *stages: tuple[str, ...]) -> None:
        """Stand two programs in for `pg_dump` and `psql`."""
        monkeypatch.setattr(
            "aiida_agents.sandbox.copy.postgres_copy_argv",
            lambda config: stages[0],
        )
        monkeypatch.setattr(
            "aiida_agents.sandbox.copy.postgres_restore_argv",
            lambda config, database: stages[1],
        )

    def test_it_pipes_one_program_into_the_next(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`pg_dump | psql` is two processes, and the second reads the first."""
        self._stub_argv(
            monkeypatch,
            (sys.executable, "-c", "print('rows')"),
            (sys.executable, "-c", "import sys; assert sys.stdin.read().strip()"),
        )

        copy_database(_pg(), "sandbox_db")

    def test_a_failing_stage_carries_its_own_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Postgres already said what was wrong; "it didn't work" costs a round
        trip to find out again."""
        self._stub_argv(
            monkeypatch,
            (sys.executable, "-c", "print('rows')"),
            (sys.executable, "-c", "import sys; sys.exit('database does not exist')"),
        )

        with pytest.raises(PostgresUnavailableError, match="database does not exist"):
            copy_database(_pg(), "sandbox_db")

    def test_the_stage_that_failed_first_is_not_blamed_for_the_broken_pipe(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`psql` exiting leaves `pg_dump` writing into a closed pipe, so it
        fails too. Reporting the earliest failure then handed the user a broken
        pipe to debug instead of the database `psql` could not find."""
        self._stub_argv(
            monkeypatch,
            # More than a pipe buffer, so the write cannot land before the
            # reader is gone: this fails every time, not one run in twenty.
            (sys.executable, "-c", "import sys; sys.stdout.write('x' * 10_000_000)"),
            (sys.executable, "-c", "import sys; sys.exit('database does not exist')"),
        )

        with pytest.raises(PostgresUnavailableError) as raised:
            copy_database(_pg(), "sandbox_db")

        assert "database does not exist" in str(raised.value)
        assert "roken pipe" not in str(raised.value)

    def test_a_failing_stage_does_not_leave_the_next_one_running(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Raising on the first stage's return code skipped `communicate` for
        the second, leaving its `stderr` pipe undrained: `psql` reporting a
        pipe-buffer's worth of broken statements blocked there rather than
        exiting, with nobody left to read it."""
        started, _ = _spy_popen(monkeypatch)
        self._stub_argv(
            monkeypatch,
            (sys.executable, "-c", "import sys; sys.exit('no such database')"),
            (
                sys.executable,
                "-c",
                "import sys; sys.stdin.buffer.read(); sys.stderr.write('x' * 200_000)",
            ),
        )

        with pytest.raises(PostgresUnavailableError, match="no such database"):
            copy_database(_pg(), "sandbox_db")

        # Both reaped: `None` here is a process still blocked on its pipe.
        assert [process.poll() for process in started] == [1, 0]

    def test_a_missing_program_is_reported_by_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`pg_dump` absent is the common case on a machine that only talks to
        a remote server."""
        self._stub_argv(
            monkeypatch, ("pg_dump_that_is_not_installed",), (sys.executable, "-c", "")
        )

        with pytest.raises(PostgresUnavailableError, match="pg_dump_that_is_not"):
            copy_database(_pg(), "sandbox_db")

    def test_the_password_is_handed_over_rather_than_prompted_for(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """It is already in `config.json`. Prompting for it would teach people
        to type their database password at whatever asks."""
        _, environment = _spy_popen(monkeypatch)
        self._stub_argv(
            monkeypatch, (sys.executable, "-c", "pass"), (sys.executable, "-c", "pass")
        )

        copy_database(_pg(), "sandbox_db")

        assert environment["PGPASSWORD"] == "pw"
