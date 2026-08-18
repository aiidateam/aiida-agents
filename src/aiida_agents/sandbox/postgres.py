"""Making the PostgreSQL copy, rather than telling somebody how to make it.

``init`` used to print a ``createdb``, a ``pg_dump | psql`` pipeline and five
``GRANT`` statements, and expect a researcher to run them. They do not: they
are here to answer questions about their data, they did not ask for a safety
feature, and most of them have never written SQL. A setup step nobody performs
is a feature nobody has.

Two things stand between us and doing it for them, and only one is real.

**Creating the database needs a privilege the profile's user does not have.**
AiiDA creates its database users with ``CREATE USER ... WITH PASSWORD``, no
``CREATEDB`` (see ``Postgres.create_dbuser``), so connecting with the
credentials already in the profile and issuing ``CREATE DATABASE`` fails on
exactly the step that matters. This is why SQLAlchemy alone is not enough,
although it is a perfectly good way to run the statement once something has
handed you a connection that may.

What can hand it over is :class:`aiida.manage.external.postgres.Postgres`,
which ``verdi presto`` already uses: it tries psycopg as the current user and
falls back to ``sudo su postgres``, so anyone who has a PostgreSQL profile has
been through it once already. No new dependency, and no new trust: it is the
same route that created the database being copied.

**Copying the data needs ``pg_dump``.** SQLAlchemy cannot express it; a
row-by-row copy through the ORM is what ``verdi archive`` does, slowly. The
alternative in SQL, ``CREATE DATABASE ... TEMPLATE``, refuses to run while any
other session is connected to the source, which with a daemon running is most
of the time.

So: a superuser connection creates the empty database, and ``pg_dump | psql``
fills it as the profile's own user, who owns it by then. Where neither can be
done (a managed server, a remote host, no ``sudo``, no ``pg_dump``), the
caller is told what could not be done and which database it needed.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess

from aiida_agents.sandbox.copy import StorageConfig

logger = logging.getLogger(__name__)

__all__ = ["PostgresUnavailableError", "copy_database", "create_database"]


class PostgresUnavailableError(RuntimeError):
    """No privileged connection, or a command that failed.

    Carries what to tell the user, because the caller's fallback is to print
    the commands and this is the sentence that explains why.
    """


def create_database(config: StorageConfig, database: str) -> None:
    """Create the sandbox's database, owned by the profile's own user.

    Raises:
        PostgresUnavailableError: If no connection with the privilege to create
            a database could be found, or the creation failed.
    """
    from aiida.manage.external.postgres import Postgres

    owner = str(config.get("database_username") or "")
    dbinfo = {
        "host": config.get("database_hostname") or "localhost",
        "port": config.get("database_port") or 5432,
    }

    try:
        # aiida's Postgres is untyped, and this is the one place we cross
        # into it, which is why the ignores sit together rather than spread.
        postgres = Postgres(  # type: ignore[no-untyped-call]
            interactive=True, quiet=False, dbinfo=dbinfo
        )
        postgres.determine_setup()
    except Exception as exc:
        msg = f"no PostgreSQL connection that may create a database: {exc}"
        raise PostgresUnavailableError(msg) from exc

    if not postgres.is_connected:
        msg = "no PostgreSQL connection that may create a database"
        raise PostgresUnavailableError(msg)

    try:
        if postgres.db_exists(database):  # type: ignore[no-untyped-call]
            logger.info("database %r already exists; reusing it", database)
            return
        postgres.create_db(owner, database)  # type: ignore[no-untyped-call]
    except Exception as exc:
        msg = f"could not create the database {database!r}: {exc}"
        raise PostgresUnavailableError(msg) from exc


def _stopped_writing(returncode: int, message: str) -> bool:
    """Whether a stage failed only because the next one stopped reading.

    `psql` exiting on "database does not exist" leaves `pg_dump` writing into a
    closed pipe, which kills it with `SIGPIPE` (or, for anything that handles
    the signal, an `EPIPE` message). Reporting that instead of what `psql` said
    hands the user a broken pipe to debug rather than the missing database.
    """
    return returncode == -signal.SIGPIPE or "broken pipe" in message.lower()


def copy_database(config: StorageConfig, database: str) -> None:
    """Fill ``database`` from the profile's own, with ``pg_dump | psql``.

    The password comes from the profile rather than a prompt: it is already in
    ``config.json``, and prompting for something the caller can read teaches
    people to type their database password at whatever asks.

    Raises:
        PostgresUnavailableError: If either program is missing or fails,
            carrying what it said.
    """
    from aiida_agents.sandbox.copy import postgres_copy_argv, postgres_restore_argv

    stages = (
        postgres_copy_argv(config),
        postgres_restore_argv(config, database),
    )
    environment = dict(os.environ)
    if password := config.get("database_password"):
        environment["PGPASSWORD"] = str(password)

    processes: list[subprocess.Popen[bytes]] = []
    try:
        for stage in stages:
            previous = processes[-1].stdout if processes else None
            processes.append(
                subprocess.Popen(  # noqa: S603
                    stage,
                    stdin=previous,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=environment,
                )
            )
            if previous is not None:
                # The upstream holds the read end too; closing it here is what
                # lets it see EOF if the downstream exits early.
                previous.close()
    except (OSError, subprocess.SubprocessError) as exc:
        msg = f"could not run {stages[0][0]!r}: {exc}"
        raise PostgresUnavailableError(msg) from exc

    # Every stage is drained before any failure is reported. Raising at the
    # first non-zero return code left the rest of the pipeline uncommunicated
    # with, and a stage with more than a pipe buffer to say about the truncated
    # input it just got blocks writing it, with nobody to read it.
    causes: list[str] = []
    consequences: list[str] = []
    try:
        for process, stage in zip(processes, stages, strict=True):
            _, stderr = process.communicate()
            if process.returncode:
                detail = stderr.decode(errors="replace").strip().splitlines()
                message = f"{stage[0]} failed: {detail[-1] if detail else 'no output'}"
                if _stopped_writing(process.returncode, message):
                    consequences.append(message)
                else:
                    causes.append(message)
    finally:
        # Reached on Ctrl-C, where the drain above did not happen.
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()

    # The earliest real failure: an upstream one explains the mess downstream,
    # and a downstream one is never explained by the pipe it left behind.
    for reported in (causes, consequences):
        if reported:
            raise PostgresUnavailableError(reported[0])
