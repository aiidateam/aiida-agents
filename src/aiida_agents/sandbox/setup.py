"""Setting up the read-only profile the sandbox runs against, and proving it is one.

The containment in :mod:`aiida_agents.sandbox.runner` is a Postgres role that
cannot write. This module helps create that role and --- more importantly ---
lets a caller *check* it, rather than trusting that whoever set it up got the
grants right.

Two deliberate choices about how far this goes.

**It prints SQL rather than executing it.** Creating a role needs Postgres
superuser, and a tool that asks for superuser to configure a safety feature is
a worse bargain than one that shows you three statements you can read. The
statements are generated from the profile's own storage config, so there is
nothing to look up or mistype.

**Verification asks Postgres, not us.** :func:`verify_read_only` does not try a
write and see what happens --- a successful probe would leave a node in
somebody's database. It asks Postgres directly whether the connected role holds
INSERT on the node table, which is the same question with no side effect.
"""

from __future__ import annotations

import typing as t
from dataclasses import dataclass

__all__ = [
    "ReadOnlyCheck",
    "profile_setup_command",
    "readonly_role_sql",
    "sandbox_profile_exists",
    "verify_read_only",
]

#: Tables the role must not be able to write. ``db_dbnode`` is the one that
#: matters --- it is where provenance lives --- and a role that cannot insert
#: there cannot record anything AiiDA would consider a result.
_GUARDED_TABLE = "db_dbnode"


@dataclass(frozen=True)
class ReadOnlyCheck:
    """Whether a profile's database role can write, and how we know."""

    read_only: bool
    detail: str

    def __bool__(self) -> bool:
        return self.read_only


def sandbox_profile_exists(profile: str) -> bool:
    """Whether ``profile`` is a profile AiiDA knows about.

    Worth asking before doing anything heavier, because "you have not set the
    sandbox up yet" and "your query has a bug" are different problems and a
    stack trace about an unknown profile reads as the second.

    A broken or unreadable AiiDA config answers False: the caller's next move
    is to tell the user to set the profile up, which is the right advice either
    way.
    """
    try:
        from aiida.manage.configuration import get_config

        return profile in {p.name for p in get_config().profiles}
    except Exception:  # pragma: no cover - a broken AiiDA config is not our news
        return False


def quote_identifier(name: str) -> str:
    """A Postgres identifier, safe to drop into generated SQL.

    Unquoted identifiers may only hold letters, digits and underscores, so a
    perfectly ordinary profile name like ``gsoc-psql`` became a syntax error in
    the middle of the SQL people were told to paste as a superuser. Postgres
    quotes with doubled ``"``, and an embedded ``"`` is escaped by doubling it.

    Quoting also makes the identifier case-sensitive, which is what the caller
    meant: a role named ``Sandbox`` should be the role named ``Sandbox``.
    """
    escaped = name.replace('"', '""')
    return f'"{escaped}"'


def _quote_literal(value: str) -> str:
    """A Postgres string literal. Single quotes are escaped by doubling."""
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


def readonly_role_sql(database: str, role: str, password: str) -> str:
    """SQL that creates a role able to read ``database`` and nothing more.

    ``GRANT SELECT`` is issued twice on purpose: once for the tables that exist
    now, and once as a default privilege so a future AiiDA migration adding a
    table does not silently leave it unreadable. The reverse mistake --- a new
    table the role can *write* --- cannot happen, because no INSERT, UPDATE or
    DELETE is ever granted.

    Every identifier is quoted and the password is emitted as a proper literal.
    None of these values is attacker-supplied --- the database name comes from
    the user's own profile --- but this SQL is handed to somebody to run **as a
    superuser**, which is the last place to be relaxed about quoting.

    Args:
        database: The database the sandbox profile will read.
        role: Name for the new Postgres role.
        password: Password for it. Ends up in the profile's storage config, so
            it protects the role rather than the data.
    """
    quoted_role = quote_identifier(role)
    return "\n".join(
        [
            f"CREATE ROLE {quoted_role} LOGIN PASSWORD {_quote_literal(password)};",
            f"GRANT CONNECT ON DATABASE {quote_identifier(database)} TO {quoted_role};",
            f"GRANT USAGE ON SCHEMA public TO {quoted_role};",
            f"GRANT SELECT ON ALL TABLES IN SCHEMA public TO {quoted_role};",
            "ALTER DEFAULT PRIVILEGES IN SCHEMA public "
            f"GRANT SELECT ON TABLES TO {quoted_role};",
        ]
    )


def profile_setup_command(
    storage_config: dict[str, t.Any], role: str, password: str, sandbox_name: str
) -> str:
    """The ``verdi`` invocation that registers the sandbox profile.

    Every value except the credentials is copied from the source profile, so
    the sandbox reaches *the same database* --- which is the point. A scratch
    database with no data could not answer anything worth asking.

    Four details are not decoration, and each was wrong here at some point:

    ``--no-set-as-default`` is the important one. That option defaults to
    *yes*, so the earlier version of this command made the write-refusing
    profile the user's default the moment they pasted it --- and then every
    ``verdi`` command and every agent run they made afterwards would hit a
    database that cannot write.

    ``--non-interactive``, with ``--first-name`` / ``--last-name`` /
    ``--institution`` supplied. Those three are *required* options; leaving
    them out did not fail, it dropped the user into a prompt session, which is
    not what a copy-pasteable one-liner should do.

    ``--broker none``, not ``--no-use-rabbitmq``. The latter still parses in
    aiida-core 2.8 but is hidden and deprecated. The sandbox only ever reads,
    so it has no processes to control and wants no broker at all.

    ``repository_uri`` is read without a fallback. A missing one used to become
    a ``<same as source profile>`` placeholder that got pasted verbatim; the
    caller checks for it and says so instead.
    """
    return " ".join(
        [
            "verdi profile setup core.psql_dos",
            "--non-interactive",
            f"--profile-name {sandbox_name}",
            "--no-set-as-default",
            "--email sandbox@localhost",
            "--first-name Agents --last-name Sandbox --institution local",
            "--broker none",
            f"--database-name {storage_config.get('database_name', '')}",
            f"--database-hostname {storage_config.get('database_hostname', 'localhost')}",
            f"--database-port {storage_config.get('database_port', 5432)}",
            f"--database-username {role}",
            f"--database-password {password}",
            f"--repository-uri {storage_config.get('repository_uri', '')}",
        ]
    )


#: Asks Postgres what the connected role may do. A read-only question about
#: write permission, so running it proves the point without creating anything.
_PRIVILEGE_PROBE = f"""\
from aiida import load_profile
from aiida.manage import get_manager
from sqlalchemy import text

load_profile({{profile!r}})
session = get_manager().get_profile_storage().get_session()
row = session.execute(
    text("SELECT current_user, has_table_privilege(current_user, '{_GUARDED_TABLE}', 'INSERT')")
).one()
print(f"{{{{row[0]}}}}|{{{{row[1]}}}}")
"""


def verify_read_only(profile: str, timeout: float = 60.0) -> ReadOnlyCheck:
    """Ask Postgres whether ``profile``'s role can insert into the node table.

    Run this before pointing the sandbox at a profile, and again whenever the
    database has been migrated: a grant that was right in March is not
    self-maintaining.

    Args:
        profile: Name of the AiiDA profile to inspect.
        timeout: Seconds to allow for the probe.

    Returns:
        A :class:`ReadOnlyCheck`. ``read_only`` is False whenever the answer is
        anything other than a clear "no" --- an unreachable database or an
        unparseable reply is not evidence of safety.
    """
    import subprocess
    import sys

    try:
        completed = subprocess.run(  # noqa: S603
            [sys.executable, "-c", _PRIVILEGE_PROBE.format(profile=profile)],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return ReadOnlyCheck(
            False, f"timed out asking profile {profile!r} about itself"
        )

    if completed.returncode != 0:
        return ReadOnlyCheck(
            False,
            f"could not inspect profile {profile!r}: "
            f"{completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else 'no output'}",
        )

    return interpret_probe(completed.stdout, profile)


def interpret_probe(stdout: str, profile: str) -> ReadOnlyCheck:
    """Read the probe's output.

    Separated from the subprocess call so the interesting half --- deciding what
    counts as proof --- is testable without a database.
    """
    line = next(
        (candidate for candidate in reversed(stdout.splitlines()) if "|" in candidate),
        "",
    )
    user, _, privilege = line.partition("|")
    answer = privilege.strip().lower()

    if answer == "false":
        return ReadOnlyCheck(
            True, f"role {user.strip()!r} cannot insert into {_GUARDED_TABLE}"
        )
    if answer == "true":
        return ReadOnlyCheck(
            False,
            f"role {user.strip()!r} CAN insert into {_GUARDED_TABLE} -- "
            f"profile {profile!r} is not read-only and must not be used as a sandbox",
        )
    return ReadOnlyCheck(
        False,
        f"could not tell whether {profile!r} is read-only (got {stdout.strip()!r})",
    )
