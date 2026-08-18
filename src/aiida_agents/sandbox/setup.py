"""SQL for a read-only PostgreSQL role, for anyone who wants one by hand.

Not part of the sandbox's containment, and not used by any command here: what
makes the sandbox safe is that it is a copy (:mod:`aiida_agents.sandbox.copy`),
which is what `ADR-11 </docs/adr/11-code-execution.md>`_ settles and what
``sandbox check`` proves. A role that cannot write can still be put under that
copy as a second layer, and :func:`readonly_role_sql` writes the statements for
it, because generating them from the profile's own storage config leaves
nothing to look up or mistype. Running them needs a Postgres superuser, so that
is the reader's to do.
"""

from __future__ import annotations

__all__ = ["readonly_role_sql"]


def _quote_identifier(name: str) -> str:
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
    quoted_role = _quote_identifier(role)
    return "\n".join(
        [
            f"CREATE ROLE {quoted_role} LOGIN PASSWORD {_quote_literal(password)};",
            f"GRANT CONNECT ON DATABASE {_quote_identifier(database)} TO {quoted_role};",
            f"GRANT USAGE ON SCHEMA public TO {quoted_role};",
            f"GRANT SELECT ON ALL TABLES IN SCHEMA public TO {quoted_role};",
            "ALTER DEFAULT PRIVILEGES IN SCHEMA public "
            f"GRANT SELECT ON TABLES TO {quoted_role};",
        ]
    )
