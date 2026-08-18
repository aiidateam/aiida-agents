"""Tests for the read-only role SQL.

One thing is worth doubting here and it needs no database: the SQL must grant
reading and *only* reading, including for tables a future AiiDA migration has
not created yet.
"""

from __future__ import annotations

import pytest

from aiida_agents.sandbox.setup import readonly_role_sql


class TestRoleSql:
    """What the generated SQL grants."""

    @pytest.fixture
    def sql(self) -> str:
        return readonly_role_sql("aiida_db", "ro_role", "s3cret")

    def test_it_grants_select(self, sql: str) -> None:
        assert "GRANT SELECT ON ALL TABLES" in sql

    @pytest.mark.parametrize("write", ["INSERT", "UPDATE", "DELETE", "TRUNCATE"])
    def test_it_grants_no_write_of_any_kind(self, sql: str, write: str) -> None:
        """The whole point. A single stray grant here voids the containment."""
        assert write not in sql

    def test_future_tables_are_covered(self, sql: str) -> None:
        """A migration adding a table must not leave it unreadable."""
        assert "ALTER DEFAULT PRIVILEGES" in sql

    def test_the_role_can_connect(self, sql: str) -> None:
        assert "LOGIN PASSWORD" in sql
        assert 'GRANT CONNECT ON DATABASE "aiida_db"' in sql

    def test_the_named_role_is_used_throughout(self, sql: str) -> None:
        assert sql.count("ro_role") == 5

    def test_no_superuser_or_ownership_is_granted(self, sql: str) -> None:
        assert "SUPERUSER" not in sql
        assert "OWNER" not in sql

    def test_a_hyphenated_database_name_is_valid_sql(self) -> None:
        """Issue #73: `gsoc-psql` was a syntax error in the middle of the SQL.

        Unquoted Postgres identifiers may hold only letters, digits and
        underscores, and `-` reads as subtraction. Profile names like
        `gsoc-psql` are entirely ordinary, so this broke setup for whoever
        happened to use one -- as a wall of SQL they had been told to paste as
        a superuser.
        """
        sql = readonly_role_sql("gsoc-psql", "ro_role", "pw")

        assert 'GRANT CONNECT ON DATABASE "gsoc-psql"' in sql

    @pytest.mark.parametrize(
        "database", ["gsoc-psql", "aiida db", "MixedCase", "aiida.test", "2024-runs"]
    )
    def test_awkward_database_names_are_quoted(self, database: str) -> None:
        assert f'"{database}"' in readonly_role_sql(database, "ro_role", "pw")

    def test_a_quote_in_an_identifier_cannot_end_the_identifier(self) -> None:
        """Doubling is how Postgres escapes a `"` inside a quoted identifier.

        None of these values is attacker-supplied, but this SQL is run as a
        superuser, which is the last place to leave an injection shaped hole.
        """
        sql = readonly_role_sql('db"name', "ro_role", "pw")

        assert '"db""name"' in sql

    def test_a_quote_in_the_password_cannot_end_the_literal(self) -> None:
        sql = readonly_role_sql("aiida_db", "ro_role", "pw'; DROP DATABASE x; --")

        assert "'pw''; DROP DATABASE x; --'" in sql
        assert "DROP DATABASE" not in sql.replace("'pw''; DROP DATABASE x; --'", ""), (
            "the injection must stay inside the literal"
        )
