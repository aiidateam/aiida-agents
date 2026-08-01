"""Tests for creating and verifying the read-only profile.

Two things are worth doubting here and neither needs a database.

The SQL must grant reading and *only* reading, including for tables a future
AiiDA migration has not created yet. And the verification must treat anything
short of a clear "no" as a failure --- an unreachable database is not evidence
of safety, and reading it as one would be the single worst bug in this feature.
"""

from __future__ import annotations

import pytest

from aiida_agents.sandbox.setup import (
    ReadOnlyCheck,
    interpret_probe,
    profile_setup_command,
    readonly_role_sql,
)


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
        assert "GRANT CONNECT ON DATABASE aiida_db" in sql

    def test_the_named_role_is_used_throughout(self, sql: str) -> None:
        assert sql.count("ro_role") == 5

    def test_no_superuser_or_ownership_is_granted(self, sql: str) -> None:
        assert "SUPERUSER" not in sql
        assert "OWNER" not in sql


class TestProfileCommand:
    """The verdi invocation that registers the sandbox profile."""

    @pytest.fixture
    def storage(self) -> dict[str, object]:
        return {
            "database_name": "aiida_db",
            "database_hostname": "db.example",
            "database_port": 6543,
            "repository_uri": "file:///data/repo",
        }

    def test_it_points_at_the_same_database(self, storage: dict[str, object]) -> None:
        """A scratch database with no data could not answer anything useful."""
        command = profile_setup_command(storage, "ro_role", "pw", "agents-sandbox")

        assert "--database-name aiida_db" in command
        assert "--database-hostname db.example" in command
        assert "--database-port 6543" in command

    def test_it_connects_as_the_restricted_role(
        self, storage: dict[str, object]
    ) -> None:
        command = profile_setup_command(storage, "ro_role", "pw", "agents-sandbox")

        assert "--database-username ro_role" in command

    def test_it_names_the_sandbox_profile(self, storage: dict[str, object]) -> None:
        command = profile_setup_command(storage, "ro_role", "pw", "my-sandbox")

        assert "--profile-name my-sandbox" in command

    def test_a_missing_repository_is_flagged_not_invented(self) -> None:
        """A silently wrong repository path is worse than an obvious placeholder."""
        command = profile_setup_command({}, "r", "p", "s")

        assert "<same as source profile>" in command


class TestInterpretingTheProbe:
    """What counts as proof that a profile cannot write.

    The probe asks Postgres whether the connected role holds INSERT on the node
    table. Everything here is about refusing to read ambiguity as safety.
    """

    def test_no_insert_privilege_is_read_only(self) -> None:
        check = interpret_probe("ro_role|false\n", "agents-sandbox")

        assert check.read_only
        assert "cannot insert" in check.detail

    def test_insert_privilege_is_not_read_only(self) -> None:
        check = interpret_probe("aiida|true\n", "agents-sandbox")

        assert not check.read_only

    def test_a_writable_profile_is_named_as_unusable(self) -> None:
        """The message has to stop someone using it, not merely inform them."""
        check = interpret_probe("aiida|true\n", "agents-sandbox")

        assert "must not be used as a sandbox" in check.detail

    @pytest.mark.parametrize(
        "stdout",
        ["", "\n", "some warning with no answer in it", "role|maybe", "|", "garbage"],
    )
    def test_anything_unclear_is_not_read_only(self, stdout: str) -> None:
        """An unparseable answer is not evidence of safety."""
        assert not interpret_probe(stdout, "p").read_only

    def test_the_answer_is_taken_from_the_last_line(self) -> None:
        """AiiDA logs warnings on profile load; the answer comes after them."""
        noisy = "Warning: something\nDeprecation: something else\nro_role|false\n"

        assert interpret_probe(noisy, "p").read_only

    def test_the_role_name_is_reported(self) -> None:
        """So a check run against the wrong profile is obvious."""
        assert "ro_role" in interpret_probe("ro_role|false", "p").detail

    def test_a_check_is_falsy_when_not_read_only(self) -> None:
        assert not ReadOnlyCheck(False, "nope")
        assert ReadOnlyCheck(True, "yes")
