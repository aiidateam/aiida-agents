"""Tests for the disposable copy that generated code reads.

One rule is load-bearing here and the rest is detail: **a sandbox profile must
never share deletable storage with a real one**. Issue #73 is what happens
without it --- a maintainer deleted the sandbox profile, agreed to delete its
data, and lost the database it was pointing at, which was his own.

So :func:`shares_storage` is tested harder than anything else in this file, and
in particular tested for what it does when it *cannot tell*.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aiida_agents.sandbox.copy import (
    SandboxStorage,
    copy_sqlite_storage,
    postgres_copy_commands,
    sandbox_profile_dictionary,
    shares_storage,
    storage_locations,
)

SQLITE = "core.sqlite_dos"
POSTGRES = "core.psql_dos"


def _pg(
    database: str = "aiida_db", repository: str = "/data/repo"
) -> dict[str, object]:
    return {
        "database_name": database,
        "database_hostname": "localhost",
        "database_port": 5432,
        "database_username": "aiida",
        "database_password": "pw",
        "repository_uri": repository,
    }


class TestSharingIsRefused:
    """The rule the module exists for."""

    def test_the_same_sqlite_directory_is_sharing(self) -> None:
        config = {"filepath": "/data/storage"}

        assert shares_storage(SQLITE, config, SQLITE, dict(config))

    def test_the_same_postgres_database_is_sharing(self) -> None:
        assert shares_storage(POSTGRES, _pg(), POSTGRES, _pg())

    def test_a_separate_database_but_the_same_repository_is_still_sharing(self) -> None:
        """Half a copy is not a copy.

        A sandbox with its own database and the user's repository still loses
        them every file their nodes refer to.
        """
        assert shares_storage(
            POSTGRES,
            _pg(database="aiida_db"),
            POSTGRES,
            _pg(database="aiida_db_agents_sandbox"),
        )

    def test_a_genuinely_separate_copy_is_not_sharing(self) -> None:
        assert not shares_storage(
            POSTGRES,
            _pg(database="aiida_db", repository="/data/repo"),
            POSTGRES,
            _pg(database="aiida_db_sandbox", repository="/data/sandbox/repo"),
        )

    def test_separate_sqlite_directories_are_not_sharing(self) -> None:
        assert not shares_storage(
            SQLITE, {"filepath": "/data/real"}, SQLITE, {"filepath": "/data/copy"}
        )

    def test_the_same_directory_written_differently_is_still_sharing(self) -> None:
        """Paths are compared resolved, not as strings.

        `/data/storage` and `/data/./sub/../storage` are one directory, and a
        string comparison would call them two.
        """
        assert shares_storage(
            SQLITE,
            {"filepath": "/data/storage"},
            SQLITE,
            {"filepath": "/data/./sub/../storage"},
        )


class TestFailingClosed:
    """What happens when the answer is not knowable.

    Every case here returns True. "I cannot tell whether these are separate"
    and "these are separate" must not lead to the same action, and only one of
    those two mistakes destroys data.
    """

    @pytest.mark.parametrize(
        "backend, config",
        [
            pytest.param("core.sqlite_zip", {"filepath": "/x"}, id="unknown-backend"),
            pytest.param(SQLITE, {}, id="sqlite-with-no-path"),
            pytest.param(POSTGRES, {}, id="postgres-with-nothing"),
            pytest.param(
                POSTGRES, {"database_name": "aiida"}, id="postgres-with-no-repository"
            ),
            pytest.param(
                POSTGRES, {"repository_uri": "/r"}, id="postgres-with-no-database"
            ),
        ],
    )
    def test_an_unreadable_config_counts_as_sharing(
        self, backend: str, config: dict[str, object]
    ) -> None:
        assert shares_storage(backend, config, SQLITE, {"filepath": "/somewhere/else"})
        assert shares_storage(SQLITE, {"filepath": "/somewhere/else"}, backend, config)

    def test_an_unreadable_config_yields_no_locations(self) -> None:
        assert storage_locations("core.sqlite_zip", {"filepath": "/x"}) == frozenset()


class TestCopyingSqliteStorage:
    """The copy has to be complete, or the backend will not open it."""

    @pytest.fixture
    def storage(self, tmp_path: Path) -> Path:
        source = tmp_path / "source"
        (source / "container" / "loose").mkdir(parents=True)
        (source / "container" / "sandbox").mkdir()
        (source / "database.sqlite").write_bytes(b"not really a database")
        (source / "container" / "config.json").write_text("{}")
        return source

    def test_the_database_and_the_repository_both_come_across(
        self, storage: Path, tmp_path: Path
    ) -> None:
        target = tmp_path / "copy"
        copy_sqlite_storage(storage, target)

        assert (target / "database.sqlite").read_bytes() == b"not really a database"
        assert (target / "container" / "config.json").exists()

    def test_the_object_stores_scratch_directory_is_copied_too(
        self, storage: Path, tmp_path: Path
    ) -> None:
        """A real bug, found by running it rather than by reading it.

        The first version skipped `container/sandbox/` on the reasonable
        grounds that a copy which never writes has no use for a scratch
        directory for in-flight writes. But `Container.is_initialised` checks
        that the directory is *present*, so the copy loaded as an uninitialised
        container and every query raised `UnreachableStorage`.
        """
        target = tmp_path / "copy"
        copy_sqlite_storage(storage, target)

        assert (target / "container" / "sandbox").is_dir()

    def test_nothing_in_the_source_is_left_behind(
        self, storage: Path, tmp_path: Path
    ) -> None:
        target = tmp_path / "copy"
        copy_sqlite_storage(storage, target)

        expected = {path.relative_to(storage) for path in storage.rglob("*")}
        actual = {path.relative_to(target) for path in target.rglob("*")}
        assert expected == actual

    def test_an_existing_target_is_refused_rather_than_merged(
        self, storage: Path, tmp_path: Path
    ) -> None:
        """Refreshing is teardown then rebuild, never an overwrite in place.

        Copying over a directory something may still be reading is how you get
        a half-old, half-new storage that opens and answers wrongly.
        """
        target = tmp_path / "copy"
        target.mkdir()

        with pytest.raises(FileExistsError, match="tear the sandbox down"):
            copy_sqlite_storage(storage, target)

    def test_a_missing_source_says_so(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            copy_sqlite_storage(tmp_path / "nope", tmp_path / "copy")


class TestPostgresCopyCommands:
    def test_it_dumps_rather_than_using_a_template(self) -> None:
        """`CREATE DATABASE ... TEMPLATE` is faster and unusable here.

        It refuses to run while any other session is connected to the source,
        and with a daemon running or a `verdi shell` open that is most of the
        time. `pg_dump` works against a database in use.
        """
        commands = " ".join(
            command for _, command in postgres_copy_commands(_pg(), "s")
        )

        assert "pg_dump" in commands
        assert "TEMPLATE" not in commands.upper()

    def test_the_database_is_created_before_it_is_filled(self) -> None:
        steps = [command for _, command in postgres_copy_commands(_pg(), "sandbox_db")]

        assert "createdb" in steps[0]
        assert "pg_dump" in steps[1]

    def test_every_step_explains_itself(self) -> None:
        """These are pasted into a terminal by hand; an unexplained one is a
        command somebody runs without knowing what it does."""
        assert all(explanation for explanation, _ in postgres_copy_commands(_pg(), "s"))

    def test_awkward_database_names_are_quoted(self) -> None:
        commands = " ".join(
            command
            for _, command in postgres_copy_commands(_pg(database="gsoc-psql"), "s-box")
        )

        assert '"gsoc-psql"' in commands
        assert '"s-box"' in commands


class TestTheClonedProfile:
    @pytest.fixture
    def source(self) -> dict[str, object]:
        return {
            "PROFILE_UUID": "1111",
            "default_user_email": "someone@example.com",
            "storage": {"backend": SQLITE, "config": {"filepath": "/data/real"}},
            "process_control": {"backend": "core.rabbitmq", "config": {"port": 5672}},
            "options": {"runner.poll.interval": 1},
            "test_profile": False,
        }

    @pytest.fixture
    def storage(self) -> SandboxStorage:
        return SandboxStorage(SQLITE, {"filepath": "/data/copy"})

    def test_it_points_at_the_copy(
        self, source: dict[str, object], storage: SandboxStorage
    ) -> None:
        result = sandbox_profile_dictionary(source, storage)

        assert result["storage"]["config"]["filepath"] == "/data/copy"

    def test_it_does_not_point_at_the_source(
        self, source: dict[str, object], storage: SandboxStorage
    ) -> None:
        result = sandbox_profile_dictionary(source, storage)

        assert not shares_storage(
            SQLITE,
            {"filepath": "/data/real"},
            result["storage"]["backend"],
            result["storage"]["config"],
        )

    def test_the_uuid_is_regenerated(
        self, source: dict[str, object], storage: SandboxStorage
    ) -> None:
        """Two profiles sharing a UUID are two profiles AiiDA cannot tell apart."""
        result = sandbox_profile_dictionary(source, storage)

        assert result["PROFILE_UUID"] != "1111"

    def test_the_broker_is_not_carried_over(
        self, source: dict[str, object], storage: SandboxStorage
    ) -> None:
        """The sandbox runs nothing, so it needs no queues --- and pointing it
        at the source profile's would let generated code reach a daemon."""
        result = sandbox_profile_dictionary(source, storage)

        assert result["process_control"]["backend"] is None

    def test_the_user_is_carried_over(
        self, source: dict[str, object], storage: SandboxStorage
    ) -> None:
        """The copy holds the same users; a different default would not resolve."""
        result = sandbox_profile_dictionary(source, storage)

        assert result["default_user_email"] == "someone@example.com"
