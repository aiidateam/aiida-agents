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

from aiida_agents.sandbox.copy import (  # noqa: F401
    copy_sqlite_storage,
    postgres_copy_commands,
    Location,
    Overlap,
    PathLocation,
    ProfileStorage,
    StorageConfig,
    SharingProfile,
    profiles_sharing_storage,
    sandbox_profile_dictionary,
    shares_storage,
    storage_locations,
    storage_overlap,
)

SQLITE = "core.sqlite_dos"
POSTGRES = "core.psql_dos"
ARCHIVE = "core.sqlite_zip"
UNREADABLE = "thirdparty.custom_dos"


def _shares(
    backend_a: str,
    config_a: StorageConfig,
    backend_b: str,
    config_b: StorageConfig,
) -> bool:
    """`shares_storage` spelled from the two halves each case is written in.

    The module compares `ProfileStorage` values, which is what stops a caller
    pairing one profile's backend with another's config. These tests are about
    which configurations share storage, so they name the halves and let this
    do the pairing once.
    """
    return shares_storage(
        ProfileStorage(backend_a, config_a), ProfileStorage(backend_b, config_b)
    )


def _locations(backend: str, config: StorageConfig) -> frozenset[Location]:
    return storage_locations(ProfileStorage(backend, config))


def _overlap(
    backend_a: str,
    config_a: StorageConfig,
    backend_b: str,
    config_b: StorageConfig,
) -> Overlap:
    return storage_overlap(
        ProfileStorage(backend_a, config_a), ProfileStorage(backend_b, config_b)
    )


def _pg(database: str = "aiida_db", repository: str = "/data/repo") -> StorageConfig:
    return {
        "database_name": database,
        "database_hostname": "localhost",
        "database_port": 5432,
        "database_username": "aiida",
        "database_password": "pw",
        "repository_uri": repository,
    }


class TestWhichPairsShareStorage:
    """The rule the module exists for, as the table of pairs it comes down to.

    One case per way two profiles can be related, because the interesting part
    is the boundary: an archive and a directory at one path share it, a name
    that merely starts the same does not, and anything unreadable counts as
    sharing rather than as separate.
    """

    SANDBOX = "/data/agents-sandbox/storage"

    @pytest.mark.parametrize(
        "a, b, shared, why",
        [
            # The same storage, spelled in every way it can be spelled.
            (
                (SQLITE, {"filepath": "/data/s"}),
                (SQLITE, {"filepath": "/data/s"}),
                True,
                "one directory",
            ),
            (
                (SQLITE, {"filepath": "/data/s"}),
                (SQLITE, {"filepath": "/data/./x/../s"}),
                True,
                "one directory, written differently: paths are resolved",
            ),
            ((POSTGRES, _pg()), (POSTGRES, _pg()), True, "one database"),
            (
                (POSTGRES, _pg(database="a")),
                (POSTGRES, _pg(database="b")),
                True,
                "own database, shared repository: half a copy is not a copy",
            ),
            (
                (ARCHIVE, {"filepath": "/data/e.aiida"}),
                (ARCHIVE, {"filepath": "/data/./e.aiida"}),
                True,
                "one archive, which `--delete-data` unlinks",
            ),
            # A path is a path, whatever the profile calls the thing there.
            (
                (SQLITE, {"filepath": "/data/thing"}),
                (ARCHIVE, {"filepath": "/data/thing"}),
                True,
                "a directory and an archive at one path",
            ),
            (
                (POSTGRES, {"database_name": "d", "repository_uri": "/data/thing"}),
                (ARCHIVE, {"filepath": "/data/thing"}),
                True,
                "a repository and an archive at one path",
            ),
            (
                (
                    POSTGRES,
                    {
                        "database_name": "d",
                        "repository_uri": Path("/data/My Drive").as_uri(),
                    },
                ),
                (SQLITE, {"filepath": "/data/My Drive"}),
                True,
                "a percent-encoded `file://` repository is the directory it names",
            ),
            # Containment, because `teardown` removes its root recursively.
            (
                (SQLITE, {"filepath": SANDBOX}),
                (ARCHIVE, {"filepath": f"{SANDBOX}/e.aiida"}),
                True,
                "an archive inside the sandbox directory",
            ),
            (
                (ARCHIVE, {"filepath": f"{SANDBOX}/e.aiida"}),
                (SQLITE, {"filepath": SANDBOX}),
                True,
                "the same, the other way round",
            ),
            (
                (POSTGRES, {"database_name": "d", "repository_uri": "/data/repo"}),
                (SQLITE, {"filepath": "/data/repo/nested"}),
                True,
                "a profile inside a Postgres repository",
            ),
            # Genuinely separate.
            (
                (SQLITE, {"filepath": "/data/real"}),
                (SQLITE, {"filepath": "/data/copy"}),
                False,
                "two directories",
            ),
            (
                (SQLITE, {"filepath": "/data/s"}),
                (SQLITE, {"filepath": "/data/s-2"}),
                False,
                "a name that merely starts the same",
            ),
            (
                (POSTGRES, _pg(database="a", repository="/data/a")),
                (POSTGRES, _pg(database="b", repository="/data/b")),
                False,
                "a whole copy: own database, own repository",
            ),
            (
                (SQLITE, {"filepath": SANDBOX}),
                (ARCHIVE, {"filepath": "/data/e.aiida"}),
                False,
                "an archive elsewhere, which is the false positive #90 was about",
            ),
            (
                (POSTGRES, _pg()),
                (ARCHIVE, {"filepath": "/data/e.aiida"}),
                False,
                "a Postgres profile and an archive elsewhere",
            ),
        ],
    )
    def test_a_pair_shares_storage_or_does_not(
        self,
        a: tuple[str, StorageConfig],
        b: tuple[str, StorageConfig],
        shared: bool,
        why: str,
    ) -> None:
        assert _shares(*a, *b) is shared, why
        assert _shares(*b, *a) is shared, f"{why}, and order must not matter"


class TestFailingClosed:
    """What happens when the answer is not knowable.

    Every case here returns True. "I cannot tell whether these are separate"
    and "these are separate" must not lead to the same action, and only one of
    those two mistakes destroys data.
    """

    @pytest.mark.parametrize(
        "backend, config",
        [
            pytest.param(UNREADABLE, {"filepath": "/x"}, id="unknown-backend"),
            pytest.param(SQLITE, {}, id="sqlite-with-no-path"),
            pytest.param(ARCHIVE, {}, id="archive-with-no-path"),
            pytest.param(POSTGRES, {}, id="postgres-with-nothing"),
            pytest.param(
                POSTGRES, {"database_name": "aiida"}, id="postgres-with-no-repository"
            ),
            pytest.param(
                POSTGRES, {"repository_uri": "/r"}, id="postgres-with-no-database"
            ),
            pytest.param(SQLITE, {"filepath": "storage"}, id="sqlite-relative-path"),
            pytest.param(ARCHIVE, {"filepath": "export.aiida"}, id="archive-relative"),
            pytest.param(
                POSTGRES,
                {"database_name": "aiida", "repository_uri": "repo"},
                id="postgres-relative-repository",
            ),
            pytest.param(SQLITE, {"filepath": 42}, id="sqlite-with-a-number"),
            pytest.param(
                SQLITE, {"filepath": Path("/data/real")}, id="sqlite-with-a-path-object"
            ),
            pytest.param(
                POSTGRES,
                {"database_name": "d", "database_hostname": ["h"]},
                id="postgres-with-an-unhashable-host",
            ),
        ],
    )
    def test_an_unreadable_config_counts_as_sharing(
        self, backend: str, config: StorageConfig
    ) -> None:
        """Relative, non-string and unhashable values are in here deliberately.

        `Path.resolve` would answer a relative path against whatever directory
        the command was run from, `Path(42)` raises, and an unhashable hostname
        took `frozenset` down with it. None of them may become "separate", and
        none may reach the user as a traceback out of the one check that guards
        their data.
        """
        assert _shares(backend, config, SQLITE, {"filepath": "/somewhere/else"})
        assert _shares(SQLITE, {"filepath": "/somewhere/else"}, backend, config)

    def test_an_unreadable_config_yields_no_locations(self) -> None:
        assert _locations(UNREADABLE, {"filepath": "/x"}) == frozenset()

    def test_an_archive_yields_its_own_location(self) -> None:
        """Named, because falling through to nothing here is what made every
        archive profile read as sharing storage with the sandbox."""
        assert _locations(ARCHIVE, {"filepath": "/data/e.aiida"}) == frozenset(
            {PathLocation(Path("/data/e.aiida"))}
        )


class TestSayingWhichKindOfNotSeparate:
    """`shares_storage` decides; `storage_overlap` says why.

    Both readings stop the same commands, and they need different words in
    front of the user: "this is the same directory as your own profile" is
    their problem to fix, "this backend is one I cannot read" is not.
    """

    def test_the_same_location_is_a_proven_overlap(self) -> None:
        config = {"filepath": "/data/storage"}

        assert _overlap(SQLITE, config, SQLITE, dict(config)) is Overlap.SHARED

    def test_different_locations_are_separate(self) -> None:
        assert (
            _overlap(
                SQLITE, {"filepath": "/data/real"}, SQLITE, {"filepath": "/data/copy"}
            )
            is Overlap.SEPARATE
        )

    @pytest.mark.parametrize(
        "backend, config",
        [
            pytest.param(UNREADABLE, {"filepath": "/x"}, id="unknown-backend"),
            pytest.param(SQLITE, {}, id="no-path-to-compare"),
        ],
    )
    def test_what_cannot_be_read_is_unknown_rather_than_shared(
        self, backend: str, config: dict[str, object]
    ) -> None:
        assert (
            _overlap(SQLITE, {"filepath": "/data/real"}, backend, config)
            is Overlap.UNKNOWN
        )

    def test_shares_storage_acts_on_unknown_exactly_as_on_shared(self) -> None:
        """The distinction is for the message, never for the decision."""
        assert _shares(SQLITE, {"filepath": "/data/real"}, UNREADABLE, {})


class _Profile:
    """The three attributes `profiles_sharing_storage` reads off a profile."""

    def __init__(self, name: str, backend: str, config: dict[str, object]) -> None:
        self.name = name
        self.storage_backend = backend
        self.storage_config = config


class _Config:
    def __init__(self, *profiles: _Profile) -> None:
        self.profiles = list(profiles)


class TestWhatTheSandboxCouldNotBeClearedOf:
    """The list `check`, `teardown` and `doctor` all report from."""

    @pytest.fixture
    def sandbox(self) -> _Profile:
        return _Profile("agents-sandbox", SQLITE, {"filepath": "/data/copy"})

    def test_a_separate_profile_is_not_listed(self, sandbox: _Profile) -> None:
        config = _Config(sandbox, _Profile("real", SQLITE, {"filepath": "/r"}))

        assert profiles_sharing_storage(config, "agents-sandbox") == []

    def test_an_archive_profile_is_not_listed(self, sandbox: _Profile) -> None:
        """The bug this suite grew out of, at the layer that reported it."""
        config = _Config(
            sandbox,
            _Profile("real", SQLITE, {"filepath": "/data/real"}),
            _Profile("dev-archive", ARCHIVE, {"filepath": "/data/export.aiida"}),
        )

        assert profiles_sharing_storage(config, "agents-sandbox") == []

    def test_a_shared_profile_is_listed_with_the_overlap_proved(
        self, sandbox: _Profile
    ) -> None:
        config = _Config(sandbox, _Profile("real", SQLITE, {"filepath": "/data/copy"}))

        assert profiles_sharing_storage(config, "agents-sandbox") == [
            SharingProfile(name="real", backend=SQLITE, overlap=Overlap.SHARED)
        ]

    def test_an_unreadable_profile_is_listed_as_unknown(
        self, sandbox: _Profile
    ) -> None:
        config = _Config(sandbox, _Profile("odd", UNREADABLE, {"filepath": "/x"}))

        assert profiles_sharing_storage(config, "agents-sandbox") == [
            SharingProfile(name="odd", backend=UNREADABLE, overlap=Overlap.UNKNOWN)
        ]

    def test_a_proven_overlap_reads_as_one(self) -> None:
        reason = SharingProfile(
            name="real", backend=SQLITE, overlap=Overlap.SHARED
        ).describe()

        assert "shares storage with 'real'" in reason

    @pytest.mark.parametrize(
        "backend",
        [
            pytest.param(UNREADABLE, id="a-backend-nobody-has-heard-of"),
            pytest.param(POSTGRES, id="a-backend-we-know-but-a-config-we-cannot-read"),
        ],
    )
    def test_what_could_not_be_read_is_not_blamed_on_the_backend(
        self, backend: str
    ) -> None:
        """The unreadable half is as often a config with a field missing, and
        it can be either profile's. Saying "this backend cannot be read" would
        be false for a `core.psql_dos` profile with no repository, which is the
        commoner way to land here than a third-party plugin."""
        reason = SharingProfile(
            name="odd", backend=backend, overlap=Overlap.UNKNOWN
        ).describe()

        assert "cannot be told apart from 'odd'" in reason
        assert backend in reason
        assert "destroy" not in reason


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
    def storage(self) -> ProfileStorage:
        return ProfileStorage(SQLITE, {"filepath": "/data/copy"})

    def test_it_points_at_the_copy(
        self, source: dict[str, object], storage: ProfileStorage
    ) -> None:
        result = sandbox_profile_dictionary(source, storage)

        assert result["storage"]["config"]["filepath"] == "/data/copy"

    def test_it_does_not_point_at_the_source(
        self, source: dict[str, object], storage: ProfileStorage
    ) -> None:
        result = sandbox_profile_dictionary(source, storage)

        assert not shares_storage(
            ProfileStorage(SQLITE, {"filepath": "/data/real"}),
            ProfileStorage(result["storage"]["backend"], result["storage"]["config"]),
        )

    def test_the_uuid_is_regenerated(
        self, source: dict[str, object], storage: ProfileStorage
    ) -> None:
        """Two profiles sharing a UUID are two profiles AiiDA cannot tell apart."""
        result = sandbox_profile_dictionary(source, storage)

        assert result["PROFILE_UUID"] != "1111"

    def test_the_broker_is_not_carried_over(
        self, source: dict[str, object], storage: ProfileStorage
    ) -> None:
        """The sandbox runs nothing, so it needs no queues --- and pointing it
        at the source profile's would let generated code reach a daemon."""
        result = sandbox_profile_dictionary(source, storage)

        assert result["process_control"]["backend"] is None

    def test_the_user_is_carried_over(
        self, source: dict[str, object], storage: ProfileStorage
    ) -> None:
        """The copy holds the same users; a different default would not resolve."""
        result = sandbox_profile_dictionary(source, storage)

        assert result["default_user_email"] == "someone@example.com"
