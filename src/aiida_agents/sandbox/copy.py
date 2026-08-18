"""A disposable copy of a profile's storage, for generated code to run against.

The rule this module exists to enforce:

    **A sandbox profile must never share deletable storage with a real one.**

:func:`shares_storage` is that rule as a function, and it fails closed: two
profiles it cannot compare are treated as sharing, because "I could not tell"
and "they are separate" must not lead to the same action. Issue `#73
<https://github.com/aiidateam/aiida-agents/issues/73>`_ is what the other
answer costs, and `ADR-11 </docs/adr/11-code-execution.md>`_ records why the
sandbox is a copy rather than a restricted view of the real thing.
"""

from __future__ import annotations

import enum
import shutil
import typing as t
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

from rich.filesize import decimal
from typing_extensions import assert_never

__all__ = [
    "FILEPATH_BACKENDS",
    "SUPPORTED_BACKENDS",
    "DatabaseLocation",
    "Location",
    "Overlap",
    "PathLocation",
    "ProfileStorage",
    "SharingProfile",
    "StorageConfig",
    "ConfigLike",
    "ProfileLike",
    "WritableConfigLike",
    "profile_storage",
    "profiles_sharing_storage",
    "register_profile",
    "copy_storage_directory",
    "postgres_copy_argv",
    "postgres_restore_argv",
    "postgres_sandbox_storage",
    "repository_path",
    "SOURCE_KEY",
    "sandbox_profile_dictionary",
    "sandbox_profile_exists",
    "sandbox_source",
    "sandbox_storage_root",
    "shares_storage",
    "storage_locations",
    "storage_overlap",
    "storage_size",
]

# Backends this module knows how to copy. Anything else is refused by name
# rather than attempted, because a copy that half worked would be worse than
# no sandbox: it would look like containment.
SUPPORTED_BACKENDS: t.Final = frozenset({"core.psql_dos", "core.sqlite_dos"})

# Backends whose whole storage is the one path under `filepath`, and so can be
# compared without loading them. Deliberately wider than SUPPORTED_BACKENDS: an
# imported archive cannot be copied into a sandbox, but it can be told apart
# from one, and until it could, every config holding an archive read as sharing
# storage with the sandbox.
FILEPATH_BACKENDS: t.Final = frozenset({"core.sqlite_dos", "core.sqlite_zip"})

# Read and written a chunk at a time so the copy can be reported as it goes.
# Large enough that the reporting costs nothing against gigabytes of packs.
_COPY_CHUNK_BYTES: t.Final = 4 * 1024 * 1024

#: A profile's ``storage.config`` as it comes out of ``config.json``: whatever
#: the backend put there, which is why every read of it is defensive.
StorageConfig: t.TypeAlias = dict[str, t.Any]


class ProfileLike(t.Protocol):
    """The three things this module reads off an AiiDA ``Profile``.

    A protocol rather than the class itself, because importing ``aiida`` at
    module scope is what the lazy imports below exist to avoid, and because a
    test that hands this a stub then has the stub checked rather than trusted.
    """

    @property
    def name(self) -> str: ...

    @property
    def storage_backend(self) -> str: ...

    @property
    def storage_config(self) -> StorageConfig | None: ...


class ConfigLike(t.Protocol):
    """The one thing this module reads off an AiiDA ``Config``."""

    @property
    def profiles(self) -> t.Sequence[ProfileLike]: ...


class WritableConfigLike(ConfigLike, t.Protocol):
    """A configuration that can also be added to, for :func:`register_profile`.

    Separate from :class:`ConfigLike` so that reading does not demand a config
    that can be written: the stubs the tests hand the read path have no reason
    to grow an ``add_profile``.
    """

    def add_profile(self, profile: t.Any) -> None: ...

    def store(self) -> None: ...


@dataclass(frozen=True)
class ProfileStorage:
    """A profile's storage: the backend that reads it, and where it keeps data.

    One value rather than two arguments, so that pairing one profile's backend
    with another's config is not something a caller can write.
    """

    backend: str
    config: StorageConfig


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


def sandbox_storage_root(sandbox_name: str) -> Path:
    """Directory holding the copy, kept beside AiiDA's own storage.

    Under the AiiDA config directory rather than a temporary directory,
    because the copy is expensive to make and is meant to be reused across
    sessions --- ``/tmp`` would silently rebuild it after every reboot.
    """
    from aiida.manage.configuration.settings import AiiDAConfigDir

    return Path(AiiDAConfigDir.get()) / "agents-sandbox" / sandbox_name


class Overlap(enum.Enum):
    """How one profile's storage compares with another's.

    ``UNKNOWN`` is acted on exactly like ``SHARED``. They are kept apart only so
    the user can be told which one they have, because "these are the same
    directory" and "there is not enough here to tell" call for different next
    steps. The second is the rarer, and comes from a plugin backend or a
    configuration written by something other than ``verdi``.
    """

    SEPARATE = "separate"
    SHARED = "shared"
    UNKNOWN = "unknown"


@dataclass(frozen=True, kw_only=True)
class SharingProfile:
    """A profile that could not be proved separate from the one being checked.

    ``overlap`` cannot be ``SEPARATE``: such a profile is not one of these at
    all. Keyword-only because ``name`` and ``backend`` are both strings, and
    swapping them names the wrong thing in a sentence that still reads.
    """

    name: str
    backend: str
    overlap: t.Literal[Overlap.SHARED, Overlap.UNKNOWN]

    def describe(self) -> str:
        """The reason, as a clause that follows the checked profile's name.

        Here rather than at the call sites because ``sandbox check``,
        ``sandbox teardown`` and ``doctor`` all report it, and three wordings
        of one finding would drift.
        """
        if self.overlap is Overlap.SHARED:
            return (
                f"shares storage with {self.name!r}, so deleting either "
                "destroys the other's data"
            )
        if self.overlap is Overlap.UNKNOWN:
            # Not "its backend cannot be read": the unreadable half is as often
            # an incomplete configuration, and it can be either profile's.
            return (
                f"cannot be told apart from {self.name!r} (storage backend "
                f"{self.backend!r}): the two configurations do not say enough "
                "to prove they are separate"
            )
        assert_never(self.overlap)  # pragma: no cover


def _storage_path(value: object) -> Path | None:
    """A configuration value read as a place on disk, or ``None`` if it is not one.

    The one place a configuration value becomes a comparable path. These come
    from ``config.json``, where a path is a string, so anything else fails
    closed; so does a relative path, which :meth:`Path.resolve` would answer
    against whatever directory the command was run from.
    """
    if not isinstance(value, str):
        return None
    # Postgres writes its repository as a `file://` URL, the SQLite backends a
    # plain path. Both name a place on disk, and the URL is decoded the way
    # `aiida.storage.psql_dos.backend.get_filepath_container` decodes it ---
    # `Path.as_uri` percent-encodes, so stripping the scheme by hand leaves
    # `/home/u/My%20Drive/...`, which never matches the directory it names.
    text = url2pathname(urlparse(value).path) if value.startswith("file://") else value
    path = Path(text)
    return path.resolve() if path.is_absolute() else None


def _database_location(config: StorageConfig) -> DatabaseLocation | None:
    """The database a Postgres profile uses, or ``None`` if it cannot be read.

    Wrong types fail closed rather than raising out of the comparison: these
    end up in a set, and an unhashable hostname took ``frozenset`` down with it.
    """
    name = config.get("database_name")
    if not isinstance(name, str) or not name:
        return None
    host = config.get("database_hostname") or "localhost"
    port = config.get("database_port") or 5432
    if not isinstance(host, str) or not isinstance(port, int):
        return None
    return DatabaseLocation(host=host, port=port, name=name)


@dataclass(frozen=True)
class PathLocation:
    """Storage kept somewhere on disk: a directory, or a single archive file."""

    path: Path

    def overlaps(self, other: Location) -> bool:
        """Whether destroying this place would destroy ``other`` as well.

        Containment counts, not only equality. ``sandbox teardown`` removes its
        root with :func:`shutil.rmtree`, so a profile whose storage sits inside
        another's goes with it, and comparing the two paths for equality would
        call them separate right up until the moment one deleted the other.
        """
        if not isinstance(other, PathLocation):
            return False
        return (
            self.path == other.path
            or self.path.is_relative_to(other.path)
            or other.path.is_relative_to(self.path)
        )


@dataclass(frozen=True)
class DatabaseLocation:
    """Storage kept in a database on a server, named the way a client reaches it."""

    host: str
    port: int
    name: str

    def overlaps(self, other: Location) -> bool:
        """Whether this is the same database. Two of them nest in no sense."""
        return self == other


#: Somewhere a profile keeps data. A Postgres profile has two, one of each.
Location: t.TypeAlias = PathLocation | DatabaseLocation


def storage_locations(storage: ProfileStorage) -> frozenset[Location]:
    """Everywhere ``verdi profile delete --delete-data`` would destroy.

    Read from the configuration, so two profiles can be compared without either
    being loadable. An empty set means "I could not tell", which
    :func:`shares_storage` treats as sharing.
    """
    # core.sqlite_dos keeps its database and container in one directory;
    # core.sqlite_zip is a single archive file, read-only in every respect but
    # this one --- `SqliteZipBackend.delete` unlinks it, so `--delete-data`
    # destroys it just the same. One kind of location for both, because two
    # profiles naming one path share it whatever each of them calls it.
    if storage.backend in FILEPATH_BACKENDS:
        path = _storage_path(storage.config.get("filepath"))
        return frozenset({PathLocation(path)}) if path is not None else frozenset()

    if storage.backend == "core.psql_dos":
        locations: set[Location] = set()
        database = _database_location(storage.config)
        if database is not None:
            locations.add(database)
        repository = _storage_path(storage.config.get("repository_uri"))
        if repository is not None:
            locations.add(PathLocation(repository))
        # A Postgres profile with neither is not something we can reason about.
        return frozenset(locations) if len(locations) == 2 else frozenset()

    return frozenset()


def profile_storage(profile: ProfileLike) -> ProfileStorage:
    """The storage of an AiiDA profile, as this module compares it.

    The one place AiiDA's profile object is read, so a missing
    ``storage_config`` becomes an empty one here rather than at four call sites.
    """
    return ProfileStorage(profile.storage_backend, profile.storage_config or {})


def storage_overlap(a: ProfileStorage, b: ProfileStorage) -> Overlap:
    """How two profiles' storage compares, and why when it is not separate.

    :func:`shares_storage` answers the same question yes-or-no. Callers that
    report it to a user need the distinction, and both come from here so there
    is one implementation of the rule.
    """
    locations_a = storage_locations(a)
    locations_b = storage_locations(b)
    if not locations_a or not locations_b:
        return Overlap.UNKNOWN
    shared = any(one.overlaps(other) for one in locations_a for other in locations_b)
    return Overlap.SHARED if shared else Overlap.SEPARATE


def shares_storage(a: ProfileStorage, b: ProfileStorage) -> bool:
    """Whether deleting one profile's data would destroy the other's.

    **Fails closed.** A backend this module does not understand, or a config
    missing the fields that identify its storage, returns True. Being unable to
    prove two profiles are separate is not evidence that they are, and the
    consequence of guessing wrong in the other direction is the bug this whole
    module exists to prevent.

    Sharing *either* the database or the repository counts. A sandbox with its
    own database but the real repository still loses the user their files.
    """
    return storage_overlap(a, b) is not Overlap.SEPARATE


def profiles_sharing_storage(config: ConfigLike, name: str) -> list[SharingProfile]:
    """Every other profile whose data would go with ``name``'s.

    The one implementation of this question. ``sandbox check`` asks it before
    passing, ``sandbox teardown`` asks it before deleting anything, ``doctor``
    asks it to fill in a row, and ``sandbox init`` asks it before registering.
    Four expressions of one rule would drift, and the drift would be a profile
    somebody deletes.

    Returns:
        One entry per profile that could not be proved separate, carrying which
        of the two reasons it was. Empty when the sandbox is genuinely
        self-contained.
    """
    profiles = {profile.name: profile for profile in config.profiles}
    target = profile_storage(profiles[name])
    found = []
    for other in config.profiles:
        if other.name == name:
            continue
        overlap = storage_overlap(target, profile_storage(other))
        if overlap is not Overlap.SEPARATE:
            found.append(
                SharingProfile(
                    name=other.name, backend=other.storage_backend, overlap=overlap
                )
            )
    return found


def register_profile(
    config: WritableConfigLike, name: str, dictionary: StorageConfig
) -> None:
    """Add a profile to the AiiDA configuration and persist it.

    Here rather than inline at the call sites so AiiDA's untyped configuration
    API is crossed in exactly one place.
    """
    from aiida.manage.configuration.profile import Profile

    config.add_profile(Profile(name, dictionary))
    config.store()


def storage_size(path: Path) -> int:
    """Total bytes of the files under ``path``.

    What a copy of it will write, and so what the disk must have room for and
    what a progress bar counts up to.

    Raises:
        FileNotFoundError: If ``path`` is not a directory.
    """
    if not path.is_dir():
        msg = f"storage directory {path} does not exist"
        raise FileNotFoundError(msg)
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _copy_file(
    source: str, target: str, *, progress: Callable[[int], None] | None
) -> None:
    """``shutil.copy2`` in chunks, reporting each one as it lands.

    ``copytree`` calls its copy function once per file, and a packed
    disk-objectstore is two files of several gigabytes, so counting files would
    leave the bar at nothing until it is nearly finished.
    """
    with open(source, "rb") as reader, open(target, "wb") as writer:
        while chunk := reader.read(_COPY_CHUNK_BYTES):
            writer.write(chunk)
            if progress is not None:
                progress(len(chunk))
    shutil.copystat(source, target)


def copy_storage_directory(
    source: Path, target: Path, *, progress: Callable[[int], None] | None = None
) -> None:
    """Copy a storage directory wholesale.

    The whole directory, not just ``database.sqlite``: the disk-objectstore
    container beside it holds every file the nodes refer to, and a database
    whose repository is missing answers half of the questions and raises on the
    rest.

    Args:
        source: The source profile's ``filepath``.
        target: Where the copy goes. Must not already exist.
        progress: Called with the number of bytes as each chunk lands, for a
            caller showing how far along the copy is.

    Raises:
        FileNotFoundError: If ``source`` is not a directory.
        FileExistsError: If ``target`` exists --- refreshing is an explicit
            teardown, never an overwrite of something already in use.
        OSError: If the filesystem has less room than the source occupies.
    """
    required = storage_size(source)
    if target.exists():
        msg = f"{target} already exists; tear the sandbox down before rebuilding it"
        raise FileExistsError(msg)

    target.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(target.parent).free
    if free < required:
        msg = (
            f"copying {source} needs {decimal(required)} and {target.parent} has "
            f"{decimal(free)} free"
        )
        raise OSError(msg)

    # Everything, verbatim. The first version skipped disk-objectstore's
    # `sandbox/` scratch directory on the grounds that a copy which never writes
    # has no use for in-flight writes. That was true and it did not matter:
    # `Container.is_initialised` checks the directory is *present*, so the copy
    # loaded as an uninitialised container and every query raised
    # `UnreachableStorage`. Copy the layout as it is and let the backend decide
    # what it needs.
    try:
        shutil.copytree(
            source, target, copy_function=partial(_copy_file, progress=progress)
        )
    except OSError:
        # A half-written copy is worse than none: it answers queries wrongly,
        # and the next attempt refuses it as something already in use.
        shutil.rmtree(target, ignore_errors=True)
        raise


def _connection_arguments(config: StorageConfig) -> tuple[str, ...]:
    host = config.get("database_hostname") or "localhost"
    port = config.get("database_port") or 5432
    user = config.get("database_username") or ""
    return ("--host", str(host), "--port", str(port), "--username", str(user))


def postgres_copy_argv(config: StorageConfig) -> tuple[str, ...]:
    """``pg_dump`` reading the profile's own database, for the pipeline
    that fills the sandbox's.

    ``pg_dump`` rather than ``CREATE DATABASE ... TEMPLATE``: the template form
    is much faster and refuses to run while any other session is connected to
    the source, which with a daemon running is most of the time.
    """
    connection = _connection_arguments(config)
    source = str(config.get("database_name") or "")
    return ("pg_dump", *connection, "--no-owner", "--no-privileges", source)


def postgres_restore_argv(config: StorageConfig, database: str) -> tuple[str, ...]:
    """``psql`` writing into the copy, reading what :func:`postgres_copy_argv` sends."""
    return ("psql", *_connection_arguments(config), "--quiet", database)


def repository_path(config: StorageConfig) -> Path | None:
    """Where a Postgres profile keeps its files, or ``None`` if it does not say.

    The other half of copying such a profile. ``pg_dump`` moves the database,
    which holds the *hashes* of every file; the objects those hashes name live
    in a disk-objectstore container on the filesystem, and a copy without it
    raises ``UnreachableStorage`` on the first node anybody opens.
    """
    return _storage_path(config.get("repository_uri"))


def postgres_sandbox_storage(
    source_config: StorageConfig, *, database: str, repository: Path
) -> ProfileStorage:
    """The sandbox's own Postgres storage, cloned from the source profile's.

    Server, port and credentials come across because the copy lives in the same
    server; only the database and the repository are its own.

    The repository is written as a ``file://`` URL because that is what
    aiida-core requires of ``repository_uri``: given a plain path,
    ``get_filepath_container`` raises ``ConfigurationError`` and the registered
    profile cannot be opened at all.
    """
    return ProfileStorage(
        "core.psql_dos",
        {
            **source_config,
            "database_name": database,
            "repository_uri": repository.as_uri(),
        },
    )


#: Where the sandbox records which profile it was copied from. A profile
#: called `agents-sandbox` says what it is and not what it holds, and with
#: several real profiles in a configuration that is the question anybody
#: reading `verdi profile list` actually has.
SOURCE_KEY: t.Final = "agents_sandbox_source"


def sandbox_source(profile: ProfileLike) -> str | None:
    """The profile a sandbox was copied from, if it recorded one.

    ``None`` for a sandbox built before this was written down, which is why
    every caller has to phrase it as an addition rather than an assumption.
    """
    dictionary = getattr(profile, "dictionary", None) or {}
    source = dictionary.get(SOURCE_KEY)
    return source if isinstance(source, str) else None


def sandbox_profile_dictionary(
    source: StorageConfig, storage: ProfileStorage, *, source_name: str = ""
) -> dict[str, t.Any]:
    """A profile configuration for the sandbox, cloned from the source profile's.

    Cloning rather than ``verdi profile setup``. That command initialises fresh
    storage and creates a default user, which is both unnecessary (the copy is
    already initialised, and already has the users) and impossible against a
    read-only role, which refuses the insert --- the reason the old setup path
    could not complete at all.

    Three fields are deliberately not carried over:

    ``PROFILE_UUID`` is regenerated. Two profiles sharing a UUID are two
    profiles AiiDA cannot tell apart.

    ``process_control`` is emptied. The sandbox runs nothing, so it needs no
    broker; pointing it at the source profile's queues would let generated code
    reach a daemon.

    ``options`` are dropped. They configure a working profile --- polling
    intervals, daemon timeouts --- and none of it applies to something that only
    answers queries.

    One field is added: the name of the profile this was copied from, so that
    everything downstream can say *which* profile the sandbox mirrors rather
    than leaving somebody with four profiles to guess.
    """
    return {
        "storage": {"backend": storage.backend, "config": dict(storage.config)},
        "process_control": {"backend": None, "config": None},
        "default_user_email": source.get("default_user_email"),
        "PROFILE_UUID": uuid.uuid4().hex,
        "test_profile": False,
        "options": {},
        SOURCE_KEY: source_name,
    }
