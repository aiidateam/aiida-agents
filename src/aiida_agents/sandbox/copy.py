"""A disposable copy of a profile's storage, for generated code to read.

The sandbox used to be a second profile pointing at **the same** database and
repository as the user's own, reached through a read-only Postgres role. That
choice weighed a shared database against an *empty* one --- an empty database
cannot answer "which structures did I relax last month" --- and missed that a
**copy** is neither.

The cost of missing it was a maintainer's database. Deleting the sandbox
profile and agreeing to delete its data deleted the storage underneath both
profiles (`#73 <https://github.com/aiidateam/aiida-agents/issues/73>`_). A
read-only role is no defence: the destructive command is run by the user, as
themselves, against a profile they were told was disposable.

So the rule this module exists to enforce is:

    **A sandbox profile must never share deletable storage with a real one.**

:func:`shares_storage` is that rule as a function, and it fails closed --- two
profiles it cannot compare are treated as sharing, because "I could not tell"
and "they are separate" must not lead to the same action.

Copying rather than restricting also settles three other things at once:

* Setup needs no write to the source. The old flow printed a
  ``verdi profile setup`` command that could never complete, because that
  command creates a default user and the read-only role refuses the insert.
  A copy is registered by cloning the source profile's own configuration,
  which already describes initialised storage.
* SQLite gets containment for the first time. It has no roles and no
  ``GRANT``, so the old design simply had nothing to offer the default
  ``verdi presto`` backend; a copy works the same way for both.
* The read-only role becomes belt-and-braces rather than the whole mechanism.
  A write that reaches the copy costs a refresh, not someone's data.
"""

from __future__ import annotations

import shutil
import typing as t
import uuid
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "SandboxStorage",
    "profiles_sharing_storage",
    "register_profile",
    "copy_sqlite_storage",
    "postgres_copy_commands",
    "sandbox_profile_dictionary",
    "sandbox_storage_root",
    "shares_storage",
    "storage_locations",
]

#: Backends this module knows how to copy. Anything else is refused by name
#: rather than attempted, because a copy that half worked would be worse than
#: no sandbox: it would look like containment.
SUPPORTED_BACKENDS = frozenset({"core.psql_dos", "core.sqlite_dos"})


@dataclass(frozen=True)
class SandboxStorage:
    """Where a sandbox's copied storage lives."""

    backend: str
    config: dict[str, t.Any]


def sandbox_storage_root(sandbox_name: str) -> Path:
    """Directory holding the copy, kept beside AiiDA's own storage.

    Under the AiiDA config directory rather than a temporary directory,
    because the copy is expensive to make and is meant to be reused across
    sessions --- ``/tmp`` would silently rebuild it after every reboot.
    """
    from aiida.manage.configuration.settings import AiiDAConfigDir

    return Path(AiiDAConfigDir.get()) / "agents-sandbox" / sandbox_name


def storage_locations(backend: str, config: dict[str, t.Any]) -> frozenset[str]:
    """Everything ``verdi profile delete --delete-data`` would destroy.

    Canonical strings rather than paths so that two profiles can be compared
    without either being loadable. An empty set means "I could not tell", which
    :func:`shares_storage` treats as sharing.
    """
    if backend == "core.sqlite_dos":
        filepath = config.get("filepath")
        return (
            frozenset({f"dir:{Path(filepath).resolve()}"}) if filepath else frozenset()
        )

    if backend == "core.psql_dos":
        locations = set()
        database = config.get("database_name")
        if database:
            host = config.get("database_hostname") or "localhost"
            port = config.get("database_port") or 5432
            locations.add(f"pg://{host}:{port}/{database}")
        repository = config.get("repository_uri")
        if repository:
            locations.add(
                f"dir:{Path(str(repository).removeprefix('file://')).resolve()}"
            )
        # A Postgres profile with neither is not something we can reason about.
        return frozenset(locations) if len(locations) == 2 else frozenset()

    return frozenset()


def shares_storage(
    backend_a: str,
    config_a: dict[str, t.Any],
    backend_b: str,
    config_b: dict[str, t.Any],
) -> bool:
    """Whether deleting one profile's data would destroy the other's.

    **Fails closed.** A backend this module does not understand, or a config
    missing the fields that identify its storage, returns True. Being unable to
    prove two profiles are separate is not evidence that they are, and the
    consequence of guessing wrong in the other direction is the bug this whole
    module exists to prevent.

    Sharing *either* the database or the repository counts. A sandbox with its
    own database but the real repository still loses the user their files.
    """
    locations_a = storage_locations(backend_a, config_a)
    locations_b = storage_locations(backend_b, config_b)
    if not locations_a or not locations_b:
        return True
    return bool(locations_a & locations_b)


def profiles_sharing_storage(config: t.Any, name: str) -> list[str]:
    """Every other profile whose data would go with ``name``'s.

    The one implementation of this question. ``sandbox check`` asks it before
    passing, ``sandbox teardown`` asks it before deleting anything, ``doctor``
    asks it to fill in a row, and ``sandbox init`` asks it before registering.
    Four expressions of one rule would drift, and the drift would be a profile
    somebody deletes.

    Returns:
        Profile names, empty when the sandbox is genuinely self-contained.
    """
    profiles = {profile.name: profile for profile in config.profiles}
    target = profiles[name]
    return [
        other.name
        for other in config.profiles
        if other.name != name
        and shares_storage(
            target.storage_backend,
            target.storage_config or {},
            other.storage_backend,
            other.storage_config or {},
        )
    ]


def register_profile(config: t.Any, name: str, dictionary: dict[str, t.Any]) -> None:
    """Add a profile to the AiiDA configuration and persist it.

    Here rather than inline at the call sites so AiiDA's untyped configuration
    API is crossed in exactly one place.
    """
    from aiida.manage.configuration.profile import Profile

    config.add_profile(Profile(name, dictionary))
    config.store()


def copy_sqlite_storage(source: Path, target: Path) -> None:
    """Copy a ``core.sqlite_dos`` storage directory wholesale.

    The whole directory, not just ``database.sqlite``: the disk-objectstore
    container beside it holds every file the nodes refer to, and a database
    whose repository is missing answers half of the questions and raises on the
    rest.

    Args:
        source: The source profile's ``filepath``.
        target: Where the copy goes. Must not already exist.

    Raises:
        FileNotFoundError: If ``source`` is not a directory.
        FileExistsError: If ``target`` exists --- refreshing is an explicit
            teardown, never an overwrite of something already in use.
    """
    if not source.is_dir():
        msg = f"storage directory {source} does not exist"
        raise FileNotFoundError(msg)
    if target.exists():
        msg = f"{target} already exists; tear the sandbox down before rebuilding it"
        raise FileExistsError(msg)

    target.parent.mkdir(parents=True, exist_ok=True)
    # Everything, verbatim. The first version skipped disk-objectstore's
    # `sandbox/` scratch directory on the grounds that a copy which never writes
    # has no use for in-flight writes. That was true and it did not matter:
    # `Container.is_initialised` checks the directory is *present*, so the copy
    # loaded as an uninitialised container and every query raised
    # `UnreachableStorage`. Copy the layout as it is and let the backend decide
    # what it needs.
    shutil.copytree(source, target)


def postgres_copy_commands(
    config: dict[str, t.Any], sandbox_database: str
) -> list[tuple[str, str]]:
    """The shell commands that copy a Postgres database, with explanations.

    Printed rather than run for the same reason the role SQL is: creating a
    database needs a privilege the profile's own user usually does not have, and
    asking for a superuser connection in order to configure a safety feature is
    a worse bargain than showing somebody two commands they can read.

    ``pg_dump | psql`` rather than ``CREATE DATABASE ... TEMPLATE ...``. The
    template form is much faster, and fails outright while any other session is
    connected to the source --- which, with a daemon running or a ``verdi
    shell`` open, is most of the time.

    Returns:
        ``(explanation, command)`` pairs, in the order they must be run.
    """
    host = config.get("database_hostname") or "localhost"
    port = config.get("database_port") or 5432
    user = config.get("database_username") or ""
    source = config.get("database_name") or ""
    connection = f"--host {host} --port {port} --username {user}"

    return [
        (
            "Create the database the copy will live in",
            f'createdb {connection} "{sandbox_database}"',
        ),
        (
            "Copy the data across (works while the source profile is in use)",
            f'pg_dump {connection} --no-owner --no-privileges "{source}" '
            f'| psql {connection} --quiet "{sandbox_database}"',
        ),
    ]


def sandbox_profile_dictionary(
    source: dict[str, t.Any], storage: SandboxStorage
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
    """
    return {
        "storage": {"backend": storage.backend, "config": dict(storage.config)},
        "process_control": {"backend": None, "config": None},
        "default_user_email": source.get("default_user_email"),
        "PROFILE_UUID": uuid.uuid4().hex,
        "test_profile": False,
        "options": {},
    }
