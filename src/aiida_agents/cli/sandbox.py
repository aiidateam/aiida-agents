"""``aiida-agents sandbox`` --- build and verify the profile generated code reads.

Three commands over one idea: the sandbox is a **disposable copy** of the
user's storage, never the storage itself.

``init`` makes the copy and registers a profile for it. ``check`` proves the
copy shares nothing with a real profile, which is the property that matters.
``teardown`` removes it, and can do so safely precisely because nothing else is
pointing at what it deletes.

There is deliberately no ``refresh``. Rebuilding is ``teardown`` then ``init``,
which is two commands that say what they do, where a single word promising to
bring the copy up to date would hide both the cost (the whole repository again)
and the loss (anything the sandbox holds that the source does not). The version
worth having is an incremental sync against the source rather than a fresh
copy, and that wants `verdi collab
<https://github.com/aiidateam/aiida-core/pull/7516>`_ underneath it.

The design this replaces pointed the sandbox at the same database through a
read-only role, and cost a maintainer his data when he deleted the sandbox
profile and agreed to delete its data (issue #73). ``check`` exists so that can
never be true again without somebody being told.
"""

from __future__ import annotations

import contextlib
import logging
import shutil
import typing as t
from collections.abc import Callable, Iterator
from pathlib import Path

import rich_click as click
from rich.filesize import decimal
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from rich.markup import escape

from aiida_agents.cli.output import console

logger = logging.getLogger(__name__)

__all__ = ["sandbox"]

#: Default name for the sandbox profile. Named for what it is, so
#: ``verdi profile list`` explains itself.
DEFAULT_PROFILE = "agents-sandbox"


def _agreed_to_copy(source: Path, target: Path, size: int, *, yes: bool) -> bool:
    """Say what the copy costs, and get agreement unless ``yes``.

    The copy is the whole repository, which on a real profile runs to
    gigabytes. Announcing it after the fact is how somebody finds out from
    ``df``, so the size, both paths and the room left go out before anything is
    written.
    """
    anchor = target
    while not anchor.exists():
        anchor = anchor.parent

    console.print(f"This copies [bold]{decimal(size)}[/bold]")
    # `soft_wrap` so a path longer than the terminal breaks where the terminal
    # breaks it, rather than being reflowed with a space in the middle of a
    # directory name, which is a path nobody can paste back.
    console.print(f"  from [cyan]{escape(str(source))}[/cyan]", soft_wrap=True)
    console.print(f"  to   [cyan]{escape(str(target))}[/cyan]", soft_wrap=True)
    console.print(f"  [dim]{decimal(shutil.disk_usage(anchor).free)} free there[/dim]")
    return yes or click.confirm("Proceed?")


@contextlib.contextmanager
def _copy_progress(size: int) -> Iterator[Callable[[int], None]]:
    """A progress bar counting bytes, yielding the callback that advances it.

    Bytes rather than files: a packed disk-objectstore is a handful of very
    large files, so a file count would sit at nothing and then finish.
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Copying storage", total=size)
        yield lambda advance: progress.advance(task, advance)


def _refuse_if_sharing(source: object, sandbox: object, profile: object) -> None:
    """Stop unless the sandbox's storage is provably the source's alone.

    Called before the copy as well as before registration: a source directory
    that contains the sandbox root would otherwise be copied into itself, and
    the refusal would arrive after the gigabytes had landed.
    """
    from aiida_agents.sandbox.copy import shares_storage

    if shares_storage(source, sandbox):  # type: ignore[arg-type]
        raise click.ClickException(
            "Refusing to build a sandbox that shares storage with "
            f"{profile.name!r}. Deleting it would take the real data with it."  # type: ignore[attr-defined]
        )


def _copy_postgres(config: dict[str, t.Any], database: str, *, yes: bool) -> bool:
    """Make the PostgreSQL copy, or say what could not be done.

    Returns:
        True when the copy is in place and the profile may be registered.
    """
    from aiida_agents.sandbox.postgres import (
        PostgresUnavailableError,
        copy_database,
        create_database,
    )

    host = config.get("database_hostname") or "localhost"
    console.print(
        f"This copies the database "
        f"[bold]{escape(str(config.get('database_name')))}[/bold]"
    )
    console.print(
        f"  to [cyan]{escape(database)}[/cyan] on [cyan]{escape(str(host))}[/cyan]",
        soft_wrap=True,
    )
    console.print(
        "  [dim]Creating a database needs a PostgreSQL superuser, so this may "
        "ask for a password.[/dim]"
    )
    if not yes and not click.confirm("Proceed?"):
        return False

    try:
        create_database(config, database)
        with console.status("Copying the database ..."):
            copy_database(config, database)
    except PostgresUnavailableError as exc:
        console.print(f"\n[yellow]![/yellow] {exc}")
        console.print(
            f"Create a database called [cyan]{escape(database)}[/cyan] on "
            f"[cyan]{escape(str(host))}[/cyan], owned by "
            f"[cyan]{escape(str(config.get('database_username')))}[/cyan], then run "
            "[cyan]aiida-agents sandbox init[/cyan] again: it will find the "
            "database and carry on from there.",
            soft_wrap=True,
        )
        return False

    console.print(f"[green]✓[/green] Copied into [cyan]{escape(database)}[/cyan]")
    return True


def _source_profile(name: str | None) -> object:
    from aiida.manage.configuration import get_config

    try:
        return get_config().get_profile(name)
    except Exception as exc:
        raise click.ClickException(f"Could not read profile: {exc}") from exc


@click.group("sandbox")
def sandbox() -> None:
    """Build and verify the disposable copy generated code runs against."""


@sandbox.command("init")
@click.option(
    "--profile",
    default=None,
    help="Source profile to copy. Default: the default profile.",
)
@click.option(
    "--sandbox-name",
    default=DEFAULT_PROFILE,
    show_default=True,
    help="Name for the new profile.",
)
@click.option("--yes", is_flag=True, help="Do not ask before copying.")
def init(profile: str | None, sandbox_name: str, yes: bool) -> None:
    """Copy the profile's storage and register a sandbox profile for it.

    On SQLite this is done for you: the storage directory is copied and the
    profile registered, with nothing to paste and no superuser involved.

    On PostgreSQL the database is copied through a connection that may create
    databases, and the files beside it are copied from disk. Where no such
    connection can be found, the commands are printed for you to run instead.
    """
    from aiida.manage.configuration import get_config

    from aiida_agents.sandbox.copy import (
        SUPPORTED_BACKENDS,
        ProfileStorage,
        copy_storage_directory,
        postgres_sandbox_storage,
        register_profile,
        repository_path,
        sandbox_profile_dictionary,
        sandbox_storage_root,
        storage_size,
    )

    config = get_config()
    source = _source_profile(profile)
    backend = source.storage_backend  # type: ignore[attr-defined]
    storage_config = source.storage_config or {}  # type: ignore[attr-defined]

    if backend not in SUPPORTED_BACKENDS:
        raise click.ClickException(
            f"Profile {source.name!r} uses {backend!r}, which this cannot copy. "  # type: ignore[attr-defined]
            f"Supported: {', '.join(sorted(SUPPORTED_BACKENDS))}."
        )
    if sandbox_name in {existing.name for existing in config.profiles}:
        raise click.ClickException(
            f"Profile {sandbox_name!r} already exists. Remove it with `sandbox "
            "teardown` first, then run this again to rebuild it."
        )

    root = sandbox_storage_root(sandbox_name)

    if backend == "core.sqlite_dos":
        storage = Path(storage_config["filepath"])
        target = root / "storage"
        new_storage = ProfileStorage(backend, {"filepath": str(target)})
        # Before the copy, not after. A source directory that contains the
        # sandbox root would be copied into itself, and the check below would
        # then refuse to register what had just been written to disk.
        _refuse_if_sharing(ProfileStorage(backend, storage_config), new_storage, source)
        try:
            size = storage_size(storage)
            if not _agreed_to_copy(storage, target, size, yes=yes):
                return
            with _copy_progress(size) as advance:
                copy_storage_directory(storage, target, progress=advance)
        except (FileNotFoundError, FileExistsError, OSError) as exc:
            raise click.ClickException(str(exc)) from exc
    else:
        sandbox_database = (
            f"{storage_config.get('database_name', 'aiida')}_agents_sandbox"
        )
        repository = root / "repository"
        new_storage = postgres_sandbox_storage(
            storage_config, database=sandbox_database, repository=repository
        )
        # Before the copy here too: a check that arrives afterwards has
        # already created a database and moved the data into it.
        _refuse_if_sharing(ProfileStorage(backend, storage_config), new_storage, source)
        # The database holds the hash of every file and none of the files. They
        # live in a container on disk, and a copy without it raises
        # `UnreachableStorage` on the first node anybody opens. Resolved before
        # the database is created, so a profile that does not say where its
        # files are fails here rather than after a dump has been restored.
        source_repository = repository_path(storage_config)
        if source_repository is None:
            raise click.ClickException(
                f"Profile {source.name!r} does not say where its files are: "  # type: ignore[attr-defined]
                "its `repository_uri` is missing or not a path."
            )
        if not _database_exists(new_storage.config) and not _copy_postgres(
            storage_config, sandbox_database, yes=yes
        ):
            return
        if not repository.exists():
            try:
                # Announced and agreed to like the SQLite copy: this is the
                # large half of a PostgreSQL profile, and `_copy_postgres`
                # above asked about the database only. Declining here leaves
                # the copied database, which a second `init` reuses.
                size = storage_size(source_repository)
                if not _agreed_to_copy(source_repository, repository, size, yes=yes):
                    return
                with _copy_progress(size) as advance:
                    copy_storage_directory(
                        source_repository, repository, progress=advance
                    )
            except (FileNotFoundError, FileExistsError, OSError) as exc:
                raise click.ClickException(str(exc)) from exc

    register_profile(
        config,
        sandbox_name,
        sandbox_profile_dictionary(
            source.dictionary,  # type: ignore[attr-defined]
            new_storage,
            source_name=source.name,  # type: ignore[attr-defined]
        ),
    )
    console.print(
        f"[green]✓[/green] Registered [cyan]{escape(sandbox_name)}[/cyan], a copy of "
        f"[cyan]{escape(str(source.name))}[/cyan]"  # type: ignore[attr-defined]
    )
    console.print(
        f"  Verify it with [cyan]aiida-agents sandbox check "
        f"--profile {escape(sandbox_name)}[/cyan]"
    )


def _database_exists(config: dict[str, object]) -> bool:
    """Whether the Postgres copy has been made yet.

    ``init`` has nothing more to do until the copy exists, so it checks rather
    than registering a profile pointing at a database that is not there, and
    rerunning after the printed commands picks up where it left off.
    """
    try:
        from sqlalchemy import URL, create_engine

        # Built rather than formatted: a password with an `@` or a `/` in it
        # turns a hand-written URL into a different host and database, and the
        # answer here would be "no such database" for one that exists.
        url = URL.create(
            "postgresql",
            username=str(config.get("database_username") or ""),
            password=str(config.get("database_password") or ""),
            host=str(config.get("database_hostname") or "localhost"),
            port=int(str(config.get("database_port") or 5432)),
            database=str(config.get("database_name") or ""),
        )
        create_engine(url).connect().close()
    except Exception:
        return False
    return True


@sandbox.command("check")
@click.option(
    "--profile", default=DEFAULT_PROFILE, show_default=True, help="Profile to verify."
)
def check(profile: str) -> None:
    """Verify the sandbox shares no storage with any real profile.

    This is the check that matters, and the only one: a sandbox sharing a
    database or a repository with a real profile is a profile whose deletion
    destroys somebody's work. Being able to write to it is not a fault ---
    a scratch profile you can iterate in is what it is for.
    """
    from aiida.manage.configuration import get_config

    from aiida_agents.sandbox.copy import (
        Overlap,
        profiles_sharing_storage,
        sandbox_source,
    )

    config = get_config()
    try:
        target = config.get_profile(profile)
    except Exception as exc:
        raise click.ClickException(f"No profile {profile!r}: {exc}") from exc

    failures = profiles_sharing_storage(config, profile)
    if failures:
        # A proven overlap and a config that could not be read both stop the
        # command, and saying which one it was is the difference between "your
        # sandbox is pointed at your own storage" and "some profile here does
        # not say where its storage is". Only the first is about their data.
        proven = any(entry.overlap is Overlap.SHARED for entry in failures)
        verdict = "must not be used" if proven else "cannot be cleared for use"
        console.print(f"[red]✗[/red] {escape(repr(profile))} {verdict} as a sandbox:")
        for entry in failures:
            console.print(f"    {escape(entry.describe())}")
        # Both remedies when there is one of each: `init` compares the copy
        # against its source alone, so a rebuild clears the overlap and leaves
        # the unreadable profile failing the next check for the same reason.
        if proven:
            console.print(
                f"  Delete it with [cyan]verdi profile delete --keep-data "
                f"{escape(profile)}[/cyan], then rebuild with [cyan]aiida-agents "
                "sandbox init[/cyan]."
            )
        if any(entry.overlap is Overlap.UNKNOWN for entry in failures):
            console.print(
                "  What cannot be read is treated as sharing until it can be ruled out."
            )
        raise SystemExit(1)

    origin = sandbox_source(target)
    copied = f", a copy of {escape(repr(origin))}" if origin else ""
    console.print(
        f"[green]✓[/green] {escape(repr(profile))}{copied} shares no storage with "
        "any profile"
    )


@sandbox.command("teardown")
@click.option(
    "--profile", default=DEFAULT_PROFILE, show_default=True, help="Sandbox to remove."
)
@click.option("--yes", is_flag=True, help="Do not ask for confirmation.")
def teardown(profile: str, yes: bool) -> None:
    """Remove the sandbox profile and the copy it points at.

    Safe by construction rather than by care: ``init`` refuses to register a
    sandbox that shares storage with a real profile, so there is nothing here
    for this to delete but the copy.
    """
    from aiida.manage.configuration import get_config

    from aiida_agents.sandbox.copy import (
        profiles_sharing_storage,
        sandbox_source,
        sandbox_storage_root,
        storage_size,
    )

    config = get_config()
    if profile not in {existing.name for existing in config.profiles}:
        console.print(f"No profile {profile!r}; nothing to tear down.")
        return

    sharing = profiles_sharing_storage(config, profile)
    if sharing:
        reasons = "\n".join(f"  {entry.describe()}" for entry in sharing)
        raise click.ClickException(
            f"Refusing to delete {profile!r} and its storage:\n{reasons}\n"
            "Remove the profile by hand if you are certain."
        )

    # Named rather than described. "Its copy" reads as a copy of the profile,
    # which is the one thing this does not delete: the profile entry is the
    # cheap half, and the storage it points at is what takes the disk with it.
    target = config.get_profile(profile)
    origin = sandbox_source(target)
    root = sandbox_storage_root(profile)
    copied = f", a copy of [cyan]{escape(origin)}[/cyan]," if origin else ","
    console.print(
        f"This removes the profile [cyan]{escape(profile)}[/cyan]{copied} and"
    )
    if root.is_dir():
        console.print(
            f"  [bold]{decimal(storage_size(root))}[/bold] of copied storage at "
            f"[cyan]{escape(str(root))}[/cyan]",
            soft_wrap=True,
        )
    else:
        console.print(
            f"  nothing on disk: [cyan]{escape(str(root))}[/cyan] is not there"
        )
    console.print(
        "  [dim]Anything the sandbox holds and the source does not goes with it.[/dim]"
    )
    if not yes and not click.confirm("Delete?"):
        return

    # AiiDA's own deletion, which drops a PostgreSQL database and removes a
    # SQLite directory without this having to know which. Asking it to delete
    # the storage is the call that cost a maintainer his database in #73, and
    # what has changed since is not the caution around the call: it is that the
    # profile no longer points at the user's storage. It points at a copy that
    # shares nothing with it, which is the property `check` proves and the one
    # the check above insists on before this line runs.
    storage_removed = True
    try:
        config.delete_profile(profile, delete_storage=True)
    except Exception as exc:
        # A database that cannot be reached, a directory already gone. The
        # profile still has to go, and what was left behind has to be said.
        logger.info("aiida could not delete the storage of %r", profile, exc_info=True)
        config.delete_profile(profile, delete_storage=False)
        storage_removed = False
        console.print(
            f"[yellow]![/yellow] Removed the profile, but not its storage: {exc}"
        )
    config.store()  # type: ignore[no-untyped-call]
    # The wrapper directory the copy lived in, which aiida does not know about.
    shutil.rmtree(root, ignore_errors=True)
    # Only when it went. The line above already reports the other outcome, and
    # a PostgreSQL database left behind is not removed by taking its directory.
    if storage_removed:
        console.print(
            f"[green]✓[/green] Removed {escape(repr(profile))} and the storage it copied"
        )
