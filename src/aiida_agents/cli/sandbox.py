"""``aiida-agents sandbox`` --- build and verify the profile generated code reads.

Four commands over one idea: the sandbox is a **disposable copy** of the user's
storage, never the storage itself.

``init`` makes the copy and registers a profile for it. ``check`` proves the
copy shares nothing with a real profile, which is the property that matters.
``refresh`` rebuilds it when it has drifted. ``teardown`` removes it, and can
do so safely precisely because nothing else is pointing at what it deletes.

The design this replaces pointed the sandbox at the same database through a
read-only role, and cost a maintainer his data when he deleted the sandbox
profile and agreed to delete its data (issue #73). ``check`` exists so that can
never be true again without somebody being told.
"""

from __future__ import annotations

import secrets
import shutil

import rich_click as click

from rich.markup import escape

from aiida_agents.cli.output import console

__all__ = ["sandbox"]

#: Default name for the read-only role and the profile that uses it. Named for
#: what it is, so ``verdi profile list`` explains itself.
DEFAULT_ROLE = "aiida_agents_readonly"
DEFAULT_PROFILE = "agents-sandbox"


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
@click.option(
    "--role",
    default=DEFAULT_ROLE,
    show_default=True,
    help="PostgreSQL role to read the copy through. Optional hardening.",
)
def init(profile: str | None, sandbox_name: str, role: str) -> None:
    """Copy the profile's storage and register a sandbox profile for it.

    On SQLite this is done for you: the storage directory is copied and the
    profile registered, with nothing to paste and no superuser involved.

    On PostgreSQL the data lives in the server, so copying it needs a role that
    may create databases. The commands are printed for you to run, then rerun
    this with the copy in place.
    """
    from aiida.manage.configuration import get_config

    from aiida_agents.sandbox.copy import (
        SUPPORTED_BACKENDS,
        ProfileStorage,
        copy_sqlite_storage,
        postgres_copy_commands,
        register_profile,
        sandbox_profile_dictionary,
        sandbox_storage_root,
        shares_storage,
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
            f"Profile {sandbox_name!r} already exists. Use `sandbox refresh` to "
            "rebuild it, or `sandbox teardown` to remove it first."
        )

    root = sandbox_storage_root(sandbox_name)

    if backend == "core.sqlite_dos":
        from pathlib import Path

        target = root / "storage"
        console.print(f"Copying storage to [cyan]{target}[/cyan] ...")
        try:
            copy_sqlite_storage(Path(storage_config["filepath"]), target)
        except (FileNotFoundError, FileExistsError, OSError) as exc:
            raise click.ClickException(str(exc)) from exc
        new_storage = ProfileStorage(backend, {"filepath": str(target)})
    else:
        sandbox_database = (
            f"{storage_config.get('database_name', 'aiida')}_agents_sandbox"
        )
        repository = root / "repository"
        console.print(
            "[bold]The copy lives in PostgreSQL, so these two run as a role that "
            "may create databases:[/bold]\n"
        )
        for explanation, command in postgres_copy_commands(
            storage_config, sandbox_database
        ):
            console.print(f"[dim]# {explanation}[/dim]")
            console.print(f"[dim]{command}[/dim]\n")
        console.print(
            "[bold]Optional, as a second layer:[/bold] read the copy through a "
            "role that cannot write it.\n"
        )
        from aiida_agents.sandbox.setup import readonly_role_sql

        password = secrets.token_urlsafe(24)
        console.print(
            f"[dim]{readonly_role_sql(sandbox_database, role, password)}[/dim]\n"
        )
        console.print(
            f"Then rerun [cyan]aiida-agents sandbox init --sandbox-name "
            f"{sandbox_name}[/cyan] to register the profile."
        )
        new_storage = ProfileStorage(
            backend,
            {
                **storage_config,
                "database_name": sandbox_database,
                "repository_uri": str(repository),
            },
        )
        if not _database_exists(new_storage.config):
            return

    # The whole point of the exercise, checked rather than assumed.
    if shares_storage(ProfileStorage(backend, storage_config), new_storage):
        raise click.ClickException(
            "Refusing to register a sandbox that shares storage with "
            f"{source.name!r}. Deleting it would take the real data with it."  # type: ignore[attr-defined]
        )

    register_profile(
        config,
        sandbox_name,
        sandbox_profile_dictionary(source.dictionary, new_storage),  # type: ignore[attr-defined]
    )
    console.print(f"[green]✓[/green] Registered profile [cyan]{sandbox_name}[/cyan]")
    console.print(
        f"  Verify it with [cyan]aiida-agents sandbox check "
        f"--profile {sandbox_name}[/cyan]"
    )


def _database_exists(config: dict[str, object]) -> bool:
    """Whether the Postgres copy has been made yet.

    ``init`` prints the copy commands and then has nothing more to do until
    somebody runs them, so it checks rather than registering a profile pointing
    at a database that is not there.
    """
    try:
        from sqlalchemy import create_engine

        url = (
            f"postgresql://{config.get('database_username')}:"
            f"{config.get('database_password')}@{config.get('database_hostname')}:"
            f"{config.get('database_port')}/{config.get('database_name')}"
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

    This is the check that matters. A sandbox sharing a database or a
    repository with a real profile is a profile whose deletion destroys
    somebody's work, and no amount of read-only role makes that safe.

    Where a read-only role is also in use, it is verified too --- but as a
    second layer, not the mechanism.
    """
    from aiida.manage.configuration import get_config

    from aiida_agents.sandbox.copy import Overlap, profiles_sharing_storage
    from aiida_agents.sandbox.setup import verify_read_only

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

    console.print(f"[green]✓[/green] {profile!r} shares no storage with any profile")

    if target.storage_backend == "core.psql_dos":
        result = verify_read_only(profile)
        mark = "[green]✓[/green]" if result.read_only else "[yellow]![/yellow]"
        console.print(f"{mark} {result.detail}")


@sandbox.command("refresh")
@click.option(
    "--profile", default=DEFAULT_PROFILE, show_default=True, help="Sandbox to rebuild."
)
@click.option(
    "--source", default=None, help="Profile to copy from. Default: the default profile."
)
@click.pass_context
def refresh(ctx: click.Context, profile: str, source: str | None) -> None:
    """Rebuild the copy from the source profile.

    The copy drifts the moment the user runs anything, which is the accepted
    cost of not sharing their storage. This is how it catches up.
    """
    ctx.invoke(teardown, profile=profile, yes=True)
    ctx.invoke(init, profile=source, sandbox_name=profile, role=DEFAULT_ROLE)


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
        sandbox_storage_root,
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

    if not yes and not click.confirm(f"Delete profile {profile!r} and its copy?"):
        return

    config.delete_profile(profile, delete_storage=False)
    config.store()  # type: ignore[no-untyped-call]
    shutil.rmtree(sandbox_storage_root(profile), ignore_errors=True)
    console.print(f"[green]✓[/green] Removed {profile!r} and its copied storage")
