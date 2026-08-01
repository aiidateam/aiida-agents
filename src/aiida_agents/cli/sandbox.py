"""``aiida-agents sandbox`` --- set up and verify the read-only execution profile.

Two commands, and the second is the one that matters. ``init`` writes out the
SQL and the ``verdi`` invocation for a profile that reaches the same database
through a role that cannot write. ``check`` asks Postgres whether that actually
came out right, which is the only way anyone should be satisfied that it did.
"""

from __future__ import annotations

import secrets

import rich_click as click

from aiida_agents.cli.output import console

__all__ = ["sandbox"]

#: Default name for the read-only role and the profile that uses it. Named for
#: what it is, so ``verdi profile list`` explains itself.
DEFAULT_ROLE = "aiida_agents_readonly"
DEFAULT_PROFILE = "agents-sandbox"


@click.group("sandbox")
def sandbox() -> None:
    """Set up and verify the read-only profile generated code runs against."""


@sandbox.command("init")
@click.option(
    "--profile",
    default=None,
    help="Source profile to mirror. Default: the default profile.",
)
@click.option(
    "--role", default=DEFAULT_ROLE, show_default=True, help="Postgres role to create."
)
@click.option(
    "--sandbox-name",
    default=DEFAULT_PROFILE,
    show_default=True,
    help="Name for the new profile.",
)
def init(profile: str | None, role: str, sandbox_name: str) -> None:
    """Print the SQL and `verdi` command that create the read-only profile.

    Deliberately prints rather than runs. Creating a role needs Postgres
    superuser, and asking for superuser in order to configure a safety feature
    is a worse bargain than showing you three statements you can read first.
    Every value is taken from the source profile, so there is nothing to look
    up or mistype.
    """
    from aiida.manage.configuration import get_config

    from aiida_agents.sandbox.setup import profile_setup_command, readonly_role_sql

    config = get_config()
    try:
        source = config.get_profile(profile)
    except Exception as exc:
        raise click.ClickException(f"Could not read profile: {exc}") from exc

    storage = source.storage_config or {}
    database = storage.get("database_name")
    if not database:
        raise click.ClickException(
            f"Profile {source.name!r} does not use a PostgreSQL storage backend, "
            "so there is no role to restrict."
        )

    password = secrets.token_urlsafe(24)

    console.print(f"[bold]Sandbox for profile [cyan]{source.name}[/cyan][/bold]\n")
    console.print("[bold]1.[/bold] Run this as a Postgres superuser:\n")
    console.print(f"[dim]{readonly_role_sql(database, role, password)}[/dim]\n")
    console.print("[bold]2.[/bold] Register the profile:\n")
    console.print(
        f"[dim]{profile_setup_command(storage, role, password, sandbox_name)}[/dim]\n"
    )
    console.print("[bold]3.[/bold] Confirm it really cannot write:\n")
    console.print(f"[dim]aiida-agents sandbox check --profile {sandbox_name}[/dim]\n")
    console.print(
        "[yellow]Do not skip step 3.[/yellow] Nothing else in this system can "
        "tell a read-only profile from a writable one."
    )


@sandbox.command("check")
@click.option(
    "--profile", default=DEFAULT_PROFILE, show_default=True, help="Profile to verify."
)
def check(profile: str) -> None:
    """Verify a profile's database role cannot write.

    Asks Postgres whether the connected role holds INSERT on the node table.
    That is a read-only question, so running this proves the point without
    creating anything.

    Worth re-running after a database migration: a grant that was correct in
    March is not self-maintaining.
    """
    from aiida_agents.sandbox.setup import verify_read_only

    result = verify_read_only(profile)
    if result.read_only:
        console.print(f"[green]✓[/green] {result.detail}")
        return

    console.print(f"[red]✗[/red] {result.detail}")
    raise SystemExit(1)
