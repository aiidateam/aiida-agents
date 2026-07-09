"""The ``aiida-agents config`` command group and the data behind it.

``config show`` renders each effective setting with its env var and source,
``config init`` scaffolds a ``.env`` template, and ``config path`` reports where
configuration is read from. The row/template builders (:func:`_config_rows`,
:func:`_env_template`) derive everything from ``_SETTINGS_GROUPS`` so the surface
never drifts from the settings themselves.
"""

from __future__ import annotations

import os
from pathlib import Path

import rich_click as click
from pydantic import ValidationError
from pydantic_settings import BaseSettings
from rich.table import Table

from aiida_agents._settings import (
    _SETTINGS_GROUPS,
    ModelSettings,
    _dotenv_keys,
    find_unrecognized_settings,
)
from aiida_agents.cli._guards import _report_setting_problems
from aiida_agents.cli.output import console
from aiida_agents.cli.session import _resolve_model_settings


def _env_var_for(cls: type[BaseSettings], field_name: str) -> str:
    """Env var that controls ``field_name`` on a settings group.

    A field maps to ``<env_prefix><field_name>`` unless it carries an explicit
    ``validation_alias`` (e.g. ``OLLAMA_BASE_URL``), matching how the settings
    groups actually read the environment.
    """
    alias = cls.model_fields[field_name].validation_alias
    if isinstance(alias, str):
        return alias
    prefix = cls.model_config.get("env_prefix", "")
    return f"{prefix}{field_name}".upper()


def _group_label(cls: type[BaseSettings]) -> str:
    """Short label for a settings group (``ModelSettings`` -> ``model``).

    Disambiguates fields that share a name across groups (both the model and
    Ollama settings expose ``base_url``), so a rendered table or template never
    shows one label for two different settings.
    """
    return cls.__name__.removesuffix("Settings").lower()


def _display_value(obj: BaseSettings, field_name: str) -> str:
    """User-facing string for a field's effective value, masking secrets."""
    value = getattr(obj, field_name)
    if field_name.endswith("_key"):
        return "set" if value and value != "api-key-not-set" else "unset"
    if value is None:
        return "unset"
    return str(value)


def _env_template() -> str:
    """A commented ``.env`` scaffold listing every recognised setting.

    Iterates :data:`_SETTINGS_GROUPS` (the same source the typo-detector and
    ``config show`` use), so the keys are correct and complete by construction: a
    generated template can never teach a typo, and no setting is silently absent.
    Every line is commented out and carries the field's default; secrets are
    listed with an empty value, never a default. Uncomment and edit to override.
    """
    lines = [
        "# aiida-agents configuration.",
        "# Uncomment a line and set its value to override the default.",
        "",
    ]
    for cls in _SETTINGS_GROUPS:
        lines.append(f"# --- {_group_label(cls)} ---")
        for field_name, field in cls.model_fields.items():
            env_var = _env_var_for(cls, field_name)
            # A field's description goes on its own comment line(s) above the key,
            # so uncommenting the key never drags an inline comment into the value.
            if field.description:
                lines.extend(f"#   {line}" for line in field.description.splitlines())
            if field_name.endswith("_key") or field.is_required():
                # Secrets get no default; a required field has none to show.
                lines.append(f"# {env_var}=")
            else:
                default = field.default
                lines.append(f"# {env_var}={'' if default is None else default}")
        lines.append("")
    return "\n".join(lines)


def _invalid_fields(exc: ValidationError) -> dict[str, str]:
    """Field name -> concise reason for each field-level error in ``exc``.

    Whole-model errors (a cross-field ``model_validator``, which carry no field
    location) are omitted; those fields still render as ``(invalid)`` and the
    full reason is available from ``aiida-agents check``.
    """
    return {
        str(err["loc"][-1]): err["msg"].removeprefix("Value error, ")
        for err in exc.errors()
        if err["loc"]
    }


def _config_rows(
    provider: str | None, model: str | None
) -> list[tuple[str, str, str, str, str]]:
    """``(group, setting, value, env var, source)`` for every recognised setting.

    Iterates :data:`_SETTINGS_GROUPS` so it stays in step with ``config init`` and
    the typo-detector, and no setting is silently omitted. ``source`` is where the
    effective value came from: a CLI ``flag``, the process ``env``, the ``.env``
    file, or the field ``default``.
    """
    env_keys = {key.upper() for key in os.environ}
    dotenv_keys = _dotenv_keys(Path(".env"))
    flagged = {
        _env_var_for(ModelSettings, "provider"): provider is not None,
        _env_var_for(ModelSettings, "model"): model is not None,
    }

    def source(env_var: str) -> str:
        if flagged.get(env_var):
            return "flag"
        if env_var.upper() in env_keys:
            return "env"
        if env_var.upper() in dotenv_keys:
            return ".env"
        return "default"

    rows: list[tuple[str, str, str, str, str]] = []
    for cls in _SETTINGS_GROUPS:
        # ModelSettings honours the --provider/--model overrides; the other
        # groups read only env / .env / defaults. config show is the debugging
        # tool, so a bad value in one group must not blank the whole table: catch
        # its ValidationError, render that group's fields as invalid (the reason
        # on the offending field), and carry on with the other groups.
        try:
            obj: BaseSettings | None = (
                _resolve_model_settings(provider, model)
                if cls is ModelSettings
                else cls()
            )
            invalid: dict[str, str] = {}
        except ValidationError as exc:
            obj = None
            invalid = _invalid_fields(exc)
        for field_name in cls.model_fields:
            env_var = _env_var_for(cls, field_name)
            if obj is None:
                reason = invalid.get(field_name)
                value = f"(invalid: {reason})" if reason else "(invalid)"
            else:
                value = _display_value(obj, field_name)
            rows.append(
                (_group_label(cls), field_name, value, env_var, source(env_var))
            )
    return rows


@click.group("config")
def config() -> None:
    """Inspect effective configuration."""


@config.command("show")
@click.pass_context
def config_show(ctx: click.Context) -> None:
    """Print the effective settings and where each value comes from."""
    # Flag (but don't fail on) a mistyped key: this is where you come to debug
    # configuration, so it must still render the table.
    _report_setting_problems(find_unrecognized_settings())
    table = Table(title="aiida-agents configuration")
    table.add_column("Group", style="dim")
    table.add_column("Setting", style="bold")
    # ``fold`` (not the table default ``ellipsis``) on the wide columns, so a
    # long value or env var wraps instead of hiding characters: both are read or
    # copied verbatim (a truncated URL, path, or ``AIIDA_AGENTS_MAX_TOK…`` is not
    # actionable). The short columns keep the default.
    table.add_column("Value", overflow="fold")
    table.add_column("Env var", style="dim", overflow="fold")
    table.add_column("Source", style="cyan")
    for group, setting, value, env_var, src in _config_rows(
        ctx.obj["provider"], ctx.obj["model"]
    ):
        table.add_row(group, setting, value, env_var, src)
    console.print(table)


@config.command("path")
def config_path() -> None:
    """Show where configuration is read from."""
    # click.echo (not console.print): the resolved path is long and must stay on
    # one line so it is copyable; rich would soft-wrap it to the terminal width.
    env_file = Path(".env").resolve()
    marker = "found" if env_file.is_file() else "not present"
    click.echo(f".env file: {env_file} ({marker})")
    click.echo(
        "Settings come from environment variables and this .env file. "
        "Run `aiida-agents config show` to see effective values and sources."
    )


@config.command("init")
@click.option(
    "--path",
    "path_str",
    type=click.Path(dir_okay=False),
    default=".env",
    show_default=True,
    help="Where to write the template.",
)
@click.option("--force", is_flag=True, help="Overwrite an existing file.")
def config_init(path_str: str, force: bool) -> None:
    """Write a commented ``.env`` template with the recognised settings."""
    target = Path(path_str)
    if target.exists() and not force:
        raise click.ClickException(
            f"{target} already exists; pass --force to overwrite."
        )
    try:
        target.write_text(_env_template(), encoding="utf-8")
    except OSError as exc:
        # e.g. a --path into a directory that does not exist: a clean error, not
        # a raw FileNotFoundError traceback.
        raise click.ClickException(f"Could not write {target}: {exc}") from exc
    click.echo(f"✓ Wrote {target}. Uncomment and edit the settings you want to change.")
