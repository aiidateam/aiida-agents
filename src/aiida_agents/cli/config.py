"""Data behind ``config show``: effective settings, their env var, and source."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings

from aiida_agents._settings import _SETTINGS_GROUPS, ModelSettings
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


def _dotenv_keys(path: Path) -> set[str]:
    """Upper-cased keys assigned in a ``.env`` file, for config-source display."""
    keys: set[str] = set()
    if not path.is_file():
        return keys
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key = line.split("=", 1)[0].strip().removeprefix("export ").strip()
        keys.add(key.upper())
    return keys


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
        return "(unset)"
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
        # groups read only env / .env / defaults.
        obj = (
            _resolve_model_settings(provider, model) if cls is ModelSettings else cls()
        )
        for field_name in cls.model_fields:
            env_var = _env_var_for(cls, field_name)
            rows.append(
                (
                    _group_label(cls),
                    field_name,
                    _display_value(obj, field_name),
                    env_var,
                    source(env_var),
                )
            )
    return rows
