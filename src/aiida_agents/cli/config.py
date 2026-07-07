"""Data behind ``config show``: effective settings, their env var, and source."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings

from aiida_agents._settings import (
    LoggingSettings,
    ModelSettings,
    OllamaSettings,
    RagSettings,
    ReplSettings,
)
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


def _env_template() -> str:
    """A commented ``.env`` scaffold listing every recognised setting.

    Derived from the settings classes, so the keys are correct by construction:
    a generated template can never teach a typo. Every line is commented out and
    carries the field's default; secrets are listed with an empty value, never a
    default. Uncomment and edit a line to override.
    """
    groups = (ModelSettings, OllamaSettings, ReplSettings, RagSettings, LoggingSettings)
    lines = [
        "# aiida-agents configuration.",
        "# Uncomment a line and set its value to override the default.",
        "",
    ]
    for cls in groups:
        lines.append(f"# --- {cls.__name__.removesuffix('Settings').lower()} ---")
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


def _group_name(obj: BaseSettings) -> str:
    """Short label for a settings group (``ModelSettings`` -> ``model``).

    Disambiguates fields that share a name across groups (both the provider and
    Ollama settings expose ``base_url``), so the rendered table never shows one
    label for two different settings.
    """
    return type(obj).__name__.removesuffix("Settings").lower()


def _config_rows(
    provider: str | None, model: str | None
) -> list[tuple[str, str, str, str, str]]:
    """``(group, setting, value, env var, source)`` for the user-facing settings.

    ``source`` is where the effective value came from: a CLI ``flag``, the
    process ``env``, the ``.env`` file, or the field ``default``.
    """
    settings = _resolve_model_settings(provider, model)
    ollama = OllamaSettings()
    repl = ReplSettings()
    rag = RagSettings()

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

    def secret(value: str | None) -> str:
        return "set" if value and value != "api-key-not-set" else "unset"

    plain: list[tuple[BaseSettings, str, str]] = [
        (settings, "provider", str(settings.provider)),
        (settings, "model", settings.model),
        (settings, "base_url", settings.base_url or "(none)"),
        (settings, "max_tokens", str(settings.max_tokens)),
        (
            settings,
            "context_length",
            str(settings.context_length)
            if settings.context_length is not None
            else "(Ollama default)",
        ),
        (ollama, "base_url", ollama.base_url),
        (repl, "history_max_turns", str(repl.history_max_turns)),
        (repl, "vi_mode", str(repl.vi_mode)),
        (rag, "embed_backend", str(rag.embed_backend)),
        (rag, "embed_model", rag.embed_model),
        (rag, "vector_db_path", str(rag.vector_db_path)),
    ]
    rows = [
        (
            _group_name(obj),
            field,
            value,
            _env_var_for(type(obj), field),
            source(_env_var_for(type(obj), field)),
        )
        for obj, field, value in plain
    ]
    for field in (
        "openai_api_key",
        "anthropic_api_key",
        "openrouter_api_key",
        "api_key",
    ):
        env_var = _env_var_for(ModelSettings, field)
        rows.append(
            (
                _group_name(settings),
                field,
                secret(getattr(settings, field)),
                env_var,
                source(env_var),
            )
        )
    return rows
