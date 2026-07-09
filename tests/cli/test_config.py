"""Tests for cli/config.py: the config command group and the row/template data."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from aiida_agents._settings import SettingProblem
from aiida_agents.cli import cli
from aiida_agents.cli.config import _config_rows


def _one_typo() -> list[SettingProblem]:
    return [SettingProblem("AIIDA_AGENS_PROVIDER", "AIIDA_AGENTS_PROVIDER")]


def test_config_show_command_runs() -> None:
    """``config show`` renders the settings table without touching AiiDA or a model."""
    result = CliRunner().invoke(cli, ["config", "show"])
    assert result.exit_code == 0
    # The table actually rendered with real setting names and default values
    # (short tokens, so they survive rich's column folding at any width).
    assert "provider" in result.output
    assert "ollama" in result.output  # default provider
    assert "qwen3.5:2b" in result.output  # default model


def test_config_rows_report_value_and_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each row carries its group, resolved value, env var, and where it came from."""
    monkeypatch.setenv("AIIDA_AGENTS_MODEL", "some-model")
    monkeypatch.delenv("AIIDA_AGENTS_PROVIDER", raising=False)

    from_env = {row.setting: row for row in _config_rows(provider=None, model=None)}
    model_row = from_env["model"]
    assert model_row.value == "some-model"
    assert model_row.env_var == "AIIDA_AGENTS_MODEL"
    assert model_row.source == "env"
    assert from_env["provider"].source == "default"

    from_flag = {
        row.setting: row for row in _config_rows(provider="openai", model=None)
    }
    assert from_flag["provider"].value == "openai"
    assert from_flag["provider"].source == "flag"


def test_config_rows_disambiguate_base_url_by_group() -> None:
    """The two ``base_url`` fields are told apart by group and env var, so the
    rendered table never shows one label for two different settings.
    """
    base_urls = {
        row.group: row.env_var
        for row in _config_rows(None, None)
        if row.setting == "base_url"
    }
    assert base_urls == {
        "model": "AIIDA_AGENTS_BASE_URL",
        "ollama": "OLLAMA_BASE_URL",
    }


def test_config_rows_mask_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    """API keys are reported as set/unset, never echoed."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-super-secret")
    rows = {row.setting: row for row in _config_rows(None, None)}
    assert rows["openrouter_api_key"].value == "set"
    assert "sk-super-secret" not in str(rows)


def test_config_show_flags_but_still_runs_on_typo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`config show` reports the typo yet renders the table (exit 0), to aid debugging."""
    monkeypatch.setattr("aiida_agents.cli.config.find_unrecognized_settings", _one_typo)
    result = CliRunner().invoke(cli, ["config", "show"])
    assert result.exit_code == 0
    assert "AIIDA_AGENS_PROVIDER" in result.output


def test_env_template_covers_exactly_the_recognised_settings() -> None:
    """The template scaffolds every recognised setting and nothing else: it can't
    teach a typo, and no setting is silently missing.
    """
    from aiida_agents._settings import _known_env_var_names
    from aiida_agents.cli.config import _env_template

    keys = {
        key
        for line in _env_template().splitlines()
        if line.startswith("# ") and "=" in line
        # env-var lines are `# NAME=...`; description lines (`#   ... x=y`) and
        # section headers are excluded because a key token has no spaces.
        for key in [line[2:].split("=", 1)[0]]
        if " " not in key
    }
    assert keys == _known_env_var_names()


def test_config_rows_cover_all_recognised_settings() -> None:
    """`config show` covers exactly the recognised settings, so it stays in step
    with `config init` and the typo-detector (one source, no drift).
    """
    from aiida_agents._settings import _known_env_var_names

    env_vars = {row.env_var for row in _config_rows(None, None)}
    assert env_vars == _known_env_var_names()


def test_config_init_writes_template_and_refuses_overwrite(tmp_path: Path) -> None:
    """`config init` writes a template, then declines to clobber it without --force."""
    first = CliRunner().invoke(cli, ["config", "init"])
    assert first.exit_code == 0
    written = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "AIIDA_AGENTS_PROVIDER=ollama" in written

    second = CliRunner().invoke(cli, ["config", "init"])
    assert second.exit_code == 1
    assert "already exists" in second.output

    forced = CliRunner().invoke(cli, ["config", "init", "--force"])
    assert forced.exit_code == 0


def test_config_path_reports_the_env_file(tmp_path: Path) -> None:
    """`config path` names the .env it reads and whether it is present."""
    result = CliRunner().invoke(cli, ["config", "path"])
    assert result.exit_code == 0
    assert ".env" in result.output
    assert "not present" in result.output


def test_config_rows_mark_invalid_group_without_crashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bad value renders as `(invalid: ...)` instead of raising, so `config show`
    (the debugging tool) still shows the healthy groups.
    """
    monkeypatch.setenv("AIIDA_AGENTS_MAX_TOKENS", "0")  # violates gt=0
    rows = {row.setting: row for row in _config_rows(None, None)}  # must not raise
    assert rows["max_tokens"].value.startswith("(invalid")
    assert rows["tool_retries"].value == "3"  # a healthy group is untouched


def test_config_init_missing_parent_dir_is_clean() -> None:
    """`config init --path` into a non-existent directory is a clean error."""
    result = CliRunner().invoke(cli, ["config", "init", "--path", "no/such/dir/.env"])
    assert result.exit_code == 1
    assert "Could not write" in result.output
    assert not isinstance(result.exception, FileNotFoundError)
