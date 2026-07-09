"""Tests for cli/commands.py: the root group and the core model-facing commands."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from aiida_agents.cli import cli


def test_cli_exposes_expected_commands() -> None:
    """The top-level group lists every subcommand we ship."""
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    for command in ("ask", "chat", "check", "config", "doctor", "mcp", "rag", "warm"):
        assert command in result.output


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(["-h"], id="root"),
        pytest.param(["config", "-h"], id="group"),
        pytest.param(["ask", "-h"], id="command"),
        pytest.param(["rag", "build", "-h"], id="nested-subcommand"),
    ],
)
def test_dash_h_is_a_help_alias(args: list[str]) -> None:
    """`-h` actually renders help (not just exits 0) at the group and every level."""
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0
    assert "Show this message and exit" in result.output


@pytest.mark.parametrize(
    "command, expected_calls",
    [
        pytest.param("check", ["reachable"], id="check-probes-never-generates"),
        pytest.param("warm", ["generate"], id="warm-generates"),
    ],
)
def test_check_and_warm_use_distinct_probes(
    monkeypatch: pytest.MonkeyPatch, command: str, expected_calls: list[str]
) -> None:
    """`check` probes reachability and never generates; `warm` runs the generation
    probe that loads the model. Pins the deliberate split (a check stays cheap and
    side-effect-free): both probes are stubbed, so the recorded calls prove which
    one each command drives.
    """
    calls: list[str] = []
    monkeypatch.setattr(
        "aiida_agents.cli.commands._check_reachable",
        lambda settings: calls.append("reachable"),
    )
    monkeypatch.setattr(
        "aiida_agents.cli.commands._probe_model",
        lambda settings: calls.append("generate"),
    )
    result = CliRunner().invoke(cli, [command])
    assert result.exit_code == 0
    assert calls == expected_calls


@pytest.mark.parametrize(
    "exc, expected",
    [
        pytest.param(ValueError(), "", id="empty-message"),
        pytest.param(ValueError("boom"), "boom", id="single-line"),
        pytest.param(ValueError("first\nsecond"), "first", id="first-line-only"),
        pytest.param(ValueError("\n\n  \nreal"), "real", id="skips-blank-lines"),
        pytest.param(ValueError("x" * 200), "x" * 100, id="truncated-to-100"),
    ],
)
def test_short_reason_summarizes_exception(exc: Exception, expected: str) -> None:
    """A failing health check yields a one-line, bounded detail for the `doctor`
    table. An empty-message exception (a bare ``ValueError``, or
    ``asyncio.TimeoutError``) yields '' instead of crashing the whole report with
    ``IndexError`` (regression for the ``str(exc).splitlines()[0]`` guard).
    """
    from aiida_agents.cli.commands import _short_reason

    assert _short_reason(exc) == expected


def test_version_option_prints_version() -> None:
    """`--version` reports the installed package version and exits cleanly."""
    from importlib.metadata import version

    result = CliRunner().invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert version("aiida-agents") in result.output


def test_check_reports_invalid_setting_value_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bad setting *value* is a clean CLI error, not a raw pydantic traceback.

    Regression: `check`/`warm`/`doctor` build settings outside their try, so an
    invalid `AIIDA_AGENTS_*` value used to surface as an uncaught
    `ValidationError`; now `_resolve_settings_or_fail` converts it.
    """
    from pydantic import ValidationError

    monkeypatch.setenv("AIIDA_AGENTS_PROVIDER", "bogus")
    result = CliRunner().invoke(cli, ["check"])
    assert result.exit_code == 1
    assert "Invalid configuration" in result.output
    assert "provider" in result.output
    assert not isinstance(result.exception, ValidationError)


def test_provider_flag_rejects_unknown_choice() -> None:
    """`--provider` is a `click.Choice` derived from the provider `Literal`, so a
    bad value is a clean 'invalid choice' usage error listing the real providers
    (complements the env-var path above, which pydantic validates).
    """
    result = CliRunner().invoke(cli, ["--provider", "bogus", "check"])
    assert result.exit_code == 2  # Click usage error, before any work
    assert "bogus" in result.output
    assert "ollama" in result.output  # the derived choices are listed
