"""Tests for cli/_guards.py: the fail-fast guard on a mistyped settings key."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from aiida_agents._settings import SettingProblem
from aiida_agents.cli import cli


def _one_typo() -> list[SettingProblem]:
    return [SettingProblem("AIIDA_AGENS_PROVIDER", "AIIDA_AGENTS_PROVIDER")]


def test_action_command_fails_fast_on_mistyped_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A command that acts on config exits 2, before doing work, on a typo'd key."""
    monkeypatch.setattr(
        "aiida_agents.cli._guards.find_unrecognized_settings", _one_typo
    )
    result = CliRunner().invoke(cli, ["rag", "status"])
    assert result.exit_code == 2
    assert "AIIDA_AGENS_PROVIDER" in result.output
    assert "did you mean AIIDA_AGENTS_PROVIDER" in result.output


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(["doctor", "--help"], id="command-help"),
        pytest.param(["rag", "--help"], id="group-help"),
        pytest.param(["rag", "status", "-h"], id="subcommand-h"),
    ],
)
def test_help_bypasses_the_settings_check(
    monkeypatch: pytest.MonkeyPatch, args: list[str]
) -> None:
    """--help / -h resolve before the guard runs, so they work despite a typo."""
    monkeypatch.setattr(
        "aiida_agents.cli._guards.find_unrecognized_settings", _one_typo
    )
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0
    # help rendered and the guard never fired (no "did you mean" from _one_typo)
    assert "Show this message and exit" in result.output
    assert "did you mean" not in result.output
