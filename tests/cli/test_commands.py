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


@pytest.mark.parametrize(
    "command, probe",
    [
        pytest.param("check", "_check_reachable", id="check"),
        pytest.param("warm", "_probe_model", id="warm"),
    ],
)
def test_probe_failure_is_diagnosed_and_exits(
    monkeypatch: pytest.MonkeyPatch, command: str, probe: str
) -> None:
    """A probe failure routes through `_diagnose_probe_failure` and exits 1,
    never a raw traceback.
    """
    from aiida_agents.cli import commands

    def _boom(settings: object) -> None:
        raise RuntimeError("endpoint down")

    diagnosed: list[str] = []
    monkeypatch.setattr(f"aiida_agents.cli.commands.{probe}", _boom)
    monkeypatch.setattr(
        commands,
        "_diagnose_probe_failure",
        lambda settings, exc: diagnosed.append(str(exc)),
    )

    result = CliRunner().invoke(cli, [command])

    assert result.exit_code == 1
    assert diagnosed == ["endpoint down"]


class _FakeResult:
    def __init__(self, output: object) -> None:
        self.output = output

    def new_messages(self) -> list[object]:
        return []


def test_ask_prints_the_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    """A one-shot `ask` prints the agent's reply and exits cleanly."""
    from aiida_agents.cli import commands

    async def _fake_ask(
        agent: object, question: str, message_history: object = None
    ) -> _FakeResult:
        return _FakeResult("42 is the answer")

    monkeypatch.setattr(
        commands, "_build_agent", lambda settings, profile, agent_type: object()
    )
    monkeypatch.setattr(commands, "ask", _fake_ask)

    result = CliRunner().invoke(cli, ["ask", "what is 6 times 7"])

    assert result.exit_code == 0
    assert "42 is the answer" in result.output


@pytest.mark.parametrize(
    "argv, expected_agent",
    [
        pytest.param(["ask", "hi"], "analysis", id="default-is-analysis"),
        pytest.param(
            ["--agent", "execution", "ask", "hi"], "execution", id="long-flag"
        ),
        pytest.param(["-a", "execution", "ask", "hi"], "execution", id="short-flag"),
        pytest.param(
            ["--agent", "EXECUTION", "ask", "hi"], "execution", id="case-fold"
        ),
    ],
)
def test_agent_flag_selects_the_agent(
    monkeypatch: pytest.MonkeyPatch, argv: list[str], expected_agent: str
) -> None:
    """`--agent`/`-a` picks which agent `_build_agent` builds (default analysis),
    case-insensitively. Pins the flag→agent_type wiring since it flips a default.
    """
    from aiida_agents.cli import commands

    captured: dict[str, str] = {}

    def _fake_build(settings: object, profile: object, agent_type: str) -> object:
        captured["agent_type"] = agent_type
        return object()

    async def _fake_ask(
        agent: object, question: str, message_history: object = None
    ) -> _FakeResult:
        return _FakeResult("ok")

    monkeypatch.setattr(commands, "_build_agent", _fake_build)
    monkeypatch.setattr(commands, "ask", _fake_ask)

    result = CliRunner().invoke(cli, argv)

    assert result.exit_code == 0
    assert captured["agent_type"] == expected_agent


def test_agent_flag_rejects_unknown_choice() -> None:
    """`--agent` is a `click.Choice`, so a bad value is a clean usage error
    listing the real agents (mirrors the `--provider` guard).
    """
    result = CliRunner().invoke(cli, ["--agent", "bogus", "ask", "hi"])
    assert result.exit_code == 2  # Click usage error, before any work
    assert "bogus" in result.output
    assert "execution" in result.output  # the derived choices are listed


def test_ask_rejects_a_deferred_write(monkeypatch: pytest.MonkeyPatch) -> None:
    """A write proposed in one-shot mode can't be approved interactively, so `ask`
    explains and exits 2 rather than silently dropping it.
    """
    from pydantic_ai.tools import DeferredToolRequests

    from aiida_agents.cli import commands

    async def _fake_ask(
        agent: object, question: str, message_history: object = None
    ) -> _FakeResult:
        return _FakeResult(DeferredToolRequests(approvals=[]))

    monkeypatch.setattr(
        commands, "_build_agent", lambda settings, profile, agent_type: object()
    )
    monkeypatch.setattr(commands, "ask", _fake_ask)

    result = CliRunner().invoke(cli, ["ask", "please submit a workflow"])

    assert result.exit_code == 2
    assert "interactive approval" in result.output
