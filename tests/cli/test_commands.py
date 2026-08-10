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

    def all_messages(self) -> list[object]:
        """No tool calls -- the grounding check reads these on every reply.

        Empty rather than absent: a fake that omits it would make the check
        raise, and making the check tolerate a missing method would let it skip
        silently in production too.
        """
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


def test_ask_reports_a_provider_failure_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A provider error is a clean CLI message, not a raw traceback.

    Regression from end-to-end testing: `chat` already caught this at its own
    boundary, but `ask` ran the agent unguarded, so a free-tier router
    returning malformed tool-call JSON (ModelHTTPError) reached the user as a
    Python traceback.
    """
    from aiida_agents.cli import commands

    async def _boom(
        agent: object, question: str, message_history: object = None
    ) -> object:
        raise RuntimeError("upstream returned malformed tool-call JSON")

    monkeypatch.setattr(
        commands, "_build_agent", lambda settings, profile, agent_type: object()
    )
    monkeypatch.setattr(commands, "ask", _boom)

    result = CliRunner().invoke(cli, ["ask", "anything"])

    assert result.exit_code == 1
    assert "Agent run failed" in result.output
    assert "malformed tool-call JSON" in result.output
    # Converted, not leaked as an uncaught traceback.
    assert not isinstance(result.exception, RuntimeError)


class TestChatStartup:
    """What `chat` builds before the REPL takes over.

    Regression from end-to-end testing: `chat` eagerly built one agent from
    ``--agent`` before starting the loop, which made the documented default
    entry point -- plain `aiida-agents chat`, whose ``--agent`` is ``auto`` --
    crash on startup, because "auto" is a routing decision and not an agent
    ``_build_agent`` can build. Every user of the default hit it immediately.
    """

    @staticmethod
    def _spy(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
        """Run `chat` without a terminal, recording what it built and passed."""
        from aiida_agents.cli import commands

        seen: dict[str, object] = {}

        def _build(settings: object, profile: object, agent_type: str) -> object:
            seen["built"] = agent_type
            return f"agent:{agent_type}"

        def _repl(agent: object, settings: object, **kwargs: object) -> None:
            seen["agent"] = agent
            seen["agent_type"] = kwargs.get("agent_type")

        monkeypatch.setattr(commands, "_build_agent", _build)
        monkeypatch.setattr(commands, "_run_repl", _repl)
        return seen

    def test_auto_builds_no_agent_up_front(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The crash itself: nothing may ask _build_agent for "auto"."""
        seen = self._spy(monkeypatch)

        result = CliRunner().invoke(cli, ["chat"])

        assert result.exit_code == 0
        assert "built" not in seen

    def test_auto_hands_the_repl_no_agent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The REPL routes each question and builds that specialist on first use."""
        seen = self._spy(monkeypatch)

        CliRunner().invoke(cli, ["chat"])

        assert seen["agent"] is None
        assert seen["agent_type"] == "auto"

    def test_a_named_specialist_is_still_built_before_the_loop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fixing auto must not cost `-a analysis` its startup failure."""
        seen = self._spy(monkeypatch)

        CliRunner().invoke(cli, ["-a", "analysis", "chat"])

        assert seen["built"] == "analysis"
        assert seen["agent"] == "agent:analysis"


class TestSandboxCommands:
    """The `sandbox` group, which builds and verifies the disposable copy.

    Every test here is really about one rule: a sandbox profile must never
    share deletable storage with a real one. Issue #73 is the cost of getting
    that wrong -- a maintainer deleted the sandbox profile, agreed to delete
    its data, and lost his own database with it.
    """

    def test_the_group_is_registered(self) -> None:
        result = CliRunner().invoke(cli, ["--help"])

        assert "sandbox" in result.output

    @staticmethod
    def _config(*profiles: object) -> object:
        class _Config:
            def __init__(self) -> None:
                self.profiles = list(profiles)
                self.deleted: dict[str, object] = {}

            def get_profile(self, name: object = None) -> object:
                for profile in self.profiles:
                    if profile.name == name:  # type: ignore[attr-defined]
                        return profile
                raise ValueError(f"no profile {name!r}")

            def delete_profile(self, name: str, delete_storage: bool = True) -> None:
                self.deleted = {"name": name, "delete_storage": delete_storage}

            def store(self) -> None:
                return None

        return _Config()

    @staticmethod
    def _profile(
        name: str, backend: str = "core.sqlite_dos", filepath: str | None = None
    ) -> object:
        class _Profile:
            pass

        profile = _Profile()
        profile.name = name  # type: ignore[attr-defined]
        profile.storage_backend = backend  # type: ignore[attr-defined]
        profile.storage_config = {"filepath": filepath or f"/data/{name}"}  # type: ignore[attr-defined]
        return profile

    def _patch(self, monkeypatch: pytest.MonkeyPatch, config: object) -> None:
        monkeypatch.setattr("aiida.manage.configuration.get_config", lambda: config)

    def test_check_fails_when_the_sandbox_shares_storage(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The exact shape of the design this replaced."""
        config = self._config(
            self._profile("real", filepath="/data/shared"),
            self._profile("agents-sandbox", filepath="/data/shared"),
        )
        self._patch(monkeypatch, config)

        result = CliRunner().invoke(cli, ["sandbox", "check"])

        assert result.exit_code == 1
        assert "real" in result.output

    def test_check_passes_a_genuinely_separate_copy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = self._config(
            self._profile("real", filepath="/data/real"),
            self._profile("agents-sandbox", filepath="/data/copy"),
        )
        self._patch(monkeypatch, config)

        result = CliRunner().invoke(cli, ["sandbox", "check"])

        assert result.exit_code == 0

    def test_check_fails_on_a_backend_it_cannot_reason_about(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fails closed. Not knowing whether two profiles are separate is not
        the same as knowing they are, and only one of those mistakes is safe."""
        config = self._config(
            self._profile("real", filepath="/data/real"),
            self._profile("agents-sandbox", backend="core.sqlite_zip"),
        )
        self._patch(monkeypatch, config)

        result = CliRunner().invoke(cli, ["sandbox", "check"])

        assert result.exit_code == 1

    def test_init_refuses_a_backend_it_cannot_copy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Naming the limit beats a copy that half worked and looked contained."""
        config = self._config(self._profile("real", backend="core.sqlite_zip"))
        self._patch(monkeypatch, config)

        result = CliRunner().invoke(cli, ["sandbox", "init", "--profile", "real"])

        assert result.exit_code != 0
        assert "cannot copy" in result.output

    def test_init_refuses_to_overwrite_an_existing_sandbox(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Rebuilding is teardown then init, so nothing in use is copied over."""
        config = self._config(self._profile("real"), self._profile("agents-sandbox"))
        self._patch(monkeypatch, config)

        result = CliRunner().invoke(cli, ["sandbox", "init", "--profile", "real"])

        assert result.exit_code != 0
        assert "refresh" in result.output

    def test_teardown_refuses_a_sandbox_that_shares_storage(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The last line of defence, for a sandbox registered before this
        rule existed. Deleting it is what destroyed a real database."""
        config = self._config(
            self._profile("real", filepath="/data/shared"),
            self._profile("agents-sandbox", filepath="/data/shared"),
        )
        self._patch(monkeypatch, config)

        result = CliRunner().invoke(cli, ["sandbox", "teardown", "--yes"])

        assert result.exit_code != 0
        assert config.deleted == {}, "nothing may be deleted"  # type: ignore[attr-defined]

    def test_teardown_never_asks_aiida_to_delete_the_storage(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`delete_storage=True` is the call that did the damage in #73.

        The copy's directory is removed here directly, so AiiDA is never asked
        to delete storage on our behalf -- not even for a profile we believe is
        self-contained.
        """
        config = self._config(
            self._profile("real", filepath="/data/real"),
            self._profile("agents-sandbox", filepath="/data/copy"),
        )
        self._patch(monkeypatch, config)

        result = CliRunner().invoke(cli, ["sandbox", "teardown", "--yes"])

        assert result.exit_code == 0
        assert config.deleted["delete_storage"] is False  # type: ignore[attr-defined]

    def test_teardown_on_a_missing_sandbox_is_not_an_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch(monkeypatch, self._config(self._profile("real")))

        result = CliRunner().invoke(cli, ["sandbox", "teardown", "--yes"])

        assert result.exit_code == 0
        assert "nothing to tear down" in result.output
