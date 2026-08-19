"""Tests for cli/commands.py: the root group and the core model-facing commands."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from aiida_agents.cli import cli


def test_cli_exposes_expected_commands() -> None:
    """The top-level group lists every subcommand we ship."""
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    for command in ("ask", "chat", "config", "doctor", "mcp", "rag"):
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


def test_version_option_prints_version() -> None:
    """`--version` reports the installed package version and exits cleanly."""
    from importlib.metadata import version

    result = CliRunner().invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert version("aiida-agents") in result.output


def test_doctor_reports_invalid_setting_value_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bad setting *value* is a clean CLI error, not a raw pydantic traceback.

    Regression: `doctor` builds settings outside its try, so an invalid
    `AIIDA_AGENTS_*` value used to surface as an uncaught `ValidationError`;
    now `_resolve_settings_or_fail` converts it.
    """
    from pydantic import ValidationError

    monkeypatch.setenv("AIIDA_AGENTS_PROVIDER", "bogus")
    result = CliRunner().invoke(cli, ["doctor"])
    assert result.exit_code == 1
    assert "Invalid configuration" in result.output
    assert "provider" in result.output
    assert not isinstance(result.exception, ValidationError)


def test_provider_flag_rejects_unknown_choice() -> None:
    """`--provider` is a `click.Choice` derived from the provider `Literal`, so a
    bad value is a clean 'invalid choice' usage error listing the real providers
    (complements the env-var path above, which pydantic validates).
    """
    result = CliRunner().invoke(cli, ["--provider", "bogus", "doctor"])
    assert result.exit_code == 2  # Click usage error, before any work
    assert "bogus" in result.output
    assert "ollama" in result.output  # the derived choices are listed


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

            def add_profile(self, profile: object) -> None:
                self.profiles.append(profile)

            def store(self) -> None:
                return None

        return _Config()

    @staticmethod
    def _profile(
        name: str,
        backend: str = "core.sqlite_dos",
        filepath: str | None = None,
        *,
        config: dict[str, object] | None = None,
    ) -> object:
        class _Profile:
            pass

        if config is None:
            config = {"filepath": filepath or f"/data/{name}"}

        profile = _Profile()
        profile.name = name  # type: ignore[attr-defined]
        profile.storage_backend = backend  # type: ignore[attr-defined]
        # One dict, referenced by both: a stub whose `dictionary` and
        # `storage_config` disagree describes a profile that cannot exist.
        profile.storage_config = config  # type: ignore[attr-defined]
        # What `sandbox_profile_dictionary` clones the sandbox's own from.
        profile.dictionary = {  # type: ignore[attr-defined]
            "storage": {"backend": backend, "config": profile.storage_config},  # type: ignore[attr-defined]
            "process_control": {"backend": None, "config": None},
            "default_user_email": "someone@example.com",
            "PROFILE_UUID": "1111",
            "options": {},
            "test_profile": False,
        }
        return profile

    def _patch(self, monkeypatch: pytest.MonkeyPatch, config: object) -> None:
        monkeypatch.setattr("aiida.manage.configuration.get_config", lambda: config)

    @pytest.mark.parametrize(
        "others, code, expected, why",
        [
            pytest.param(
                [("real", "core.sqlite_dos", "/data/copy")],
                1,
                "shares storage with 'real'",
                "the exact shape of the design this replaced",
                id="shares-storage",
            ),
            pytest.param(
                [("real", "core.sqlite_dos", "/data/real")],
                0,
                "shares no storage",
                "a genuinely separate copy",
                id="separate",
            ),
            pytest.param(
                [("odd", "thirdparty.custom_dos", "/data/odd")],
                1,
                "cannot be told apart from 'odd'",
                "fails closed, and says which of the two it is",
                id="unreadable-backend",
            ),
            pytest.param(
                [("dev-archive", "core.sqlite_zip", "/data/e.aiida")],
                0,
                "shares no storage",
                "an archive is a built-in backend, not an unreadable one",
                id="archive-elsewhere",
            ),
            pytest.param(
                [
                    ("real", "core.sqlite_dos", "/data/copy"),
                    ("odd", "thirdparty.custom_dos", "/data/odd"),
                ],
                1,
                "verdi profile delete --keep-data",
                "both remedies, because rebuilding fixes only one of them",
                id="one-of-each",
            ),
        ],
    )
    def test_check_reports_what_it_found(
        self,
        monkeypatch: pytest.MonkeyPatch,
        others: list[tuple[str, str, str]],
        code: int,
        expected: str,
        why: str,
    ) -> None:
        """The sandbox is at `/data/copy` throughout; the other profiles vary."""
        config = self._config(
            *(
                self._profile(name, backend=backend, filepath=filepath)
                for name, backend, filepath in others
            ),
            self._profile("agents-sandbox", filepath="/data/copy"),
        )
        self._patch(monkeypatch, config)

        result = CliRunner().invoke(cli, ["sandbox", "check"])
        output = " ".join(result.output.split())

        assert result.exit_code == code, why
        assert expected in output, why

    def test_check_does_not_claim_to_have_found_what_it_could_not_read(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A researcher reading "deleting this destroys their data" concludes
        their sandbox is pointed at their own database. Where the truth is that
        a backend could not be read, saying so is the difference between a
        problem they can fix and one they cannot."""
        config = self._config(
            self._profile("odd", backend="thirdparty.custom_dos"),
            self._profile("agents-sandbox", filepath="/data/copy"),
        )
        self._patch(monkeypatch, config)

        result = CliRunner().invoke(cli, ["sandbox", "check"])

        assert "destroys" not in " ".join(result.output.split())

    def test_init_refuses_a_backend_it_cannot_copy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Naming the limit beats a copy that half worked and looked contained."""
        config = self._config(self._profile("real", backend="core.sqlite_zip"))
        self._patch(monkeypatch, config)

        result = CliRunner().invoke(cli, ["sandbox", "init", "--profile", "real"])

        assert result.exit_code != 0
        assert "cannot copy" in result.output

    def _real_storage(self, tmp_path: Path) -> Path:
        """A storage directory with something in it worth counting."""
        storage = tmp_path / "real-storage"
        (storage / "container").mkdir(parents=True)
        (storage / "database.sqlite").write_bytes(b"x" * 4096)
        return storage

    def test_init_says_what_the_copy_costs_and_waits_for_an_answer(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The copy is the whole repository, gigabytes on a real profile.

        Finding that out from `df` afterwards is the behaviour this replaces,
        so the size and the destination go out first and "n" stops it before
        anything is written.
        """
        storage = self._real_storage(tmp_path)
        config = self._config(self._profile("real", filepath=str(storage)))
        self._patch(monkeypatch, config)
        monkeypatch.setattr(
            "aiida_agents.sandbox.copy.sandbox_storage_root",
            lambda name: tmp_path / "sandbox" / name,
        )

        result = CliRunner().invoke(
            cli, ["sandbox", "init", "--profile", "real"], input="n\n"
        )
        output = " ".join(result.output.split())

        assert result.exit_code == 0, "declining is an answer, not an error"
        assert "4.1 kB" in output
        assert str(tmp_path / "sandbox" / "agents-sandbox" / "storage") in output
        assert not (tmp_path / "sandbox").exists(), "nothing may be copied on 'n'"

    def test_init_yes_copies_without_asking(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Scripted use has to stay possible, the same way `teardown --yes` does."""
        storage = self._real_storage(tmp_path)
        config = self._config(self._profile("real", filepath=str(storage)))
        self._patch(monkeypatch, config)
        monkeypatch.setattr(
            "aiida_agents.sandbox.copy.sandbox_storage_root",
            lambda name: tmp_path / "sandbox" / name,
        )

        result = CliRunner().invoke(
            cli, ["sandbox", "init", "--profile", "real", "--yes"]
        )

        assert result.exit_code == 0
        copy = tmp_path / "sandbox" / "agents-sandbox" / "storage"
        assert (copy / "database.sqlite").read_bytes() == b"x" * 4096

    def test_init_refuses_an_overlapping_target_before_copying_anything(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A source that contains the sandbox root would be copied into itself.

        The separation check used to run after the copy, so the refusal arrived
        once the gigabytes were already on disk, and left them there.
        """
        storage = self._real_storage(tmp_path)
        config = self._config(self._profile("real", filepath=str(storage)))
        self._patch(monkeypatch, config)
        # The sandbox root inside the source: the shape that copies into itself.
        monkeypatch.setattr(
            "aiida_agents.sandbox.copy.sandbox_storage_root",
            lambda name: storage / "agents-sandbox" / name,
        )

        result = CliRunner().invoke(
            cli, ["sandbox", "init", "--profile", "real", "--yes"]
        )

        assert result.exit_code != 0
        assert "shares storage" in result.output
        assert not (storage / "agents-sandbox").exists(), "nothing may be copied"

    def test_init_refuses_to_overwrite_an_existing_sandbox(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Rebuilding is teardown then init, so nothing in use is copied over.

        Named in the message, because "already exists" without a way forward is
        where a reader stops.
        """
        config = self._config(self._profile("real"), self._profile("agents-sandbox"))
        self._patch(monkeypatch, config)

        result = CliRunner().invoke(cli, ["sandbox", "init", "--profile", "real"])

        assert result.exit_code != 0
        assert "teardown" in result.output

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

    def test_teardown_asks_aiida_to_delete_the_storage(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`delete_storage=True` is the call that did the damage in #73.

        It is the supported path, and it is the only one that drops a
        PostgreSQL database rather than printing a `dropdb` whose password the
        user does not have. What makes it safe is the check that runs
        immediately before, which counts containment and fails closed; the test
        above pins that nothing is deleted when it finds anything.
        """
        config = self._config(
            self._profile("real", filepath="/data/real"),
            self._profile("agents-sandbox", filepath="/data/copy"),
        )
        self._patch(monkeypatch, config)

        result = CliRunner().invoke(cli, ["sandbox", "teardown", "--yes"])

        assert result.exit_code == 0
        assert config.deleted["delete_storage"] is True  # type: ignore[attr-defined]

    @staticmethod
    def _postgres_profile(name: str) -> object:
        return TestSandboxCommands._profile(
            name,
            backend="core.psql_dos",
            config={
                "database_name": f"aiida_{name}",
                "database_hostname": "localhost",
                "database_port": 5432,
                "database_username": "aiida",
                "database_password": "pw",
                "repository_uri": f"file:///data/{name}/repository",
            },
        )

    def test_init_makes_the_postgres_copy_rather_than_printing_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Five commands and five GRANTs is a setup step nobody performs, and a
        setup step nobody performs is a feature nobody has."""
        self._patch(monkeypatch, self._config(self._postgres_profile("real")))
        created: list[tuple[object, str]] = []
        copied: list[object] = []
        monkeypatch.setattr(
            "aiida_agents.sandbox.postgres.create_database",
            lambda config, database: created.append((config, database)),
        )
        monkeypatch.setattr(
            "aiida_agents.sandbox.postgres.copy_database",
            lambda config, database: copied.append(database),
        )
        monkeypatch.setattr(
            "aiida_agents.cli.sandbox._database_exists", lambda config: False
        )

        result = CliRunner().invoke(
            cli, ["sandbox", "init", "--profile", "real", "--yes"]
        )

        assert [database for _, database in created] == ["aiida_real_agents_sandbox"]
        assert copied == ["aiida_real_agents_sandbox"]
        assert "Copied into" in result.output

    def test_init_copies_the_postgres_repository_as_well(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """`pg_dump` moves the database, which holds the hash of every file and
        none of the files.

        They live in a disk-objectstore container on disk, and the sandbox's
        was only ever a path in a URI: nothing created it, so `sandbox check`
        reported `UnreachableStorage: the container is not initialised yet` and
        no node could ever have been opened.
        """
        files = tmp_path / "real-repository"
        (files / "container").mkdir(parents=True)
        (files / "container" / "config.json").write_text("{}")
        source = self._postgres_profile("real")
        source.storage_config["repository_uri"] = files.as_uri()  # type: ignore[attr-defined]
        self._patch(monkeypatch, self._config(source))
        monkeypatch.setattr(
            "aiida_agents.sandbox.copy.sandbox_storage_root",
            lambda name: tmp_path / "sandbox" / name,
        )
        monkeypatch.setattr(
            "aiida_agents.sandbox.postgres.create_database", lambda config, name: None
        )
        monkeypatch.setattr(
            "aiida_agents.sandbox.postgres.copy_database", lambda config, database: None
        )
        monkeypatch.setattr(
            "aiida_agents.cli.sandbox._database_exists", lambda config: False
        )

        result = CliRunner().invoke(
            cli, ["sandbox", "init", "--profile", "real", "--yes"]
        )

        copied = tmp_path / "sandbox" / "agents-sandbox" / "repository"
        assert result.exit_code == 0, result.output
        assert (copied / "container" / "config.json").exists()

    def test_init_says_what_the_postgres_repository_copy_costs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The database question was asked and the repository was copied
        unannounced, and the repository is the large half of a PostgreSQL
        profile: gigabytes went with no size, no destination and no free-space
        line, where the SQLite path reports all three."""
        files = tmp_path / "real-repository"
        (files / "container").mkdir(parents=True)
        (files / "container" / "config.json").write_text("{}")
        source = self._postgres_profile("real")
        source.storage_config["repository_uri"] = files.as_uri()  # type: ignore[attr-defined]
        config = self._config(source)
        self._patch(monkeypatch, config)
        monkeypatch.setattr(
            "aiida_agents.sandbox.copy.sandbox_storage_root",
            lambda name: tmp_path / "sandbox" / name,
        )
        monkeypatch.setattr(
            "aiida_agents.sandbox.postgres.create_database", lambda config, name: None
        )
        monkeypatch.setattr(
            "aiida_agents.sandbox.postgres.copy_database", lambda config, database: None
        )
        monkeypatch.setattr(
            "aiida_agents.cli.sandbox._database_exists", lambda config: False
        )

        # Yes to the database, no to the repository.
        result = CliRunner().invoke(
            cli, ["sandbox", "init", "--profile", "real"], input="y\nn\n"
        )
        output = " ".join(result.output.split())

        assert result.exit_code == 0
        assert str(files) in output, "the source it would read"
        assert "free there" in output, "and whether it fits"
        assert not (tmp_path / "sandbox" / "agents-sandbox" / "repository").exists()
        registered = [profile.name for profile in config.profiles]  # type: ignore[attr-defined]
        assert registered == ["real"], "nothing registered on 'n'"

    def test_init_falls_back_to_printing_when_it_cannot(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A managed server, a remote host, no sudo. The printed commands were
        the whole feature until now and are still the only road there."""
        from aiida_agents.sandbox.postgres import PostgresUnavailableError

        self._patch(monkeypatch, self._config(self._postgres_profile("real")))

        def _refuse(config: object, database: str) -> None:
            raise PostgresUnavailableError("no superuser connection here")

        monkeypatch.setattr("aiida_agents.sandbox.postgres.create_database", _refuse)
        monkeypatch.setattr(
            "aiida_agents.cli.sandbox._database_exists", lambda config: False
        )

        result = CliRunner().invoke(
            cli, ["sandbox", "init", "--profile", "real", "--yes"]
        )
        output = " ".join(result.output.split())

        assert "no superuser connection here" in output
        assert "aiida_real_agents_sandbox" in output, "name the database it needed"
        assert "init" in output, "and how to carry on once it exists"

    def test_a_profile_named_like_rich_markup_survives_the_output(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`aiida[test]` is a legal profile name, and rich read `[test]` as a
        style tag and dropped it.

        The command then reported a profile the user does not have, which is
        the same failure as naming the wrong one.
        """
        config = self._config(
            self._profile("real", filepath="/data/real"),
            self._profile("aiida[test]", filepath="/data/real"),
        )
        self._patch(monkeypatch, config)

        result = CliRunner().invoke(
            cli, ["sandbox", "check", "--profile", "aiida[test]"]
        )

        assert "aiida[test]" in result.output

    def test_teardown_names_the_storage_it_is_about_to_remove(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """It asked "delete profile X and its copy?", which reads as a copy of
        the profile: the one thing it does not delete. Both halves are named
        now, and the storage half by size and path."""
        storage = self._real_storage(tmp_path)
        config = self._config(
            self._profile("real", filepath="/data/real"),
            self._profile("agents-sandbox", filepath=str(storage)),
        )
        self._patch(monkeypatch, config)
        monkeypatch.setattr(
            "aiida_agents.sandbox.copy.sandbox_storage_root", lambda name: storage
        )

        result = CliRunner().invoke(cli, ["sandbox", "teardown"], input="n\n")
        output = " ".join(result.output.split())

        assert "4.1 kB" in output
        assert str(storage) in output
        assert storage.exists(), "nothing may be removed on 'n'"

    def test_teardown_removes_the_profile_even_if_the_storage_will_not_go(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A database that cannot be reached, a directory already gone.

        The profile still has to go, or the next `init` refuses because it
        exists, and what was left behind has to be said rather than implied by
        a traceback.
        """

        def _refuse(name: str, delete_storage: bool = True) -> None:
            if delete_storage:
                raise RuntimeError("could not connect to the server")

        config = self._config(
            self._profile("real", filepath="/data/real"),
            self._profile("agents-sandbox", filepath="/data/copy"),
        )
        monkeypatch.setattr(config, "delete_profile", _refuse)
        self._patch(monkeypatch, config)

        result = CliRunner().invoke(cli, ["sandbox", "teardown", "--yes"])
        output = " ".join(result.output.split())

        assert result.exit_code == 0
        assert "not its storage" in output
        assert "could not connect to the server" in output
        # It went on to claim the storage had gone, one line under the warning
        # that it had not, and for PostgreSQL the database is still there.
        assert "the storage it copied" not in output

    def test_teardown_on_a_missing_sandbox_is_not_an_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch(monkeypatch, self._config(self._profile("real")))

        result = CliRunner().invoke(cli, ["sandbox", "teardown", "--yes"])

        assert result.exit_code == 0
        assert "nothing to tear down" in result.output


class TestRootOptionsWorkOnEitherSideOfTheCommand:
    """Issue #79: `--agent` was position-sensitive, and said so badly.

    Click binds a flag to the group or the command by position, so `--agent`,
    declared on the root group, stopped being recognised the moment a
    subcommand name was read. `aiida-agents ask -a codegen ...` --- the
    spelling every other CLI a researcher uses would accept --- failed with a
    bare "No such option: -a", which names the option without saying that it
    exists, where it goes, or that moving it two words left would fix it.
    """

    @staticmethod
    def _agent_type(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> str | None:
        """The specialist `ask` resolved to, without running anything."""
        from aiida_agents.cli import commands as commands_module

        seen: dict[str, str] = {}

        def _capture(agent_type: str, question: str, settings: object) -> list[object]:
            seen["agent"] = agent_type
            raise SystemExit(0)

        monkeypatch.setattr(commands_module, "_resolve_plan", _capture)
        CliRunner().invoke(cli, argv)
        return seen.get("agent")

    @pytest.mark.parametrize(
        "argv",
        [
            pytest.param(["-a", "codegen", "ask", "q"], id="before-the-command"),
            pytest.param(["ask", "-a", "codegen", "q"], id="after-the-command"),
        ],
    )
    def test_the_agent_can_be_named_in_either_position(
        self, monkeypatch: pytest.MonkeyPatch, argv: list[str]
    ) -> None:
        assert self._agent_type(monkeypatch, argv) == "codegen"

    def test_the_more_specific_position_wins(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Given both, the one nearer the command is the one meant."""
        argv = ["-a", "analysis", "ask", "-a", "codegen", "q"]

        assert self._agent_type(monkeypatch, argv) == "codegen"

    def test_the_duplicate_option_does_not_clobber_the_root_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The failure mode this design has to avoid.

        If the subcommand's copy carried a default, that default would silently
        beat an explicit `-a codegen` written before the command. It carries
        none, so "not given" stays distinguishable from "given the default".
        """
        argv = ["-a", "codegen", "ask", "q"]

        assert self._agent_type(monkeypatch, argv) == "codegen"

    def test_naming_neither_still_routes_automatically(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert self._agent_type(monkeypatch, ["ask", "q"]) == "auto"

    @pytest.mark.parametrize("flag", ["--provider", "--model", "--profile", "--agent"])
    def test_every_root_option_is_accepted_after_the_command(self, flag: str) -> None:
        """Not just `-a`.

        All four root options had the same trap; the issue names `--agent` only
        because that is the one being used at the time. Fixing one would leave
        it set for whoever reaches for `--model` next.
        """
        result = CliRunner().invoke(cli, ["ask", "--help"])

        assert flag in result.output

    def test_a_misplaced_option_elsewhere_says_where_it_goes(self) -> None:
        """`--agent` on `rag` is still an error --- but a useful one.

        Accepting it there would be pretending it does something. The parse
        still fails; what changed is that the message names the option and
        shows the working spelling.
        """
        result = CliRunner().invoke(cli, ["rag", "status", "-a", "codegen"])

        assert result.exit_code != 0
        assert "goes before the command" in result.output
        assert "No such option" not in result.output

    def test_an_option_that_really_does_not_exist_is_unaffected(self) -> None:
        """The rewrite is scoped to root options, not to every parse failure."""
        result = CliRunner().invoke(cli, ["rag", "status", "--not-an-option"])

        assert result.exit_code != 0
        assert "goes before the command" not in result.output
