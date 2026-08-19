"""Tests for cli/doctor.py: the doctor command's health-check helpers."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from click.testing import CliRunner

from aiida.engine.daemon.client import (
    DaemonNotRunningException,
    DaemonStalePidException,
    DaemonTimeoutException,
)

from aiida_agents._settings import ModelSettings, _Provider
from aiida_agents.cli import cli
from aiida_agents.cli.doctor import _DiagnosticRow


class _Profile:
    name = "test-profile"


class _StorageProfile:
    """A profile with somewhere to keep data, for the sandbox check."""

    def __init__(self, name: str, filepath: str) -> None:
        self.name = name
        self.storage_backend = "core.sqlite_dos"
        self.storage_config = {"filepath": filepath}


class _SandboxConfig:
    """An AiiDA config holding a real profile and a sandbox beside it.

    `sandbox` names the sandbox profile; pass the same filepath as the real one
    to model the storage-sharing case that issue #73 was about.
    """

    def __init__(self, sandbox: str | None, filepath: str = "/data/copy") -> None:
        self.profiles = [_StorageProfile("real", "/data/real")]
        if sandbox is not None:
            self.profiles.append(_StorageProfile(sandbox, filepath))


class _Index:
    def __init__(self, built: bool, stale: tuple[object, ...] = ()) -> None:
        self.built = built
        # doctor distinguishes "never built" from "built but unreachable", so
        # the stub carries both facts the real IndexStatus exposes.
        self.stale = stale


# Verbatim from ``aiida.engine.daemon.client.call_client``. 162 characters, with
# the remedy in the second sentence, so ``_short_reason``'s 100-char cut lands
# mid-word and drops it. Copied rather than imported because AiiDA builds it
# inline at the raise site.
_STALE_PID_MSG = (
    "The daemon could not be reached, seemingly because of a stale PID file. "
    "Either stop or start the daemon to remove it and restore the daemon to a "
    "functional state."
)


class _DaemonClient:
    """Stub AiiDA daemon client exposing what the doctor check reads."""

    def __init__(
        self,
        *,
        running: bool = True,
        workers: int = 1,
        raises: Exception | None = None,
        response: dict[str, object] | None = None,
    ) -> None:
        self._running = running
        self._workers = workers
        self._raises = raises
        self._response = response

    @property
    def is_daemon_running(self) -> bool:
        return self._running

    # Widened from AiiDA's own `dict[str, t.Any]`: a case returns an error
    # payload, which carries no int at all.
    def get_numprocesses(self) -> dict[str, object]:
        # Running is not reachable: mid-restart the circus endpoint can be down
        # while the daemon is alive, which the check reports as a failed row.
        if self._raises is not None:
            raise self._raises
        if self._response is not None:
            return self._response
        return {"numprocesses": self._workers}


def _patch_all_checks_passing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub every subsystem probe to succeed, so a test can then break exactly
    one and assert the others are unaffected.
    """
    from aiida_agents.cli import doctor
    from aiida_agents.cli.agent import _Reachability

    monkeypatch.setattr("aiida.load_profile", lambda profile: _Profile())
    monkeypatch.setattr(
        "aiida.engine.daemon.client.get_daemon_client", lambda: _DaemonClient()
    )
    monkeypatch.setattr(
        "aiida_agents.cli.agent._probe_reachable",
        lambda settings: _Reachability("http://endpoint", 3, model_ok=True),
    )
    monkeypatch.setattr("aiida_agents.rag.store.index_status", lambda: _Index(True))
    monkeypatch.setattr(
        "aiida.manage.configuration.get_config",
        lambda: _SandboxConfig("agents-sandbox"),
    )
    monkeypatch.setattr(doctor, "_module_missing", lambda name: False)


def _rows_by_label(profile: str | None = None) -> dict[str, _DiagnosticRow]:
    from aiida_agents.cli.doctor import _run_diagnostics

    settings = ModelSettings(provider="ollama", model="m")
    return {row.label: row for row in _run_diagnostics(settings, profile)}


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
    from aiida_agents.cli.doctor import _short_reason

    assert _short_reason(exc) == expected


def test_run_diagnostics_all_checks_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """With every subsystem healthy, every check reports a passing row."""
    _patch_all_checks_passing(monkeypatch)
    rows = _rows_by_label()

    assert list(rows) == [
        "AiiDA profile loads",
        "Daemon running and reachable",
        "Model reachable (ollama:m)",
        "RAG index built",
        "Codegen sandbox (disposable copy)",
        "Docs toolchain (sphinx)",
    ]
    assert all(row.ok for row in rows.values())


@pytest.mark.parametrize(
    "client, expected_ok, needle",
    [
        pytest.param(
            _DaemonClient(running=True, workers=2), True, "2 worker(s)", id="reachable"
        ),
        pytest.param(
            _DaemonClient(running=False), False, "verdi daemon start", id="not-running"
        ),
        pytest.param(
            _DaemonClient(running=True, workers=0),
            False,
            "verdi daemon incr",
            id="zero-workers",
        ),
        pytest.param(
            _DaemonClient(running=True, raises=RuntimeError("circus unreachable")),
            False,
            "circus unreachable",
            id="running-but-unreachable",
        ),
        # AiiDA's own exception types carrying AiiDA's own wording, because what
        # is being pinned is that the remedy survives the row. `is_daemon_running`
        # reads the PID file, so a stale one is precisely how it says yes while
        # the round-trip fails: this row's main real failure, not an edge case.
        pytest.param(
            _DaemonClient(running=True, raises=DaemonStalePidException(_STALE_PID_MSG)),
            False,
            "stale PID file; run `verdi daemon start`",
            id="stale-pid-file",
        ),
        # One condition, one sentence: the daemon can stop between the PID-file
        # read and the round-trip, and that must not become a second wording.
        pytest.param(
            _DaemonClient(
                running=True,
                raises=DaemonNotRunningException("The daemon is not running."),
            ),
            False,
            "not running; run `verdi daemon start`",
            id="stopped-mid-check",
        ),
        pytest.param(
            _DaemonClient(
                running=True,
                raises=DaemonTimeoutException("Connection to the daemon timed out."),
            ),
            False,
            "verdi daemon restart",
            id="timed-out",
        ),
        # circus answers a command-level failure in the payload instead of
        # raising (circus.commands.base.error), and that payload has no
        # `numprocesses`. Reading the count off it finds nothing, which must not
        # read as a healthy daemon. Caught by CodeRabbit on the PR.
        pytest.param(
            _DaemonClient(
                running=True,
                response={"status": "error", "reason": "command not supported"},
            ),
            False,
            "command not supported",
            id="error-payload",
        ),
        pytest.param(
            _DaemonClient(running=True, response={"status": "ok"}),
            False,
            "verdi daemon restart",
            id="no-worker-count",
        ),
    ],
)
def test_check_daemon(
    monkeypatch: pytest.MonkeyPatch,
    client: _DaemonClient,
    expected_ok: bool,
    needle: str,
) -> None:
    """The daemon check separates running-and-reachable from every way it is not:
    stopped, zero workers, or up but with an unreachable circus endpoint.
    """
    from aiida_agents.cli.doctor import _check_daemon

    monkeypatch.setattr("aiida.engine.daemon.client.get_daemon_client", lambda: client)
    row = _check_daemon()

    assert row.ok is expected_ok
    assert needle in row.detail
    # AiiDA phrases the stale-PID remedy in a second sentence, 162 characters in,
    # so routing it through `_short_reason` (which truncates at 100) used to cut
    # the row off mid-word with the fix still to come.
    assert len(row.detail) < 100


@pytest.mark.parametrize(
    "target, failing_label",
    [
        pytest.param("aiida.load_profile", "AiiDA profile loads", id="profile"),
        pytest.param(
            "aiida_agents.cli.agent._probe_reachable",
            "Model reachable (ollama:m)",
            id="model",
        ),
        pytest.param(
            "aiida_agents.rag.store.index_status", "RAG index built", id="rag"
        ),
        pytest.param(
            "aiida_agents.sandbox.copy.profiles_sharing_storage",
            "Codegen sandbox (disposable copy)",
            id="sandbox",
        ),
    ],
)
def test_run_diagnostics_isolates_one_failure(
    monkeypatch: pytest.MonkeyPatch, target: str, failing_label: str
) -> None:
    """One failing check yields a failed row carrying the reason, and never
    aborts the report: every other check still runs and passes.
    """
    _patch_all_checks_passing(monkeypatch)

    def _boom(*args: object, **kwargs: object) -> object:
        raise RuntimeError("kaboom")

    monkeypatch.setattr(target, _boom)
    rows = _rows_by_label()

    assert rows[failing_label].ok is False
    assert "kaboom" in rows[failing_label].detail
    for label, row in rows.items():
        if label != failing_label:
            assert row.ok is True


@pytest.mark.parametrize(
    "provider, model_ok, expected_ok, detail_needle",
    [
        pytest.param("ollama", False, False, "not pulled", id="ollama-missing-fails"),
        pytest.param("openrouter", False, True, "not listed", id="cloud-missing-ok"),
    ],
)
def test_run_diagnostics_model_availability_policy(
    monkeypatch: pytest.MonkeyPatch,
    provider: _Provider,
    model_ok: bool,
    expected_ok: bool,
    detail_needle: str,
) -> None:
    """An unadvertised model fails the check for Ollama (authoritative listing)
    but passes with a note for a cloud endpoint (listing may be partial).
    """
    _patch_all_checks_passing(monkeypatch)
    from aiida_agents.cli.agent import _Reachability
    from aiida_agents.cli.doctor import _run_diagnostics

    monkeypatch.setattr(
        "aiida_agents.cli.agent._probe_reachable",
        lambda settings: _Reachability("http://endpoint", 3, model_ok=model_ok),
    )
    settings = ModelSettings(provider=provider, model="m")
    rows = {row.label: row for row in _run_diagnostics(settings, None)}
    model_row = rows[f"Model reachable ({provider}:m)"]

    assert model_row.ok is expected_ok
    assert detail_needle in model_row.detail


@pytest.mark.parametrize(
    "break_target, label, needle",
    [
        pytest.param(
            lambda mp: mp.setattr(
                "aiida_agents.rag.store.index_status",
                lambda: _Index(False),
            ),
            "RAG index built",
            "rag build",
            id="unbuilt-rag-index",
        ),
        pytest.param(
            lambda mp: mp.setattr(
                "aiida_agents.rag.store.index_status",
                lambda: _Index(False, stale=(object(),)),
            ),
            "RAG index built",
            "--force",
            id="stale-rag-index",
        ),
        pytest.param(
            lambda mp: mp.setattr(
                "aiida_agents.cli.doctor._module_missing", lambda name: True
            ),
            "Docs toolchain (sphinx)",
            "rag build",
            id="missing-docs-toolchain",
        ),
        pytest.param(
            lambda mp: mp.setattr(
                "aiida.manage.configuration.get_config", lambda: _SandboxConfig(None)
            ),
            "Codegen sandbox (disposable copy)",
            "sandbox init",
            id="unset-sandbox",
        ),
    ],
)
def test_run_diagnostics_flags_optional_provisioning(
    monkeypatch: pytest.MonkeyPatch,
    break_target: Callable[[pytest.MonkeyPatch], None],
    label: str,
    needle: str,
) -> None:
    """A not-yet-built RAG index and a missing docs toolchain each fail their own
    row with an actionable hint, without failing anything else.
    """
    _patch_all_checks_passing(monkeypatch)
    break_target(monkeypatch)
    rows = _rows_by_label()

    assert rows[label].ok is False
    assert needle in rows[label].detail


@pytest.mark.parametrize(
    "rows, exit_code",
    [
        pytest.param([("Profile", True, ""), ("Model", True, "ok")], 0, id="all-ok"),
        pytest.param([("Profile", True, ""), ("Model", False, "down")], 1, id="a-fail"),
    ],
)
def test_doctor_exit_code_reflects_health(
    monkeypatch: pytest.MonkeyPatch, rows: list[tuple[str, bool, str]], exit_code: int
) -> None:
    """`doctor` exits non-zero when any check fails, zero when all pass."""
    from aiida_agents.cli import doctor

    monkeypatch.setattr(
        doctor,
        "_run_diagnostics",
        lambda settings, profile: [_DiagnosticRow(*row) for row in rows],
    )
    result = CliRunner().invoke(cli, ["doctor"])

    assert result.exit_code == exit_code


def test_a_sandbox_sharing_storage_fails_the_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sandbox that shares storage is worse than no sandbox at all.

    With none configured the Codegen agent knows it cannot run code and says
    so. With a *sharing* one, everything looks healthy right up until somebody
    deletes the profile they were told was disposable and takes the real
    database with it, which is exactly what issue #73 records. Existing is
    therefore not enough for this row to pass.
    """
    _patch_all_checks_passing(monkeypatch)
    monkeypatch.setattr(
        "aiida.manage.configuration.get_config",
        lambda: _SandboxConfig("agents-sandbox", filepath="/data/real"),
    )
    row = _rows_by_label()["Codegen sandbox (disposable copy)"]

    assert row.ok is False
    assert "real" in row.detail


def test_a_sharing_sandbox_is_not_sent_to_a_command_that_would_refuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This row used to end in "rebuild with `aiida-agents sandbox refresh`".

    Teardown refuses a sandbox in exactly this state, so that advice sent the
    reader to a command that could not run, on the one row where they most need
    a next step that works. There is no `refresh` at all now, which makes the
    assertion cheap and worth keeping: the row must not name a remedy that
    cannot be carried out.
    """
    _patch_all_checks_passing(monkeypatch)
    monkeypatch.setattr(
        "aiida.manage.configuration.get_config",
        lambda: _SandboxConfig("agents-sandbox", filepath="/data/real"),
    )
    row = _rows_by_label()["Codegen sandbox (disposable copy)"]

    assert "refresh" not in row.detail


def test_an_invalid_sandbox_setting_names_the_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The row says which setting is wrong, and the report still prints.

    Left to the broad ``except Exception`` this rendered as "1 validation error
    for SandboxSettings", naming neither the setting nor the fix. It stays a row
    rather than becoming a raised error because diagnosing a broken setup is the
    whole job: the other checks are still worth seeing.
    """
    from aiida_agents.cli.doctor import _check_sandbox

    monkeypatch.setenv("AIIDA_AGENTS_SNIPPET_TIMEOUT", "10000")

    row = _check_sandbox()

    assert row.ok is False
    assert "snippet_timeout" in row.detail
    assert "less than or equal to 300" in row.detail
    assert "\n" not in row.detail  # one table cell, one line
