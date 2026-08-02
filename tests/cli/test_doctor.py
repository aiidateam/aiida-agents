"""Tests for cli/doctor.py: the doctor command's health-check helpers."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from click.testing import CliRunner

from aiida_agents._settings import ModelSettings, _Provider
from aiida_agents.cli import cli
from aiida_agents.cli.doctor import _DiagnosticRow
from aiida_agents.sandbox.setup import ReadOnlyCheck


class _Profile:
    name = "test-profile"


class _Index:
    def __init__(self, built: bool, stale: tuple[object, ...] = ()) -> None:
        self.built = built
        # doctor distinguishes "never built" from "built but unreachable", so
        # the stub carries both facts the real IndexStatus exposes.
        self.stale = stale


def _patch_all_checks_passing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub every subsystem probe to succeed, so a test can then break exactly
    one and assert the others are unaffected.
    """
    from aiida_agents.cli import doctor
    from aiida_agents.cli.agent import _Reachability

    monkeypatch.setattr("aiida.load_profile", lambda profile: _Profile())
    monkeypatch.setattr(
        "aiida_agents.cli.agent._probe_reachable",
        lambda settings: _Reachability("http://endpoint", 3, model_ok=True),
    )
    monkeypatch.setattr("aiida_agents.rag.store.index_status", lambda: _Index(True))
    monkeypatch.setattr(
        "aiida_agents.sandbox.setup.sandbox_profile_exists", lambda profile: True
    )
    monkeypatch.setattr(
        "aiida_agents.sandbox.setup.verify_read_only",
        lambda profile, timeout=60.0: ReadOnlyCheck(True, "role cannot insert"),
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
        "Model reachable (ollama:m)",
        "RAG index built",
        "Codegen sandbox (read-only profile)",
        "Docs toolchain (sphinx)",
    ]
    assert all(row.ok for row in rows.values())


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
            "aiida_agents.sandbox.setup.verify_read_only",
            "Codegen sandbox (read-only profile)",
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
                "aiida_agents.sandbox.setup.sandbox_profile_exists",
                lambda profile: False,
            ),
            "Codegen sandbox (read-only profile)",
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


def test_a_writable_sandbox_profile_fails_the_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sandbox that can write is worse than no sandbox at all.

    With none configured the Codegen agent knows it cannot run code and says
    so. With a *writable* one it runs code believing the database will refuse
    the writes, which is the one situation the read-only role exists to
    prevent. Existing is therefore not enough for this row to pass.
    """
    _patch_all_checks_passing(monkeypatch)
    monkeypatch.setattr(
        "aiida_agents.sandbox.setup.verify_read_only",
        lambda profile, timeout=60.0: ReadOnlyCheck(False, "role CAN insert"),
    )
    row = _rows_by_label()["Codegen sandbox (read-only profile)"]

    assert row.ok is False
    assert "CAN insert" in row.detail
