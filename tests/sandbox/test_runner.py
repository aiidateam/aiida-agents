"""Tests for running a snippet in a subprocess.

Run against real subprocesses rather than a mocked ``subprocess.run``: the
behaviour worth doubting is what happens on a timeout, a traceback and a
non-zero exit, and a mock would assert that we call the function we can already
see we call. These do not load an AiiDA profile --- that path needs a database,
and what is tested here is the containment around the code, not AiiDA itself.
"""

from __future__ import annotations

import typing as t

import pytest

from aiida_agents.sandbox.runner import (
    MAX_OUTPUT_CHARS,
    MAX_TIMEOUT_SECONDS,
    SandboxResult,
    run_in_sandbox,
)


class TestRunning:
    def test_output_comes_back(self) -> None:
        result = run_in_sandbox("print(6 * 7)")

        assert result.ok
        assert result.stdout.strip() == "42"

    def test_a_clean_run_is_ok(self) -> None:
        assert run_in_sandbox("x = 1").ok

    def test_the_duration_is_recorded(self) -> None:
        assert run_in_sandbox("print(1)").duration_seconds > 0


class TestFailures:
    """A traceback is a result, not an exception for the caller to handle."""

    def test_a_raising_snippet_returns_rather_than_raises(self) -> None:
        result = run_in_sandbox("raise ValueError('boom')")

        assert not result.ok
        assert "ValueError: boom" in result.stderr

    def test_the_traceback_is_what_a_retry_needs(self) -> None:
        """The whole point of running it: hand the model its own error back."""
        result = run_in_sandbox("import json\njson.loads('{')")

        assert "JSONDecodeError" in result.stderr

    def test_a_failure_is_not_a_timeout(self) -> None:
        assert run_in_sandbox("raise ValueError()").timed_out is False


class TestTimeout:
    def test_an_endless_loop_is_killed(self) -> None:
        result = run_in_sandbox("while True: pass", timeout=1.0)

        assert result.timed_out
        assert not result.ok

    def test_a_timeout_is_distinguishable_from_a_crash(self) -> None:
        """They call for different next moves, so they must not collapse."""
        result = run_in_sandbox("while True: pass", timeout=1.0)

        assert "Timed out" in result.summary()

    def test_the_timeout_is_capped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A generated argument must not turn into an unbounded wait."""
        import subprocess

        captured: dict[str, float] = {}
        real_run = subprocess.run

        def _spy(*args: t.Any, **kwargs: t.Any) -> t.Any:
            captured["timeout"] = float(kwargs["timeout"])
            return real_run(*args, **kwargs)

        monkeypatch.setattr(subprocess, "run", _spy)

        run_in_sandbox("pass", timeout=10_000)

        assert captured["timeout"] == MAX_TIMEOUT_SECONDS


class TestGuardIntegration:
    """Refused code must never reach a subprocess."""

    def test_forbidden_code_is_refused_before_running(self) -> None:
        result = run_in_sandbox("import os\nprint(os.listdir('/'))")

        assert result.refused
        assert not result.ok
        assert result.stdout == ""

    def test_a_refusal_explains_itself(self) -> None:
        result = run_in_sandbox("import subprocess")

        assert "Refused before running" in result.summary()
        assert "subprocess" in result.summary()

    def test_a_refusal_is_not_reported_as_a_crash(self) -> None:
        result = run_in_sandbox("import os")

        assert result.stderr == ""
        assert result.timed_out is False


class TestOutputLimits:
    def test_a_flood_of_output_is_truncated(self) -> None:
        """Ten thousand rows would otherwise fill a model's whole context."""
        result = run_in_sandbox(f"print('x' * {MAX_OUTPUT_CHARS * 2})")

        assert len(result.stdout) < MAX_OUTPUT_CHARS * 2
        assert "truncated" in result.stdout


class TestSummary:
    """One paragraph a model or a user can act on."""

    def test_a_silent_success_says_so(self) -> None:
        """Empty output reads as a failure otherwise."""
        assert "printed nothing" in run_in_sandbox("x = 1").summary()

    def test_a_successful_run_summarises_as_its_output(self) -> None:
        assert run_in_sandbox("print('hello')").summary() == "hello"

    def test_a_raise_is_labelled_as_one(self) -> None:
        assert run_in_sandbox("raise ValueError('boom')").summary().startswith("Raised")

    def test_the_three_outcomes_are_distinguishable(self) -> None:
        """Refused, timed out and raised call for different next moves."""
        refused = run_in_sandbox("import os").summary()
        raised = run_in_sandbox("raise ValueError()").summary()
        timed_out = run_in_sandbox("while True: pass", timeout=1.0).summary()

        assert len({refused, raised, timed_out}) == 3


class TestResultShape:
    def test_a_bare_result_is_not_refused(self) -> None:
        assert SandboxResult(ok=True).refused is False


class TestProfileSelection:
    """Which profile the subprocess loads.

    The entire safety argument rests on this being a read-only profile, so it
    must be the one the caller named and never a default resolved inside the
    subprocess where nobody chose it.
    """

    def test_the_named_profile_is_loaded(self) -> None:
        from aiida_agents.sandbox.runner import _script

        script = _script("print(1)", profile="readonly-sandbox")

        assert "load_profile('readonly-sandbox')" in script

    def test_the_profile_is_loaded_before_the_code(self) -> None:
        from aiida_agents.sandbox.runner import _script

        script = _script("print(1)", profile="ro")

        assert script.index("load_profile") < script.index("print(1)")

    def test_no_profile_means_no_load(self) -> None:
        """A snippet touching no database should not need one."""
        from aiida_agents.sandbox.runner import _script

        assert _script("print(1)", profile=None) == "print(1)"

    def test_nothing_falls_back_to_the_default_profile(self) -> None:
        """A default resolved in the subprocess is one the caller never chose."""
        from aiida_agents.sandbox.runner import _script

        assert "load_profile()" not in _script("print(1)", profile=None)
