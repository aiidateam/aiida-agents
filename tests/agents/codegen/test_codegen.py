"""Tests for the Codegen agent's wiring.

The claims worth pinning are about its *boundaries*, not its prose: that it
holds exactly the tools it should, that none of them can write, and that the
one which executes code refuses to run anything when no read-only profile has
been set up. The last is the case where a quiet fallback would be worst --
running against the user's writable profile because the sandbox was missing.
"""

from __future__ import annotations

import typing as t

import pytest

EXPECTED_TOOLS = {"search_aiida_examples", "run_python_snippet", "search_aiida_docs"}


@pytest.fixture
def codegen_agent(monkeypatch: pytest.MonkeyPatch) -> t.Any:
    from pydantic_ai.models.test import TestModel

    from aiida_agents.agents.codegen import get_agent

    monkeypatch.setattr(
        "aiida_agents.agents.codegen.get_model", lambda **kwargs: TestModel()
    )
    return get_agent()


class TestToolSurface:
    def test_it_exposes_exactly_the_expected_tools(self, codegen_agent: t.Any) -> None:
        """Pinned, so widening this surface is a deliberate edit."""
        from aiida_agents.agents.codegen import _TOOLS

        assert {tool.__name__ for tool in _TOOLS} == EXPECTED_TOOLS

    def test_it_holds_no_write_tool(self, codegen_agent: t.Any) -> None:
        """Its safety rests on having nothing that can write, not on a prompt."""
        from aiida_agents.agents.codegen import _TOOLS

        forbidden = {
            "execute_workflow_spec",
            "import_structure",
            "execute_workflow_batch",
        }
        assert {tool.__name__ for tool in _TOOLS} & forbidden == set()

    def test_it_declares_no_deferred_output(self, codegen_agent: t.Any) -> None:
        """Nothing here can pause for approval, so advertising it would mislead.

        Checked on the built agent rather than the source: the docstring
        explains *why* there is no deferred output, and a grep would trip on
        the explanation.
        """
        assert codegen_agent.output_type is str


class TestRegistration:
    def test_the_factory_builds_it_by_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from pydantic_ai.models.test import TestModel

        from aiida_agents.agents import get_agent

        monkeypatch.setattr(
            "aiida_agents.agents.codegen.get_model", lambda **kwargs: TestModel()
        )
        assert get_agent("codegen") is not None

    def test_the_planner_can_route_to_it(self) -> None:
        """A specialist the planner has not been told about is unreachable."""
        from aiida_agents.agents.planner import _SPECIALISTS

        assert "codegen" in _SPECIALISTS

    def test_the_planner_prompt_describes_it(self) -> None:
        """Listing it without describing it gets it chosen for the wrong things."""
        from importlib.resources import files

        prompt = files("aiida_agents.agents.planner").joinpath("prompt.md").read_text()
        assert "**`codegen`**" in prompt

    def test_the_agent_flag_accepts_it(self) -> None:
        from aiida_agents.cli.agent import _AGENT_CHOICES

        assert "codegen" in _AGENT_CHOICES

    def test_it_is_a_runnable_specialist_for_the_cli(self) -> None:
        """Otherwise `_build_agent` rejects it the way it rejects "auto"."""
        from aiida_agents.cli.agent import _SPECIALISTS

        assert "codegen" in _SPECIALISTS


class TestExecutionToolSafety:
    """What happens when no read-only profile is configured."""

    def test_it_refuses_to_run_rather_than_falling_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The worst possible fallback is the user's writable profile."""
        from aiida_agents.tools.codegen import execution

        monkeypatch.setattr(
            "aiida_agents.sandbox.copy.sandbox_profile_exists",
            lambda profile: False,
        )
        ran: list[str] = []
        monkeypatch.setattr(
            "aiida_agents.sandbox.run_in_sandbox",
            lambda *a, **k: ran.append("ran"),
        )

        result = execution.run_python_snippet("print(1)")

        assert ran == []
        assert "sandbox init" in result

    def test_the_refusal_forbids_claiming_it_ran(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiida_agents.tools.codegen import execution

        monkeypatch.setattr(
            "aiida_agents.sandbox.copy.sandbox_profile_exists",
            lambda profile: False,
        )

        assert "do NOT claim to have run it" in execution.run_python_snippet("print(1)")

    def test_it_refuses_a_profile_that_is_not_separate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`sandbox check` proves separation; this proves it again when it counts.

        The setting is a profile name and nothing stops it naming the user's
        own profile, which is issue #73 with one extra step.
        """
        from aiida_agents.sandbox.copy import Overlap, SharingProfile
        from aiida_agents.tools.codegen import execution

        monkeypatch.setattr(
            "aiida_agents.sandbox.copy.sandbox_profile_exists",
            lambda profile: True,
        )
        monkeypatch.setattr(
            "aiida_agents.sandbox.copy.profiles_sharing_storage",
            lambda config, name: [
                SharingProfile(
                    name="real", backend="core.sqlite_dos", overlap=Overlap.SHARED
                )
            ],
        )
        ran: list[str] = []
        monkeypatch.setattr(
            "aiida_agents.sandbox.run_in_sandbox",
            lambda *a, **k: ran.append("ran"),
        )

        result = execution.run_python_snippet("print(1)")

        assert ran == []
        assert "shares storage" in result
        assert "do NOT claim to have run it" in result

    def test_a_check_that_cannot_be_made_refuses_too(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Whatever went wrong, separation was not proved."""
        from aiida_agents.tools.codegen import execution

        monkeypatch.setattr(
            "aiida_agents.sandbox.copy.sandbox_profile_exists",
            lambda profile: True,
        )

        def _boom(config: object, name: str) -> list[object]:
            raise RuntimeError("no config here")

        monkeypatch.setattr("aiida_agents.sandbox.copy.profiles_sharing_storage", _boom)
        ran: list[str] = []
        monkeypatch.setattr(
            "aiida_agents.sandbox.run_in_sandbox",
            lambda *a, **k: ran.append("ran"),
        )

        assert "not run" in execution.run_python_snippet("print(1)")
        assert ran == []

    def test_it_runs_against_the_configured_profile(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Never a default resolved elsewhere: this setting is the safety boundary."""
        from aiida_agents.tools.codegen import execution

        monkeypatch.setenv("AIIDA_AGENTS_SANDBOX_PROFILE", "my-readonly")
        monkeypatch.setattr(
            "aiida_agents.sandbox.copy.sandbox_profile_exists",
            lambda profile: True,
        )
        monkeypatch.setattr(
            "aiida_agents.sandbox.copy.profiles_sharing_storage",
            lambda config, name: [],
        )
        seen: dict[str, t.Any] = {}

        class _Result:
            ok, refused, timed_out, duration_seconds = True, False, False, 0.1

            def summary(self) -> str:
                return "42"

        def _fake_run(
            code: str, profile: str | None = None, timeout: float = 0
        ) -> t.Any:
            seen["profile"] = profile
            return _Result()

        monkeypatch.setattr("aiida_agents.sandbox.run_in_sandbox", _fake_run)

        assert execution.run_python_snippet("print(42)") == "42"
        assert seen["profile"] == "my-readonly"

    def test_invalid_settings_refuse_rather_than_raise(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A bad setting is refused in words, never handed over as a traceback.

        ``snippet_timeout`` is bounded, so an out-of-range value now fails at
        load. Letting that escape would put a pydantic ``ValidationError`` in
        front of the model, which would set about debugging a snippet that was
        never the problem.
        """
        from aiida_agents.tools.codegen import execution

        monkeypatch.setenv("AIIDA_AGENTS_SNIPPET_TIMEOUT", "10000")

        result = execution.run_python_snippet("print(42)")

        assert "snippet_timeout" in result
        assert "less than or equal to 300" in result
        # The standing rule for every refusal path: never claim it ran.
        assert "do NOT claim to have run it" in result
