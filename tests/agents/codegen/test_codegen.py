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

EXPECTED_TOOLS = {"search_aiida_code", "run_aiida_code", "search_aiida_docs"}


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
            "aiida_agents.sandbox.setup.sandbox_profile_exists",
            lambda profile: False,
        )
        ran: list[str] = []
        monkeypatch.setattr(
            "aiida_agents.sandbox.run_in_sandbox",
            lambda *a, **k: ran.append("ran"),
        )

        result = execution.run_aiida_code("print(1)")

        assert ran == []
        assert "sandbox init" in result

    def test_the_refusal_forbids_claiming_it_ran(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiida_agents.tools.codegen import execution

        monkeypatch.setattr(
            "aiida_agents.sandbox.setup.sandbox_profile_exists",
            lambda profile: False,
        )

        assert "do NOT claim to have run it" in execution.run_aiida_code("print(1)")

    def test_it_runs_against_the_configured_profile(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Never a default resolved elsewhere: this setting is the safety boundary."""
        from aiida_agents.tools.codegen import execution

        monkeypatch.setenv("AIIDA_AGENTS_SANDBOX_PROFILE", "my-readonly")
        monkeypatch.setattr(
            "aiida_agents.sandbox.setup.sandbox_profile_exists",
            lambda profile: True,
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

        assert execution.run_aiida_code("print(42)") == "42"
        assert seen["profile"] == "my-readonly"
