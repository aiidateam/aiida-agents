"""Tests for the planning agent.

Whether a *real* model plans well is measured by the opt-in eval tier
(``tests/evals``). These pin the parts that must hold regardless of the model:
that a reply is read strictly, that anything unusable degrades to the read-only
specialist rather than the one that can submit, and that the planner has no
tools.
"""

from __future__ import annotations

import logging

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from aiida_agents.agents.planner import MAX_STEPS, Step, _parse_plan, get_planner, plan


def _replying(text: str) -> FunctionModel:
    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(text)])

    return FunctionModel(model_fn)


class TestParsingASingleStep:
    """The common case: one specialist, and usually the user's own words."""

    @pytest.mark.parametrize(
        "reply,expected",
        [
            ("analysis: how many workchains finished", "analysis"),
            ("execution: relax the structure at pk 512", "execution"),
            ("Execution: relax pk 512", "execution"),
            ("  analysis:   count the nodes  ", "analysis"),
        ],
    )
    def test_one_line_yields_one_step(self, reply: str, expected: str) -> None:
        steps = _parse_plan(reply)
        assert steps is not None
        assert len(steps) == 1
        assert steps[0].specialist == expected
        assert steps[0].task

    @pytest.mark.parametrize("reply", ["analysis", "execution", " Analysis \n"])
    def test_a_bare_specialist_name_is_still_a_plan(self, reply: str) -> None:
        """The output format before this agent could plan.

        A model answering that way is being unambiguous, not wrong, so it is
        read as one step over the user's original request rather than rejected.
        """
        steps = _parse_plan(reply)
        assert steps is not None
        assert len(steps) == 1
        assert steps[0].task == "", "an empty task means 'use the request as given'"


class TestParsingAMultiStepPlan:
    def test_two_steps_keep_their_order_and_tasks(self) -> None:
        steps = _parse_plan(
            "analysis: find out why pk 1234 failed\n"
            "execution: resubmit it with a longer wallclock"
        )

        assert steps == [
            Step("analysis", "find out why pk 1234 failed"),
            Step("execution", "resubmit it with a longer wallclock"),
        ]

    @pytest.mark.parametrize(
        "reply",
        [
            # Numbered.
            "1. analysis: diagnose pk 1\n2. execution: resubmit pk 1",
            # Bulleted.
            "- analysis: diagnose pk 1\n- execution: resubmit pk 1",
            # Fenced, with blank lines.
            "```\nanalysis: diagnose pk 1\n\nexecution: resubmit pk 1\n```",
        ],
    )
    def test_formatting_a_model_adds_is_tolerated(self, reply: str) -> None:
        """Strict about content, tolerant of decoration.

        Numbering, bullets and code fences are how a model formats a list, not
        a difference in what it planned. Rejecting them would send perfectly
        good plans to the fallback.
        """
        steps = _parse_plan(reply)
        assert steps is not None
        assert [s.specialist for s in steps] == ["analysis", "execution"]
        assert steps[0].task == "diagnose pk 1"


class TestAnUnusablePlanIsRejectedWhole:
    """A plan missing a step is worse than no plan.

    The steps that did run would then be acting on a premise nobody
    established, which is harder to notice than an outright fallback.
    """

    @pytest.mark.parametrize(
        "reply",
        [
            # A specialist that does not exist -- silently dropping this line
            # would run a truncated plan.
            "analysis: diagnose pk 1\nvalidator: check the inputs",
            # No task.
            "analysis: diagnose pk 1\nexecution:",
            # No specialist at all.
            "analysis: diagnose pk 1\nthen resubmit it",
            # Prose instead of a plan.
            "I think you should probably ask the execution agent about this.",
            "",
        ],
    )
    def test_any_unusable_line_rejects_the_plan(self, reply: str) -> None:
        assert _parse_plan(reply) is None

    def test_a_plan_longer_than_the_cap_is_rejected(self) -> None:
        """Past a few steps a plan is speculation about findings not yet made."""
        reply = "\n".join(f"analysis: step {i}" for i in range(MAX_STEPS + 1))

        assert _parse_plan(reply) is None

    def test_a_plan_at_the_cap_is_accepted(self) -> None:
        reply = "\n".join(f"analysis: step {i}" for i in range(MAX_STEPS))

        steps = _parse_plan(reply)
        assert steps is not None and len(steps) == MAX_STEPS


class TestFallbackIsAlwaysTheReadOnlySpecialist:
    """A planning failure must never land on the agent that can write.

    Planning is a model call, so it can fail or ramble. Degrading to
    ``execution`` would mean an unparsed reply could put a user in front of a
    submission flow they never asked for; degrading to ``analysis`` costs them
    at worst a redirect.
    """

    def test_an_unusable_reply_falls_back(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        import aiida_agents.agents.planner as planner_module

        rambling = Agent(_replying("hmm, hard to say really"), output_type=str)
        monkeypatch.setattr(planner_module, "get_planner", lambda *a, **k: rambling)

        with caplog.at_level(logging.WARNING):
            steps = plan("anything")

        assert steps == [Step("analysis", "")]
        assert "not a usable plan" in caplog.text

    def test_a_failing_planner_falls_back(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        import aiida_agents.agents.planner as planner_module

        def boom(*args: object, **kwargs: object) -> object:
            raise RuntimeError("provider is down")

        monkeypatch.setattr(planner_module, "get_planner", boom)

        with caplog.at_level(logging.WARNING):
            steps = plan("anything")

        assert steps == [Step("analysis", "")]
        assert "planning failed" in caplog.text

    def test_an_over_long_plan_falls_back_rather_than_truncating(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Running the first N steps of a rejected plan would be a different plan."""
        import aiida_agents.agents.planner as planner_module

        long_plan = "\n".join(f"execution: step {i}" for i in range(MAX_STEPS + 2))
        agent = Agent(_replying(long_plan), output_type=str)
        monkeypatch.setattr(planner_module, "get_planner", lambda *a, **k: agent)

        assert plan("anything") == [Step("analysis", "")]


class TestPlannerShape:
    def test_planner_has_no_tools(self) -> None:
        """Planning is a decision, not an action -- it must not reach the database."""
        from pydantic_ai.models.test import TestModel

        agent = get_planner()
        fake = TestModel(call_tools=[])
        with agent.override(model=fake):
            agent.run_sync("ping")

        params = fake.last_model_request_parameters
        assert params is not None
        assert params.function_tools == []


class TestHandingOneStepsAnswerToTheNext:
    """The context handoff is the whole point of a multi-step plan.

    It is explicit text rather than replayed message history: the specialists
    hold different tools, so one agent's history references tools the other
    does not have.
    """

    def test_a_first_step_gets_its_task_verbatim(self) -> None:
        from aiida_agents.cli.agent import _step_prompt

        prompt = _step_prompt(Step("analysis", "diagnose pk 1"), "original", None)

        assert prompt == "diagnose pk 1"

    def test_an_empty_task_falls_back_to_the_users_own_words(self) -> None:
        """Rephrasing a request we were not asked to rephrase can only lose detail."""
        from aiida_agents.cli.agent import _step_prompt

        prompt = _step_prompt(Step("analysis", ""), "why did pk 1234 fail?", None)

        assert prompt == "why did pk 1234 fail?"

    def test_a_later_step_carries_the_previous_answer_and_its_source(self) -> None:
        from aiida_agents.cli.agent import _StepResult, _step_prompt

        previous = _StepResult("analysis", "It exceeded its wallclock in the SCF step.")
        prompt = _step_prompt(
            Step("execution", "resubmit with a longer wallclock"), "original", previous
        )

        assert "resubmit with a longer wallclock" in prompt
        assert "It exceeded its wallclock in the SCF step." in prompt
        # Attributed, so the receiving agent weighs it as a finding rather than
        # as the user's own instruction.
        assert "analysis agent" in prompt
        # And told not to fill gaps in it. Whitespace-normalised: the template
        # wraps, and the test should pin the instruction, not the line breaks.
        flowed = " ".join(prompt.split())
        assert "Do not restate values it does not contain" in flowed
