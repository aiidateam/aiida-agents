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
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from aiida_agents.agents.planner import (
    MAX_STEPS,
    _SPECIALISTS,
    _SYSTEM_PROMPT,
    Specialist,
    Step,
    _as_specialist,
    _parse_plan,
    get_planner,
    plan,
)


def _conversation() -> list[ModelMessage]:
    """One prior turn, as the REPL records it: the user's words, then the answer.

    A function rather than a module constant, so no two cases share a list.
    """
    return [
        ModelRequest(parts=[UserPromptPart(content="search for silicon structures")]),
        ModelResponse(parts=[TextPart("I found PK 105 and PK 150.")]),
    ]


def _replying(text: str) -> FunctionModel:
    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(text)])

    return FunctionModel(model_fn)


class TestRoutingAFollowUp:
    """What reaches the model when a follow-up is routed."""

    def test_the_conversation_reaches_the_model_in_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ "the former" is only routable alongside what it refers back to.

        Asserted as the whole transcript rather than as membership: a history
        replayed reversed, or twice, resolves the reference to the wrong turn
        and would satisfy an ``in`` check either way.
        """
        seen: dict[str, list[ModelMessage]] = {}

        def capture(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen["messages"] = messages
            return ModelResponse(parts=[TextPart("analysis")])

        monkeypatch.setattr(
            "aiida_agents.agents.planner.get_planner",
            lambda model_settings=None, ollama_settings=None: Agent(
                FunctionModel(capture), instructions=_SYSTEM_PROMPT, output_type=str
            ),
        )

        steps = plan("relax the former", message_history=_conversation())

        assert [
            part.content
            for message in seen["messages"]
            for part in message.parts
            if isinstance(part, UserPromptPart | TextPart)
            and isinstance(part.content, str)
        ] == [
            "search for silicon structures",
            "I found PK 105 and PK 150.",
            "relax the former",
        ]
        assert [step.specialist for step in steps] == ["analysis"]

    @pytest.mark.parametrize(
        "history",
        [
            pytest.param(None, id="first-turn"),
            pytest.param(_conversation(), id="follow-up"),
        ],
    )
    def test_the_planner_keeps_its_routing_prompt(
        self, history: list[ModelMessage] | None
    ) -> None:
        """The prompt is what makes a reply parseable; without it a turn routes blind.

        pydantic-ai emits a ``system_prompt`` only on a run that starts from an
        empty history, so a planner built with one loses it the moment a
        follow-up arrives. Both turns are checked, so neither can regress alone.

        Driven through the real :func:`get_planner` with only its model swapped:
        how that function attaches the prompt is the thing under test, and a
        stand-in agent built here would pin this test's own wiring. Asserted on
        the prompt arriving rather than on which field carries it.
        """
        seen: list[list[ModelMessage]] = []

        def capture(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen.append(messages)
            return ModelResponse(parts=[TextPart("analysis")])

        agent = get_planner()
        with agent.override(model=FunctionModel(capture)):
            agent.run_sync("relax the former", message_history=history)

        (messages,) = seen
        delivered = [
            message.instructions
            for message in messages
            if isinstance(message, ModelRequest)
        ] + [
            part.content
            for message in messages
            for part in message.parts
            if isinstance(part, SystemPromptPart)
        ]
        assert _SYSTEM_PROMPT in delivered, (
            "the routing prompt never reached the model; it has to choose a "
            "specialist without knowing what the specialists are"
        )


class TestParsingASingleStep:
    """The common case: one specialist, and usually the user's own words."""

    @pytest.mark.parametrize(
        "reply,expected",
        [
            ("analysis: how many workchains finished", "analysis"),
            ("execution: relax the structure at pk 512", "execution"),
            ("Execution: relax pk 512", "execution"),
            ("  analysis:   count the nodes  ", "analysis"),
            ("codegen: tabulate every relaxation with its final energy", "codegen"),
            ("Codegen: tabulate the relaxations", "codegen"),
        ],
    )
    def test_one_line_yields_one_step(self, reply: str, expected: str) -> None:
        steps = _parse_plan(reply)
        assert steps is not None
        assert len(steps) == 1
        assert steps[0].specialist == expected
        assert steps[0].task

    @pytest.mark.parametrize(
        "reply", ["analysis", "execution", "codegen", " Analysis \n"]
    )
    def test_a_bare_specialist_name_is_still_a_plan(self, reply: str) -> None:
        """The output format before this agent could plan.

        A model answering that way is being unambiguous, not wrong, so it is
        read as one step over the user's original request rather than rejected.
        """
        steps = _parse_plan(reply)
        assert steps is not None
        assert len(steps) == 1
        assert steps[0].task == "", "an empty task means 'use the request as given'"


class TestEverySpecialistIsReachable:
    """A plan may name any specialist, and gets the one it named.

    These exist because of a real bug: the narrowing helper tested for
    ``execution`` and returned ``analysis`` for anything else, so when
    ``codegen`` was added, every step routed to it ran the Analysis agent
    instead. Nothing raised. The Analysis agent simply answered the question
    with its own tools, plausibly enough that only reading the debug log would
    have shown it. The loop over ``_SPECIALISTS`` is deliberate: a fourth
    specialist is then covered on the day it is added, not the day someone
    notices.
    """

    @pytest.mark.parametrize("name", _SPECIALISTS)
    def test_a_specialist_narrows_to_itself(self, name: Specialist) -> None:
        assert _as_specialist(name) == name

    @pytest.mark.parametrize("name", _SPECIALISTS)
    def test_a_step_reaches_the_specialist_it_names(self, name: Specialist) -> None:
        steps = _parse_plan(f"{name}: do the thing")
        assert steps == [Step(name, "do the thing")]

    def test_an_unknown_name_is_refused_rather_than_defaulted(self) -> None:
        """Defaulting is what hid the bug; the helper now says so out loud."""
        with pytest.raises(ValueError, match="is not a specialist"):
            _as_specialist("diagnosis")


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
