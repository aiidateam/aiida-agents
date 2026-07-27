"""Tests for the routing agent.

Whether a *real* model routes well is measured by the opt-in eval tier
(``tests/evals``). These pin the parts that must hold regardless of the model:
that the reply is read strictly, that an unreadable reply degrades to the
read-only specialist rather than to the one that can submit, and that the
router has no tools.
"""

from __future__ import annotations

import logging

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from aiida_agents.agents.router import _parse_choice, get_router, route


def _replying(text: str) -> FunctionModel:
    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(text)])

    return FunctionModel(model_fn)


class TestParsingTheReply:
    @pytest.mark.parametrize(
        "reply,expected",
        [
            ("analysis", "analysis"),
            ("execution", "execution"),
            # A model that adds punctuation or casing is still unambiguous.
            ("Execution.", "execution"),
            (" ANALYSIS \n", "analysis"),
            ('"execution"', "execution"),
            # Wrapped in a sentence, but naming exactly one specialist.
            ("This should go to the execution agent.", "execution"),
        ],
    )
    def test_reads_an_unambiguous_reply(self, reply: str, expected: str) -> None:
        assert _parse_choice(reply) == expected

    @pytest.mark.parametrize(
        "reply",
        [
            # Names both: resolving this by whichever is checked first would be
            # a guess dressed as a decision.
            "not execution, use analysis",
            "either analysis or execution would work",
            # Names neither.
            "I'm not sure what you mean",
            "",
        ],
    )
    def test_refuses_an_ambiguous_reply(self, reply: str) -> None:
        assert _parse_choice(reply) is None


class TestFallbackIsAlwaysTheReadOnlySpecialist:
    """A routing failure must never land on the agent that can write.

    Routing is a model call, so it can fail or ramble. Degrading to
    ``execution`` would mean an unparsed reply could put a user in front of a
    submission flow they never asked for; degrading to ``analysis`` costs them
    at worst a redirect.
    """

    def test_unreadable_reply_falls_back_to_analysis(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        import aiida_agents.agents.router as router_module

        rambling = Agent(_replying("hmm, could be either one really"), output_type=str)
        monkeypatch.setattr(router_module, "get_router", lambda *a, **k: rambling)

        with caplog.at_level(logging.WARNING):
            assert route("anything") == "analysis"

        assert "names no single specialist" in caplog.text

    def test_a_failing_router_falls_back_to_analysis(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        import aiida_agents.agents.router as router_module

        def boom(*args: object, **kwargs: object) -> object:
            raise RuntimeError("provider is down")

        monkeypatch.setattr(router_module, "get_router", boom)

        with caplog.at_level(logging.WARNING):
            assert route("anything") == "analysis"

        assert "routing failed" in caplog.text


class TestRouterShape:
    def test_router_has_no_tools(self) -> None:
        """Routing is a decision, not an action -- it must not reach the database."""
        from pydantic_ai.models.test import TestModel

        agent = get_router()
        fake = TestModel(call_tools=[])
        with agent.override(model=fake):
            agent.run_sync("ping")

        params = fake.last_model_request_parameters
        assert params is not None
        assert params.function_tools == []

    def test_scripted_reply_is_routed(self) -> None:
        agent = get_router()
        with agent.override(model=_replying("execution")):
            assert agent.run_sync("relax this structure").output.strip() == "execution"
