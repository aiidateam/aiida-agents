"""Grounding evals against a real model. Opt-in --- never runs in CI.

``test_harness.py`` proves the checks fire; this module points them at an
actual model, which is the only way to catch the failures that matter. They
are behavioural, so they need a real model to exhibit them, and they cost
tokens and require a built index --- hence opt-in rather than part of the
suite.

Run with::

    AIIDA_AGENTS_EVAL=1 hatch test tests/evals -m llm --log-cli-level=INFO

It runs against whatever model your ``.env`` (or exported environment)
configures, and logs the resolved provider/model on every case. Check that line
first when a run looks wrong: the very first real-model run of this tier failed
ten cases against a model it had never contacted, because the project-wide
``_isolate_cwd`` fixture hides ``.env`` from every test.

Expect these to be *flaky by nature*: a weaker model fails them more often
than a strong one, and that is information rather than a broken test. A
failure here means "this model, with these prompts, answered ungroundedly on
this question" --- read the assertion output, which prints the tool sequence
and the answer, before deciding whether the fix is the prompt, the corpus, or
the model.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest
from pydantic_ai import Agent

from tests.evals._harness import (
    RunTrace,
    assert_cited,
    assert_consulted_docs,
    assert_grounded_quantities,
    copy_project_env,
    trace_run,
)

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.llm,
    pytest.mark.skipif(
        os.environ.get("AIIDA_AGENTS_EVAL") != "1",
        reason="real-model eval; set AIIDA_AGENTS_EVAL=1 to run",
    ),
]


@pytest.fixture(autouse=True)
def _developer_env(_isolate_cwd: None) -> None:
    """Undo, for this tier only, the chdir that hides the developer's ``.env``.

    Depends on ``_isolate_cwd`` by name rather than relying on fixture ordering:
    the copy has to land in the temp directory *after* the project-wide autouse
    fixture has chdir'd into it, or it goes to the repository root instead.
    """
    copy_project_env(Path.cwd())


@pytest.fixture
def analysis_agent() -> Agent:
    from aiida_agents.agents.analysis import get_agent

    return get_agent()


@pytest.fixture
def execution_agent() -> Agent:
    from aiida_agents.agents.execution import get_agent

    return get_agent()


def _run(agent: Agent, prompt: str) -> RunTrace:
    """Run one eval case, logging the trace.

    Reviewing an eval means reading what the model actually said, including on
    the cases that passed, so surface it with ``--log-cli-level=INFO``. Failures
    do not depend on this: the assertion messages carry the tool sequence and
    the answer themselves.
    """
    from aiida_agents._settings import ModelSettings

    # Name the model in every case, passing or failing. A grounding result is
    # meaningless without knowing which model produced it, and an eval that
    # silently resolved to something other than what you configured is exactly
    # the failure this tier hit first -- it must be visible, not inferred.
    resolved = ModelSettings()
    trace = trace_run(agent, prompt)
    logger.info(
        "eval case %r [%s/%s]\n  tools: %s\n  answer: %s",
        prompt,
        resolved.provider,
        resolved.model,
        trace.tool_names,
        trace.answer,
    )
    return trace


class TestKnowledgeQuestionsReachTheDocs:
    """A question about how AiiDA works must be answered from the docs."""

    @pytest.mark.parametrize(
        "prompt",
        [
            "What is a CalcJobNode?",
            "How does the provenance graph track calculations?",
            "What does KpointsData represent?",
        ],
    )
    def test_analysis_agent_searches_and_cites(
        self, analysis_agent: Agent, prompt: str
    ) -> None:
        trace = _run(analysis_agent, prompt)
        assert_consulted_docs(trace)
        assert_cited(trace)


class TestExecutionAgentDoesNotRouteAroundTheDocs:
    """Regression: naming a WorkChain used to bypass documentation entirely.

    The execution prompt's imperative ``discover -> ... -> execute`` progression
    was introduced by "asks to set up a workflow", which these questions match
    on the surface -- so the agent walked a six-step sequence with no
    documentation step and answered from memory. These are the cases that
    defect produced.
    """

    @pytest.mark.parametrize(
        "prompt",
        [
            "What kpoints_distance should I use for PwBandsWorkChain?",
            "What does the clean_workdir input of PwRelaxWorkChain do?",
        ],
    )
    def test_parameter_question_consults_docs(
        self, execution_agent: Agent, prompt: str
    ) -> None:
        trace = _run(execution_agent, prompt)
        assert_consulted_docs(trace)

    @pytest.mark.parametrize(
        "prompt",
        [
            "What kpoints_distance should I use for PwBandsWorkChain?",
            "What ecutwfc is recommended for silicon with SSSP efficiency?",
        ],
    )
    def test_no_invented_physics_values(
        self, execution_agent: Agent, prompt: str
    ) -> None:
        """Every cutoff or spacing stated must appear in something a tool returned."""
        trace = _run(execution_agent, prompt)
        assert_grounded_quantities(trace, prompt)


class TestSetupRequestsStillRouteToTheProgression:
    """Guards the *other* direction: the routing fix must not over-correct.

    Splitting knowledge questions away from setup requests risks the opposite
    failure -- an actual "run this" request detouring into documentation
    instead of discovering entry points. These pin that setup still discovers.
    """

    @pytest.mark.parametrize(
        "prompt,expected_tool",
        [
            ("What workflows can I run?", "list_workflows"),
            ("Set up a multiply_add calculation for me", "list_workflows"),
        ],
    )
    def test_setup_request_discovers_entry_points(
        self,
        execution_agent: Agent,
        prompt: str,
        expected_tool: str,
    ) -> None:
        trace = _run(execution_agent, prompt)
        assert trace.called(expected_tool), (
            f"setup request did not call {expected_tool!r}; "
            f"tools called: {trace.tool_names}"
        )


class TestMissingIndexIsReportedNotPaperedOver:
    """With no index at all, the honest answer is "I can't check"."""

    def test_agent_admits_it_cannot_look_things_up(
        self,
        analysis_agent: Agent,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("AIIDA_AGENTS_VECTOR_DB_PATH", str(tmp_path / "empty"))

        trace = _run(analysis_agent, "What is a CalcJobNode?")

        lowered = trace.answer.lower()
        assert any(
            phrase in lowered
            for phrase in ("not been built", "unavailable", "rag build", "cannot")
        ), (
            "agent answered a docs question with no index instead of saying it "
            f"could not look it up:\n{trace.answer}"
        )


class TestRoutingPicksTheRightSpecialist:
    """Whether a real model routes well -- the one thing only a model can answer.

    The unit tests in ``tests/agents/router`` pin the parsing and the fallback;
    they cannot tell you whether "relax this structure" reads as execution to
    an actual model. ADR-09 makes mis-routing an accepted failure mode on the
    grounds that it is measurable, and this is the measurement.
    """

    @pytest.mark.parametrize(
        "question,expected",
        [
            # Unambiguously execution: something is to be run or set up.
            ("relax the silicon structure at pk 512", "execution"),
            ("what workflows can I run?", "execution"),
            ("submit a band structure calculation", "execution"),
            # Asked in order to configure a run, and the historical-statistics
            # tool lives on execution.
            ("what ecutwfc did my successful relaxations use?", "execution"),
            # Unambiguously analysis: about data that already exists.
            ("how many workchains finished successfully?", "analysis"),
            ("why did pk 1234 fail?", "analysis"),
            ("show me the structures with the highest band gap", "analysis"),
            ("what is a CalcJobNode?", "analysis"),
        ],
    )
    def test_request_routes_to_the_expected_specialist(
        self, question: str, expected: str
    ) -> None:
        from aiida_agents.agents.router import route

        chosen = route(question)
        logger.info("routed %r -> %s (expected %s)", question, chosen, expected)
        assert chosen == expected

    def test_a_mixed_request_routes_to_the_read_only_specialist(self) -> None:
        """ "Diagnose then resubmit" must not jump straight to submitting.

        Until multi-step coordination exists, the honest handling of a mixed
        request is the half that has to happen first -- and the half that
        cannot write. Routing it to execution would offer a resubmission before
        the user has been shown a reason for one.
        """
        from aiida_agents.agents.router import route

        chosen = route("why did pk 1234 fail, and resubmit it with a longer wallclock")
        logger.info("routed the mixed request -> %s", chosen)
        assert chosen == "analysis"
