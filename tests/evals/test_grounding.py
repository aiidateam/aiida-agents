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
import re
from pathlib import Path

import pytest
from aiida import orm
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
            ("What workflows can I run?", "list_process_entry_points"),
            ("Set up a multiply_add calculation for me", "list_process_entry_points"),
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


class TestPlanningPicksTheRightSpecialist:
    """Whether a real model routes well -- the one thing only a model can answer.

    The unit tests in ``tests/agents/planner`` pin the parsing and the fallback;
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
            # A preview is still execution's: only it holds the tools that
            # build a spec. Routed to analysis, a real session answered in
            # prose and invented a cost figure rather than saying it could not.
            (
                "show me what a re-run of pk 1234 would look like with a higher "
                "cutoff, don't submit it",
                "execution",
            ),
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
        from aiida_agents.agents.planner import plan

        steps = plan(question)
        chosen = steps[0].specialist
        logger.info("routed %r -> %s (expected %s)", question, chosen, expected)
        assert chosen == expected

    def test_a_mixed_request_routes_to_the_read_only_specialist(self) -> None:
        """ "Diagnose then resubmit" must not jump straight to submitting.

        Until multi-step coordination exists, the honest handling of a mixed
        request is the half that has to happen first -- and the half that
        cannot write. Routing it to execution would offer a resubmission before
        the user has been shown a reason for one.
        """
        from aiida_agents.agents.planner import plan

        steps = plan("why did pk 1234 fail, and resubmit it with a longer wallclock")
        logger.info(
            "planned the mixed request -> %s", [(s.specialist, s.task) for s in steps]
        )
        assert steps[0].specialist == "analysis", (
            "diagnosis has to happen before a resubmission can be built sensibly"
        )


class TestNoInventedNumbersFromDatabaseResults:
    """The fabrication that actually shipped, as an eval case.

    A real run went: the router replied with garbage, the strict parser
    refused it and fell back to the read-only analysis agent -- correctly --
    but that agent has no tool for aggregate statistics. Rather than saying
    so, it searched the database, found real structure labels and no cutoffs,
    and answered "ecutwfc values vary widely (e.g., 42 for gold, 8 for
    different alkali metals)".

    'Au' and 'K' were real query results. 42 and 8 came from nowhere. Bolting
    invented numbers onto retrieved labels is the worst failure available
    here: it reads as sourced and cannot be checked.

    It slipped through because the grounding rule was written under a heading
    about *retrieved documentation*, so it read as governing
    ``search_aiida_docs`` rather than ``query_nodes``. The rule is now about
    any tool's output; these cases hold it to that.
    """

    @pytest.mark.parametrize(
        "prompt",
        [
            "what ecutwfc did my successful relaxations use?",
            "what cutoffs do the structures in my database typically use?",
        ],
    )
    def test_analysis_agent_invents_no_numbers(
        self, analysis_agent: Agent, prompt: str
    ) -> None:
        """Asked for statistics it has no tool for, it must not estimate them."""
        trace = _run(analysis_agent, prompt)
        assert_grounded_quantities(trace, prompt)

    def test_analysis_agent_says_statistics_are_not_its_tool(
        self, analysis_agent: Agent
    ) -> None:
        """The honest answer names the agent that does have the tool."""
        trace = _run(analysis_agent, "what ecutwfc did my successful relaxations use?")

        lowered = trace.answer.lower()
        # `"not" in lowered` was the original test and certified anything:
        # "another", "cannot", "nothing" and "note" all contain it, so no
        # answer in English could fail. Look for the actual refusal instead.
        redirected = "execution agent" in lowered
        declined = any(
            phrase in lowered
            for phrase in (
                "cannot answer",
                "can't answer",
                "do not have",
                "don't have",
                "no tool",
                "not able to",
                "unable to",
            )
        )
        assert redirected or declined, (
            "analysis agent neither redirected to the execution agent nor said "
            f"it could not answer:\n{trace.answer}"
        )

    def test_execution_agent_quotes_the_units_it_was_given(
        self, execution_agent: Agent
    ) -> None:
        """A cutoff must not be relabelled between Ry and eV on the way out."""
        trace = _run(execution_agent, "what ecutwfc did my successful runs use?")

        if "ecutwfc" in trace.all_output:
            # The original searched for the substring "ev", which appears in
            # "however", "several", "every" and "relevant" -- so it failed on
            # ordinary prose while saying nothing about units. What matters is
            # a *number* labelled eV, which is what the unit regex already
            # finds; a bare mention of electronvolts in passing is not the bug.
            mislabelled = re.findall(r"\d+(?:\.\d+)?\s*(?:eV|meV)\b", trace.answer)
            assert not mislabelled, (
                f"answer labelled a cutoff in eV ({', '.join(mislabelled)}); "
                f"query_run_context reports Ry:\n{trace.answer}"
            )


class TestFailureDiagnosisReachesTheDiagnosisTool:
    """A failure question has to be resolved, not narrated.

    ``diagnose_process_failure`` exists because the three things a diagnosis
    needs --- which process really broke, what its exit code means, what the
    workflow already tried --- were being inferred by the model from a log.
    Adding the tool does not make the model reach for it, and the prompt
    ordering that tells it to is exactly the kind of instruction that decays
    silently as the prompt grows. This is the measurement.

    The fixture is a real nested failure (``failed_multiply_add``): the work
    chain exits 400 saying only that a sub-process failed, and the cause ---
    exit 410, a negative sum --- is one level down. An agent that stops at the
    top-level code has not answered the question.
    """

    def test_a_failure_question_reaches_the_diagnosis_tool(
        self, analysis_agent: Agent, failed_multiply_add: orm.WorkChainNode
    ) -> None:
        trace = _run(analysis_agent, f"why did pk {failed_multiply_add.pk} fail?")

        assert trace.called("diagnose_process_failure"), (
            "a 'why did it fail' question was answered without diagnosing it; "
            f"tools called: {trace.tool_names}"
        )

    def test_the_diagnosis_comes_before_the_log(
        self, analysis_agent: Agent, failed_multiply_add: orm.WorkChainNode
    ) -> None:
        """The prompt's ladder, pinned.

        Reading the report first is not wrong, only wasteful and prone to
        stopping at a symptom: the diagnosis already resolves what the log has
        to be read to work out.
        """
        trace = _run(analysis_agent, f"why did pk {failed_multiply_add.pk} fail?")

        names = trace.tool_names
        if "get_process_report" not in names:
            return
        assert names.index("diagnose_process_failure") < names.index(
            "get_process_report"
        ), f"read the log before diagnosing; tools called: {names}"

    def test_the_answer_reaches_the_calculation_not_just_the_workchain(
        self, analysis_agent: Agent, failed_multiply_add: orm.WorkChainNode
    ) -> None:
        """The root cause is a level below the pk the user asked about.

        Asserted against the tool output rather than the prose, so a model that
        phrases the cause in its own words still passes and one that never
        obtained it still fails.
        """
        trace = _run(analysis_agent, f"why did pk {failed_multiply_add.pk} fail?")

        assert "410" in trace.all_output, (
            "never retrieved the nested calculation's exit code, so any cause "
            f"named in the answer was inferred:\n{trace.answer}"
        )

    def test_a_failure_explanation_invents_no_numbers(
        self, analysis_agent: Agent, failed_multiply_add: orm.WorkChainNode
    ) -> None:
        """An invented exit code or parameter reads exactly like a real one."""
        prompt = f"why did pk {failed_multiply_add.pk} fail?"
        trace = _run(analysis_agent, prompt)

        assert_grounded_quantities(trace, prompt)


class TestMultiStepPlanning:
    """Does a real model split a request only when it genuinely needs splitting?

    Two ways to be wrong, and both cost something. Planning one step for a
    request that needs two produces an answer built on a premise nobody
    checked. Planning two for a request that needs one spends an extra model
    call and an extra chance to drift.
    """

    @pytest.mark.parametrize(
        "prompt",
        [
            "why did pk 1234 fail, and resubmit it with a longer wallclock",
            "find my most recent failed relaxation and resubmit it with a higher cutoff",
        ],
    )
    def test_a_diagnose_then_act_request_gets_more_than_one_step(
        self, prompt: str
    ) -> None:
        from aiida_agents.agents.planner import plan

        steps = plan(prompt)
        logger.info("planned %r -> %s", prompt, [(s.specialist, s.task) for s in steps])

        assert len(steps) >= 2, "the resubmission depends on what the diagnosis finds"
        assert steps[0].specialist == "analysis"
        assert steps[-1].specialist == "execution"

    @pytest.mark.parametrize(
        "prompt",
        [
            "how many workchains finished successfully?",
            "what workflows can I run?",
            "relax the silicon structure at pk 512",
            "why did pk 1234 fail?",
        ],
    )
    def test_a_single_specialist_request_stays_one_step(self, prompt: str) -> None:
        """Including "relax pk 512" -- the execution agent does discovery itself.

        And "why did pk 1234 fail?" -- the user did not ask for a
        resubmission, so planning one would be acting beyond the request.
        """
        from aiida_agents.agents.planner import plan

        steps = plan(prompt)
        logger.info("planned %r -> %s", prompt, [(s.specialist, s.task) for s in steps])

        assert len(steps) == 1, "no step here depends on another step's findings"
