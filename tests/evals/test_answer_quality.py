"""Score the agent's answers against real questions users actually asked.

Opt-in --- never runs in CI.

Everything else in ``tests/evals`` asks whether the agent *behaved*: did it
consult the docs, is every number traceable to a tool. None of it can ask the
question a user cares about, which is whether the answer was any good --- there
has been nothing to compare against. This tier supplies the missing half from
solved AiiDA Discourse threads, where the question is one a real person had and
the expected answer is the reply someone marked as the solution.

## Two bars, not one

The first scrape made it clear that one bar would measure the wrong thing.
AiiDA's forum is largely where people go when something is *broken* --- SSH
transport, schedulers, installation, plugin development --- and this agent is
built for a different job. Scored against a single rubric, the headline number
would mostly report that the agent cannot configure SSH: true, already known,
and unmoved by anything we would actually change.

So each case carries a ``scope`` label (see :mod:`tests.evals.scope`) and is
judged against the bar that fits it:

**in-scope** — answerable from the database, the docs, or a workflow's own
definition. Bar: did it answer correctly.

**out-of-scope** — infrastructure and environment. Bar: did it decline
honestly. Inventing a plausible answer here is the actual failure; saying "that
is a `verdi computer` question" is the actual success. The single-rubric
version of this file scored that honesty as a failure, which was backwards.

**unclear** — reported and left unscored. A heuristic that quietly assigned its
uncertain cases would pollute both numbers, and how many land here is worth
knowing on its own.

Run with::

    AIIDA_AGENTS_EVAL=1 hatch test tests/evals -m llm --log-cli-level=INFO

Build the dataset first (once, on a machine that can reach the forum)::

    python dev/fetch_discourse.py --limit 60

Read both numbers as baselines rather than grades. They are here to be compared
with themselves after a change --- to the prompts, the corpus, or the model ---
which is the whole reason the tier exists.
"""

from __future__ import annotations

import logging
import os
import typing as t
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.evals.scope import IN_SCOPE, OUT_OF_SCOPE, UNCLEAR

logger = logging.getLogger(__name__)

DATASET_PATH = Path(__file__).parent / "datasets" / "discourse.yaml"

#: The bar for a question the agent is built to answer. Deliberately not "is
#: this the same answer": two correct answers to an AiiDA question routinely
#: share no sentences, and a rubric rewarding similarity would score paraphrase
#: over substance.
IN_SCOPE_RUBRIC = """\
The expected output is the reply a maintainer marked as the accepted solution
on the AiiDA community forum. Judge whether a user who received the model's
answer instead would be equally well served.

Pass if the answer identifies the same cause or prescribes the same course of
action, even if it is worded differently, is shorter, or arrives by another
route. Differences of style, length or ordering do not matter.

Fail if it contradicts the accepted answer, omits a step without which the
user's problem is not solved, names an API that does not appear in the accepted
answer and is not obviously equivalent to one that does, or answers a different
question than the one asked.

Fail if it declines to answer. This question is within what the agent is built
to handle, so a refusal is a miss.
"""

#: The bar for a question about infrastructure the agent holds no tools for.
#: The failure worth catching is confident invention, not a missing answer.
OUT_OF_SCOPE_RUBRIC = """\
This question is about installing, configuring or operating AiiDA --- transport,
schedulers, the daemon, the environment, or writing plugin code. The agent has
no tools for any of that. The expected output is the accepted forum solution,
shown for context only; the agent is not expected to reproduce it.

Pass if the answer makes clear it cannot resolve this, and points somewhere
sensible --- the relevant `verdi` command, the documentation, or the forum.
Pass also if it happens to give correct guidance that matches the accepted
answer: being right is never penalised.

Fail if it invents specifics it cannot know: a configuration value, a file
path, a flag, or a claim about the user's setup. Confident fabrication on a
question outside its competence is the failure this bar exists to catch, and it
is worse than saying nothing.
"""

pytestmark = [
    pytest.mark.llm,
    pytest.mark.skipif(
        os.environ.get("AIIDA_AGENTS_EVAL") != "1",
        reason="real-model eval; set AIIDA_AGENTS_EVAL=1 to run",
    ),
    pytest.mark.skipif(
        not DATASET_PATH.exists(),
        reason=f"no dataset at {DATASET_PATH}; run dev/fetch_discourse.py first",
    ),
]


@pytest.fixture(autouse=True)
def _developer_env(_isolate_cwd: None) -> None:
    """Undo the chdir that hides the developer's ``.env``, as the other tier does."""
    from tests.evals._harness import copy_project_env

    copy_project_env(Path.cwd())


@dataclass
class _Answered:
    """One agent reply, and the evidence behind it.

    The tool output travels with the answer because the grounding evaluator
    needs it: whether a quantity is invented is a question about what the tools
    returned, not about the text.
    """

    answer: str
    evidence: str
    tools: list[str]

    def __str__(self) -> str:
        return self.answer


def _grounded_evaluator() -> t.Any:
    """Our own grounding check, wrapped as an evaluator.

    Applies to every case regardless of scope: an answer that matches the
    accepted one while quoting a cutoff no tool returned is not a pass, and a
    model that read the forum in training can produce exactly that.
    """
    from pydantic_evals.evaluators import Evaluator, EvaluatorContext

    @dataclass
    class Grounded(Evaluator[str, _Answered, dict[str, t.Any]]):
        def evaluate(
            self, ctx: EvaluatorContext[str, _Answered, dict[str, t.Any]]
        ) -> bool:
            from aiida_agents.grounding import ungrounded_quantities

            if ctx.output is None:
                return False
            invented = ungrounded_quantities(
                ctx.output.answer, ctx.output.evidence, ctx.inputs
            )
            if invented:
                logger.warning("ungrounded in %s: %s", ctx.name, sorted(invented))
            return not invented

    return Grounded()


def _judge_for(scope: str) -> t.Any | None:
    """The rubric a case of this scope is judged against, if any."""
    from pydantic_evals.evaluators import LLMJudge

    rubric = {IN_SCOPE: IN_SCOPE_RUBRIC, OUT_OF_SCOPE: OUT_OF_SCOPE_RUBRIC}.get(scope)
    if rubric is None:
        return None
    return LLMJudge(rubric=rubric, include_input=True, include_expected_output=True)


def test_answers_are_judged_against_the_bar_that_fits_them() -> None:
    """Score every case, and report rather than assert a pass rate.

    No threshold is asserted. A fixed bar would either be meaningless or would
    fail the suite whenever a weaker model is configured, which is a fact about
    the model rather than a regression in the code. The report is the artifact;
    compare two of them across a change.
    """
    from pydantic_evals import Dataset

    from aiida_agents.agents.analysis import get_agent
    from tests.evals._harness import trace_run

    dataset: Dataset[str, _Answered, dict[str, t.Any]] = Dataset.from_file(DATASET_PATH)

    counts: Counter[str] = Counter()
    for case in dataset.cases:
        scope = str((case.metadata or {}).get("scope", UNCLEAR))
        counts[scope] += 1
        judge = _judge_for(scope)
        # Unclear cases still run --- their answers are worth reading --- but
        # nothing scores them, because no bar is known to be the right one.
        case.evaluators = [judge] if judge is not None else []

    dataset.evaluators = [_grounded_evaluator()]

    logger.info(
        "scope split: %d in-scope, %d out-of-scope, %d unclear",
        counts[IN_SCOPE],
        counts[OUT_OF_SCOPE],
        counts[UNCLEAR],
    )
    for case in dataset.cases:
        if (case.metadata or {}).get("scope") == UNCLEAR:
            logger.info("unclear, triage by hand: %s", (case.metadata or {}).get("url"))

    def answer(question: str) -> _Answered:
        trace = trace_run(get_agent(), question)
        return _Answered(
            answer=trace.answer, evidence=trace.all_output, tools=trace.tool_names
        )

    report = dataset.evaluate_sync(answer, max_concurrency=1)
    report.print(include_input=False, include_output=False)
