"""Score the agent's answers against real questions users actually asked.

Opt-in --- never runs in CI.

Everything else in ``tests/evals`` asks whether the agent *behaved*: did it
consult the docs, is every number traceable to a tool. None of it can ask the
question a user cares about, which is whether the answer was any good --- there
has been nothing to compare against. This tier supplies the missing half from
solved AiiDA Discourse threads, where the question is one a real person had and
the expected answer is the reply someone marked as the solution.

Two things are scored per case, and they are independent:

``correctness``
    An ``LLMJudge`` comparing the agent's answer to the accepted one. Not for
    wording --- the rubric asks whether a user would be equally well served,
    which is the only comparison that survives two correct answers phrased
    differently.

``grounded``
    Our own :mod:`aiida_agents.grounding` check, wrapped as an evaluator. An
    answer that matches the accepted one while quoting a cutoff no tool
    returned is not a pass; a model that has read the forum in training can
    produce exactly that.

Run with::

    AIIDA_AGENTS_EVAL=1 hatch test tests/evals -m llm --log-cli-level=INFO

Build the dataset first (once, and on a machine that can reach the forum)::

    python dev/fetch_discourse.py --limit 60

Expect a mediocre absolute score and read it as a baseline, not a grade. The
number is here to be compared with itself after a change --- to the prompts,
the corpus, or the model --- which is the whole reason the tier exists.
"""

from __future__ import annotations

import logging
import os
import typing as t
from dataclasses import dataclass
from pathlib import Path

import pytest

logger = logging.getLogger(__name__)

DATASET_PATH = Path(__file__).parent / "datasets" / "discourse.yaml"

#: What the judge is asked. Deliberately not "is this the same answer": two
#: correct answers to an AiiDA question routinely share no sentences, and a
#: rubric that rewards similarity would score paraphrase over substance.
RUBRIC = """\
The expected output is the reply that a maintainer marked as the accepted
solution on the AiiDA community forum. Judge whether a user who received the
model's answer instead would be equally well served.

Pass if the answer identifies the same cause or prescribes the same course of
action, even if it is worded differently, is shorter, or arrives by another
route. Differences of style, length or ordering do not matter.

Fail if it contradicts the accepted answer, omits a step without which the
user's problem is not solved, names an API that does not appear in the
accepted answer and is not obviously equivalent to one that does, or answers a
different question than the one asked.

If the answer says it cannot help and directs the user somewhere sensible,
that is a fail for this rubric but not a serious one: it is honest and wrong,
not confident and wrong.
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


def _make_evaluators() -> tuple[t.Any, ...]:
    """The two scores, built lazily so import does not need pydantic-evals."""
    from pydantic_evals.evaluators import Evaluator, EvaluatorContext, LLMJudge

    @dataclass
    class Grounded(Evaluator[str, _Answered, dict[str, t.Any]]):
        """Every quantity in the answer appears in some tool's output.

        The same function the CLI runs on every reply, so the check that guards
        a shipped answer and the check that scores this suite cannot drift.
        """

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

    return (
        LLMJudge(rubric=RUBRIC, include_input=True, include_expected_output=True),
        Grounded(),
    )


def test_answers_match_the_accepted_forum_solution() -> None:
    """Score every case, and report rather than assert a pass rate.

    No threshold is asserted. A fixed bar would either be low enough to be
    meaningless or would fail the suite whenever a weaker model is configured,
    which is a fact about the model rather than a regression in the code. The
    report is the artifact; compare two of them across a change.
    """
    from pydantic_evals import Dataset

    from aiida_agents.agents.analysis import get_agent
    from tests.evals._harness import trace_run

    dataset: Dataset[str, _Answered, dict[str, t.Any]] = Dataset.from_file(DATASET_PATH)
    dataset.evaluators = list(_make_evaluators())

    def answer(question: str) -> _Answered:
        trace = trace_run(get_agent(), question)
        return _Answered(
            answer=trace.answer, evidence=trace.all_output, tools=trace.tool_names
        )

    report = dataset.evaluate_sync(answer, max_concurrency=1)
    report.print(include_input=False, include_output=False)
