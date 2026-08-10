"""Tests for the eval harness itself, with no model provider involved.

An assertion that never fails is worse than no assertion: it produces a green
suite that certifies nothing. So before the harness is pointed at a real model,
these tests script the exact failures it exists to catch --- an agent that
skips the docs, and one that invents a cutoff --- and require it to catch them,
then script the corresponding correct behaviour and require it to pass.

Everything here runs on ``FunctionModel``, so it is deterministic, offline, and
belongs in CI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic_ai import Agent

from aiida_agents import grounding
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from tests.evals import _harness
from tests.evals._harness import (
    assert_cited,
    assert_consulted_docs,
    assert_grounded_quantities,
    copy_project_env,
    quantities_in,
    trace_run,
    ungrounded_quantities,
)

#: Stands in for a real retrieval hit. Contains 60 and 480 so an answer quoting
#: those is grounded, and omits 0.15 so an answer quoting *that* is not.
_DOCS_EXCERPT = (
    "[howto/pw  §  Cutoffs]\n"
    "For this pseudopotential family a wavefunction cutoff of 60.0 Ry and a "
    "charge density cutoff of 480.0 Ry are recommended."
)


def _scripted_agent(*turns: list[Any]) -> Agent:
    """An agent whose model replays ``turns``, one ModelResponse per step.

    Each turn is a list of parts. A turn holding a ``ToolCallPart`` drives the
    agent to call that tool for real; a turn holding a ``TextPart`` ends the
    run with that answer.
    """
    script = iter(turns)

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=next(script))

    agent = Agent(FunctionModel(model_fn))

    @agent.tool_plain
    def search_aiida_docs(query: str) -> str:
        """Stand-in for the real documentation search."""
        return _DOCS_EXCERPT

    return agent


class TestFabricationIsCaught:
    """The check must fire on an invented quantity and stay quiet on a quoted one."""

    def test_quantity_absent_from_every_tool_output_is_reported(self) -> None:
        """The exact defect found by hand: retrieval worked, the number didn't come from it.

        The agent searches, gets a real excerpt, and then states a k-point
        spacing that appears nowhere in it. This passed every existing test in
        the repo while being precisely the failure users hit.
        """
        agent = _scripted_agent(
            [ToolCallPart("search_aiida_docs", {"query": "pw cutoffs"})],
            [TextPart("Use a kpoints_distance of 0.15 Å⁻¹ for silicon.")],
        )

        trace = trace_run(agent, "what kpoints_distance for silicon?")

        assert trace.called("search_aiida_docs")  # retrieval genuinely happened
        assert ungrounded_quantities(trace) == {"0.15"}
        with pytest.raises(AssertionError, match="absent from every tool output"):
            assert_grounded_quantities(trace)

    def test_quantity_quoted_from_the_excerpt_passes(self) -> None:
        agent = _scripted_agent(
            [ToolCallPart("search_aiida_docs", {"query": "pw cutoffs"})],
            [TextPart("The docs recommend ecutwfc of 60 Ry [howto/pw  §  Cutoffs].")],
        )

        trace = trace_run(agent, "what ecutwfc?")

        assert ungrounded_quantities(trace) == set()
        assert_grounded_quantities(trace)

    def test_value_the_user_supplied_is_not_called_invented(self) -> None:
        """Repeating the user's own number back is not fabrication."""
        agent = _scripted_agent(
            [ToolCallPart("search_aiida_docs", {"query": "cutoff"})],
            [TextPart("Running with your requested ecutwfc of 90 Ry.")],
        )
        prompt = "submit a relax with ecutwfc 90 Ry"

        trace = trace_run(agent, prompt)

        assert ungrounded_quantities(trace, prompt) == set()


class TestSkippedRetrievalIsCaught:
    def test_answering_with_no_tool_call_fails(self) -> None:
        """The routing defect: a knowledge question answered straight from memory."""
        agent = _scripted_agent([TextPart("Just use 0.2 Å⁻¹, that's standard.")])

        trace = trace_run(agent, "what kpoints_distance for PwBandsWorkChain?")

        assert trace.tool_names == []
        with pytest.raises(AssertionError, match="without calling"):
            assert_consulted_docs(trace)

    def test_answering_after_searching_passes(self) -> None:
        agent = _scripted_agent(
            [ToolCallPart("search_aiida_docs", {"query": "cutoffs"})],
            [TextPart("60 Ry [howto/pw  §  Cutoffs].")],
        )

        assert_consulted_docs(trace_run(agent, "what ecutwfc?"))


class TestCitationIsCaught:
    def test_uncited_answer_fails(self) -> None:
        agent = _scripted_agent(
            [ToolCallPart("search_aiida_docs", {"query": "cutoffs"})],
            [TextPart("The recommended cutoff is 60 Ry.")],
        )

        with pytest.raises(AssertionError, match="cites no source"):
            assert_cited(trace_run(agent, "what ecutwfc?"))

    def test_cited_answer_passes(self) -> None:
        agent = _scripted_agent(
            [ToolCallPart("search_aiida_docs", {"query": "cutoffs"})],
            [TextPart("The recommended cutoff is 60 Ry [howto/pw  §  Cutoffs].")],
        )

        assert_cited(trace_run(agent, "what ecutwfc?"))


class TestQuantityDetection:
    """What counts as a physics claim, and what is just a number in prose."""

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("a cutoff of 60 Ry", {"60.0"}),
            ("spacing of 0.15 Å⁻¹", {"0.15"}),
            ("set ecutwfc to 45", {"45.0"}),
            ("conv_thr of 1e-8", {"1e-08"}),
            # Prose numbers are not physics claims; flagging them would bury
            # the real signal in noise.
            ("step 2 of 3", set()),
            ("found 12 structures", set()),
            ("PK 1234 finished", set()),
        ],
    )
    def test_only_units_and_named_parameters_count(
        self, text: str, expected: set[str]
    ) -> None:
        assert quantities_in(text) == expected

    def test_same_value_written_differently_is_still_grounded(self) -> None:
        """60 in the answer must match 60.0 in the tool output, or the check cries wolf."""
        agent = _scripted_agent(
            [ToolCallPart("search_aiida_docs", {"query": "cutoffs"})],
            [TextPart("Use 60 Ry and 480 Ry [howto/pw  §  Cutoffs].")],
        )

        # The excerpt says "60.0 Ry" / "480.0 Ry"; the answer says "60" / "480".
        assert ungrounded_quantities(trace_run(agent, "cutoffs?")) == set()


class TestDeveloperEnvReachesSettings:
    """The eval tier must run against the configured model, not a silent default.

    Regression from the first real-model run: ``tests/conftest.py`` chdirs every
    test into an empty temp directory so a developer's ``.env`` cannot leak into
    hermetic tests. The eval tier inherited that, so ``ModelSettings`` found no
    ``.env``, fell back to ``ollama``/``qwen3.5:2b``, and reported ten grounding
    failures for a model that had never been contacted.
    """

    @staticmethod
    def _project_with_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """A fake project holding a ``.env``, and an empty cwd like conftest's."""
        project = tmp_path / "project"
        project.mkdir()
        (project / ".env").write_text(
            "AIIDA_AGENTS_PROVIDER=anthropic\nAIIDA_AGENTS_MODEL=some-configured-model\n"
        )
        monkeypatch.setattr(_harness, "project_env_file", lambda: project / ".env")

        workdir = tmp_path / "work"
        workdir.mkdir()
        monkeypatch.chdir(workdir)
        return project

    def test_settings_read_the_copied_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Before the copy the defaults win; after it, the developer's config does."""
        from aiida_agents._settings import ModelSettings

        # Exported variables would mask the whole mechanism under test.
        monkeypatch.delenv("AIIDA_AGENTS_PROVIDER", raising=False)
        monkeypatch.delenv("AIIDA_AGENTS_MODEL", raising=False)
        self._project_with_env(tmp_path, monkeypatch)

        # The bug: an empty cwd means the bare defaults, silently.
        assert ModelSettings().provider == "ollama"
        assert ModelSettings().model == "qwen3.5:2b"

        copy_project_env(Path.cwd())

        assert ModelSettings().provider == "anthropic"
        assert ModelSettings().model == "some-configured-model"

    def test_copies_into_the_given_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._project_with_env(tmp_path, monkeypatch)

        written = copy_project_env(Path.cwd())

        assert written == Path.cwd() / ".env"
        assert written.is_file()

    def test_absent_project_env_is_not_an_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Config may come from exported shell variables, which survive the chdir."""
        monkeypatch.setattr(_harness, "project_env_file", lambda: tmp_path / "nope.env")

        assert copy_project_env(tmp_path) is None


class TestTraceCapture:
    def test_records_call_order_and_pairs_outputs_to_calls(self) -> None:
        agent = _scripted_agent(
            [ToolCallPart("search_aiida_docs", {"query": "first"})],
            [ToolCallPart("search_aiida_docs", {"query": "second"})],
            [TextPart("done")],
        )

        trace = trace_run(agent, "q")

        assert trace.tool_names == ["search_aiida_docs", "search_aiida_docs"]
        assert [c.args for c in trace.calls] == [
            {"query": "first"},
            {"query": "second"},
        ]
        assert all(c.output == _DOCS_EXCERPT for c in trace.calls)
        assert trace.answer == "done"


class TestEvidenceMustPresentTheNumberAsAQuantity:
    """The oracle's own failure mode, reported as issue #81.

    Grounding used to accept a claim whose bare number appeared *anywhere* in
    the evidence. Real tool output is dense with incidental integers -- pks,
    exit codes, counts, ports -- so a fabricated cutoff was certified the
    moment some unrelated node happened to have that pk. Since this same path
    is both the shipped CLI warning and the eval oracle, it was certifying the
    exact class of invention it exists to catch.
    """

    def test_a_pk_does_not_ground_a_cutoff(self) -> None:
        """Julian's case, verbatim: 60 is a pk and 8 is a count."""
        evidence = "{'pk': 60, 'exit_status': 501, 'count': 8}"
        answer = (
            "ecutwfc varies widely (e.g., 60 Ry for gold, 8 for the alkali metals)."
        )

        assert "60.0" in grounding.ungrounded_quantities(answer, evidence)

    def test_a_number_the_evidence_labels_is_grounded(self) -> None:
        """The other direction, or the fix would just be "flag everything"."""
        evidence = "{'ecutwfc': 60.0, 'ecutwfc_unit': 'Ry'}"

        assert (
            grounding.ungrounded_quantities("The cutoff was 60 Ry.", evidence) == set()
        )

    def test_a_unit_in_the_evidence_grounds_the_claim(self) -> None:
        evidence = "recommended cutoff is 60 Ry for this family"

        assert grounding.ungrounded_quantities("Use 60 Ry.", evidence) == set()

    def test_a_bare_count_still_grounds_a_bare_number(self) -> None:
        """The over-fire direction.

        A bare number sharing a sentence with a parameter name is a guess about
        what the sentence means, so it stays on the loose bar. Holding it to
        the strict one would flag "I checked 3 relaxations and the cutoff was
        fine", and a warning that fires on correct answers gets scrolled past.
        """
        evidence = "{'count': 3, 'ecutwfc': 60.0, 'unit': 'Ry'}"
        answer = "I checked 3 relaxations and the cutoff of 60 Ry was fine."

        assert grounding.ungrounded_quantities(answer, evidence) == set()

    def test_a_realistic_tool_dump_does_not_ground_everything(self) -> None:
        """The large-evidence case CI never exercised.

        Every small integer appears somewhere in a real dump, which is what
        made the old bare-number rule vacuous in practice rather than only in
        principle. The dump below grounds none of the invented cutoffs.
        """
        evidence = repr(
            [
                {
                    "pk": pk,
                    "uuid": f"0000-{pk}",
                    "exit_status": status,
                    "process_label": "PwCalculation",
                    "num_machines": 1,
                    "seconds": pk * 3,
                }
                for pk, status in zip(range(30, 90), [0, 501, 400] * 20)
            ]
        )
        answer = "Typical values are 40 Ry for the wavefunction and 80 Ry for the charge density."

        assert grounding.ungrounded_quantities(answer, evidence) == {"40.0", "80.0"}

    def test_units_are_matched_on_word_boundaries(self) -> None:
        """`K`, `A` and `eV` sit inside ordinary words.

        Unbounded, the unit pattern cut letters out of "Kelvin", "Analysis" and
        "every" before the text was scanned for numbers.
        """
        assert grounding.quantities_in("However, several Analysis steps ran.") == set()
