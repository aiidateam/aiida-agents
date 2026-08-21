"""Tests for ``build_resubmission_spec`` and ``submit_process_batch``.

Rebuilding is tested against real finished processes, because the whole point
of the tool is that it reads what a run *actually* used rather than what
someone remembers it using --- a fixture of inputs would test the merge logic
while removing the part worth doubting.

Submission itself is not re-tested here: ``submit_process_batch`` delegates
to ``submit_process_spec``, which has its own coverage. What is tested is the
batch's own contract --- the cap, the rejection of an empty list, and that the
count it reports matches what it was given.
"""

from __future__ import annotations

import typing as t

import pytest
from aiida import orm

from aiida_agents.tools._orm import WrongNodeType
from aiida_agents.tools.execution import resubmission
from aiida_agents.tools.execution.resubmission import (
    MAX_BATCH,
    build_resubmission_spec,
    submit_process_batch,
)


class TestRebuildingFromTheNode:
    """What the original ran, not what anyone remembers it running."""

    def test_the_entry_point_comes_from_the_process_itself(
        self, multiply_add_workchain: orm.WorkChainNode
    ) -> None:
        spec = build_resubmission_spec(str(multiply_add_workchain.pk))

        assert spec["entry_point"] == "core.arithmetic.multiply_add"

    def test_every_original_input_is_carried_over(
        self, multiply_add_workchain: orm.WorkChainNode
    ) -> None:
        """Including the ones nobody would think to mention."""
        spec = build_resubmission_spec(str(multiply_add_workchain.pk))

        inputs = spec["inputs"]
        assert inputs["x"] == 2
        assert inputs["y"] == 3
        assert inputs["z"] == 4
        assert "code" in inputs

    def test_a_node_valued_input_comes_back_as_a_reference(
        self, multiply_add_workchain: orm.WorkChainNode
    ) -> None:
        """So the rebuilt spec is the same plain document submission accepts."""
        spec = build_resubmission_spec(str(multiply_add_workchain.pk))

        assert set(spec["inputs"]["code"]) == {"pk"}

    def test_the_spec_records_what_it_was_rebuilt_from(
        self, multiply_add_workchain: orm.WorkChainNode
    ) -> None:
        spec = build_resubmission_spec(str(multiply_add_workchain.pk))

        assert spec["metadata"]["resubmitted_from"] == multiply_add_workchain.pk

    def test_a_failed_run_can_be_rebuilt_too(
        self, failed_multiply_add: orm.WorkChainNode
    ) -> None:
        """The common case: re-run the thing that broke, with a change."""
        spec = build_resubmission_spec(str(failed_multiply_add.pk))

        assert spec["inputs"]["z"] == -100

    def test_the_rebuilt_spec_actually_submits(
        self, multiply_add_workchain: orm.WorkChainNode
    ) -> None:
        """The claim that matters: it round-trips through the resolver."""
        from aiida_agents.tools.execution.submit import _resolve_inputs

        spec = build_resubmission_spec(str(multiply_add_workchain.pk))

        resolved = _resolve_inputs(spec["entry_point"], spec["inputs"])
        assert set(resolved) >= {"x", "y", "z", "code"}


class TestOverrides:
    """Changing one thing without discarding the rest."""

    def test_an_override_replaces_only_what_it_names(
        self, multiply_add_workchain: orm.WorkChainNode
    ) -> None:
        spec = build_resubmission_spec(str(multiply_add_workchain.pk), {"z": 99})

        assert spec["inputs"]["z"] == 99
        assert spec["inputs"]["x"] == 2  # untouched

    def test_a_nested_override_merges_rather_than_replaces(self) -> None:
        """Raising one cutoff must not discard the other parameters.

        Tested on the merge directly: no core process nests parameters, and the
        rule is what would silently lose a whole ``SYSTEM`` card on a real
        PwRelaxWorkChain resubmission.
        """
        original = {"parameters": {"SYSTEM": {"ecutwfc": 30.0, "ecutrho": 240.0}}}
        overrides = {"parameters": {"SYSTEM": {"ecutwfc": 80.0}}}

        merged = resubmission._deep_merge(original, overrides)

        assert merged["parameters"]["SYSTEM"] == {"ecutwfc": 80.0, "ecutrho": 240.0}

    def test_a_flat_link_label_is_rebuilt_into_its_namespace(self) -> None:
        """AiiDA stores base.pw.parameters as 'base__pw__parameters'."""
        nested = resubmission._nest({"base__pw__parameters": {"x": 1}, "structure": 2})

        assert nested == {"base": {"pw": {"parameters": {"x": 1}}}, "structure": 2}


class TestRejectedRebuilds:
    def test_a_data_node_is_rejected(
        self, silicon_structure: orm.StructureData
    ) -> None:
        with pytest.raises(WrongNodeType, match="not a process node"):
            build_resubmission_spec(str(silicon_structure.pk))

    def test_a_process_with_no_entry_point_is_rejected(
        self, multiply_add_workchain: orm.WorkChainNode, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A calcfunction or a script-run process cannot be rebuilt this way."""
        monkeypatch.setattr(
            type(multiply_add_workchain), "process_type", "", raising=False
        )

        with pytest.raises(ValueError, match="records no entry point"):
            build_resubmission_spec(str(multiply_add_workchain.pk))


class TestBatchContract:
    """The batch's own rules, independent of what submitting does."""

    def test_an_empty_batch_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty list"):
            submit_process_batch([])

    @pytest.mark.parametrize("bad", [None, "not a list", {}])
    def test_a_non_list_is_rejected(self, bad: t.Any) -> None:
        with pytest.raises(ValueError, match="non-empty list"):
            submit_process_batch(bad)

    def test_a_batch_over_the_cap_is_refused_with_the_reason(self) -> None:
        """An approval prompt nobody reads is not an approval."""
        specs = [{"entry_point": "core.arithmetic.add", "inputs": {}}] * (MAX_BATCH + 1)

        with pytest.raises(ValueError, match=f"capped at {MAX_BATCH}"):
            submit_process_batch(specs)  # type: ignore[arg-type]

    def test_the_cap_itself_is_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Off-by-one in the other direction: exactly MAX_BATCH is fine."""
        _stub_submission(monkeypatch, pk=1)

        result = submit_process_batch([{"entry_point": "x"}] * MAX_BATCH)

        assert result["requested"] == MAX_BATCH
        assert result["submitted"] == MAX_BATCH

    def test_it_reports_how_many_it_was_asked_for(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """So a trimmed batch cannot read as one that ran whole."""
        _stub_submission(monkeypatch, pk=7)

        result = submit_process_batch([{"entry_point": "x"}] * 3)

        assert result["requested"] == 3
        assert [r["pk"] for r in result["results"]] == [7, 7, 7]


def _stub_submission(
    monkeypatch: pytest.MonkeyPatch,
    pk: int = 1,
    fail_prepare_at: int | None = None,
    fail_run_at: int | None = None,
    trace: list[str] | None = None,
) -> None:
    """Stub the three seams a batch passes through, optionally failing at one.

    Deliberately stubs `_validate_spec`/`_prepare_submission`/`_run_submission`
    rather than `submit_process_spec`. The old tests stubbed the latter,
    which is exactly why the partial-submit bug survived them: with validation
    and submission behind one stub, no test could tell whether the batch
    validated everything before submitting anything.

    `trace`, when given, records the order the seams were reached, so a test
    can assert on the interleaving rather than only the outcome.
    """
    prepared = {"n": 0}
    ran = {"n": 0}

    def _validate(spec: object) -> tuple[str, dict[str, object]]:
        return "x", {"a": 1}

    def _prepare(entry_point: str, inputs: object) -> tuple[object, object]:
        prepared["n"] += 1
        if trace is not None:
            trace.append(f"prepare {prepared['n']}")
        if fail_prepare_at == prepared["n"]:
            msg = "input port 'x' is required"
            raise ValueError(msg)
        return object(), {}

    def _run(
        entry_point: str, process_class: object, resolved: object
    ) -> dict[str, int]:
        ran["n"] += 1
        if trace is not None:
            trace.append(f"run {ran['n']}")
        if fail_run_at == ran["n"]:
            msg = "the daemon went away"
            raise RuntimeError(msg)
        return {"pk": pk}

    monkeypatch.setattr(
        "aiida_agents.tools.execution.spec_execution._validate_spec", _validate
    )
    monkeypatch.setattr(
        "aiida_agents.tools.execution.submit._prepare_submission", _prepare
    )
    monkeypatch.setattr("aiida_agents.tools.execution.submit._run_submission", _run)


class TestTheBatchIsAllOrNothing:
    """The guarantee the docstring makes, which the code did not keep.

    Issue #76. The implementation was
    `[submit_process_spec(spec) for spec in specs]`, and that call validates
    *and submits* each spec in turn. A bad spec at position N therefore left
    1..N-1 already on the daemon and then raised, returning no BatchResult at
    all -- so the user who approved twenty got fourteen running and a traceback
    naming none of them.
    """

    def test_every_spec_is_validated_before_any_is_submitted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The interleaving is the whole property, so assert on it directly."""
        trace: list[str] = []
        _stub_submission(monkeypatch, trace=trace)

        submit_process_batch([{"entry_point": "x"}] * 3)

        assert trace == [
            "prepare 1",
            "prepare 2",
            "prepare 3",
            "run 1",
            "run 2",
            "run 3",
        ]

    def test_a_bad_spec_submits_nothing_at_all(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The regression. With the old code the first two were already gone."""
        trace: list[str] = []
        _stub_submission(monkeypatch, fail_prepare_at=3, trace=trace)

        with pytest.raises(ValueError):
            submit_process_batch([{"entry_point": "x"}] * 5)

        assert not [step for step in trace if step.startswith("run")]

    def test_the_error_names_which_spec_and_says_nothing_ran(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The specs came from a query, so "the third one" is how it is found."""
        _stub_submission(monkeypatch, fail_prepare_at=3)

        with pytest.raises(ValueError, match="Spec 3 of 5") as exc_info:
            submit_process_batch([{"entry_point": "x"}] * 5)

        assert "none of the batch was submitted" in str(exc_info.value)

    def test_a_failure_while_submitting_reports_what_already_went_out(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Validation can be atomic; submission cannot.

        There is no transaction across the daemon, so a batch really can split
        here. The one thing that must not happen then is losing the pks of what
        did go out -- a user who cannot tell what is running is the worst
        outcome, worse than the split itself.
        """
        _stub_submission(monkeypatch, pk=42, fail_run_at=3)

        with pytest.raises(RuntimeError, match="Submitted 2 of 4") as exc_info:
            submit_process_batch([{"entry_point": "x"}] * 4)

        message = str(exc_info.value)
        assert "42, 42" in message, "the pks already running must be recoverable"
        assert "Do not resubmit" in message
