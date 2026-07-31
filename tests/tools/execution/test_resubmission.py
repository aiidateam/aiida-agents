"""Tests for ``build_resubmission_spec`` and ``execute_workflow_batch``.

Rebuilding is tested against real finished processes, because the whole point
of the tool is that it reads what a run *actually* used rather than what
someone remembers it using --- a fixture of inputs would test the merge logic
while removing the part worth doubting.

Submission itself is not re-tested here: ``execute_workflow_batch`` delegates
to ``execute_workflow_spec``, which has its own coverage. What is tested is the
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
    execute_workflow_batch,
)


class TestRebuildingFromTheNode:
    """What the original ran, not what anyone remembers it running."""

    def test_the_entry_point_comes_from_the_process_itself(
        self, multiply_add_workchain: orm.WorkChainNode
    ) -> None:
        spec = build_resubmission_spec(str(multiply_add_workchain.pk))

        assert spec["workflow_type"] == "core.arithmetic.multiply_add"

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

        resolved = _resolve_inputs(spec["workflow_type"], spec["inputs"])
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
            execute_workflow_batch([])

    @pytest.mark.parametrize("bad", [None, "not a list", {}])
    def test_a_non_list_is_rejected(self, bad: t.Any) -> None:
        with pytest.raises(ValueError, match="non-empty list"):
            execute_workflow_batch(bad)

    def test_a_batch_over_the_cap_is_refused_with_the_reason(self) -> None:
        """An approval prompt nobody reads is not an approval."""
        specs = [{"workflow_type": "core.arithmetic.add", "inputs": {}}] * (
            MAX_BATCH + 1
        )

        with pytest.raises(ValueError, match=f"capped at {MAX_BATCH}"):
            execute_workflow_batch(specs)  # type: ignore[arg-type]

    def test_the_cap_itself_is_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Off-by-one in the other direction: exactly MAX_BATCH is fine."""
        monkeypatch.setattr(
            "aiida_agents.tools.execution.spec_execution.execute_workflow_spec",
            lambda spec: {"pk": 1},
        )

        result = execute_workflow_batch([{"workflow_type": "x"}] * MAX_BATCH)

        assert result["requested"] == MAX_BATCH
        assert result["submitted"] == MAX_BATCH

    def test_it_reports_how_many_it_was_asked_for(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """So a trimmed batch cannot read as one that ran whole."""
        monkeypatch.setattr(
            "aiida_agents.tools.execution.spec_execution.execute_workflow_spec",
            lambda spec: {"pk": 7},
        )

        result = execute_workflow_batch([{"workflow_type": "x"}] * 3)

        assert result["requested"] == 3
        assert [r["pk"] for r in result["results"]] == [7, 7, 7]
