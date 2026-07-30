"""Tests for ``diagnose_process_failure``.

Everything here runs against real failed processes, in-process, because the
three things this tool reads are all produced by the engine and none of them
can be faked convincingly: the ``called`` tree, the exit codes a process class
declares, and the ``considered_handlers`` extra a restart work chain writes as
it goes. A mock of any of those would pin the mock's shape rather than AiiDA's.

Two failures are built:

``failed_multiply_add`` (from ``tests/conftest.py``)
    ``MultiplyAddWorkChain`` with a negative ``z``, so the nested
    ``ArithmeticAddCalculation`` exits 410 and the work chain exits 400. A real
    two-level failure chain, with no restart machinery involved. Shared with the
    eval tier, which needs the same failure to ask a model about.

``exhausted_workchain``
    A ``BaseRestartWorkChain`` whose handler recognises exit 410 and restarts,
    but does not actually fix it, so the loop runs out of iterations and the
    work chain gives up. This is the case the tool exists for: the remedy has
    already been tried, twice, and recommending it again would send the user
    round the same loop.
"""

from __future__ import annotations

import typing as t

import pytest
from aiida import orm
from aiida.calculations.arithmetic.add import ArithmeticAddCalculation
from aiida.engine import (
    BaseRestartWorkChain,
    ProcessHandlerReport,
    run_get_node,
    while_,
)
from aiida.engine.processes.workchains.utils import process_handler

from aiida_agents.tools._orm import WrongNodeType
from aiida_agents.tools.analysis.diagnostics import (
    _handler_exit_statuses,
    diagnose_process_failure,
)


class RestartDemo(BaseRestartWorkChain):
    """A restart work chain that recognises one failure but cannot fix it.

    The handler fires every time and restarts unchanged, which is what makes
    the run exhaust its iterations -- and what puts a repeated, ineffective
    remedy into the record for the diagnosis to find.
    """

    _process_class = ArithmeticAddCalculation

    @classmethod
    def define(cls, spec: t.Any) -> None:
        super().define(spec)
        spec.expose_inputs(ArithmeticAddCalculation, namespace="add")
        spec.expose_outputs(ArithmeticAddCalculation)
        spec.outline(
            cls.setup,
            # mypy checks these against WorkChain, but they are inherited from
            # BaseRestartWorkChain, which it does not accept as the same thing.
            while_(cls.should_run_process)(  # type: ignore[arg-type]
                cls.run_process,  # type: ignore[arg-type]
                cls.inspect_process,  # type: ignore[arg-type]
            ),
            cls.results,
        )

    def setup(self) -> None:
        super().setup()
        self.ctx.inputs = dict(self.exposed_inputs(ArithmeticAddCalculation, "add"))

    @process_handler(
        priority=500,
        exit_codes=ArithmeticAddCalculation.exit_codes.ERROR_NEGATIVE_NUMBER,
    )
    def handle_negative_sum(self, node: orm.CalcJobNode) -> ProcessHandlerReport:
        """Restart the calculation, leaving the operands as they were.

        Second paragraph, which the diagnosis should not quote.
        """
        self.ctx.inputs["y"] = orm.Int(node.inputs.y.value)
        return ProcessHandlerReport(True)

    @process_handler(
        priority=100,
        exit_codes=ArithmeticAddCalculation.exit_codes.ERROR_READING_OUTPUT_FILE,
    )
    def handle_unreadable_output(self, node: orm.CalcJobNode) -> None:
        """Unrelated remedy, for an exit code this run never produces."""


class RecoveringRestartDemo(RestartDemo):
    """The same work chain, with a handler that actually fixes the problem."""

    @process_handler(
        priority=500,
        exit_codes=ArithmeticAddCalculation.exit_codes.ERROR_NEGATIVE_NUMBER,
    )
    def handle_negative_sum(self, node: orm.CalcJobNode) -> ProcessHandlerReport:
        """Flip the sign of the operand so the sum comes out positive."""
        self.ctx.inputs["y"] = orm.Int(abs(node.inputs.y.value))
        return ProcessHandlerReport(True)


@pytest.fixture(scope="module")
def recovered_workchain(arithmetic_add_code: orm.InstalledCode) -> orm.WorkChainNode:
    """A restart work chain that hit exit 410, fixed it, and finished cleanly."""
    _, node = run_get_node(
        RecoveringRestartDemo,
        add={"x": orm.Int(2), "y": orm.Int(-100), "code": arithmetic_add_code},
    )
    assert isinstance(node, orm.WorkChainNode)
    assert node.exit_status == 0, "the fixture is only useful if the run recovered"
    return node


@pytest.fixture(scope="module")
def exhausted_workchain(arithmetic_add_code: orm.InstalledCode) -> orm.WorkChainNode:
    """A restart work chain that applied its remedy twice and still failed."""
    _, node = run_get_node(
        RestartDemo,
        add={"x": orm.Int(2), "y": orm.Int(-100), "code": arithmetic_add_code},
        max_iterations=orm.Int(2),
    )
    assert isinstance(node, orm.WorkChainNode)
    assert node.exit_status, "the fixture is only useful if the run actually failed"
    return node


class TestFailureChain:
    """Finding the process that actually broke."""

    def test_the_chain_reaches_the_calculation_below_the_workchain(
        self, failed_multiply_add: orm.WorkChainNode
    ) -> None:
        diagnosis = diagnose_process_failure(str(failed_multiply_add.pk))

        assert diagnosis["failed"] is True
        labels = [step["process_label"] for step in diagnosis["failure_chain"]]
        assert labels == ["MultiplyAddWorkChain", "ArithmeticAddCalculation"]

    def test_the_root_cause_is_the_calculation_not_the_workchain(
        self, failed_multiply_add: orm.WorkChainNode
    ) -> None:
        """The work chain's own 400 only says a sub-process failed."""
        diagnosis = diagnose_process_failure(str(failed_multiply_add.pk))
        root = diagnosis["root_cause"]

        assert root is not None
        assert root["process_label"] == "ArithmeticAddCalculation"
        assert root["exit_status"] == 410
        assert diagnosis["exit_status"] == 400

    def test_the_exit_codes_meaning_comes_from_the_process_class(
        self, failed_multiply_add: orm.WorkChainNode
    ) -> None:
        diagnosis = diagnose_process_failure(str(failed_multiply_add.pk))
        root = diagnosis["root_cause"]

        assert root is not None
        assert (
            root["exit_code_meaning"] == "The sum of the operands is a negative number."
        )

    def test_the_failed_calcjob_is_offered_for_its_output_files(
        self, failed_multiply_add: orm.WorkChainNode
    ) -> None:
        """The pk to hand to list_retrieved_files, where the code's own error is."""
        diagnosis = diagnose_process_failure(str(failed_multiply_add.pk))
        root = diagnosis["root_cause"]

        assert root is not None
        assert diagnosis["retrieved_files_pk"] == root["pk"]

    def test_asking_about_the_calculation_directly_also_works(
        self, failed_multiply_add: orm.WorkChainNode
    ) -> None:
        calcjob = next(
            child
            for child in failed_multiply_add.called
            if isinstance(child, orm.CalcJobNode)
        )

        diagnosis = diagnose_process_failure(str(calcjob.pk))

        assert diagnosis["failed"] is True
        assert diagnosis["root_cause"] is not None
        assert diagnosis["root_cause"]["pk"] == calcjob.pk


class TestHandlingAttempted:
    """What the restart machinery already tried."""

    def test_the_same_remedy_is_shown_applied_on_every_iteration(
        self, exhausted_workchain: orm.WorkChainNode
    ) -> None:
        """The point of the tool: this remedy has been tried and did not work."""
        diagnosis = diagnose_process_failure(str(exhausted_workchain.pk))

        applied = [a for a in diagnosis["handling_attempted"] if a["applied"]]
        assert [a["handler"] for a in applied] == [
            "handle_negative_sum",
            "handle_negative_sum",
        ]
        assert [a["iteration"] for a in applied] == [1, 2]
        assert applied[0]["do_break"] is True

    def test_the_attempt_carries_the_handlers_own_description(
        self, exhausted_workchain: orm.WorkChainNode
    ) -> None:
        """Quoted from the plugin's docstring, first paragraph only."""
        diagnosis = diagnose_process_failure(str(exhausted_workchain.pk))

        applied = next(a for a in diagnosis["handling_attempted"] if a["applied"])
        assert applied["description"] == (
            "Restart the calculation, leaving the operands as they were."
        )

    def test_a_workchain_with_no_restart_machinery_reports_no_attempts(
        self, failed_multiply_add: orm.WorkChainNode
    ) -> None:
        diagnosis = diagnose_process_failure(str(failed_multiply_add.pk))

        assert diagnosis["handling_attempted"] == []


class TestKnownRemedies:
    """What the work chain could still do about this exit code."""

    def test_remedies_are_narrowed_to_handlers_covering_the_failure(
        self, exhausted_workchain: orm.WorkChainNode
    ) -> None:
        """The handler for a different exit code is not offered."""
        diagnosis = diagnose_process_failure(str(exhausted_workchain.pk))

        names = [r["name"] for r in diagnosis["known_remedies"]]
        assert "handle_negative_sum" in names
        assert "handle_unreadable_output" not in names

    def test_a_remedy_carries_its_priority_and_description(
        self, exhausted_workchain: orm.WorkChainNode
    ) -> None:
        diagnosis = diagnose_process_failure(str(exhausted_workchain.pk))

        remedy = next(
            r for r in diagnosis["known_remedies"] if r["name"] == "handle_negative_sum"
        )
        assert remedy["priority"] == 500
        assert remedy["enabled"] is True
        assert remedy["handles_exit_statuses"] == [410]
        assert remedy["description"] == (
            "Restart the calculation, leaving the operands as they were."
        )

    def test_a_plain_workchain_offers_no_remedies(
        self, failed_multiply_add: orm.WorkChainNode
    ) -> None:
        """MultiplyAddWorkChain registers no handlers; nothing is invented."""
        diagnosis = diagnose_process_failure(str(failed_multiply_add.pk))

        assert diagnosis["known_remedies"] == []


class TestExitCodeIntrospection:
    """The one unofficial read, and its fallback."""

    def test_declared_exit_codes_are_recovered_from_the_handler(self) -> None:
        handler = next(
            h
            for h in RestartDemo.get_process_handlers()
            if h.__name__ == "handle_negative_sum"
        )

        assert _handler_exit_statuses(handler) == [410]

    def test_an_unreadable_declaration_degrades_to_unknown(self) -> None:
        """A shape change upstream must cost filtering, not raise."""

        def not_a_handler() -> None:
            pass

        assert _handler_exit_statuses(not_a_handler) is None


class TestNonFailures:
    """A process that did not fail, and things that are not processes."""

    def test_a_successful_process_is_reported_as_not_failed(
        self, add_calc: orm.CalcJobNode
    ) -> None:
        diagnosis = diagnose_process_failure(str(add_calc.pk))

        assert diagnosis["failed"] is False
        assert diagnosis["root_cause"] is None
        assert diagnosis["failure_chain"] == []
        assert diagnosis["known_remedies"] == []

    def test_a_healthy_process_yields_no_diagnosis_fields(
        self, add_calc: orm.CalcJobNode
    ) -> None:
        """Nothing is populated speculatively for a process that never faltered."""
        diagnosis = diagnose_process_failure(str(add_calc.pk))

        assert diagnosis["handling_attempted"] == []
        assert diagnosis["retrieved_files_pk"] is None

    def test_a_recovered_run_still_shows_what_it_had_to_do(
        self, recovered_workchain: orm.WorkChainNode
    ) -> None:
        """Succeeding after two restarts is not the same as succeeding.

        Reported even though ``failed`` is false: the trail is a fact about the
        run, and suppressing it hides trouble behind a green result.
        """
        diagnosis = diagnose_process_failure(str(recovered_workchain.pk))

        assert diagnosis["failed"] is False
        assert diagnosis["root_cause"] is None
        applied = [a for a in diagnosis["handling_attempted"] if a["applied"]]
        assert [a["handler"] for a in applied] == ["handle_negative_sum"]

    def test_a_data_node_is_rejected(
        self, silicon_structure: orm.StructureData
    ) -> None:
        with pytest.raises(WrongNodeType, match="not a process node"):
            diagnose_process_failure(str(silicon_structure.pk))
