"""Tests for the Quantum ESPRESSO convergence tool the dev stub contributes.

Skipped unless ``dev/qe_rag_stub`` is installed (``uv sync --group qe``), which
CI does not do --- so these do not run there. That is the cost of keeping
Quantum ESPRESSO's output format out of the core tool layer: the code lives in
a plugin, and the plugin is optional. Run them locally before touching the
parser.

The parser is tested against pw.x output rather than through a calculation,
because the shapes that matter --- a cycle that converged, one that oscillated,
one that stalled while still descending --- cannot be produced on demand by
running anything. The node handling is tested separately against a real
CalcJob, so the two halves are covered without either standing in for the
other.
"""

from __future__ import annotations

import pytest
from aiida import orm

pytest.importorskip(
    "qe_rag_stub",
    reason="dev/qe_rag_stub is not installed; run `uv sync --group qe`",
)

from qe_rag_stub.convergence import (  # noqa: E402 - after the skip guard
    parse_scf_trace,
    read_scf_convergence,
)

# A converged cycle, trimmed to the lines the parser reads.
CONVERGED = """
     Self-consistent Calculation

     iteration #  1     ecut=    30.00 Ry     beta= 0.70
     Davidson diagonalization with overlap
     total energy              =     -22.65 Ry
     estimated scf accuracy    <       0.16 Ry

     iteration #  2     ecut=    30.00 Ry     beta= 0.70
     estimated scf accuracy    <       0.01 Ry

     iteration #  3     ecut=    30.00 Ry     beta= 0.70
     estimated scf accuracy    <       4.9E-05 Ry

     iteration #  4     ecut=    30.00 Ry     beta= 0.70
     estimated scf accuracy    <       1.2D-09 Ry

     End of self-consistent calculation

     convergence has been achieved in   4 iterations
"""

# Ran out of iterations while still descending: the case where more steps is
# the proportionate fix.
DESCENDING_BUT_UNFINISHED = """
     iteration #  1     ecut=    30.00 Ry     beta= 0.70
     estimated scf accuracy    <       0.90 Ry
     iteration #  2     ecut=    30.00 Ry     beta= 0.70
     estimated scf accuracy    <       0.30 Ry
     iteration #  3     ecut=    30.00 Ry     beta= 0.70
     estimated scf accuracy    <       0.09 Ry
     iteration #  4     ecut=    30.00 Ry     beta= 0.70
     estimated scf accuracy    <       0.02 Ry

     convergence NOT achieved after   4 iterations: stopping
"""

# Never settled: more iterations would not have helped.
OSCILLATING = """
     iteration #  1     ecut=    30.00 Ry     beta= 0.70
     estimated scf accuracy    <       0.50 Ry
     iteration #  2     ecut=    30.00 Ry     beta= 0.70
     estimated scf accuracy    <       0.05 Ry
     iteration #  3     ecut=    30.00 Ry     beta= 0.70
     estimated scf accuracy    <       0.40 Ry
     iteration #  4     ecut=    30.00 Ry     beta= 0.70
     estimated scf accuracy    <       0.06 Ry
     iteration #  5     ecut=    30.00 Ry     beta= 0.70
     estimated scf accuracy    <       0.38 Ry

     convergence NOT achieved after   5 iterations: stopping
"""

# Descending, but the last iterations barely move.
STALLED = """
     iteration #  1     ecut=    30.00 Ry     beta= 0.70
     estimated scf accuracy    <       0.90 Ry
     iteration #  2     ecut=    30.00 Ry     beta= 0.70
     estimated scf accuracy    <       0.11 Ry
     iteration #  3     ecut=    30.00 Ry     beta= 0.70
     estimated scf accuracy    <       0.10 Ry
     iteration #  4     ecut=    30.00 Ry     beta= 0.70
     estimated scf accuracy    <       0.09 Ry

     convergence NOT achieved after   4 iterations: stopping
"""

# Killed mid-cycle: no verdict line at all.
INTERRUPTED = """
     iteration #  1     ecut=    30.00 Ry     beta= 0.70
     estimated scf accuracy    <       0.70 Ry
     iteration #  2     ecut=    30.00 Ry     beta= 0.70
     estimated scf accuracy    <       0.20 Ry
"""


class TestVerdict:
    """Whether the cycle reached self-consistency."""

    def test_a_converged_cycle_is_recognised(self) -> None:
        parsed = parse_scf_trace(CONVERGED)

        assert parsed["converged"] is True
        assert parsed["iterations_run"] == 4

    def test_an_unconverged_cycle_is_recognised(self) -> None:
        parsed = parse_scf_trace(OSCILLATING)

        assert parsed["converged"] is False
        assert parsed["iterations_run"] == 5

    def test_an_interrupted_cycle_reports_no_verdict(self) -> None:
        """A killed job neither succeeded nor gave up; saying either is wrong."""
        parsed = parse_scf_trace(INTERRUPTED)

        assert parsed["converged"] is None
        assert parsed["iterations_run"] == 2

    def test_the_reported_count_wins_over_counting_lines(self) -> None:
        """pw.x states the count; the 'iteration #' lines undercount a restart."""
        text = CONVERGED.replace("in   4 iterations", "in   9 iterations")

        assert parse_scf_trace(text)["iterations_run"] == 9


class TestAccuracySeries:
    """The numbers themselves."""

    def test_every_accuracy_is_captured_in_order(self) -> None:
        parsed = parse_scf_trace(CONVERGED)

        assert parsed["scf_accuracy"] == [0.16, 0.01, 4.9e-05, 1.2e-09]

    def test_a_fortran_d_exponent_is_read_as_a_number(self) -> None:
        """pw.x prints 1.2D-09 in some builds; a string here would be useless."""
        parsed = parse_scf_trace(CONVERGED)

        assert parsed["final_accuracy_ry"] == pytest.approx(1.2e-09)

    def test_an_output_with_no_cycle_yields_an_empty_series(self) -> None:
        parsed = parse_scf_trace("some other program's output entirely")

        assert parsed["scf_accuracy"] == []
        assert parsed["final_accuracy_ry"] is None
        assert parsed["converged"] is None


class TestShape:
    """The distinction the tool exists to draw."""

    def test_a_monotonic_descent_is_decreasing_and_not_stalled(self) -> None:
        """Ran out of iterations while converging: more steps is the fix."""
        parsed = parse_scf_trace(DESCENDING_BUT_UNFINISHED)

        assert parsed["trend"] == "decreasing"
        assert parsed["stalled"] is False

    def test_a_cycle_that_never_settles_is_oscillating(self) -> None:
        """More iterations would not have helped; this is the mixing case."""
        parsed = parse_scf_trace(OSCILLATING)

        assert parsed["trend"] == "oscillating"

    def test_a_descent_that_stops_moving_is_stalled(self) -> None:
        """Decreasing and stalled at once -- the two are independent facts."""
        parsed = parse_scf_trace(STALLED)

        assert parsed["trend"] == "decreasing"
        assert parsed["stalled"] is True

    def test_a_single_iteration_yields_no_trend(self) -> None:
        parsed = parse_scf_trace("estimated scf accuracy    <       0.5 Ry")

        assert parsed["trend"] == "too_few_iterations"
        assert parsed["stalled"] is None


class TestAgainstARealNode:
    """The half the text fixtures cannot cover: getting to the file at all."""

    def test_it_reads_the_calculations_retrieved_output(
        self, add_calc: orm.CalcJobNode
    ) -> None:
        """A real CalcJob whose retrieved folder holds an ``aiida.out``.

        Not a Quantum ESPRESSO run --- there is none to be had in this test
        environment --- so the trace comes back empty. That is the point: the
        node handling, the option lookup and the file read are exercised
        independently of whether the contents happen to be a pw.x cycle.
        """
        result = read_scf_convergence(str(add_calc.pk))

        assert result["pk"] == add_calc.pk
        assert result["filename"] == "aiida.out"
        assert result["scf_accuracy"] == []
        assert result["converged"] is None

    def test_a_workchain_is_rejected_with_the_reason(
        self, multiply_add_workchain: orm.WorkChainNode
    ) -> None:
        """Only a CalcJob retrieves files; the error says where to get the pk."""
        with pytest.raises(ValueError, match="not a calculation"):
            read_scf_convergence(str(multiply_add_workchain.pk))

    def test_a_missing_output_file_names_what_was_retrieved(
        self, add_calc: orm.CalcJobNode, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The recovery is to read a different file, so list the candidates."""
        monkeypatch.setattr(
            orm.CalcJobNode, "get_option", lambda self, name: "not-there.out"
        )

        with pytest.raises(ValueError, match="retrieved no 'not-there.out'"):
            read_scf_convergence(str(add_calc.pk))


class TestPluginContribution:
    """The stub's own wiring, which nothing else exercises."""

    def test_the_stub_contributes_the_tool_as_a_read(self) -> None:
        from aiida_agents.plugins import discover_plugins

        plugin = next(p for p in discover_plugins() if p.name == "quantumespresso")

        contributed = {tool.name: tool.writes for tool in plugin.tools}
        assert contributed == {"read_scf_convergence": False}

    def test_the_stub_contributes_guidance_on_using_it(self) -> None:
        """Without the fragment, nothing tells the agent when the tool applies."""
        from aiida_agents.plugins import discover_plugins

        plugin = next(p for p in discover_plugins() if p.name == "quantumespresso")

        assert plugin.prompt_fragment is not None
        assert "read_scf_convergence" in plugin.prompt_fragment
