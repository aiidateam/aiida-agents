"""Tests for ``wait_for_process``.

The already-terminated paths run against real finished processes from the
session fixtures. The *waiting* itself cannot be produced that way --- every
process in this suite runs in-process and is terminal the moment it exists ---
so the running case drives a stub node whose ``is_terminated`` flips after a
set number of polls. That stub stands in for the daemon, which is the only
thing that could otherwise change the state mid-wait.

Sleeping is patched out throughout. The behaviour worth pinning is when the
loop gives up and what it reports, not how long a test takes to find out.
"""

from __future__ import annotations

import typing as t

import pytest
from aiida import orm

from aiida_agents.tools import execution as execution_tools
from aiida_agents.tools._orm import WrongNodeType
from aiida_agents.tools.execution import waiting

# Patched by name on the module under test: mypy will not accept a re-exported
# ``time``/``orm`` attribute, and naming the real modules keeps the patch target
# explicit anyway.
import time as _time_module
from aiida_agents.tools.execution.waiting import (
    _MAX_TIMEOUT_SECONDS,
    wait_for_process,
)


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the poll interval free, so a timeout test is not a slow test."""
    monkeypatch.setattr(_time_module, "sleep", lambda _seconds: None)


class _NeverFinishes:
    """A process node that stays running however often it is polled."""

    pk = 4242
    process_label = "PwRelaxWorkChain"
    exit_status = None
    exit_message = None
    is_terminated = False

    def __init__(self) -> None:
        from aiida.engine import ProcessState

        self.process_state = ProcessState.WAITING
        self.polls = 0


@pytest.fixture
def running_process(monkeypatch: pytest.MonkeyPatch, no_sleep: None) -> _NeverFinishes:
    """Resolve any identifier to a process that never terminates."""
    node = _NeverFinishes()

    def _load(_identifier: str) -> t.Any:
        node.polls += 1
        return node

    monkeypatch.setattr(waiting, "load_node", _load)
    # isinstance(node, orm.ProcessNode) must pass for the stub to get past the
    # type guard; patching the check keeps the stub small and honest about what
    # it is standing in for.
    monkeypatch.setattr(
        orm, "ProcessNode", (orm.ProcessNode, _NeverFinishes), raising=False
    )
    return node


class TestAlreadyFinished:
    """The common case: the process is done before the wait starts."""

    def test_a_finished_calculation_returns_immediately(
        self, add_calc: orm.CalcJobNode
    ) -> None:
        result = wait_for_process(str(add_calc.pk))

        assert result["terminated"] is True
        assert result["pk"] == add_calc.pk
        assert result["exit_status"] == 0

    def test_its_outputs_are_reported_with_their_link_labels(
        self, add_calc: orm.CalcJobNode
    ) -> None:
        """The link label is the handle: 'output_structure' is what chains."""
        result = wait_for_process(str(add_calc.pk))

        labels = {output["link_label"] for output in result["outputs"]}
        assert "sum" in labels
        assert all(output["pk"] for output in result["outputs"])

    def test_a_failed_process_is_terminated_not_hidden(
        self, failed_multiply_add: orm.WorkChainNode
    ) -> None:
        """Terminated says it stopped, exit_status says how it went."""
        result = wait_for_process(str(failed_multiply_add.pk))

        assert result["terminated"] is True
        assert result["exit_status"] == 400


class TestStillRunning:
    """Giving up is a statement about the wait, not about the process."""

    def test_a_timeout_reports_the_process_as_unfinished(
        self, running_process: _NeverFinishes
    ) -> None:
        result = wait_for_process("4242", timeout_seconds=1)

        assert result["terminated"] is False
        assert result["state"] == "waiting"

    def test_no_outputs_are_reported_while_it_runs(
        self, running_process: _NeverFinishes
    ) -> None:
        """A half-written result read as final is how a chain goes wrong."""
        result = wait_for_process("4242", timeout_seconds=1)

        assert result["outputs"] == []

    def test_it_polls_more_than_once_before_giving_up(
        self, running_process: _NeverFinishes
    ) -> None:
        """Otherwise 'waiting' would just be one status check with extra steps."""
        wait_for_process("4242", timeout_seconds=6)

        assert running_process.polls > 1

    def test_the_state_is_re_read_rather_than_cached(
        self, running_process: _NeverFinishes
    ) -> None:
        """The daemon writes the state from another process, so reload each poll."""
        before = running_process.polls

        wait_for_process("4242", timeout_seconds=4)

        assert running_process.polls > before + 1


class TestBounds:
    """The wait cannot be talked into blocking the agent indefinitely."""

    def test_an_over_long_timeout_is_capped(
        self, running_process: _NeverFinishes, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Asking for ten times the cap still gives up at the cap.

        A real wait would take half an hour, so the clock is driven forward in
        large steps instead. The assertion is on how long the loop *believed*
        it waited: uncapped it would have kept going to ten times this.
        """
        ticks = iter(float(n) * _MAX_TIMEOUT_SECONDS / 2 for n in range(100))
        monkeypatch.setattr(_time_module, "monotonic", lambda: next(ticks))

        result = wait_for_process("4242", timeout_seconds=_MAX_TIMEOUT_SECONDS * 10)

        assert result["terminated"] is False
        assert result["waited_seconds"] <= _MAX_TIMEOUT_SECONDS * 1.5

    @pytest.mark.parametrize("bad", [0, -1])
    def test_a_non_positive_timeout_is_rejected(
        self, bad: int, add_calc: orm.CalcJobNode
    ) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            wait_for_process(str(add_calc.pk), timeout_seconds=bad)

    def test_a_data_node_is_rejected(
        self, silicon_structure: orm.StructureData
    ) -> None:
        with pytest.raises(WrongNodeType, match="not a process node"):
            wait_for_process(str(silicon_structure.pk))


def test_it_is_exported_for_the_agent_and_the_server() -> None:
    """A tool nothing can reach is not a feature."""
    assert execution_tools.wait_for_process is wait_for_process
