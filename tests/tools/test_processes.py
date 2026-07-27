"""Tests for ``aiida_agents.tools.processes``.

These tools wrap the ORM / ``aiida-restapi`` thinly, so the tests target only
what is *ours*: the output-dict contract, pk/uuid acceptance, the missing-node
behaviour (the function lets a bare ``NotExistent`` propagate), and the
``%process%`` filter plus state-from-attributes assembly. The surface adapters
that turn that ``NotExistent`` into a ``ToolError`` (MCP) or ``ModelRetry``
(agent) are covered in ``tests/mcp/test_errors.py`` and
``tests/agents/test_errors.py``. See ``tests/conftest.py`` for the real,
session-scoped process fixtures.
"""

from __future__ import annotations

import pytest
from aiida import orm
from aiida.common import timezone

from aiida_agents.tools.processes import (
    get_process_report,
    get_process_status,
    list_processes,
)


@pytest.mark.parametrize("by", ["pk", "uuid"])
def test_get_process_status(add_calc: orm.CalcJobNode, by: str) -> None:
    """A finished process's full status is returned, by pk or uuid alike.

    The pk/uuid axis is the regression guard for the identifier handling; the
    rest pins the output-dict contract the tool exposes to the agent.
    """
    identifier = str(add_calc.pk) if by == "pk" else add_calc.uuid

    assert get_process_status(identifier) == {
        "pk": add_calc.pk,
        "process_label": "ArithmeticAddCalculation",
        "process_type": "aiida.calculations:core.arithmetic.add",
        "state": "finished",
        "exit_status": 0,
        "exit_message": None,
    }


@pytest.mark.usefixtures("aiida_profile")
def test_get_process_status_not_found() -> None:
    """The bare function lets a ``NotExistent`` propagate, naming the identifier.

    Each surface adapts that exception at its own boundary; the conversions live
    in ``tests/mcp/test_errors.py`` (ToolError) and ``tests/agents/test_errors.py``
    (ModelRetry).
    """
    from aiida.common.exceptions import NotExistent

    with pytest.raises(NotExistent, match="987654321"):
        get_process_status("987654321")


@pytest.mark.usefixtures("aiida_profile")
def test_get_process_status_rejects_non_process_node() -> None:
    """A valid pk for a *data* node raises WrongNodeType, not a cryptic crash.

    Regression for the wrong-tool case: a model passing a data-node pk to
    get_process_status used to hit ``AttributeError`` on ``node.process_label``;
    now it gets an AiiDA exception the surfaces turn into a clear message.
    """
    from aiida_agents.tools._orm import WrongNodeType

    data = orm.Int(42).store()
    with pytest.raises(WrongNodeType, match="not a process node"):
        get_process_status(str(data.pk))


def test_list_processes(add_calc: orm.CalcJobNode) -> None:
    """The process filter selects process nodes and pulls state from attributes."""
    records = list_processes(limit=50)

    # The ``%process%`` node-type filter excludes plain data nodes.
    assert all("process" in record["node_type"] for record in records)

    # State and exit status come from the per-node attributes lookup (the N+1).
    calc = next((r for r in records if r["pk"] == add_calc.pk), None)
    assert calc is not None
    assert calc["state"] == "finished"
    assert calc["exit_status"] == 0


@pytest.mark.parametrize("by", ["pk", "uuid"])
def test_get_process_report_calcjob(add_calc: orm.CalcJobNode, by: str) -> None:
    """A CalcJob's report includes its identity and log-message count, by pk or uuid."""
    identifier = str(add_calc.pk) if by == "pk" else add_calc.uuid

    result = get_process_report(identifier)

    assert result["pk"] == add_calc.pk
    assert result["process_label"] == "ArithmeticAddCalculation"
    assert result["node_type"] == "CalcJobNode"
    assert result["state"] == "finished"
    assert result["exit_status"] == 0
    assert result["exit_message"] is None
    # Mirrors verdi process report's calcjob format: identity, scheduler
    # output/errors, then the log messages.
    assert str(add_calc.pk) in result["report"]
    assert "LOG MESSAGES" in result["report"]


def test_get_process_report_workchain(
    multiply_add_workchain: orm.WorkChainNode,
) -> None:
    """A WorkChain's report includes its own REPORT-level log messages."""
    result = get_process_report(str(multiply_add_workchain.pk))

    assert result["pk"] == multiply_add_workchain.pk
    assert result["process_label"] == "MultiplyAddWorkChain"
    assert result["node_type"] == "WorkChainNode"
    assert result["state"] == "finished"
    # The workchain logs a REPORT message when it submits its sub-calculation.
    assert "Submitted the" in result["report"]


def test_get_process_report_includes_attached_log_messages(
    add_calc: orm.CalcJobNode,
) -> None:
    """A log message attached to the node surfaces verbatim in its report."""
    orm.Log(
        time=timezone.now(),
        loggername="test.logger",
        levelname="ERROR",
        dbnode_id=add_calc.pk,
        message="boom, something broke",
    ).store()

    result = get_process_report(str(add_calc.pk))

    assert "boom, something broke" in result["report"]
    assert "ERROR" in result["report"]


def test_get_process_report_invalid_levelname_raises(
    add_calc: orm.CalcJobNode,
) -> None:
    """An unrecognised levelname is a clear ValueError, not a KeyError from AiiDA."""
    with pytest.raises(ValueError, match="not a recognised log level"):
        get_process_report(str(add_calc.pk), levelname="NOT_A_LEVEL")


@pytest.mark.usefixtures("aiida_profile")
def test_get_process_report_not_found() -> None:
    """The bare function lets a ``NotExistent`` propagate, naming the identifier."""
    from aiida.common.exceptions import NotExistent

    with pytest.raises(NotExistent, match="987654321"):
        get_process_report("987654321")


@pytest.mark.usefixtures("aiida_profile")
def test_get_process_report_rejects_non_process_node() -> None:
    """A valid pk for a *data* node raises WrongNodeType, not a cryptic crash."""
    from aiida_agents.tools._orm import WrongNodeType

    data = orm.Int(42).store()
    with pytest.raises(WrongNodeType, match="not a process node"):
        get_process_report(str(data.pk))


class TestReadingRetrievedFiles:
    """The code's own output must be reachable, not just AiiDA's log.

    ``get_process_report`` returns what AiiDA recorded about a run. When a
    simulation fails for a physics reason -- "convergence NOT achieved", a bad
    pseudopotential -- the code writes that in its own output file, which AiiDA
    stores in the calculation's ``retrieved`` folder and nothing could open.
    ``get_node_outputs`` showed the agent that the folder existed while leaving
    it unable to look inside, so diagnosis stopped at an exit code.
    """

    def test_lists_the_files_the_calculation_brought_back(
        self, add_calc: orm.CalcJobNode
    ) -> None:
        from aiida_agents.tools.processes import list_retrieved_files

        listing = list_retrieved_files(str(add_calc.pk))

        assert listing["pk"] == add_calc.pk
        assert listing["retrieved_pk"] == add_calc.outputs.retrieved.pk
        names = [f["name"] for f in listing["files"]]
        # The scheduler streams are always retrieved; aiida.out is this
        # calculation's own output.
        assert "_scheduler-stdout.txt" in names
        assert "aiida.out" in names
        assert all(f["size_bytes"] >= 0 for f in listing["files"])

    def test_reads_a_file_the_agent_previously_could_not_reach(
        self, add_calc: orm.CalcJobNode
    ) -> None:
        from aiida_agents.tools.processes import get_retrieved_file

        result = get_retrieved_file(str(add_calc.pk), "aiida.out")

        assert result["pk"] == add_calc.pk
        assert result["filename"] == "aiida.out"
        assert result["truncated"] is False
        assert result["lines_returned"] == result["lines_total"]
        # ArithmeticAddCalculation writes the sum; the point is that whatever
        # the code wrote is now readable, not what it happens to say.
        assert isinstance(result["content"], str)

    @staticmethod
    def _calc_with_output(
        computer: orm.Computer, text: str, filename: str = "big.out"
    ) -> orm.CalcJobNode:
        """A stored CalcJob whose retrieved folder holds one known file.

        Built rather than reusing ``add_calc``: a stored node's repository is
        immutable, so a file of a chosen length cannot be added to a fixture
        run after the fact.
        """
        from aiida.common.links import LinkType

        calc = orm.CalcJobNode(computer=computer)
        calc.set_option("resources", {"num_machines": 1, "num_mpiprocs_per_machine": 1})
        calc.store()

        folder = orm.FolderData()
        folder.base.repository.put_object_from_bytes(text.encode(), filename)
        folder.base.links.add_incoming(
            calc, link_type=LinkType.CREATE, link_label="retrieved"
        )
        folder.store()
        return calc

    def test_tail_lines_returns_the_end_and_flags_truncation(
        self, arithmetic_add_code: orm.InstalledCode
    ) -> None:
        """A failing code says why at the end, so tailing must be honest about it."""
        from aiida_agents.tools.processes import get_retrieved_file

        calc = self._calc_with_output(
            arithmetic_add_code.computer,
            "\n".join(f"line {i}" for i in range(50)),
        )

        result = get_retrieved_file(str(calc.pk), "big.out", tail_lines=5)

        assert result["lines_total"] == 50
        assert result["lines_returned"] == 5
        assert result["truncated"] is True
        assert result["content"].endswith("line 49")

    def test_whole_small_file_is_not_flagged_as_truncated(
        self, arithmetic_add_code: orm.InstalledCode
    ) -> None:
        """Truncation must mean truncation, or the flag is noise."""
        from aiida_agents.tools.processes import get_retrieved_file

        calc = self._calc_with_output(arithmetic_add_code.computer, "a\nb\nc")

        result = get_retrieved_file(str(calc.pk), "big.out")

        assert result["truncated"] is False
        assert result["lines_returned"] == result["lines_total"] == 3

    def test_oversized_file_is_capped_keeping_the_end(
        self, arithmetic_add_code: orm.InstalledCode
    ) -> None:
        """Past the char cap, keep the tail -- that is where the error is."""
        from aiida_agents.tools.processes import _MAX_CHARS, get_retrieved_file

        text = (
            "\n".join(f"padding line {i}" for i in range(4000)) + "\nFINAL ERROR HERE"
        )
        calc = self._calc_with_output(arithmetic_add_code.computer, text)

        result = get_retrieved_file(str(calc.pk), "big.out")

        assert len(text) > _MAX_CHARS  # the case under test is real
        assert result["truncated"] is True
        assert len(result["content"]) <= _MAX_CHARS
        assert result["content"].endswith("FINAL ERROR HERE")

    def test_unknown_filename_names_what_is_available(
        self, add_calc: orm.CalcJobNode
    ) -> None:
        from aiida_agents.tools.processes import get_retrieved_file

        with pytest.raises(FileNotFoundError, match="Available:"):
            get_retrieved_file(str(add_calc.pk), "no-such-file.out")

    def test_workchain_is_told_to_find_its_calcjob(
        self, multiply_add_workchain: orm.WorkChainNode
    ) -> None:
        """A WorkChain retrieves nothing itself; guessing which child was meant is wrong."""
        from aiida_agents.tools._orm import WrongNodeType
        from aiida_agents.tools.processes import list_retrieved_files

        with pytest.raises(WrongNodeType, match="get_node_outputs|query_nodes"):
            list_retrieved_files(str(multiply_add_workchain.pk))

    def test_data_node_is_rejected_clearly(
        self, silicon_structure: orm.StructureData
    ) -> None:
        from aiida_agents.tools._orm import WrongNodeType
        from aiida_agents.tools.processes import list_retrieved_files

        with pytest.raises(WrongNodeType, match="not a CalcJob"):
            list_retrieved_files(str(silicon_structure.pk))
