"""Project-wide pytest fixtures.

All fixtures that create AiiDA nodes use a real, temporary SQLite profile spun
up by pytest-aiida. No mocks for database interaction — tests run against actual
AiiDA storage so the ORM behaviour is always exercised.

Docs: https://docs.pytest.org/en/stable/how-to/fixtures.html#conftest-py-sharing-fixtures-across-multiple-files
"""

from __future__ import annotations

import pytest
from aiida import orm
from aiida.common.links import LinkType


@pytest.fixture(scope="session")
def aiida_profile(aiida_profile_clean):
    """Session-scoped clean AiiDA profile provided by pytest-aiida."""
    return aiida_profile_clean


@pytest.fixture
def add_calc(aiida_profile) -> orm.CalcJobNode:
    """A finished ArithmeticAddCalculation with two Int inputs and three outputs.

    Mirrors the minimal node graph used in aiida-core's own test suite so
    fixture data is recognisable and well-understood.
    """
    computer = orm.Computer(
        label="localhost",
        hostname="localhost",
        transport_type="core.local",
        scheduler_type="core.direct",
    ).store()
    computer.set_minimum_job_poll_interval(0)

    code = orm.InstalledCode(
        label="add",
        computer=computer,
        filepath_executable="/bin/add",
    ).store()

    x = orm.Int(1).store()
    y = orm.Int(2).store()

    calc = orm.CalcJobNode()
    calc.computer = computer
    calc.set_option("resources", {"num_machines": 1})
    calc.base.attributes.set("process_label", "ArithmeticAddCalculation")
    calc.base.attributes.set("process_state", "finished")
    calc.base.attributes.set("exit_status", 0)
    calc.base.attributes.set(
        "process_type", "aiida.calculations:core.arithmetic.add"
    )
    calc.store()

    calc.base.links.add_incoming(x, link_type=LinkType.INPUT_CALC, link_label="x")
    calc.base.links.add_incoming(y, link_type=LinkType.INPUT_CALC, link_label="y")
    calc.base.links.add_incoming(
        code, link_type=LinkType.INPUT_CALC, link_label="code"
    )

    result = orm.Int(3).store()
    remote = orm.RemoteData(computer=computer, remote_path="/tmp").store()
    retrieved = orm.FolderData().store()

    result.base.links.add_incoming(
        calc, link_type=LinkType.CREATE, link_label="sum"
    )
    remote.base.links.add_incoming(
        calc, link_type=LinkType.CREATE, link_label="remote_folder"
    )
    retrieved.base.links.add_incoming(
        calc, link_type=LinkType.CREATE, link_label="retrieved"
    )

    return calc


@pytest.fixture
def multiply_add_workchain(aiida_profile, add_calc) -> orm.WorkChainNode:
    """A finished MultiplyAddWorkChain that called one calcfunction and one calcjob.

    The workchain returns a single 'result' output so get_node_outputs can be
    tested for both call_calc and return link types.
    """
    wc = orm.WorkChainNode()
    wc.base.attributes.set("process_label", "MultiplyAddWorkChain")
    wc.base.attributes.set("process_state", "finished")
    wc.base.attributes.set("exit_status", 0)
    wc.store()

    # Calcfunction sub-process (multiply step)
    multiply = orm.CalcFunctionNode()
    multiply.base.attributes.set("process_state", "finished")
    multiply.base.attributes.set("exit_status", 0)
    multiply.store()
    multiply.base.links.add_incoming(
        wc, link_type=LinkType.CALL_CALC, link_label="multiply"
    )

    # Reuse the add_calc fixture as the second sub-process
    add_calc.base.links.add_incoming(
        wc, link_type=LinkType.CALL_CALC, link_label="add"
    )

    result = orm.Int(6).store()
    result.base.links.add_incoming(
        wc, link_type=LinkType.RETURN, link_label="result"
    )

    return wc


@pytest.fixture
def silicon_structure(aiida_profile) -> orm.StructureData:
    """A minimal two-atom silicon unit cell stored in the database."""
    structure = orm.StructureData(
        cell=[[0, 2.715, 2.715], [2.715, 0, 2.715], [2.715, 2.715, 0]]
    )
    structure.append_atom(position=(0.0, 0.0, 0.0), symbols="Si")
    structure.append_atom(position=(1.3575, 1.3575, 1.3575), symbols="Si")
    structure.store()
    return structure