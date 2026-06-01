"""Project-wide pytest fixtures & hooks.

The process fixtures below run real AiiDA calculations/workflows in-process (no
daemon or broker needed). They are **session-scoped**: each is executed once for
the whole test run, not per test, since spinning up the engine is expensive.
The shared profile is never cleaned between tests, so the tools' tests assert by
node identity (the pk/uuid they created) rather than against global counts.
"""

from __future__ import annotations

import pytest
from aiida import orm

# Pull in AiiDA's test fixtures (``aiida_profile``, ``aiida_localhost``, ...).
# ``aiida_profile`` is session-scoped and autouse: it loads a temporary
# ``core.sqlite_dos`` profile that needs no external services, so the MCP tools
# run against a real database, not mocks.
pytest_plugins = ["aiida.tools.pytest_fixtures"]


@pytest.fixture(scope="session")
def arithmetic_add_code(
    tmp_path_factory: pytest.TempPathFactory,
) -> orm.InstalledCode:
    """A configured localhost computer with a ``core.arithmetic.add`` code.

    Built directly (rather than via the function-scoped ``aiida_localhost`` /
    ``aiida_code_installed`` fixtures) so it can be session-scoped and shared by
    the run fixtures below.

    :return: The stored ``InstalledCode`` running ``/bin/bash`` on localhost.
    """
    computer = orm.Computer(
        label="localhost-agents-tests",
        hostname="localhost",
        workdir=str(tmp_path_factory.mktemp("aiida-work")),
        transport_type="core.local",
        scheduler_type="core.direct",
    ).store()
    computer.set_minimum_job_poll_interval(0)
    computer.configure()

    return orm.InstalledCode(
        label="bash",
        computer=computer,
        filepath_executable="/bin/bash",
        default_calc_job_plugin="core.arithmetic.add",
    ).store()


@pytest.fixture(scope="session")
def add_calc(arithmetic_add_code: orm.InstalledCode) -> orm.CalcJobNode:
    """A real, finished ``ArithmeticAddCalculation`` run (session-scoped).

    Runs ``core.arithmetic.add`` with ``x=2``, ``y=3`` in-process, producing a
    genuine process node: inputs ``x``, ``y`` and ``code`` (``input_calc``);
    outputs ``sum`` (=5), ``remote_folder`` and ``retrieved`` (``create``).

    :return: The stored ``CalcJobNode`` for the completed calculation.
    """
    from aiida.calculations.arithmetic.add import ArithmeticAddCalculation
    from aiida.engine import run_get_node

    _, node = run_get_node(
        ArithmeticAddCalculation,
        x=orm.Int(2),
        y=orm.Int(3),
        code=arithmetic_add_code,
    )
    assert isinstance(node, orm.CalcJobNode)
    return node


@pytest.fixture(scope="session")
def multiply_add_workchain(
    arithmetic_add_code: orm.InstalledCode,
) -> orm.WorkChainNode:
    """A real, finished ``MultiplyAddWorkChain`` run (session-scoped).

    Runs ``core.arithmetic.multiply_add`` with ``x=2``, ``y=3``, ``z=4``
    in-process. The work chain multiplies ``x * y`` (a calcfunction) and adds
    ``z`` (an ``ArithmeticAddCalculation``), so it yields a process tree: the
    top ``WorkChainNode`` with ``input_work`` links, ``call_calc`` links to a
    ``CalcFunctionNode`` and a ``CalcJobNode``, and a ``result`` (=10) return.

    :return: The stored top-level ``WorkChainNode`` for the completed run.
    """
    from aiida.engine import run_get_node
    from aiida.workflows.arithmetic.multiply_add import MultiplyAddWorkChain

    _, node = run_get_node(
        MultiplyAddWorkChain,
        x=orm.Int(2),
        y=orm.Int(3),
        z=orm.Int(4),
        code=arithmetic_add_code,
    )
    assert isinstance(node, orm.WorkChainNode)
    return node


@pytest.fixture(scope="session")
def silicon_structure() -> orm.StructureData:
    """A stored two-atom silicon ``StructureData`` (session-scoped).

    The only ``StructureData`` created in the test session, so structure
    searches can assert on it by identity.

    :return: The stored ``StructureData`` (formula ``Si2``, two sites).
    """
    structure = orm.StructureData(cell=[[3.0, 0, 0], [0, 3.0, 0], [0, 0, 3.0]])
    structure.append_atom(position=(0, 0, 0), symbols="Si")
    structure.append_atom(position=(1.5, 1.5, 1.5), symbols="Si")
    structure.store()
    return structure
