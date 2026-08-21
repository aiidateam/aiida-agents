"""Project-wide pytest fixtures & hooks.

The process fixtures below run real AiiDA calculations/workflows in-process (no
daemon or broker needed). They are **session-scoped**: each is executed once for
the whole test run, not per test, since spinning up the engine is expensive.
The shared profile is never cleaned between tests, so the tools' tests assert by
node identity (the pk/uuid they created) rather than against global counts.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from aiida import orm
from aiida.calculations.arithmetic.add import ArithmeticAddCalculation
from aiida.common.exceptions import IncompatibleStorageSchema
from aiida.engine import run_get_node
from aiida.manage import get_manager
from aiida.workflows.arithmetic.multiply_add import MultiplyAddWorkChain

# Pull in AiiDA's test fixtures (``aiida_profile``, ``aiida_localhost``, ...).
# ``aiida_profile`` is session-scoped and autouse: it loads a temporary
# ``core.sqlite_dos`` profile that needs no external services, so the MCP tools
# run against a real database, not mocks.
pytest_plugins = ["aiida.tools.pytest_fixtures"]

if TYPE_CHECKING:
    from aiida.manage.manager import Manager


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]
    """Run each test from a clean temp directory.

    pydantic-settings reads a ``.env`` from the CWD, so running from the repo
    root leaks a developer's local ``.env`` (e.g. a cloud provider selected
    without an exported key) into ``ModelSettings`` / ``get_agent`` and breaks
    otherwise-hermetic tests. A test that needs a ``.env`` writes its own and
    chdirs to it, overriding this.
    """
    monkeypatch.chdir(tmp_path)


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


@pytest.fixture(scope="session")
def failed_multiply_add(
    arithmetic_add_code: orm.InstalledCode,
) -> orm.WorkChainNode:
    """A real, *failed* ``MultiplyAddWorkChain`` run (session-scoped).

    Runs ``core.arithmetic.multiply_add`` with ``z=-100``, so ``x * y + z`` is
    negative: the nested ``ArithmeticAddCalculation`` exits 410 and the work
    chain exits 400. The nesting is the point --- the work chain's own code says
    only that a sub-process failed, and the cause is one level down --- so this
    is the fixture for anything that has to find a root cause rather than read
    a top-level exit status.

    :return: The stored top-level ``WorkChainNode`` for the failed run.
    """
    _, node = run_get_node(
        MultiplyAddWorkChain,
        x=orm.Int(2),
        y=orm.Int(3),
        z=orm.Int(-100),
        code=arithmetic_add_code,
    )
    assert isinstance(node, orm.WorkChainNode)
    assert node.exit_status, "the fixture is only useful if the run actually failed"
    return node


@pytest.fixture
def without_plugins(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build agents as if no plugin were installed.

    Two tests pin an agent's tool surface by *exact* equality, which is only
    meaningful for the tools this package registers itself. A plugin may
    contribute more --- ``dev/qe_rag_stub`` does, once ``uv sync --group qe``
    has run --- and without this those assertions fail on a developer's machine
    while passing in CI, where no plugin is installed. That is the worst
    direction for a test to be wrong in.

    Plugin contribution is not left untested by stubbing it here; it is tested
    where it belongs, in ``tests/plugins/``.
    """
    from aiida_agents.agents import analysis

    monkeypatch.setattr(analysis, "discover_plugins", lambda: ())


@pytest.fixture
def unopened_profile_storage(monkeypatch: pytest.MonkeyPatch) -> Iterator[Manager]:
    """Put the manager back in the lazy state a fresh process starts in.

    The session ``aiida_profile`` fixture has opened the storage long before any
    test runs, so "does this entry point open it?" is otherwise unaskable.
    ``Manager.reset_profile_storage`` would *close* the backend, and the
    session-scoped node fixtures above hold ORM objects bound to it, so the rest
    of the suite would fail on a dead engine. Detaching it instead leaves the
    original intact for ``monkeypatch`` to reattach on teardown; whatever the
    test opened in its place is closed here first.

    :return: The manager, detached from its storage backend.
    """
    manager = get_manager()
    monkeypatch.setattr(manager, "_profile_storage", None)
    yield manager
    if manager.profile_storage_loaded:
        manager.get_profile_storage().close()


@pytest.fixture
def unmigrated_storage_error() -> IncompatibleStorageSchema:
    """AiiDA's real wording for a profile an ``aiida-core`` upgrade left behind.

    Quoted rather than paraphrased because its shape is what the error handling
    under test has to survive: it opens with a blank line, and the command that
    resolves it sits indented on a line of its own.

    :return: The exception a storage on an older schema raises when opened.
    """
    return IncompatibleStorageSchema(
        "\nDatabase schema version `main_0001` is incompatible with the "
        "required schema version `main_0002`.\nTo migrate the database schema "
        "version to the current one, run the following command:\n\n"
        "    verdi -p test storage migrate\n"
    )
