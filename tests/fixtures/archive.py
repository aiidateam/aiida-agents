"""A miniature of the archive the query tool is built for.

Shaped after the ``fermi_surfaces`` archive rather than invented: there, a group
holds ~15k ``CalcJobNode`` s which carry the structural metadata as *extras*,
and the ``StructureData`` they came from sits two hops upstream::

    StructureData --input_calc--> CalcJobNode --create--> RemoteData --input_calc--> CalcJobNode
                                  (upstream)                                        (in the group,
                                                                                     carries extras)

Two details of that shape matter and are easy to get wrong.

*Which* extras sit where. Extras are a ``Node`` feature, so both the structures
and the calculations carry them, and some keys appear on both: ``formula_hill``
is on a structure *and* on the calculation derived from it, so a query for it
matches two different nodes unless an entity type says which is meant. Other
keys belong to one side only -- ``bravais_lattice`` to structures, and the
electronic results (``insulator``, ``pw_bandgap``, ``spacegroup_number``) to
calculations. "How many *structures* are metallic" is therefore answered by 0,
not because a structure could not carry ``insulator``, but because in this data
none does.

How far apart they are. The structure is a transitive ancestor of the grouped
calculation, two hops up behind the remote folder, so ``with_outgoing`` finds
nothing and only ``with_descendants`` reaches it.

This module is a plain builder, not a pytest plugin, so an evaluation harness or
a script can put the same content into any profile it likes. The pytest wrapper
lives in ``tests/tools/conftest.py``.

Every row in ``CALCULATIONS`` earns its place by making some wrong
implementation return a different answer; see the table there.
"""

from __future__ import annotations

import typing as t
from dataclasses import dataclass

from aiida import orm
from aiida.common.links import LinkType

GROUP_LABEL = "test/fermi/calculations"


@dataclass(frozen=True)
class Calculation:
    """One calculation, plus the structure it came from."""

    formula: str
    spacegroup: int
    insulator: bool
    bandgap: float
    overlapping_semicore: bool
    exit_status: int
    bravais_lattice: str
    """Structure-side extra; no calculation carries it."""
    in_group: bool = True


# Each row exists to make a specific mistake visible:
#
# | property                                  | catches                                   |
# |-------------------------------------------|-------------------------------------------|
# | insulator and spacegroup<195 sets OVERLAP | a faked OR union (summing subcounts)      |
# | bandgaps 9.0 and 10.0                     | a text sort: "10.0" < "9.0"               |
# | Ag is not in the group                    | group scoping being ignored               |
# | Cu/VO2 failed, the rest passed            | a filter on a joined entity being dropped |
# | CoV is unique                             | ==, !==, like, in                         |
# | overlapping_semicore splits the metals    | multi-condition AND                       |
# | formula_hill is on structure AND calc     | an entity type being ignored              |
CALCULATIONS: tuple[Calculation, ...] = (
    #          formula  sg   insulator  gap    semicore  exit  bravais    in_group
    Calculation("Si", 227, True, 1.12, False, 0, "cF"),
    Calculation("Cu", 225, False, 0.0, False, 302, "cF"),
    Calculation("MgO", 225, True, 10.0, False, 0, "cF"),
    Calculation("BN", 194, True, 9.0, True, 0, "hP"),
    Calculation("VO2", 14, False, 0.0, True, 400, "mP"),
    Calculation("CoV", 216, False, 0.0, False, 0, "cP"),
    Calculation("Ag", 225, False, 0.0, False, 0, "cF", in_group=False),
)


@dataclass(frozen=True)
class QueryArchive:
    """Handle on the built content: the nodes, and the truth about them."""

    group_label: str
    rows: tuple[Calculation, ...]
    calculations: dict[str, orm.CalcJobNode]
    """Grouped calcjob per formula -- the node carrying the extras."""
    structures: dict[str, orm.StructureData]
    """The StructureData two hops upstream of each calcjob."""

    def expect(
        self, predicate: t.Callable[[Calculation], bool] = lambda _: True
    ) -> int:
        """Count rows in the group matching `predicate`.

        Ground truth derived from the source rows, so a test never hardcodes a
        number that could drift from the data.
        """
        return sum(1 for row in self.rows if row.in_group and predicate(row))

    def formulas(self, predicate: t.Callable[[Calculation], bool]) -> set[str]:
        """Formulas in the group matching `predicate`."""
        return {row.formula for row in self.rows if row.in_group and predicate(row)}


def _computer() -> orm.Computer:
    """A local computer, reused if the profile already has one."""
    try:
        return orm.Computer.collection.get(label="localhost")
    except Exception:  # noqa: BLE001 - NotExistent, but keep the builder profile-agnostic
        computer = orm.Computer(
            label="localhost",
            hostname="localhost",
            transport_type="core.local",
            scheduler_type="core.direct",
            workdir="/tmp",
        ).store()
        computer.configure()
        return computer


def _calcjob(computer: orm.Computer) -> orm.CalcJobNode:
    node = orm.CalcJobNode(computer=computer)
    node.set_option("resources", {"num_machines": 1})
    return node


def _build_chain(
    row: Calculation, computer: orm.Computer
) -> tuple[orm.StructureData, orm.CalcJobNode]:
    """structure -> upstream calc -> remote folder -> the calc that gets grouped."""
    structure = orm.StructureData(cell=[[4.0, 0, 0], [0, 4.0, 0], [0, 0, 4.0]])
    structure.append_atom(position=(0, 0, 0), symbols="Co")
    structure.store()
    # `formula_hill` is shared with the calculation below, `bravais_lattice` is
    # not, and `insulator` is deliberately absent from this side.
    structure.base.extras.set_many(
        {"formula_hill": row.formula, "bravais_lattice": row.bravais_lattice}
    )

    upstream = _calcjob(computer)
    upstream.base.links.add_incoming(
        structure, link_type=LinkType.INPUT_CALC, link_label="structure"
    )
    upstream.store()

    remote = orm.RemoteData(remote_path="/tmp/scratch", computer=computer)
    remote.base.links.add_incoming(
        upstream, link_type=LinkType.CREATE, link_label="remote_folder"
    )
    remote.store()
    # only seal once the outgoing link exists: a sealed node rejects new links
    upstream.set_exit_status(0)
    upstream.seal()

    calculation = _calcjob(computer)
    calculation.base.links.add_incoming(
        remote, link_type=LinkType.INPUT_CALC, link_label="parent_folder"
    )
    calculation.store()
    calculation.set_exit_status(row.exit_status)
    calculation.seal()
    calculation.base.extras.set_many(
        {
            "formula_hill": row.formula,
            "spacegroup_number": row.spacegroup,
            "insulator": row.insulator,
            "pw_bandgap": row.bandgap,
            "overlapping_semicore": row.overlapping_semicore,
        }
    )
    return structure, calculation


def build_query_archive(
    group_label: str = GROUP_LABEL,
    rows: tuple[Calculation, ...] = CALCULATIONS,
) -> QueryArchive:
    """Store the content and return a handle on it.

    Additive: it stores nodes and never cleans the profile, so it is safe
    alongside session-scoped fixtures that other tests depend on.

    :param group_label: label for the group the calculations are added to.
    :param rows: the calculations to build; defaults to :data:`CALCULATIONS`.
    :return: the handle, for looking up nodes and computing expected counts.
    """
    computer = _computer()
    group = orm.Group(label=group_label).store()

    structures: dict[str, orm.StructureData] = {}
    calculations: dict[str, orm.CalcJobNode] = {}
    members: list[orm.Node] = []
    for row in rows:
        structure, calculation = _build_chain(row, computer)
        structures[row.formula] = structure
        calculations[row.formula] = calculation
        if row.in_group:
            members.append(calculation)

    group.add_nodes(members)
    return QueryArchive(
        group_label=group_label,
        rows=rows,
        calculations=calculations,
        structures=structures,
    )
