"""Tests for ``aiida_agents.tools.query_builder``.

The pipeline splits, so the tests do too: ``_lower`` is pure and checked by dict
equality with no database at all, while ``query_nodes`` runs against the
miniature archive in ``tests.fixtures.archive`` -- whose every row exists to
make some wrong implementation return a different answer.

Expected counts are derived from that fixture's source rows via
``archive.expect(...)``, never written out as literals, so the data and the
assertions cannot drift apart.
"""

from __future__ import annotations

import typing as t

import pytest
from aiida import orm
from aiida.calculations.arithmetic.add import ArithmeticAddCalculation
from aiida.orm.implementation.querybuilder import EntityRelationships
from aiida.workflows.arithmetic.multiply_add import MultiplyAddWorkChain
from pydantic import ValidationError

from aiida_agents.tools.query_builder import (
    MAX_LIMIT,
    FieldFilter,
    FilterGroup,
    JoinKeyword,
    PathItem,
    QuerySpec,
    QueryValidationError,
    _lower,
    _qualify_field,
    query_nodes,
)
from tests.fixtures import QueryArchive

CALC = "CalcJobNode"


# --------------------------------------------------------- pure: no database


def test_qualify_field() -> None:
    """Bare names are extras keys; columns and containers are left alone."""
    assert _qualify_field("pk") == "pk"
    assert _qualify_field("node_type") == "node_type"
    assert _qualify_field("extras") == "extras"
    assert _qualify_field("attributes") == "attributes"
    assert _qualify_field("extras.insulator") == "extras.insulator"
    assert _qualify_field("attributes.exit_status") == "attributes.exit_status"
    assert _qualify_field("spacegroup_number") == "extras.spacegroup_number"


def test_flat_form_lowers_to_a_single_entity_path() -> None:
    """The flat shorthand becomes a canonical one-entity path."""
    spec = QuerySpec.model_validate(
        {
            "entity_type": "StructureData",
            "filters": {"field": "insulator", "operator": "==", "value": False},
            "count_only": True,
        }
    )
    assert _lower(spec) == {
        "path": [
            {
                "entity_type": "data.core.structure.StructureData.",
                "orm_base": "node",
                "tag": "node",
                "outerjoin": False,
            }
        ],
        "filters": {"node": {"extras.insulator": False}},
    }


def test_group_label_lowers_to_a_join() -> None:
    """`group_label` is sugar for a group entity joined by `with_group`."""
    spec = QuerySpec.model_validate(
        {"entity_type": "node", "group_label": "some/group", "count_only": True}
    )
    lowered = _lower(spec)
    assert [item["tag"] for item in lowered["path"]] == ["_group", "node"]
    assert lowered["path"][1]["joining_keyword"] == "with_group"
    assert lowered["path"][1]["joining_value"] == "_group"
    assert lowered["filters"]["_group"] == {"label": "some/group"}


@pytest.mark.parametrize("field", ["path", "filters", "project", "sort"])
def test_explicit_null_is_treated_as_absent(field: str) -> None:
    """A model saying `"filters": null` must not fail the whole spec."""
    spec = QuerySpec.model_validate(
        {"entity_type": "node", field: None, "count_only": True}
    )
    assert _lower(spec)["path"][0]["entity_type"] == ""


def test_lower_nested_or() -> None:
    """'(A AND B) OR C' becomes QueryBuilder's nested and/or dicts."""
    spec = QuerySpec.model_validate(
        {
            "filters": FilterGroup(
                logic="OR",
                conditions=[
                    FilterGroup(
                        logic="AND",
                        conditions=[
                            FieldFilter(field="a", operator="==", value=1),
                            FieldFilter(field="b", operator="==", value=2),
                        ],
                    ),
                    FieldFilter(field="c", operator=">", value=3),
                ],
            )
        }
    )
    assert _lower(spec)["filters"]["node"] == {
        "or": [
            {"and": [{"extras.a": 1}, {"extras.b": 2}]},
            {"extras.c": {">": 3}},
        ]
    }


def test_lower_provenance_path() -> None:
    """A join lowers to the path entry ``QueryBuilder.from_dict`` expects."""
    spec = QuerySpec.model_validate(
        {
            "path": [
                {"entity_type": CALC, "tag": "calc"},
                {
                    "entity_type": "StructureData",
                    "tag": "st",
                    "joining_keyword": "with_descendants",
                    "joining_value": "calc",
                },
            ],
            "count_only": True,
        }
    )
    assert _lower(spec)["path"][1] == {
        "entity_type": "data.core.structure.StructureData.",
        "orm_base": "node",
        "tag": "st",
        "outerjoin": False,
        "joining_keyword": "with_descendants",
        "joining_value": "calc",
    }


def test_lower_process_plugin_adds_a_process_type_filter() -> None:
    """A process plugin shares its node_type with every plugin of the same
    kind (all CalcJobs are one `CalcJobNode`), so its entity alias must also
    lower to a `process_type` filter -- that's the only thing narrowing the
    query to this specific plugin rather than any CalcJob."""
    spec = QuerySpec.model_validate(
        {"entity_type": "ArithmeticAddCalculation", "count_only": True}
    )
    lowered = _lower(spec)
    assert lowered["path"][0]["entity_type"] == orm.CalcJobNode.class_node_type
    assert lowered["filters"]["node"] == {
        "process_type": ArithmeticAddCalculation.build_process_type()
    }


def test_lower_process_plugin_filter_combines_with_a_user_filter() -> None:
    """The implicit process_type filter is ANDed with, not replaced by, a
    user-supplied filter on the same tag."""
    spec = QuerySpec.model_validate(
        {
            "entity_type": "core.arithmetic.add",  # entry-point name, not class name
            "filters": {"field": "pk", "operator": ">", "value": 0},
            "count_only": True,
        }
    )
    assert _lower(spec)["filters"]["node"] == {
        "and": [
            {"process_type": ArithmeticAddCalculation.build_process_type()},
            {"pk": {">": 0}},
        ]
    }


def test_lower_workchain_plugin_adds_a_process_type_filter() -> None:
    """The same process_type resolution applies to WorkChain plugins, not just
    CalcJob plugins."""
    spec = QuerySpec.model_validate(
        {"entity_type": "MultiplyAddWorkChain", "count_only": True}
    )
    lowered = _lower(spec)
    assert lowered["path"][0]["entity_type"] == orm.WorkChainNode.class_node_type
    assert lowered["filters"]["node"] == {
        "process_type": MultiplyAddWorkChain.build_process_type()
    }


def test_lower_edge_filters_uses_the_edge_qualifier() -> None:
    """edge_filters lowers 'label'/'type' as-is -- no extras/attributes prefixing,
    unlike an entity's own filters."""
    spec = QuerySpec.model_validate(
        {
            "path": [
                {"entity_type": CALC, "tag": "calc"},
                {
                    "entity_type": "StructureData",
                    "tag": "st",
                    "joining_keyword": "with_outgoing",
                    "joining_value": "calc",
                    "edge_filters": {
                        "field": "label",
                        "operator": "==",
                        "value": "output_structure",
                    },
                },
            ],
            "count_only": True,
        }
    )
    assert _lower(spec)["path"][1]["edge_filters"] == {"label": "output_structure"}


@pytest.mark.parametrize(
    "spec_dict,match",
    [
        pytest.param(
            {"entity_type": "Structur"}, "Unknown entity_type", id="unknown-entity"
        ),
        pytest.param(
            {
                "path": [
                    {"entity_type": "node", "tag": "a"},
                    {
                        "entity_type": "node",
                        "tag": "b",
                        "joining_keyword": "with_group",
                        "joining_value": "nope",
                    },
                ]
            },
            "does not name an earlier entity",
            id="dangling-join",
        ),
        pytest.param(
            {
                "path": [
                    {"entity_type": "group", "tag": "g"},
                    {
                        "entity_type": "node",
                        "tag": "n",
                        "joining_keyword": "with_authinfo",
                        "joining_value": "g",
                    },
                ]
            },
            "not a valid join",
            id="illegal-join-for-entity",
        ),
        pytest.param(
            {"entity_type": "node", "project": {"nope": ["pk"]}},
            "Unknown tag",
            id="unknown-tag",
        ),
        pytest.param(
            {"filters": {"field": "pk", "operator": "in", "value": 5}},
            "needs a list of values",
            id="in-needs-a-list",
        ),
        pytest.param(
            {"sort": [{"field": "pw_bandgap", "direction": "desc"}]},
            "requires a 'cast'",
            id="extras-sort-needs-cast",
        ),
        pytest.param(
            {
                "path": [
                    {"entity_type": CALC, "tag": "calc"},
                    {
                        "entity_type": "StructureData",
                        "tag": "st",
                        "joining_keyword": "with_descendants",
                        "joining_value": "calc",
                        "edge_filters": {
                            "field": "label",
                            "operator": "==",
                            "value": "x",
                        },
                    },
                ]
            },
            "does not support filtering the edge",
            id="edge-filters-on-transitive-join",
        ),
        pytest.param(
            {
                "path": [
                    {"entity_type": CALC, "tag": "calc"},
                    {
                        "entity_type": "StructureData",
                        "tag": "st",
                        "joining_keyword": "with_outgoing",
                        "joining_value": "calc",
                        "edge_filters": {"field": "pk", "operator": "==", "value": 1},
                    },
                ]
            },
            "may only use 'label' or 'type'",
            id="edge-filters-bad-field",
        ),
        pytest.param(
            {"entity_type": "node", "group_label": "no-such-group-123"},
            "Group with label 'no-such-group-123' does not exist",
            id="nonexistent-group",
        ),
    ],
)
def test_invalid_specs_say_what_is_wrong(
    spec_dict: dict[str, t.Any], match: str
) -> None:
    """A bad spec fails before the database, naming the problem."""
    spec = QuerySpec.model_validate(spec_dict)
    with pytest.raises(QueryValidationError, match=match):
        query_nodes(spec)


def test_nonexistent_group_is_not_confused_with_an_empty_one(
    query_archive: QueryArchive,
) -> None:
    """A typo'd group_label must fail loudly, not silently return a total of 0.

    Pins the regression where `group_label` lowering into a plain join/filter
    dropped the group-existence check: an existing-but-empty group and a group
    that was never created would otherwise both report 0 matches.
    """
    with pytest.raises(QueryValidationError, match="does not exist"):
        query_nodes(
            QuerySpec.model_validate(
                {"entity_type": "node", "group_label": "definitely-not-a-real-group"}
            )
        )
    # the real group, by contrast, is valid and just happens to exclude "node"
    # (its members are CalcJobNodes), so it must return 0 without raising.
    result = query_nodes(
        QuerySpec.model_validate(
            {
                "entity_type": "StructureData",
                "group_label": query_archive.group_label,
                "count_only": True,
            }
        )
    )
    assert result["total"] == 0


def test_unknown_entity_suggests_a_close_match() -> None:
    """The error is a repair hint, not just a rejection."""
    spec = QuerySpec.model_validate({"entity_type": "StructureDat"})
    with pytest.raises(QueryValidationError, match="Did you mean.*structuredata"):
        query_nodes(spec)


@pytest.mark.parametrize("limit", [0, -1, MAX_LIMIT + 1])
def test_limit_is_bounded(limit: int) -> None:
    """A limit outside 1..MAX_LIMIT is rejected by the schema."""
    with pytest.raises(ValidationError):
        QuerySpec(limit=limit)


def test_invalid_operator_is_rejected_by_the_schema() -> None:
    """'!=' is not AiiDA syntax; '!==' is."""
    with pytest.raises(ValidationError):
        FieldFilter(field="insulator", operator="!=", value=True)  # type: ignore[arg-type]


def test_join_keywords_are_derived_from_entity_relationships() -> None:
    """`JoinKeyword` is generated from `EntityRelationships`, not hand-typed.

    Pins the derivation itself: every keyword AiiDA recognises for any entity
    must be a valid `joining_keyword`, and nothing else is.
    """
    expected = {
        keyword for keywords in EntityRelationships.values() for keyword in keywords
    }
    assert set(t.get_args(JoinKeyword)) == expected


def test_bogus_join_keyword_is_rejected_by_the_schema() -> None:
    """A join keyword AiiDA has never heard of is rejected before validation.

    `JoinKeyword` is a runtime-derived Literal (mypy cannot see its members
    statically, hence no `type: ignore` needed here), so this is the only
    check that a bogus keyword is actually rejected.
    """
    with pytest.raises(ValidationError):
        PathItem(tag="x", joining_keyword="with_nonsense")


# ------------------------------------------- against the miniature archive


def test_entity_type_disambiguates_a_shared_extras_key(
    query_archive: QueryArchive,
) -> None:
    """`formula_hill` is on the structure *and* on its calculation.

    Extras are a Node feature, so the same key can sit on either side. Without an
    entity type a query for it matches both nodes; the entity type is what says
    which was meant.
    """
    formula = {"field": "formula_hill", "operator": "==", "value": "Si"}

    def total(entity_type: str) -> int:
        return query_nodes(
            QuerySpec.model_validate(
                {"entity_type": entity_type, "filters": formula, "count_only": True}
            )
        )["total"]

    assert total("StructureData") == 1
    assert total(CALC) == 1
    assert total("node") == 2  # both, which is the point


def test_a_calculation_only_key_does_not_match_structures(
    query_archive: QueryArchive,
) -> None:
    """ "How many metallic structures" is 0 here, and that is the honest answer.

    Not because a structure could not carry `insulator` -- extras are defined on
    the base Node -- but because in this data none does; the electronic results
    are recorded on the calculations. Silently answering with the calculation
    count instead is the bug worth never repeating.
    """
    metallic = {"field": "insulator", "operator": "==", "value": False}
    structures = query_nodes(
        QuerySpec.model_validate(
            {
                "entity_type": "StructureData",
                "group_label": query_archive.group_label,
                "filters": metallic,
                "count_only": True,
            }
        )
    )
    calculations = query_nodes(
        QuerySpec.model_validate(
            {
                "entity_type": CALC,
                "group_label": query_archive.group_label,
                "filters": metallic,
                "count_only": True,
            }
        )
    )
    assert structures["total"] == 0
    assert calculations["total"] == query_archive.expect(lambda row: not row.insulator)


def test_a_structure_only_key_does_not_match_calculations(
    query_archive: QueryArchive,
) -> None:
    """The converse: `bravais_lattice` is recorded on structures, not calcs."""
    lattice = {"field": "bravais_lattice", "operator": "==", "value": "cF"}
    structures = query_nodes(
        QuerySpec.model_validate(
            {"entity_type": "StructureData", "filters": lattice, "count_only": True}
        )
    )
    calculations = query_nodes(
        QuerySpec.model_validate(
            {"entity_type": CALC, "filters": lattice, "count_only": True}
        )
    )
    assert structures["total"] == sum(
        1 for row in query_archive.rows if row.bravais_lattice == "cF"
    )
    assert calculations["total"] == 0


def test_nested_or_equals_the_manual_union(query_archive: QueryArchive) -> None:
    """The OR count matches the union of the two sets, not the sum of them."""
    insulators = query_archive.formulas(lambda row: row.insulator)
    low_spacegroup = query_archive.formulas(lambda row: row.spacegroup < 195)
    expected = len(insulators | low_spacegroup)
    # the sets overlap, so summing the subcounts would give a different answer
    assert len(insulators) + len(low_spacegroup) != expected

    result = query_nodes(
        QuerySpec.model_validate(
            {
                "entity_type": CALC,
                "group_label": query_archive.group_label,
                "filters": {
                    "logic": "OR",
                    "conditions": [
                        {"field": "insulator", "operator": "==", "value": True},
                        {"field": "spacegroup_number", "operator": "<", "value": 195},
                    ],
                },
                "count_only": True,
            }
        )
    )
    assert result["total"] == expected


def test_and_group_combines_conditions(query_archive: QueryArchive) -> None:
    """AND narrows: metals without overlapping semicore states."""
    expected = query_archive.expect(
        lambda row: not row.insulator and not row.overlapping_semicore
    )
    result = query_nodes(
        QuerySpec.model_validate(
            {
                "entity_type": CALC,
                "group_label": query_archive.group_label,
                "filters": {
                    "logic": "AND",
                    "conditions": [
                        {"field": "insulator", "operator": "==", "value": False},
                        {
                            "field": "overlapping_semicore",
                            "operator": "==",
                            "value": False,
                        },
                    ],
                },
                "count_only": True,
            }
        )
    )
    assert result["total"] == expected


def test_group_scoping_excludes_outsiders(query_archive: QueryArchive) -> None:
    """A calculation outside the group is not returned."""
    result = query_nodes(
        QuerySpec.model_validate(
            {
                "entity_type": CALC,
                "group_label": query_archive.group_label,
                "project": ["pk", "formula_hill"],
                "limit": MAX_LIMIT,
            }
        )
    )
    formulas = {record["formula_hill"] for record in result["records"]}
    assert result["total"] == query_archive.expect()
    assert formulas == query_archive.formulas(lambda _: True)
    assert "Ag" not in formulas  # stored, but not a member of the group


def test_sort_uses_numeric_order_not_text_order(query_archive: QueryArchive) -> None:
    """cast='f' ranks 10.0 above 9.0, which a text sort would not."""
    ranked = sorted(
        (row for row in query_archive.rows if row.in_group),
        key=lambda row: -row.bandgap,
    )[:2]
    result = query_nodes(
        QuerySpec.model_validate(
            {
                "entity_type": CALC,
                "group_label": query_archive.group_label,
                "sort": [{"field": "pw_bandgap", "direction": "desc", "cast": "f"}],
                "project": ["formula_hill", "pw_bandgap"],
                "limit": 2,
            }
        )
    )
    assert [record["formula_hill"] for record in result["records"]] == [
        row.formula for row in ranked
    ]


def test_records_carry_their_extras(query_archive: QueryArchive) -> None:
    """The default projection returns the extras dict, minus AiiDA's internals."""
    result = query_nodes(
        QuerySpec.model_validate(
            {
                "entity_type": CALC,
                "group_label": query_archive.group_label,
                "filters": {"field": "formula_hill", "operator": "==", "value": "Si"},
                "limit": 1,
            }
        )
    )
    record = result["records"][0]
    assert record["extras"] == {
        "formula_hill": "Si",
        "spacegroup_number": 227,
        "insulator": True,
        "pw_bandgap": 1.12,
        "overlapping_semicore": False,
    }
    assert isinstance(record["ctime"], str)  # serialised, not a datetime


def test_count_only_returns_no_records(query_archive: QueryArchive) -> None:
    """count_only reports the total without fetching rows."""
    spec = QuerySpec.model_validate(
        {
            "entity_type": CALC,
            "group_label": query_archive.group_label,
            "count_only": True,
        }
    )
    assert query_nodes(spec) == {"total": query_archive.expect(), "records": []}


def test_total_is_the_full_count_not_the_page(query_archive: QueryArchive) -> None:
    """`total` counts every match; `records` is capped by `limit`."""
    result = query_nodes(
        QuerySpec.model_validate(
            {"entity_type": CALC, "group_label": query_archive.group_label, "limit": 2}
        )
    )
    assert result["total"] == query_archive.expect()
    assert len(result["records"]) == 2


def test_provenance_join_reaches_upstream_structures(
    query_archive: QueryArchive,
) -> None:
    """Structures upstream of the failed calculations.

    The query a single-entity spec cannot express at any depth. The structure is
    two hops up (via the upstream calc and its remote folder), so this needs the
    transitive `with_descendants`, not a direct link.
    """
    failed = query_archive.formulas(lambda row: row.exit_status != 0)
    result = query_nodes(
        QuerySpec.model_validate(
            {
                "path": [
                    {"entity_type": "group", "tag": "g"},
                    {
                        "entity_type": CALC,
                        "tag": "calc",
                        "joining_keyword": "with_group",
                        "joining_value": "g",
                    },
                    {
                        "entity_type": "StructureData",
                        "tag": "st",
                        "joining_keyword": "with_descendants",
                        "joining_value": "calc",
                    },
                ],
                "filters": {
                    "g": {
                        "field": "label",
                        "operator": "==",
                        "value": query_archive.group_label,
                    },
                    "calc": {
                        "field": "attributes.exit_status",
                        "operator": "!==",
                        "value": 0,
                    },
                },
                "project": {"st": ["pk"], "calc": ["formula_hill"]},
                "limit": MAX_LIMIT,
            }
        )
    )
    assert result["total"] == len(failed)
    assert {record["calc.formula_hill"] for record in result["records"]} == failed
    assert {record["st.pk"] for record in result["records"]} == {
        query_archive.structures[formula].pk for formula in failed
    }


def test_direct_join_does_not_reach_a_transitive_ancestor(
    query_archive: QueryArchive,
) -> None:
    """`with_outgoing` is one hop; the structure is further up.

    Pins the difference between the two joins, which is easy to confuse and
    silently returns nothing when confused.
    """
    result = query_nodes(
        QuerySpec.model_validate(
            {
                "path": [
                    {"entity_type": CALC, "tag": "calc"},
                    {
                        "entity_type": "StructureData",
                        "tag": "st",
                        "joining_keyword": "with_outgoing",
                        "joining_value": "calc",
                    },
                ],
                "filters": {
                    "calc": {
                        "field": "pk",
                        "operator": "==",
                        "value": query_archive.calculations["Si"].pk,
                    }
                },
                "count_only": True,
            }
        )
    )
    assert result["total"] == 0


def test_edge_filters_narrow_to_a_specific_link(query_archive: QueryArchive) -> None:
    """edge_filters distinguishes a specific link from any link of that kind.

    Every archive row's upstream calc creates exactly one RemoteData, labelled
    'remote_folder' -- entity filters alone cannot express "only that link",
    since both sides of the join are the same for any CalcJob->RemoteData edge.
    `with_incoming=upstream` on the RemoteData means "upstream sends to this
    entity" -- i.e. upstream creates the remote folder.

    Scoped to the archive's own upstream pks (derived via the real incoming
    links, not assumed) rather than the bare entity type: the shared session
    profile may carry other CalcJob->RemoteData 'remote_folder' edges from
    unrelated fixtures (e.g. a real ArithmeticAddCalculation run also creates
    one), which a global count would wrongly fold in.
    """
    upstream_pks = [
        calc.base.links.get_incoming()
        .all()[0]
        .node.base.links.get_incoming()
        .all()[0]
        .node.pk
        for calc in query_archive.calculations.values()
    ]

    def total(edge_label: str) -> int:
        return query_nodes(
            QuerySpec.model_validate(
                {
                    "path": [
                        {"entity_type": CALC, "tag": "upstream"},
                        {
                            "entity_type": "RemoteData",
                            "tag": "remote",
                            "joining_keyword": "with_incoming",
                            "joining_value": "upstream",
                            "edge_filters": {
                                "field": "label",
                                "operator": "==",
                                "value": edge_label,
                            },
                        },
                    ],
                    "filters": {
                        "upstream": {
                            "field": "pk",
                            "operator": "in",
                            "value": upstream_pks,
                        }
                    },
                    "count_only": True,
                }
            )
        )["total"]

    assert total("remote_folder") == len(upstream_pks)
    assert total("no-such-label") == 0


def test_multi_tag_projection_keeps_values_with_their_own_tag(
    query_archive: QueryArchive,
) -> None:
    """Columns are labelled by path order, not by the order tags were listed.

    QueryBuilder emits columns in path order, so a `project` dict written in a
    different order must not shift every value onto the wrong key.
    """
    result = query_nodes(
        QuerySpec.model_validate(
            {
                "path": [
                    {"entity_type": CALC, "tag": "calc"},
                    {
                        "entity_type": "StructureData",
                        "tag": "st",
                        "joining_keyword": "with_descendants",
                        "joining_value": "calc",
                    },
                ],
                "filters": {
                    "calc": {
                        "field": "pk",
                        "operator": "==",
                        "value": query_archive.calculations["Si"].pk,
                    }
                },
                # deliberately the reverse of the path order
                "project": {"st": ["pk"], "calc": ["pk", "formula_hill"]},
            }
        )
    )
    record = result["records"][0]
    assert record["calc.pk"] == query_archive.calculations["Si"].pk
    assert record["calc.formula_hill"] == "Si"
    assert record["st.pk"] == query_archive.structures["Si"].pk


@pytest.mark.parametrize(
    "operator,value",
    [
        ("==", "Si"),
        ("!==", "Si"),
        ("like", "S%"),
        ("in", ["Si", "Cu"]),
        (">", "A"),
        (">=", "A"),
        ("<", "Z"),
        ("<=", "Z"),
    ],
)
def test_every_operator_executes(
    query_archive: QueryArchive, operator: str, value: t.Any
) -> None:
    """Each advertised operator reaches the database without error.

    aiida-core has no declarative operator list, so this pins the Literal against
    the backend: '!=' shipped once and is not valid AiiDA syntax.
    """
    spec = QuerySpec.model_validate(
        {
            "entity_type": CALC,
            "group_label": query_archive.group_label,
            "filters": {"field": "formula_hill", "operator": operator, "value": value},
            "count_only": True,
        }
    )
    assert isinstance(query_nodes(spec)["total"], int)


def test_process_plugin_entity_type_excludes_other_calcjobs(
    add_calc: orm.CalcJobNode, query_archive: QueryArchive
) -> None:
    """`ArithmeticAddCalculation` must not also match the archive's plain,
    process-less `CalcJobNode`s -- both share `node_type`, and only the
    process_type filter tells them apart."""
    plugin_total = query_nodes(
        QuerySpec.model_validate(
            {"entity_type": "ArithmeticAddCalculation", "count_only": True}
        )
    )["total"]
    calcjob_total = query_nodes(
        QuerySpec.model_validate({"entity_type": CALC, "count_only": True})
    )["total"]
    assert 0 < plugin_total < calcjob_total


def test_process_plugin_entity_type_by_entry_point_name(
    add_calc: orm.CalcJobNode,
) -> None:
    """The registered entry-point name works as `entity_type` too, not just
    the class name."""
    result = query_nodes(
        QuerySpec.model_validate(
            {
                "entity_type": "core.arithmetic.add",
                "filters": {"field": "pk", "operator": "==", "value": add_calc.pk},
                "count_only": True,
            }
        )
    )
    assert result["total"] == 1


def test_workchain_plugin_entity_type(
    multiply_add_workchain: orm.WorkChainNode,
) -> None:
    """A WorkChain plugin resolves the same way a CalcJob plugin does."""
    result = query_nodes(
        QuerySpec.model_validate(
            {
                "entity_type": "MultiplyAddWorkChain",
                "filters": {
                    "field": "pk",
                    "operator": "==",
                    "value": multiply_add_workchain.pk,
                },
                "count_only": True,
            }
        )
    )
    assert result["total"] == 1
