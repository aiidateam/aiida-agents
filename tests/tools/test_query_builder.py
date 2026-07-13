"""Tests for ``aiida_agents.tools.query_builder``.

Covers filter translation (single, AND group, nested OR), field qualification,
schema validation (operators, limits, IN operator), sorting rules, group
scoping, and the structured return shape.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from aiida import orm

from aiida_agents.tools.query_builder import (
    FieldFilter,
    FilterGroup,
    QueryNodesSpec,
    SortSpec,
    _qualify_field,
    _translate_filter,
    query_nodes,
)


def test_qualify_field() -> None:
    """Native fields and explicit prefixes are preserved; bare extras are prefixed."""
    assert _qualify_field("pk") == "pk"
    assert _qualify_field("uuid") == "uuid"
    assert _qualify_field("ctime") == "ctime"
    assert _qualify_field("node_type") == "node_type"
    assert _qualify_field("extras.insulator") == "extras.insulator"
    assert _qualify_field("attributes.process_state") == "attributes.process_state"
    assert _qualify_field("spacegroup_number") == "extras.spacegroup_number"
    assert _qualify_field("insulator") == "extras.insulator"


def test_translate_filter_single() -> None:
    """A single FieldFilter translates directly into QueryBuilder filter syntax."""
    eq_filter = FieldFilter(field="insulator", operator="==", value=True)
    assert _translate_filter(eq_filter) == {"extras.insulator": True}

    gt_filter = FieldFilter(field="spacegroup_number", operator=">", value=100)
    assert _translate_filter(gt_filter) == {"extras.spacegroup_number": {">": 100}}

    neq_filter = FieldFilter(field="insulator", operator="!==", value=False)
    assert _translate_filter(neq_filter) == {"extras.insulator": {"!==": False}}


def test_translate_filter_and_group() -> None:
    """A FilterGroup with logic='AND' translates to an 'and' list of conditions."""
    group = FilterGroup(
        logic="AND",
        conditions=[
            FieldFilter(field="insulator", operator="==", value=True),
            FieldFilter(field="spacegroup_number", operator="<", value=195),
        ],
    )
    assert _translate_filter(group) == {
        "and": [
            {"extras.insulator": True},
            {"extras.spacegroup_number": {"<": 195}},
        ]
    }


def test_translate_filter_nested_or() -> None:
    """Nested groups like '(A AND B) OR C' translate to nested 'or'/'and' dicts."""
    nested = FilterGroup(
        logic="OR",
        conditions=[
            FilterGroup(
                logic="AND",
                conditions=[
                    FieldFilter(field="a", operator="==", value=1),
                    FieldFilter(field="b", operator="==", value=2),
                ],
            ),
            FieldFilter(field="c", operator="==", value=3),
        ],
    )
    assert _translate_filter(nested) == {
        "or": [
            {
                "and": [
                    {"extras.a": 1},
                    {"extras.b": 2},
                ]
            },
            {"extras.c": 3},
        ]
    }


def test_in_operator_validation() -> None:
    """Passing a non-list/set value to operator 'in' raises a clear ValueError."""
    bad_filter = FieldFilter(field="spacegroup_number", operator="in", value=195)
    with pytest.raises(ValueError, match="Operator 'in' requires a list of values"):
        _translate_filter(bad_filter)


def test_operator_schema_validation() -> None:
    """Invalid operators are rejected eagerly at the schema level."""
    with pytest.raises(ValidationError):
        FieldFilter(field="insulator", operator="!=", value=True)  # type: ignore[arg-type]


def test_limit_bounds() -> None:
    """Limit must be between 1 and MAX_LIMIT."""
    with pytest.raises(ValidationError):
        QueryNodesSpec(limit=0)
    with pytest.raises(ValidationError):
        QueryNodesSpec(limit=-1)
    with pytest.raises(ValidationError):
        QueryNodesSpec(limit=51)
    # Valid bounds pass
    assert QueryNodesSpec(limit=1).limit == 1
    assert QueryNodesSpec(limit=50).limit == 50


@pytest.mark.usefixtures("add_calc")
def test_missing_cast_value_error() -> None:
    """Sorting by an extras field without explicit cast raises ValueError."""
    spec = QueryNodesSpec(
        sort=[SortSpec(field="pw_bandgap", direction="desc")],
    )
    with pytest.raises(
        ValueError, match="Sorting by extras field 'pw_bandgap' requires a 'cast'"
    ):
        query_nodes(spec)


@pytest.mark.usefixtures("add_calc")
def test_count_only_vs_records() -> None:
    """count_only=True returns empty records list; False returns records plus total."""
    count_spec = QueryNodesSpec(count_only=True)
    res_count = query_nodes(count_spec)
    assert isinstance(res_count, dict)
    assert "total" in res_count
    assert "records" in res_count
    assert res_count["total"] > 0
    assert res_count["records"] == []

    records_spec = QueryNodesSpec(count_only=False, limit=2)
    res_records = query_nodes(records_spec)
    assert isinstance(res_records, dict)
    assert res_records["total"] == res_count["total"]
    assert len(res_records["records"]) <= 2
    for record in res_records["records"]:
        assert "pk" in record
        assert "uuid" in record
        assert "node_type" in record
        assert "ctime" in record
        assert isinstance(record["ctime"], str)


@pytest.mark.usefixtures("add_calc")
def test_group_scoping() -> None:
    """Scoping to a group returns only nodes in that group; missing group raises clearly."""
    orm.Group(label="test_query_group").store()
    spec_empty = QueryNodesSpec(group_label="test_query_group", count_only=True)
    assert query_nodes(spec_empty)["total"] == 0

    with pytest.raises(
        ValueError, match="Group with label 'non_existent_group' does not exist"
    ):
        query_nodes(QueryNodesSpec(group_label="non_existent_group"))


def test_nested_or_count_equals_manual_union(aiida_profile: object) -> None:
    """Verify that a nested-OR filter count matches the exact manual union of sub-queries.

    This pins the #25 bug fix where OR queries previously required manual split calls.
    """
    node1 = orm.Int(1).store()
    node1.base.extras.set_many({"flag_a": True, "flag_b": True, "flag_c": False})

    node2 = orm.Int(2).store()
    node2.base.extras.set_many({"flag_a": True, "flag_b": False, "flag_c": False})

    node3 = orm.Int(3).store()
    node3.base.extras.set_many({"flag_a": False, "flag_b": False, "flag_c": True})

    node4 = orm.Int(4).store()
    node4.base.extras.set_many({"flag_a": False, "flag_b": False, "flag_c": False})

    # Query: (flag_a == True AND flag_b == True) OR (flag_c == True)
    # Expected matches: node1 (A and B) and node3 (C) -> 2 nodes.
    spec = QueryNodesSpec(
        filters=FilterGroup(
            logic="OR",
            conditions=[
                FilterGroup(
                    logic="AND",
                    conditions=[
                        FieldFilter(field="flag_a", operator="==", value=True),
                        FieldFilter(field="flag_b", operator="==", value=True),
                    ],
                ),
                FieldFilter(field="flag_c", operator="==", value=True),
            ],
        ),
        count_only=True,
    )
    result = query_nodes(spec)
    assert result["total"] >= 2

    # Verify directly against manual QueryBuilder union of the two branches
    qb1 = orm.QueryBuilder()
    qb1.append(
        orm.Node,
        filters={"and": [{"extras.flag_a": True}, {"extras.flag_b": True}]},
        project="id",
    )
    pks1 = {row[0] for row in qb1.iterall()}

    qb2 = orm.QueryBuilder()
    qb2.append(orm.Node, filters={"extras.flag_c": True}, project="id")
    pks2 = {row[0] for row in qb2.iterall()}

    union_pks = pks1 | pks2
    assert result["total"] == len(union_pks)
    assert node1.pk in union_pks
    assert node3.pk in union_pks
