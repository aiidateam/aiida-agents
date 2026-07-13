"""Generic node-query tool: structured filter spec -> AiiDA QueryBuilder.

Prototype for the "generic QueryBuilder tool" direction discussed as an
alternative to adding more hardcoded, single-purpose tools like
`query_nodes_by_extras`.

Design goals, informed by testing `query_nodes_by_extras` against
Natasha's archive:

1. Native AND/OR support. `query_nodes_by_extras` has no OR — when
   asked an OR question, the agent worked around it with three separate
   tool calls (count A, count B, count A-and-B) and then reported one
   sub-count as if it were the combined answer (off by ~4,500 in
   testing). This tool instead translates a nested filter spec directly
   into AiiDA QueryBuilder's own `{"and": [...]}` / `{"or": [...]}`
   filter syntax, so the database evaluates the logic correctly in a
   single query. The model never has to do the boolean arithmetic
   itself.

2. Native sorting. `query_nodes_by_extras` had no sort — the agent was
   sorting whatever partial record set it happened to fetch, client-side,
   with no guarantee the true top-N were even in that set. This tool
   sorts via QueryBuilder's `order_by` before applying `limit`, so
   ranking queries are correct regardless of how many total matches
   exist.

3. Count queries never fetch full records. `count_only=True` calls
   `qb.count()` directly and never projects/serializes rows — avoiding
   the token-overflow issue Julian flagged with large result sets.

4. Bounded, not arbitrary. The model supplies a structured spec
   (fields, operators, logic groups, sort, limit), not raw QueryBuilder
   Python. `limit` is hard-capped. This is a middle ground between the
   old hardcoded-per-pattern tools and letting the model write and
   execute arbitrary QueryBuilder code itself.

Validate standalone behavior with `dev/query_builder/verify_query_nodes.py` (direct function calls, no LLM involved).
"""

from __future__ import annotations

import datetime
import logging
import typing as t

from pydantic import BaseModel, Field

from aiida import orm
from aiida.common.exceptions import NotExistent

logger = logging.getLogger(__name__)

MAX_LIMIT = 50
DEFAULT_LIMIT = 5
DEFAULT_PROJECT = ["pk", "uuid", "node_type", "ctime", "extras"]

# Fields that live on the node itself, not inside `extras`. Anything not
# in this set is assumed to be an extras key and gets the "extras."
# prefix added automatically, so the model doesn't need to know AiiDA's
# internal QueryBuilder field-naming convention.
NATIVE_NODE_FIELDS = {
    "pk",
    "id",
    "uuid",
    "node_type",
    "ctime",
    "mtime",
    "label",
    "description",
}


FilterOpLiteral = t.Literal["==", "!==", ">", ">=", "<", "<=", "in", "like"]


class FilterOperator(str):
    """Comparison operators, matching AiiDA QueryBuilder's own filter syntax."""

    EQ: t.Literal["=="] = "=="
    NEQ: t.Literal["!=="] = "!=="
    GT: t.Literal[">"] = ">"
    GTE: t.Literal[">="] = ">="
    LT: t.Literal["<"] = "<"
    LTE: t.Literal["<="] = "<="
    IN: t.Literal["in"] = "in"
    LIKE: t.Literal["like"] = "like"


class FieldFilter(BaseModel):
    """A single field comparison, e.g. spacegroup_number >= 195."""

    field: str = Field(
        description=(
            "Field name to filter on. Extras fields (e.g. 'spacegroup_number', "
            "'insulator') are given as-is; the 'extras.' prefix is added "
            "automatically. Native node fields (pk, id, uuid, node_type, ctime, "
            "mtime, label, description) are used as-is."
        )
    )
    operator: FilterOpLiteral = Field(
        default=FilterOperator.EQ,
        description="One of: ==, !==, >, >=, <, <=, in, like",
    )
    value: t.Any = Field(description="Value to compare against.")


class FilterGroup(BaseModel):
    """A group of conditions combined with AND/OR, allowing nesting."""

    logic: t.Literal["AND", "OR"] = "AND"
    conditions: list[t.Union["FieldFilter", "FilterGroup"]] = Field(
        default_factory=list,
        description=(
            "List of FieldFilter or nested FilterGroup objects, combined "
            "using `logic`. Nesting groups allows expressions like "
            "'(A AND B) OR C'."
        ),
    )


FilterGroup.model_rebuild()


class SortSpec(BaseModel):
    field: str = Field(description="Field to sort by (extras or native node field).")
    direction: t.Literal["asc", "desc"] = "asc"
    cast: t.Literal["f", "i", "t", "b", "d"] | None = Field(
        default=None,
        description=(
            "Required when sorting by an extras field, since extras are "
            "stored as JSON and AiiDA needs to know how to compare values: "
            "'f' float, 'i' integer, 't' text, 'b' boolean, 'd' date. "
            "Not needed for native node fields."
        ),
    )


class QueryNodesSpec(BaseModel):
    """Structured query spec for `query_nodes`."""

    group_label: str | None = Field(
        default=None,
        description="Restrict the search to nodes in this group, if given.",
    )
    filters: FieldFilter | FilterGroup | None = Field(
        default=None,
        description=(
            "Filter conditions: a single FieldFilter, or a FilterGroup "
            "combining multiple conditions with AND/OR. Omit to match all "
            "nodes (within the group, if given)."
        ),
    )
    sort: list[SortSpec] = Field(
        default_factory=list, description="Sort order, applied in list order."
    )
    project: list[str] = Field(
        default_factory=lambda: list(DEFAULT_PROJECT),
        description="Fields to return per record (defaults to pk, uuid, node_type, ctime, extras). Ignored when count_only is True.",
    )
    limit: int = Field(
        default=DEFAULT_LIMIT,
        ge=1,
        le=MAX_LIMIT,
        description=f"Max records to return (must be between 1 and {MAX_LIMIT} to avoid token overflow). Ignored when count_only is True.",
    )
    count_only: bool = Field(
        default=False,
        description="If True, return only the total matching count (never fetches full records).",
    )


def _qualify_field(field: str) -> str:
    """Add the 'extras.' prefix unless the field is a native node field."""
    if (
        field in NATIVE_NODE_FIELDS
        or field.startswith("extras.")
        or field.startswith("attributes.")
    ):
        return field
    return f"extras.{field}"


def _translate_filter(node: FieldFilter | FilterGroup) -> dict[str, t.Any]:
    """Recursively translate our filter spec into AiiDA QueryBuilder's
    native nested filter dict syntax, e.g.:

        {"and": [{"extras.insulator": True}, {"or": [...]}]}
    """
    if isinstance(node, FieldFilter):
        if node.operator == FilterOperator.IN and not isinstance(
            node.value, (list, tuple, set)
        ):
            raise ValueError(
                f"Operator 'in' requires a list of values for field '{node.field}', "
                f"got {type(node.value).__name__}: {node.value!r}. "
                "Please provide a list, e.g. [value1, value2]."
            )
        field = _qualify_field(node.field)
        if node.operator == FilterOperator.EQ:
            return {field: node.value}
        return {field: {node.operator: node.value}}

    # FilterGroup
    key = "and" if node.logic == "AND" else "or"
    return {key: [_translate_filter(c) for c in node.conditions]}


def _serialize_value(val: t.Any) -> t.Any:
    """Serialize values like raw datetimes and filter internal _aiida_* keys."""
    if isinstance(val, (datetime.datetime, datetime.date)):
        return str(val)
    if isinstance(val, dict):
        return {
            k: _serialize_value(v)
            for k, v in val.items()
            if not str(k).startswith("_aiida_")
        }
    if isinstance(val, list):
        return [_serialize_value(v) for v in val]
    return val


def query_nodes(spec: QueryNodesSpec) -> dict[str, t.Any]:
    """Query AiiDA nodes using a structured filter spec, with native
    AND/OR logic, sorting, and group scoping.

    Use this for any node/dataset query — filtering by extras or native
    fields, combining conditions with AND/OR (including nested groups),
    ranking/sorting by a field, and optionally scoping to a group.
    Prefer `count_only=True` for "how many" questions; it never fetches
    full records.

    Args:
        spec: The structured query spec (see QueryNodesSpec).

    Returns:
        A dictionary `{"total": int, "records": list[dict]}` containing the
        total matching count and up to `limit` records (`records` is empty when
        count_only is True).
    """
    logger.debug("query_nodes(spec=%r)", spec)

    qb = orm.QueryBuilder()

    node_kwargs: dict[str, t.Any] = {}
    if spec.group_label is not None:
        try:
            orm.Group.collection.get(label=spec.group_label)
        except NotExistent as exc:
            raise ValueError(
                f"Group with label '{spec.group_label}' does not exist. "
                "Use query_nodes or search tools to check available groups."
            ) from exc
        qb.append(orm.Group, filters={"label": spec.group_label}, tag="group")
        node_kwargs["with_group"] = "group"

    filters = _translate_filter(spec.filters) if spec.filters is not None else {}

    if spec.count_only:
        qb.append(orm.Node, tag="node", filters=filters, **node_kwargs)
        count = qb.count()
        logger.debug("query_nodes: returned count %d (count_only=True)", count)
        return {"total": count, "records": []}

    project_fields = [_qualify_field(p) for p in spec.project]
    qb.append(
        orm.Node, tag="node", filters=filters, project=project_fields, **node_kwargs
    )
    total = qb.count()

    for sort_spec in spec.sort:
        field = _qualify_field(sort_spec.field)
        if field.startswith("extras.") and sort_spec.cast is None:
            raise ValueError(
                f"Sorting by extras field '{sort_spec.field}' requires a 'cast' "
                "(one of: f, i, t, b, d) since extras are stored as JSON."
            )
        order_value: str | dict[str, t.Any] = sort_spec.direction
        if sort_spec.cast is not None:
            order_value = {"order": sort_spec.direction, "cast": sort_spec.cast}
        qb.order_by({"node": {field: order_value}})

    qb.limit(spec.limit)

    records = [
        {k: _serialize_value(v) for k, v in zip(spec.project, row, strict=True)}
        for row in qb.iterall()
    ]
    logger.debug("query_nodes: returned %d records (total %d)", len(records), total)
    return {"total": total, "records": records}
