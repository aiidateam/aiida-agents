"""Generic node-query tool: a structured spec lowered to AiiDA's QueryBuilder.

The model supplies a :class:`QuerySpec` -- entities, joins, filters, sort,
projection -- rather than raw QueryBuilder code. It runs as a small pipeline::

    spec -> normalise -> validate -> lower -> execute -> serialise

``lower`` is a pure function producing the dict that ``QueryBuilder.from_dict``
already accepts (the same shape as aiida-restapi's ``QueryBuilderDict``), so the
whole translation is testable without a database, and only ``_execute`` touches
AiiDA.

Design notes:

1. Native AND/OR. A nested filter spec becomes QueryBuilder's own
   ``{"and": [...]}`` / ``{"or": [...]}``, so the database evaluates the logic
   and the model never does boolean arithmetic itself.
2. Native joins. A ``path`` of entities with join keywords expresses provenance
   queries ("structures that are inputs to a failed workchain") which a
   single-entity query cannot reach at any depth.
3. Native sorting, applied before ``limit``, so ranking is correct regardless of
   how many total matches exist.
4. Counts never fetch records: ``count_only`` returns the total alone.
5. Bounded, not arbitrary: a structured spec, a hard-capped ``limit``, and
   validation with repair hints before the query runs.

The closed sets (join keywords, entity types, fields) are derived from
aiida-core rather than hand-maintained; only the operator list is hand-written,
because aiida-core resolves operators through branch chains with no declarative
source. It is pinned by a test that executes each one.
"""

from __future__ import annotations

import datetime
import difflib
import logging
import typing as t
from functools import lru_cache

from aiida import orm
from aiida.orm.implementation.querybuilder import EntityRelationships
from aiida.plugins.entry_point import get_entry_point_names, load_entry_point
from pydantic import BaseModel, Field, model_validator

from ._types import QueryResult

logger = logging.getLogger(__name__)

MAX_LIMIT = 50
DEFAULT_LIMIT = 5
DEFAULT_PROJECT = ["pk", "uuid", "node_type", "ctime", "extras"]
NODE_TAG = "node"
GROUP_TAG = "_group"

#: Operators accepted in a filter. AiiDA's "not equal" is ``!==``, not ``!=``.
FilterOp = t.Literal["==", "!==", ">", ">=", "<", "<=", "in", "like"]

#: Join keywords, mirroring ``EntityRelationships``; validated per entity below.
JoinKeyword = t.Literal[
    "with_ancestors",
    "with_authinfo",
    "with_comment",
    "with_computer",
    "with_descendants",
    "with_group",
    "with_incoming",
    "with_log",
    "with_node",
    "with_outgoing",
    "with_user",
]

# Containers and columns addressed as-is. Any other bare name is taken to be an
# extras key and gets the ``extras.`` prefix, so the model does not need to know
# AiiDA's field-naming convention.
PASSTHROUGH_FIELDS = frozenset(
    {
        "pk",
        "id",
        "uuid",
        "node_type",
        "process_type",
        "ctime",
        "mtime",
        "label",
        "description",
        "extras",
        "attributes",
        "type_string",
        "time",
    }
)


class QueryValidationError(ValueError):
    """A spec that cannot be lowered, reported before the query runs."""


@lru_cache(maxsize=1)
def _entity_index() -> dict[str, str]:
    """Map entity aliases to canonical ``node_type`` strings.

    ``class_node_type`` already encodes subtree matching (``data.Data.`` matches
    every ``Data`` subclass), so abstract levels need no ``like`` patterns.
    """
    index: dict[str, str] = {"node": "", "group": "group.core"}
    for entry_point_group in ("aiida.data", "aiida.node"):
        for name in get_entry_point_names(entry_point_group):
            try:
                cls = load_entry_point(entry_point_group, name)
            except Exception:  # noqa: BLE001 - a broken plugin must not break queries
                logger.debug("entity_index: cannot load %s:%s", entry_point_group, name)
                continue
            node_type = getattr(cls, "class_node_type", None)
            if node_type is not None:
                index[name.lower()] = node_type
                index[cls.__name__.lower()] = node_type
    for cls in (orm.Data, orm.ProcessNode, orm.CalculationNode, orm.WorkflowNode):
        index[cls.__name__.lower()] = cls.class_node_type
    index["data"] = orm.Data.class_node_type
    index["process"] = orm.ProcessNode.class_node_type
    index["calculation"] = orm.CalculationNode.class_node_type
    index["workflow"] = orm.WorkflowNode.class_node_type
    return index


def _suggest(name: str, valid: t.Iterable[str]) -> str:
    """A "did you mean" hint, or a sample of what is valid."""
    close = difflib.get_close_matches(name, sorted(valid), n=3, cutoff=0.5)
    if close:
        return f" Did you mean: {', '.join(close)}?"
    return f" Valid values include: {', '.join(sorted(valid)[:8])}."


class FieldFilter(BaseModel):
    """A single comparison, e.g. ``spacegroup_number >= 195``."""

    field: str = Field(
        description=(
            "Field to filter on. Extras keys (e.g. 'spacegroup_number') are given "
            "bare and get the 'extras.' prefix automatically. Node columns (pk, "
            "uuid, node_type, ctime, mtime, label, description) and explicit "
            "'attributes.x' / 'extras.x' paths are used as given."
        )
    )
    operator: FilterOp = Field(
        default="==",
        description="One of: ==, !==, >, >=, <, <=, in, like. Not-equal is '!=='.",
    )
    value: t.Any = Field(description="Value to compare against; a list for 'in'.")


class FilterGroup(BaseModel):
    """Conditions combined with AND/OR, nestable to any depth."""

    logic: t.Literal["AND", "OR"] = "AND"
    conditions: list[FieldFilter | FilterGroup] = Field(
        default_factory=list,
        description=(
            "FieldFilter or nested FilterGroup objects combined with `logic`. "
            "Nesting expresses e.g. '(A AND B) OR C'."
        ),
    )


FilterGroup.model_rebuild()

FilterTree = FieldFilter | FilterGroup


class PathItem(BaseModel):
    """One entity in the query path, optionally joined to an earlier one."""

    entity_type: str = Field(
        default="node",
        description=(
            "Entity to query: an alias ('StructureData', 'process', 'data', "
            "'group'), or a full node type ('data.core.structure.StructureData.'). "
            "Abstract levels match their whole subtree."
        ),
    )
    tag: str = Field(
        description="Name for this entity, referenced by filters/project/sort."
    )
    joining_keyword: JoinKeyword | None = Field(
        default=None,
        description=(
            "How this entity relates to `joining_value`. For provenance: "
            "'with_incoming' (has inputs from), 'with_outgoing' (is an input to), "
            "'with_ancestors' / 'with_descendants' (transitive). Omit on the first item."
        ),
    )
    joining_value: str | None = Field(
        default=None, description="The tag of the earlier entity this one joins to."
    )
    outerjoin: bool = Field(default=False, description="Keep rows with no match.")


class SortSpec(BaseModel):
    """Sort order, applied server-side before `limit`."""

    field: str = Field(description="Field to sort by.")
    direction: t.Literal["asc", "desc"] = "asc"
    cast: t.Literal["f", "i", "t", "b", "d"] | None = Field(
        default=None,
        description=(
            "Required when sorting an extras/attributes field, since they are "
            "stored as JSON: 'f' float, 'i' int, 't' text, 'b' bool, 'd' date."
        ),
    )
    tag: str | None = Field(
        default=None, description="Which path entity to sort; defaults to the last."
    )


class QuerySpec(BaseModel):
    """A query: canonically a `path` plus per-tag filters.

    A flat single-entity form is accepted as shorthand and rewritten into that
    canonical shape, so simple questions stay simple.

    Flat (most queries)::

        {"entity_type": "StructureData", "group_label": "my/group",
         "filters": {"field": "insulator", "operator": "==", "value": false},
         "count_only": true}

    Path (provenance)::

        {"path": [{"entity_type": "WorkChainNode", "tag": "wc"},
                  {"entity_type": "StructureData", "tag": "st",
                   "joining_keyword": "with_outgoing", "joining_value": "wc"}],
         "filters": {"wc": {"field": "attributes.exit_status", "operator": "!==", "value": 0}},
         "project": {"st": ["pk", "formula_hill"]}}
    """

    path: list[PathItem] = Field(
        default_factory=list,
        description=(
            "Entities to query, in order; later items join to earlier ones by tag. "
            "Omit and give 'entity_type' instead for a single-entity query."
        ),
    )
    filters: dict[str, FilterTree] = Field(
        default_factory=dict,
        description=(
            "Filters per path tag. For a single-entity query, a bare filter may be "
            "given and is applied to that entity."
        ),
    )
    project: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "Fields to return per tag. For a single-entity query a bare list may be "
            f"given. Defaults to {DEFAULT_PROJECT}. Ignored when count_only is True."
        ),
    )
    sort: list[SortSpec] = Field(default_factory=list, description="Sort order.")
    limit: int = Field(
        default=DEFAULT_LIMIT,
        ge=1,
        le=MAX_LIMIT,
        description=f"Max records to return (1-{MAX_LIMIT}). Ignored when count_only is True.",
    )
    offset: int = Field(default=0, ge=0, description="Records to skip.")
    count_only: bool = Field(
        default=False,
        description="Return only the total; never fetches records. Use for 'how many'.",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalise(cls, data: t.Any) -> t.Any:
        """Rewrite the flat single-entity shorthand into a canonical path.

        Keeps simple queries simple for the model while leaving exactly one
        internal shape for everything downstream.
        """
        if not isinstance(data, dict):
            return data
        data = dict(data)
        entity_type = data.pop("entity_type", None)
        group_label = data.pop("group_label", None)

        # An explicit null is how a model most often says "no filters"; treat it
        # as absent rather than rejecting the whole spec over it.
        for key in ("path", "filters", "project", "sort"):
            if data.get(key, ...) is None:
                del data[key]

        if not data.get("path"):
            node: dict[str, t.Any] = {
                "entity_type": entity_type or "node",
                "tag": NODE_TAG,
            }
            path: list[dict[str, t.Any]] = []
            if group_label is not None:
                path.append({"entity_type": "group", "tag": GROUP_TAG})
                node["joining_keyword"] = "with_group"
                node["joining_value"] = GROUP_TAG
            path.append(node)
            data["path"] = path

        if _is_filter_tree(data.get("filters")):
            data["filters"] = {NODE_TAG: data["filters"]}
        if isinstance(data.get("project"), list):
            data["project"] = {NODE_TAG: data["project"]}

        if group_label is not None:
            filters = data.setdefault("filters", {})
            if isinstance(filters, dict):
                filters[GROUP_TAG] = {
                    "field": "label",
                    "operator": "==",
                    "value": group_label,
                }
        return data


def _is_filter_tree(value: t.Any) -> bool:
    """True for a bare filter tree, as opposed to a ``{tag: tree}`` mapping.

    Accepts both the JSON form a model sends and the model objects a Python
    caller is likely to build.
    """
    if isinstance(value, (FieldFilter, FilterGroup)):
        return True
    return isinstance(value, dict) and ("field" in value or "logic" in value)


def _orm_base(entity_type: str) -> str:
    """The QueryBuilder ``orm_base`` for a canonical entity type."""
    if entity_type.startswith("group"):
        return "group"
    if entity_type.startswith("computer"):
        return "computer"
    if entity_type.startswith("user"):
        return "user"
    return "node"


def _validate_spec(spec: QuerySpec) -> None:
    """Check a spec against aiida-core's own metadata, before any query runs.

    :param spec: the spec to check.
    :raises QueryValidationError: if an entity type, join or tag is unknown; the
        message names the offending value and what would be valid instead.
    """
    index = _entity_index()
    declared: set[str] = set()

    if not spec.path:
        msg = "A query needs at least one entity: give 'entity_type' or 'path'."
        raise QueryValidationError(msg)

    tags = [item.tag for item in spec.path]
    duplicates = {tag for tag in tags if tags.count(tag) > 1}
    if duplicates:
        msg = f"Duplicate path tags: {', '.join(sorted(duplicates))}. Tags must be unique."
        raise QueryValidationError(msg)

    for position, item in enumerate(spec.path):
        if item.entity_type.lower() not in index and not item.entity_type.endswith("."):
            msg = (
                f"Unknown entity_type {item.entity_type!r} for tag {item.tag!r}."
                + _suggest(item.entity_type, index)
            )
            raise QueryValidationError(msg)

        if position == 0 and item.joining_keyword is not None:
            msg = (
                f"The first path entity ({item.tag!r}) cannot join to anything; "
                "remove its joining_keyword."
            )
            raise QueryValidationError(msg)
        if position > 0 and item.joining_keyword is None:
            msg = (
                f"Path entity {item.tag!r} needs a joining_keyword saying how it "
                f"relates to an earlier entity ({', '.join(sorted(declared))})."
            )
            raise QueryValidationError(msg)

        if item.joining_keyword is not None:
            canonical = index.get(item.entity_type.lower(), item.entity_type)
            orm_base = _orm_base(canonical)
            legal = EntityRelationships[orm_base]
            if item.joining_keyword not in legal:
                msg = (
                    f"{item.joining_keyword!r} is not a valid join from {orm_base} "
                    f"(tag {item.tag!r}). Valid here: {', '.join(sorted(legal))}."
                )
                raise QueryValidationError(msg)
            if item.joining_value not in declared:
                msg = (
                    f"joining_value {item.joining_value!r} on tag {item.tag!r} does not "
                    f"name an earlier entity. Declared so far: "
                    f"{', '.join(sorted(declared)) or '(none)'}."
                )
                raise QueryValidationError(msg)
        declared.add(item.tag)

    referenced = (
        set(spec.filters)
        | set(spec.project)
        | {sort.tag for sort in spec.sort if sort.tag is not None}
    )
    for tag in sorted(referenced - declared):
        msg = f"Unknown tag {tag!r}." + _suggest(tag, declared)
        raise QueryValidationError(msg)


def _qualify_field(field: str) -> str:
    """Prefix a bare extras key with ``extras.``; leave columns and containers alone."""
    if field in PASSTHROUGH_FIELDS or field.startswith(("extras.", "attributes.")):
        return field
    return f"extras.{field}"


def _lower_filter(node: FilterTree) -> dict[str, t.Any]:
    """Translate a filter tree into QueryBuilder's nested filter dict."""
    if isinstance(node, FieldFilter):
        if node.operator == "in" and not isinstance(node.value, (list, tuple, set)):
            msg = (
                f"Operator 'in' needs a list of values for field {node.field!r}, got "
                f"{type(node.value).__name__}: {node.value!r}. Example: [1, 2, 3]."
            )
            raise QueryValidationError(msg)
        field = _qualify_field(node.field)
        if node.operator == "==":
            return {field: node.value}
        return {field: {node.operator: node.value}}
    key = "and" if node.logic == "AND" else "or"
    return {key: [_lower_filter(condition) for condition in node.conditions]}


def _lower(spec: QuerySpec) -> dict[str, t.Any]:
    """Translate a validated spec into a ``QueryBuilder.from_dict`` dict.

    Pure: touches no ORM class, no QueryBuilder and no database, so the whole
    translation can be tested by comparing dicts.

    :param spec: a spec that has passed :func:`_validate_spec`.
    :return: the QueryBuilder dict (also aiida-restapi's ``QueryBuilderDict`` shape).
    :raises QueryValidationError: if a filter or sort cannot be expressed.
    """
    index = _entity_index()
    path: list[dict[str, t.Any]] = []
    for item in spec.path:
        canonical = index.get(item.entity_type.lower(), item.entity_type)
        entry: dict[str, t.Any] = {
            "entity_type": canonical,
            "orm_base": _orm_base(canonical),
            "tag": item.tag,
            "outerjoin": item.outerjoin,
        }
        if item.joining_keyword is not None:
            entry["joining_keyword"] = item.joining_keyword
            entry["joining_value"] = item.joining_value
        path.append(entry)

    lowered: dict[str, t.Any] = {"path": path}

    if spec.filters:
        lowered["filters"] = {
            tag: _lower_filter(tree) for tag, tree in spec.filters.items()
        }

    if not spec.count_only:
        project = spec.project or {spec.path[-1].tag: list(DEFAULT_PROJECT)}
        lowered["project"] = {
            tag: [_qualify_field(field) for field in fields]
            for tag, fields in project.items()
        }

    if spec.sort:
        order_by: list[dict[str, t.Any]] = []
        for sort in spec.sort:
            field = _qualify_field(sort.field)
            if field.startswith(("extras.", "attributes.")) and sort.cast is None:
                msg = (
                    f"Sorting by {sort.field!r} requires a 'cast' (one of: f, i, t, b, d) "
                    "because extras and attributes are stored as JSON."
                )
                raise QueryValidationError(msg)
            value: str | dict[str, t.Any] = (
                {"order": sort.direction, "cast": sort.cast}
                if sort.cast is not None
                else sort.direction
            )
            order_by.append({sort.tag or spec.path[-1].tag: {field: value}})
        lowered["order_by"] = order_by

    return lowered


def _serialize_value(value: t.Any) -> t.Any:
    """Make a projected value JSON-safe and drop AiiDA's internal extras keys."""
    if isinstance(value, (datetime.datetime, datetime.date)):
        return str(value)
    if isinstance(value, dict):
        return {
            key: _serialize_value(item)
            for key, item in value.items()
            if not str(key).startswith("_aiida_")
        }
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    return value


def _record_keys(lowered: dict[str, t.Any]) -> list[str]:
    """Output keys for the projected columns, in QueryBuilder's row order.

    Reported under the name that identifies the value, not the storage path:
    ``extras.pw_bandgap`` and ``attributes.exit_status`` come back as
    ``pw_bandgap`` and ``exit_status``. A tag prefix is added only when the query
    spans entities, so single-entity results stay flat.

    Walks ``path`` rather than ``project``: QueryBuilder emits columns in path
    order, which is not necessarily the order the caller listed the tags in.
    """
    project: dict[str, list[str]] = lowered.get("project", {})
    multi = len(project) > 1
    keys: list[str] = []
    for item in lowered["path"]:
        tag = item["tag"]
        for field in project.get(tag, []):
            bare = (
                field.split(".", 1)[1]
                if field.startswith(("extras.", "attributes."))
                else field
            )
            keys.append(f"{tag}.{bare}" if multi else bare)
    return keys


def _execute(lowered: dict[str, t.Any], spec: QuerySpec) -> QueryResult:
    """Run a lowered query. The only part that touches AiiDA."""
    qb = orm.QueryBuilder.from_dict(lowered)
    total = qb.count()
    if spec.count_only:
        return {"total": total, "records": []}

    qb.limit(spec.limit)
    qb.offset(spec.offset)
    keys = _record_keys(lowered)
    records = [
        {key: _serialize_value(value) for key, value in zip(keys, row, strict=True)}
        for row in qb.iterall()
    ]
    return {"total": total, "records": records}


def query_nodes(spec: QuerySpec) -> QueryResult:
    """Query AiiDA with filters, AND/OR logic, joins, sorting and group scoping.

    Use for any question about what is in the database: how many nodes match,
    which ones rank highest, and how they relate to each other. Filters combine
    with AND/OR natively, and a `path` of entities expresses provenance
    questions such as "structures that are inputs to a failed workchain".

    Prefer `count_only=True` for "how many" questions: it returns the total
    without fetching records.

    :param spec: the structured query (see :class:`QuerySpec`).
    :return: the total number of matches, and up to `limit` records (empty
        when `count_only` is True).
    :raises QueryValidationError: if the spec names an unknown entity, an
        invalid join or an unknown tag; the message says what is valid instead.
    """
    logger.debug("query_nodes(spec=%r)", spec)
    _validate_spec(spec)
    lowered = _lower(spec)
    logger.debug("query_nodes: lowered to %r", lowered)
    result = _execute(lowered, spec)
    logger.debug(
        "query_nodes: total %d, returned %d records",
        result["total"],
        len(result["records"]),
    )
    return result
