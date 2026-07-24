"""Generic node-query tool: a structured spec lowered to AiiDA's QueryBuilder.

The model supplies a :class:`QuerySpec` -- entities, joins, filters, sort,
projection -- rather than raw QueryBuilder code. It runs as a small pipeline::

    spec -> normalise -> validate -> lower -> execute -> serialise

``lower`` is a pure function producing the dict that ``QueryBuilder.from_dict``
already accepts (the same shape as aiida-restapi's ``QueryBuilderDict``), so the
whole translation is testable without a database. ``validate`` also touches the
database, but only to confirm a group named by label actually exists -- a typo'd
`group_label` would otherwise be indistinguishable from an existing, empty group,
both silently returning a total of 0.

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
6. Installed plugins are queryable by name. A specific ``CalcJob``/``WorkChain``
   plugin (e.g. ``PwBandsWorkChain``) shares its ``node_type`` with every other
   plugin of the same kind, so entity resolution also carries an implicit
   ``process_type`` filter for these -- see :class:`EntityAlias`.
7. Joins can filter the edge, not just the entities either side. A ``path``
   item's ``edge_filters`` narrows by ``link_label``/``link_type`` (e.g. "only
   the *return* links"), which entity filters alone cannot express. Only valid
   on ``with_incoming``/``with_outgoing``: transitive joins have no single
   edge to filter, and aiida-core does not validate that itself -- it raises
   an unrelated ``AttributeError`` instead, so :func:`_validate_spec` catches
   it first.

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
from aiida.common.exceptions import MultipleObjectsError, NotExistent
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

#: Every join keyword ``EntityRelationships`` recognises for any entity, so the
#: schema itself cannot fall behind if aiida-core ever adds or renames one; the
#: legality of a keyword *for a given entity* is still checked dynamically
#: against ``EntityRelationships`` in :func:`_validate_spec`, this only bounds
#: what the model may even attempt to write.
_JOIN_KEYWORDS: tuple[str, ...] = tuple(
    sorted(
        {keyword for keywords in EntityRelationships.values() for keyword in keywords}
    )
)
JoinKeyword = t.Literal[_JOIN_KEYWORDS]  # type: ignore[valid-type]

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

#: The only fields a link (edge) actually has -- unlike node fields, there is
#: no extras/attributes prefixing to apply.
EDGE_FIELDS = frozenset({"label", "type"})

#: Joins whose edge_filters actually apply. Transitive joins (with_ancestors,
#: with_descendants) walk a recursive path-table with no per-edge join, so
#: aiida-core raises a raw, unhelpful AttributeError if edge_filters is given
#: there instead of validating it (verified against the installed aiida-core).
EDGE_FILTERABLE_JOINS = frozenset({"with_incoming", "with_outgoing"})


class QueryValidationError(ValueError):
    """A spec that cannot be lowered, reported before the query runs."""


class EntityAlias(t.NamedTuple):
    """What an entity alias resolves to.

    Plain ``Data``/``Node`` subclasses are fully identified by ``node_type``
    (``class_node_type`` already encodes subtree matching, e.g. ``data.Data.``
    matches every ``Data`` subclass). Process plugins (a specific ``CalcJob`` or
    ``WorkChain``, e.g. installed from ``aiida-quantumespresso``) are not: they
    have no ``class_node_type`` of their own and share one generic node class
    with every other plugin of the same kind, so ``CalcJobNode`` alone cannot
    tell a ``PwCalculation`` from an ``ArithmeticAddCalculation``. Their
    concrete identity lives in ``process_type`` instead, which must be filtered
    on in addition to the (shared) ``node_type``.
    """

    node_type: str
    process_type: str | None = None


@lru_cache(maxsize=1)
def _entity_index() -> dict[str, EntityAlias]:
    """Map entity aliases to what they resolve to, derived from aiida-core."""
    index: dict[str, EntityAlias] = {
        "node": EntityAlias(""),
        "group": EntityAlias("group.core"),
    }
    for entry_point_group in ("aiida.data", "aiida.node"):
        for name in get_entry_point_names(entry_point_group):
            try:
                cls = load_entry_point(entry_point_group, name)
            except Exception:  # noqa: BLE001 - a broken plugin must not break queries
                logger.debug("entity_index: cannot load %s:%s", entry_point_group, name)
                continue
            node_type = getattr(cls, "class_node_type", None)
            if node_type is not None:
                index[name.lower()] = EntityAlias(node_type)
                index[cls.__name__.lower()] = EntityAlias(node_type)
    # Process plugins (installed CalcJobs/WorkChains): identified by
    # `process_type`, not `node_type`, which only distinguishes their kind
    # (calcjob vs workchain etc.), not the specific plugin.
    for entry_point_group in ("aiida.calculations", "aiida.workflows"):
        for name in get_entry_point_names(entry_point_group):
            try:
                cls = load_entry_point(entry_point_group, name)
            except Exception:  # noqa: BLE001 - a broken plugin must not break queries
                logger.debug("entity_index: cannot load %s:%s", entry_point_group, name)
                continue
            node_class = getattr(cls, "_node_class", None)
            node_type = getattr(node_class, "class_node_type", None)
            if node_type is None:
                continue
            alias = EntityAlias(node_type, cls.build_process_type())
            index[name.lower()] = alias
            index[cls.__name__.lower()] = alias
    for cls in (orm.Data, orm.ProcessNode, orm.CalculationNode, orm.WorkflowNode):
        index[cls.__name__.lower()] = EntityAlias(cls.class_node_type)
    index["data"] = EntityAlias(orm.Data.class_node_type)
    index["process"] = EntityAlias(orm.ProcessNode.class_node_type)
    index["calculation"] = EntityAlias(orm.CalculationNode.class_node_type)
    index["workflow"] = EntityAlias(orm.WorkflowNode.class_node_type)
    return index


def _resolve_entity(entity_type: str, index: dict[str, EntityAlias]) -> EntityAlias:
    """Resolve an alias, or pass through a raw ``node_type`` string as-is."""
    return index.get(entity_type.lower(), EntityAlias(entity_type))


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
            "'group'), an installed plugin's class or entry-point name "
            "('PwBandsWorkChain', 'quantumespresso.pw.bands'), or a full node "
            "type ('data.core.structure.StructureData.'). Abstract levels match "
            "their whole subtree; a plugin name matches only that plugin."
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
    edge_filters: FilterTree | None = Field(
        default=None,
        description=(
            "Filter the link itself, not the entities it connects -- e.g. only "
            "'return' links, or a specific link_label. Fields: 'label' "
            "(link_label), 'type' (link_type, e.g. 'create', 'return', "
            "'input_calc', 'call_calc', 'call_work'). Only valid with "
            "joining_keyword 'with_incoming' or 'with_outgoing'; aiida-core does "
            "not support edge filtering on transitive joins."
        ),
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
                   "joining_keyword": "with_outgoing", "joining_value": "wc",
                   "edge_filters": {"field": "label", "operator": "==",
                                    "value": "output_structure"}}],
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
        if (
            item.edge_filters is not None
            and item.joining_keyword not in EDGE_FILTERABLE_JOINS
        ):
            valid = " or ".join(f"{kw!r}" for kw in sorted(EDGE_FILTERABLE_JOINS))
            msg = (
                f"edge_filters on tag {item.tag!r} needs joining_keyword {valid}; "
                f"aiida-core does not support filtering the edge for "
                f"{item.joining_keyword!r}."
            )
            raise QueryValidationError(msg)

        if item.joining_keyword is not None:
            orm_base = _orm_base(_resolve_entity(item.entity_type, index).node_type)
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

    _validate_group_labels(spec, index)


def _validate_group_labels(spec: QuerySpec, index: dict[str, EntityAlias]) -> None:
    """A ``group`` entity filtered by an exact ``label`` must name a real group.

    Without this, a typo'd `group_label` is indistinguishable from an existing
    but empty group -- both would otherwise silently return a total of 0.
    """
    for item in spec.path:
        if _orm_base(_resolve_entity(item.entity_type, index).node_type) != "group":
            continue
        tree = spec.filters.get(item.tag)
        if (
            not isinstance(tree, FieldFilter)
            or tree.field != "label"
            or tree.operator != "=="
        ):
            continue
        try:
            orm.Group.collection.get(label=tree.value)
        except (NotExistent, MultipleObjectsError) as exc:
            msg = f"Group with label {tree.value!r} does not exist."
            raise QueryValidationError(msg) from exc


def _qualify_field(field: str) -> str:
    """Prefix a bare extras key with ``extras.``; leave columns and containers alone."""
    if field in PASSTHROUGH_FIELDS or field.startswith(("extras.", "attributes.")):
        return field
    return f"extras.{field}"


def _qualify_edge_field(field: str) -> str:
    """A link has only `label`/`type` -- no extras/attributes prefixing applies."""
    if field not in EDGE_FIELDS:
        msg = (
            f"edge_filters may only use 'label' or 'type', got {field!r}. "
            "'label' is the link_label, 'type' is the link_type "
            "(e.g. 'create', 'return', 'input_calc')."
        )
        raise QueryValidationError(msg)
    return field


def _lower_filter(
    node: FilterTree, qualify: t.Callable[[str], str] = _qualify_field
) -> dict[str, t.Any]:
    """Translate a filter tree into QueryBuilder's nested filter dict.

    :param qualify: how to resolve a bare field name. Node filters (the
        default) prefix bare extras keys with ``extras.``; edge filters pass
        :func:`_qualify_edge_field` instead, since a link's only fields are
        ``label``/``type`` with no extras/attributes concept.
    """
    if isinstance(node, FieldFilter):
        if node.operator == "in" and not isinstance(node.value, (list, tuple, set)):
            msg = (
                f"Operator 'in' needs a list of values for field {node.field!r}, got "
                f"{type(node.value).__name__}: {node.value!r}. Example: [1, 2, 3]."
            )
            raise QueryValidationError(msg)
        field = qualify(node.field)
        if node.operator == "==":
            return {field: node.value}
        return {field: {node.operator: node.value}}
    key = "and" if node.logic == "AND" else "or"
    return {key: [_lower_filter(condition, qualify) for condition in node.conditions]}


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
    process_type_filters: dict[str, dict[str, t.Any]] = {}
    for item in spec.path:
        alias = _resolve_entity(item.entity_type, index)
        entry: dict[str, t.Any] = {
            "entity_type": alias.node_type,
            "orm_base": _orm_base(alias.node_type),
            "tag": item.tag,
            "outerjoin": item.outerjoin,
        }
        if item.joining_keyword is not None:
            entry["joining_keyword"] = item.joining_keyword
            entry["joining_value"] = item.joining_value
        if item.edge_filters is not None:
            entry["edge_filters"] = _lower_filter(
                item.edge_filters, _qualify_edge_field
            )
        path.append(entry)
        # A process plugin (e.g. a specific WorkChain) shares its node_type with
        # every other plugin of the same kind; process_type is what actually
        # narrows the query to that plugin.
        if alias.process_type is not None:
            process_type_filters[item.tag] = {"process_type": alias.process_type}

    lowered: dict[str, t.Any] = {"path": path}

    filters: dict[str, t.Any] = {
        tag: _lower_filter(tree) for tag, tree in spec.filters.items()
    }
    for tag, process_type_filter in process_type_filters.items():
        filters[tag] = (
            {"and": [process_type_filter, filters[tag]]}
            if tag in filters
            else process_type_filter
        )
    if filters:
        lowered["filters"] = filters

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
