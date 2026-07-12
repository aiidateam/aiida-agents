"""Introspection of AiiDA profile schema for QueryBuilder validation.

Pulls entity types, join keywords, and available fields per type from the
running AiiDA profile. These are injected as context/enums into the model's
prompt so it generates valid QueryBuilderDict specs.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from aiida.plugins.entry_point import get_entry_point_names, load_entry_point

logger = logging.getLogger(__name__)

# The 11 join keywords defined in AiiDA's QueryBuilder.
# See: aiida.orm.querybuilder.EntityRelationships
VALID_JOIN_KEYWORDS = {
    "with_incoming",
    "with_outgoing",
    "with_descendants",
    "with_ancestors",
    "with_group",
    "with_node",
    "with_user",
    "with_computer",
    "with_comment",
    "with_log",
    "with_authinfo",
}

# The ORM base types that can appear in a QueryBuilder path.
VALID_ORM_BASES = {
    "node",
    "group",
    "computer",
    "user",
    "comment",
    "log",
}


@lru_cache(maxsize=1)
def get_entity_types() -> dict[str, str]:
    """Introspect the profile and return a mapping of entity type string
    to ORM base class (e.g., 'data.core.int.Int.' -> 'node').

    This is the ground truth of what types exist in your AiiDA profile,
    including plugin types.
    """
    types: dict[str, str] = {}

    # Node types from entry points (data, calculations, workflows, etc.)
    for group in ("aiida.data", "aiida.node"):
        for name in get_entry_point_names(group):
            try:
                cls = load_entry_point(group, name)
            except Exception:
                continue
            if node_type := getattr(cls, "class_node_type", None):
                types[node_type] = "node"

    # Note: non-node ORM bases (group, computer, user, comment, log) don't have
    # entry-point-style entity_type strings. In QueryBuilder.from_dict, you specify
    # them via orm_base only, not entity_type. So we don't add them here.

    logger.debug("introspection: found %d entity types", len(types))
    return types


@lru_cache(maxsize=1)
def get_projectable_fields() -> dict[str, set[str]]:
    """Introspect and return the set of projectable fields per ORM base.

    Returns a dict mapping base type (e.g., 'node') to the set of field
    names that can be projected (e.g., {'id', 'uuid', 'node_type', 'ctime',
    'attributes', 'extras', ...}).

    This is what you can put in project[tag] = [list of these fields].
    """
    fields: dict[str, set[str]] = {}

    # Node fields
    node_fields = {
        "id",
        "uuid",
        "node_type",
        "ctime",
        "mtime",
        "label",
        "description",
        "attributes",
        "extras",
        "repository_metadata",
    }
    fields["node"] = node_fields

    # Group fields
    group_fields = {"id", "uuid", "label", "description", "type_string"}
    fields["group"] = group_fields

    # Computer fields
    computer_fields = {
        "id",
        "uuid",
        "label",
        "hostname",
        "description",
        "enabled",
        "transport_type",
        "scheduler_type",
    }
    fields["computer"] = computer_fields

    # User fields
    user_fields = {"id", "email", "first_name", "last_name", "institution"}
    fields["user"] = user_fields

    # Comment fields
    comment_fields = {"id", "uuid", "content", "ctime", "mtime"}
    fields["comment"] = comment_fields

    # Log fields
    log_fields = {"id", "uuid", "loggername", "levelname", "message", "ctime"}
    fields["log"] = log_fields

    logger.debug("introspection: found projectable fields per base type")
    return fields


def get_filterable_fields() -> dict[str, set[str]]:
    """Introspect and return the set of filterable fields per ORM base.

    For nodes, this includes both native fields (attributes.*, extras.*)
    and computed properties. Filterable fields are those you can use in
    filters[tag] = {...}.

    Returns a dict mapping base type to the set of field names.
    """
    fields: dict[str, set[str]] = {}

    # Node: can filter by native + attributes.* + extras.* + computed props
    node_fields = {
        "id",
        "uuid",
        "node_type",
        "ctime",
        "mtime",
        "label",
        "description",
        "attributes",
        "extras",
        # Computed/relationship fields
        "user_id",
        "computer_id",
        "dbcomputer_id",
        "dbuser_id",
    }
    fields["node"] = node_fields

    # Other bases: native fields only
    fields["group"] = {"id", "uuid", "label", "description", "type_string", "user_id"}
    fields["computer"] = {"id", "uuid", "label", "hostname", "enabled"}
    fields["user"] = {"id", "email", "first_name", "last_name"}
    fields["comment"] = {"id", "uuid", "ctime", "mtime"}
    fields["log"] = {"id", "uuid", "ctime", "loggername", "levelname"}

    logger.debug("introspection: found filterable fields per base type")
    return fields


def orm_base_for_entity_type(entity_type: str) -> str | None:
    """Look up the ORM base class ('node', 'group', etc.) for an entity type.

    Returns None if the entity type is not recognized.
    """
    types = get_entity_types()
    return types.get(entity_type)


def suggest_entity_type(partial: str) -> list[str]:
    """Suggest valid entity types based on a partial match.

    Useful for error messages: "data.core.Integer unknown; did you mean
    data.core.int.Int. or data.core.float.Float.?"
    """
    types = get_entity_types()
    partial_lower = partial.lower()
    matches = [t for t in types.keys() if partial_lower in t.lower()]
    return sorted(matches)[:5]  # Return top 5 suggestions
