"""Strict validation of QueryBuilderDict against the AiiDA profile schema.

When the model generates an invalid QueryBuilderDict, this returns a
ModelRetry-compatible error message with specific suggestions (e.g.,
"data.core.Integer unknown; did you mean data.core.int.Int.?") so the
model learns and corrects itself.
"""

from __future__ import annotations

import logging

from aiida_restapi.models.querybuilder import QueryBuilderDict, QueryBuilderPathItem

from .introspection import (
    VALID_JOIN_KEYWORDS,
    VALID_ORM_BASES,
    get_entity_types,
    get_projectable_fields,
    suggest_entity_type,
)

logger = logging.getLogger(__name__)


class QueryValidationError(Exception):
    """Raised when a QueryBuilderDict fails validation with an actionable message."""

    pass


def validate_querybuilder_dict(spec: QueryBuilderDict) -> None:
    """Validate a QueryBuilderDict strictly against the AiiDA profile schema.

    Raises QueryValidationError with actionable suggestions if validation fails.
    This error is caught by the agent and returned as a ModelRetry, so the model
    learns the correct format and tries again.

    Args:
        spec: The QueryBuilderDict to validate.

    Raises:
        QueryValidationError: If any part of the spec is invalid. The message
            is human-readable and includes specific suggestions.
    """
    logger.debug(
        "validate_querybuilder_dict(path=%r, filters=%r)", spec.path, spec.filters
    )

    entity_types = get_entity_types()
    projectable = get_projectable_fields()
    tags_in_path: set[str] = set()

    # Validate path: entity types and joining keywords
    for i, item in enumerate(spec.path):
        if isinstance(item, str):
            # Simple entity type string
            if item not in entity_types:
                suggestions = suggest_entity_type(item)
                msg = f"Path item {i}: entity type '{item}' is unknown."
                if suggestions:
                    msg += f" Did you mean one of: {', '.join(suggestions[:3])}?"
                raise QueryValidationError(msg)
        elif isinstance(item, QueryBuilderPathItem):
            # Full path item with optional joining keyword
            base = item.orm_base
            if base not in VALID_ORM_BASES:
                raise QueryValidationError(
                    f"Path item {i}: orm_base '{base}' is invalid. "
                    f"Valid bases: {', '.join(sorted(VALID_ORM_BASES))}."
                )

            # Validate entity_type only if provided and non-empty (required for nodes, optional for groups/computers/etc)
            if item.entity_type is not None and item.entity_type != "":
                types_to_check = (
                    item.entity_type
                    if isinstance(item.entity_type, list)
                    else [item.entity_type]
                )
                for et in types_to_check:
                    if et not in entity_types:
                        suggestions = suggest_entity_type(et)
                        msg = f"Path item {i}: entity type '{et}' is unknown."
                        if suggestions:
                            msg += (
                                f" Did you mean one of: {', '.join(suggestions[:3])}?"
                            )
                        raise QueryValidationError(msg)

            # Validate joining_keyword if present
            if item.joining_keyword is not None:
                if item.joining_keyword not in VALID_JOIN_KEYWORDS:
                    raise QueryValidationError(
                        f"Path item {i}: joining_keyword '{item.joining_keyword}' is invalid. "
                        f"Valid keywords: {', '.join(sorted(VALID_JOIN_KEYWORDS))}."
                    )

            # Track tag for later validation
            if item.tag:
                tags_in_path.add(item.tag)

    # Validate filters: tags must exist in path, field names must be valid
    if spec.filters:
        for tag, filter_dict in spec.filters.items():
            if tag not in tags_in_path:
                raise QueryValidationError(
                    f"Filters: tag '{tag}' not found in path. "
                    f"Valid tags: {', '.join(sorted(tags_in_path)) if tags_in_path else '(none, add tags to path items)'}."
                )

            # For now, we don't deeply validate individual field names since they can be
            # nested (attributes.foo.bar, extras.x.y) and complex operators. A basic
            # check could go here, but QueryBuilder.from_dict will catch real errors.

    # Validate project: tags and field names
    if spec.project:
        for tag, fields in spec.project.items():
            if tag not in tags_in_path:
                raise QueryValidationError(
                    f"Project: tag '{tag}' not found in path. "
                    f"Valid tags: {', '.join(sorted(tags_in_path)) if tags_in_path else '(none, add tags to path items)'}."
                )

            # Get the ORM base for this tag to check projectable fields
            tag_base: str | None = None
            for item in spec.path:
                if isinstance(item, QueryBuilderPathItem) and item.tag == tag:
                    tag_base = item.orm_base
                    break

            if tag_base and tag_base in projectable:
                fields_to_check = fields if isinstance(fields, list) else [fields]
                for field in fields_to_check:
                    if field not in projectable[tag_base]:
                        raise QueryValidationError(
                            f"Project['{tag}']: field '{field}' is not projectable for "
                            f"orm_base '{tag_base}'. Valid fields: "
                            f"{', '.join(sorted(projectable[tag_base]))}."
                        )

    logger.debug("validate_querybuilder_dict: passed validation")
