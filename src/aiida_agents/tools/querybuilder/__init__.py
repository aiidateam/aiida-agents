"""Generic QueryBuilder tool for the Analysis Agent.

accept arbitrary QueryBuilderDict specs,
validate strictly against the AiiDA profile schema (entity types, join
keywords, field names), and execute with preview (count + first rows).

Validation errors are specific and actionable, so the model learns and
corrects itself via a ModelRetry loop.
"""

from __future__ import annotations

import logging
import typing as t

from aiida_restapi.models.querybuilder import QueryBuilderDict

from .execution import execute_querybuilder
from .validation import QueryValidationError, validate_querybuilder_dict

logger = logging.getLogger(__name__)


def query(spec: QueryBuilderDict) -> dict[str, t.Any]:
    """Query AiiDA using a structured QueryBuilderDict spec.

    This is the Analysis Agent's generic query tool. It accepts an arbitrary
    QueryBuilder specification (path, filters, projections, sorting, paging),
    validates it strictly against the AiiDA profile schema, and executes it
    with a preview of the first few results.

    The validation is strict: if your spec has invalid entity types, join
    keywords, or field names, you'll get a specific error message suggesting
    corrections, so you can fix it and try again. This replaces the need to
    run Python code directly.

    Args:
        spec: A QueryBuilderDict with:
          - path: list of entity types / path items (nodes, groups, users, etc.)
          - filters: field conditions per tag
          - project: fields to return per tag
          - limit, offset: pagination
          - order_by: sort order

    Returns:
        A dict with:
          - query_dict: the spec you provided (for reference)
          - total_count: total matching records
          - preview_rows: first N rows (preview only, not the full result)
          - preview_count: how many rows in the preview

    Raises:
        QueryValidationError (returned as ModelRetry): if the spec violates
            the schema (unknown entity type, invalid join keyword, bad field
            name, etc.). The error message is actionable and specific.
    """
    logger.debug("query(spec=%r)", spec)

    # Step 1: Validate strictly
    try:
        validate_querybuilder_dict(spec)
    except QueryValidationError as e:
        logger.warning("query validation failed: %s", e)
        raise  # Caught by the agent as a ModelRetry

    # Step 2: Execute and return preview
    try:
        result = execute_querybuilder(spec)
        logger.debug(
            "query: returned count=%d, preview_count=%d",
            result["total_count"],
            result["preview_count"],
        )
        return result
    except ValueError as e:
        logger.error("query execution failed: %s", e)
        raise QueryValidationError(f"Query execution failed: {e}") from e


__all__ = ["query"]
