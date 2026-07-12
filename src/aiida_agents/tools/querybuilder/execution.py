"""Execute a validated QueryBuilderDict and return results with preview."""

from __future__ import annotations

import logging
import typing as t

from aiida import orm
from aiida_restapi.models.querybuilder import QueryBuilderDict

logger = logging.getLogger(__name__)

MAX_PREVIEW_ROWS = 5


def execute_querybuilder(spec: QueryBuilderDict) -> dict[str, t.Any]:
    """Execute a validated QueryBuilderDict and return results with preview.

    Returns a dict with:
      - query_dict: the original QueryBuilderDict (for reference/debugging)
      - total_count: total matching records (fast COUNT query)
      - preview_rows: first N rows (without limit, limited to MAX_PREVIEW_ROWS here)
      - preview_count: how many rows are in the preview

    This lets the user sanity-check the query result before building a full
    answer on it.

    Args:
        spec: A validated QueryBuilderDict.

    Returns:
        A dict with query, total_count, preview_rows, and preview_count.

    Raises:
        ValueError: If QueryBuilder.from_dict fails (syntax/schema error).
    """
    logger.debug("execute_querybuilder(path=%r)", spec.path)

    # Convert spec to dict for QueryBuilder
    query_dict = spec.model_dump(exclude_none=True)

    try:
        # Build the QueryBuilder from the dict
        qb = orm.QueryBuilder.from_dict(query_dict)
    except Exception as e:
        logger.error("QueryBuilder.from_dict failed: %s", e)
        raise ValueError(f"QueryBuilder construction failed: {e}") from e

    # Get total count (fast)
    try:
        total_count = qb.count()
    except Exception as e:
        logger.error("count() failed: %s", e)
        raise ValueError(f"count() failed: {e}") from e

    # Get preview rows (limited to MAX_PREVIEW_ROWS)
    preview_rows: list[dict[str, t.Any]] = []
    try:
        # Set a preview limit
        qb.limit(MAX_PREVIEW_ROWS)

        # Fetch preview
        for row in qb.iterall():
            # Row is typically a tuple or list; convert to dict if we have project info
            if isinstance(row, (tuple, list)):
                # If spec.project is set, zip with field names; otherwise just pass as-is
                if spec.project:
                    # This is a simplified approach; real logic would be more complex
                    preview_rows.append({"values": row})
                else:
                    preview_rows.append({"values": row})
            else:
                preview_rows.append({"value": row})
    except Exception as e:
        logger.error("iterall() failed: %s", e)
        raise ValueError(f"Preview fetch failed: {e}") from e

    logger.debug(
        "execute_querybuilder: total_count=%d, preview_rows=%d",
        total_count,
        len(preview_rows),
    )

    return {
        "query_dict": query_dict,
        "total_count": total_count,
        "preview_rows": preview_rows,
        "preview_count": len(preview_rows),
    }
