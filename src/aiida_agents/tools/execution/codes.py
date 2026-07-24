"""Discovery of the configured codes a calculation can actually run on.

Submitting a real calculation needs a ``code`` input, given as a
``{"label": "name@computer"}`` reference (see ``tools.execution.submit``). The
agent has no way to invent that label correctly, so this tool lists what is
actually configured in the profile, optionally narrowed to the codes that can
run a given calculation entry point.
"""

from __future__ import annotations

import logging
import typing as t

from aiida import orm
from pydantic import Field

from .._types import CodeRecord

logger = logging.getLogger(__name__)

__all__ = ["list_codes"]


def list_codes(
    entry_point: t.Annotated[
        str | None,
        Field(
            description=(
                "Only list codes whose default calculation plugin is this entry "
                "point (e.g. 'core.arithmetic.add', 'quantumespresso.pw'). "
                "Omit to list every configured code."
            )
        ),
    ] = None,
    include_hidden: t.Annotated[
        bool,
        Field(description="Include codes the user has marked as hidden."),
    ] = False,
) -> list[CodeRecord]:
    """List the codes configured in this profile, for use as a `code` input.

    A calculation's `code` input is given as `{"label": "name@computer"}`; this
    tool reports the `full_label` to use for each configured code, so the label
    never has to be guessed. Filter by `entry_point` to see only the codes that
    can run a given calculation.

    Args:
        entry_point: Restrict to codes whose default calculation plugin matches.
        include_hidden: Include codes marked hidden (excluded by default).

    Returns:
        One record per configured code, with the `full_label` to submit with.
    """
    logger.debug(
        "list_codes(entry_point=%r, include_hidden=%r)", entry_point, include_hidden
    )

    qb = orm.QueryBuilder()
    filters: dict[str, t.Any] = {}
    if entry_point is not None:
        filters["attributes.input_plugin"] = entry_point
    # ``orm.Code`` (not ``orm.AbstractCode``) is the queryable base: it resolves
    # to both InstalledCode and PortableCode and nothing else. AbstractCode has
    # no entry point of its own, and querying it silently matches unrelated Data
    # nodes (a StructureData comes back as a "code").
    qb.append(orm.Code, tag="code", filters=filters or None)
    qb.order_by({"code": {"ctime": "desc"}})

    records: list[CodeRecord] = []
    for (code,) in qb.iterall():
        if code.is_hidden and not include_hidden:
            continue
        computer = code.computer
        records.append(
            {
                "pk": t.cast(int, code.pk),
                "uuid": str(code.uuid),
                "label": code.label,
                "full_label": code.full_label,
                "computer": computer.label if computer is not None else None,
                "default_calc_job_plugin": code.default_calc_job_plugin,
                "node_type": type(code).__name__,
            }
        )

    logger.debug("list_codes: returned %d records", len(records))
    return records
