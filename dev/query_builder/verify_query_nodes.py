"""Verify the generic query_nodes tool directly, with no agent/LLM involved.

This isolates tool correctness from model correctness here we
call query_nodes() ourselves with hand-built specs, so there's no
question of whether the *model* constructed the right filters. This
tells us whether the tool itself gets the right answers, before it's
ever wired into the agent's tool list.

Includes the OR case that query_nodes_by_extras got wrong when the agent
tried to work around its missing OR support (agent reported 9313;
correct answer is 13829). Here it's expressed as a single native OR
filter, no multi-call workaround needed.

Usage:
    uv run python dev/query_builder/verify_query_nodes_direct.py
"""

from __future__ import annotations

import logging
import sys

from aiida_agents.tools.query_builder import (
    FieldFilter,
    FilterGroup,
    QueryNodesSpec,
    SortSpec,
    query_nodes,
)

GROUP = "workchain/PBEsol/wannier/lumi/final/bxsf"


def case_metallic() -> tuple[str, int, int | list]:
    spec = QueryNodesSpec(
        group_label=GROUP,
        filters=FieldFilter(field="insulator", operator="==", value=False),
        count_only=True,
    )
    return "metallic_count", 5597, query_nodes(spec)


def case_cubic() -> tuple[str, int, int | list]:
    spec = QueryNodesSpec(
        group_label=GROUP,
        filters=FieldFilter(field="spacegroup_number", operator=">=", value=195),
        count_only=True,
    )
    return "cubic_count", 2781, query_nodes(spec)


def case_metallic_no_overlap() -> tuple[str, int, int | list]:
    spec = QueryNodesSpec(
        group_label=GROUP,
        filters=FilterGroup(
            logic="AND",
            conditions=[
                FieldFilter(field="insulator", operator="==", value=False),
                FieldFilter(field="overlapping_semicore", operator="==", value=False),
            ],
        ),
        count_only=True,
    )
    return "metallic_no_overlap_semicore", 5597, query_nodes(spec)


def case_cov() -> tuple[str, int, int | list]:
    spec = QueryNodesSpec(
        group_label=GROUP,
        filters=FieldFilter(field="formula_hill", operator="==", value="CoV"),
        count_only=True,
    )
    return "cov_formula", 1, query_nodes(spec)


def case_negative_control() -> tuple[str, int, int | list]:
    spec = QueryNodesSpec(
        group_label=GROUP,
        filters=FilterGroup(
            logic="AND",
            conditions=[
                FieldFilter(field="spacegroup_number", operator=">=", value=195),
                FieldFilter(field="formula_hill", operator="==", value="XYZ123"),
            ],
        ),
        count_only=True,
    )
    return "negative_control", 0, query_nodes(spec)


def case_or_logic() -> tuple[str, int, int | list]:
    """The case query_nodes_by_extras got wrong: agent reported 9313,
    correct combined OR count is 13829 (9313 + 12130 - 7614, per the
    earlier manual sub-query breakdown). Expressed here as a single
    native OR filter instead of a multi-call workaround.
    """
    spec = QueryNodesSpec(
        group_label=GROUP,
        filters=FilterGroup(
            logic="OR",
            conditions=[
                FieldFilter(field="insulator", operator="==", value=True),
                FieldFilter(field="spacegroup_number", operator="<", value=195),
            ],
        ),
        count_only=True,
    )
    return "or_logic", 13829, query_nodes(spec)


def case_range_query() -> tuple[str, int | None, int | list]:
    """No independently-verified ground truth yet for this one — the
    agent reported 5905 via query_nodes_by_extras, but that hasn't been
    checked against a raw QueryBuilder script. Treat the expected value
    here as "to be confirmed", not verified ground truth.
    """
    spec = QueryNodesSpec(
        group_label=GROUP,
        filters=FilterGroup(
            logic="AND",
            conditions=[
                FieldFilter(field="spacegroup_number", operator=">=", value=100),
                FieldFilter(field="spacegroup_number", operator="<=", value=195),
            ],
        ),
        count_only=True,
    )
    return "range_query", None, query_nodes(spec)


def case_ranking() -> tuple[str, None, list]:
    spec = QueryNodesSpec(
        group_label=GROUP,
        filters=None,
        sort=[SortSpec(field="pw_bandgap", direction="desc", cast="f")],
        project=["id", "extras"],
        limit=5,
        count_only=False,
    )
    return "ranking_query", None, query_nodes(spec)


CASES = [
    case_metallic,
    case_cubic,
    case_metallic_no_overlap,
    case_cov,
    case_negative_control,
    case_or_logic,
    case_range_query,
    case_ranking,
]


def main() -> None:
    from aiida import load_profile
    from aiida_agents._logging import suppress_noisy_loggers
    from aiida_agents._settings import LoggingSettings

    logging.basicConfig(level=LoggingSettings().log_level, stream=sys.stderr)
    if logging.getLogger().getEffectiveLevel() <= logging.DEBUG:
        suppress_noisy_loggers()
    load_profile()

    header = f"{'case':<32} {'expected':>10} {'actual':>10} {'status':>8}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for case_fn in CASES:
        case_id, expected, actual = case_fn()
        if expected is None:
            status = "n/a"
            actual_display = (
                actual if not isinstance(actual, list) else f"[{len(actual)} rows]"
            )
        elif isinstance(actual, list):
            status = "n/a"
            actual_display = f"[{len(actual)} rows]"
        else:
            status = "PASS" if actual == expected else "FAIL"
            actual_display = actual
        print(
            f"{case_id:<32} {str(expected):>10} {str(actual_display):>10} {status:>8}"
        )

        if case_id == "ranking_query" and isinstance(actual, list):
            print("  top 5 by pw_bandgap:")
            for row in actual:
                extras = row.get("extras", {})
                print(f"    pk={row.get('id')} pw_bandgap={extras.get('pw_bandgap')}")

    print("=" * len(header))
    print(
        "\nNote: range_query has no independently verified ground truth yet "
        "(only the agent's own prior claim of 5905, unconfirmed). "
        "ranking_query has no ground truth to compare against — verify the "
        "top-5 PKs above against a raw QueryBuilder order_by if needed."
    )


if __name__ == "__main__":
    main()
