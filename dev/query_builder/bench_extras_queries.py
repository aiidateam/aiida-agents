"""Benchmark the extras-query tool against Natasha's ground-truth numbers.

For each test case, runs the agent on a natural-language query and inspects
`all_messages()` to separately
capture:

  - tool_result       the value query_nodes_by_extras actually returned
                       (None if the tool was never called at all)
  - agent_text_answer  a best-effort number pulled from the agent's final
                       prose output (fuzzy — natural language, not
                       structured data)

These are checked independently against `expected`, since earlier manual
testing found cases where the tool returned the correct count but the
agent's prose answer misreported it (e.g. tool returned 2781, agent said
"2,881"), and one case where a repeated query wasn't re-verified against
the tool at all.

Usage:
    uv run python dev/query_builder/bench_extras_queries.py
    uv run python dev/query_builder/bench_extras_queries.py --group "other/group/label"
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys
from dataclasses import dataclass
from typing import Any

from pydantic_ai.messages import ToolReturnPart

from aiida_agents.agents import get_agent

DEFAULT_GROUP = "workchain/PBEsol/wannier/lumi/final/bxsf"
TOOL_NAME = "query_nodes"


@dataclass
class TestCase:
    id: str
    query_template: str  # use "{group}" as placeholder
    expected: int | None  # None = no known ground truth, just observe behavior


@dataclass
class CaseResult:
    id: str
    query: str
    expected: int | None
    tool_called: bool
    tool_result: Any = None
    agent_text_answer: int | None = None
    raw_output: str = ""

    @property
    def tool_matches_expected(self) -> bool | None:
        if self.expected is None or not self.tool_called:
            return None
        res_val = (
            self.tool_result["total"]
            if isinstance(self.tool_result, dict) and "total" in self.tool_result
            else self.tool_result
        )
        return res_val == self.expected

    @property
    def agent_matches_tool(self) -> bool | None:
        if not self.tool_called or self.agent_text_answer is None:
            return None
        res_val = (
            self.tool_result["total"]
            if isinstance(self.tool_result, dict) and "total" in self.tool_result
            else self.tool_result
        )
        return self.agent_text_answer == res_val


TEST_CASES: list[TestCase] = [
    TestCase(
        id="cubic_count",
        query_template=(
            "how many structures with spacegroup number 195 or above are "
            "in the group {group}?"
        ),
        expected=2781,
    ),
    TestCase(
        id="metallic_count",
        query_template=(
            "how many metallic structures (insulator is false) are in the "
            "group {group}?"
        ),
        expected=5597,
    ),
    TestCase(
        id="metallic_no_overlap_semicore",
        query_template=(
            "how many structures with insulator false and "
            "overlapping_semicore false are in the group {group}?"
        ),
        expected=5597,
    ),
    TestCase(
        id="cov_formula",
        query_template="how many nodes have formula_hill CoV in the group {group}?",
        expected=1,
    ),
    TestCase(
        id="negative_control",
        query_template=(
            "how many structures with spacegroup_number >= 195 are in the "
            "group {group} and formula_hill XYZ123?"
        ),
        expected=0,
    ),
    # "Break it" cases below have no known ground truth yet — the point is
    # to observe *whether* the tool can express the query at all, and
    # whether the agent silently approximates instead of refusing.
    TestCase(
        id="or_logic",
        query_template=(
            "how many structures are insulator=true OR spacegroup_number < "
            "195 in the group {group}?"
        ),
        expected=None,
    ),
    TestCase(
        id="range_query",
        query_template=(
            "how many structures have spacegroup_number between 100 and "
            "195 in the group {group}?"
        ),
        expected=None,
    ),
    TestCase(
        id="ranking_query",
        query_template=(
            "what are the 5 structures with the highest pw_bandgap in the "
            "group {group}?"
        ),
        expected=None,
    ),
]


def _extract_agent_number(text: str) -> int | None:
    """Best-effort pull of the first number-like token from prose output.

    Fuzzy by nature. Good enough for flagging mismatches like the
    2781-vs-2881 case; not a substitute for reading raw_output when a
    case looks suspicious.
    """
    match = re.search(r"\b\d{1,3}(?:,\d{3})*\b", text)
    if not match:
        return None
    return int(match.group(0).replace(",", ""))


async def run_case(case: TestCase, group: str) -> CaseResult:
    query = case.query_template.format(group=group)
    agent = get_agent()
    result = await agent.run(query)

    tool_called = False
    tool_result: Any = None

    for msg in result.all_messages():
        for part in msg.parts:
            if isinstance(part, ToolReturnPart) and part.tool_name == TOOL_NAME:
                tool_called = True
                tool_result = part.content

    raw_output = str(result.output)
    agent_text_answer = _extract_agent_number(raw_output)

    return CaseResult(
        id=case.id,
        query=query,
        expected=case.expected,
        tool_called=tool_called,
        tool_result=tool_result,
        agent_text_answer=agent_text_answer,
        raw_output=raw_output,
    )


def _fmt(value: Any) -> str:
    return "—" if value is None else str(value)


def _fmt_bool(value: bool | None) -> str:
    if value is None:
        return "n/a"
    return "PASS" if value else "FAIL"


def print_summary(results: list[CaseResult]) -> None:
    header = (
        f"{'case':<32} {'expected':>9} {'tool_result':>12} "
        f"{'agent_said':>11} {'tool_ok':>8} {'agent_ok':>9} {'called':>7}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for r in results:
        tool_res_display = (
            r.tool_result["total"]
            if isinstance(r.tool_result, dict) and "total" in r.tool_result
            else r.tool_result
        )
        print(
            f"{r.id:<32} {_fmt(r.expected):>9} {_fmt(tool_res_display):>12} "
            f"{_fmt(r.agent_text_answer):>11} {_fmt_bool(r.tool_matches_expected):>8} "
            f"{_fmt_bool(r.agent_matches_tool):>9} {r.tool_called!s:>7}"
        )
    print("=" * len(header))

    known = [r for r in results if r.expected is not None]
    tool_pass = sum(1 for r in known if r.tool_matches_expected)
    agent_pass = sum(1 for r in known if r.agent_matches_tool)
    print(
        f"\nTool correctness (vs. ground truth): {tool_pass}/{len(known)}"
        f"\nAgent prose matches tool result:      {agent_pass}/{len(known)}"
    )

    not_called = [r for r in results if not r.tool_called]
    if not_called:
        print(
            f"\nNote: tool was NOT called for {len(not_called)} case(s): "
            f"{', '.join(r.id for r in not_called)}"
        )
        print("(check raw_output below — may be a refusal, an approximation,")
        print(" or an answer pulled from conversation history instead of a fresh call)")


def print_raw_outputs(results: list[CaseResult]) -> None:
    print("\n" + "=" * 80)
    print("RAW AGENT OUTPUTS (for cases with no ground truth, or mismatches)")
    print("=" * 80)
    for r in results:
        interesting = (
            r.expected is None
            or r.tool_matches_expected is False
            or r.agent_matches_tool is False
            or not r.tool_called
        )
        if interesting:
            print(f"\n--- {r.id} ---")
            print(f"query: {r.query}")
            print(f"tool_called: {r.tool_called}  tool_result: {r.tool_result}")
            print(f"raw_output:\n{r.raw_output}")


async def main_async(group: str, case_ids: list[str] | None) -> None:
    from aiida import load_profile
    from aiida_agents._logging import suppress_noisy_loggers
    from aiida_agents._settings import LoggingSettings

    logging.basicConfig(level=LoggingSettings().log_level, stream=sys.stderr)
    if logging.getLogger().getEffectiveLevel() <= logging.DEBUG:
        suppress_noisy_loggers()
    load_profile()

    cases = TEST_CASES
    if case_ids:
        cases = [c for c in TEST_CASES if c.id in case_ids]
        if not cases:
            print(f"No matching case ids: {case_ids}")
            print(f"Available: {[c.id for c in TEST_CASES]}")
            sys.exit(1)

    results: list[CaseResult] = []
    for case in cases:
        print(f"Running: {case.id} ...", file=sys.stderr)
        result = await run_case(case, group)
        results.append(result)

    print_summary(results)
    print_raw_outputs(results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        default=DEFAULT_GROUP,
        help=f"Group label to scope queries to (default: {DEFAULT_GROUP})",
    )
    parser.add_argument(
        "--cases",
        nargs="*",
        default=None,
        help="Specific case ids to run (default: all). "
        f"Available: {[c.id for c in TEST_CASES]}",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args.group, args.cases))


if __name__ == "__main__":
    main()
