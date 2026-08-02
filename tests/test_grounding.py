"""Tests for the runtime grounding check.

This is the backstop that does not depend on the model cooperating. The prompt
version of the same rule was measured at five ignored instructions out of five,
so these hold the check to catching what the prompt did not.
"""

from __future__ import annotations

import pytest

from aiida_agents.grounding import (
    labelled_python_blocks,
    python_blocks,
    syntax_errors,
    ungrounded_symbols,
    quantities_in,
    tool_output_text,
    ungrounded_quantities,
)


class TestTheFabricationThatShipped:
    """The real answers that prompted this check, as test cases."""

    def test_number_bolted_onto_a_real_label_is_caught(self) -> None:
        """'42 for gold' -- where the gold was retrieved and the 42 was not.

        The hardest case to see by eye: the sentence contains real, retrieved
        content, so it reads as sourced. Only the number is invented.
        """
        evidence = "[{'formula_hill': 'Au', 'ecutwfc': None}, {'formula_hill': 'K'}]"
        answer = "ecutwfc varies widely (e.g., 42 for gold, 8 for the alkali metals)"

        assert ungrounded_quantities(answer, evidence) == {"42.0", "8.0"}

    def test_invented_log_line_is_caught(self) -> None:
        """A run that made no query at all and quoted a log that never existed."""
        answer = "The log shows ecutwfc=0.05 Ry, later refined to ecutwfc = 0.03 Ry"

        assert ungrounded_quantities(answer, evidence="") == {"0.05", "0.03"}

    def test_correctly_quoted_statistics_pass(self) -> None:
        """The good runs must stay quiet, or the warning becomes noise."""
        evidence = "{'median_ecutwfc': 60.0, 'median_kpoints_distance': 0.15}"
        answer = "Your successful runs used an ecutwfc of 60.0 Ry and 0.15 1/A spacing."

        assert ungrounded_quantities(answer, evidence) == set()


class TestItDoesNotCryWolf:
    """A check that fires on prose would be switched off, and protect nothing."""

    @pytest.mark.parametrize(
        "answer",
        [
            "I found 12 structures across 3 groups.",
            "Step 2 of 4 completed.",
            "Process pk 335625 finished with exit status 0.",
        ],
    )
    def test_prose_numbers_are_not_physics_claims(self, answer: str) -> None:
        assert ungrounded_quantities(answer, evidence="") == set()

    def test_the_users_own_value_is_not_called_invented(self) -> None:
        prompt = "submit a relax with ecutwfc 90 Ry"
        answer = "Submitting with your requested ecutwfc of 90 Ry."

        assert ungrounded_quantities(answer, evidence="", prompt=prompt) == set()

    def test_the_same_value_written_differently_still_matches(self) -> None:
        """60 in the answer must match 60.0 in the tool return."""
        assert ungrounded_quantities("use 60 Ry", evidence="{'ecutwfc': 60.0}") == set()


class TestQuantityDetection:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("a cutoff of 60 Ry", {"60.0"}),
            ("spacing of 0.15 A-1", {"0.15"}),
            ("set ecutwfc to 45", {"45.0"}),
            ("conv_thr of 1e-8", {"1e-08"}),
            ("found 12 structures", set()),
        ],
    )
    def test_only_units_and_named_parameters_count(
        self, text: str, expected: set[str]
    ) -> None:
        assert quantities_in(text) == expected


class TestEvidenceExtraction:
    def test_tool_returns_are_flattened_including_nested_values(self) -> None:
        """A number inside a nested dict is still a number the model was given."""
        from pydantic_ai.messages import ModelRequest, ToolReturnPart

        messages = [
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name="query_run_context",
                        content={"stats": {"median_ecutwfc": 60.0}},
                        tool_call_id="a",
                    )
                ]
            )
        ]

        assert "60.0" in tool_output_text(messages)

    def test_no_tool_calls_yields_no_evidence(self) -> None:
        """The severe case: an answer produced without calling anything."""
        assert tool_output_text([]) == ""


class TestPercentages:
    """The gap a real session found: an invented percentage passed unflagged.

    An agent asked to preview a resubmission answered in prose and offered
    "~40% more resources" -- a number no tool produced, reading as measured.
    Percentages were not treated as quantities at all, so nothing caught it.
    """

    def test_an_invented_percentage_is_reported(self) -> None:
        assert ungrounded_quantities("it needs ~40% more resources", "") == {"40.0"}

    def test_the_word_percent_counts_too(self) -> None:
        assert ungrounded_quantities("about 30 percent faster", "") == {"30.0"}

    def test_a_percentage_the_tools_returned_is_grounded(self) -> None:
        assert ungrounded_quantities("40% failed", "failure_rate 40%") == set()

    def test_a_percentage_is_grounded_by_its_fraction(self) -> None:
        """query_run_context reports success_rate as 0.67, not 67%.

        An answer saying "67%" is repeating that correctly. Flagging the
        conversion would train people to ignore the warning that matters.
        """
        assert ungrounded_quantities("67% of runs succeeded", "success_rate: 0.67") == (
            set()
        )

    def test_a_wrong_conversion_is_still_reported(self) -> None:
        """0.67 does not ground "80%" -- only the right conversion passes."""
        assert ungrounded_quantities("80% of runs succeeded", "success_rate: 0.67") == {
            "80.0"
        }

    def test_ordinary_prose_numbers_are_still_ignored(self) -> None:
        assert ungrounded_quantities("found 12 structures in step 2 of 3", "") == set()


class TestPythonBlocks:
    """Which parts of an answer are offered as code."""

    def test_a_labelled_fence_is_found(self) -> None:
        assert python_blocks("```python\nx = 1\n```") == ["x = 1\n"]

    def test_an_unlabelled_fence_counts_too(self) -> None:
        """Models routinely omit the language; the user pastes it regardless."""
        assert python_blocks("```\nx = 1\n```") == ["x = 1\n"]

    def test_prose_with_no_fence_yields_nothing(self) -> None:
        assert python_blocks("Use the QueryBuilder to do this.") == []

    def test_several_blocks_are_all_returned(self) -> None:
        answer = "First:\n```python\na = 1\n```\nThen:\n```python\nb = 2\n```"

        assert len(python_blocks(answer)) == 2


class TestSyntaxErrors:
    """Code that cannot run at all."""

    def test_valid_code_reports_nothing(self) -> None:
        assert syntax_errors("```python\nqb = QueryBuilder()\n```") == []

    def test_an_unclosed_paren_is_reported(self) -> None:
        problems = syntax_errors("```python\nqb = QueryBuilder(\n```")

        assert len(problems) == 1
        assert "never closed" in problems[0]

    def test_the_line_number_is_given(self) -> None:
        """So a long snippet's problem can be found without re-reading it."""
        problems = syntax_errors("```python\na = 1\nb = (\n```")

        assert "line 2" in problems[0]

    def test_a_shell_transcript_is_not_parsed_as_python(self) -> None:
        """A block of verdi commands is not meant to be a module."""
        assert syntax_errors("```\nverdi process list -a\n```") == []

    def test_a_repl_transcript_is_not_parsed_as_python(self) -> None:
        assert syntax_errors("```\n>>> node.pk\n334407\n```") == []

    def test_an_empty_block_is_not_an_error(self) -> None:
        assert syntax_errors("```python\n```") == []


class TestUngroundedSymbols:
    """AiiDA names in generated code that no tool output mentions."""

    def test_an_invented_import_is_flagged(self) -> None:
        answer = "```python\nfrom aiida.orm import MagicNode\n```"

        assert ungrounded_symbols(answer, "QueryBuilder and CalcJobNode") == {
            "MagicNode"
        }

    def test_a_retrieved_import_is_not_flagged(self) -> None:
        answer = "```python\nfrom aiida.orm import QueryBuilder\n```"

        assert ungrounded_symbols(answer, "use the QueryBuilder like so") == set()

    def test_several_names_on_one_import_are_checked_separately(self) -> None:
        answer = "```python\nfrom aiida.orm import QueryBuilder, Invented\n```"

        assert ungrounded_symbols(answer, "QueryBuilder") == {"Invented"}

    def test_a_non_aiida_import_is_ignored(self) -> None:
        """numpy is not ours to vouch for, and flagging it would be noise."""
        answer = "```python\nimport numpy\nfrom pathlib import Path\n```"

        assert ungrounded_symbols(answer, "") == set()

    def test_a_submodule_import_is_still_checked(self) -> None:
        answer = "```python\nfrom aiida.orm.nodes.data import Invented\n```"

        assert ungrounded_symbols(answer, "") == {"Invented"}

    def test_method_calls_are_deliberately_not_checked(self) -> None:
        """`.append` belongs to lists too; flagging it would drown the signal."""
        answer = "```python\nqb.append(CalcJobNode)\nqb.made_this_up()\n```"

        assert ungrounded_symbols(answer, "") == set()

    def test_unparseable_code_yields_no_symbol_findings(self) -> None:
        """syntax_errors reports that; reporting it twice helps nobody."""
        answer = "```python\nfrom aiida.orm import (\n```"

        assert ungrounded_symbols(answer, "") == set()

    def test_prose_naming_a_class_is_not_a_finding(self) -> None:
        """Only code is checked. Discussing a class is not claiming it exists."""
        answer = "You could imagine a MadeUpNode for this."

        assert ungrounded_symbols(answer, "") == set()


class TestSyntaxCheckingIsQuiet:
    """Only blocks that say they are Python get syntax-checked.

    Found in live testing: an analysis answer that illustrated a point with an
    unlabelled block of `verdi` commands produced "the Python above does not
    parse" every time. A warning that fires on correct output teaches people to
    ignore it, which is the exact failure this module is built to avoid.
    """

    @pytest.mark.parametrize(
        "block",
        [
            "pip install aiida-core",
            "key: value\n  nested: 1",
            "for each structure in group:\n    submit relax(structure)",
            "SELECT * FROM db_dbnode;",
        ],
    )
    def test_an_unlabelled_non_python_block_is_not_flagged(self, block: str) -> None:
        assert syntax_errors(f"```\n{block}\n```") == []

    def test_a_labelled_python_block_is_still_checked(self) -> None:
        """The check must not have been quietened into uselessness."""
        assert syntax_errors("```python\nqb = QueryBuilder(\n```")

    def test_a_labelled_python_block_that_parses_is_clean(self) -> None:
        assert syntax_errors("```python\nqb = QueryBuilder()\n```") == []

    def test_symbols_are_still_checked_in_unlabelled_blocks(self) -> None:
        """A non-Python block parses to nothing, so it cannot false-positive."""
        answer = "```\nfrom aiida.orm import MadeUpNode\n```"

        assert ungrounded_symbols(answer, "") == {"MadeUpNode"}

    def test_a_block_labelled_another_language_is_ignored_entirely(self) -> None:
        assert syntax_errors("```bash\npip install x\n```") == []
        assert labelled_python_blocks("```bash\nx = (\n```") == []
