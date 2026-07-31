"""Tests for the runtime grounding check.

This is the backstop that does not depend on the model cooperating. The prompt
version of the same rule was measured at five ignored instructions out of five,
so these hold the check to catching what the prompt did not.
"""

from __future__ import annotations

import pytest

from aiida_agents.grounding import (
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
