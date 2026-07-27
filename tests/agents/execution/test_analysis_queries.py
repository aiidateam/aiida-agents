"""Test suite for query_analysis_agent tool."""

from __future__ import annotations

import pytest
from typing import Any

from aiida_agents.tools.execution.analysis_queries import query_analysis_agent


class TestAnalysisQueries:
    """Verify query_analysis_agent queries return expected structure and data."""

    def test_query_past_workflows_returns_expected_structure(self) -> None:
        """Querying past workflows must return required structured fields."""
        res = query_analysis_agent(
            query_type="past_successful_workflows",
            filters={"workflow_type": "PwRelaxWorkChain", "structure_type": "metallic"},
        )
        assert res["query_type"] == "past_successful_workflows"
        assert res["workflow_type"] == "PwRelaxWorkChain"
        assert "count" in res
        assert "success_rate" in res
        assert "median_ecutwfc" in res
        assert "common_parameters" in res
        assert "common_failure_modes" in res
        assert "example_structures" in res
        assert "structure_type_filter_note" in res, (
            "structure_type_filter_note must be present so the model knows "
            "structure_type is not a DB-level filter"
        )

    def test_query_available_codes_returns_list(self, arithmetic_add_code: Any) -> None:
        """Querying available codes returns a list of codes with expected structure.

        Uses the ``arithmetic_add_code`` session fixture to guarantee at least
        one real code exists in the in-memory test profile.
        """
        res = query_analysis_agent(
            query_type="available_codes",
            filters={},
        )
        assert res["query_type"] == "available_codes"
        assert isinstance(res.get("codes"), list)
        # The conftest registers a 'bash' code — at least that one must appear
        assert len(res["codes"]) > 0
        # Every entry must have the expected schema fields
        for code in res["codes"]:
            assert "label" in code
            assert "plugin" in code

    def test_query_failed_attempts_structured(self) -> None:
        """Querying failed attempts should return structured failure modes."""
        res = query_analysis_agent(
            query_type="failed_attempts",
            filters={"workflow_type": "PwRelaxWorkChain"},
        )
        assert res["query_type"] == "failed_attempts"
        assert "attempts" in res
        assert isinstance(res["attempts"], list)

    def test_invalid_query_type_raises_error(self) -> None:
        """Unknown query_type must raise ValueError."""
        with pytest.raises(ValueError, match="Unknown query_type"):
            query_analysis_agent("invalid_query_type", {})


class TestAnalysisQueriesEdgeCases:
    """Verify behavior on empty results and unknown filters."""

    def test_query_past_workflows_empty_results(self) -> None:
        """A workflow with no matching runs returns the full shape, not an error.

        The workflow named here is not installed, so it exercises the
        no-matches path. This test owns the *structural* contract -- every
        statistics key present, defaults filled in, nothing raised -- so a
        caller can read the result without branching.

        What the note should *say* in that situation is asserted by
        ``TestEntryPointSpellingsAllFindTheSameRuns`` instead. It used to be
        checked here as "Using defaults", which encoded the very conflation
        that hid a real bug: an unregistered name and a registered workflow
        with no history are different facts and were reported identically.
        """
        res = query_analysis_agent(
            query_type="past_successful_workflows",
            filters={"workflow_type": "aiida.workflows:ExoticWorkChain"},
        )
        assert res["query_type"] == "past_successful_workflows"
        assert res["count"] == 0
        assert res["success_rate"] == 0.0
        assert res["median_ecutwfc"] is None
        assert res["common_parameters"] == {}
        assert res["note"]
        assert "structure_type_filter_note" in res

    def test_query_available_codes_empty_results(self) -> None:
        """Querying an unknown code should return an empty codes list."""
        res = query_analysis_agent(
            query_type="available_codes",
            filters={"code": "unknown-abinit-code"},
        )
        assert res["query_type"] == "available_codes"
        assert isinstance(res["codes"], list)
        assert len(res["codes"]) == 0


class TestEntryPointSpellingsAllFindTheSameRuns:
    """A workflow's history must be found however the agent names the workflow.

    Regression from executing the scenarios in ``docs/gsoc/agent-scenarios.md``
    against real nodes. Nodes store ``process_label`` as the class name, but the
    lookup derived it with ``workflow_type.split(":")[-1]``, which only works
    for the legacy ``aiida.workflows:PwRelaxWorkChain`` spelling used in the
    prompt's examples. ``list_workflows()`` hands the agent modern entry points
    (``core.arithmetic.multiply_add``), which have no colon, were passed through
    whole, and matched nothing.

    The result was silent: "No prior runs of this workflow found. Using
    defaults." on a database containing them, so the agent built inputs
    believing there was no history to draw on.
    """

    @pytest.mark.parametrize(
        "workflow_type",
        [
            "core.arithmetic.multiply_add",  # what list_workflows() reports
            "aiida.workflows:core.arithmetic.multiply_add",  # what process_type stores
            "MultiplyAddWorkChain",  # the class name itself
        ],
    )
    def test_every_spelling_finds_the_run(
        self,
        multiply_add_workchain: Any,
        workflow_type: str,
    ) -> None:
        res = query_analysis_agent(
            query_type="past_successful_workflows",
            filters={"workflow_type": workflow_type},
        )

        assert res["count"] >= 1, (
            f"{workflow_type!r} found no runs of "
            f"{multiply_add_workchain.process_label!r}; note was: {res.get('note')!r}"
        )
        assert res["success_rate"] > 0

    def test_unresolvable_workflow_says_so_instead_of_claiming_no_history(
        self,
    ) -> None:
        """An unregistered name must not be reported as "no prior runs".

        The two are different facts and lead the agent to different actions:
        one means proceed with defaults, the other means the request itself was
        wrong. Conflating them is what made the original bug invisible.
        """
        res = query_analysis_agent(
            query_type="past_successful_workflows",
            filters={"workflow_type": "definitely.not.installed"},
        )

        assert res["count"] == 0
        assert "not a registered entry point" in res["note"]
