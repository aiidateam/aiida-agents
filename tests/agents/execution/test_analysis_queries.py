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
        """Querying a workflow without historical data should return count=0 cleanly without errors."""
        res = query_analysis_agent(
            query_type="past_successful_workflows",
            filters={"workflow_type": "aiida.workflows:ExoticWorkChain"},
        )
        assert res["query_type"] == "past_successful_workflows"
        assert res["count"] == 0
        assert "note" in res
        assert "Using defaults" in res["note"]
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
