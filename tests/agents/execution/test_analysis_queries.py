"""Test suite for query_analysis_agent tool."""

from __future__ import annotations

import pytest

from aiida_agents.tools.execution.analysis_queries import query_analysis_agent


class TestAnalysisQueries:
    """Verify query_analysis_agent queries return expected structure and data."""

    def test_query_past_workflows_returns_expected_structure(self):
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

    def test_query_available_codes_returns_list(self):
        """Querying available codes returns code list and recommended version."""
        res = query_analysis_agent(
            query_type="available_codes",
            filters={"code": "qe-pw"},
        )
        assert res["query_type"] == "available_codes"
        assert isinstance(res.get("codes"), list)
        assert len(res["codes"]) > 0
        assert res.get("recommended_version") == "qe-pw-6.8"

    def test_query_failed_attempts_structured(self):
        """Querying failed attempts should return structured failure modes."""
        res = query_analysis_agent(
            query_type="failed_attempts",
            filters={"workflow_type": "PwRelaxWorkChain"},
        )
        assert res["query_type"] == "failed_attempts"
        assert "attempts" in res
        assert isinstance(res["attempts"], list)

    def test_invalid_query_type_raises_error(self):
        """Unknown query_type must raise ValueError."""
        with pytest.raises(ValueError, match="Unknown query_type"):
            query_analysis_agent("invalid_query_type", {})


class TestAnalysisQueriesEdgeCases:
    """Verify behavior on empty results and unknown filters."""

    def test_query_past_workflows_empty_results(self):
        """Querying a workflow without historical data should return count=0 cleanly without errors."""
        res = query_analysis_agent(
            query_type="past_successful_workflows",
            filters={"workflow_type": "aiida.workflows:ExoticWorkChain"},
        )
        assert res["query_type"] == "past_successful_workflows"
        assert res["count"] == 0
        assert "note" in res
        assert "Using defaults" in res["note"]

    def test_query_available_codes_empty_results(self):
        """Querying an unknown code should return an empty codes list."""
        res = query_analysis_agent(
            query_type="available_codes",
            filters={"code": "unknown-abinit-code"},
        )
        assert res["query_type"] == "available_codes"
        assert isinstance(res["codes"], list)
        assert len(res["codes"]) == 0
