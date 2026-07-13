"""Test suite for full workflow integration and execution pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import DeferredToolRequests

from aiida_agents.agents.execution import get_agent
from aiida_agents.tools.execution.analysis_queries import query_analysis_agent
from aiida_agents.tools.workflows.schemas import WorkflowSpec
from aiida_agents.tools.workflows.spec_generation import generate_workflow_spec
from aiida_agents.tools.workflows.spec_validation import validate_workflow_spec


class TestFullWorkflowIntegration:
    """Test full workflow progression: query → generate → validate → execution."""

    def test_full_successful_workflow(self) -> None:
        """Complete workflow should succeed: query → generate → validate."""
        # Step 1: Query for context
        context = query_analysis_agent(
            query_type="past_successful_workflows",
            filters={
                "workflow_type": "aiida.workflows:PwRelaxWorkChain",
                "structure_type": "metallic",
            },
        )
        assert isinstance(context["count"], int)
        assert context["count"] >= 0
        assert "success_rate" in context
        assert "structure_type_filter_note" in context, (
            "structure_type_filter_note must be present to inform the model "
            "that structure_type is not a DB-level filter"
        )

        # Step 2: Generate spec

        spec = generate_workflow_spec(
            description=f"Based on {context['count']} successful runs",
            workflow_type="aiida.workflows:PwRelaxWorkChain",
            structure_type="metallic",
            optimization_level="standard",
        )
        assert spec["workflow_type"] == "aiida.workflows:PwRelaxWorkChain"

        # Step 3: Add structure (user provides this)
        spec["inputs"]["structure"] = 12345

        # Step 4: Validate
        result = validate_workflow_spec(spec, structure_type="metallic")
        assert result["valid"] is True, f"Validation failed: {result.get('errors')}"

    def test_workflow_with_regeneration_on_error(self) -> None:
        """Workflow should allow regeneration when validation fails."""
        # Generate with bad parameters
        spec: WorkflowSpec = {
            "workflow_type": "aiida.workflows:PwRelaxWorkChain",
            "inputs": {
                "structure": 12345,
                "pw_code_label": "qe-pw-6.8",
                "pseudo_family": "SSSP/1.3/PBE/efficiency",
                "parameters": {
                    "system": {
                        "ecutwfc": 10,  # TOO LOW
                        "ecutrho": 80,
                    }
                },
            },
        }

        # Validate (should fail)
        result = validate_workflow_spec(spec, structure_type="metallic")
        assert result["valid"] is False
        assert len(result.get("suggestions", [])) > 0

        # Regenerate with correction
        spec_corrected = generate_workflow_spec(
            description="Geometry optimization (corrected)",
            workflow_type="aiida.workflows:PwRelaxWorkChain",
            structure_type="metallic",
            optimization_level="high_accuracy",  # Boost ecutwfc
        )
        spec_corrected["inputs"]["structure"] = 12345

        # Revalidate (should pass)
        result = validate_workflow_spec(spec_corrected, structure_type="metallic")
        assert result["valid"] is True

    def test_workflow_across_different_structure_types(self) -> None:
        """Workflow should work for metallic, insulator, semiconductor."""
        for structure_type in ["metallic", "insulator", "semiconductor"]:
            query_analysis_agent(
                query_type="past_successful_workflows",
                filters={
                    "workflow_type": "aiida.workflows:PwRelaxWorkChain",
                    "structure_type": structure_type,
                },
            )

            spec = generate_workflow_spec(
                description=f"Opt for {structure_type}",
                workflow_type="aiida.workflows:PwRelaxWorkChain",
                structure_type=structure_type,
                optimization_level="standard",
            )
            spec["inputs"]["structure"] = 12345

            result = validate_workflow_spec(spec, structure_type=structure_type)
            assert result["valid"] is True, (
                f"Failed for {structure_type}: {result.get('errors')}"
            )

    def test_verify_submit_workflow_invoked(self) -> None:
        """Verify that calling submit_workflow actually invokes the submission tool/function."""
        from aiida_agents.tools.submit import submit_workflow

        with patch("aiida_agents.tools.submit._prepare_submission") as mock_prep:
            mock_process = MagicMock()
            mock_prep.return_value = (mock_process, {"x": 2, "y": 3})

            with patch("aiida_agents.tools.submit._run_submission") as mock_run:
                mock_run.return_value = {
                    "pk": 999,
                    "uuid": "1234-5678-90ab",
                    "workflow": "core.arithmetic.add",
                    "state": "created",
                }

                res = submit_workflow("core.arithmetic.add", {"x": 2, "y": 3})
                mock_prep.assert_called_once_with(
                    "core.arithmetic.add", {"x": 2, "y": 3}
                )
                mock_run.assert_called_once_with(
                    "core.arithmetic.add", mock_process, {"x": 2, "y": 3}
                )
                assert res["pk"] == 999
                assert res["state"] == "created"

    def test_full_workflow_generation_validation_execution(self) -> None:
        """Test: user request → spec generation → validation → ready for execution preview (approval)."""
        # 1. User asks for geometry optimization
        spec = generate_workflow_spec(
            description="Geometry optimization of CoV",
            workflow_type="aiida.workflows:PwRelaxWorkChain",
            structure_type="metallic",
            optimization_level="standard",
        )
        spec["inputs"]["structure"] = 12345

        # 2. Validation passes
        val_result = validate_workflow_spec(spec, structure_type="metallic")
        assert val_result["valid"] is True

        # 3. Execution preview: agent invokes submit_workflow (requires_approval=True pauses for human check)
        agent = get_agent()
        with agent.override(model=TestModel(call_tools=["execute_workflow_spec"])):
            res = agent.run_sync("Please submit the calculation.")
            assert isinstance(res.output, DeferredToolRequests)
            assert len(res.output.approvals) == 1
            assert res.output.approvals[0].tool_name == "execute_workflow_spec"


class TestEdgeCasesAndFallbacks:
    """Test robustness: fallbacks when historical data or queries fail."""

    def test_spec_generation_fallback_if_query_fails(self) -> None:
        """Even with no prior context, generate_workflow_spec must produce valid spec using defaults."""
        spec = generate_workflow_spec(
            description="No historical data available",
            workflow_type="aiida.workflows:PwRelaxWorkChain",
            structure_type="metallic",
            optimization_level="standard",
        )
        assert spec["workflow_type"] == "aiida.workflows:PwRelaxWorkChain"
        assert "inputs" in spec
        assert "parameters" in spec["inputs"]
        assert spec["inputs"]["parameters"]["system"]["ecutwfc"] == 65
