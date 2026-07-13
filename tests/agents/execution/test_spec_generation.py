"""Test suite for workflow specification generation tool."""

from __future__ import annotations

import pytest

from aiida_agents.tools.workflows.schemas import KNOWN_WORKFLOWS, WorkflowSpec
from aiida_agents.tools.workflows.spec_generation import generate_workflow_spec
from aiida_agents.tools.workflows.spec_validation import validate_workflow_spec


class TestSpecGeneration:
    """Test standard spec generation outputs across workflows."""

    def test_spec_is_dict_not_string(self) -> None:
        """Spec should be a dict (JSON-serializable), not a string."""
        spec = generate_workflow_spec(
            description="Test relax",
            workflow_type="aiida.workflows:PwRelaxWorkChain",
            structure_type="metallic",
            optimization_level="standard",
        )
        assert isinstance(spec, dict), "Spec must be dict, not string"

    def test_spec_has_required_keys(self) -> None:
        """Spec must have workflow_type, inputs, metadata."""
        spec = generate_workflow_spec(
            description="Test relax",
            workflow_type="aiida.workflows:PwRelaxWorkChain",
            structure_type="metallic",
            optimization_level="standard",
        )
        assert "workflow_type" in spec
        assert "inputs" in spec
        assert "metadata" in spec

    def test_spec_inputs_has_parameters(self) -> None:
        """Inputs must contain parameters sub-dict."""
        spec = generate_workflow_spec(
            description="Test relax",
            workflow_type="aiida.workflows:PwRelaxWorkChain",
            structure_type="metallic",
            optimization_level="standard",
        )
        assert "parameters" in spec["inputs"]
        assert isinstance(spec["inputs"]["parameters"], dict)

    def test_spec_parameters_have_numeric_values(self) -> None:
        """Numeric parameters should not be strings."""
        spec = generate_workflow_spec(
            description="Test relax",
            workflow_type="aiida.workflows:PwRelaxWorkChain",
            structure_type="metallic",
            optimization_level="standard",
        )
        params = spec["inputs"]["parameters"]
        # In structured relax params, system holds ecutwfc/ecutrho
        assert isinstance(params["system"]["ecutwfc"], (int, float))
        assert isinstance(params["system"]["ecutrho"], (int, float))

    def test_all_known_workflows_produce_valid_specs(self) -> None:
        """Test every known workflow generates a valid specification."""
        for wf_type in KNOWN_WORKFLOWS.keys():
            spec = generate_workflow_spec(
                description=f"Test {wf_type}",
                workflow_type=wf_type,
                structure_type="insulator",
                optimization_level="standard",
            )
            assert spec["workflow_type"] == wf_type
            assert "inputs" in spec


class TestSpecGenerationEdgeCases:
    """Test edge cases and conflict handling for spec generation and validation."""

    def test_unknown_workflow_type(self) -> None:
        """If KNOWN_WORKFLOWS doesn't have the requested workflow, generate_workflow_spec should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown workflow_type"):
            generate_workflow_spec(
                description="Test unknown workflow",
                workflow_type="aiida.workflows:ExoticWorkChain",
                structure_type="metallic",
                optimization_level="standard",
            )

    def test_invalid_structure_type_or_optimization_level(self) -> None:
        """Ensure boundary validation rejects invalid structure_type or optimization_level."""
        with pytest.raises(ValueError, match="structure_type must be"):
            generate_workflow_spec(
                description="Test bad structure",
                workflow_type="aiida.workflows:PwRelaxWorkChain",
                structure_type="alien_crystal",
                optimization_level="standard",
            )

        with pytest.raises(ValueError, match="optimization_level must be"):
            generate_workflow_spec(
                description="Test bad level",
                workflow_type="aiida.workflows:PwRelaxWorkChain",
                structure_type="metallic",
                optimization_level="ultra_super_high",
            )

    def test_conflict_between_toplevel_and_nested_parameters(self) -> None:
        """Test handling when user provides both flat ecutwfc and nested system.ecutwfc (conflict)."""
        spec: WorkflowSpec = {
            "workflow_type": "aiida.workflows:PwRelaxWorkChain",
            "inputs": {
                "pw_code_label": "qe-pw-6.8",
                "structure": "node_123",
                "pseudo_family": "SSSP/1.3/PBE/efficiency",
                "parameters": {
                    "ecutwfc": 40.0,  # Top-level conflicting value
                    "system": {
                        "ecutwfc": 65.0,  # Nested value
                        "ecutrho": 520.0,
                    },
                },
            },
        }
        result = validate_workflow_spec(spec, structure_type="metallic")
        # Validation checks both parameter instances. Since 40.0 and 65.0 are checked against range,
        # ensure no unhandled exception occurs and both get evaluated cleanly.
        assert isinstance(result, dict)
        assert "valid" in result
