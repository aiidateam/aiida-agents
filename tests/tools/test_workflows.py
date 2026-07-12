"""Tests for workflow specification tools (`aiida_agents.tools.workflows`).

Tests the standalone behavior of generate_workflow_spec and validate_workflow_spec
tools independent of agent orchestration or external services.
"""

from __future__ import annotations

import pytest

from aiida_agents.tools.workflows import (
    KNOWN_WORKFLOWS,
    execute_workflow_spec,
    generate_workflow_spec,
    get_workflow_templates,
    validate_workflow_spec,
)


class TestGenerateWorkflowSpecTool:
    """Test tool-level contract and output structure of generate_workflow_spec."""

    def test_output_has_required_top_level_keys(self) -> None:
        """The tool returns a dict with workflow_type, inputs, and metadata."""
        spec = generate_workflow_spec(
            description="Relaxation test",
            workflow_type="aiida.workflows:PwRelaxWorkChain",
            structure_type="metallic",
            optimization_level="standard",
        )
        assert isinstance(spec, dict)
        assert spec["workflow_type"] == "aiida.workflows:PwRelaxWorkChain"
        assert isinstance(spec["inputs"], dict)
        assert "parameters" in spec["inputs"]
        assert "metadata" in spec

    @pytest.mark.parametrize(
        "opt_level, expected_ecutwfc",
        [
            ("standard", 65.0),
            ("high_accuracy", 70.0),
        ],
    )
    def test_optimization_levels_scale_cutoffs(
        self, opt_level: str, expected_ecutwfc: float
    ) -> None:
        """Different optimization levels should adjust ecutwfc defaults accordingly."""
        spec = generate_workflow_spec(
            description=f"Level {opt_level}",
            workflow_type="aiida.workflows:PwRelaxWorkChain",
            structure_type="metallic",
            optimization_level=opt_level,
        )
        ecutwfc = spec["inputs"]["parameters"]["system"]["ecutwfc"]
        assert ecutwfc == expected_ecutwfc

    def test_bands_workflow_includes_bands_parameters(self) -> None:
        """PwBandsWorkChain spec generation should include bands-specific inputs."""
        spec = generate_workflow_spec(
            description="Band structure calculation",
            workflow_type="aiida.workflows:PwBandsWorkChain",
            structure_type="semiconductor",
            optimization_level="standard",
        )
        assert spec["workflow_type"] == "aiida.workflows:PwBandsWorkChain"
        assert "bands_kpoints_distance" in spec["inputs"]

    def test_vasp_workflow_spec_generation(self) -> None:
        """VASP spec generation includes ENCUT, EDIFF, and PREC."""
        spec = generate_workflow_spec(
            description="VASP relax",
            workflow_type="aiida.workflows:VaspRelaxWorkChain",
            structure_type="metallic",
            optimization_level="standard",
        )
        assert spec["workflow_type"] == "aiida.workflows:VaspRelaxWorkChain"
        assert "ENCUT" in spec["inputs"]["parameters"]
        assert spec["inputs"]["parameters"]["ENCUT"] == 520.0
        assert spec["inputs"]["parameters"]["PREC"] == "Accurate"


class TestValidateWorkflowSpecTool:
    """Test tool-level contract and output structure of validate_workflow_spec."""

    def test_valid_spec_returns_true_with_no_errors(self) -> None:
        """A complete, valid specification returns valid=True and empty error list."""
        spec = {
            "workflow_type": "aiida.workflows:PwRelaxWorkChain",
            "inputs": {
                "structure": "node_123",
                "pw_code_label": "qe-pw-6.8",
                "pseudo_family": "SSSP/1.3/PBE/efficiency",
                "parameters": {"system": {"ecutwfc": 65, "ecutrho": 520}},
            },
        }
        res = validate_workflow_spec(spec, structure_type="metallic")
        assert res["valid"] is True
        assert res["errors"] == []

    def test_invalid_spec_returns_false_and_suggestions(self) -> None:
        """An invalid ecutwfc returns valid=False and actionable suggestions."""
        spec = {
            "workflow_type": "aiida.workflows:PwRelaxWorkChain",
            "inputs": {
                "structure": "node_123",
                "pw_code_label": "qe-pw-6.8",
                "pseudo_family": "SSSP/1.3/PBE/efficiency",
                "parameters": {"system": {"ecutwfc": 10, "ecutrho": 80}},
            },
        }
        res = validate_workflow_spec(spec, structure_type="metallic")
        assert res["valid"] is False
        assert len(res["errors"]) > 0
        assert any("ecutwfc" in err["error"] for err in res["errors"])
        assert len(res["suggestions"]) > 0


class TestGetWorkflowTemplatesTool:
    """Test tool-level contract and output of get_workflow_templates."""

    def test_get_all_templates_when_none(self) -> None:
        """If workflow_type is None, returns catalog of all known templates."""
        res = get_workflow_templates()
        assert res["query_type"] == "all_workflow_templates"
        assert res["total_workflows"] >= 8
        assert "aiida.workflows:PwRelaxWorkChain" in res["workflows"]
        assert "aiida.workflows:VaspRelaxWorkChain" in res["workflows"]

    def test_get_specific_template(self) -> None:
        """If workflow_type is provided, returns detailed schema and bounds."""
        res = get_workflow_templates("aiida.workflows:VaspRelaxWorkChain")
        assert res["workflow_type"] == "aiida.workflows:VaspRelaxWorkChain"
        assert "parameters_schema" in res
        assert "ENCUT" in res["parameters_schema"]
        assert "EDIFF" in res["parameters_schema"]

    def test_unknown_workflow_template(self) -> None:
        """If workflow_type is unknown, returns error and list of available workflows."""
        res = get_workflow_templates("aiida.workflows:UnknownWorkChain")
        assert "error" in res
        assert "available_workflows" in res


class TestExecuteWorkflowSpecTool:
    """Test tool-level contract and validation in execute_workflow_spec."""

    def test_invalid_or_missing_keys_raises_error(self) -> None:
        """Raises error if validated_spec is not a proper dict."""
        with pytest.raises(ValueError, match="missing required 'workflow_type'"):
            execute_workflow_spec({})

        with pytest.raises(ValueError, match="missing required 'inputs'"):
            execute_workflow_spec({"workflow_type": "aiida.workflows:PwRelaxWorkChain"})

    def test_unknown_workflow_raises_error(self) -> None:
        """Raises error if workflow is unknown."""
        with pytest.raises(ValueError, match="not known"):
            execute_workflow_spec(
                {
                    "workflow_type": "aiida.workflows:Unknown",
                    "inputs": {"structure": 123},
                }
            )


def test_known_workflows_export() -> None:
    """Verify KNOWN_WORKFLOWS contains expected default entry points."""
    assert "aiida.workflows:PwRelaxWorkChain" in KNOWN_WORKFLOWS
    assert "aiida.workflows:PwBandsWorkChain" in KNOWN_WORKFLOWS
    assert "aiida.workflows:VaspWorkChain" in KNOWN_WORKFLOWS
    assert "aiida.workflows:VaspRelaxWorkChain" in KNOWN_WORKFLOWS
